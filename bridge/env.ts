/**
 * RL environment bridge: JSONL protocol over stdio wrapping the headless
 * OpenFront engine. One agent-controlled human player vs the map's nations.
 *
 * stdin (one JSON per line):
 *   {"op": "reset", "map": "Onion", "seed": "abc"}
 *   {"op": "step", "intents": [Intent...], "ticks": 10}
 *   {"op": "close"}
 *
 * stdout (one JSON per line): observation payloads (see buildObs). Tile
 * state ships as base64(gzip(uint16-le grid)) - ~30-80KB per step at the
 * 10-tick decision cadence. All engine logging goes to stderr.
 *
 * Legality: the bridge reports action-type masks and per-player validity
 * (exact engine calls). Tile arguments are validated by the engine at
 * execution; intents the engine would silently discard are counted in the
 * obs head as "wasted" so the env can penalize them - without a cost, a
 * doomed boat/build is reward-identical to noop and strictly better in
 * expectation (occasional lottery win), so the policy farms them
 * (region-snapping happens Python-side, see DESIGN.md head 3).
 */

import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import { Config } from "../openfront/src/core/configuration/Config";
import { DoomsdayClockExecution } from "../openfront/src/core/execution/DoomsdayClockExecution";
import { Executor } from "../openfront/src/core/execution/ExecutionManager";
import { RecomputeRailClusterExecution } from "../openfront/src/core/execution/RecomputeRailClusterExecution";
import { SpawnTimerExecution } from "../openfront/src/core/execution/SpawnTimerExecution";
import { WinCheckExecution } from "../openfront/src/core/execution/WinCheckExecution";
import {
  Game,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  Difficulty,
  PlayerInfo,
  PlayerType,
  UnitType,
} from "../openfront/src/core/game/Game";
import { createGame } from "../openfront/src/core/game/GameImpl";
import { GameUpdateType } from "../openfront/src/core/game/GameUpdates";
import { createNationsForGame } from "../openfront/src/core/game/NationCreation";
import { genTerrainFromBin } from "../openfront/src/core/game/TerrainMapLoader";
import { PseudoRandom } from "../openfront/src/core/PseudoRandom";
import type {
  GameConfig,
  GameStartInfo,
  Intent,
  StampedIntent,
} from "../openfront/src/core/Schemas";
import { simpleHash } from "../openfront/src/core/Util";
import type { MapManifest } from "../openfront/src/core/game/TerrainMapLoader";
import { bordersNeutralLand, buildObsParts, terrainPayload } from "./common";

// The engine logs to console.log; stdout must stay pure JSONL.
console.log = console.info = console.warn = (...args: unknown[]) =>
  process.stderr.write(args.map(String).join(" ") + "\n");

const AGENT_CLIENT_ID = "AGENTRL1";
const REPO_ROOT = path.resolve(__dirname, "..");
const MAPS_DIR = path.join(REPO_ROOT, "openfront", "resources", "maps");

function mapDirName(mapType: GameMapType): string {
  const key = Object.keys(GameMapType).find(
    (k) => GameMapType[k as keyof typeof GameMapType] === mapType,
  )!;
  // Map dirs are fully lowercase; matters on case-sensitive filesystems.
  return key.toLowerCase();
}

function seedToGameID(seed: string): string {
  // Deterministic 8-char alnum ID (GAME_ID_REGEX) from an arbitrary seed.
  const alphabet =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let h = simpleHash(`rl-${seed}`);
  let out = "";
  for (let i = 0; i < 8; i++) {
    h = (h * 1103515245 + 12345) & 0x7fffffff;
    out += alphabet[h % alphabet.length];
  }
  return out;
}

class EnvSession {
  game!: Game;
  executor!: Executor;
  gameID = "";
  gameConfig!: GameConfig;
  turns: { turnNumber: number; intents: StampedIntent[] }[] = [];
  startTime = 0;
  lastWinner: unknown = null;

  async reset(
    mapKey: string,
    seed: string,
    bots: number,
    difficulty: string,
    nations: number | "default" | "disabled" = "default",
  ): Promise<{ head: Record<string, unknown>; tiles: Buffer }> {
    const mapType = GameMapType[mapKey as keyof typeof GameMapType];
    if (!mapType) throw new Error(`unknown map ${mapKey}`);
    const diff = Difficulty[difficulty as keyof typeof Difficulty];
    if (!diff) throw new Error(`unknown difficulty ${difficulty}`);

    const gameConfig: GameConfig = {
      gameMap: mapType,
      gameMapSize: GameMapSize.Normal,
      gameMode: GameMode.FFA,
      gameType: GameType.Singleplayer,
      difficulty: diff,
      nations,
      donateGold: true,
      donateTroops: true,
      bots,
      infiniteGold: false,
      infiniteTroops: false,
      instantBuild: false,
      randomSpawn: false,
    };
    const gameID = seedToGameID(seed);
    const gameStart: GameStartInfo = {
      gameID,
      lobbyCreatedAt: Date.now(),
      config: gameConfig,
      players: [
        { clientID: AGENT_CLIENT_ID, username: "Agent", clanTag: null },
      ],
    };
    this.gameID = gameID;
    this.gameConfig = gameConfig;
    this.turns = [];
    this.startTime = Date.now();
    this.lastWinner = null;
    const config = new Config(gameConfig, null, false);

    const dir = path.join(MAPS_DIR, mapDirName(mapType));
    const manifest = JSON.parse(
      fs.readFileSync(path.join(dir, "manifest.json"), "utf8"),
    ) as MapManifest;
    const gameMap = await genTerrainFromBin(
      manifest.map,
      new Uint8Array(fs.readFileSync(path.join(dir, "map.bin"))),
    );
    const miniGameMap = await genTerrainFromBin(
      manifest.map4x,
      new Uint8Array(fs.readFileSync(path.join(dir, "map4x.bin"))),
    );

    // Mirror createGameRunner() exactly (humans consume random.nextID()
    // first, then nations) so recorded games replay bit-identically in the
    // real OpenFront client.
    const random = new PseudoRandom(simpleHash(gameID));
    const humans = gameStart.players.map(
      (p) =>
        new PlayerInfo(p.username, PlayerType.Human, p.clientID, random.nextID()),
    );
    const gameNations = createNationsForGame(
      gameStart,
      manifest.nations,
      manifest.additionalNations ?? [],
      humans.length,
      random,
    );

    this.game = createGame(
      humans,
      gameNations,
      gameMap,
      miniGameMap,
      config,
      manifest.teamGameSpawnAreas,
    );
    this.executor = new Executor(this.game, gameID, AGENT_CLIENT_ID);
    // Mirror GameRunner.init(): singleplayer has no spawn timer - the spawn
    // phase ends the moment the human (agent) picks a spawn.
    if (gameConfig.gameType !== GameType.Singleplayer) {
      this.game.addExecution(new SpawnTimerExecution());
    }
    if (config.spawnNations()) {
      this.game.addExecution(...this.executor.nationExecutions());
    }
    if (config.bots() > 0) {
      this.game.addExecution(...this.executor.spawnTribes(config.bots()));
    }
    this.game.addExecution(new WinCheckExecution());
    if (config.doomsdayClockConfig().enabled) {
      this.game.addExecution(new DoomsdayClockExecution());
    }
    if (!config.isUnitDisabled(UnitType.Factory)) {
      this.game.addExecution(
        new RecomputeRailClusterExecution(this.game.railNetwork()),
      );
    }

    // Advance one tick so executions initialize; agent spawns via step().
    this.game.executeNextTick();
    return buildObsParts(this.game, AGENT_CLIENT_ID, null);
  }

  /** Count intents the engine would silently discard at execution. Uses
   * the same engine calls the executions run at init, against the same
   * state the action masks were built from, so the env can penalize
   * wasted intents (otherwise a doomed boat/build is reward-identical to
   * noop and the policy farms them as free lottery tickets). */
  countWasted(intents: Intent[]): number {
    const agent = this.game.playerByClientID(AGENT_CLIENT_ID);
    if (!agent || !agent.isAlive()) return 0;
    let wasted = 0;
    for (const intent of intents) {
      if (intent.type === "boat") {
        // TransportShipExecution.init(): boat cap, reachable target shore,
        // launchable own shore, non-friendly destination owner.
        if (agent.canBuild(UnitType.TransportShip, intent.dst) === false) {
          wasted++;
        }
      } else if (intent.type === "build_unit") {
        // ConstructionExecution / NukeExecution: ownership + structure
        // spacing for buildings, silo cooldown + spawn immunity for nukes.
        if (agent.canBuild(intent.unit, intent.tile) === false) {
          wasted++;
        }
      } else if (intent.type === "attack" && intent.targetID === null) {
        // AttackExecution on terra nullius fizzles without neutral border.
        if (!bordersNeutralLand(this.game, agent)) {
          wasted++;
        }
      } else if (intent.type === "upgrade_structure") {
        // UpgradeStructureExecution: unit exists, owned, upgradable type,
        // not constructing / marked for deletion.
        const unit = this.game.unit(intent.unitId);
        if (!unit || unit.owner() !== agent || !agent.canUpgradeUnit(unit)) {
          wasted++;
        }
      } else if (intent.type === "move_warship") {
        // MoveWarshipExecution: valid water position and at least one own
        // active warship in the same water component.
        const comp = this.game.getWaterComponent(intent.tile);
        const ships = new Map(
          agent.units(UnitType.Warship).map((u) => [u.id(), u]),
        );
        const movable =
          comp !== null &&
          intent.unitIds.some((id) => {
            const w = ships.get(id);
            return (
              w !== undefined &&
              w.isActive() &&
              this.game.hasWaterComponent(w.tile(), comp)
            );
          });
        if (!movable) wasted++;
      } else if (intent.type === "cancel_boat") {
        // BoatRetreatExecution: own in-flight transport with that id.
        if (
          !agent
            .units(UnitType.TransportShip)
            .some((u) => u.id() === intent.unitID)
        ) {
          wasted++;
        }
      } else if (intent.type === "delete_unit") {
        // DeleteUnitExecution: own active unit on own land territory,
        // delete cooldown expired, not spawn phase.
        const unit = this.game.unit(intent.unitId);
        const owner = unit ? this.game.owner(unit.tile()) : null;
        if (
          !unit ||
          unit.owner() !== agent ||
          !unit.isActive() ||
          !this.game.isLand(unit.tile()) ||
          !(owner?.isPlayer() && owner.id() === agent.id()) ||
          this.game.inSpawnPhase() ||
          !agent.canDeleteUnit()
        ) {
          wasted++;
        }
      } else if (intent.type === "cancel_attack") {
        // RetreatExecution on a nonexistent attack id is a no-op.
        if (!agent.outgoingAttacks().some((a) => a.id() === intent.attackID)) {
          wasted++;
        }
      } else if (intent.type === "embargo") {
        // EmbargoExecution: starting an existing embargo / stopping a
        // nonexistent one changes nothing.
        const target = this.game.hasPlayer(intent.targetID)
          ? this.game.player(intent.targetID)
          : null;
        const has = target !== null && agent.hasEmbargoAgainst(target);
        if (target === null || (intent.action === "start" ? has : !has)) {
          wasted++;
        }
      } else if (intent.type === "targetPlayer") {
        // TargetPlayerExecution gates on canTarget.
        const target = this.game.hasPlayer(intent.target)
          ? this.game.player(intent.target)
          : null;
        if (target === null || !agent.canTarget(target)) wasted++;
      } else if (intent.type === "allianceExtension") {
        // AllianceExtensionExecution needs an active alliance inside the
        // extension window that the agent hasn't already agreed to extend.
        const other = this.game.hasPlayer(intent.recipient)
          ? this.game.player(intent.recipient)
          : null;
        if (other === null || agent.allianceInfo(other)?.canExtend !== true) {
          wasted++;
        }
      }
    }
    return wasted;
  }

  step(
    intents: Intent[],
    ticks: number,
  ): { head: Record<string, unknown>; tiles: Buffer } {
    const wasted = this.countWasted(intents);
    const stamped: StampedIntent[] = intents.map((i) => ({
      ...i,
      clientID: AGENT_CLIENT_ID,
    })) as StampedIntent[];
    if (stamped.length > 0) {
      const turn = { turnNumber: this.game.ticks(), intents: stamped };
      this.turns.push(turn);
      for (const exec of this.executor.createExecs(turn)) {
        this.game.addExecution(exec);
      }
    }

    let winner: unknown = null;
    for (let t = 0; t < ticks; t++) {
      const updates = this.game.executeNextTick();
      const winUpdates = updates[GameUpdateType.Win];
      if (winUpdates && winUpdates.length > 0) {
        winner = (winUpdates[0] as { winner: unknown }).winner;
        this.lastWinner = winner;
        break;
      }
    }
    const parts = buildObsParts(this.game, AGENT_CLIENT_ID, winner);
    parts.head.wasted = wasted;
    return parts;
  }

  /** GameRecord JSON loadable by the real OpenFront client replay viewer. */
  saveRecord(outPath: string): object {
    const end = Date.now();
    const record = {
      info: {
        gameID: this.gameID,
        lobbyCreatedAt: this.startTime,
        config: this.gameConfig,
        players: [
          {
            clientID: AGENT_CLIENT_ID,
            username: "Agent",
            clanTag: null,
            persistentID: null,
            stats: {},
          },
        ],
        start: this.startTime,
        end,
        duration: Math.floor((end - this.startTime) / 1000),
        num_turns: this.game.ticks(),
        winner: this.lastWinner ?? undefined,
        lobbyFillTime: 0,
      },
      version: "v0.0.2",
      gitCommit: "DEV",
      subdomain: "rl",
      domain: "localhost",
      turns: this.turns,
    };
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    fs.writeFileSync(outPath, JSON.stringify(record));
    return { saved: outPath, gameID: this.gameID, turns: this.turns.length };
  }

  terrain(): object {
    return terrainPayload(this.game);
  }
}

async function main() {
  const session = new EnvSession();
  const rl = readline.createInterface({ input: process.stdin });
  const write = (obj: object) => process.stdout.write(JSON.stringify(obj) + "\n");
  // Obs responses ship the tile state as a raw binary frame after the JSON
  // header line ("tilesBin" = byte length) - no gzip, no base64. The tile
  // codec was pure CPU overhead on both sides of the pipe.
  const writeObs = (
    parts: { head: Record<string, unknown>; tiles: Buffer },
    extra: object = {},
  ) => {
    process.stdout.write(
      JSON.stringify({ ...parts.head, ...extra, tilesBin: parts.tiles.length }) + "\n",
    );
    process.stdout.write(parts.tiles);
  };

  for await (const line of rl) {
    if (!line.trim()) continue;
    let msg: {
      op: string;
      map?: string;
      seed?: string;
      bots?: number;
      difficulty?: string;
      nations?: number | "default" | "disabled";
      intents?: Intent[];
      ticks?: number;
      path?: string;
    };
    try {
      msg = JSON.parse(line);
    } catch {
      write({ error: "bad json" });
      continue;
    }
    try {
      if (msg.op === "reset") {
        const parts = await session.reset(
          msg.map ?? "Onion",
          msg.seed ?? "0",
          msg.bots ?? 100,
          msg.difficulty ?? "Medium",
          msg.nations ?? "default",
        );
        writeObs(parts, session.terrain());
      } else if (msg.op === "step") {
        writeObs(session.step(msg.intents ?? [], msg.ticks ?? 10));
      } else if (msg.op === "save_record") {
        write(session.saveRecord(msg.path ?? "/tmp/openfront_record.json"));
      } else if (msg.op === "close") {
        break;
      } else {
        write({ error: `unknown op ${msg.op}` });
      }
    } catch (err) {
      write({ error: String(err instanceof Error ? err.stack : err) });
    }
  }
  process.exit(0);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
