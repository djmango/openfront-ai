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
 * state ships as base64(gzip(uint16-le grid)) — ~30-80KB per step at the
 * 10-tick decision cadence. All engine logging goes to stderr.
 *
 * Legality: the bridge reports action-type masks and per-player validity
 * (exact engine calls). Tile arguments are validated by the engine at
 * execution; illegal tile picks become no-ops the policy learns to avoid
 * (region-snapping happens Python-side, see DESIGN.md head 3).
 */

import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import * as zlib from "zlib";
import { Config } from "../openfront/src/core/configuration/Config";
import { Executor } from "../openfront/src/core/execution/ExecutionManager";
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
import type { MapManifest } from "../openfront/src/core/game/TerrainMapFileLoader";

// The engine logs to console.log; stdout must stay pure JSONL.
console.log = console.info = console.warn = (...args: unknown[]) =>
  process.stderr.write(args.map(String).join(" ") + "\n");

const AGENT_CLIENT_ID = "AGENTRL1";
const REPO_ROOT = path.resolve(__dirname, "..");
const MAPS_DIR = path.join(REPO_ROOT, "openfront", "resources", "maps");

const STRUCTURES = [
  UnitType.City,
  UnitType.Port,
  UnitType.DefensePost,
  UnitType.MissileSilo,
  UnitType.SAMLauncher,
  UnitType.Factory,
];
const LAUNCHABLE = [UnitType.AtomBomb, UnitType.HydrogenBomb, UnitType.MIRV];

function mapDirName(mapType: GameMapType): string {
  const key = Object.keys(GameMapType).find(
    (k) => GameMapType[k as keyof typeof GameMapType] === mapType,
  )!;
  return key.charAt(0).toLowerCase() + key.slice(1);
}

class EnvSession {
  game!: Game;
  executor!: Executor;
  agentSmallID = -1;

  async reset(mapKey: string, seed: string): Promise<object> {
    const mapType = GameMapType[mapKey as keyof typeof GameMapType];
    if (!mapType) throw new Error(`unknown map ${mapKey}`);

    const gameConfig: GameConfig = {
      gameMap: mapType,
      gameMapSize: GameMapSize.Normal,
      gameMode: GameMode.FFA,
      gameType: GameType.Singleplayer,
      difficulty: Difficulty.Medium,
      nations: "default",
      donateGold: true,
      donateTroops: true,
      bots: 100,
      infiniteGold: false,
      infiniteTroops: false,
      instantBuild: false,
    };
    const gameID = `rl-${seed}`;
    const gameStart: GameStartInfo = {
      gameID,
      lobbyCreatedAt: Date.now(),
      config: gameConfig,
      players: [],
    };
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

    const random = new PseudoRandom(simpleHash(gameID));
    const nations = createNationsForGame(
      gameStart,
      manifest.nations,
      manifest.additionalNations ?? [],
      1, // one human slot
      random,
    );

    const agentInfo = new PlayerInfo(
      "Agent",
      PlayerType.Human,
      AGENT_CLIENT_ID,
      random.nextID(),
    );

    this.game = createGame(
      [agentInfo],
      nations,
      gameMap,
      miniGameMap,
      config,
      manifest.teamGameSpawnAreas,
    );
    this.executor = new Executor(this.game, gameID, AGENT_CLIENT_ID);
    this.game.addExecution(new SpawnTimerExecution());
    if (config.spawnNations()) {
      this.game.addExecution(...this.executor.nationExecutions());
    }
    if (config.bots() > 0) {
      this.game.addExecution(...this.executor.spawnTribes(config.bots()));
    }
    this.game.addExecution(new WinCheckExecution());

    // Advance one tick so executions initialize; agent spawns via step().
    this.game.executeNextTick();
    return this.buildObs(null);
  }

  step(intents: Intent[], ticks: number): object {
    const stamped: StampedIntent[] = intents.map((i) => ({
      ...i,
      clientID: AGENT_CLIENT_ID,
    })) as StampedIntent[];
    if (stamped.length > 0) {
      for (const exec of this.executor.createExecs({
        turnNumber: this.game.ticks(),
        intents: stamped,
      })) {
        this.game.addExecution(exec);
      }
    }

    let winner: unknown = null;
    for (let t = 0; t < ticks; t++) {
      const updates = this.game.executeNextTick();
      const winUpdates = updates[GameUpdateType.Win];
      if (winUpdates && winUpdates.length > 0) {
        winner = (winUpdates[0] as { winner: unknown }).winner;
        break;
      }
    }
    return this.buildObs(winner);
  }

  private agent() {
    return this.game.playerByClientID(AGENT_CLIENT_ID) ?? null;
  }

  private buildObs(winner: unknown): object {
    const game = this.game;
    const numTiles = game.width() * game.height();
    const tiles = zlib
      .gzipSync(
        Buffer.from(
          game.tileStateBuffer().buffer,
          game.tileStateBuffer().byteOffset,
          numTiles * 2,
        ),
        { level: 1 },
      )
      .toString("base64");

    const agent = this.agent();
    const me = agent?.smallID() ?? -1;
    this.agentSmallID = me;

    // Terrain ships once per reset (it is immutable); Python caches it.
    return {
      tick: game.ticks(),
      width: game.width(),
      height: game.height(),
      spawnPhase: game.inSpawnPhase(),
      winner,
      me,
      alive: agent?.isAlive() ?? false,
      tiles,
      entities: this.entities(),
      legal: this.legality(),
    };
  }

  terrain(): object {
    const game = this.game;
    const numTiles = game.width() * game.height();
    const buf = new Uint8Array(numTiles);
    for (let ref = 0; ref < numTiles; ref++) buf[ref] = game.terrainByte(ref);
    return {
      terrain: zlib.gzipSync(Buffer.from(buf), { level: 6 }).toString("base64"),
    };
  }

  private entities(): object {
    const game = this.game;
    const players = game.players().map((p) => ({
      id: p.smallID(),
      pid: p.id(), // persistent ID; intents reference players by this
      type: p.type(),
      troops: Math.round(p.troops()),
      gold: p.gold().toString(),
      tiles: p.numTilesOwned(),
      alive: p.isAlive(),
      traitor: p.isTraitor(),
      embargoes: p.getEmbargoes().map((e) => e.target.smallID()),
      reqsIn: p.incomingAllianceRequests().map((r) => r.requestor().smallID()),
      reqsOut: p.outgoingAllianceRequests().map((r) => r.recipient().smallID()),
    }));

    const alliances: number[][] = [];
    const seen = new Set<string>();
    for (const p of game.players()) {
      for (const a of p.alliances()) {
        const x = a.requestor().smallID();
        const y = a.recipient().smallID();
        const key = x < y ? `${x}:${y}` : `${y}:${x}`;
        if (!seen.has(key)) {
          seen.add(key);
          alliances.push([x, y, a.expiresAt()]);
        }
      }
    }

    const units = game.players().flatMap((p) =>
      p.units().map((u) => {
        const tt = u.targetTile();
        return {
          uid: u.id(),
          type: u.type(),
          owner: p.smallID(),
          x: game.x(u.tile()),
          y: game.y(u.tile()),
          tx: tt !== undefined ? game.x(tt) : null,
          ty: tt !== undefined ? game.y(tt) : null,
          samLock: u.targetedBySAM(),
          level: u.level(),
          constructing: u.isUnderConstruction(),
          troops: Math.round(u.troops()),
        };
      }),
    );

    const attacks = game.players().flatMap((p) =>
      p.outgoingAttacks().map((a) => ({
        aid: a.id(),
        from: p.smallID(),
        to: a.target().isPlayer() ? a.target().smallID() : 0,
        troops: Math.round(a.troops()),
        retreating: a.retreating(),
      })),
    );

    return { players, alliances, units, attacks };
  }

  /** Exact per-action legality from engine calls; Python builds masks. */
  private legality(): object {
    const game = this.game;
    const agent = this.agent();
    if (!agent || !agent.isAlive()) {
      return { spawn: game.inSpawnPhase(), actions: {} };
    }
    const others = game.players().filter(
      (p) => p !== agent && p.isAlive(),
    );
    const gold = agent.gold();
    const buildable = [...STRUCTURES, ...LAUNCHABLE, UnitType.Warship].filter(
      (t) => gold >= game.unitInfo(t).cost(game, agent),
    );
    return {
      spawn: game.inSpawnPhase(),
      actions: {
        attackable: others
          .filter((p) => agent.sharesBorderWith(p) && !agent.isFriendly(p))
          .map((p) => p.smallID()),
        allianceRequestable: others
          .filter((p) => agent.canSendAllianceRequest(p))
          .map((p) => p.smallID()),
        allianceRejectable: agent
          .incomingAllianceRequests()
          .map((r) => r.requestor().smallID()),
        breakable: agent.alliances().map((a) => a.other(agent).smallID()),
        targetable: others
          .filter((p) => agent.canTarget(p))
          .map((p) => p.smallID()),
        donatableGold: others
          .filter((p) => agent.canDonateGold(p))
          .map((p) => p.smallID()),
        donatableTroops: others
          .filter((p) => agent.canDonateTroops(p))
          .map((p) => p.smallID()),
        embargoable: others
          .filter((p) => !agent.hasEmbargoAgainst(p))
          .map((p) => p.smallID()),
        buildableTypes: buildable,
        hasSilo: agent
          .units(UnitType.MissileSilo)
          .some((u) => !u.isUnderConstruction()),
        troops: Math.round(agent.troops()),
        gold: gold.toString(),
        attacks: agent.outgoingAttacks().map((a) => a.id()),
        boats: agent.units(UnitType.Transport).map((u) => u.id()),
        warships: agent.units(UnitType.Warship).map((u) => u.id()),
        upgradable: agent
          .units()
          .filter((u) => game.unitInfo(u.type()).upgradable)
          .map((u) => u.id()),
      },
    };
  }
}

async function main() {
  const session = new EnvSession();
  const rl = readline.createInterface({ input: process.stdin });
  const write = (obj: object) => process.stdout.write(JSON.stringify(obj) + "\n");

  for await (const line of rl) {
    if (!line.trim()) continue;
    let msg: {
      op: string;
      map?: string;
      seed?: string;
      intents?: Intent[];
      ticks?: number;
    };
    try {
      msg = JSON.parse(line);
    } catch {
      write({ error: "bad json" });
      continue;
    }
    try {
      if (msg.op === "reset") {
        const obs = await session.reset(msg.map ?? "Onion", msg.seed ?? "0");
        write({ ...obs, ...session.terrain() });
      } else if (msg.op === "step") {
        write(session.step(msg.intents ?? [], msg.ticks ?? 10));
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
