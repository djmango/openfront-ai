/**
 * Headless RL env session - shared by bridge/env.ts and bridge/engine_daemon.ts.
 */
import * as fs from "fs";
import * as path from "path";
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
import type { MapManifest } from "../openfront/src/core/game/TerrainMapFileLoader";
import { buildObsParts, terrainPayload } from "./common";

export const AGENT_CLIENT_ID = "AGENTRL1";

export function repoRoot(): string {
  return path.resolve(__dirname, "..");
}

export function mapsDir(): string {
  return path.join(repoRoot(), "openfront", "resources", "maps");
}

function mapDirName(mapType: GameMapType): string {
  const key = Object.keys(GameMapType).find(
    (k) => GameMapType[k as keyof typeof GameMapType] === mapType,
  )!;
  return key.toLowerCase();
}

export function seedToGameID(seed: string): string {
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

export class EnvSession {
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

    const dir = path.join(mapsDir(), mapDirName(mapType));
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

    this.game.executeNextTick();
    return buildObsParts(this.game, AGENT_CLIENT_ID, null);
  }

  step(
    intents: Intent[],
    ticks: number,
  ): { head: Record<string, unknown>; tiles: Buffer } {
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
    return buildObsParts(this.game, AGENT_CLIENT_ID, winner);
  }

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
