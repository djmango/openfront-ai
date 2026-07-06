/**
 * Headless OpenFront game runner that plays bot/nation-only games and dumps
 * tile-state snapshots for autoencoder training.
 *
 * Usage (from openfront-ai/):
 *   openfront/node_modules/.bin/tsx datagen/generate.ts --map Onion --games 2
 *
 * Output layout, per game (format v2):
 *   data/<map>/<gameID>/
 *     terrain.bin       uint8[w*h]  immutable terrain bytes (land/ocean/shore/magnitude)
 *     states/t<tick>.bin.gz   gzipped uint16-le[w*h] per snapshot (owner id bits
 *                       0-11, fallout bit 13, defense bonus bit 14)
 *     states/t<tick>.json.gz  gzipped JSON: full entity state per snapshot -
 *                       players (stats, diplomacy, relations), alliances,
 *                       units, attacks in flight
 *     meta.json         dims, snapshot tick list, winner
 */
import * as fs from "fs";
import * as path from "path";
import * as zlib from "zlib";
import {
  Difficulty,
  Game,
  GameMapSize,
  GameMapType,
  GameMode,
  GameType,
  UnitType,
} from "../openfront/src/core/game/Game";
import { createGame } from "../openfront/src/core/game/GameImpl";
import { GameUpdateType } from "../openfront/src/core/game/GameUpdates";
import { createNationsForGame } from "../openfront/src/core/game/NationCreation";
import { Config } from "../openfront/src/core/configuration/Config";
import { DoomsdayClockExecution } from "../openfront/src/core/execution/DoomsdayClockExecution";
import { Executor } from "../openfront/src/core/execution/ExecutionManager";
import { RecomputeRailClusterExecution } from "../openfront/src/core/execution/RecomputeRailClusterExecution";
import { SpawnTimerExecution } from "../openfront/src/core/execution/SpawnTimerExecution";
import { WinCheckExecution } from "../openfront/src/core/execution/WinCheckExecution";
import { PseudoRandom } from "../openfront/src/core/PseudoRandom";
import { GameConfig, GameStartInfo } from "../openfront/src/core/Schemas";
import { simpleHash } from "../openfront/src/core/Util";
import { loadFreshTerrain, snapshotEntities } from "./common";

const REPO_ROOT = path.join(__dirname, "..");

interface Snapshot {
  tick: number;
}

async function runGame(opts: {
  mapType: GameMapType;
  gameID: string;
  outDir: string;
  snapshotEvery: number;
  maxTicks: number;
}): Promise<void> {
  const { mapType, gameID, outDir, snapshotEvery, maxTicks } = opts;

  const gameConfig: GameConfig = {
    gameMap: mapType,
    gameMapSize: GameMapSize.Normal,
    gameMode: GameMode.FFA,
    gameType: GameType.Singleplayer,
    difficulty: Difficulty.Medium,
    nations: "default",
    donateGold: false,
    donateTroops: false,
    bots: 100,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    randomSpawn: false,
  };

  const gameStart: GameStartInfo = {
    gameID,
    lobbyCreatedAt: Date.now(),
    config: gameConfig,
    players: [],
  };

  const config = new Config(gameConfig, null, false);
  const terrain = await loadFreshTerrain(mapType, GameMapSize.Normal);
  const random = new PseudoRandom(simpleHash(gameID));
  const nations = createNationsForGame(
    gameStart,
    terrain.nations,
    terrain.additionalNations,
    0,
    random,
  );

  const game: Game = createGame(
    [],
    nations,
    terrain.gameMap,
    terrain.miniGameMap,
    config,
    terrain.teamGameSpawnAreas,
  );

  const executor = new Executor(game, gameID, undefined);
  game.addExecution(new SpawnTimerExecution());
  if (config.spawnNations()) {
    game.addExecution(...executor.nationExecutions());
  }
  if (config.bots() > 0) {
    game.addExecution(...executor.spawnTribes(config.bots()));
  }
  game.addExecution(new WinCheckExecution());
  if (config.doomsdayClockConfig().enabled) {
    game.addExecution(new DoomsdayClockExecution());
  }
  if (!config.isUnitDisabled(UnitType.Factory)) {
    game.addExecution(new RecomputeRailClusterExecution(game.railNetwork()));
  }

  fs.mkdirSync(outDir, { recursive: true });

  const w = game.width();
  const h = game.height();
  const numTiles = w * h;

  // Terrain is immutable; dump once.
  const terrainBuf = new Uint8Array(numTiles);
  for (let ref = 0; ref < numTiles; ref++) {
    terrainBuf[ref] = game.terrainByte(ref);
  }
  fs.writeFileSync(path.join(outDir, "terrain.bin"), terrainBuf);

  const statesDir = path.join(outDir, "states");
  fs.mkdirSync(statesDir, { recursive: true });
  const snapshots: Snapshot[] = [];
  let winner: unknown = null;
  const startedAt = Date.now();

  while (game.ticks() < maxTicks) {
    const updates = game.executeNextTick();

    const winUpdates = updates[GameUpdateType.Win];
    if (winUpdates && winUpdates.length > 0) {
      winner = (winUpdates[0] as { winner: unknown }).winner;
    }

    const pastSpawn = !game.inSpawnPhase();
    if (pastSpawn && game.ticks() % snapshotEvery === 0) {
      const stem = `t${String(game.ticks()).padStart(6, "0")}`;
      const raw = Buffer.from(
        game.tileStateBuffer().buffer,
        game.tileStateBuffer().byteOffset,
        numTiles * 2,
      );
      fs.writeFileSync(
        path.join(statesDir, `${stem}.bin.gz`),
        zlib.gzipSync(raw, { level: 6 }),
      );
      fs.writeFileSync(
        path.join(statesDir, `${stem}.json.gz`),
        zlib.gzipSync(JSON.stringify(snapshotEntities(game)), { level: 6 }),
      );
      snapshots.push({ tick: game.ticks() });
    }

    if (winner !== null) break;
  }

  const elapsedS = (Date.now() - startedAt) / 1000;
  const meta = {
    formatVersion: 3,
    gameID,
    map: mapType,
    width: w,
    height: h,
    snapshotEvery,
    finalTick: game.ticks(),
    winner,
    dtypeStates: "uint16-le",
    dtypeTerrain: "uint8",
    snapshots,
  };
  fs.writeFileSync(path.join(outDir, "meta.json"), JSON.stringify(meta));
  console.log(
    `[${gameID}] done: ${game.ticks()} ticks, ${snapshots.length} snapshots, ` +
      `${elapsedS.toFixed(1)}s (${(game.ticks() / elapsedS).toFixed(0)} ticks/s), ` +
      `winner=${JSON.stringify(winner)}`,
  );
}

async function main() {
  const args = process.argv.slice(2);
  const getArg = (name: string, fallback: string): string => {
    const i = args.indexOf(`--${name}`);
    return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : fallback;
  };

  const mapKey = getArg("map", "Onion");
  const numGames = parseInt(getArg("games", "1"), 10);
  // 10 ticks = 1s of game time; a typical nuke flight (20-60 ticks at speed
  // 10) now spans 2-6 snapshots instead of 0-2.
  const snapshotEvery = parseInt(getArg("every", "10"), 10);
  const maxTicks = parseInt(getArg("max-ticks", "15000"), 10);
  const seedBase = getArg("seed", `${Date.now() % 100000}`);

  const mapType = GameMapType[mapKey as keyof typeof GameMapType];
  if (!mapType) {
    throw new Error(
      `Unknown map key "${mapKey}". Valid keys: ${Object.keys(GameMapType).join(", ")}`,
    );
  }

  for (let g = 0; g < numGames; g++) {
    const gameID = `${mapKey.toLowerCase()}-${seedBase}-${g}`;
    const outDir = path.join(REPO_ROOT, "data", mapKey.toLowerCase(), gameID);
    await runGame({ mapType, gameID, outDir, snapshotEvery, maxTicks });
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
