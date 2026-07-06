/**
 * Headless OpenFront game runner that plays bot/nation-only games and dumps
 * tile-state snapshots for autoencoder training.
 *
 * Usage (from openfront-ae/):
 *   openfront/node_modules/.bin/tsx datagen/generate.ts --map Onion --games 2
 *
 * Output layout, per game:
 *   data/<map>/<gameID>/
 *     terrain.bin       uint8[w*h]  immutable terrain bytes (land/ocean/shore/magnitude)
 *     states/t<tick>.bin.gz  gzipped uint16-le[w*h] per snapshot (owner id bits
 *                       0-11, fallout bit 13, defense bonus bit 14)
 *     meta.json         dims, snapshot ticks, per-snapshot player stats, winner
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
import { GameMapLoader, MapData } from "../openfront/src/core/game/GameMapLoader";
import { GameUpdateType } from "../openfront/src/core/game/GameUpdates";
import { createNationsForGame } from "../openfront/src/core/game/NationCreation";
import {
  genTerrainFromBin,
  MapManifest,
} from "../openfront/src/core/game/TerrainMapLoader";
import { Config } from "../openfront/src/core/configuration/Config";
import { DoomsdayClockExecution } from "../openfront/src/core/execution/DoomsdayClockExecution";
import { Executor } from "../openfront/src/core/execution/ExecutionManager";
import { RecomputeRailClusterExecution } from "../openfront/src/core/execution/RecomputeRailClusterExecution";
import { SpawnTimerExecution } from "../openfront/src/core/execution/SpawnTimerExecution";
import { WinCheckExecution } from "../openfront/src/core/execution/WinCheckExecution";
import { PseudoRandom } from "../openfront/src/core/PseudoRandom";
import { GameConfig, GameStartInfo } from "../openfront/src/core/Schemas";
import { simpleHash } from "../openfront/src/core/Util";

const REPO_ROOT = path.join(__dirname, "..");
const MAPS_DIR = path.join(REPO_ROOT, "openfront", "resources", "maps");

class NodeMapLoader implements GameMapLoader {
  getMapData(map: GameMapType): MapData {
    const key = Object.keys(GameMapType).find(
      (k) => GameMapType[k as keyof typeof GameMapType] === map,
    );
    if (!key) throw new Error(`Unknown map: ${map}`);
    const dir = path.join(MAPS_DIR, key.toLowerCase());
    return {
      mapBin: async () => new Uint8Array(fs.readFileSync(path.join(dir, "map.bin"))),
      map4xBin: async () =>
        new Uint8Array(fs.readFileSync(path.join(dir, "map4x.bin"))),
      map16xBin: async () =>
        new Uint8Array(fs.readFileSync(path.join(dir, "map16x.bin"))),
      manifest: async () =>
        JSON.parse(
          fs.readFileSync(path.join(dir, "manifest.json"), "utf8"),
        ) as MapManifest,
      webpPath: "",
    };
  }
}

interface PlayerSnapshot {
  smallID: number;
  name: string;
  type: string;
  troops: number;
  gold: string;
  tiles: number;
  alive: boolean;
}

interface Snapshot {
  tick: number;
  players: PlayerSnapshot[];
}

function snapshotPlayers(game: Game): PlayerSnapshot[] {
  return game.players().map((p) => ({
    smallID: p.smallID(),
    name: p.name(),
    type: p.type(),
    troops: Math.round(p.troops()),
    gold: p.gold().toString(),
    tiles: p.numTilesOwned(),
    alive: p.isAlive(),
  }));
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
  };

  const gameStart: GameStartInfo = {
    gameID,
    lobbyCreatedAt: Date.now(),
    config: gameConfig,
    players: [],
  };

  const config = new Config(gameConfig, null, false);
  // Deliberately avoid loadTerrainMap(): it caches the mutable GameMap object
  // across games, so a second game would inherit the first game's ownership
  // state. Generating from the binary gives each game a fresh map.
  const loader = new NodeMapLoader().getMapData(mapType);
  const manifest = await loader.manifest();
  const terrain = {
    nations: manifest.nations,
    additionalNations: manifest.additionalNations ?? [],
    gameMap: await genTerrainFromBin(manifest.map, await loader.mapBin()),
    miniGameMap: await genTerrainFromBin(
      manifest.map4x,
      await loader.map4xBin(),
    ),
    teamGameSpawnAreas: manifest.teamGameSpawnAreas,
  };
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
      const raw = Buffer.from(
        game.tileStateBuffer().buffer,
        game.tileStateBuffer().byteOffset,
        numTiles * 2,
      );
      fs.writeFileSync(
        path.join(statesDir, `t${String(game.ticks()).padStart(6, "0")}.bin.gz`),
        zlib.gzipSync(raw, { level: 6 }),
      );
      snapshots.push({ tick: game.ticks(), players: snapshotPlayers(game) });
    }

    if (winner !== null) break;
  }

  const elapsedS = (Date.now() - startedAt) / 1000;
  const meta = {
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
  const snapshotEvery = parseInt(getArg("every", "25"), 10);
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
