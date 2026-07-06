/**
 * Replay archived human games (downloaded via scripts/fetch_games.py) through
 * the headless engine and dump snapshots in the same format as generate.ts.
 *
 * The engine is deterministic: feeding the recorded per-turn intents back
 * through the Executor regenerates the full state trajectory. Integrity is
 * verified against the state hashes embedded in the record (clients report
 * game.hash() every 10 ticks during live play); any mismatch marks the game
 * desynced and it is skipped.
 *
 * Usage (from openfront-ai/):
 *   openfront/node_modules/.bin/tsx datagen/replay.ts \
 *     --records records --out data-human [--every 10] [--limit N]
 *
 * Output layout per game: data-human/<map>/<gameID>/ with terrain.bin,
 * states/t<tick>.{bin,json}.gz and meta.json (formatVersion 3, plus
 * source/gitCommit/hash-verification fields).
 */
import * as fs from "fs";
import * as path from "path";
import * as zlib from "zlib";
import {
  Game,
  GameMapType,
  GameType,
  PlayerInfo,
  PlayerType,
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
import { GameRecord, GameStartInfo } from "../openfront/src/core/Schemas";
import { decompressGameRecord, simpleHash } from "../openfront/src/core/Util";
import { loadFreshTerrain, snapshotEntities } from "./common";

interface ReplayResult {
  ok: boolean;
  reason?: string;
  ticks: number;
  snapshots: number;
  hashesChecked: number;
}

async function replayGame(
  record: GameRecord,
  outDir: string,
  snapshotEvery: number,
): Promise<ReplayResult> {
  const info: GameStartInfo = record.info;
  const gameConfig = info.config;

  const mapType = gameConfig.gameMap as GameMapType;
  if (!Object.values(GameMapType).includes(mapType)) {
    return {
      ok: false,
      reason: `unknown map ${gameConfig.gameMap}`,
      ticks: 0,
      snapshots: 0,
      hashesChecked: 0,
    };
  }

  const config = new Config(gameConfig, null, false);
  const terrain = await loadFreshTerrain(mapType, gameConfig.gameMapSize);
  const random = new PseudoRandom(simpleHash(info.gameID));

  // Mirrors createGameRunner(): humans first (consuming random.nextID() in
  // player order), then nations, so IDs match the live game exactly.
  const humans = info.players.map(
    (p) =>
      new PlayerInfo(
        p.username,
        PlayerType.Human,
        p.clientID,
        random.nextID(),
        p.isLobbyCreator ?? false,
        p.clanTag,
        p.friends ?? [],
      ),
  );
  const nations = createNationsForGame(
    info,
    terrain.nations,
    terrain.additionalNations,
    humans.length,
    random,
  );

  const game: Game = createGame(
    humans,
    nations,
    terrain.gameMap,
    terrain.miniGameMap,
    config,
    terrain.teamGameSpawnAreas,
  );
  const executor = new Executor(game, info.gameID, undefined);

  // Mirrors GameRunner.init().
  if (gameConfig.gameType !== GameType.Singleplayer) {
    game.addExecution(new SpawnTimerExecution());
  }
  if (config.spawnNations()) {
    game.addExecution(...executor.nationExecutions());
  }
  if (config.isRandomSpawn()) {
    game.addExecution(...executor.spawnPlayers());
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

  const terrainBuf = new Uint8Array(numTiles);
  for (let ref = 0; ref < numTiles; ref++) {
    terrainBuf[ref] = game.terrainByte(ref);
  }
  fs.writeFileSync(path.join(outDir, "terrain.bin"), terrainBuf);

  const statesDir = path.join(outDir, "states");
  fs.mkdirSync(statesDir, { recursive: true });
  const snapshots: { tick: number }[] = [];
  let winner: unknown = null;
  let hashesChecked = 0;

  for (const turn of record.turns) {
    game.addExecution(...executor.createExecs(turn));
    const updates = game.executeNextTick();

    // Verify state hashes where the record has them. The engine emits a
    // Hash update every 10 ticks; live clients reported the same value and
    // the server stored it on the matching turn.
    const hashUpdates = updates[GameUpdateType.Hash] as
      | { tick: number; hash: number }[]
      | undefined;
    for (const hu of hashUpdates ?? []) {
      const archived = record.turns[hu.tick]?.hash;
      if (archived === null || archived === undefined) continue;
      hashesChecked++;
      if (archived !== hu.hash) {
        return {
          ok: false,
          reason: `desync at tick ${hu.tick} (ours ${hu.hash} != archived ${archived})`,
          ticks: game.ticks(),
          snapshots: snapshots.length,
          hashesChecked,
        };
      }
    }

    const winUpdates = updates[GameUpdateType.Win];
    if (winUpdates && winUpdates.length > 0) {
      winner = (winUpdates[0] as { winner: unknown }).winner;
    }

    if (!game.inSpawnPhase() && game.ticks() % snapshotEvery === 0) {
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
  }

  const meta = {
    formatVersion: 3,
    source: "human-replay",
    gitCommit: record.gitCommit,
    gameID: info.gameID,
    map: mapType,
    mapSize: gameConfig.gameMapSize,
    gameMode: gameConfig.gameMode,
    numHumans: humans.length,
    width: w,
    height: h,
    snapshotEvery,
    finalTick: game.ticks(),
    winner: winner ?? (record.info as { winner?: unknown }).winner ?? null,
    hashesChecked,
    dtypeStates: "uint16-le",
    dtypeTerrain: "uint8",
    snapshots,
  };
  fs.writeFileSync(path.join(outDir, "meta.json"), JSON.stringify(meta));
  return {
    ok: true,
    ticks: game.ticks(),
    snapshots: snapshots.length,
    hashesChecked,
  };
}

function loadRecord(file: string): GameRecord {
  const buf = fs.readFileSync(file);
  const json = file.endsWith(".gz")
    ? zlib.gunzipSync(buf).toString("utf8")
    : buf.toString("utf8");
  // Archived turns are sparse (empty turns dropped); re-expand so index in
  // the array == turn number == tick.
  return decompressGameRecord(JSON.parse(json) as GameRecord);
}

async function main() {
  const args = process.argv.slice(2);
  const getArg = (name: string, fallback: string): string => {
    const i = args.indexOf(`--${name}`);
    return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : fallback;
  };

  const recordsDir = getArg("records", "records");
  const outRoot = getArg("out", "data-human");
  const snapshotEvery = parseInt(getArg("every", "10"), 10);
  const limit = parseInt(getArg("limit", "0"), 10);

  const files = fs
    .readdirSync(recordsDir, { recursive: true })
    .map(String)
    .filter((f) => f.endsWith(".json") || f.endsWith(".json.gz"))
    .map((f) => path.join(recordsDir, f))
    .sort();
  const todo = limit > 0 ? files.slice(0, limit) : files;

  let ok = 0;
  let failed = 0;
  for (const file of todo) {
    let record: GameRecord;
    try {
      record = loadRecord(file);
    } catch (e) {
      console.error(`[${path.basename(file)}] unreadable: ${e}`);
      failed++;
      continue;
    }
    const gameID = record.info.gameID;
    const mapKey = String(record.info.config.gameMap)
      .toLowerCase()
      .replace(/\s+/g, "");
    const outDir = path.join(outRoot, mapKey, gameID);
    if (fs.existsSync(path.join(outDir, "meta.json"))) {
      continue; // already replayed
    }
    const started = Date.now();
    try {
      const res = await replayGame(record, outDir, snapshotEvery);
      const secs = ((Date.now() - started) / 1000).toFixed(1);
      if (res.ok) {
        ok++;
        console.log(
          `[${gameID}] ok: ${res.ticks} ticks, ${res.snapshots} snapshots, ` +
            `${res.hashesChecked} hashes verified, ${secs}s`,
        );
      } else {
        failed++;
        fs.rmSync(outDir, { recursive: true, force: true });
        console.error(`[${gameID}] FAILED: ${res.reason} (${secs}s)`);
      }
    } catch (e) {
      failed++;
      fs.rmSync(outDir, { recursive: true, force: true });
      console.error(`[${gameID}] crashed: ${e}`);
    }
  }
  console.log(`replay done: ${ok} ok, ${failed} failed, ${todo.length} total`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
