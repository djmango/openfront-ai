/**
 * Generates a small, curriculum-representative set of self-play GameRecord
 * files - bots + AI nations only, zero human players - at the RL
 * curriculum's actual bot counts (see rust/ofcore/src/curriculum.rs
 * `stages()`), for the outcome-parity gate machinery in
 * rust/engine/src/replay.rs / rust/engine/src/bin/outcome_gate.rs.
 *
 * Motivation: the existing 78-record archive (records/0c4c7d7993c9/) that
 * outcome_gate normally runs against is entirely 400-bot/125-human
 * mega-games - an extreme edge case far outside anything the RL curriculum
 * actually trains on (which tops out at 150 bots and never has more than a
 * handful of human/nation opponents in early stages). This script produces
 * a few games per curriculum bot-count bucket (0, 5, 10, 30, 50, 80, 120,
 * 150) so parity can be checked in the regime that actually matters for RL
 * training validity.
 *
 * Config quirk this script works around: a Singleplayer-type game (as used
 * by rust/oftrain's RlSession::reset / bridge/env.ts) only ends its spawn
 * phase when a HUMAN player spawns - see `SpawnExecution.tick()`
 * (TS) / `spawn.rs` (native), both of which gate `endSpawnPhase()` on
 * `playerType === Human`. With zero human players that never fires, and
 * since real bot/nation AI behavior (PlayerExecution, TribeExecution,
 * WinCheckExecution, and the "already spawned" branch of NationExecution)
 * is gated on `!inSpawnPhase()`, the game would sit frozen in a
 * perpetual-respawn loop forever (verified empirically - see the
 * curriculum-parity devlog). Using gameType "Public" instead (the type
 * real archived multiplayer games use) adds a `SpawnTimerExecution` that
 * force-ends the spawn phase after `numSpawnPhaseTurns()` ticks
 * regardless of players, which is what actually lets bots/nations play.
 * `maxTimerValue` is also set so a terminal condition (max_timer) is
 * guaranteed within the tick budget even if no one reaches the land-share
 * win threshold naturally.
 *
 * Each record has zero recorded human intents (turns are entirely empty -
 * `info.num_turns` alone drives GameRecord.decompress()'s zero-padding on
 * both the TS and native side), since bots/nations act autonomously from
 * engine-internal AI, not from replayed intents. This makes generation and
 * replay the same operation: there is nothing to "record" beyond the wire
 * config, so the record is produced directly (no separate live driving
 * pass needed to capture intents).
 *
 * As an extra cross-check beyond the standard `datagen/replay.ts
 * --outcome-oracle` pipeline (run separately, see
 * scripts/run_curriculum_parity_gate.sh), this script also calls the same
 * `replayOutcome()` right after writing each record and saves the result
 * to manifest.json, so the "ground truth" TS outcome is captured
 * immediately at generation time rather than purely trusting a later
 * re-replay.
 *
 * Usage (from openfront-ai/):
 *   openfront/node_modules/.bin/tsx datagen/gen_curriculum_parity.ts \
 *     --out records/curriculum-parity-v1 [--games-per-bucket 5] \
 *     [--ticks 4500] [--max-timer 6] [--buckets 0,5,10]
 */
import * as fs from "fs";
import * as path from "path";
import * as zlib from "zlib";
import { Difficulty, GameMapType, GameMapSize, GameMode, GameType } from "../openfront/src/core/game/Game";
import { GameConfig, GameRecord, GameStartInfo } from "../openfront/src/core/Schemas";
import { decompressGameRecord } from "../openfront/src/core/Util";
import { replayOutcome, type GameOutcome } from "./replay";

const REPO_ROOT = path.join(__dirname, "..");

interface Bucket {
  bots: number;
  nations: number | "default";
  difficulty: Difficulty;
  // Curriculum map keys (rust/ofcore/src/curriculum.rs ALL_MAPS / per-stage
  // maps lists use the enum KEY, e.g. "BetweenTwoSeas"; resolved to the
  // actual GameMapType string value below).
  mapKeys: string[];
  stageLabel: string;
}

// One representative curriculum stage per distinct bot count. bots=30 and
// bots=80 each appear in two stages (Easy/Medium and Medium/Hard
// respectively); we use the first (lower-difficulty) occurrence for both -
// see rust/ofcore/src/curriculum.rs `stages()`.
const BUCKETS: Bucket[] = [
  {
    bots: 0,
    nations: 1,
    difficulty: Difficulty.Easy,
    mapKeys: ["Onion"],
    stageLabel: "stage0 (bots=0, nations=1, Easy)",
  },
  {
    bots: 5,
    nations: 3,
    difficulty: Difficulty.Easy,
    mapKeys: ["Onion", "Pangaea"],
    stageLabel: "stage2 (bots=5, nations=3, Easy)",
  },
  {
    bots: 10,
    nations: 6,
    difficulty: Difficulty.Easy,
    mapKeys: ["Pangaea", "Caucasus"],
    stageLabel: "stage3 (bots=10, nations=6, Easy)",
  },
  {
    bots: 30,
    nations: "default",
    difficulty: Difficulty.Easy,
    mapKeys: ["Pangaea", "Caucasus", "BlackSea"],
    stageLabel: "stage4 (bots=30, nations=default, Easy)",
  },
  {
    bots: 50,
    nations: "default",
    difficulty: Difficulty.Medium,
    mapKeys: ["World", "Asia", "BlackSea"],
    stageLabel: "stage6 (bots=50, nations=default, Medium)",
  },
  {
    bots: 80,
    nations: "default",
    difficulty: Difficulty.Medium,
    mapKeys: ["World", "Asia", "BetweenTwoSeas", "Caucasus"],
    stageLabel: "stage7 (bots=80, nations=default, Medium)",
  },
  {
    bots: 120,
    nations: "default",
    difficulty: Difficulty.Hard,
    mapKeys: ["Onion", "Pangaea", "Caucasus", "BlackSea", "BetweenTwoSeas", "World", "Asia"],
    stageLabel: "stage9 (bots=120, nations=default, Hard)",
  },
  {
    bots: 150,
    nations: "default",
    difficulty: Difficulty.Impossible,
    mapKeys: ["Onion", "Pangaea", "Caucasus", "BlackSea", "BetweenTwoSeas", "World", "Asia"],
    stageLabel: "stage10 (bots=150, nations=default, Impossible)",
  },
];

interface ManifestEntry {
  gameId: string;
  file: string;
  bots: number;
  nations: number | "default";
  difficulty: string;
  map: string;
  mapKey: string;
  stageLabel: string;
  seedIndex: number;
  numTurns: number;
  generationOutcome: GameOutcome;
}

function resolveMap(key: string): GameMapType {
  const value = (GameMapType as unknown as Record<string, GameMapType>)[key];
  if (!value) {
    throw new Error(`Unknown curriculum map key "${key}"`);
  }
  return value;
}

function buildConfig(
  mapKey: string,
  bots: number,
  nations: number | "default",
  difficulty: Difficulty,
  maxTimerMinutes: number,
): GameConfig {
  return {
    gameMap: resolveMap(mapKey),
    gameMapSize: GameMapSize.Normal,
    gameMode: GameMode.FFA,
    // "Public" (not "Singleplayer") so a SpawnTimerExecution is added and
    // the spawn phase force-ends after numSpawnPhaseTurns() ticks - see
    // the module doc comment above for why this matters with 0 humans.
    gameType: GameType.Public,
    difficulty,
    nations,
    donateGold: true,
    donateTroops: true,
    bots,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    randomSpawn: false,
    maxTimerValue: maxTimerMinutes,
  } as GameConfig;
}

function buildRecord(
  gameId: string,
  config: GameConfig,
  numTurns: number,
): GameRecord {
  const info = {
    gameID: gameId,
    lobbyCreatedAt: Date.now(),
    config,
    players: [],
    num_turns: numTurns,
    winner: undefined,
  } as unknown as GameStartInfo & { num_turns: number };
  return {
    info,
    version: "curriculum-parity-v1",
    turns: [],
    gitCommit: "DEV",
  } as unknown as GameRecord;
}

function writeRecord(outDir: string, gameId: string, record: GameRecord): string {
  const file = path.join(outDir, `${gameId}.json.gz`);
  fs.writeFileSync(file, zlib.gzipSync(JSON.stringify(record), { level: 6 }));
  return file;
}

async function main() {
  const args = process.argv.slice(2);
  const getArg = (name: string, fallback: string): string => {
    const i = args.indexOf(`--${name}`);
    return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : fallback;
  };

  const outDir = path.resolve(REPO_ROOT, getArg("out", "records/curriculum-parity-v1"));
  const gamesPerBucket = parseInt(getArg("games-per-bucket", "5"), 10);
  const numTurns = parseInt(getArg("ticks", "4500"), 10);
  const maxTimerMinutes = parseInt(getArg("max-timer", "6"), 10);
  const bucketFilter = getArg("buckets", "");
  const wantedBots = bucketFilter
    ? new Set(bucketFilter.split(",").map((s) => parseInt(s.trim(), 10)))
    : null;

  fs.mkdirSync(outDir, { recursive: true });
  const manifest: ManifestEntry[] = [];

  for (const bucket of BUCKETS) {
    if (wantedBots && !wantedBots.has(bucket.bots)) continue;
    console.log(`=== bucket bots=${bucket.bots} (${bucket.stageLabel}) ===`);
    for (let g = 0; g < gamesPerBucket; g++) {
      const mapKey = bucket.mapKeys[g % bucket.mapKeys.length];
      const gameId = `curr-b${String(bucket.bots).padStart(3, "0")}-s${g}-${mapKey.toLowerCase()}`;
      const config = buildConfig(mapKey, bucket.bots, bucket.nations, bucket.difficulty, maxTimerMinutes);
      // Sparse on disk (turns=[], num_turns=N) - both the native `decompress()`
      // and TS `decompressGameRecord()` zero-pad this to N empty-intent turns,
      // so it round-trips identically through the file-based oracle/native
      // replay paths. Write it to disk BEFORE decompressing in-memory for the
      // inline replayOutcome() check below, since decompressGameRecord()
      // mutates `record.turns` in place.
      const record = buildRecord(gameId, config, numTurns);
      const file = writeRecord(outDir, gameId, record);

      const started = Date.now();
      const outcome = await replayOutcome(decompressGameRecord(record as never));
      const secs = ((Date.now() - started) / 1000).toFixed(1);
      console.log(
        `[${gameId}] map=${mapKey} winner=${JSON.stringify(outcome.winner)} ` +
          `terminalTick=${outcome.terminalTick} reason=${outcome.terminalReason} ` +
          `landShare=${outcome.winnerLandShare?.toFixed(3)} (${secs}s)`,
      );
      manifest.push({
        gameId,
        file: path.relative(REPO_ROOT, file),
        bots: bucket.bots,
        nations: bucket.nations,
        difficulty: bucket.difficulty,
        map: config.gameMap as string,
        mapKey,
        stageLabel: bucket.stageLabel,
        seedIndex: g,
        numTurns,
        generationOutcome: outcome,
      });
    }
  }

  // Sibling to outDir, NOT inside it: datagen/replay.ts's --outcome-oracle
  // (and the plain replay CLI) glob every *.json/*.json.gz file under the
  // records dir as a game record, so a manifest.json living inside would
  // get mistaken for one and crash decompressGameRecord() on it.
  const manifestPath = `${outDir}.manifest.json`;
  fs.writeFileSync(manifestPath, JSON.stringify({ generatedAt: new Date().toISOString(), numTurns, maxTimerMinutes, records: manifest }, null, 2));
  console.log(`\nwrote ${manifest.length} records -> ${outDir}`);
  console.log(`wrote manifest -> ${manifestPath}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
