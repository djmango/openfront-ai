/**
 * Generate a synthetic bot/nation-only GameRecord for the native-vs-TS
 * bot-AI parity investigation: zero human players, so the entire
 * trajectory is driven by autonomous bot/nation AI decisions (no human
 * intents to replay) - `nextTick()` on both engines re-derives the same
 * AI decisions from identical starting state, seed, and PRNG streams (in
 * theory).
 *
 * Because bots/nations act from their own per-tick AI logic rather than
 * recorded intents, the GameRecord itself needs no `turns` payload at all:
 * `info.num_turns` alone is enough for both engines' `decompressGameRecord`
 * to pad in `--max-ticks` all-empty turns (see openfront/src/core/Util.ts
 * and rust/engine/src/record.rs). `gameType: "Public"` (not Singleplayer)
 * mirrors real archived games so `SpawnTimerExecution` ends the spawn
 * phase on a timer instead of waiting for a human to place a spawn.
 *
 * Usage (from openfront-ai/):
 *   openfront/node_modules/.bin/tsx scripts/gen_selfplay_record.ts \
 *     --map BlackSea --bots 50 --nations default --difficulty Medium \
 *     --seed bot-ai-parity-1 --max-ticks 6000 --out records/selfplay/bs50.json.gz
 */
import * as fs from "fs";
import * as path from "path";
import * as zlib from "zlib";
import { Difficulty, GameMapSize, GameMapType, GameMode, GameType } from "../openfront/src/core/game/Game";
import { GameConfig } from "../openfront/src/core/Schemas";
import { simpleHash } from "../openfront/src/core/Util";

function seedToGameID(seed: string): string {
  // Same scheme as bridge/env.ts's seedToGameID: deterministic 8-char
  // alnum id (GAME_ID_REGEX) from an arbitrary seed string.
  const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  let h = simpleHash(`selfplay-${seed}`);
  let out = "";
  for (let i = 0; i < 8; i++) {
    h = (h * 1103515245 + 12345) & 0x7fffffff;
    out += alphabet[h % alphabet.length];
  }
  return out;
}

function main() {
  const args = process.argv.slice(2);
  const getArg = (name: string, fallback: string): string => {
    const i = args.indexOf(`--${name}`);
    return i >= 0 && args[i + 1] !== undefined ? args[i + 1] : fallback;
  };

  const mapKey = getArg("map", "BlackSea");
  const bots = parseInt(getArg("bots", "50"), 10);
  const nationsArg = getArg("nations", "default");
  const nations: number | "default" | "disabled" =
    nationsArg === "default" || nationsArg === "disabled" ? nationsArg : parseInt(nationsArg, 10);
  const difficulty = getArg("difficulty", "Medium");
  const seed = getArg("seed", "bot-ai-parity-1");
  const maxTicks = parseInt(getArg("max-ticks", "6000"), 10);
  const outPath = getArg("out", `records/selfplay/${mapKey.toLowerCase()}-${bots}bots.json.gz`);

  const mapType = GameMapType[mapKey as keyof typeof GameMapType];
  if (!mapType) {
    throw new Error(`Unknown map key "${mapKey}". Valid keys: ${Object.keys(GameMapType).join(", ")}`);
  }
  const diff = Difficulty[difficulty as keyof typeof Difficulty];
  if (!diff) {
    throw new Error(`Unknown difficulty "${difficulty}"`);
  }

  const gameID = seedToGameID(seed);
  const config: GameConfig = {
    gameMap: mapType,
    gameMapSize: GameMapSize.Normal,
    gameMode: GameMode.FFA,
    gameType: GameType.Public,
    difficulty: diff,
    nations,
    donateGold: false,
    donateTroops: false,
    bots,
    infiniteGold: false,
    infiniteTroops: false,
    instantBuild: false,
    randomSpawn: false,
  };

  const record = {
    info: {
      gameID,
      lobbyCreatedAt: 0,
      config,
      players: [] as unknown[],
      num_turns: maxTicks,
      winner: undefined,
    },
    version: "v0.0.2",
    gitCommit: "DEV",
    subdomain: "bot-ai-parity",
    domain: "localhost",
    turns: [] as unknown[],
  };

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, zlib.gzipSync(Buffer.from(JSON.stringify(record)), { level: 6 }));
  console.log(
    `wrote ${outPath}: gameID=${gameID} map=${mapKey} bots=${bots} nations=${nationsArg} ` +
      `difficulty=${difficulty} maxTicks=${maxTicks} seed=${seed}`,
  );
}

main();
