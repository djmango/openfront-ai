/**
 * Seed -> game id ground truth from the real TypeScript core.
 *
 * Two independent sources are printed for every seed:
 *   - `real`: the actual exported `seedToGameID` from bridge/session.ts
 *             (the production code path the TS EnvSession uses).
 *   - `emu`:  a verbatim inline copy that additionally records every
 *             intermediate value, so the double-precision semantics can be
 *             pinned down and re-implemented natively.
 * A mismatch between `real` and `emu` would mean the recorded semantics are
 * not the ones TS actually runs.
 *
 * Usage: openfront/node_modules/.bin/tsx scripts/seed_game_ids.ts [seed ...]
 * Output: NDJSON on stdout.
 */
import * as fs from "fs";
import { seedToGameID } from "../bridge/session";

const DEFAULT_SEEDS = [
  // The stage-0 RL seed and the other engine seeds.
  "parity",
  "alpha",
  "bravo",
  "charlie",
  // `rl-<seed>`-shaped and raw player-id-shaped strings.
  "rl-parity",
  "rl-0",
  "rl-1",
  "rl-2",
  "rl-3",
  "rl-12345",
  "rl-stage0",
  "rl-stage1",
  "rl-v10",
  "rl-ep",
  "rl-seed42",
  "rl-curriculum",
  // Raw player ids (what `prng_dump.rs` hashes too).
  "AGENTRL1",
  "DUMPPLAYER0",
  "DUMPPLAYER1",
  // Plain numeric / short seeds.
  "0",
  "1",
  "2",
  "42",
  "12345",
  "1337",
  "",
  "seed",
  "seed-0",
  "test-seed-0001",
  "0123456789abcdef",
  // Boundary-ish: a value whose simpleHash is large.
  "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz",
];

const ALPHABET =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

function simpleHashInline(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash;
  }
  return Math.abs(hash);
}

function emu(seed: string) {
  let h = simpleHashInline(`rl-${seed}`);
  const h0 = h;
  const steps: { hIn: number; prodExact: number; exactMasked: number; hDouble: number; int32: number; masked: number }[] = [];
  let out = "";
  for (let i = 0; i < 8; i++) {
    const hIn = h;
    // Exact product as a bigint: what an infinitely wide integer would give.
    const prodExact = BigInt(hIn) * 1103515245n + 12345n;
    // What a wrapping-u32 implementation would produce (the current native
    // semantics): low 31 bits of the exact integer product.
    const exactMasked = Number((prodExact % 0x100000000n) & 0x7fffffffn);
    // The double expression TS actually evaluates (this is the line in
    // bridge/session.ts that rounds above 2^53).
    const hDouble = hIn * 1103515245 + 12345;
    const masked = hDouble & 0x7fffffff; // ToInt32 then & 0x7fffffff
    steps.push({
      hIn,
      prodExact: Number(prodExact),
      exactMasked,
      hDouble,
      int32: hDouble | 0,
      masked,
    });
    h = masked;
    out += ALPHABET[h % ALPHABET.length];
  }
  return { h0, out, steps };
}

function main() {
  const argv = process.argv.slice(2);
  // Either a list of seeds, or a single path to a seeds file (one per line) -
  // the same file the native `seed_ids` bin reads, so the two tables are
  // computed over an identical seed set.
  let seeds: string[];
  if (argv.length === 1 && fs.existsSync(argv[0]) && fs.statSync(argv[0]).isFile()) {
    seeds = fs
      .readFileSync(argv[0], "utf8")
      .split("\n")
      .map((l) => l.replace(/\r$/, ""))
      .filter((l) => l.length > 0);
  } else if (argv.length > 0) {
    seeds = argv;
  } else {
    seeds = DEFAULT_SEEDS;
  }
  for (const seed of seeds) {
    const real = seedToGameID(seed);
    const e = emu(seed);
    const line = {
      seed,
      real_game_id: real,
      emu_game_id: e.out,
      emu_matches_real: real === e.out,
      h0: e.h0,
      steps: e.steps,
    };
    process.stdout.write(JSON.stringify(line) + "\n");
  }
}

main();