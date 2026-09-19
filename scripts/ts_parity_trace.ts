/**
 * TS-core reference trace for the CUDA port, in the *exact* format the
 * native oracle emits (rust/engine/src/bin/parity_trace.rs -> 
 * rust/parity-traces/parity_stage0_seedparity.jsonl).
 *
 * Why this exists: parity so far was measured against the native Rust
 * engine. The CUDA port must match the *TypeScript core*. This script
 * produces the TS-produced oracle in the same machine-checkable NDJSON
 * shape, so a reimplementation can be diffed against TS rather than
 * native.
 *
 * It drives bridge/session.ts `EnvSession` — the TS analogue of the Rust
 * `RlSession` — headlessly under node/tsx. That matters: `EnvSession.step`
 * is the TS source that the native `RlSession::step` was written to
 * mirror (same stamping, same `turnNumber = ticks()` capture, same
 * break-on-win), so this episode is the TS-side ground truth.
 *
 * Stage-0 configuration (the closest TS analogue of the Rust stage 0):
 *   map=Pangaea, seed="parity", bots=2, difficulty=Easy, nations=0,
 *   n_agents=1 (clientID AGENTRL1), decision_ticks=15, decisions=24.
 * That is exactly the knob set the native trace header records.
 *
 * The plane hashes reuse the native definition verbatim:
 *   FNV-1a 64, offset basis 0xcbf29ce484222325, prime 0x100000001b3, over
 *   raw bytes. The state plane feeds each u16 as two little-endian bytes;
 *   the terrain plane is raw bytes. Printed lowercase, 0x-prefixed, 16
 *   digits.
 *
 * Usage (from openfront-ai/):
 *   openfront/node_modules/.bin/tsx scripts/ts_parity_trace.ts
 *   openfront/node_modules/.bin/tsx scripts/ts_parity_trace.ts \
 *     --map Pangaea --seed parity --bots 2 --difficulty Easy --nations 0 \
 *     --decisions 24 --decision-ticks 15 --expand-troops 50 \
 *     --out rust/parity-traces/parity_stage0_seedparity.ts.jsonl
 */
import * as fs from "fs";
import * as path from "path";
import { EnvSession } from "../bridge/session";
import type { Intent } from "../openfront/src/core/Schemas";

// ---------------------------------------------------------------------------
// FNV-1a 64 (the contract — identical to rust/engine/src/bin/parity_trace.rs)
// ---------------------------------------------------------------------------

const FNV_OFFSET = 0xcbf29ce484222325n;
const FNV_PRIME = 0x100000001b3n;
const MASK64 = 0xffffffffffffffffn;

/** FNV-1a 64 over raw bytes. */
function fnv1a64Bytes(bytes: Uint8Array): bigint {
  let h = FNV_OFFSET;
  for (let i = 0; i < bytes.length; i++) {
    h = ((h ^ BigInt(bytes[i])) * FNV_PRIME) & MASK64;
  }
  return h;
}

/** FNV-1a 64 over a u16 plane, each word fed as two little-endian bytes. */
function fnv1a64U16LE(words: Uint16Array): bigint {
  let h = FNV_OFFSET;
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    h = ((h ^ BigInt(w & 0xff)) * FNV_PRIME) & MASK64;
    h = ((h ^ BigInt((w >>> 8) & 0xff)) * FNV_PRIME) & MASK64;
  }
  return h;
}

function hex64(h: bigint): string {
  return "0x" + h.toString(16).padStart(16, "0");
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

interface Args {
  map: string;
  seed: string;
  bots: number;
  difficulty: string;
  nations: number | "default" | "disabled";
  nAgents: number;
  decisions: number;
  decisionTicks: number;
  expandTroops: number;
  spawnTile: number | null;
  out: string;
}

function parseArgs(argv: string[]): Args {
  const repo = path.join(__dirname, "..");
  const a: Args = {
    map: "Pangaea",
    seed: "parity",
    bots: 2,
    difficulty: "Easy",
    nations: 0,
    nAgents: 1,
    decisions: 24,
    decisionTicks: 15,
    expandTroops: 50,
    spawnTile: null,
    out: path.join(repo, "rust/parity-traces/parity_stage0_seedparity.ts.jsonl"),
  };
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    const val = argv[i + 1];
    switch (key) {
      case "--map":
        a.map = val;
        i++;
        break;
      case "--seed":
        a.seed = val;
        i++;
        break;
      case "--bots":
        a.bots = parseInt(val, 10);
        i++;
        break;
      case "--difficulty":
        a.difficulty = val;
        i++;
        break;
      case "--nations":
        a.nations = /^\d+$/.test(val) ? parseInt(val, 10) : (val as Args["nations"]);
        i++;
        break;
      case "--n-agents":
      case "--nagents":
        a.nAgents = parseInt(val, 10);
        i++;
        break;
      case "--decisions":
        a.decisions = parseInt(val, 10);
        i++;
        break;
      case "--decision-ticks":
        a.decisionTicks = parseInt(val, 10);
        i++;
        break;
      case "--expand-troops":
        a.expandTroops = parseInt(val, 10);
        i++;
        break;
      case "--spawn-tile":
        a.spawnTile = parseInt(val, 10);
        i++;
        break;
      case "--out":
        a.out = val;
        i++;
        break;
      default:
        break;
    }
  }
  return a;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.nAgents !== 1) {
    // RlSession/EnvSession only wire AGENTRL1 in bridge/session.ts; be explicit.
    throw new Error(
      `ts_parity_trace.ts only supports n_agents=1 (bridge/session.ts wires a single AGENTRL1); got ${args.nAgents}`,
    );
  }

  const session = new EnvSession();
  await session.reset(args.map, args.seed, args.bots, args.difficulty, args.nations);

  const game = session.game;
  const width = game.width();
  const height = game.height();
  const numTiles = width * height;

  // --- PRE-SPAWN map facts (immediately after reset's one init tick) --------
  const terrainBytes = new Uint8Array(numTiles);
  for (let ref = 0; ref < numTiles; ref++) terrainBytes[ref] = game.terrainByte(ref);
  const terrainHash = fnv1a64Bytes(terrainBytes);

  const preState = game.tileStateBuffer();
  const preSpawnStateHash = fnv1a64U16LE(preState);

  const landTilesEngine = game.numLandTiles();
  let landTilesFromTerrain = 0;
  for (let ref = 0; ref < numTiles; ref++) {
    if (terrainBytes[ref] & 0x80) landTilesFromTerrain++;
  }
  const tickAfterReset = game.ticks();

  // --- Scripted spawn tile: first legal region tile (row-major) ------------
  // Mirrors parity_trace.rs::first_legal_region_tile: land && !impassable &&
  // magnitude < 31 && owner == 0. magnitude is masked to 5 bits (max 31), so
  // `< 31` is equivalent to `!impassable`; both are kept for exactness.
  let spawnTile = args.spawnTile;
  if (spawnTile === null) {
    spawnTile = -1;
    for (let ref = 0; ref < numTiles; ref++) {
      if (
        game.isLand(ref) &&
        !game.isImpassable(ref) &&
        game.magnitude(ref) < 31 &&
        game.ownerID(ref) === 0
      ) {
        spawnTile = ref;
        break;
      }
    }
    if (spawnTile < 0) throw new Error(`no legal spawn tile found on map ${args.map}`);
  }
  const spawnGx = Math.floor((spawnTile % width) / 8);
  const spawnGy = Math.floor(Math.floor(spawnTile / width) / 8);

  // --- Header --------------------------------------------------------------
  const script = {
    decision_0: [{ type: "spawn", tile: spawnTile }],
    decisions_1_to_N: [{ type: "attack", targetID: null, troops: args.expandTroops }],
    note:
      "No RNG, no policy: identical intent list on every decision. `expand` is the ofcore translate form " +
      "(ofcore/src/translate.rs:253): an attack with targetID null (TerraNullius).",
    ticks_per_decision: args.decisionTicks,
    decisions_total: args.decisions,
    n_agents: args.nAgents,
  };

  const header = {
    type: "header",
    engine: "openfront-ts",
    repo_root: path.join(__dirname, ".."),
    stage: 0,
    stage_name: "Easy",
    map: args.map,
    seed: args.seed,
    game_id: session.gameID,
    bots: args.bots,
    difficulty: args.difficulty,
    nations: String(args.nations),
    n_agents: args.nAgents,
    decision_ticks: args.decisionTicks,
    map_width: width,
    map_height: height,
    tick_after_reset: tickAfterReset,
    spawn_tile: spawnTile,
    spawn_region: { gx: spawnGx, gy: spawnGy },
    pre_spawn: {
      terrain_hash: hex64(terrainHash),
      terrain_plane_bytes: terrainBytes.length,
      state_hash: hex64(preSpawnStateHash),
      state_plane_words: preState.length,
      land_tiles_engine: landTilesEngine,
      land_tiles_from_terrain: landTilesFromTerrain,
    },
    hash: {
      algo: "fnv1a64",
      offset_basis: "0xcbf29ce484222325",
      prime: "0x100000001b3",
      state_plane: "width*height u16, each word as 2 LE bytes",
      obs_mask_buffers: "n_agents*PER_AGENT f32, each float as 4 LE bytes",
      print: "lowercase hex, 0x-prefixed, 16 digits",
    },
    ffi_cfg:
      `repo_root=${path.join(__dirname, "..")}, map=${args.map}, seed=${args.seed}, bots=${args.bots}, ` +
      `difficulty=${args.difficulty}, nations=${String(args.nations)}, n_agents=${args.nAgents}, ` +
      `ticks_per_decision=${args.decisionTicks}, stage=0`,
    script,
  };

  fs.mkdirSync(path.dirname(args.out), { recursive: true });
  const fd = fs.openSync(args.out, "w");
  const write = (o: unknown) => fs.writeSync(fd, JSON.stringify(o) + "\n");
  write(header);

  // --- Scripted episode ----------------------------------------------------
  let decisionsRun = 0;
  for (let d = 0; d < args.decisions; d++) {
    const intents: Intent[] =
      d === 0
        ? ([{ type: "spawn", tile: spawnTile }] as unknown as Intent[])
        : ([{ type: "attack", targetID: null, troops: args.expandTroops }] as unknown as Intent[]);

    const { head } = session.step(intents, args.decisionTicks);
    const tick = game.ticks();

    const stateHash = fnv1a64U16LE(game.tileStateBuffer());

    const players = game
      .allPlayers()
      .slice()
      .sort((a, b) => a.smallID() - b.smallID())
      .map((p) => {
        const tiles = p.numTilesOwned();
        const gold = p.gold();
        const safe = BigInt(Number.MAX_SAFE_INTEGER);
        return {
          small_id: p.smallID(),
          tiles_owned: tiles,
          alive: tiles > 0,
          engine_alive: p.isAlive(),
          troops: Math.round(p.troops()),
          // native prints gold as a JSON number; TS gold is a bigint, so keep
          // the numeric form while it is exactly representable.
          gold: gold <= safe ? Number(gold) : gold.toString(),
        };
      });

    const winner = (head as { winner?: unknown }).winner ?? null;
    const terminal = winner !== null;

    write({
      type: "decision",
      decision: d,
      tick,
      spawn_phase: game.inSpawnPhase(),
      state_hash: hex64(stateHash),
      players,
      terminal,
      winner,
      intents,
    });

    decisionsRun = d + 1;
    if (terminal) break;
  }

  write({
    type: "footer",
    final_tick: game.ticks(),
    decisions_run: decisionsRun,
  });
  fs.closeSync(fd);

  process.stderr.write(
    `[ts_parity_trace] wrote ${args.out} (map ${args.map}, seed ${args.seed}, ${width}x${height}, ` +
      `spawn_tile=${spawnTile}, land_tiles=${landTilesEngine}, terrain=${hex64(terrainHash)}, ` +
      `pre_spawn_state=${hex64(preSpawnStateHash)}, decisions=${decisionsRun})\n`,
  );

  // Echo the trace to stdout (the harness prints the first 20 lines).
  const text = fs.readFileSync(args.out, "utf8");
  process.stdout.write(text);
}

main().catch((e) => {
  process.stderr.write(String(e?.stack ?? e) + "\n");
  process.exit(1);
});