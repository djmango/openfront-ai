//! Parity oracle for a CUDA reimplementation of the OpenFront RL environment.
//!
//! Emits a deterministic, machine-checkable reference trace of a scripted
//! episode so a reimplementation can be diffed against the real engine.
//!
//! Usage (from `rust/`):
//!   nix shell nixpkgs#gcc nixpkgs#pkg-config -c cargo run --release \
//!     -p openfront-engine --bin parity_trace -- \
//!     --seed parity --stage 0 --decisions 24 \
//!     --out /tmp/parity_stage0.jsonl
//!
//! The trace is NDJSON: line 1 is a header (the exact script, config, hash
//! definition and the PRE-SPAWN map facts), then one JSON object per decision.
//!
//! Two engines are driven in lockstep with the same scripted intents:
//!
//!   * a direct `RlSession` (built exactly the way `puffer_ffi::new_state`
//!     builds one, `puffer_ffi.rs:968-977`) - source of the per-player facts
//!     (`troops`/`gold`) that the C ABI does not expose; and
//!   * the real C ABI env (`ofenv_create`/`ofenv_set_stage`/`ofenv_reset`/
//!     `ofenv_step`) - source of the `ofenv_obs` / `ofenv_mask` buffers
//!     "exactly as the C ABI would see them", the reward itemisation from
//!     `ofenv_meta`, and `ofenv_terminal`.
//!
//! Both call `RlSession::reset` with identical inputs, so they are the same
//! deterministic episode. That is not assumed: after every decision the
//! packed state plane is hashed from *both* (`ofenv_tiles` vs
//! `RlSession::tile_state`) and the traces are only combined if the hashes
//! match (`lockstep_ok`); a mismatch is reported, never papered over.
//!
//! Hash definition (must be matched exactly by the CUDA side):
//!   FNV-1a 64-bit, offset basis 0xcbf29ce484222325, prime 0x100000001b3,
//!   over the raw bytes. The state plane feeds each `u16` as two
//!   little-endian bytes; the obs/mask buffers feed each `f32` as four
//!   little-endian bytes. Every hash prints as lowercase hex with `0x`.

use clap::Parser;
use openfront_engine::puffer_ffi::{
    ofenv_create, ofenv_destroy, ofenv_last_error, ofenv_mask, ofenv_obs, ofenv_reward,
    ofenv_reset, ofenv_set_stage, ofenv_step, ofenv_terminal, ofenv_tiles, stage_params,
    OFENV_MASK_PER_AGENT, OFENV_OBS_PER_AGENT,
};
use openfront_engine::rl::RlSession;
use serde_json::{json, Value};
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::io::Write;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// FNV-1a 64 (the contract)
// ---------------------------------------------------------------------------

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[inline]
fn fnv_step(h: u64, b: u8) -> u64 {
    (h ^ b as u64).wrapping_mul(FNV_PRIME)
}

/// FNV-1a 64 over raw bytes.
fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |h, &b| fnv_step(h, b))
}

/// FNV-1a 64 over a `u16` plane, each word fed as two little-endian bytes.
fn fnv1a64_u16_le(words: &[u16]) -> u64 {
    let mut h = FNV_OFFSET;
    for w in words {
        let b = w.to_le_bytes();
        h = fnv_step(h, b[0]);
        h = fnv_step(h, b[1]);
    }
    h
}

/// FNV-1a 64 over an `f32` buffer, each float fed as four little-endian bytes
/// (its IEEE-754 bit pattern).
fn fnv1a64_f32_le(vals: &[f32]) -> u64 {
    let mut h = FNV_OFFSET;
    for v in vals {
        for b in v.to_bits().to_le_bytes() {
            h = fnv_step(h, b);
        }
    }
    h
}

fn hex64(h: u64) -> String {
    format!("0x{h:016x}")
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = "parity_trace")]
#[command(about = "Emit a deterministic reference trace of a scripted RL episode")]
struct Args {
    /// OpenFront repo root (the dir containing `openfront/resources/maps`).
    #[arg(long, default_value = "/opt/data/workspaces/skg/openfront-ai")]
    repo: PathBuf,
    /// Episode seed (`seed_to_game_id` input).
    #[arg(long, default_value = "parity")]
    seed: String,
    /// V10 curriculum stage index; picks bots / nations / difficulty / ticks.
    #[arg(long, default_value_t = 0)]
    stage: usize,
    /// Map key override (default: the stage's map pool's first map).
    #[arg(long)]
    map: Option<String>,
    /// Number of RL humans (1 or 2; RlSession clamps).
    #[arg(long, default_value_t = 1)]
    n_agents: u32,
    /// Total decisions to script (decision 0 = spawn, the rest = expand).
    #[arg(long, default_value_t = 24)]
    decisions: u32,
    /// Troop count sent by every scripted `expand` intent.
    #[arg(long, default_value_t = 50)]
    expand_troops: i64,
    /// Explicit spawn tile (default: first legal region tile, row-major).
    #[arg(long)]
    spawn_tile: Option<u32>,
    /// Override the stage's ticks-per-decision. Diagnostics only (e.g. `1` to
    /// bisect the first diverging tick at tick resolution); the reference
    /// traces use the stage's own value.
    #[arg(long)]
    ticks_per_decision: Option<u32>,
    /// Decision index at which the obs/mask buffer hashes + reward
    /// itemisation are emitted.
    #[arg(long, default_value_t = 1)]
    obs_decision: u32,
    /// Trace output path (NDJSON). Parent dirs are created.
    #[arg(long)]
    out: Option<PathBuf>,
}

fn last_error() -> String {
    let p = ofenv_last_error();
    if p.is_null() {
        return "<no last_error>".into();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

fn cstring(s: String) -> CString {
    CString::new(s).unwrap_or_default()
}

/// First tile (row-major) that the featurizer's spawn legality accepts:
/// `land && magnitude < 31 && owner == 0` (`ofcore/src/feat.rs:1392-1420`).
/// The first such tile necessarily sits in the first legal `/8` region.
fn first_legal_region_tile(session: &RlSession) -> Option<(u32, u32, u32)> {
    let w = session.game.width();
    let h = session.game.height();
    for y in 0..h {
        for x in 0..w {
            let t = y * w + x;
            if session.game.is_land(t)
                && !session.game.is_impassable(t)
                && session.game.map.magnitude(t) < 31
                && session.game.map.owner_id(t) == 0
            {
                return Some((t, x / 8, y / 8));
            }
        }
    }
    None
}

fn main() {
    let args = Args::parse();

    // ---------------------------------------------------------------------
    // Stage params (same resolution the FFI's `ofenv_set_stage` uses,
    // puffer_ffi.rs:1377-1402).
    // ---------------------------------------------------------------------
    let params = match stage_params(args.stage) {
        Some(p) => p,
        None => {
            eprintln!("stage {} has no V10 params", args.stage);
            std::process::exit(2);
        }
    };
    let map = args
        .map
        .clone()
        .unwrap_or_else(|| params.maps.first().cloned().unwrap_or_default());
    let nations_value: Value = if params.nations_default {
        Value::String("default".into())
    } else {
        Value::from(params.nations as i64)
    };
    let decision_ticks = args
        .ticks_per_decision
        .unwrap_or(params.decision_ticks)
        .max(1);

    let out_path = args.out.clone().unwrap_or_else(|| {
        args.repo
            .join("rust/parity-traces")
            .join(format!("parity_stage{}_seed{}.jsonl", args.stage, args.seed))
    });
    if let Some(parent) = out_path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            eprintln!("mkdir {}: {e}", parent.display());
            std::process::exit(2);
        }
    }
    let mut file = match std::fs::File::create(&out_path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("create {}: {e}", out_path.display());
            std::process::exit(2);
        }
    };

    // ---------------------------------------------------------------------
    // (1) Fresh RlSession at this stage / seed.
    // ---------------------------------------------------------------------
    let (mut session, _head, _ents, _legal, _terrain, _duo) = match RlSession::reset(
        &args.repo,
        &map,
        &args.seed,
        params.bots,
        &params.difficulty,
        nations_value.clone(),
        args.n_agents,
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("RlSession::reset failed: {e}");
            std::process::exit(2);
        }
    };

    // ---------------------------------------------------------------------
    // (2) PRE-SPAWN map facts.
    //
    // `RlSession::reset` runs exactly one init tick before returning
    // (`rl.rs:201`, mirroring `bridge/env.ts`), so "pre-spawn" here means
    // "immediately after reset, before any decision/step", not "before the
    // map was ever ticked".
    // ---------------------------------------------------------------------
    let width = session.game.width();
    let height = session.game.height();
    let terrain_hash = fnv1a64(session.game.map.terrain_bytes());
    let pre_state = session.tile_state().to_vec();
    let pre_spawn_state_hash = fnv1a64_u16_le(&pre_state);
    let land_tiles_engine = session.game.num_land_tiles();
    let land_tiles_terrain = session
        .game
        .map
        .terrain_bytes()
        .iter()
        .filter(|b| *b & 0x80 != 0)
        .count() as u32;
    let tick_after_reset = session.game.ticks();

    // ---------------------------------------------------------------------
    // (2b) The C ABI env, same config.
    // ---------------------------------------------------------------------
    let nations_cfg = if params.nations_default {
        "default".to_string()
    } else {
        params.nations.to_string()
    };
    let cfg_text = format!(
        "repo_root={}, map={}, seed={}, bots={}, difficulty={}, nations={}, \
         n_agents={}, ticks_per_decision={}, stage={}",
        args.repo.display(),
        map,
        args.seed,
        params.bots,
        params.difficulty,
        nations_cfg,
        args.n_agents,
        decision_ticks,
        args.stage
    );
    let cfg_c = cstring(cfg_text.clone());
    let env = ofenv_create(cfg_c.as_ptr() as *const c_char);
    if env.is_null() {
        eprintln!("ofenv_create failed: {}", last_error());
        eprintln!("cfg = {cfg_text}");
        std::process::exit(2);
    }
    if ofenv_set_stage(env, args.stage as c_int) != 0 {
        eprintln!("ofenv_set_stage({}) failed: {}", args.stage, last_error());
        std::process::exit(2);
    }
    if ofenv_reset(env) != 0 {
        eprintln!("ofenv_reset failed: {}", last_error());
        std::process::exit(2);
    }
    let env = env as *mut c_void;

    // ---------------------------------------------------------------------
    // (3) The deterministic script.
    // ---------------------------------------------------------------------
    let spawn_tile = match args.spawn_tile {
        Some(t) => t,
        None => match first_legal_region_tile(&session) {
            Some((t, _, _)) => t,
            None => {
                eprintln!("no legal spawn tile found on map {map}");
                std::process::exit(2);
            }
        },
    };
    let spawn_gx = spawn_tile % width / 8;
    let spawn_gy = spawn_tile / width / 8;
    let script = json!({
        "decision_0": [{"type": "spawn", "tile": spawn_tile}],
        "decisions_1_to_N": [
            {"type": "attack", "targetID": null, "troops": args.expand_troops}
        ],
        "note": "No RNG, no policy: identical intent list on every decision. \
                 `expand` is the ofcore translate form (ofcore/src/translate.rs:253): \
                 an attack with targetID null (TerraNullius).",
        "ticks_per_decision": decision_ticks,
        "decisions_total": args.decisions,
        "n_agents": args.n_agents,
    });

    // ---------------------------------------------------------------------
    // Header line.
    // ---------------------------------------------------------------------
    let header = json!({
        "type": "header",
        "engine": "openfront-native",
        "repo_root": args.repo.display().to_string(),
        "stage": args.stage,
        "stage_name": params.name,
        "map": map,
        "seed": args.seed,
        "game_id": session.game_id(),
        "bots": params.bots,
        "difficulty": params.difficulty,
        "nations": nations_cfg,
        "n_agents": args.n_agents,
        "decision_ticks": decision_ticks,
        "map_width": width,
        "map_height": height,
        "tick_after_reset": tick_after_reset,
        "spawn_tile": spawn_tile,
        "spawn_region": {"gx": spawn_gx, "gy": spawn_gy},
        "pre_spawn": {
            "terrain_hash": hex64(terrain_hash),
            "terrain_plane_bytes": session.game.map.terrain_bytes().len(),
            "state_hash": hex64(pre_spawn_state_hash),
            "state_plane_words": pre_state.len(),
            "land_tiles_engine": land_tiles_engine,
            "land_tiles_from_terrain": land_tiles_terrain,
        },
        "hash": {
            "algo": "fnv1a64",
            "offset_basis": "0xcbf29ce484222325",
            "prime": "0x100000001b3",
            "state_plane": "width*height u16, each word as 2 LE bytes",
            "obs_mask_buffers": "n_agents*PER_AGENT f32, each float as 4 LE bytes",
            "print": "lowercase hex, 0x-prefixed, 16 digits"
        },
        "ffi_cfg": cfg_text,
        "obs_per_agent": OFENV_OBS_PER_AGENT,
        "mask_per_agent": OFENV_MASK_PER_AGENT,
        "obs_hash_decision": args.obs_decision,
        "script": script,
    });
    writeln!(file, "{}", header).expect("write header");
    file.flush().ok();

    // ---------------------------------------------------------------------
    // (4) Scripted episode.
    // ---------------------------------------------------------------------
    let expansions = &script["decisions_1_to_N"];
    let mut lockstep_all_ok = true;
    let mut decisions_run = 0u32;

    for d in 0..args.decisions {
        let intents: Vec<Value> = if d == 0 {
            vec![json!({"type": "spawn", "tile": spawn_tile})]
        } else {
            expansions.as_array().cloned().unwrap_or_default()
        };
        let intents_text = serde_json::to_string(&Value::Array(intents.clone())).unwrap();

        // Direct session (per-player facts + engine state plane).
        let (head, _e, _l, _duo) = session.step(&intents, decision_ticks);
        let wasted = head.get("wasted").and_then(Value::as_u64).unwrap_or(0);

        // C ABI env (obs/mask/reward/terminal).
        let intents_c = cstring(intents_text.clone());
        if ofenv_step(env, intents_c.as_ptr() as *const c_char) != 0 {
            eprintln!("ofenv_step(decision {d}) failed: {}", last_error());
            std::process::exit(2);
        }

        let tick = session.game.ticks();

        // State plane hash from both engines + lockstep check.
        let state = session.tile_state();
        let state_hash = fnv1a64_u16_le(state);
        let mut tn: c_int = 0;
        let tp = ofenv_tiles(env, &mut tn);
        let ffi_state_hash = if tp.is_null() {
            None
        } else {
            let ffi_tiles = unsafe { std::slice::from_raw_parts(tp, tn as usize) };
            Some(fnv1a64_u16_le(ffi_tiles))
        };
        let lockstep_ok = ffi_state_hash == Some(state_hash);
        if !lockstep_ok {
            lockstep_all_ok = false;
        }

        // Per-player facts.
        let players: Vec<Value> = session
            .game
            .players_in_order()
            .iter()
            .map(|p| {
                json!({
                    "small_id": p.small_id,
                    "tiles_owned": p.tiles_owned,
                    "alive": p.tiles_owned > 0,
                    "engine_alive": p.alive,
                    "troops": p.troops,
                    "gold": p.gold,
                })
            })
            .collect();

        // FFI verdicts for this decision.
        let meta_ptr = openfront_engine::puffer_ffi::ofenv_meta(env);
        let meta: Value = if meta_ptr.is_null() {
            Value::Null
        } else {
            let s = unsafe { CStr::from_ptr(meta_ptr) }.to_string_lossy().into_owned();
            serde_json::from_str(&s).unwrap_or(Value::Null)
        };
        let terminal = ofenv_terminal(env, 0) != 0;
        let reward = ofenv_reward(env, 0);
        let winner = meta.get("winner").cloned().unwrap_or(Value::Null);

        let mut line = json!({
            "type": "decision",
            "decision": d,
            "tick": tick,
            "spawn_phase": session.game.in_spawn_phase(),
            "state_hash": hex64(state_hash),
            "players": players,
            "terminal": terminal,
            "ffi_terminal": terminal,
            "reward_agent0": reward,
            "won": openfront_engine::puffer_ffi::ofenv_won(env) != 0,
            "winner": winner,
            "wasted": wasted,
            "intents": intents,
            "ffi_state_hash": ffi_state_hash.map(hex64),
            "lockstep_ok": lockstep_ok,
        });

        if d == args.obs_decision {
            let mut on: c_int = 0;
            let op = ofenv_obs(env, &mut on);
            let (obs_hash, obs_len) = if op.is_null() {
                (None, 0usize)
            } else {
                let buf = unsafe { std::slice::from_raw_parts(op, on as usize) };
                (Some(fnv1a64_f32_le(buf)), on as usize)
            };
            let mut mn: c_int = 0;
            let mp = ofenv_mask(env, &mut mn);
            let (mask_hash, mask_len) = if mp.is_null() {
                (None, 0usize)
            } else {
                let buf = unsafe { std::slice::from_raw_parts(mp, mn as usize) };
                (Some(fnv1a64_f32_le(buf)), mn as usize)
            };
            line["obs_hash"] = json!(obs_hash.map(hex64));
            line["mask_hash"] = json!(mask_hash.map(hex64));
            line["obs_len"] = json!(obs_len);
            line["mask_len"] = json!(mask_len);
            line["reward_components"] = meta
                .get("agents")
                .and_then(|a| a.get(0))
                .and_then(|a| a.get("reward_components"))
                .cloned()
                .unwrap_or(Value::Null);
            line["agent_meta"] = meta.get("agents").cloned().unwrap_or(Value::Null);
        }

        writeln!(file, "{line}").expect("write decision");
        file.flush().ok();
        decisions_run = d + 1;

        if terminal {
            break;
        }
    }

    let meta = json!({
        "type": "footer",
        "lockstep_all_ok": lockstep_all_ok,
        "final_tick": session.game.ticks(),
        "decisions_run": decisions_run,
    });
    writeln!(file, "{meta}").expect("write footer");
    file.flush().ok();
    ofenv_destroy(env);

    eprintln!(
        "[parity_trace] wrote {} (stage {}, seed {}, map {}, {}x{}, land_tiles={}, \
         terrain={}, pre_spawn_state={}, lockstep_all_ok={})",
        out_path.display(),
        args.stage,
        args.seed,
        map,
        width,
        height,
        land_tiles_engine,
        hex64(terrain_hash),
        hex64(pre_spawn_state_hash),
        lockstep_all_ok
    );
}
