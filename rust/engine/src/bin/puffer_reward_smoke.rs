//! Smoke test for the curriculum-shaped reward + stage table wired into the
//! C ABI (`openfront_engine::puffer_ffi`).
//!
//! Creates a `map=pangaea, n_agents=1` env, prints `ofenv_stage_count()` and
//! the V10 stage table, then for two stages runs `ofenv_set_stage`, prints the
//! stage's `ofenv_stage_info` JSON, and plays 200 decisions with EMPTY intents
//! at that stage's `decision_ticks`, printing the per-decision reward stream,
//! terminal events and cumulative reward per stage.
//!
//! Why a spawn decision first: the engine only leaves the spawn phase when the
//! human picks a spawn tile (`execution/spawn.rs`), exactly like the trainer's
//! `spawn_randomly` fallback. Empty intents alone would sit in the spawn phase
//! forever and every decision would be reward 0. The spawn is labelled; the 200
//! measured decisions use truly empty intents.
//!
//! Run:
//!   cargo run --release -p openfront-engine --bin puffer_reward_smoke

use std::ffi::{c_void, CStr, CString};

use openfront_engine::puffer_ffi::{
    ofenv_create, ofenv_destroy, ofenv_last_error, ofenv_mask, ofenv_meta, ofenv_reward,
    ofenv_set_stage, ofenv_stage_count, ofenv_stage_info, ofenv_step, ofenv_terminal,
    stage_params, OFENV_GW_MAX, OFENV_MASK_TILE_OFF,
};
use serde_json::Value;

const REPO_ROOT: &str = "/opt/data/workspaces/skg/openfront-ai";
const DECISIONS: usize = 200;
/// The two stages exercised below (both use the broad-16 map pool).
const RUN_STAGES: [usize; 2] = [0, 20];

fn cstr(p: *const std::os::raw::c_char) -> String {
    if p.is_null() {
        return "<null>".into();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

fn meta_of(env: *mut c_void) -> Value {
    serde_json::from_str(&cstr(ofenv_meta(env))).unwrap_or(Value::Null)
}

/// First `/8` region that contains at least one spawnable tile, as
/// (gh, gw, region_y, region_x). The FFI mask is region-level (a `1` means the
/// 8x8 region has a land/unowned/passable tile somewhere in it), so the caller
/// has to probe the region's tiles to find the valid one.
fn first_legal_region(env: *mut c_void, meta: &Value) -> Option<(usize, usize, usize, usize)> {
    let gh = meta["gh"].as_u64().unwrap_or(0) as usize;
    let gw = meta["gw"].as_u64().unwrap_or(0) as usize;
    let mut n = 0i32;
    let p = ofenv_mask(env, &mut n);
    if p.is_null() || gh == 0 || gw == 0 {
        return None;
    }
    let block = unsafe { std::slice::from_raw_parts(p, n as usize) };
    for y in 0..gh {
        for x in 0..gw {
            if block[OFENV_MASK_TILE_OFF + y * OFENV_GW_MAX + x] != 0.0 {
                return Some((gh, gw, y, x));
            }
        }
    }
    None
}

fn run_stage(env: *mut c_void, stage: usize) {
    let rc = ofenv_set_stage(env, stage as i32);
    if rc != 0 {
        println!("\n=== stage {stage}: ofenv_set_stage FAILED: {} ===", cstr(ofenv_last_error()));
        return;
    }
    println!("\n=== stage {stage} ===");
    println!("stage_info: {}", cstr(ofenv_stage_info(env)));

    let meta = meta_of(env);
    let ticks = meta["ticks_per_decision"].as_u64().unwrap_or(0);
    let width = meta["width"].as_u64().unwrap_or(0) as usize;
    println!(
        "run: {DECISIONS} decisions, ticks_per_decision={ticks}, bots={}, nations={}, width={width}",
        meta["bots"], meta["nations"]
    );

    // One spawn decision so the engine leaves the spawn phase (see module doc).
    // The /8 mask only marks a region as spawnable; probe the region's tiles
    // until one is accepted (a rejected tile is a no-op the engine counts as
    // wasted, it does not advance play).
    let mut spawned = false;
    let mut attempts = 0usize;
    'rounds: for _round in 0..4 {
        let m = meta_of(env);
        if !m["spawn_phase"].as_bool().unwrap_or(true) {
            spawned = true;
            break;
        }
        let width = m["width"].as_u64().unwrap_or(0) as usize;
        let height = m["height"].as_u64().unwrap_or(0) as usize;
        let Some((_gh, _gw, ry, rx)) = first_legal_region(env, &m) else {
            break;
        };
        for dy in 0..8usize {
            for dx in 0..8usize {
                let (ty, tx) = (ry * 8 + dy, rx * 8 + dx);
                if ty >= height || tx >= width {
                    continue;
                }
                let tile = ty * width + tx;
                let spawn = CString::new(format!(
                    "[{{\"type\":\"spawn\",\"tile\":{tile},\"clientID\":\"AGENTRL1\"}}]"
                ))
                .unwrap();
                let _ = ofenv_step(env, spawn.as_ptr());
                attempts += 1;
                if !meta_of(env)["spawn_phase"].as_bool().unwrap_or(true) {
                    spawned = true;
                    break 'rounds;
                }
            }
        }
    }
    let m = meta_of(env);
    println!(
        "spawn: attempts={attempts} spawned={spawned} tick={} spawn_phase={} on_map={}",
        m["tick"], m["spawn_phase"], m["agents"][0]["on_map"]
    );

    let mut cumulative = 0.0f64;
    let mut terminals: Vec<usize> = Vec::new();
    let mut last_tick = 0i64;
    println!("decision stream (empty intents):");
    for d in 0..DECISIONS {
        // EMPTY intents: NULL pointer.
        let rc = ofenv_step(env, std::ptr::null());
        if rc != 0 {
            println!("  d={d:03} ofenv_step FAILED: {}", cstr(ofenv_last_error()));
            break;
        }
        let reward = ofenv_reward(env, 0);
        let terminal = ofenv_terminal(env, 0);
        let m = meta_of(env);
        last_tick = m["tick"].as_i64().unwrap_or(0);
        cumulative += reward;
        if terminal == 1 {
            terminals.push(d);
        }
        println!(
            "  d={d:03} tick={last_tick:>5} reward={reward:+.9} cumulative={cumulative:+.9} terminal={terminal}"
        );
        if terminal == 1 {
            break;
        }
    }
    println!(
        "stage {stage} summary: decisions_run={} last_tick={last_tick} cumulative_reward={cumulative:+.9} terminal_events={terminals:?}",
        if terminals.is_empty() { DECISIONS } else { terminals[0] + 1 }
    );
    let final_meta = meta_of(env);
    if let Some(agents) = final_meta["agents"].as_array() {
        if let Some(a) = agents.first() {
            println!(
                "  final agent0: me={} on_map={} tiles={} share={:.6} reward={:+.9} terminal={}",
                a["me"], a["on_map"], a["tiles"], a["share"], a["reward"], a["terminal"]
            );
            println!("  final reward_components: {}", a["reward_components"]);
        }
    }
}

fn main() {
    let cfg = format!("repo_root={REPO_ROOT},map=pangaea,n_agents=1,seed=reward-smoke");
    let c = CString::new(cfg).unwrap();
    let env = ofenv_create(c.as_ptr());
    if env.is_null() {
        eprintln!("ofenv_create failed: {}", cstr(ofenv_last_error()));
        std::process::exit(1);
    }

    let n = ofenv_stage_count();
    println!("ofenv_stage_count() = {n}");

    println!("\nV10 stage table (index, name, decision_ticks, bots, nations, win_at, env_target):");
    println!(
        "{:>5}  {:<10} {:>14} {:>5} {:>7} {:>7} {:>10}",
        "idx", "name", "decision_ticks", "bots", "nations", "win_at", "env_target"
    );
    for i in 0..n as usize {
        match stage_params(i) {
            Some(p) => println!(
                "{:>5}  {:<10} {:>14} {:>5} {:>7} {:>7.3} {:>10}",
                p.index, p.name, p.decision_ticks, p.bots, p.nations, p.win_at, p.env_target
            ),
            None => println!("{:>5}  <no params>", i),
        }
    }

    for stage in RUN_STAGES {
        run_stage(env, stage);
    }

    ofenv_destroy(env);
}