//! Smoke test for the `puffer_ffi` C ABI: drives a real `RlSession` through
//! the exact `extern "C"` surface a PufferLib-style C env would call.
//!
//! Usage (from rust/):
//!   nix shell nixpkgs#gcc -c cargo run --release -p openfront-engine \
//!     --bin puffer_smoke
//!
//! The task-specified config is the default (map=plains). NOTE: this
//! checkout has NO `plains` map key under openfront/resources/maps (Plains
//! is a TerrainType, not a GameMapType), so the default run reports the real
//! missing-map error. Pass a config override as argv[1] (or set
//! `PUFFER_SMOKE_CFG`) to drive a real map, e.g.
//!   ... --bin puffer_smoke "repo_root=/opt/data/workspaces/skg/openfront-ai,
//!     map=pangaea, seed=smoke, bots=3, difficulty=Easy, n_agents=1,
//!     ticks_per_decision=8"
//!
//! Runs 200 decisions with EMPTY intents and prints obs/mask sizes, the
//! nonzero mask entries per action, reward, terminal count, tick count and
//! the meta head. Exits non-zero on any FFI error.

use std::ffi::{c_char, c_int, c_void, CStr, CString};

use openfront_engine::puffer_ffi::{
    ofenv_create, ofenv_destroy, ofenv_last_error, ofenv_mask, ofenv_mask_size, ofenv_meta,
    ofenv_obs, ofenv_obs_size, ofenv_reset, ofenv_reward, ofenv_step, ofenv_terminal,
    ofenv_tiles,
    OFENV_MASK_ACTIONS_N, OFENV_MASK_PER_AGENT, OFENV_MASK_PTARGET_N, OFENV_MASK_PTARGET_OFF,
    OFENV_MASK_UTARGET_N, OFENV_MASK_UTARGET_OFF, OFENV_OBS_PER_AGENT,
};

const DEFAULT_CFG: &str = "repo_root=/opt/data/workspaces/skg/openfront-ai, map=plains, \
                            seed=smoke, bots=3, difficulty=Easy, n_agents=1, \
                            ticks_per_decision=8";

fn resolve_cfg() -> String {
    if let Some(a) = std::env::args().nth(1) {
        return a;
    }
    if let Ok(c) = std::env::var("PUFFER_SMOKE_CFG") {
        if !c.trim().is_empty() {
            return c;
        }
    }
    DEFAULT_CFG.to_string()
}

fn last_error() -> String {
    let p = ofenv_last_error();
    if p.is_null() {
        return "<no last_error>".into();
    }
    unsafe { CStr::from_ptr(p) }
        .to_string_lossy()
        .into_owned()
}

fn meta_json(env: *mut c_void) -> String {
    let p = ofenv_meta(env);
    if p.is_null() {
        return "<null meta>".into();
    }
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// Nonzero count in `mask` row `action` of a per-agent [N_ACTIONS, n] table.
fn row_nonzero(mask: &[f32], base: usize, off: usize, row: usize, n: usize) -> usize {
    let start = base + off + row * n;
    mask[start..start + n].iter().filter(|v| **v != 0.0).count()
}

fn main() {
    let cfg_text = resolve_cfg();
    let cfg = CString::new(cfg_text.clone()).unwrap();
    let env = ofenv_create(cfg.as_ptr() as *const c_char);
    if env.is_null() {
        eprintln!("ofenv_create FAILED: {}", last_error());
        eprintln!("cfg = {cfg_text}");
        std::process::exit(1);
    }
    println!("ofenv_create ok: cfg = {cfg_text}");

    if ofenv_reset(env) != 0 {
        eprintln!("ofenv_reset FAILED: {}", last_error());
        std::process::exit(1);
    }
    println!(
        "ofenv_reset ok: obs_size={} mask_size={} (per agent: obs={} mask={})",
        ofenv_obs_size(env),
        ofenv_mask_size(env),
        OFENV_OBS_PER_AGENT,
        OFENV_MASK_PER_AGENT
    );

    // Post-reset snapshot.
    let mut n: c_int = 0;
    let obs_ptr = ofenv_obs(env, &mut n as *mut c_int);
    let obs_n = n as usize;
    let mut mn: c_int = 0;
    let mask_ptr = ofenv_mask(env, &mut mn as *mut c_int);
    let mask_n = mn as usize;
    if obs_ptr.is_null() || mask_ptr.is_null() {
        eprintln!("ofenv_obs/ofenv_mask returned NULL: {}", last_error());
        std::process::exit(1);
    }
    let obs = unsafe { std::slice::from_raw_parts(obs_ptr, obs_n) };
    let mask = unsafe { std::slice::from_raw_parts(mask_ptr, mask_n) };
    println!(
        "reset obs: n={} nonzero={} scalars_head={:?}",
        obs_n,
        obs.iter().filter(|v| **v != 0.0).count(),
        &obs[..12]
    );
    println!(
        "reset mask: n={} nonzero={}",
        mask_n,
        mask.iter().filter(|v| **v != 0.0).count()
    );
    // Exercise ofenv_tiles() too (packed owner/fallout/defense state).
    let mut tn: c_int = 0;
    let tiles_ptr = ofenv_tiles(env, &mut tn as *mut c_int);
    if tiles_ptr.is_null() {
        eprintln!("ofenv_tiles returned NULL: {}", last_error());
        std::process::exit(1);
    }
    let tiles = unsafe { std::slice::from_raw_parts(tiles_ptr, tn as usize) };
    let owned = tiles
        .iter()
        .filter(|t| **t & openfront_engine::puffer_ffi::OFENV_TILE_OWNER_MASK != 0)
        .count();
    println!(
        "reset tiles: n={} owned={} sample={:?}",
        tn,
        owned,
        &tiles[..3.min(tiles.len())]
    );

    let empty = CString::new("[]").unwrap();
    let mut terminal_count = 0usize;
    let mut total_reward = 0.0f64;
    let mut last_tick = 0i64;
    const STEPS: usize = 200;
    for step in 1..=STEPS {
        if ofenv_step(env, empty.as_ptr()) != 0 {
            eprintln!("ofenv_step {step} FAILED: {}", last_error());
            std::process::exit(1);
        }
        let r = ofenv_reward(env, 0);
        let t = ofenv_terminal(env, 0);
        total_reward += r;
        terminal_count += t as usize;
        let meta = meta_json(env);
        let v: serde_json::Value = serde_json::from_str(&meta).unwrap_or(serde_json::Value::Null);
        last_tick = v["tick"].as_i64().unwrap_or(last_tick);
        if step <= 3 || step % 25 == 0 || step == STEPS {
            println!(
                "step {:3}: reward={:+.6} terminal={} tick={} spawn_phase={} winner={}",
                step,
                r,
                t,
                v["tick"].as_i64().unwrap_or(-1),
                v["spawn_phase"],
                v["winner"],
            );
        }
    }

    // Final buffers.
    let obs_ptr = ofenv_obs(env, &mut n as *mut c_int);
    let obs_n = n as usize;
    let mask_ptr = ofenv_mask(env, &mut mn as *mut c_int);
    let mask_n = mn as usize;
    let mask = unsafe { std::slice::from_raw_parts(mask_ptr, mask_n) };
    let obs = unsafe { std::slice::from_raw_parts(obs_ptr, obs_n) };

    println!("\n--- final sizes ---");
    println!("obs_size={obs_n} (per agent {OFENV_OBS_PER_AGENT})");
    println!("mask_size={mask_n} (per agent {OFENV_MASK_PER_AGENT})");
    println!(
        "obs nonzero={} total_reward={:+.6} terminal_count={terminal_count}/{STEPS} tick={last_tick}",
        obs.iter().filter(|v| **v != 0.0).count(),
        total_reward
    );

    println!("\n--- mask nonzero entries per action (agent 0) ---");
    let names = ["noop", "attack", "expand", "boat", "build", "launch_nuke",
                 "alliance_request", "alliance_reject", "break_alliance", "donate_gold",
                 "donate_troops", "embargo", "retreat", "spawn", "upgrade_structure",
                 "move_warship", "cancel_boat", "delete_unit", "embargo_stop",
                 "target_player", "alliance_extension"];
    for a in 0..OFENV_MASK_ACTIONS_N {
        let bit = mask[a];
        let pn = row_nonzero(
            mask,
            0,
            OFENV_MASK_PTARGET_OFF,
            a,
            OFENV_MASK_PTARGET_N / OFENV_MASK_ACTIONS_N,
        );
        let un = row_nonzero(
            mask,
            0,
            OFENV_MASK_UTARGET_OFF,
            a,
            OFENV_MASK_UTARGET_N / OFENV_MASK_ACTIONS_N,
        );
        println!(
            "  a={a:2} {:<18} action_bit={:.0} ptarget_nonzero={pn} utarget_nonzero={un}",
            names.get(a).copied().unwrap_or("?"),
            bit
        );
    }

    println!("\n--- meta head ---");
    let meta = meta_json(env);
    // Keep the head short but complete enough to eyeball.
    println!("{}", &meta[..meta.len().min(1200)]);

    // --- phase 2 (opt-in via PUFFER_SMOKE_SPAWN=1): prove the mask is real
    // legality, not just the spawn-phase stub. Pick spawn candidates from the
    // mask's legal_tile plane, spawn once, then step with empty intents.
    if std::env::var("PUFFER_SMOKE_SPAWN").as_deref() == Ok("1") {
        let (gw, gh) = {
            let meta = meta_json(env);
            let v: serde_json::Value =
                serde_json::from_str(&meta).unwrap_or(serde_json::Value::Null);
            (
                v["gw"].as_u64().unwrap_or(0) as usize,
                v["gh"].as_u64().unwrap_or(0) as usize,
            )
        };
        let width = {
            let meta = meta_json(env);
            let v: serde_json::Value =
                serde_json::from_str(&meta).unwrap_or(serde_json::Value::Null);
            v["width"].as_u64().unwrap_or(0) as usize
        };
        let mask_ptr = ofenv_mask(env, &mut mn as *mut c_int);
        let mask = unsafe { std::slice::from_raw_parts(mask_ptr, mn as usize) };
        let tile_plane =
            &mask[openfront_engine::puffer_ffi::OFENV_MASK_TILE_OFF
                ..openfront_engine::puffer_ffi::OFENV_MASK_TILE_OFF
                    + openfront_engine::puffer_ffi::OFENV_MASK_TILE_N];
        let gw_max = openfront_engine::puffer_ffi::OFENV_GW_MAX as usize;
        let mut candidates: Vec<u64> = Vec::new();
        'outer: for gy in 0..gh {
            for gx in 0..gw {
                if tile_plane[gy * gw_max + gx] == 0.0 {
                    continue;
                }
                // Center tile of the region; the engine validates the exact
                // spawn tile (land, unowned, passable) itself.
                let tile = ((gy * 8 + 4) as u64) * width as u64 + (gx * 8 + 4) as u64;
                candidates.push(tile);
                if candidates.len() >= 8 {
                    break 'outer;
                }
            }
        }
        println!(
            "\n--- phase 2: PUFFER_SMOKE_SPAWN=1, {} legal_tile regions, spawning ---",
            candidates.len()
        );
        for (i, tile) in candidates.iter().enumerate() {
            let intent = CString::new(format!(r#"[{{"type":"spawn","tile":{tile}}}]"#)).unwrap();
            if ofenv_step(env, intent.as_ptr()) != 0 {
                eprintln!("phase 2 spawn step FAILED: {}", last_error());
                std::process::exit(1);
            }
            let meta = meta_json(env);
            let v: serde_json::Value =
                serde_json::from_str(&meta).unwrap_or(serde_json::Value::Null);
            let on_map = v["agents"][0]["on_map"].as_bool().unwrap_or(false);
            println!("  spawn candidate {i} tile={tile}: on_map={on_map} tick={}", v["tick"]);
            if on_map {
                break;
            }
        }
        // A few empty-intent steps on the spawned agent (territory/troops grow).
        for _ in 0..10 {
            if ofenv_step(env, empty.as_ptr()) != 0 {
                eprintln!("phase 2 step FAILED: {}", last_error());
                std::process::exit(1);
            }
        }
        let obs_ptr = ofenv_obs(env, &mut n as *mut c_int);
        let obs = unsafe { std::slice::from_raw_parts(obs_ptr, n as usize) };
        let mask_ptr = ofenv_mask(env, &mut mn as *mut c_int);
        let mask = unsafe { std::slice::from_raw_parts(mask_ptr, mn as usize) };
        println!(
            "  post-spawn obs scalars={:?}",
            &obs[..openfront_engine::puffer_ffi::OFENV_OBS_SCALARS_N]
        );
        for a in 0..OFENV_MASK_ACTIONS_N {
            let pn = row_nonzero(
                mask,
                0,
                OFENV_MASK_PTARGET_OFF,
                a,
                OFENV_MASK_PTARGET_N / OFENV_MASK_ACTIONS_N,
            );
            let un = row_nonzero(
                mask,
                0,
                OFENV_MASK_UTARGET_OFF,
                a,
                OFENV_MASK_UTARGET_N / OFENV_MASK_ACTIONS_N,
            );
            println!(
                "  post-spawn a={a:2} {:<18} action_bit={:.0} ptarget_nonzero={pn} utarget_nonzero={un}",
                names.get(a).copied().unwrap_or("?"),
                mask[a]
            );
        }
        println!("  post-spawn meta: {}", &meta_json(env)[..400.min(meta_json(env).len())]);
    }

    ofenv_destroy(env);
    println!("\nofenv_destroy ok");
}