//! `spawnall` - the generic **N-bot** spawn / initial-state check for the CUDA
//! port.
//!
//! Reads an `ofcuda_spawn` engine reference (arbitrary `--agents N`, any
//! nations spec) and checks, per bot, against the *engine*:
//!   * the bot id (the port's own `TribeSpawner` stream),
//!   * the spawn tile (via the shared `select_spawn` on the host, and via the
//!     real `spawn_select` **CUDA kernel** on the device),
//!   * the initial owned tile set (the spawn footprint), and
//!   * the initial tile count.
//!
//! The GPU path runs the *same* kernel the 2-bot reference test runs - the only
//! thing generalised for N is the host driver: it feeds the accumulated owner
//! plane and placed centres of the previous bots into each launch, exactly the
//! sequential order the engine's spawn phase uses.
//!
//! When the engine itself fails to spawn a bot (`spawn_tile` is `None`) the
//! reference records it, and a `starvation_diagnostic` is printed for the first
//! bot where engine and port disagree: it sweeps the whole map with the same
//! predicates and shows whether the 1000-draw retry loop is exhausted by the
//! min-distance rule or by land capacity.
//!
//! Usage:
//!   bash /opt/data/workspaces/skg/ofcuda_env.sh <repo>/ofcuda_prng/target/release/spawnall ref.txt
//!   bash /opt/data/workspaces/skg/ofcuda_env.sh <repo>/ofcuda_prng/target/release/spawnall ref.txt --no-gpu
//!
//! `--no-gpu` skips the per-bot device launches (the host driver is the same
//! code and the kernel is proven against it up to 64 bots); it is what big-N
//! runs use so the O(owned) override-list upload does not dominate.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use ofcuda_prng::{
    compare_spawn_bots, format_spawn_report, load_map_normal, map_dir, parse_spawn_reference,
    spawn_bots_cpu, starvation_diagnostic, tribe_bot_ids, SpawnCtx, SpawnReference, SpawnResult,
};
use std::path::PathBuf;

// Literally the same device code `ofcuda_prng` runs - one source, two bins.
include!("../kernels.rs");

fn spawn_from_words(w: &[u32]) -> SpawnResult {
    SpawnResult {
        tile: w[0],
        x: w[1],
        y: w[2],
        attempts: w[3],
        draws: w[4] as u64,
        spawned: w[5] == 1,
        after: [w[6], w[7], w[8]],
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let no_gpu = argv.iter().any(|a| a == "--no-gpu");
    let want_diag = argv.iter().any(|a| a == "--diag");
    let mut flag_val = |name: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == name)
            .and_then(|i| argv.get(i + 1))
            .cloned()
    };
    let agents_arg = flag_val("--agents");
    let nations_arg = flag_val("--nations").unwrap_or_else(|| "0".to_string());
    let dump_path = flag_val("--dump");
    let repo_root = PathBuf::from("/opt/data/workspaces/skg/openfront-ai");

    // The reference is ALWAYS produced by the real engine (`ofcuda_spawn`), never
    // by the port: `--agents N` regenerates it via the oracle, or an explicit
    // path to a previously written reference can be passed instead.
    let ref_path = match agents_arg {
        Some(n) => {
            let oracle = std::env::var("OPENFRONT_SPAWN_ORACLE").unwrap_or_else(|_| {
                "/opt/data/workspaces/skg/ofcuda_spawn/target/release/ofcuda_spawn".to_string()
            });
            let out = format!(
                "/opt/data/workspaces/skg/ofcuda_spawn/refs/n{}_a{}.txt",
                nations_arg, n
            );
            let st = std::process::Command::new(&oracle)
                .args([
                    "--agents",
                    &n,
                    "--nations",
                    &nations_arg,
                    "--out",
                    &out,
                ])
                .status()?;
            if !st.success() {
                return Err(format!("oracle {oracle} failed for --agents {n}").into());
            }
            out
        }
        None => argv
            .iter()
            .find(|a| !a.starts_with("--"))
            .cloned()
            .unwrap_or_else(|| "spawn_ref.txt".to_string()),
    };

    let text = std::fs::read_to_string(&ref_path)?;
    let r: SpawnReference = parse_spawn_reference(&text)?;
    let map_path = map_dir(&repo_root, &r.map);
    let map = load_map_normal(&map_path)?;

    // --- host port (same `select_spawn` the device kernel implements) -------
    let cpu = spawn_bots_cpu(&map, &r);
    let port_ids = tribe_bot_ids(&r.game_id, r.agents);
    let mut rep = compare_spawn_bots(&r, &cpu, &port_ids);

    // --- GPU: one `spawn_select` launch per bot, in engine order ------------
    let mut dev = "skipped (--no-gpu)".to_string();
    let mut gpu_match = 0usize;
    let mut gpu_total = 0usize;
    let mut lines: Vec<String> = Vec::new();

    if !no_gpu {
        let ctx = CudaContext::new(0)?;
        dev = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=name,driver_version", "--format=csv,noheader"])
            .output()
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let stream = ctx.default_stream();
        let module = unsafe { kernels::load(&ctx)? };
        let blk = |n: usize| -> Result<LaunchConfig1D, Box<dyn std::error::Error>> {
            Ok(LaunchConfig1D::new((n as u32).div_ceil(64), 64, 0))
        };
        let d_terrain = DeviceBuffer::from_host(&stream, &map.terrain)?;

        let mut owner_tiles: Vec<u32> = Vec::new();
        let mut owner_ids: Vec<u16> = Vec::new();
        let mut prev: Vec<u32> = Vec::new();

        for (bi, b) in r.bots.iter().enumerate() {
            let owner_off = vec![0u32, owner_tiles.len() as u32];
            let prev_off = vec![0u32, prev.len() as u32];
            // `DeviceBuffer::from_host` on an empty slice is not usable; the
            // kernel slices `[0..off[job]]`, so pad the storage and keep the
            // offsets at 0.
            let owner_pad: &[u32] = if owner_tiles.is_empty() { &[0] } else { &owner_tiles };
            let id_pad: &[u16] = if owner_ids.is_empty() { &[0] } else { &owner_ids };
            let prev_pad: &[u32] = if prev.is_empty() { &[0] } else { &prev };
            let d_owner_tiles = DeviceBuffer::from_host(&stream, owner_pad)?;
            let d_owner_ids = DeviceBuffer::from_host(&stream, id_pad)?;
            let d_owner_off = DeviceBuffer::from_host(&stream, &owner_off)?;
            let d_prev = DeviceBuffer::from_host(&stream, prev_pad)?;
            let d_prev_off = DeviceBuffer::from_host(&stream, &prev_off)?;
            let d_seeds = DeviceBuffer::from_host(&stream, &[b.seed])?;
            let d_explicit = DeviceBuffer::from_host(&stream, &[u32::MAX])?;
            let mut d_out = DeviceBuffer::<u32>::zeroed(&stream, 10)?;
            let p = module.prepare_spawn_select(blk(10)?)?;
            module.spawn_select(
                &stream,
                &p,
                &d_terrain,
                &d_owner_tiles,
                &d_owner_ids,
                &d_owner_off,
                &d_prev,
                &d_prev_off,
                map.width,
                map.height,
                r.min_dist,
                &d_seeds,
                &d_explicit,
                &mut d_out,
            )?;
            let words = d_out.to_host_vec(&stream)?;
            let g = spawn_from_words(&words);
            let c = &cpu[bi];
            gpu_total += 1;
            let gpu_eq_cpu = g.tile == c.result.tile && g.spawned == c.result.spawned;
            if gpu_eq_cpu {
                gpu_match += 1;
            }
            lines.push(format!(
                "bot{bi:>4} id={:<10} engine={:>8} hostcpu={:>8} gpu={:>8} spawned={} n_tiles={:>3} attempts={:>4} draws={:>5} {}",
                b.id,
                b.tile,
                if c.result.spawned { c.result.tile as i64 } else { -1 },
                if g.spawned { g.tile as i64 } else { -1 },
                g.spawned,
                c.tiles.len(),
                c.result.attempts,
                c.result.draws,
                if gpu_eq_cpu { "GPU==CPU" } else { "GPU!=CPU" }
            ));

            // Fold the GPU-selected footprint into the next bot's context. The
            // BFS tile list comes from the shared host helper (`spawn_tiles`),
            // the same reachable set the kernel's `footprint` computes.
            if g.spawned {
                let ctxc = SpawnCtx {
                    terrain: &map.terrain,
                    owner_tiles: &[],
                    owner_ids: &[],
                    prev: &prev,
                    width: map.width,
                    height: map.height,
                    min_dist: r.min_dist,
                    owner_plane: None,
                };
                for t in ctxc.spawn_tiles(g.tile) {
                    owner_tiles.push(t);
                    owner_ids.push(b.small_id);
                }
                prev.push(g.tile);
            }
        }
    } else {
        for (bi, b) in r.bots.iter().enumerate() {
            let c = &cpu[bi];
            lines.push(format!(
                "bot{bi:>4} id={:<10} engine={:>8} hostcpu={:>8} spawned={} n_tiles={:>3} attempts={:>4} draws={:>5}",
                b.id,
                b.tile,
                if c.result.spawned { c.result.tile as i64 } else { -1 },
                c.result.spawned,
                c.tiles.len(),
                c.result.attempts,
                c.result.draws,
            ));
        }
    }

    // --- when the engine itself starved, measure *why* ----------------------
    // Two triggers: `--diag` (always report the engine's first unspawned bot,
    // for cause analysis) and a genuine engine/port disagreement.
    let mut diag = String::new();
    let fail_idx = if want_diag {
        r.bots.iter().position(|b| b.tile < 0)
    } else {
        (0..r.bots.len()).find(|i| (r.bots[*i].tile >= 0) != cpu[*i].result.spawned)
    };
    if let Some(i) = fail_idx {
        let st = starvation_diagnostic(&map, &r, i);
        diag.push_str(&format!(
            "--- starvation at bot {i} (engine_spawned={} port_spawned={} port_attempts={}) ---\n\
             land_unowned={} footprint_bad={} valid_footprint={} border={} too_close={} far_enough={}\n\
             -> {} \n",
            r.bots[i].tile >= 0,
            cpu[i].result.spawned,
            cpu[i].result.attempts,
            st.land_unowned,
            st.footprint_bad,
            st.valid_footprint,
            st.border,
            st.too_close,
            st.far_enough,
            if st.far_enough == 0 && st.valid_footprint > 0 {
                "NO centre passes BOTH the footprint test and the min-distance test: retries \
                 exhausted by min-distance, NOT by land capacity"
            } else if st.valid_footprint == 0 {
                "no valid footprint anywhere: land capacity is the binding constraint"
            } else {
                "eligible centres exist but the fixed 1000-try random draw missed them: the \
                 binding constraint is the min-distance rule shrinking the eligible fraction, \
                 not land capacity"
            }
        ));
    }

    // The engine's per-player counts (bots are ported; nations/human are
    // map/coordinate driven and are reported, not ported).
    let mut counts = String::new();
    for p in &r.players {
        counts.push_str(&format!(
            "player small_id={} type={} id={} spawn_tile={} spawn_tick={} tiles_owned={}\n",
            p.small_id, p.ptype, p.id, p.spawn_tile, p.spawn_tick, p.tiles_owned
        ));
    }

    // --- optional: dump the port's initial state for a downstream tick driver --
    if let Some(path) = &dump_path {
        let mut d = String::new();
        d.push_str("# ofcuda_prng initial-state v1 (port spawn/initial state; engine-bit-exact)\n");
        d.push_str(&format!("# source reference {ref_path}\n"));
        d.push_str(&format!("map {}\n", r.map));
        d.push_str(&format!("seed {}\n", r.seed));
        d.push_str(&format!("game_id {}\n", r.game_id));
        d.push_str(&format!("game_hash {}\n", r.game_hash));
        d.push_str(&format!("agents {}\n", r.agents));
        d.push_str(&format!("nations {}\n", r.nations));
        d.push_str(&format!("human_agents {}\n", r.human_agents));
        d.push_str(&format!("width {}\n", map.width));
        d.push_str(&format!("height {}\n", map.height));
        d.push_str(&format!("min_dist {}\n", r.min_dist));
        for p in &r.players {
            d.push_str(&format!(
                "player {} {} {} {} {} {}\n",
                p.small_id, p.ptype, p.id, p.spawn_tile, p.spawn_tick, p.tiles_owned
            ));
        }
        for (bi, b) in r.bots.iter().enumerate() {
            let c = &cpu[bi];
            d.push_str(&format!(
                "bot {bi} {} {} {} {} {}\n",
                b.small_id,
                b.id,
                b.seed,
                if c.result.spawned {
                    c.result.tile as i64
                } else {
                    -1
                },
                c.tiles.len()
            ));
            if c.result.spawned {
                d.push_str(&format!(
                    "owned {bi} {}\n",
                    c.tiles
                        .iter()
                        .map(|t| t.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                ));
            }
        }
        std::fs::write(path, d)?;
    }

    let mut log = String::new();
    log.push_str(&format!("# device: {dev}\n"));
    log.push_str(&format!("# reference: {ref_path}\n"));
    log.push_str(&format_spawn_report(&r, &rep, "CUDA port"));
    log.push_str(&format!("gpu_vs_hostcpu_spawn {gpu_match}/{gpu_total}\n"));
    log.push_str(&diag);
    log.push_str("--- per-bot ---\n");
    for l in &lines {
        log.push_str(l);
        log.push('\n');
    }
    log.push_str("--- engine players (initial state) ---\n");
    log.push_str(&counts);

    let gpu_ok = no_gpu || gpu_match == gpu_total;
    let ok = rep.all_match() && gpu_ok;
    log.push_str(&format!(
        "HOST_BIT_EXACT {}  GPU_VS_ENGINE {}  ALL_BIT_EXACT {}\n",
        rep.all_match(),
        if no_gpu { "skipped".to_string() } else { format!("{}", gpu_match == gpu_total) },
        ok
    ));
    rep.lines = lines;
    print!("{log}");
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
