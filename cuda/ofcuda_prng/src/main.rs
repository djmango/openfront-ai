//! `ofcuda_prng` - GPU (cuda-oxide) side of the PRNG + spawn-tile parity level.
//!
//! Goal: prove that a Rust CUDA kernel reproduces the engine's `PseudoRandom`
//! stream and the engine's spawn-tile selection bit-for-bit, by comparing
//! element by element against a dump produced by the *real* engine
//! (`rust/engine/src/bin/prng_dump.rs`, which drives an actual `RlSession`).
//!
//! Kernels (one thread per output element - `DisjointSlice::get_mut` only
//! hands a thread its own index, so each thread replays the short PRNG prefix
//! it needs):
//!   * `stream24`   - the first 24 `next()` values as raw u32
//!   * `draws16`    - 16 `next_int(min,max)` per range
//!   * `sem`        - `chance` / `rand_element` (empty-list rule included) /
//!                    `shuffle_array`, plus the 3 following raw draws
//!   * `pstream24`  - the same stream seeded by `simple_hash(player_id)`
//!   * `hash_strings` - `simple_hash` itself, over the real game/player ids
//!   * `spawn_select` - the whole spawn selection: `rand_tile`'s two draws per
//!                    attempt, `is_land`/`has_owner`/`is_border`, the
//!                    min-distance rule, the BFS footprint and the 1000-attempt
//!                    loop. Jobs are the 2 bots + 1 human of each seed.
//!
//! The four traps the engine's PRNG has, and how this port handles them:
//!   1. the **12 warm-up draws** in `PseudoRandom::new` (`prng.rs:30-32`) - done
//!      in `Prng::new` on the device;
//!   2. `next()` is `(t as u32) / 2^32`, a **u32 division**, not an f64 one
//!      (`prng.rs:44`) - the device does `next_u32() as f64 / 4294967296.0`,
//!      which is the same exact value;
//!   3. `rand_element` on an **empty list draws nothing** (`prng.rs:86-92`) -
//!      `sem` reproduces the rule and the following draws catch a port that
//!      consumed one anyway;
//!   4. `shuffle_array` does exactly **len - 1** draws (`prng.rs:103-110`).
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use ofcuda_prng::{
    PortOutput, Reference, SpawnResult, compare, compare_ports, cpu_port_output, format_report,
    hash_strings, hex, load_map_normal, map_dir, parse_reference, spawn_jobs,
};
use std::path::PathBuf;


// The CUDA kernel module lives in `kernels.rs` so both bins (`ofcuda_prng` and
// `spawnall`) embed *literally the same* device code instead of a second copy.
include!("kernels.rs");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let ref_path = args.next().unwrap_or_else(|| "reference.txt".to_string());
    let repo_root = PathBuf::from("/opt/data/workspaces/skg/openfront-ai");

    let text = std::fs::read_to_string(&ref_path)?;
    let r: Reference = parse_reference(&text)?;
    let map_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| map_dir(&repo_root, &r.map));
    let map = load_map_normal(&map_path)?;
    println!(
        "# gpu: map {} {}x{}  terrain_bytes {}",
        map_path.display(),
        map.width,
        map.height,
        map.terrain.len()
    );

    let n_seeds = r.seeds.len();
    let seed_hashes: Vec<i32> = r.seeds.iter().map(|s| s.game_hash).collect();
    let ranges: Vec<i32> = vec![0, 100, 40, 80, 0, 7, 0, 5];
    let pstream_hashes: Vec<i32> = r.seeds.iter().map(|s| s.pstream_hash).collect();

    // Hash arena: the real ids the engine used, in `hash_strings` order.
    let strings = hash_strings(&r);
    let mut arena: Vec<u8> = Vec::new();
    let mut off: Vec<u32> = vec![0];
    for s in &strings {
        arena.extend_from_slice(s.as_bytes());
        off.push(arena.len() as u32);
    }

    // Spawn jobs: per job the owner overrides and the already-placed centres.
    let jobs = spawn_jobs(&r);
    let mut job_seeds: Vec<i32> = Vec::new();
    let mut job_explicit: Vec<u32> = Vec::new();
    let mut owner_tiles: Vec<u32> = Vec::new();
    let mut owner_ids: Vec<u16> = Vec::new();
    let mut owner_off: Vec<u32> = vec![0];
    let mut prev: Vec<u32> = Vec::new();
    let mut prev_off: Vec<u32> = vec![0];
    for s in r.seeds.iter() {
        for sp in s.spawns.iter() {
            job_seeds.push(sp.seed);
            job_explicit.push(u32::MAX);
            for (t, v) in &sp.owners {
                owner_tiles.push(*t);
                owner_ids.push(*v);
            }
            owner_off.push(owner_tiles.len() as u32);
            for t in &sp.prev {
                prev.push(*t);
            }
            prev_off.push(prev.len() as u32);
        }
        if let Some(h) = &s.human {
            job_seeds.push(h.seed);
            job_explicit.push(h.tile);
            owner_off.push(owner_tiles.len() as u32);
            prev_off.push(prev.len() as u32);
        }
    }
    assert_eq!(job_seeds.len(), jobs.len());

    let ctx = CudaContext::new(0)?;
    // Proof that the kernels really execute on the GPU: the host asks the
    // driver which device it is about to use, and every value compared below is
    // read back out of a `DeviceBuffer` that only a kernel launch wrote.
    let dev = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,driver_version", "--format=csv,noheader"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    println!("# device: {dev}");
    let stream = ctx.default_stream();
    // SAFETY: this package owns the embedded device bundle for `kernels`.
    let module = unsafe { kernels::load(&ctx)? };

    let d_seeds = DeviceBuffer::from_host(&stream, &seed_hashes)?;
    let d_ranges = DeviceBuffer::from_host(&stream, &ranges)?;
    let d_pstream = DeviceBuffer::from_host(&stream, &pstream_hashes)?;
    let d_arena = DeviceBuffer::from_host(&stream, &arena)?;
    let d_off = DeviceBuffer::from_host(&stream, &off)?;
    let d_terrain = DeviceBuffer::from_host(&stream, &map.terrain)?;
    let d_owner_tiles = DeviceBuffer::from_host(&stream, &owner_tiles)?;
    let d_owner_ids = DeviceBuffer::from_host(&stream, &owner_ids)?;
    let d_owner_off = DeviceBuffer::from_host(&stream, &owner_off)?;
    let d_prev = DeviceBuffer::from_host(&stream, &prev)?;
    let d_prev_off = DeviceBuffer::from_host(&stream, &prev_off)?;
    let d_job_seeds = DeviceBuffer::from_host(&stream, &job_seeds)?;
    let d_job_explicit = DeviceBuffer::from_host(&stream, &job_explicit)?;

    let blk = |n: usize| -> Result<LaunchConfig1D, Box<dyn std::error::Error>> {
        Ok(LaunchConfig1D::new((n as u32).div_ceil(64), 64, 0))
    };

    // --- (a) 24 raw u32 per seed -------------------------------------------
    let mut d_stream = DeviceBuffer::<u32>::zeroed(&stream, n_seeds * 24)?;
    let p = module.prepare_stream24(blk(n_seeds * 24)?)?;
    module.stream24(&stream, &p, &d_seeds, &mut d_stream)?;
    let gpu_stream = d_stream.to_host_vec(&stream)?;

    // --- (b) next_int draws -------------------------------------------------
    let mut d_draws = DeviceBuffer::<i32>::zeroed(&stream, n_seeds * 4 * 16)?;
    let p = module.prepare_draws16(blk(n_seeds * 4 * 16)?)?;
    module.draws16(&stream, &p, &d_seeds, &d_ranges, &mut d_draws)?;
    let gpu_draws = d_draws.to_host_vec(&stream)?;

    // --- (c) semantics ------------------------------------------------------
    let mut d_sem = DeviceBuffer::<u32>::zeroed(&stream, n_seeds * 54)?;
    let p = module.prepare_sem(blk(n_seeds * 54)?)?;
    module.sem(&stream, &p, &d_seeds, &mut d_sem)?;
    let gpu_sem = d_sem.to_host_vec(&stream)?;

    // --- (d) simple_hash(player_id)-seeded stream ---------------------------
    let mut d_ps = DeviceBuffer::<u32>::zeroed(&stream, n_seeds * 24)?;
    let p = module.prepare_pstream24(blk(n_seeds * 24)?)?;
    module.pstream24(&stream, &p, &d_pstream, &mut d_ps)?;
    let gpu_ps = d_ps.to_host_vec(&stream)?;

    // --- (e) simple_hash over the real ids ---------------------------------
    let mut d_hash = DeviceBuffer::<u32>::zeroed(&stream, strings.len())?;
    let p = module.prepare_hash_strings(blk(strings.len())?)?;
    module.hash_strings(&stream, &p, &d_arena, &d_off, &mut d_hash)?;
    let gpu_hash = d_hash.to_host_vec(&stream)?;

    // --- (f) spawn selection on the GPU ------------------------------------
    let mut d_spawn = DeviceBuffer::<u32>::zeroed(&stream, job_seeds.len() * 10)?;
    let p = module.prepare_spawn_select(blk(job_seeds.len() * 10)?)?;
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
        30,
        &d_job_seeds,
        &d_job_explicit,
        &mut d_spawn,
    )?;
    let gpu_spawn = d_spawn.to_host_vec(&stream)?;

    // ---------------------------------------------------------------------
    // Assemble the port output
    // ---------------------------------------------------------------------
    let mut gpu = PortOutput::default();
    for si in 0..n_seeds {
        gpu.streams.push(gpu_stream[si * 24..si * 24 + 24].to_vec());
        let mut per_range = Vec::new();
        for ri in 0..4 {
            let j = si * 4 + ri;
            per_range.push(gpu_draws[j * 16..j * 16 + 16].to_vec());
        }
        gpu.draws.push(per_range);
        let b = si * 54;
        gpu.chance100
            .push(gpu_sem[b..b + 16].iter().map(|v| *v as u8).collect());
        gpu.chance2
            .push(gpu_sem[b + 16..b + 32].iter().map(|v| *v as u8).collect());
        gpu.chance1_all.push(gpu_sem[b + 32] == 1);
        gpu.rand7
            .push(gpu_sem[b + 33..b + 36].iter().map(|v| *v as i32).collect());
        gpu.rand7_next.push(gpu_sem[b + 36..b + 39].to_vec());
        gpu.rand_empty_consumed.push(gpu_sem[b + 39] as i32);
        gpu.rand_empty_next.push(gpu_sem[b + 40..b + 43].to_vec());
        gpu.shuffle8
            .push(gpu_sem[b + 43..b + 51].iter().map(|v| *v as i32).collect());
        gpu.shuffle8_next.push(gpu_sem[b + 51..b + 54].to_vec());
        gpu.pstream.push(gpu_ps[si * 24..si * 24 + 24].to_vec());
    }
    for (i, s) in strings.iter().enumerate() {
        gpu.hashes.push((s.clone(), gpu_hash[i] as i32));
    }
    let mut at = 0usize;
    for s in &r.seeds {
        let mut v = Vec::new();
        for _ in &s.spawns {
            v.push(spawn_from_words(&gpu_spawn[at * 10..at * 10 + 10]));
            at += 1;
        }
        if s.human.is_some() {
            v.push(spawn_from_words(&gpu_spawn[at * 10..at * 10 + 10]));
            at += 1;
        }
        gpu.spawns.push(v);
    }

    // ---------------------------------------------------------------------
    // Compare: GPU vs engine, then CPU vs engine, then GPU vs CPU
    // ---------------------------------------------------------------------
    let mut rep = compare(&r, &gpu);
    let cpu = cpu_port_output(&r, &map);
    let cpu_rep = compare(&r, &cpu);
    rep.cross = compare_ports(&gpu, &cpu);

    let mut log = String::new();
    log.push_str(&format!("# device: {dev}\n"));
    log.push_str(&format!("# gpu reference: {ref_path}\n"));
    log.push_str(&format_report(&r, &rep, "CUDA kernel"));
    log.push_str(&format!(
        "--- CPU companion (same crate, no CUDA) ---\n{}",
        format_report(&r, &cpu_rep, "CPU port")
    ));

    // Raw tables, so the comparison can be read without re-deriving anything.
    log.push_str("--- raw streams (engine vs GPU) ---\n");
    for (si, s) in r.seeds.iter().enumerate() {
        log.push_str(&format!("seed{si} engine {}\n", hex(&s.stream)));
        log.push_str(&format!("seed{si} gpu    {}\n", hex(&gpu.streams[si])));
    }
    log.push_str("--- spawn table (engine vs GPU) ---\n");
    for (si, s) in r.seeds.iter().enumerate() {
        for (bi, sp) in s.spawns.iter().enumerate() {
            log.push_str(&format!(
                "seed{si} bot{bi} engine_tile={} gpu_tile={} engine_seed={}\n",
                sp.tile, gpu.spawns[si][bi].tile, sp.seed
            ));
        }
        if let Some(h) = &s.human {
            let bi = s.spawns.len();
            log.push_str(&format!(
                "seed{si} human engine_tile={} gpu_tile={}\n",
                h.tile, gpu.spawns[si][bi].tile
            ));
        }
    }

    let ok = rep.all_match() && cpu_rep.all_match() && rep.cross.matched == rep.cross.total;
    log.push_str(&format!("GPU_VS_ENGINE_BIT_EXACT {}\n", rep.all_match()));
    log.push_str(&format!("CPU_VS_ENGINE_BIT_EXACT {}\n", cpu_rep.all_match()));
    log.push_str(&format!(
        "GPU_VS_CPU_CROSS {}/{}\n",
        rep.cross.matched, rep.cross.total
    ));
    log.push_str(&format!("ALL_BIT_EXACT {ok}\n"));

    print!("{log}");
    std::fs::write("comparison_gpu.txt", &log)?;
    if !ok {
        eprintln!("FAILED: port differs from the engine reference");
        std::process::exit(1);
    }
    Ok(())
}

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
