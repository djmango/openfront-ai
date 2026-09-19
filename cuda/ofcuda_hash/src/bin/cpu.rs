//! CPU-only companion to `ofcuda_hash`: the same dump, the same serializer, the
//! same hash definition, no CUDA in the dependency graph at all.
//!
//! It exists so the GPU numbers have an independent reference that does not
//! share a single line of device code, and so the harness can be reproduced on
//! a host without a GPU.

use ofcuda_hash::{
    FNV_OFFSET_BASIS, FNV_PRIME, fnv1a_bytes, fnv1a_bytes_chunked_ordered, hex64, load_dump,
    naive_combine, serialize_u16_le, state_hash, terrain_hash,
};
use std::path::PathBuf;

const EXPECTED_TERRAIN_HASH: u64 = 0xebff_a87c_2568_cc58;
const EXPECTED_POST_RESET_STATE_HASH: u64 = 0x6334_dfb9_8045_3d25;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dump_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: ofcuda_hash_cpu <dump-dir> [--max-ticks N]")?;
    let mut max_ticks = usize::MAX;
    let argv: Vec<String> = std::env::args().skip(2).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--max-ticks" => {
                max_ticks = argv[i + 1].parse()?;
                i += 2;
            }
            other => return Err(format!("unknown arg {other}").into()),
        }
    }

    let dump = load_dump(&dump_dir)?;
    let n = dump.header.tiles;
    let n_ticks = dump.ticks.len().min(max_ticks);

    println!("# ofcuda_hash_cpu - CPU reference (no CUDA in the dependency graph)");
    println!("# dump dir        {}", dump.dir.display());
    println!("# mode            {}", dump.header.mode);
    println!(
        "# grid            {}x{} tiles {}",
        dump.header.width, dump.header.height, n
    );
    println!(
        "# fnv             1a64 offset={:#018x} prime={:#018x}",
        FNV_OFFSET_BASIS, FNV_PRIME
    );

    let mut failures = 0u64;

    // --- terrain ---------------------------------------------------------
    let t = terrain_hash(&dump.terrain);
    println!();
    println!("== terrain hash ==");
    println!("terrain_hash_cpu_reference   {}", hex64(t));
    println!(
        "terrain_hash_engine_expected {}",
        hex64(dump.header.terrain_hash)
    );
    println!(
        "terrain_hash_published_const {}",
        hex64(EXPECTED_TERRAIN_HASH)
    );
    println!("terrain_matches_engine       {}", t == dump.header.terrain_hash);
    println!(
        "terrain_matches_published    {}",
        t == EXPECTED_TERRAIN_HASH
    );
    if t != EXPECTED_TERRAIN_HASH {
        failures += 1;
    }

    // --- post-reset state -------------------------------------------------
    if let Some(post) = &dump.post_reset {
        let h = state_hash(post);
        println!();
        println!("== post-reset state hash ==");
        println!("post_reset_state_hash_cpu       {}", hex64(h));
        println!(
            "post_reset_state_engine_expected {}",
            hex64(dump.header.post_reset_state_hash.unwrap_or(0))
        );
        println!(
            "post_reset_state_published_const {}",
            hex64(EXPECTED_POST_RESET_STATE_HASH)
        );
        // Serializer round-trip: u16 plane -> LE bytes -> FNV.
        let bytes = serialize_u16_le(post);
        println!(
            "post_reset_via_serializer       {}",
            hex64(fnv1a_bytes(FNV_OFFSET_BASIS, &bytes))
        );
        if h != EXPECTED_POST_RESET_STATE_HASH {
            failures += 1;
        }
    }

    // --- chunk-count invariance (non-degenerate probes) --------------------
    println!();
    println!("== chunk-count invariance (non-degenerate probes) ==");
    let synthetic = ofcuda_hash::synthetic_state(n);
    let mut densest = (0usize, 0usize);
    for i in 0..dump.ticks.len() {
        let b = serialize_u16_le(&dump.plane(i));
        let nz = ofcuda_hash::nonzero_bytes(&b);
        if nz > densest.1 {
            densest = (i, nz);
        }
    }
    let real_plane = dump.plane(densest.0);
    let real_label = format!("real plane tick {}", dump.ticks[densest.0].tick);
    for (label, plane) in [
        ("synthetic".to_string(), &synthetic),
        (real_label, &real_plane),
    ] {
        let bytes = serialize_u16_le(plane);
        let serial = fnv1a_bytes(FNV_OFFSET_BASIS, &bytes);
        let nz = ofcuda_hash::nonzero_bytes(&bytes);
        println!(
            "probe {label} nonzero_bytes {nz}/{} serial {}",
            bytes.len(),
            hex64(serial)
        );
        let mut invariant = true;
        for chunk in [1usize, 2, 3, 5, 7, 64, 1024, 4096, 65536, 250_000, 1_000_000] {
            let v = fnv1a_bytes_chunked_ordered(FNV_OFFSET_BASIS, &bytes, chunk);
            let chunks = bytes.len().div_ceil(chunk);
            let ok = v == serial;
            invariant &= ok;
            println!(
                "  chunk_bytes={chunk:<8} chunks={chunks:<8} hash={} equal={ok}",
                hex64(v)
            );
        }
        println!("  chunk_count_invariance_cpu {invariant}");
        if !invariant {
            failures += 1;
        }
        let mut naive_disagreements = 0;
        for chunk in [1usize, 2, 3, 5, 7, 64, 1024, 4096, 65536, 250_000, 1_000_000] {
            if naive_combine(FNV_OFFSET_BASIS, &bytes, chunk) != serial {
                naive_disagreements += 1;
            }
        }
        println!("  naive_combine_wrong_by_design disagreeing_chunk_sizes={naive_disagreements}/11");
    }

    // --- per-tick ---------------------------------------------------------
    println!();
    println!("== per-tick state hash: CPU reference vs engine ({} ticks) ==", n_ticks);
    let mut matches = 0usize;
    let mut first = None;
    for idx in 0..n_ticks {
        let tl = &dump.ticks[idx];
        let h = state_hash(&dump.plane(idx));
        let ok = h == tl.state_hash;
        if ok {
            matches += 1;
        } else if first.is_none() {
            first = Some((tl.tick, h, tl.state_hash));
        }
        println!(
            "  tick {:<8} cpu {:<20} engine {:<20} {}",
            tl.tick,
            hex64(h),
            hex64(tl.state_hash),
            if ok { "ok" } else { "MISMATCH" }
        );
    }
    println!("per_tick_checked  {n_ticks}");
    println!("per_tick_matches  {matches}");
    match first {
        None => println!("first_disagreement none"),
        Some((t, c, e)) => {
            println!(
                "first_disagreement tick={t} cpu={} expected={}",
                hex64(c),
                hex64(e)
            );
            failures += 1;
        }
    }
    if matches != n_ticks {
        failures += 1;
    }

    println!();
    println!("verdict {}", if failures == 0 { "PASS" } else { "FAIL" });
    if failures != 0 {
        std::process::exit(1);
    }
    Ok(())
}