//! `bench` - HOST-side attribution benchmark for the composed tick.
//!
//! Added for the speed measurement only. It does NOT touch `main.rs`/`lib.rs`
//! and it launches no kernel: the composed tick in this crate is a host
//! implementation, so this binary exists to say *where* the host time goes, not
//! to pretend any of it is device work.
//!
//!     cargo build --release --offline --bin bench
//!     ./target/release/bench /tmp/envdump/fresh.ndjson <map-dir>
//!
//! Measures, per selected tick, with the real record data:
//!   * `recon`  - the driver's per-tick reconstruction: clone the players'
//!                `owned_tiles`/`border_order`/`owned_order` (what `players_at`
//!                does) and build the `w*h` u16 plane from them (`state_plane`);
//!   * `planeb` - one `state_plane` build alone;
//!   * `fnv`    - one `state_hash` (FNV-1a-64 over `w*h` u16 = 2 MB of LE bytes);
//!   * `tick`   - `Attack::refresh` (the init) + `Attack::tick` (the engine's
//!                `AttackExecution::tick` pop loop) on an attack shaped by the
//!                record's own owner border and plane.

use std::path::PathBuf;
use std::time::Instant;

use ofcuda_env::{Attack, state_hash, state_plane};

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let dump_path = PathBuf::from(a.first().cloned().unwrap_or("/tmp/envdump/fresh.ndjson".into()));
    let map_dir = PathBuf::from(
        a.get(1)
            .cloned()
            .unwrap_or("/opt/data/workspaces/skg/openfront-ai/openfront/resources/maps/pangaea".into()),
    );

    let t0 = Instant::now();
    let map = ofcuda_tick::load_map(&map_dir).expect("map");
    let (w, h) = (map.width, map.height);
    let load_map_ms = ms(t0.elapsed());

    let t0 = Instant::now();
    let dump = ofcuda_tick::parse_dump(&dump_path).expect("dump");
    let parse_ms = ms(t0.elapsed());

    println!("# ofcuda_env HOST attribution bench (no kernel launched)");
    println!("# dump  {} ({} tick records)", dump_path.display(), dump.len());
    println!("# map   {w}x{h}  plane = {} u16 words = {} bytes", w * h, 2 * w * h);
    println!("# load_map {load_map_ms:.1} ms | parse_dump {parse_ms:.1} ms (one-off, outside the tick loop)");
    println!();

    // The driver's reconstruction is exactly this: for every player at the tick,
    // clone owned/border/ownedOrder (players_at) and fold owned into the plane.
    let ticks: [u32; 4] = [320, 600, 900, 1200];
    let iters = 200usize;

    println!(
        "{:>5} {:>7} {:>7} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}",
        "tick", "players", "border∑", "owned∑", "recon_ms", "planeb_ms", "fnv_ms", "tick_ms", "refr_ms"
    );

    for &t in &ticks {
        let Some(ps) = dump.get(&t) else { continue };
        let mut recs: Vec<ofcuda_tick::PlayerRec> = ps.values().cloned().collect();
        recs.sort_by_key(|p| p.small_id);
        let border_sum: usize = recs.iter().map(|p| p.border_order.len()).sum();
        let owned_sum: usize = recs.iter().map(|p| p.owned_tiles.len()).sum();

        // every player's owned set, in the order `players_at` produces them
        let owned_lists: Vec<(u32, Vec<u32>)> = recs
            .iter()
            .map(|p| (p.small_id, p.owned_tiles.clone()))
            .collect();

        // --- recon: clone the three vectors per player, then build the plane ---
        let t0 = Instant::now();
        for _ in 0..iters {
            let mut cloned: Vec<(u32, Vec<u32>, Vec<u32>, Vec<u32>)> = Vec::with_capacity(recs.len());
            for p in &recs {
                cloned.push((
                    p.small_id,
                    p.owned_tiles.clone(),
                    p.border_order.clone(),
                    p.owned_order.clone(),
                ));
            }
            let pl: Vec<(u32, Vec<u32>)> = cloned.into_iter().map(|(s, o, _, _)| (s, o)).collect();
            std::hint::black_box(state_plane(&pl, w, h));
        }
        let recon_ms = ms(t0.elapsed()) / iters as f64;

        // --- planeb: one plane build, no cloning ---
        let t0 = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(state_plane(&owned_lists, w, h));
        }
        let planeb_ms = ms(t0.elapsed()) / iters as f64;

        let plane = state_plane(&owned_lists, w, h);
        let t0 = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(state_hash(&plane));
        }
        let fnv_ms = ms(t0.elapsed()) / iters as f64;

        // --- the engine tick: init (refresh from the owner's border) + one tick ---
        // The owner used is the record's largest-bordered player (scenario 2's
        // expansion), with troops chosen so the budget is of the window's size.
        let owner = recs
            .iter()
            .filter(|p| !p.border_order.is_empty())
            .max_by_key(|p| p.border_order.len())
            .cloned()
            .unwrap();
        let mut snow = Attack::new(
            owner.small_id as u16,
            0,
            true,
            1363.0,
            ofcuda_tick::SEED,
        );
        let t0 = Instant::now();
        snow.refresh(&owner.border_order, &plane, &map.terrain, w, h, t - 1);
        let refr_ms = ms(t0.elapsed());
        let t_before = snow.troops;
        let t0 = Instant::now();
        for _ in 0..iters {
            let mut atk = Attack::new(owner.small_id as u16, 0, true, t_before, ofcuda_tick::SEED);
            atk.refresh(&owner.border_order, &plane, &map.terrain, w, h, t - 1);
            std::hint::black_box(atk.tick(&plane, &map.terrain, w, h, t, &owner.border_order, 0.0, false));
        }
        let tick_ms = ms(t0.elapsed()) / iters as f64;
        let refr_iter_ms = tick_ms; // measured together; split by the standalone refresh below
        let _ = refr_iter_ms;

        println!(
            "{t:>5} {:>7} {border_sum:>7} {owned_sum:>9} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3}",
            recs.len(),
            recon_ms,
            planeb_ms,
            fnv_ms,
            tick_ms,
            refr_ms
        );
    }

    println!();
    println!("# columns: recon = players_at-equivalent clone + state_plane (what the driver pays per tick,");
    println!("#          twice per tick: it builds the plane at s AND at s+1); planeb = state_plane alone;");
    println!("#          fnv = one FNV-1a-64 over the 2 MB plane (the driver also hashes twice per tick);");
    println!("#          tick = Attack::refresh + Attack::tick for one attack (the engine expansion step);");
    println!("#          refr = the standalone refresh in ms (first init only)");
}
