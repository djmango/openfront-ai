//! Breaks down the per-decision cost of the trainer's FFI path
//! (`ofenv_step` -> `rebuild_buffers`) against the cost of the simulation step
//! itself, using the same widgets the FFI layer calls:
//!
//!   * `RlSession::step`            - intent submission + N engine ticks
//!   * `obs_typed::entities_typed`  - per-player obs struct
//!   * `obs_typed::legality_typed`  - per-agent action legality
//!   * `ofcore::feat::make_lut`     - slot LUT over the live roster
//!   * owners plane build           - the `owners_slotted` loop (trimmed u8
//!                                    plane, LUT-mapped, one alloc per call)
//!   * `ofcore::feat::featurize`    - the 4270-float observation tensor
//!
//! Run:
//!   cargo run --release -p openfront-engine --example bench_ffi_overhead
use openfront_engine::obs_typed::{entities_typed, legality_typed};
use openfront_engine::rl::RlSession;
use openfront_engine::session::AGENT_CLIENT_ID;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

fn root() -> PathBuf {
    PathBuf::from(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .expect("engine crate manifest dir has a grandparent"),
    )
}

const OWNER_MASK: u16 = 0x0FFF; // mirror of puffer_ffi::OWNER_MASK

macro_rules! timeit {
    ($label:expr, $n:expr, $body:block) => {{
        let t0 = Instant::now();
        for _ in 0..$n {
            $body
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / $n as f64;
        println!("{:34} {:9.1} us/call", $label, us);
        us
    }};
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut map = "Pangaea".to_string();
    let mut bots = 3u32;
    let mut difficulty = "Easy".to_string();
    let mut iters = 200usize;
    let mut warmup = 200usize;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--map" => {
                map = argv[i + 1].clone();
                i += 2;
            }
            "--bots" => {
                bots = argv[i + 1].parse().unwrap();
                i += 2;
            }
            "--difficulty" => {
                difficulty = argv[i + 1].clone();
                i += 2;
            }
            "--iters" => {
                iters = argv[i + 1].parse().unwrap();
                i += 2;
            }
            "--warmup" => {
                warmup = argv[i + 1].parse().unwrap();
                i += 2;
            }
            other => {
                eprintln!("unknown flag {other}, ignoring");
                i += 1;
            }
        }
    }

    let r = root();
    let (mut session, _head, _ents, _legal, _terrain, _duo) = RlSession::reset(
        &r,
        &map,
        "ffi-overhead-0",
        bots,
        &difficulty,
        serde_json::Value::from(0u32),
        1,
    )
    .expect("RlSession::reset failed");

    for _ in 0..warmup {
        let _ = session.step(&[], 15);
    }

    let width = session.game.width() as usize;
    let height = session.game.height() as usize;
    let (hr, wr) = (height, width);
    let tiles = session.tile_state().len();
    println!(
        "{map} bots={bots} {difficulty}: {width}x{height}, tile plane {tiles} u16 ({:.1} MiB), regions {}x{}",
        tiles as f64 * 2.0 / 1048576.0,
        hr / 8,
        wr / 8
    );

    // 1. the simulation step alone
    let step_us = timeit!("RlSession::step(15 ticks)", iters, {
        let _ = black_box(session.step(&[], 15));
    });

    // 2. typed obs widgets rebuilt by ofenv_step
    let ents = entities_typed(&session.game);
    let legal = legality_typed(&session.game, AGENT_CLIENT_ID);
    let ents_us = timeit!("entities_typed()", iters, {
        let e = black_box(entities_typed(&session.game));
        black_box(e.players.len());
    });
    let legal_us = timeit!("legality_typed()", iters, {
        let l = black_box(legality_typed(&session.game, AGENT_CLIENT_ID));
        black_box(l.present);
    });

    // 3. LUT + owners plane (the `owners_slotted` loop in puffer_ffi)
    let ids: Vec<usize> = ents.players.iter().map(|p| p.id).collect();
    let lut = ofcore::feat::make_lut(&ids);
    let lut_us = timeit!("make_lut(roster)", iters, {
        black_box(ofcore::feat::make_lut(&ids));
    });
    let owners_us = timeit!("owners plane (alloc+1M LUT walk)", iters, {
        let mut tbl = [0u8; 0x1000];
        for (i, v) in lut.iter().enumerate().take(0x1000) {
            tbl[i] = *v;
        }
        let packed = session.tile_state();
        let mut out = vec![0u8; hr * wr];
        for y in 0..hr {
            let src_row = y * width;
            let dst_row = y * wr;
            let src = &packed[src_row..src_row + wr];
            let dst = &mut out[dst_row..dst_row + wr];
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d = tbl[(*s & OWNER_MASK) as usize];
            }
        }
        black_box(out.len());
    });

    // 4. the observation tensor itself
    let tick = session.game.ticks() as i64;
    let spawn_phase = session.game.in_spawn_phase();
    let owners_slotted = {
        let packed = session.tile_state();
        let mut out = vec![0u8; hr * wr];
        for y in 0..hr {
            let src_row = y * width;
            let dst_row = y * wr;
            for x in 0..wr {
                let owner = (packed[src_row + x] & OWNER_MASK) as usize;
                out[dst_row + x] = lut.get(owner).copied().unwrap_or(0);
            }
        }
        out
    };
    let gh = hr / ofcore::feat::REGION;
    let gw = wr / ofcore::feat::REGION;
    let land: Vec<u8> = vec![0u8; hr * wr];
    let feat_us = timeit!("featurize(gh,gw,...)", iters, {
        let f = black_box(ofcore::feat::featurize(
            gh,
            gw,
            &lut,
            &land,
            &land,
            &owners_slotted,
            tick,
            spawn_phase,
            true,
            0i64,
            &ents,
            &legal,
        ));
        black_box(f.stat.len());
    });

    let total = step_us + ents_us + legal_us + lut_us + owners_us + feat_us;
    println!("\n--- breakdown (per decision) ---");
    for (name, us) in [
        ("step", step_us),
        ("entities_typed", ents_us),
        ("legality_typed", legal_us),
        ("make_lut", lut_us),
        ("owners plane", owners_us),
        ("featurize", feat_us),
    ] {
        println!("{name:18} {us:8.1} us  {:5.1}%", 100.0 * us / total);
    }
    println!("total              {total:8.1} us");
}