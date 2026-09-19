//! `ofhash_oracle` - plane dumper for the CUDA hash yardstick.
//!
//! The CUDA side (`ofcuda_hash`) must reproduce, bit-exactly, the engine's
//! per-tick state hash. That needs the *state planes themselves*, not just the
//! hashes - so this crate drives the real engine (linked as a library) and
//! writes the planes out, together with the hashes the engine computes over
//! exactly those planes.
//!
//! Hash definition (identical to `rust/engine/src/bin/parity_trace.rs`):
//!   FNV-1a 64, offset basis 0xcbf29ce484222325, prime 0x100000001b3, over raw
//!   bytes. The state plane feeds each `u16` as two little-endian bytes; the
//!   terrain plane is raw bytes.
//!
//! Two modes:
//!
//!   --mode parity   Rebuilds the scripted stage-0 Pangaea episode exactly the
//!                   way `parity_trace` does (`RlSession::reset` + spawn then
//!                   `expand` intents, one tick per decision) and dumps the
//!                   terrain plane, the post-reset state plane and one state
//!                   plane per tick. These are the planes behind the two
//!                   published constants (terrain 0xebffa87c2568cc58,
//!                   post-reset state 0x6334dfb980453d25).
//!
//!   --mode record   Replays a `GameRecord` exactly the way `tick_dump` does
//!                   (`bootstrap::game_from_record` + `turn_to_executions` +
//!                   `execute_next_tick`) and dumps the terrain plane plus one
//!                   state plane per tick, with the engine's own per-tick FNV
//!                   state hash and per-tick player `gameHashBits`.
//!
//! Output (in --out DIR):
//!   terrain.bin          width*height u8
//!   planes.bin           ticks * width*height u16, little-endian, in tick order
//!   post_reset.bin       parity mode: width*height u16 LE, right after reset
//!   expected.jsonl       line 0 = header, then one line per tick
//!
//! Everything written is read-only with respect to the engine: no engine file
//! is modified, and the trainer's `libopenfront_engine.so` is never touched.

use openfront_engine::game::Game;
use openfront_engine::record::GameRecord;
use openfront_engine::rl::RlSession;
use serde_json::{json, Value};
use std::io::Write;
use std::path::{Path, PathBuf};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

#[inline]
fn fnv_step(h: u64, b: u8) -> u64 {
    (h ^ b as u64).wrapping_mul(FNV_PRIME)
}

pub fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |h, &b| fnv_step(h, b))
}

pub fn fnv1a64_u16_le(words: &[u16]) -> u64 {
    let mut h = FNV_OFFSET;
    for w in words {
        let b = w.to_le_bytes();
        h = fnv_step(h, b[0]);
        h = fnv_step(h, b[1]);
    }
    h
}

fn hex64(h: u64) -> String {
    format!("0x{h:016x}")
}

fn write_u16_le(path: &Path, words: &[u16]) {
    let mut buf = Vec::with_capacity(words.len() * 2);
    for w in words {
        buf.extend_from_slice(&w.to_le_bytes());
    }
    std::fs::write(path, &buf).expect("write planes");
}

fn load_record(path: &Path) -> GameRecord {
    let raw = std::fs::read(path).expect("read record");
    let bytes = if path.extension().and_then(|s| s.to_str()) == Some("gz") {
        use std::io::Read;
        let mut out = Vec::new();
        flate2::read::GzDecoder::new(&raw[..])
            .read_to_end(&mut out)
            .expect("gunzip");
        out
    } else {
        raw
    };
    GameRecord::from_json_bytes(&bytes)
        .expect("parse record")
        .decompress()
}

fn game_hash_f64(game: &Game) -> f64 {
    let mut h = 1.0_f64;
    for p in game.players_in_order() {
        h += openfront_engine::hash::player_hash_js(p);
    }
    h
}

struct Args {
    mode: String,
    repo: PathBuf,
    record: Option<PathBuf>,
    out: PathBuf,
    decisions: u32,
    ticks_per_decision: u32,
    replay_ticks: u32,
    from_tick: u32,
}

fn parse_args() -> Args {
    let mut a = Args {
        mode: "record".into(),
        repo: PathBuf::from("/opt/data/workspaces/skg/openfront-ai"),
        record: None,
        out: PathBuf::from("/tmp/ofhash_dump"),
        decisions: 130,
        ticks_per_decision: 1,
        replay_ticks: 200,
        from_tick: 1,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let key = argv[i].as_str();
        let mut val = || -> String {
            let v = argv.get(i + 1).cloned().unwrap_or_default();
            v
        };
        match key {
            "--mode" => {
                a.mode = val();
                i += 2;
            }
            "--repo" => {
                a.repo = PathBuf::from(val());
                i += 2;
            }
            "--record" => {
                a.record = Some(PathBuf::from(val()));
                i += 2;
            }
            "--out" => {
                a.out = PathBuf::from(val());
                i += 2;
            }
            "--decisions" => {
                a.decisions = val().parse().unwrap();
                i += 2;
            }
            "--ticks-per-decision" => {
                a.ticks_per_decision = val().parse().unwrap();
                i += 2;
            }
            "--replay-ticks" => {
                a.replay_ticks = val().parse().unwrap();
                i += 2;
            }
            "--from-tick" => {
                a.from_tick = val().parse().unwrap();
                i += 2;
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
    }
    a
}

fn main() {
    let args = parse_args();
    std::fs::create_dir_all(&args.out).expect("mkdir out");
    match args.mode.as_str() {
        "parity" => run_parity(&args),
        "record" => run_record(&args),
        other => {
            eprintln!("unknown mode {other}");
            std::process::exit(2);
        }
    }
}

/// Scripted stage-0 Pangaea episode, byte-for-byte the input set
/// `parity_trace` uses by default: map Pangaea, seed "parity", bots 2,
/// difficulty Easy, nations 0, n_agents 1.
fn run_parity(args: &Args) {
    let (mut session, _head, _ents, _legal, _terrain, _duo) = RlSession::reset(
        &args.repo,
        "Pangaea",
        "parity",
        2,
        "Easy",
        Value::from(0i64),
        1,
    )
    .expect("RlSession::reset");

    let width = session.game.width();
    let height = session.game.height();
    let n = (width as usize) * (height as usize);

    let terrain: Vec<u8> = (0..n)
        .map(|t| session.game.terrain_byte(t as u32))
        .collect();
    let terrain_hash = fnv1a64(&terrain);
    std::fs::write(args.out.join("terrain.bin"), &terrain).expect("write terrain");

    let post_reset: Vec<u16> = session.tile_state().to_vec();
    let post_hash = fnv1a64_u16_le(&post_reset);
    write_u16_le(&args.out.join("post_reset.bin"), &post_reset);

    // Spawn tile: first legal region tile, row-major - identical rule to
    // `parity_trace::first_legal_region_tile`.
    let mut spawn_tile = 0u32;
    'outer: for y in 0..height {
        for x in 0..width {
            let t = y * width + x;
            if session.game.is_land(t)
                && !session.game.is_impassable(t)
                && session.game.map.magnitude(t) < 31
                && session.game.map.owner_id(t) == 0
            {
                spawn_tile = t;
                break 'outer;
            }
        }
    }

    let mut planes: Vec<u16> = Vec::with_capacity(args.decisions as usize * n);
    let mut expected = String::new();
    for d in 0..args.decisions {
        let intents: Vec<Value> = if d == 0 {
            vec![json!({"type": "spawn", "tile": spawn_tile})]
        } else {
            vec![json!({"type": "attack", "targetID": null, "troops": 50})]
        };
        session.step(&intents, args.ticks_per_decision);
        let plane = session.tile_state();
        let h = fnv1a64_u16_le(plane);
        planes.extend_from_slice(plane);
        expected.push_str(&format!(
            "{}\n",
            json!({
                "tick": session.game.ticks(),
                "index": d,
                "state_hash": hex64(h),
                "game_hash_bits": game_hash_f64(&session.game).to_bits().to_string(),
            })
        ));
    }
    write_u16_le(&args.out.join("planes.bin"), &planes);

    let header = json!({
        "type": "header",
        "mode": "parity",
        "width": width,
        "height": height,
        "tiles": n,
        "ticks": args.decisions,
        "ticks_per_decision": args.ticks_per_decision,
        "terrain_hash": hex64(terrain_hash),
        "post_reset_state_hash": hex64(post_hash),
        "spawn_tile": spawn_tile,
        "game_id": session.game_id(),
    });
    let path = args.out.join("expected.jsonl");
    let mut f = std::fs::File::create(&path).expect("create expected.jsonl");
    writeln!(f, "{header}").unwrap();
    f.write_all(expected.as_bytes()).unwrap();
    f.flush().unwrap();

    println!("mode parity grid {width}x{height} tiles {n} ticks {}", args.decisions);
    println!("terrain_hash {}", hex64(terrain_hash));
    println!("post_reset_state_hash {}", hex64(post_hash));
    println!("spawn_tile {spawn_tile} game_id {}", session.game_id());
    println!("wrote {}", args.out.display());
}

/// Replay a `GameRecord` the way `tick_dump` does and dump one state plane per tick.
fn run_record(args: &Args) {
    let record_path = args.record.clone().expect("--record required");
    let record = load_record(&record_path);
    let mut game = openfront_engine::bootstrap::game_from_record(&args.repo, &record)
        .expect("game_from_record");

    let width = game.width();
    let height = game.height();
    let n = (width as usize) * (height as usize);

    let terrain: Vec<u8> = (0..n).map(|t| game.terrain_byte(t as u32)).collect();
    let terrain_hash = fnv1a64(&terrain);
    std::fs::write(args.out.join("terrain.bin"), &terrain).expect("write terrain");

    let mut planes: Vec<u16> = Vec::new();
    let mut expected = String::new();
    let mut kept = 0u32;
    for turn in &record.turns {
        if kept >= args.replay_ticks {
            break;
        }
        let gid = game.game_id.clone();
        for execution in
            openfront_engine::execution::intent::turn_to_executions(&mut game, &gid, &turn.intents)
        {
            game.add_execution(execution);
        }
        game.execute_next_tick();

        if game.ticks() < args.from_tick {
            continue;
        }
        let plane = game.tile_state_buffer();
        let h = fnv1a64_u16_le(plane);
        planes.extend_from_slice(plane);
        expected.push_str(&format!(
            "{}\n",
            json!({
                "tick": game.ticks(),
                "index": kept,
                "state_hash": hex64(h),
                "game_hash_bits": game_hash_f64(&game).to_bits().to_string(),
            })
        ));
        kept += 1;
    }
    write_u16_le(&args.out.join("planes.bin"), &planes);

    let header = json!({
        "type": "header",
        "mode": "record",
        "record": record_path.display().to_string(),
        "game_id": record.info.game_id,
        "width": width,
        "height": height,
        "tiles": n,
        "ticks": kept,
        "from_tick": args.from_tick,
        "terrain_hash": hex64(terrain_hash),
    });
    let path = args.out.join("expected.jsonl");
    let mut f = std::fs::File::create(&path).expect("create expected.jsonl");
    writeln!(f, "{header}").unwrap();
    f.write_all(expected.as_bytes()).unwrap();
    f.flush().unwrap();

    println!("mode record grid {width}x{height} tiles {n} ticks {kept}");
    println!("terrain_hash {}", hex64(terrain_hash));
    println!("game_id {}", record.info.game_id);
    println!("wrote {}", args.out.display());
}