//! Dump `seed -> game id` for the engine's own id derivation, as NDJSON, so it
//! can be diffed against the TypeScript core's `seedToGameID`.
//!
//! The whole point of this binary is that it calls the *production* function
//! (`openfront_engine::session::seed_to_game_id`, the one `RlSession::reset`
//! and `stub`/`daemon` backends use) rather than a copy - a copy is exactly
//! how the TS/native id divergence stayed hidden.
//!
//! Output line shape (one per seed, NDJSON, stdout):
//!   {"seed":"parity","game_id":"QqkIuyke","h0":358883488}
//!
//! Usage (from `rust/`):
//!   cargo run --release -p openfront-engine --bin seed_ids -- <seeds.txt>
//! Seeds are read one per line from the file (blank lines skipped), or taken
//! from argv when no file is given.

use openfront_engine::session::seed_to_game_id;
use openfront_engine::util::simple_hash;
use serde_json::json;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let seeds: Vec<String> = if let Some(path) = args.first() {
        std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("read seeds file {path}: {e}"))
            .lines()
            .map(|l| l.trim_end_matches(['\r', '\n']).to_string())
            .filter(|l| !l.is_empty())
            .collect()
    } else {
        args
    };
    if seeds.is_empty() {
        eprintln!("usage: seed_ids [seeds-file | seed ...]");
        std::process::exit(2);
    }
    for seed in &seeds {
        let h0 = simple_hash(&format!("rl-{seed}"));
        println!(
            "{}",
            json!({ "seed": seed, "game_id": seed_to_game_id(seed), "h0": h0 })
        );
    }
}