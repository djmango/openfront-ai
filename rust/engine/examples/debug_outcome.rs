//! Quick one-off: replay a single record natively and print the full
//! GameOutcome (winner, terminal tick/reason, land share, final ranking) so
//! it can be diffed against the cached TS oracle entry by hand.
//!
//! Usage: cargo run --release -p openfront-engine --example debug_outcome -- <repo_root> <record_path>

use openfront_engine::replay::replay_outcome_native;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let repo = std::path::PathBuf::from(&args[1]);
    let record = std::path::PathBuf::from(&args[2]);
    match replay_outcome_native(&repo, &record) {
        Ok(outcome) => println!("{}", serde_json::to_string_pretty(&outcome).unwrap()),
        Err(e) => eprintln!("error: {e}"),
    }
}
