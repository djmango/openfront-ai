//! Phase 1 harness CLI: replay GameRecords with hash verification.

use clap::Parser;
use openfront_engine::backend::Backend;
use openfront_engine::record::GameRecord;
use openfront_engine::replay::{replay_record, ReplayOptions};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "openfront-replay")]
#[command(about = "Replay GameRecords with hash verification (Rust harness)")]
struct Args {
    /// Path to GameRecord JSON
    record: PathBuf,
    /// Repo root (openfront-ai checkout with openfront/ submodule)
    #[arg(long, default_value = ".")]
    repo: PathBuf,
    /// TS engine (default, hash-verified) or growing Rust native / stub
    #[arg(long, value_enum, default_value_t = Backend::Ts)]
    backend: Backend,
}

fn main() {
    let args = Args::parse();

    if matches!(args.backend, Backend::Stub) {
        let bytes = std::fs::read(&args.record).unwrap_or_else(|e| {
            eprintln!("read {}: {e}", args.record.display());
            std::process::exit(1);
        });
        let _record = GameRecord::from_json_bytes(&bytes).unwrap_or_else(|e| {
            eprintln!("parse record: {e}");
            std::process::exit(1);
        });
    }

    let result = replay_record(
        &args.record,
        &ReplayOptions {
            backend: args.backend,
            repo_root: args.repo,
        },
    );

    println!(
        "ok={} ticks={} hashes_checked={}",
        result.ok, result.ticks, result.hashes_checked
    );
    if let Some(r) = &result.reason {
        println!("reason: {r}");
    }
    if !result.ok {
        std::process::exit(1);
    }
}
