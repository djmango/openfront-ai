//! ofexport - filter SpatialAE encoder tensors into `.encoder.safetensors`.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use ofhub::export::{export_encoder, ExportArgs};

#[derive(Parser, Debug)]
#[command(
    name = "ofexport",
    about = "Filter encoder prefixes from a safetensors AE into .encoder.safetensors"
)]
struct Cli {
    /// Input safetensors (full AE or already-filtered).
    #[arg(long)]
    input: PathBuf,
    /// Output `.encoder.safetensors` path.
    #[arg(long)]
    out: PathBuf,
    /// Optional JSON sidecar path (default: out with .json).
    #[arg(long)]
    meta_out: Option<PathBuf>,
    #[arg(long)]
    expected_down: Option<i64>,
    #[arg(long)]
    expected_c: Option<i64>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    export_encoder(ExportArgs {
        input: cli.input,
        out: cli.out,
        meta_out: cli.meta_out,
        expected_down: cli.expected_down,
        expected_c: cli.expected_c,
    })
}
