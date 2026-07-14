//! ofae - Spatial AE training / prefeaturize / encoder export (Rust port of ae/).

mod checkpoint;
mod data;
mod model;
mod prefeat;
mod train;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "ofae", about = "OpenFront spatial AE trainer")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Train SpatialAE v3.1 from prefeaturized caches.
    Train(train::TrainArgs),
    /// Convert snapshot games into AE training caches.
    #[command(name = "prefeaturize")]
    Prefeaturize(prefeat::PrefeaturizeArgs),
    /// Filter a full AE safetensors into encoder-only weights for oftrain.
    ExportEncoder {
        #[arg(long)]
        ckpt: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Train(args) => train::run(args),
        Cmd::Prefeaturize(args) => prefeat::run(args),
        Cmd::ExportEncoder { ckpt, out } => {
            let meta = serde_json::json!({});
            checkpoint::export_encoder_file(&ckpt, &out, &meta)
        }
    }
}
