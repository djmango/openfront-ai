//! ofshowcase - daemon / hub / archive orchestration.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use ofhub::archive;
use ofhub::daemon;
use ofhub::hub;
use ofhub::paths::{clips_dir, data_dir, records_dir, state_path};

#[derive(Parser, Debug)]
#[command(name = "ofshowcase", about = "Showcase hub / archive / eval daemon")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Background worker: pull policy, oftrain --watch, render clips, write state.json.
    Daemon,
    /// HTTP hub: /, /watch, /play, /status (featured map = latest).
    Hub {
        #[arg(long, env = "HUB_PORT", default_value_t = 8988)]
        port: u16,
    },
    /// Serve GameRecords + clips for the OpenFront client archive API.
    Archive {
        #[arg(long, default_value_os_t = records_dir())]
        records: PathBuf,
        #[arg(long, default_value_t = 8987)]
        port: u16,
        #[arg(long, default_value = "127.0.0.1")]
        bind: String,
        #[arg(long, default_value_os_t = state_path())]
        state: PathBuf,
        #[arg(long, default_value_os_t = clips_dir())]
        clips: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let _ = data_dir();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Daemon => daemon::run_daemon().await?,
        Cmd::Hub { port } => hub::run_hub(port).await?,
        Cmd::Archive {
            records,
            port,
            bind,
            state,
            clips,
        } => {
            archive::run_archive(records, port, bind, Some(state), Some(clips)).await?
        }
    }
    Ok(())
}
