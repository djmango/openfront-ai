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
    /// One-shot: watch + SoftGL MODEL-overlay WebM for one or more maps (hard-fail).
    ///
    /// Prefers live archive+vite when healthy, otherwise spins a patched
    /// SoftGL client worktree. Example:
    ///   ofshowcase clip --map Onion --map Pangaea
    Clip {
        /// Map key(s). Repeatable. Default: full curriculum pool.
        #[arg(long = "map")]
        maps: Vec<String>,
        #[arg(long, env = "RUN_NAME", default_value = "ppo_v10")]
        run_name: String,
        #[arg(long, env = "STAGE", default_value_t = 27)]
        stage: i64,
        #[arg(long, env = "SHOWCASE_WATCH_STAGE")]
        watch_stage: Option<i64>,
        #[arg(long, env = "SHOWCASE_BOTS", default_value_t = 24)]
        bots: i64,
        #[arg(long, env = "SHOWCASE_NATIONS", default_value = "4")]
        nations: String,
        #[arg(long, env = "SHOWCASE_DIFFICULTY", default_value = "Easy")]
        difficulty: String,
        /// Local policy `.safetensors` (skips HF pull).
        #[arg(long)]
        policy: Option<PathBuf>,
        /// Watch device (`cpu`, `cuda`, `cuda:0`, …). Defaults to `cpu` for
        /// one-shot clips so busy training pods do not OOM; daemon still uses
        /// `SHOWCASE_DEVICE` / cuda.
        #[arg(long, env = "SHOWCASE_DEVICE")]
        device: Option<String>,
        /// Delete existing record/clip artifacts and regenerate.
        #[arg(long, default_value_t = false)]
        force: bool,
    },
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
        Cmd::Clip {
            maps,
            run_name,
            stage,
            watch_stage,
            bots,
            nations,
            difficulty,
            policy,
            device,
            force,
        } => {
            if let Some(dev) = device {
                std::env::set_var("SHOWCASE_DEVICE", dev);
            }
            let state = daemon::run_clip(daemon::ClipConfig {
                run_name,
                stage,
                watch_stage: watch_stage.unwrap_or(stage),
                nations,
                bots,
                difficulty,
                maps,
                policy,
                force,
            })
            .await?;
            println!("{}", serde_json::to_string_pretty(&state)?);
        }
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
