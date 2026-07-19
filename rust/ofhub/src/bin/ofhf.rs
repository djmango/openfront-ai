//! ofhf - Hugging Face pull / push / sync-loop / revision.

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use ofhub::hf;
use ofhub::paths::{POLICY_REPO, REPLAYS_REPO, data_dir, replay_spool_dir, repo_root};
use ofhub::sync::{self, SyncConfig};
use std::process::Command;

#[derive(Parser, Debug)]
#[command(name = "ofhf", about = "Hugging Face sync for openfront-rl checkpoints")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Restore latest.safetensors + latest.state.json into a checkpoint dir.
    Pull {
        #[arg(long, env = "HF_RUN_PREFIX", default_value = "ppo_v10")]
        run_prefix: String,
        #[arg(long)]
        checkpoint_dir: PathBuf,
        #[arg(long, env = "HF_REPO_ID", default_value = POLICY_REPO)]
        repo_id: String,
    },
    /// Upload current oftrain interchange files once.
    Push {
        #[arg(long)]
        checkpoint_dir: PathBuf,
        #[arg(long, env = "HF_RUN_PREFIX", default_value = "ppo_v10")]
        run_prefix: String,
        #[arg(long, env = "HF_REPO_ID", default_value = POLICY_REPO)]
        repo_id: String,
        #[arg(long)]
        dry_run: bool,
    },
    /// Background sync loop (fail loud without HF_TOKEN).
    SyncLoop {
        #[arg(long)]
        checkpoint_dir: PathBuf,
        #[arg(long, env = "HF_RUN_PREFIX", default_value = "ppo_v10")]
        run_prefix: String,
        #[arg(long, env = "HF_REPO_ID", default_value = POLICY_REPO)]
        repo_id: String,
        #[arg(long, env = "HF_SYNC_INTERVAL_SECONDS", default_value_t = 600.0)]
        interval: f64,
        #[arg(long, default_value_t = 5)]
        max_retries: u32,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        once: bool,
    },
    /// Print the remote revision (blob oid / etag) for a run's weights.
    Revision {
        #[arg(long, default_value = "ppo_v10")]
        run_name: String,
    },
    /// Download AE encoder safetensors into a directory.
    PullAe {
        #[arg(long, default_value = "weights/ae")]
        ae_dir: PathBuf,
    },
    /// Pack local GameRecord spool into parquet shards on openfront-replays.
    Replays {
        #[arg(long, default_value_os_t = replay_spool_dir())]
        spool: PathBuf,
        #[arg(long, default_value_t = 1000)]
        shard_size: usize,
        #[arg(long, env = "HF_REPLAYS_REPO", default_value = REPLAYS_REPO)]
        repo: String,
        #[arg(long)]
        dry_run: bool,
        #[arg(long, default_value_t = 1)]
        min_files: usize,
    },
    /// Download latest openfront-replays parquet rows into records/ for archive Watch.
    ReplaysPull {
        #[arg(long, env = "HF_REPLAYS_REPO", default_value = REPLAYS_REPO)]
        repo: String,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, default_value_t = 100)]
        limit: usize,
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

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Pull {
            run_prefix,
            checkpoint_dir,
            repo_id,
        } => {
            let client = hf::client_with_optional_token()?;
            let ok =
                sync::restore_latest(&client, &repo_id, &run_prefix, &checkpoint_dir).await?;
            if !ok {
                anyhow::bail!("pull failed: no complete safetensors pair");
            }
        }
        Cmd::Push {
            checkpoint_dir,
            run_prefix,
            repo_id,
            dry_run,
        } => {
            sync::run_sync(SyncConfig {
                checkpoint_dir,
                repo_id,
                run_prefix,
                dry_run,
                once: true,
                ..SyncConfig::default()
            })
            .await?;
        }
        Cmd::SyncLoop {
            checkpoint_dir,
            run_prefix,
            repo_id,
            interval,
            max_retries,
            dry_run,
            once,
        } => {
            sync::run_sync(SyncConfig {
                checkpoint_dir,
                repo_id,
                run_prefix,
                interval_secs: interval,
                max_retries,
                dry_run,
                once,
                restore_latest: false,
            })
            .await?;
        }
        Cmd::Revision { run_name } => {
            let client = hf::client_with_optional_token()?;
            let rev = hf::policy_revision(&client, &run_name).await?;
            println!("{rev}");
        }
        Cmd::PullAe { ae_dir } => {
            sync::pull_ae_encoders(&ae_dir).await?;
        }
        Cmd::Replays {
            spool,
            shard_size,
            repo,
            dry_run,
            min_files,
        } => {
            let script = repo_root().join("scripts/hf_replay_upload.py");
            let mut cmd = Command::new("uv");
            cmd.args([
                "run",
                "--with",
                "pyarrow",
                "python",
                script.to_str().unwrap_or("scripts/hf_replay_upload.py"),
                "--spool",
                spool.to_str().unwrap_or("/data/replay-spool"),
                "--shard-size",
                &shard_size.to_string(),
                "--repo",
                &repo,
                "--min-files",
                &min_files.to_string(),
            ]);
            if dry_run {
                cmd.arg("--dry-run");
            }
            let status = cmd.status()?;
            if !status.success() {
                anyhow::bail!("hf_replay_upload.py failed with {status}");
            }
        }
        Cmd::ReplaysPull { repo, out, limit } => {
            let script = repo_root().join("scripts/hf_replay_pull.py");
            let out_dir = out.unwrap_or_else(|| data_dir().join("records").join("hf-replays"));
            let mut cmd = Command::new("uv");
            cmd.args([
                "run",
                "--with",
                "pyarrow",
                "python",
                script.to_str().unwrap_or("scripts/hf_replay_pull.py"),
                "--repo",
                &repo,
                "--out",
                out_dir.to_str().unwrap_or("/data/records/hf-replays"),
                "--limit",
                &limit.to_string(),
            ]);
            let status = cmd.status()?;
            if !status.success() {
                anyhow::bail!("hf_replay_pull.py failed with {status}");
            }
        }
    }
    Ok(())
}
