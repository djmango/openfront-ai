//! Structured training metrics (JSONL), port of the useful bits of
//! Python's TensorBoard logging without pulling in a TB dependency.
//!
//! One JSON object per line under `{ckpt_dir}/metrics.jsonl`. Scalars are
//! enough for the cloud sync / crash-loop path; histograms stay as mean/
//! std summaries so the file stays small.

use serde::Serialize;
use serde_json::json;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct MetricsWriter {
    path: PathBuf,
}

impl MetricsWriter {
    pub fn create(ckpt_dir: &str) -> anyhow::Result<Self> {
        let path = Path::new(ckpt_dir).join("metrics.jsonl");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        // Touch the file so it's present even before the first log line.
        OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self { path })
    }

    pub fn log<T: Serialize>(&self, row: &T) -> anyhow::Result<()> {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        serde_json::to_writer(&mut f, row)?;
        f.write_all(b"\n")?;
        Ok(())
    }

    pub fn log_update(
        &self,
        update: u64,
        stage: usize,
        pg: f64,
        vf: f64,
        ent: f64,
        entq: f64,
        win_rate: Option<f64>,
        lr: f64,
        env_steps: u64,
        eval_win: Option<f64>,
        eval_score: Option<f64>,
    ) -> anyhow::Result<()> {
        self.log(&json!({
            "update": update,
            "stage": stage,
            "loss/pg": pg,
            "loss/vf": vf,
            "loss/ent": ent,
            "loss/entq": entq,
            "lr": lr,
            "env_steps": env_steps,
            "win_rate": win_rate,
            "eval/win": eval_win,
            "eval/score": eval_score,
        }))
    }

    pub fn log_actor_batches(
        &self,
        update: u64,
        mean_size: f64,
        singleton_fraction: f64,
        shapes_per_dispatch: f64,
        dispatches: usize,
        padding_ratio: f64,
    ) -> anyhow::Result<()> {
        self.log(&json!({
            "update": update,
            "actor_batch/mean_size": mean_size,
            "actor_batch/singleton_fraction": singleton_fraction,
            "actor_batch/shapes_per_dispatch": shapes_per_dispatch,
            "actor_batch/dispatches": dispatches,
            "actor_batch/padding_ratio": padding_ratio,
        }))
    }
}
