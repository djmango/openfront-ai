mod batch;
mod bridge;
mod engine;
mod gpu_util;
#[cfg(feature = "native-engine")]
mod native;
mod policy;
mod train;
mod vecenv;

use clap::Parser;
use tch::Device;

/// Rust PPO trainer for OpenFront (v8 port of `rl/ppo.py`). See
/// `rust/DEVLOG.md` for status, known deviations from the Python model,
/// and benchmark results.
#[derive(Parser, Debug)]
#[command(name = "oftrain")]
struct Args {
    /// Number of parallel env workers PER GPU/shard (each spawns its own
    /// Node bridge subprocess). Total envs = num_envs * num_gpus.
    #[arg(long, default_value_t = 4)]
    num_envs: usize,

    /// Number of GPU replicas (data-parallel shards). 1 = single-device.
    /// >1 requires `--device cuda[:N]`; shards always use cuda:0..N-1
    /// (see `train::Config::devices`).
    #[arg(long, default_value_t = 1)]
    num_gpus: usize,

    /// Curriculum stage index (0 = simplest single-map stage).
    #[arg(long, default_value_t = 0)]
    stage: usize,

    #[arg(long, default_value_t = 3000)]
    max_episode_ticks: i64,

    /// Steps collected per env before each PPO update.
    #[arg(long, default_value_t = 64)]
    rollout_len: usize,

    #[arg(long, default_value_t = 1_000_000)]
    updates: u64,

    #[arg(long, default_value_t = 3e-4)]
    lr: f64,

    #[arg(long, default_value_t = 0.999)]
    gamma: f32,

    #[arg(long, default_value_t = 0.95)]
    lambda_: f32,

    #[arg(long, default_value_t = 0.2)]
    clip: f32,

    #[arg(long, default_value_t = 0.5)]
    vf_coef: f32,

    /// Anneals linearly to `ent_coef_final` over `ent_anneal_updates`.
    #[arg(long, default_value_t = 0.01)]
    ent_coef: f32,

    #[arg(long, default_value_t = 0.002)]
    ent_coef_final: f32,

    #[arg(long, default_value_t = 4000)]
    ent_anneal_updates: u64,

    #[arg(long, default_value_t = 0.002)]
    entq_coef: f32,

    /// LR decays by this factor per curriculum stage (matches `rl/ppo.py`).
    #[arg(long, default_value_t = 0.85)]
    stage_lr_decay: f64,

    #[arg(long, default_value_t = 2)]
    epochs: usize,

    #[arg(long, default_value_t = 4)]
    minibatches: usize,

    /// "cpu", "cuda", or "cuda:N".
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Simulation backend: "node" (JSONL subprocess per env) or "native"
    /// (in-process Rust engine; requires the `native-engine` feature and
    /// passing parity gates - see DEVLOG).
    #[arg(long, default_value = "node")]
    engine: engine::EngineKind,

    #[arg(long, default_value_t = 1)]
    log_every: u64,

    #[arg(long, default_value_t = 200)]
    ckpt_every: u64,

    #[arg(long, default_value = "checkpoints")]
    ckpt_dir: String,
}

fn parse_device(s: &str) -> Device {
    if s == "cpu" {
        Device::Cpu
    } else if s == "cuda" {
        Device::Cuda(0)
    } else if let Some(idx) = s.strip_prefix("cuda:") {
        Device::Cuda(idx.parse().unwrap_or(0))
    } else {
        Device::Cpu
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    tch::manual_seed(0);
    let device = parse_device(&args.device);
    println!("[oftrain] device={device:?}");

    let cfg = train::Config {
        num_envs: args.num_envs,
        num_gpus: args.num_gpus,
        stage: args.stage,
        max_episode_ticks: args.max_episode_ticks,
        rollout_len: args.rollout_len,
        updates: args.updates,
        lr: args.lr,
        gamma: args.gamma,
        lambda: args.lambda_,
        clip: args.clip,
        vf_coef: args.vf_coef,
        ent_coef: args.ent_coef,
        ent_coef_final: args.ent_coef_final,
        ent_anneal_updates: args.ent_anneal_updates,
        entq_coef: args.entq_coef,
        stage_lr_decay: args.stage_lr_decay,
        epochs: args.epochs,
        minibatches: args.minibatches,
        device,
        engine: args.engine,
        log_every: args.log_every,
        ckpt_every: args.ckpt_every,
        ckpt_dir: args.ckpt_dir,
    };
    train::run(cfg)
}
