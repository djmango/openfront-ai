mod autoscale;
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

    /// Matches `rl/ppo.py --max-episode-ticks` (default 15000). The port
    /// originally shipped 3000, which truncated every stage-0 episode
    /// before the 80%-ownership win condition was reachable - see devlog.
    #[arg(long, default_value_t = 15000)]
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

    /// Clamps the value-loss target to [-ret_clip, ret_clip] before the
    /// MSE loss ever sees it (see `train::Config::ret_clip`'s doc for the
    /// full rationale/incident this fixes). 0.0 disables it entirely.
    /// Default derived from `gamma=0.999`'s ~1000-tick effective horizon
    /// and this reward function's worst-single-step magnitude (~6.5, the
    /// larger of `W_DELTA_GAIN`/`W_DELTA_LOSS`) - generous enough that a
    /// real long/extreme episode's return shouldn't ever hit it, but far
    /// below the billions/trillions the unclipped version actually reached
    /// during the 2026-07-12 incident.
    #[arg(long, default_value_t = 3000.0)]
    ret_clip: f32,

    /// PPO2-style value-prediction clipping (see `train::Config::vf_clip`'s
    /// doc for the full mechanism/rationale). 0.0 disables it. Default
    /// (50.0) intentionally much tighter than a single worst-case
    /// per-step reward (~6.5) would suggest is needed for *normal*
    /// learning - its job is specifically to bound how far a single
    /// update can move any one sample's prediction, not to accommodate
    /// legitimate variance in the returns themselves (that's `ret_clip`'s
    /// job). Combined with `clip_grad_norm(0.5)` already bounding the
    /// aggregate step, this is deliberately the tighter constraint of the
    /// two on the value head specifically.
    #[arg(long, default_value_t = 50.0)]
    vf_clip: f32,

    /// Anneals linearly to `ent_coef_final` over `ent_anneal_updates`.
    #[arg(long, default_value_t = 0.01)]
    ent_coef: f32,

    #[arg(long, default_value_t = 0.002)]
    ent_coef_final: f32,

    #[arg(long, default_value_t = 4000)]
    ent_anneal_updates: u64,

    /// Adaptive entropy floor (port of `rl/ppo.py --ent-floor`, 0 = off):
    /// when mean policy entropy drops below this, the entropy coef scales
    /// up (x1.3/update, cap 5.0) until it recovers.
    #[arg(long, default_value_t = 2.5)]
    ent_floor: f32,

    #[arg(long, default_value_t = 0.002)]
    entq_coef: f32,

    /// LR decays by this factor per curriculum stage (matches `rl/ppo.py`).
    #[arg(long, default_value_t = 0.85)]
    stage_lr_decay: f64,

    #[arg(long, default_value_t = 2)]
    epochs: usize,

    #[arg(long, default_value_t = 4)]
    minibatches: usize,

    /// Manual bf16 mixed precision for the policy net's conv towers
    /// (grid towers, local net, tile heads); logits/loss/optimizer state
    /// stay f32. tch-rs 0.24's `autocast()` has no dtype selector (always
    /// picks fp16 on CUDA), so this is a hand-rolled cast-in/cast-out path
    /// instead - see `policy.rs`/DEVLOG. Works (slower) on CPU too, so
    /// it's smoke-testable without a GPU.
    #[arg(long, default_value_t = false)]
    amp: bool,

    /// Real foveated crop: the fine-grid branch becomes a fixed
    /// `policy::FOVEATE_SIZE`x`FOVEATE_SIZE` window centered on the agent's
    /// own-tile centroid instead of the whole map (coarse branch is
    /// unaffected - always the full map). Off by default, matching the
    /// existing legacy fallback (fine == whole map) - see `policy.rs`
    /// module doc / `PolicyNet::foveate`.
    #[arg(long, default_value_t = false)]
    foveate: bool,

    /// GridTower channel width override (default from `policy::GC` = 256).
    /// Applies to both the coarse and fine grid towers. Smaller values
    /// (e.g. 128) trade capacity for speed - see `--blocks`.
    #[arg(long, default_value_t = policy::GC)]
    gc: i64,

    /// GridTower residual-block count override (default from
    /// `policy::BLOCKS` = 4). Smaller values (e.g. 2) trade capacity for
    /// speed - see `--gc`.
    #[arg(long, default_value_t = policy::BLOCKS)]
    blocks: i64,

    /// Pin the CPU-side observation/choice tensors' backing memory and use
    /// non-blocking H2D copies for the batch-build CPU->GPU upload (see
    /// `batch::to_device_maybe_pinned`). No-op unless `--device`/shards
    /// are CUDA - not exercisable end-to-end on this Mac's CPU-only
    /// libtorch build, see DEVLOG/final report for how this was verified.
    #[arg(long, default_value_t = false)]
    pinned_h2d: bool,

    /// "cpu", "cuda", or "cuda:N".
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Simulation backend: "native" (in-process Rust engine; requires the
    /// `native-engine` feature) or "node" (JSONL subprocess per env, kept
    /// as the parity-testing fallback). Native is ~10x faster ticking and
    /// validated end-to-end (curriculum advances, wins fire correctly) at
    /// the curriculum's early-stage bot counts (0/5/10, where outcome
    /// parity vs the TS engine is 67-100%); parity is weaker at high bot
    /// counts (30+, "wrong narrow leader in a crowded field" - see
    /// docs/devlog.html's curriculum-parity-check section) so re-check
    /// that gate before relying on native at later curriculum stages.
    #[arg(long, default_value = "native")]
    engine: engine::EngineKind,

    /// Fraction (0.0-1.0) of env workers, evenly spread across every
    /// shard's index range, that run the real Node/TS engine instead of
    /// `--engine`'s choice for the rest. Exists to hedge the native
    /// engine's known parity gaps (see `--engine`'s doc comment - weaker at
    /// 30+ bot counts) by keeping some ground-truth-accurate episodes
    /// flowing even while training mostly on native's ~10x-faster ticking.
    /// 0.0 (default) is a pure single-engine run, identical to omitting
    /// this flag entirely. 0.2 = 1 Node env per 5 (evenly spread, not
    /// clumped - see `train::engine_for_idx`). Requires `openfront/`'s
    /// node_modules installed (`bridge::Bridge::spawn` shells out to its
    /// `tsx`) whenever this is > 0, even if `--engine native`.
    #[arg(long, default_value_t = 0.0)]
    node_fraction: f64,

    #[arg(long, default_value_t = 1)]
    log_every: u64,

    #[arg(long, default_value_t = 200)]
    ckpt_every: u64,

    #[arg(long, default_value = "checkpoints")]
    ckpt_dir: String,

    /// Resume from a previously-saved checkpoint (e.g.
    /// `checkpoints/latest.ot`). Restores weights and training state
    /// (curriculum stage, entropy-floor scale, learning rate, total env
    /// steps, win-rate window, update counter) from the `.state.json`
    /// sidecar saved alongside it - see `train::TrainState`. AdamW's
    /// momentum/variance state is not restored (tch-rs exposes no
    /// optimizer state_dict save/load) and rebuilds over the first few
    /// dozen updates post-resume.
    #[arg(long)]
    resume: Option<String>,

    /// Opt-in: automatically grow `--num-envs` at runtime toward the
    /// `--target-gpu-util` set point instead of relying on manual
    /// trial-and-error (see the "V8 Rust PPO Trainer" devlog entry that
    /// found `--num-envs 4` gave ~40% util vs 64's 98-100% on the same
    /// A100 box, by hand). Off by default so existing training behavior/
    /// configs are unaffected. See `autoscale.rs` for the decision logic
    /// (grow-only in this version - see its module doc for why) and
    /// `train::run`'s update loop for where it's checked.
    #[arg(long, default_value_t = false)]
    auto_scale_envs: bool,

    /// Target GPU utilization set point for `--auto-scale-envs`, as a 0-1
    /// fraction (0.95 = 95%), compared against `GpuSnapshot::min_mean_util`
    /// (the worst GPU's running mean - see `gpu_util.rs`) converted to the
    /// same 0-1 scale. No effect without `--auto-scale-envs`.
    #[arg(long, default_value_t = 0.95)]
    target_gpu_util: f64,

    /// Floor for `--auto-scale-envs`: never scale below this many envs per
    /// shard. Unset defaults to `--num-envs` (never scale below whatever
    /// the run was started with). No effect without `--auto-scale-envs`.
    #[arg(long)]
    min_envs: Option<usize>,

    /// Ceiling for `--auto-scale-envs`, per shard (same "per shard" unit
    /// as `--num-envs`/`--min-envs`). 0 (the default) means "derive
    /// automatically" from CPU headroom (see `autoscale::cpu_env_cap_per_shard`:
    /// logical CPUs minus a small reserved margin, divided across
    /// `--num-gpus` shards) - each env worker is one OS thread plus, for
    /// `--engine node`, its own Node bridge subprocess, so this exists to
    /// keep autoscale from oversubscribing the CPU chasing GPU headroom
    /// that IPC/engine-tick latency won't actually let it use. No effect
    /// without `--auto-scale-envs`.
    #[arg(long, default_value_t = 0)]
    max_envs: usize,

    /// How often (in PPO updates) `--auto-scale-envs` re-evaluates GPU
    /// utilization and possibly resizes. Checking every update would let
    /// one noisy sample cause needless resize churn; too rarely leaves the
    /// GPU under-fed longer than necessary after startup.
    #[arg(long, default_value_t = 5)]
    autoscale_check_every: u64,

    /// Envs added per `--auto-scale-envs` growth step (per shard). Small
    /// steps converge more slowly but overshoot the target less; see
    /// `autoscale::next_env_count`'s hysteresis band for the other half of
    /// the anti-thrashing story.
    #[arg(long, default_value_t = 4)]
    autoscale_step: usize,
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

/// Mirrors PyTorch's own `torch/__init__.py::_preload_cuda_deps()`: when
/// libtorch is a "split" pip wheel install (each CUDA component - cublas,
/// cudnn, cusparse, nccl, etc. - its own separate `nvidia-*-cu12` package,
/// as opposed to one monolithic system libtorch), Python's loader
/// `ctypes.CDLL(path, mode=RTLD_GLOBAL)`s each of these .so files in a
/// specific dependency order *before* the C extension that actually calls
/// into CUDA loads. A plain Rust binary linking against the same libraries
/// via normal ELF DT_NEEDED/lazy-binding does NOT replicate this - `ldd`
/// resolves every library fine, but `cudaGetDeviceCount` still fails with
/// "CUDA unknown error" at runtime (reproduced directly: this exact
/// combination of libs works from `python3 -c "import torch; ..."` in the
/// same environment/LD_LIBRARY_PATH, but panics from `oftrain` otherwise -
/// see docs/devlog or the PR that added this for the full bisection). Only
/// matters for a split pip-wheel libtorch, so failing silently (missing
/// `nvidia/` dir entirely, e.g. a monolithic/system libtorch) is fine - the
/// binary just proceeds exactly as it did before this existed.
fn preload_cuda_deps() {
    // LIBTORCH's own lib dir is a sibling of the `nvidia/` package dir this
    // hunts for (`<venv>/.../site-packages/{torch,nvidia}/`) - reuse
    // whichever the binary was actually linked against (LD_LIBRARY_PATH,
    // set by the launch script) rather than hardcoding a venv path.
    let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") else { return };
    // Rough dependency order (cublasLt before cublas, cusparse/cublas
    // before cusolver, etc.) - matches the order torch's own preloader
    // uses; harmless if a library has no such ordering constraint.
    const ORDER: &[&str] = &[
        "cusparselt", "nvtx", "nvjitlink", "cuda_nvrtc", "cuda_runtime", "cuda_cupti", "cublas",
        "cufft", "curand", "cudnn", "cusparse", "cusolver", "nccl", "cufile", "nvshmem",
    ];
    let dirs: Vec<&str> = ld_path.split(':').collect();
    // Drive the walk from ORDER, not from LD_LIBRARY_PATH's own entry
    // order, so load order matches torch's regardless of how the launch
    // script happened to list these directories.
    for pkg in ORDER {
        let Some(base) = dirs.iter().find(|dir| {
            std::path::Path::new(dir).parent().and_then(|p| p.file_name()).and_then(|n| n.to_str())
                == Some(*pkg)
        }) else {
            continue;
        };
        let Ok(entries) = std::fs::read_dir(base) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
            if !name.starts_with("lib") || !name.contains(".so") {
                continue;
            }
            let Ok(cpath) = std::ffi::CString::new(path.as_os_str().to_string_lossy().as_bytes())
            else {
                continue;
            };
            unsafe {
                libc::dlopen(cpath.as_ptr(), libc::RTLD_GLOBAL | libc::RTLD_NOW);
            }
        }
    }
}

fn main() -> anyhow::Result<()> {
    preload_cuda_deps();
    if std::env::var("OFTRAIN_EXPLICIT_CUINIT").is_ok() {
        unsafe {
            let handle = libc::dlopen(c"libcuda.so.1".as_ptr(), libc::RTLD_NOW | libc::RTLD_GLOBAL);
            if handle.is_null() {
                eprintln!("[oftrain] dlopen(libcuda.so.1) failed: {:?}", std::ffi::CStr::from_ptr(libc::dlerror()));
            } else {
                let sym = libc::dlsym(handle, c"cuInit".as_ptr());
                if sym.is_null() {
                    eprintln!("[oftrain] dlsym(cuInit) failed");
                } else {
                    let cu_init: extern "C" fn(u32) -> i32 = std::mem::transmute(sym);
                    let rc = cu_init(0);
                    eprintln!("[oftrain] explicit cuInit(0) -> {rc}");
                }
            }
        }
    }
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
        ret_clip: args.ret_clip,
        vf_clip: args.vf_clip,
        ent_coef: args.ent_coef,
        ent_coef_final: args.ent_coef_final,
        ent_anneal_updates: args.ent_anneal_updates,
        ent_floor: args.ent_floor,
        entq_coef: args.entq_coef,
        stage_lr_decay: args.stage_lr_decay,
        epochs: args.epochs,
        minibatches: args.minibatches,
        amp: args.amp,
        foveate: args.foveate,
        gc: args.gc,
        blocks: args.blocks,
        pinned_h2d: args.pinned_h2d,
        device,
        engine: args.engine,
        node_fraction: args.node_fraction.clamp(0.0, 1.0),
        log_every: args.log_every,
        ckpt_every: args.ckpt_every,
        ckpt_dir: args.ckpt_dir,
        resume: args.resume,
        auto_scale_envs: args.auto_scale_envs,
        target_gpu_util: args.target_gpu_util,
        // Never scale below whatever the run was explicitly started with
        // unless the user gave an explicit floor of their own.
        min_envs: args.min_envs.unwrap_or(args.num_envs),
        max_envs: args.max_envs,
        autoscale_check_every: args.autoscale_check_every,
        autoscale_step: args.autoscale_step,
    };
    train::run(cfg)
}
