//! Threaded rollout collection + clipped PPO update. Port of the training
//! loop in the (Python) `rl/ppo.py`, restructured for tch/threads instead
//! of multiprocessing: `spawn_worker` gives every env its own OS thread
//! driving a `Bridge` subprocess, lockstepped with a collector thread via
//! a pair of channels per env - functionally a Gym `SyncVectorEnv`.
//!
//! ## Pipelined actor/learner (the "why is the GPU idle" fix)
//!
//! Rollout collection is CPU/IPC-bound (each step round-trips through a
//! Node subprocess per env to advance the actual game simulation) while
//! the minibatch loop is GPU-bound. Running them sequentially - collect a
//! full rollout, *then* train on it - means the GPU sits idle for the
//! entire collection phase every update; measured on a 4-GPU box this was
//! ~9-10s idle out of every ~25s update (`min_mean_util` stuck at ~40%
//! even though the GPU is genuinely ~100% busy *during* the minibatch
//! loop - see DEVLOG). Fix: split each shard into an `ActorShard` (owns
//! the env workers + a frozen snapshot of the policy, used only for
//! `act()`) and a `LearnerShard` (the trained policy/optimizer), and every
//! update collect the *next* rollout on the actor (unchanged, so this is
//! safe to read concurrently) while training the learner on the
//! *previous* rollout - both phases run on separate OS threads. With
//! `--persistent-actors`, one actor thread per GPU owns its ActorShard,
//! policy VarStore, AE, terrain cache, and env workers for the entire run;
//! the legacy default uses one scoped collector thread per update. This is
//! the standard "collect batch k+1 with
//! actor v(k-1) while training learner v(k-1)->v(k) on batch k" one-step-
//! lag pipeline. Persistent refreshes serialize each learner VarStore to
//! an owned CPU byte vector; the actor deserializes and synchronizes it on
//! its own thread, so no VarStore/Tensor reference crosses a channel. The
//! actor is refreshed from the learner's just-updated weights after training
//! finishes (so the *next* update's collection uses the newest weights
//! available, one version behind the learner it's paired with for
//! training).
//! The same flag starts one persistent learner owner per GPU. Each constructs
//! and retains its LearnerShard, ShardBatch, and shuffle RNG on one stable OS
//! thread. At each optimizer barrier owners send flat `Vec<f32>` gradients to
//! a CPU hub, which reduces in shard-index order and returns CPU averages;
//! no Tensor or VarStore crosses threads. Checkpoint weights are written by
//! owner 0 and coordinator-ordered sidecars retain their format/order.
//!
//! Phase 1 deliberately does not implement live env-resize commands.
//! Combining `--persistent-actors` with `--auto-scale-envs` selects the
//! legacy path (with a startup warning), preserving existing autoscaling.
//!
//! ## Dual env-group pipelining (`--pipeline-groups`)
//!
//! Inside a single `collect_rollout`, workers are split into two halves
//! (Python `rl/ppo.py` v4.1): encode+act(g0) → send(g0) → encode+act(g1)
//! while g0's engines step → recv(g0) → … . With one env the second group
//! is empty and the path degenerates to the classic lockstep loop. Default
//! on.
//!
//! Multi-GPU (see `LearnerShard`/`ActorShard`): one `PolicyNet`/`VarStore`
//! replica per device, each owning a disjoint slice of envs, in a single
//! process/thread. This mirrors `rl/ppo.py`'s DDP mode (torchrun ranks
//! each own `envs/world` environments and a full local rollout/epoch/
//! minibatch loop, with gradients flat-all-reduced-and-averaged once per
//! optimizer step - see the comment above `dist.all_reduce(flat)` there)
//! rather than wrapping in `nn.parallel.DistributedDataParallel`. `tch`/
//! `torch-sys` has no NCCL bindings. The legacy path uses P2P copies;
//! persistent owners use explicit CPU flat-gradient messages so CUDA objects
//! remain strictly thread-owned. Both implement "average grad before step"
//! semantics; the CPU hub prioritizes correctness over NCCL throughput.
//!
//! ## Remaining Python-parity gaps (oftrain)
//!
//! - **AdamW optimizer-state restore**: tch-rs `COptimizer` exposes no
//!   moment getters / `state_dict`. Handled via `--resume-warmup-updates`
//!   (default 100) so LR warms in while moments rebuild after `--resume`
//!   (documented on `Config::resume` / `Config::resume_warmup_updates`).
//! - **fp16 host storage**: `--compact-rollout` stores foveated grids as
//!   host fp16. The legacy path stays f32; `--fp16-rollout` only changes
//!   its H2D transfer dtype.
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use ofcore::feat::ACTIONS;
use ofcore::translate::Choice;
use rand::{RngCore, SeedableRng};
use rand::seq::SliceRandom;
use tch::nn::OptimizerConfig;
use tch::{Cuda, Device, Kind, Tensor, nn};

use crate::autoscale;
use crate::batch::{self, ChoiceScalars};
use crate::engine::EngineKind;
use crate::gpu_util::GpuUtilSampler;
use crate::metrics::MetricsWriter;
use crate::policy::{self, PolicyNet};
use crate::vecenv::{EnvWorker, EpisodeInfo, PreparedObs};

/// Pixel budget for one forward/backward sub-batch during the PPO update
/// (mirrors `rl/ppo.py::MAX_UPD_PIX`). When `mb_size * (gh*gw + cgh*cgw)`
/// exceeds this, the minibatch is further split and grads accumulate with
/// `w_sub = sub_len / mb_len` weights before a single optimizer step.
const MAX_UPD_PIX: usize = 1_600_000;

/// Value-loss form. Default `Mse` matches Python `F.mse_loss`; `Huber`
/// remains available as a stabilizer escape hatch (`--value-loss huber`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueLoss {
    Huber,
    Mse,
}

/// Port of `rl/ppo.py`'s entropy-floor controller constants: hold the
/// controller for the first N updates (spawn-heavy startup rollouts read
/// artificially low entropy), and cap the multiplicative scale so the
/// entropy term can't dwarf the policy gradient (the old Python cap of 30
/// did exactly that - see the Jul 9 v7 audit in the devlog).
const ENT_GRACE_UPDATES: u64 = 20;
const ENT_SCALE_MAX: f64 = 5.0;

/// Linear LR warmup over the first N updates of *every* stage (including
/// stage 0's very start and every curriculum advance). Every value-loss
/// instability episode observed this session started within the first
/// ~10-40 updates after a fresh/stage-advance policy snapshot began
/// training on genuinely new data - exactly the highest-variance,
/// least-reliable-gradient-estimate window PPO ever sees, since the
/// value function hasn't had a chance to calibrate to the new distribution
/// yet. A full-strength optimizer step against a badly wrong value
/// function's advantage estimates is the most likely trigger for the
/// initial disruption that every other fix in this session (Huber,
/// ret_clip, adv_clip, the value soft-bound, the adaptive return clip)
/// has been bounding the *damage* from rather than reducing how often it
/// happens in the first place. Standard warmup mitigation: scale LR
/// linearly from 0 up to its target value over this many updates,
/// applied every update (a no-op once warmup completes, so this doesn't
/// interact with the existing stage-advance LR decay at all).
///
/// A live run with a 20-update warmup confirmed the mechanism works
/// exactly as intended - v-loss stayed healthy (0.02-0.9) for the *entire*
/// warmup window and recent_reward hit a new best (28.2, sustained
/// 15-19 for many updates, far above every prior run's ~3.5-4.9
/// plateau) - but then instability reignited at *exactly* update
/// 20-21, the update right after warmup completed and LR snapped to
/// full strength. That's about as direct a confirmation of the warmup
/// hypothesis as a live run can give, and an equally direct signal that
/// 20 updates isn't a long enough runway - raised 5x to give the value
/// function much more time to actually stabilize before facing a
/// full-strength step.
const LR_WARMUP_UPDATES: u64 = 100;

/// `update`/`warmup_start` are both absolute update indices (not reset to
/// 0 at stage boundaries) - see the call site's doc for why warmup resets
/// on every curriculum advance, not just the run's start.
fn lr_warmup_frac(update: u64, warmup_start: u64, warmup_updates: u64) -> f64 {
    ((update - warmup_start + 1) as f64 / warmup_updates.max(1) as f64).min(1.0)
}

/// How many running standard deviations (see `RetStat`) the adaptive
/// return-clip bound allows - generous relative to a well-behaved return
/// distribution (matching `adv_clip`'s own default of 10 std-devs for the
/// same reasoning), tight relative to the "value head plateaued in the
/// tens of thousands while typical returns were single digits" gap this
/// closes.
const RET_ADAPTIVE_N_STD: f64 = 10.0;

#[derive(Clone)]
pub struct Config {
    /// Envs per shard (per device) *at startup*. Total envs = num_envs *
    /// devices().len(). If `auto_scale_envs` is on, the *live* per-shard
    /// count can grow past this at runtime (see `autoscale.rs`/`run()`'s
    /// update loop) - anything that sizes a buffer or indexes per-env data
    /// must use the actual live count (e.g. `ActorShard::workers.len()` or
    /// a rollout's actual buffer width), never this field, or it will
    /// silently misindex after a scale-up. Logging/startup-only uses (e.g.
    /// the initial spawn-count message) are fine as-is.
    pub num_envs: usize,
    /// Number of GPU replicas/shards. 1 = original single-device path.
    /// >1 requires `device` to be `Cuda(_)`; shards use `Cuda(0..num_gpus)`.
    pub num_gpus: usize,
    pub stage: usize,
    pub max_episode_ticks: i64,
    pub rollout_len: usize,
    pub updates: u64,
    pub lr: f64,
    pub gamma: f32,
    pub lambda: f32,
    pub clip: f32,
    pub vf_coef: f32,
    /// `--ret-clip` (0.0 = disabled): clamps the value-loss *target*
    /// (`ret = advantage + rollout-time value`, see `train_update`) to
    /// `[-ret_clip, ret_clip]` before it's ever used in the MSE loss. Real
    /// fix (not just a guard) for the 2026-07-12 value-loss-explosion
    /// incident (docs/devlog.html): squared-error loss scales with the
    /// *square* of the target, so a legitimately-large-but-rare return (a
    /// long episode's accumulated reward, or a big terminal bonus - not
    /// necessarily a bug, see the reward-scale analysis in that entry)
    /// still produces a disproportionately large gradient even with
    /// grad-norm clipping applied every step, and can kick off a
    /// self-reinforcing bootstrap-value feedback loop (GAE uses the
    /// rollout-time value as `next_value`, so a bad prediction poisons
    /// every earlier timestep's own return too). Clamping the target
    /// directly is a coarser fix than PPO2-style value-prediction clipping
    /// (which would need the old per-sample value threaded into the
    /// minibatch loss separately, a bigger refactor) but stops the exact
    /// observed failure mode without touching the advantage/policy-loss
    /// path at all - the return is only ever consumed as this one target.
    pub ret_clip: f32,
    /// Clamps the *normalized* advantage to [-adv_clip, adv_clip] (0.0 =
    /// disabled) - see the clamping site in `train_update`'s batch-build
    /// stage for why this, not just `vf_clip`, was needed to fully close
    /// the 2026-07-12 incident (the policy loss's gradient scales with the
    /// advantage directly, and normalization alone doesn't bound a single
    /// outlier's normalized magnitude when the rest of the batch's
    /// advantages are near zero - the same return-spike was poisoning
    /// *both* the value and policy losses' gradients, and only the value
    /// side had been fixed before this). Default 10.0 - generous (10
    /// population std-devs) relative to a well-behaved advantage
    /// distribution, tight relative to the hundreds-of-std-devs outliers
    /// actually observed live.
    pub adv_clip: f32,
    /// Huber-loss delta for the value loss (replaces plain MSE - see the
    /// loss computation in `train_update`'s minibatch loop for the full
    /// incident history/rationale). Below `delta`, behaves like ordinary
    /// squared error; beyond it, the loss - and critically, its *gradient*
    /// - grows only linearly, bounded by `delta`, no matter how extreme the
    /// target or prediction is. This is what actually fixed the
    /// 2026-07-12 value-loss-explosion incident: `--ret-clip` (bounding
    /// only the target) and a PPO2-style prediction-clipping attempt
    /// (bounding the prediction relative to its old value, but still
    /// selecting the *unclipped* branch's unbounded gradient whenever the
    /// prediction had already drifted far enough - confirmed live, `v`
    /// still reached 310 quadrillion with that active) both failed to
    /// actually cap the gradient magnitude in the one case that matters -
    /// Huber does, by construction, unconditionally.
    pub vf_clip: f32,
    /// Entropy coefficient anneals linearly `ent_coef -> ent_coef_final`
    /// over `ent_anneal_updates` (matches `rl/ppo.py`'s schedule), with
    /// the adaptive entropy-floor multiplier (`ent_floor`) on top.
    pub ent_coef: f32,
    pub ent_coef_final: f32,
    pub ent_anneal_updates: u64,
    /// Adaptive entropy floor (port of `rl/ppo.py --ent-floor`, 0 = off):
    /// when mean discrete-head policy entropy drops below this, the
    /// entropy coef scales up (x1.3/update, cap `ENT_SCALE_MAX`) until it
    /// recovers. Without it the policy collapses to near-zero entropy
    /// within a handful of updates on low-variance early stages and never
    /// recovers (observed on every A100 run before this was ported).
    pub ent_floor: f32,
    pub entq_coef: f32,
    /// `lr * stage_lr_decay ^ stage`, applied on curriculum advance.
    pub stage_lr_decay: f64,
    pub epochs: usize,
    pub minibatches: usize,
    /// `--amp`: manual bf16 mixed precision for the policy net's conv
    /// towers (see `policy::PolicyNet::amp` doc). CPU-safe (bf16 works on
    /// CPU, just slower - useful for correctness smoke tests without a
    /// GPU), off by default. Frozen AE bf16 is more strictly gated on CUDA
    /// plus `persistent_actors`; legacy/nonpersistent AE inference stays f32.
    pub amp: bool,
    /// `--foveate`: real foveated crop for the fine-grid branch (see
    /// `policy::PolicyNet::foveate` doc). Default on (Python v7 parity).
    pub foveate: bool,
    /// Fine AE encoder safetensors path (`--ckpt`).
    pub ae_ckpt: String,
    /// Optional coarse /16 AE encoder safetensors (`--coarse-ckpt`).
    pub coarse_ckpt: Option<String>,
    /// `--gc`/`--blocks`: GridTower size overrides (see `policy::GC`/
    /// `policy::BLOCKS` defaults).
    pub gc: i64,
    pub blocks: i64,
    /// `--pinned-h2d`: pin the CPU-side observation/choice tensors' backing
    /// memory and use non-blocking H2D copies for the batch-build
    /// CPU->GPU upload (see `batch::to_device_maybe_pinned`). No-op unless
    /// `device`/shard devices are CUDA.
    pub pinned_h2d: bool,
    /// `--fp16-rollout`: after AE encode, H2D fine/coarse grids as Half
    /// then cast to Float on device (halves PCIe transfer). Host
    /// `PreparedObs.grid` stays f32. Default off (opt-in); pod_train_v8
    /// enables it via EXTRA_ARGS.
    pub fp16_rollout: bool,
    /// Host-owned fp16 foveated rollout payload. Effective only together
    /// with `foveate`; no actor device tensor crosses into learner threads.
    pub compact_rollout: bool,
    /// `--pipeline-groups`: split env workers into two halves and overlap
    /// act(g1) with step(g0) inside each rollout step (Python v4.1
    /// dual-group pipelining). Default on; with N=1 the second group is
    /// empty and behavior matches the single-group path.
    pub pipeline_groups: bool,
    /// Opt-in persistent ownership. One long-lived OS thread per shard owns
    /// actor CUDA state; one stable learner thread per GPU owns that shard's
    /// learner CUDA state and PPO execution. Gradient and weight messages are
    /// packed CPU-only f32 vectors. Disabled by default; autoscaling selects
    /// the legacy path.
    pub persistent_actors: bool,
    pub device: Device,
    /// Which simulation backend envs run against (Node bridge or the
    /// in-process native engine) for the `1.0 - node_fraction` majority of
    /// workers.
    pub engine: EngineKind,
    /// Fraction of workers that run the Node engine regardless of
    /// `engine`'s choice, evenly spread by index (see `engine_for_idx`).
    /// 0.0 = every worker uses `engine` unchanged.
    pub node_fraction: f64,
    pub log_every: u64,
    /// `--eval-every`: updates between fixed-seed greedy eval passes
    /// (0 = off). See `run_eval`.
    pub eval_every: u64,
    /// `--eval-episodes`: fresh workers per greedy eval pass.
    pub eval_episodes: usize,
    pub ckpt_every: u64,
    pub ckpt_dir: String,
    /// `--init`: warm-start weights from a `.safetensors` (preferred) or
    /// legacy `.ot` VarStore dump without restoring `TrainState`
    /// (BC→RL / exported weights). Ignored when `resume` is also set.
    pub init: Option<String>,
    /// `--resume`: path to a previously-saved `.safetensors` (preferred) or
    /// legacy `.ot` weights file (its training-state sidecar is found by
    /// swapping the extension for `.state.json` - see `TrainState`).
    /// Restores weights, curriculum stage, entropy-floor scale, learning
    /// rate, total env steps, and the win-rate window; the update counter
    /// resumes from where it left off. AdamW's momentum/variance state is
    /// NOT restored (tch-rs exposes no optimizer state_dict save/load -
    /// see module doc / `--resume-warmup-updates`) and rebuilds over the
    /// post-resume warmup window.
    pub resume: Option<String>,
    /// `--resume-warmup-updates`: extra LR warmup length after `--resume`
    /// while Adam moments rebuild (tch COptimizer has no state dump).
    /// Stage-advance warmup still uses `LR_WARMUP_UPDATES`; this only
    /// stretches the *first* post-resume window. Default 100; 0 disables.
    pub resume_warmup_updates: u64,
    /// `--value-loss`: `Mse` (Python `F.mse_loss` parity, default) or
    /// `Huber` (Rust stabilizer escape hatch).
    pub value_loss: ValueLoss,

    /// `--auto-scale-envs`: opt-in runtime growth of the per-shard env
    /// count toward `target_gpu_util` (see `autoscale.rs`). Off by
    /// default - existing configs/behavior are unaffected.
    pub auto_scale_envs: bool,
    /// `--target-gpu-util`: 0-1 fraction set point for `auto_scale_envs`.
    pub target_gpu_util: f64,
    /// `--min-envs`: per-shard floor for `auto_scale_envs` (defaults to
    /// `num_envs` if the user didn't pass an explicit value - see
    /// `main.rs`).
    pub min_envs: usize,
    /// `--max-envs`: per-shard ceiling for `auto_scale_envs`. 0 means
    /// "derive from CPU headroom" (see `autoscale::cpu_env_cap_per_shard`,
    /// resolved once in `run()`).
    pub max_envs: usize,
    /// `--autoscale-check-every`: how often (in updates) to re-evaluate.
    pub autoscale_check_every: u64,
    /// `--autoscale-step`: envs added per growth step (per shard).
    pub autoscale_step: usize,
}

/// Restart-proof training state, saved as a JSON sidecar next to every
/// weights checkpoint (`<ckpt>.state.json` alongside
/// `<ckpt>.safetensors` / legacy `<ckpt>.ot`) - port of `rl/ppo.py`'s
/// `state.json`/embedded-checkpoint-state pattern. Small and cheap enough
/// to write every checkpoint without it being the bottleneck.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainState {
    pub update: u64,
    pub stage: usize,
    pub ent_scale: f64,
    pub lr_now: f64,
    pub total_env_steps: u64,
    pub recent_wins: Vec<f64>,
}

fn state_sidecar_path(ckpt_path: &str) -> String {
    let stem = ckpt_path
        .strip_suffix(".safetensors")
        .or_else(|| ckpt_path.strip_suffix(".ot"))
        .unwrap_or(ckpt_path);
    format!("{stem}.state.json")
}

#[cfg(test)]
mod sidecar_path_tests {
    use super::{atomic_tmp_path, state_sidecar_path};

    #[test]
    fn strips_safetensors_and_ot() {
        assert_eq!(
            state_sidecar_path("checkpoints/latest.safetensors"),
            "checkpoints/latest.state.json"
        );
        assert_eq!(
            state_sidecar_path("checkpoints/latest.ot"),
            "checkpoints/latest.state.json"
        );
        assert_eq!(
            state_sidecar_path("checkpoints/policy_update10.safetensors"),
            "checkpoints/policy_update10.state.json"
        );
    }

    #[test]
    fn atomic_tmp_keeps_serializer_extension() {
        assert_eq!(
            atomic_tmp_path("checkpoints/latest.safetensors"),
            "checkpoints/latest.tmp.safetensors"
        );
        assert_eq!(
            atomic_tmp_path("checkpoints/policy_update10.ot"),
            "checkpoints/policy_update10.tmp.ot"
        );
        assert_eq!(
            atomic_tmp_path("checkpoints/latest.state.json"),
            "checkpoints/latest.state.tmp.json"
        );
    }
}

/// Atomic write (tmp file + rename) so a kill mid-save can never leave a
/// torn/half-written checkpoint or state file behind - matches
/// `rl/ppo.py`'s `policy.pt.tmp` -> `policy.pt` rename pattern.
fn atomic_tmp_path(path: &str) -> String {
    let p = std::path::Path::new(path);
    match (p.file_stem(), p.extension()) {
        (Some(stem), Some(ext)) => {
            let name = format!("{}.tmp.{}", stem.to_string_lossy(), ext.to_string_lossy());
            p.with_file_name(name).to_string_lossy().into_owned()
        }
        _ => format!("{path}.tmp"),
    }
}

fn save_atomic(path: &str, write: impl FnOnce(&str) -> Result<()>) -> Result<()> {
    // VarStore chooses the serializer from the filename extension. Keep
    // `.safetensors`/`.ot` on the temporary path; `foo.safetensors.tmp`
    // silently selects PyTorch zip serialization and then becomes
    // unreadable after it is renamed back to `.safetensors`.
    let tmp = atomic_tmp_path(path);
    write(&tmp)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn save_checkpoint(vs: &nn::VarStore, path: &str, state: &TrainState) -> Result<()> {
    save_atomic(path, |tmp| Ok(vs.save(tmp)?))?;
    save_checkpoint_state(path, state)
}

fn save_checkpoint_state(path: &str, state: &TrainState) -> Result<()> {
    let state_path = state_sidecar_path(path);
    save_atomic(&state_path, |tmp| {
        std::fs::write(tmp, serde_json::to_string_pretty(state)?)?;
        Ok(())
    })?;
    Ok(())
}

impl Config {
    fn devices(&self) -> Vec<Device> {
        if self.num_gpus <= 1 {
            return vec![self.device];
        }
        match self.device {
            // CPU multi-shard is only meaningful as a local (no-GPU) test
            // of the sharding/grad-sync plumbing itself - every "shard"
            // still lands on the same physical device.
            Device::Cpu => vec![Device::Cpu; self.num_gpus],
            _ => (0..self.num_gpus).map(Device::Cuda).collect(),
        }
    }
}

struct Worker {
    choice_tx: Sender<Choice>,
    stage_tx: Sender<usize>,
    obs_rx: Receiver<Result<(PreparedObs, f64, bool, Option<EpisodeInfo>), String>>,
    handle: JoinHandle<()>,
}

/// Which engine worker `idx` (a stable, global env index - see both call
/// sites below) should run, given `default` (the `1.0 - node_fraction`
/// majority engine) and `node_fraction`. Uses an error-diffusion spread
/// (same idea as Bresenham line drawing) rather than `idx < node_fraction *
/// total`, so a 0.2 fraction gives one Node env per 5 spread evenly across
/// the whole index range instead of clumped at the start - matters because
/// autoscale-grown envs (see the second call site) get appended at
/// ever-increasing indices, and clumping would make the *ratio* drift as
/// more envs get added rather than staying fixed at `node_fraction`.
fn engine_for_idx(idx: usize, default: EngineKind, node_fraction: f64) -> EngineKind {
    if node_fraction <= 0.0 {
        return default;
    }
    if node_fraction >= 1.0 {
        return EngineKind::Node;
    }
    let before = (idx as f64 * node_fraction).floor() as i64;
    let after = ((idx + 1) as f64 * node_fraction).floor() as i64;
    if after > before {
        EngineKind::Node
    } else {
        default
    }
}

fn spawn_worker(
    idx: usize,
    stage: usize,
    max_ticks: i64,
    engine: EngineKind,
) -> Result<(Worker, PreparedObs)> {
    let (choice_tx, choice_rx) = mpsc::channel::<Choice>();
    let (stage_tx, stage_rx) = mpsc::channel::<usize>();
    let (obs_tx, obs_rx) = mpsc::channel();
    let (init_tx, init_rx) = mpsc::channel::<Result<PreparedObs, String>>();
    let handle = std::thread::Builder::new()
        .name(format!("env{idx}"))
        .spawn(move || {
            let mut w = match EnvWorker::new(idx, stage, max_ticks, engine) {
                Ok(w) => w,
                Err(e) => {
                    let _ = init_tx.send(Err(format!("{e:#}")));
                    return;
                }
            };
            let first = w.prepare();
            if init_tx.send(Ok(first)).is_err() {
                return;
            }
            while let Ok(choice) = choice_rx.recv() {
                // Curriculum advance: take the newest pending stage (if any);
                // takes effect at this env's *next* episode reset inside
                // `step()`, matching `rl/vec.py::set_stage`'s per-episode
                // sampling (never mid-episode).
                let mut new_stage = None;
                while let Ok(s) = stage_rx.try_recv() {
                    new_stage = Some(s);
                }
                if let Some(s) = new_stage {
                    w.set_stage(s);
                }
                let result = w.step(&choice).map_err(|e| format!("{e:#}"));
                if obs_tx.send(result).is_err() {
                    break;
                }
            }
            w.close();
        })?;
    let first = init_rx
        .recv()
        .map_err(|_| anyhow!("env {idx} died before first obs"))?
        .map_err(|e| anyhow!("env {idx}: {e}"))?;
    Ok((
        Worker {
            choice_tx,
            stage_tx,
            obs_rx,
            handle,
        },
        first,
    ))
}

fn action_needs_player(a: i64) -> bool {
    policy::needs_player(ACTIONS[a as usize])
}
fn action_needs_tile(a: i64) -> bool {
    policy::needs_tile(ACTIONS[a as usize])
}
fn action_needs_quantity(a: i64) -> bool {
    policy::needs_quantity(ACTIONS[a as usize])
}

const ACT_DISCRETE_FIELDS: usize = 5;
const ACT_FLOAT_FIELDS: usize = 3;

/// Host representation of one policy act batch. Discrete and floating-point
/// outputs stay in their native types, so packing does not change any bits.
struct PackedActHost {
    /// Row-major `(action, player, tile, build, nuke)` values.
    discrete: Vec<i64>,
    /// Row-major `(quantity, logp, value)` values.
    floats: Vec<f32>,
    len: usize,
}

impl PackedActHost {
    fn from_parts(discrete: Vec<i64>, floats: Vec<f32>, len: usize) -> Result<Self> {
        let expected_discrete = len
            .checked_mul(ACT_DISCRETE_FIELDS)
            .ok_or_else(|| anyhow!("act batch length {len} overflows discrete packed size"))?;
        let expected_floats = len
            .checked_mul(ACT_FLOAT_FIELDS)
            .ok_or_else(|| anyhow!("act batch length {len} overflows float packed size"))?;
        if discrete.len() != expected_discrete || floats.len() != expected_floats {
            return Err(anyhow!(
                "packed act result size mismatch for batch {len}: got {} discrete / {} float, expected {expected_discrete} / {expected_floats}",
                discrete.len(),
                floats.len()
            ));
        }
        Ok(Self {
            discrete,
            floats,
            len,
        })
    }

    fn row(&self, i: usize) -> Result<(&[i64], &[f32])> {
        if i >= self.len {
            return Err(anyhow!(
                "packed act row {i} out of bounds for batch {}",
                self.len
            ));
        }
        let d = i * ACT_DISCRETE_FIELDS;
        let f = i * ACT_FLOAT_FIELDS;
        Ok((
            &self.discrete[d..d + ACT_DISCRETE_FIELDS],
            &self.floats[f..f + ACT_FLOAT_FIELDS],
        ))
    }
}

/// Pack policy outputs by dtype before crossing the device boundary. A single
/// mixed-dtype tensor would either lose i64 precision or promote all f32
/// values to f64 and increase transfer volume. These are therefore the
/// minimal two exact native-width device-to-host transfers.
fn transfer_act_results(
    a: &Tensor,
    player: &Tensor,
    tile: &Tensor,
    build: &Tensor,
    nuke: &Tensor,
    qty: &Tensor,
    logp: &Tensor,
    value: &Tensor,
    len: usize,
) -> Result<PackedActHost> {
    let discrete_cpu = Tensor::stack(&[a, player, tile, build, nuke], 1).to_device(Device::Cpu);
    let floats_cpu = Tensor::stack(&[qty, logp, value], 1).to_device(Device::Cpu);
    let discrete: Vec<i64> = discrete_cpu.reshape([-1]).try_into()?;
    let floats: Vec<f32> = floats_cpu.reshape([-1]).try_into()?;
    PackedActHost::from_parts(discrete, floats, len)
}

/// One (T, N) rollout buffer slot (N = envs in this shard).
struct Step {
    obs: PreparedObs,
    choice: ChoiceScalars,
    logp: f32,
    value: f32,
    reward: f32,
    done: bool,
}

/// Owns one shard's env workers and a snapshot of the policy used *only*
/// for `act()` during rollout collection - kept intentionally separate
/// from `LearnerShard` so a collector thread can read `policy`/`cur_obs`
/// concurrently with a learner thread mutating its own (different)
/// `VarStore` on the same device, with no shared mutable state between
/// the two (see module doc).
struct ActorShard {
    device: Device,
    workers: Vec<Worker>,
    cur_obs: Vec<PreparedObs>,
    /// CPU-only compact payloads are recycled after the learner drops the
    /// final immutable Step view. CUDA tensors never enter this arena.
    compact_host_arena: Arc<crate::vecenv::CompactHostArena>,
    /// Libtorch's CUDA current stream is thread-local. The persistent actor
    /// creates, uses, and drops this VarStore and every tensor derived from it
    /// on its one owner thread; no policy/AE tensor is sent through a channel.
    vs: nn::VarStore,
    policy: PolicyNet,
    ae: Option<crate::ae::AePair>,
    /// Shares that same actor-thread CUDA stream ownership. Cached terrain,
    /// shared fine/coarse inputs, and encoder outputs never enter a rollout.
    terrain_cache: crate::ae::TerrainDeviceCache,
}

/// One GPU replica's trainable weights/optimizer. `run()` holds one
/// `LearnerShard` per device (paired index-for-index with `ActorShard`);
/// with `num_gpus=1` there's exactly one and `sync_grads` degenerates to a
/// no-op for a 1-shard list.
struct LearnerShard {
    device: Device,
    vs: nn::VarStore,
    policy: PolicyNet,
    opt: nn::Optimizer,
}

/// Everything a training update needs out of one shard's rollout: the
/// (T, N) step buffer, the bootstrap value for GAE (computed with the
/// *same* actor policy that produced every other value estimate in the
/// buffer, for internal consistency), and any episodes that finished
/// during collection.
struct RolloutResult {
    buffer: Vec<Vec<Step>>,
    bootstrap_v: Vec<f32>,
    ep_infos: Vec<EpisodeInfo>,
    policy_version: u64,
    collect_seconds: f64,
}

fn validate_rollout_set(results: &[RolloutResult], expected_policy_version: u64) -> Result<()> {
    anyhow::ensure!(!results.is_empty(), "no rollout shards returned");
    for (shard, result) in results.iter().enumerate() {
        anyhow::ensure!(
            result.policy_version == expected_policy_version,
            "rollout shard {shard} used policy version {}, expected {}",
            result.policy_version,
            expected_policy_version
        );
        anyhow::ensure!(!result.buffer.is_empty(), "rollout shard {shard} is empty");
        let envs = result.buffer[0].len();
        anyhow::ensure!(envs > 0, "rollout shard {shard} has no envs");
        anyhow::ensure!(
            result.buffer.iter().all(|row| row.len() == envs),
            "rollout shard {shard} changed env width mid-rollout"
        );
        anyhow::ensure!(
            result.bootstrap_v.len() == envs,
            "rollout shard {shard} bootstrap width {} != rollout width {envs}",
            result.bootstrap_v.len()
        );
    }
    Ok(())
}

/// Collects one full (rollout_len, num_envs) rollout on `actor`'s policy
/// snapshot. Safe to run concurrently with a `LearnerShard` training on a
/// *different* update's data - this function never touches any
/// `LearnerShard` state, only `actor`'s own workers/`cur_obs`/`vs`/
/// `policy`, all owned exclusively by the caller's `&mut ActorShard`.
fn choice_from_act_vecs(
    i: usize,
    a_v: &[i64],
    player_v: &[i64],
    tile_v: &[i64],
    build_v: &[i64],
    nuke_v: &[i64],
    qty_v: &[f32],
) -> (Choice, ChoiceScalars) {
    choice_from_act_values(
        a_v[i],
        player_v[i],
        tile_v[i],
        build_v[i],
        nuke_v[i],
        qty_v[i],
    )
}

fn choice_from_act_values(
    act: i64,
    player: i64,
    tile: i64,
    build: i64,
    nuke: i64,
    qty: f32,
) -> (Choice, ChoiceScalars) {
    let np = action_needs_player(act);
    let nt = action_needs_tile(act);
    let nq = action_needs_quantity(act);
    let is_build = ACTIONS[act as usize] == "build";
    let is_nuke = ACTIONS[act as usize] == "launch_nuke";
    let choice = Choice {
        action: act,
        player_slot: np.then_some(player),
        tile_region: nt.then_some(tile),
        build_type: is_build.then_some(build),
        nuke_type: is_nuke.then_some(nuke),
        quantity_frac: nq.then_some(qty as f64),
    };
    let scalars = ChoiceScalars {
        action: act,
        player_slot: if np { player } else { -1 },
        tile_region: if nt { tile } else { -1 },
        build_type: if is_build { build } else { -1 },
        nuke_type: if is_nuke { nuke } else { -1 },
        quantity_frac: if nq { qty } else { -1.0 },
    };
    (choice, scalars)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ActorShape {
    hr: usize,
    wr: usize,
    gh: usize,
    gw: usize,
    cgh: usize,
    cgw: usize,
}

impl ActorShape {
    fn of(obs: &PreparedObs) -> Self {
        Self {
            hr: obs.ae_raw.hr,
            wr: obs.ae_raw.wr,
            gh: obs.gh,
            gw: obs.gw,
            // Fresh actor observations have cgh/cgw == 0 until AE encode.
            // The compact policy's native coarse shape is nevertheless
            // determined exactly by the fine grid.
            cgh: obs.gh.div_ceil(2),
            cgw: obs.gw.div_ceil(2),
        }
    }
}

#[derive(Debug)]
struct ActorShapeBuckets {
    /// New bucket-major position -> original worker-slice position.
    order: Vec<usize>,
    ranges: Vec<std::ops::Range<usize>>,
}

/// Stable exact-shape partition. Bucket order is first occurrence order and
/// each bucket retains worker order, making scheduling deterministic even for
/// interleaved map shapes.
fn actor_shape_buckets(items: &[PreparedObs]) -> ActorShapeBuckets {
    let mut grouped: Vec<(ActorShape, Vec<usize>)> = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let shape = ActorShape::of(item);
        if let Some((_, indices)) = grouped.iter_mut().find(|(key, _)| *key == shape) {
            indices.push(index);
        } else {
            grouped.push((shape, vec![index]));
        }
    }

    let mut order = Vec::with_capacity(items.len());
    let mut ranges = Vec::with_capacity(grouped.len());
    for (_, indices) in grouped {
        let begin = order.len();
        order.extend(indices);
        ranges.push(begin..order.len());
    }
    ActorShapeBuckets { order, ranges }
}

/// Reorders without cloning the large host observation payloads.
/// `order[new_position]` names the old position to place there.
fn permute_slice<T>(items: &mut [T], order: &[usize]) {
    debug_assert_eq!(items.len(), order.len());
    let mut old_at_position: Vec<usize> = (0..items.len()).collect();
    let mut position_of_old: Vec<usize> = (0..items.len()).collect();
    for new_position in 0..items.len() {
        let wanted_old = order[new_position];
        let current_position = position_of_old[wanted_old];
        if current_position == new_position {
            continue;
        }
        let displaced_old = old_at_position[new_position];
        items.swap(new_position, current_position);
        old_at_position.swap(new_position, current_position);
        position_of_old[wanted_old] = new_position;
        position_of_old[displaced_old] = current_position;
    }
}

fn inverse_permutation(order: &[usize]) -> Vec<usize> {
    let mut inverse = vec![0; order.len()];
    for (new_position, &old_position) in order.iter().enumerate() {
        inverse[old_position] = new_position;
    }
    inverse
}

struct ActorRow {
    choice: Choice,
    scalars: ChoiceScalars,
    logp: f32,
    value: f32,
}

fn act_contiguous_obs(
    actor: &mut ActorShard,
    cfg: &Config,
    start: usize,
    end: usize,
) -> Result<PackedActHost> {
    let n = end - start;
    // Every device tensor is created, consumed, and synchronously copied to
    // PackedActHost on this persistent actor thread.
    let obs_t = if let Some(ae) = actor.ae.as_ref() {
        if cfg.compact_rollout && cfg.foveate {
            batch::build_compact_rollout_obs(
                &mut actor.cur_obs[start..end],
                actor.device,
                cfg.pinned_h2d,
                cfg.fp16_rollout,
                ae,
                &mut actor.terrain_cache,
                &actor.compact_host_arena,
            )?
        } else {
            batch::build_rollout_obs(
                &mut actor.cur_obs[start..end],
                actor.device,
                cfg.pinned_h2d,
                cfg.fp16_rollout,
                ae,
                &mut actor.terrain_cache,
            )?
        }
    } else {
        let obs_refs: Vec<&PreparedObs> = actor.cur_obs[start..end].iter().collect();
        batch::build_obs(&obs_refs, actor.device, cfg.pinned_h2d, cfg.fp16_rollout)
    };
    let (a, player, tile, build, nuke, qty, logp, value) =
        tch::no_grad(|| actor.policy.act(&obs_t, false));
    transfer_act_results(&a, &player, &tile, &build, &nuke, &qty, &logp, &value, n)
}

/// Encode + act for a contiguous worker slice `[start, end)`. Returns
/// per-env choice scalars / logp / value aligned to the slice (length
/// `end - start`).
fn act_group(
    actor: &mut ActorShard,
    cfg: &Config,
    start: usize,
    end: usize,
) -> Result<(Vec<ChoiceScalars>, Vec<f32>, Vec<f32>)> {
    let n = end - start;
    if n == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }

    // Bucketing is specific to the persistent AE + compact actor path. Any
    // pre-encoded/compact extras are deliberately left on the established
    // mixed builder, as are legacy/no-AE and non-foveated configurations.
    let can_bucket = actor.ae.is_some()
        && cfg.compact_rollout
        && cfg.foveate
        && actor.cur_obs[start..end]
            .iter()
            .all(|obs| obs.compact.is_none() && obs.grid.is_none() && obs.grid_coarse.is_none());
    let buckets = actor_shape_buckets(&actor.cur_obs[start..end]);
    let use_buckets = can_bucket && buckets.ranges.len() > 1;
    let mut rows: Vec<Option<ActorRow>> = (0..n).map(|_| None).collect();

    if use_buckets {
        // Stochastic parity: each categorical remains an independent draw
        // from exactly the same per-observation logits, and each beta remains
        // an independent draw from the same (a,b), so regrouping has exact
        // joint-distribution parity. It cannot preserve seeded samples
        // bit-for-bit: libtorch multinomial consumes its thread-global stream
        // per call, while sample_beta_host intentionally seeds from entropy.
        // First-occurrence bucket order plus stable membership makes the call
        // schedule deterministic without claiming nonexistent sample identity.
        let inverse = inverse_permutation(&buckets.order);
        permute_slice(&mut actor.cur_obs[start..end], &buckets.order);
        // Restore worker observation order even when encode/inference returns
        // an error; a successful call scatters only CPU-owned scalar rows.
        let result = (|| -> Result<()> {
            for range in &buckets.ranges {
                let packed = act_contiguous_obs(
                    actor,
                    cfg,
                    start + range.start,
                    start + range.end,
                )?;
                for bucket_row in 0..range.len() {
                    let original = buckets.order[range.start + bucket_row];
                    let (discrete, floats) = packed.row(bucket_row)?;
                    let (choice, scalars) = choice_from_act_values(
                        discrete[0],
                        discrete[1],
                        discrete[2],
                        discrete[3],
                        discrete[4],
                        floats[0],
                    );
                    rows[original] = Some(ActorRow {
                        choice,
                        scalars,
                        logp: floats[1],
                        value: floats[2],
                    });
                }
            }
            Ok(())
        })();
        permute_slice(&mut actor.cur_obs[start..end], &inverse);
        result?;
    } else {
        let packed = act_contiguous_obs(actor, cfg, start, end)?;
        for (i, row) in rows.iter_mut().enumerate() {
            let (discrete, floats) = packed.row(i)?;
            let (choice, scalars) = choice_from_act_values(
                discrete[0],
                discrete[1],
                discrete[2],
                discrete[3],
                discrete[4],
                floats[0],
            );
            *row = Some(ActorRow {
                choice,
                scalars,
                logp: floats[1],
                value: floats[2],
            });
        }
    }

    let mut scalars = Vec::with_capacity(n);
    let mut logp_v = Vec::with_capacity(n);
    let mut value_v = Vec::with_capacity(n);
    for (i, row) in rows.into_iter().enumerate() {
        let row = row.ok_or_else(|| anyhow!("missing actor result for env {}", start + i))?;
        actor.workers[start + i]
            .choice_tx
            .send(row.choice)
            .map_err(|_| anyhow!("env {} choice channel closed", start + i))?;
        scalars.push(row.scalars);
        logp_v.push(row.logp);
        value_v.push(row.value);
    }
    Ok((scalars, logp_v, value_v))
}

#[cfg(test)]
mod packed_act_tests {
    use super::*;
    use std::sync::Arc;

    fn choice_bits(
        c: &Choice,
    ) -> (
        i64,
        Option<i64>,
        Option<i64>,
        Option<i64>,
        Option<i64>,
        Option<u64>,
    ) {
        (
            c.action,
            c.player_slot,
            c.tile_region,
            c.build_type,
            c.nuke_type,
            c.quantity_frac.map(f64::to_bits),
        )
    }

    fn scalar_bits(c: &ChoiceScalars) -> (i64, i64, i64, i64, i64, u32) {
        (
            c.action,
            c.player_slot,
            c.tile_region,
            c.build_type,
            c.nuke_type,
            c.quantity_frac.to_bits(),
        )
    }

    fn shaped_obs(id: usize, gh: usize, gw: usize) -> PreparedObs {
        let (hr, wr) = (gh * 8, gw * 8);
        let plane = gh * gw;
        let mut grid = vec![0.0; policy::C_GRID as usize * plane];
        for (i, value) in grid.iter_mut().enumerate() {
            *value = ((i * 13 + id * 17) % 97) as f32 / 97.0;
        }
        PreparedObs {
            compact: None,
            grid: Some(grid),
            grid_coarse: None,
            cgh: 0,
            cgw: 0,
            ae_raw: crate::ae::AeRaw {
                owners: vec![0; hr * wr],
                static_terrain: crate::ae::StaticTerrain {
                    key: crate::ae::TerrainCacheKey {
                        env_id: id as u64,
                        episode: 0,
                        static_id: id as u64,
                        hr,
                        wr,
                    },
                    map: Arc::from(format!("shape-{gh}x{gw}")),
                    land_mag: vec![0.0; 2 * hr * wr].into(),
                },
                fallout: crate::ae::pack_fallout(&vec![0; hr * wr], hr, wr),
                stat: vec![0.0; 6 * plane],
                hr,
                wr,
            },
            ego: vec![0.0; 3 * plane],
            db: vec![0.0; plane],
            transient: vec![0.0; ofcore::feat::N_TRANSIENT * plane],
            legal_tile: vec![1.0; plane],
            gh,
            gw,
            players: (0..ofcore::feat::MAX_SLOTS * ofcore::feat::P_FEAT)
                .map(|i| ((i * 7 + id * 11) % 53) as f32 / 53.0)
                .collect(),
            pmask: [1.0; ofcore::feat::MAX_SLOTS],
            scalars: {
                let mut values = [0.1; ofcore::feat::N_SCALARS];
                values[0] = id as f32;
                values
            },
            me_slot: 0,
            legal_actions: [1.0; ofcore::feat::N_ACTIONS],
            legal_ptarget: vec![
                1.0;
                ofcore::feat::N_ACTIONS * ofcore::feat::MAX_SLOTS
            ],
            legal_build: [1.0; ofcore::feat::N_BUILD],
            legal_nuke: [1.0; ofcore::feat::N_NUKE],
            local: vec![0.1; 5 * policy::LOCAL as usize * policy::LOCAL as usize],
        }
    }

    #[test]
    fn exact_shape_buckets_are_stable_and_permutation_restores_worker_order() {
        let mut obs = vec![
            shaped_obs(0, 6, 8),
            shaped_obs(1, 8, 6),
            shaped_obs(2, 6, 8),
            shaped_obs(3, 8, 6),
            shaped_obs(4, 6, 8),
        ];
        let buckets = actor_shape_buckets(&obs);
        assert_eq!(buckets.order, vec![0, 2, 4, 1, 3]);
        assert_eq!(buckets.ranges, vec![0..3, 3..5]);

        permute_slice(&mut obs, &buckets.order);
        assert_eq!(
            obs.iter().map(|item| item.scalars[0] as usize).collect::<Vec<_>>(),
            buckets.order
        );
        permute_slice(&mut obs, &inverse_permutation(&buckets.order));
        assert_eq!(
            obs.iter().map(|item| item.scalars[0] as usize).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
    }

    fn policy_rows(
        policy: &PolicyNet,
        items: &[PreparedObs],
        greedy: bool,
    ) -> Vec<(Vec<i64>, Vec<f32>)> {
        let buckets = actor_shape_buckets(items);
        let mut rows: Vec<Option<(Vec<i64>, Vec<f32>)>> =
            (0..items.len()).map(|_| None).collect();
        for range in &buckets.ranges {
            let refs: Vec<&PreparedObs> = buckets.order[range.clone()]
                .iter()
                .map(|&i| &items[i])
                .collect();
            let full = batch::build_obs(&refs, Device::Cpu, false, false);
            let compact = PolicyNet::compact_observation(&full);
            let (a, p, t, b, n, q, lp, v) = tch::no_grad(|| policy.act(&compact, greedy));
            let packed =
                transfer_act_results(&a, &p, &t, &b, &n, &q, &lp, &v, range.len()).unwrap();
            for bucket_row in 0..range.len() {
                let original = buckets.order[range.start + bucket_row];
                let (discrete, floats) = packed.row(bucket_row).unwrap();
                rows[original] = Some((discrete.to_vec(), floats.to_vec()));
            }
        }
        rows.into_iter().map(Option::unwrap).collect()
    }

    fn initialize_test_policy(vs: &nn::VarStore) {
        for (name, mut tensor) in vs.variables() {
            let salt = name.bytes().map(usize::from).sum::<usize>();
            let values: Vec<f32> = (0..tensor.numel())
                .map(|i| ((i * 37 + salt * 11) % 101) as f32 / 500.0 - 0.1)
                .collect();
            let values = Tensor::from_slice(&values).view(tensor.size().as_slice());
            tch::no_grad(|| tensor.copy_(&values));
        }
    }

    #[test]
    fn interleaved_exact_shape_greedy_outputs_scatter_like_singletons() {
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = PolicyNet::new(&vs.root(), false, true, 8, 1);
        initialize_test_policy(&vs);
        let items = vec![
            shaped_obs(0, 6, 8),
            shaped_obs(1, 8, 6),
            shaped_obs(2, 6, 8),
            shaped_obs(3, 8, 6),
            shaped_obs(4, 6, 8),
        ];
        let bucketed = policy_rows(&policy, &items, true);
        for (i, item) in items.iter().enumerate() {
            let singleton = policy_rows(&policy, std::slice::from_ref(item), true)
                .pop()
                .unwrap();
            let normalized = |row: &(Vec<i64>, Vec<f32>)| {
                choice_from_act_values(
                    row.0[0], row.0[1], row.0[2], row.0[3], row.0[4], row.1[0],
                )
                .1
            };
            let actual_choice = normalized(&bucketed[i]);
            let expected_choice = normalized(&singleton);
            assert_eq!(
                (
                    actual_choice.action,
                    actual_choice.player_slot,
                    actual_choice.tile_region,
                    actual_choice.build_type,
                    actual_choice.nuke_type,
                ),
                (
                    expected_choice.action,
                    expected_choice.player_slot,
                    expected_choice.tile_region,
                    expected_choice.build_type,
                    expected_choice.nuke_type,
                ),
                "semantic discrete choice row {i}"
            );
            assert!(
                (actual_choice.quantity_frac - expected_choice.quantity_frac).abs() <= 1e-6,
                "semantic quantity row {i}"
            );
            // Raw unused heads may choose different members of an exact tie;
            // they are deliberately not part of ChoiceScalars or logp.
            for (field, (&actual, &expected)) in bucketed[i].1[1..]
                .iter()
                .zip(&singleton.1[1..])
                .enumerate()
            {
                assert!(
                    (actual - expected).abs() <= 1e-5,
                    "logp/value field {field} row {i}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn interleaved_exact_shape_stochastic_samples_remain_valid() {
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = PolicyNet::new(&vs.root(), false, true, 8, 1);
        initialize_test_policy(&vs);
        let items = vec![
            shaped_obs(0, 6, 8),
            shaped_obs(1, 8, 6),
            shaped_obs(2, 6, 8),
            shaped_obs(3, 8, 6),
        ];
        let sampled = policy_rows(&policy, &items, false);
        for (i, (discrete, floats)) in sampled.iter().enumerate() {
            assert!((0..ofcore::feat::N_ACTIONS as i64).contains(&discrete[0]));
            assert!((0..ofcore::feat::MAX_SLOTS as i64).contains(&discrete[1]));
            assert!((0..ofcore::feat::N_BUILD as i64).contains(&discrete[3]));
            assert!((0..ofcore::feat::N_NUKE as i64).contains(&discrete[4]));
            assert!(discrete[2] >= 0, "negative tile row {i}");
            assert!((1e-4..=1.0 - 1e-4).contains(&floats[0]));
            assert!(floats[1].is_finite() && floats[2].is_finite());
        }
    }

    #[test]
    fn packed_transfer_matches_individual_reference_unpack_in_batch_order() {
        // Covers no-mask, player/tile/quantity, build, and nuke actions. The
        // reference vectors are the eight individual conversions act_group
        // used before packing.
        let actions = [0i64, 1, 4, 5, 9];
        let players = [10i64, 11, 12, 13, 14];
        let tiles = [20i64, 21, 22, 23, 24];
        let builds = [1i64, 2, 3, 4, 5];
        let nukes = [4i64, 3, 2, 1, 0];
        let quantities = [0.125f32, 0.25, 0.5, 0.75, 0.875];
        let logps = [-0.01f32, -1.25, -2.5, -3.75, -5.0];
        let values = [100.0f32, -20.0, 3.5, -0.0, 1.0 / 3.0];

        let a = Tensor::from_slice(&actions);
        let player = Tensor::from_slice(&players);
        let tile = Tensor::from_slice(&tiles);
        let build = Tensor::from_slice(&builds);
        let nuke = Tensor::from_slice(&nukes);
        let qty = Tensor::from_slice(&quantities);
        let logp = Tensor::from_slice(&logps);
        let value = Tensor::from_slice(&values);
        let packed = transfer_act_results(
            &a,
            &player,
            &tile,
            &build,
            &nuke,
            &qty,
            &logp,
            &value,
            actions.len(),
        )
        .unwrap();

        let a_v: Vec<i64> = (&a).try_into().unwrap();
        let player_v: Vec<i64> = (&player).try_into().unwrap();
        let tile_v: Vec<i64> = (&tile).try_into().unwrap();
        let build_v: Vec<i64> = (&build).try_into().unwrap();
        let nuke_v: Vec<i64> = (&nuke).try_into().unwrap();
        let qty_v: Vec<f32> = (&qty).try_into().unwrap();
        let logp_v: Vec<f32> = (&logp).try_into().unwrap();
        let value_v: Vec<f32> = (&value).try_into().unwrap();

        for i in 0..actions.len() {
            let (discrete, floats) = packed.row(i).unwrap();
            assert_eq!(
                discrete,
                &[a_v[i], player_v[i], tile_v[i], build_v[i], nuke_v[i]]
            );
            assert_eq!(
                floats.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                [qty_v[i], logp_v[i], value_v[i]]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );

            let packed_choice = choice_from_act_values(
                discrete[0],
                discrete[1],
                discrete[2],
                discrete[3],
                discrete[4],
                floats[0],
            );
            let reference_choice =
                choice_from_act_vecs(i, &a_v, &player_v, &tile_v, &build_v, &nuke_v, &qty_v);
            assert_eq!(
                choice_bits(&packed_choice.0),
                choice_bits(&reference_choice.0)
            );
            assert_eq!(
                scalar_bits(&packed_choice.1),
                scalar_bits(&reference_choice.1)
            );
        }
    }

    #[test]
    fn packed_transfer_preserves_native_integer_and_float_bounds_exactly() {
        let discrete_inputs = [
            [i64::MIN, i64::MAX],
            [-(1i64 << 54), 1i64 << 54],
            [-(1i64 << 24) - 1, (1i64 << 24) + 1],
            [-1, 0],
            [123_456_789_012_345, -123_456_789_012_345],
        ];
        let float_inputs = [
            [f32::MIN, f32::MAX],
            [-0.0, f32::MIN_POSITIVE],
            [f32::EPSILON, -f32::EPSILON],
        ];
        let d: Vec<Tensor> = discrete_inputs
            .iter()
            .map(|values| Tensor::from_slice(values))
            .collect();
        let f: Vec<Tensor> = float_inputs
            .iter()
            .map(|values| Tensor::from_slice(values))
            .collect();
        let packed =
            transfer_act_results(&d[0], &d[1], &d[2], &d[3], &d[4], &f[0], &f[1], &f[2], 2)
                .unwrap();

        for row in 0..2 {
            let (actual_d, actual_f) = packed.row(row).unwrap();
            assert_eq!(actual_d, &discrete_inputs.map(|field| field[row]));
            assert_eq!(
                actual_f.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                float_inputs.map(|field| field[row].to_bits()).to_vec()
            );
        }
    }

    #[test]
    fn packed_host_rejects_size_overflow_mismatches_and_out_of_bounds_rows() {
        assert!(PackedActHost::from_parts(Vec::new(), Vec::new(), usize::MAX).is_err());
        assert!(PackedActHost::from_parts(vec![0; 4], vec![0.0; 3], 1).is_err());
        assert!(PackedActHost::from_parts(vec![0; 5], vec![0.0; 2], 1).is_err());

        let one = PackedActHost::from_parts(vec![0; 5], vec![0.0; 3], 1).unwrap();
        assert!(one.row(1).is_err());
        let empty = PackedActHost::from_parts(Vec::new(), Vec::new(), 0).unwrap();
        assert!(empty.row(0).is_err());
    }
}

fn recv_group(
    actor: &mut ActorShard,
    start: usize,
    end: usize,
    scalars: &[ChoiceScalars],
    logp_v: &[f32],
    value_v: &[f32],
    step_row: &mut [Option<Step>],
    ep_infos: &mut Vec<EpisodeInfo>,
) -> Result<()> {
    for (j, i) in (start..end).enumerate() {
        let (next_obs, reward, done, info) = actor.workers[i]
            .obs_rx
            .recv()
            .map_err(|_| anyhow!("env {i} obs channel closed"))?
            .map_err(|e| anyhow!("env {i}: {e}"))?;
        if let Some(info) = info {
            ep_infos.push(info);
        }
        let prev_obs = std::mem::replace(&mut actor.cur_obs[i], next_obs);
        step_row[i] = Some(Step {
            obs: prev_obs,
            choice: scalars[j].clone(),
            logp: logp_v[j],
            value: value_v[j],
            reward: reward as f32,
            done,
        });
    }
    Ok(())
}

fn collect_rollout(actor: &mut ActorShard, cfg: &Config, policy_version: u64) -> Result<RolloutResult> {
    let collect_start = Instant::now();
    let n = actor.workers.len();
    let mut buffer: Vec<Vec<Step>> = Vec::with_capacity(cfg.rollout_len);
    let mut ep_infos = Vec::new();

    // Dual env-group pipelining (Python v4.1): split workers into two
    // halves; while group 0's engines step, encode+act group 1 (and vice
    // versa). Disabled or N<=1 → single lockstep group.
    let pipeline = cfg.pipeline_groups && n > 1;
    let half = if pipeline { n.div_ceil(2) } else { n };
    let g0 = (0, half);
    let g1 = (half, n);

    // Prime: act+send group 0 before the step loop (matches Python's
    // `pack0 = act_group(groups[0]); vec.send_group(...)` before `for t`).
    let (mut pack0_sc, mut pack0_lp, mut pack0_v) = act_group(actor, cfg, g0.0, g0.1)?;

    for t in 0..cfg.rollout_len {
        let mut step_row: Vec<Option<Step>> = (0..n).map(|_| None).collect();

        let mut pack1: Option<(Vec<ChoiceScalars>, Vec<f32>, Vec<f32>)> = None;
        if g1.0 < g1.1 {
            // Overlaps group 0 stepping (choices already in flight).
            pack1 = Some(act_group(actor, cfg, g1.0, g1.1)?);
        }

        recv_group(
            actor,
            g0.0,
            g0.1,
            &pack0_sc,
            &pack0_lp,
            &pack0_v,
            &mut step_row,
            &mut ep_infos,
        )?;

        if t + 1 < cfg.rollout_len {
            // Next-step act for g0 overlaps g1 stepping when pack1 is live.
            let next = act_group(actor, cfg, g0.0, g0.1)?;
            pack0_sc = next.0;
            pack0_lp = next.1;
            pack0_v = next.2;
        }

        if let Some((sc1, lp1, v1)) = pack1.as_ref() {
            recv_group(
                actor,
                g1.0,
                g1.1,
                sc1,
                lp1,
                v1,
                &mut step_row,
                &mut ep_infos,
            )?;
        }

        buffer.push(
            step_row
                .into_iter()
                .map(|s| s.expect("every env must produce a step"))
                .collect(),
        );
    }

    let bootstrap_v: Vec<f32> = {
        let buckets = actor_shape_buckets(&actor.cur_obs);
        let can_bucket = actor.ae.is_some()
            && cfg.compact_rollout
            && cfg.foveate
            && buckets.ranges.len() > 1
            && actor
                .cur_obs
                .iter()
                .all(|obs| obs.compact.is_none() && obs.grid.is_none() && obs.grid_coarse.is_none());
        if can_bucket {
            let ae = actor.ae.as_ref().expect("checked above");
            let mut values = vec![0.0; n];
            for range in &buckets.ranges {
                let obs_refs: Vec<&PreparedObs> = buckets.order[range.clone()]
                    .iter()
                    .map(|&i| &actor.cur_obs[i])
                    .collect();
                let obs_t = batch::build_compact_obs_with_ae(
                    &obs_refs,
                    actor.device,
                    cfg.pinned_h2d,
                    cfg.fp16_rollout,
                    ae,
                    &mut actor.terrain_cache,
                )?;
                let bucket_values: Vec<f32> =
                    (&tch::no_grad(|| actor.policy.value_only(&obs_t))).try_into()?;
                anyhow::ensure!(
                    bucket_values.len() == range.len(),
                    "bootstrap bucket width mismatch"
                );
                for (bucket_row, value) in bucket_values.into_iter().enumerate() {
                    values[buckets.order[range.start + bucket_row]] = value;
                }
            }
            values
        } else {
            let obs_t = if let Some(ae) = actor.ae.as_ref() {
                let obs_refs: Vec<&PreparedObs> = actor.cur_obs.iter().collect();
                if cfg.compact_rollout && cfg.foveate {
                    batch::build_compact_obs_with_ae(
                        &obs_refs,
                        actor.device,
                        cfg.pinned_h2d,
                        cfg.fp16_rollout,
                        ae,
                        &mut actor.terrain_cache,
                    )?
                } else {
                    batch::build_obs_with_ae_cached(
                        &obs_refs,
                        actor.device,
                        cfg.pinned_h2d,
                        cfg.fp16_rollout,
                        ae,
                        &mut actor.terrain_cache,
                    )?
                }
            } else {
                let obs_refs: Vec<&PreparedObs> = actor.cur_obs.iter().collect();
                batch::build_obs(&obs_refs, actor.device, cfg.pinned_h2d, cfg.fp16_rollout)
            };
            let v = tch::no_grad(|| actor.policy.value_only(&obs_t));
            (&v).try_into()?
        }
    };

    // Libtorch uses thread-local CUDA streams. Finish actor work before
    // producing the CPU-only result. This is required by the legacy path
    // before ActorShard moves back to the coordinator, and gives the
    // persistent command a strict completion boundary before refresh/eval.
    if let Device::Cuda(index) = actor.device {
        Cuda::synchronize(index as i64);
    }

    Ok(RolloutResult {
        buffer,
        bootstrap_v,
        ep_infos,
        policy_version,
        collect_seconds: collect_start.elapsed().as_secs_f64(),
    })
}

/// Result of a fixed-seed greedy eval pass (`run_eval`).
pub struct EvalResult {
    pub win: f64,
    pub score: f64,
    pub episodes: usize,
}

/// Deployment-style eval: fresh fixed-seed envs (worker `i` always plays
/// seed `w{i}-ep0` at this stage), greedy actions, one episode per env.
/// Mirrors `rl/ppo.py::run_eval`.
fn run_eval(
    policy: &PolicyNet,
    ae: &crate::ae::AePair,
    device: Device,
    stage: usize,
    episodes: usize,
    max_ticks: i64,
    engine: EngineKind,
    pinned_h2d: bool,
    fp16_rollout: bool,
    compact_rollout: bool,
) -> Result<EvalResult> {
    if episodes == 0 {
        return Ok(EvalResult {
            win: 0.0,
            score: 0.0,
            episodes: 0,
        });
    }
    let mut workers = Vec::with_capacity(episodes);
    let mut cur_obs = Vec::with_capacity(episodes);
    for i in 0..episodes {
        let (w, obs) = spawn_worker(i, stage, max_ticks, engine)?;
        workers.push(w);
        cur_obs.push(obs);
    }
    let stages = ofcore::curriculum::stages();
    let decision_ticks = stages[stage.min(stages.len().saturating_sub(1))]
        .decision_ticks
        .max(1) as u64;
    let step_cap = (max_ticks as u64 / decision_ticks) + 64;
    let mut results: Vec<Option<EpisodeInfo>> = vec![None; episodes];
    let mut terrain_cache = crate::ae::TerrainDeviceCache::new(device);

    for _ in 0..step_cap {
        let pending: Vec<usize> = (0..episodes).filter(|&i| results[i].is_none()).collect();
        if pending.is_empty() {
            break;
        }
        let refs: Vec<&PreparedObs> = pending.iter().map(|&i| &cur_obs[i]).collect();
        let obs_t = if compact_rollout {
            batch::build_compact_obs_with_ae(
                &refs,
                device,
                pinned_h2d,
                fp16_rollout,
                ae,
                &mut terrain_cache,
            )?
        } else {
            batch::build_obs_with_ae_cached(
                &refs,
                device,
                pinned_h2d,
                fp16_rollout,
                ae,
                &mut terrain_cache,
            )?
        };
        let (a, player, tile, build, nuke, qty, _logp, _value) =
            tch::no_grad(|| policy.act(&obs_t, true));
        let a_v: Vec<i64> = (&a).try_into()?;
        let player_v: Vec<i64> = (&player).try_into()?;
        let tile_v: Vec<i64> = (&tile).try_into()?;
        let build_v: Vec<i64> = (&build).try_into()?;
        let nuke_v: Vec<i64> = (&nuke).try_into()?;
        let qty_v: Vec<f32> = (&qty).try_into()?;

        for (bi, &ei) in pending.iter().enumerate() {
            let act = a_v[bi];
            let np = action_needs_player(act);
            let nt = action_needs_tile(act);
            let nq = action_needs_quantity(act);
            let is_build = ACTIONS[act as usize] == "build";
            let is_nuke = ACTIONS[act as usize] == "launch_nuke";
            let choice = Choice {
                action: act,
                player_slot: np.then_some(player_v[bi]),
                tile_region: nt.then_some(tile_v[bi]),
                build_type: is_build.then_some(build_v[bi]),
                nuke_type: is_nuke.then_some(nuke_v[bi]),
                quantity_frac: nq.then_some(qty_v[bi] as f64),
            };
            workers[ei]
                .choice_tx
                .send(choice)
                .map_err(|_| anyhow!("eval env {ei} choice channel closed"))?;
        }
        for (bi, &ei) in pending.iter().enumerate() {
            let _ = bi;
            let (next_obs, _reward, _done, info) = workers[ei]
                .obs_rx
                .recv()
                .map_err(|_| anyhow!("eval env {ei} obs channel closed"))?
                .map_err(|e| anyhow!("eval env {ei}: {e}"))?;
            cur_obs[ei] = next_obs;
            if let Some(info) = info {
                results[ei] = Some(info);
            }
        }
    }

    for w in workers {
        drop(w.choice_tx);
        let _ = w.handle.join();
    }

    let finished: Vec<&EpisodeInfo> = results.iter().filter_map(|r| r.as_ref()).collect();
    if finished.is_empty() {
        return Ok(EvalResult {
            win: 0.0,
            score: 0.0,
            episodes: 0,
        });
    }
    let n = finished.len() as f64;
    let win = finished
        .iter()
        .map(|e| if e.won { 1.0 } else { 0.0 })
        .sum::<f64>()
        / n;
    let score = finished.iter().map(|e| e.score).sum::<f64>() / n;
    Ok(EvalResult {
        win,
        score,
        episodes: finished.len(),
    })
}

/// Actor commands contain only ordinary CPU data. `weights` is a complete
/// VarStore byte serialization, never a VarStore or Tensor handle.
#[derive(Clone)]
struct CpuWeightMeta {
    name: String,
    shape: Vec<i64>,
    len: i64,
}

#[derive(Clone)]
struct CpuWeightSnapshot {
    meta: Vec<CpuWeightMeta>,
    values: Vec<f32>,
}

fn snapshot_weights(vs: &nn::VarStore) -> Result<CpuWeightSnapshot> {
    tch::no_grad(|| {
        let mut variables: Vec<_> = vs.variables().into_iter().collect();
        variables.sort_by(|(a, _), (b, _)| a.cmp(b));
        let meta = variables
            .iter()
            .map(|(name, tensor)| CpuWeightMeta {
                name: name.clone(),
                shape: tensor.size(),
                len: tensor.numel() as i64,
            })
            .collect();
        let flattened: Vec<Tensor> = variables
            .iter()
            .map(|(_, tensor)| tensor.flatten(0, -1))
            .collect();
        let refs: Vec<&Tensor> = flattened.iter().collect();
        let packed = Tensor::cat(&refs, 0)
            .to_device(Device::Cpu)
            .to_kind(Kind::Float);
        let values = Vec::<f32>::try_from(packed)?;
        Ok(CpuWeightSnapshot { meta, values })
    })
}

fn apply_weight_snapshot(vs: &nn::VarStore, snapshot: &CpuWeightSnapshot) -> Result<()> {
    let mut variables = vs.variables();
    anyhow::ensure!(
        variables.len() == snapshot.meta.len(),
        "weight snapshot variable count {} != destination {}",
        snapshot.meta.len(),
        variables.len()
    );
    let cpu = Tensor::from_slice(&snapshot.values);
    let mut offset = 0i64;
    for item in &snapshot.meta {
        let mut destination = variables
            .remove(&item.name)
            .ok_or_else(|| anyhow!("weight snapshot destination missing {}", item.name))?;
        anyhow::ensure!(
            destination.size() == item.shape,
            "weight snapshot shape mismatch for {}: {:?} != {:?}",
            item.name,
            item.shape,
            destination.size()
        );
        let source = cpu
            .narrow(0, offset, item.len)
            .view(item.shape.as_slice())
            .to_device(destination.device());
        tch::no_grad(|| destination.f_copy_(&source))?;
        offset += item.len;
    }
    anyhow::ensure!(
        offset as usize == snapshot.values.len(),
        "weight snapshot payload length mismatch"
    );
    Ok(())
}

enum ActorCommand {
    Collect { id: u64 },
    Refresh {
        id: u64,
        policy_version: u64,
        weights: CpuWeightSnapshot,
    },
    SetStage { id: u64, stage: usize },
    Eval { id: u64, stage: usize, episodes: usize },
    Shutdown { id: u64 },
}

impl ActorCommand {
    fn id(&self) -> u64 {
        match self {
            Self::Collect { id }
            | Self::Refresh { id, .. }
            | Self::SetStage { id, .. }
            | Self::Eval { id, .. }
            | Self::Shutdown { id } => *id,
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Collect { .. } => "collect",
            Self::Refresh { .. } => "refresh",
            Self::SetStage { .. } => "set-stage",
            Self::Eval { .. } => "eval",
            Self::Shutdown { .. } => "shutdown",
        }
    }
}

enum ActorReply {
    Ready { envs: usize },
    Collected { id: u64, result: RolloutResult },
    Eval { id: u64, result: EvalResult },
    Ack { id: u64 },
    Failed { id: u64, command: &'static str, error: String },
}

/// Enforces strict command ordering and monotonic policy snapshots.
#[derive(Debug)]
struct ActorProtocol {
    next_command_id: u64,
    policy_version: u64,
    stopped: bool,
}

impl ActorProtocol {
    fn new(policy_version: u64) -> Self {
        Self { next_command_id: 1, policy_version, stopped: false }
    }
    fn accept(&mut self, command: &ActorCommand) -> Result<()> {
        anyhow::ensure!(!self.stopped, "command received after shutdown");
        anyhow::ensure!(
            command.id() == self.next_command_id,
            "actor command ordering violation: got id {}, expected {}",
            command.id(),
            self.next_command_id
        );
        self.next_command_id += 1;
        if let ActorCommand::Refresh { policy_version, .. } = command {
            anyhow::ensure!(
                *policy_version > self.policy_version,
                "stale actor policy refresh: got version {}, current {}",
                policy_version,
                self.policy_version
            );
            self.policy_version = *policy_version;
        }
        if matches!(command, ActorCommand::Shutdown { .. }) {
            self.stopped = true;
        }
        Ok(())
    }
}

fn close_actor(actor: ActorShard) {
    for worker in actor.workers {
        drop(worker.choice_tx);
        let _ = worker.handle.join();
    }
}

fn build_actor_shard(
    shard_index: usize,
    device: Device,
    cfg: &Config,
    stage: usize,
    initial_weights: CpuWeightSnapshot,
) -> Result<ActorShard> {
    // All CUDA-bearing actor resources are created and destroyed on the
    // persistent actor thread.
    let vs = nn::VarStore::new(device);
    let policy = PolicyNet::new(&vs.root(), cfg.amp, cfg.foveate, cfg.gc, cfg.blocks);
    apply_weight_snapshot(&vs, &initial_weights)?;
    if let Device::Cuda(index) = device {
        Cuda::synchronize(index as i64);
    }
    let path = std::path::Path::new(&cfg.ae_ckpt);
    anyhow::ensure!(
        path.exists(),
        "AE checkpoint not found at {} — run scripts/export_safetensors.py \
         (or scripts/fetch_ae_encoders.sh) first",
        path.display()
    );
    let coarse = cfg.coarse_ckpt.as_ref().map(std::path::Path::new);
    let ae = Some(crate::ae::AePair::load(
        path, coarse, device, cfg.amp, true,
    )?);
    let mut workers = Vec::with_capacity(cfg.num_envs);
    let mut cur_obs = Vec::with_capacity(cfg.num_envs);
    for local_i in 0..cfg.num_envs {
        let idx = shard_index * cfg.num_envs + local_i;
        let engine = engine_for_idx(idx, cfg.engine, cfg.node_fraction);
        let (worker, obs) = spawn_worker(idx, stage, cfg.max_episode_ticks, engine)?;
        workers.push(worker);
        cur_obs.push(obs);
    }
    Ok(ActorShard {
        device,
        workers,
        cur_obs,
        compact_host_arena: Arc::new(crate::vecenv::CompactHostArena::default()),
        vs,
        policy,
        ae,
        terrain_cache: crate::ae::TerrainDeviceCache::new_persistent_actor(device),
    })
}

fn actor_loop(
    shard_index: usize,
    device: Device,
    cfg: Config,
    stage: usize,
    initial_policy_version: u64,
    initial_weights: CpuWeightSnapshot,
    command_rx: Receiver<ActorCommand>,
    reply_tx: Sender<ActorReply>,
) {
    let mut actor = match build_actor_shard(shard_index, device, &cfg, stage, initial_weights) {
        Ok(actor) => actor,
        Err(error) => {
            let _ = reply_tx.send(ActorReply::Failed {
                id: 0,
                command: "initialize",
                error: format!("{error:#}"),
            });
            return;
        }
    };
    if reply_tx.send(ActorReply::Ready { envs: actor.workers.len() }).is_err() {
        close_actor(actor);
        return;
    }
    let mut protocol = ActorProtocol::new(initial_policy_version);
    while let Ok(command) = command_rx.recv() {
        let id = command.id();
        let name = command.name();
        if let Err(error) = protocol.accept(&command) {
            let _ = reply_tx.send(ActorReply::Failed {
                id,
                command: name,
                error: format!("{error:#}"),
            });
            break;
        }
        let reply: Result<ActorReply> = match command {
            ActorCommand::Collect { id } => collect_rollout(&mut actor, &cfg, protocol.policy_version)
                .map(|result| ActorReply::Collected { id, result }),
            ActorCommand::Refresh { id, weights, .. } => {
                apply_weight_snapshot(&actor.vs, &weights).map(|_| {
                    if let Device::Cuda(index) = actor.device {
                        Cuda::synchronize(index as i64);
                    }
                    ActorReply::Ack { id }
                })
            }
            ActorCommand::SetStage { id, stage } => {
                for worker in &actor.workers {
                    let _ = worker.stage_tx.send(stage);
                }
                Ok(ActorReply::Ack { id })
            }
            ActorCommand::Eval { id, stage, episodes } => actor
                .ae
                .as_ref()
                .ok_or_else(|| anyhow!("eval requires AE encoders"))
                .and_then(|ae| run_eval(
                    &actor.policy,
                    ae,
                    actor.device,
                    stage,
                    episodes,
                    cfg.max_episode_ticks,
                    cfg.engine,
                    cfg.pinned_h2d,
                    cfg.fp16_rollout,
                    cfg.compact_rollout && cfg.foveate,
                ))
                .map(|result| ActorReply::Eval { id, result }),
            ActorCommand::Shutdown { id } => Ok(ActorReply::Ack { id }),
        };
        match reply {
            Ok(reply) => {
                let shutdown = protocol.stopped;
                if reply_tx.send(reply).is_err() || shutdown {
                    break;
                }
            }
            Err(error) => {
                let _ = reply_tx.send(ActorReply::Failed {
                    id,
                    command: name,
                    error: format!("{error:#}"),
                });
                break;
            }
        }
    }
    if let Device::Cuda(index) = actor.device {
        Cuda::synchronize(index as i64);
    }
    close_actor(actor);
}

fn actor_reply_error(shard_index: usize, expected: &str, reply: &ActorReply) -> anyhow::Error {
    match reply {
        ActorReply::Failed { id, command, error } => anyhow!(
            "persistent actor {shard_index} command {id} ({command}) failed: {error}"
        ),
        _ => anyhow!("persistent actor {shard_index} protocol error: expected {expected}"),
    }
}

struct PersistentActor {
    shard_index: usize,
    command_tx: Sender<ActorCommand>,
    reply_rx: Receiver<ActorReply>,
    handle: Option<JoinHandle<()>>,
    next_command_id: u64,
    pending_collect_id: Option<u64>,
}

impl PersistentActor {
    fn spawn(
        shard_index: usize,
        device: Device,
        cfg: Config,
        stage: usize,
        initial_policy_version: u64,
        initial_weights: CpuWeightSnapshot,
    ) -> Result<Self> {
        let (command_tx, command_rx) = mpsc::channel();
        let (reply_tx, reply_rx) = mpsc::channel();
        let handle = std::thread::Builder::new()
            .name(format!("actor-gpu{shard_index}"))
            .spawn(move || actor_loop(
                shard_index,
                device,
                cfg,
                stage,
                initial_policy_version,
                initial_weights,
                command_rx,
                reply_tx,
            ))?;
        let mut actor = Self {
            shard_index,
            command_tx,
            reply_rx,
            handle: Some(handle),
            next_command_id: 1,
            pending_collect_id: None,
        };
        match actor.recv_reply("initialization")? {
            ActorReply::Ready { envs } => {
                anyhow::ensure!(envs > 0, "persistent actor {shard_index} initialized with no envs");
                Ok(actor)
            }
            reply => Err(actor_reply_error(shard_index, "ready", &reply)),
        }
    }
    fn recv_reply(&mut self, phase: &str) -> Result<ActorReply> {
        let started = Instant::now();
        loop {
            match self.reply_rx.recv_timeout(Duration::from_secs(15)) {
                Ok(reply) => return Ok(reply),
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    eprintln!(
                        "[watchdog] actor shard {} still in {phase} after {:.0}s",
                        self.shard_index,
                        started.elapsed().as_secs_f64()
                    );
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    let shard_index = self.shard_index;
                    let status = self.join_status();
                    return Err(anyhow!(
                        "persistent actor {shard_index} reply channel closed{status}"
                    ));
                }
            }
        }
    }
    fn join_status(&mut self) -> String {
        let Some(handle) = self.handle.take() else { return String::new() };
        match handle.join() {
            Ok(()) => " (thread exited)".to_string(),
            Err(payload) => {
                let reason = payload.downcast_ref::<&str>().copied()
                    .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("non-string panic");
                format!(" (thread panicked: {reason})")
            }
        }
    }
    fn next_id(&mut self) -> u64 {
        let id = self.next_command_id;
        self.next_command_id += 1;
        id
    }
    fn send_collect(&mut self) -> Result<()> {
        anyhow::ensure!(self.pending_collect_id.is_none(), "persistent actor {} already collecting", self.shard_index);
        let id = self.next_id();
        self.command_tx.send(ActorCommand::Collect { id })?;
        self.pending_collect_id = Some(id);
        Ok(())
    }
    fn finish_collect(&mut self) -> Result<RolloutResult> {
        let expected = self.pending_collect_id.take()
            .ok_or_else(|| anyhow!("persistent actor {} has no pending collect", self.shard_index))?;
        match self.recv_reply("rollout collection")? {
            ActorReply::Collected { id, result } if id == expected => Ok(result),
            reply => Err(actor_reply_error(self.shard_index, "matching collect result", &reply)),
        }
    }
    fn request_ack(&mut self, command: ActorCommand) -> Result<()> {
        let expected = command.id();
        let phase = command.name();
        self.command_tx.send(command)?;
        match self.recv_reply(phase)? {
            ActorReply::Ack { id } if id == expected => Ok(()),
            reply => Err(actor_reply_error(self.shard_index, "matching acknowledgement", &reply)),
        }
    }
    fn refresh(&mut self, policy_version: u64, weights: CpuWeightSnapshot) -> Result<()> {
        let id = self.next_id();
        self.request_ack(ActorCommand::Refresh { id, policy_version, weights })
    }
    fn set_stage(&mut self, stage: usize) -> Result<()> {
        let id = self.next_id();
        self.request_ack(ActorCommand::SetStage { id, stage })
    }
    fn eval(&mut self, stage: usize, episodes: usize) -> Result<EvalResult> {
        let id = self.next_id();
        self.command_tx.send(ActorCommand::Eval { id, stage, episodes })?;
        match self.recv_reply("evaluation")? {
            ActorReply::Eval { id: reply_id, result } if reply_id == id => Ok(result),
            reply => Err(actor_reply_error(self.shard_index, "matching eval result", &reply)),
        }
    }
    fn shutdown(mut self) -> Result<()> {
        let id = self.next_id();
        self.request_ack(ActorCommand::Shutdown { id })?;
        let status = self.join_status();
        anyhow::ensure!(!status.contains("panicked"), "actor shutdown failed{status}");
        Ok(())
    }
}

impl Drop for PersistentActor {
    fn drop(&mut self) {
        if self.handle.is_none() {
            return;
        }
        let id = self.next_id();
        let _ = self.command_tx.send(ActorCommand::Shutdown { id });
        let _ = self.reply_rx.recv_timeout(Duration::from_secs(5));
        let _ = self.join_status();
    }
}

enum LearnerCommand {
    Train {
        id: u64,
        rollout: RolloutResult,
        lr: f64,
        ent_coef: f32,
        /// Global running statistic at the start of the update.  Every
        /// owner therefore computes the same adaptive return bound.
        ret_stat: RetStat,
    },
    SaveWeights { id: u64, path: String },
    Shutdown { id: u64 },
}

impl LearnerCommand {
    fn id(&self) -> u64 {
        match self {
            Self::Train { id, .. } | Self::SaveWeights { id, .. } | Self::Shutdown { id } => *id,
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Train { .. } => "train",
            Self::SaveWeights { .. } => "save-weights",
            Self::Shutdown { .. } => "shutdown",
        }
    }
}

enum GradientDecision {
    Apply(Arc<Vec<f32>>),
    Discard,
    Abort(String),
}

enum LearnerReply {
    Ready { shard: usize, params: i64 },
    Gradient {
        id: u64,
        shard: usize,
        epoch: usize,
        minibatch: usize,
        finite: bool,
        values: Vec<f32>,
    },
    Trained {
        id: u64,
        shard: usize,
        losses: (f64, f64, f64, f64),
        weights: CpuWeightSnapshot,
        ret_stat: RetStat,
        train_seconds: f64,
        snapshot_seconds: f64,
    },
    Ack { id: u64, shard: usize },
    Failed { id: u64, shard: usize, command: &'static str, error: String },
}

#[derive(Debug)]
struct LearnerProtocol {
    next_command_id: u64,
    stopped: bool,
}

impl LearnerProtocol {
    fn new() -> Self {
        Self { next_command_id: 1, stopped: false }
    }
    fn accept(&mut self, command: &LearnerCommand) -> Result<()> {
        anyhow::ensure!(!self.stopped, "learner command received after shutdown");
        anyhow::ensure!(
            command.id() == self.next_command_id,
            "learner command ordering violation: got id {}, expected {}",
            command.id(),
            self.next_command_id
        );
        self.next_command_id += 1;
        if matches!(command, LearnerCommand::Shutdown { .. }) {
            self.stopped = true;
        }
        Ok(())
    }
}

fn learner_loop(
    shard_index: usize,
    device: Device,
    cfg: Config,
    initial_weights: CpuWeightSnapshot,
    initial_lr: f64,
    mut rng: rand::rngs::SmallRng,
    command_rx: Receiver<LearnerCommand>,
    reply_tx: Sender<LearnerReply>,
    gradient_rx: Receiver<GradientDecision>,
) {
    let init_started = Instant::now();
    eprintln!("[phase] persistent learner initialization started");
    let initialized = (|| -> Result<LearnerShard> {
        let vs = nn::VarStore::new(device);
        let policy = PolicyNet::new(&vs.root(), cfg.amp, cfg.foveate, cfg.gc, cfg.blocks);
        apply_weight_snapshot(&vs, &initial_weights)?;
        let opt = nn::AdamW::default().build(&vs, initial_lr)?;
        if let Device::Cuda(index) = device {
            Cuda::synchronize(index as i64);
        }
        Ok(LearnerShard { device, vs, policy, opt })
    })();
    let mut learner = match initialized {
        Ok(learner) => {
            eprintln!(
                "[phase] persistent learner initialization finished in {:.3}s",
                init_started.elapsed().as_secs_f64()
            );
            learner
        }
        Err(error) => {
            let _ = reply_tx.send(LearnerReply::Failed {
                id: 0,
                shard: shard_index,
                command: "initialize",
                error: format!("{error:#}"),
            });
            return;
        }
    };
    let params = learner
        .vs
        .trainable_variables()
        .iter()
        .map(|tensor| tensor.numel() as i64)
        .sum();
    if reply_tx.send(LearnerReply::Ready { shard: shard_index, params }).is_err() {
        return;
    }
    let mut protocol = LearnerProtocol::new();
    while let Ok(command) = command_rx.recv() {
        let id = command.id();
        let name = command.name();
        if let Err(error) = protocol.accept(&command) {
            let _ = reply_tx.send(LearnerReply::Failed {
                id,
                shard: shard_index,
                command: name,
                error: format!("{error:#}"),
            });
            break;
        }
        let reply: Result<LearnerReply> = match command {
            LearnerCommand::Train {
                id,
                rollout,
                lr,
                ent_coef,
                ret_stat: initial_ret_stat,
            } => {
                learner.opt.set_lr(lr);
                let adaptive_ret_bound =
                    (RET_ADAPTIVE_N_STD * initial_ret_stat.std()).max(1.0);
                // Keep only this shard's exact contribution here. The hub
                // folds these partials into the global statistic in shard
                // order after all owners finish.
                let mut ret_stat = RetStat::default();
                let train_start = Instant::now();
                eprintln!("[phase] learner shard {shard_index} command {id} PPO started");
                let gradient_reply = reply_tx.clone();
                let mut sync = |owned: &mut LearnerShard,
                                epoch: usize,
                                minibatch: usize,
                                finite: bool|
                 -> Result<bool> {
                    let values = if finite {
                        flat_grad_to_cpu(owned)?
                    } else {
                        Vec::new()
                    };
                    gradient_reply.send(LearnerReply::Gradient {
                        id,
                        shard: shard_index,
                        epoch,
                        minibatch,
                        finite,
                        values,
                    })?;
                    match gradient_rx.recv()? {
                        GradientDecision::Apply(values) => {
                            apply_cpu_flat_grad(owned, values.as_slice())?;
                            Ok(true)
                        }
                        GradientDecision::Discard => Ok(false),
                        GradientDecision::Abort(error) => Err(anyhow!(error)),
                    }
                };
                let losses = train_update(
                    std::slice::from_mut(&mut learner),
                    std::slice::from_ref(&rollout),
                    &cfg,
                    &mut rng,
                    ent_coef,
                    &mut ret_stat,
                    true,
                    Some(adaptive_ret_bound),
                    Some(&mut sync),
                );
                let train_seconds = train_start.elapsed().as_secs_f64();
                losses.and_then(|losses| {
                    eprintln!(
                        "[phase] learner shard {shard_index} command {id} PPO finished in {train_seconds:.3}s; \
                         packed CPU snapshot started"
                    );
                    let snapshot_start = Instant::now();
                    let weights = snapshot_weights(&learner.vs)?;
                    let snapshot_seconds = snapshot_start.elapsed().as_secs_f64();
                    eprintln!(
                        "[phase] learner command {id} packed CPU snapshot finished in \
                         {snapshot_seconds:.3}s ({} MiB)",
                        weights.values.len() * std::mem::size_of::<f32>() / (1024 * 1024)
                    );
                    Ok(LearnerReply::Trained {
                        id,
                        shard: shard_index,
                        losses,
                        weights,
                        ret_stat,
                        train_seconds,
                        snapshot_seconds,
                    })
                })
            }
            LearnerCommand::SaveWeights { id, path } => {
                (|| -> Result<()> {
                    if shard_index == 0 {
                        save_atomic(&path, |tmp| Ok(learner.vs.save(tmp)?))?;
                    }
                    Ok(())
                })()
                .map(|_| LearnerReply::Ack { id, shard: shard_index })
            }
            LearnerCommand::Shutdown { id } => Ok(LearnerReply::Ack { id, shard: shard_index }),
        };
        match reply {
            Ok(reply) => {
                let stopped = protocol.stopped;
                if reply_tx.send(reply).is_err() || stopped {
                    break;
                }
            }
            Err(error) => {
                let _ = reply_tx.send(LearnerReply::Failed {
                    id,
                    shard: shard_index,
                    command: name,
                    error: format!("{error:#}"),
                });
                break;
            }
        }
    }
    if let Device::Cuda(index) = device {
        Cuda::synchronize(index as i64);
    }
}

struct TrainReply {
    losses: (f64, f64, f64, f64),
    weights: Vec<CpuWeightSnapshot>,
    train_seconds: f64,
    snapshot_seconds: f64,
    /// Test-only copies of the hub result at every optimizer barrier. This
    /// proves parity before clipping/Adam can obscure the source of drift.
    #[cfg(test)]
    averaged_gradients: Vec<Vec<f32>>,
}

struct PersistentLearner {
    command_txs: Vec<Sender<LearnerCommand>>,
    gradient_txs: Vec<Sender<GradientDecision>>,
    reply_rx: Receiver<LearnerReply>,
    handles: Vec<Option<JoinHandle<()>>>,
    next_command_id: u64,
    ret_stat: RetStat,
    epochs: usize,
    minibatches: usize,
}

impl PersistentLearner {
    fn spawn(
        devices: &[Device],
        cfg: Config,
        initial_weights: CpuWeightSnapshot,
        initial_lr: f64,
        mut rng: rand::rngs::SmallRng,
    ) -> Result<(Self, i64)> {
        let (reply_tx, reply_rx) = mpsc::channel();
        let mut command_txs = Vec::with_capacity(devices.len());
        let mut gradient_txs = Vec::with_capacity(devices.len());
        let mut handles = Vec::with_capacity(devices.len());
        // DDP ranks use the same epoch permutation over their disjoint local
        // batches. Giving each owner a different permutation is mathematically
        // equivalent in exact arithmetic, but changes floating-point reduction
        // order. For duplicated-rollout parity that produces g0 != g1, and
        // Adam's first-step normalization can amplify the tiny discrepancy.
        let shuffle_seed = rng.next_u64();
        for (shard, &device) in devices.iter().enumerate() {
            let (command_tx, command_rx) = mpsc::channel();
            let (gradient_tx, gradient_rx) = mpsc::channel();
            let thread_reply = reply_tx.clone();
            let thread_cfg = cfg.clone();
            let thread_weights = initial_weights.clone();
            let thread_rng = rand::rngs::SmallRng::seed_from_u64(shuffle_seed);
            let handle = std::thread::Builder::new()
                .name(format!("learner-gpu{shard}"))
                .spawn(move || learner_loop(
                    shard,
                    device,
                    thread_cfg,
                    thread_weights,
                    initial_lr,
                    thread_rng,
                    command_rx,
                    thread_reply,
                    gradient_rx,
                ))?;
            command_txs.push(command_tx);
            gradient_txs.push(gradient_tx);
            handles.push(Some(handle));
        }
        drop(reply_tx);
        let mut learner = Self {
            command_txs,
            gradient_txs,
            reply_rx,
            handles,
            next_command_id: 1,
            ret_stat: RetStat::default(),
            epochs: cfg.epochs,
            minibatches: cfg.minibatches.max(1),
        };
        let mut params = vec![None; devices.len()];
        for _ in devices {
            match learner.recv_reply("initialization")? {
                LearnerReply::Ready { shard, params: count } if shard < params.len() => {
                    params[shard] = Some(count);
                }
                reply => return Err(learner_reply_error("ready", &reply)),
            }
        }
        let first = params[0].ok_or_else(|| anyhow!("learner shard 0 did not initialize"))?;
        anyhow::ensure!(
            params.iter().all(|p| *p == Some(first)),
            "persistent learner parameter counts differ across shards: {params:?}"
        );
        Ok((learner, first))
    }
    fn next_id(&mut self) -> u64 {
        let id = self.next_command_id;
        self.next_command_id += 1;
        id
    }
    fn join_status(&mut self) -> String {
        let mut statuses = Vec::new();
        for (shard, handle) in self.handles.iter_mut().enumerate() {
            let Some(handle) = handle.take() else { continue };
            match handle.join() {
                Ok(()) => statuses.push(format!("shard {shard} exited")),
                Err(payload) => {
                    let reason = payload.downcast_ref::<&str>().copied()
                        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                        .unwrap_or("non-string panic");
                    statuses.push(format!("shard {shard} panicked: {reason}"));
                }
            }
        }
        if statuses.is_empty() { String::new() } else { format!(" ({})", statuses.join(", ")) }
    }
    fn recv_reply(&mut self, phase: &str) -> Result<LearnerReply> {
        let started = Instant::now();
        loop {
            match self.reply_rx.recv_timeout(Duration::from_secs(15)) {
                Ok(reply) => return Ok(reply),
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    if self
                        .handles
                        .iter()
                        .any(|handle| handle.as_ref().is_some_and(JoinHandle::is_finished))
                    {
                        self.abort_barrier("learner owner exited during barrier");
                        let finished: Vec<usize> = self
                            .handles
                            .iter()
                            .enumerate()
                            .filter_map(|(shard, handle)| {
                                handle.as_ref().is_some_and(JoinHandle::is_finished).then_some(shard)
                            })
                            .collect();
                        return Err(anyhow!(
                            "persistent learner owner(s) {finished:?} exited while waiting in {phase}"
                        ));
                    }
                    eprintln!(
                        "[watchdog] learner still in {phase} after {:.0}s",
                        started.elapsed().as_secs_f64()
                    );
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    let status = self.join_status();
                    return Err(anyhow!("persistent learner reply channel closed{status}"));
                }
            }
        }
    }
    fn train(&mut self, rollouts: Vec<RolloutResult>, lr: f64, ent_coef: f32) -> Result<TrainReply> {
        anyhow::ensure!(
            rollouts.len() == self.command_txs.len(),
            "persistent learner got {} rollouts for {} shards",
            rollouts.len(),
            self.command_txs.len()
        );
        let id = self.next_id();
        for (tx, rollout) in self.command_txs.iter().zip(rollouts) {
            tx.send(LearnerCommand::Train {
                id,
                rollout,
                lr,
                ent_coef,
                ret_stat: self.ret_stat,
            })?;
        }
        let world = self.command_txs.len();
        #[cfg(test)]
        let mut averaged_gradients = Vec::with_capacity(self.epochs * self.minibatches);
        for epoch in 0..self.epochs {
            for minibatch in 0..self.minibatches {
                let mut packets: Vec<Option<(bool, Vec<f32>)>> =
                    (0..world).map(|_| None).collect();
                for _ in 0..world {
                    match self.recv_reply("CPU gradient barrier")? {
                        LearnerReply::Gradient {
                            id: reply_id,
                            shard,
                            epoch: reply_epoch,
                            minibatch: reply_mb,
                            finite,
                            values,
                        } if reply_id == id
                            && reply_epoch == epoch
                            && reply_mb == minibatch
                            && shard < world
                            && packets[shard].is_none() =>
                        {
                            packets[shard] = Some((finite, values));
                        }
                        reply => {
                            self.abort_barrier("gradient barrier protocol failure");
                            return Err(learner_reply_error("matching gradient packet", &reply));
                        }
                    }
                }
                let all_finite = packets
                    .iter()
                    .all(|packet| packet.as_ref().is_some_and(|(finite, _)| *finite));
                if !all_finite {
                    for tx in &self.gradient_txs {
                        let _ = tx.send(GradientDecision::Discard);
                    }
                    continue;
                }
                let ordered: Vec<Vec<f32>> = packets
                    .into_iter()
                    .map(|packet| packet.expect("all packets present").1)
                    .collect();
                let average = Arc::new(average_cpu_gradients(&ordered)?);
                #[cfg(test)]
                averaged_gradients.push(average.as_ref().clone());
                for tx in &self.gradient_txs {
                    tx.send(GradientDecision::Apply(average.clone()))?;
                }
            }
        }
        let mut trained: Vec<Option<((f64, f64, f64, f64), CpuWeightSnapshot, RetStat, f64, f64)>> =
            (0..world).map(|_| None).collect();
        for _ in 0..world {
            match self.recv_reply("training and CPU weight snapshots")? {
                LearnerReply::Trained {
                    id: reply_id,
                    shard,
                    losses,
                    weights,
                    ret_stat,
                    train_seconds,
                    snapshot_seconds,
                } if reply_id == id && shard < world && trained[shard].is_none() => {
                    trained[shard] =
                        Some((losses, weights, ret_stat, train_seconds, snapshot_seconds));
                }
                reply => return Err(learner_reply_error("matching train result", &reply)),
            }
        }
        let mut losses = (0.0, 0.0, 0.0, 0.0);
        let mut weights = Vec::with_capacity(world);
        let mut train_seconds = 0.0f64;
        let mut snapshot_seconds = 0.0f64;
        for entry in trained {
            let (local, snapshot, stat, train_s, snapshot_s) = entry.expect("all shards trained");
            losses.0 += local.0 / world as f64;
            losses.1 += local.1 / world as f64;
            losses.2 += local.2 / world as f64;
            losses.3 += local.3 / world as f64;
            self.ret_stat.add_batch(
                stat.count,
                stat.sum,
                stat.sum_sq,
            );
            weights.push(snapshot);
            train_seconds = train_seconds.max(train_s);
            snapshot_seconds = snapshot_seconds.max(snapshot_s);
        }
        Ok(TrainReply {
            losses,
            weights,
            train_seconds,
            snapshot_seconds,
            #[cfg(test)]
            averaged_gradients,
        })
    }
    fn abort_barrier(&self, error: &str) {
        for tx in &self.gradient_txs {
            let _ = tx.send(GradientDecision::Abort(error.to_string()));
        }
    }
    fn save_weights(&mut self, path: &str) -> Result<()> {
        let id = self.next_id();
        for tx in &self.command_txs {
            tx.send(LearnerCommand::SaveWeights {
                id,
                path: path.to_string(),
            })?;
        }
        let mut seen = vec![false; self.command_txs.len()];
        for _ in 0..self.command_txs.len() {
            match self.recv_reply("checkpoint save")? {
                LearnerReply::Ack { id: reply_id, shard }
                    if reply_id == id && shard < seen.len() && !seen[shard] =>
                {
                    seen[shard] = true;
                }
                reply => return Err(learner_reply_error("matching save acknowledgement", &reply)),
            }
        }
        Ok(())
    }
    fn shutdown(mut self) -> Result<()> {
        let id = self.next_id();
        for tx in &self.command_txs {
            tx.send(LearnerCommand::Shutdown { id })?;
        }
        let mut seen = vec![false; self.command_txs.len()];
        for _ in 0..self.command_txs.len() {
            match self.recv_reply("shutdown")? {
                LearnerReply::Ack { id: reply_id, shard }
                    if reply_id == id && shard < seen.len() && !seen[shard] =>
                {
                    seen[shard] = true;
                }
                reply => return Err(learner_reply_error("matching shutdown acknowledgement", &reply)),
            }
        }
        let status = self.join_status();
        anyhow::ensure!(!status.contains("panicked"), "learner shutdown failed{status}");
        Ok(())
    }
}

impl Drop for PersistentLearner {
    fn drop(&mut self) {
        if self.handles.iter().all(Option::is_none) {
            return;
        }
        let id = self.next_id();
        self.abort_barrier("persistent learner group dropped");
        for tx in &self.command_txs {
            let _ = tx.send(LearnerCommand::Shutdown { id });
        }
        for _ in 0..self.command_txs.len() {
            let _ = self.reply_rx.recv_timeout(Duration::from_secs(5));
        }
        let _ = self.join_status();
    }
}

fn learner_reply_error(expected: &str, reply: &LearnerReply) -> anyhow::Error {
    match reply {
        LearnerReply::Failed { id, shard, command, error } => {
            anyhow!("persistent learner shard {shard} command {id} ({command}) failed: {error}")
        }
        _ => anyhow!("persistent learner protocol error: expected {expected}"),
    }
}

fn flat_grad_to_cpu(shard: &LearnerShard) -> Result<Vec<f32>> {
    let parts: Vec<Tensor> = shard
        .vs
        .trainable_variables()
        .iter()
        .map(|v| v.grad().reshape([-1]))
        .collect();
    let flat = Tensor::cat(&parts, 0).to_device(Device::Cpu).to_kind(Kind::Float);
    Ok((&flat).try_into()?)
}

fn apply_cpu_flat_grad(shard: &mut LearnerShard, values: &[f32]) -> Result<()> {
    let flat = Tensor::from_slice(values).to_device(shard.device);
    let mut offset = 0i64;
    for variable in shard.vs.trainable_variables() {
        let mut grad = variable.grad();
        let len = grad.numel() as i64;
        let source = flat.narrow(0, offset, len).reshape(grad.size());
        grad.f_copy_(&source)?;
        offset += len;
    }
    anyhow::ensure!(offset as usize == values.len(), "flat gradient length mismatch");
    Ok(())
}

fn average_cpu_gradients(ordered: &[Vec<f32>]) -> Result<Vec<f32>> {
    anyhow::ensure!(!ordered.is_empty(), "cannot average zero gradient shards");
    let len = ordered[0].len();
    anyhow::ensure!(
        ordered.iter().all(|gradient| gradient.len() == len),
        "gradient shard lengths differ"
    );
    let mut average = vec![0.0f32; len];
    // Fixed shard-index order makes the floating-point reduction repeatable
    // regardless of which owner reached the barrier first.
    for gradient in ordered {
        for (sum, value) in average.iter_mut().zip(gradient) {
            *sum += *value;
        }
    }
    let world = ordered.len() as f32;
    for value in &mut average {
        *value /= world;
    }
    Ok(average)
}

/// Averages gradients across all shards and writes the average back onto
/// every shard's own device, in place - equivalent to
/// `dist.all_reduce(flat); flat /= world` in `rl/ppo.py`, minus the NCCL
/// collective (see module doc). No-op for a single shard.
///
/// Flattens every shard's per-parameter grads into one contiguous 1D
/// tensor before crossing devices, mirroring the "flat" in that Python
/// reference, rather than copying+averaging each of the ~hundreds of
/// individual parameter tensors one at a time: each per-parameter
/// `to_device`/`copy_` pays its own CUDA kernel-launch + host-sync round
/// trip, and with O(n_params) of those on every optimizer step this adds
/// up. One `cat` + one cross-device add per shard fixes that; the
/// assumption (true for this architecture - `PolicyNet::evaluate` densely
/// computes every head every forward pass, masking contributions post-hoc
/// rather than branching) is every trainable variable gets a defined grad
/// every backward pass, so all shards' flattened grad vectors line up
/// index-for-index.
fn sync_grads(shards: &[LearnerShard]) {
    if shards.len() <= 1 {
        return;
    }
    let hub = shards[0].device;
    let n = shards.len() as f64;
    let var_lists: Vec<Vec<Tensor>> = shards.iter().map(|s| s.vs.trainable_variables()).collect();

    let flats: Vec<Tensor> = var_lists
        .iter()
        .map(|vars| {
            let parts: Vec<Tensor> = vars.iter().map(|v| v.grad().reshape([-1])).collect();
            Tensor::cat(&parts, 0)
        })
        .collect();

    let mut acc = flats[0].to_device(hub);
    for f in &flats[1..] {
        acc += f.to_device(hub);
    }
    let avg = acc / n;

    for (shard, vars) in shards.iter().zip(&var_lists) {
        let avg_local = avg.to_device(shard.device);
        let mut offset: i64 = 0;
        for v in vars {
            let mut g = v.grad();
            let numel = g.numel() as i64;
            let piece = avg_local.narrow(0, offset, numel).reshape(g.size());
            let _ = g.f_copy_(&piece);
            offset += numel;
        }
    }
}

/// True if any shard's (pg, v, ent, entq) loss tuple has a non-finite
/// (NaN/Inf) component - see `train_update`'s NaN guard (docs/devlog.html's
/// 2026-07-12 entry) for why this gates skipping `opt.step()` for a
/// minibatch rather than applying a poisoned gradient.
fn any_loss_non_finite(losses: &[(f64, f64, f64, f64)]) -> bool {
    losses.iter().any(|(pg, v, ent, entq)| {
        !pg.is_finite() || !v.is_finite() || !ent.is_finite() || !entq.is_finite()
    })
}

/// Running (all-updates-so-far) mean/variance of the value-loss target,
/// used to derive an *adaptive* second bound on top of the fixed
/// `ret_clip` ceiling (see `train_update`'s batch-build stage for where
/// this is applied). `ret_clip` alone is a single guessed constant - fine
/// as an absolute safety ceiling, but it does nothing to help the value
/// head when the *typical* return scale is much smaller than that
/// ceiling (observed live: healthy early updates had v-loss ~0.02-10, but
/// once the value head drifted, it plateaued in the tens-of-thousands
/// range - well under ret_clip=3000's *square* but nowhere near the
/// scale the value head should actually be predicting). Adapting the
/// bound to `N_STD` standard deviations of what returns have actually
/// looked like so far gives the value head a target scale that tracks
/// the real data instead of a fixed guess, while `ret_clip` still caps
/// the absolute worst case unconditionally.
///
/// Uses plain sum/sum-of-squares accumulation (not Welford) specifically
/// because it composes trivially across the per-shard parallel threads in
/// `train_update`'s batch-build stage: each shard accumulates its own
/// partial (count, sum, sum_sq) locally with no cross-thread
/// synchronization, and merging them into the single running total after
/// the parallel scope closes is exact plain addition - no merge formula
/// needed.
#[derive(Default, Clone, Copy)]
pub struct RetStat {
    count: f64,
    sum: f64,
    sum_sq: f64,
}

impl RetStat {
    fn add_batch(&mut self, count: f64, sum: f64, sum_sq: f64) {
        self.count += count;
        self.sum += sum;
        self.sum_sq += sum_sq;
    }

    fn mean(&self) -> f64 {
        if self.count < 1.0 {
            0.0
        } else {
            self.sum / self.count
        }
    }

    /// Population std of everything seen so far. Deliberately returns
    /// `f64::INFINITY` until there's enough data to estimate it at all
    /// (fewer than 2 samples) - callers use this to mean "don't apply the
    /// adaptive bound yet", the same way `ret_clip=0.0` means "disabled"
    /// for the fixed bound.
    fn std(&self) -> f64 {
        if self.count < 2.0 {
            return f64::INFINITY;
        }
        let m = self.mean();
        (self.sum_sq / self.count - m * m).max(0.0).sqrt().max(1e-3)
    }
}

/// Runs GAE + the `epochs` x `minibatches` clipped-PPO update for one
/// update's worth of rollouts (one `RolloutResult` per learner shard).
/// Pure compute over `learners`/`pending` - never touches any
/// `ActorShard`, so it's safe to run concurrently with the *next*
/// update's `collect_rollout` calls (see module doc).
fn train_update(
    learners: &mut [LearnerShard],
    pending: &[RolloutResult],
    cfg: &Config,
    rng: &mut rand::rngs::SmallRng,
    ent_coef: f32,
    ret_stat: &mut RetStat,
    exclusive_owner: bool,
    adaptive_ret_bound_override: Option<f64>,
    mut persistent_sync: Option<
        &mut dyn FnMut(&mut LearnerShard, usize, usize, bool) -> Result<bool>,
    >,
) -> Result<(f64, f64, f64, f64)> {
    debug_assert!(!exclusive_owner || learners.len() == 1);
    let t_len = cfg.rollout_len;
    // Derive the live per-shard env count from the actual rollout data,
    // not `cfg.num_envs` - `auto_scale_envs` can grow every `ActorShard`'s
    // workers between updates (see `run()`), and every shard is grown in
    // lockstep (an all-or-nothing spawn across shards, rolled back on any
    // single failure - see `run()`) specifically so every `RolloutResult`
    // in `pending` has the same buffer width and this single shared `n`
    // stays valid for every shard's minibatch/index-tensor math below.
    // Trusting the *original* `cfg.num_envs` here after a scale-up would
    // silently misindex/corrupt the GAE and minibatch buffers the very
    // next update.
    let n = pending
        .first()
        .and_then(|r| r.buffer.first())
        .map(|row| row.len())
        .unwrap_or(cfg.num_envs);
    let total = t_len * n;
    let minibatch_size = (total / cfg.minibatches.max(1)).max(1);
    // Sums over every (epoch, minibatch) pair, averaged across shards -
    // returned as means, matching `rl/ppo.py`'s `*_sum / n_mb` (whose
    // entropy mean drives the entropy-floor controller; last-minibatch
    // snapshots read artificially noisy/low).
    let mut loss_sums = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let mut n_mb: usize = 0;

    // Per-shard full-rollout tensors, built once (CPU repack + one
    // host->device upload) and resident on that shard's device for the
    // whole update. `epochs`>1 revisits this same rollout under different
    // minibatch shuffles - rebuilding+re-uploading the observation grid
    // from scratch for every (epoch, minibatch) pair would dominate
    // update wall-clock while barely touching the GPU's compute units
    // (see DEVLOG). Minibatches instead `index_select` a GPU-resident
    // shuffled-index slice out of these.
    struct ShardBatch {
        obs: policy::Obs,
        choice: policy::ChoiceBatch,
        adv: Tensor,
        ret: Tensor,
        old_logp: Tensor,
    }
    let batch_build_t0 = Instant::now();
    // Per-shard: GAE (plain Rust, negligible) + one CPU repack/host->device
    // upload of the whole rollout (the actual cost here - tens of MB of
    // observation grid per shard). Spawn one thread per shard instead of a
    // sequential `for` loop: `RolloutResult`/`Device` hold no GPU tensor
    // handles (only plain floats/ints until `build_obs` allocates fresh
    // tensors *inside* each thread), so this is safe, and without it the
    // 4 shards' multi-second CPU repacks and independent-GPU H2D transfers
    // were serialized one after another instead of overlapping (measured:
    // ~8-9s sequential for 4 shards vs ~2s once parallelized - this was
    // the single largest remaining non-overlapped, GPU-idle chunk of
    // update wall-clock; see DEVLOG).
    // Read *before* this update's data is folded in, so the bound applied
    // to this batch reflects only prior updates (matching how a live
    // system would have to work - you can't normalize against data you
    // haven't seen yet). `RET_ADAPTIVE_N_STD * std` only kicks in once
    // `ret_stat` has enough samples (see `RetStat::std`'s doc); until
    // then this is `f64::INFINITY` and the adaptive bound is a no-op.
    let adaptive_ret_bound = adaptive_ret_bound_override
        .unwrap_or_else(|| (RET_ADAPTIVE_N_STD * ret_stat.std()).max(1.0));
    macro_rules! build_shard {
        ($gi:expr, $device:expr, $result:expr) => {{
                    let gi = $gi;
                    let result = $result;
                    let device = $device;
                    let buffer = &result.buffer;
                    let bootstrap_v = &result.bootstrap_v;
                    // Guards the uniform-per-shard-env-count invariant
                    // `n`'s derivation above relies on (see this
                    // function's top) - would only trip if a future
                    // change lets shards' env counts drift apart.
                    debug_assert_eq!(buffer.first().map(|row| row.len()).unwrap_or(n), n);
                    let mut adv = vec![vec![0.0f32; n]; t_len];
                    let mut last_gae = vec![0.0f32; n];
                    for t in (0..t_len).rev() {
                        for e in 0..n {
                            let next_value = if t == t_len - 1 { bootstrap_v[e] } else { buffer[t + 1][e].value };
                            let mask = if buffer[t][e].done { 0.0 } else { 1.0 };
                            let delta = buffer[t][e].reward + cfg.gamma * next_value * mask - buffer[t][e].value;
                            last_gae[e] = delta + cfg.gamma * cfg.lambda * mask * last_gae[e];
                            adv[t][e] = last_gae[e];
                        }
                    }

                    let mut adv_flat = vec![0.0f32; total];
                    let mut ret_flat = vec![0.0f32; total];
                    let mut old_logp_flat = vec![0.0f32; total];
                    // Tracks the pre-clamp raw return separately from
                    // `ret_flat` itself, so the diagnostic below still
                    // reports the *true* extreme value even once
                    // `--ret-clip` (see Config::ret_clip's doc) is
                    // clamping it out of the actual loss target - the
                    // whole point of this diagnostic is visibility into
                    // what the return *would have been*, now that the fix
                    // for the incident it's named after means `ret_flat`
                    // itself won't show it anymore.
                    let mut max_raw_ret: (f32, usize) = (0.0, 0);
                    for t in 0..t_len {
                        for e in 0..n {
                            let flat_idx = t * n + e;
                            adv_flat[flat_idx] = adv[t][e];
                            let ret = adv[t][e] + buffer[t][e].value;
                            if ret.abs() > max_raw_ret.0.abs() {
                                max_raw_ret = (ret, flat_idx);
                            }
                            let ret = if cfg.ret_clip > 0.0 { ret.clamp(-cfg.ret_clip, cfg.ret_clip) } else { ret };
                            // Adaptive bound (see RetStat's doc) applied
                            // on top of the fixed ret_clip ceiling above -
                            // always at least as tight, never looser.
                            ret_flat[flat_idx] = ret.clamp(-adaptive_ret_bound as f32, adaptive_ret_bound as f32);
                            old_logp_flat[flat_idx] = buffer[t][e].logp;
                        }
                    }
                    // Diagnostic for the 2026-07-12 value-loss-explosion
                    // incident (docs/devlog.html) - the original
                    // native-vs-Node-engine reward-mismatch hypothesis was
                    // revised after live data showed flagged env indices
                    // were a mix of both engines, not exclusively Node;
                    // current understanding is this is an inherent
                    // early-stage-0 PPO value instability, now bounded (not
                    // just guarded against NaN) by `--ret-clip`. Logs the
                    // *global* env-worker index (matching `spawn_worker`'s
                    // `idx = gi * cfg.num_envs + local_i` and hence
                    // `engine_for_idx`) in case a future recurrence's
                    // engine distribution ever looks different from this
                    // session's mixed result.
                    let (r, flat_idx) = max_raw_ret;
                    if r.abs() > 1000.0 {
                        let e_local = flat_idx % n;
                        let global_idx = gi * n + e_local;
                        eprintln!(
                            "[train] WARNING: extreme return {r:.1} (pre-clamp) at global env-worker \
                             index {global_idx} (shard={gi} local_e={e_local} t={}, value={:.1}) - \
                             see docs/devlog.html's 2026-07-12 NaN-guard entry",
                            flat_idx / n,
                            buffer[flat_idx / n][e_local].value
                        );
                    }
                    {
                        let adv_mean = adv_flat.iter().sum::<f32>() / total as f32;
                        let adv_var = adv_flat.iter().map(|x| (x - adv_mean).powi(2)).sum::<f32>() / total as f32;
                        let adv_std = adv_var.sqrt().max(1e-8);
                        for v in adv_flat.iter_mut() {
                            *v = (*v - adv_mean) / adv_std;
                        }
                        // Advantage clipping (see Config::adv_clip's doc) -
                        // closes the *other* half of the 2026-07-12
                        // incident this session spent hours chasing.
                        // Fixing the value loss's gradient boundedness
                        // (Huber, prior commit) was necessary but not
                        // sufficient: the policy loss's gradient w.r.t.
                        // logits is proportional to `adv_t` too (`surr1 =
                        // ratio * adv_t`, and d(ratio)/d(logp) = ratio, so
                        // d(loss)/d(logp) scales directly with adv_t) - and
                        // normalization *by itself* doesn't bound a single
                        // outlier's *normalized* magnitude at all. If one
                        // sample's raw advantage is a legitimate-but-rare
                        // outlier (the same GAE/return spikes this whole
                        // incident is about) while most of the batch's
                        // advantages are near zero, the population std
                        // that outlier gets divided by is *itself* tiny,
                        // so its normalized value can land tens or
                        // hundreds of std-devs out - directly explaining
                        // why entropy collapse and value explosion were
                        // observed happening together every single time:
                        // the same root return-spike was poisoning both
                        // loss terms' gradients simultaneously, and only
                        // one of the two was ever actually fixed before
                        // now.
                        if cfg.adv_clip > 0.0 {
                            for v in adv_flat.iter_mut() {
                                *v = v.clamp(-cfg.adv_clip, cfg.adv_clip);
                            }
                        }
                    }
                    let mut choice_flat: Vec<ChoiceScalars> = Vec::with_capacity(total);
                    for row in buffer.iter() {
                        for s in row.iter() {
                            choice_flat.push(s.choice.clone());
                        }
                    }
                    let mut obs_flat: Vec<&PreparedObs> = Vec::with_capacity(total);
                    for row in buffer.iter() {
                        for s in row {
                            obs_flat.push(&s.obs);
                        }
                    }
                    // Local partial sums for RetStat's global running
                    // total - see RetStat's doc for why plain sum/sum_sq
                    // accumulation (not Welford) composes trivially here:
                    // this shard's thread never touches any other
                    // shard's state, and the caller just adds these three
                    // numbers into the running total after every shard's
                    // thread has finished.
                    let local_count = ret_flat.len() as f64;
                    let local_sum: f64 = ret_flat.iter().map(|&x| x as f64).sum();
                    let local_sum_sq: f64 = ret_flat.iter().map(|&x| (x as f64) * (x as f64)).sum();

                    let obs = batch::build_obs(&obs_flat, device, cfg.pinned_h2d, cfg.fp16_rollout);
                    let choice = batch::build_choice_batch(&choice_flat, device, cfg.pinned_h2d);
                    let adv = Tensor::from_slice(&adv_flat).to_device(device);
                    let ret = Tensor::from_slice(&ret_flat).to_device(device);
                    let old_logp = Tensor::from_slice(&old_logp_flat).to_device(device);
                    // ShardBatch crosses from this short-lived builder thread
                    // to newly spawned learner threads. Complete its
                    // thread-local stream before handing device tensors over.
                    if let Device::Cuda(index) = device {
                        Cuda::synchronize(index as i64);
                    }
                    (ShardBatch { obs, choice, adv, ret, old_logp }, local_count, local_sum, local_sum_sq)
        }};
    }
    let batch_started = Instant::now();
    if exclusive_owner {
        eprintln!("[phase] learner ShardBatch construction started");
    }
    let shard_results: Vec<(ShardBatch, f64, f64, f64)> = if exclusive_owner {
        vec![build_shard!(0, learners[0].device, &pending[0])]
    } else {
        std::thread::scope(|s| {
            let handles: Vec<_> = learners
                .iter()
                .zip(pending.iter())
                .enumerate()
                .map(|(gi, (shard, result))| {
                    let device = shard.device;
                    s.spawn(move || build_shard!(gi, device, result))
                })
                .collect();
            handles.into_iter().map(|h| {
                h.join().unwrap_or_else(|e| {
                    // `.expect()` on a `thread::Result` only ever prints
                    // "Any { .. }" - the panic payload's `Box<dyn Any>` only
                    // downcasts cleanly for the exact type the panicking
                    // code used (usually `&str`/`String` for a `panic!()`,
                    // but a `.unwrap()` on a tensor op's `Result` carries a
                    // different payload type tch/libtorch chooses, which is
                    // exactly the case this session's crashes hit and had
                    // zero visibility into). Try both common payload shapes
                    // before giving up, so a future crash's actual message
                    // - not just its opaque type - ends up in the log.
                    let msg = e
                        .downcast_ref::<&str>()
                        .map(|s| s.to_string())
                        .or_else(|| e.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| format!("{e:?}"));
                    panic!("batch-build thread panicked: {msg}");
                })
            }).collect()
        })
    };
    if std::env::var("OFTRAIN_DIAG").is_ok() {
        println!(
            "[diag] batch_build_s={:.3}",
            batch_build_t0.elapsed().as_secs_f64()
        );
    }
    // Fold this update's per-shard partial sums into the running total
    // *after* they were used to compute `adaptive_ret_bound` above - the
    // bound applied to a given batch always reflects only prior updates
    // (see the comment there), never this one's own data.
    for (_, count, sum, sum_sq) in &shard_results {
        ret_stat.add_batch(*count, *sum, *sum_sq);
    }
    let mut shard_batches: Vec<ShardBatch> =
        shard_results.into_iter().map(|(batch, ..)| batch).collect();
    if exclusive_owner {
        eprintln!(
            "[phase] learner ShardBatch construction finished in {:.3}s",
            batch_started.elapsed().as_secs_f64()
        );
    }

    for _epoch in 0..cfg.epochs {
        let epoch_started = Instant::now();
        if exclusive_owner {
            eprintln!("[phase] learner PPO epoch {}/{} started", _epoch + 1, cfg.epochs);
        }
        // Per-shard shuffled index tensor, built once per epoch (CPU
        // shuffle + one tiny (total,) i64 upload) and resident on that
        // shard's device - minibatches `narrow` a contiguous slice of it
        // (a view, no host round trip) to `index_select` the full-batch
        // tensors above.
        let mut idx_vec: Vec<i64> = (0..total as i64).collect();
        let idx_tensors: Vec<Tensor> = learners
            .iter()
            .map(|shard| {
                idx_vec.shuffle(rng);
                Tensor::from_slice(&idx_vec).to_device(shard.device)
            })
            .collect();
        let n_minibatches = cfg.minibatches.max(1);

        // Pixel budget from the padded rollout Obs (shared across shards).
        let (gh, gw) = {
            let s = shard_batches[0].obs.grid.size();
            (s[2] as usize, s[3] as usize)
        };
        let (cgh, cgw) = match &shard_batches[0].obs.grid_coarse {
            Some(gc) => {
                let s = gc.size();
                (s[2] as usize, s[3] as usize)
            }
            None => (gh.div_ceil(2), gw.div_ceil(2)),
        };
        let pix_per = (gh * gw + cgh * cgw).max(1);
        let sub_size = (MAX_UPD_PIX / pix_per).max(1) as i64;

        for m in 0..n_minibatches {
            let start = (m * minibatch_size) as i64;
            let len = if m == n_minibatches - 1 {
                total as i64 - start
            } else {
                minibatch_size as i64
            };
            let mb_t0 = Instant::now();
            // Further split when `mb_size * pix_per > MAX_UPD_PIX` (mirror
            // `rl/ppo.py`). All samples share padded grid dims, so no
            // shape-grouping is needed — just chop and accumulate grads
            // with `w_sub = sub_len / mb_len` before one optimizer step.
            let n_subs = ((len + sub_size - 1) / sub_size) as usize;
            for shard in learners.iter_mut() {
                shard.opt.zero_grad();
            }
            let mut mb_loss_sums = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            let mut discard_mb = false;

            for s_i in 0..n_subs {
                let sub_start = start + (s_i as i64) * sub_size;
                let sub_len = if s_i == n_subs - 1 {
                    start + len - sub_start
                } else {
                    sub_size
                };
                let w_sub = sub_len as f64 / len as f64;
                // Forward + backward for every shard on its own OS thread:
                // `backward()` for shard 0 would otherwise fully finish,
                // including its implicit device sync, before shard 1 even
                // starts on a plain sequential loop.
                macro_rules! backward_shard {
                    ($shard:expr, $sb:expr, $idx_t:expr) => {{
                                let shard = $shard;
                                let sb = $sb;
                                let idx_t = $idx_t;
                                let obs_t = sb.obs.index_select(&idx_t);
                                let choice_t = sb.choice.index_select(&idx_t);
                                let adv_t = sb.adv.index_select(0, &idx_t);
                                let ret_t = sb.ret.index_select(0, &idx_t);
                                let old_logp_t = sb.old_logp.index_select(0, &idx_t);

                                let (logp, ent, ent_q, value) =
                                    shard.policy.evaluate(&obs_t, &choice_t);
                                // Bound log-ratio before exp (see prior
                                // pg_loss trillion-spike incident).
                                let log_ratio = (&logp - &old_logp_t).clamp(-20.0, 20.0);
                                let ratio = log_ratio.exp();
                                let surr1 = &ratio * &adv_t;
                                let surr2 = ratio
                                    .clamp(1.0 - cfg.clip as f64, 1.0 + cfg.clip as f64)
                                    * &adv_t;
                                let pg_loss = -surr1.minimum(&surr2).mean(Kind::Float);
                                // Value loss: Huber (default Rust
                                // stabilizer; see Config::vf_clip) or MSE
                                // (Python F.mse_loss). Phase 5 (final)
                                // switches the CLI default to mse once
                                // training is stable under Huber.
                                let v_loss = match cfg.value_loss {
                                    ValueLoss::Huber => value.huber_loss(
                                        &ret_t,
                                        tch::Reduction::Mean,
                                        cfg.vf_clip.max(1e-3) as f64,
                                    ),
                                    ValueLoss::Mse => value.mse_loss(&ret_t, tch::Reduction::Mean),
                                };
                                let ent_loss = ent.mean(Kind::Float);
                                let n_active = choice_t
                                    .quantity_frac
                                    .ge(0.0)
                                    .to_kind(Kind::Float)
                                    .sum(Kind::Float)
                                    .clamp_min(1.0);
                                let entq_loss = ent_q.sum(Kind::Float) / &n_active;
                                let loss = (&pg_loss + cfg.vf_coef as f64 * &v_loss
                                    - ent_coef as f64 * &ent_loss
                                    - cfg.entq_coef as f64 * &entq_loss)
                                    * w_sub;

                                // Grads accumulate across pixel-budget
                                // subs (zero_grad ran once above).
                                loss.backward();

                                (
                                    f64::try_from(&pg_loss).unwrap_or(0.0),
                                    f64::try_from(&v_loss).unwrap_or(0.0),
                                    f64::try_from(&ent_loss).unwrap_or(0.0),
                                    f64::try_from(&entq_loss).unwrap_or(0.0),
                                )
                    }};
                }
                let per_shard_losses: Vec<(f64, f64, f64, f64)> = if exclusive_owner {
                    vec![backward_shard!(
                        &mut learners[0],
                        &mut shard_batches[0],
                        idx_tensors[0].narrow(0, sub_start, sub_len)
                    )]
                } else {
                    std::thread::scope(|s| {
                        let handles: Vec<_> = learners
                            .iter_mut()
                            .zip(shard_batches.iter_mut())
                            .zip(idx_tensors.iter())
                            .map(|((shard, sb), idx_full)| {
                                let idx_t = idx_full.narrow(0, sub_start, sub_len);
                                s.spawn(move || backward_shard!(shard, sb, idx_t))
                            })
                            .collect();
                        handles.into_iter().map(|h| {
                            h.join().unwrap_or_else(|e| {
                                let msg = e
                                    .downcast_ref::<&str>()
                                    .map(|s| s.to_string())
                                    .or_else(|| e.downcast_ref::<String>().cloned())
                                    .unwrap_or_else(|| format!("{e:?}"));
                                panic!("backward thread panicked: {msg}");
                            })
                        }).collect()
                    })
                };
                let n_shards = per_shard_losses.len() as f64;
                // Skip the whole minibatch if any sub-batch went non-finite
                // (see docs/devlog.html's 2026-07-12 NaN-guard entry).
                if any_loss_non_finite(&per_shard_losses) {
                    eprintln!(
                        "[train] WARNING: non-finite loss in epoch={_epoch} mb={m} sub={s_i} \
                         (per-shard pg/v/ent/entq={per_shard_losses:?}) - discarding this \
                         minibatch's gradients without stepping the optimizer (see \
                         docs/devlog.html's 2026-07-12 NaN-guard entry)"
                    );
                    discard_mb = true;
                    break;
                }
                for (pg, v, ent, entq) in &per_shard_losses {
                    mb_loss_sums.0 += pg / n_shards * w_sub;
                    mb_loss_sums.1 += v / n_shards * w_sub;
                    mb_loss_sums.2 += ent / n_shards * w_sub;
                    mb_loss_sums.3 += entq / n_shards * w_sub;
                }
            }

            // Persistent multi-GPU owners cannot inspect another thread's
            // tensors.  They rendezvous here with a CPU flat-gradient
            // message; the hub deterministically reduces in shard order and
            // writes the average back through this callback.  A non-finite
            // shard participates with `finite=false`, causing every owner to
            // discard the same minibatch.
            if let Some(sync) = persistent_sync.as_deref_mut() {
                debug_assert_eq!(learners.len(), 1);
                let global_finite = sync(&mut learners[0], _epoch, m, !discard_mb)?;
                if !global_finite {
                    discard_mb = true;
                }
            }
            if discard_mb {
                for shard in learners.iter_mut() {
                    shard.opt.zero_grad();
                }
                n_mb += 1;
                continue;
            }
            loss_sums.0 += mb_loss_sums.0;
            loss_sums.1 += mb_loss_sums.1;
            loss_sums.2 += mb_loss_sums.2;
            loss_sums.3 += mb_loss_sums.3;
            n_mb += 1;
            let fwdbwd_dt = mb_t0.elapsed().as_secs_f64();

            // DDP-equivalent sync: average grads across shards (no-op for
            // 1 shard) so every replica's optimizer step is identical and
            // weights never drift apart.
            let sync_t0 = Instant::now();
            if persistent_sync.is_none() {
                sync_grads(learners);
            }
            let sync_dt = sync_t0.elapsed().as_secs_f64();
            let step_t0 = Instant::now();
            for shard in learners.iter_mut() {
                // Matches `rl/ppo.py`'s `clip_grad_norm_(..., 0.5)`.
                shard.opt.clip_grad_norm(0.5);
                shard.opt.step();
            }
            let step_dt = step_t0.elapsed().as_secs_f64();
            if std::env::var("OFTRAIN_DIAG").is_ok() {
                println!(
                    "[diag] epoch={_epoch} mb={m} subs={n_subs} fwdbwd_s={fwdbwd_dt:.3} \
                     sync_s={sync_dt:.3} step_s={step_dt:.3}"
                );
            }
        }
        if exclusive_owner {
            eprintln!(
                "[phase] learner PPO epoch {}/{} finished in {:.3}s",
                _epoch + 1,
                cfg.epochs,
                epoch_started.elapsed().as_secs_f64()
            );
        }
    }

    let d = n_mb.max(1) as f64;
    Ok((
        loss_sums.0 / d,
        loss_sums.1 / d,
        loss_sums.2 / d,
        loss_sums.3 / d,
    ))
}

pub fn run(mut cfg: Config) -> Result<()> {
    if cfg.persistent_actors && cfg.auto_scale_envs {
        println!(
            "[train] WARNING: --persistent-actors Phase 1 does not support \
             --auto-scale-envs commands; selecting the legacy collector path \
             so existing autoscale behavior is preserved"
        );
        cfg.persistent_actors = false;
    }
    std::fs::create_dir_all(&cfg.ckpt_dir)?;

    // Resume / init: load weights before shards spawn. `--resume` restores
    // TrainState; `--init` is warm-start only (fresh counters/stage).
    let mut resumed_state: Option<TrainState> = None;
    let mut hub_vs: Option<nn::VarStore> = if let Some(resume_path) = &cfg.resume {
        let mut snapshot = nn::VarStore::new(Device::Cpu);
        let _ = PolicyNet::new(&snapshot.root(), cfg.amp, cfg.foveate, cfg.gc, cfg.blocks);
        snapshot.load(resume_path)?;
        let state_path = state_sidecar_path(resume_path);
        resumed_state = match std::fs::read_to_string(&state_path) {
            Ok(s) => Some(serde_json::from_str(&s)?),
            Err(e) => {
                println!(
                    "[train] WARNING: resuming weights from {resume_path} but no readable state \
                     sidecar at {state_path} ({e}); starting update/stage/entropy-scale counters \
                     from scratch with the resumed weights"
                );
                None
            }
        };
        println!(
            "[train] resumed weights from {resume_path}{}",
            resumed_state
                .as_ref()
                .map(|s| format!(
                    " (state: update={} stage={} ent_scale={:.3} lr_now={:.2e} total_env_steps={})",
                    s.update, s.stage, s.ent_scale, s.lr_now, s.total_env_steps
                ))
                .unwrap_or_default()
        );
        Some(snapshot)
    } else if let Some(init_path) = &cfg.init {
        let mut snapshot = nn::VarStore::new(Device::Cpu);
        let _ = PolicyNet::new(&snapshot.root(), cfg.amp, cfg.foveate, cfg.gc, cfg.blocks);
        snapshot.load(init_path)?;
        println!(
            "[train] warm-started weights from {init_path} (--init; fresh TrainState / optimizer)"
        );
        Some(snapshot)
    } else {
        None
    };
    let start_stage = resumed_state.as_ref().map(|s| s.stage).unwrap_or(cfg.stage);

    let metrics = MetricsWriter::create(&cfg.ckpt_dir)?;
    println!("[train] metrics -> {}/metrics.jsonl", cfg.ckpt_dir);

    let devices = cfg.devices();
    let persistent_learner_enabled = cfg.persistent_actors;
    if persistent_learner_enabled && hub_vs.is_none() {
        let snapshot = nn::VarStore::new(Device::Cpu);
        let _ = PolicyNet::new(&snapshot.root(), cfg.amp, cfg.foveate, cfg.gc, cfg.blocks);
        hub_vs = Some(snapshot);
    }
    // Build the packed host snapshot once. Phase 2 previously wrote the
    // complete VarStore archive once for the actor and then again, after the
    // "all envs ready" line, for the learner. Besides making startup silent
    // for a long time on large policies, archive creation is unnecessary for
    // an in-process transfer.
    let mut persistent_initial_weights = if persistent_learner_enabled {
        Some(snapshot_weights(
            hub_vs.as_ref().expect("CPU hub initialized"),
        )?)
    } else {
        None
    };
    let total_envs = cfg.num_envs * devices.len();
    println!(
        "[train] spawning {} env workers across {} shard(s) {:?} (stage={}, max_ticks={})...",
        total_envs,
        devices.len(),
        devices,
        start_stage,
        cfg.max_episode_ticks
    );

    let mut actors: Vec<ActorShard> = Vec::with_capacity(devices.len());
    let mut persistent_actors: Vec<PersistentActor> = Vec::with_capacity(devices.len());
    let mut learners: Vec<LearnerShard> = Vec::with_capacity(devices.len());
    for (gi, &device) in devices.iter().enumerate() {
        if persistent_learner_enabled {
            let initial_version = resumed_state.as_ref().map(|s| s.update).unwrap_or(0);
            let weights = persistent_initial_weights
                .as_ref()
                .expect("persistent initial weights available")
                .clone();
            persistent_actors.push(PersistentActor::spawn(
                gi,
                device,
                cfg.clone(),
                start_stage,
                initial_version,
                weights,
            )?);
            continue;
        }
        let mut workers = Vec::with_capacity(cfg.num_envs);
        let mut cur_obs = Vec::with_capacity(cfg.num_envs);
        if !cfg.persistent_actors {
            for local_i in 0..cfg.num_envs {
                let idx = gi * cfg.num_envs + local_i;
                let worker_engine = engine_for_idx(idx, cfg.engine, cfg.node_fraction);
                let (w, obs) = spawn_worker(idx, start_stage, cfg.max_episode_ticks, worker_engine)?;
                workers.push(w);
                cur_obs.push(obs);
            }
        }
        let mut learner_vs = nn::VarStore::new(device);
        let learner_policy =
            PolicyNet::new(&learner_vs.root(), cfg.amp, cfg.foveate, cfg.gc, cfg.blocks);
        if let Some(hub) = &hub_vs {
            learner_vs.copy(hub)?;
        } else {
            hub_vs = Some({
                // Keep a CPU-independent handle to shard 0's weights so
                // every later shard starts from bit-identical initial
                // parameters (VarStore::copy handles the cross-device
                // transfer).
                let mut snapshot = nn::VarStore::new(Device::Cpu);
                let _ = PolicyNet::new(&snapshot.root(), cfg.amp, cfg.foveate, cfg.gc, cfg.blocks);
                snapshot.copy(&learner_vs)?;
                snapshot
            });
        }
        let lr_init = resumed_state.as_ref().map(|s| s.lr_now).unwrap_or(cfg.lr);
        let opt = nn::AdamW::default().build(&learner_vs, lr_init)?;

        if gi == 0 {
            println!(
                "[train] AE fine={} coarse={}",
                cfg.ae_ckpt,
                cfg.coarse_ckpt.as_deref().unwrap_or("(2x pool fallback)")
            );
        }

        learners.push(LearnerShard {
            device,
            vs: learner_vs,
            policy: learner_policy,
            opt,
        });
        if cfg.persistent_actors {
            let initial_version = resumed_state.as_ref().map(|s| s.update).unwrap_or(0);
            let weights = snapshot_weights(&learners[gi].vs)?;
            persistent_actors.push(PersistentActor::spawn(
                gi,
                device,
                cfg.clone(),
                start_stage,
                initial_version,
                weights,
            )?);
        } else {
            let mut actor_vs = nn::VarStore::new(device);
            let actor_policy =
                PolicyNet::new(&actor_vs.root(), cfg.amp, cfg.foveate, cfg.gc, cfg.blocks);
            actor_vs.copy(&learners[gi].vs)?;
            let path = std::path::Path::new(&cfg.ae_ckpt);
            if !path.exists() {
                anyhow::bail!(
                    "AE checkpoint not found at {} — run scripts/export_safetensors.py \
                     (or scripts/fetch_ae_encoders.sh) first",
                    path.display()
                );
            }
            let coarse = cfg.coarse_ckpt.as_ref().map(std::path::Path::new);
            // The legacy ActorShard can move between ephemeral collector
            // threads. Keep its AE byte-for-byte on the established f32 path
            // even when policy AMP is enabled.
            let actor_ae = Some(crate::ae::AePair::load(
                path, coarse, device, cfg.amp, false,
            )?);
            actors.push(ActorShard {
                device,
                workers,
                cur_obs,
                compact_host_arena: Arc::new(crate::vecenv::CompactHostArena::default()),
                vs: actor_vs,
                policy: actor_policy,
                ae: actor_ae,
                terrain_cache: crate::ae::TerrainDeviceCache::new(device),
            });
        }
    }
    // Legacy initialization and VarStore copies ran on the coordinator.
    // Persistent owners synchronize their own initialization before Ready.
    if !persistent_learner_enabled {
        for &device in &devices {
            if let Device::Cuda(index) = device {
                Cuda::synchronize(index as i64);
            }
        }
    }
    println!("[train] all {total_envs} envs ready");

    let mut rng = Some(rand::rngs::SmallRng::from_entropy());
    let mut persistent_learner = if persistent_learner_enabled {
        let initial_weights = persistent_initial_weights
            .take()
            .expect("persistent initial weights available");
        let initial_lr = resumed_state.as_ref().map(|s| s.lr_now).unwrap_or(cfg.lr);
        let (learner, params) = PersistentLearner::spawn(
            &devices,
            cfg.clone(),
            initial_weights,
            initial_lr,
            rng.take().expect("learner RNG available"),
        )?;
        Some((learner, params))
    } else {
        None
    };
    let n_params: i64 = persistent_learner
        .as_ref()
        .map(|(_, params)| *params)
        .unwrap_or_else(|| {
            learners[0]
                .vs
                .trainable_variables()
                .iter()
                .map(|t| t.numel() as i64)
                .sum()
        });
    println!(
        "[train] policy params: {n_params} per shard x {} shard(s) on {:?}",
        devices.len(),
        devices
    );

    let gpu_sampler = if devices.iter().any(|d| matches!(d, Device::Cuda(_))) {
        Some(GpuUtilSampler::start(Duration::from_millis(500)))
    } else {
        None
    };

    // Resolve `auto_scale_envs`' bounds once up front (cheap, and only
    // ever logged/used when the flag is actually on): `max_envs=0` means
    // "derive from CPU headroom" (see `autoscale::cpu_env_cap_per_shard`),
    // and `max < min` (e.g. `--max-envs` set below `--num-envs`) is
    // resolved by raising max to min rather than left as a state that
    // could later hang or panic the resize logic below.
    let (autoscale_min_envs, autoscale_max_envs) = if cfg.auto_scale_envs {
        let min_envs = cfg.min_envs.max(1);
        let mut max_envs = if cfg.max_envs == 0 {
            let auto_cap = autoscale::cpu_env_cap_per_shard(devices.len());
            println!(
                "[autoscale] --max-envs=0 (auto): cpu-derived cap = {auto_cap} envs/shard \
                 ({} logical cpu(s) available, {} shard(s))",
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(0),
                devices.len()
            );
            auto_cap
        } else {
            cfg.max_envs
        };
        if max_envs < min_envs {
            println!(
                "[autoscale] WARNING: --max-envs ({max_envs}) < effective --min-envs ({min_envs}); \
                 raising max to min so scaling never hangs/panics"
            );
            max_envs = min_envs;
        }
        println!(
            "[autoscale] enabled: target_gpu_util={:.0}% min_envs={min_envs} max_envs={max_envs} \
             check_every={} step={}",
            cfg.target_gpu_util * 100.0,
            cfg.autoscale_check_every,
            cfg.autoscale_step
        );
        (min_envs, max_envs)
    } else {
        (cfg.min_envs, cfg.max_envs)
    };
    // Monotonic id for newly-spawned envs past the initial batch (RNG
    // seed/thread-name/log uniqueness only - doesn't need to encode shard
    // index the way the startup loop's `idx` does).
    let mut next_env_idx = total_envs;

    // Persistent mode derives one stable RNG per learner owner above.
    // Legacy mode keeps using the same coordinator-owned instance.
    // Persists across every update in this run (see RetStat's doc) - a
    // fresh, empty RetStat on every process restart is an acceptable cold
    // start (its adaptive bound is a no-op until ~2 updates' worth of
    // data has accumulated; RET_ADAPTIVE_N_STD covers everything else).
    let mut ret_stat = RetStat::default();
    let mut ep_rewards: Vec<f64> = Vec::new();
    let mut ep_lengths: Vec<i64> = Vec::new();
    let train_start = Instant::now();
    let mut total_env_steps: u64 = resumed_state
        .as_ref()
        .map(|s| s.total_env_steps)
        .unwrap_or(0);
    let cfg_ref = &cfg;

    // Curriculum advancement (port of `rl/ppo.py`'s win-rate gate, see
    // `ofcore::curriculum::WINDOW`/`Stage::win_at`): a single stage shared
    // across every shard/env (this is one process, unlike Python's
    // multi-rank DDP, so no cross-rank sync is needed - just a plain local
    // rolling window). Only non-rehearsal episodes played *at* the current
    // stage count; advancing resets the window and rebroadcasts the new
    // stage to every env worker thread (see `spawn_worker`'s `stage_rx`)
    // plus decays the learning rate on every shard's optimizer.
    let stages = ofcore::curriculum::stages();
    let mut curr_stage = start_stage;
    let mut recent_wins: std::collections::VecDeque<f64> = resumed_state
        .as_ref()
        .map(|s| s.recent_wins.iter().copied().collect())
        .unwrap_or_else(|| std::collections::VecDeque::with_capacity(ofcore::curriculum::WINDOW));
    let mut lr_now = resumed_state.as_ref().map(|s| s.lr_now).unwrap_or(cfg.lr);
    // Adaptive entropy-floor multiplier (port of `rl/ppo.py`'s
    // `ent_scale`): multiplicative on top of the linear anneal, nudged
    // after each update from that update's measured mean entropy.
    let mut ent_scale: f64 = resumed_state.as_ref().map(|s| s.ent_scale).unwrap_or(1.0);
    let start_update = resumed_state.as_ref().map(|s| s.update).unwrap_or(0);
    // See LR_WARMUP_UPDATES's doc - resets on every curriculum advance
    // too, not just the run's very start, since the value function faces
    // the same "brand new data distribution" shock either way.
    // After `--resume`, AdamW moments are cold (tch cannot restore them);
    // `--resume-warmup-updates` (default 100) sets the first warmup length.
    // 0 → fall back to the ordinary stage warmup (`LR_WARMUP_UPDATES`).
    let mut lr_warmup_updates = if cfg.resume.is_some() && cfg.resume_warmup_updates > 0 {
        cfg.resume_warmup_updates
    } else {
        LR_WARMUP_UPDATES
    };
    let mut lr_warmup_start_update = start_update;

    // Prime the pipeline: collect the very first rollout (using the
    // actors' initial, freshly-copied-from-learner weights) before the
    // update loop starts overlapping collection with training.
    let prime_started = Instant::now();
    eprintln!(
        "[phase] prime rollout started ({} envs x {} steps)",
        total_envs, cfg.rollout_len
    );
    let mut pending: Vec<RolloutResult> = if cfg.persistent_actors {
        for actor in &mut persistent_actors {
            actor.send_collect()?;
        }
        persistent_actors
            .iter_mut()
            .map(PersistentActor::finish_collect)
            .collect::<Result<Vec<_>>>()?
    } else {
        std::thread::scope(|s| {
            let handles: Vec<_> = actors
                .iter_mut()
                .map(|actor| s.spawn(move || collect_rollout(actor, cfg_ref, start_update)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().map_err(|_| anyhow!("collector thread panicked"))?)
                .collect::<Result<Vec<_>>>()
        })?
    };
    validate_rollout_set(&pending, start_update)?;
    eprintln!(
        "[phase] prime rollout finished in {:.3}s",
        prime_started.elapsed().as_secs_f64()
    );

    for update in start_update..cfg.updates {
        let update_start = Instant::now();
        let expected_pending_version =
            if update == start_update { start_update } else { update - 1 };
        validate_rollout_set(&pending, expected_pending_version)?;

        // Overlap: collect update `update+1`'s rollout on every shard's
        // (frozen-for-this-round) actor concurrently with training the
        // learner on `pending` (this update's, already-collected data).
        // See module doc for why this is safe (disjoint state) and what
        // it's fixing (GPU idling during collection).
        // Linear anneal ent_coef -> ent_coef_final so late training commits
        // instead of exploring forever, times the adaptive entropy-floor
        // scale (both match `rl/ppo.py`).
        let frac = (update as f64 / cfg.ent_anneal_updates.max(1) as f64).min(1.0);
        let ent_coef_now = ((cfg.ent_coef as f64
            + (cfg.ent_coef_final as f64 - cfg.ent_coef as f64) * frac)
            * ent_scale) as f32;
        // LR warmup (see LR_WARMUP_UPDATES's doc) - a no-op once warmup
        // completes (frac saturates at 1.0), so this is safe to apply on
        // every single update rather than only during the warmup window.
        let warmup_frac = lr_warmup_frac(update, lr_warmup_start_update, lr_warmup_updates);
        if !persistent_learner_enabled {
            for shard in learners.iter_mut() {
                shard.opt.set_lr(lr_now * warmup_frac);
            }
        }
        // `collect_dt`/`train_dt` used to both be measured from
        // (effectively) the same start instant to the same point *after*
        // this whole scope returns - since the scope can't return until
        // both the (synchronous, main-thread) `train_update` call *and*
        // every collector thread's `.join()` have completed, they always
        // read out nearly identical wall-clock values regardless of which
        // phase actually dominates, making it impossible to tell collect-
        // vs-train-bound apart from the log (this is why 8-GPU runs showed
        // `collect_s == train_s == update_s` on every line - not a real
        // tie, just an instrumentation bug). Fix: time `train_update`
        // itself (the main thread's own synchronous work) directly, and
        // separately time from just before spawning the collectors to
        // just after every one of them has joined - if collection is the
        // long pole, `collect_dt` will now correctly read higher than
        // `train_dt` (the `.join()` calls after `train_update` returns
        // will block, adding to `collect_dt` but not `train_dt`).
        let collect_start = Instant::now();
        let (train_result, next_pending, train_dt, learner_weights, learner_snapshot_dt) =
            if persistent_learner_enabled {
                for actor in &mut persistent_actors {
                    actor.send_collect()?;
                }
                let train_t0 = Instant::now();
                let train_reply = persistent_learner
                    .as_mut()
                    .expect("persistent learner initialized")
                    .0
                    .train(std::mem::take(&mut pending), lr_now * warmup_frac, ent_coef_now);
                let next_pending = persistent_actors
                    .iter_mut()
                    .map(PersistentActor::finish_collect)
                    .collect::<Result<Vec<_>>>();
                match train_reply {
                    Ok(reply) => (
                        Ok(reply.losses),
                        next_pending,
                        reply.train_seconds,
                        Some(reply.weights),
                        reply.snapshot_seconds,
                    ),
                    Err(error) => (
                        Err(error),
                        next_pending,
                        train_t0.elapsed().as_secs_f64(),
                        None,
                        0.0,
                    ),
                }
            } else if cfg.persistent_actors {
            for actor in &mut persistent_actors {
                actor.send_collect()?;
            }
            let train_t0 = Instant::now();
            let train_result = train_update(
                &mut learners,
                &pending,
                cfg_ref,
                rng.as_mut().expect("legacy learner RNG available"),
                ent_coef_now,
                &mut ret_stat,
                false,
                None,
                None,
            );
            let train_dt = train_t0.elapsed().as_secs_f64();
            let next_pending = persistent_actors
                .iter_mut()
                .map(PersistentActor::finish_collect)
                .collect::<Result<Vec<_>>>();
            (train_result, next_pending, train_dt, None, 0.0)
        } else {
            std::thread::scope(|s| {
                let collect_handles: Vec<_> = actors
                    .iter_mut()
                    .map(|actor| s.spawn(move || collect_rollout(actor, cfg_ref, update)))
                    .collect();
                let train_t0 = Instant::now();
                let train_result = train_update(
                    &mut learners,
                    &pending,
                    cfg_ref,
                    rng.as_mut().expect("legacy learner RNG available"),
                    ent_coef_now,
                    &mut ret_stat,
                    false,
                    None,
                    None,
                );
                let train_dt = train_t0.elapsed().as_secs_f64();
                let next_pending: Result<Vec<RolloutResult>> = collect_handles
                    .into_iter()
                    .map(|h| h.join().map_err(|_| anyhow!("collector thread panicked"))?)
                    .collect();
                (train_result, next_pending, train_dt, None, 0.0)
            })
        };
        let collect_dt = collect_start.elapsed().as_secs_f64();
        let last_losses = train_result?;
        let next_pending = next_pending?;
        validate_rollout_set(&next_pending, update)?;
        // Actual env count behind `next_pending` (each shard's rollout
        // just collected) - not the startup `total_envs`, which goes
        // stale the moment `auto_scale_envs` grows any shard. Derived
        // straight from the collected data rather than
        // `actors[..].workers.len()` so it's correct regardless of
        // exactly when in this iteration a resize lands.
        let live_total_envs: usize = next_pending
            .iter()
            .map(|r| r.buffer.first().map(|row| row.len()).unwrap_or(0))
            .sum();

        // Entropy floor controller (port of `rl/ppo.py`): nudge the coef
        // scale toward keeping measured mean entropy above the floor, with
        // hysteresis so it doesn't oscillate. Multiplicative so it composes
        // with the anneal. Held for ENT_GRACE_UPDATES at startup
        // (spawn-heavy startup rollouts read artificially low entropy).
        // Discrete heads only: the Beta quantity head's differential
        // entropy lives on another scale.
        if cfg.ent_floor > 0.0 && update > ENT_GRACE_UPDATES {
            let ent_mean = last_losses.2;
            let floor = cfg.ent_floor as f64;
            if ent_mean < floor {
                ent_scale = (ent_scale * 1.3).min(ENT_SCALE_MAX);
            } else if ent_mean > floor * 1.4 {
                ent_scale = (ent_scale / 1.3).max(1.0);
            } else {
                // Anywhere above the floor decays (slowly): without this,
                // a scale pushed up by a transient dip is trapped forever
                // when the bonus holds entropy inside [floor, floor*1.4).
                ent_scale = (ent_scale / 1.05).max(1.0);
            }
        }

        let mut advanced = false;
        let debug_eps = std::env::var("OFTRAIN_DEBUG_EPISODES").is_ok();
        for result in &next_pending {
            for info in &result.ep_infos {
                if debug_eps {
                    eprintln!(
                        "[ep] reward={:.3} len={} tiles={:.1} tick={} place={}/{} score={:.3} won={} wasted={} stage={} rehearsal={} map={}",
                        info.reward,
                        info.length,
                        info.final_tiles,
                        info.final_tick,
                        info.place,
                        info.n_players,
                        info.score,
                        info.won,
                        info.wasted,
                        info.stage,
                        info.rehearsal,
                        info.map
                    );
                }
                ep_rewards.push(info.reward);
                ep_lengths.push(info.length);
                if info.stage == curr_stage && !info.rehearsal {
                    if recent_wins.len() == ofcore::curriculum::WINDOW {
                        recent_wins.pop_front();
                    }
                    recent_wins.push_back(if info.won { 1.0 } else { 0.0 });
                    let win_rate = recent_wins.iter().sum::<f64>() / recent_wins.len() as f64;
                    if debug_eps {
                        eprintln!(
                            "[win_rate] {:.3} (window={}/{})",
                            win_rate,
                            recent_wins.len(),
                            ofcore::curriculum::WINDOW
                        );
                    }
                    if recent_wins.len() == ofcore::curriculum::WINDOW
                        && win_rate > stages[curr_stage].win_at
                        && curr_stage < stages.len() - 1
                    {
                        curr_stage += 1;
                        recent_wins.clear();
                        advanced = true;
                    }
                }
            }
        }
        if advanced {
            lr_now = cfg.lr * cfg.stage_lr_decay.powi(curr_stage as i32);
            lr_warmup_start_update = update + 1;
            lr_warmup_updates = LR_WARMUP_UPDATES;
            if cfg.persistent_actors {
                for actor in &mut persistent_actors {
                    actor.set_stage(curr_stage)?;
                }
            } else {
                for actor in &actors {
                    for w in &actor.workers {
                        let _ = w.stage_tx.send(curr_stage);
                    }
                }
            }
            // The *next* iteration's top-of-loop warmup logic recomputes
            // and re-applies the correct (freshly-reset) warmup-scaled LR
            // before it's ever used for an actual optimizer step, so what
            // gets set here doesn't matter for training itself - keeping
            // it at the un-warmed-up `lr_now` just means anything that
            // reads the optimizer's LR *before* the next iteration starts
            // (there's currently nothing that does) sees the stage's
            // target rate, not a stale pre-advance value.
            if !persistent_learner_enabled {
                for shard in learners.iter_mut() {
                    shard.opt.set_lr(lr_now);
                }
            }
            let st = &stages[curr_stage];
            println!(
                "=== curriculum advance -> stage {curr_stage}: maps={:?} bots={} {} lr->{lr_now:.2e}",
                st.maps, st.bots, st.difficulty
            );
        }
        total_env_steps += (live_total_envs * cfg.rollout_len) as u64;

        // Refresh every actor from its paired learner's just-updated
        // weights, now that training has finished (and the collection
        // that ran concurrently with it is done reading the *old*
        // weights) - the next update's collection will use these.
        let refresh_start = Instant::now();
        if cfg.persistent_actors {
            let snapshots = if persistent_learner_enabled {
                learner_weights.ok_or_else(|| {
                    anyhow!("persistent learner completed without a weight snapshot")
                })?
            } else {
                learners
                    .iter()
                    .map(|learner| snapshot_weights(&learner.vs))
                    .collect::<Result<Vec<_>>>()?
            };
            for (actor, weights) in persistent_actors.iter_mut().zip(snapshots) {
                actor.refresh(update + 1, weights)?;
            }
        } else {
            for (actor, learner) in actors.iter_mut().zip(learners.iter()) {
                actor.vs.copy(&learner.vs)?;
            }
            // VarStore::copy schedules CUDA device-to-device copies and may return
            // before they complete. The next loop iteration runs actor inference
            // while the learner updates on another thread/stream, so crossing this
            // ownership boundary without a wait can expose partially copied actor
            // weights (or learner storage being mutated while still read). This
            // was the root cause of delayed, non-deterministic device asserts.
            for actor in &actors {
                if let Device::Cuda(index) = actor.device {
                    Cuda::synchronize(index as i64);
                }
            }
        }
        let refresh_dt = refresh_start.elapsed().as_secs_f64();
        let actor_work_dt = next_pending
            .iter()
            .map(|result| result.collect_seconds)
            .fold(0.0f64, f64::max);
        pending = next_pending;
        eprintln!(
            "[phase] update {update} train+collect+refresh finished in {:.3}s \
             (train={train_dt:.3}s collect={collect_dt:.3}s refresh={refresh_dt:.3}s)",
            update_start.elapsed().as_secs_f64()
        );

        // Auto-scale check: deliberately placed here, after this
        // update's `pending`/`next_pending` swap and *before* the next
        // loop iteration spawns its `collect_rollout` calls - growing
        // `actor.workers`/`actor.cur_obs` mid-rollout (inside
        // `collect_rollout`'s per-step send/recv loop) would desync the
        // `n = actor.workers.len()` it captured at the top of that call.
        if cfg.auto_scale_envs && update % cfg.autoscale_check_every.max(1) == 0 {
            let gpu_util_frac = gpu_sampler
                .as_ref()
                .map(|g| g.snapshot().min_mean_util() / 100.0);
            let current = actors[0].workers.len();
            let target_n = autoscale::next_env_count(
                current,
                gpu_util_frac,
                cfg.target_gpu_util,
                autoscale_min_envs,
                autoscale_max_envs,
                cfg.autoscale_step,
            );
            if target_n > current {
                let add = target_n - current;
                // Grow every shard by the same amount in lockstep so all
                // shards keep an identical env count (see `train_update`'s
                // derivation of a single shared `n` from shard 0's data,
                // and this module's doc for why uniform growth is the
                // simplifying choice) - spawn everything first, and only
                // commit (push onto the real shards) if every single spawn
                // across every shard succeeded; otherwise close whatever
                // was spawned in this attempt and keep the old count. A
                // partial commit would leave shards with different env
                // counts, which the rest of this file assumes never
                // happens.
                let mut spawned: Vec<(usize, Worker, PreparedObs)> =
                    Vec::with_capacity(add * actors.len());
                let mut spawn_err: Option<anyhow::Error> = None;
                'grow: for gi in 0..actors.len() {
                    for _ in 0..add {
                        let worker_engine =
                            engine_for_idx(next_env_idx, cfg.engine, cfg.node_fraction);
                        match spawn_worker(
                            next_env_idx,
                            curr_stage,
                            cfg.max_episode_ticks,
                            worker_engine,
                        ) {
                            Ok((w, obs)) => {
                                next_env_idx += 1;
                                spawned.push((gi, w, obs));
                            }
                            Err(e) => {
                                spawn_err = Some(e);
                                break 'grow;
                            }
                        }
                    }
                }
                let gpu_str = gpu_util_frac
                    .map(|f| format!("{:.1}%", f * 100.0))
                    .unwrap_or_else(|| "n/a (no GPU)".to_string());
                match spawn_err {
                    Some(e) => {
                        println!(
                            "[autoscale] scale-up {current} -> {target_n} envs/shard FAILED ({e:#}); \
                             closing partially-spawned workers, staying at {current}"
                        );
                        for (_, w, _) in spawned {
                            drop(w.choice_tx);
                            let _ = w.handle.join();
                        }
                    }
                    None => {
                        for (gi, w, obs) in spawned {
                            actors[gi].workers.push(w);
                            actors[gi].cur_obs.push(obs);
                        }
                        println!(
                            "[autoscale] all shards: {current} -> {target_n} envs (gpu_util={gpu_str} \
                             target={:.0}% cpu_cap={autoscale_max_envs})",
                            cfg.target_gpu_util * 100.0
                        );
                    }
                }
            }
        }

        let mut last_eval: Option<EvalResult> = None;
        if cfg.eval_every > 0 && update % cfg.eval_every == 0 {
            let t_eval = Instant::now();
            eprintln!(
                "[phase] update {update} evaluation started (stage={curr_stage}, episodes={})",
                cfg.eval_episodes
            );
            let ev = if cfg.persistent_actors {
                persistent_actors[0].eval(curr_stage, cfg.eval_episodes)?
            } else {
                let ae = actors[0]
                    .ae
                    .as_ref()
                    .ok_or_else(|| anyhow!("eval requires AE encoders"))?;
                run_eval(
                    &learners[0].policy,
                    ae,
                    learners[0].device,
                    curr_stage,
                    cfg.eval_episodes,
                    cfg.max_episode_ticks,
                    cfg.engine,
                    cfg.pinned_h2d,
                    cfg.fp16_rollout,
                    cfg.compact_rollout && cfg.foveate,
                )?
            };
            println!(
                "[eval] stage {curr_stage}  win {:.2}  score {:.2}  ({} eps, {:.0}s)",
                ev.win,
                ev.score,
                ev.episodes,
                t_eval.elapsed().as_secs_f64()
            );
            // Always persist eval rows even when this update isn't a
            // log_every tick (so sparse eval schedules still land in JSONL).
            if update % cfg.log_every != 0 && update != cfg.updates - 1 {
                let win_rate = if recent_wins.is_empty() {
                    None
                } else {
                    Some(recent_wins.iter().sum::<f64>() / recent_wins.len() as f64)
                };
                let _ = metrics.log_update(
                    update,
                    curr_stage,
                    last_losses.0,
                    last_losses.1,
                    last_losses.2,
                    last_losses.3,
                    win_rate,
                    lr_now,
                    total_env_steps,
                    Some(ev.win),
                    Some(ev.score),
                );
            }
            last_eval = Some(ev);
        }

        if update % cfg.log_every == 0 || update == cfg.updates - 1 {
            let dt = update_start.elapsed().as_secs_f64();
            let total_dt = train_start.elapsed().as_secs_f64();
            let sps = (live_total_envs * cfg.rollout_len) as f64 / dt.max(1e-6);
            let recent_n = ep_rewards.len().min(50);
            let recent_reward = if recent_n > 0 {
                ep_rewards[ep_rewards.len() - recent_n..]
                    .iter()
                    .sum::<f64>()
                    / recent_n as f64
            } else {
                0.0
            };
            let win_rate = if recent_wins.is_empty() {
                None
            } else {
                Some(recent_wins.iter().sum::<f64>() / recent_wins.len() as f64)
            };
            let gpu_str = match &gpu_sampler {
                Some(s) => {
                    let gpu = s.snapshot();
                    let per_gpu_str = gpu
                        .util_per_gpu
                        .iter()
                        .enumerate()
                        .map(|(i, u)| format!("gpu{i}={u:.0}%"))
                        .collect::<Vec<_>>()
                        .join(" ");
                    format!(
                        " gpu_mem%={:.0} min_mean_util%={:.0} [{per_gpu_str}]",
                        gpu.mem_pct,
                        gpu.min_mean_util()
                    )
                }
                None => String::new(),
            };
            println!(
                "[update {:>5}] steps/s={:>7.1} decisions_total={:>9} eps_done={:>5} recent_reward={:>8.3} \
                 pg={:>+.4} v={:>.4} ent={:>.3} entq={:>+.3} ecoef={:.4} stage={} lr={:.2e} elapsed={:.0}s \
                 update_s={:.1} collect_s={:.1} train_s={:.1} actor_work_s={:.1} \
                 learner_snapshot_s={:.3} refresh_s={:.3}{gpu_str}",
                update,
                sps,
                total_env_steps,
                ep_rewards.len(),
                recent_reward,
                last_losses.0,
                last_losses.1,
                last_losses.2,
                last_losses.3,
                ent_coef_now,
                curr_stage,
                lr_now,
                total_dt,
                dt,
                collect_dt,
                train_dt,
                actor_work_dt,
                learner_snapshot_dt,
                refresh_dt,
            );
            let (eval_win, eval_score) = match &last_eval {
                Some(ev) => (Some(ev.win), Some(ev.score)),
                None => (None, None),
            };
            if let Err(e) = metrics.log_update(
                update,
                curr_stage,
                last_losses.0,
                last_losses.1,
                last_losses.2,
                last_losses.3,
                win_rate,
                lr_now,
                total_env_steps,
                eval_win,
                eval_score,
            ) {
                eprintln!("[train] WARNING: metrics log failed: {e:#}");
            }
        }

        if cfg.ckpt_every > 0 && (update % cfg.ckpt_every == 0) && update > 0 {
            let state = TrainState {
                update: update + 1, // resume must start at the *next* update, not repeat this one
                stage: curr_stage,
                ent_scale,
                lr_now,
                total_env_steps,
                recent_wins: recent_wins.iter().copied().collect(),
            };
            let path = format!("{}/policy_update{}.safetensors", cfg.ckpt_dir, update);
            if persistent_learner_enabled {
                persistent_learner
                    .as_mut()
                    .expect("persistent learner initialized")
                    .0
                    .save_weights(&path)?;
                save_checkpoint_state(&path, &state)?;
            } else {
                save_checkpoint(&learners[0].vs, &path, &state)?;
            }
            // Fixed-name pointer at the latest checkpoint so a restart-loop
            // wrapper (or a fresh pod after total disk loss) always has one
            // unambiguous thing to resume from, without parsing filenames -
            // matches `rl/ppo.py`'s single always-current `policy.pt`.
            // Extension is `.safetensors` so tch `VarStore::save` writes the
            // safetensors interchange format (legacy `.ot` still loads).
            let latest = format!("{}/latest.safetensors", cfg.ckpt_dir);
            if persistent_learner_enabled {
                persistent_learner
                    .as_mut()
                    .expect("persistent learner initialized")
                    .0
                    .save_weights(&latest)?;
                save_checkpoint_state(&latest, &state)?;
            } else {
                save_checkpoint(&learners[0].vs, &latest, &state)?;
            }
            println!("[train] checkpoint saved: {path} (update={})", state.update);
        }
    }

    let final_state = TrainState {
        update: cfg.updates,
        stage: curr_stage,
        ent_scale,
        lr_now,
        total_env_steps,
        recent_wins: recent_wins.iter().copied().collect(),
    };
    let final_path = format!("{}/policy_final.safetensors", cfg.ckpt_dir);
    let latest_path = format!("{}/latest.safetensors", cfg.ckpt_dir);
    if persistent_learner_enabled {
        let learner = &mut persistent_learner
            .as_mut()
            .expect("persistent learner initialized")
            .0;
        learner.save_weights(&final_path)?;
        save_checkpoint_state(&final_path, &final_state)?;
        learner.save_weights(&latest_path)?;
        save_checkpoint_state(&latest_path, &final_state)?;
    } else {
        save_checkpoint(&learners[0].vs, &final_path, &final_state)?;
        save_checkpoint(&learners[0].vs, &latest_path, &final_state)?;
    }
    if let Some((learner, _)) = persistent_learner {
        learner.shutdown()?;
    }
    if cfg.persistent_actors {
        for actor in persistent_actors {
            actor.shutdown()?;
        }
    } else {
        for actor in actors {
            for w in actor.workers {
                drop(w.choice_tx);
                let _ = w.handle.join();
            }
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn unused_lock_hint(_m: &Arc<Mutex<()>>) {}

#[cfg(test)]
mod persistent_actor_tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn protocol_accepts_complete_lifecycle_in_order() {
        let mut protocol = ActorProtocol::new(7);
        protocol.accept(&ActorCommand::Collect { id: 1 }).unwrap();
        protocol
            .accept(&ActorCommand::Refresh {
                id: 2,
                policy_version: 8,
                weights: CpuWeightSnapshot {
                    meta: Vec::new(),
                    values: vec![1.0, 2.0, 3.0],
                },
            })
            .unwrap();
        protocol
            .accept(&ActorCommand::SetStage { id: 3, stage: 2 })
            .unwrap();
        protocol
            .accept(&ActorCommand::Eval {
                id: 4,
                stage: 2,
                episodes: 8,
            })
            .unwrap();
        protocol.accept(&ActorCommand::Shutdown { id: 5 }).unwrap();
        assert_eq!(protocol.policy_version, 8);
        assert!(protocol.stopped);
        assert!(
            protocol.accept(&ActorCommand::Collect { id: 6 }).is_err(),
            "post-shutdown work must be rejected"
        );
    }

    #[test]
    fn protocol_rejects_reordered_and_stale_refreshes() {
        let mut reordered = ActorProtocol::new(3);
        let err = reordered
            .accept(&ActorCommand::Collect { id: 2 })
            .unwrap_err();
        assert!(err.to_string().contains("expected 1"));

        let mut stale = ActorProtocol::new(3);
        let err = stale
            .accept(&ActorCommand::Refresh {
                id: 1,
                policy_version: 3,
                weights: CpuWeightSnapshot {
                    meta: Vec::new(),
                    values: Vec::new(),
                },
            })
            .unwrap_err();
        assert!(err.to_string().contains("stale actor policy refresh"));
    }

    #[test]
    fn actor_failure_reply_preserves_command_context() {
        let reply = ActorReply::Failed {
            id: 11,
            command: "collect",
            error: "env 4 obs channel closed".to_string(),
        };
        let error = actor_reply_error(2, "collect result", &reply).to_string();
        assert!(error.contains("actor 2"));
        assert!(error.contains("command 11 (collect)"));
        assert!(error.contains("env 4 obs channel closed"));
    }

    #[test]
    fn packed_weight_message_is_independent_cpu_data() {
        tch::manual_seed(91);
        let source = nn::VarStore::new(Device::Cpu);
        let _ = PolicyNet::new(&source.root(), false, false, 8, 1);
        let snapshot = snapshot_weights(&source).unwrap();
        assert!(!snapshot.values.is_empty());

        let destination = nn::VarStore::new(Device::Cpu);
        let _ = PolicyNet::new(&destination.root(), false, false, 8, 1);
        apply_weight_snapshot(&destination, &snapshot).unwrap();
        let source_vars = source.variables();
        let destination_vars = destination.variables();
        assert_eq!(source_vars.len(), destination_vars.len());
        for (name, source_tensor) in source_vars {
            let destination_tensor = &destination_vars[&name];
            assert_eq!(
                (&source_tensor - destination_tensor)
                    .abs()
                    .max()
                    .double_value(&[]),
                0.0,
                "weight {name} changed across CPU serialization"
            );
        }
    }

    fn empty_rollout() -> RolloutResult {
        RolloutResult {
            buffer: Vec::new(),
            bootstrap_v: Vec::new(),
            ep_infos: Vec::new(),
            policy_version: 0,
            collect_seconds: 0.0,
        }
    }

    #[test]
    fn learner_protocol_accepts_train_save_shutdown_lifecycle() {
        let mut protocol = LearnerProtocol::new();
        protocol
            .accept(&LearnerCommand::Train {
                id: 1,
                rollout: empty_rollout(),
                lr: 1e-4,
                ent_coef: 0.01,
                ret_stat: RetStat::default(),
            })
            .unwrap();
        protocol
            .accept(&LearnerCommand::SaveWeights {
                id: 2,
                path: "checkpoint.safetensors".to_string(),
            })
            .unwrap();
        protocol.accept(&LearnerCommand::Shutdown { id: 3 }).unwrap();
        assert!(protocol.stopped);
        assert!(
            protocol
                .accept(&LearnerCommand::Shutdown { id: 4 })
                .is_err()
        );
    }

    #[test]
    fn learner_protocol_rejects_reordering_and_propagates_errors() {
        let mut protocol = LearnerProtocol::new();
        let error = protocol
            .accept(&LearnerCommand::SaveWeights {
                id: 2,
                path: "wrong-order.safetensors".to_string(),
            })
            .unwrap_err();
        assert!(error.to_string().contains("expected 1"));

        let reply = LearnerReply::Failed {
            id: 7,
            shard: 2,
            command: "train",
            error: "batch upload failed".to_string(),
        };
        let error = learner_reply_error("train result", &reply).to_string();
        assert!(error.contains("command 7 (train)"));
        assert!(error.contains("batch upload failed"));
    }

    fn parity_config() -> Config {
        Config {
            num_envs: 1,
            num_gpus: 1,
            stage: 0,
            max_episode_ticks: 10,
            rollout_len: 2,
            updates: 1,
            lr: 1e-4,
            gamma: 0.99,
            lambda: 0.95,
            clip: 0.2,
            vf_coef: 0.5,
            ret_clip: 3000.0,
            adv_clip: 10.0,
            vf_clip: 50.0,
            ent_coef: 0.01,
            ent_coef_final: 0.002,
            ent_anneal_updates: 100,
            ent_floor: 0.0,
            entq_coef: 0.002,
            stage_lr_decay: 0.85,
            epochs: 2,
            minibatches: 1,
            amp: false,
            foveate: false,
            ae_ckpt: String::new(),
            coarse_ckpt: None,
            gc: 8,
            blocks: 1,
            pinned_h2d: false,
            fp16_rollout: false,
            compact_rollout: false,
            pipeline_groups: false,
            persistent_actors: true,
            device: Device::Cpu,
            engine: EngineKind::Native,
            node_fraction: 0.0,
            log_every: 1,
            eval_every: 0,
            eval_episodes: 0,
            ckpt_every: 0,
            ckpt_dir: String::new(),
            init: None,
            resume: None,
            resume_warmup_updates: 0,
            value_loss: ValueLoss::Mse,
            auto_scale_envs: false,
            target_gpu_util: 0.95,
            min_envs: 1,
            max_envs: 1,
            autoscale_check_every: 5,
            autoscale_step: 1,
        }
    }

    fn parity_obs() -> PreparedObs {
        let (gh, gw) = (4usize, 4usize);
        let plane = gh * gw;
        PreparedObs {
            compact: None,
            grid: Some(vec![0.1; policy::C_GRID as usize * plane]),
            grid_coarse: None,
            cgh: 0,
            cgw: 0,
            ae_raw: crate::ae::AeRaw {
                owners: vec![0; gh * 8 * gw * 8],
                static_terrain: crate::ae::StaticTerrain {
                    key: crate::ae::TerrainCacheKey {
                        env_id: 0,
                        episode: 0,
                        static_id: 0,
                        hr: gh * 8,
                        wr: gw * 8,
                    },
                    map: Arc::from("parity"),
                    land_mag: vec![0.0; 2 * gh * 8 * gw * 8].into(),
                },
                fallout: crate::ae::pack_fallout(
                    &vec![0u8; gh * 8 * gw * 8],
                    gh * 8,
                    gw * 8,
                ),
                stat: vec![0.0; 6 * plane],
                hr: gh * 8,
                wr: gw * 8,
            },
            ego: vec![0.0; 3 * plane],
            db: vec![0.0; plane],
            transient: vec![0.0; ofcore::feat::N_TRANSIENT * plane],
            legal_tile: vec![1.0; plane],
            gh,
            gw,
            players: vec![0.1; ofcore::feat::MAX_SLOTS * ofcore::feat::P_FEAT],
            pmask: [1.0; ofcore::feat::MAX_SLOTS],
            scalars: [0.1; ofcore::feat::N_SCALARS],
            me_slot: 0,
            legal_actions: [1.0; ofcore::feat::N_ACTIONS],
            legal_ptarget: vec![
                1.0;
                ofcore::feat::N_ACTIONS * ofcore::feat::MAX_SLOTS
            ],
            legal_build: [1.0; ofcore::feat::N_BUILD],
            legal_nuke: [1.0; ofcore::feat::N_NUKE],
            local: vec![0.1; 5 * policy::LOCAL as usize * policy::LOCAL as usize],
        }
    }

    fn parity_rollout() -> RolloutResult {
        let wait = ACTIONS.iter().position(|&action| action == "noop").unwrap() as i64;
        let step = |reward| Step {
            obs: parity_obs(),
            choice: ChoiceScalars {
                action: wait,
                player_slot: -1,
                tile_region: -1,
                build_type: -1,
                nuke_type: -1,
                quantity_frac: -1.0,
            },
            logp: -1.0,
            value: 0.25,
            reward,
            done: false,
        };
        RolloutResult {
            buffer: vec![vec![step(0.5)], vec![step(-0.25)]],
            bootstrap_v: vec![0.1],
            ep_infos: Vec::new(),
            policy_version: 0,
            collect_seconds: 0.0,
        }
    }

    fn parity_learner(cfg: &Config, weights: &CpuWeightSnapshot) -> LearnerShard {
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = PolicyNet::new(&vs.root(), false, false, cfg.gc, cfg.blocks);
        apply_weight_snapshot(&vs, weights).unwrap();
        let opt = nn::AdamW::default().build(&vs, cfg.lr).unwrap();
        LearnerShard {
            device: Device::Cpu,
            vs,
            policy,
            opt,
        }
    }

    #[test]
    fn persistent_owner_train_path_matches_legacy_single_shard_math() {
        tch::manual_seed(404);
        let cfg = parity_config();
        let source = nn::VarStore::new(Device::Cpu);
        let _ = PolicyNet::new(&source.root(), false, false, cfg.gc, cfg.blocks);
        let weights = snapshot_weights(&source).unwrap();
        let mut legacy = parity_learner(&cfg, &weights);
        let mut owned = parity_learner(&cfg, &weights);
        let mut legacy_rng = rand::rngs::SmallRng::seed_from_u64(919);
        let mut owned_rng = rand::rngs::SmallRng::seed_from_u64(919);
        let mut legacy_ret = RetStat::default();
        let mut owned_ret = RetStat::default();
        let legacy_rollout = parity_rollout();
        let owned_rollout = parity_rollout();

        let legacy_losses = train_update(
            std::slice::from_mut(&mut legacy),
            std::slice::from_ref(&legacy_rollout),
            &cfg,
            &mut legacy_rng,
            0.01,
            &mut legacy_ret,
            false,
            None,
            None,
        )
        .unwrap();
        let owned_losses = train_update(
            std::slice::from_mut(&mut owned),
            std::slice::from_ref(&owned_rollout),
            &cfg,
            &mut owned_rng,
            0.01,
            &mut owned_ret,
            true,
            None,
            None,
        )
        .unwrap();

        for (legacy_loss, owned_loss) in [
            (legacy_losses.0, owned_losses.0),
            (legacy_losses.1, owned_losses.1),
            (legacy_losses.2, owned_losses.2),
            (legacy_losses.3, owned_losses.3),
        ] {
            assert!((legacy_loss - owned_loss).abs() < 1e-6);
        }
        assert_eq!(legacy_ret.count, owned_ret.count);
        assert_eq!(legacy_ret.sum, owned_ret.sum);
        assert_eq!(legacy_ret.sum_sq, owned_ret.sum_sq);
        let owned_vars = owned.vs.variables();
        for (name, legacy_tensor) in legacy.vs.variables() {
            let max_diff = (&legacy_tensor - &owned_vars[&name])
                .abs()
                .max()
                .double_value(&[]);
            assert!(max_diff < 1e-6, "{name} differs by {max_diff}");
        }
    }

    #[test]
    fn cpu_gradient_hub_reduces_in_shard_order_and_rejects_shape_mismatch() {
        let averaged = average_cpu_gradients(&[
            vec![1.0, 8.0, -3.0],
            vec![3.0, 4.0, 1.0],
            vec![2.0, 0.0, 2.0],
            vec![6.0, 4.0, 0.0],
        ])
        .unwrap();
        assert_eq!(averaged, vec![3.0, 4.0, 0.0]);
        assert!(average_cpu_gradients(&[vec![1.0], vec![1.0, 2.0]]).is_err());
    }

    #[test]
    fn persistent_one_and_two_shard_updates_have_ddp_parity() {
        tch::manual_seed(606);
        let cfg = parity_config();
        let source = nn::VarStore::new(Device::Cpu);
        let _ = PolicyNet::new(&source.root(), false, false, cfg.gc, cfg.blocks);
        let weights = snapshot_weights(&source).unwrap();
        for repetition in 0..3 {
            let (mut one, _) = PersistentLearner::spawn(
                &[Device::Cpu],
                cfg.clone(),
                weights.clone(),
                cfg.lr,
                rand::rngs::SmallRng::seed_from_u64(77),
            )
            .unwrap();
            let (mut two, _) = PersistentLearner::spawn(
                &[Device::Cpu, Device::Cpu],
                cfg.clone(),
                weights.clone(),
                cfg.lr,
                rand::rngs::SmallRng::seed_from_u64(77),
            )
            .unwrap();
            let one_reply = one.train(vec![parity_rollout()], cfg.lr, 0.01).unwrap();
            let two_reply = two
                .train(vec![parity_rollout(), parity_rollout()], cfg.lr, 0.01)
                .unwrap();

            assert_eq!(
                one_reply.averaged_gradients.len(),
                two_reply.averaged_gradients.len(),
                "repetition {repetition}: optimizer barrier count differs"
            );
            let mut max_gradient_diff = 0.0f32;
            for (barrier, (single, ddp)) in one_reply
                .averaged_gradients
                .iter()
                .zip(&two_reply.averaged_gradients)
                .enumerate()
            {
                let barrier_max = single
                    .iter()
                    .zip(ddp)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                max_gradient_diff = max_gradient_diff.max(barrier_max);
                assert!(
                    barrier_max < 1e-7,
                    "repetition {repetition}: pre-step averaged gradient differs at barrier \
                     {barrier}: max_diff={barrier_max:e}"
                );
            }

            let max_replica_diff = two_reply.weights[0]
                .values
                .iter()
                .zip(&two_reply.weights[1].values)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let max_weight_diff = one_reply.weights[0]
                .values
                .iter()
                .zip(&two_reply.weights[0].values)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "[ddp-parity] repetition={repetition} max_gradient_diff={max_gradient_diff:e} \
                 max_replica_diff={max_replica_diff:e} max_weight_diff={max_weight_diff:e}"
            );
            assert!(
                max_replica_diff < 1e-7,
                "repetition {repetition}: two-shard replicas diverged: \
                 max_diff={max_replica_diff:e}"
            );
            assert!(
                max_weight_diff < 1e-6,
                "repetition {repetition}: 1-vs-2 shard weight mismatch: \
                 max_diff={max_weight_diff:e}; pre-step max_diff={max_gradient_diff:e}"
            );
            one.shutdown().unwrap();
            two.shutdown().unwrap();
        }
    }

    #[test]
    fn non_finite_shard_discards_every_owner_at_barrier() {
        tch::manual_seed(707);
        let mut cfg = parity_config();
        cfg.epochs = 1;
        let source = nn::VarStore::new(Device::Cpu);
        let _ = PolicyNet::new(&source.root(), false, false, cfg.gc, cfg.blocks);
        let weights = snapshot_weights(&source).unwrap();
        let (mut group, _) = PersistentLearner::spawn(
            &[Device::Cpu, Device::Cpu],
            cfg.clone(),
            weights.clone(),
            cfg.lr,
            rand::rngs::SmallRng::seed_from_u64(88),
        )
        .unwrap();
        let mut poisoned = parity_rollout();
        poisoned.buffer[0][0].reward = f32::NAN;
        let reply = group
            .train(vec![poisoned, parity_rollout()], cfg.lr, 0.01)
            .unwrap();
        assert_eq!(reply.weights[0].values, weights.values);
        assert_eq!(reply.weights[1].values, weights.values);
        group.shutdown().unwrap();
    }

    #[test]
    fn persistent_learner_thread_trains_saves_and_shuts_down() {
        tch::manual_seed(505);
        let cfg = parity_config();
        let source = nn::VarStore::new(Device::Cpu);
        let _ = PolicyNet::new(&source.root(), false, false, cfg.gc, cfg.blocks);
        let weights = snapshot_weights(&source).unwrap();
        let (mut learner, params) = PersistentLearner::spawn(
            &[Device::Cpu],
            cfg.clone(),
            weights,
            cfg.lr,
            rand::rngs::SmallRng::seed_from_u64(1234),
        )
        .unwrap();
        assert!(params > 0);
        let reply = learner.train(vec![parity_rollout()], cfg.lr, 0.01).unwrap();
        assert!(reply.train_seconds >= 0.0);
        assert!(reply.snapshot_seconds >= 0.0);
        assert!(!reply.weights[0].values.is_empty());

        let path = std::env::temp_dir().join(format!(
            "oftrain-persistent-learner-{}.safetensors",
            std::process::id()
        ));
        learner.save_weights(path.to_str().unwrap()).unwrap();
        let mut loaded = nn::VarStore::new(Device::Cpu);
        let _ = PolicyNet::new(&loaded.root(), false, false, cfg.gc, cfg.blocks);
        loaded.load(&path).unwrap();
        assert_eq!(loaded.variables().len(), source.variables().len());
        std::fs::remove_file(&path).unwrap();
        learner.shutdown().unwrap();
    }
}

#[cfg(test)]
mod adv_clip_tests {
    /// Mirrors the exact normalize-then-clamp sequence in `train_update`'s
    /// batch-build stage (see `Config::adv_clip`'s doc for why both steps,
    /// in this order, are needed): confirms a single extreme outlier
    /// advantage - the same failure mode this whole incident is about -
    /// gets clamped to a bounded magnitude after normalization, while a
    /// well-behaved batch with no outliers is left untouched (normalizing
    /// a batch that's already within the clip range should never trigger
    /// clamping at all).
    fn normalize_and_clamp(adv: &mut [f32], clip: f32) {
        let total = adv.len() as f32;
        let mean = adv.iter().sum::<f32>() / total;
        let var = adv.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / total;
        let std = var.sqrt().max(1e-8);
        for v in adv.iter_mut() {
            *v = (*v - mean) / std;
        }
        if clip > 0.0 {
            for v in adv.iter_mut() {
                *v = v.clamp(-clip, clip);
            }
        }
    }

    #[test]
    fn one_extreme_outlier_among_many_tiny_advantages_gets_clamped() {
        // Exactly the shape of the live incident: ~2048 samples, all but
        // one near zero, one legitimate-but-rare outlier return spike.
        let mut adv = vec![0.01f32; 2047];
        adv.push(4000.0);
        normalize_and_clamp(&mut adv, 10.0);
        for &v in &adv {
            assert!(
                v.abs() <= 10.0 + 1e-4,
                "every normalized advantage must be clamped to +-10, got {v}"
            );
        }
        // The outlier should still be *at* the clip boundary (not
        // collapsed to 0 or left unclamped) - clamping should engage, not
        // silently no-op.
        assert!(
            (adv[2047] - 10.0).abs() < 1e-3,
            "the outlier should sit exactly at the clip boundary, got {}",
            adv[2047]
        );
    }

    #[test]
    fn well_behaved_batch_is_never_clamped() {
        let mut adv: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 25.0).collect();
        let before = adv.clone();
        normalize_and_clamp(&mut adv, 10.0);
        for (a, b) in adv.iter().zip(before.iter()) {
            let mean = before.iter().sum::<f32>() / before.len() as f32;
            let var = before.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / before.len() as f32;
            let std = var.sqrt().max(1e-8);
            let expected = (b - mean) / std;
            assert!(
                (a - expected).abs() < 1e-4,
                "no clamping should engage for a well-behaved batch"
            );
        }
    }

    #[test]
    fn zero_disables_clamping_entirely() {
        // Same shape as the "one outlier among many" test above (2047
        // near-zero + 1 outlier) - with only ~100 samples total, the
        // outlier itself would dominate the population std enough that
        // its own normalized value isn't actually that extreme (a
        // smaller-scale version of this same test with 99 tiny samples
        // instead of 2047 gives a normalized outlier of only ~10, not
        // "huge" - the whole point of this incident is that it takes a
        // *large* population of near-zero advantages for one outlier's
        // normalized magnitude to blow up, so the test needs that same
        // scale to actually exercise the failure mode it's named for).
        let mut adv = vec![0.01f32; 2047];
        adv.push(4000.0);
        normalize_and_clamp(&mut adv, 0.0);
        // Population std ends up dominated by the outlier itself here
        // (~88, mostly from its own contribution to the variance sum), so
        // its own normalized value lands around ~45 - well above the
        // adv_clip=10.0 default the other test confirms clamping engages
        // at, which is exactly the point: unclamped, this is still a
        // 40+-std-dev outlier feeding straight into the policy gradient.
        assert!(
            adv[2047].abs() > 20.0,
            "with adv_clip=0.0 (disabled), the outlier must NOT be clamped down near +-10, got {}",
            adv[2047]
        );
    }
}

#[cfg(test)]
mod huber_value_loss_tests {
    use tch::{Kind, Tensor};

    /// The core property the 2026-07-12 fix relies on: Huber loss's
    /// *gradient* w.r.t. the prediction stays bounded by `delta`
    /// regardless of how extreme the target is - unlike plain MSE, whose
    /// gradient is `2*(value-target)`, unbounded as the error grows. This
    /// is what a plain magnitude assertion on the loss *value* wouldn't
    /// catch (a huge loss value alone isn't the problem - see the
    /// PPO2-clipping attempt this replaced, which also produced *bounded*
    /// clipped-branch losses but still let `max(unclipped, clipped)` select
    /// an unbounded gradient) - so this test differentiates w.r.t. the
    /// prediction directly and checks the resulting gradient, not the loss.
    #[test]
    fn huber_loss_gradient_stays_bounded_for_extreme_targets() {
        let delta = 50.0;
        for &target in &[0.0, 100.0, 1_000.0, 1e9, 1e15] {
            let value = Tensor::from_slice(&[0.0f32]).set_requires_grad(true);
            let ret = Tensor::from_slice(&[target as f32]);
            let loss = value.huber_loss(&ret, tch::Reduction::Mean, delta);
            loss.backward();
            let grad = f64::try_from(value.grad()).unwrap();
            assert!(
                grad.abs() <= delta + 1e-4,
                "gradient magnitude must stay <= delta={delta} (Huber's exact bound beyond \
                 the quadratic region, regardless of how much larger the error is) even for \
                 target={target}, got {grad}"
            );
        }
    }

    /// Below `delta`, Huber must behave identically to plain MSE (matching
    /// "normal"/healthy training exactly, per Config::vf_clip's doc) - only
    /// large errors should ever see different behavior.
    #[test]
    fn huber_loss_matches_mse_for_small_errors() {
        let delta = 50.0;
        let value = Tensor::from_slice(&[1.0f32, 2.0, -3.0]);
        let ret = Tensor::from_slice(&[1.5f32, 2.2, -3.1]);
        let huber = value.huber_loss(&ret, tch::Reduction::Mean, delta);
        let mse = (&value - &ret).pow_tensor_scalar(2).mean(Kind::Float);
        let (h, m) = (f64::try_from(&huber).unwrap(), f64::try_from(&mse).unwrap());
        // Huber's small-error branch is 0.5*(err)^2 (half of MSE's err^2);
        // both scale identically with error, only the constant differs.
        assert!(
            (h - 0.5 * m).abs() < 1e-5,
            "huber={h} should be ~0.5*mse={m} for small errors"
        );
    }
}

#[cfg(test)]
mod lr_warmup_tests {
    use super::lr_warmup_frac;

    #[test]
    fn ramps_linearly_from_the_first_update_to_full_strength() {
        assert!((lr_warmup_frac(0, 0, 20) - 1.0 / 20.0).abs() < 1e-9);
        assert!((lr_warmup_frac(9, 0, 20) - 10.0 / 20.0).abs() < 1e-9);
        assert!((lr_warmup_frac(19, 0, 20) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn never_exceeds_full_strength_after_warmup_completes() {
        assert_eq!(lr_warmup_frac(20, 0, 20), 1.0);
        assert_eq!(lr_warmup_frac(1_000_000, 0, 20), 1.0);
    }

    #[test]
    fn resets_correctly_from_a_nonzero_warmup_start_update() {
        // Mirrors a curriculum advance at update 137: the very next
        // update (138) must be back at the start of a fresh ramp, not
        // treated as update 138 of an already-long-since-finished ramp.
        assert!((lr_warmup_frac(138, 138, 20) - 1.0 / 20.0).abs() < 1e-9);
        assert_eq!(lr_warmup_frac(158, 138, 20), 1.0);
    }

    #[test]
    fn a_warmup_window_of_zero_updates_never_panics_and_is_full_strength_immediately() {
        assert_eq!(lr_warmup_frac(0, 0, 0), 1.0);
    }
}

#[cfg(test)]
mod ret_stat_tests {
    use super::RetStat;

    #[test]
    fn std_is_infinite_before_enough_data_so_the_adaptive_bound_is_a_no_op() {
        let mut s = RetStat::default();
        assert!(s.std().is_infinite());
        s.add_batch(1.0, 5.0, 25.0);
        // Still only one sample total - can't estimate variance yet.
        assert!(s.std().is_infinite());
        s.add_batch(1.0, 5.0, 25.0);
        assert!(s.std().is_finite());
    }

    #[test]
    fn mean_and_std_match_a_known_distribution() {
        // 1, 2, 3, 4, 5: mean=3, population variance=2, std=sqrt(2).
        let mut s = RetStat::default();
        for x in [1.0, 2.0, 3.0, 4.0, 5.0] {
            s.add_batch(1.0, x, x * x);
        }
        assert!((s.mean() - 3.0).abs() < 1e-9, "mean={}", s.mean());
        assert!((s.std() - 2.0f64.sqrt()).abs() < 1e-9, "std={}", s.std());
    }

    #[test]
    fn merging_partial_batches_matches_merging_all_samples_at_once() {
        // The whole point of using plain sum/sum_sq (see RetStat's doc):
        // splitting the same data across several add_batch calls (as the
        // per-shard threads in train_update do) must give an identical
        // result to accumulating it all in one call.
        let mut split = RetStat::default();
        split.add_batch(2.0, 1.0 + 2.0, 1.0 + 4.0);
        split.add_batch(3.0, 3.0 + 4.0 + 5.0, 9.0 + 16.0 + 25.0);

        let mut whole = RetStat::default();
        whole.add_batch(
            5.0,
            1.0 + 2.0 + 3.0 + 4.0 + 5.0,
            1.0 + 4.0 + 9.0 + 16.0 + 25.0,
        );

        assert!((split.mean() - whole.mean()).abs() < 1e-9);
        assert!((split.std() - whole.std()).abs() < 1e-9);
    }
}

#[cfg(test)]
mod ratio_clamp_tests {
    use tch::{Kind, Tensor};

    /// Replicates the exact `log_ratio`/`ratio`/`surr1`/`surr2`/`pg_loss`
    /// formula from the minibatch loop above. A live run's pg_loss spiked
    /// to 1.48 TRILLION from a single sample whose `logp - old_logp_t`
    /// (the pre-fix unclamped input to `.exp()`) got large enough that the
    /// resulting astronomical ratio, paired with a negative advantage,
    /// made `min(surr1, surr2)` select the *unclamped* branch - this test
    /// reproduces that exact shape (huge log-ratio, negative advantage)
    /// and asserts the fixed formula keeps pg_loss bounded.
    fn pg_loss(logp: &Tensor, old_logp: &Tensor, adv: &Tensor, clip: f64) -> f64 {
        let log_ratio = (logp - old_logp).clamp(-20.0, 20.0);
        let ratio = log_ratio.exp();
        let surr1 = &ratio * adv;
        let surr2 = ratio.clamp(1.0 - clip, 1.0 + clip) * adv;
        f64::try_from(-surr1.minimum(&surr2).mean(Kind::Float)).unwrap()
    }

    #[test]
    fn pg_loss_stays_bounded_for_a_huge_log_ratio_with_negative_advantage() {
        // Exactly the pathological combination: policy drifted far enough
        // during minibatch updates that logp - old_logp is enormous, and
        // the advantage is negative (so min() would otherwise pick the
        // unbounded surr1 branch).
        let logp = Tensor::from_slice(&[1000.0f32]);
        let old_logp = Tensor::from_slice(&[0.0f32]);
        let adv = Tensor::from_slice(&[-10.0f32]); // adv_clip's default bound
        let loss = pg_loss(&logp, &old_logp, &adv, 0.2);
        assert!(loss.is_finite(), "pg_loss must stay finite, got {loss}");
        // exp(20) * 10 is the worst case the clamp permits; give some slack.
        assert!(loss.abs() < 1.0e10, "pg_loss must stay bounded, got {loss}");
    }

    #[test]
    fn pg_loss_is_unaffected_by_the_clamp_for_ordinary_ratios() {
        // A log-ratio well within the clamp's range must produce the exact
        // same result as the unclamped formula would - the fix must not
        // perturb ordinary, healthy PPO updates.
        let logp = Tensor::from_slice(&[0.05f32, -0.1, 0.2]);
        let old_logp = Tensor::from_slice(&[0.0f32, 0.0, 0.0]);
        let adv = Tensor::from_slice(&[1.0f32, -1.0, 0.5]);
        let clamped = pg_loss(&logp, &old_logp, &adv, 0.2);
        let ratio = (&logp - &old_logp).exp();
        let surr1 = &ratio * &adv;
        let surr2 = ratio.clamp(0.8, 1.2) * &adv;
        let unclamped = f64::try_from(-surr1.minimum(&surr2).mean(Kind::Float)).unwrap();
        assert!(
            (clamped - unclamped).abs() < 1e-5,
            "clamp must be a no-op for ordinary ratios: {clamped} vs {unclamped}"
        );
    }
}

#[cfg(test)]
mod nan_guard_tests {
    use super::*;

    #[test]
    fn all_finite_losses_are_not_flagged() {
        let losses = vec![(0.02, 0.14, 2.1, -0.08), (0.01, 0.09, 3.6, -0.08)];
        assert!(!any_loss_non_finite(&losses));
    }

    #[test]
    fn a_single_nan_value_loss_is_flagged() {
        let losses = vec![(0.02, 0.14, 2.1, -0.08), (0.01, f64::NAN, 3.6, -0.08)];
        assert!(any_loss_non_finite(&losses));
    }

    #[test]
    fn infinite_pg_loss_is_flagged() {
        let losses = vec![(f64::INFINITY, 0.14, 2.1, -0.08)];
        assert!(any_loss_non_finite(&losses));
    }

    #[test]
    fn nan_in_entropy_or_entq_is_also_flagged() {
        assert!(any_loss_non_finite(&[(0.0, 0.0, f64::NAN, 0.0)]));
        assert!(any_loss_non_finite(&[(0.0, 0.0, 0.0, f64::NAN)]));
    }

    #[test]
    fn a_merely_large_but_finite_loss_is_not_flagged() {
        // The actual incident this guards against saw v climb into the
        // trillions *before* going NaN - large-but-finite values should
        // still train (badly, but not corrupt weights outright); only the
        // NaN/Inf endpoint itself should skip the optimizer step.
        let losses = vec![(0.1, 1_265_838_468_017.77, 3.3, -0.27)];
        assert!(!any_loss_non_finite(&losses));
    }
}

#[cfg(test)]
mod engine_mix_tests {
    use super::*;

    #[test]
    fn zero_fraction_is_always_default() {
        for idx in 0..50 {
            assert_eq!(
                engine_for_idx(idx, EngineKind::Native, 0.0),
                EngineKind::Native
            );
        }
    }

    #[test]
    fn full_fraction_is_always_node() {
        for idx in 0..50 {
            assert_eq!(
                engine_for_idx(idx, EngineKind::Native, 1.0),
                EngineKind::Node
            );
        }
    }

    #[test]
    fn one_fifth_gives_exactly_one_node_per_five_evenly_spread() {
        let picks: Vec<usize> = (0..50)
            .filter(|&i| engine_for_idx(i, EngineKind::Native, 0.2) == EngineKind::Node)
            .collect();
        assert_eq!(picks.len(), 10, "50 * 0.2 == 10 node envs, got {picks:?}");
        // Evenly spread, not clumped: consecutive picks should be exactly
        // 5 apart, and the ratio should hold over any prefix, not just the
        // full range (matters once autoscale appends more envs at
        // ever-increasing indices - see engine_for_idx's doc comment).
        for w in picks.windows(2) {
            assert_eq!(
                w[1] - w[0],
                5,
                "picks should be evenly spread 5 apart, got {picks:?}"
            );
        }
        for prefix_len in [5, 15, 25, 35, 45] {
            let n = (0..prefix_len)
                .filter(|&i| engine_for_idx(i, EngineKind::Native, 0.2) == EngineKind::Node)
                .count();
            assert_eq!(
                n,
                prefix_len / 5,
                "ratio should hold at prefix {prefix_len}, got {n} node envs"
            );
        }
    }

    #[test]
    fn respects_default_engine_choice_for_the_majority() {
        // node_fraction only carves Node envs OUT of the majority - it
        // should never silently override an explicit `--engine node` run
        // into anything but all-Node.
        for idx in 0..20 {
            assert_eq!(engine_for_idx(idx, EngineKind::Node, 0.2), EngineKind::Node);
        }
    }
}
