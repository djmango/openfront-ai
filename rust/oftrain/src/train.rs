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
//! thread. At each optimizer barrier an optional feature-gated NCCL path
//! reduces flat gradients on each owner's current CUDA stream; builds or
//! launches without NCCL retain the CPU `Vec<f32>` hub. A single owner steps
//! its already-final device gradient directly. No Tensor or VarStore crosses
//! threads. Checkpoint weights are written by owner 0 and coordinator-ordered
//! sidecars retain their format/order.
//!
//! Persistent owners deliberately do not accept live env-resize commands.
//! Stage-aware sizing and GPU-util autoscale both checkpoint at an update
//! boundary and exit with a machine-readable restart request
//! (`restart_request.json`); pod_train relaunches at the new per-shard count.
//! Legacy (non-persistent) runs still grow env workers in-process.
//!
//! ## Dual env-group pipelining (`--pipeline-groups`)
//!
//! Inside a single `collect_rollout`, workers are split into two halves
//! (Python `rl/ppo.py` v4.1): encode+act(g0) → send(g0) → encode+act(g1)
//! while g0's engines step → recv(g0) → … . With one env the second group
//! is empty and the path degenerates to the classic lockstep loop. Default
//! on.
//! Gated persistent runs may instead use `--work-conserving-actors`: each
//! worker publishes its CPU `PreparedObs` as soon as stepping finishes, and
//! the actor drains a ready queue into microbatches bounded by target/max
//! size and `--actor-max-wait-ms`. Dispatch prefers the oldest env's exact
//! shape (zero padding); mixed-shape compact coalescing is only a fallback
//! after `max_wait`, still bounded by `--actor-max-padding-waste`. AE encode
//! and foveated crop stay exact-shape. Rollout slots remain T-major/N-minor
//! regardless of completion order. The fixed-half collector remains the
//! default/non-persistent fallback.
//!
//! Multi-GPU (see `LearnerShard`/`ActorShard`): one `PolicyNet`/`VarStore`
//! replica per device, each owning a disjoint slice of envs, in a single
//! process/thread. This mirrors `rl/ppo.py`'s DDP mode (torchrun ranks
//! each own `envs/world` environments and a full local rollout/epoch/
//! minibatch loop, with gradients flat-all-reduced-and-averaged once per
//! optimizer step - see the comment above `dist.all_reduce(flat)` there)
//! rather than wrapping in `nn.parallel.DistributedDataParallel`. `tch`/
//! `torch-sys` has no NCCL bindings. The legacy path uses P2P copies;
//! persistent owners use the small exact-libtorch NCCL shim when compiled and
//! initialized, or explicit CPU flat-gradient messages otherwise. Both
//! implement "average grad before step" semantics.
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
use std::collections::VecDeque;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use ofcore::curriculum::RewardConfig;
use ofcore::feat::ACTIONS;
use ofcore::translate::Choice;
use rand::seq::SliceRandom;
use rand::{RngCore, SeedableRng};
use tch::nn::OptimizerConfig;
use tch::{Cuda, Device, Kind, Tensor, nn};

use crate::autoscale;
use crate::batch::{self, ChoiceScalars};
use crate::engine::EngineKind;
use crate::gpu_util::GpuUtilSampler;
use crate::metrics::MetricsWriter;
use crate::policy::{self, PolicyNet};
use crate::recurrent::ActorRecurrentState;
use crate::vecenv::{ActionOutcome, EnvTransition, EnvWorker, EpisodeInfo, PreparedObs};

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
    /// Versioned schedule identity; V10 is the only live schedule.
    pub curriculum_schedule: ofcore::curriculum::CurriculumSchedule,
    /// Allows V8.3/V8.6 checkpoint -> V10 schedule + anti-spiral reward profile.
    pub migrate_v86_to_v10: bool,
    /// Optional env workers-per-shard target for every curriculum stage.
    /// A target change is applied by checkpointing and restarting so
    /// persistent actor/learner CUDA ownership never changes threads.
    pub stage_env_targets: Vec<usize>,
    pub max_episode_ticks: i64,
    pub rollout_len: usize,
    pub updates: u64,
    pub lr: f64,
    pub gamma: f32,
    pub reward_config: RewardConfig,
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
    /// `lr * stage_lr_decay ^ stage`, applied on curriculum advance/demote
    /// and recomputed on resume (floored by `stage_lr_floor`).
    pub stage_lr_decay: f64,
    /// Lower bound for stage-decayed LR (default [`ofcore::curriculum::V10_STAGE_LR_FLOOR`]).
    pub stage_lr_floor: f64,
    /// Max multiplier on stage LR when win-rate is far below the stage gate
    /// (`1.0` disables). See [`ofcore::curriculum::performance_lr_scale`].
    pub lr_perf_max_boost: f64,
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
    /// `PreparedObs.grid` stays f32. Default off (opt-in); pod_train_v10
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
    /// learner CUDA state and PPO execution. Weights remain packed CPU-only
    /// messages; gradients use owner-local NCCL when available and a packed
    /// CPU fallback otherwise. Disabled by default. With `--auto-scale-envs`,
    /// growth uses the same restart-resize path as stage env targets (no
    /// live mid-run spawn under persistent CUDA ownership).
    pub persistent_actors: bool,
    /// Opt-in recurrent actor state. This is restricted to persistent actors
    /// so the device hidden tensor never changes owner threads.
    pub recurrent_policy: bool,
    /// Per-environment hidden width in f32 values (default 256).
    pub recurrent_hidden_size: usize,
    /// Timesteps per truncated recurrent backpropagation chunk.
    pub bptt_chunk_len: usize,
    /// Opt-in ready-queue scheduler for persistent actors. The default-off
    /// fallback remains the fixed two-half collector.
    pub work_conserving_actors: bool,
    /// Hard maximum coalesced actor inference batch.
    pub actor_max_batch: usize,
    /// Preferred cross-shape compact inference batch.
    pub actor_target_batch: usize,
    /// Maximum fraction of padded compact coarse cells in one dispatch.
    pub actor_max_padding_waste: f64,
    /// Maximum age of the oldest item before a partial microbatch dispatches.
    pub actor_max_wait: Duration,
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
    /// Run evaluation on its own long-lived owner thread. The coordinator
    /// only submits immutable CPU weight snapshots and polls completed work
    /// at update boundaries. When no eval device is available the existing
    /// synchronous path is retained.
    pub async_eval: bool,
    /// Dedicated device for asynchronous evaluation. `main` resolves the
    /// automatic spare-GPU default; `None` selects synchronous evaluation.
    pub eval_device: Option<Device>,
    pub ckpt_every: u64,
    pub ckpt_dir: String,
    /// Keep only the newest N numbered `policy_update*.safetensors` locally
    /// (plus matching `.state.json`). Curriculum advance/demote milestones,
    /// `latest.*`, and `best_eval.*` are never pruned. HF already retains the
    /// full historical backlog — local prune is disk hygiene only. `0` disables.
    pub ckpt_keep_last: usize,
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
    /// Stage-advance warmup still uses V10's stage warmup; this only
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
    /// Architecture/checkpoint contract. Recurrent full resumes require 2.
    #[serde(default)]
    pub checkpoint_schema_version: u32,
    /// Per-environment hidden state reset semantics.
    #[serde(default)]
    pub hidden_reset_policy: String,
    pub update: u64,
    pub stage: usize,
    pub ent_scale: f64,
    pub lr_now: f64,
    pub total_env_steps: u64,
    pub recent_wins: Vec<f64>,
    /// Qualifying closeout episodes only (max fixed-land share >= 0.45).
    #[serde(default)]
    pub recent_conversions: Vec<f64>,
    /// On-stage non-rehearsal death outcomes (1.0 died / 0.0 survived episode).
    #[serde(default)]
    pub recent_deaths: Vec<f64>,
    /// Best fixed-seed evaluation associated with this checkpoint. Optional
    /// for backward compatibility with sidecars written before async eval.
    #[serde(default)]
    pub best_eval_win: Option<f64>,
    #[serde(default)]
    pub best_eval_score: Option<f64>,
    /// Curriculum/sizing identity is persisted so a supervisor with stale
    /// launch flags cannot silently resume under another schedule.
    #[serde(default)]
    pub curriculum_schedule: Option<String>,
    /// Reward semantics identity. Required for V8.3 resumes.
    #[serde(default)]
    pub reward_profile: Option<String>,
    /// Reserved backward-compatible pass-through for return normalization.
    #[serde(default)]
    pub return_stats: Option<serde_json::Value>,
    #[serde(default)]
    pub stage_env_targets: Vec<usize>,
    #[serde(default)]
    pub envs_per_shard: usize,
    /// Non-null only on the checkpoint that requests a stage-boundary
    /// restart at a new equal per-shard env count.
    #[serde(default)]
    pub requested_env_target: Option<usize>,
}

impl TrainState {
    fn schedule(&self) -> Result<ofcore::curriculum::CurriculumSchedule> {
        match self.curriculum_schedule.as_deref() {
            Some("v10") => Ok(ofcore::curriculum::CurriculumSchedule::V10),
            Some(id) => anyhow::bail!(
                "checkpoint has unsupported curriculum schedule {id:?}; V10 is required"
            ),
            None => anyhow::bail!(
                "checkpoint is missing curriculum schedule; migrate to V10 explicitly"
            ),
        }
    }
}

fn expand_v10_sidecar_if_needed(state: &mut TrainState) {
    let targets_len = state.stage_env_targets.len();
    let remapped = if targets_len == ofcore::curriculum::V10_LEGACY_LEN {
        let as35 = state
            .stage
            .saturating_add(20)
            .min(ofcore::curriculum::V10_PREV35_LEN - 1);
        Some(ofcore::curriculum::remap_v10_stage_35_to_100(as35))
    } else if targets_len == ofcore::curriculum::V10_PREV35_LEN {
        Some(ofcore::curriculum::remap_v10_stage_35_to_100(state.stage))
    } else {
        None
    };
    if let Some(new_stage) = remapped {
        let old = state.stage;
        state.stage = new_stage;
        state.stage_env_targets.clear();
        state.recent_wins.clear();
        state.recent_conversions.clear();
        state.recent_deaths.clear();
        println!(
            "[train] V10 100-stage expand: stage {old} -> {new_stage} \
             (from {targets_len}-slot sidecar; cleared windows; env targets reset)"
        );
    }
}

fn reconcile_resume_schedule(
    state: &mut TrainState,
    requested: ofcore::curriculum::CurriculumSchedule,
    migrate_v86_to_v10: bool,
) -> Result<()> {
    anyhow::ensure!(
        requested == ofcore::curriculum::CurriculumSchedule::V10,
        "oftrain only supports the V10 curriculum"
    );
    match state.curriculum_schedule.as_deref() {
        Some("v10") => {
            anyhow::ensure!(
                !migrate_v86_to_v10,
                "--migrate-v86-to-v10 was supplied but checkpoint already uses v10"
            );
            anyhow::ensure!(
                state.reward_profile.as_deref() == Some(ofcore::curriculum::V10_REWARD_PROFILE),
                "V10 checkpoint reward profile mismatch: expected {:?}, found {:?}",
                ofcore::curriculum::V10_REWARD_PROFILE,
                state.reward_profile
            );
            expand_v10_sidecar_if_needed(state);
            Ok(())
        }
        Some(ofcore::curriculum::LEGACY_V83_SCHEDULE_ID) if migrate_v86_to_v10 => {
            anyhow::ensure!(
                state.reward_profile.as_deref() == Some(ofcore::curriculum::V86_REWARD_PROFILE),
                "V86-to-V10 migration requires reward profile {:?}, found {:?}",
                ofcore::curriculum::V86_REWARD_PROFILE,
                state.reward_profile
            );
            state.curriculum_schedule =
                Some(ofcore::curriculum::CurriculumSchedule::V10.id().to_string());
            state.reward_profile = Some(ofcore::curriculum::V10_REWARD_PROFILE.to_string());
            state.stage = ofcore::curriculum::V10_CLOSEOUT_STAGE;
            state.stage_env_targets.clear();
            state.recent_wins.clear();
            state.recent_conversions.clear();
            state.recent_deaths.clear();
            state.best_eval_win = None;
            state.best_eval_score = None;
            state.requested_env_target = None;
            Ok(())
        }
        Some(ofcore::curriculum::LEGACY_V83_SCHEDULE_ID) => {
            anyhow::bail!("V8.3/V8.6 checkpoint requires --migrate-v86-to-v10 to resume under V10")
        }
        Some(id) => anyhow::bail!(
            "checkpoint curriculum schedule {id:?} is incompatible with V10; only v10 resumes and explicit v8.3/V86 -> V10 migration are supported"
        ),
        None => anyhow::bail!(
            "checkpoint is missing curriculum_schedule; only explicit v8.3/V86 -> V10 migration or native v10 sidecars are supported"
        ),
    }
}

/// Detect sidecars whose `stage` was rewritten downward while `lr_now` still
/// matches a much higher stage (observed: stage 28→8 at update 11571 with
/// `lr_now` left at `2.6e-6`). Restore the implied stage, then always
/// recompute `lr_now` from the (possibly corrected) stage + floor.
fn reconcile_resume_stage_and_lr(
    state: &mut TrainState,
    base_lr: f64,
    decay: f64,
    floor: f64,
) {
    const MIN_IMPLIED_STAGE_GAP: usize = 3;
    if let Some(implied) =
        ofcore::curriculum::imply_stage_from_learning_rate(state.lr_now, base_lr, decay)
    {
        let uncapped_for_sidecar =
            base_lr * decay.powi(state.stage as i32);
        if implied >= state.stage + MIN_IMPLIED_STAGE_GAP
            && state.lr_now < uncapped_for_sidecar * 0.5
        {
            println!(
                "[train] refusing unexplained resume stage drop: sidecar stage={} \
                 but lr_now={:.2e} implies stage~{implied}; restoring stage",
                state.stage, state.lr_now
            );
            state.stage = implied;
            state.recent_wins.clear();
            state.recent_conversions.clear();
            state.recent_deaths.clear();
        }
    }
    let corrected =
        ofcore::curriculum::stage_learning_rate(base_lr, decay, state.stage, floor);
    if (corrected - state.lr_now).abs() > 1e-15 {
        println!(
            "[train] recomputed resume lr_now: {:.2e} -> {corrected:.2e} \
             (stage={}, decay={decay}, floor={floor:.2e})",
            state.lr_now, state.stage
        );
        state.lr_now = corrected;
    }
}

#[cfg(test)]
mod resume_stage_lr_tests {
    use super::{reconcile_resume_stage_and_lr, TrainState};
    use ofcore::curriculum::{V10_REWARD_PROFILE, V10_STAGE_LR_FLOOR};

    fn state(stage: usize, lr_now: f64) -> TrainState {
        TrainState {
            checkpoint_schema_version: 2,
            hidden_reset_policy: "episode_done".into(),
            update: 1,
            stage,
            ent_scale: 1.0,
            lr_now,
            total_env_steps: 0,
            recent_wins: vec![1.0; 4],
            recent_conversions: vec![],
            recent_deaths: vec![],
            best_eval_win: None,
            best_eval_score: None,
            curriculum_schedule: Some("v10".into()),
            reward_profile: Some(V10_REWARD_PROFILE.into()),
            return_stats: None,
            stage_env_targets: vec![],
            envs_per_shard: 0,
            requested_env_target: None,
        }
    }

    #[test]
    fn restores_stage_when_lr_implies_much_higher_stage() {
        let base: f64 = 2.5e-4;
        let decay: f64 = 0.85;
        let lr28 = base * decay.powi(28);
        let mut st = state(8, lr28);
        reconcile_resume_stage_and_lr(&mut st, base, decay, V10_STAGE_LR_FLOOR);
        assert_eq!(st.stage, 28);
        assert!((st.lr_now - V10_STAGE_LR_FLOOR).abs() < 1e-15);
        assert!(st.recent_wins.is_empty());
    }

    #[test]
    fn recomputes_lr_to_floor_without_stage_rewrite() {
        let base: f64 = 2.5e-4;
        let decay: f64 = 0.85;
        let mut st = state(24, base * decay.powi(24));
        reconcile_resume_stage_and_lr(&mut st, base, decay, V10_STAGE_LR_FLOOR);
        assert_eq!(st.stage, 24);
        assert!((st.lr_now - V10_STAGE_LR_FLOOR).abs() < 1e-15);
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
struct EnvResizeRequest {
    format: u32,
    reason: String,
    update: u64,
    stage: usize,
    current_envs_per_shard: usize,
    requested_envs_per_shard: usize,
    num_shards: usize,
    checkpoint: String,
}

fn restart_request_path(ckpt_dir: &str) -> String {
    format!("{ckpt_dir}/restart_request.json")
}

fn should_advance(recent_wins: &std::collections::VecDeque<f64>, gate: f64) -> bool {
    recent_wins.len() == ofcore::curriculum::WINDOW
        && recent_wins.iter().sum::<f64>() / recent_wins.len() as f64 > gate
}

fn conversion_gate_v10(stage: usize) -> Option<(usize, f64)> {
    let gate = ofcore::curriculum::v10_win_at_for_stage(stage);
    if stage == ofcore::curriculum::V10_CLOSEOUT_STAGE {
        Some((20, gate))
    } else if stage == ofcore::curriculum::V10_BRIDGE_STAGE {
        Some((16, gate))
    } else {
        None
    }
}

fn window_mean(window: &VecDeque<f64>) -> Option<f64> {
    if window.is_empty() {
        None
    } else {
        Some(window.iter().sum::<f64>() / window.len() as f64)
    }
}

fn should_advance_v10(
    stage: usize,
    recent_wins: &VecDeque<f64>,
    recent_conversions: &VecDeque<f64>,
    recent_deaths: &VecDeque<f64>,
    win_gate: f64,
) -> bool {
    should_advance(recent_wins, win_gate)
        && conversion_gate_v10(stage).is_none_or(|(minimum, gate)| {
            recent_conversions.len() >= minimum
                && recent_conversions.iter().sum::<f64>() / recent_conversions.len() as f64 > gate
        })
        && recent_deaths.len() == ofcore::curriculum::WINDOW
        && window_mean(recent_deaths)
            .is_some_and(|death_rate| death_rate < ofcore::curriculum::V10_ADVANCE_MAX_DEATH_RATE)
}

fn should_demote_v10(
    stage: usize,
    recent_wins: &VecDeque<f64>,
    recent_deaths: &VecDeque<f64>,
) -> bool {
    stage > 0
        && recent_wins.len() == ofcore::curriculum::WINDOW
        && recent_deaths.len() == ofcore::curriculum::WINDOW
        && window_mean(recent_wins)
            .is_some_and(|wr| wr < ofcore::curriculum::V10_DEMOTE_MAX_WIN_RATE)
        && window_mean(recent_deaths)
            .is_some_and(|dr| dr > ofcore::curriculum::V10_DEMOTE_MIN_DEATH_RATE)
}

#[allow(clippy::too_many_arguments)]
fn record_advancement_result(
    _schedule: ofcore::curriculum::CurriculumSchedule,
    current_stage: usize,
    episode_stage: usize,
    rehearsal: bool,
    won: bool,
    died: bool,
    closeout_reached: bool,
    converted: bool,
    recent_wins: &mut VecDeque<f64>,
    recent_conversions: &mut VecDeque<f64>,
    recent_deaths: &mut VecDeque<f64>,
) -> bool {
    if rehearsal || episode_stage != current_stage {
        return false;
    }
    if recent_wins.len() == ofcore::curriculum::WINDOW {
        recent_wins.pop_front();
    }
    recent_wins.push_back(if won { 1.0 } else { 0.0 });
    if recent_deaths.len() == ofcore::curriculum::WINDOW {
        recent_deaths.pop_front();
    }
    recent_deaths.push_back(if died { 1.0 } else { 0.0 });
    if closeout_reached {
        if recent_conversions.len() == ofcore::curriculum::WINDOW {
            recent_conversions.pop_front();
        }
        recent_conversions.push_back(if converted { 1.0 } else { 0.0 });
    }
    true
}

fn requested_stage_env_target(
    targets: &[usize],
    stage: usize,
    current_envs_per_shard: usize,
) -> Option<usize> {
    targets
        .get(stage)
        .copied()
        .filter(|&target| target != current_envs_per_shard)
}

/// Like [`requested_stage_env_target`], but apply `--max-envs` before deciding
/// whether a restart is warranted.
///
/// Without this, a stage floor of 24 with `MAX_ENVS=16` requests a restart on
/// every advance/demote, then clamps back to 14 on boot — a full cold restart
/// with no net env-count change (and it also skips in-process `set_stage`).
fn requested_stage_env_target_for_resize(
    targets: &[usize],
    stage: usize,
    current_envs_per_shard: usize,
    auto_scale_envs: bool,
    max_envs: usize,
) -> Option<usize> {
    let target = requested_stage_env_target(targets, stage, current_envs_per_shard)?;
    let capped = clamp_resolved_envs_to_autoscale_max(target, auto_scale_envs, max_envs);
    (capped != current_envs_per_shard).then_some(capped)
}

/// Resolve per-shard env count at process start.
///
/// Stage schedule values are a *floor* within a stage: an autoscale restart
/// (or operator `--num-envs`) above the floor must stick across resume.
/// Explicit `TrainState.requested_env_target` always wins (stage shrink/grow
/// or GPU-util autoscale restart). Cold start with no resume uses the stage
/// floor when present.
fn resolve_startup_envs_per_shard(
    cli_num_envs: usize,
    stage_target: Option<usize>,
    resumed: Option<&TrainState>,
    fulfilling_restart: bool,
) -> usize {
    if let Some(req) = resumed.and_then(|s| s.requested_env_target) {
        return req.max(1);
    }
    let Some(stage) = stage_target else {
        return cli_num_envs.max(1);
    };
    if fulfilling_restart {
        // pod_train already applied restart_request.json → --num-envs.
        return cli_num_envs.max(1);
    }
    if let Some(prev) = resumed.map(|s| s.envs_per_shard).filter(|&n| n > 0) {
        return prev.max(stage);
    }
    stage
}

/// Cap a pending resize request to `--max-envs` when autoscale is on, so a
/// stale oversized `requested_env_target` (or an old restart request) cannot
/// OOM the next boot.
fn clamp_resolved_envs_to_autoscale_max(
    resolved: usize,
    auto_scale_envs: bool,
    max_envs: usize,
) -> usize {
    if auto_scale_envs && max_envs > 0 {
        resolved.min(max_envs).max(1)
    } else {
        resolved.max(1)
    }
}

fn state_sidecar_path(ckpt_path: &str) -> String {
    let stem = ckpt_path
        .strip_suffix(".safetensors")
        .or_else(|| ckpt_path.strip_suffix(".ot"))
        .unwrap_or(ckpt_path);
    format!("{stem}.state.json")
}

fn manifest_path(ckpt_path: &str) -> std::path::PathBuf {
    std::path::Path::new(ckpt_path)
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join("manifest.json")
}

fn read_architecture_manifest(ckpt_path: &str, expected_schema: u64) -> Result<serde_json::Value> {
    anyhow::ensure!(
        ckpt_path.ends_with(".safetensors"),
        "schema-aware policy loading requires a .safetensors checkpoint"
    );
    let path = manifest_path(ckpt_path);
    let text = std::fs::read_to_string(&path).map_err(|error| {
        anyhow!(
            "checkpoint manifest {} is required: {error}",
            path.display()
        )
    })?;
    let manifest: serde_json::Value = serde_json::from_str(&text)?;
    anyhow::ensure!(
        manifest["format"] == "oftrain-safetensors",
        "unsupported checkpoint format"
    );
    anyhow::ensure!(
        manifest["manifest_schema_version"] == 1,
        "unsupported manifest schema"
    );
    anyhow::ensure!(
        manifest["architecture"]["schema_version"] == expected_schema,
        "checkpoint architecture schema mismatch: expected v{expected_schema}, found {}",
        manifest["architecture"]["schema_version"]
    );
    Ok(manifest)
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

#[cfg(test)]
mod v10_state_and_gate_tests {
    use super::*;
    use std::collections::VecDeque;

    fn state_for_schedule(id: Option<&str>, stage: usize) -> TrainState {
        TrainState {
            checkpoint_schema_version: 1,
            hidden_reset_policy: "none".to_string(),
            update: 1,
            stage,
            ent_scale: 1.0,
            lr_now: 1e-4,
            total_env_steps: 32,
            recent_wins: vec![0.0; ofcore::curriculum::WINDOW],
            recent_conversions: vec![],
            recent_deaths: vec![],
            best_eval_win: None,
            best_eval_score: None,
            curriculum_schedule: id.map(str::to_string),
            reward_profile: Some(ofcore::curriculum::V10_REWARD_PROFILE.to_string()),
            return_stats: None,
            stage_env_targets: ofcore::curriculum::V10_ENV_TARGETS.to_vec(),
            envs_per_shard: 24,
            requested_env_target: Some(12),
        }
    }

    #[test]
    fn gate_requires_a_full_window_and_strictly_exceeds_threshold() {
        let mut wins = VecDeque::from(vec![1.0; ofcore::curriculum::WINDOW - 1]);
        assert!(!should_advance(&wins, 0.35), "39 wins must not advance");

        wins = VecDeque::from(
            (0..ofcore::curriculum::WINDOW)
                .map(|index| if index < 14 { 1.0 } else { 0.0 })
                .collect::<Vec<_>>(),
        );
        assert!(
            !should_advance(&wins, 0.35),
            "exactly 0.35 is below the strict gate"
        );
        wins[14] = 1.0;
        assert!(should_advance(&wins, 0.35));
    }

    #[test]
    fn advancement_windows_exclude_rehearsal_off_stage_and_record_v10_signals() {
        let mut wins = VecDeque::new();
        let mut conversions = VecDeque::new();
        let mut deaths = VecDeque::new();
        assert!(!record_advancement_result(
            ofcore::curriculum::CurriculumSchedule::V10,
            5,
            5,
            true,
            true,
            false,
            true,
            true,
            &mut wins,
            &mut conversions,
            &mut deaths,
        ));
        assert!(!record_advancement_result(
            ofcore::curriculum::CurriculumSchedule::V10,
            5,
            4,
            false,
            true,
            false,
            true,
            true,
            &mut wins,
            &mut conversions,
            &mut deaths,
        ));
        assert!(wins.is_empty() && conversions.is_empty() && deaths.is_empty());

        assert!(record_advancement_result(
            ofcore::curriculum::CurriculumSchedule::V10,
            5,
            5,
            false,
            false,
            true,
            false,
            false,
            &mut wins,
            &mut conversions,
            &mut deaths,
        ));
        assert_eq!(wins, VecDeque::from([0.0]));
        assert_eq!(deaths, VecDeque::from([1.0]));
        assert!(conversions.is_empty());

        assert!(record_advancement_result(
            ofcore::curriculum::CurriculumSchedule::V10,
            5,
            5,
            false,
            true,
            false,
            true,
            true,
            &mut wins,
            &mut conversions,
            &mut deaths,
        ));
        assert_eq!(conversions, VecDeque::from([1.0]));
    }

    #[test]
    fn prune_numbered_checkpoints_keeps_newest_only() {
        let dir = std::env::temp_dir().join(format!(
            "oftrain-prune-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        for idx in [10u64, 20, 30, 40, 50] {
            let path = dir.join(format!("policy_update{idx}.safetensors"));
            std::fs::write(&path, b"weights").unwrap();
            std::fs::write(state_sidecar_path(path.to_str().unwrap()), b"{}").unwrap();
        }
        std::fs::write(dir.join("latest.safetensors"), b"latest").unwrap();
        std::fs::write(
            dir.join("curriculum_advance_u50_s1_to_2.safetensors"),
            b"mile",
        )
        .unwrap();
        prune_numbered_checkpoints(dir.to_str().unwrap(), 2).unwrap();
        assert!(!dir.join("policy_update10.safetensors").exists());
        assert!(!dir.join("policy_update30.safetensors").exists());
        assert!(dir.join("policy_update40.safetensors").exists());
        assert!(dir.join("policy_update50.safetensors").exists());
        assert!(dir.join("latest.safetensors").exists());
        assert!(
            dir.join("curriculum_advance_u50_s1_to_2.safetensors")
                .exists()
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn curriculum_note_summarizes_advance() {
        let stages =
            ofcore::curriculum::stages_for_schedule(ofcore::curriculum::CurriculumSchedule::V10);
        let transition = CurriculumTransition {
            event: "advance",
            from_stage: 0,
            to_stage: 1,
            win_rate: 0.975,
            conversion_rate: 1.0,
            death_rate: 0.05,
            win_gate: 0.95,
            window_size: ofcore::curriculum::WINDOW,
        };
        let note = curriculum_transition_note(
            &transition,
            123,
            ofcore::curriculum::CurriculumSchedule::V10,
            &stages,
        );
        assert_eq!(note["event"], "advance");
        assert_eq!(note["from_stage"], 0);
        assert_eq!(note["to_stage"], 1);
        let summary = note["summary"].as_str().unwrap();
        assert!(summary.contains("Advanced stage 0→1"));
        assert!(summary.contains("update 123"));
    }

    #[test]
    fn v10_advance_requires_death_ceiling_and_demotes_on_spiral() {
        let closeout = ofcore::curriculum::V10_CLOSEOUT_STAGE;
        let wins = VecDeque::from(vec![1.0; ofcore::curriculum::WINDOW]);
        let conversions = VecDeque::from(vec![1.0; 20]);
        let mut deaths = VecDeque::from(vec![0.6; ofcore::curriculum::WINDOW]);
        assert!(!should_advance_v10(
            closeout,
            &wins,
            &conversions,
            &deaths,
            0.70
        ));
        deaths = VecDeque::from(vec![0.4; ofcore::curriculum::WINDOW]);
        assert!(should_advance_v10(
            closeout,
            &wins,
            &conversions,
            &deaths,
            0.70
        ));

        let lose = VecDeque::from(vec![0.0; ofcore::curriculum::WINDOW]);
        let slaughter = VecDeque::from(vec![1.0; ofcore::curriculum::WINDOW]);
        assert!(!should_demote_v10(0, &lose, &slaughter));
        assert!(should_demote_v10(8, &lose, &slaughter));
        let ok_wins = VecDeque::from(vec![0.2; ofcore::curriculum::WINDOW]);
        assert!(!should_demote_v10(8, &ok_wins, &slaughter));
    }

    #[test]
    fn train_state_round_trips_v10_sizing() {
        let state = state_for_schedule(Some("v10"), 6);
        let restored: TrainState =
            serde_json::from_str(&serde_json::to_string(&state).unwrap()).unwrap();
        assert_eq!(
            restored.schedule().unwrap(),
            ofcore::curriculum::CurriculumSchedule::V10
        );
        assert_eq!(
            restored.stage_env_targets,
            ofcore::curriculum::V10_ENV_TARGETS
        );
        assert_eq!(restored.requested_env_target, Some(12));

        let legacy: TrainState = serde_json::from_str(
            r#"{"update":1,"stage":0,"ent_scale":1.0,"lr_now":0.0001,
                "total_env_steps":32,"recent_wins":[]}"#,
        )
        .unwrap();
        assert!(legacy.schedule().is_err());
    }

    #[test]
    fn policy_manifest_is_machine_readable_and_versioned() {
        let state = state_for_schedule(Some("v10"), 4);
        let manifest = policy_manifest_value(
            128,
            2,
            true,
            256,
            16,
            32,
            "weights/ae/fine.encoder.safetensors",
            Some("weights/ae/coarse.encoder.safetensors"),
            &state,
        );
        assert_eq!(manifest["format"], "oftrain-safetensors");
        assert_eq!(manifest["manifest_schema_version"], 1);
        assert_eq!(manifest["architecture"]["schema_version"], 2);
        assert_eq!(manifest["architecture"]["recurrent"]["hidden_size"], 256);
        assert_eq!(
            manifest["architecture"]["recurrent"]["context_schema"],
            "action-outcome-v1"
        );
        assert_eq!(
            manifest["architecture"]["recurrent"]["context_features"],
            14
        );
        assert_eq!(manifest["architecture"]["recurrent"]["bptt_length"], 16);
        assert_eq!(manifest["architecture"]["recurrent"]["rollout_length"], 32);
        assert_eq!(
            manifest["architecture"]["dimensions"]["grid_channels"],
            policy::C_GRID
        );
        assert_eq!(
            manifest["architecture"]["dimensions"]["grid_tower_channels"],
            128
        );
        assert_eq!(manifest["autoencoders"]["fine"]["format"], "safetensors");
        assert_eq!(manifest["update"], 1);
        assert_eq!(manifest["stage"], 4);
        assert_eq!(manifest["curriculum_schedule"], "v10");
    }

    #[test]
    fn resize_request_is_machine_readable_and_per_shard() {
        let request = EnvResizeRequest {
            format: 1,
            reason: "curriculum_stage_env_target".to_string(),
            update: 99,
            stage: 6,
            current_envs_per_shard: 24,
            requested_envs_per_shard: 12,
            num_shards: 4,
            checkpoint: "checkpoints/latest.safetensors".to_string(),
        };
        let json = serde_json::to_value(&request).unwrap();
        assert_eq!(json["requested_envs_per_shard"], 12);
        assert_eq!(json["num_shards"], 4);
        assert_eq!(json["reason"], "curriculum_stage_env_target");
    }

    #[test]
    fn stage_resize_plans_equal_shard_shrink_and_growth_but_not_a_noop() {
        let targets = [24, 12, 20];
        assert_eq!(requested_stage_env_target(&targets, 1, 24), Some(12));
        assert_eq!(requested_stage_env_target(&targets, 2, 12), Some(20));
        assert_eq!(requested_stage_env_target(&targets, 0, 24), None);

        let v10 = ofcore::curriculum::V10_ENV_TARGETS;
        assert_eq!(requested_stage_env_target(&v10, 54, 20), Some(16));
        assert_eq!(requested_stage_env_target(&v10, 58, 16), Some(12));
        assert_eq!(requested_stage_env_target(&v10, 82, 10), Some(8));
    }

    #[test]
    fn stage_resize_skips_noop_when_floor_exceeds_max_envs() {
        let v10 = ofcore::curriculum::V10_ENV_TARGETS;
        // Stages 0-45 want 24, but A40 pods cap at MAX_ENVS=16.
        assert_eq!(
            requested_stage_env_target_for_resize(&v10, 10, 14, true, 14),
            None
        );
        assert_eq!(
            requested_stage_env_target_for_resize(&v10, 10, 12, true, 14),
            Some(14)
        );
        // Without a max-envs cap, the raw floor still requests a restart.
        assert_eq!(
            requested_stage_env_target_for_resize(&v10, 10, 14, false, 14),
            Some(24)
        );
        // Late-stage shrink below the cap still restarts.
        assert_eq!(
            requested_stage_env_target_for_resize(&v10, 58, 14, true, 14),
            Some(12)
        );
    }

    #[test]
    fn startup_envs_keep_autoscale_gains_above_stage_floor() {
        let mut resumed = state_for_schedule(Some("v10"), 68);
        resumed.envs_per_shard = 16;
        resumed.requested_env_target = None;
        assert_eq!(
            resolve_startup_envs_per_shard(10, Some(10), Some(&resumed), false),
            16
        );
        resumed.requested_env_target = Some(20);
        assert_eq!(
            resolve_startup_envs_per_shard(10, Some(10), Some(&resumed), false),
            20
        );
        assert_eq!(
            resolve_startup_envs_per_shard(20, Some(10), Some(&resumed), true),
            20
        );
        assert_eq!(
            resolve_startup_envs_per_shard(24, Some(10), None, false),
            10
        );
    }

    #[test]
    fn resume_accepts_v10_and_expands_old_v10_sidecars() {
        let mut state = state_for_schedule(Some("v10"), 2);
        state.stage_env_targets = vec![24; ofcore::curriculum::V10_LEGACY_LEN];
        reconcile_resume_schedule(
            &mut state,
            ofcore::curriculum::CurriculumSchedule::V10,
            false,
        )
        .unwrap();
        assert_eq!(
            state.schedule().unwrap(),
            ofcore::curriculum::CurriculumSchedule::V10
        );
        assert!(state.stage >= ofcore::curriculum::V10_EASY_RAMP_LEN);
        assert!(state.stage_env_targets.is_empty());
        assert!(state.recent_wins.is_empty());
    }

    #[test]
    fn migrate_v86_to_v10_rewrites_legacy_v83_sidecar() {
        let mut state = state_for_schedule(Some(ofcore::curriculum::LEGACY_V83_SCHEDULE_ID), 14);
        state.reward_profile = Some(ofcore::curriculum::V86_REWARD_PROFILE.to_string());
        state.recent_wins = vec![1.0; ofcore::curriculum::WINDOW];
        state.recent_conversions = vec![1.0; 20];
        state.recent_deaths = vec![1.0; ofcore::curriculum::WINDOW];
        state.best_eval_win = Some(0.5);
        state.best_eval_score = Some(0.6);
        reconcile_resume_schedule(
            &mut state,
            ofcore::curriculum::CurriculumSchedule::V10,
            true,
        )
        .unwrap();
        assert_eq!(
            state.schedule().unwrap(),
            ofcore::curriculum::CurriculumSchedule::V10
        );
        assert_eq!(state.stage, ofcore::curriculum::V10_CLOSEOUT_STAGE);
        assert_eq!(
            state.reward_profile.as_deref(),
            Some(ofcore::curriculum::V10_REWARD_PROFILE)
        );
        assert!(state.recent_wins.is_empty());
        assert!(state.recent_conversions.is_empty());
        assert!(state.recent_deaths.is_empty());
        assert_eq!(state.best_eval_win, None);
        assert_eq!(state.best_eval_score, None);
        assert_eq!(state.requested_env_target, None);
    }

    #[test]
    fn reject_non_v10_resume_without_supported_migration() {
        let mut state = state_for_schedule(Some(ofcore::curriculum::LEGACY_V83_SCHEDULE_ID), 5);
        state.reward_profile = Some(ofcore::curriculum::V86_REWARD_PROFILE.to_string());
        let error = reconcile_resume_schedule(
            &mut state,
            ofcore::curriculum::CurriculumSchedule::V10,
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("migrate-v86-to-v10"));

        let mut wrong_profile =
            state_for_schedule(Some(ofcore::curriculum::LEGACY_V83_SCHEDULE_ID), 5);
        wrong_profile.reward_profile = Some("wrong".to_string());
        assert!(
            reconcile_resume_schedule(
                &mut wrong_profile,
                ofcore::curriculum::CurriculumSchedule::V10,
                true,
            )
            .is_err()
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

fn note_sidecar_path(ckpt_path: &str) -> String {
    let stem = ckpt_path
        .strip_suffix(".safetensors")
        .or_else(|| ckpt_path.strip_suffix(".ot"))
        .unwrap_or(ckpt_path);
    format!("{stem}.note.json")
}

fn parse_policy_update_index(name: &str) -> Option<u64> {
    let rest = name.strip_prefix("policy_update")?;
    let digits = rest.strip_suffix(".safetensors")?;
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Drop oldest numbered `policy_update*` checkpoints once local retention
/// exceeds `keep_last`. Never touches curriculum milestones, `latest.*`, or
/// `best_eval.*` — HF sync already keeps the full historical backlog.
fn prune_numbered_checkpoints(ckpt_dir: &str, keep_last: usize) -> Result<()> {
    if keep_last == 0 {
        return Ok(());
    }
    let dir = std::path::Path::new(ckpt_dir);
    if !dir.is_dir() {
        return Ok(());
    }
    let mut numbered: Vec<(u64, std::path::PathBuf)> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if let Some(idx) = parse_policy_update_index(name) {
            numbered.push((idx, path));
        }
    }
    if numbered.len() <= keep_last {
        return Ok(());
    }
    numbered.sort_by_key(|(idx, _)| *idx);
    let drop_n = numbered.len() - keep_last;
    for (_, path) in numbered.into_iter().take(drop_n) {
        let state = state_sidecar_path(path.to_str().unwrap_or(""));
        if let Err(e) = std::fs::remove_file(&path) {
            eprintln!("[train] WARNING: failed to prune {}: {e}", path.display());
        } else {
            println!("[train] pruned local checkpoint {}", path.display());
        }
        let _ = std::fs::remove_file(state);
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct CurriculumTransition {
    event: &'static str,
    from_stage: usize,
    to_stage: usize,
    win_rate: f64,
    conversion_rate: f64,
    death_rate: f64,
    win_gate: f64,
    window_size: usize,
}

fn nations_label(nations: &ofcore::curriculum::Nations) -> String {
    match nations {
        ofcore::curriculum::Nations::Default => "default".to_string(),
        ofcore::curriculum::Nations::Exact(n) => n.to_string(),
    }
}

fn curriculum_transition_note(
    transition: &CurriculumTransition,
    update: u64,
    schedule: ofcore::curriculum::CurriculumSchedule,
    stages: &[ofcore::curriculum::Stage],
) -> serde_json::Value {
    let from = &stages[transition.from_stage];
    let to = &stages[transition.to_stage];
    let summary = match transition.event {
        "demote" => format!(
            "Demoted stage {}→{} at update {} (WR={:.1}% < demote ceil; DR={:.1}%; CR={:.1}%). \
             {} bots={} nations={} {} → {} bots={} nations={} {}.",
            transition.from_stage,
            transition.to_stage,
            update,
            transition.win_rate * 100.0,
            transition.death_rate * 100.0,
            transition.conversion_rate * 100.0,
            from.maps.join("+"),
            from.bots,
            nations_label(&from.nations),
            from.difficulty,
            to.maps.join("+"),
            to.bots,
            nations_label(&to.nations),
            to.difficulty,
        ),
        _ => format!(
            "Advanced stage {}→{} at update {} (WR={:.1}% vs gate {:.1}%; DR={:.1}%; CR={:.1}%). \
             {} bots={} nations={} {} → {} bots={} nations={} {}.",
            transition.from_stage,
            transition.to_stage,
            update,
            transition.win_rate * 100.0,
            transition.win_gate * 100.0,
            transition.death_rate * 100.0,
            transition.conversion_rate * 100.0,
            from.maps.join("+"),
            from.bots,
            nations_label(&from.nations),
            from.difficulty,
            to.maps.join("+"),
            to.bots,
            nations_label(&to.nations),
            to.difficulty,
        ),
    };
    serde_json::json!({
        "schema": 1,
        "event": transition.event,
        "update": update,
        "from_stage": transition.from_stage,
        "to_stage": transition.to_stage,
        "curriculum_schedule": schedule.id(),
        "win_rate": transition.win_rate,
        "conversion_rate": transition.conversion_rate,
        "death_rate": transition.death_rate,
        "window_size": transition.window_size,
        "win_gate": transition.win_gate,
        "from": {
            "maps": from.maps,
            "bots": from.bots,
            "nations": nations_label(&from.nations),
            "difficulty": from.difficulty,
            "decision_ticks": from.decision_ticks,
            "win_at": from.win_at,
        },
        "to": {
            "maps": to.maps,
            "bots": to.bots,
            "nations": nations_label(&to.nations),
            "difficulty": to.difficulty,
            "decision_ticks": to.decision_ticks,
            "win_at": to.win_at,
        },
        "summary": summary,
    })
}

fn curriculum_milestone_path(
    ckpt_dir: &str,
    transition: &CurriculumTransition,
    update: u64,
) -> String {
    format!(
        "{}/curriculum_{}_u{}_s{}_to_{}.safetensors",
        ckpt_dir, transition.event, update, transition.from_stage, transition.to_stage,
    )
}

fn policy_manifest_value(
    gc: i64,
    blocks: i64,
    recurrent_policy: bool,
    recurrent_hidden_size: usize,
    bptt_chunk_len: usize,
    rollout_len: usize,
    ae_ckpt: &str,
    coarse_ckpt: Option<&str>,
    state: &TrainState,
) -> serde_json::Value {
    let mut architecture = serde_json::json!({
        "name": "oftrain-policy",
        "schema_version": 1,
        "dimensions": {
            "grid_channels": policy::C_GRID,
            "fine_grid_channels": policy::C_GRID_FINE,
            "player_features": policy::P_FEAT,
            "scalars": policy::N_SCALARS,
            "local_planes": policy::N_LOCAL,
            "actions": policy::N_ACTIONS,
            "build_types": policy::N_BUILD,
            "nuke_types": policy::N_NUKE,
            "quantity_params": 2,
            "grid_tower_channels": gc,
            "grid_tower_blocks": blocks,
            "hidden": policy::HIDDEN,
            "player_hidden": policy::PC,
            "local_hidden": policy::LC,
            "transformer_layers": policy::TF_LAYERS,
            "attention_heads": policy::N_HEAD,
        },
    });
    if recurrent_policy {
        architecture["schema_version"] = 2.into();
        architecture["recurrent"] = serde_json::json!({
            "cell": "gru",
            "hidden_size": recurrent_hidden_size,
            "context_schema": policy::RECURRENT_CONTEXT_SCHEMA,
            "context_features": policy::RECURRENT_CONTEXT_FLOATS,
            "context_embedding": policy::RECURRENT_CONTEXT_EMBEDDED,
            "bptt_length": bptt_chunk_len,
            "rollout_length": rollout_len,
            "residual_initialization": "zero-output-projection",
            "hidden_reset_policy": "episode_done",
        });
    }
    serde_json::json!({
        "format": "oftrain-safetensors",
        "manifest_schema_version": 1,
        "architecture": architecture,
        "autoencoders": {
            "fine": {"ref": ae_ckpt, "format": "safetensors"},
            "coarse": coarse_ckpt.map(|reference| {
                serde_json::json!({"ref": reference, "format": "safetensors"})
            }),
        },
        "checkpoint": {
            "weights": "latest.safetensors",
            "state": "latest.state.json",
        },
        "update": state.update,
        "stage": state.stage,
        "curriculum_schedule": state.schedule().map(|schedule| schedule.id()).unwrap_or("unknown"),
        "reward_profile": state.reward_profile,
    })
}

fn save_policy_manifest(cfg: &Config, state: &TrainState) -> Result<()> {
    let path = format!("{}/manifest.json", cfg.ckpt_dir);
    let manifest = policy_manifest_value(
        cfg.gc,
        cfg.blocks,
        cfg.recurrent_policy,
        cfg.recurrent_hidden_size,
        cfg.bptt_chunk_len,
        cfg.rollout_len,
        &cfg.ae_ckpt,
        cfg.coarse_ckpt.as_deref(),
        state,
    );
    save_atomic(&path, |tmp| {
        std::fs::write(tmp, serde_json::to_string_pretty(&manifest)?)?;
        Ok(())
    })
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
    /// Legacy fixed-group collectors receive directly from each worker.
    /// Ready-queue persistent actors instead use ActorShard::ready_rx.
    obs_rx: Option<Receiver<EnvStepResult>>,
    handle: JoinHandle<()>,
}

type EnvStepResult = Result<EnvTransition, String>;

struct ReadyEnv {
    env: usize,
    result: EnvStepResult,
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
    reward_config: RewardConfig,
    curriculum_schedule: ofcore::curriculum::CurriculumSchedule,
) -> Result<(Worker, PreparedObs)> {
    spawn_worker_routed(
        idx,
        stage,
        max_ticks,
        engine,
        reward_config,
        curriculum_schedule,
        None,
    )
}

fn spawn_worker_routed(
    idx: usize,
    stage: usize,
    max_ticks: i64,
    engine: EngineKind,
    reward_config: RewardConfig,
    curriculum_schedule: ofcore::curriculum::CurriculumSchedule,
    ready_route: Option<(usize, Sender<ReadyEnv>)>,
) -> Result<(Worker, PreparedObs)> {
    let routed = ready_route.is_some();
    let (choice_tx, choice_rx) = mpsc::channel::<Choice>();
    let (stage_tx, stage_rx) = mpsc::channel::<usize>();
    let (obs_tx, obs_rx) = mpsc::channel();
    let (init_tx, init_rx) = mpsc::channel::<Result<PreparedObs, String>>();
    let handle = std::thread::Builder::new()
        .name(format!("env{idx}"))
        .spawn(move || {
            let mut w = match EnvWorker::new(
                idx,
                stage,
                max_ticks,
                engine,
                reward_config,
                curriculum_schedule,
            ) {
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
                let sent = match &ready_route {
                    Some((env, ready_tx)) => ready_tx.send(ReadyEnv { env: *env, result }).is_ok(),
                    None => obs_tx.send(result).is_ok(),
                };
                if !sent {
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
            obs_rx: (!routed).then_some(obs_rx),
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
    /// CPU copy of the actor's recurrent input row, for learner BPTT.
    hidden_in: Vec<f32>,
    /// Previous action/result consumed with `obs`, stored explicitly so the
    /// learner does not need to infer temporal alignment.
    context: ActionOutcome,
    /// Outcome produced by this step (including terminal actions).
    outcome: ActionOutcome,
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
    /// One CPU-only completion queue shared by gated persistent env workers.
    ready_rx: Option<Receiver<ReadyEnv>>,
    /// CPU-only compact payloads are recycled after the learner drops the
    /// final immutable Step view. CUDA tensors never enter this arena.
    compact_host_arena: Arc<crate::vecenv::CompactHostArena>,
    /// Libtorch's CUDA current stream is thread-local. The persistent actor
    /// creates, uses, and drops this VarStore and every tensor derived from it
    /// on its one owner thread; no policy/AE tensor is sent through a channel.
    vs: nn::VarStore,
    policy: PolicyNet,
    /// Present only for --recurrent-policy persistent actors. Device tensor
    /// lifetime is wholly contained by the actor owner thread.
    recurrent: Option<ActorRecurrentState>,
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
    actor_batches: ActorBatchStats,
}

#[derive(Clone, Debug, Default)]
struct ActorBatchStats {
    dispatches: usize,
    observations: usize,
    singletons: usize,
    shape_dispatches: usize,
    padded_cells: usize,
    allocated_cells: usize,
}

impl ActorBatchStats {
    fn observe(&mut self, envs: &[usize], observations: &[PreparedObs]) {
        let mut shapes = Vec::new();
        let mut max_h = 0usize;
        let mut max_w = 0usize;
        let mut native = 0usize;
        for &env in envs {
            let shape = ActorShape::of(&observations[env]);
            if !shapes.contains(&shape) {
                shapes.push(shape);
            }
            max_h = max_h.max(shape.cgh);
            max_w = max_w.max(shape.cgw);
            native += shape.cgh * shape.cgw;
        }
        let allocated = envs.len() * max_h * max_w;
        self.dispatches += 1;
        self.observations += envs.len();
        self.singletons += usize::from(envs.len() == 1);
        self.shape_dispatches += shapes.len();
        self.padded_cells += allocated.saturating_sub(native);
        self.allocated_cells += allocated;
    }

    fn mean_size(&self) -> f64 {
        self.observations as f64 / self.dispatches.max(1) as f64
    }
    fn singleton_fraction(&self) -> f64 {
        self.singletons as f64 / self.dispatches.max(1) as f64
    }
    fn shapes_per_dispatch(&self) -> f64 {
        self.shape_dispatches as f64 / self.dispatches.max(1) as f64
    }
    fn padding_ratio(&self) -> f64 {
        self.padded_cells as f64 / self.allocated_cells.max(1) as f64
    }
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

#[derive(Debug)]
struct ReadyItem {
    env: usize,
    ready_at: Duration,
    sequence: u64,
}

#[derive(Debug)]
struct ReadyShapeBucket {
    shape: ActorShape,
    items: VecDeque<ReadyItem>,
}

/// Stable exact-shape ready queue. Shape buckets retain first-occurrence
/// order, while each bucket is FIFO by publication order.
#[derive(Debug)]
struct ReadyScheduler {
    buckets: Vec<ReadyShapeBucket>,
    target_batch: usize,
    max_batch: usize,
    max_padding_waste: f64,
    max_wait: Duration,
    next_sequence: u64,
}

impl ReadyScheduler {
    fn new(
        target_batch: usize,
        max_batch: usize,
        max_padding_waste: f64,
        max_wait: Duration,
    ) -> Self {
        let max_batch = max_batch.max(1);
        Self {
            buckets: Vec::new(),
            target_batch: target_batch.max(1).min(max_batch),
            max_batch,
            max_padding_waste: max_padding_waste.clamp(0.0, 1.0),
            max_wait,
            next_sequence: 0,
        }
    }

    fn push(&mut self, env: usize, shape: ActorShape, ready_at: Duration) {
        let item = ReadyItem {
            env,
            ready_at,
            sequence: self.next_sequence,
        };
        self.next_sequence += 1;
        if let Some(bucket) = self.buckets.iter_mut().find(|bucket| bucket.shape == shape) {
            bucket.items.push_back(item);
        } else {
            self.buckets.push(ReadyShapeBucket {
                shape,
                items: VecDeque::from([item]),
            });
        }
    }

    fn next_deadline(&self) -> Option<Duration> {
        self.buckets
            .iter()
            .filter_map(|bucket| bucket.items.front())
            .map(|item| item.ready_at.saturating_add(self.max_wait))
            .min()
    }

    fn take_batch(&mut self, now: Duration) -> Option<Vec<usize>> {
        let (oldest_bucket, oldest_ready_at) = self
            .buckets
            .iter()
            .enumerate()
            .filter_map(|(idx, bucket)| {
                bucket
                    .items
                    .front()
                    .map(|item| (idx, item.sequence, item.ready_at))
            })
            .min_by_key(|(_, sequence, _)| *sequence)
            .map(|(idx, _, ready_at)| (idx, ready_at))?;
        let same_ready = self.buckets[oldest_bucket].items.len();
        let waited_out = now.saturating_sub(oldest_ready_at) >= self.max_wait;
        // Wait for more peers of the oldest env's shape — not merely any
        // ready env — so mixed curriculum maps don't force singleton AE
        // dispatches the moment total_ready hits target_batch.
        if same_ready < self.target_batch && !waited_out {
            return None;
        }

        // Same-shape prefer: fill from the globally oldest env's shape bucket
        // first (zero padding, exact AE shapes). Mixed-shape coalescing is a
        // fallback only after max_wait when that bucket is still short of
        // target_batch — still subject to max_padding_waste. Inference
        // batching only; does not change obs or sampled action for any env.
        let mut selected_buckets = Vec::new();
        let same_shape_n = same_ready.min(self.max_batch);
        for _ in 0..same_shape_n {
            selected_buckets.push(oldest_bucket);
        }

        if waited_out && selected_buckets.len() < self.target_batch {
            let primary = self.buckets[oldest_bucket].shape;
            let mut sum_area = selected_buckets.len() * primary.cgh * primary.cgw;
            let mut max_h = primary.cgh;
            let mut max_w = primary.cgw;
            let mut taken_from: Vec<usize> = self.buckets.iter().map(|b| b.items.len()).collect();
            taken_from[oldest_bucket] -= same_shape_n;

            let mut others: Vec<(u64, usize, ActorShape)> = self
                .buckets
                .iter()
                .enumerate()
                .filter(|(idx, _)| *idx != oldest_bucket)
                .flat_map(|(bucket, values)| {
                    values
                        .items
                        .iter()
                        .map(move |item| (item.sequence, bucket, values.shape))
                })
                .collect();
            others.sort_unstable_by_key(|candidate| candidate.0);
            for &(_, bucket, shape) in &others {
                if selected_buckets.len() >= self.max_batch {
                    break;
                }
                if taken_from[bucket] == 0 {
                    continue;
                }
                let next_n = selected_buckets.len() + 1;
                let next_h = max_h.max(shape.cgh);
                let next_w = max_w.max(shape.cgw);
                let next_sum = sum_area + shape.cgh * shape.cgw;
                let allocated = next_n * next_h * next_w;
                let waste = 1.0 - next_sum as f64 / allocated.max(1) as f64;
                if waste > self.max_padding_waste {
                    continue;
                }
                selected_buckets.push(bucket);
                taken_from[bucket] -= 1;
                sum_area = next_sum;
                max_h = next_h;
                max_w = next_w;
            }
        }

        Some(
            selected_buckets
                .into_iter()
                .map(|bucket| self.buckets[bucket].items.pop_front().unwrap().env)
                .collect(),
        )
    }

    fn is_empty(&self) -> bool {
        self.buckets.iter().all(|bucket| bucket.items.is_empty())
    }
}

/// Fixed `(T, N)` storage filled in arbitrary completion order.
struct RolloutSlots<T> {
    rows: Vec<Vec<Option<T>>>,
    next_t: Vec<usize>,
}

impl<T> RolloutSlots<T> {
    fn new(t: usize, n: usize) -> Self {
        Self {
            rows: (0..t).map(|_| (0..n).map(|_| None).collect()).collect(),
            next_t: vec![0; n],
        }
    }

    fn insert(&mut self, t: usize, env: usize, value: T) -> Result<()> {
        anyhow::ensure!(env < self.next_t.len(), "rollout env {env} out of range");
        anyhow::ensure!(
            t == self.next_t[env],
            "rollout env {env} completed t={t}, expected t={}",
            self.next_t[env]
        );
        let slot = self
            .rows
            .get_mut(t)
            .and_then(|row| row.get_mut(env))
            .ok_or_else(|| anyhow!("rollout slot ({t}, {env}) out of range"))?;
        anyhow::ensure!(slot.is_none(), "duplicate rollout slot ({t}, {env})");
        *slot = Some(value);
        self.next_t[env] += 1;
        Ok(())
    }

    fn next_t(&self, env: usize) -> usize {
        self.next_t[env]
    }

    fn finish(self) -> Result<Vec<Vec<T>>> {
        self.rows
            .into_iter()
            .enumerate()
            .map(|(t, row)| {
                row.into_iter()
                    .enumerate()
                    .map(|(env, slot)| {
                        slot.ok_or_else(|| anyhow!("missing rollout slot ({t}, {env})"))
                    })
                    .collect()
            })
            .collect()
    }
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
    choice: Option<Choice>,
    scalars: ChoiceScalars,
    logp: f32,
    value: f32,
    hidden_in: Vec<f32>,
    context: ActionOutcome,
}

struct ActorBatchHost {
    packed: PackedActHost,
    hidden_in: Vec<Vec<f32>>,
}

fn act_contiguous_obs(
    actor: &mut ActorShard,
    cfg: &Config,
    start: usize,
    end: usize,
    envs: &[usize],
) -> Result<ActorBatchHost> {
    let n = end - start;
    anyhow::ensure!(envs.len() == n, "actor env identity width mismatch");
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
    let (action, hidden_in) = if let Some(state) = actor.recurrent.as_mut() {
        let hidden = state.gather(envs);
        let contexts: Vec<ActionOutcome> = actor.cur_obs[start..end]
            .iter()
            .map(|obs| obs.prev_action.clone())
            .collect();
        let context_t = crate::recurrent::context_tensor(&contexts, actor.device);
        let (action, hidden_out) = tch::no_grad(|| {
            actor
                .policy
                .act_with_state(&obs_t, &hidden, &context_t, false)
        });
        let hidden_cpu: Vec<f32> = hidden.to_device(Device::Cpu).reshape([-1]).try_into()?;
        let hidden_rows = hidden_cpu
            .chunks_exact(state.hidden_size())
            .map(<[f32]>::to_vec)
            .collect();
        state.scatter(envs, &hidden_out)?;
        (action, hidden_rows)
    } else {
        (
            tch::no_grad(|| actor.policy.act(&obs_t, false)),
            vec![Vec::new(); n],
        )
    };
    let (a, player, tile, build, nuke, qty, logp, value) = action;
    Ok(ActorBatchHost {
        packed: transfer_act_results(&a, &player, &tile, &build, &nuke, &qty, &logp, &value, n)?,
        hidden_in,
    })
}

/// Act on arbitrary ready envs without cloning their large host payloads.
/// The observations are temporarily packed into a contiguous prefix, restored
/// even on inference failure, and only CPU scalar choices leave this function.
fn act_ready_batch(actor: &mut ActorShard, cfg: &Config, envs: &[usize]) -> Result<Vec<ActorRow>> {
    anyhow::ensure!(!envs.is_empty(), "cannot act on an empty ready batch");
    let n = actor.cur_obs.len();
    let mut selected = vec![false; n];
    for &env in envs {
        anyhow::ensure!(env < n, "ready env {env} out of range");
        anyhow::ensure!(!selected[env], "duplicate ready env {env}");
        selected[env] = true;
    }
    let mut order = Vec::with_capacity(n);
    order.extend_from_slice(envs);
    order.extend((0..n).filter(|&env| !selected[env]));
    let inverse = inverse_permutation(&order);
    permute_slice(&mut actor.cur_obs, &order);
    let packed = act_contiguous_obs(actor, cfg, 0, envs.len(), envs);
    permute_slice(&mut actor.cur_obs, &inverse);
    let packed = packed?;

    let mut rows = Vec::with_capacity(envs.len());
    for row in 0..envs.len() {
        let (discrete, floats) = packed.packed.row(row)?;
        let (choice, scalars) = choice_from_act_values(
            discrete[0],
            discrete[1],
            discrete[2],
            discrete[3],
            discrete[4],
            floats[0],
        );
        rows.push(ActorRow {
            choice: Some(choice),
            scalars,
            logp: floats[1],
            value: floats[2],
            hidden_in: packed.hidden_in[row].clone(),
            context: actor.cur_obs[envs[row]].prev_action.clone(),
        });
    }
    Ok(rows)
}

/// Encode + act for a contiguous worker slice `[start, end)`. Returns
/// per-env choice scalars / logp / value aligned to the slice (length
/// `end - start`).
fn act_group(
    actor: &mut ActorShard,
    cfg: &Config,
    start: usize,
    end: usize,
) -> Result<Vec<ActorRow>> {
    let n = end - start;
    if n == 0 {
        return Ok(Vec::new());
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
                let identities: Vec<usize> = buckets.order[range.clone()]
                    .iter()
                    .map(|&original| start + original)
                    .collect();
                let packed = act_contiguous_obs(
                    actor,
                    cfg,
                    start + range.start,
                    start + range.end,
                    &identities,
                )?;
                for bucket_row in 0..range.len() {
                    let original = buckets.order[range.start + bucket_row];
                    let (discrete, floats) = packed.packed.row(bucket_row)?;
                    let (choice, scalars) = choice_from_act_values(
                        discrete[0],
                        discrete[1],
                        discrete[2],
                        discrete[3],
                        discrete[4],
                        floats[0],
                    );
                    rows[original] = Some(ActorRow {
                        choice: Some(choice),
                        scalars,
                        logp: floats[1],
                        value: floats[2],
                        hidden_in: packed.hidden_in[bucket_row].clone(),
                        context: actor.cur_obs[start + range.start + bucket_row]
                            .prev_action
                            .clone(),
                    });
                }
            }
            Ok(())
        })();
        permute_slice(&mut actor.cur_obs[start..end], &inverse);
        result?;
    } else {
        let identities: Vec<usize> = (start..end).collect();
        let packed = act_contiguous_obs(actor, cfg, start, end, &identities)?;
        for (i, row) in rows.iter_mut().enumerate() {
            let (discrete, floats) = packed.packed.row(i)?;
            let (choice, scalars) = choice_from_act_values(
                discrete[0],
                discrete[1],
                discrete[2],
                discrete[3],
                discrete[4],
                floats[0],
            );
            *row = Some(ActorRow {
                choice: Some(choice),
                scalars,
                logp: floats[1],
                value: floats[2],
                hidden_in: packed.hidden_in[i].clone(),
                context: actor.cur_obs[start + i].prev_action.clone(),
            });
        }
    }

    let mut completed = Vec::with_capacity(n);
    for (i, row) in rows.into_iter().enumerate() {
        let mut row = row.ok_or_else(|| anyhow!("missing actor result for env {}", start + i))?;
        actor.workers[start + i]
            .choice_tx
            .send(row.choice.take().expect("unsent actor choice"))
            .map_err(|_| anyhow!("env {} choice channel closed", start + i))?;
        completed.push(row);
    }
    Ok(completed)
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
            prev_action: ActionOutcome::default(),
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
            legal_ptarget: vec![1.0; ofcore::feat::N_ACTIONS * ofcore::feat::MAX_SLOTS],
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
            obs.iter()
                .map(|item| item.scalars[0] as usize)
                .collect::<Vec<_>>(),
            buckets.order
        );
        permute_slice(&mut obs, &inverse_permutation(&buckets.order));
        assert_eq!(
            obs.iter()
                .map(|item| item.scalars[0] as usize)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
    }

    #[test]
    fn ready_scheduler_prefers_same_shape_over_mixed_fifo() {
        let shape_a = ActorShape::of(&shaped_obs(0, 6, 8));
        let shape_b = ActorShape::of(&shaped_obs(1, 8, 6));
        let mut ready = ReadyScheduler::new(2, 2, 1.0, Duration::from_millis(5));
        ready.push(0, shape_a, Duration::ZERO);
        ready.push(1, shape_b, Duration::ZERO);
        ready.push(2, shape_a, Duration::from_millis(1));
        ready.push(3, shape_b, Duration::from_millis(1));
        // Oldest is shape_a: pack both a's before touching b's.
        assert_eq!(ready.take_batch(Duration::from_millis(1)), Some(vec![0, 2]));
        assert_eq!(ready.take_batch(Duration::from_millis(1)), Some(vec![1, 3]));

        ready.push(4, shape_b, Duration::from_millis(2));
        assert_eq!(ready.take_batch(Duration::from_millis(6)), None);
        assert_eq!(ready.next_deadline(), Some(Duration::from_millis(7)));
        assert_eq!(ready.take_batch(Duration::from_millis(7)), Some(vec![4]));
        assert!(ready.is_empty());
    }

    #[test]
    fn ready_scheduler_padding_bound_splits_fifo_without_bypass() {
        let small = ActorShape::of(&shaped_obs(0, 4, 4));
        let large = ActorShape::of(&shaped_obs(1, 20, 20));
        let mut ready = ReadyScheduler::new(2, 8, 0.20, Duration::from_millis(5));
        ready.push(7, small, Duration::ZERO);
        ready.push(8, large, Duration::from_millis(1));
        ready.push(9, large, Duration::from_millis(2));

        // Before wait expires, same-shape count < target → hold (don't fire a
        // singleton just because other shapes are ready).
        assert_eq!(ready.take_batch(Duration::from_millis(2)), None);
        // After wait: oldest small dispatches alone; larges pack next.
        assert_eq!(ready.take_batch(Duration::from_millis(5)), Some(vec![7]));
        assert_eq!(ready.take_batch(Duration::from_millis(5)), Some(vec![8, 9]));
        assert!(ready.is_empty());
    }

    #[test]
    fn ready_scheduler_mixed_shape_fallback_only_after_wait() {
        // compact areas 2x2=4 and 2x3=6 → waste = 1-10/12 ≈ 0.167 < 0.25.
        let small = ActorShape::of(&shaped_obs(0, 4, 4));
        let large = ActorShape::of(&shaped_obs(1, 4, 5));
        let mut ready = ReadyScheduler::new(2, 8, 0.25, Duration::from_millis(10));
        ready.push(0, small, Duration::ZERO);
        ready.push(1, large, Duration::from_millis(1));
        // Before wait: hold for same-shape peers even though another shape is ready.
        assert_eq!(ready.take_batch(Duration::from_millis(5)), None);
        // After wait: same-shape first (small), then mix in large to hit target.
        assert_eq!(
            ready.take_batch(Duration::from_millis(10)),
            Some(vec![0, 1])
        );
        assert!(ready.is_empty());
    }

    #[test]
    fn rollout_slots_are_t_major_complete_despite_out_of_order_envs() {
        let mut slots = RolloutSlots::new(3, 3);
        for &(t, env) in &[
            (0, 2),
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 2),
            (2, 2),
            (1, 1),
            (2, 0),
            (2, 1),
        ] {
            slots.insert(t, env, (t, env)).unwrap();
        }
        assert_eq!(
            slots.finish().unwrap(),
            vec![
                vec![(0, 0), (0, 1), (0, 2)],
                vec![(1, 0), (1, 1), (1, 2)],
                vec![(2, 0), (2, 1), (2, 2)],
            ]
        );
    }

    #[test]
    fn recurrent_step_payloads_remain_t_major() {
        let make_step = |t: usize, env: usize| {
            let mut context = ActionOutcome::default();
            context.action = (t * 10 + env) as i64;
            let mut outcome = context.clone();
            outcome.success = true;
            Step {
                obs: shaped_obs(env, 2, 2),
                hidden_in: vec![env as f32, t as f32],
                context,
                outcome,
                choice: ChoiceScalars {
                    action: 0,
                    player_slot: -1,
                    tile_region: -1,
                    build_type: -1,
                    nuke_type: -1,
                    quantity_frac: -1.0,
                },
                logp: 0.0,
                value: 0.0,
                reward: 0.0,
                done: false,
            }
        };
        let mut slots = RolloutSlots::new(2, 3);
        for &(t, env) in &[(0, 2), (0, 0), (0, 1), (1, 1), (1, 2), (1, 0)] {
            slots.insert(t, env, make_step(t, env)).unwrap();
        }
        let rows = slots.finish().unwrap();
        for (t, row) in rows.iter().enumerate() {
            for (env, step) in row.iter().enumerate() {
                assert_eq!(step.hidden_in, vec![env as f32, t as f32]);
                assert_eq!(step.context.action, (t * 10 + env) as i64);
                assert!(step.outcome.success);
            }
        }
    }

    #[test]
    fn staggered_mock_envs_do_not_block_fast_ready_batches() {
        const T: usize = 3;
        let shape = ActorShape::of(&shaped_obs(0, 6, 8));
        let latency = [1u64, 1, 100];
        let mut ready = ReadyScheduler::new(2, 2, 1.0, Duration::from_millis(10));
        for env in 0..latency.len() {
            ready.push(env, shape, Duration::ZERO);
        }
        let mut slots = RolloutSlots::new(T, latency.len());
        let mut pending = vec![None; latency.len()];
        let mut events: Vec<(u64, usize)> = Vec::new();
        let mut dispatches: Vec<(u64, Vec<usize>)> = Vec::new();
        let mut now = 0u64;
        let mut completed = 0;

        while completed < T * latency.len() {
            while let Some(batch) = ready.take_batch(Duration::from_millis(now)) {
                for &env in &batch {
                    let t = slots.next_t(env);
                    assert!(pending[env].replace(t).is_none());
                    events.push((now + latency[env], env));
                }
                dispatches.push((now, batch));
            }
            if completed == T * latency.len() {
                break;
            }
            let event_time = events.iter().map(|event| event.0).min();
            let deadline = ready
                .next_deadline()
                .map(|deadline| deadline.as_millis() as u64);
            now = match (event_time, deadline) {
                (Some(event), Some(deadline)) => event.min(deadline),
                (Some(event), None) => event,
                (None, Some(deadline)) => deadline,
                (None, None) => panic!("mock scheduler deadlocked"),
            };
            let mut arrived: Vec<usize> = events
                .iter()
                .filter(|event| event.0 == now)
                .map(|event| event.1)
                .collect();
            events.retain(|event| event.0 != now);
            arrived.sort_unstable();
            for env in arrived {
                let t = pending[env].take().unwrap();
                slots.insert(t, env, (t, env)).unwrap();
                completed += 1;
                if t + 1 < T {
                    ready.push(env, shape, Duration::from_millis(now));
                }
            }
        }

        assert!(
            dispatches
                .iter()
                .any(|(at, batch)| *at < 100 && batch.iter().all(|&env| env != 2)),
            "fast envs never formed another GPU batch before the slow env returned: {dispatches:?}"
        );
        let rows = slots.finish().unwrap();
        for (t, row) in rows.iter().enumerate() {
            assert_eq!(row, &vec![(t, 0), (t, 1), (t, 2)]);
        }
    }

    #[test]
    fn staggered_ready_batches_gather_hidden_by_env_not_batch_position() {
        let shape = ActorShape::of(&shaped_obs(0, 6, 8));
        let mut ready = ReadyScheduler::new(2, 2, 1.0, Duration::ZERO);
        let mut state = ActorRecurrentState::new(3, 2, Device::Cpu);

        ready.push(2, shape, Duration::ZERO);
        ready.push(0, shape, Duration::ZERO);
        let first = ready.take_batch(Duration::ZERO).unwrap();
        assert_eq!(first, vec![2, 0]);
        let first_in = state.gather(&first);
        let first_delta = Tensor::from_slice(&[20.0f32, 20.0, 10.0, 10.0]).view([2, 2]);
        state.scatter(&first, &(first_in + first_delta)).unwrap();

        // Env 2 now appears in a different row alongside a previously unseen
        // env. Its row must start from 20, while env 1 starts from zero.
        ready.push(1, shape, Duration::from_millis(1));
        ready.push(2, shape, Duration::from_millis(1));
        let second = ready.take_batch(Duration::from_millis(1)).unwrap();
        assert_eq!(second, vec![1, 2]);
        let second_in: Vec<f32> = state.gather(&second).reshape([-1]).try_into().unwrap();
        assert_eq!(second_in, vec![0.0, 0.0, 20.0, 20.0]);
    }

    fn policy_rows(
        policy: &PolicyNet,
        items: &[PreparedObs],
        greedy: bool,
    ) -> Vec<(Vec<i64>, Vec<f32>)> {
        let buckets = actor_shape_buckets(items);
        let mut rows: Vec<Option<(Vec<i64>, Vec<f32>)>> = (0..items.len()).map(|_| None).collect();
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
                choice_from_act_values(row.0[0], row.0[1], row.0[2], row.0[3], row.0[4], row.1[0]).1
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
            for (field, (&actual, &expected)) in
                bucketed[i].1[1..].iter().zip(&singleton.1[1..]).enumerate()
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
    rows: &[ActorRow],
    step_row: &mut [Option<Step>],
    ep_infos: &mut Vec<EpisodeInfo>,
) -> Result<()> {
    for (j, i) in (start..end).enumerate() {
        let transition = actor.workers[i]
            .obs_rx
            .as_ref()
            .ok_or_else(|| anyhow!("env {i} has no legacy obs receiver"))?
            .recv()
            .map_err(|_| anyhow!("env {i} obs channel closed"))?
            .map_err(|e| anyhow!("env {i}: {e}"))?;
        if let Some(info) = transition.info {
            ep_infos.push(info);
        }
        let prev_obs = std::mem::replace(&mut actor.cur_obs[i], transition.next_obs);
        if transition.done {
            if let Some(state) = actor.recurrent.as_mut() {
                state.reset(i)?;
            }
        }
        step_row[i] = Some(Step {
            obs: prev_obs,
            hidden_in: rows[j].hidden_in.clone(),
            context: rows[j].context.clone(),
            outcome: transition.outcome,
            choice: rows[j].scalars.clone(),
            logp: rows[j].logp,
            value: rows[j].value,
            reward: transition.reward as f32,
            done: transition.done,
        });
    }
    Ok(())
}

fn actor_value_rows(actor: &ActorShard, obs: &policy::Obs, envs: &[usize]) -> Result<Tensor> {
    if let Some(state) = actor.recurrent.as_ref() {
        let hidden = state.gather(envs);
        let contexts: Vec<ActionOutcome> = envs
            .iter()
            .map(|&env| actor.cur_obs[env].prev_action.clone())
            .collect();
        let context = crate::recurrent::context_tensor(&contexts, actor.device);
        Ok(tch::no_grad(|| {
            actor.policy.value_with_state(obs, &hidden, &context).0
        }))
    } else {
        Ok(tch::no_grad(|| actor.policy.value_only(obs)))
    }
}

fn bootstrap_values(actor: &mut ActorShard, cfg: &Config) -> Result<Vec<f32>> {
    let n = actor.cur_obs.len();
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
            let envs = &buckets.order[range.clone()];
            let bucket_values: Vec<f32> = (&actor_value_rows(actor, &obs_t, envs)?).try_into()?;
            anyhow::ensure!(
                bucket_values.len() == range.len(),
                "bootstrap bucket width mismatch"
            );
            for (bucket_row, value) in bucket_values.into_iter().enumerate() {
                values[buckets.order[range.start + bucket_row]] = value;
            }
        }
        Ok(values)
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
        let envs: Vec<usize> = (0..n).collect();
        let v = actor_value_rows(actor, &obs_t, &envs)?;
        Ok((&v).try_into()?)
    }
}

struct PendingAct {
    t: usize,
    scalars: ChoiceScalars,
    logp: f32,
    value: f32,
    hidden_in: Vec<f32>,
    context: ActionOutcome,
}

fn collect_rollout_ready(
    actor: &mut ActorShard,
    cfg: &Config,
    policy_version: u64,
) -> Result<RolloutResult> {
    let collect_start = Instant::now();
    let n = actor.workers.len();
    anyhow::ensure!(n > 0, "ready collector has no envs");
    let mut scheduler = ReadyScheduler::new(
        cfg.actor_target_batch,
        cfg.actor_max_batch,
        cfg.actor_max_padding_waste,
        cfg.actor_max_wait,
    );
    for env in 0..n {
        scheduler.push(env, ActorShape::of(&actor.cur_obs[env]), Duration::ZERO);
    }
    let mut pending: Vec<Option<PendingAct>> = (0..n).map(|_| None).collect();
    let mut slots = RolloutSlots::new(cfg.rollout_len, n);
    let mut episode_slots = RolloutSlots::new(cfg.rollout_len, n);
    let target = cfg.rollout_len * n;
    let mut completed = 0usize;
    let mut actor_batches = ActorBatchStats::default();

    while completed < target {
        while let Some(envs) = scheduler.take_batch(collect_start.elapsed()) {
            actor_batches.observe(&envs, &actor.cur_obs);
            let rows = act_ready_batch(actor, cfg, &envs)?;
            anyhow::ensure!(rows.len() == envs.len(), "ready act batch width mismatch");
            for (env, mut row) in envs.into_iter().zip(rows) {
                anyhow::ensure!(
                    pending[env].is_none(),
                    "env {env} already has an action in flight"
                );
                let t = slots.next_t(env);
                anyhow::ensure!(t < cfg.rollout_len, "env {env} exceeded rollout length");
                actor.workers[env]
                    .choice_tx
                    .send(row.choice.take().expect("unsent actor choice"))
                    .map_err(|_| anyhow!("env {env} choice channel closed"))?;
                pending[env] = Some(PendingAct {
                    t,
                    scalars: row.scalars,
                    logp: row.logp,
                    value: row.value,
                    hidden_in: row.hidden_in,
                    context: row.context,
                });
            }
        }
        if completed == target {
            break;
        }

        let now = collect_start.elapsed();
        let timeout = scheduler
            .next_deadline()
            .map(|deadline| deadline.saturating_sub(now));
        let ready_rx = actor
            .ready_rx
            .as_ref()
            .ok_or_else(|| anyhow!("work-conserving actor has no ready receiver"))?;
        let message = match timeout {
            Some(duration) => match ready_rx.recv_timeout(duration) {
                Ok(message) => message,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    return Err(anyhow!("ready env channel closed"));
                }
            },
            None => ready_rx
                .recv()
                .map_err(|_| anyhow!("ready env channel closed"))?,
        };
        anyhow::ensure!(message.env < n, "ready env {} out of range", message.env);
        let env = message.env;
        let action = pending[env]
            .take()
            .ok_or_else(|| anyhow!("env {env} published without an action in flight"))?;
        let transition = message
            .result
            .map_err(|error| anyhow!("env {env}: {error}"))?;
        let prev_obs = std::mem::replace(&mut actor.cur_obs[env], transition.next_obs);
        if transition.done {
            if let Some(state) = actor.recurrent.as_mut() {
                state.reset(env)?;
            }
        }
        episode_slots.insert(action.t, env, transition.info)?;
        slots.insert(
            action.t,
            env,
            Step {
                obs: prev_obs,
                hidden_in: action.hidden_in,
                context: action.context,
                outcome: transition.outcome,
                choice: action.scalars,
                logp: action.logp,
                value: action.value,
                reward: transition.reward as f32,
                done: transition.done,
            },
        )?;
        completed += 1;
        if slots.next_t(env) < cfg.rollout_len {
            scheduler.push(
                env,
                ActorShape::of(&actor.cur_obs[env]),
                collect_start.elapsed(),
            );
        }
    }
    anyhow::ensure!(
        scheduler.is_empty(),
        "ready items remain after rollout completion"
    );
    anyhow::ensure!(
        pending.iter().all(Option::is_none),
        "actions remain in flight after rollout completion"
    );
    let buffer = slots.finish()?;
    let ep_infos = episode_slots
        .finish()?
        .into_iter()
        .flatten()
        .flatten()
        .collect();
    let bootstrap_v = bootstrap_values(actor, cfg)?;
    if let Device::Cuda(index) = actor.device {
        Cuda::synchronize(index as i64);
    }
    eprintln!(
        "[actor-batch] mean_size={:.2} singleton_fraction={:.3} \
         shapes_per_dispatch={:.2} dispatches={} padding_ratio={:.3}",
        actor_batches.mean_size(),
        actor_batches.singleton_fraction(),
        actor_batches.shapes_per_dispatch(),
        actor_batches.dispatches,
        actor_batches.padding_ratio(),
    );
    Ok(RolloutResult {
        buffer,
        bootstrap_v,
        ep_infos,
        policy_version,
        collect_seconds: collect_start.elapsed().as_secs_f64(),
        actor_batches,
    })
}

fn collect_rollout(
    actor: &mut ActorShard,
    cfg: &Config,
    policy_version: u64,
) -> Result<RolloutResult> {
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
    let mut pack0 = act_group(actor, cfg, g0.0, g0.1)?;

    for t in 0..cfg.rollout_len {
        let mut step_row: Vec<Option<Step>> = (0..n).map(|_| None).collect();

        let mut pack1: Option<Vec<ActorRow>> = None;
        if g1.0 < g1.1 {
            // Overlaps group 0 stepping (choices already in flight).
            pack1 = Some(act_group(actor, cfg, g1.0, g1.1)?);
        }

        recv_group(actor, g0.0, g0.1, &pack0, &mut step_row, &mut ep_infos)?;

        if t + 1 < cfg.rollout_len {
            // Next-step act for g0 overlaps g1 stepping when pack1 is live.
            let next = act_group(actor, cfg, g0.0, g0.1)?;
            pack0 = next;
        }

        if let Some(rows) = pack1.as_ref() {
            recv_group(actor, g1.0, g1.1, rows, &mut step_row, &mut ep_infos)?;
        }

        buffer.push(
            step_row
                .into_iter()
                .map(|s| s.expect("every env must produce a step"))
                .collect(),
        );
    }

    let bootstrap_v = bootstrap_values(actor, cfg)?;

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
        actor_batches: ActorBatchStats::default(),
    })
}

/// Result of a fixed-seed greedy eval pass (`run_eval`).
pub struct EvalResult {
    pub win: f64,
    pub score: f64,
    pub episodes: usize,
    pub details: Vec<(usize, EpisodeInfo)>,
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
    recurrent_policy: bool,
    reward_config: RewardConfig,
    curriculum_schedule: ofcore::curriculum::CurriculumSchedule,
) -> Result<EvalResult> {
    if episodes == 0 {
        return Ok(EvalResult {
            win: 0.0,
            score: 0.0,
            episodes: 0,
            details: Vec::new(),
        });
    }
    let mut workers = Vec::with_capacity(episodes);
    let mut cur_obs = Vec::with_capacity(episodes);
    for i in 0..episodes {
        let (w, obs) = spawn_worker(
            i,
            stage,
            max_ticks,
            engine,
            reward_config,
            curriculum_schedule,
        )?;
        workers.push(w);
        cur_obs.push(obs);
    }
    let stages = ofcore::curriculum::stages_for_schedule(curriculum_schedule);
    let decision_ticks = stages[stage.min(stages.len().saturating_sub(1))]
        .decision_ticks
        .max(1) as u64;
    let step_cap = (max_ticks as u64 / decision_ticks) + 64;
    let mut results: Vec<Option<EpisodeInfo>> = vec![None; episodes];
    let mut terrain_cache = crate::ae::TerrainDeviceCache::new(device);
    let mut recurrent = recurrent_policy
        .then(|| ActorRecurrentState::new(episodes, policy::RECURRENT_HIDDEN as usize, device));

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
            if let Some(state) = recurrent.as_mut() {
                let hidden = state.gather(&pending);
                let contexts: Vec<ActionOutcome> = pending
                    .iter()
                    .map(|&env| cur_obs[env].prev_action.clone())
                    .collect();
                let context = crate::recurrent::context_tensor(&contexts, device);
                let (action, hidden_out) =
                    tch::no_grad(|| policy.act_with_state(&obs_t, &hidden, &context, true));
                state.scatter(&pending, &hidden_out)?;
                action
            } else {
                tch::no_grad(|| policy.act(&obs_t, true))
            };
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
            let transition = workers[ei]
                .obs_rx
                .as_ref()
                .ok_or_else(|| anyhow!("eval env {ei} has no obs receiver"))?
                .recv()
                .map_err(|_| anyhow!("eval env {ei} obs channel closed"))?
                .map_err(|e| anyhow!("eval env {ei}: {e}"))?;
            cur_obs[ei] = transition.next_obs;
            if let Some(info) = transition.info {
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
            details: Vec::new(),
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
        details: results
            .into_iter()
            .enumerate()
            .filter_map(|(index, info)| info.map(|info| (index, info)))
            .collect(),
    })
}

/// Minimal inference-only entry point used by the paired old-policy
/// benchmark. Loading into a fully constructed VarStore is intentionally
/// strict: a v5 (43-channel/14-action) or otherwise incompatible checkpoint
/// fails here instead of being partially loaded and mislabeled.
pub struct BenchmarkConfig<'a> {
    pub checkpoint: &'a str,
    pub output: &'a str,
    pub ae_ckpt: &'a str,
    pub coarse_ckpt: Option<&'a str>,
    pub stage: usize,
    pub episodes: usize,
    pub max_ticks: i64,
    pub engine: EngineKind,
    pub device: Device,
    pub amp: bool,
    pub foveate: bool,
    pub gc: i64,
    pub blocks: i64,
    pub recurrent_policy: bool,
    pub pinned_h2d: bool,
    pub fp16_rollout: bool,
    pub compact_rollout: bool,
    pub reward_config: RewardConfig,
    pub curriculum_schedule: ofcore::curriculum::CurriculumSchedule,
}

pub fn run_benchmark(cfg: BenchmarkConfig<'_>) -> Result<()> {
    let mut vs = nn::VarStore::new(cfg.device);
    let policy = PolicyNet::new_with_recurrence(
        &vs.root(),
        cfg.amp,
        cfg.foveate,
        cfg.gc,
        cfg.blocks,
        cfg.recurrent_policy,
    );
    vs.load(cfg.checkpoint)?;
    let ae = crate::ae::AePair::load(
        std::path::Path::new(cfg.ae_ckpt),
        cfg.coarse_ckpt.map(std::path::Path::new),
        cfg.device,
        cfg.amp,
        false,
    )?;
    let result = run_eval(
        &policy,
        &ae,
        cfg.device,
        cfg.stage,
        cfg.episodes,
        cfg.max_ticks,
        cfg.engine,
        cfg.pinned_h2d,
        cfg.fp16_rollout,
        cfg.compact_rollout,
        cfg.recurrent_policy,
        cfg.reward_config,
        cfg.curriculum_schedule,
    )?;
    let engine = match cfg.engine {
        EngineKind::Node => "node-ts",
        EngineKind::Native => "native-rust",
    };
    let stages = ofcore::curriculum::stages_for_schedule(cfg.curriculum_schedule);
    let stage_cfg = &stages[cfg.stage];
    let nations = match stage_cfg.nations {
        ofcore::curriculum::Nations::Default => serde_json::Value::String("default".into()),
        ofcore::curriculum::Nations::Exact(n) => serde_json::Value::from(n),
    };
    let episodes: Vec<_> = result
        .details
        .iter()
        .map(|(index, info)| {
            serde_json::json!({
                "index": index,
                "seed": format!("w{index}-ep0"),
                "map": info.map,
                "bots": stage_cfg.bots,
                "difficulty": stage_cfg.difficulty,
                "nations": nations.clone(),
                "decision_ticks": stage_cfg.decision_ticks,
                "won": info.won,
                "place": info.place,
                "n_players": info.n_players,
                "score": info.score,
                "final_tick": info.final_tick,
                "final_tiles": info.final_tiles,
                "final_land_share": info.final_land_share,
                "max_land_share": info.max_land_share,
                "closeout_reached": info.closeout_reached,
                "closeout_entry_tick": info.closeout_entry_tick,
                "decisions_after_closeout": info.decisions_after_closeout,
                "converted": info.converted,
                "timeout_after_closeout": info.timeout_after_closeout,
                "post_closeout_churn_pairs": info.post_closeout_churn_pairs,
                "reward_components": {
                    "strength": info.reward_components.strength,
                    "strength_delta": info.reward_components.strength_delta,
                    "dominance": info.reward_components.dominance,
                    "closeout": info.reward_components.closeout,
                    "action_churn": info.reward_components.action_churn,
                    "boat_outcome": info.reward_components.boat_outcome,
                    "tempo": info.reward_components.tempo,
                    "embargo_outcome": info.reward_components.embargo_outcome,
                    "combat_outcome": info.reward_components.combat_outcome,
                    "survival": info.reward_components.survival,
                    "diplo_panic": info.reward_components.diplo_panic,
                    "combat_action": info.reward_components.combat_action,
                    "waste": info.reward_components.waste,
                    "death": info.reward_components.death,
                    "terminal": info.reward_components.terminal,
                },
                "action_pair_counts": {
                    "boat_cancel_boat": info.action_pair_counts.boat_cancel_boat,
                    "embargo_embargo_stop": info.action_pair_counts.embargo_embargo_stop,
                    "attack_retreat": info.action_pair_counts.attack_retreat,
                    "retreat_attack": info.action_pair_counts.retreat_attack,
                    "total": info.action_pair_counts.total(),
                },
            })
        })
        .collect();
    let report = serde_json::json!({
        "format": 1,
        "mode": "indirect-scripted-bot",
        "runner": "rust",
        "checkpoint": cfg.checkpoint,
        "engine": engine,
        "stage": cfg.stage,
        "max_ticks": cfg.max_ticks,
        "schema": {
            "grid_channels": policy::C_GRID,
            "player_features": policy::P_FEAT,
            "scalars": policy::N_SCALARS,
            "local_planes": policy::N_LOCAL,
            "actions": policy::N_ACTIONS,
            "build_types": policy::N_BUILD,
            "nuke_types": policy::N_NUKE,
            "quantity": "beta",
        },
        "episodes": episodes,
    });
    std::fs::write(cfg.output, serde_json::to_vec_pretty(&report)?)?;
    println!(
        "[benchmark] wrote {} completed episodes to {}",
        result.episodes, cfg.output
    );
    Ok(())
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
    /// Shared immutable backing makes actor refresh and asynchronous eval
    /// submissions cheap clones rather than another full policy copy.
    meta: Arc<[CpuWeightMeta]>,
    values: Arc<[f32]>,
}

fn snapshot_weights(vs: &nn::VarStore) -> Result<CpuWeightSnapshot> {
    tch::no_grad(|| {
        let mut variables: Vec<_> = vs.variables().into_iter().collect();
        variables.sort_by(|(a, _), (b, _)| a.cmp(b));
        let meta: Vec<CpuWeightMeta> = variables
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
        Ok(CpuWeightSnapshot {
            meta: meta.into(),
            values: values.into(),
        })
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
    let cpu = Tensor::from_slice(snapshot.values.as_ref());
    let mut offset = 0i64;
    for item in snapshot.meta.iter() {
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
        // Snapshots are always f32 (learner). Actor inference weights may
        // already be BF16 (see `cast_actor_inference_weights_bf16`).
        let source = cpu
            .narrow(0, offset, item.len)
            .view(item.shape.as_slice())
            .to_device(destination.device())
            .to_kind(destination.kind());
        tch::no_grad(|| destination.f_copy_(&source))?;
        offset += item.len;
    }
    anyhow::ensure!(
        offset as usize == snapshot.values.len(),
        "weight snapshot payload length mismatch"
    );
    Ok(())
}

/// Conv towers / tile heads the actor runs under `--amp` via
/// `conv2d_bf16`. Storing them as BF16 in the actor VarStore halves that
/// shard of policy VRAM; the learner keeps f32 for Adam. One-step-lag
/// dual VarStores stay intentional — this only shrinks the actor copy.
fn actor_inference_weight_bf16(name: &str) -> bool {
    name.starts_with("grid_coarse.")
        || name.starts_with("grid_fine.")
        || name.starts_with("local.")
        || name.starts_with("htc1.")
        || name.starts_with("htc2.")
        || name.starts_with("htf1.")
        || name.starts_with("htf2.")
}

fn cast_actor_inference_weights_bf16(vs: &nn::VarStore) {
    tch::no_grad(|| {
        for (name, mut tensor) in vs.variables() {
            if actor_inference_weight_bf16(&name) && tensor.kind() != Kind::BFloat16 {
                let bf16 = tensor.to_kind(Kind::BFloat16);
                tensor.set_data(&bf16);
            }
        }
    });
}

#[derive(Clone)]
struct EvalReportContext {
    losses: (f64, f64, f64, f64),
    win_rate: Option<f64>,
    lr_now: f64,
    total_env_steps: u64,
}

struct EvalJob {
    update: u64,
    stage: usize,
    weights: CpuWeightSnapshot,
    state: TrainState,
    report: EvalReportContext,
}

struct EvalCompletion {
    update: u64,
    stage: usize,
    result: EvalResult,
    elapsed_seconds: f64,
    promoted: bool,
    best: Option<(f64, f64)>,
    report: EvalReportContext,
}

fn eval_is_better(candidate: &EvalResult, best: Option<(f64, f64)>) -> bool {
    if !candidate.win.is_finite() || !candidate.score.is_finite() {
        return false;
    }
    match best {
        None => true,
        Some((best_win, best_score)) => {
            candidate.win > best_win || (candidate.win == best_win && candidate.score > best_score)
        }
    }
}

fn best_eval_path(ckpt_dir: &str) -> String {
    format!("{ckpt_dir}/best_eval.safetensors")
}

fn save_snapshot_checkpoint(
    snapshot: &CpuWeightSnapshot,
    cfg: &Config,
    path: &str,
    state: &TrainState,
) -> Result<()> {
    // Materialize directly on CPU. The immutable snapshot remains owned by
    // the eval job and no training CUDA state or VarStore is touched.
    let vs = nn::VarStore::new(Device::Cpu);
    let _ = PolicyNet::new_with_recurrence(
        &vs.root(),
        cfg.amp,
        cfg.foveate,
        cfg.gc,
        cfg.blocks,
        cfg.recurrent_policy,
    );
    apply_weight_snapshot(&vs, snapshot)?;
    save_checkpoint(&vs, path, state)
}

enum EvalCommand {
    Run(EvalJob),
    Shutdown,
}

#[derive(Default)]
struct EvalFlight {
    update: Option<u64>,
}

impl EvalFlight {
    fn reserve(&mut self, update: u64) -> bool {
        if self.update.is_some() {
            false
        } else {
            self.update = Some(update);
            true
        }
    }

    fn complete(&mut self, update: u64) -> Result<()> {
        anyhow::ensure!(
            self.update == Some(update),
            "eval completion update {update} does not match in-flight {:?}",
            self.update
        );
        self.update = None;
        Ok(())
    }

    fn is_busy(&self) -> bool {
        self.update.is_some()
    }
}

struct AsyncEval {
    command_tx: mpsc::SyncSender<EvalCommand>,
    result_rx: Receiver<std::result::Result<EvalCompletion, String>>,
    handle: Option<JoinHandle<()>>,
    flight: EvalFlight,
}

impl AsyncEval {
    fn spawn(cfg: Config, device: Device, initial_best: Option<(f64, f64)>) -> Result<Self> {
        let (command_tx, command_rx) = mpsc::sync_channel(1);
        let (result_tx, result_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let handle = std::thread::Builder::new()
            .name("oftrain-async-eval".to_string())
            .spawn(move || {
                // Every CUDA-bearing eval object is created, used, and
                // destroyed on this owner thread.
                let initialized = (|| -> Result<(nn::VarStore, PolicyNet, crate::ae::AePair)> {
                    let vs = nn::VarStore::new(device);
                    let policy = PolicyNet::new_with_recurrence(
                        &vs.root(),
                        cfg.amp,
                        cfg.foveate,
                        cfg.gc,
                        cfg.blocks,
                        cfg.recurrent_policy,
                    );
                    let path = std::path::Path::new(&cfg.ae_ckpt);
                    anyhow::ensure!(
                        path.exists(),
                        "AE checkpoint not found at {}",
                        path.display()
                    );
                    let coarse = cfg.coarse_ckpt.as_ref().map(std::path::Path::new);
                    let ae = crate::ae::AePair::load(path, coarse, device, cfg.amp, true)?;
                    if let Device::Cuda(index) = device {
                        Cuda::synchronize(index as i64);
                    }
                    Ok((vs, policy, ae))
                })();
                let (vs, policy, ae) = match initialized {
                    Ok(resources) => {
                        let _ = ready_tx.send(Ok(()));
                        resources
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(format!("{error:#}")));
                        return;
                    }
                };
                let mut best = initial_best;
                while let Ok(command) = command_rx.recv() {
                    let EvalCommand::Run(mut job) = command else {
                        break;
                    };
                    let started = Instant::now();
                    let outcome = (|| -> Result<EvalCompletion> {
                        apply_weight_snapshot(&vs, &job.weights)?;
                        if let Device::Cuda(index) = device {
                            Cuda::synchronize(index as i64);
                        }
                        let result = run_eval(
                            &policy,
                            &ae,
                            device,
                            job.stage,
                            cfg.eval_episodes,
                            cfg.max_episode_ticks,
                            cfg.engine,
                            cfg.pinned_h2d,
                            cfg.fp16_rollout,
                            cfg.compact_rollout && cfg.foveate,
                            cfg.recurrent_policy,
                            cfg.reward_config,
                            cfg.curriculum_schedule,
                        )?;
                        let promoted = eval_is_better(&result, best);
                        if promoted {
                            best = Some((result.win, result.score));
                            job.state.best_eval_win = Some(result.win);
                            job.state.best_eval_score = Some(result.score);
                            save_snapshot_checkpoint(
                                &job.weights,
                                &cfg,
                                &best_eval_path(&cfg.ckpt_dir),
                                &job.state,
                            )?;
                        }
                        Ok(EvalCompletion {
                            update: job.update,
                            stage: job.stage,
                            result,
                            elapsed_seconds: started.elapsed().as_secs_f64(),
                            promoted,
                            best,
                            report: job.report,
                        })
                    })();
                    if result_tx
                        .send(outcome.map_err(|e| format!("{e:#}")))
                        .is_err()
                    {
                        break;
                    }
                }
                if let Device::Cuda(index) = device {
                    Cuda::synchronize(index as i64);
                }
            })?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                command_tx,
                result_rx,
                handle: Some(handle),
                flight: EvalFlight::default(),
            }),
            Ok(Err(error)) => {
                let _ = handle.join();
                Err(anyhow!("async eval initialization failed: {error}"))
            }
            Err(_) => {
                let _ = handle.join();
                Err(anyhow!("async eval thread exited during initialization"))
            }
        }
    }

    fn submit(&mut self, job: EvalJob) -> Result<bool> {
        if !self.flight.reserve(job.update) {
            return Ok(false);
        }
        let update = job.update;
        if let Err(error) = self.command_tx.try_send(EvalCommand::Run(job)) {
            self.flight.complete(update)?;
            return Err(anyhow!("async eval command failed: {error}"));
        }
        Ok(true)
    }

    fn poll(&mut self) -> Result<Option<EvalCompletion>> {
        match self.result_rx.try_recv() {
            Ok(Ok(completion)) => {
                self.flight.complete(completion.update)?;
                Ok(Some(completion))
            }
            Ok(Err(error)) => {
                self.flight.update = None;
                Err(anyhow!("asynchronous evaluation failed: {error}"))
            }
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => Err(anyhow!(
                "asynchronous evaluation thread exited unexpectedly"
            )),
        }
    }

    fn shutdown(mut self) -> Result<Option<EvalCompletion>> {
        let _ = self.command_tx.send(EvalCommand::Shutdown);
        let completion = if self.flight.is_busy() {
            match self.result_rx.recv() {
                Ok(Ok(completion)) => {
                    self.flight.complete(completion.update)?;
                    Some(completion)
                }
                Ok(Err(error)) => {
                    return Err(anyhow!("asynchronous evaluation failed: {error}"));
                }
                Err(_) => return Err(anyhow!("asynchronous evaluation result channel closed")),
            }
        } else {
            None
        };
        if self
            .handle
            .take()
            .expect("async eval join handle available")
            .join()
            .is_err()
        {
            return Err(anyhow!("asynchronous evaluation thread panicked"));
        }
        Ok(completion)
    }
}

impl Drop for AsyncEval {
    fn drop(&mut self) {
        let _ = self.command_tx.send(EvalCommand::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn report_eval_completion(
    completion: EvalCompletion,
    boundary_update: u64,
    metrics: &MetricsWriter,
) -> Result<Option<(f64, f64)>> {
    let suffix = if completion.promoted {
        " promoted=best_eval"
    } else {
        ""
    };
    println!(
        "[eval] update {} (reported_at={boundary_update}) stage {}  win {:.2}  score {:.2}  \
         ({} eps, {:.0}s){suffix}",
        completion.update,
        completion.stage,
        completion.result.win,
        completion.result.score,
        completion.result.episodes,
        completion.elapsed_seconds,
    );
    // Always append an eval row. In the asynchronous case the training row
    // for this evaluated update may already be on disk; retaining the launch
    // update here prevents a stale completion from being attributed to the
    // boundary where it happened to finish.
    metrics.log_update(
        completion.update,
        completion.stage,
        completion.report.losses.0,
        completion.report.losses.1,
        completion.report.losses.2,
        completion.report.losses.3,
        completion.report.win_rate,
        completion.report.lr_now,
        completion.report.total_env_steps,
        Some(completion.result.win),
        Some(completion.result.score),
    )?;
    Ok(completion.best)
}

#[cfg(test)]
mod async_eval_tests {
    use super::*;

    fn result(win: f64, score: f64) -> EvalResult {
        EvalResult {
            win,
            score,
            episodes: 8,
            details: Vec::new(),
        }
    }

    #[test]
    fn best_eval_uses_win_then_score_tie_break() {
        assert!(eval_is_better(&result(0.5, 1.0), None));
        assert!(eval_is_better(&result(0.6, -10.0), Some((0.5, 100.0))));
        assert!(eval_is_better(&result(0.5, 2.0), Some((0.5, 1.0))));
        assert!(!eval_is_better(&result(0.5, 1.0), Some((0.5, 1.0))));
        assert!(!eval_is_better(&result(0.4, 100.0), Some((0.5, 1.0))));
    }

    #[test]
    fn flight_rejects_overlap_and_checks_stale_completion_update() {
        let mut flight = EvalFlight::default();
        assert!(flight.reserve(17));
        assert!(!flight.reserve(18));
        let error = flight.complete(18).unwrap_err();
        assert!(error.to_string().contains("does not match"));
        assert!(flight.is_busy());
        flight.complete(17).unwrap();
        assert!(flight.reserve(19));
    }

    #[test]
    fn command_submission_is_nonblocking_when_worker_is_busy() {
        let (tx, _rx) = mpsc::sync_channel(1);
        tx.try_send(1u8).unwrap();
        let started = Instant::now();
        assert!(matches!(tx.try_send(2u8), Err(mpsc::TrySendError::Full(2))));
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn stale_result_is_logged_against_evaluated_update() {
        let dir = std::env::temp_dir().join(format!(
            "oftrain-eval-attribution-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let metrics = MetricsWriter::create(dir.to_str().unwrap()).unwrap();
        let completion = EvalCompletion {
            update: 7,
            stage: 2,
            result: result(0.75, 3.5),
            elapsed_seconds: 1.0,
            promoted: false,
            best: Some((0.8, 4.0)),
            report: EvalReportContext {
                losses: (1.0, 2.0, 3.0, 4.0),
                win_rate: Some(0.5),
                lr_now: 1e-4,
                total_env_steps: 99,
            },
        };
        report_eval_completion(completion, 12, &metrics).unwrap();
        let line = std::fs::read_to_string(dir.join("metrics.jsonl")).unwrap();
        let row: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        assert_eq!(row["update"], 7);
        assert_eq!(row["stage"], 2);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn best_atomic_paths_do_not_alias_latest() {
        let dir = std::env::temp_dir().join(format!(
            "oftrain-best-paths-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let best = best_eval_path(dir.to_str().unwrap());
        let latest = dir.join("latest.safetensors");
        std::fs::write(&latest, b"latest-sentinel").unwrap();
        let seen_tmp = std::cell::RefCell::new(String::new());
        save_atomic(&best, |tmp| {
            *seen_tmp.borrow_mut() = tmp.to_string();
            std::fs::write(tmp, b"best")?;
            Ok(())
        })
        .unwrap();
        let state = TrainState {
            checkpoint_schema_version: 1,
            hidden_reset_policy: "none".to_string(),
            update: 8,
            stage: 2,
            ent_scale: 1.0,
            lr_now: 1e-4,
            total_env_steps: 99,
            recent_wins: vec![1.0],
            recent_conversions: vec![],
            recent_deaths: vec![],
            best_eval_win: Some(0.75),
            best_eval_score: Some(3.5),
            curriculum_schedule: Some("v10".to_string()),
            reward_profile: Some(ofcore::curriculum::V10_REWARD_PROFILE.to_string()),
            return_stats: None,
            stage_env_targets: ofcore::curriculum::V10_ENV_TARGETS.to_vec(),
            envs_per_shard: 24,
            requested_env_target: None,
        };
        save_checkpoint_state(&best, &state).unwrap();
        assert!(seen_tmp.borrow().ends_with("best_eval.tmp.safetensors"));
        assert_eq!(std::fs::read(&best).unwrap(), b"best");
        let saved_state: TrainState = serde_json::from_str(
            &std::fs::read_to_string(dir.join("best_eval.state.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(saved_state.update, 8);
        assert_eq!(saved_state.best_eval_win, Some(0.75));
        assert_eq!(saved_state.envs_per_shard, 24);
        assert_eq!(std::fs::read(&latest).unwrap(), b"latest-sentinel");
        assert!(!std::path::Path::new(seen_tmp.borrow().as_str()).exists());
        std::fs::remove_dir_all(dir).unwrap();
    }
}

enum ActorCommand {
    Collect {
        id: u64,
    },
    Refresh {
        id: u64,
        policy_version: u64,
        weights: CpuWeightSnapshot,
    },
    SetStage {
        id: u64,
        stage: usize,
    },
    Eval {
        id: u64,
        stage: usize,
        episodes: usize,
    },
    Shutdown {
        id: u64,
    },
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
    Ready {
        envs: usize,
    },
    Collected {
        id: u64,
        result: RolloutResult,
    },
    Eval {
        id: u64,
        result: EvalResult,
    },
    Ack {
        id: u64,
    },
    Failed {
        id: u64,
        command: &'static str,
        error: String,
    },
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
        Self {
            next_command_id: 1,
            policy_version,
            stopped: false,
        }
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
    let policy = PolicyNet::new_with_recurrence(
        &vs.root(),
        cfg.amp,
        cfg.foveate,
        cfg.gc,
        cfg.blocks,
        cfg.recurrent_policy,
    );
    apply_weight_snapshot(&vs, &initial_weights)?;
    if cfg.amp {
        cast_actor_inference_weights_bf16(&vs);
    }
    if let Device::Cuda(index) = device {
        Cuda::synchronize(index as i64);
    }
    let path = std::path::Path::new(&cfg.ae_ckpt);
    anyhow::ensure!(
        path.exists(),
        "AE checkpoint not found at {} - run `ofae train` / `bash scripts/fetch_ae_encoders.sh` \
         (or scripts/fetch_ae_encoders.sh) first",
        path.display()
    );
    let coarse = cfg.coarse_ckpt.as_ref().map(std::path::Path::new);
    let ae = Some(crate::ae::AePair::load(
        path, coarse, device, cfg.amp, true,
    )?);
    let (ready_tx, ready_rx) = if cfg.work_conserving_actors {
        let (tx, rx) = mpsc::channel();
        (Some(tx), Some(rx))
    } else {
        (None, None)
    };
    let mut workers = Vec::with_capacity(cfg.num_envs);
    let mut cur_obs = Vec::with_capacity(cfg.num_envs);
    for local_i in 0..cfg.num_envs {
        let idx = shard_index * cfg.num_envs + local_i;
        let engine = engine_for_idx(idx, cfg.engine, cfg.node_fraction);
        let route = ready_tx.as_ref().map(|tx| (local_i, tx.clone()));
        let (worker, obs) = spawn_worker_routed(
            idx,
            stage,
            cfg.max_episode_ticks,
            engine,
            cfg.reward_config,
            cfg.curriculum_schedule,
            route,
        )?;
        workers.push(worker);
        cur_obs.push(obs);
    }
    let recurrent = cfg
        .recurrent_policy
        .then(|| ActorRecurrentState::new(workers.len(), cfg.recurrent_hidden_size, device));
    Ok(ActorShard {
        device,
        workers,
        cur_obs,
        ready_rx,
        compact_host_arena: Arc::new(crate::vecenv::CompactHostArena::default()),
        vs,
        policy,
        recurrent,
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
    if reply_tx
        .send(ActorReply::Ready {
            envs: actor.workers.len(),
        })
        .is_err()
    {
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
            ActorCommand::Collect { id } => {
                let collected = if cfg.work_conserving_actors {
                    collect_rollout_ready(&mut actor, &cfg, protocol.policy_version)
                } else {
                    collect_rollout(&mut actor, &cfg, protocol.policy_version)
                };
                collected.map(|result| ActorReply::Collected { id, result })
            }
            ActorCommand::Refresh { id, weights, .. } => apply_weight_snapshot(&actor.vs, &weights)
                .map(|_| {
                    if cfg.amp {
                        cast_actor_inference_weights_bf16(&actor.vs);
                    }
                    if let Device::Cuda(index) = actor.device {
                        Cuda::synchronize(index as i64);
                    }
                    ActorReply::Ack { id }
                }),
            ActorCommand::SetStage { id, stage } => {
                for worker in &actor.workers {
                    let _ = worker.stage_tx.send(stage);
                }
                Ok(ActorReply::Ack { id })
            }
            ActorCommand::Eval {
                id,
                stage,
                episodes,
            } => actor
                .ae
                .as_ref()
                .ok_or_else(|| anyhow!("eval requires AE encoders"))
                .and_then(|ae| {
                    run_eval(
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
                        cfg.recurrent_policy,
                        cfg.reward_config,
                        cfg.curriculum_schedule,
                    )
                })
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
        ActorReply::Failed { id, command, error } => {
            anyhow!("persistent actor {shard_index} command {id} ({command}) failed: {error}")
        }
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
            .spawn(move || {
                actor_loop(
                    shard_index,
                    device,
                    cfg,
                    stage,
                    initial_policy_version,
                    initial_weights,
                    command_rx,
                    reply_tx,
                )
            })?;
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
                anyhow::ensure!(
                    envs > 0,
                    "persistent actor {shard_index} initialized with no envs"
                );
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
        let Some(handle) = self.handle.take() else {
            return String::new();
        };
        match handle.join() {
            Ok(()) => " (thread exited)".to_string(),
            Err(payload) => {
                let reason = payload
                    .downcast_ref::<&str>()
                    .copied()
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
        anyhow::ensure!(
            self.pending_collect_id.is_none(),
            "persistent actor {} already collecting",
            self.shard_index
        );
        let id = self.next_id();
        self.command_tx.send(ActorCommand::Collect { id })?;
        self.pending_collect_id = Some(id);
        Ok(())
    }
    fn finish_collect(&mut self) -> Result<RolloutResult> {
        let expected = self.pending_collect_id.take().ok_or_else(|| {
            anyhow!(
                "persistent actor {} has no pending collect",
                self.shard_index
            )
        })?;
        match self.recv_reply("rollout collection")? {
            ActorReply::Collected { id, result } if id == expected => Ok(result),
            reply => Err(actor_reply_error(
                self.shard_index,
                "matching collect result",
                &reply,
            )),
        }
    }
    fn request_ack(&mut self, command: ActorCommand) -> Result<()> {
        let expected = command.id();
        let phase = command.name();
        self.command_tx.send(command)?;
        match self.recv_reply(phase)? {
            ActorReply::Ack { id } if id == expected => Ok(()),
            reply => Err(actor_reply_error(
                self.shard_index,
                "matching acknowledgement",
                &reply,
            )),
        }
    }
    fn refresh(&mut self, policy_version: u64, weights: CpuWeightSnapshot) -> Result<()> {
        let id = self.next_id();
        self.request_ack(ActorCommand::Refresh {
            id,
            policy_version,
            weights,
        })
    }
    fn set_stage(&mut self, stage: usize) -> Result<()> {
        let id = self.next_id();
        self.request_ack(ActorCommand::SetStage { id, stage })
    }
    fn eval(&mut self, stage: usize, episodes: usize) -> Result<EvalResult> {
        let id = self.next_id();
        self.command_tx.send(ActorCommand::Eval {
            id,
            stage,
            episodes,
        })?;
        match self.recv_reply("evaluation")? {
            ActorReply::Eval {
                id: reply_id,
                result,
            } if reply_id == id => Ok(result),
            reply => Err(actor_reply_error(
                self.shard_index,
                "matching eval result",
                &reply,
            )),
        }
    }
    fn shutdown(mut self) -> Result<()> {
        let id = self.next_id();
        self.request_ack(ActorCommand::Shutdown { id })?;
        let status = self.join_status();
        anyhow::ensure!(
            !status.contains("panicked"),
            "actor shutdown failed{status}"
        );
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
    SaveWeights {
        id: u64,
        path: String,
    },
    Shutdown {
        id: u64,
    },
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
    ApplyCpu(Arc<Vec<f32>>),
    Collective,
    Discard,
    Abort(String),
}

enum LearnerReply {
    Ready {
        shard: usize,
        params: i64,
    },
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
        timings: TrainTimings,
    },
    Ack {
        id: u64,
        shard: usize,
    },
    Failed {
        id: u64,
        shard: usize,
        command: &'static str,
        error: String,
    },
}

#[derive(Debug)]
struct LearnerProtocol {
    next_command_id: u64,
    stopped: bool,
}

impl LearnerProtocol {
    fn new() -> Self {
        Self {
            next_command_id: 1,
            stopped: false,
        }
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
    gradient_rx: Option<Receiver<GradientDecision>>,
    mut nccl_comm: Option<crate::nccl::Comm>,
) {
    let init_started = Instant::now();
    eprintln!("[phase] persistent learner initialization started");
    let initialized = (|| -> Result<LearnerShard> {
        let vs = nn::VarStore::new(device);
        let policy = PolicyNet::new_with_recurrence(
            &vs.root(),
            cfg.amp,
            cfg.foveate,
            cfg.gc,
            cfg.blocks,
            cfg.recurrent_policy,
        );
        apply_weight_snapshot(&vs, &initial_weights)?;
        let opt = nn::AdamW::default().build(&vs, initial_lr)?;
        if let Device::Cuda(index) = device {
            Cuda::synchronize(index as i64);
        }
        Ok(LearnerShard {
            device,
            vs,
            policy,
            opt,
        })
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
    if reply_tx
        .send(LearnerReply::Ready {
            shard: shard_index,
            params,
        })
        .is_err()
    {
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
                mut rollout,
                lr,
                ent_coef,
                ret_stat: initial_ret_stat,
            } => {
                learner.opt.set_lr(lr);
                let adaptive_ret_bound = (RET_ADAPTIVE_N_STD * initial_ret_stat.std()).max(1.0);
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
                    let values = if finite && nccl_comm.is_none() {
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
                    match gradient_rx
                        .as_ref()
                        .expect("gradient hub receiver")
                        .recv()?
                    {
                        GradientDecision::ApplyCpu(values) => {
                            apply_cpu_flat_grad(owned, values.as_slice())?;
                            Ok(true)
                        }
                        GradientDecision::Collective => {
                            let comm = nccl_comm.as_mut().ok_or_else(|| {
                                anyhow!("NCCL collective requested without communicator")
                            })?;
                            let mut flat = flat_grad_on_device(owned);
                            // Any error after entering a collective is fatal.
                            // Falling back here could strand another rank or
                            // apply a partially reduced optimizer step.
                            comm.all_reduce_average(
                                &mut flat,
                                &format!("command={id} epoch={epoch} minibatch={minibatch}"),
                            )?;
                            apply_device_flat_grad(owned, &flat)?;
                            Ok(true)
                        }
                        GradientDecision::Discard => Ok(false),
                        GradientDecision::Abort(error) => Err(anyhow!(error)),
                    }
                };
                let mut timings = TrainTimings::default();
                let losses = if gradient_rx.is_some() {
                    train_update(
                        std::slice::from_mut(&mut learner),
                        std::slice::from_mut(&mut rollout),
                        &cfg,
                        &mut rng,
                        ent_coef,
                        &mut ret_stat,
                        true,
                        Some(adaptive_ret_bound),
                        Some(&mut sync),
                        &mut timings,
                    )
                } else {
                    // A single persistent owner already has the final gradient
                    // in place. Avoid flattening it through CPU and rewriting
                    // it to the same device.
                    train_update(
                        std::slice::from_mut(&mut learner),
                        std::slice::from_mut(&mut rollout),
                        &cfg,
                        &mut rng,
                        ent_coef,
                        &mut ret_stat,
                        true,
                        Some(adaptive_ret_bound),
                        None,
                        &mut timings,
                    )
                };
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
                        timings,
                    })
                })
            }
            LearnerCommand::SaveWeights { id, path } => (|| -> Result<()> {
                if shard_index == 0 {
                    save_atomic(&path, |tmp| Ok(learner.vs.save(tmp)?))?;
                }
                Ok(())
            })()
            .map(|_| LearnerReply::Ack {
                id,
                shard: shard_index,
            }),
            LearnerCommand::Shutdown { id } => Ok(LearnerReply::Ack {
                id,
                shard: shard_index,
            }),
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
    timings: TrainTimings,
    /// Test-only copies of the hub result at every optimizer barrier. This
    /// proves parity before clipping/Adam can obscure the source of drift.
    #[cfg(test)]
    averaged_gradients: Vec<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, Default)]
struct TrainTimings {
    batch_build_seconds: f64,
    gradient_sync_seconds: f64,
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
    nccl_enabled: bool,
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
        // ncclCommInitAll is the only cross-rank initialization point. Each
        // resulting rank handle is then moved once into its permanent owner.
        // Startup failure is safe to fall back from because no collective or
        // optimizer step has begun.
        let mut nccl_comms: Vec<Option<crate::nccl::Comm>> = match crate::nccl::try_init(devices) {
            Ok(Some(comms)) => {
                anyhow::ensure!(
                    comms.len() == devices.len(),
                    "NCCL returned {} communicators for {} learners",
                    comms.len(),
                    devices.len()
                );
                eprintln!(
                    "[nccl] device-resident gradient all-reduce enabled for {} learner owners",
                    devices.len()
                );
                comms.into_iter().map(Some).collect()
            }
            Ok(None) => {
                if devices.len() > 1 {
                    eprintln!("[nccl] unavailable; using CPU persistent gradient hub");
                }
                (0..devices.len()).map(|_| None).collect()
            }
            Err(error) => {
                eprintln!(
                    "[nccl] initialization failed before training ({error:#}); using CPU persistent gradient hub"
                );
                (0..devices.len()).map(|_| None).collect()
            }
        };
        let nccl_enabled = nccl_comms.iter().all(Option::is_some);
        // DDP ranks use the same epoch permutation over their disjoint local
        // batches. Giving each owner a different permutation is mathematically
        // equivalent in exact arithmetic, but changes floating-point reduction
        // order. For duplicated-rollout parity that produces g0 != g1, and
        // Adam's first-step normalization can amplify the tiny discrepancy.
        let shuffle_seed = rng.next_u64();
        let use_gradient_hub = devices.len() > 1;
        for (shard, &device) in devices.iter().enumerate() {
            let (command_tx, command_rx) = mpsc::channel();
            let (gradient_tx, gradient_rx) = if use_gradient_hub {
                let (tx, rx) = mpsc::channel();
                (Some(tx), Some(rx))
            } else {
                (None, None)
            };
            let thread_reply = reply_tx.clone();
            let thread_cfg = cfg.clone();
            let thread_weights = initial_weights.clone();
            let thread_rng = rand::rngs::SmallRng::seed_from_u64(shuffle_seed);
            let nccl_comm = nccl_comms[shard].take();
            let handle = std::thread::Builder::new()
                .name(format!("learner-gpu{shard}"))
                .spawn(move || {
                    learner_loop(
                        shard,
                        device,
                        thread_cfg,
                        thread_weights,
                        initial_lr,
                        thread_rng,
                        command_rx,
                        thread_reply,
                        gradient_rx,
                        nccl_comm,
                    )
                })?;
            command_txs.push(command_tx);
            gradient_txs.extend(gradient_tx);
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
            nccl_enabled,
        };
        let mut params = vec![None; devices.len()];
        for _ in devices {
            match learner.recv_reply("initialization")? {
                LearnerReply::Ready {
                    shard,
                    params: count,
                } if shard < params.len() => {
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
            let Some(handle) = handle.take() else {
                continue;
            };
            match handle.join() {
                Ok(()) => statuses.push(format!("shard {shard} exited")),
                Err(payload) => {
                    let reason = payload
                        .downcast_ref::<&str>()
                        .copied()
                        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                        .unwrap_or("non-string panic");
                    statuses.push(format!("shard {shard} panicked: {reason}"));
                }
            }
        }
        if statuses.is_empty() {
            String::new()
        } else {
            format!(" ({})", statuses.join(", "))
        }
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
                                handle
                                    .as_ref()
                                    .is_some_and(JoinHandle::is_finished)
                                    .then_some(shard)
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
    fn train(
        &mut self,
        rollouts: Vec<RolloutResult>,
        lr: f64,
        ent_coef: f32,
    ) -> Result<TrainReply> {
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
        if world > 1 {
            for epoch in 0..self.epochs {
                for minibatch in 0..self.minibatches {
                    let mut packets: Vec<Option<(bool, Vec<f32>)>> =
                        (0..world).map(|_| None).collect();
                    for _ in 0..world {
                        match self.recv_reply("finite gradient barrier")? {
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
                                return Err(learner_reply_error(
                                    "matching gradient packet",
                                    &reply,
                                ));
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
                    if self.nccl_enabled {
                        anyhow::ensure!(
                            packets.iter().all(|packet| packet
                                .as_ref()
                                .is_some_and(|(_, values)| values.is_empty())),
                            "NCCL readiness packet unexpectedly contained host gradients"
                        );
                        // Enter the collective only after every owner reports
                        // a finite minibatch, so no rank can be stranded.
                        for tx in &self.gradient_txs {
                            tx.send(GradientDecision::Collective)?;
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
                        tx.send(GradientDecision::ApplyCpu(average.clone()))?;
                    }
                }
            }
        }
        let mut trained: Vec<
            Option<(
                (f64, f64, f64, f64),
                CpuWeightSnapshot,
                RetStat,
                f64,
                f64,
                TrainTimings,
            )>,
        > = (0..world).map(|_| None).collect();
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
                    timings,
                } if reply_id == id && shard < world && trained[shard].is_none() => {
                    trained[shard] = Some((
                        losses,
                        weights,
                        ret_stat,
                        train_seconds,
                        snapshot_seconds,
                        timings,
                    ));
                }
                reply => return Err(learner_reply_error("matching train result", &reply)),
            }
        }
        let mut losses = (0.0, 0.0, 0.0, 0.0);
        let mut weights = Vec::with_capacity(world);
        let mut train_seconds = 0.0f64;
        let mut snapshot_seconds = 0.0f64;
        let mut timings = TrainTimings::default();
        for entry in trained {
            let (local, snapshot, stat, train_s, snapshot_s, local_timings) =
                entry.expect("all shards trained");
            losses.0 += local.0 / world as f64;
            losses.1 += local.1 / world as f64;
            losses.2 += local.2 / world as f64;
            losses.3 += local.3 / world as f64;
            self.ret_stat.add_batch(stat.count, stat.sum, stat.sum_sq);
            weights.push(snapshot);
            train_seconds = train_seconds.max(train_s);
            snapshot_seconds = snapshot_seconds.max(snapshot_s);
            timings.batch_build_seconds = timings
                .batch_build_seconds
                .max(local_timings.batch_build_seconds);
            timings.gradient_sync_seconds = timings
                .gradient_sync_seconds
                .max(local_timings.gradient_sync_seconds);
        }
        Ok(TrainReply {
            losses,
            weights,
            train_seconds,
            snapshot_seconds,
            timings,
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
                LearnerReply::Ack {
                    id: reply_id,
                    shard,
                } if reply_id == id && shard < seen.len() && !seen[shard] => {
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
                LearnerReply::Ack {
                    id: reply_id,
                    shard,
                } if reply_id == id && shard < seen.len() && !seen[shard] => {
                    seen[shard] = true;
                }
                reply => {
                    return Err(learner_reply_error(
                        "matching shutdown acknowledgement",
                        &reply,
                    ));
                }
            }
        }
        let status = self.join_status();
        anyhow::ensure!(
            !status.contains("panicked"),
            "learner shutdown failed{status}"
        );
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
        LearnerReply::Failed {
            id,
            shard,
            command,
            error,
        } => {
            anyhow!("persistent learner shard {shard} command {id} ({command}) failed: {error}")
        }
        _ => anyhow!("persistent learner protocol error: expected {expected}"),
    }
}

fn flat_grad_to_cpu(shard: &LearnerShard) -> Result<Vec<f32>> {
    let flat = flat_grad_on_device(shard).to_device(Device::Cpu);
    Ok((&flat).try_into()?)
}

fn flat_grad_on_device(shard: &LearnerShard) -> Tensor {
    let parts: Vec<Tensor> = shard
        .vs
        .trainable_variables()
        .iter()
        .map(|v| v.grad().reshape([-1]))
        .collect();
    Tensor::cat(&parts, 0).to_kind(Kind::Float)
}

fn apply_cpu_flat_grad(shard: &mut LearnerShard, values: &[f32]) -> Result<()> {
    let flat = Tensor::from_slice(values).to_device(shard.device);
    apply_device_flat_grad(shard, &flat)
}

fn apply_device_flat_grad(shard: &mut LearnerShard, flat: &Tensor) -> Result<()> {
    anyhow::ensure!(
        flat.device() == shard.device,
        "flat gradient device mismatch: {:?} != {:?}",
        flat.device(),
        shard.device
    );
    let mut offset = 0i64;
    for variable in shard.vs.trainable_variables() {
        let mut grad = variable.grad();
        let len = grad.numel() as i64;
        let source = flat.narrow(0, offset, len).reshape(grad.size());
        grad.f_copy_(&source)?;
        offset += len;
    }
    anyhow::ensure!(
        offset as usize == flat.numel(),
        "flat gradient length mismatch"
    );
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

fn read_loss_scalars(losses: [&Tensor; 4]) -> (f64, f64, f64, f64) {
    // One packed D2H read replaces four scalar synchronizations. PPO losses
    // are Float tensors, so widening each returned f32 preserves every bit.
    let packed = Tensor::stack(&losses, 0).to_device(Device::Cpu);
    let values = Vec::<f32>::try_from(&packed).unwrap_or_else(|_| vec![0.0; 4]);
    (
        values[0] as f64,
        values[1] as f64,
        values[2] as f64,
        values[3] as f64,
    )
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

fn bptt_ranges(t_len: usize, chunk_len: usize) -> Vec<std::ops::Range<usize>> {
    let chunk_len = chunk_len.max(1);
    (0..t_len)
        .step_by(chunk_len)
        .map(|start| start..(start + chunk_len).min(t_len))
        .collect()
}

fn shuffled_sequence_envs(n: usize, rng: &mut rand::rngs::SmallRng) -> Vec<i64> {
    let mut envs: Vec<i64> = (0..n as i64).collect();
    envs.shuffle(rng);
    envs
}

/// Runs GAE + the `epochs` x `minibatches` clipped-PPO update for one
/// update's worth of rollouts (one `RolloutResult` per learner shard).
/// Pure compute over `learners`/`pending` - never touches any
/// `ActorShard`, so it's safe to run concurrently with the *next*
/// update's `collect_rollout` calls (see module doc).
fn train_update(
    learners: &mut [LearnerShard],
    pending: &mut [RolloutResult],
    cfg: &Config,
    rng: &mut rand::rngs::SmallRng,
    ent_coef: f32,
    ret_stat: &mut RetStat,
    exclusive_owner: bool,
    adaptive_ret_bound_override: Option<f64>,
    mut persistent_sync: Option<
        &mut dyn FnMut(&mut LearnerShard, usize, usize, bool) -> Result<bool>,
    >,
    timings: &mut TrainTimings,
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
        hidden_in: Option<Tensor>,
        context: Option<Tensor>,
        reset_before: Option<Tensor>,
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
                    let mut hidden_flat = cfg.recurrent_policy.then(|| {
                        vec![0.0f32; total * policy::RECURRENT_HIDDEN as usize]
                    });
                    let mut context_flat = cfg.recurrent_policy.then(|| {
                        vec![0.0f32; total * crate::recurrent::CONTEXT_FLOATS]
                    });
                    let mut reset_flat = cfg.recurrent_policy.then(|| vec![0.0f32; total]);
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
                            if let Some(hidden) = hidden_flat.as_mut() {
                                let state = &buffer[t][e].hidden_in;
                                debug_assert_eq!(state.len(), policy::RECURRENT_HIDDEN as usize);
                                let begin = flat_idx * policy::RECURRENT_HIDDEN as usize;
                                hidden[begin..begin + policy::RECURRENT_HIDDEN as usize]
                                    .copy_from_slice(state);
                            }
                            if let Some(context) = context_flat.as_mut() {
                                let values = buffer[t][e].context.as_floats();
                                let begin = flat_idx * crate::recurrent::CONTEXT_FLOATS;
                                context[begin..begin + crate::recurrent::CONTEXT_FLOATS]
                                    .copy_from_slice(&values);
                            }
                            if let Some(reset) = reset_flat.as_mut() {
                                reset[flat_idx] =
                                    (t > 0 && buffer[t - 1][e].done) as u8 as f32;
                            }
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
                    let hidden_in = hidden_flat.map(|values| {
                        Tensor::from_slice(&values)
                            .view([total as i64, policy::RECURRENT_HIDDEN])
                            .to_device(device)
                    });
                    let context = context_flat.map(|values| {
                        Tensor::from_slice(&values)
                            .view([total as i64, crate::recurrent::CONTEXT_FLOATS as i64])
                            .to_device(device)
                    });
                    let reset_before =
                        reset_flat.map(|values| Tensor::from_slice(&values).to_device(device));
                    // Legacy multi-shard batches cross from this short-lived
                    // builder thread to a different consumer thread, so its
                    // stream must complete first. An exclusive persistent
                    // owner constructs and consumes on this same stable thread;
                    // synchronizing there only drains its own stream early.
                    if !exclusive_owner && let Device::Cuda(index) = device {
                        Cuda::synchronize(index as i64);
                    }
                    (
                        ShardBatch {
                            obs,
                            choice,
                            adv,
                            ret,
                            old_logp,
                            hidden_in,
                            context,
                            reset_before,
                        },
                        local_count,
                        local_sum,
                        local_sum_sq,
                    )
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
            handles
                .into_iter()
                .map(|h| {
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
                })
                .collect()
        })
    };
    if std::env::var("OFTRAIN_DIAG").is_ok() {
        println!(
            "[diag] batch_build_s={:.3}",
            batch_build_t0.elapsed().as_secs_f64()
        );
    }
    timings.batch_build_seconds += batch_build_t0.elapsed().as_secs_f64();
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
    // Pipelined collect keeps rollout N+1's host obs alive while we train
    // on N. Once ShardBatch is on-device, drop N's host PreparedObs / compact
    // arena refs so peak RSS is one host rollout + one GPU batch, not two
    // host rollouts stacked through the whole PPO loop.
    for result in pending.iter_mut() {
        for row in result.buffer.iter_mut() {
            for step in row.iter_mut() {
                step.obs.release_rollout_payload();
            }
        }
    }

    for _epoch in 0..cfg.epochs {
        let epoch_started = Instant::now();
        if exclusive_owner {
            eprintln!(
                "[phase] learner PPO epoch {}/{} started",
                _epoch + 1,
                cfg.epochs
            );
        }
        // Per-shard shuffled index tensor, built once per epoch (CPU
        // shuffle + one tiny (total,) i64 upload) and resident on that
        // shard's device - minibatches `narrow` a contiguous slice of it
        // (a view, no host round trip) to `index_select` the full-batch
        // tensors above.
        let mut idx_vec: Vec<i64> = if cfg.recurrent_policy {
            shuffled_sequence_envs(n, rng)
        } else {
            (0..total as i64).collect()
        };
        let idx_tensors: Vec<Tensor> = learners
            .iter()
            .map(|shard| {
                if !cfg.recurrent_policy {
                    idx_vec.shuffle(rng);
                }
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
        let sub_size_steps = (MAX_UPD_PIX / pix_per).max(1);
        let sub_size = if cfg.recurrent_policy {
            (sub_size_steps / t_len.max(1)).max(1) as i64
        } else {
            sub_size_steps as i64
        };

        for m in 0..n_minibatches {
            let unit_total = if cfg.recurrent_policy { n } else { total };
            let unit_minibatch_size = if cfg.recurrent_policy {
                (n / n_minibatches).max(1)
            } else {
                minibatch_size
            };
            let start = (m * unit_minibatch_size) as i64;
            let len = if m == n_minibatches - 1 {
                unit_total as i64 - start
            } else {
                unit_minibatch_size as i64
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
                        let (logp, ent, ent_q, value, quantity, adv_t, ret_t, old_logp_t) = if cfg
                            .recurrent_policy
                        {
                            let hidden_all = sb.hidden_in.as_ref().expect("recurrent hidden batch");
                            let context_all = sb.context.as_ref().expect("recurrent context batch");
                            let reset_all =
                                sb.reset_before.as_ref().expect("recurrent reset batch");
                            let mut logps = Vec::with_capacity(t_len);
                            let mut ents = Vec::with_capacity(t_len);
                            let mut entqs = Vec::with_capacity(t_len);
                            let mut values = Vec::with_capacity(t_len);
                            let mut quantities = Vec::with_capacity(t_len);
                            let mut target_indices =
                                Vec::with_capacity(t_len.div_ceil(cfg.bptt_chunk_len));
                            for range in bptt_ranges(t_len, cfg.bptt_chunk_len) {
                                let first = &idx_t + (range.start * n) as i64;
                                // Actor state at the boundary is a
                                // detached BPTT initial condition.
                                let hidden = hidden_all.index_select(0, &first);
                                // One time-major gather per field and
                                // one full trunk/head pass per BPTT
                                // chunk. Only the GRU recurrence is
                                // evaluated timestep by timestep.
                                let time_offsets = Tensor::arange_start(
                                    range.start as i64,
                                    range.end as i64,
                                    (Kind::Int64, shard.device),
                                ) * n as i64;
                                let chunk_idx =
                                    (time_offsets.unsqueeze(1) + idx_t.unsqueeze(0)).flatten(0, -1);
                                let obs_chunk = sb.obs.index_select(&chunk_idx);
                                let choice_chunk = sb.choice.index_select(&chunk_idx);
                                let context_chunk = context_all.index_select(0, &chunk_idx);
                                let reset_chunk = reset_all.index_select(0, &chunk_idx);
                                let (lp, en, eq, v, _) = shard.policy.evaluate_sequence_fused(
                                    &obs_chunk,
                                    &choice_chunk,
                                    &hidden,
                                    &context_chunk,
                                    &reset_chunk,
                                    range.len() as i64,
                                );
                                logps.push(lp);
                                ents.push(en);
                                entqs.push(eq);
                                values.push(v);
                                quantities.push(choice_chunk.quantity_frac);
                                target_indices.push(chunk_idx);
                            }
                            let target_idx =
                                Tensor::cat(&target_indices.iter().collect::<Vec<_>>(), 0);
                            (
                                Tensor::cat(&logps.iter().collect::<Vec<_>>(), 0),
                                Tensor::cat(&ents.iter().collect::<Vec<_>>(), 0),
                                Tensor::cat(&entqs.iter().collect::<Vec<_>>(), 0),
                                Tensor::cat(&values.iter().collect::<Vec<_>>(), 0),
                                Tensor::cat(&quantities.iter().collect::<Vec<_>>(), 0),
                                sb.adv.index_select(0, &target_idx),
                                sb.ret.index_select(0, &target_idx),
                                sb.old_logp.index_select(0, &target_idx),
                            )
                        } else {
                            let obs_t = sb.obs.index_select(&idx_t);
                            let choice_t = sb.choice.index_select(&idx_t);
                            let (lp, en, eq, v) = shard.policy.evaluate(&obs_t, &choice_t);
                            (
                                lp,
                                en,
                                eq,
                                v,
                                choice_t.quantity_frac,
                                sb.adv.index_select(0, &idx_t),
                                sb.ret.index_select(0, &idx_t),
                                sb.old_logp.index_select(0, &idx_t),
                            )
                        };
                        // Bound log-ratio before exp (see prior
                        // pg_loss trillion-spike incident).
                        let log_ratio = (&logp - &old_logp_t).clamp(-20.0, 20.0);
                        let ratio = log_ratio.exp();
                        let surr1 = &ratio * &adv_t;
                        let surr2 =
                            ratio.clamp(1.0 - cfg.clip as f64, 1.0 + cfg.clip as f64) * &adv_t;
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
                        let n_active = quantity
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

                        read_loss_scalars([&pg_loss, &v_loss, &ent_loss, &entq_loss])
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
                        handles
                            .into_iter()
                            .map(|h| {
                                h.join().unwrap_or_else(|e| {
                                    let msg = e
                                        .downcast_ref::<&str>()
                                        .map(|s| s.to_string())
                                        .or_else(|| e.downcast_ref::<String>().cloned())
                                        .unwrap_or_else(|| format!("{e:?}"));
                                    panic!("backward thread panicked: {msg}");
                                })
                            })
                            .collect()
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
            // tensors. They first rendezvous here with finite status. The
            // selected backend then either enters owner-local NCCL or sends a
            // CPU flat gradient to the deterministic hub. A non-finite shard
            // causes every owner to discard the same minibatch before any
            // collective starts.
            if let Some(sync) = persistent_sync.as_deref_mut() {
                let sync_t0 = Instant::now();
                debug_assert_eq!(learners.len(), 1);
                let global_finite = sync(&mut learners[0], _epoch, m, !discard_mb)?;
                timings.gradient_sync_seconds += sync_t0.elapsed().as_secs_f64();
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
            timings.gradient_sync_seconds += sync_dt;
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
    anyhow::ensure!(
        cfg.actor_max_batch > 0,
        "--actor-max-batch must be at least 1"
    );
    if cfg.recurrent_policy {
        anyhow::ensure!(
            cfg.recurrent_hidden_size == policy::RECURRENT_HIDDEN as usize,
            "recurrent hidden size must be {}",
            policy::RECURRENT_HIDDEN
        );
        anyhow::ensure!(cfg.bptt_chunk_len > 0, "BPTT chunk length must be positive");
        // Minibatch vs env-count is re-checked after startup env resolve below.
        // Do not require minibatches <= every stage_env_targets entry: those are
        // within-stage floors, and GPU-util autoscale may run above a late-stage
        // floor with a matching --minibatches derived from the live count.
    }
    if cfg.work_conserving_actors && !cfg.persistent_actors {
        println!(
            "[train] WARNING: --work-conserving-actors requires --persistent-actors; \
             selecting the legacy collector path"
        );
        cfg.work_conserving_actors = false;
    }
    anyhow::ensure!(
        !cfg.work_conserving_actors || (cfg.compact_rollout && cfg.foveate),
        "--work-conserving-actors requires --compact-rollout and --foveate=true"
    );
    if cfg.persistent_actors && cfg.auto_scale_envs {
        println!(
            "[train] --persistent-actors + --auto-scale-envs: GPU-util growth uses \
             restart_request.json (same path as stage env targets); live in-process \
             spawn remains legacy-only"
        );
    }
    let stage_count = ofcore::curriculum::stages_for_schedule(cfg.curriculum_schedule).len();
    anyhow::ensure!(
        cfg.stage_env_targets.is_empty() || cfg.stage_env_targets.len() == stage_count,
        "stage env target count must match the curriculum stage count"
    );
    std::fs::create_dir_all(&cfg.ckpt_dir)?;

    // Resume / init: load weights before shards spawn. `--resume` restores
    // TrainState; `--init` is warm-start only (fresh counters/stage).
    let mut resumed_state: Option<TrainState> = None;
    let mut hub_vs: Option<nn::VarStore> = if let Some(resume_path) = &cfg.resume {
        if cfg.recurrent_policy {
            let manifest = read_architecture_manifest(resume_path, 2)?;
            let recurrent = &manifest["architecture"]["recurrent"];
            anyhow::ensure!(
                recurrent["cell"] == "gru",
                "V8.2 resume requires a GRU manifest"
            );
            anyhow::ensure!(
                recurrent["hidden_size"] == cfg.recurrent_hidden_size,
                "V8.2 recurrent hidden-size mismatch"
            );
            anyhow::ensure!(
                recurrent["context_schema"] == policy::RECURRENT_CONTEXT_SCHEMA
                    && recurrent["context_features"] == policy::RECURRENT_CONTEXT_FLOATS,
                "V8.2 recurrent context schema mismatch"
            );
            anyhow::ensure!(
                recurrent["bptt_length"] == cfg.bptt_chunk_len
                    && recurrent["rollout_length"] == cfg.rollout_len,
                "recurrent BPTT/rollout configuration mismatch"
            );
            anyhow::ensure!(
                recurrent["hidden_reset_policy"] == "episode_done",
                "unsupported V8.2 hidden reset policy"
            );
        }
        let mut snapshot = nn::VarStore::new(Device::Cpu);
        let _ = PolicyNet::new_with_recurrence(
            &snapshot.root(),
            cfg.amp,
            cfg.foveate,
            cfg.gc,
            cfg.blocks,
            cfg.recurrent_policy,
        );
        snapshot.load(resume_path)?;
        let state_path = state_sidecar_path(resume_path);
        resumed_state = match std::fs::read_to_string(&state_path) {
            Ok(s) => Some(serde_json::from_str(&s)?),
            Err(e) if cfg.recurrent_policy => {
                anyhow::bail!("V8.2 resume requires TrainState sidecar {state_path}: {e}")
            }
            Err(e) => {
                println!(
                    "[train] WARNING: resuming weights from {resume_path} but no readable state \
                     sidecar at {state_path} ({e}); starting update/stage/entropy-scale counters \
                     from scratch with the resumed weights"
                );
                None
            }
        };
        if let Some(state) = resumed_state.as_ref().filter(|_| cfg.recurrent_policy) {
            anyhow::ensure!(
                state.checkpoint_schema_version == 2,
                "V8.2 resume requires TrainState checkpoint_schema_version=2"
            );
            anyhow::ensure!(
                state.hidden_reset_policy == "episode_done",
                "V8.2 resume requires TrainState hidden_reset_policy=episode_done"
            );
        }
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
        if cfg.recurrent_policy {
            let _ = read_architecture_manifest(init_path, 2)?;
        }
        let mut snapshot = nn::VarStore::new(Device::Cpu);
        let _ = PolicyNet::new_with_recurrence(
            &snapshot.root(),
            cfg.amp,
            cfg.foveate,
            cfg.gc,
            cfg.blocks,
            cfg.recurrent_policy,
        );
        snapshot.load(init_path)?;
        println!(
            "[train] warm-started weights from {init_path} (--init; fresh TrainState / optimizer)"
        );
        Some(snapshot)
    } else {
        None
    };
    if let Some(state) = &mut resumed_state {
        let saved_schedule_id = state
            .curriculum_schedule
            .clone()
            .unwrap_or_else(|| "<missing>".to_string());
        reconcile_resume_schedule(state, cfg.curriculum_schedule, cfg.migrate_v86_to_v10)?;
        reconcile_resume_stage_and_lr(state, cfg.lr, cfg.stage_lr_decay, cfg.stage_lr_floor);
        if saved_schedule_id != cfg.curriculum_schedule.id() {
            println!(
                "[train] migrated curriculum schedule {} -> {}",
                saved_schedule_id,
                cfg.curriculum_schedule.id()
            );
        }
        if !state.stage_env_targets.is_empty() {
            anyhow::ensure!(
                state.stage_env_targets.len() == stage_count,
                "checkpoint has {} stage env targets, expected {}",
                state.stage_env_targets.len(),
                stage_count
            );
            if cfg.stage_env_targets != state.stage_env_targets {
                println!("[train] restoring stage env targets from TrainState");
                cfg.stage_env_targets.clone_from(&state.stage_env_targets);
            }
        }
    }
    let start_stage = resumed_state.as_ref().map(|s| s.stage).unwrap_or(cfg.stage);
    anyhow::ensure!(
        start_stage < stage_count,
        "curriculum stage {start_stage} is out of range"
    );
    let restart_path = restart_request_path(&cfg.ckpt_dir);
    let fulfilling_restart = std::path::Path::new(&restart_path).exists();
    let stage_target = cfg.stage_env_targets.get(start_stage).copied();
    let resolved = clamp_resolved_envs_to_autoscale_max(
        resolve_startup_envs_per_shard(
            cfg.num_envs,
            stage_target,
            resumed_state.as_ref(),
            fulfilling_restart,
        ),
        cfg.auto_scale_envs,
        cfg.max_envs,
    );
    if cfg.num_envs != resolved {
        println!(
            "[train] stage {start_stage} env sizing: {} -> {resolved} envs/shard \
             (stage_floor={:?}, fulfilling_restart={fulfilling_restart})",
            cfg.num_envs, stage_target
        );
    }
    cfg.num_envs = resolved;
    if cfg.recurrent_policy {
        anyhow::ensure!(
            cfg.minibatches.max(1) <= cfg.num_envs,
            "recurrent PPO requires minibatches <= envs per shard \
             (minibatches={}, envs/shard={})",
            cfg.minibatches,
            cfg.num_envs
        );
    }
    // main() may default `--min-envs` from `stage_env_targets[--stage]`
    // (often an early-stage 24) before this resume/stage-floor resolve.
    // Snap the autoscale floor down to the live startup count so growth
    // steps from reality instead of jumping to a stale stage-5 default.
    if cfg.auto_scale_envs && cfg.min_envs > cfg.num_envs {
        println!(
            "[autoscale] lowering min_envs {} -> {} to match resolved startup count",
            cfg.min_envs, cfg.num_envs
        );
        cfg.min_envs = cfg.num_envs;
    }

    let metrics = MetricsWriter::create(&cfg.ckpt_dir)?;
    println!("[train] metrics -> {}/metrics.jsonl", cfg.ckpt_dir);

    let devices = cfg.devices();
    let persistent_learner_enabled = cfg.persistent_actors;
    if persistent_learner_enabled && hub_vs.is_none() {
        let snapshot = nn::VarStore::new(Device::Cpu);
        let _ = PolicyNet::new_with_recurrence(
            &snapshot.root(),
            cfg.amp,
            cfg.foveate,
            cfg.gc,
            cfg.blocks,
            cfg.recurrent_policy,
        );
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
                let (w, obs) = spawn_worker(
                    idx,
                    start_stage,
                    cfg.max_episode_ticks,
                    worker_engine,
                    cfg.reward_config,
                    cfg.curriculum_schedule,
                )?;
                workers.push(w);
                cur_obs.push(obs);
            }
        }
        let mut learner_vs = nn::VarStore::new(device);
        let learner_policy = PolicyNet::new_with_recurrence(
            &learner_vs.root(),
            cfg.amp,
            cfg.foveate,
            cfg.gc,
            cfg.blocks,
            cfg.recurrent_policy,
        );
        if let Some(hub) = &hub_vs {
            learner_vs.copy(hub)?;
        } else {
            hub_vs = Some({
                // Keep a CPU-independent handle to shard 0's weights so
                // every later shard starts from bit-identical initial
                // parameters (VarStore::copy handles the cross-device
                // transfer).
                let mut snapshot = nn::VarStore::new(Device::Cpu);
                let _ = PolicyNet::new_with_recurrence(
                    &snapshot.root(),
                    cfg.amp,
                    cfg.foveate,
                    cfg.gc,
                    cfg.blocks,
                    cfg.recurrent_policy,
                );
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
            let actor_policy = PolicyNet::new_with_recurrence(
                &actor_vs.root(),
                cfg.amp,
                cfg.foveate,
                cfg.gc,
                cfg.blocks,
                cfg.recurrent_policy,
            );
            actor_vs.copy(&learners[gi].vs)?;
            if cfg.amp {
                cast_actor_inference_weights_bf16(&actor_vs);
            }
            let path = std::path::Path::new(&cfg.ae_ckpt);
            if !path.exists() {
                anyhow::bail!(
                    "AE checkpoint not found at {} - run `ofae train` / `bash scripts/fetch_ae_encoders.sh` \
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
                ready_rx: None,
                compact_host_arena: Arc::new(crate::vecenv::CompactHostArena::default()),
                vs: actor_vs,
                policy: actor_policy,
                recurrent: None,
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
    // A prior process may have intentionally exited for this exact target.
    // Once every equal shard is live, the machine-readable request is
    // fulfilled and can be removed.
    let request_path = restart_request_path(&cfg.ckpt_dir);
    if std::path::Path::new(&request_path).exists() {
        std::fs::remove_file(&request_path)?;
        println!(
            "[train] fulfilled stage env resize request at {} envs/shard",
            cfg.num_envs
        );
    }

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
    let stages = ofcore::curriculum::stages_for_schedule(cfg.curriculum_schedule);
    let mut curr_stage = start_stage;
    let mut recent_wins: std::collections::VecDeque<f64> = resumed_state
        .as_ref()
        .map(|s| s.recent_wins.iter().copied().collect())
        .unwrap_or_else(|| std::collections::VecDeque::with_capacity(ofcore::curriculum::WINDOW));
    let mut recent_conversions: std::collections::VecDeque<f64> = resumed_state
        .as_ref()
        .map(|s| s.recent_conversions.iter().copied().collect())
        .unwrap_or_else(|| std::collections::VecDeque::with_capacity(ofcore::curriculum::WINDOW));
    let mut recent_deaths: std::collections::VecDeque<f64> = resumed_state
        .as_ref()
        .map(|s| s.recent_deaths.iter().copied().collect())
        .unwrap_or_else(|| std::collections::VecDeque::with_capacity(ofcore::curriculum::WINDOW));
    let return_stats = resumed_state
        .as_ref()
        .and_then(|state| state.return_stats.clone());
    let mut lr_now = resumed_state.as_ref().map(|s| s.lr_now).unwrap_or(cfg.lr);
    // Adaptive entropy-floor multiplier (port of `rl/ppo.py`'s
    // `ent_scale`): multiplicative on top of the linear anneal, nudged
    // after each update from that update's measured mean entropy.
    let mut ent_scale: f64 = resumed_state.as_ref().map(|s| s.ent_scale).unwrap_or(1.0);
    let start_update = resumed_state.as_ref().map(|s| s.update).unwrap_or(0);
    // V10 warmup resets on every curriculum advance too, not just the run's
    // very start, since the value function faces the same "brand new data
    // distribution" shock either way.
    // After `--resume`, AdamW moments are cold (tch cannot restore them);
    // `--resume-warmup-updates` (default 100) sets the first warmup length.
    // 0 → fall back to the ordinary V10 stage warmup.
    let mut lr_warmup_updates = if cfg.resume.is_some() && cfg.resume_warmup_updates > 0 {
        cfg.resume_warmup_updates
    } else {
        ofcore::curriculum::V10_LR_WARMUP_UPDATES
    };
    let mut lr_warmup_start_update = start_update;
    let mut current_best = resumed_state
        .as_ref()
        .and_then(|state| state.best_eval_win.zip(state.best_eval_score));
    let mut async_eval = match (cfg.async_eval, cfg.eval_device) {
        (true, Some(device)) if cfg.eval_every > 0 => {
            Some(AsyncEval::spawn(cfg.clone(), device, current_best)?)
        }
        _ => None,
    };
    let mut requested_env_target: Option<usize> = None;
    let mut resize_reason = String::new();

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
        if let Some(eval) = async_eval.as_mut() {
            if let Some(completion) = eval.poll()? {
                current_best = report_eval_completion(completion, update, &metrics)?;
            }
        }
        let expected_pending_version = if update == start_update {
            start_update
        } else {
            update - 1
        };
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
        // LR warmup - a no-op once warmup
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
        let (
            train_result,
            next_pending,
            train_dt,
            learner_weights,
            learner_snapshot_dt,
            batch_build_dt,
            gradient_sync_dt,
        ) = if persistent_learner_enabled {
            for actor in &mut persistent_actors {
                actor.send_collect()?;
            }
            let train_t0 = Instant::now();
            let train_reply = persistent_learner
                .as_mut()
                .expect("persistent learner initialized")
                .0
                .train(
                    std::mem::take(&mut pending),
                    lr_now * warmup_frac,
                    ent_coef_now,
                );
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
                    reply.timings.batch_build_seconds,
                    reply.timings.gradient_sync_seconds,
                ),
                Err(error) => (
                    Err(error),
                    next_pending,
                    train_t0.elapsed().as_secs_f64(),
                    None,
                    0.0,
                    0.0,
                    0.0,
                ),
            }
        } else if cfg.persistent_actors {
            for actor in &mut persistent_actors {
                actor.send_collect()?;
            }
            let train_t0 = Instant::now();
            let mut timings = TrainTimings::default();
            let train_result = train_update(
                &mut learners,
                &mut pending,
                cfg_ref,
                rng.as_mut().expect("legacy learner RNG available"),
                ent_coef_now,
                &mut ret_stat,
                false,
                None,
                None,
                &mut timings,
            );
            let train_dt = train_t0.elapsed().as_secs_f64();
            let next_pending = persistent_actors
                .iter_mut()
                .map(PersistentActor::finish_collect)
                .collect::<Result<Vec<_>>>();
            (
                train_result,
                next_pending,
                train_dt,
                None,
                0.0,
                timings.batch_build_seconds,
                timings.gradient_sync_seconds,
            )
        } else {
            std::thread::scope(|s| {
                let collect_handles: Vec<_> = actors
                    .iter_mut()
                    .map(|actor| s.spawn(move || collect_rollout(actor, cfg_ref, update)))
                    .collect();
                let train_t0 = Instant::now();
                let mut timings = TrainTimings::default();
                let train_result = train_update(
                    &mut learners,
                    &mut pending,
                    cfg_ref,
                    rng.as_mut().expect("legacy learner RNG available"),
                    ent_coef_now,
                    &mut ret_stat,
                    false,
                    None,
                    None,
                    &mut timings,
                );
                let train_dt = train_t0.elapsed().as_secs_f64();
                let next_pending: Result<Vec<RolloutResult>> = collect_handles
                    .into_iter()
                    .map(|h| h.join().map_err(|_| anyhow!("collector thread panicked"))?)
                    .collect();
                (
                    train_result,
                    next_pending,
                    train_dt,
                    None,
                    0.0,
                    timings.batch_build_seconds,
                    timings.gradient_sync_seconds,
                )
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
        let mut demoted = false;
        let mut curriculum_transitions: Vec<CurriculumTransition> = Vec::new();
        let debug_eps = std::env::var("OFTRAIN_DEBUG_EPISODES").is_ok();
        for result in &next_pending {
            for info in &result.ep_infos {
                if debug_eps {
                    eprintln!(
                        "[ep] reward={:.3} components[str={:.3} delta={:.3} dom={:.3} churn={:.3} waste={:.3} death={:.3} terminal={:.3}] churn_pairs[boat_cancel={} embargo_stop={} attack_retreat={} retreat_attack={}] len={} tiles={:.1} tick={} place={}/{} score={:.3} won={} wasted={} stage={} rehearsal={} map={}",
                        info.reward,
                        info.reward_components.strength,
                        info.reward_components.strength_delta,
                        info.reward_components.dominance,
                        info.reward_components.action_churn,
                        info.reward_components.waste,
                        info.reward_components.death,
                        info.reward_components.terminal,
                        info.action_pair_counts.boat_cancel_boat,
                        info.action_pair_counts.embargo_embargo_stop,
                        info.action_pair_counts.attack_retreat,
                        info.action_pair_counts.retreat_attack,
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
                if let Err(e) = metrics.log(&serde_json::json!({
                    "event": "episode",
                    "update": update,
                    "stage": info.stage,
                    "map": &info.map,
                    "rehearsal": info.rehearsal,
                    "reward": info.reward,
                    "reward/strength": info.reward_components.strength,
                    "reward/strength_delta": info.reward_components.strength_delta,
                    "reward/dominance": info.reward_components.dominance,
                    "reward/closeout": info.reward_components.closeout,
                    "reward/action_churn": info.reward_components.action_churn,
                    "reward/boat_outcome": info.reward_components.boat_outcome,
                    "reward/tempo": info.reward_components.tempo,
                    "reward/embargo_outcome": info.reward_components.embargo_outcome,
                    "reward/combat_outcome": info.reward_components.combat_outcome,
                    "reward/survival": info.reward_components.survival,
                    "reward/diplo_panic": info.reward_components.diplo_panic,
                    "reward/combat_action": info.reward_components.combat_action,
                    "reward/attack_commit": info.reward_components.attack_commit,
                    "reward/waste": info.reward_components.waste,
                    "reward/death": info.reward_components.death,
                    "reward/terminal": info.reward_components.terminal,
                    "action_pairs/boat_cancel_boat": info.action_pair_counts.boat_cancel_boat,
                    "boats/useful_landing": info.boat_outcome_counts.useful_landing,
                    "boats/own_shore_return": info.boat_outcome_counts.own_shore_return,
                    "boats/cancelled": info.boat_outcome_counts.cancelled,
                    "boats/destroyed": info.boat_outcome_counts.destroyed,
                    "embargo/bad_stops": info.embargo_bad_stops,
                    "embargo/good_stops": info.embargo_good_stops,
                    "combat/premature_retreats": info.premature_retreats,
                    "combat/thrash_reengages": info.thrash_reengages,
                    "action_pairs/embargo_embargo_stop": info.action_pair_counts.embargo_embargo_stop,
                    "action_pairs/attack_retreat": info.action_pair_counts.attack_retreat,
                    "action_pairs/retreat_attack": info.action_pair_counts.retreat_attack,
                    "action_pairs/total": info.action_pair_counts.total(),
                    "final_land_share": info.final_land_share,
                    "max_land_share": info.max_land_share,
                    "closeout_reached": info.closeout_reached as u8,
                    "closeout_entry_tick": info.closeout_entry_tick,
                    "decisions_after_closeout": info.decisions_after_closeout,
                    "converted": info.converted as u8,
                    "timeout_after_closeout": info.timeout_after_closeout as u8,
                    "post_closeout_churn_pairs": info.post_closeout_churn_pairs,
                })) {
                    eprintln!("[train] WARNING: episode reward-component log failed: {e:#}");
                }
                ep_rewards.push(info.reward);
                ep_lengths.push(info.length);
                if record_advancement_result(
                    cfg.curriculum_schedule,
                    curr_stage,
                    info.stage,
                    info.rehearsal,
                    info.won,
                    info.died,
                    info.closeout_reached,
                    info.converted,
                    &mut recent_wins,
                    &mut recent_conversions,
                    &mut recent_deaths,
                ) {
                    let win_rate = recent_wins.iter().sum::<f64>() / recent_wins.len() as f64;
                    if debug_eps {
                        eprintln!(
                            "[win_rate] {:.3} (window={}/{})",
                            win_rate,
                            recent_wins.len(),
                            ofcore::curriculum::WINDOW
                        );
                    }
                    let gate_passed = should_advance_v10(
                        curr_stage,
                        &recent_wins,
                        &recent_conversions,
                        &recent_deaths,
                        stages[curr_stage].win_at,
                    );
                    let demote = should_demote_v10(curr_stage, &recent_wins, &recent_deaths);
                    if demote {
                        let from_stage = curr_stage;
                        let win_rate = window_mean(&recent_wins).unwrap_or(0.0);
                        let conversion_rate = window_mean(&recent_conversions).unwrap_or(0.0);
                        let death_rate = window_mean(&recent_deaths).unwrap_or(0.0);
                        let win_gate = stages[from_stage].win_at;
                        let window_size = recent_wins.len();
                        curr_stage -= 1;
                        curriculum_transitions.push(CurriculumTransition {
                            event: "demote",
                            from_stage,
                            to_stage: curr_stage,
                            win_rate,
                            conversion_rate,
                            death_rate,
                            win_gate,
                            window_size,
                        });
                        recent_wins.clear();
                        recent_conversions.clear();
                        recent_deaths.clear();
                        demoted = true;
                    } else if curr_stage < stages.len() - 1 && gate_passed {
                        let from_stage = curr_stage;
                        let win_rate = window_mean(&recent_wins).unwrap_or(0.0);
                        let conversion_rate = window_mean(&recent_conversions).unwrap_or(0.0);
                        let death_rate = window_mean(&recent_deaths).unwrap_or(0.0);
                        let win_gate = stages[from_stage].win_at;
                        let window_size = recent_wins.len();
                        curr_stage += 1;
                        curriculum_transitions.push(CurriculumTransition {
                            event: "advance",
                            from_stage,
                            to_stage: curr_stage,
                            win_rate,
                            conversion_rate,
                            death_rate,
                            win_gate,
                            window_size,
                        });
                        recent_wins.clear();
                        recent_conversions.clear();
                        recent_deaths.clear();
                        advanced = true;
                    }
                }
            }
        }
        // Performance-scaled LR: inverse to how close we are to the gate.
        // Stage advance/demote still resets warmup; every update recomputes
        // lr_now from stage baseline × (gate - wr) boost.
        {
            let wr = window_mean(&recent_wins);
            let gate = stages[curr_stage].win_at;
            lr_now = ofcore::curriculum::effective_learning_rate(
                cfg.lr,
                cfg.stage_lr_decay,
                curr_stage,
                cfg.stage_lr_floor,
                wr,
                gate,
                cfg.lr_perf_max_boost,
            );
        }
        if advanced || demoted {
            lr_warmup_start_update = update + 1;
            lr_warmup_updates = ofcore::curriculum::V10_LR_WARMUP_UPDATES;
            let live_envs_per_shard = live_total_envs / devices.len();
            requested_env_target = requested_stage_env_target_for_resize(
                &cfg.stage_env_targets,
                curr_stage,
                live_envs_per_shard,
                cfg.auto_scale_envs,
                cfg.max_envs,
            );
            if requested_env_target.is_none() {
                if let Some(floor) = cfg.stage_env_targets.get(curr_stage).copied() {
                    let capped = clamp_resolved_envs_to_autoscale_max(
                        floor,
                        cfg.auto_scale_envs,
                        cfg.max_envs,
                    );
                    if floor != live_envs_per_shard && capped == live_envs_per_shard {
                        println!(
                            "[train] stage {curr_stage} env floor {floor} capped to \
                             max_envs={} (live={live_envs_per_shard}); skipping noop resize",
                            if cfg.auto_scale_envs { cfg.max_envs } else { floor }
                        );
                    }
                }
            }
            if requested_env_target.is_some() {
                resize_reason = if demoted {
                    "curriculum_demote_env_target".to_string()
                } else {
                    "curriculum_stage_env_target".to_string()
                };
            }
            if requested_env_target.is_none() && cfg.persistent_actors {
                for actor in &mut persistent_actors {
                    actor.set_stage(curr_stage)?;
                }
            } else if requested_env_target.is_none() {
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
            if demoted {
                println!(
                    "=== curriculum demote -> stage {curr_stage}: maps={:?} bots={} {} lr->{lr_now:.2e}",
                    st.maps, st.bots, st.difficulty
                );
            } else {
                println!(
                    "=== curriculum advance -> stage {curr_stage}: maps={:?} bots={} {} lr->{lr_now:.2e}",
                    st.maps, st.bots, st.difficulty
                );
            }
            if let Some(target) = requested_env_target {
                println!(
                    "[train] stage {curr_stage} requests {target} envs/shard;                      checkpointing at update boundary for a persistent-owner-safe restart"
                );
            }
            // Persist one immutable milestone per advance/demote so HF sync
            // can keep a card-backed historical trail (not just latest.*).
            for transition in &curriculum_transitions {
                let note = curriculum_transition_note(
                    transition,
                    update,
                    cfg.curriculum_schedule,
                    &stages,
                );
                let path = curriculum_milestone_path(&cfg.ckpt_dir, transition, update);
                let state = TrainState {
                    checkpoint_schema_version: if cfg.recurrent_policy { 2 } else { 1 },
                    hidden_reset_policy: if cfg.recurrent_policy {
                        "episode_done".to_string()
                    } else {
                        "none".to_string()
                    },
                    update: update + 1,
                    stage: transition.to_stage,
                    ent_scale,
                    lr_now,
                    total_env_steps,
                    recent_wins: recent_wins.iter().copied().collect(),
                    recent_conversions: recent_conversions.iter().copied().collect(),
                    recent_deaths: recent_deaths.iter().copied().collect(),
                    best_eval_win: current_best.map(|best| best.0),
                    best_eval_score: current_best.map(|best| best.1),
                    curriculum_schedule: Some(cfg.curriculum_schedule.id().to_string()),
                    reward_profile: Some(cfg.reward_config.reward_profile_id().to_string()),
                    return_stats: return_stats.clone(),
                    stage_env_targets: cfg.stage_env_targets.clone(),
                    envs_per_shard: live_total_envs / devices.len(),
                    requested_env_target: None,
                };
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
                let note_path = note_sidecar_path(&path);
                save_atomic(&note_path, |tmp| {
                    std::fs::write(tmp, serde_json::to_string_pretty(&note)?)?;
                    Ok(())
                })?;
                println!(
                    "[train] curriculum {} milestone saved: {} ({})",
                    transition.event,
                    path,
                    note["summary"].as_str().unwrap_or("")
                );
            }
        }
        total_env_steps += (live_total_envs * cfg.rollout_len) as u64;
        let eval_due = cfg.eval_every > 0 && update % cfg.eval_every == 0;
        let can_submit_async = async_eval
            .as_ref()
            .is_some_and(|eval| !eval.flight.is_busy());
        let eval_weights = if eval_due && (async_eval.is_none() || can_submit_async) {
            Some(if persistent_learner_enabled {
                learner_weights
                    .as_ref()
                    .ok_or_else(|| anyhow!("eval requires persistent learner weights"))?
                    .first()
                    .cloned()
                    .ok_or_else(|| anyhow!("eval requires at least one learner weight snapshot"))?
            } else {
                snapshot_weights(&learners[0].vs)?
            })
        } else {
            None
        };

        // Refresh every actor from its paired learner's just-updated
        // weights, now that training has finished (and the collection
        // that ran concurrently with it is done reading the *old*
        // weights) - the next update's collection will use these.
        let refresh_start = Instant::now();
        if requested_env_target.is_none() && cfg.persistent_actors {
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
        } else if requested_env_target.is_none() {
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
        let actor_batch_stats =
            next_pending
                .iter()
                .fold(ActorBatchStats::default(), |mut total, result| {
                    let stats = &result.actor_batches;
                    total.dispatches += stats.dispatches;
                    total.observations += stats.observations;
                    total.singletons += stats.singletons;
                    total.shape_dispatches += stats.shape_dispatches;
                    total.padded_cells += stats.padded_cells;
                    total.allocated_cells += stats.allocated_cells;
                    total
                });
        if actor_batch_stats.dispatches > 0 {
            metrics.log_actor_batches(
                update,
                actor_batch_stats.mean_size(),
                actor_batch_stats.singleton_fraction(),
                actor_batch_stats.shapes_per_dispatch(),
                actor_batch_stats.dispatches,
                actor_batch_stats.padding_ratio(),
            )?;
        }
        pending = next_pending;
        eprintln!(
            "[phase] update {update} train+collect+refresh finished in {:.3}s \
             (train={train_dt:.3}s collect={collect_dt:.3}s refresh={refresh_dt:.3}s)",
            update_start.elapsed().as_secs_f64()
        );

        // Auto-scale check: after this update's pending swap and *before*
        // the next collect. Persistent owners cannot live-spawn workers
        // (CUDA ownership stays on the actor thread), so growth sets
        // `requested_env_target` and shares the restart path below with
        // curriculum stage env targets. Legacy collectors still grow
        // in-process.
        if requested_env_target.is_none()
            && cfg.auto_scale_envs
            && update % cfg.autoscale_check_every.max(1) == 0
        {
            let gpu_snap = gpu_sampler.as_ref().map(|g| g.snapshot());
            let gpu_util_frac = gpu_snap.as_ref().map(|s| s.min_mean_util() / 100.0);
            let gpu_mem_frac = gpu_snap.as_ref().map(|s| s.mem_pct / 100.0);
            let current = live_total_envs / devices.len().max(1);
            let target_n = autoscale::next_env_count(
                current,
                gpu_util_frac,
                gpu_mem_frac,
                cfg.target_gpu_util,
                autoscale_min_envs,
                autoscale_max_envs,
                cfg.autoscale_step,
            );
            let gpu_str = format!(
                "util={} mem={}",
                gpu_util_frac
                    .map(|f| format!("{:.1}%", f * 100.0))
                    .unwrap_or_else(|| "n/a".to_string()),
                gpu_mem_frac
                    .map(|f| format!("{:.1}%", f * 100.0))
                    .unwrap_or_else(|| "n/a".to_string()),
            );
            if target_n < current {
                if cfg.persistent_actors {
                    requested_env_target = Some(target_n);
                    resize_reason = "gpu_mem_autoscale".to_string();
                    println!(
                        "[autoscale] persistent: {current} -> {target_n} envs/shard \
                         ({gpu_str}; VRAM shrink); checkpointing for restart"
                    );
                } else {
                    println!(
                        "[autoscale] mem pressure wants {current} -> {target_n} envs/shard \
                         ({gpu_str}) but legacy live-shrink is unsupported; holding"
                    );
                }
            } else if target_n > current {
                if cfg.persistent_actors {
                    requested_env_target = Some(target_n);
                    resize_reason = "gpu_util_autoscale".to_string();
                    println!(
                        "[autoscale] persistent: {current} -> {target_n} envs/shard \
                         ({gpu_str} target={:.0}%); checkpointing for restart",
                        cfg.target_gpu_util * 100.0
                    );
                } else {
                    let add = target_n - current;
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
                                cfg.reward_config,
                                cfg.curriculum_schedule,
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
                                "[autoscale] all shards: {current} -> {target_n} envs ({gpu_str} \
                                 target={:.0}% cpu_cap={autoscale_max_envs})",
                                cfg.target_gpu_util * 100.0
                            );
                        }
                    }
                }
            }
        }

        if let Some(target) = requested_env_target {
            if let Some(eval) = async_eval.take() {
                if let Some(completion) = eval.shutdown()? {
                    current_best = report_eval_completion(completion, update, &metrics)?;
                }
            }
            let current_envs_per_shard = live_total_envs / devices.len();
            let reason = if resize_reason.is_empty() {
                "curriculum_stage_env_target".to_string()
            } else {
                resize_reason.clone()
            };
            let state = TrainState {
                checkpoint_schema_version: if cfg.recurrent_policy { 2 } else { 1 },
                hidden_reset_policy: if cfg.recurrent_policy {
                    "episode_done".to_string()
                } else {
                    "none".to_string()
                },
                update: update + 1,
                stage: curr_stage,
                ent_scale,
                lr_now,
                total_env_steps,
                recent_wins: recent_wins.iter().copied().collect(),
                recent_conversions: recent_conversions.iter().copied().collect(),
                recent_deaths: recent_deaths.iter().copied().collect(),
                best_eval_win: current_best.map(|best| best.0),
                best_eval_score: current_best.map(|best| best.1),
                curriculum_schedule: Some(cfg.curriculum_schedule.id().to_string()),
                reward_profile: Some(cfg.reward_config.reward_profile_id().to_string()),
                return_stats: return_stats.clone(),
                stage_env_targets: cfg.stage_env_targets.clone(),
                envs_per_shard: current_envs_per_shard,
                requested_env_target: Some(target),
            };
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
            save_policy_manifest(&cfg, &state)?;
            let request = EnvResizeRequest {
                format: 1,
                reason,
                update: update + 1,
                stage: curr_stage,
                current_envs_per_shard,
                requested_envs_per_shard: target,
                num_shards: devices.len(),
                checkpoint: latest,
            };
            let request_path = restart_request_path(&cfg.ckpt_dir);
            save_atomic(&request_path, |tmp| {
                std::fs::write(tmp, serde_json::to_string_pretty(&request)?)?;
                Ok(())
            })?;
            if let Some((learner, _)) = persistent_learner.take() {
                learner.shutdown()?;
            }
            if cfg.persistent_actors {
                for actor in persistent_actors {
                    actor.shutdown()?;
                }
            } else {
                for actor in actors {
                    for worker in actor.workers {
                        drop(worker.choice_tx);
                        let _ = worker.handle.join();
                    }
                }
            }
            println!(
                "[train] resize checkpoint complete: {request_path}; exiting cleanly \
                 for restart at {target} envs/shard"
            );
            return Ok(());
        }

        // Legacy live autoscale already applied above when !persistent_actors.

        if eval_due {
            let win_rate = if recent_wins.is_empty() {
                None
            } else {
                Some(recent_wins.iter().sum::<f64>() / recent_wins.len() as f64)
            };
            let report = EvalReportContext {
                losses: last_losses,
                win_rate,
                lr_now,
                total_env_steps,
            };
            let state = TrainState {
                checkpoint_schema_version: if cfg.recurrent_policy { 2 } else { 1 },
                hidden_reset_policy: if cfg.recurrent_policy {
                    "episode_done".to_string()
                } else {
                    "none".to_string()
                },
                update: update + 1,
                stage: curr_stage,
                ent_scale,
                lr_now,
                total_env_steps,
                recent_wins: recent_wins.iter().copied().collect(),
                recent_conversions: recent_conversions.iter().copied().collect(),
                recent_deaths: recent_deaths.iter().copied().collect(),
                best_eval_win: current_best.map(|best| best.0),
                best_eval_score: current_best.map(|best| best.1),
                curriculum_schedule: Some(cfg.curriculum_schedule.id().to_string()),
                reward_profile: Some(cfg.reward_config.reward_profile_id().to_string()),
                return_stats: return_stats.clone(),
                stage_env_targets: cfg.stage_env_targets.clone(),
                envs_per_shard: live_total_envs / devices.len(),
                requested_env_target: None,
            };
            if let Some(eval) = async_eval.as_mut() {
                if let Some(weights) = eval_weights {
                    let submitted = eval.submit(EvalJob {
                        update,
                        stage: curr_stage,
                        weights,
                        state,
                        report,
                    })?;
                    debug_assert!(submitted);
                    eprintln!(
                        "[phase] update {update} asynchronous evaluation submitted \
                         (stage={curr_stage}, episodes={})",
                        cfg.eval_episodes
                    );
                } else {
                    eprintln!(
                        "[eval] update {update} skipped: previous asynchronous evaluation \
                         is still in flight"
                    );
                }
            } else {
                let weights = eval_weights
                    .ok_or_else(|| anyhow!("synchronous eval missing weight snapshot"))?;
                let started = Instant::now();
                eprintln!(
                    "[phase] update {update} synchronous evaluation started \
                     (stage={curr_stage}, episodes={})",
                    cfg.eval_episodes
                );
                let result = if cfg.persistent_actors {
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
                        cfg.recurrent_policy,
                        cfg.reward_config,
                        cfg.curriculum_schedule,
                    )?
                };
                let promoted = eval_is_better(&result, current_best);
                let best = if promoted {
                    let next_best = Some((result.win, result.score));
                    let mut promoted_state = state;
                    promoted_state.best_eval_win = Some(result.win);
                    promoted_state.best_eval_score = Some(result.score);
                    save_snapshot_checkpoint(
                        &weights,
                        &cfg,
                        &best_eval_path(&cfg.ckpt_dir),
                        &promoted_state,
                    )?;
                    next_best
                } else {
                    current_best
                };
                current_best = report_eval_completion(
                    EvalCompletion {
                        update,
                        stage: curr_stage,
                        result,
                        elapsed_seconds: started.elapsed().as_secs_f64(),
                        promoted,
                        best,
                        report,
                    },
                    update,
                    &metrics,
                )?;
            }
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
                 batch_build_s={:.3} gradient_sync_s={:.3} learner_snapshot_s={:.3} \
                 refresh_s={:.3}{gpu_str}",
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
                batch_build_dt,
                gradient_sync_dt,
                learner_snapshot_dt,
                refresh_dt,
            );
            if let Err(e) = metrics.log(&serde_json::json!({
                "event": "host_pipeline_timing",
                "update": update,
                "timing/batch_build_s": batch_build_dt,
                "timing/gradient_sync_s": gradient_sync_dt,
            })) {
                eprintln!("[train] WARNING: host-pipeline timing log failed: {e:#}");
            }
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
                None,
                None,
            ) {
                eprintln!("[train] WARNING: metrics log failed: {e:#}");
            }
        }

        if cfg.ckpt_every > 0 && (update % cfg.ckpt_every == 0) && update > 0 {
            let state = TrainState {
                checkpoint_schema_version: if cfg.recurrent_policy { 2 } else { 1 },
                hidden_reset_policy: if cfg.recurrent_policy {
                    "episode_done".to_string()
                } else {
                    "none".to_string()
                },
                update: update + 1, // resume must start at the *next* update, not repeat this one
                stage: curr_stage,
                ent_scale,
                lr_now,
                total_env_steps,
                recent_wins: recent_wins.iter().copied().collect(),
                recent_conversions: recent_conversions.iter().copied().collect(),
                recent_deaths: recent_deaths.iter().copied().collect(),
                best_eval_win: current_best.map(|best| best.0),
                best_eval_score: current_best.map(|best| best.1),
                curriculum_schedule: Some(cfg.curriculum_schedule.id().to_string()),
                reward_profile: Some(cfg.reward_config.reward_profile_id().to_string()),
                return_stats: return_stats.clone(),
                stage_env_targets: cfg.stage_env_targets.clone(),
                envs_per_shard: live_total_envs / devices.len(),
                requested_env_target: None,
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
            save_policy_manifest(&cfg, &state)?;
            println!("[train] checkpoint saved: {path} (update={})", state.update);
            if let Err(e) = prune_numbered_checkpoints(&cfg.ckpt_dir, cfg.ckpt_keep_last) {
                eprintln!("[train] WARNING: checkpoint prune failed: {e:#}");
            }
        }
    }

    if let Some(eval) = async_eval.take() {
        if let Some(completion) = eval.shutdown()? {
            current_best = report_eval_completion(completion, cfg.updates, &metrics)?;
        }
    }
    let final_state = TrainState {
        checkpoint_schema_version: if cfg.recurrent_policy { 2 } else { 1 },
        hidden_reset_policy: if cfg.recurrent_policy {
            "episode_done".to_string()
        } else {
            "none".to_string()
        },
        update: cfg.updates,
        stage: curr_stage,
        ent_scale,
        lr_now,
        total_env_steps,
        recent_wins: recent_wins.iter().copied().collect(),
        recent_conversions: recent_conversions.iter().copied().collect(),
        recent_deaths: recent_deaths.iter().copied().collect(),
        best_eval_win: current_best.map(|best| best.0),
        best_eval_score: current_best.map(|best| best.1),
        curriculum_schedule: Some(cfg.curriculum_schedule.id().to_string()),
        reward_profile: Some(cfg.reward_config.reward_profile_id().to_string()),
        return_stats,
        stage_env_targets: cfg.stage_env_targets.clone(),
        envs_per_shard: pending
            .iter()
            .map(|rollout| rollout.buffer.first().map(|row| row.len()).unwrap_or(0))
            .sum::<usize>()
            / devices.len(),
        requested_env_target: None,
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
    save_policy_manifest(&cfg, &final_state)?;
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
                    meta: Vec::new().into(),
                    values: vec![1.0, 2.0, 3.0].into(),
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
                    meta: Vec::new().into(),
                    values: Vec::new().into(),
                },
            })
            .unwrap_err();
        assert!(err.to_string().contains("stale actor policy refresh"));
    }

    #[test]
    fn stage_resize_boundary_uses_ordered_shutdown_not_a_live_resize_command() {
        let mut protocol = ActorProtocol::new(12);
        protocol.accept(&ActorCommand::Collect { id: 1 }).unwrap();
        protocol.accept(&ActorCommand::Shutdown { id: 2 }).unwrap();
        assert!(protocol.stopped);
        assert!(
            protocol.accept(&ActorCommand::Collect { id: 3 }).is_err(),
            "a resize restart must leave no old actor accepting work"
        );
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
    fn actor_inference_weights_cast_to_bf16_and_refresh_preserves_kind() {
        tch::manual_seed(92);
        let source = nn::VarStore::new(Device::Cpu);
        let _ = PolicyNet::new(&source.root(), true, false, 8, 1);
        let snapshot = snapshot_weights(&source).unwrap();

        let actor = nn::VarStore::new(Device::Cpu);
        let _ = PolicyNet::new(&actor.root(), true, false, 8, 1);
        apply_weight_snapshot(&actor, &snapshot).unwrap();
        cast_actor_inference_weights_bf16(&actor);
        for (name, tensor) in actor.variables() {
            if actor_inference_weight_bf16(&name) {
                assert_eq!(tensor.kind(), Kind::BFloat16, "{name}");
            } else {
                assert_eq!(tensor.kind(), Kind::Float, "{name}");
            }
        }

        // Learner snapshot is f32; refresh must cast into the BF16 slots.
        apply_weight_snapshot(&actor, &snapshot).unwrap();
        for (name, tensor) in actor.variables() {
            if actor_inference_weight_bf16(&name) {
                assert_eq!(tensor.kind(), Kind::BFloat16, "{name}");
            }
        }
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
            actor_batches: ActorBatchStats::default(),
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
        protocol
            .accept(&LearnerCommand::Shutdown { id: 3 })
            .unwrap();
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
            curriculum_schedule: ofcore::curriculum::CurriculumSchedule::V10,
            migrate_v86_to_v10: false,
            stage_env_targets: Vec::new(),
            max_episode_ticks: 10,
            rollout_len: 2,
            updates: 1,
            lr: 1e-4,
            gamma: 0.99,
            reward_config: RewardConfig {
                gamma: 0.99,
                v81_dom_coef: 0.0,
                v81_min_stage: 4,
                v81_potential_clamp: 2.0,
                v81_dominant_loss: false,
                v81_dominance_threshold: 0.55,
                v81_delta_loss_dominant: 5.25,
                v81_churn_coef: 0.0,
                v81_churn_window: 2,
                v81_churn_min_stage: 4,
                v83_close_coef: 4.0,
                v83_churn_coef: 0.06,
                v84_boat_useful: 0.0,
                v84_boat_destroyed: 0.0,
                v84_boat_cancelled: 0.0,
                v84_boat_own_shore: 0.0,
                v84_boat_min_stage: 4,
                v84_tempo_coef: 0.0,
                v84_tempo_min_stage: 4,
                v84_fast_win_coef: 0.0,
                v85_tempo_share_threshold: 0.0,
                v85_extra_win_bonus: 0.0,
                v85_embargo_bad_stop: 0.0,
                v85_embargo_good_stop: 0.0,
                v85_embargo_min_stage: 4,
                v85_premature_retreat: 0.0,
                v85_thrash_reengage: 0.0,
                v85_combat_min_stage: 4,
                v86_delta_loss: 0.0,
                v86_attack_symmetric_loss: false,
                v86_skip_combat_churn: false,
                v86_death_penalty: 0.0,
                v10_survival_coef: 0.0,
                v10_diplo_panic: 0.0,
                v10_diplo_panic_share: 0.35,
                v10_diplo_panic_tick_frac: 0.55,
                v10_combat_action: 0.0,
                v10_attack_commit: 0.0,
                v10_attack_switch: 0.0,
                v10_timeout_closeout: 0.0,
                v10_closeout_entry: 0.0,
            },
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
            stage_lr_floor: ofcore::curriculum::V10_STAGE_LR_FLOOR,
            lr_perf_max_boost: 4.0,
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
            recurrent_policy: false,
            recurrent_hidden_size: 256,
            bptt_chunk_len: 16,
            work_conserving_actors: false,
            actor_max_batch: 32,
            actor_target_batch: 8,
            actor_max_padding_waste: 0.35,
            actor_max_wait: Duration::from_millis(15),
            device: Device::Cpu,
            engine: EngineKind::Native,
            node_fraction: 0.0,
            log_every: 1,
            eval_every: 0,
            eval_episodes: 0,
            async_eval: false,
            eval_device: None,
            ckpt_every: 0,
            ckpt_dir: String::new(),
            ckpt_keep_last: 0,
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
            prev_action: ActionOutcome::default(),
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
                fallout: crate::ae::pack_fallout(&vec![0u8; gh * 8 * gw * 8], gh * 8, gw * 8),
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
            legal_ptarget: vec![1.0; ofcore::feat::N_ACTIONS * ofcore::feat::MAX_SLOTS],
            legal_build: [1.0; ofcore::feat::N_BUILD],
            legal_nuke: [1.0; ofcore::feat::N_NUKE],
            local: vec![0.1; 5 * policy::LOCAL as usize * policy::LOCAL as usize],
        }
    }

    fn parity_rollout() -> RolloutResult {
        let wait = ACTIONS.iter().position(|&action| action == "noop").unwrap() as i64;
        let step = |reward| Step {
            obs: parity_obs(),
            hidden_in: Vec::new(),
            context: ActionOutcome::default(),
            outcome: ActionOutcome::default(),
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
            actor_batches: ActorBatchStats::default(),
        }
    }

    fn parity_learner(cfg: &Config, weights: &CpuWeightSnapshot) -> LearnerShard {
        let vs = nn::VarStore::new(Device::Cpu);
        let policy = PolicyNet::new_with_recurrence(
            &vs.root(),
            false,
            false,
            cfg.gc,
            cfg.blocks,
            cfg.recurrent_policy,
        );
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
        let mut legacy_timings = TrainTimings::default();
        let mut owned_timings = TrainTimings::default();
        let mut legacy_rollout = parity_rollout();
        let mut owned_rollout = parity_rollout();

        let legacy_losses = train_update(
            std::slice::from_mut(&mut legacy),
            std::slice::from_mut(&mut legacy_rollout),
            &cfg,
            &mut legacy_rng,
            0.01,
            &mut legacy_ret,
            false,
            None,
            None,
            &mut legacy_timings,
        )
        .unwrap();
        let owned_losses = train_update(
            std::slice::from_mut(&mut owned),
            std::slice::from_mut(&mut owned_rollout),
            &cfg,
            &mut owned_rng,
            0.01,
            &mut owned_ret,
            true,
            None,
            None,
            &mut owned_timings,
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
    fn packed_loss_read_preserves_individual_scalar_bits() {
        let tensors = [
            Tensor::from(1.25f32),
            Tensor::from(-0.0f32),
            Tensor::from(f32::from_bits(0x3f80_0001)),
            Tensor::from(-17.5f32),
        ];
        let packed = read_loss_scalars([&tensors[0], &tensors[1], &tensors[2], &tensors[3]]);
        let individual = (
            f64::try_from(&tensors[0]).unwrap(),
            f64::try_from(&tensors[1]).unwrap(),
            f64::try_from(&tensors[2]).unwrap(),
            f64::try_from(&tensors[3]).unwrap(),
        );
        assert_eq!(
            [
                packed.0.to_bits(),
                packed.1.to_bits(),
                packed.2.to_bits(),
                packed.3.to_bits(),
            ],
            [
                individual.0.to_bits(),
                individual.1.to_bits(),
                individual.2.to_bits(),
                individual.3.to_bits(),
            ]
        );
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
            assert!(
                one.gradient_txs.is_empty(),
                "world=1 must not create a persistent CPU gradient hub"
            );
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

            assert!(one_reply.averaged_gradients.is_empty());
            assert_eq!(
                two_reply.averaged_gradients.len(),
                cfg.epochs * cfg.minibatches.max(1)
            );

            let max_replica_diff = two_reply.weights[0]
                .values
                .iter()
                .zip(two_reply.weights[1].values.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let max_weight_diff = one_reply.weights[0]
                .values
                .iter()
                .zip(two_reply.weights[0].values.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            eprintln!(
                "[ddp-parity] repetition={repetition} max_replica_diff={max_replica_diff:e} \
                 max_weight_diff={max_weight_diff:e}"
            );
            assert!(
                max_replica_diff < 1e-7,
                "repetition {repetition}: two-shard replicas diverged: \
                 max_diff={max_replica_diff:e}"
            );
            assert!(
                max_weight_diff < 1e-6,
                "repetition {repetition}: 1-vs-2 shard weight mismatch: \
                 max_diff={max_weight_diff:e}"
            );
            one.shutdown().unwrap();
            two.shutdown().unwrap();
        }
    }

    #[test]
    fn recurrent_sequence_chunks_shuffle_whole_environment_trajectories() {
        assert_eq!(bptt_ranges(5, 2), vec![0..2, 2..4, 4..5]);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(44);
        let order = shuffled_sequence_envs(7, &mut rng);
        let mut sorted = order.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..7).collect::<Vec<_>>());
        assert_ne!(order, (0..7).collect::<Vec<_>>());
        for env in order {
            let flat: Vec<i64> = bptt_ranges(5, 2)
                .into_iter()
                .flat_map(|range| range.map(|t| t as i64 * 7 + env))
                .collect();
            assert_eq!(
                flat.iter().map(|index| index % 7).collect::<Vec<_>>(),
                vec![env; 5]
            );
        }
    }

    #[test]
    fn recurrent_one_and_two_shard_gradients_match_on_cpu() {
        tch::manual_seed(808);
        let mut cfg = parity_config();
        cfg.recurrent_policy = true;
        // Exercise the fused multi-step train-update path. The rollout below
        // terminates env 0 at t=0, so t=1 also covers an in-chunk hidden reset.
        cfg.bptt_chunk_len = 2;
        cfg.epochs = 1;
        let source = nn::VarStore::new(Device::Cpu);
        let _ =
            PolicyNet::new_with_recurrence(&source.root(), false, false, cfg.gc, cfg.blocks, true);
        let weights = snapshot_weights(&source).unwrap();
        let rollout = || {
            let mut rollout = parity_rollout();
            for (t, row) in rollout.buffer.iter_mut().enumerate() {
                for step in row {
                    step.hidden_in = vec![0.0; policy::RECURRENT_HIDDEN as usize];
                    step.context.action = t as i64;
                }
            }
            rollout.buffer[0][0].done = true;
            rollout
        };
        let (mut one, _) = PersistentLearner::spawn(
            &[Device::Cpu],
            cfg.clone(),
            weights.clone(),
            cfg.lr,
            rand::rngs::SmallRng::seed_from_u64(55),
        )
        .unwrap();
        let (mut two, _) = PersistentLearner::spawn(
            &[Device::Cpu, Device::Cpu],
            cfg.clone(),
            weights,
            cfg.lr,
            rand::rngs::SmallRng::seed_from_u64(55),
        )
        .unwrap();
        let one_reply = one.train(vec![rollout()], cfg.lr, 0.01).unwrap();
        let two_reply = two.train(vec![rollout(), rollout()], cfg.lr, 0.01).unwrap();
        assert!(one_reply.averaged_gradients.is_empty());
        assert_eq!(
            two_reply.averaged_gradients.len(),
            cfg.epochs * cfg.minibatches.max(1)
        );
        assert_eq!(two_reply.weights[0].values, two_reply.weights[1].values);
        let max_weight = one_reply.weights[0]
            .values
            .iter()
            .zip(two_reply.weights[0].values.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_weight < 1e-6,
            "recurrent weight mismatch: {max_weight:e}"
        );
        one.shutdown().unwrap();
        two.shutdown().unwrap();
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
