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
//! *previous* rollout - both phases run on separate OS threads inside one
//! `std::thread::scope`. This is the standard "collect batch k+1 with
//! actor v(k-1) while training learner v(k-1)->v(k) on batch k" one-step-
//! lag pipeline (an `Arc<Mutex<VarStore>>`-free way to get real overlap
//! given `tch` `Tensor`s aren't `Sync`); the actor is refreshed from the
//! learner's just-updated weights right after each update's training
//! finishes (so the *next* update's collection uses the newest weights
//! available, one version behind the learner it's paired with for
//! training).
//!
//! Multi-GPU (see `LearnerShard`/`ActorShard`): one `PolicyNet`/`VarStore`
//! replica per device, each owning a disjoint slice of envs, in a single
//! process/thread. This mirrors `rl/ppo.py`'s DDP mode (torchrun ranks
//! each own `envs/world` environments and a full local rollout/epoch/
//! minibatch loop, with gradients flat-all-reduced-and-averaged once per
//! optimizer step - see the comment above `dist.all_reduce(flat)` there)
//! rather than wrapping in `nn.parallel.DistributedDataParallel`. `tch`/
//! `torch-sys` has no NCCL bindings, so `sync_grads` below does the same
//! "average grad before step" semantics via plain `Tensor::to(device)` P2P
//! copies instead of an `ncclAllReduce`; correct and plenty fast for an
//! 11M-param policy on a single node, just not as low-latency as real NCCL
//! would be across nodes.

use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Result};
use ofcore::feat::ACTIONS;
use ofcore::translate::Choice;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use tch::nn::OptimizerConfig;
use tch::{nn, Device, Kind, Tensor};

use crate::autoscale;
use crate::batch::{self, ChoiceScalars};
use crate::gpu_util::GpuUtilSampler;
use crate::policy::{self, PolicyNet};
use crate::engine::EngineKind;
use crate::vecenv::{EnvWorker, EpisodeInfo, PreparedObs};

/// Port of `rl/ppo.py`'s entropy-floor controller constants: hold the
/// controller for the first N updates (spawn-heavy startup rollouts read
/// artificially low entropy), and cap the multiplicative scale so the
/// entropy term can't dwarf the policy gradient (the old Python cap of 30
/// did exactly that - see the Jul 9 v7 audit in the devlog).
const ENT_GRACE_UPDATES: u64 = 20;
const ENT_SCALE_MAX: f64 = 5.0;

/// How many running standard deviations (see `RetStat`) the adaptive
/// return-clip bound allows - generous relative to a well-behaved return
/// distribution (matching `adv_clip`'s own default of 10 std-devs for the
/// same reasoning), tight relative to the "value head plateaued in the
/// tens of thousands while typical returns were single digits" gap this
/// closes.
const RET_ADAPTIVE_N_STD: f64 = 10.0;

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
    /// GPU), off by default.
    pub amp: bool,
    /// `--foveate`: real foveated crop for the fine-grid branch (see
    /// `policy::PolicyNet::foveate` doc). Off by default (legacy
    /// whole-map-as-fine fallback).
    pub foveate: bool,
    /// `--gc`/`--blocks`: GridTower size overrides (see `policy::GC`/
    /// `policy::BLOCKS` defaults).
    pub gc: i64,
    pub blocks: i64,
    /// `--pinned-h2d`: pin the CPU-side observation/choice tensors' backing
    /// memory and use non-blocking H2D copies for the batch-build
    /// CPU->GPU upload (see `batch::to_device_maybe_pinned`). No-op unless
    /// `device`/shard devices are CUDA.
    pub pinned_h2d: bool,
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
    pub ckpt_every: u64,
    pub ckpt_dir: String,
    /// `--resume`: path to a previously-saved `.ot` weights file (its
    /// training-state sidecar is found by swapping the `.ot` extension for
    /// `.state.json` - see `TrainState`). Restores weights, curriculum
    /// stage, entropy-floor scale, learning rate, total env steps, and the
    /// win-rate window; the update counter resumes from where it left off.
    /// AdamW's momentum/variance state is NOT restored (tch-rs exposes no
    /// optimizer state_dict save/load - see module doc) and rebuilds over
    /// the first few dozen post-resume updates, a deliberate, documented
    /// gap rather than a silent one.
    pub resume: Option<String>,

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
/// weights checkpoint (`<ckpt>.state.json` alongside `<ckpt>.ot`) - port of
/// `rl/ppo.py`'s `state.json`/embedded-checkpoint-state pattern. Small and
/// cheap enough to write every checkpoint without it being the bottleneck.
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
    match ckpt_path.strip_suffix(".ot") {
        Some(stem) => format!("{stem}.state.json"),
        None => format!("{ckpt_path}.state.json"),
    }
}

/// Atomic write (tmp file + rename) so a kill mid-save can never leave a
/// torn/half-written checkpoint or state file behind - matches
/// `rl/ppo.py`'s `policy.pt.tmp` -> `policy.pt` rename pattern.
fn save_atomic(path: &str, write: impl FnOnce(&str) -> Result<()>) -> Result<()> {
    let tmp = format!("{path}.tmp");
    write(&tmp)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn save_checkpoint(vs: &nn::VarStore, path: &str, state: &TrainState) -> Result<()> {
    save_atomic(path, |tmp| Ok(vs.save(tmp)?))?;
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
    if after > before { EngineKind::Node } else { default }
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
    let handle = std::thread::Builder::new().name(format!("env{idx}")).spawn(move || {
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
    Ok((Worker { choice_tx, stage_tx, obs_rx, handle }, first))
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
    vs: nn::VarStore,
    policy: PolicyNet,
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
}

/// Collects one full (rollout_len, num_envs) rollout on `actor`'s policy
/// snapshot. Safe to run concurrently with a `LearnerShard` training on a
/// *different* update's data - this function never touches any
/// `LearnerShard` state, only `actor`'s own workers/`cur_obs`/`vs`/
/// `policy`, all owned exclusively by the caller's `&mut ActorShard`.
fn collect_rollout(actor: &mut ActorShard, cfg: &Config) -> Result<RolloutResult> {
    let n = actor.workers.len();
    let mut buffer: Vec<Vec<Step>> = Vec::with_capacity(cfg.rollout_len);
    let mut ep_infos = Vec::new();

    for _ in 0..cfg.rollout_len {
        let obs_refs: Vec<&PreparedObs> = actor.cur_obs.iter().collect();
        let obs_t = batch::build_obs(&obs_refs, actor.device, cfg.pinned_h2d);
        let (a, player, tile, build, nuke, qty, logp, value) = tch::no_grad(|| actor.policy.act(&obs_t, false));

        let a_v: Vec<i64> = (&a).try_into()?;
        let player_v: Vec<i64> = (&player).try_into()?;
        let tile_v: Vec<i64> = (&tile).try_into()?;
        let build_v: Vec<i64> = (&build).try_into()?;
        let nuke_v: Vec<i64> = (&nuke).try_into()?;
        let qty_v: Vec<f32> = (&qty).try_into()?;
        let logp_v: Vec<f32> = (&logp).try_into()?;
        let value_v: Vec<f32> = (&value).try_into()?;

        // Phase 1 (send): issue every env's next choice before blocking on
        // any recv, so all `n` worker threads tick concurrently instead of
        // serializing on this shard's own env count.
        let mut step_choices = Vec::with_capacity(n);
        for i in 0..n {
            let act = a_v[i];
            let np = action_needs_player(act);
            let nt = action_needs_tile(act);
            let nq = action_needs_quantity(act);
            let is_build = ACTIONS[act as usize] == "build";
            let is_nuke = ACTIONS[act as usize] == "launch_nuke";
            let choice = Choice {
                action: act,
                player_slot: np.then_some(player_v[i]),
                tile_region: nt.then_some(tile_v[i]),
                build_type: is_build.then_some(build_v[i]),
                nuke_type: is_nuke.then_some(nuke_v[i]),
                quantity_frac: nq.then_some(qty_v[i] as f64),
            };
            let scalars = ChoiceScalars {
                action: act,
                player_slot: if np { player_v[i] } else { -1 },
                tile_region: if nt { tile_v[i] } else { -1 },
                build_type: if is_build { build_v[i] } else { -1 },
                nuke_type: if is_nuke { nuke_v[i] } else { -1 },
                quantity_frac: if nq { qty_v[i] } else { -1.0 },
            };
            actor.workers[i].choice_tx.send(choice).map_err(|_| anyhow!("env {i} choice channel closed"))?;
            step_choices.push(scalars);
        }

        // Phase 2 (recv).
        let mut step_row = Vec::with_capacity(n);
        for i in 0..n {
            let (next_obs, reward, done, info) = actor.workers[i]
                .obs_rx
                .recv()
                .map_err(|_| anyhow!("env {i} obs channel closed"))?
                .map_err(|e| anyhow!("env {i}: {e}"))?;
            if let Some(info) = info {
                ep_infos.push(info);
            }
            let prev_obs = std::mem::replace(&mut actor.cur_obs[i], next_obs);
            step_row.push(Step {
                obs: prev_obs,
                choice: step_choices[i].clone(),
                logp: logp_v[i],
                value: value_v[i],
                reward: reward as f32,
                done,
            });
        }
        buffer.push(step_row);
    }

    let bootstrap_v: Vec<f32> = {
        let obs_refs: Vec<&PreparedObs> = actor.cur_obs.iter().collect();
        let obs_t = batch::build_obs(&obs_refs, actor.device, cfg.pinned_h2d);
        let v = tch::no_grad(|| actor.policy.value_only(&obs_t));
        (&v).try_into()?
    };

    Ok(RolloutResult { buffer, bootstrap_v, ep_infos })
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
    losses.iter().any(|(pg, v, ent, entq)| !pg.is_finite() || !v.is_finite() || !ent.is_finite() || !entq.is_finite())
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
) -> Result<(f64, f64, f64, f64)> {
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
    let n = pending.first().and_then(|r| r.buffer.first()).map(|row| row.len()).unwrap_or(cfg.num_envs);
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
    let adaptive_ret_bound = (RET_ADAPTIVE_N_STD * ret_stat.std()).max(1.0);
    let shard_results: Vec<(ShardBatch, f64, f64, f64)> = std::thread::scope(|s| {
        let handles: Vec<_> = learners
            .iter()
            .zip(pending.iter())
            .enumerate()
            .map(|(gi, (shard, result))| {
                let device = shard.device;
                s.spawn(move || {
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

                    let obs = batch::build_obs(&obs_flat, device, cfg.pinned_h2d);
                    let choice = batch::build_choice_batch(&choice_flat, device, cfg.pinned_h2d);
                    let adv = Tensor::from_slice(&adv_flat).to_device(device);
                    let ret = Tensor::from_slice(&ret_flat).to_device(device);
                    let old_logp = Tensor::from_slice(&old_logp_flat).to_device(device);
                    (ShardBatch { obs, choice, adv, ret, old_logp }, local_count, local_sum, local_sum_sq)
                })
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
    });
    if std::env::var("OFTRAIN_DIAG").is_ok() {
        println!("[diag] batch_build_s={:.3}", batch_build_t0.elapsed().as_secs_f64());
    }
    // Fold this update's per-shard partial sums into the running total
    // *after* they were used to compute `adaptive_ret_bound` above - the
    // bound applied to a given batch always reflects only prior updates
    // (see the comment there), never this one's own data.
    for (_, count, sum, sum_sq) in &shard_results {
        ret_stat.add_batch(*count, *sum, *sum_sq);
    }
    let mut shard_batches: Vec<ShardBatch> = shard_results.into_iter().map(|(batch, ..)| batch).collect();

    for _epoch in 0..cfg.epochs {
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

        for m in 0..n_minibatches {
            let start = (m * minibatch_size) as i64;
            let len = if m == n_minibatches - 1 { total as i64 - start } else { minibatch_size as i64 };
            let mb_t0 = Instant::now();
            // Forward + backward for every shard on its own OS thread:
            // `opt.zero_grad()`/`backward()` for shard 0 would otherwise
            // fully finish, including its implicit device sync, before
            // shard 1 even starts on a plain sequential loop.
            let per_shard_losses: Vec<(f64, f64, f64, f64)> = std::thread::scope(|s| {
                let handles: Vec<_> = learners
                    .iter_mut()
                    .zip(shard_batches.iter_mut())
                    .zip(idx_tensors.iter())
                    .map(|((shard, sb), idx_full)| {
                        let idx_t = idx_full.narrow(0, start, len);
                        s.spawn(move || {
                            let obs_t = sb.obs.index_select(&idx_t);
                            let choice_t = sb.choice.index_select(&idx_t);
                            let adv_t = sb.adv.index_select(0, &idx_t);
                            let ret_t = sb.ret.index_select(0, &idx_t);
                            let old_logp_t = sb.old_logp.index_select(0, &idx_t);

                            let (logp, ent, ent_q, value) = shard.policy.evaluate(&obs_t, &choice_t);
                            // `surr2`'s ratio is clamped to PPO's trust
                            // region [1-clip, 1+clip], but `surr1` - needed
                            // unclamped so `min(surr1, surr2)` stays a
                            // correct *pessimistic* bound for legitimate
                            // policy drift - multiplies the *raw* ratio by
                            // adv_t with nothing bounding it. A live run's
                            // pg_loss spiked to 1.48 TRILLION in a single
                            // update, immediately recovering the next
                            // update: `old_logp` is a frozen snapshot from
                            // rollout collection and `logp` is recomputed
                            // under the policy *after* several epochs of
                            // minibatch updates on the same rollout, so if
                            // even one sample's policy has drifted far
                            // enough in that window, `logp - old_logp_t`
                            // can be large enough that `.exp()` alone turns
                            // it into an astronomical ratio - and because
                            // `min(surr1, surr2)` picks whichever branch is
                            // *more negative*, that exact pathological case
                            // (huge ratio, negative advantage) is precisely
                            // when the unbounded `surr1` branch gets
                            // selected instead of the safe, clamped one.
                            // Bounding the log-ratio before exponentiating
                            // (same "fix the network's own output, not
                            // just a downstream aggregate" pattern as
                            // LOGIT_CLAMP_MAX/sanitize_value) keeps this
                            // pathological case bounded without touching
                            // the clipped trust-region behavior for any
                            // sample whose drift is within a sane range.
                            let log_ratio = (&logp - &old_logp_t).clamp(-20.0, 20.0);
                            let ratio = log_ratio.exp();
                            let surr1 = &ratio * &adv_t;
                            let surr2 = ratio.clamp(1.0 - cfg.clip as f64, 1.0 + cfg.clip as f64) * &adv_t;
                            let pg_loss = -surr1.minimum(&surr2).mean(Kind::Float);
                            // Huber (delta=`vf_clip`) instead of plain MSE for
                            // the value loss - see Config::vf_clip's doc for
                            // the full incident history. A PPO2-style
                            // "clip-and-take-worse-of-two" attempt (tried
                            // first, see git history) did NOT work: that
                            // formula's `max(unclipped, clipped)` can still
                            // select the *unclipped* branch's gradient
                            // whenever the raw prediction has drifted far
                            // enough, so it regularizes normal training but
                            // provides no actual ceiling once a prediction
                            // has already diverged - confirmed live, `v`
                            // still reached 310 quadrillion with that
                            // approach active, and even `pg_loss` got
                            // corrupted (poisoned gradients flowing back
                            // through the shared conv-tower backbone both
                            // heads sit on). Huber's gradient magnitude is
                            // bounded by `delta` by construction, everywhere,
                            // for any error size - it degrades gracefully to
                            // linear (bounded-gradient) loss beyond the
                            // delta threshold instead of MSE's unbounded
                            // quadratic gradient. This is the actual fix, not
                            // another clamp: no single sample's target or
                            // prediction, however extreme, can ever
                            // contribute more than a `delta`-bounded
                            // gradient, so the shared-backbone poisoning
                            // pathway is closed off at the source.
                            let v_loss = value.huber_loss(&ret_t, tch::Reduction::Mean, cfg.vf_clip.max(1e-3) as f64);
                            let ent_loss = ent.mean(Kind::Float);
                            // `ent_q` (Beta quantity-head entropy) is 0 for
                            // every sample whose action doesn't use a
                            // quantity_frac - averaging over the *full*
                            // batch (as `rl/ppo.py`'s original mistake would
                            // be) scales the bonus down by
                            // n_active/batch_size, weakening exploration on
                            // quantity actions. Python divides by n_active
                            // (`ppo.py` `ent_qm = ent_q.sum() / n_q`); match
                            // that.
                            let n_active = choice_t.quantity_frac.ge(0.0).to_kind(Kind::Float).sum(Kind::Float).clamp_min(1.0);
                            let entq_loss = ent_q.sum(Kind::Float) / &n_active;
                            let loss = &pg_loss + cfg.vf_coef as f64 * &v_loss
                                - ent_coef as f64 * &ent_loss
                                - cfg.entq_coef as f64 * &entq_loss;

                            shard.opt.zero_grad();
                            loss.backward();

                            (
                                f64::try_from(&pg_loss).unwrap_or(0.0),
                                f64::try_from(&v_loss).unwrap_or(0.0),
                                f64::try_from(&ent_loss).unwrap_or(0.0),
                                f64::try_from(&entq_loss).unwrap_or(0.0),
                            )
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| {
                        h.join().unwrap_or_else(|e| {
                            // See the batch-build thread's identical fix
                            // above for why this doesn't just `.expect()`.
                            let msg = e
                                .downcast_ref::<&str>()
                                .map(|s| s.to_string())
                                .or_else(|| e.downcast_ref::<String>().cloned())
                                .unwrap_or_else(|| format!("{e:?}"));
                            panic!("backward thread panicked: {msg}");
                        })
                    })
                    .collect()
            });
            let n_shards = per_shard_losses.len() as f64;
            // Neither `rl/ppo.py` nor this port clip/normalize the value
            // target (`ret`) itself - only the advantage is normalized, and
            // only *after* `ret` is derived from the raw (unnormalized) one,
            // which is correct GAE, not a bug. That means an unusually
            // large true return (a long high-reward episode, a big terminal
            // bonus, or - not yet ruled out - a reward-scale mismatch
            // between the native and Node engines under `--node-fraction`,
            // since this run was the first to ever combine that with a
            // full-scale multi-GPU run) can still occasionally push the
            // *value loss* (squared, then `vf_coef`-scaled) into the range
            // where a `clip_grad_norm(0.5)`-bounded step still overshoots
            // badly enough to compound into NaN over a few updates (see
            // docs/devlog.html's 2026-07-12 entry for the live incident this
          // guards against - `v` climbed from 5.47 to 56.4 TRILLION in one
            // update, then to outright NaN eight updates later). Rather
            // than fix the exact trigger (unconfirmed) by guessing at a
            // reward-scale change, guard the *symptom* directly: skip
            // applying a minibatch's update outright if any shard's loss
            // came back non-finite, so one poisoned rollout costs one
            // discarded minibatch instead of turning every future update
            // (and the checkpoint born from it) into unrecoverable garbage.
            if any_loss_non_finite(&per_shard_losses) {
                eprintln!(
                    "[train] WARNING: non-finite loss in epoch={_epoch} mb={m} \
                     (per-shard pg/v/ent/entq={per_shard_losses:?}) - discarding this \
                     minibatch's gradients without stepping the optimizer (see \
                     docs/devlog.html's 2026-07-12 NaN-guard entry)"
                );
                // Still clears the poisoned grads (zero_grad() runs at the
                // top of next minibatch's forward-pass closure regardless,
                // but doing it here too means a mid-epoch panic/early exit
                // between minibatches can never leave a shard holding NaN
                // grads for `sync_grads`/logging elsewhere to observe).
                for shard in learners.iter_mut() {
                    shard.opt.zero_grad();
                }
                n_mb += 1;
                continue;
            }
            for (pg, v, ent, entq) in &per_shard_losses {
                loss_sums.0 += pg / n_shards;
                loss_sums.1 += v / n_shards;
                loss_sums.2 += ent / n_shards;
                loss_sums.3 += entq / n_shards;
            }
            n_mb += 1;
            let fwdbwd_dt = mb_t0.elapsed().as_secs_f64();

            // DDP-equivalent sync: average grads across shards (no-op for
            // 1 shard) so every replica's optimizer step is identical and
            // weights never drift apart.
            let sync_t0 = Instant::now();
            sync_grads(learners);
            let sync_dt = sync_t0.elapsed().as_secs_f64();
            let step_t0 = Instant::now();
            for shard in learners.iter_mut() {
                // Matches `rl/ppo.py`'s `clip_grad_norm_(..., 0.5)`.
                shard.opt.clip_grad_norm(0.5);
                shard.opt.step();
            }
            let step_dt = step_t0.elapsed().as_secs_f64();
            if std::env::var("OFTRAIN_DIAG").is_ok() {
                println!("[diag] epoch={_epoch} mb={m} fwdbwd_s={fwdbwd_dt:.3} sync_s={sync_dt:.3} step_s={step_dt:.3}");
            }
        }
    }

    let d = n_mb.max(1) as f64;
    Ok((loss_sums.0 / d, loss_sums.1 / d, loss_sums.2 / d, loss_sums.3 / d))
}

pub fn run(cfg: Config) -> Result<()> {
    std::fs::create_dir_all(&cfg.ckpt_dir)?;

    // Resume: load weights + training state before anything else needs
    // them - env workers need the resumed curriculum stage at spawn time,
    // and every shard copies from `hub_vs` below, so it has to be seeded
    // with the resumed weights (not shard 0's freshly-initialized ones)
    // before the shard loop runs.
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
    } else {
        None
    };
    let start_stage = resumed_state.as_ref().map(|s| s.stage).unwrap_or(cfg.stage);

    let devices = cfg.devices();
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
    let mut learners: Vec<LearnerShard> = Vec::with_capacity(devices.len());
    for (gi, &device) in devices.iter().enumerate() {
        let mut workers = Vec::with_capacity(cfg.num_envs);
        let mut cur_obs = Vec::with_capacity(cfg.num_envs);
        for local_i in 0..cfg.num_envs {
            let idx = gi * cfg.num_envs + local_i;
            let worker_engine = engine_for_idx(idx, cfg.engine, cfg.node_fraction);
            let (w, obs) = spawn_worker(idx, start_stage, cfg.max_episode_ticks, worker_engine)?;
            workers.push(w);
            cur_obs.push(obs);
        }
        let mut learner_vs = nn::VarStore::new(device);
        let learner_policy = PolicyNet::new(&learner_vs.root(), cfg.amp, cfg.foveate, cfg.gc, cfg.blocks);
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

        let mut actor_vs = nn::VarStore::new(device);
        let actor_policy = PolicyNet::new(&actor_vs.root(), cfg.amp, cfg.foveate, cfg.gc, cfg.blocks);
        actor_vs.copy(&learner_vs)?;

        actors.push(ActorShard { device, workers, cur_obs, vs: actor_vs, policy: actor_policy });
        learners.push(LearnerShard { device, vs: learner_vs, policy: learner_policy, opt });
    }
    println!("[train] all {total_envs} envs ready");

    let n_params: i64 = learners[0].vs.trainable_variables().iter().map(|t| t.numel() as i64).sum();
    println!("[train] policy params: {n_params} per shard x {} shard(s) on {:?}", learners.len(), devices);

    let gpu_sampler =
        if devices.iter().any(|d| matches!(d, Device::Cuda(_))) { Some(GpuUtilSampler::start(Duration::from_millis(500))) } else { None };

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
                std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
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

    let mut rng = rand::rngs::SmallRng::from_entropy();
    // Persists across every update in this run (see RetStat's doc) - a
    // fresh, empty RetStat on every process restart is an acceptable cold
    // start (its adaptive bound is a no-op until ~2 updates' worth of
    // data has accumulated; RET_ADAPTIVE_N_STD covers everything else).
    let mut ret_stat = RetStat::default();
    let mut ep_rewards: Vec<f64> = Vec::new();
    let mut ep_lengths: Vec<i64> = Vec::new();
    let train_start = Instant::now();
    let mut total_env_steps: u64 = resumed_state.as_ref().map(|s| s.total_env_steps).unwrap_or(0);
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

    // Prime the pipeline: collect the very first rollout (using the
    // actors' initial, freshly-copied-from-learner weights) before the
    // update loop starts overlapping collection with training.
    let mut pending: Vec<RolloutResult> = std::thread::scope(|s| {
        let handles: Vec<_> = actors.iter_mut().map(|actor| s.spawn(move || collect_rollout(actor, cfg_ref))).collect();
        handles.into_iter().map(|h| h.join().expect("collector thread panicked")).collect::<Result<Vec<_>>>()
    })?;

    for update in start_update..cfg.updates {
        let update_start = Instant::now();

        // Overlap: collect update `update+1`'s rollout on every shard's
        // (frozen-for-this-round) actor concurrently with training the
        // learner on `pending` (this update's, already-collected data).
        // See module doc for why this is safe (disjoint state) and what
        // it's fixing (GPU idling during collection).
        // Linear anneal ent_coef -> ent_coef_final so late training commits
        // instead of exploring forever, times the adaptive entropy-floor
        // scale (both match `rl/ppo.py`).
        let frac = (update as f64 / cfg.ent_anneal_updates.max(1) as f64).min(1.0);
        let ent_coef_now =
            ((cfg.ent_coef as f64 + (cfg.ent_coef_final as f64 - cfg.ent_coef as f64) * frac) * ent_scale) as f32;
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
        let (train_result, next_pending, train_dt) = std::thread::scope(|s| {
            let collect_handles: Vec<_> =
                actors.iter_mut().map(|actor| s.spawn(move || collect_rollout(actor, cfg_ref))).collect();

            let train_t0 = Instant::now();
            let train_result = train_update(&mut learners, &pending, cfg_ref, &mut rng, ent_coef_now, &mut ret_stat);
            let train_dt = train_t0.elapsed().as_secs_f64();

            let next_pending: Result<Vec<RolloutResult>> =
                collect_handles.into_iter().map(|h| h.join().expect("collector thread panicked")).collect();
            (train_result, next_pending, train_dt)
        });
        let collect_dt = collect_start.elapsed().as_secs_f64();
        let last_losses = train_result?;
        let next_pending = next_pending?;
        // Actual env count behind `next_pending` (each shard's rollout
        // just collected) - not the startup `total_envs`, which goes
        // stale the moment `auto_scale_envs` grows any shard. Derived
        // straight from the collected data rather than
        // `actors[..].workers.len()` so it's correct regardless of
        // exactly when in this iteration a resize lands.
        let live_total_envs: usize =
            next_pending.iter().map(|r| r.buffer.first().map(|row| row.len()).unwrap_or(0)).sum();

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
                        info.reward, info.length, info.final_tiles, info.final_tick,
                        info.place, info.n_players, info.score, info.won, info.wasted,
                        info.stage, info.rehearsal, info.map
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
                        eprintln!("[win_rate] {:.3} (window={}/{})", win_rate, recent_wins.len(), ofcore::curriculum::WINDOW);
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
            for actor in &actors {
                for w in &actor.workers {
                    let _ = w.stage_tx.send(curr_stage);
                }
            }
            for shard in learners.iter_mut() {
                shard.opt.set_lr(lr_now);
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
        for (actor, learner) in actors.iter_mut().zip(learners.iter()) {
            actor.vs.copy(&learner.vs)?;
        }
        pending = next_pending;

        // Auto-scale check: deliberately placed here, after this
        // update's `pending`/`next_pending` swap and *before* the next
        // loop iteration spawns its `collect_rollout` calls - growing
        // `actor.workers`/`actor.cur_obs` mid-rollout (inside
        // `collect_rollout`'s per-step send/recv loop) would desync the
        // `n = actor.workers.len()` it captured at the top of that call.
        if cfg.auto_scale_envs && update % cfg.autoscale_check_every.max(1) == 0 {
            let gpu_util_frac = gpu_sampler.as_ref().map(|g| g.snapshot().min_mean_util() / 100.0);
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
                let mut spawned: Vec<(usize, Worker, PreparedObs)> = Vec::with_capacity(add * actors.len());
                let mut spawn_err: Option<anyhow::Error> = None;
                'grow: for gi in 0..actors.len() {
                    for _ in 0..add {
                        let worker_engine = engine_for_idx(next_env_idx, cfg.engine, cfg.node_fraction);
                        match spawn_worker(next_env_idx, curr_stage, cfg.max_episode_ticks, worker_engine) {
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
                let gpu_str = gpu_util_frac.map(|f| format!("{:.1}%", f * 100.0)).unwrap_or_else(|| "n/a (no GPU)".to_string());
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

        if update % cfg.log_every == 0 || update == cfg.updates - 1 {
            let dt = update_start.elapsed().as_secs_f64();
            let total_dt = train_start.elapsed().as_secs_f64();
            let sps = (live_total_envs * cfg.rollout_len) as f64 / dt.max(1e-6);
            let recent_n = ep_rewards.len().min(50);
            let recent_reward = if recent_n > 0 {
                ep_rewards[ep_rewards.len() - recent_n..].iter().sum::<f64>() / recent_n as f64
            } else {
                0.0
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
                    format!(" gpu_mem%={:.0} min_mean_util%={:.0} [{per_gpu_str}]", gpu.mem_pct, gpu.min_mean_util())
                }
                None => String::new(),
            };
            println!(
                "[update {:>5}] steps/s={:>7.1} decisions_total={:>9} eps_done={:>5} recent_reward={:>8.3} \
                 pg={:>+.4} v={:>.4} ent={:>.3} entq={:>+.3} ecoef={:.4} stage={} lr={:.2e} elapsed={:.0}s \
                 update_s={:.1} collect_s={:.1} train_s={:.1}{gpu_str}",
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
            );
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
            let path = format!("{}/policy_update{}.ot", cfg.ckpt_dir, update);
            save_checkpoint(&learners[0].vs, &path, &state)?;
            // Fixed-name pointer at the latest checkpoint so a restart-loop
            // wrapper (or a fresh pod after total disk loss) always has one
            // unambiguous thing to resume from, without parsing filenames -
            // matches `rl/ppo.py`'s single always-current `policy.pt`.
            save_checkpoint(&learners[0].vs, &format!("{}/latest.ot", cfg.ckpt_dir), &state)?;
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
    save_checkpoint(&learners[0].vs, &format!("{}/policy_final.ot", cfg.ckpt_dir), &final_state)?;
    save_checkpoint(&learners[0].vs, &format!("{}/latest.ot", cfg.ckpt_dir), &final_state)?;
    for actor in actors {
        for w in actor.workers {
            drop(w.choice_tx);
            let _ = w.handle.join();
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn unused_lock_hint(_m: &Arc<Mutex<()>>) {}

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
            assert!(v.abs() <= 10.0 + 1e-4, "every normalized advantage must be clamped to +-10, got {v}");
        }
        // The outlier should still be *at* the clip boundary (not
        // collapsed to 0 or left unclamped) - clamping should engage, not
        // silently no-op.
        assert!((adv[2047] - 10.0).abs() < 1e-3, "the outlier should sit exactly at the clip boundary, got {}", adv[2047]);
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
            assert!((a - expected).abs() < 1e-4, "no clamping should engage for a well-behaved batch");
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
        assert!((h - 0.5 * m).abs() < 1e-5, "huber={h} should be ~0.5*mse={m} for small errors");
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
        whole.add_batch(5.0, 1.0 + 2.0 + 3.0 + 4.0 + 5.0, 1.0 + 4.0 + 9.0 + 16.0 + 25.0);

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
        assert!((clamped - unclamped).abs() < 1e-5, "clamp must be a no-op for ordinary ratios: {clamped} vs {unclamped}");
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
            assert_eq!(engine_for_idx(idx, EngineKind::Native, 0.0), EngineKind::Native);
        }
    }

    #[test]
    fn full_fraction_is_always_node() {
        for idx in 0..50 {
            assert_eq!(engine_for_idx(idx, EngineKind::Native, 1.0), EngineKind::Node);
        }
    }

    #[test]
    fn one_fifth_gives_exactly_one_node_per_five_evenly_spread() {
        let picks: Vec<usize> =
            (0..50).filter(|&i| engine_for_idx(i, EngineKind::Native, 0.2) == EngineKind::Node).collect();
        assert_eq!(picks.len(), 10, "50 * 0.2 == 10 node envs, got {picks:?}");
        // Evenly spread, not clumped: consecutive picks should be exactly
        // 5 apart, and the ratio should hold over any prefix, not just the
        // full range (matters once autoscale appends more envs at
        // ever-increasing indices - see engine_for_idx's doc comment).
        for w in picks.windows(2) {
            assert_eq!(w[1] - w[0], 5, "picks should be evenly spread 5 apart, got {picks:?}");
        }
        for prefix_len in [5, 15, 25, 35, 45] {
            let n = (0..prefix_len).filter(|&i| engine_for_idx(i, EngineKind::Native, 0.2) == EngineKind::Node).count();
            assert_eq!(n, prefix_len / 5, "ratio should hold at prefix {prefix_len}, got {n} node envs");
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
