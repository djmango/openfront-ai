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

use crate::batch::{self, ChoiceScalars};
use crate::gpu_util::GpuUtilSampler;
use crate::policy::{self, PolicyNet};
use crate::engine::EngineKind;
use crate::vecenv::{EnvWorker, EpisodeInfo, PreparedObs};

pub struct Config {
    /// Envs per shard (per device). Total envs = num_envs * devices().len().
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
    /// Entropy coefficient anneals linearly `ent_coef -> ent_coef_final`
    /// over `ent_anneal_updates` (matches `rl/ppo.py`'s schedule; the
    /// adaptive entropy-floor multiplier on top of this is not ported -
    /// see DEVLOG).
    pub ent_coef: f32,
    pub ent_coef_final: f32,
    pub ent_anneal_updates: u64,
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
    /// in-process native engine).
    pub engine: EngineKind,
    pub log_every: u64,
    pub ckpt_every: u64,
    pub ckpt_dir: String,
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
) -> Result<(f64, f64, f64, f64)> {
    let n = cfg.num_envs;
    let t_len = cfg.rollout_len;
    let total = t_len * n;
    let minibatch_size = (total / cfg.minibatches.max(1)).max(1);
    let mut last_losses = (0.0f64, 0.0f64, 0.0f64, 0.0f64);

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
    let mut shard_batches: Vec<ShardBatch> = std::thread::scope(|s| {
        let handles: Vec<_> = learners
            .iter()
            .zip(pending.iter())
            .map(|(shard, result)| {
                let device = shard.device;
                s.spawn(move || {
                    let buffer = &result.buffer;
                    let bootstrap_v = &result.bootstrap_v;
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
                    for t in 0..t_len {
                        for e in 0..n {
                            let flat_idx = t * n + e;
                            adv_flat[flat_idx] = adv[t][e];
                            ret_flat[flat_idx] = adv[t][e] + buffer[t][e].value;
                            old_logp_flat[flat_idx] = buffer[t][e].logp;
                        }
                    }
                    {
                        let adv_mean = adv_flat.iter().sum::<f32>() / total as f32;
                        let adv_var = adv_flat.iter().map(|x| (x - adv_mean).powi(2)).sum::<f32>() / total as f32;
                        let adv_std = adv_var.sqrt().max(1e-8);
                        for v in adv_flat.iter_mut() {
                            *v = (*v - adv_mean) / adv_std;
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
                    let obs = batch::build_obs(&obs_flat, device, cfg.pinned_h2d);
                    let choice = batch::build_choice_batch(&choice_flat, device, cfg.pinned_h2d);
                    let adv = Tensor::from_slice(&adv_flat).to_device(device);
                    let ret = Tensor::from_slice(&ret_flat).to_device(device);
                    let old_logp = Tensor::from_slice(&old_logp_flat).to_device(device);
                    ShardBatch { obs, choice, adv, ret, old_logp }
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("batch-build thread panicked")).collect()
    });
    if std::env::var("OFTRAIN_DIAG").is_ok() {
        println!("[diag] batch_build_s={:.3}", batch_build_t0.elapsed().as_secs_f64());
    }

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
                            let ratio = (&logp - &old_logp_t).exp();
                            let surr1 = &ratio * &adv_t;
                            let surr2 = ratio.clamp(1.0 - cfg.clip as f64, 1.0 + cfg.clip as f64) * &adv_t;
                            let pg_loss = -surr1.minimum(&surr2).mean(Kind::Float);
                            let v_loss = (&value - &ret_t).pow_tensor_scalar(2).mean(Kind::Float);
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
                handles.into_iter().map(|h| h.join().expect("backward thread panicked")).collect()
            });
            last_losses = per_shard_losses[0];
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

    Ok(last_losses)
}

pub fn run(cfg: Config) -> Result<()> {
    std::fs::create_dir_all(&cfg.ckpt_dir)?;
    let devices = cfg.devices();
    let total_envs = cfg.num_envs * devices.len();
    println!(
        "[train] spawning {} env workers across {} shard(s) {:?} (stage={}, max_ticks={})...",
        total_envs,
        devices.len(),
        devices,
        cfg.stage,
        cfg.max_episode_ticks
    );

    let mut actors: Vec<ActorShard> = Vec::with_capacity(devices.len());
    let mut learners: Vec<LearnerShard> = Vec::with_capacity(devices.len());
    let mut hub_vs: Option<nn::VarStore> = None;
    for (gi, &device) in devices.iter().enumerate() {
        let mut workers = Vec::with_capacity(cfg.num_envs);
        let mut cur_obs = Vec::with_capacity(cfg.num_envs);
        for local_i in 0..cfg.num_envs {
            let idx = gi * cfg.num_envs + local_i;
            let (w, obs) = spawn_worker(idx, cfg.stage, cfg.max_episode_ticks, cfg.engine)?;
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
        let opt = nn::AdamW::default().build(&learner_vs, cfg.lr)?;

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

    let mut rng = rand::rngs::SmallRng::from_entropy();
    let mut ep_rewards: Vec<f64> = Vec::new();
    let mut ep_lengths: Vec<i64> = Vec::new();
    let train_start = Instant::now();
    let mut total_env_steps: u64 = 0;
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
    let mut curr_stage = cfg.stage;
    let mut recent_wins: std::collections::VecDeque<f64> = std::collections::VecDeque::with_capacity(ofcore::curriculum::WINDOW);
    let mut lr_now = cfg.lr;

    // Prime the pipeline: collect the very first rollout (using the
    // actors' initial, freshly-copied-from-learner weights) before the
    // update loop starts overlapping collection with training.
    let mut pending: Vec<RolloutResult> = std::thread::scope(|s| {
        let handles: Vec<_> = actors.iter_mut().map(|actor| s.spawn(move || collect_rollout(actor, cfg_ref))).collect();
        handles.into_iter().map(|h| h.join().expect("collector thread panicked")).collect::<Result<Vec<_>>>()
    })?;

    for update in 0..cfg.updates {
        let update_start = Instant::now();

        // Overlap: collect update `update+1`'s rollout on every shard's
        // (frozen-for-this-round) actor concurrently with training the
        // learner on `pending` (this update's, already-collected data).
        // See module doc for why this is safe (disjoint state) and what
        // it's fixing (GPU idling during collection).
        // Linear anneal ent_coef -> ent_coef_final so late training commits
        // instead of exploring forever (matches `rl/ppo.py`; the adaptive
        // entropy-floor multiplier on top of this schedule is not ported -
        // see DEVLOG).
        let frac = (update as f64 / cfg.ent_anneal_updates.max(1) as f64).min(1.0);
        let ent_coef_now = (cfg.ent_coef as f64 + (cfg.ent_coef_final as f64 - cfg.ent_coef as f64) * frac) as f32;
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
            let train_result = train_update(&mut learners, &pending, cfg_ref, &mut rng, ent_coef_now);
            let train_dt = train_t0.elapsed().as_secs_f64();

            let next_pending: Result<Vec<RolloutResult>> =
                collect_handles.into_iter().map(|h| h.join().expect("collector thread panicked")).collect();
            (train_result, next_pending, train_dt)
        });
        let collect_dt = collect_start.elapsed().as_secs_f64();
        let last_losses = train_result?;
        let next_pending = next_pending?;

        let mut advanced = false;
        for result in &next_pending {
            for info in &result.ep_infos {
                ep_rewards.push(info.reward);
                ep_lengths.push(info.length);
                if info.stage == curr_stage && !info.rehearsal {
                    if recent_wins.len() == ofcore::curriculum::WINDOW {
                        recent_wins.pop_front();
                    }
                    recent_wins.push_back(if info.won { 1.0 } else { 0.0 });
                    let win_rate = recent_wins.iter().sum::<f64>() / recent_wins.len() as f64;
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
        total_env_steps += (total_envs * cfg.rollout_len) as u64;

        // Refresh every actor from its paired learner's just-updated
        // weights, now that training has finished (and the collection
        // that ran concurrently with it is done reading the *old*
        // weights) - the next update's collection will use these.
        for (actor, learner) in actors.iter_mut().zip(learners.iter()) {
            actor.vs.copy(&learner.vs)?;
        }
        pending = next_pending;

        if update % cfg.log_every == 0 || update == cfg.updates - 1 {
            let dt = update_start.elapsed().as_secs_f64();
            let total_dt = train_start.elapsed().as_secs_f64();
            let sps = (total_envs * cfg.rollout_len) as f64 / dt.max(1e-6);
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
            let path = format!("{}/policy_update{}.ot", cfg.ckpt_dir, update);
            learners[0].vs.save(&path)?;
            println!("[train] checkpoint saved: {path}");
        }
    }

    learners[0].vs.save(format!("{}/policy_final.ot", cfg.ckpt_dir))?;
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
