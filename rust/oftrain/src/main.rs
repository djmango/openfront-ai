mod ae;
mod autoscale;
mod batch;
mod bridge;
mod engine;
mod gpu_util;
mod metrics;
#[cfg(feature = "native-engine")]
mod native;
mod nccl;
mod policy;
mod recurrent;
mod train;
mod vecenv;
mod watch;

use clap::Parser;
use tch::{Cuda, Device};

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

    /// Opt into the V8.1 curriculum gates. Stage identities/maps are
    /// unchanged; only stages 4+ use recalibrated crowded-map win gates.
    #[arg(long, default_value_t = false, conflicts_with = "v9_curriculum")]
    v81_curriculum: bool,

    /// Opt into the V8.1.1 high-player bridge schedule. Stage 5 is the
    /// V8.1 30-bot map pool at Easy, stage 6 repeats it at Medium, and the
    /// World/Asia challenge moves to stage 7. Incompatible with V8.1 state
    /// unless the explicit stage-5 migration flag is also supplied.
    #[arg(
        long,
        default_value_t = false,
        conflicts_with_all = ["v81_curriculum", "v9_curriculum"]
    )]
    v811_curriculum: bool,

    /// Opt into the recurrent V8.2 broad-map schedule. Stages 0-4 are
    /// unchanged; Easy progresses through 30/50/80 players before Medium.
    #[arg(
        long,
        default_value_t = false,
        conflicts_with_all = ["v81_curriculum", "v811_curriculum", "v9_curriculum"]
    )]
    v82_curriculum: bool,

    /// Opt into the V8.3 closeout schedule and reward profile.
    #[arg(
        long,
        default_value_t = false,
        conflicts_with_all = ["v81_curriculum", "v811_curriculum", "v82_curriculum", "v9_curriculum"]
    )]
    v83_curriculum: bool,

    /// Opt into the parallel V9 sparse-win curriculum (high-gate micro-ladder).
    /// Pair with `--v9-sparse-win` for the ±1 terminal reward profile.
    #[arg(
        long,
        default_value_t = false,
        conflicts_with_all = ["v81_curriculum", "v811_curriculum", "v82_curriculum", "v83_curriculum"]
    )]
    v9_curriculum: bool,

    /// Per-stage env worker targets, per GPU/shard. Accepts either one
    /// comma-separated value per stage (`24,24,...`) or ranges such as
    /// `0-5=24,6=12,7=10,8+=8`. Schedule flags select versioned defaults;
    /// V8.2 uses 16 envs at its eight-map bridge, then 12/10/8 by load.
    /// A target change checkpoints and exits for supervisor restart.
    #[arg(long)]
    stage_env_targets: Option<String>,

    /// Matches `rl/ppo.py --max-episode-ticks` (default 15000). The port
    /// originally shipped 3000, which truncated every stage-0 episode
    /// before the 80%-ownership win condition was reachable - see devlog.
    #[arg(long, default_value_t = 15000)]
    max_episode_ticks: i64,

    /// Steps collected per env before each PPO update.
    #[arg(long, default_value_t = 32)]
    rollout_len: usize,

    #[arg(long, default_value_t = 1_000_000)]
    updates: u64,

    #[arg(long, default_value_t = 2.5e-4)]
    lr: f64,

    #[arg(long, default_value_t = 0.999)]
    gamma: f32,

    /// V8.1 potential-based closeout shaping coefficient K_DOM (0 disables).
    #[arg(long, default_value_t = 0.0)]
    v81_dom_coef: f64,

    /// First curriculum stage where V8.1 shaping may apply.
    #[arg(long, default_value_t = 4)]
    v81_min_stage: usize,

    /// Symmetric clamp magnitude for the V8.1 log strength-ratio potential.
    #[arg(long, default_value_t = 2.0)]
    v81_potential_clamp: f64,

    /// Relax loss aversion after reaching the dominance threshold.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    v81_dominant_loss: bool,

    /// Normalized composite-strength share that counts as dominant.
    #[arg(long, default_value_t = 0.55)]
    v81_dominance_threshold: f64,

    /// Strength-loss weight while dominant (global legacy weight is 6.5).
    #[arg(long, default_value_t = 5.25)]
    v81_delta_loss_dominant: f64,

    /// Penalty for reversing a matching recent action (0 disables).
    #[arg(long, default_value_t = 0.0)]
    v81_churn_coef: f64,

    /// Number of prior decisions searched for a matching inverse action.
    /// Default 16 matches BPTT length so delayed undos (boat → other → cancel)
    /// are still attributed; 2 only caught immediately adjacent reversals.
    #[arg(long, default_value_t = 16)]
    v81_churn_window: usize,

    /// First curriculum stage where the action-churn penalty may apply.
    #[arg(long, default_value_t = 4)]
    v81_churn_min_stage: usize,

    /// V8.3 land-share closeout potential coefficient.
    #[arg(long, default_value_t = 4.0)]
    v83_close_coef: f64,

    /// V8.3 closeout-region action-churn penalty.
    #[arg(long, default_value_t = 0.06)]
    v83_churn_coef: f64,

    /// V8.4 boat-outcome: reward for a sourced landing attack.
    #[arg(long, default_value_t = 0.0)]
    v84_boat_useful: f64,

    /// V8.4 boat-outcome: penalty when a boat is destroyed without landing.
    #[arg(long, default_value_t = 0.0)]
    v84_boat_destroyed: f64,

    /// V8.4 boat-outcome: mild cancel penalty (churn already covers the pair).
    #[arg(long, default_value_t = 0.0)]
    v84_boat_cancelled: f64,

    /// V8.4 boat-outcome: penalty for returning to own shore without invading.
    #[arg(long, default_value_t = 0.0)]
    v84_boat_own_shore: f64,

    #[arg(long, default_value_t = 4)]
    v84_boat_min_stage: usize,

    /// V8.4 late-game tempo pressure while dominant.
    #[arg(long, default_value_t = 0.0)]
    v84_tempo_coef: f64,

    #[arg(long, default_value_t = 4)]
    v84_tempo_min_stage: usize,

    /// V8.4 terminal bonus for faster wins: coef * (1 - tick/max_ticks).
    #[arg(long, default_value_t = 0.0)]
    v84_fast_win_coef: f64,

    /// V8.5: tempo share threshold (0 = use --v81-dominance-threshold).
    #[arg(long, default_value_t = 0.0)]
    v85_tempo_share_threshold: f64,

    /// V8.5: extra terminal bonus on win (on top of W_WIN).
    #[arg(long, default_value_t = 0.0)]
    v85_extra_win_bonus: f64,

    /// V8.5: penalty for embargo_stop while Hostile/Distrustful.
    #[arg(long, default_value_t = 0.0)]
    v85_embargo_bad_stop: f64,

    /// V8.5: small reward for embargo_stop after relation recovered.
    #[arg(long, default_value_t = 0.0)]
    v85_embargo_good_stop: f64,

    #[arg(long, default_value_t = 4)]
    v85_embargo_min_stage: usize,

    /// V8.5: penalty for retreating a just-opened attack.
    #[arg(long, default_value_t = 0.0)]
    v85_premature_retreat: f64,

    /// V8.5: penalty for re-attacking right after retreat.
    #[arg(long, default_value_t = 0.0)]
    v85_thrash_reengage: f64,

    #[arg(long, default_value_t = 4)]
    v85_combat_min_stage: usize,

    /// V8.6: override W_DELTA_LOSS when > 0 (soften attack variance tax).
    #[arg(long, default_value_t = 0.0)]
    v86_delta_loss: f64,

    /// V8.6: price strength losses like gains while an attack is open.
    #[arg(long, default_value_t = false)]
    v86_attack_symmetric_loss: bool,

    /// V8.6: skip flat attack↔retreat churn (combat sticky already prices it).
    #[arg(long, default_value_t = false)]
    v86_skip_combat_churn: bool,

    /// V8.6: override W_DEATH when > 0.
    #[arg(long, default_value_t = 0.0)]
    v86_death_penalty: f64,

    /// V9: terminal win/loss only (+1 / -1). Disables dense strength shaping.
    #[arg(long, default_value_t = false)]
    v9_sparse_win: bool,

    /// Resume a V8.4 reward-profile checkpoint under V8.5 coeffs (weights unchanged).
    #[arg(long, default_value_t = false, requires_all = ["v83_curriculum", "resume"])]
    migrate_v84_to_v85: bool,

    /// Resume a V8.5 reward-profile checkpoint under V8.6 coeffs (weights unchanged).
    #[arg(long, default_value_t = false, requires_all = ["v83_curriculum", "resume"])]
    migrate_v85_to_v86: bool,

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

    /// Clamps the normalized advantage magnitude (see
    /// `train::Config::adv_clip`'s doc - the policy-loss-side counterpart
    /// to `vf_clip`/Huber on the value-loss side, both needed to fully fix
    /// the 2026-07-12 incident). 0.0 disables it.
    #[arg(long, default_value_t = 10.0)]
    adv_clip: f32,

    /// Huber-loss delta for the value loss, replacing plain MSE (see
    /// `train::Config::vf_clip`'s doc for the full mechanism and the two
    /// prior attempts - target clamping, then PPO2-style prediction
    /// clamping - that didn't actually bound the gradient in the case that
    /// matters). Below this error magnitude, behaves like ordinary squared
    /// error (matching typical/healthy training exactly); beyond it, the
    /// loss grows only linearly, so no single sample can ever contribute
    /// more than a `vf_clip`-bounded gradient regardless of how extreme
    /// the target or prediction is. Default 50.0 - well above the healthy
    /// v-loss range (~0.05-0.5) seen early in training, so it doesn't
    /// interfere with normal learning at all, while still bounding the
    /// pathological case (millions/billions/quadrillions, all observed
    /// live before this fix) to a sane per-sample gradient contribution.
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

    /// Number of minibatches per shard. The default gives Python's
    /// 128-sample minibatches for the default 4 envs x 32 rollout.
    #[arg(long, default_value_t = 1)]
    minibatches: usize,

    /// Manual bf16 mixed precision for the policy net's conv towers
    /// (grid towers, local net, tile heads); logits/loss/optimizer state
    /// stay f32. tch-rs 0.24's `autocast()` has no dtype selector (always
    /// picks fp16 on CUDA), so this is a hand-rolled cast-in/cast-out path
    /// instead - see `policy.rs`/DEVLOG. Works (slower) on CPU too, so
    /// it's smoke-testable without a GPU. Frozen AE bf16 additionally
    /// requires CUDA and --persistent-actors; all other AE paths stay f32.
    #[arg(long, default_value_t = false)]
    amp: bool,

    /// Real foveated crop: the fine-grid branch becomes a fixed
    /// `policy::FOVEATE_SIZE`x`FOVEATE_SIZE` window centered on the agent's
    /// own-tile centroid instead of the whole map (coarse branch is
    /// unaffected - always the full map). Default on to match Python v7;
    /// pass `--foveate=false` for the legacy whole-map-as-fine path.
    #[arg(long, default_value_t = true)]
    foveate: bool,

    /// Frozen fine AE encoder safetensors (from
    /// `scripts/export_safetensors.py` on `ae_v31_d8c32.pt`). Required for
    /// production obs parity (`C_GRID=89`).
    #[arg(long, default_value = "weights/ae/ae_v31_d8c32.encoder.safetensors")]
    ckpt: String,

    /// Optional frozen coarse /16 AE encoder safetensors (from
    /// `ae_v31_d16c32.pt`). When set, the coarse stream uses a native /16
    /// latent instead of 2x-pooling the fine grid.
    #[arg(long)]
    coarse_ckpt: Option<String>,

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

    /// H2D fine/coarse grids as fp16 then cast to f32 on device (halves
    /// PCIe bytes for the big AE planes). Host `PreparedObs` stays f32.
    /// Opt-in (default off); `pod_train_v8.sh` enables via EXTRA_ARGS.
    /// No-op on CPU (Half round-trip skipped).
    #[arg(long, default_value_t = false)]
    fp16_rollout: bool,

    /// Store foveated rollout grids as compact host fp16 windows with
    /// explicit origins/masks. Requires --foveate; legacy behavior is kept
    /// when disabled or when foveation is off.
    #[arg(long, default_value_t = false)]
    compact_rollout: bool,

    /// Split env workers into two halves and overlap act(g1) with step(g0)
    /// inside each rollout step (Python v4.1 dual-group pipelining).
    /// Default on; with one env the second group is empty.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pipeline_groups: bool,

    /// Keep one actor OS thread alive per GPU for the full run and, for the
    /// one-GPU Phase 2 path, keep the learner on its own stable owner thread.
    /// CUDA state never crosses channels. Incompatible utilization autoscale
    /// is disabled without changing ownership mode.
    #[arg(long, default_value_t = false)]
    persistent_actors: bool,

    /// Enable per-environment recurrent policy state. Requires persistent
    /// CUDA actors. The hidden width is set by --recurrent-hidden-size.
    #[arg(long, default_value_t = false, requires = "persistent_actors")]
    recurrent_policy: bool,

    /// Number of f32 values in each environment's recurrent hidden state.
    #[arg(long, default_value_t = 256)]
    recurrent_hidden_size: usize,

    /// Timesteps per truncated-BPTT chunk for recurrent PPO.
    #[arg(long, default_value_t = 16)]
    bptt_chunk_len: usize,

    /// Let persistent actors batch whichever envs are ready instead of
    /// waiting for fixed worker halves. Requires --persistent-actors.
    #[arg(long, default_value_t = false)]
    work_conserving_actors: bool,

    /// Maximum ready envs in one work-conserving actor inference batch.
    #[arg(long, default_value_t = 32)]
    actor_max_batch: usize,

    /// Preferred ready envs before dispatch (bounded by --actor-max-batch).
    #[arg(long, default_value_t = 8)]
    actor_target_batch: usize,

    /// Maximum compact-coarse padding waste allowed while coalescing shapes.
    #[arg(long, default_value_t = 0.25)]
    actor_max_padding_waste: f64,

    /// Maximum time the oldest ready observation waits for a batch.
    #[arg(long, default_value_t = 2)]
    actor_max_wait_ms: u64,

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

    /// Updates between fixed-seed greedy eval passes (0 = off). Default 50
    /// (tighter than Python's 300 so short smoke runs still see a number).
    #[arg(long, default_value_t = 50)]
    eval_every: u64,

    /// Episodes per greedy eval pass (fresh workers, seeds `w{i}-ep0`).
    #[arg(long, default_value_t = 8)]
    eval_episodes: usize,

    /// Evaluate on a dedicated owner thread without pausing training.
    /// Disable to retain the original synchronous actor evaluation.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    async_eval: bool,

    /// Dedicated asynchronous evaluation device (for example `cuda:2`).
    /// If omitted, `cuda:1` is selected when training uses one GPU on
    /// cuda:0 and at least two CUDA devices are visible. Otherwise eval
    /// falls back to the synchronous path.
    #[arg(long)]
    eval_device: Option<String>,

    /// Run one fixed-seed greedy benchmark and write its per-episode JSON,
    /// then exit without training. The checkpoint must match the current
    /// Rust observation/action schema exactly; VarStore loading is strict.
    #[arg(long)]
    benchmark_out: Option<String>,

    /// Run one greedy Node-engine episode, save GameRecord + `.debug.json`, exit.
    #[arg(long, default_value_t = false)]
    watch: bool,

    /// Policy safetensors for `--watch` (also accepted via `--resume`/`--init`).
    #[arg(long)]
    policy: Option<String>,

    /// GameRecord JSON output path for `--watch`.
    #[arg(long)]
    record: Option<String>,

    /// Episode seed for `--watch` (default watch0).
    #[arg(long, default_value = "watch0")]
    seed: String,

    /// Map override for `--watch` (default: first map in stage pool).
    #[arg(long)]
    map: Option<String>,

    /// Bot count override for `--watch`.
    #[arg(long)]
    bots: Option<u32>,

    /// Difficulty override for `--watch`.
    #[arg(long)]
    difficulty: Option<String>,

    /// Nations override for `--watch` (`default` / `disabled` / integer).
    #[arg(long)]
    nations: Option<String>,

    /// Max post-spawn decisions for `--watch`.
    #[arg(long, default_value_t = 1200)]
    max_steps: usize,

    /// Write `.debug.json` sidecar with `--watch` (default true).
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    debug: bool,

    #[arg(long, default_value_t = 200)]
    ckpt_every: u64,

    #[arg(long, default_value = "checkpoints")]
    ckpt_dir: String,

    /// Warm-start policy weights from a `.safetensors` (preferred) or
    /// legacy `.ot` VarStore dump (BC→RL or a previously exported
    /// checkpoint) without restoring TrainState. Ignored when `--resume`
    /// is also set. Policy interchange is safetensors-only via Rust
    /// `VarStore` — there is no Python `.pt` converter.
    #[arg(long)]
    init: Option<String>,

    /// Explicit strict V8.1 schema-v1 safetensors migration into the
    /// recurrent V8.2 policy. Existing tensors are copied exactly and only
    /// `recurrent.*` tensors retain their V8.2 initialization.
    #[arg(
        long,
        requires = "recurrent_policy",
        conflicts_with_all = ["init", "resume"]
    )]
    init_v81_recurrent: Option<String>,

    /// Resume from a previously-saved checkpoint (e.g.
    /// `checkpoints/latest.safetensors`; legacy `.ot` still accepted).
    /// Restores weights and training state (curriculum stage,
    /// entropy-floor scale, learning rate, total env steps, win-rate
    /// window, update counter) from the `.state.json` sidecar saved
    /// alongside it - see `train::TrainState`. AdamW's momentum/variance
    /// state is not restored (tch-rs exposes no optimizer state_dict
    /// save/load); use `--resume-warmup-updates` (default 100) so LR
    /// warms back in while moments rebuild.
    #[arg(long)]
    resume: Option<String>,

    /// Permit the one supported cross-schedule resume: a V8.1 stage-5
    /// checkpoint becomes V8.1.1 stage 5 (the new Easy bridge), with its
    /// old Medium-stage win window cleared.
    #[arg(
        long,
        alias = "migrate-v81-stage5",
        default_value_t = false,
        requires_all = ["v811_curriculum", "resume"]
    )]
    migrate_v81_stage5_to_v811: bool,

    /// Permit a V8.1.1 stage-5 checkpoint to adopt the V8.2 stage 5.
    /// The old win window and env sizing are cleared.
    #[arg(
        long,
        default_value_t = false,
        requires_all = ["v82_curriculum", "resume"]
    )]
    migrate_v811_stage5_to_v82: bool,

    /// Permit only a V8.2 stage-5 checkpoint to adopt V8.3 stage 5.
    #[arg(
        long,
        default_value_t = false,
        requires_all = ["v83_curriculum", "resume"]
    )]
    migrate_v82_to_v83: bool,

    /// Permit a V8.3 checkpoint to adopt the V8.4 reward profile (same
    /// curriculum schedule / weights; BPTT/rollout may change).
    #[arg(
        long,
        default_value_t = false,
        requires_all = ["v83_curriculum", "resume"]
    )]
    migrate_v83_to_v84: bool,

    /// Extra LR warmup updates applied after `--resume` while AdamW
    /// moments rebuild from scratch (tch cannot dump/restore optimizer
    /// state). 0 disables the post-resume boost (stage warmup still
    /// applies). Default 100.
    #[arg(long, default_value_t = 100)]
    resume_warmup_updates: u64,

    /// Value-loss form: `mse` (default; Python `F.mse_loss` parity) or
    /// `huber` (Rust stabilizer escape hatch after the 2026-07-12
    /// explosion).
    #[arg(long, default_value = "mse", value_parser = ["huber", "mse"])]
    value_loss: String,

    /// Opt-in: automatically grow `--num-envs` toward `--target-gpu-util`.
    /// With `--persistent-actors`, growth checkpoints and exits via
    /// `restart_request.json` (same path as stage env targets); legacy
    /// collectors still spawn workers in-process. See `autoscale.rs` and
    /// `train::run`. Off by default; `pod_train_v8.sh` enables it for V8.3+.
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

fn parse_stage_env_targets(spec: &str, stage_count: usize) -> anyhow::Result<Vec<usize>> {
    let parts: Vec<&str> = spec
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    anyhow::ensure!(!parts.is_empty(), "--stage-env-targets cannot be empty");
    if !parts.iter().any(|part| part.contains('=')) {
        let values = parts
            .iter()
            .map(|value| value.parse::<usize>().map_err(anyhow::Error::from))
            .collect::<anyhow::Result<Vec<_>>>()?;
        anyhow::ensure!(
            values.len() == stage_count,
            "--stage-env-targets requires {stage_count} values, got {}",
            values.len()
        );
        anyhow::ensure!(
            values.iter().all(|&value| value > 0),
            "env targets must be positive"
        );
        return Ok(values);
    }

    let mut values = vec![0; stage_count];
    let mut assigned = vec![false; stage_count];
    for part in parts {
        let (range, value) = part
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("mixed list/range stage env target {part:?}"))?;
        let value: usize = value.parse()?;
        anyhow::ensure!(value > 0, "env targets must be positive");
        let (start, end) = if let Some(start) = range.strip_suffix('+') {
            (start.parse::<usize>()?, stage_count.saturating_sub(1))
        } else if let Some((start, end)) = range.split_once('-') {
            (start.parse::<usize>()?, end.parse::<usize>()?)
        } else {
            let stage = range.parse::<usize>()?;
            (stage, stage)
        };
        anyhow::ensure!(
            start <= end && end < stage_count,
            "stage target range {range:?} is outside 0..{}",
            stage_count.saturating_sub(1)
        );
        for stage in start..=end {
            anyhow::ensure!(
                !assigned[stage],
                "stage {stage} has more than one env target"
            );
            values[stage] = value;
            assigned[stage] = true;
        }
    }
    let missing: Vec<usize> = assigned
        .iter()
        .enumerate()
        .filter_map(|(stage, assigned)| (!assigned).then_some(stage))
        .collect();
    anyhow::ensure!(
        missing.is_empty(),
        "missing env targets for stages {missing:?}"
    );
    Ok(values)
}

fn resolve_eval_device(
    explicit: Option<&str>,
    async_eval: bool,
    train_device: Device,
    num_gpus: usize,
    cuda_devices: i64,
) -> anyhow::Result<Option<Device>> {
    if !async_eval {
        return Ok(None);
    }
    if let Some(value) = explicit {
        let device = match value {
            "cuda" => Device::Cuda(0),
            value if value.starts_with("cuda:") => {
                let index = value["cuda:".len()..]
                    .parse::<usize>()
                    .map_err(|_| anyhow::anyhow!("invalid --eval-device {value:?}"))?;
                Device::Cuda(index)
            }
            _ => anyhow::bail!("--eval-device must be cuda or cuda:N"),
        };
        if let Device::Cuda(index) = device {
            anyhow::ensure!(
                (index as i64) < cuda_devices,
                "--eval-device cuda:{index} is not visible ({cuda_devices} CUDA device(s))"
            );
            let occupied = match train_device {
                Device::Cuda(train_index) if num_gpus <= 1 => index == train_index,
                Device::Cuda(_) => index < num_gpus,
                _ => false,
            };
            anyhow::ensure!(
                !occupied,
                "--eval-device cuda:{index} is used by training; select a spare GPU"
            );
        }
        return Ok(Some(device));
    }
    if num_gpus == 1 && train_device == Device::Cuda(0) && cuda_devices > 1 {
        Ok(Some(Device::Cuda(1)))
    } else {
        Ok(None)
    }
}

#[cfg(test)]
mod eval_device_tests {
    use super::resolve_eval_device;
    use tch::Device;

    #[test]
    fn defaults_to_cuda_one_only_for_single_cuda_zero_training() {
        assert_eq!(
            resolve_eval_device(None, true, Device::Cuda(0), 1, 2).unwrap(),
            Some(Device::Cuda(1))
        );
        assert_eq!(
            resolve_eval_device(None, true, Device::Cuda(1), 1, 2).unwrap(),
            None
        );
        assert_eq!(
            resolve_eval_device(None, true, Device::Cuda(0), 2, 4).unwrap(),
            None
        );
    }

    #[test]
    fn explicit_device_must_be_visible_and_spare() {
        assert_eq!(
            resolve_eval_device(Some("cuda:3"), true, Device::Cuda(0), 2, 4).unwrap(),
            Some(Device::Cuda(3))
        );
        assert!(
            resolve_eval_device(Some("cuda:1"), true, Device::Cuda(0), 2, 4)
                .unwrap_err()
                .to_string()
                .contains("used by training")
        );
        assert!(resolve_eval_device(Some("cpu"), true, Device::Cuda(0), 1, 2).is_err());
    }

    #[test]
    fn disabling_async_eval_forces_synchronous_fallback() {
        assert_eq!(
            resolve_eval_device(Some("cuda:1"), false, Device::Cuda(0), 1, 2).unwrap(),
            None
        );
    }
}

#[cfg(test)]
mod stage_env_target_tests {
    use super::parse_stage_env_targets;

    #[test]
    fn parses_v81_range_string() {
        assert_eq!(
            parse_stage_env_targets("0-5=24,6=12,7=10,8+=8", 11).unwrap(),
            vec![24, 24, 24, 24, 24, 24, 12, 10, 8, 8, 8]
        );
    }

    #[test]
    fn parses_full_list_and_rejects_gaps_or_duplicates() {
        assert_eq!(parse_stage_env_targets("4,4,3", 3).unwrap(), vec![4, 4, 3]);
        assert!(parse_stage_env_targets("0=4,2+=2", 4).is_err());
        assert!(parse_stage_env_targets("0-2=4,2+=2", 4).is_err());
        assert!(parse_stage_env_targets("4,4", 3).is_err());
    }
}

#[cfg(test)]
mod curriculum_flag_tests {
    use super::Args;
    use clap::Parser;

    #[test]
    fn versioned_curricula_are_opt_in_and_mutually_exclusive() {
        let defaults = Args::try_parse_from(["oftrain"]).unwrap();
        assert!(!defaults.v81_curriculum);
        assert!(!defaults.v811_curriculum);
        assert!(!defaults.v82_curriculum);
        assert!(!defaults.v83_curriculum);
        assert!(!defaults.v9_curriculum);
        assert!(!defaults.v9_sparse_win);

        let v811 = Args::try_parse_from(["oftrain", "--v811-curriculum"]).unwrap();
        assert!(v811.v811_curriculum);
        let v82 = Args::try_parse_from(["oftrain", "--v82-curriculum"]).unwrap();
        assert!(v82.v82_curriculum);
        let v83 = Args::try_parse_from(["oftrain", "--v83-curriculum"]).unwrap();
        assert!(v83.v83_curriculum);
        let v9 = Args::try_parse_from(["oftrain", "--v9-curriculum", "--v9-sparse-win"]).unwrap();
        assert!(v9.v9_curriculum);
        assert!(v9.v9_sparse_win);
        assert!(
            Args::try_parse_from(["oftrain", "--v81-curriculum", "--v811-curriculum"]).is_err()
        );
        assert!(Args::try_parse_from(["oftrain", "--v81-curriculum", "--v82-curriculum"]).is_err());
        assert!(Args::try_parse_from(["oftrain", "--v82-curriculum", "--v83-curriculum"]).is_err());
        assert!(
            Args::try_parse_from(["oftrain", "--v811-curriculum", "--v82-curriculum"]).is_err()
        );
        assert!(Args::try_parse_from(["oftrain", "--v83-curriculum", "--v9-curriculum"]).is_err());
    }

    #[test]
    fn migration_requires_v811_and_resume() {
        assert!(Args::try_parse_from(["oftrain", "--migrate-v81-stage5-to-v811"]).is_err());
        let args = Args::try_parse_from([
            "oftrain",
            "--v811-curriculum",
            "--resume",
            "latest.safetensors",
            "--migrate-v81-stage5",
        ])
        .unwrap();
        assert!(args.migrate_v81_stage5_to_v811);

        assert!(Args::try_parse_from(["oftrain", "--migrate-v811-stage5-to-v82"]).is_err());
        let args = Args::try_parse_from([
            "oftrain",
            "--v82-curriculum",
            "--resume",
            "latest.safetensors",
            "--migrate-v811-stage5-to-v82",
        ])
        .unwrap();
        assert!(args.migrate_v811_stage5_to_v82);

        assert!(Args::try_parse_from(["oftrain", "--migrate-v82-to-v83"]).is_err());
        let args = Args::try_parse_from([
            "oftrain",
            "--v83-curriculum",
            "--resume",
            "latest.safetensors",
            "--migrate-v82-to-v83",
        ])
        .unwrap();
        assert!(args.migrate_v82_to_v83);
    }

    #[cfg(feature = "native-engine")]
    #[test]
    fn v82_maps_all_load_through_the_rust_engine() {
        use openfront_engine::core::terrain::{GameMapSize, load_fresh_terrain};
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repository root");
        for map in ofcore::curriculum::V82_MAPS {
            let terrain = load_fresh_terrain(&root, map, GameMapSize::Normal)
                .unwrap_or_else(|error| panic!("V8.2 map key {map:?} failed to load: {error}"));
            assert!(terrain.game_map.width > 0, "{map} has zero width");
            assert!(terrain.game_map.height > 0, "{map} has zero height");
            assert!(!terrain.nations.is_empty(), "{map} has no nations");
        }
    }
}

#[cfg(test)]
mod recurrent_flag_tests {
    use super::Args;
    use clap::Parser;

    #[test]
    fn recurrent_policy_is_opt_in_and_requires_persistent_actors() {
        let defaults = Args::try_parse_from(["oftrain"]).unwrap();
        assert!(!defaults.recurrent_policy);
        assert_eq!(defaults.recurrent_hidden_size, 256);
        assert_eq!(defaults.bptt_chunk_len, 16);
        assert!(Args::try_parse_from(["oftrain", "--recurrent-policy"]).is_err());

        let enabled = Args::try_parse_from([
            "oftrain",
            "--persistent-actors",
            "--recurrent-policy",
            "--recurrent-hidden-size",
            "128",
            "--bptt-chunk-len",
            "32",
        ])
        .unwrap();
        assert!(enabled.recurrent_policy);
        assert_eq!(enabled.recurrent_hidden_size, 128);
        assert_eq!(enabled.bptt_chunk_len, 32);

        let warm = Args::try_parse_from([
            "oftrain",
            "--persistent-actors",
            "--recurrent-policy",
            "--init-v81-recurrent",
            "v81.safetensors",
        ])
        .unwrap();
        assert_eq!(warm.init_v81_recurrent.as_deref(), Some("v81.safetensors"));
        assert!(
            Args::try_parse_from([
                "oftrain",
                "--persistent-actors",
                "--recurrent-policy",
                "--init-v81-recurrent",
                "v81.safetensors",
                "--resume",
                "v81.safetensors",
            ])
            .is_err()
        );
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
    let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") else {
        return;
    };
    // Rough dependency order (cublasLt before cublas, cusparse/cublas
    // before cusolver, etc.) - matches the order torch's own preloader
    // uses; harmless if a library has no such ordering constraint.
    const ORDER: &[&str] = &[
        "cusparselt",
        "nvtx",
        "nvjitlink",
        "cuda_nvrtc",
        "cuda_runtime",
        "cuda_cupti",
        "cublas",
        "cufft",
        "curand",
        "cudnn",
        "cusparse",
        "cusolver",
        "nccl",
        "cufile",
        "nvshmem",
    ];
    let dirs: Vec<&str> = ld_path.split(':').collect();
    // Drive the walk from ORDER, not from LD_LIBRARY_PATH's own entry
    // order, so load order matches torch's regardless of how the launch
    // script happened to list these directories.
    for pkg in ORDER {
        let Some(base) = dirs.iter().find(|dir| {
            std::path::Path::new(dir)
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                == Some(*pkg)
        }) else {
            continue;
        };
        let Ok(entries) = std::fs::read_dir(base) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                continue;
            };
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
                eprintln!(
                    "[oftrain] dlopen(libcuda.so.1) failed: {:?}",
                    std::ffi::CStr::from_ptr(libc::dlerror())
                );
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
    anyhow::ensure!(
        args.actor_max_batch > 0,
        "--actor-max-batch must be positive"
    );
    anyhow::ensure!(
        args.actor_target_batch > 0 && args.actor_target_batch <= args.actor_max_batch,
        "--actor-target-batch must be in 1..=--actor-max-batch"
    );
    anyhow::ensure!(
        args.actor_max_padding_waste.is_finite()
            && (0.0..=1.0).contains(&args.actor_max_padding_waste),
        "--actor-max-padding-waste must be in [0, 1]"
    );
    anyhow::ensure!(
        args.v81_dom_coef.is_finite() && args.v81_dom_coef >= 0.0,
        "--v81-dom-coef must be finite and non-negative"
    );
    anyhow::ensure!(
        args.v81_potential_clamp.is_finite() && args.v81_potential_clamp >= 0.0,
        "--v81-potential-clamp must be finite and non-negative"
    );
    anyhow::ensure!(
        args.v81_dominance_threshold.is_finite()
            && (0.0..=1.0).contains(&args.v81_dominance_threshold),
        "--v81-dominance-threshold must be in [0, 1]"
    );
    anyhow::ensure!(
        args.v81_delta_loss_dominant.is_finite() && args.v81_delta_loss_dominant >= 0.0,
        "--v81-delta-loss-dominant must be finite and non-negative"
    );
    anyhow::ensure!(
        args.v81_churn_coef.is_finite() && args.v81_churn_coef >= 0.0,
        "--v81-churn-coef must be finite and non-negative"
    );
    anyhow::ensure!(
        args.v83_close_coef.is_finite() && args.v83_close_coef >= 0.0,
        "--v83-close-coef must be finite and non-negative"
    );
    anyhow::ensure!(
        args.v83_churn_coef.is_finite() && args.v83_churn_coef >= 0.0,
        "--v83-churn-coef must be finite and non-negative"
    );
    anyhow::ensure!(
        args.v84_boat_useful.is_finite(),
        "--v84-boat-useful must be finite"
    );
    anyhow::ensure!(
        args.v84_boat_destroyed.is_finite(),
        "--v84-boat-destroyed must be finite"
    );
    anyhow::ensure!(
        args.v84_boat_cancelled.is_finite(),
        "--v84-boat-cancelled must be finite"
    );
    anyhow::ensure!(
        args.v84_boat_own_shore.is_finite(),
        "--v84-boat-own-shore must be finite"
    );
    anyhow::ensure!(
        args.v84_tempo_coef.is_finite() && args.v84_tempo_coef >= 0.0,
        "--v84-tempo-coef must be finite and non-negative"
    );
    anyhow::ensure!(
        args.v84_fast_win_coef.is_finite() && args.v84_fast_win_coef >= 0.0,
        "--v84-fast-win-coef must be finite and non-negative"
    );
    anyhow::ensure!(
        args.v85_tempo_share_threshold.is_finite()
            && (0.0..=1.0).contains(&args.v85_tempo_share_threshold),
        "--v85-tempo-share-threshold must be in [0, 1]"
    );
    anyhow::ensure!(
        args.v85_extra_win_bonus.is_finite() && args.v85_extra_win_bonus >= 0.0,
        "--v85-extra-win-bonus must be finite and non-negative"
    );
    anyhow::ensure!(
        args.v85_embargo_bad_stop.is_finite(),
        "--v85-embargo-bad-stop must be finite"
    );
    anyhow::ensure!(
        args.v85_embargo_good_stop.is_finite(),
        "--v85-embargo-good-stop must be finite"
    );
    anyhow::ensure!(
        args.v85_premature_retreat.is_finite(),
        "--v85-premature-retreat must be finite"
    );
    anyhow::ensure!(
        args.v85_thrash_reengage.is_finite(),
        "--v85-thrash-reengage must be finite"
    );
    anyhow::ensure!(
        args.v86_delta_loss.is_finite() && args.v86_delta_loss >= 0.0,
        "--v86-delta-loss must be finite and non-negative"
    );
    anyhow::ensure!(
        args.v86_death_penalty.is_finite() && args.v86_death_penalty >= 0.0,
        "--v86-death-penalty must be finite and non-negative"
    );
    let reward_config = ofcore::curriculum::RewardConfig {
        gamma: args.gamma as f64,
        v81_dom_coef: args.v81_dom_coef,
        v81_min_stage: args.v81_min_stage,
        v81_potential_clamp: args.v81_potential_clamp,
        v81_dominant_loss: args.v81_dominant_loss,
        v81_dominance_threshold: args.v81_dominance_threshold,
        v81_delta_loss_dominant: args.v81_delta_loss_dominant,
        v81_churn_coef: args.v81_churn_coef,
        v81_churn_window: args.v81_churn_window,
        v81_churn_min_stage: args.v81_churn_min_stage,
        v83_close_coef: args.v83_close_coef,
        v83_churn_coef: args.v83_churn_coef,
        v84_boat_useful: args.v84_boat_useful,
        v84_boat_destroyed: args.v84_boat_destroyed,
        v84_boat_cancelled: args.v84_boat_cancelled,
        v84_boat_own_shore: args.v84_boat_own_shore,
        v84_boat_min_stage: args.v84_boat_min_stage,
        v84_tempo_coef: args.v84_tempo_coef,
        v84_tempo_min_stage: args.v84_tempo_min_stage,
        v84_fast_win_coef: args.v84_fast_win_coef,
        v85_tempo_share_threshold: args.v85_tempo_share_threshold,
        v85_extra_win_bonus: args.v85_extra_win_bonus,
        v85_embargo_bad_stop: args.v85_embargo_bad_stop,
        v85_embargo_good_stop: args.v85_embargo_good_stop,
        v85_embargo_min_stage: args.v85_embargo_min_stage,
        v85_premature_retreat: args.v85_premature_retreat,
        v85_thrash_reengage: args.v85_thrash_reengage,
        v85_combat_min_stage: args.v85_combat_min_stage,
        v86_delta_loss: args.v86_delta_loss,
        v86_attack_symmetric_loss: args.v86_attack_symmetric_loss,
        v86_skip_combat_churn: args.v86_skip_combat_churn,
        v86_death_penalty: args.v86_death_penalty,
        v9_sparse_win: args.v9_sparse_win,
    };
    anyhow::ensure!(
        args.v9_curriculum == args.v9_sparse_win,
        "--v9-curriculum and --v9-sparse-win must be used together"
    );
    let curriculum_schedule = if args.v9_curriculum {
        ofcore::curriculum::CurriculumSchedule::V9
    } else if args.v83_curriculum {
        ofcore::curriculum::CurriculumSchedule::V83
    } else if args.v82_curriculum {
        ofcore::curriculum::CurriculumSchedule::V82
    } else if args.v811_curriculum {
        ofcore::curriculum::CurriculumSchedule::V811
    } else if args.v81_curriculum {
        ofcore::curriculum::CurriculumSchedule::V81
    } else {
        ofcore::curriculum::CurriculumSchedule::Legacy
    };
    let stage_count = ofcore::curriculum::stages_for_schedule(curriculum_schedule).len();
    anyhow::ensure!(
        args.stage < stage_count,
        "--stage {} is outside 0..{}",
        args.stage,
        stage_count - 1
    );
    let stage_env_targets = match args.stage_env_targets.as_deref() {
        Some(spec) => parse_stage_env_targets(spec, stage_count)?,
        None if curriculum_schedule == ofcore::curriculum::CurriculumSchedule::V82 => {
            ofcore::curriculum::V82_ENV_TARGETS.to_vec()
        }
        None if curriculum_schedule == ofcore::curriculum::CurriculumSchedule::V83 => {
            ofcore::curriculum::V83_ENV_TARGETS.to_vec()
        }
        None if curriculum_schedule == ofcore::curriculum::CurriculumSchedule::V9 => {
            ofcore::curriculum::V9_ENV_TARGETS.to_vec()
        }
        None if curriculum_schedule == ofcore::curriculum::CurriculumSchedule::V811 => {
            ofcore::curriculum::V811_ENV_TARGETS.to_vec()
        }
        None if curriculum_schedule == ofcore::curriculum::CurriculumSchedule::V81 => {
            ofcore::curriculum::V81_ENV_TARGETS.to_vec()
        }
        None => Vec::new(),
    };
    let initial_num_envs = stage_env_targets
        .get(args.stage)
        .copied()
        .unwrap_or(args.num_envs);
    tch::manual_seed(0);
    let device = parse_device(&args.device);
    anyhow::ensure!(
        !args.recurrent_policy || matches!(device, Device::Cuda(_)),
        "--recurrent-policy requires a CUDA --device"
    );
    if args.recurrent_policy {
        anyhow::ensure!(
            args.recurrent_hidden_size == policy::RECURRENT_HIDDEN as usize,
            "--recurrent-hidden-size must match the policy core ({})",
            policy::RECURRENT_HIDDEN
        );
        anyhow::ensure!(args.bptt_chunk_len > 0, "--bptt-chunk-len must be positive");
    }
    let eval_device = resolve_eval_device(
        args.eval_device.as_deref(),
        args.async_eval,
        device,
        args.num_gpus,
        Cuda::device_count(),
    )?;
    println!("[oftrain] device={device:?}");
    println!("[oftrain] curriculum schedule={}", curriculum_schedule.id());
    if curriculum_schedule == ofcore::curriculum::CurriculumSchedule::V9
        || curriculum_schedule == ofcore::curriculum::CurriculumSchedule::V83
    {
        println!(
            "[oftrain] reward profile={}",
            reward_config.reward_profile_id()
        );
    }
    println!(
        "[oftrain] v81 reward: min_stage={} K_DOM={} gamma={} phi_clamp={} \
         dominant_loss={} threshold={} W_DELTA_LOSS_DOMINANT={} \
         churn_coef={} churn_window={} churn_min_stage={}",
        reward_config.v81_min_stage,
        reward_config.v81_dom_coef,
        reward_config.gamma,
        reward_config.v81_potential_clamp,
        reward_config.v81_dominant_loss,
        reward_config.v81_dominance_threshold,
        reward_config.v81_delta_loss_dominant,
        reward_config.v81_churn_coef,
        reward_config.v81_churn_window,
        reward_config.v81_churn_min_stage,
    );
    if args.async_eval {
        match eval_device {
            Some(eval_device) => println!("[oftrain] async eval device={eval_device:?}"),
            None => println!(
                "[oftrain] no spare eval GPU selected; using synchronous evaluation \
                 (set --eval-device cuda:N to override)"
            ),
        }
    }

    if args.watch {
        let policy = args
            .policy
            .as_deref()
            .or(args.resume.as_deref())
            .or(args.init.as_deref())
            .ok_or_else(|| anyhow::anyhow!("--watch requires --policy (or --resume/--init)"))?;
        let run = std::path::Path::new(policy)
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("policy");
        let record = args
            .record
            .clone()
            .unwrap_or_else(|| format!("records-rl/{run}_s{}_{}.json", args.stage, args.seed));
        return watch::run_watch(watch::WatchConfig {
            policy,
            record: std::path::PathBuf::from(record),
            ae_ckpt: &args.ckpt,
            coarse_ckpt: args.coarse_ckpt.as_deref(),
            stage: args.stage,
            seed: args.seed.clone(),
            map: args.map.clone(),
            bots: args.bots,
            difficulty: args.difficulty.clone(),
            nations: args.nations.clone(),
            max_steps: args.max_steps,
            debug: args.debug,
            device,
            amp: args.amp,
            foveate: args.foveate,
            gc: args.gc,
            blocks: args.blocks,
            curriculum_schedule,
            reward_config,
        });
    }

    if let Some(out) = &args.benchmark_out {
        let checkpoint = args
            .resume
            .as_deref()
            .or(args.init.as_deref())
            .ok_or_else(|| anyhow::anyhow!("--benchmark-out requires --resume or --init"))?;
        return train::run_benchmark(train::BenchmarkConfig {
            checkpoint,
            output: out,
            ae_ckpt: &args.ckpt,
            coarse_ckpt: args.coarse_ckpt.as_deref(),
            stage: args.stage,
            episodes: args.eval_episodes,
            max_ticks: args.max_episode_ticks,
            engine: args.engine,
            device,
            amp: args.amp,
            foveate: args.foveate,
            gc: args.gc,
            blocks: args.blocks,
            recurrent_policy: args.recurrent_policy,
            pinned_h2d: args.pinned_h2d,
            fp16_rollout: args.fp16_rollout,
            compact_rollout: args.compact_rollout,
            reward_config,
            curriculum_schedule,
        });
    }

    let cfg = train::Config {
        num_envs: initial_num_envs,
        num_gpus: args.num_gpus,
        stage: args.stage,
        curriculum_schedule,
        migrate_v81_stage5_to_v811: args.migrate_v81_stage5_to_v811,
        migrate_v811_stage5_to_v82: args.migrate_v811_stage5_to_v82,
        migrate_v82_to_v83: args.migrate_v82_to_v83,
        migrate_v83_to_v84: args.migrate_v83_to_v84,
        migrate_v84_to_v85: args.migrate_v84_to_v85,
        migrate_v85_to_v86: args.migrate_v85_to_v86,
        stage_env_targets,
        max_episode_ticks: args.max_episode_ticks,
        rollout_len: args.rollout_len,
        updates: args.updates,
        lr: args.lr,
        gamma: args.gamma,
        reward_config,
        lambda: args.lambda_,
        clip: args.clip,
        vf_coef: args.vf_coef,
        ret_clip: args.ret_clip,
        adv_clip: args.adv_clip,
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
        ae_ckpt: args.ckpt,
        coarse_ckpt: args.coarse_ckpt,
        gc: args.gc,
        blocks: args.blocks,
        pinned_h2d: args.pinned_h2d,
        fp16_rollout: args.fp16_rollout,
        compact_rollout: args.compact_rollout,
        pipeline_groups: args.pipeline_groups,
        persistent_actors: args.persistent_actors,
        recurrent_policy: args.recurrent_policy,
        recurrent_hidden_size: args.recurrent_hidden_size,
        bptt_chunk_len: args.bptt_chunk_len,
        work_conserving_actors: args.work_conserving_actors,
        actor_max_batch: args.actor_max_batch,
        actor_target_batch: args.actor_target_batch,
        actor_max_padding_waste: args.actor_max_padding_waste,
        actor_max_wait: std::time::Duration::from_millis(args.actor_max_wait_ms),
        device,
        engine: args.engine,
        node_fraction: args.node_fraction.clamp(0.0, 1.0),
        log_every: args.log_every,
        eval_every: args.eval_every,
        eval_episodes: args.eval_episodes,
        async_eval: args.async_eval,
        eval_device,
        ckpt_every: args.ckpt_every,
        ckpt_dir: args.ckpt_dir,
        init: args.init,
        init_v81_recurrent: args.init_v81_recurrent,
        resume: args.resume,
        resume_warmup_updates: args.resume_warmup_updates,
        value_loss: match args.value_loss.as_str() {
            "huber" => train::ValueLoss::Huber,
            _ => train::ValueLoss::Mse,
        },
        auto_scale_envs: args.auto_scale_envs,
        target_gpu_util: args.target_gpu_util,
        // Never scale below whatever the run was explicitly started with
        // (`--num-envs`) unless the user gave an explicit floor of their own.
        // Do not seed from `initial_num_envs` (stage_env_targets[--stage]),
        // which is often an early-stage 24 and would fight late-stage floors.
        min_envs: args.min_envs.unwrap_or(args.num_envs),
        max_envs: args.max_envs,
        autoscale_check_every: args.autoscale_check_every,
        autoscale_step: args.autoscale_step,
    };
    train::run(cfg)
}
