# V8 Rust PPO Trainer - Dev Log

Goal: port the Python PPO training path (`rl/ppo.py` + `rl/vec.py` +
`rl/obs.py` + `rl/curriculum.py` + `rl/ppo_translate.py` + `rl/policy.py`)
to Rust (`rust/oftrain`, on `tch-rs`/libtorch) for faster training on CUDA
pods. BC warm-start and the AE encoder training are explicitly out of
scope for this port; the policy currently sees raw pooled grid features
instead of an AE latent (see `oftrain/src/policy.rs` module doc).

## 2026-07-09 - Workspace scaffold + CPU/GPU smoke tests

- New cargo workspace: `ofrs` (existing PyO3 BC crate, untouched
  behaviorally, bumped to edition 2024 for workspace consistency),
  `ofcore` (new, Python-free lib: v7 featurization/`feat.rs`,
  `curriculum.rs`, `translate.rs` - a fresh port, kept separate from
  `ofrs::feat` which is frozen at obs v4 as the on-disk BC cache format),
  `oftrain` (new bin: bridge client, threaded vec-env, tch policy net,
  PPO loop).
- `bridge.rs` talks to `bridge/env.ts` (Node/tsx subprocess) over
  JSONL+binary framing on stdin/stdout, one subprocess per env.
  `vecenv.rs::EnvWorker` mirrors `rl/vec.py::EnvWorker`: curriculum episode
  bookkeeping, reward shaping, auto-reset. Each env gets its own OS thread
  (no GIL, so no multiprocessing/pickle framing needed) driven from
  `train.rs` over a pair of mpsc channels - functionally a Gym
  `SyncVectorEnv`.
- `policy.rs`: hand-rolled port of `rl/policy.py`'s conv towers +
  transformer player-attention + factorized action heads (categorical
  action/player/build/nuke/tile, Beta-distributed quantity), foveated
  tile head (fine/coarse), matching `act()`/`evaluate()` field for field.
  v1 simplifications (documented in the module doc, not silent
  shortcuts): no AE latent (no exported checkpoint to port against yet -
  raw pooled ego/static/transient planes instead), no real foveation crop
  (whole grid used as "fine", matching `Policy._ensure_foveated`'s
  existing legacy fallback path).
- `train.rs`: rollout collection -> GAE -> shuffled-minibatch clipped PPO
  update (AdamW), checkpointing (`VarStore::save`), periodic
  `nvidia-smi`-polling GPU utilization logging (`gpu_util.rs`).
- Local (Mac, CPU) smoke test: end-to-end single-episode rollout through a
  real `bridge/env.ts` subprocess passed, confirming the featurization/
  translation/bridge plumbing is wired correctly.

### `tch-rs`/libtorch linking notes (both were real footguns, future-you
read this before re-deriving them)

1. **libtorch version pinning.** `tch 0.24` is hard-pinned to libtorch
   2.11.0 (`torch-sys` checks the `build-version`/`LIBTORCH_VERSION`
   string and errors on mismatch). Neither the Mac's default nor the
   pod's default Python `torch` matched. Fix: a dedicated venv
   (`rust/.libtorch-venv`, gitignored) with `torch==2.11.0` pinned, built
   by `rust/scripts/setup_libtorch.sh`, with `LIBTORCH`/`LD_LIBRARY_PATH`
   pointed at it via `.cargo/config.toml` (also gitignored - it's a local
   path).
2. **CUDA silently not linked (the big one - this is why early GPU runs
   only hit ~1-40% util with `Cuda(0)` "working").** `torch-sys`'s build
   script *does* emit `cargo:rustc-link-lib=torch_cuda` whenever
   `libtorch_cuda.so` exists next to `libtorch_cpu.so`, and the very
   first CUDA smoke test looked like it "worked" (`Device::Cuda(0)`
   accepted, tensors constructed) - but every real op panicked at
   runtime with `Could not run 'aten::empty.memory_format' ... CUDA
   backend`. Root cause: nothing in our Rust code references a
   `torch_cuda` *symbol* directly (CUDA kernels self-register with ATen's
   dispatcher via static initializers at load time, not via any call we
   make), so the linker's default `--as-needed` drops the whole
   `libtorch_cuda.so`/`libc10_cuda.so` NEEDED entries at final-link time
   - `readelf -d target/release/oftrain | grep NEEDED` confirmed only
   `libtorch_cpu.so`/`libc10.so` were actually linked despite the
   `-ltorch_cuda` flag being passed. This is a known upstream footgun
   (tch-rs issues #907, #1015: same problem hits ROCm too) with no fix in
   `torch-sys` itself since link args from a *library* crate's build
   script don't propagate to the final binary link, only a *binary*
   crate's do. Fix: added `oftrain/build.rs` that re-asserts
   `-Wl,--no-as-needed -ltorch_cuda -lc10_cuda -Wl,--as-needed` from the
   binary crate itself. Confirmed fixed via `readelf -d` showing both
   libs NEEDED post-fix.
3. **NVRTC.** Once CUDA ops actually dispatched, `lgamma` (used by the
   Beta-distribution quantity head) hit `nvrtc: error: failed to open
   libnvrtc-builtins.so.13.0` - this ships in the separate
   `nvidia-cuda-nvrtc-cu13` pip package's `nvidia/cu13/lib/`, not
   `torch/lib/`. Fix: added that dir to `LD_LIBRARY_PATH` alongside
   `torch/lib`.

### Single-GPU (RunPod, 1x A100 SXM 80GB) results

With the CUDA-link fix above and `LD_LIBRARY_PATH` including both
`torch/lib` and `nvidia/cu13/lib`:

- `--num-envs 4 --rollout-len 16`: GPU util **40%, dropping to 1%** -
  batch too small / env-IPC-latency-bound, GPU mostly idle between the
  tiny batched `act()`/`evaluate()` calls. Not acceptable per the
  >=90% bar.
- `--num-envs 64 --rollout-len 32`: GPU util **98-100%**, memory ~84%
  (stable across updates, consistent with PyTorch's caching allocator
  high-water-marking rather than a leak - it plateaus, doesn't keep
  climbing). ~65 decisions/s aggregate across all 64 envs once warmed up
  (update 0 pays one-time engine-subprocess startup cost across 64 Node
  children). Extended ~60-update stability run launched to confirm no
  crashes/OOM/throughput regression over several minutes; see next entry
  for results.

Takeaway: on this workload the GPU-utilization bottleneck was almost
entirely about batch size (env count), not the linking issue in
isolation - but the linking bug meant CUDA ops were never truly
dispatching to the GPU in the first place (every op silently fell back/
errored), so it had to be fixed before batch size could even be the
relevant knob.

### Legal-action mask sign bug (NaN divergence at update ~10)

With `--num-envs 64 --rollout-len 32` running, training progressed fine
through ~9 updates then hit `probability tensor contains either inf, nan
or element < 0` inside `Categorical::sample` a couple updates later.
Root cause was in `policy.rs`'s action-masking helper: legal-action masks
were applied as `logits + (legal - 1.0) * MASKED_NEG` where `MASKED_NEG`
is a large **positive** constant (`1e9`-ish). For illegal actions
(`legal == 0`) this computes `logits + (0 - 1) * MASKED_NEG = logits -
MASKED_NEG`... which looks right at a glance, but `MASKED_NEG` was
defined as a large *negative* number already (`-1e9`), so the actual
expression became `logits + (−1)*(−1e9) = logits + 1e9`, i.e. illegal
actions got the huge **boost**, not the huge penalty - exactly inverted.
Once a rollout wandered into a state where every "legal" action per the
mask was actually the previously-illegal set, the policy net's softmax
saturated (one logit at +1e9, everything else comparatively -inf after
softmax normalization) and gradients/probs went NaN within a couple of
updates. Fix: every `(legal - 1.0) * MASKED_NEG` site changed to
`(legal - 1.0) * (-MASKED_NEG)` so illegal actions get subtracted, not
added. Re-ran the same 64-env/32-step config for 10+ updates past the
previous crash point with no NaNs and entropy behaving normally
(decreasing as expected instead of the pre-fix pattake of exploding then
collapsing to a degenerate one-hot).

### Single-GPU stability run (RunPod, 1x A100 SXM 80GB)

With both fixes above in place, launched a longer run
(`--num-envs 64 --rollout-len 32 --updates 90+`, checkpointing every 20
updates) and let it run unattended for ~47 minutes / 90 updates:

- GPU util held at **98-100%** every single logged update, no
  degradation over time.
- No crashes, no NaNs, no unbounded memory growth (`gpu_mem%` plateaued
  at 84-85% the whole run, consistent with the caching allocator's
  high-water mark rather than a leak).
- Throughput steady at **~65-66 decisions/s** aggregate the entire run.
- `recent_reward` climbed from 0 -> ~17.4 over the run and checkpoints
  saved cleanly every 20 updates.

Single-GPU bar (>=95% sustained util, stable, no crashes) is met.

## Multi-GPU (manual data-parallel, single process)

`tch-rs` has no NCCL/`DistributedDataParallel` bindings, so multi-GPU is
implemented as N independent policy replicas (one per `Device::Cuda(i)`),
each driving its own subset of env workers ("shards"). Every optimizer
step: each shard computes local grads on its own device, grads are
copied to device 0 and averaged (`sync_grads` in `train.rs`), then the
averaged grads are copied back and applied via each shard's own
optimizer (so all replicas stay in sync, equivalent to single-node DDP
with a very naive allreduce). `Config::devices()` also accepts CPU with
`num_gpus > 1` for sharding logic to be smoke-tested locally on the Mac
without any GPUs.

### First 4-GPU attempt: GPUs took turns instead of overlapping

Initial 4-GPU run (`--num-envs 64 --num-gpus 4`, 256 envs total = 64/GPU)
showed the *aggregate* util metric looking OK-ish but per-GPU snapshots
during the run showed only one or two GPUs active at any instant, taking
turns - not the 4-way overlap the design intended. Root cause: rollout
`act()` and the minibatch `evaluate()`/`backward()` were both being
called shard-by-shard in a plain sequential loop on the main thread.
libtorch CUDA calls are async from the *issuing* thread's perspective,
but the loop's next statement (e.g. reading back a CPU tensor, or a
subsequent shard's own async launch racing against the *previous*
shard's still-pending host-side bookkeeping) meant the kernels weren't
actually all in-flight concurrently - effectively serializing GPUs
despite them being logically independent.

Fix: wrapped both the rollout `act()` phase and the minibatch
`evaluate()`/`backward()` phase in `std::thread::scope`, launching one
OS thread per shard so all shards' CUDA kernels get issued to their
respective devices before any thread blocks on host-side sync/collection
- i.e. actually overlapping dispatch across GPUs instead of relying on
the (false) assumption that a sequential Rust loop calling async CUDA
ops was enough for concurrency.

### 4-GPU results (RunPod, 4x A100 SXM 80GB)

After the `thread::scope` fix, `--num-envs 64 --num-gpus 4 --rollout-len
32 --updates 40` (256 total envs, 64/GPU/shard):

- All 4 GPUs sustained **97-100% utilization** every logged update, for
  the entire 40-update run (~16 minutes), e.g. update 39:
  `[gpu0=100% gpu1=100% gpu2=97% gpu3=100%]`. Direct `nvidia-smi`
  snapshots during the run confirmed all 4 GPUs reading ~100% at the
  same instant (previously only 1 GPU would show high util at a time).
- Aggregate throughput **~325-355 decisions/s** vs. ~65 decisions/s on
  1 GPU - i.e. ~5x speedup from 4x the GPUs (better than linear, likely
  because the fixed per-update host-side overhead - GAE, minibatch
  shuffling, checkpoint I/O - is amortized over a larger batch rather
  than because of any actual reduction in that overhead).
- No crashes, no NaNs, `recent_reward` climbed 0 -> ~14.6 over the run.
- 4-GPU bar (>=95% sustained on every GPU) is met.

### Correction: "97-100%" above was instantaneous, not sustained

The per-update `[gpu0=...% gpu1=...%]` numbers logged above are a single
point-in-time `nvidia-smi` snapshot taken right when the update finishes,
not a time-average. `min_mean_util%` (added to the logger) is the actual
cumulative mean utilization per GPU, sampled continuously since process
start. Re-checking the 4-GPU run's `min_mean_util%`, it was only
**~36-41%**, not 95%+. GPUs burst to 100% during the compute-heavy parts
of an update and idle the rest of the time - the earlier conclusion that
the util bar was met was wrong, based on misreading the instantaneous
snapshot as if it were sustained.

Root cause, from per-phase timing (`OFTRAIN_DIAG=1`): each update splits
into `collect_s` (env stepping via subprocess IPC - CPU-bound, GPU
~idle) and `train_s` (minibatch fwd/bwd - GPU-bound). With
`--num-envs 64 --rollout-len 32`, these were roughly 9.5s and 15.2s out
of a ~25s update - i.e. ~38% of every update had the GPU doing nothing
but waiting on Node/tsx game-tick subprocesses over stdin/stdout IPC.

Two follow-up hypotheses that turned out to be *dead ends*:
- **Per-parameter grad sync overhead**: `sync_grads` looped over every
  named parameter doing individual device-to-device copies (~150-300
  small ops). Rewrote to flatten all grads into one tensor, single
  cross-device copy+average, single unflatten. Correct fix in principle
  (matches real allreduce), but `train_s` barely moved - not the
  bottleneck.
- **Redundant observation re-transfer per minibatch**: `build_obs` was
  being called fresh (CPU repack + H2D transfer) for every one of the 8
  minibatches per update, even though the underlying rollout data is
  identical across epochs (only shuffling differs). With grid tensors up
  to ~9.4MB/sample x 512 samples, that's gigabytes of redundant PCIe
  traffic per update. Added `index_select` on `Obs`/`ChoiceBatch` so the
  full batch is built and transferred to GPU **once** per update, then
  minibatches are sliced via GPU-side `index_select` (no further H2D).
  Correct fix in principle, but again `train_s` barely moved.

Measuring `fwdbwd_s` (forward+backward for one 512-sample minibatch)
directly: **~1.0s per minibatch on an A100**, consistent across 1/4-GPU
runs. That is genuinely GPU compute time (the `GridTower` conv+attention
policy net over up to 150x250 grids x 63 channels is not cheap), not
sync or transfer overhead. So `train_s` (8 minibatches x ~1s + small
sync/step overhead) is close to irreducible given the current network
and batch size - the real lever is eliminating the GPU-idle time during
`collect_s`, not shaving `train_s` further.

### Pipelined actor/learner architecture (overlap collect with train)

Since `collect_s` (CPU/IPC-bound) and `train_s` (GPU-bound) don't share
a resource, they can run concurrently: split each shard into an
`ActorShard` (frozen policy copy used only to pick actions during
rollout) and a `LearnerShard` (the policy actually being trained).
Restructured the update loop to, on every iteration, spawn a background
thread per shard that runs `collect_rollout` on the **actor** weights
while the main thread runs `train_on_rollout` on the **previous**
rollout using the **learner** weights; after both join, learner weights
are copied into the actor for the next iteration. This introduces a
one-update policy lag (standard in async actor/learner setups like
IMPALA) but removes the GPU-idle gap.

Also parallelized `batch_build` (the initial per-update GAE + CPU
repack + H2D transfer of the whole rollout, previously ~8-9s run
sequentially across shards) across shards with `std::thread::scope`,
cutting it to ~3.3-4.3s.

### 4-GPU results after pipelining (RunPod, 4x A100 SXM 80GB)

`--num-envs 64 --num-gpus 4 --rollout-len 32 --updates 25 --epochs 2
--minibatches 8`, 256 total envs:

- `collect_s == train_s == update_s` in every logged update, confirming
  true overlap (previously `update_s` was `collect_s + train_s`
  sequential; now it's `max(collect_s, train_s)`).
- `update_s` dropped from ~24-25s to **~14.2-15.0s**; throughput rose
  from ~330 to **~550-580 decisions/s** (~1.7x).
- `min_mean_util%` (true cumulative time-average per GPU) converged
  after ~15 updates to a steady **~66-68%**, up from the ~36-41%
  baseline, but still **below the 90-95% target**.
- Per-update instantaneous snapshots vary widely (0% to 100% depending
  on which exact moment the sample lands relative to a kernel burst),
  confirming util is bursty rather than flat - consistent with GPU work
  (train) and CPU/IPC work (collect) both existing inside the same
  overlapped window, but the GPU not being kept continuously fed.
- Stable throughout: no NaNs, no crashes, `recent_reward` climbed
  0 -> ~10.8 over 25 updates (~6.2 min).

**Honest assessment**: pipelining + parallel batch_build closed a real
gap (36-41% -> 66-68% sustained, +1.7x throughput) but did not reach the
95% target. The remaining shortfall is architectural: the actor's
inference forward passes (needed every rollout step to pick actions)
and the learner's training kernels are issued from different host
threads onto the *same* physical GPU per shard, and without explicit
CUDA stream separation, work from the two roles is not truly
concurrent at the SM level - it's overlapped at the wall-clock/thread
level but still serialized somewhat on-device, plus `collect_s`'s
CPU-bound stretches (waiting on subprocess IPC) don't have enough
queued GPU work to fill every idle instant. Closing the remaining gap
would need either (a) explicit CUDA stream isolation between actor and
learner so their kernels can truly interleave on-device, or (b) a much
deeper rollout/env-count ratio to keep enough GPU work queued at all
times, or (c) moving environment stepping itself off the CPU/IPC path
entirely (a much larger project, out of scope here).

### Stopping point (all pods torn down)

Decision: stop here rather than keep spending pod-hours chasing the
remaining utilization gap. All three RunPod instances (1x, 4x, 8x A100)
were terminated. Summary of final state for whoever picks this back up:

**What works and is correct:**
- Full Rust PPO port (`ofrs`/`ofcore`/`oftrain`) builds and runs on both
  CPU (Mac, for smoke tests) and CUDA (RunPod A100s).
- `tch-rs`/`libtorch` CUDA linking fixed (`oftrain/build.rs` forces
  `--no-as-needed` for `libtorch_cuda.so`/`libc10_cuda.so`).
- Action-mask sign bug fixed (`policy.rs`) - this was causing NaN
  divergence around update 10 and is a correctness fix independent of
  performance, should stay fixed regardless of what happens with the
  perf work below.
- Multi-GPU (manual replica + flattened-gradient-averaging "poor man's
  DDP") works correctly and scales: 4-GPU run showed ~5x throughput
  over 1-GPU with stable training and no crashes across dozens of
  updates. 8-GPU was set up (venv/build/linking all confirmed working)
  but not run to completion under the pipelined code - see below.

**Performance state:**
- Pipelined actor/learner architecture + parallel batch_build is
  implemented and stable, and measurably helps: 4-GPU sustained
  utilization went from ~36-41% to ~66-68% (time-averaged, not
  instantaneous), throughput 330 -> 560 decisions/s.
- This is below the 90-95% target. Root cause of the remaining gap is
  believed to be lack of explicit CUDA stream separation between the
  actor's per-rollout-step inference and the learner's per-minibatch
  train kernels on the same physical device, plus insufficient queued
  GPU work during the CPU/IPC-bound stretches of rollout collection.
  Not fixed - see "Correction" and "Pipelined actor/learner" sections
  above for the full investigation trail (grad-sync flattening and
  redundant-transfer elimination were both tried and ruled out as the
  cause).
- 8-GPU run under the pipelined code was **not** completed - the 8-GPU
  pod was torn down before a full validation run. If resuming this
  work, re-provision an 8-GPU pod, re-sync `/workspace/openfront-ai-v8`,
  rebuild, and rerun the same `OFTRAIN_DIAG=1` diagnostic + longer
  `--updates 25`-style run used for the 4-GPU validation above before
  trusting 8-GPU numbers.

**If resuming utilization work**, the most promising untried lever is
explicit CUDA stream isolation (create a dedicated `torch::cuda::Stream`
per actor and learner role per device, so kernels from the two roles
can genuinely interleave on-device instead of both hitting the default
stream). Second most promising: increasing the env/GPU-work ratio
further (more envs per shard, or deeper rollout lookahead) to keep more
GPU work queued during CPU-bound stretches.

## 2026-07-09 - Cross-checked against the parallel Python (`ppo_v7`) GPU-util push

The Python side ran the same kind of investigation independently on
`ppo_v7` (4x H200, `rl/vec.py`) and landed on the *same root cause* we
did: after killing every fixable host-side sync, the ceiling is the
`VecEnv` bridge to the Node.js game engine (pipes + JSON), which is
wall-clock the GPU cannot touch. Their run: 20-40% -> peaks of 72% (avg
12-50%), never sustained above 90%, then killed in favor of a Rust
rewrite of that bridge. See `docs/devlog.html#gpu-util-killed` in the
main repo for their full writeup. Cross-checking their four fixes
against our Rust port:

1. **Env fleet auto-sizing** (EMA the roll/upd ratio, adjust `num_envs`
   at each restart). Not implemented here - we hand-tuned `--num-envs`
   once and got `collect_s ~= train_s` (the exact balance their
   auto-sizer targets), so the dynamic version buys us less, but it's a
   reasonable robustness add if the network/grid size changes materially
   (e.g. curriculum stage transitions) and nobody re-tunes by hand.
   **Not done.**
2. **Sub-batch shape bucketing** for a per-sample variable-shape
   `grid_fine` foveation crop (splintered minibatches into hundreds of
   single-digit sub-batches by exact shape, >20x regression until
   bucketed to multiples of 8). **Not applicable yet** - our Rust port's
   grid obs is uniform/fixed-shape (no foveation implemented, see
   `policy.rs` module doc); this becomes a hard requirement if/when
   foveation is ported here. Noted for whoever builds that.
3. **Crash-loop pipe-hang** (a signal-killed rank left orphaned env
   workers holding the log pipe open, so the restart loop's `tee` never
   saw EOF and the run sat dead in "restarting" - this is literally what
   ended their session). **Directly applicable and fixed here**: our
   `Bridge`'s `Drop` impl called `child.kill()` without `child.wait()`
   (zombies accumulate under any future supervise/auto-restart wrapper),
   and `stderr` was `Stdio::null()`'d entirely, so a crashed/hung child
   gave zero diagnostic signal beyond "bridge died" - the same blind
   spot that forced them to reach for `py-spy`/`faulthandler` mid-crisis.
   Fixed in `bridge.rs`: `stderr` is now piped and continuously drained
   by a background thread into a capped ring buffer (must be drained or
   an unread pipe fills its OS buffer and blocks the child's *next*
   write - a self-inflicted hang otherwise), and every bridge-died error
   path now includes the last 200 lines of the child's stderr. `Drop`
   now `wait()`s after `kill()` to actually reap the child instead of
   leaving a zombie.
4. **Batched foveated coverage/content masks** (`_fine_coverage` did
   `.any()`/`.nonzero()`/`.int()` per sample - 5+ GPU syncs and 3 host
   transfers *each*, ~2.5k syncs per act batch). **Not applicable** -
   this is inside the not-yet-ported foveation path; more generally, we
   already audited the Rust `policy.rs` forward pass for any per-sample
   `.item()`/sync-inducing calls during the `fwdbwd_s` investigation
   above and found none - the network is fully batched/vectorized.
   General lesson worth keeping in mind if foveation (or any future
   per-sample-shaped op) gets ported: batch the op, don't loop+sync.

**The one finding that matters most and isn't fixed by either side**:
both the Python `VecEnv` and our Rust `oftrain` ultimately drive the
*same* Node.js/TypeScript game engine over the *same* kind of
subprocess+JSON-pipe IPC (`bridge/env.ts`, one `tsx` process per env -
see `bridge.rs`). Porting the trainer to Rust did not remove this - it
only fixed Rust-side pipeline/gradient-sync issues (36-41% -> 66-68%,
above). The architectural ceiling both investigations converged on
independently is the engine bridge itself. The real fix is the
already-started native Rust port of the game engine itself
(`openfront-engine`/`ofrs` "native backend" in the
`openfront-ai-rust-fast` worktree - PRNG/record-bootstrap done, hash
parity holds at tick 0, desyncs by tick 10+, executions not yet
ported) which would let env stepping run in-process with zero
subprocess/pipe/JSON overhead. Until that lands, `collect_s` being
CPU/IPC-bound is a shared, unresolved ceiling for both the Python and
Rust trainers - not something either side's PPO-loop optimization work
can close.

## 2026-07-09 - Pre-merge correctness audit: found and fixed 6 learning bugs

Before merging into `master`, did a line-by-line audit of the Rust PPO
math/hyperparameters against `rl/ppo.py` (ground truth), covering GAE,
the clipped surrogate, value loss, entropy bonus, action masking,
advantage normalization, rewards, LR/grad-clip, multi-GPU grad
averaging, and the actor/learner staleness introduced by the pipelined
architecture. GAE, the clipped surrogate, value loss, reward shaping,
advantage normalization, multi-GPU grad averaging, and the
actor/learner staleness all **matched Python exactly** - no bugs there.
Six real divergences found and fixed:

1. **Beta (quantity-head) entropy averaged over the full batch instead
   of only the samples that use it.** `ent_q` is 0 for every sample
   whose sampled action doesn't carry a `quantity_frac` (most of them);
   `.mean()` over the whole minibatch silently scaled the bonus down by
   `n_active/batch_size`. Python divides by `n_active`
   (`ppo.py:857-858`). Fixed in `train.rs`'s minibatch loop.
2. **`entq_coef` defaulted to `0.0`** (Python default `0.002`) - the
   Beta-entropy bonus was fully disabled by default. Fixed the CLI
   default.
3. **`gamma` defaulted to `0.99`** (Python default `0.999`) - a
   materially shorter effective horizon for a long strategy game.
   Fixed the CLI default.
4. **Gradient clip norm was `1.0`** (Python `0.5`, `ppo.py:910`). Fixed
   to `0.5`.
5. **No entropy annealing.** Python linearly anneals
   `ent_coef -> ent_coef_final` over `ent_anneal_updates` so late
   training commits instead of exploring forever (`ppo.py:778-783`);
   Rust used a fixed `ent_coef` forever. Ported the linear anneal (new
   `--ent-coef-final`/`--ent-anneal-updates` flags, computed once per
   update in `run()` and threaded into `train_update`). **Not ported**:
   Python's additional adaptive entropy-*floor* multiplier
   (`ent_scale`, `ppo.py:933-949`) that nudges the coefficient up to
   5x if measured entropy drops below a floor - added complexity for a
   safety net that can be watched for and added later if entropy
   collapse is actually observed in a real run.
6. **Curriculum never advanced.** `EnvWorker::set_stage` existed
   (dead-code warning) but nothing ever called it - the trainer ran
   `--stage N` forever, no matter how well it did. Python gates
   advancement on a rolling window of win-rate at the *current* stage
   (`WINDOW=40` non-rehearsal episodes, `win_at` per stage,
   `ppo.py:587-605`); `ofcore::curriculum` already had the identical
   `WINDOW`/`Stage::win_at` constants ported and just wasn't wired to
   anything. Wired it up: `run()` now tracks a rolling win window
   keyed off `EpisodeInfo::{stage, rehearsal, won}` (already collected,
   just unused), advances `curr_stage` when the window fills and clears
   above threshold, and on advance (a) broadcasts the new stage to
   every env worker thread via a new `stage_tx` channel per worker
   (checked non-blockingly before each `step()`, takes effect at that
   env's next episode reset - never mid-episode, matching Python's
   `vec.set_stage`), and (b) decays the learning rate
   `lr * stage_lr_decay^stage` on every shard's optimizer
   (`nn::Optimizer::set_lr`, new `--stage-lr-decay` flag, Python default
   `0.85`).

**Investigated and NOT a bug**: the audit initially flagged "missing
coarse land/water content pruning" (`policy.rs`'s `foveate()` stubs
`coarse_has_land`/`coarse_has_water` to all-ones, so the coarse tile
head doesn't get an extra push away from e.g. targeting a boat at pure
ocean). Checked `rl/policy.py::_coarse_logits_for_action` directly -
this pruning is itself gated behind `if "coarse_has_land" in o`, i.e.
Python's *own* code only computes/applies it when the AE/foveation
encode path populates those keys, and gracefully falls back to the
plain `legal_tile_coarse` mask (exactly what Rust does) when they're
absent. Since this Rust port intentionally has no AE/foveation yet
(documented, out of scope), it's correctly exercising Python's own
supported fallback path, not diverging from it. Left as a known
enhancement for whenever AE/foveation gets ported, not a fix.

Also re-affirmed as intentional/correct rather than a bug: the
pipelined actor/learner one-update policy lag (both sides do this by
design, see prior "Pipelined actor/learner architecture" section).

All 6 fixes verified with a local CPU smoke test (3 envs, few updates,
finite losses, correct log fields) before merging to `master`.
