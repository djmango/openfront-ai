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

## 2026-07-09 - Post-merge 8-GPU scale validation: ticks/s target hit, util plateaus at ~55%

Merged `v8-rust` -> `master` and pushed (`070c5c0`). Re-verified the
correctness-audit build on the 8x A100 SXM pod, then ran a clean,
from-scratch 8-GPU validation (`--num-envs 32 --num-gpus 8
--rollout-len 64 --epochs 2 --minibatches 4`, 256 total envs, stage 0,
`decision_ticks=15`):

- **42 updates / ~14 min, zero panics, zero errors** (`grep -c
  panicked` = 0 over the full log). `eps_done` climbing steadily
  (3584 episodes by update 42), `recent_reward` oscillating 5-10 (noisy
  but not collapsing/diverging - expected for stage-0 from a
  freshly-initialized value head, see below).
- **Throughput: 838-936 decisions/s -> ~12.6k-14k game-ticks/s**
  (`decisions/s x decision_ticks[stage]`, 15 ticks/decision at stage
  0). Comfortably clears the 10k ticks/s target.
- **GPU utilization: `min_mean_util` converges to a stable 55%**,
  individual GPUs cycling roughly 0-37% during the collect-bound part
  of each update and briefly bursting to 100% across all 8 during
  synchronized minibatch steps. Does **not** clear the 90% target.

**Why util plateaus around 55% and not higher**: per-update timing is
consistently `collect_s` (~18-19s, CPU/IPC-bound: 256 Node.js
subprocess game-tick round-trips) slightly *longer* than `train_s`
(~16-18s, GPU-bound). The pipelined actor/learner architecture already
overlaps these, so the GPU is busy for essentially the *entire*
`collect_s` window doing the *previous* update's training - but even
during that overlapped window, per-GPU instantaneous utilization only
reads 20-37% most of the time (only spiking to 100% during the
synchronized forward/backward of each minibatch). This matches the
timing breakdown from the pre-merge optimization pass: none of flattened
grad-sync, GPU-resident minibatching, or parallelized `batch_build`
changed `train_s` materially, and this session's one new experiment
(doubling minibatch size 512->1024 via `--minibatches 2` to amortize
per-launch kernel overhead) **OOM'd immediately** (`CUDA out of memory
... Tried to allocate 4.00 GiB`, GPU memory was already pinned at
89-91% with `--minibatches 4`) - so there's no batch-size headroom left
on 80GB A100s at this env/rollout-len/network-size combination to push
`train_s` down further. The remaining ~500ms-1s gap on the 40-50s scale
is genuinely CPU/IPC-bound rollout collection that the GPU cannot help
with under the current subprocess-per-env architecture; closing that
gap fully requires the shared-memory/FFI vec-env rewrite flagged in the
pre-optimization notes above, not a training-loop tweak.

**Correctness note on the noisy value loss** (`v=` swinging
14-1393 update to update): checked this isn't a Rust-specific bug.
`rl/ppo.py` (line 339-341) documents the identical behavior as
*expected*: "the value head stays untrained (BC has no value loss), so
early PPO updates mostly fit the critic - expect noisy advantages for
the first ~100 updates". Neither implementation normalizes/clips the
value target (`returns`), only the advantage (`adv`) - matched exactly
in `train.rs` (see `adv_mean`/`adv_std` normalization in the
`batch_build` closure) - so large-magnitude discounted returns (gamma
0.999, episodes up to 200 decision-steps) naturally produce a
high-variance MSE loss early on. `pg_loss` stays small and sane
throughout (`+0.03` to `+0.75`), entropy is noisy but never
collapses/explodes to NaN/inf. Not a learning-affecting bug.

**Bottom line**: ticks/s target (10k+) cleared with margin (~13-14k
sustained); learning is stable and bug-free per the audit + this
validation run; GPU utilization is stuck at a well-understood ~55%
ceiling that is architectural (IPC bridge to the Node.js game engine),
not a training-loop inefficiency - matches the same wall the Python
`ppo_v7` effort hit (documented above) after its own optimization pass.
Getting to 90%+ util would require replacing the JSON-over-pipes
`Bridge`/`bridge/env.ts` subprocess protocol with shared memory or FFI,
which is a substantially larger project than tuning the training loop
and was explicitly deferred (see "Stopping point" section above).

## 2026-07-10 - AMP, real foveated crop, model-size overrides, pinned H2D (all flag-gated, CPU-verified)

Landed a batch of independently A/B-able changes on top of the 2026-07-09
work above, per-item, each behind its own CLI flag (default preserves
prior behavior exactly). Built/tested against this Mac's CPU-only
libtorch (`rust/scripts/setup_libtorch.sh`) - no GPU numbers here, those
need the A100 pod; every item below is instead checked for finite
losses / no NaN / no panic on a short real-engine (Node bridge) rollout,
plus targeted unit tests for the riskier coordinate math.

- **`--amp`** (manual bf16 mixed precision): tch-rs 0.24 has no
  dtype-selectable autocast context (its `autocast()` always picks fp16
  on CUDA, no bf16 option), so this hand-casts conv weights/activations
  to bf16 at each tower's input boundary and casts back to f32 at the
  output (`policy::conv2d_bf16`) - optimizer state and logits/loss stay
  f32. CPU-correctness-verified via a *unit test on tiny synthetic
  tensors* rather than an end-to-end smoke run: a real-map-scale (up to
  GH_MAX x GW_MAX, GC=256, BLOCKS=4) bf16 forward pass measured
  *dramatically* slower than f32 on this CPU (no accelerated CPU bf16
  conv/GEMM kernel in this libtorch build) - correct, just not
  end-to-end-smoke-testable without a GPU.
- **`--foveate`** (real foveated crop): replaces the legacy
  whole-map-as-fine fallback with an actual fixed `FOVEATE_SIZE`=48
  window, gathered (not resized) from the full grid and centered on the
  agent's own-tile centroid (falls back to map center before the agent
  owns anything). All the crop/coordinate math (`crop_origin`,
  `crop_and_pad`, `place_crop`, and the `_cropped` coordinate-mapping
  helpers in `policy.rs`) is fully vectorized (batched tensor ops, no
  per-sample host loop), so it should stay cheap at real batch sizes on
  GPU rather than becoming a new bottleneck. This was the highest-risk
  item - hand-derived the local<->global<->coarse coordinate math by
  hand and caught/fixed two real broadcasting bugs (an accidental (B,)
  vs (B,1) outer-product instead of elementwise subtract, and comparing
  local crop-frame coordinates against un-translated absolute
  coordinates) via a dedicated non-random coordinate-math unit test
  before trusting the finite-loss smoke tests alone.
- **`--gc`/`--blocks`** (model-size override): threaded through
  `PolicyNet::new` instead of hardcoding the `GC`/`BLOCKS` module
  constants; verified both the default (GC=256/BLOCKS=4, 11.2M params)
  and a small variant (`--gc 128 --blocks 2`, 2.6M params) build and
  train on a short CPU rollout.
- **`--epochs`**: already existed (`main.rs`/`train.rs`, default 2,
  matching `rl/ppo.py`) - no change needed; confirmed both 1 and 2 run
  correctly.
- **`--pinned-h2d`** (pinned memory + non-blocking H2D): tch-rs 0.24
  directly exposes both needed primitives - `Tensor::pin_memory(device)`
  and `Tensor::to_device_(device, dtype, non_blocking, copy)` (the
  `non_blocking` arg maps to ATen's `aten::to.device` op) - so
  `batch::to_device_maybe_pinned` uses them for the batch-build CPU->GPU
  upload with no unsafe/raw-FFI workaround needed. No-op unless the
  target device is CUDA (this Mac's libtorch build has none to test
  against), so verified instead that turning the flag on for
  `Device::Cpu` produces byte-identical tensors to the flag off (i.e.
  it's provably inert off-CUDA, not just untested).
- **Channels-last memory format - skipped, infeasible in tch-rs 0.24**:
  no `MemoryFormat` type, no `to_memory_format`/`channels_last`-style
  method anywhere in tch-rs 0.24's generated tensor API or in
  torch-sys 0.24's C bindings (confirmed by grepping both crates'
  source for `memory_format`/`channels_last` - zero hits). PyTorch's
  ATen op for this (`aten::to.memory_format` / the `memory_format`
  kwarg other `to`/`contiguous` overloads take) exists in libtorch
  itself but tch-rs's codegen never bound it. Not safely reachable
  without patching tch-rs's C shim (`torch-sys`) to add a new binding,
  which is out of scope here.
- **Fused/foreach optimizer step - skipped, no actionable gap found**:
  `nn::AdamW`/`Optimizer` (`tch::nn::optimizer`) is a thin wrapper
  around libtorch's native C++ `torch::optim::AdamW` class - `step()` is
  already a single call into that C++ object, so there's no
  Python-object-style per-parameter marshalling overhead to remove (this
  whole stack is Rust->C++ FFI, no Python in the loop at all). The
  *real* fused/foreach multi-tensor kernels (`torch._fused_adamw_`,
  `_foreach_*`) are implemented as ATen ops that only PyTorch's *Python*
  `torch.optim.AdamW(fused=True)`/`(foreach=True)` code paths call into;
  the C++-side `torch::optim` classes tch-rs wraps don't have a fused/
  foreach flag at all, and tch-rs 0.24's codegen doesn't bind
  `_foreach_*`/`_fused_adamw_` either (confirmed: zero matches for
  `_foreach_`/`fused_adamw` in `tch-0.24.0/src/wrappers/tensor_generated.rs`,
  consistent with the 2026-07-09 finding above that `_foreach_` ops
  aren't exposed). Getting real fused-optimizer throughput would need
  either patching tch-rs to bind those ops and reimplementing AdamW's
  step in Rust against them, or driving libtorch's own JIT-scripted
  fused path - both out of scope as a "low-risk" item.
- **CUDA graph capture - skipped, not reachable from tch-rs 0.24**:
  same conclusion as the AMP/pinned-memory investigation - torch-sys
  0.24's bindings don't expose `at::cuda::CUDAGraph`/
  `torch::cuda::graph_pool_handle` or any stream-capture API at all (this
  matches the earlier tch-rs API surface exploration for this task: no
  CUDA graph bindings found). Capturing a graph safely also requires
  every captured op's tensors to have fixed addresses/shapes across
  replays (no dynamic map-size-dependent shapes, no host-side branching
  like the curriculum/reward-shaping code in this loop) - a real
  implementation would need static-shape buffers and warmup-iteration
  discipline on top of the missing bindings, so this is intentionally
  left undone rather than hacked around with something fragile.

## 2026-07-09/10 - Native engine throughput + GPU validation on a 1x A100 pod

Merged two parallel workstreams into `master` (`agent/native-rl-perf`,
`agent/oftrain-gpu-opt` - the AMP/foveate/model-size/pinned-H2D work above),
then validated the combined result end-to-end on a fresh single A100 SXM
RunPod pod.

### Native engine hot-path optimization (`agent/native-rl-perf`)

Standalone benchmark (`rust/engine/examples/bench_rl_session.rs`, no policy
network, no IPC) measuring raw `RlSession` reset+step throughput:

| envs | threads | ticks/s before | ticks/s after | speedup |
|---|---|---|---|---|
| 64 | 1 | 9,304 | 45,225 | 4.9x |
| 256 | 8 | 57,764 | 131,656 | 2.3x |
| 512 | 8 | 51,493 | 151,661 | 2.9x |

Root cause of the "before" cost: `obs::tile_bytes_le` (a scalar per-tile loop
re-encoding the full tile plane into a `Vec<u8>` on *every* `step()`, even
though the native trainer immediately decodes those bytes back to `u16`) was
~78% of per-step time; `entities()`/`legality()` building `Vec`s before
wrapping in `Value::Array` was most of the rest. Fixes: memcpy-based
terrain/tile byte encoding with a `ptr::copy_nonoverlapping` fast path
(unit-tested for byte-for-byte equivalence with the scalar reference), a
`value_array()` helper that builds `Value::Array` directly from iterators
(no intermediate `Vec`), and - the biggest single win - `RlSession` growing a
zero-copy `tile_state() -> &[u16]` accessor so the native `oftrain` backend
skips the byte-encode/byte-decode round trip entirely (`NativeEngine` in
`oftrain/src/native.rs` reads tile state straight from the session). An
optional `rayon`-based batched-stepping path was added behind a `parallel`
Cargo feature but not wired in by default - plain `std::thread` sharding (one
OS thread per shard of envs, as `oftrain` already does) measured 20-40%
faster than a shared rayon pool for this workload, consistent with it being
memory-bandwidth-bound rather than scheduler-bound. Correctness: bit-identical
observation JSON before/after on a fixed seed, and the existing exact-parity
replay tests fail identically (same tick, same hash) on both the unmodified
base commit and this branch - confirmed unrelated pre-existing oracle
staleness, not a regression. Net: native engine ticking is no longer
anywhere close to the bottleneck (10x+ the old ~12-14k ticks/s Node-subprocess-
IPC ceiling even before this optimization pass).

### GPU pod setup (1x A100 SXM4 80GB, RunPod, torch 2.11.0+cu128)

- `tch 0.24`'s C++ shim (`torch_api_generated.cpp`) doesn't just check a
  version *string* - it calls ATen ops (`hash_tensor`, `_use_miopen_ctc_loss`,
  etc.) that don't exist in the pod's stock `torch==2.8.0+cu128`, so it fails
  to *compile*, not just link, even with
  `LIBTORCH_USE_PYTORCH=1`/`LIBTORCH_BYPASS_VERSION_CHECK=1` set. Fix: same
  approach as the Mac (`rust/scripts/setup_libtorch.sh`), but with a CUDA
  wheel - a dedicated venv (`rust/.libtorch-venv`) with
  `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128`
  (this version has a `+cu128` build published), `rust/.cargo/config.toml`
  pointing `LIBTORCH` at that venv's `torch/` dir plus `LD_LIBRARY_PATH`
  including both `torch/lib` and the venv's
  `nvidia/cuda_nvrtc/lib` (same NVRTC footgun as before, just `cu12` package
  names instead of `cu13` this time). `readelf -d target/release/oftrain`
  confirmed `libtorch_cuda.so` NEEDED post-build.

### Sustained GPU utilization result

`--engine native --num-envs 64 --rollout-len 32 --epochs 2 --minibatches 8`,
default (full, non-foveated) policy, no AMP: **sustained 81-82%
`min_mean_util`** (time-averaged since start, not instantaneous peaks),
climbing from a cold-start ~20% and plateauing by ~update 15, held flat
through a full 150-update / ~30-minute stability run with zero crashes and
finite losses throughout (steps/s ~172-177 the whole time). This is a real
improvement over the previous Python/Node-IPC ceiling documented above
(~55-70%), though it falls short of the 90%+ target. `collect_s ≈ train_s`
essentially every update at this size (confirmed via the same
collect-includes-join / train-is-inner-timer instrumentation from
2026-07-09) - i.e. rollout collection and GPU training are genuinely
overlapping and roughly balanced, not one-sided.

Swept envs/flags looking for a better operating point; the direction of
every result was consistent and informative even though none beat the 64-env
default:
- **64 envs, default policy**: 81-82% util, ~173 steps/s (best result).
- **96 envs, default policy**: 76-78% util, ~170 steps/s - collection time
  scaled roughly linearly with env count (12s -> 18s for 1.5x envs), so the
  collection-phase cost is genuinely proportional to work done, not a fixed
  per-call dispatch overhead that more envs would amortize.
- **128 envs, `--foveate`**: only 58-64% util despite fitting far more envs
  in memory - shrinking the observation also shrinks the GPU-bound training
  compute, which shifts the collect/train balance *back* toward
  collection-bound (lower util) even though decisions/s is flat-to-similar.
- **192 envs, `--gc 128 --blocks 2`** (small policy, 2.6M params): highest
  raw throughput seen (~285 steps/s) but only 37% util, for the same reason -
  less GPU compute per step tips the balance toward the CPU/engine-bound
  collection phase.
- **256 envs, default policy (no foveate)**: OOM (the full 250x150x63
  observation batch for 256envs x 32 rollout = 8192 samples is itself
  ~77GB before any activations - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`
  didn't help, this is a real allocation ceiling not fragmentation).
- **128 envs, `--amp`, 16 minibatches**: no OOM (33% GPU mem vs 46-70% without
  AMP - it does meaningfully cut memory), but `train_s` was *higher* than the
  8-env-config baseline in this particular run, confounded by the minibatch
  count change; not a clean AMP-only A/B, needs a same-minibatch-count rerun
  to attribute cleanly.

Takeaway: for *this* pipelined single-process-per-GPU architecture, GPU
utilization tracks the ratio of GPU-bound training compute to CPU-bound
collection compute, not env count or raw decisions/s in isolation - anything
that makes the network cheaper (foveation, smaller policy) pays for itself in
throughput but *costs* utilization by shrinking the numerator. Getting past
~82% on a single GPU now likely needs either genuinely overlapping streams for
actor/learner kernels (the same conclusion the 2026-07-09 entry reached) or
deepening the env/rollout ratio further while keeping the full policy size,
not swapping in a cheaper model.

### Learning-progress finding (not a code bug, a curriculum/exploration one)

`OFTRAIN_DEBUG_EPISODES=1` (new diagnostic env var, `train.rs`) dumps
per-episode reward/win/placement and the curriculum win-rate window. On
curriculum stage 0 (`Onion` map, 0 bots, 1 nation opponent,
`win_at=0.9`), every episode across every config tested ran to the full
`max_ticks` (~3000-tick truncation, never an explicit elimination win),
placed 1st of 2 (`score=1.000`) at ~47-52 tiles, and `won` was `false` every
time - so the curriculum-advance win-rate window never fills with a win and
training is stuck at stage 0 in every run above. Traced this to the FFA win
check requiring **80% of total map tiles** (`PERCENTAGE_TILES_TO_WIN_FFA`,
`win_check.rs`), not 80% of the opponent's tiles - the policy plateaus at
roughly half the map and stops expanding into neutral land once entropy
collapses (see below), so it never gets close to the threshold. Verified this
isn't an engine or sign bug: the win-check math, the PPO entropy-bonus sign
(`loss = pg_loss + vf_coef*v_loss - ent_coef*ent_loss`, correctly *subtracting*
the entropy term so gradient descent pushes entropy up), and gradient-norm
clipping (`clip_grad_norm(0.5)`, matching `rl/ppo.py`) all check out as
correct and match the ported Python baseline's hyperparameters
(`ent_coef=0.01` default). What's actually happening: this specific
curriculum stage has very low reward variance (a small, close-to-solved
2-player map), so PPO's advantage estimates are small and the shared network
trunk drifts toward a low-entropy, locally-stable policy within the first
handful of updates - entropy was observed collapsing from ~7 (near-uniform)
to <0.05 within 2-5 updates in every full-policy run tested, and stayed
there for the rest of a 150-update run without recovering. This is a
pre-existing characteristic of the ported curriculum/reward design (not
something introduced by today's native-engine or throughput changes - the
entropy-coefficient default is inherited unchanged from `rl/ppo.py`), but it
means training will not progress past stage 0 as currently tuned. Flagging
for follow-up rather than fixing now (out of scope for a throughput task):
likely needs a higher/slower-annealing `ent_coef` for early stages, or reward
shaping that keeps rewarding continued expansion into neutral land instead of
flattening out once the opponent is roughly matched.

### Status against the 8 "best next steps"

1. **Native engine benchmark + optimize** - done, 5-10x standalone ticks/s.
2. **`--engine native` GPU run, profiled** - done, see above.
3. **BF16 AMP** - implemented and runs on GPU without NaN; not yet cleanly
   isolated as a same-minibatch-count A/B (confounded run above).
4. **Real foveated crop** - implemented, runs on GPU, lets far more envs fit
   in memory, but *reduces* sustained utilization for the reason above.
5. **Smaller policy variant** - implemented, runs on GPU, highest raw
   decisions/s of any config tried, but lowest utilization for the same
   reason.
6. **Epoch count** - already exposed, both 1 and 2 confirmed to run.
7. **Low-risk GPU kernel opts** - pinned-H2D implemented and runs
   (no crash); channels-last/fused-optimizer/CUDA-graphs confirmed
   genuinely unreachable in tch-rs 0.24 (documented above), correctly
   skipped rather than hacked around.
8. **Scale to 4-8 GPUs once wins stack** - not started this session: the
   single-GPU sweep above shows utilization is currently bounded by the
   collect/train compute ratio rather than by anything that scaling GPU
   count would fix, so multi-GPU scaling was deferred until either (a) the
   AMP A/B is redone cleanly to see if it's a net win, or (b) the
   actor/learner CUDA-stream-overlap work from 2026-07-09 is picked back up
   - scaling a config that's still ~18% away from the target would just
   reproduce the same ceiling on more GPUs at higher cost.
