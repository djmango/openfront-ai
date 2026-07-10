//! Pure, dependency-free (no `tch`, no trainer state) auto-scaling policy
//! for the number of parallel env workers, opt-in via `--auto-scale-envs`
//! (see `main.rs`/`train::Config`). Kept separate from `train.rs` so the
//! actual decision logic is unit-testable without a GPU, a policy net, or
//! a running trainer - see the `#[cfg(test)]` module below for the cases
//! that matter (below/at/above target, already at a bound, no GPU signal).
//!
//! ## Why this exists
//!
//! The 2026-07-09 A100 sweep (see DEVLOG's "law" writeup) found
//! `--num-envs 4` sustaining ~40% GPU utilization (and dipping to ~1% when
//! IPC-bound) while `--num-envs 64` on the same box sustained 98-100% -
//! discovered by hand, one manual restart at a time. This module makes
//! that search automatic: grow the env count while the GPU is measurably
//! under the target utilization, using the same "worst GPU's running
//! mean" signal (`GpuSnapshot::min_mean_util`, see `gpu_util.rs`'s module
//! doc for why the minimum-not-average is the right metric) that already
//! drives the existing utilization logging.
//!
//! ## Scale-down: not implemented (v1)
//!
//! `next_env_count` only ever grows or holds steady, never shrinks. Two
//! reasons: (1) low GPU utilization realistically means "the run started
//! with too few envs for this box", a static mis-sizing that autoscale
//! should correct once and be done with - it is not evidence of active
//! overshoot the way, say, high memory pressure would be, so there is no
//! symmetric "scale back down" signal to react to; (2) shrinking safely
//! requires tearing down `EnvWorker` threads/bridge subprocesses mid-run
//! (`spawn_worker`'s returned handle *does* support this - drop the
//! `choice_tx` sender to make the worker's `recv()` loop exit, then join -
//! see `run()`'s own shutdown code) and, more subtly, requires either
//! accepting a transient shape mismatch across shards for one update or
//! synchronizing the shrink across every shard atomically the same way
//! growth is (see `run()`); that's real additional complexity in exchange
//! for a case v1 doesn't need. If real runs show the GPU signal
//! oscillating enough to want shrinking (e.g. thermal throttling, another
//! process contending for the GPU), add hysteresis-gated shrink here as a
//! v2 - the `next_env_count` signature already takes everything a shrink
//! decision would need (`current`, `gpu_util_frac`, `target`, `min`).
//!
//! ## Degrading without a GPU
//!
//! `gpu_util_frac: None` (no CUDA device, so no `GpuUtilSampler` - see
//! `train::run`) always holds the current count steady. This is a
//! deliberate no-op, not a CPU-only scaling heuristic: CPU headroom alone
//! only tells you it's *safe* to add more envs, never that it's
//! *beneficial* to (that's what the GPU signal is for), and scaling up
//! forever with no way to tell "enough" on a CPU-only box would just burn
//! CPU for identical throughput. CPU headroom is used instead as a hard
//! ceiling (`cpu_env_cap_per_shard`, backing `--max-envs 0`'s "auto"
//! default) that bounds growth regardless of GPU signal.

use std::thread;

/// Reserve a couple of logical CPUs for the main/training thread and the
/// per-update helper threads `train_update` spawns per shard (the
/// batch-build and backward-pass threads, see `train.rs`) so the
/// auto-derived cap leaves the trainer's own machinery room to run, not
/// just env workers. Deliberately
/// small and fixed rather than scaled with shard count - env workers
/// mostly block on subprocess IPC or engine ticking, not CPU-bound
/// compute, so they oversubscribe far more gracefully than the trainer's
/// own compute threads would.
const RESERVED_CPU_THREADS: usize = 2;

/// Hysteresis band (as a 0-1 fraction of GPU utilization) around `target`:
/// only grow when utilization reads clearly below target
/// (`util < target - GROW_BAND`), otherwise hold. Without this, noise in
/// a single utilization sample near the target would flip the decision
/// every `--autoscale-check-every` updates - each flip means spawning (or,
/// if v2 adds shrink, tearing down) real OS threads/subprocesses, so it's
/// worth damping even though v1's grow-only policy can never oscillate
/// the count itself (see module doc); this just avoids needless resize
/// churn once already close to the target.
const GROW_BAND: f64 = 0.03;

/// Logical CPUs available to this process, minus the reserved margin
/// (floor of 1 so a tiny/CPU-starved box still gets *some* cap rather than
/// zero). Backs `--max-envs 0`'s "auto" default - see `cpu_env_cap_per_shard`.
pub fn cpu_total_env_cap() -> usize {
    let logical = thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    logical.saturating_sub(RESERVED_CPU_THREADS).max(1)
}

/// `cpu_total_env_cap` divided across `num_shards` (each shard spawns its
/// own env workers - see `train::Config::num_envs`'s "per shard" doc), so
/// the derived per-shard `--max-envs` default keeps the *total* env count
/// across all shards within the machine's CPU headroom rather than
/// multiplying the single-shard cap by `num_shards`.
pub fn cpu_env_cap_per_shard(num_shards: usize) -> usize {
    (cpu_total_env_cap() / num_shards.max(1)).max(1)
}

/// Pure decision function: given the current per-shard env count and the
/// latest GPU utilization reading (a 0-1 fraction, or `None` if there's no
/// GPU signal - see module doc), returns the env count that should be live
/// after this check. Never panics on a malformed `(min, max)` - if
/// `min > max` (e.g. `--max-envs` set below `--num-envs`/`--min-envs`),
/// `max` is silently raised to `min` so the range is always non-empty and
/// `current` always ends up clamped into a valid range instead of the
/// trainer hanging or panicking on a bad CLI combination.
pub fn next_env_count(current: usize, gpu_util_frac: Option<f64>, target: f64, min: usize, max: usize, step: usize) -> usize {
    let max = max.max(min);
    let current = current.clamp(min, max);
    let Some(util) = gpu_util_frac else {
        // No GPU signal: hold steady (see module doc's "degrading
        // without a GPU" section) rather than guessing.
        return current;
    };
    if current >= max {
        return current;
    }
    if util < target - GROW_BAND {
        (current + step.max(1)).min(max)
    } else {
        // At or above target (minus the hysteresis band): nothing to do.
        // v1 never shrinks even when comfortably above target - see
        // module doc.
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grows_when_clearly_below_target() {
        let next = next_env_count(8, Some(0.40), 0.95, 4, 64, 4);
        assert_eq!(next, 12);
    }

    #[test]
    fn holds_steady_inside_the_hysteresis_band() {
        // 0.93 is above `target - GROW_BAND` (0.95 - 0.03 = 0.92), so this
        // should hold rather than keep growing on every check.
        let next = next_env_count(8, Some(0.93), 0.95, 4, 64, 4);
        assert_eq!(next, 8);
    }

    #[test]
    fn holds_steady_exactly_at_target() {
        let next = next_env_count(8, Some(0.95), 0.95, 4, 64, 4);
        assert_eq!(next, 8);
    }

    #[test]
    fn never_shrinks_when_above_target() {
        let next = next_env_count(16, Some(1.0), 0.95, 4, 64, 4);
        assert_eq!(next, 16);
    }

    #[test]
    fn clamps_growth_step_to_the_max_bound() {
        // Below target, but the step would overshoot max - clamp, don't
        // exceed it.
        let next = next_env_count(62, Some(0.10), 0.95, 4, 64, 4);
        assert_eq!(next, 64);
    }

    #[test]
    fn already_at_max_holds_even_when_far_below_target() {
        let next = next_env_count(64, Some(0.01), 0.95, 4, 64, 4);
        assert_eq!(next, 64);
    }

    #[test]
    fn current_above_max_gets_clamped_down_to_max() {
        // Shouldn't happen in practice (current is always <= a
        // previously-returned value <= max), but the function must not
        // return something out of range if it ever does.
        let next = next_env_count(100, Some(0.10), 0.95, 4, 64, 4);
        assert_eq!(next, 64);
    }

    #[test]
    fn current_below_min_gets_clamped_up_to_min() {
        let next = next_env_count(1, Some(1.0), 0.95, 4, 64, 4);
        assert_eq!(next, 4);
    }

    #[test]
    fn no_gpu_signal_holds_steady_regardless_of_bounds() {
        let next = next_env_count(8, None, 0.95, 4, 64, 4);
        assert_eq!(next, 8);
    }

    #[test]
    fn nonsensical_min_greater_than_max_never_panics_and_stays_in_range() {
        // --max-envs set below --min-envs/--num-envs: max gets raised to
        // min instead of producing an empty/invalid range.
        let next = next_env_count(4, Some(0.10), 0.95, 32, 8, 4);
        assert_eq!(next, 32);
    }

    #[test]
    fn zero_step_still_makes_progress() {
        // `step.max(1)` inside the function means a misconfigured
        // `--autoscale-step 0` can't wedge growth forever.
        let next = next_env_count(4, Some(0.10), 0.95, 4, 64, 0);
        assert_eq!(next, 5);
    }

    #[test]
    fn cpu_env_cap_per_shard_is_never_zero_and_divides_across_shards() {
        let total = cpu_total_env_cap();
        assert!(total >= 1);
        let one_shard = cpu_env_cap_per_shard(1);
        assert_eq!(one_shard, total);
        let many_shards = cpu_env_cap_per_shard(total.max(2));
        assert!(many_shards >= 1);
        assert!(many_shards <= one_shard);
    }
}
