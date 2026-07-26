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
//! ## Scale-down: VRAM pressure (v2)
//!
//! Grow-only util search overshot on dense late-stage maps: host OOM-killer
//! SIGKILLed `oftrain` (exit 137) every ~30–40 min at 20 envs/shard while
//! util still looked "low enough to grow." Persistent owners already resize
//! via `restart_request.json`, so shrinking is the same path as growing.
//!
//! Policy:
//! - `mem >= MEM_SHRINK_FRAC` → step down toward `--min-envs`
//! - `mem >= MEM_BLOCK_GROW_FRAC` → hold (never grow into pressure)
//! - else, if `util < target - GROW_BAND`, jump to
//!   `n' = n * target / util` (linear util∝envs estimate), step-aligned and
//!   capped by VRAM headroom (`n * MEM_BLOCK / mem`) and `--max-envs`.
//!   One restart replaces the old `+step` climb that cost ~8 cold restarts
//!   to go 24→40 on A100s.
//!
//! ## Degrading without a GPU
//!
//! `gpu_util_frac: None` (no CUDA device, so no `GpuUtilSampler` - see
//! `train::run`) always holds the current count steady unless VRAM
//! pressure is reported independently. CPU headroom alone only tells you
//! it's *safe* to add more envs, never that it's *beneficial*. CPU
//! headroom is used instead as a hard ceiling (`cpu_env_cap_per_shard`,
//! backing `--max-envs 0`'s "auto" default).

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
/// every `--autoscale-check-every` updates - each flip means spawning (or
/// tearing down) real OS threads/subprocesses via restart, so it's worth
/// damping.
const GROW_BAND: f64 = 0.03;

/// Block util-driven growth once aggregate VRAM used/total reaches this
/// fraction. Leaves headroom below the shrink tripwire so we don't grow
/// into an immediate shrink restart.
pub const MEM_BLOCK_GROW_FRAC: f64 = 0.90;

/// Shrink one `--autoscale-step` when aggregate VRAM used/total reaches
/// this fraction. Tuned from A40 runs that OOM-killed near ~90–95% mem
/// with 20 envs on bot-heavy stage-6 maps.
pub const MEM_SHRINK_FRAC: f64 = 0.94;

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

/// Estimate envs needed to hit `target` util, assuming rough linearity
/// `util ∝ envs` (valid while collect-bound / under-saturated - the A100
/// sweep that motivated this module). Cap by VRAM headroom so a single
/// jump does not overshoot `MEM_BLOCK_GROW_FRAC`. Round the delta up to a
/// multiple of `step` (minimum one step).
///
/// `n' = min(max, max(n+step, n * target/util), n * MEM_BLOCK/mem)`.
fn util_jump_target(
    current: usize,
    util: f64,
    mem: Option<f64>,
    target: f64,
    step: usize,
    max: usize,
) -> usize {
    let util = util.max(0.01);
    let mut estimated = ((current as f64) * (target / util)).ceil() as usize;
    if let Some(mem) = mem.filter(|m| *m > 0.0) {
        // Headroom to the grow-block line: if mem is 60% of VRAM, we can
        // scale envs by ~0.90/0.60 before hitting MEM_BLOCK (assuming mem
        // also scales roughly with env count).
        let mem_cap = ((current as f64) * (MEM_BLOCK_GROW_FRAC / mem)).floor() as usize;
        estimated = estimated.min(mem_cap.max(current));
    }
    let min_next = current.saturating_add(step).min(max);
    let desired = estimated.max(min_next).min(max);
    let delta = desired.saturating_sub(current);
    let stepped_delta = ((delta + step - 1) / step) * step;
    current.saturating_add(stepped_delta).min(max).max(min_next)
}

/// Pure decision function: given the current per-shard env count and the
/// latest GPU utilization / memory readings (0-1 fractions, or `None` if
/// there's no signal), returns the env count that should be live after
/// this check. Never panics on a malformed `(min, max)` - if `min > max`
/// (e.g. `--max-envs` set below `--num-envs`/`--min-envs`), `max` is
/// silently raised to `min` so the range is always non-empty and
/// `current` always ends up clamped into a valid range instead of the
/// trainer hanging or panicking on a bad CLI combination.
pub fn next_env_count(
    current: usize,
    gpu_util_frac: Option<f64>,
    gpu_mem_frac: Option<f64>,
    target: f64,
    min: usize,
    max: usize,
    step: usize,
) -> usize {
    let max = max.max(min);
    let current = current.clamp(min, max);
    let step = step.max(1);

    // VRAM pressure wins over util growth. Shrink even when util is
    // unknown - OOM kills do not wait for a clean util sample.
    if let Some(mem) = gpu_mem_frac {
        if mem >= MEM_SHRINK_FRAC && current > min {
            return current.saturating_sub(step).max(min);
        }
        if mem >= MEM_BLOCK_GROW_FRAC {
            return current;
        }
    }

    let Some(util) = gpu_util_frac else {
        // No GPU util signal: hold steady (see module doc) rather than
        // guessing. Mem pressure already handled above.
        return current;
    };
    if current >= max {
        return current;
    }
    if util < target - GROW_BAND {
        util_jump_target(current, util, gpu_mem_frac, target, step, max)
    } else {
        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grows_when_clearly_below_target() {
        // Util wants ceil(8*0.95/0.40)=19, but mem headroom caps at
        // floor(8*0.90/0.50)=14 → step-align up to 16.
        let next = next_env_count(8, Some(0.40), Some(0.50), 0.95, 4, 64, 4);
        assert_eq!(next, 16);
    }

    #[test]
    fn holds_steady_inside_the_hysteresis_band() {
        // 0.93 is above `target - GROW_BAND` (0.95 - 0.03 = 0.92), so this
        // should hold rather than keep growing on every check.
        let next = next_env_count(8, Some(0.93), Some(0.50), 0.95, 4, 64, 4);
        assert_eq!(next, 8);
    }

    #[test]
    fn holds_steady_exactly_at_target() {
        let next = next_env_count(8, Some(0.95), Some(0.50), 0.95, 4, 64, 4);
        assert_eq!(next, 8);
    }

    #[test]
    fn never_shrinks_on_high_util_alone() {
        let next = next_env_count(16, Some(1.0), Some(0.50), 0.95, 4, 64, 4);
        assert_eq!(next, 16);
    }

    #[test]
    fn shrinks_when_vram_is_critical() {
        // 0.95 >= MEM_SHRINK_FRAC (0.94) → shrink
        let next = next_env_count(20, Some(0.40), Some(0.95), 0.85, 8, 20, 2);
        assert_eq!(next, 18);
    }

    #[test]
    fn shrink_stops_at_min_envs() {
        let next = next_env_count(10, Some(0.40), Some(0.95), 0.85, 8, 20, 4);
        assert_eq!(next, 8);
    }

    #[test]
    fn grows_under_previous_mem_block_threshold() {
        // 0.85 < MEM_BLOCK so growth is allowed, but mem headroom only
        // fits ~one step (floor(16*0.90/0.85)=16) so we take min +step.
        let next = next_env_count(16, Some(0.40), Some(0.85), 0.85, 8, 32, 2);
        assert_eq!(next, 18);
    }

    #[test]
    fn blocks_growth_when_vram_is_elevated() {
        // 0.91 >= MEM_BLOCK_GROW_FRAC (0.90) → hold
        let next = next_env_count(16, Some(0.40), Some(0.91), 0.85, 8, 20, 2);
        assert_eq!(next, 16);
    }

    #[test]
    fn shrinks_on_vram_even_without_util_signal() {
        let next = next_env_count(20, None, Some(0.95), 0.85, 8, 20, 2);
        assert_eq!(next, 18);
    }

    #[test]
    fn clamps_growth_step_to_the_max_bound() {
        // Below target, but the jump would overshoot max - clamp, don't
        // exceed it.
        let next = next_env_count(62, Some(0.10), Some(0.50), 0.95, 4, 64, 4);
        assert_eq!(next, 64);
    }

    #[test]
    fn already_at_max_holds_even_when_far_below_target() {
        let next = next_env_count(64, Some(0.01), Some(0.50), 0.95, 4, 64, 4);
        assert_eq!(next, 64);
    }

    #[test]
    fn current_above_max_gets_clamped_down_to_max() {
        // Shouldn't happen in practice (current is always <= a
        // previously-returned value <= max), but the function must not
        // return something out of range if it ever does.
        let next = next_env_count(100, Some(0.10), Some(0.50), 0.95, 4, 64, 4);
        assert_eq!(next, 64);
    }

    #[test]
    fn current_below_min_gets_clamped_up_to_min() {
        let next = next_env_count(1, Some(1.0), Some(0.50), 0.95, 4, 64, 4);
        assert_eq!(next, 4);
    }

    #[test]
    fn no_gpu_signal_holds_steady_regardless_of_bounds() {
        let next = next_env_count(8, None, None, 0.95, 4, 64, 4);
        assert_eq!(next, 8);
    }

    #[test]
    fn nonsensical_min_greater_than_max_never_panics_and_stays_in_range() {
        // --max-envs set below --min-envs/--num-envs: max gets raised to
        // min instead of producing an empty/invalid range.
        let next = next_env_count(4, Some(0.10), Some(0.50), 0.95, 32, 8, 4);
        assert_eq!(next, 32);
    }

    #[test]
    fn zero_step_still_makes_progress() {
        // `step.max(1)` inside the function means a misconfigured
        // `--autoscale-step 0` can't wedge growth forever. Mem headroom
        // at 50% caps the jump below the pure util estimate (38).
        let next = next_env_count(4, Some(0.10), Some(0.50), 0.95, 4, 64, 0);
        assert_eq!(next, 7); // floor(4*0.90/0.50)=7
    }

    #[test]
    fn near_target_util_only_takes_minimum_step() {
        // Grow when util < target - band (0.92 - 0.03 = 0.89). At 0.88:
        // 24*0.92/0.88 ≈ 25.1 → 26.
        let next = next_env_count(24, Some(0.88), Some(0.50), 0.92, 10, 40, 2);
        assert_eq!(next, 26);
    }

    #[test]
    fn underutilized_a100_jumps_using_util_and_mem_headroom() {
        // Live ppo_v11 shape: 24 envs @ 50% util / 60% mem, target 92%.
        // Util wants ~44, mem headroom allows floor(24*0.90/0.60)=36,
        // max=40 → one restart to 36 (not eight +2 climbs to 40).
        let next = next_env_count(24, Some(0.50), Some(0.60), 0.92, 10, 40, 2);
        assert_eq!(next, 36);
    }

    #[test]
    fn low_mem_pressure_allows_full_util_jump_to_max() {
        let next = next_env_count(24, Some(0.50), Some(0.40), 0.92, 10, 40, 2);
        assert_eq!(next, 40);
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
