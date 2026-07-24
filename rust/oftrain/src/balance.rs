//! Adaptive PPO-epoch rebalancing so the one-step actor/learner pipeline
//! stays train-saturated when env collection is the long pole.
//!
//! ## Why this exists
//!
//! With persistent actors the update wall is `max(collect_s, train_s)`.
//! Mean GPU util tracks roughly `train_s / collect_s` (plus sparse actor
//! kernels during collect). On A100 V11 at the VRAM env ceiling we measured
//! `collect_s≈180`, `train_s≈82` → util stuck near ~62%. Growing envs is
//! blocked by VRAM; lengthening train via epochs is the lever that does
//! not need more memory.
//!
//! This module is pure decision logic (no CUDA / no trainer state) so the
//! hysteresis and clamp rules stay unit-testable.

/// Default target for `train_s / collect_s` under `--balance-train-collect`.
/// Slightly below 1.0 leaves a little slack so collect jitter does not
/// immediately push epochs past the useful range.
pub const DEFAULT_TARGET_RATIO: f64 = 0.92;

/// Only bump/drop epochs when the EMA ratio is this far from target.
pub const RATIO_BAND: f64 = 0.06;

/// EMA smoothing for the observed train/collect ratio (0 = instant, 1 = frozen).
pub const RATIO_EMA_ALPHA: f64 = 0.35;

/// Pure decision: given the current epoch count and a smoothed
/// `train_s / collect_s` ratio, return the next epoch count.
///
/// - Grow by 1 when `ratio < target - band` and below `max_epochs`
/// - Shrink by 1 when `ratio > target + band` and above `min_epochs`
/// - Otherwise hold
pub fn next_epochs(
    current: usize,
    train_over_collect: f64,
    target_ratio: f64,
    min_epochs: usize,
    max_epochs: usize,
) -> usize {
    let min_epochs = min_epochs.max(1);
    let max_epochs = max_epochs.max(min_epochs);
    let current = current.clamp(min_epochs, max_epochs);
    if !train_over_collect.is_finite() || train_over_collect <= 0.0 {
        return current;
    }
    let lo = target_ratio - RATIO_BAND;
    let hi = target_ratio + RATIO_BAND;
    if train_over_collect < lo && current < max_epochs {
        current + 1
    } else if train_over_collect > hi && current > min_epochs {
        current - 1
    } else {
        current
    }
}

/// Update the EMA of `train_s / collect_s`. Seeds on the first finite sample.
pub fn update_ratio_ema(prev: Option<f64>, train_s: f64, collect_s: f64) -> Option<f64> {
    if !(train_s.is_finite() && collect_s.is_finite()) || collect_s <= 1e-3 || train_s < 0.0 {
        return prev;
    }
    let sample = train_s / collect_s;
    Some(match prev {
        Some(ema) => ema * (1.0 - RATIO_EMA_ALPHA) + sample * RATIO_EMA_ALPHA,
        None => sample,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grows_when_train_is_clearly_shorter_than_collect() {
        // 82/180 ≈ 0.456 ≪ 0.92 - 0.06
        assert_eq!(next_epochs(2, 0.456, DEFAULT_TARGET_RATIO, 2, 8), 3);
    }

    #[test]
    fn holds_near_target() {
        assert_eq!(next_epochs(4, 0.93, DEFAULT_TARGET_RATIO, 2, 8), 4);
    }

    #[test]
    fn shrinks_when_train_overshoots_collect() {
        assert_eq!(next_epochs(6, 1.20, DEFAULT_TARGET_RATIO, 2, 8), 5);
    }

    #[test]
    fn respects_max_epochs_ceiling() {
        assert_eq!(next_epochs(8, 0.20, DEFAULT_TARGET_RATIO, 2, 8), 8);
    }

    #[test]
    fn respects_min_epochs_floor() {
        assert_eq!(next_epochs(2, 1.50, DEFAULT_TARGET_RATIO, 2, 8), 2);
    }

    #[test]
    fn non_finite_ratio_holds() {
        assert_eq!(next_epochs(3, f64::NAN, DEFAULT_TARGET_RATIO, 2, 8), 3);
        assert_eq!(next_epochs(3, -1.0, DEFAULT_TARGET_RATIO, 2, 8), 3);
    }

    #[test]
    fn ema_seeds_then_smooths() {
        let e0 = update_ratio_ema(None, 80.0, 200.0).unwrap();
        assert!((e0 - 0.4).abs() < 1e-9);
        let e1 = update_ratio_ema(Some(e0), 160.0, 200.0).unwrap();
        // 0.4 * 0.65 + 0.8 * 0.35 = 0.26 + 0.28 = 0.54
        assert!((e1 - 0.54).abs() < 1e-9);
    }
}
