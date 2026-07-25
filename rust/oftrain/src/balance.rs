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
//! Collect wall also drifts with stage/map length (observed 180→350s on the
//! same recipe). A static epoch count that was "good enough" at 180s falls
//! back to ~70% util when collect stretches. This module is the control
//! loop that keeps `train_s / collect_s` near a high target continuously.
//!
//! Pure decision logic (no CUDA / no trainer state) so hysteresis and clamp
//! rules stay unit-testable.

/// Default target for `train_s / collect_s` under `--balance-train-collect`.
/// Tuned for ≥90% mean util: with sparse actor work during the collect
/// tail, a 0.95 train/collect duty cycle lands mean util near 90–95%.
pub const DEFAULT_TARGET_RATIO: f64 = 0.95;

/// Only bump/drop epochs when the EMA ratio is this far from target.
pub const RATIO_BAND: f64 = 0.05;

/// When the EMA is this far *below* `(target - band)`, grow by 2 epochs
/// instead of 1 so collect spikes (180→350s) are recovered in one/two
/// updates rather than crawling +1 forever.
pub const FAST_GROW_GAP: f64 = 0.18;

/// EMA smoothing for the observed train/collect ratio (higher = faster).
pub const RATIO_EMA_ALPHA: f64 = 0.50;

/// High-water-mark blend for collect wall. Using a soft peak as the
/// denominator keeps epochs sized for recent long collects so util does
/// not collapse on the next spike after a short update.
pub const COLLECT_HWM_DECAY: f64 = 0.85;

/// Pure decision: given the current epoch count and a smoothed
/// `train_s / collect_s` ratio, return the next epoch count.
///
/// - Grow by 2 when `ratio < target - band - FAST_GROW_GAP`
/// - Grow by 1 when `ratio < target - band`
/// - Shrink by 1 when `ratio > target + band`
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
    if train_over_collect < lo - FAST_GROW_GAP && current + 1 < max_epochs {
        (current + 2).min(max_epochs)
    } else if train_over_collect < lo && current < max_epochs {
        current + 1
    } else if train_over_collect > hi && current > min_epochs {
        current - 1
    } else {
        current
    }
}

/// Soft high-water mark of collect wall time. Decays toward recent samples
/// but stays sticky on spikes so epoch sizing covers the long tail.
pub fn update_collect_hwm(prev: Option<f64>, collect_s: f64) -> Option<f64> {
    if !collect_s.is_finite() || collect_s <= 1e-3 {
        return prev;
    }
    Some(match prev {
        Some(hwm) => collect_s.max(hwm * COLLECT_HWM_DECAY),
        None => collect_s,
    })
}

/// Update the EMA of `train_s / collect_ref` (prefer collect HWM).
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
        // 82/180 ≈ 0.456 ≪ 0.95 - 0.05 - 0.18 → fast +2
        assert_eq!(next_epochs(2, 0.456, DEFAULT_TARGET_RATIO, 2, 12), 4);
    }

    #[test]
    fn grows_by_one_when_moderately_below_target() {
        // Just below lo (0.90) but inside fast-grow gap: +1
        assert_eq!(next_epochs(6, 0.88, DEFAULT_TARGET_RATIO, 4, 12), 7);
    }

    #[test]
    fn holds_near_target() {
        assert_eq!(next_epochs(8, 0.94, DEFAULT_TARGET_RATIO, 4, 12), 8);
    }

    #[test]
    fn shrinks_when_train_overshoots_collect() {
        assert_eq!(next_epochs(10, 1.15, DEFAULT_TARGET_RATIO, 4, 12), 9);
    }

    #[test]
    fn respects_max_epochs_ceiling() {
        assert_eq!(next_epochs(12, 0.20, DEFAULT_TARGET_RATIO, 4, 12), 12);
    }

    #[test]
    fn respects_min_epochs_floor() {
        assert_eq!(next_epochs(4, 1.50, DEFAULT_TARGET_RATIO, 4, 12), 4);
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
        // 0.4 * 0.50 + 0.8 * 0.50 = 0.60
        assert!((e1 - 0.60).abs() < 1e-9);
    }

    #[test]
    fn collect_hwm_sticks_to_spikes_then_decays() {
        let h0 = update_collect_hwm(None, 200.0).unwrap();
        assert!((h0 - 200.0).abs() < 1e-9);
        let h1 = update_collect_hwm(Some(h0), 350.0).unwrap();
        assert!((h1 - 350.0).abs() < 1e-9);
        let h2 = update_collect_hwm(Some(h1), 180.0).unwrap();
        // max(180, 350*0.85) = 297.5
        assert!((h2 - 297.5).abs() < 1e-9);
    }
}
