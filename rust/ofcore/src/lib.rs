//! Shared, Python-free core for the V8 Rust PPO trainer (`oftrain`).
//!
//! Fresh v7 port of the former Python `rl/obs.py` + `rl/curriculum.py` +
//! `rl/ppo_translate.py`. Re-implements current (v7) featurization from
//! live bridge / native engine JSON.

pub mod curriculum;
pub mod feat;
pub mod translate;

/// Single source of truth for the train/watch episode tick budget.
///
/// Used by `oftrain --max-episode-ticks` (clap default), watch truncation,
/// and ofhub showcase when `SHOWCASE_MAX_EPISODE_TICKS` is unset. Keep
/// `scripts/pod_train_v11.sh` in sync when changing this.
///
/// OpenFront `msPerTick() = 100`, so 21000 ticks ≈ 35 in-game minutes.
pub const DEFAULT_MAX_EPISODE_TICKS: i64 = 21_000;

/// Watch / `rl.watch` advances this many sim ticks per policy decision.
pub const WATCH_TICKS_PER_DECISION: i64 = 10;

/// Decision cap that cannot undercut [`DEFAULT_MAX_EPISODE_TICKS`].
pub const DEFAULT_WATCH_MAX_STEPS: usize =
    (DEFAULT_MAX_EPISODE_TICKS / WATCH_TICKS_PER_DECISION) as usize + 64;

/// Watch decision cap for an arbitrary tick budget (tick budget is authoritative).
pub fn watch_max_steps_for_ticks(max_episode_ticks: i64) -> usize {
    let ticks = max_episode_ticks.max(0);
    ((ticks / WATCH_TICKS_PER_DECISION) as usize).saturating_add(64).max(64)
}

#[cfg(test)]
mod episode_budget_tests {
    use super::*;

    #[test]
    fn watch_steps_cover_default_tick_budget() {
        assert_eq!(DEFAULT_WATCH_MAX_STEPS, 2164);
        assert!(
            (DEFAULT_WATCH_MAX_STEPS as i64) * WATCH_TICKS_PER_DECISION
                >= DEFAULT_MAX_EPISODE_TICKS
        );
        assert_eq!(
            watch_max_steps_for_ticks(DEFAULT_MAX_EPISODE_TICKS),
            DEFAULT_WATCH_MAX_STEPS
        );
    }
}

#[cfg(test)]
#[path = "translate_boat_build_tests.rs"]
mod translate_boat_build_tests;
