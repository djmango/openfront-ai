//! Doomsday Clock wave-schedule and drain math (TS `core/game/DoomsdayClock.ts`).
//!
//! This is the pure, integer-only calculation layer. The stateful per-tick
//! consumer - `Player` marking/draining, team grouping, crown exemption,
//! warship decay - is `crate::execution::doomsday_clock_execution`, wired into
//! `bootstrap.rs::game_from_record` the same way TS's `GameRunner.init()`
//! conditionally adds `DoomsdayClockExecution` (only when the config enables
//! it).

use super::schemas::DoomsdayClockSpeed;

const LEVELS: [i64; 6] = [300, 500, 1000, 2000, 3000, 5500]; // 3, 5, 10, 20, 30, 55%

struct WaveSchedule {
    grace_seconds: i64,
    ramp_seconds: i64,
    pause_seconds: i64,
}

fn schedule(speed: DoomsdayClockSpeed) -> WaveSchedule {
    match speed {
        DoomsdayClockSpeed::Normal => WaveSchedule {
            grace_seconds: 330,
            ramp_seconds: 270,
            pause_seconds: 30,
        },
        DoomsdayClockSpeed::Slow => WaveSchedule {
            grace_seconds: 390,
            ramp_seconds: 330,
            pause_seconds: 30,
        },
        DoomsdayClockSpeed::Fast => WaveSchedule {
            grace_seconds: 270,
            ramp_seconds: 170,
            pause_seconds: 30,
        },
        DoomsdayClockSpeed::VeryFast => WaveSchedule {
            grace_seconds: 180,
            ramp_seconds: 120,
            pause_seconds: 30,
        },
    }
}

fn required_basis_points(speed: DoomsdayClockSpeed, elapsed: i64) -> i64 {
    let s = schedule(speed);
    if elapsed <= s.grace_seconds {
        return 0;
    }
    let cycle = s.ramp_seconds + s.pause_seconds;
    let t = elapsed - s.grace_seconds;
    let i = t / cycle;
    if i as usize >= LEVELS.len() {
        return LEVELS[LEVELS.len() - 1];
    }
    let idx = i as usize;
    let into = t - i * cycle;
    let prev = if idx == 0 { 0 } else { LEVELS[idx - 1] };
    let target = LEVELS[idx];
    if into >= s.ramp_seconds {
        return target;
    }
    prev + (target - prev) * into / s.ramp_seconds
}

/// Base minimum tiles one player must own at `elapsed` game seconds.
pub fn doomsday_clock_required_tiles(speed: DoomsdayClockSpeed, land: i64, elapsed: i64) -> i64 {
    if land <= 0 {
        return 0;
    }
    required_basis_points(speed, elapsed) * land / 10000
}

/// Threshold a whole side must hold: the base per-player share scaled by the
/// side's headcount (min size 1), capped at the whole map.
pub fn doomsday_clock_side_required_tiles(
    speed: DoomsdayClockSpeed,
    land: i64,
    elapsed: i64,
    side_size: i64,
) -> i64 {
    let base = doomsday_clock_required_tiles(speed, land, elapsed);
    (base * side_size.max(1)).min(land)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoomsdayClockWaveState {
    pub current_percent: f64,
    pub target_percent: f64,
    pub growing: bool,
    pub seconds_to_next_growth: i64,
    pub wave_flash: bool,
    pub done: bool,
}

/// Display-only companion for the HUD: the live share, whether it is ramping
/// or holding, and the cue window.
pub fn doomsday_clock_wave_state(speed: DoomsdayClockSpeed, elapsed: i64) -> DoomsdayClockWaveState {
    let s = schedule(speed);
    let current_percent = required_basis_points(speed, elapsed) as f64 / 100.0;
    let cycle = s.ramp_seconds + s.pause_seconds;
    let n = LEVELS.len() as i64;
    let last = LEVELS[LEVELS.len() - 1] as f64 / 100.0;

    if elapsed <= s.grace_seconds {
        return DoomsdayClockWaveState {
            current_percent: 0.0,
            target_percent: LEVELS[0] as f64 / 100.0,
            growing: false,
            seconds_to_next_growth: s.grace_seconds - elapsed,
            wave_flash: s.grace_seconds - elapsed <= 5,
            done: false,
        };
    }

    let t = elapsed - s.grace_seconds;
    let i = t / cycle;
    if i >= n {
        return DoomsdayClockWaveState {
            current_percent,
            target_percent: last,
            growing: false,
            seconds_to_next_growth: 0,
            wave_flash: false,
            done: true,
        };
    }

    let idx = i as usize;
    let into = t - i * cycle;
    let growing = into < s.ramp_seconds;
    let is_last = i == n - 1;
    let next_ramp_start = s.grace_seconds + (i + 1) * cycle;
    DoomsdayClockWaveState {
        current_percent,
        target_percent: (if growing || is_last {
            LEVELS[idx]
        } else {
            LEVELS[idx + 1]
        }) as f64
            / 100.0,
        growing,
        seconds_to_next_growth: if growing || is_last {
            0
        } else {
            next_ramp_start - elapsed
        },
        wave_flash: into <= 5 || (!is_last && next_ramp_start - elapsed <= 5),
        done: is_last && !growing,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DoomsdayClockDrainConfig {
    pub drain_start_percent: i64,
    pub drain_max_percent: i64,
    pub drain_ramp_seconds: i64,
}

/// Troops (or warship HP) a skulled side loses this second: a linear ramp
/// from `drain_start_percent` up to `drain_max_percent` over
/// `drain_ramp_seconds`, as a percentage of `max_troops` (capacity, not
/// current). Always removes at least 1.
///
/// `max_troops` is `f64`, not `i64`: TS's caller passes `Config.maxTroops(m)`
/// straight through (a raw, un-`toInt`'d float - `2 * (tiles^0.6 * 1000 +
/// 50000) + ...`), and `doomsdayClockDrain` floors only the *final* product
/// (`Math.floor((maxTroops * pct) / 100)`). Flooring `max_troops` first (as an
/// `i64` param would force) computes a different, sometimes-smaller result:
/// e.g. `max_troops=1.5, pct=99` gives TS `floor(1.5*99/100)=1` but
/// `floor(1.5)*99/100=0` if truncated up front. Real `maxTroops` values are
/// always >= 100000 (the formula's floor), so this never bites in practice,
/// but matching TS's single-floor-at-the-end exactly costs nothing.
pub fn doomsday_clock_drain(
    max_troops: f64,
    seconds_past_warn: i64,
    cfg: &DoomsdayClockDrainConfig,
) -> i64 {
    let t = seconds_past_warn.max(0);
    let r = cfg.drain_ramp_seconds;
    let span = cfg.drain_max_percent - cfg.drain_start_percent;
    let pct = if r <= 0 || t >= r {
        cfg.drain_max_percent
    } else {
        cfg.drain_start_percent + span * t / r
    };
    ((max_troops * pct as f64) / 100.0).floor().max(1.0) as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use DoomsdayClockSpeed::{Fast, Normal, Slow, VeryFast};

    #[test]
    fn required_tiles_ramps_then_holds_during_pause() {
        let land = 10000;
        assert_eq!(doomsday_clock_required_tiles(Normal, land, 200), 0);
        assert_eq!(doomsday_clock_required_tiles(Normal, land, 330), 0);
        assert_eq!(doomsday_clock_required_tiles(Normal, land, 465), 150);
        assert_eq!(doomsday_clock_required_tiles(Normal, land, 600), 300);
        assert_eq!(doomsday_clock_required_tiles(Normal, land, 615), 300);
        assert_eq!(doomsday_clock_required_tiles(Normal, land, 630), 300);
        assert_eq!(doomsday_clock_required_tiles(Normal, land, 9999), 5500);
    }

    #[test]
    fn required_tiles_reaches_final_squeeze_per_preset() {
        let land = 10000;
        assert_eq!(doomsday_clock_required_tiles(Normal, land, 1800), 3000);
        assert_eq!(doomsday_clock_required_tiles(Normal, land, 2100), 5500);
        assert_eq!(doomsday_clock_required_tiles(Fast, land, 1440), 5500);
        assert_eq!(doomsday_clock_required_tiles(VeryFast, land, 1050), 5500);
        assert_eq!(doomsday_clock_required_tiles(Slow, land, 2520), 5500);
    }

    #[test]
    fn required_tiles_never_decreases_and_is_zero_for_no_land() {
        let land = 10000;
        let mut prev = 0;
        let mut t = 0;
        while t <= 2400 {
            let r = doomsday_clock_required_tiles(Normal, land, t);
            assert!(r >= prev);
            prev = r;
            t += 5;
        }
        assert_eq!(doomsday_clock_required_tiles(Normal, 0, 1800), 0);
    }

    #[test]
    fn side_required_tiles_scales_by_size_and_caps_at_map() {
        let land = 10000;
        assert_eq!(doomsday_clock_required_tiles(VeryFast, land, 900), 3000);
        assert_eq!(
            doomsday_clock_side_required_tiles(VeryFast, land, 900, 1),
            3000
        );
        assert_eq!(
            doomsday_clock_side_required_tiles(VeryFast, land, 900, 2),
            6000
        );
        assert_eq!(
            doomsday_clock_side_required_tiles(VeryFast, land, 900, 4),
            10000
        );
        assert_eq!(
            doomsday_clock_side_required_tiles(VeryFast, land, 900, 0),
            3000
        );
    }

    #[test]
    fn wave_state_reports_live_share_and_target_while_ramping() {
        let s = doomsday_clock_wave_state(Normal, 465);
        assert_eq!(s.current_percent, 1.5);
        assert_eq!(s.target_percent, 3.0);
        assert!(s.growing);
        assert_eq!(s.seconds_to_next_growth, 0);
        assert!(!s.done);
    }

    #[test]
    fn wave_state_counts_down_to_next_ramp_during_pause() {
        let s = doomsday_clock_wave_state(Normal, 615);
        assert!(!s.growing);
        assert_eq!(s.current_percent, 3.0);
        assert_eq!(s.target_percent, 5.0);
        assert_eq!(s.seconds_to_next_growth, 15);
    }

    #[test]
    fn wave_state_counts_down_through_grace() {
        let s = doomsday_clock_wave_state(Normal, 200);
        assert_eq!(s.current_percent, 0.0);
        assert_eq!(s.target_percent, 3.0);
        assert_eq!(s.seconds_to_next_growth, 130);
    }

    #[test]
    fn wave_state_flags_the_window_around_a_ramp_starting() {
        assert!(doomsday_clock_wave_state(VeryFast, 176).wave_flash);
        assert!(doomsday_clock_wave_state(VeryFast, 184).wave_flash);
        assert!(!doomsday_clock_wave_state(VeryFast, 250).wave_flash);
    }

    #[test]
    fn wave_state_marks_done_after_the_last_ramp() {
        let s = doomsday_clock_wave_state(VeryFast, 1100);
        assert!(s.done);
        assert_eq!(s.current_percent, 55.0);
        assert_eq!(s.seconds_to_next_growth, 0);
    }

    fn drain_cfg() -> DoomsdayClockDrainConfig {
        DoomsdayClockDrainConfig {
            drain_start_percent: 10,
            drain_max_percent: 80,
            drain_ramp_seconds: 3,
        }
    }

    #[test]
    fn drain_starts_gentle_and_grows_linearly_capping_at_max() {
        let cfg = drain_cfg();
        assert_eq!(doomsday_clock_drain(1000.0, 0, &cfg), 100);
        assert_eq!(doomsday_clock_drain(1000.0, 1, &cfg), 330);
        assert_eq!(doomsday_clock_drain(1000.0, 2, &cfg), 560);
        assert_eq!(doomsday_clock_drain(1000.0, 3, &cfg), 800);
        assert_eq!(doomsday_clock_drain(1000.0, 100, &cfg), 800);
        let d0 = doomsday_clock_drain(1000.0, 0, &cfg);
        let d1 = doomsday_clock_drain(1000.0, 1, &cfg);
        let d2 = doomsday_clock_drain(1000.0, 2, &cfg);
        assert_eq!(d1 - d0, d2 - d1);
    }

    #[test]
    fn drain_removes_at_least_one_troop_and_never_less() {
        assert_eq!(doomsday_clock_drain(1.0, 0, &drain_cfg()), 1);
    }

    #[test]
    fn drain_treats_time_before_warn_window_as_zero() {
        assert_eq!(doomsday_clock_drain(1000.0, -5, &drain_cfg()), 100);
    }

    // The rest of openfront/tests/DoomsdayClockExecution.test.ts (the "(logic)",
    // "(warship decay)", "(teams)", and "(integration)" describe blocks) is
    // ported in `crate::execution::doomsday_clock_execution`'s own test module,
    // alongside the stateful `Execution` those tests exercise.
}
