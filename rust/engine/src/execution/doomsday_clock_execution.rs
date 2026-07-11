//! Doomsday Clock (anti-stall) execution (TS `DoomsdayClockExecution.ts`).
//!
//! Once armed, every side must hold a rising share of the whole map: each
//! player in FFA, each whole team in team modes (so a team is judged on its
//! combined territory and every member shares the fate). The bar rises in
//! discrete waves (see `crate::core::doomsday_clock`), stepping up to each
//! wave's level and holding. A side below the bar is marked
//! (`in_doomsday_clock` -> blinking skull on the client) and, after the warn
//! window, every member bleeds an escalating percentage of their troops
//! until the side recovers or hits zero. The leading side (crown holder in
//! FFA, top team otherwise) is never doomed, so the game can never freeze
//! with every remaining side bled to zero.
//!
//! Runs once per second (every 10 ticks), like `WinCheckExecution`. Added
//! once per game (not per player) in `bootstrap.rs::game_from_record`,
//! gated on `doomsday_clock_config().enabled` - exactly like TS's
//! `GameRunner.init()` only calls `addExecution(new DoomsdayClockExecution())`
//! inside that same `if`.

use super::Execution;
use crate::core::schemas::unit_type::WARSHIP;
use crate::core::{doomsday_clock_drain, doomsday_clock_side_required_tiles, DoomsdayClockDrainConfig};
use crate::game::{Game, PlayerType};

/// TS `UnitInfo` for `Warship.maxHealth` (see `game.rs::build_unit` /
/// `execution/warship.rs`, which hardcode the same 1000 - native has no
/// veterancy system, so unlike TS's `UnitImpl.maxHealth()` there is no
/// veterancy adjustment to apply here).
const WARSHIP_MAX_HEALTH: i32 = 1000;

pub struct DoomsdayClockExecution {
    active: bool,
}

impl DoomsdayClockExecution {
    pub fn new() -> Self {
        Self { active: true }
    }

    /// TS `DoomsdayClockExecution.sides` - singletons in FFA, grouped by team
    /// otherwise. A `Vec<(String, Vec<u16>)>` (not a `HashMap`) preserves the
    /// first-seen team order, matching TS's `Map<Team, Player[]>` insertion
    /// order (`Array.from(byTeam.values())`); grouping order doesn't affect
    /// any single side's drain amount, but it does decide which side wins a
    /// tile-count tie for the crown (see `tick`'s `leader_idx` scan), so it
    /// must be deterministic and TS-matching.
    fn sides(game: &Game, contenders: &[u16], ffa: bool) -> Vec<Vec<u16>> {
        if ffa {
            return contenders.iter().map(|&id| vec![id]).collect();
        }
        let mut order: Vec<String> = Vec::new();
        let mut groups: Vec<Vec<u16>> = Vec::new();
        for &id in contenders {
            let Some(team) = game.player_by_small_id(id).and_then(|p| p.team.clone()) else {
                continue;
            };
            if let Some(idx) = order.iter().position(|t| *t == team) {
                groups[idx].push(id);
            } else {
                order.push(team);
                groups.push(vec![id]);
            }
        }
        groups
    }
}

impl Default for DoomsdayClockExecution {
    fn default() -> Self {
        Self::new()
    }
}

impl Execution for DoomsdayClockExecution {
    fn init(&mut self, _game: &mut Game, _tick: u32) {}

    fn tick(&mut self, game: &mut Game, tick: u32) {
        if tick % 10 != 0 {
            return; // once per second
        }
        let cfg = game.wire.doomsday_clock_config();
        if !cfg.enabled {
            return;
        }
        let troop_drain_cfg = DoomsdayClockDrainConfig {
            drain_start_percent: cfg.drain_start_percent as i64,
            drain_max_percent: cfg.drain_max_percent as i64,
            drain_ramp_seconds: cfg.drain_ramp_seconds as i64,
        };
        // Warships bleed on the same start + ramp as troops but toward a much
        // higher ceiling (warship_drain_max_percent), so a fleet at full
        // attrition sinks in ~2s. Only the max differs from the troop drain.
        let warship_drain_cfg = DoomsdayClockDrainConfig {
            drain_max_percent: cfg.warship_drain_max_percent as i64,
            ..troop_drain_cfg
        };

        let elapsed = game.elapsed_game_seconds();
        // Humans and Nations are subject to it; the small map bots are not
        // (the `!= Bot` idiom used across the codebase). `players_alive()`
        // already returns only alive players, in insertion order.
        let contenders: Vec<u16> = game
            .players_alive()
            .filter(|p| p.player_type != PlayerType::Bot)
            .map(|p| p.small_id)
            .collect();

        // The bar applies per side: each player in FFA, each whole team otherwise.
        let ffa = game.wire.game_config().game_mode == "Free For All";
        let sides = Self::sides(game, &contenders, ffa);

        // A winner is already inevitable (one side left): idle. Before the
        // first wave the bar is 0, so nobody is flagged anyway.
        if sides.len() < 2 {
            for &id in &contenders {
                game.clear_doomsday_clock(id);
            }
            return;
        }

        let land = (game.num_land_tiles() as i64 - game.num_tiles_with_fallout() as i64).max(0);

        // The leading side (the crown holder in FFA, the top team otherwise)
        // is never doomed. Doomsday Clock culls the challengers toward the
        // leader, so the leader always keeps its army: the game can never
        // freeze with every remaining side bled to zero, and the final wave
        // squeezes out everyone but the leader -> a single winner. First side
        // with the most tiles wins ties (deterministic: `sides` is built in a
        // fixed order).
        let side_tiles: Vec<i64> = sides
            .iter()
            .map(|members| {
                members
                    .iter()
                    .map(|&id| {
                        game.player_by_small_id(id)
                            .map(|p| p.tiles_owned as i64)
                            .unwrap_or(0)
                    })
                    .sum()
            })
            .collect();
        let mut leader_idx = 0usize;
        for i in 1..side_tiles.len() {
            if side_tiles[i] > side_tiles[leader_idx] {
                leader_idx = i;
            }
        }

        for (i, members) in sides.iter().enumerate() {
            // Threshold scales with the side's headcount: a team of N must
            // hold N x a solo player's share (FFA sides are size 1, unscaled).
            let required = doomsday_clock_side_required_tiles(
                cfg.speed,
                land,
                elapsed,
                members.len() as i64,
            );
            // A non-leading side below the bar skulls and drains every
            // member; the leader (and any side above the bar) clears them all.
            if i != leader_idx && side_tiles[i] < required {
                for &id in members {
                    game.enter_doomsday_clock(id);
                    let seconds_under = game.doomsday_clock_ticks(id) / 10;
                    if seconds_under >= cfg.warn_seconds as i64 {
                        let seconds_past_warn = seconds_under - cfg.warn_seconds as i64;
                        let max_troops = game.max_troops_for(id);
                        let chunk = doomsday_clock_drain(max_troops, seconds_past_warn, &troop_drain_cfg);
                        game.remove_troops(id, chunk as f64); // caps at current troops

                        // The navy bleeds on the same ramp but toward
                        // warship_drain_cfg's far higher ceiling (see above),
                        // so a doomed side's fleet is scuttled fast at full
                        // attrition. Destroyed with no attacker (matches TS's
                        // "never a credited kill" - moot natively either way,
                        // since this engine has no kill-credit/stats system
                        // at all yet for any damage source).
                        let warship_ids: Vec<i32> = game
                            .player_by_small_id(id)
                            .map(|p| {
                                p.units
                                    .iter()
                                    .filter(|u| u.unit_type == WARSHIP)
                                    .map(|u| u.id)
                                    .collect()
                            })
                            .unwrap_or_default();
                        for wid in warship_ids {
                            let dmg = doomsday_clock_drain(
                                WARSHIP_MAX_HEALTH as f64,
                                seconds_past_warn,
                                &warship_drain_cfg,
                            );
                            let destroyed = if let Some(u) = game.unit_mut(id, wid) {
                                u.health = (u.health - dmg as i32).max(0);
                                u.health == 0
                            } else {
                                false
                            };
                            if destroyed {
                                game.remove_unit(id, wid);
                            }
                        }
                    }
                }
            } else {
                for &id in members {
                    game.clear_doomsday_clock(id);
                }
            }
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::Config;
    use crate::execution::ExecEnum;
    use crate::game::{PlayerInfo, PlayerType};
    use serde_json::{json, Value};

    // Note on tuning: TS's ported test file (`DoomsdayClockExecution.test.ts`)
    // drives its "(logic)"/"(warship decay)"/"(teams)" describe blocks through
    // a hand-rolled `FakeGame`/`FakePlayer`, with a custom `sdConfig()`
    // (warnSeconds=1, drainStartPercent=10%, drainMaxPercent=80%, ...) - a
    // deliberate bypass of the real `Config` class, which (like native's
    // `Config::doomsday_clock_config`) hardcodes warn/drain to fixed defaults
    // and only exposes `enabled`/`speed` from the wire config; there is no
    // production seam to inject that tuning into the real engine on either
    // side. These tests instead exercise the *real* defaults (warnSeconds=10,
    // drainStart 2%, drainMax 6%, rampSeconds=50, warshipMax 50%) through the
    // real `Game`/`Config`/`Execution` path end-to-end, sizing troop counts to
    // the real `maxTroops` formula (dominated by a `+50000` floor term, so
    // always >= ~100k) so the multi-second warn/drain progression stays
    // observable - same behaviors under test, real production numbers.

    fn configure(game: &mut Game, mode: &str, speed: &str, extra: Value) {
        let mut value = json!({
            "gameMap": "Onion",
            "difficulty": "Medium",
            "donateGold": false,
            "donateTroops": false,
            "gameType": "Public",
            "gameMode": mode,
            "gameMapSize": "Normal",
            "nations": "disabled",
            "bots": 0,
            "infiniteGold": false,
            "infiniteTroops": false,
            "instantBuild": false,
            "randomSpawn": false,
            "doomsdayClock": { "enabled": true, "speed": speed },
        });
        for (key, item) in extra.as_object().unwrap() {
            value[key] = item.clone();
        }
        game.wire = Config::from_value(&value, true).unwrap();
    }

    fn add(game: &mut Game, id: &str, player_type: PlayerType, team: Option<&str>) -> u16 {
        game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type,
            client_id: Some(id.into()),
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: team.map(|t| t.to_string()),
        })
    }

    fn set(game: &mut Game, id: u16, tiles: i32, troops: i32) {
        if let Some(p) = game.player_by_small_id_mut(id) {
            p.tiles_owned = tiles;
            p.troops = troops;
        }
    }

    fn troops(game: &Game, id: u16) -> i32 {
        game.player_by_small_id(id).unwrap().troops
    }

    // `Game::default()`'s stub map is 1x1 (1 land tile); every test below
    // needs a real land count to exercise the wave math (`num_land_tiles()`),
    // but never touches actual tile geometry (tiles_owned is set directly),
    // so a real, fully-land `plains_game` of the right area stands in.
    fn game_with_land(land: i64) -> Game {
        crate::test_util::plains_game(land as u32, 1)
    }

    /// Registers the exec and drives real ticks up to (and including)
    /// `target_tick`, so `game.ticks()`/`elapsed_game_seconds()` progress
    /// exactly as they would in a real game (TS's `FakeGame.now`/`ticks()`
    /// stand-in, but through the real tick driver instead of a fake).
    fn advance_to(game: &mut Game, target_tick: u32) {
        while game.ticks() <= target_tick {
            game.execute_next_tick();
        }
    }

    fn setup(land: i64, mode: &str, speed: &str) -> Game {
        let mut game = game_with_land(land);
        configure(&mut game, mode, speed, json!({}));
        game.add_execution(ExecEnum::DoomsdayClock(DoomsdayClockExecution::new()));
        game
    }

    /// Advances real ticks until `id` first gets flagged, returning that tick.
    /// The bar rises continuously (unlike TS's `FakeGame`, which jumps
    /// straight to a hand-picked tick and never processes anything before
    /// it), so under real per-tick simulation a side crosses below the bar
    /// well before any fixed "wave" tick chosen for its *held* value - tests
    /// that need a controlled "0 seconds under" starting point advance to
    /// this tick, then measure forward from it, instead of assuming a fixed
    /// tick doubles as the catch moment.
    fn advance_until_flagged(game: &mut Game, id: u16) -> u32 {
        loop {
            game.execute_next_tick();
            if game.in_doomsday_clock(id) {
                return game.ticks();
            }
        }
    }

    // land 1000, veryfast 20% wave -> bar = 200 at WAVE_TICK (elapsed 760s).
    const WAVE_TICK: u32 = 7600;
    // Comparable to the real `maxTroops` formula's floor (~100k for any tile
    // count), so a 2%-6% drain removes a meaningful, gradually-growing slice
    // instead of instantly zeroing a TS-style "troops=1000" stand-in.
    const BIG_TROOPS: i32 = 100_000;

    fn two_player_game(a_tiles: i32, b_tiles: i32) -> (Game, u16, u16) {
        let mut game = setup(1000, "Free For All", "veryfast");
        let a = add(&mut game, "a", PlayerType::Human, None);
        let b = add(&mut game, "b", PlayerType::Human, None);
        set(&mut game, a, a_tiles, BIG_TROOPS);
        set(&mut game, b, b_tiles, BIG_TROOPS);
        (game, a, b)
    }

    // ---------------------------------------------------------------------
    // "(logic)"
    // ---------------------------------------------------------------------

    #[test]
    fn does_nothing_when_disabled() {
        let (mut game, _a, b) = two_player_game(400, 100);
        configure(
            &mut game,
            "Free For All",
            "veryfast",
            json!({ "doomsdayClock": { "enabled": false } }),
        );
        advance_to(&mut game, WAVE_TICK);
        assert!(!game.in_doomsday_clock(b));
        assert_eq!(troops(&game, b), BIG_TROOPS);
    }

    #[test]
    fn does_nothing_before_the_first_wave() {
        // veryfast grace runs to 180s; before it the bar is 0, nobody below it.
        let (mut game, _a, b) = two_player_game(400, 100);
        advance_to(&mut game, 500); // elapsed 50s < 180s (grace)
        assert!(!game.in_doomsday_clock(b));
        assert_eq!(troops(&game, b), BIG_TROOPS);
    }

    #[test]
    fn flags_a_player_below_the_bar_and_spares_one_above_it() {
        let (mut game, a, b) = two_player_game(400, 100);
        advance_to(&mut game, WAVE_TICK); // bar = 200
        assert!(!game.in_doomsday_clock(a));
        assert!(game.in_doomsday_clock(b));
    }

    #[test]
    fn warns_before_draining_then_drains_harder_over_time() {
        // Rather than a fixed tick, catch the exact moment `b` first drops
        // below the (continuously rising) bar - unlike TS's `FakeGame`, which
        // jumps straight to a hand-picked tick, real per-tick simulation
        // means `b` is caught well before any tick chosen to match its
        // *held* wave value (see `advance_until_flagged`'s doc comment).
        let (mut game, _a, b) = two_player_game(400, 100);
        let caught_tick = advance_until_flagged(&mut game, b);
        assert_eq!(troops(&game, b), BIG_TROOPS); // 0s under -> within the (10s) warn, no drain yet

        // Land exactly on the tick before the first past-warn drain event
        // (seconds_past_warn = 0, drainStart 2%), so this single event's
        // drain can be measured in isolation.
        advance_to(&mut game, caught_tick + 98);
        let before_first = troops(&game, b);
        game.execute_next_tick();
        let first_drain = before_first - troops(&game, b);
        assert!(first_drain > 0);

        // With drain_ramp_seconds=50 and a 2%->6% span, the integer-percent
        // ramp (`drain_start_percent + span * t / r`) only ticks up every
        // ~12.5s (t=13 is the first second where it reaches 3%), so comparing
        // t=0 to t=1 wouldn't show a difference - jump to that next integer
        // step instead. Troops are reset to a full stack right before (still
        // below the bar, still flagged - resetting troops doesn't touch
        // `marked_doomsday_clock_tick`) so the intervening events (t=1..12,
        // still at 2%) can't zero the stack before this isolated event fires.
        advance_to(&mut game, caught_tick + 228);
        set(&mut game, b, 100, BIG_TROOPS);
        game.execute_next_tick();
        let second_drain = BIG_TROOPS - troops(&game, b);
        assert!(second_drain > first_drain); // drains harder as the ramp advances
    }

    #[test]
    fn drains_an_unrecovered_player_all_the_way_to_zero() {
        let (mut game, _a, b) = two_player_game(400, 50);
        advance_to(&mut game, WAVE_TICK + 2000); // 200s of continuous drain
        assert_eq!(troops(&game, b), 0);
        assert!(game.in_doomsday_clock(b));
    }

    #[test]
    fn clears_the_mark_and_stops_draining_when_a_player_climbs_back_above_the_bar() {
        let (mut game, _a, b) = two_player_game(400, 100);
        advance_to(&mut game, WAVE_TICK);
        advance_to(&mut game, WAVE_TICK + 100); // drained once (past the 10s warn)
        let after_drain = troops(&game, b);
        assert!(game.in_doomsday_clock(b));

        set(&mut game, b, 400, after_drain); // recovered above the bar
        advance_to(&mut game, WAVE_TICK + 110);
        assert!(!game.in_doomsday_clock(b));
        assert_eq!(troops(&game, b), after_drain); // drain stopped
    }

    #[test]
    fn drops_the_mark_once_a_flagged_player_dies() {
        // Nothing clears the mark on death, so in_doomsday_clock/doomsday_clock_ticks
        // must gate on `alive` to avoid a permanently "Draining" panel and a
        // per-tick update delta for an eliminated player.
        let (mut game, _a, b) = two_player_game(400, 100);
        advance_to(&mut game, WAVE_TICK);
        assert!(game.in_doomsday_clock(b));

        game.player_by_small_id_mut(b).unwrap().alive = false;
        assert!(!game.in_doomsday_clock(b));
        assert_eq!(game.doomsday_clock_ticks(b), 0);
    }

    #[test]
    fn never_dooms_the_leading_side_even_below_the_bar() {
        // Both sides below the 200 bar; the larger (a) is the crown, so it is
        // spared and keeps its army instead of everyone bleeding to zero.
        let (mut game, a, b) = two_player_game(150, 100);
        advance_to(&mut game, WAVE_TICK);
        assert!(!game.in_doomsday_clock(a)); // leader, spared
        assert!(game.in_doomsday_clock(b)); // challenger, doomed
        advance_to(&mut game, WAVE_TICK + 300);
        assert_eq!(troops(&game, a), BIG_TROOPS); // never drained
        assert!(troops(&game, b) < BIG_TROOPS); // bled
    }

    #[test]
    fn applies_to_nations_like_players_and_excludes_map_bots() {
        let mut game = setup(1000, "Free For All", "veryfast");
        let leader = add(&mut game, "leader", PlayerType::Human, None);
        let human = add(&mut game, "human", PlayerType::Human, None);
        let nation = add(&mut game, "nation", PlayerType::Nation, None);
        let bot = add(&mut game, "bot", PlayerType::Bot, None);
        set(&mut game, leader, 400, BIG_TROOPS);
        set(&mut game, human, 100, BIG_TROOPS);
        set(&mut game, nation, 50, BIG_TROOPS);
        set(&mut game, bot, 5, BIG_TROOPS);
        // Bar 200; leader (400) is crown-exempt; human (100) and nation (50)
        // are below it; the bot is exempt by type.
        advance_to(&mut game, WAVE_TICK);
        assert!(game.in_doomsday_clock(human));
        assert!(game.in_doomsday_clock(nation)); // a nation is treated like a player
        assert!(!game.in_doomsday_clock(bot)); // map bots are never subject to it
        assert!(!game.in_doomsday_clock(leader)); // the crown is never doomed
        advance_to(&mut game, WAVE_TICK + 100);
        assert!(troops(&game, nation) < BIG_TROOPS); // drained like a player
        assert_eq!(troops(&game, bot), BIG_TROOPS); // untouched
    }

    #[test]
    fn is_deterministic_identical_scenarios_give_identical_drains() {
        let run = || {
            let (mut game, _a, b) = two_player_game(400, 100);
            advance_to(&mut game, WAVE_TICK + 200);
            troops(&game, b)
        };
        assert_eq!(run(), run());
    }

    // ---------------------------------------------------------------------
    // "(warship decay)": a flagged (sub-threshold, non-leader) side's
    // warships bleed HP on the troop start + ramp but toward a higher
    // ceiling (50% vs troops' 6%), so a doomed fleet decays faster than the
    // side's troops. Destroyed with no attacker: native has no kill-credit
    // system for any damage source at all, so TS's "never scores a kill"
    // assertion has no native counterpart to check - only the
    // destruction+removal side is portable (see `WARSHIP_MAX_HEALTH`'s doc
    // comment above).
    // ---------------------------------------------------------------------

    fn add_warship(game: &mut Game, small_id: u16) -> i32 {
        game.build_unit(small_id, WARSHIP, 0) // spawns at full WARSHIP_MAX_HEALTH
    }

    fn warship_game() -> (Game, u16, u16) {
        // a (400) is the leader, above the bar; b (100) is below it.
        two_player_game(400, 100)
    }

    #[test]
    fn warship_matches_the_troop_drain_at_the_start_of_the_ramp() {
        let (mut game, _a, b) = warship_game();
        let ship = add_warship(&mut game, b);
        let caught_tick = advance_until_flagged(&mut game, b); // flags b (within the warn window)
        let max_troops = game.max_troops_for(b);
        advance_to(&mut game, caught_tick + 99); // 10s under -> 0s past warn -> both at drainStart%
        let troop_loss = BIG_TROOPS - troops(&game, b);
        let hp_loss = WARSHIP_MAX_HEALTH - game.unit(b, ship).unwrap().health;
        // Same *pct* at t=0 past warn (both configs share drain_start_percent),
        // but max_troops (~130k here) != WARSHIP_MAX_HEALTH (1000), and each
        // is independently floored (`doomsday_clock_drain`'s single
        // floor-at-the-end - see its doc comment), so the raw loss ratios
        // aren't exactly equal (floor(130698*0.02)/130698 != exactly 0.02).
        // Assert each loss against the pure formula directly instead.
        let cfg = game.wire.doomsday_clock_config();
        let troop_cfg = DoomsdayClockDrainConfig {
            drain_start_percent: cfg.drain_start_percent as i64,
            drain_max_percent: cfg.drain_max_percent as i64,
            drain_ramp_seconds: cfg.drain_ramp_seconds as i64,
        };
        let warship_cfg = DoomsdayClockDrainConfig {
            drain_max_percent: cfg.warship_drain_max_percent as i64,
            ..troop_cfg
        };
        assert_eq!(troop_loss as i64, doomsday_clock_drain(max_troops, 0, &troop_cfg));
        assert_eq!(
            hp_loss as i64,
            doomsday_clock_drain(WARSHIP_MAX_HEALTH as f64, 0, &warship_cfg)
        );
    }

    #[test]
    fn warship_scuttles_within_two_events_at_full_attrition() {
        // Once fully ramped, a fresh full-HP (1000) warship takes
        // warship_drain_max_percent (50%, vs troops' 6% max) per per-second
        // event, so it's destroyed after its second hit, not its first -
        // still far faster than troops (which need many 6% hits), proving
        // the ship uses its own higher ceiling.
        let (mut game, _a, b) = warship_game();
        let caught_tick = advance_until_flagged(&mut game, b); // starts the side's attrition clock
        advance_to(&mut game, caught_tick + 599); // past warn(10s)+ramp(50s) -> next events at max
        let ship = add_warship(&mut game, b); // fresh ship, appears once fully ramped
        advance_to(&mut game, caught_tick + 609); // one max-rate event: 1000 -> 500
        assert_eq!(game.unit(b, ship).unwrap().health, 500);
        advance_to(&mut game, caught_tick + 619); // a second max-rate event: 500 -> 0, removed
        assert!(game.unit(b, ship).is_none());
    }

    #[test]
    fn warship_destroyed_when_drain_exceeds_remaining_hp() {
        let (mut game, _a, b) = warship_game();
        let ship = add_warship(&mut game, b);
        game.unit_mut(b, ship).unwrap().health = 1; // less than any one tick's drain
        advance_to(&mut game, WAVE_TICK);
        advance_to(&mut game, WAVE_TICK + 100);
        assert!(game.unit(b, ship).is_none());
    }

    #[test]
    fn warship_spares_the_leaders_fleet() {
        let (mut game, a, b) = warship_game();
        let leader_ship = add_warship(&mut game, a);
        add_warship(&mut game, b);
        advance_to(&mut game, WAVE_TICK);
        advance_to(&mut game, WAVE_TICK + 300); // well past the warn window
        assert_eq!(
            game.unit(a, leader_ship).unwrap().health,
            WARSHIP_MAX_HEALTH
        );
    }

    #[test]
    fn warship_not_damaged_during_the_warn_window() {
        let (mut game, _a, b) = warship_game();
        let ship = add_warship(&mut game, b);
        advance_until_flagged(&mut game, b); // flagged, 0s under -> within the 10s warn
        assert!(game.in_doomsday_clock(b));
        assert_eq!(game.unit(b, ship).unwrap().health, WARSHIP_MAX_HEALTH);
    }

    // ---------------------------------------------------------------------
    // "(teams)": the bar applies to a whole team's combined territory, and
    // every member shares the fate (skull + drain together).
    // ---------------------------------------------------------------------

    fn team_game(land: i64, teams: &[(&str, &[i32])]) -> (Game, Vec<u16>) {
        let mut game = setup(land, "Team", "veryfast");
        let mut ids = Vec::new();
        for (team, tiles) in teams {
            for (n, &t) in tiles.iter().enumerate() {
                let id = add(
                    &mut game,
                    &format!("{team}-{n}"),
                    PlayerType::Human,
                    Some(team),
                );
                set(&mut game, id, t, BIG_TROOPS);
                ids.push(id);
            }
        }
        (game, ids)
    }

    #[test]
    fn team_judges_on_combined_territory_and_skulls_every_member() {
        // Both teams size 2 -> threshold 200x2=400. Red 250+250=500 safe;
        // Blue 50+50=100 below -> both Blue skulled.
        let (mut game, ids) = team_game(1000, &[("Red", &[250, 250]), ("Blue", &[50, 50])]);
        let (red1, red2, blue1, blue2) = (ids[0], ids[1], ids[2], ids[3]);
        advance_to(&mut game, WAVE_TICK);
        assert!(!game.in_doomsday_clock(red1));
        assert!(!game.in_doomsday_clock(red2));
        assert!(game.in_doomsday_clock(blue1));
        assert!(game.in_doomsday_clock(blue2));
        advance_to(&mut game, WAVE_TICK + 100); // past the warn -> both Blue members drain
        assert!(troops(&game, blue1) < BIG_TROOPS);
        assert!(troops(&game, blue2) < BIG_TROOPS);
        assert_eq!(troops(&game, red1), BIG_TROOPS); // safe team untouched
    }

    #[test]
    fn team_spares_a_tiny_member_whose_team_is_collectively_above_the_bar() {
        // Size 2 -> threshold 400. Red 400+40=440 -> safe, so the 40-tile
        // member is NOT skulled.
        let (mut game, ids) = team_game(1000, &[("Red", &[400, 40]), ("Blue", &[50, 50])]);
        let (red_tiny, blue1) = (ids[1], ids[2]);
        advance_to(&mut game, WAVE_TICK);
        assert!(!game.in_doomsday_clock(red_tiny)); // team is collectively safe
        assert!(game.in_doomsday_clock(blue1));
    }

    #[test]
    fn team_scales_the_threshold_by_team_size() {
        // base bar 200. Red is 3 members -> threshold 600; Blue is 1 -> 200.
        // Blue leads on tiles (crown-exempt), so Red is squeezed purely by size.
        let (mut game, ids) = team_game(1000, &[("Red", &[200, 200, 100]), ("Blue", &[700])]);
        let (red1, red2, red3, blue1) = (ids[0], ids[1], ids[2], ids[3]);
        advance_to(&mut game, WAVE_TICK);
        assert!(game.in_doomsday_clock(red1)); // 500 < 200x3
        assert!(game.in_doomsday_clock(red2));
        assert!(game.in_doomsday_clock(red3));
        assert!(!game.in_doomsday_clock(blue1)); // leader
    }

    #[test]
    fn team_idles_when_only_one_team_remains() {
        let (mut game, ids) = team_game(1000, &[("Red", &[50, 50])]);
        advance_to(&mut game, WAVE_TICK);
        for id in ids {
            assert!(!game.in_doomsday_clock(id));
        }
    }

    // ---------------------------------------------------------------------
    // Integration: a real multi-tile map, real ticks. The drain is isolated
    // from normal troop dynamics by comparing an enabled run vs. a disabled
    // run that otherwise shares identical setup.
    // ---------------------------------------------------------------------

    fn give_land_tiles(game: &mut Game, small_id: u16, n: u32) -> u32 {
        let mut count = 0;
        let (w, h) = (game.width(), game.height());
        for y in 0..h {
            for x in 0..w {
                if count >= n {
                    return count;
                }
                let t = game.ref_xy(x, y);
                if game.is_land(t) && !game.has_owner(t) {
                    game.conquer(small_id, t);
                    count += 1;
                }
            }
        }
        count
    }

    #[test]
    fn integration_skulls_below_the_bar_spares_above_it_and_drains() {
        let run = |enabled: bool| {
            let mut game = crate::test_util::plains_game(60, 60);
            configure(
                &mut game,
                "Free For All",
                "veryfast",
                json!({ "doomsdayClock": { "enabled": enabled, "speed": "veryfast" } }),
            );
            let big = add(&mut game, "big", PlayerType::Human, None);
            let small = add(&mut game, "small", PlayerType::Human, None);
            let target_ticks = 3000u32; // 300s -> veryfast holding its 3% wave
            let bar = doomsday_clock_side_required_tiles(
                crate::core::DoomsdayClockSpeed::VeryFast,
                game.num_land_tiles() as i64,
                (target_ticks / 10) as i64,
                1,
            );
            give_land_tiles(&mut game, big, bar as u32 + 50); // above the bar
            give_land_tiles(&mut game, small, 3); // a sliver, below the bar
            game.player_by_small_id_mut(big).unwrap().troops = 50_000;
            game.player_by_small_id_mut(small).unwrap().troops = 50_000;
            game.add_execution(ExecEnum::DoomsdayClock(DoomsdayClockExecution::new()));
            advance_to(&mut game, target_ticks);
            (
                game.in_doomsday_clock(small),
                game.in_doomsday_clock(big),
                troops(&game, small),
            )
        };

        let (small_on, big_on, small_troops_on) = run(true);
        let (small_off, _big_off, small_troops_off) = run(false);

        assert!(small_on);
        assert!(!big_on);
        assert!(!small_off);
        assert!(small_troops_on < small_troops_off);
    }

    // ---------------------------------------------------------------------
    // Default-config wipe time, with real troop income (`PlayerExecution`)
    // flowing every tick: pins the advertised "~1 minute from caught to
    // wiped" against the *real*, non-test-tuned defaults end-to-end (both TS
    // and native hardcode the same warn(10s)/drain(2%->6% over 50s) values,
    // so this window carries over unchanged from TS's own ported assertion).
    // ---------------------------------------------------------------------

    #[test]
    fn default_drain_wipes_a_full_troop_side_in_about_a_minute() {
        use crate::execution::player::PlayerExecution;

        // 100x100 (10000 land tiles), not 60x60 (3600) - `big` alone claims
        // 4000 tiles below to stay safely above the bar even at the last wave.
        let mut game = crate::test_util::plains_game(100, 100);
        configure(
            &mut game,
            "Free For All",
            "veryfast",
            json!({ "doomsdayClock": { "enabled": true, "speed": "veryfast" } }),
        );
        let big = add(&mut game, "big", PlayerType::Human, None);
        let small = add(&mut game, "small", PlayerType::Human, None);
        give_land_tiles(&mut game, big, 4000); // safely above the bar
        give_land_tiles(&mut game, small, 3); // a sliver, caught once the bar rises
        for &id in &[big, small] {
            let spawn = game.player_by_small_id(id).unwrap().owned_tiles[0];
            let p = game.player_by_small_id_mut(id).unwrap();
            p.spawn_tile = Some(spawn);
        }
        game.add_execution(ExecEnum::Player(PlayerExecution::new(big)));
        game.add_execution(ExecEnum::Player(PlayerExecution::new(small)));
        game.add_execution(ExecEnum::DoomsdayClock(DoomsdayClockExecution::new()));

        // Run until the rising bar catches the sliver, then fill it to a full
        // stack so we measure the worst-case (longest) wipe from that moment.
        let mut caught_tick = None;
        for _ in 0..3000 {
            game.execute_next_tick();
            if game.in_doomsday_clock(small) {
                caught_tick = Some(game.ticks());
                break;
            }
        }
        let caught_tick = caught_tick.expect("bar should have caught the sliver");
        let max = game.max_troops_for(small) as i32;
        game.player_by_small_id_mut(small).unwrap().troops = max;

        let mut zero_tick = None;
        for _ in 0..1500 {
            game.execute_next_tick();
            if troops(&game, small) <= 0 {
                zero_tick = Some(game.ticks());
                break;
            }
        }
        let zero_tick = zero_tick.expect("should eventually drain to zero");
        let seconds = (zero_tick - caught_tick) as f64 / 10.0;
        // ~10s warn + ~50s drain, income included: about a minute (not ~45s).
        assert!(seconds > 50.0, "seconds={seconds}");
        assert!(seconds < 85.0, "seconds={seconds}");
    }
}
