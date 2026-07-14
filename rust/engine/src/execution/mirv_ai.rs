//! Nation MIRV coordination AI (TS `NationMIRVBehavior.ts`).
//!
//! Ported as free functions over `Game`/`PseudoRandom`, following this
//! codebase's established pattern for TS AI-behavior classes (see
//! `nuke_ai.rs`, `warship_ai.rs`). The TS class keeps `recentMirvTargets` as
//! a `static` field shared across *every* `NationMIRVBehavior` instance (all
//! nations in the match, so nobody piles on the same freshly-hit target) -
//! the native equivalent lives on `Game` itself (`recent_mirv_targets`) for
//! the same reason.
//!
//! Before this port, `nation_tick.rs`'s `consider_mirv` only replicated the
//! RNG draws (silo/gold/hesitation checks) and always discarded the result
//! (`let _ = consider_mirv(...)`) - none of `NationMIRVBehavior`'s actual
//! targeting (counter-MIRV retaliation, victory denial, steamroll-stop) or
//! launch logic existed natively. This module fills that in for real and
//! `nation_tick.rs` now calls it.

use super::mirv_execution::MirvExecution;
use super::nation_emoji::{self, NationEmojiState};
use super::nuke_execution::can_build_nuke;
use super::ExecEnum;
use crate::core::schemas::unit_type;
use crate::game::{Game, Player, PlayerType};
use crate::map::TileRef;
use crate::prng::PseudoRandom;

/// TS `MIRV_COOLDOWN_TICKS` (30s at 10 ticks/s).
const MIRV_COOLDOWN_TICKS: i64 = 300;
/// TS `EMOJI_NUKE.length` - matches `nuke_ai.rs`'s own copy (RNG-draw-count parity only).
const EMOJI_NUKE_LEN: i32 = 2;

fn hesitation_odds(difficulty: &str) -> i32 {
    match difficulty {
        "Easy" => 2,
        "Medium" => 4,
        "Hard" => 8,
        "Impossible" => 16,
        _ => 4,
    }
}

fn victory_denial_team_threshold(difficulty: &str) -> f64 {
    match difficulty {
        "Easy" => 0.9,
        "Medium" => 0.8,
        "Hard" => 0.7,
        "Impossible" => 0.6,
        _ => 0.8,
    }
}

fn victory_denial_individual_threshold(difficulty: &str) -> f64 {
    match difficulty {
        "Easy" => 0.75,
        "Medium" => 0.65,
        "Hard" => 0.55,
        "Impossible" => 0.4,
        _ => 0.65,
    }
}

fn steamroll_city_gap_multiplier(difficulty: &str) -> f64 {
    match difficulty {
        "Easy" => 2.0,
        "Medium" => 1.5,
        "Hard" => 1.25,
        "Impossible" => 1.15,
        _ => 1.5,
    }
}

fn steamroll_min_leader_cities(difficulty: &str) -> i64 {
    match difficulty {
        "Easy" => 20,
        "Medium" | "Hard" => 10,
        "Impossible" => 8,
        _ => 10,
    }
}

/// TS `NationMIRVBehavior.considerMIRV()`.
pub fn consider_mirv(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    emoji_state: &mut NationEmojiState,
) -> bool {
    if game.wire.is_unit_disabled(unit_type::MIRV) {
        return false;
    }
    if game.unit_count(small_id, unit_type::MISSILE_SILO) == 0 {
        return false;
    }
    let mirv_cost = game.structure_cost(small_id, unit_type::MIRV);
    let gold = game.player_by_small_id(small_id).map(|p| p.gold).unwrap_or(0);
    if gold < mirv_cost {
        return false;
    }

    let difficulty = game.wire.game_config().difficulty.clone();
    if random.chance(hesitation_odds(&difficulty)) {
        return false;
    }

    if let Some(target) = select_counter_mirv_target(game, small_id) {
        if !was_recently_mirved(game, target) {
            maybe_send_mirv(game, random, small_id, target, emoji_state);
            return true;
        }
    }

    if let Some(target) = select_victory_denial_target(game, small_id, &difficulty) {
        if !was_recently_mirved(game, target) {
            maybe_send_mirv(game, random, small_id, target, emoji_state);
            return true;
        }
    }

    if let Some(target) = select_steamroll_stop_target(game, small_id, &difficulty) {
        if !was_recently_mirved(game, target) {
            maybe_send_mirv(game, random, small_id, target, emoji_state);
            return true;
        }
    }

    false
}

/// TS `NationMIRVBehavior.getValidMirvTargetPlayers()`.
///
/// Mirrors `game.players()` (alive only), not `allPlayers()`.
fn valid_mirv_target_players(game: &Game, small_id: u16) -> Vec<&Player> {
    game.all_players()
        .iter()
        .filter(|p| {
            p.alive
                && p.small_id != small_id
                && p.player_type != PlayerType::Bot
                && !game.players_on_same_team(small_id, p.small_id)
        })
        .collect()
}

/// TS `NationMIRVBehavior.isInboundMIRVFrom()`.
fn is_inbound_mirv_from(game: &Game, attacker: &Player, small_id: u16) -> bool {
    attacker.units.iter().any(|u| {
        u.unit_type == unit_type::MIRV
            && u.target_tile
                .is_some_and(|t| game.has_owner(t) && game.map.owner_id(t) == small_id)
    })
}

/// TS `NationMIRVBehavior.selectCounterMirvTarget()`.
fn select_counter_mirv_target(game: &Game, small_id: u16) -> Option<u16> {
    let mut attackers: Vec<&Player> = valid_mirv_target_players(game, small_id)
        .into_iter()
        .filter(|p| is_inbound_mirv_from(game, p, small_id))
        .collect();
    if attackers.is_empty() {
        return None;
    }
    attackers.sort_by(|a, b| b.tiles_owned.cmp(&a.tiles_owned));
    Some(attackers[0].small_id)
}

/// TS `NationMIRVBehavior.selectVictoryDenialTarget()`.
fn select_victory_denial_target(game: &Game, small_id: u16, difficulty: &str) -> Option<u16> {
    let total_land = game.num_land_tiles();
    if total_land == 0 {
        return None;
    }
    let team_threshold = victory_denial_team_threshold(difficulty);
    let individual_threshold = victory_denial_individual_threshold(difficulty);

    let mut best: Option<(u16, f64)> = None;
    for p in valid_mirv_target_players(game, small_id) {
        let mut severity = 0.0;
        if let Some(team) = p.team.as_deref() {
            // TS `game.players().filter(x => x.team() === team && x.isPlayer())`.
            let team_members: Vec<&Player> = game
                .all_players()
                .iter()
                .filter(|x| x.alive && x.team.as_deref() == Some(team))
                .collect();
            let team_territory: i64 = team_members.iter().map(|x| x.tiles_owned as i64).sum();
            let team_share = team_territory as f64 / total_land as f64;
            if team_share >= team_threshold {
                // Only consider the largest team member as the target when team exceeds threshold.
                let mut largest_member: Option<u16> = None;
                let mut largest_tiles: i64 = -1;
                for m in &team_members {
                    let tiles = m.tiles_owned as i64;
                    if tiles > largest_tiles {
                        largest_tiles = tiles;
                        largest_member = Some(m.small_id);
                    }
                }
                if largest_member == Some(p.small_id) {
                    severity = team_share;
                }
            }
        } else {
            let share = p.tiles_owned as f64 / total_land as f64;
            if share >= individual_threshold {
                severity = share;
            }
        }
        if severity > 0.0 && best.is_none_or(|(_, s)| severity > s) {
            best = Some((p.small_id, severity));
        }
    }
    best.map(|(sid, _)| sid)
}

/// TS `NationMIRVBehavior.selectSteamrollStopTarget()`.
fn select_steamroll_stop_target(game: &Game, small_id: u16, difficulty: &str) -> Option<u16> {
    let valid_targets = valid_mirv_target_players(game, small_id);
    if valid_targets.is_empty() {
        return None;
    }

    // TS ranks with `p.unitCount(UnitType.City)` which sums city *levels*, not
    // unit cardinality. Using `unit_count` here made native over-trigger steamroll
    // MIRVs whenever the leader had more city buildings but a smaller level gap
    // (e.g. many L1 cities vs fewer upgraded ones) — seen as Egypt MIRV @ 15888
    // on curr-b030-s0-pangaea while TS correctly stayed put.
    let mut all_players: Vec<(u16, i64)> = game
        .all_players()
        .iter()
        .filter(|p| p.alive)
        .map(|p| (p.small_id, game.unit_level_sum(p.small_id, unit_type::CITY) as i64))
        .collect();
    if all_players.len() < 2 {
        return None;
    }
    all_players.sort_by(|a, b| b.1.cmp(&a.1));

    let (top_sid, top_cities) = all_players[0];
    if top_cities <= steamroll_min_leader_cities(difficulty) {
        return None;
    }

    let second_highest = all_players[1].1;
    let threshold = second_highest as f64 * steamroll_city_gap_multiplier(difficulty);

    if top_cities as f64 >= threshold {
        if valid_targets.iter().any(|p| p.small_id == top_sid) {
            Some(top_sid)
        } else {
            None
        }
    } else {
        None
    }
}

/// TS `NationMIRVBehavior.wasRecentlyMirved()`.
fn was_recently_mirved(game: &Game, target_small_id: u16) -> bool {
    let Some(&last_tick) = game.recent_mirv_targets.get(&target_small_id) else {
        return false;
    };
    (game.ticks() as i64 - last_tick as i64) < MIRV_COOLDOWN_TICKS
}

/// TS `NationMIRVBehavior.recordMirvHit()`.
fn record_mirv_hit(game: &mut Game, target_small_id: u16) {
    let tick = game.ticks();
    game.recent_mirv_targets.insert(target_small_id, tick);
}

/// TS `NationMIRVBehavior.calculateTerritoryCenter()` (`Util.calculateTerritoryCenter`).
fn calculate_territory_center(game: &Game, target_small_id: u16) -> Option<TileRef> {
    let border_tiles = game.border_tiles_for(target_small_id);
    if border_tiles.is_empty() {
        return None;
    }

    let mut min_x = i64::MAX;
    let mut max_x = i64::MIN;
    let mut min_y = i64::MAX;
    let mut max_y = i64::MIN;
    for &t in &border_tiles {
        let x = game.x(t) as i64;
        let y = game.y(t) as i64;
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    // Border tile coordinates are always non-negative map coordinates, so
    // integer division here matches JS `Math.floor((minX + maxX) / 2)`.
    let center_x = (min_x + max_x) / 2;
    let center_y = (min_y + max_y) / 2;
    let center_tile = game.ref_xy(center_x as u32, center_y as u32);

    if game.map.owner_id(center_tile) == target_small_id {
        return Some(center_tile);
    }

    let mut closest: Option<TileRef> = None;
    let mut closest_dist_sq = i64::MAX;
    for &t in &border_tiles {
        let dx = game.x(t) as i64 - center_x;
        let dy = game.y(t) as i64 - center_y;
        let dist_sq = dx * dx + dy * dy;
        if dist_sq < closest_dist_sq {
            closest_dist_sq = dist_sq;
            closest = Some(t);
        }
    }
    closest
}

/// TS `NationMIRVBehavior.maybeSendMIRV()`.
fn maybe_send_mirv(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    target_small_id: u16,
    emoji_state: &mut NationEmojiState,
) {
    nation_emoji::maybe_send_attack_emoji(game, random, small_id, emoji_state, target_small_id);

    let Some(center_tile) = calculate_territory_center(game, target_small_id) else {
        return;
    };
    if can_build_nuke(game, small_id, unit_type::MIRV, center_tile).is_none() {
        return;
    }

    game.add_execution(ExecEnum::Mirv(MirvExecution::new(small_id, center_tile)));
    record_mirv_hit(game, target_small_id);
    nation_emoji::send_emoji(random, game, small_id, emoji_state, None, EMOJI_NUKE_LEN);
    nation_emoji::respond_to_mirv(game, random, target_small_id);
}

// TS `NationMIRV.test.ts` - ported at the AI-decision level (the underlying
// target-selection helpers, directly and deterministically) rather than by
// replaying the TS suite's own "drive `NationExecution.tick` for 200 ticks,
// retry across 20 game ids to dodge hesitation odds" harness, since that
// harness exists only to work around `considerMIRV`'s RNG hesitation gate -
// something exercised directly and far more cheaply here by feeding
// `consider_mirv` a handful of deterministic seeds. One true end-to-end
// smoke test (`consider_mirv_launches_a_real_mirv_in_counter_retaliation_end_to_end`)
// still drives the whole `consider_mirv` -> `MirvExecution` -> spawned unit
// pipeline for real, matching `nuke_ai.rs`'s own end-to-end nuke test.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::PlayerInfo;

    fn add_player(game: &mut Game, id: &str, player_type: PlayerType, team: Option<&str>) -> u16 {
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

    fn set_difficulty(game: &mut Game, difficulty: &str) {
        let mut cfg = game.wire.game_config().clone();
        cfg.difficulty = difficulty.to_string();
        game.wire = crate::core::config::Config::new(cfg, false);
    }

    fn set_game_mode(game: &mut Game, mode: &str) {
        let mut cfg = game.wire.game_config().clone();
        cfg.game_mode = mode.to_string();
        game.wire = crate::core::config::Config::new(cfg, false);
    }

    fn set_mirv_target(game: &mut Game, owner: u16, unit_id: i32, target: TileRef) {
        if let Some(p) = game.player_by_small_id_mut(owner) {
            if let Some(u) = p.units.iter_mut().find(|u| u.id == unit_id) {
                u.target_tile = Some(target);
            }
        }
    }

    #[test]
    fn select_counter_mirv_target_prefers_the_larger_of_two_inbound_attackers() {
        let mut game = crate::test_util::plains_game(20, 20);
        let nation = add_player(&mut game, "nation", PlayerType::Nation, None);
        let small_attacker = add_player(&mut game, "small_attacker", PlayerType::Human, None);
        let big_attacker = add_player(&mut game, "big_attacker", PlayerType::Human, None);
        let bot = add_player(&mut game, "bot", PlayerType::Bot, None);

        let nation_tile = game.ref_xy(10, 10);
        game.conquer(nation, nation_tile);
        for i in 0..2 {
            game.conquer(small_attacker, game.ref_xy(i, 0));
        }
        for i in 0..5 {
            game.conquer(big_attacker, game.ref_xy(i, 1));
        }
        for i in 0..5 {
            game.conquer(bot, game.ref_xy(i, 2));
        }

        for (i, attacker) in [small_attacker, big_attacker, bot].into_iter().enumerate() {
            let mirv_tile = game.ref_xy(0, 5 + i as u32);
            let id = game.build_unit(attacker, unit_type::MIRV, mirv_tile);
            set_mirv_target(&mut game, attacker, id, nation_tile);
        }

        assert_eq!(
            select_counter_mirv_target(&game, nation),
            Some(big_attacker),
            "should retaliate against the larger of the two inbound attackers, ignoring the bot"
        );
    }

    #[test]
    fn select_counter_mirv_target_ignores_a_teammates_inbound_mirv() {
        let mut game = crate::test_util::plains_game(20, 20);
        set_game_mode(&mut game, "Team");
        let nation = add_player(&mut game, "nation", PlayerType::Nation, Some("ALPHA"));
        let teammate = add_player(&mut game, "teammate", PlayerType::Human, Some("ALPHA"));
        let nation_tile = game.ref_xy(10, 10);
        game.conquer(nation, nation_tile);

        let mirv_tile = game.ref_xy(0, 0);
        let id = game.build_unit(teammate, unit_type::MIRV, mirv_tile);
        set_mirv_target(&mut game, teammate, id, nation_tile);

        assert_eq!(select_counter_mirv_target(&game, nation), None);
    }

    #[test]
    fn select_victory_denial_target_flags_a_dominant_individual_player() {
        let mut game = crate::test_util::plains_game(10, 10); // 100 land tiles.
        let nation = add_player(&mut game, "nation", PlayerType::Nation, None);
        let dominant = add_player(&mut game, "dominant", PlayerType::Human, None);
        let small = add_player(&mut game, "small", PlayerType::Human, None);

        // 70% share clears Medium's 0.65 individual threshold.
        for i in 0..70 {
            game.conquer(dominant, game.ref_xy(i % 10, i / 10));
        }
        game.conquer(small, game.ref_xy(0, 7));
        game.conquer(nation, game.ref_xy(0, 8));

        assert_eq!(
            select_victory_denial_target(&game, nation, "Medium"),
            Some(dominant)
        );
        // Easy's higher 0.75 threshold isn't cleared by the same 70% share.
        assert_eq!(select_victory_denial_target(&game, nation, "Easy"), None);
    }

    #[test]
    fn select_victory_denial_target_targets_the_largest_member_of_a_dominant_team() {
        let mut game = crate::test_util::plains_game(10, 10); // 100 land tiles.
        set_game_mode(&mut game, "Team");
        let nation = add_player(&mut game, "nation", PlayerType::Nation, None);
        let team1 = add_player(&mut game, "team1", PlayerType::Human, Some("ALPHA"));
        let team2 = add_player(&mut game, "team2", PlayerType::Human, Some("ALPHA"));

        // team1 (60) + team2 (25) = 85% clears Medium's 0.8 team threshold;
        // team1 is clearly the largest member.
        for i in 0..60 {
            game.conquer(team1, game.ref_xy(i % 10, i / 10));
        }
        for i in 60..85 {
            game.conquer(team2, game.ref_xy(i % 10, i / 10));
        }
        game.conquer(nation, game.ref_xy(9, 9));

        assert_eq!(
            select_victory_denial_target(&game, nation, "Medium"),
            Some(team1),
            "should target the largest team member, not the smaller one"
        );
    }

    #[test]
    fn select_steamroll_stop_target_returns_the_leader_when_gap_exceeds_threshold() {
        let mut game = crate::test_util::plains_game(20, 20);
        let nation = add_player(&mut game, "nation", PlayerType::Nation, None);
        let steamroller = add_player(&mut game, "steamroller", PlayerType::Human, None);
        let second = add_player(&mut game, "second", PlayerType::Human, None);

        let steamroller_tile = game.ref_xy(0, 0);
        game.conquer(steamroller, steamroller_tile);
        for _ in 0..12 {
            game.build_unit(steamroller, unit_type::CITY, steamroller_tile);
        }
        let second_tile = game.ref_xy(1, 0);
        game.conquer(second, second_tile);
        for _ in 0..5 {
            game.build_unit(second, unit_type::CITY, second_tile);
        }
        game.conquer(nation, game.ref_xy(2, 0));

        // 12 cities > Medium's min-leader threshold (10), and 12 >= 5 * 1.5.
        assert_eq!(
            select_steamroll_stop_target(&game, nation, "Medium"),
            Some(steamroller)
        );
    }

    /// Regression: TS ranks by `unitCount(City)` (sum of levels). A leader with
    /// many L1 cities can clear a 2x *unit* gap while losing a 2x *level* gap to
    /// a second place with fewer, upgraded cities — native must not MIRV then.
    #[test]
    fn select_steamroll_stop_target_uses_city_level_sum_not_unit_cardinality() {
        let mut game = crate::test_util::plains_game(20, 20);
        let nation = add_player(&mut game, "nation", PlayerType::Nation, None);
        let many_l1 = add_player(&mut game, "many_l1", PlayerType::Human, None);
        let few_upgraded = add_player(&mut game, "few_upgraded", PlayerType::Human, None);

        let a_tile = game.ref_xy(0, 0);
        game.conquer(many_l1, a_tile);
        // 9 L1 cities: unit_count=9, level_sum=9. Easy min-leader is 20, so use Medium (10).
        // Against Medium: need >10 and >= second*1.5.
        for _ in 0..12 {
            game.build_unit(many_l1, unit_type::CITY, a_tile);
        }
        let b_tile = game.ref_xy(1, 0);
        game.conquer(few_upgraded, b_tile);
        // 5 cities upgraded to level 3 → unit_count=5, level_sum=15.
        // unit_count gap: 12 >= 5*1.5=7.5 → would wrongly fire.
        // level_sum gap: 12 >= 15*1.5=22.5 → correctly stays quiet.
        for _ in 0..5 {
            let id = game.build_unit(few_upgraded, unit_type::CITY, b_tile);
            if let Some(u) = game.unit_mut(few_upgraded, id) {
                u.level = 3;
            }
        }
        game.conquer(nation, game.ref_xy(2, 0));

        assert_eq!(
            game.unit_count(many_l1, unit_type::CITY),
            12,
            "sanity: unit cardinality gap would look like a steamroll"
        );
        assert_eq!(game.unit_level_sum(few_upgraded, unit_type::CITY), 15);
        assert_eq!(
            select_steamroll_stop_target(&game, nation, "Medium"),
            None,
            "must rank by city level sum like TS unitCount(City)"
        );
    }

    #[test]
    fn select_steamroll_stop_target_returns_none_when_leader_has_le_min_cities() {
        let mut game = crate::test_util::plains_game(20, 20);
        let nation = add_player(&mut game, "nation", PlayerType::Nation, None);
        let steamroller = add_player(&mut game, "steamroller", PlayerType::Human, None);
        let second = add_player(&mut game, "second", PlayerType::Human, None);

        let steamroller_tile = game.ref_xy(0, 0);
        game.conquer(steamroller, steamroller_tile);
        for _ in 0..10 {
            game.build_unit(steamroller, unit_type::CITY, steamroller_tile);
        }
        let second_tile = game.ref_xy(1, 0);
        game.conquer(second, second_tile);
        for _ in 0..5 {
            game.build_unit(second, unit_type::CITY, second_tile);
        }
        game.conquer(nation, game.ref_xy(2, 0));

        // Exactly at Medium's min-leader threshold (10) - TS uses `<=`, not `<`.
        assert_eq!(
            select_steamroll_stop_target(&game, nation, "Medium"),
            None
        );
    }

    #[test]
    fn calculate_territory_center_returns_the_bounding_box_center_when_owned() {
        let mut game = crate::test_util::plains_game(10, 10);
        let target = add_player(&mut game, "target", PlayerType::Human, None);
        for x in 2..8 {
            for y in 2..8 {
                game.conquer(target, game.ref_xy(x, y));
            }
        }

        assert_eq!(
            calculate_territory_center(&game, target),
            Some(game.ref_xy(4, 4))
        );
    }

    #[test]
    fn calculate_territory_center_falls_back_to_nearest_border_tile_when_center_is_unowned() {
        let mut game = crate::test_util::plains_game(10, 10);
        let target = add_player(&mut game, "target", PlayerType::Human, None);
        // An L-shape whose bounding-box center (4, 4) is NOT itself owned.
        for x in 0..5 {
            game.conquer(target, game.ref_xy(x, 0));
        }
        for y in 0..5 {
            game.conquer(target, game.ref_xy(0, y));
        }

        let center = calculate_territory_center(&game, target);
        assert!(center.is_some());
        assert_eq!(game.map.owner_id(center.unwrap()), target);
    }

    #[test]
    fn was_recently_mirved_respects_the_cooldown_window() {
        let mut game = crate::test_util::plains_game(5, 5);
        let target = add_player(&mut game, "target", PlayerType::Human, None);
        game.recent_mirv_targets.insert(target, 0);
        assert!(was_recently_mirved(&game, target));

        for _ in 0..299 {
            game.execute_next_tick();
        }
        assert_eq!(game.ticks(), 299);
        assert!(was_recently_mirved(&game, target));

        game.execute_next_tick();
        assert_eq!(game.ticks(), 300);
        assert!(!was_recently_mirved(&game, target));
    }

    /// End-to-end smoke test for the whole `consider_mirv` entry point: a
    /// nation with a missile silo, favorable gold, and an inbound MIRV from
    /// a human neighbor should autonomously queue a real `MirvExecution`
    /// that spawns a genuine in-flight MIRV unit within a couple of ticks.
    /// `considerMIRV`'s hesitation odds are a real RNG gate (unlike the
    /// deterministic target-selection helpers above), so - matching this
    /// codebase's other end-to-end AI tests - this tries a handful of seeds
    /// rather than special-casing the RNG.
    #[test]
    fn consider_mirv_launches_a_real_mirv_in_counter_retaliation_end_to_end() {
        let size = 40u32;
        for seed in 1..50i32 {
            let mut game = crate::test_util::plains_game(size, size);
            set_difficulty(&mut game, "Hard");
            let nation = add_player(&mut game, "nation", PlayerType::Nation, None);
            let attacker = add_player(&mut game, "attacker", PlayerType::Human, None);

            for x in 0..size {
                for y in 0..size {
                    if x < 5 && y < 5 {
                        continue;
                    }
                    game.conquer(nation, game.ref_xy(x, y));
                }
            }
            for x in 0..5 {
                for y in 0..5 {
                    game.conquer(attacker, game.ref_xy(x, y));
                }
            }

            let silo_tile = game.ref_xy(20, 20);
            game.build_unit(nation, unit_type::MISSILE_SILO, silo_tile);
            if let Some(p) = game.player_by_small_id_mut(nation) {
                p.gold = 1_000_000_000;
            }

            let attacker_mirv_tile = game.ref_xy(0, 0);
            let mirv_id = game.build_unit(attacker, unit_type::MIRV, attacker_mirv_tile);
            set_mirv_target(&mut game, attacker, mirv_id, silo_tile);

            for _ in 0..game.wire.spawn_immunity_duration() + 1 {
                game.execute_next_tick();
            }

            let mut random = PseudoRandom::new(seed);
            let mut emoji_state = NationEmojiState::default();
            if !consider_mirv(&mut game, &mut random, nation, &mut emoji_state) {
                continue;
            }

            // Let the queued `MirvExecution` actually init (tick 1) and spawn (tick 2).
            game.execute_next_tick();
            game.execute_next_tick();
            assert_eq!(
                game.unit_count(nation, unit_type::MIRV),
                1,
                "consider_mirv should have spawned a real in-flight MIRV unit"
            );
            return;
        }
        panic!("consider_mirv never launched a retaliatory MIRV across 50 seeds");
    }
}
