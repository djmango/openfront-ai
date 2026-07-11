//! Nation nuke-launching AI decision logic (TS `NationNukeBehavior.ts`).
//!
//! Ports `maybeSendNuke()` and its target-selection/scoring helpers as free
//! functions over `Game`/`PseudoRandom`, following this codebase's
//! established pattern (see `ai_attack.rs`) of turning a TS AI-behavior
//! *class* into free functions plus a plain-data state struct
//! (`NationNukeState`, threaded through from `nation_tick.rs`'s
//! `NationBehaviorState`) instead of porting the class/instance directly.
//!
//! `NationNukeBehavior.maybeDestroyEnemySam` and its two helpers
//! (`findEnemySamsCoveringTile`, `maybeUpgradeHelpfulSilo`) - the
//! Impossible-only SAM-overwhelm salvo planner - are fully ported (see
//! `maybe_destroy_enemy_sam` below). This was previously deferred but is now
//! done for real: every branch of `maybeSendNuke`/`findBestNukeTarget` is
//! ported.

use super::ai_attack::{find_incoming_attacker, should_attack};
use super::nation_emoji::{self, NationEmojiState};
use super::nation_structures::rand_territory_tile_array;
use super::nuke_execution::{can_build_nuke, NukeExecution};
use super::parabola;
use super::upgrade_structure::UpgradeStructureExecution;
use super::ExecEnum;
use crate::core::schemas::unit_type;
use crate::game::{Game, Player, PlayerType, Relation};
use crate::map::TileRef;
use crate::prng::PseudoRandom;
use std::collections::HashSet;

/// TS `HIGH_DENSITY_NUKE_THRESHOLD`.
const HIGH_DENSITY_NUKE_THRESHOLD: f64 = 1.0 / 75.0;
/// TS `MIN_LEVEL_SUM_FOR_HIGH_DENSITY_NUKE`.
const MIN_LEVEL_SUM_FOR_HIGH_DENSITY_NUKE: i64 = 5;
/// TS `removeOldNukeEvents`'s `maxAge` (600 ticks = 1 minute).
const RECENT_NUKE_MAX_AGE_TICKS: u32 = 600;
/// TS `EMOJI_NUKE` (`["☢️", "💥"]`) - only its length matters for RNG parity
/// (see `nation_emoji.rs`'s other `EMOJI_*_LEN` constants).
const EMOJI_NUKE_LEN: i32 = 2;
/// TS `MAX_NATION_SILO_UPGRADE_LEVEL` - cap on silo levels reachable via
/// `maybe_destroy_enemy_sam`'s upgrade fallback.
const MAX_NATION_SILO_UPGRADE_LEVEL: i32 = 5;

/// TS `Structures.types` (`City`, `DefensePost`, `SAMLauncher`, `MissileSilo`,
/// `Port`, `Factory`) - duplicated locally rather than imported, matching
/// `nuke_execution.rs`'s own local `STRUCTURE_TYPES` copy.
const STRUCTURE_TYPES: [&str; 6] = [
    unit_type::CITY,
    unit_type::DEFENSE_POST,
    unit_type::SAM_LAUNCHER,
    unit_type::MISSILE_SILO,
    unit_type::PORT,
    unit_type::FACTORY,
];

/// TS `NationNukeBehavior`'s per-instance fields (`recentlySentNukes`,
/// `atomBombsLaunched`/`hydrogenBombsLaunched` + their "perceived cost"
/// derivatives). `isHydroNation` lives on `NationBehaviorState` directly
/// (already ported before this file existed).
#[derive(Debug, Default)]
pub struct NationNukeState {
    /// TS `recentlySentNukes: [Tick, TileRef, AtomBomb|HydrogenBomb][]`.
    pub recently_sent_nukes: Vec<(u32, TileRef, String)>,
    pub atom_bombs_launched: u32,
    /// TS `atomBombPerceivedCost` - initialized to `cost(AtomBomb)` at
    /// behavior construction (`initialize_nation_behaviors`), then escalated
    /// 50% per launch to simulate saving toward a MIRV.
    pub atom_bomb_perceived_cost: i64,
    pub hydrogen_bombs_launched: u32,
    /// TS `hydrogenBombPerceivedCost` - escalated 25% per launch.
    pub hydrogen_bomb_perceived_cost: i64,
}

/// Snapshot of the nuke target's structures at decision time (TS
/// `nukeTarget.units(Structures.types)`), taken up front so the rest of
/// `maybe_send_nuke` can borrow `game` immutably without re-walking
/// `Player.units` per candidate tile.
struct StructureSnapshot {
    tile: TileRef,
    level: i32,
    unit_type: String,
}

/// TS `NationNukeBehavior.maybeSendNuke()`.
pub fn maybe_send_nuke(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    is_hydro_nation: bool,
    nuke_state: &mut NationNukeState,
    emoji_state: &mut NationEmojiState,
) {
    let silo_tiles: Vec<TileRef> = game
        .player_by_small_id(small_id)
        .map(|p| {
            p.units
                .iter()
                .filter(|u| u.unit_type == unit_type::MISSILE_SILO)
                .map(|u| u.tile as TileRef)
                .collect()
        })
        .unwrap_or_default();
    if silo_tiles.is_empty()
        || game.wire.is_unit_disabled(unit_type::MISSILE_SILO)
        || (game.wire.is_unit_disabled(unit_type::ATOM_BOMB)
            && game.wire.is_unit_disabled(unit_type::HYDROGEN_BOMB))
    {
        return;
    }

    let Some(nuke_target) = find_best_nuke_target(game, random, small_id) else {
        return;
    };

    let target_type = game
        .player_by_small_id(nuke_target)
        .map(|p| p.player_type)
        .unwrap_or(PlayerType::Human);
    if target_type == PlayerType::Bot
        || game.players_on_same_team(small_id, nuke_target)
        || !should_attack(game, random, small_id, nuke_target, false)
    {
        return;
    }

    let hydro_cost = get_perceived_nuke_cost(game, small_id, nuke_state, unit_type::HYDROGEN_BOMB);
    let atom_cost = get_perceived_nuke_cost(game, small_id, nuke_state, unit_type::ATOM_BOMB);
    let gold = game.player_by_small_id(small_id).map(|p| p.gold).unwrap_or(0);

    let nuke_type: &str;
    if !game.wire.is_unit_disabled(unit_type::HYDROGEN_BOMB) && gold >= hydro_cost {
        nuke_type = unit_type::HYDROGEN_BOMB;
    } else if !game.wire.is_unit_disabled(unit_type::ATOM_BOMB)
        && (!is_hydro_nation || is_under_heavy_attack(game, small_id))
        && gold >= atom_cost
    {
        nuke_type = unit_type::ATOM_BOMB;
    } else {
        return;
    }
    let (_inner, outer) = game.wire.nuke_magnitudes(nuke_type);
    let range = outer;

    let structures: Vec<StructureSnapshot> = game
        .player_by_small_id(nuke_target)
        .map(|p| {
            p.units
                .iter()
                .filter(|u| STRUCTURE_TYPES.contains(&u.unit_type.as_str()))
                .map(|u| StructureSnapshot {
                    tile: u.tile as TileRef,
                    level: u.level,
                    unit_type: u.unit_type.clone(),
                })
                .collect()
        })
        .unwrap_or_default();
    let structure_tiles: Vec<TileRef> = structures.iter().map(|s| s.tile).collect();

    let difficulty = game.wire.game_config().difficulty.clone();
    // Use more random tiles on Impossible difficulty to improve chances of finding a perfect SAM outranging spot.
    let num_random_tiles = if difficulty == "Impossible" { 30 } else { 10 };
    let mut all_tiles = rand_territory_tile_array(game, random, nuke_target, num_random_tiles);
    all_tiles.extend(structure_tiles.iter().copied());

    let mut seen = HashSet::new();
    let unique_tiles: Vec<TileRef> = all_tiles.into_iter().filter(|&t| seen.insert(t)).collect();

    let mut best_tile: Option<TileRef> = None;
    let mut best_value = -1.0_f64; // -1 is important, so that we can also nuke land without structures.
    remove_old_nuke_events(game, nuke_state);

    'outer: for tile in unique_tiles {
        let bb1 = bounding_box_tiles(game, tile, range as i32);
        // Add radius / 2 in case there is a piece of unwanted territory inside the outer radius that we miss.
        let bb2 = bounding_box_tiles(game, tile, (range / 2.0).floor() as i32);
        for &t in bb1.iter().chain(bb2.iter()) {
            if !is_valid_nuke_tile(game, small_id, t, nuke_target) {
                continue 'outer;
            }
        }
        let Some(spawn_tile) = can_build_nuke(game, small_id, nuke_type, tile) else {
            continue;
        };

        // In team games, avoid nuking the same position as a teammate.
        if game.wire.game_config().game_mode == "Team"
            && difficulty != "Easy"
            && is_teammate_already_nuking_this_spot(game, small_id, tile, nuke_type)
        {
            continue;
        }

        // On Hard & Impossible, avoid trajectories that can be intercepted by enemy SAMs.
        if (difficulty == "Hard" || difficulty == "Impossible")
            && is_trajectory_interceptable_by_sam(
                game,
                small_id,
                spawn_tile,
                tile,
                &HashSet::new(),
            )
        {
            continue;
        }

        // On all difficulties, avoid trajectories that cross impassable terrain
        // (the simulation aborts such launches - see NukeExecution).
        if is_trajectory_blocked_by_impassable(game, spawn_tile, tile) {
            continue;
        }

        let value = nuke_tile_score(
            game,
            tile,
            &silo_tiles,
            &structures,
            nuke_type,
            &nuke_state.recently_sent_nukes,
        );
        if value > best_value {
            best_tile = Some(tile);
            best_value = value;
        }
    }

    if let Some(bt) = best_tile {
        if best_value > 0.0 || difficulty != "Impossible" {
            send_nuke(
                game,
                random,
                small_id,
                nuke_state,
                emoji_state,
                bt,
                nuke_type,
                nuke_target,
                0,
            );
            return;
        }
    }
    if difficulty == "Impossible" {
        maybe_destroy_enemy_sam(game, random, small_id, nuke_target, nuke_state, emoji_state);
    }
}

/// TS `NationNukeBehavior.findBestNukeTarget()`.
fn find_best_nuke_target(game: &Game, random: &mut PseudoRandom, small_id: u16) -> Option<u16> {
    let difficulty = game.wire.game_config().difficulty.clone();

    // On Hard & Impossible with only 2 players left, target the only other one.
    if (difficulty == "Hard" || difficulty == "Impossible") && game.players_alive().count() == 2 {
        if let Some(other) = game.players_alive().find(|p| p.small_id != small_id) {
            return Some(other.small_id);
        }
    }

    // Retaliate against incoming attacks (Most important!)
    if let Some(attacker) = find_incoming_attacker(game, small_id) {
        return Some(attacker);
    }

    // On Impossible, the richest nation hunts very high structure density targets.
    // Restricting to the richest nation prevents every impossible nation
    // from piling onto the same compact player.
    if difficulty == "Impossible" && is_richest_nation(game, small_id) && random.chance(2) {
        if let Some(dense_target) = find_high_density_target(game, small_id) {
            return Some(dense_target);
        }
    }

    // On impossible difficulty, prioritize nuking the crown if they have more than 50% of the map.
    if difficulty == "Impossible" && game.wire.game_config().game_mode != "Team" {
        let num_tiles_without_fallout =
            game.num_land_tiles() as i64 - game.num_tiles_with_fallout() as i64;
        if num_tiles_without_fallout > 0 {
            let mut sorted_by_tiles: Vec<&Player> = game.players_alive().collect();
            sorted_by_tiles.sort_by(|a, b| b.tiles_owned.cmp(&a.tiles_owned));
            if let Some(crown) = sorted_by_tiles.first() {
                if crown.small_id != small_id && !game.is_friendly(small_id, crown.small_id) {
                    let crown_share = crown.tiles_owned as f64 / num_tiles_without_fallout as f64;
                    if crown_share > 0.5 {
                        return Some(crown.small_id);
                    }
                }
            }
        }
    }

    // Assist allies, check their targets (this is basically the same as in assistAllies, but without sending emojis).
    for ally in player_allies(game, small_id) {
        let targets = game.player_targets(ally);
        if targets.is_empty() {
            continue;
        }
        if game.relation(small_id, ally) < Relation::Friendly {
            continue;
        }
        for &target in &targets {
            if target == small_id {
                continue;
            }
            if game.is_friendly(small_id, target) {
                continue;
            }
            return Some(target);
        }
    }

    // Find the most hated player.
    // Ignore much weaker players (we don't need nukes to deal with them).
    let my_max_troops = game.max_troops_for(small_id);
    for (other, _value) in game.all_relations_sorted(small_id) {
        if game.relation(small_id, other) != Relation::Hostile {
            continue;
        }
        if game.is_friendly(small_id, other) {
            continue;
        }
        let other_max_troops = game.max_troops_for(other);
        if my_max_troops >= other_max_troops * 2.0 {
            continue;
        }
        return Some(other);
    }

    // In FFAs, nuke the crown if they're far enough ahead.
    if let Some(crown_target) = find_ffa_crown_target(game, small_id) {
        return Some(crown_target);
    }

    // In Teams, nuke the strongest team.
    if let Some(team_target) = find_strongest_team_target(game, random, small_id) {
        return Some(team_target);
    }

    None
}

fn is_richest_nation(game: &Game, small_id: u16) -> bool {
    let my_gold = game.player_by_small_id(small_id).map(|p| p.gold).unwrap_or(0);
    for other in game.players_alive() {
        if other.small_id == small_id {
            continue;
        }
        if other.player_type != PlayerType::Nation {
            continue;
        }
        if other.gold > my_gold {
            return false;
        }
    }
    true
}

fn find_high_density_target(game: &Game, small_id: u16) -> Option<u16> {
    let mut best_target: Option<u16> = None;
    let mut best_density = HIGH_DENSITY_NUKE_THRESHOLD;
    for other in game.players_alive() {
        if other.small_id == small_id {
            continue;
        }
        if other.player_type == PlayerType::Bot {
            continue;
        }
        if game.is_friendly(small_id, other.small_id) {
            continue;
        }
        let tiles_owned = other.tiles_owned;
        if tiles_owned == 0 {
            continue;
        }
        let level_sum: i64 = other
            .units
            .iter()
            .filter(|u| STRUCTURE_TYPES.contains(&u.unit_type.as_str()))
            .map(|u| u.level as i64)
            .sum();
        // Skip players with too few structures regardless of density.
        if level_sum < MIN_LEVEL_SUM_FOR_HIGH_DENSITY_NUKE {
            continue;
        }
        let density = level_sum as f64 / tiles_owned as f64;
        if density > best_density {
            best_density = density;
            best_target = Some(other.small_id);
        }
    }
    best_target
}

/// TS `NationNukeBehavior.findFFACrownTarget()`.
fn find_ffa_crown_target(game: &Game, small_id: u16) -> Option<u16> {
    let difficulty = game.wire.game_config().difficulty.clone();
    if game.wire.game_config().game_mode == "Team" {
        return None;
    }
    let mut sorted_by_tiles: Vec<&Player> = game.players_alive().collect();
    if sorted_by_tiles.len() <= 1 {
        return None;
    }
    sorted_by_tiles.sort_by(|a, b| b.tiles_owned.cmp(&a.tiles_owned));
    let first_place = sorted_by_tiles[0];

    // If we're the crown on Impossible difficulty, target 2nd place.
    if difficulty == "Impossible" && first_place.small_id == small_id && sorted_by_tiles.len() >= 2
    {
        let second_place = sorted_by_tiles[1];
        if !game.is_friendly(small_id, second_place.small_id) {
            return Some(second_place.small_id);
        }
    }

    // Don't target ourselves or allies.
    if first_place.small_id == small_id || game.is_friendly(small_id, first_place.small_id) {
        return None;
    }

    let num_tiles_without_fallout =
        game.num_land_tiles() as i64 - game.num_tiles_with_fallout() as i64;
    if num_tiles_without_fallout <= 0 {
        return None;
    }

    let first_place_share = first_place.tiles_owned as f64 / num_tiles_without_fallout as f64;
    let my_tiles = game.player_by_small_id(small_id).map(|p| p.tiles_owned).unwrap_or(0);
    let my_share = my_tiles as f64 / num_tiles_without_fallout as f64;

    let threshold = match difficulty.as_str() {
        "Easy" => 0.4,
        "Medium" => 0.3,
        "Hard" => 0.2,
        "Impossible" => 0.1,
        _ => 0.3,
    };

    // Check if first place has threshold% more tile-percentage of the map than us.
    if first_place_share - my_share > threshold {
        Some(first_place.small_id)
    } else {
        None
    }
}

/// TS `NationNukeBehavior.findStrongestTeamTarget()`.
fn find_strongest_team_target(game: &Game, random: &mut PseudoRandom, small_id: u16) -> Option<u16> {
    if game.wire.game_config().game_mode != "Team" {
        return None;
    }
    let players: Vec<&Player> = game.players_alive().collect();
    if players.len() <= 1 {
        return None;
    }

    // JS `Map` iterates in insertion order; replicate via a separately
    // tracked first-seen-team order alongside the value maps.
    let mut team_order: Vec<String> = Vec::new();
    let mut team_tiles: std::collections::HashMap<String, i64> = std::collections::HashMap::new();
    let mut team_players: std::collections::HashMap<String, Vec<u16>> =
        std::collections::HashMap::new();
    for p in &players {
        let Some(team) = p.team.clone() else {
            continue;
        };
        *team_tiles.entry(team.clone()).or_insert(0) += p.tiles_owned as i64;
        team_players
            .entry(team.clone())
            .and_modify(|v| v.push(p.small_id))
            .or_insert_with(|| {
                team_order.push(team.clone());
                vec![p.small_id]
            });
    }

    let mut sorted_teams: Vec<(String, i64)> = team_order
        .iter()
        .map(|t| (t.clone(), team_tiles[t]))
        .collect();
    sorted_teams.sort_by(|a, b| b.1.cmp(&a.1));
    if sorted_teams.is_empty() {
        return None;
    }

    let my_team = game.player_by_small_id(small_id).and_then(|p| p.team.clone());
    let mut strongest_team = sorted_teams[0].0.clone();
    if my_team.as_ref() == Some(&strongest_team) {
        if sorted_teams.len() > 1 {
            strongest_team = sorted_teams[1].0.clone();
        } else {
            return None;
        }
    }

    let target_team_players = team_players.get(&strongest_team)?;
    let valid_targets: Vec<u16> = target_team_players
        .iter()
        .copied()
        .filter(|&sid| !game.is_friendly(small_id, sid))
        .collect();
    if valid_targets.is_empty() {
        return None;
    }

    if random.chance(2) {
        // Strongest player. TS `.reduce((prev,cur) => maxTroops(prev) > maxTroops(cur) ? prev : cur)`
        // keeps `prev` only on strict `>`, so ties favor the later (current) element.
        let mut best = valid_targets[0];
        let mut best_troops = game.max_troops_for(best);
        for &sid in &valid_targets[1..] {
            let troops = game.max_troops_for(sid);
            if !(best_troops > troops) {
                best = sid;
                best_troops = troops;
            }
        }
        Some(best)
    } else {
        // Random player.
        random.rand_element(&valid_targets)
    }
}

/// TS `Player.allies()` (`this.alliances().map((a) => a.other(this))`) -
/// creation order preserved via `Game::alliances`' own insertion order
/// (broken alliances are removed via `retain`, matching TS's `_alliances`
/// list, which also drops entries on break).
fn player_allies(game: &Game, small_id: u16) -> Vec<u16> {
    game.alliances
        .iter()
        .filter_map(|a| {
            if a.requestor_small_id == small_id {
                Some(a.recipient_small_id)
            } else if a.recipient_small_id == small_id {
                Some(a.requestor_small_id)
            } else {
                None
            }
        })
        .collect()
}

/// TS `NationNukeBehavior.getPerceivedNukeCost()` - simulates saving up for a MIRV.
fn get_perceived_nuke_cost(
    game: &Game,
    small_id: u16,
    nuke_state: &NationNukeState,
    nuke_type: &str,
) -> i64 {
    // If only 2 players left, use actual cost (no point saving for MIRV).
    if game.players_alive().count() == 2 {
        return cost(game, small_id, nuke_type);
    }

    // If MIRVs are disabled, return the actual cost.
    if game.wire.is_unit_disabled(unit_type::MIRV) {
        return cost(game, small_id, nuke_type);
    }

    let gold = game.player_by_small_id(small_id).map(|p| p.gold).unwrap_or(0);

    // Save up a limited amount in team games, synced with NationStructureBehavior.
    // Saving up for a MIRV is not relevant.
    if game.wire.game_config().game_mode == "Team"
        && gold > cost(game, small_id, unit_type::HYDROGEN_BOMB)
    {
        return cost(game, small_id, nuke_type);
    }

    // Return the actual cost if we already have enough gold to buy both a MIRV and a hydro.
    if gold > cost(game, small_id, unit_type::MIRV) + cost(game, small_id, unit_type::HYDROGEN_BOMB)
    {
        return cost(game, small_id, nuke_type);
    }

    // On Hard & Impossible, ignore perceived cost when under heavy attack.
    // The nation is probably going to get destroyed soon, so go all-in on nukes.
    let difficulty = game.wire.game_config().difficulty.as_str();
    if (difficulty == "Hard" || difficulty == "Impossible") && is_under_heavy_attack(game, small_id)
    {
        return cost(game, small_id, nuke_type);
    }

    if nuke_type == unit_type::ATOM_BOMB {
        nuke_state.atom_bomb_perceived_cost
    } else {
        nuke_state.hydrogen_bomb_perceived_cost
    }
}

/// TS `NationNukeBehavior.isUnderHeavyAttack()`.
fn is_under_heavy_attack(game: &Game, small_id: u16) -> bool {
    let total_incoming_troops: f64 = game
        .incoming_attacks(small_id, false)
        .iter()
        .map(|a| a.troops)
        .sum();
    let my_troops = game.player_by_small_id(small_id).map(|p| p.troops).unwrap_or(0) as f64;
    total_incoming_troops >= my_troops
}

/// TS `NationNukeBehavior.removeOldNukeEvents()`.
fn remove_old_nuke_events(game: &Game, nuke_state: &mut NationNukeState) {
    let tick = game.ticks();
    while nuke_state
        .recently_sent_nukes
        .first()
        .is_some_and(|&(t, _, _)| t + RECENT_NUKE_MAX_AGE_TICKS < tick)
    {
        nuke_state.recently_sent_nukes.remove(0);
    }
}

/// TS `NationNukeBehavior.isTeammateAlreadyNukingThisSpot()`.
fn is_teammate_already_nuking_this_spot(
    game: &Game,
    small_id: u16,
    tile: TileRef,
    nuke_type: &str,
) -> bool {
    let (our_inner_radius, _) = game.wire.nuke_magnitudes(nuke_type);
    for p in game.players_in_order() {
        if p.small_id == small_id || !game.is_friendly(small_id, p.small_id) {
            continue;
        }
        for u in &p.units {
            if u.unit_type != unit_type::ATOM_BOMB && u.unit_type != unit_type::HYDROGEN_BOMB {
                continue;
            }
            let Some(target_tile) = u.target_tile else {
                continue;
            };
            let (teammate_inner_radius, _) = game.wire.nuke_magnitudes(&u.unit_type);
            let dist_sq = game.map.euclidean_dist_squared(tile, target_tile) as f64;
            let sum_radius = our_inner_radius + teammate_inner_radius;
            if dist_sq <= sum_radius * sum_radius {
                return true;
            }
        }
    }
    false
}

/// TS `NationNukeBehavior.isTrajectoryInterceptableBySam()`. `excluded_sam_ids`
/// mirrors TS's optional `excludedSamIds` param, used by `maybe_destroy_enemy_sam`
/// to ignore the SAM(s) it's intentionally trying to overwhelm; every other
/// caller passes an empty set (TS callers simply omit the argument).
fn is_trajectory_interceptable_by_sam(
    game: &Game,
    small_id: u16,
    spawn_tile: TileRef,
    target_tile: TileRef,
    excluded_sam_ids: &HashSet<i32>,
) -> bool {
    let speed = game.wire.default_nuke_speed();
    let trajectory = parabola::find_path_tiles(game, spawn_tile, target_tile, speed, true, true);
    if trajectory.is_empty() {
        return false;
    }

    let target_range_sq = game.wire.default_nuke_targetable_range().powi(2);

    let mut untargetable_start: i64 = -1;
    let mut untargetable_end: i64 = -1;
    for (i, &tile) in trajectory.iter().enumerate() {
        if untargetable_start == -1 {
            if game.map.euclidean_dist_squared(tile, spawn_tile) as f64 > target_range_sq {
                if (game.map.euclidean_dist_squared(tile, target_tile) as f64) < target_range_sq {
                    // Overlapping spawn & target range - no untargetable segment.
                    break;
                } else {
                    untargetable_start = i as i64;
                }
            }
        } else if (game.map.euclidean_dist_squared(tile, target_tile) as f64) < target_range_sq {
            untargetable_end = i as i64;
            break;
        }
    }

    let max_sam_range = game.wire.max_sam_range();
    let mut i: usize = 0;
    while i < trajectory.len() {
        // Skip the mid-air untargetable portion.
        if untargetable_start != -1 && untargetable_end != -1 && i as i64 == untargetable_start {
            i = untargetable_end as usize;
            continue;
        }
        let tile = trajectory[i];
        let nearby_sams =
            game.nearby_structures_any(tile, max_sam_range as u32, &[unit_type::SAM_LAUNCHER]);
        for (owner_sid, unit_id, _sam_tile, dist_sq) in nearby_sams {
            if owner_sid == small_id || game.is_friendly(small_id, owner_sid) {
                continue;
            }
            // Skip SAMs we're intentionally overwhelming.
            if excluded_sam_ids.contains(&unit_id) {
                continue;
            }
            let level = game
                .player_by_small_id(owner_sid)
                .and_then(|p| p.units.iter().find(|u| u.id == unit_id))
                .map(|u| u.level)
                .unwrap_or(1);
            let range_sq = game.wire.sam_range(level).powi(2);
            if dist_sq <= range_sq {
                return true;
            }
        }
        i += 1;
    }

    false
}

/// TS `NationNukeBehavior.isTrajectoryBlockedByImpassable()`.
fn is_trajectory_blocked_by_impassable(game: &Game, spawn_tile: TileRef, target_tile: TileRef) -> bool {
    let path = parabola::find_path_tiles(
        game,
        spawn_tile,
        target_tile,
        game.wire.default_nuke_speed(),
        true,
        true,
    );
    path.iter().any(|&t| game.is_impassable(t))
}

/// TS `NationNukeBehavior.findEnemySamsCoveringTile()` - enemy SAMs (unit id +
/// level) whose range covers `tile`, i.e. every SAM that would try to
/// intercept a nuke landing there.
fn find_enemy_sams_covering_tile(game: &Game, small_id: u16, tile: TileRef) -> Vec<(i32, i32)> {
    let max_range = game.wire.max_sam_range();
    game.nearby_structures_any(tile, max_range as u32, &[unit_type::SAM_LAUNCHER])
        .into_iter()
        .filter_map(|(owner_sid, unit_id, _sam_tile, dist_sq)| {
            if owner_sid == small_id || game.is_friendly(small_id, owner_sid) {
                return None;
            }
            let level = game
                .player_by_small_id(owner_sid)
                .and_then(|p| p.units.iter().find(|u| u.id == unit_id))
                .map(|u| u.level)
                .unwrap_or(1);
            let range = game.wire.sam_range(level);
            if dist_sq <= range * range {
                Some((unit_id, level))
            } else {
                None
            }
        })
        .collect()
}

/// TS `NationNukeBehavior.maybeUpgradeHelpfulSilo()` - upgrades the missile
/// silo (if any) that would actually have helped the just-failed overwhelm
/// attempt, preferring the one best protected by our own SAMs.
fn maybe_upgrade_helpful_silo(
    game: &mut Game,
    small_id: u16,
    failed_target_tile: TileRef,
    covering_sam_ids: &HashSet<i32>,
    total_bombs: i64,
) {
    let silos: Vec<(i32, TileRef, i32)> = game
        .player_by_small_id(small_id)
        .map(|p| {
            p.units
                .iter()
                .filter(|u| u.unit_type == unit_type::MISSILE_SILO)
                .map(|u| (u.id, u.tile as TileRef, u.level))
                .collect()
        })
        .unwrap_or_default();
    if silos.is_empty() {
        return;
    }

    // First pass: only silos whose trajectory to the failed target is
    // unblocked (not interceptable by a non-covering enemy SAM, and not
    // crossing impassable terrain) contribute slots to the overwhelm plan.
    let unblocked_silos: Vec<(i32, TileRef, i32)> = silos
        .into_iter()
        .filter(|&(_, tile, _)| {
            !is_trajectory_interceptable_by_sam(
                game,
                small_id,
                tile,
                failed_target_tile,
                covering_sam_ids,
            ) && !is_trajectory_blocked_by_impassable(game, tile, failed_target_tile)
        })
        .collect();
    if unblocked_silos.is_empty() {
        return;
    }

    // Bail out if the target is unreachable even at max silo level - crazy
    // amounts of covering SAMs, upgrading is wasted gold.
    let max_achievable_slots = unblocked_silos.len() as i64 * MAX_NATION_SILO_UPGRADE_LEVEL as i64;
    if max_achievable_slots < total_bombs {
        return;
    }

    let our_sams: Vec<(TileRef, i32)> = game
        .player_by_small_id(small_id)
        .map(|p| {
            p.units
                .iter()
                .filter(|u| u.unit_type == unit_type::SAM_LAUNCHER)
                .map(|u| (u.tile as TileRef, u.level))
                .collect()
        })
        .unwrap_or_default();

    let mut best_silo: Option<i32> = None;
    let mut best_protection: i64 = -1;
    for &(id, tile, level) in &unblocked_silos {
        if level >= MAX_NATION_SILO_UPGRADE_LEVEL {
            continue;
        }
        if !game.can_upgrade_unit(small_id, id) {
            continue;
        }
        let mut protection: i64 = 0;
        for &(sam_tile, sam_level) in &our_sams {
            let range = game.wire.sam_range(sam_level);
            let dist_sq = game.map.euclidean_dist_squared(tile, sam_tile) as f64;
            if dist_sq <= range * range {
                protection += sam_level as i64;
            }
        }
        if protection > best_protection {
            best_protection = protection;
            best_silo = Some(id);
        }
    }

    if let Some(id) = best_silo {
        game.add_execution(ExecEnum::UpgradeStructure(UpgradeStructureExecution::new(
            small_id, id,
        )));
    }
}

/// TS `NationNukeBehavior.maybeDestroyEnemySam()` - Impossible-only fallback
/// fired from `maybe_send_nuke` when no direct nuke target scored above zero:
/// plans a salvo of atom bombs sized (and time-staggered) to overwhelm one of
/// `nuke_target`'s SAMs, trying the weakest SAM first. A SAM of level N can
/// intercept N nukes before going on cooldown, so overwhelming it needs N+1
/// bombs that all survive interception and arrive within the same cooldown
/// window.
fn maybe_destroy_enemy_sam(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    nuke_target: u16,
    nuke_state: &mut NationNukeState,
    emoji_state: &mut NationEmojiState,
) {
    if game.wire.is_unit_disabled(unit_type::ATOM_BOMB) {
        return;
    }

    // Don't launch another salvo if we already have atom bombs in flight.
    let our_atom_bombs_in_flight = game
        .player_by_small_id(small_id)
        .map(|p| p.units.iter().filter(|u| u.unit_type == unit_type::ATOM_BOMB).count())
        .unwrap_or(0);
    if our_atom_bombs_in_flight > 0 {
        return;
    }

    let atom_cost = cost(game, small_id, unit_type::ATOM_BOMB);

    let mut enemy_sams: Vec<(i32, TileRef, i32)> = game
        .player_by_small_id(nuke_target)
        .map(|p| {
            p.units
                .iter()
                .filter(|u| u.unit_type == unit_type::SAM_LAUNCHER)
                .map(|u| (u.id, u.tile as TileRef, u.level))
                .collect()
        })
        .unwrap_or_default();
    if enemy_sams.is_empty() {
        return;
    }

    let our_silos: Vec<(TileRef, i32, usize)> = game
        .player_by_small_id(small_id)
        .map(|p| {
            p.units
                .iter()
                .filter(|u| u.unit_type == unit_type::MISSILE_SILO && !u.under_construction)
                .map(|u| (u.tile as TileRef, u.level, u.missile_timer_queue.len()))
                .collect()
        })
        .unwrap_or_default();
    if our_silos.is_empty() {
        return;
    }

    // Try each enemy SAM as a target, easiest (lowest level) first.
    enemy_sams.sort_by_key(|&(_, _, level)| level);

    let mut needs_more_silos = false;
    // Track the first failed attempt so we can upgrade a silo that would
    // actually have helped that plan (rather than an unrelated silo).
    let mut failed_target: Option<(TileRef, HashSet<i32>, i64)> = None;

    let nuke_speed = game.wire.default_nuke_speed();
    let sam_cooldown = game.wire.sam_cooldown() as i64;
    let max_total_arrival_spread = sam_cooldown / 2;

    for &(_, target_tile, _) in &enemy_sams {
        let covering_sams = find_enemy_sams_covering_tile(game, small_id, target_tile);
        let covering_sam_ids: HashSet<i32> = covering_sams.iter().map(|&(id, _)| id).collect();
        let total_interceptions: i64 = covering_sams.iter().map(|&(_, level)| level as i64).sum();
        let bombs_needed = total_interceptions + 1;

        // NukeExecution always picks the closest non-cooldown silo by
        // Manhattan distance to target (via nuke_spawn). Our planning must
        // mirror that order. Silos with interceptable trajectories are still
        // picked first by NukeExecution - their bombs launch but get
        // intercepted, "wasting" slots.
        let mut available_silos: Vec<(TileRef, i64, usize, bool)> = Vec::new();
        for &(silo_tile, level, queue_len) in &our_silos {
            let available_slots = level as i64 - queue_len as i64;
            if available_slots <= 0 {
                continue;
            }
            let interceptable = is_trajectory_interceptable_by_sam(
                game,
                small_id,
                silo_tile,
                target_tile,
                &covering_sam_ids,
            );
            let trajectory = parabola::find_path_tiles(game, silo_tile, target_tile, nuke_speed, true, true);
            if trajectory.is_empty() {
                continue;
            }
            // Skip silos whose trajectory crosses impassable terrain - the
            // simulation would abort these launches (see NukeExecution).
            if is_trajectory_blocked_by_impassable(game, silo_tile, target_tile) {
                continue;
            }
            available_silos.push((silo_tile, available_slots, trajectory.len(), interceptable));
        }

        // Sort by Manhattan distance to target (matching nuke_spawn's pick order).
        available_silos.sort_by_key(|&(silo_tile, ..)| game.manhattan_dist(silo_tile, target_tile));

        // Flatten into a per-bomb launch sequence matching NukeExecution's order.
        // Each silo contributes `slots` consecutive bombs before NukeExecution
        // moves to the next silo.
        let mut launch_sequence: Vec<(usize, bool)> = Vec::new();
        for &(_, slots, flight_ticks, interceptable) in &available_silos {
            for _ in 0..slots {
                launch_sequence.push((flight_ticks, interceptable));
            }
        }

        // Add extra bombs: 1 for every 5 to account for the enemy building
        // more SAMs while our bombs are in flight.
        let extra_bombs = bombs_needed / 5;
        let total_bombs = bombs_needed + extra_bombs;

        // Collect bombs from silos whose trajectory to the target is NOT
        // blocked by enemy SAMs other than the covering SAMs we're trying to
        // overwhelm.
        let unblocked_bombs: Vec<(usize, usize)> = launch_sequence
            .iter()
            .enumerate()
            .filter_map(|(index, &(flight_ticks, interceptable))| {
                if interceptable {
                    None
                } else {
                    Some((index, flight_ticks))
                }
            })
            .collect();

        if (unblocked_bombs.len() as i64) < total_bombs {
            failed_target.get_or_insert((target_tile, covering_sam_ids.clone(), total_bombs));
            needs_more_silos = true;
            continue;
        }

        // Sort unblocked bombs by flight time to find a sliding window of
        // `max_total_arrival_spread` that captures the most bombs.
        let mut sorted_by_flight = unblocked_bombs.clone();
        sorted_by_flight.sort_by_key(|&(_, flight_ticks)| flight_ticks);

        let mut best_window_start = 0usize;
        let mut best_window_count = 0usize;
        for start in 0..sorted_by_flight.len() {
            let mut end = start;
            while end < sorted_by_flight.len()
                && (sorted_by_flight[end].1 as i64) - (sorted_by_flight[start].1 as i64)
                    <= max_total_arrival_spread
            {
                end += 1;
            }
            if end - start > best_window_count {
                best_window_count = end - start;
                best_window_start = start;
            }
        }

        if (best_window_count as i64) < total_bombs {
            failed_target.get_or_insert((target_tile, covering_sam_ids.clone(), total_bombs));
            needs_more_silos = true;
            continue;
        }

        // From the window, pick `total_bombs` with the lowest launch-sequence
        // indices to minimise how many bombs we need to fire (minimise gold cost).
        let window_bombs =
            &sorted_by_flight[best_window_start..best_window_start + best_window_count];
        let mut window_by_index = window_bombs.to_vec();
        window_by_index.sort_by_key(|&(index, _)| index);
        let selected: Vec<(usize, usize)> =
            window_by_index[..total_bombs as usize].to_vec();
        let selected_set: HashSet<usize> = selected.iter().map(|&(index, _)| index).collect();
        let last_selected_index = selected.last().unwrap().0;
        let bombs_to_fire = last_selected_index + 1;

        // Compute per-bomb waitTicks so all selected bombs arrive in the
        // window. Target: spread arrivals evenly, anchored at the earliest
        // flight time in the selected set.
        let selected_flight_min = selected.iter().map(|&(_, ft)| ft).min().unwrap() as i64;
        let stagger_interval = (max_total_arrival_spread / total_bombs).max(1);
        let mut selected_idx: i64 = 0;
        let mut wait_ticks_per_bomb: Vec<u32> = Vec::with_capacity(bombs_to_fire);
        for i in 0..bombs_to_fire {
            if selected_set.contains(&i) {
                let target_arrival = selected_flight_min + selected_idx * stagger_interval;
                let flight_ticks_i = launch_sequence[i].0 as i64;
                wait_ticks_per_bomb.push((target_arrival - flight_ticks_i).max(0) as u32);
                selected_idx += 1;
            } else {
                // Wasted bomb (interceptable or out-of-window) - launch immediately.
                wait_ticks_per_bomb.push(0);
            }
        }

        // Check gold for all fired bombs (including wasted ones).
        let total_cost = atom_cost * bombs_to_fire as i64;
        let gold = game.player_by_small_id(small_id).map(|p| p.gold).unwrap_or(0);
        if gold < total_cost {
            continue;
        }

        // Fire the salvo - NukeExecution will pick silos in the same
        // Manhattan distance order we planned.
        for i in 0..bombs_to_fire {
            send_nuke(
                game,
                random,
                small_id,
                nuke_state,
                emoji_state,
                target_tile,
                unit_type::ATOM_BOMB,
                nuke_target,
                wait_ticks_per_bomb[i],
            );
        }
        return;
    }

    // Couldn't destroy any SAM - upgrade silos only if capacity was the
    // bottleneck. If we only lack gold, don't waste it upgrading silos - just
    // wait and save.
    if needs_more_silos {
        if let Some((target_tile, covering_sam_ids, total_bombs)) = failed_target {
            maybe_upgrade_helpful_silo(game, small_id, target_tile, &covering_sam_ids, total_bombs);
        }
    }
}

/// TS `NationNukeBehavior.isValidNukeTile()`.
fn is_valid_nuke_tile(game: &Game, small_id: u16, t: TileRef, nuke_target: u16) -> bool {
    let difficulty = game.wire.game_config().difficulty.as_str();
    let owner = game.map.owner_id(t);
    if owner == nuke_target {
        return true;
    }
    // On Hard & Impossible, allow TerraNullius (hit small islands) and in team games other non-friendly players.
    if difficulty == "Hard" || difficulty == "Impossible" {
        let owner_is_player = owner != 0;
        if !owner_is_player
            || (game.wire.game_config().game_mode == "Team"
                && owner_is_player
                && !game.is_friendly(small_id, owner))
        {
            return true;
        }
    }
    // On Easy & Medium, only allow tiles owned by the target player (=> nuke away from the border) to reduce nuke usage.
    false
}

/// TS `NationNukeBehavior.nukeTileScore()`.
fn nuke_tile_score(
    game: &Game,
    tile: TileRef,
    silo_tiles: &[TileRef],
    targets: &[StructureSnapshot],
    nuke_type: &str,
    recently_sent_nukes: &[(u32, TileRef, String)],
) -> f64 {
    let (_inner, outer) = game.wire.nuke_magnitudes(nuke_type);
    let outer_sq = outer * outer;
    let mut tile_value: f64 = targets
        .iter()
        .filter(|s| game.map.euclidean_dist_squared(tile, s.tile) as f64 <= outer_sq)
        .map(|s| {
            let level = s.level as f64;
            match s.unit_type.as_str() {
                unit_type::CITY => 25_000.0 * level,
                unit_type::DEFENSE_POST => 5_000.0 * level,
                unit_type::MISSILE_SILO => 50_000.0 * level,
                unit_type::PORT => 15_000.0 * level,
                unit_type::FACTORY => 15_000.0 * level,
                _ => 0.0,
            }
        })
        .sum();

    let difficulty = game.wire.game_config().difficulty.as_str();
    // On Easy, ignore SAMs entirely.
    // On Medium, apply a simple local SAM penalty.
    // On Hard & Impossible we rely on trajectory-based interception checks instead. See maybe_send_nuke().
    if difficulty == "Medium" {
        let dist50_sq = 50.0 * 50.0;
        let has_sam = targets.iter().any(|s| {
            s.unit_type == unit_type::SAM_LAUNCHER
                && game.map.euclidean_dist_squared(tile, s.tile) as f64 <= dist50_sq
        });
        if has_sam {
            return -1.0;
        }
    }

    // On Impossible difficulty and a hydrogen bomb, add value for SAMs that can be outranged.
    if difficulty == "Impossible" && nuke_type == unit_type::HYDROGEN_BOMB {
        let (_, hydro_outer) = game.wire.nuke_magnitudes(unit_type::HYDROGEN_BOMB);
        let nearby_sams =
            game.nearby_structures_any(tile, hydro_outer as u32, &[unit_type::SAM_LAUNCHER]);
        for (owner_sid, unit_id, _sam_tile, dist_sq) in nearby_sams {
            let sam_level = game
                .player_by_small_id(owner_sid)
                .and_then(|p| p.units.iter().find(|u| u.id == unit_id))
                .map(|u| u.level)
                .unwrap_or(1);
            if sam_level >= 5 {
                continue; // Can't outrange level 5+ SAMs.
            }
            let sam_range = game.wire.sam_range(sam_level);
            let dist_to_sam = dist_sq.sqrt();
            // Check if we can outrange this SAM.
            if dist_to_sam > sam_range {
                // Add significant value for destroying a SAM that we can outrange.
                tile_value += 100_000.0 * sam_level as f64;
            }
        }
    }

    // Prefer tiles that are closer to a silo (but preserve structure value).
    // `silo_tiles` is guaranteed non-empty by `maybe_send_nuke`'s early-out.
    let Some((closest_silo, _)) = crate::spatial::closest_two_tiles(game, silo_tiles, &[tile]) else {
        return tile_value;
    };
    let distance_squared = game.map.euclidean_dist_squared(tile, closest_silo) as f64;
    let distance_to_closest_silo = distance_squared.sqrt();
    let distance_penalty = distance_to_closest_silo * 30.0;
    let base_tile_value = tile_value;
    tile_value = (base_tile_value * 0.2).max(tile_value - distance_penalty); // Keep at least 20% of structure value.

    // Don't target near recent targets.
    let recent_penalty_count = recently_sent_nukes
        .iter()
        .filter(|(_tick, recent_tile, recent_nuke_type)| {
            let (recent_inner_radius, _) = game.wire.nuke_magnitudes(recent_nuke_type);
            let dist_sq = game.map.euclidean_dist_squared(tile, *recent_tile) as f64;
            dist_sq <= recent_inner_radius * recent_inner_radius
        })
        .count();
    tile_value -= recent_penalty_count as f64 * 1_000_000.0;

    tile_value
}

/// TS `NationNukeBehavior.sendNuke()`.
fn send_nuke(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    nuke_state: &mut NationNukeState,
    emoji_state: &mut NationEmojiState,
    tile: TileRef,
    nuke_type: &str,
    target_player: u16,
    wait_ticks: u32,
) {
    let tick = game.ticks();
    nuke_state
        .recently_sent_nukes
        .push((tick, tile, nuke_type.to_string()));
    if nuke_type == unit_type::ATOM_BOMB {
        nuke_state.atom_bombs_launched += 1;
        // Increase perceived cost by 50% each time to simulate saving up for a MIRV
        // (higher than hydro to make atom bombs less attractive for the lategame).
        nuke_state.atom_bomb_perceived_cost = nuke_state.atom_bomb_perceived_cost * 150 / 100;
    } else if nuke_type == unit_type::HYDROGEN_BOMB {
        nuke_state.hydrogen_bombs_launched += 1;
        // Increase perceived cost by 25% each time to simulate saving up for a MIRV.
        nuke_state.hydrogen_bomb_perceived_cost = nuke_state.hydrogen_bomb_perceived_cost * 125 / 100;
    }
    game.add_execution(ExecEnum::Nuke(NukeExecution::new(
        nuke_type, small_id, tile, None, -1.0, wait_ticks, true,
    )));
    nation_emoji::maybe_send_emoji(
        random,
        game,
        small_id,
        emoji_state,
        Some(target_player),
        EMOJI_NUKE_LEN,
    );
}

fn cost(game: &Game, small_id: u16, nuke_type: &str) -> i64 {
    game.structure_cost(small_id, nuke_type)
}

/// TS `Util.boundingBoxTiles()` - the perimeter (not filled interior) of the
/// axis-aligned box of the given `radius` around `center`.
fn bounding_box_tiles(game: &Game, center: TileRef, radius: i32) -> Vec<TileRef> {
    let mut tiles = Vec::new();
    let center_x = game.x(center) as i32;
    let center_y = game.y(center) as i32;

    let min_x = center_x - radius;
    let max_x = center_x + radius;
    let min_y = center_y - radius;
    let max_y = center_y + radius;

    // Top and bottom edges (full width).
    for x in min_x..=max_x {
        if game.is_valid_coord(x, min_y) {
            tiles.push(game.ref_xy(x as u32, min_y as u32));
        }
        if game.is_valid_coord(x, max_y) && min_y != max_y {
            tiles.push(game.ref_xy(x as u32, max_y as u32));
        }
    }

    // Left and right edges (exclude corners already added).
    for y in (min_y + 1)..max_y {
        if game.is_valid_coord(min_x, y) {
            tiles.push(game.ref_xy(min_x as u32, y as u32));
        }
        if game.is_valid_coord(max_x, y) && min_x != max_x {
            tiles.push(game.ref_xy(max_x as u32, y as u32));
        }
    }

    tiles
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::PlayerInfo;

    fn tiny_all_land_map(width: u32, height: u32) -> crate::map::GameMap {
        let n = (width * height) as usize;
        crate::map::GameMap::from_terrain_bytes(
            &crate::map::MapMeta {
                width,
                height,
                num_land_tiles: n as u32,
            },
            &vec![0x80u8; n],
        )
        .unwrap()
    }

    fn tiny_game(width: u32, height: u32, difficulty: &str, game_mode: &str) -> Game {
        let mut game = Game::default();
        game.map = tiny_all_land_map(width, height);
        game.wire = crate::core::config::Config::new(
            crate::core::schemas::GameConfig {
                game_map: "tiny".into(),
                difficulty: difficulty.into(),
                donate_gold: false,
                donate_troops: false,
                game_type: "Singleplayer".into(),
                game_mode: game_mode.into(),
                game_map_size: "Normal".into(),
                nations: crate::core::schemas::NationsConfig::Mode("default".into()),
                bots: 0,
                infinite_gold: false,
                infinite_troops: false,
                instant_build: false,
                random_spawn: false,
                doomsday_clock: None,
                disabled_units: None,
                player_teams: None,
                disable_alliances: None,
                spawn_immunity_duration: None,
                starting_gold: None,
                gold_multiplier: None,
                max_timer_value: None,
                ranked_type: None,
            },
            false,
        );
        game.end_spawn_phase();
        game
    }

    fn add_player(game: &mut Game, id: &str, player_type: PlayerType) -> u16 {
        game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type,
            client_id: Some(id.into()),
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        })
    }

    #[test]
    fn maybe_send_nuke_is_noop_without_a_missile_silo() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let attacker = add_player(&mut game, "attacker", PlayerType::Nation);
        let _target = add_player(&mut game, "target", PlayerType::Human);
        if let Some(p) = game.player_by_small_id_mut(attacker) {
            p.gold = 10_000_000;
            p.tiles_owned = 1;
        }
        let mut random = PseudoRandom::new(1);
        let mut nuke_state = NationNukeState::default();
        let mut emoji_state = NationEmojiState::default();

        maybe_send_nuke(
            &mut game,
            &mut random,
            attacker,
            false,
            &mut nuke_state,
            &mut emoji_state,
        );

        assert!(nuke_state.recently_sent_nukes.is_empty());
        assert_eq!(game.player_by_small_id(attacker).unwrap().gold, 10_000_000);
    }

    #[test]
    fn maybe_send_nuke_is_noop_when_missile_silo_is_disabled() {
        let mut game = tiny_game(20, 20, "Hard", "Free For All");
        game.wire = crate::core::config::Config::new(
            crate::core::schemas::GameConfig {
                game_map: "tiny".into(),
                difficulty: "Hard".into(),
                donate_gold: false,
                donate_troops: false,
                game_type: "Singleplayer".into(),
                game_mode: "Free For All".into(),
                game_map_size: "Normal".into(),
                nations: crate::core::schemas::NationsConfig::Mode("default".into()),
                bots: 0,
                infinite_gold: false,
                infinite_troops: false,
                instant_build: false,
                random_spawn: false,
                doomsday_clock: None,
                disabled_units: Some(vec![unit_type::MISSILE_SILO.to_string()]),
                player_teams: None,
                disable_alliances: None,
                spawn_immunity_duration: None,
                starting_gold: None,
                gold_multiplier: None,
                max_timer_value: None,
                ranked_type: None,
            },
            false,
        );
        let attacker = add_player(&mut game, "attacker", PlayerType::Nation);
        let target = add_player(&mut game, "target", PlayerType::Human);
        let silo_tile = game.ref_xy(0, 0);
        game.build_unit(attacker, unit_type::MISSILE_SILO, silo_tile);
        if let Some(p) = game.player_by_small_id_mut(attacker) {
            p.gold = 10_000_000;
            p.tiles_owned = 1;
        }
        let _ = target;

        let mut random = PseudoRandom::new(1);
        let mut nuke_state = NationNukeState::default();
        let mut emoji_state = NationEmojiState::default();

        maybe_send_nuke(
            &mut game,
            &mut random,
            attacker,
            false,
            &mut nuke_state,
            &mut emoji_state,
        );

        assert!(nuke_state.recently_sent_nukes.is_empty());
    }

    #[test]
    fn is_richest_nation_true_only_when_no_nation_has_more_gold() {
        let mut game = tiny_game(10, 10, "Medium", "Free For All");
        let me = add_player(&mut game, "me", PlayerType::Nation);
        let richer = add_player(&mut game, "richer", PlayerType::Nation);
        let human = add_player(&mut game, "human", PlayerType::Human);
        game.player_by_small_id_mut(me).unwrap().gold = 1_000;
        game.player_by_small_id_mut(richer).unwrap().gold = 2_000;
        game.player_by_small_id_mut(human).unwrap().gold = 1_000_000; // not a Nation - ignored.

        assert!(!is_richest_nation(&game, me));

        game.player_by_small_id_mut(richer).unwrap().gold = 500;
        assert!(is_richest_nation(&game, me));
    }

    #[test]
    fn find_high_density_target_ignores_bots_friendlies_and_low_structure_counts() {
        let mut game = tiny_game(30, 30, "Impossible", "Free For All");
        let me = add_player(&mut game, "me", PlayerType::Nation);
        let bot = add_player(&mut game, "bot", PlayerType::Bot);
        let sparse = add_player(&mut game, "sparse", PlayerType::Human);
        let dense = add_player(&mut game, "dense", PlayerType::Human);

        for (sid, count) in [(bot, 50), (sparse, 50), (dense, 4)] {
            for i in 0..count {
                game.conquer(sid, game.ref_xy(i, 0));
            }
        }
        // Bot: dense in structures, but excluded entirely by player_type.
        for i in 0..10 {
            game.build_unit(bot, unit_type::CITY, game.ref_xy(i, 0));
        }
        // Sparse: too few total structure levels to qualify.
        game.build_unit(sparse, unit_type::CITY, game.ref_xy(0, 0));
        // Dense: only 4 tiles owned but 6 structure levels (>= the min sum of
        // 5) at a density (1.5) way above HIGH_DENSITY_NUKE_THRESHOLD (1/75).
        for i in 0..6 {
            game.build_unit(dense, unit_type::DEFENSE_POST, game.ref_xy(i % 4, 1));
        }

        assert_eq!(find_high_density_target(&game, me), Some(dense));
    }

    #[test]
    fn get_perceived_nuke_cost_uses_actual_cost_with_two_players_left() {
        let mut game = tiny_game(10, 10, "Medium", "Free For All");
        let me = add_player(&mut game, "me", PlayerType::Nation);
        let _other = add_player(&mut game, "other", PlayerType::Human);
        let nuke_state = NationNukeState {
            atom_bomb_perceived_cost: 999_999_999,
            ..Default::default()
        };

        let cost = get_perceived_nuke_cost(&game, me, &nuke_state, unit_type::ATOM_BOMB);
        assert_eq!(cost, game.structure_cost(me, unit_type::ATOM_BOMB));
    }

    #[test]
    fn get_perceived_nuke_cost_uses_escalated_value_with_more_players_and_low_gold() {
        let mut game = tiny_game(10, 10, "Medium", "Free For All");
        let me = add_player(&mut game, "me", PlayerType::Nation);
        let _p2 = add_player(&mut game, "p2", PlayerType::Human);
        let _p3 = add_player(&mut game, "p3", PlayerType::Human);
        game.player_by_small_id_mut(me).unwrap().gold = 0;
        let nuke_state = NationNukeState {
            atom_bomb_perceived_cost: 999_999_999,
            ..Default::default()
        };

        let cost = get_perceived_nuke_cost(&game, me, &nuke_state, unit_type::ATOM_BOMB);
        assert_eq!(cost, 999_999_999);
    }

    #[test]
    fn send_nuke_escalates_perceived_cost_and_records_the_event() {
        let mut game = tiny_game(10, 10, "Medium", "Free For All");
        let me = add_player(&mut game, "me", PlayerType::Nation);
        let target = add_player(&mut game, "target", PlayerType::Human);
        let atom_cost = game.structure_cost(me, unit_type::ATOM_BOMB);
        let mut nuke_state = NationNukeState {
            atom_bomb_perceived_cost: atom_cost,
            ..Default::default()
        };
        let mut emoji_state = NationEmojiState::default();
        let mut random = PseudoRandom::new(7);
        let tile = game.ref_xy(3, 3);

        send_nuke(
            &mut game,
            &mut random,
            me,
            &mut nuke_state,
            &mut emoji_state,
            tile,
            unit_type::ATOM_BOMB,
            target,
            0,
        );

        assert_eq!(nuke_state.atom_bombs_launched, 1);
        assert_eq!(nuke_state.atom_bomb_perceived_cost, atom_cost * 150 / 100);
        assert_eq!(nuke_state.recently_sent_nukes.len(), 1);
        assert_eq!(nuke_state.recently_sent_nukes[0].1, tile);
    }

    #[test]
    fn remove_old_nuke_events_drops_only_entries_past_max_age() {
        let mut game = tiny_game(10, 10, "Medium", "Free For All");
        for _ in 0..1_000 {
            game.execute_next_tick();
        }
        let mut nuke_state = NationNukeState {
            recently_sent_nukes: vec![
                (100, 0, unit_type::ATOM_BOMB.to_string()), // 900 ticks old -> dropped.
                (500, 1, unit_type::ATOM_BOMB.to_string()), // 500 ticks old -> kept.
            ],
            ..Default::default()
        };

        remove_old_nuke_events(&game, &mut nuke_state);

        assert_eq!(nuke_state.recently_sent_nukes.len(), 1);
        assert_eq!(nuke_state.recently_sent_nukes[0].0, 500);
    }

    #[test]
    fn is_valid_nuke_tile_on_medium_only_allows_the_targets_own_tiles() {
        let mut game = tiny_game(10, 10, "Medium", "Free For All");
        let me = add_player(&mut game, "me", PlayerType::Nation);
        let target = add_player(&mut game, "target", PlayerType::Human);
        let their_tile = game.ref_xy(1, 1);
        let unowned_tile = game.ref_xy(5, 5);
        game.conquer(target, their_tile);

        assert!(is_valid_nuke_tile(&game, me, their_tile, target));
        assert!(!is_valid_nuke_tile(&game, me, unowned_tile, target));
    }

    #[test]
    fn is_valid_nuke_tile_on_hard_allows_terra_nullius_too() {
        let mut game = tiny_game(10, 10, "Hard", "Free For All");
        let me = add_player(&mut game, "me", PlayerType::Nation);
        let target = add_player(&mut game, "target", PlayerType::Human);
        let unowned_tile = game.ref_xy(5, 5);

        assert!(is_valid_nuke_tile(&game, me, unowned_tile, target));
    }

    #[test]
    fn nuke_tile_score_on_medium_penalizes_a_nearby_sam_to_negative() {
        let mut game = tiny_game(20, 20, "Medium", "Free For All");
        let target = add_player(&mut game, "target", PlayerType::Human);
        let sam_tile = game.ref_xy(10, 10);
        game.build_unit(target, unit_type::SAM_LAUNCHER, sam_tile);
        let target_tile = game.ref_xy(10, 10);
        let structures = vec![StructureSnapshot {
            tile: sam_tile,
            level: 1,
            unit_type: unit_type::SAM_LAUNCHER.to_string(),
        }];
        let silo_tiles = vec![game.ref_xy(0, 0)];

        let value = nuke_tile_score(
            &game,
            target_tile,
            &silo_tiles,
            &structures,
            unit_type::ATOM_BOMB,
            &[],
        );
        assert_eq!(value, -1.0);
    }

    #[test]
    fn nuke_tile_score_penalizes_overlap_with_a_recently_sent_nuke() {
        let mut game = tiny_game(20, 20, "Medium", "Free For All");
        let _target = add_player(&mut game, "target", PlayerType::Human);
        let tile = game.ref_xy(10, 10);
        let silo_tiles = vec![tile];
        let recently_sent = vec![(0u32, tile, unit_type::ATOM_BOMB.to_string())];

        let with_recent = nuke_tile_score(
            &game,
            tile,
            &silo_tiles,
            &[],
            unit_type::ATOM_BOMB,
            &recently_sent,
        );
        let without_recent =
            nuke_tile_score(&game, tile, &silo_tiles, &[], unit_type::ATOM_BOMB, &[]);
        assert_eq!(without_recent - with_recent, 1_000_000.0);
    }

    #[test]
    fn bounding_box_tiles_returns_only_the_perimeter() {
        let game = tiny_game(20, 20, "Medium", "Free For All");
        let center = game.ref_xy(10, 10);
        let perimeter = bounding_box_tiles(&game, center, 2);

        // A radius-2 box has an 5x5 footprint; the perimeter is everything
        // except the inner 3x3, i.e. 25 - 9 = 16 tiles.
        assert_eq!(perimeter.len(), 16);
        assert!(!perimeter.contains(&center));
        assert!(perimeter.contains(&game.ref_xy(8, 8)));
        assert!(perimeter.contains(&game.ref_xy(12, 12)));
    }

    /// End-to-end smoke test for the whole `maybe_send_nuke` entry point (not
    /// just its sub-functions): a nation with a missile silo, favorable gold,
    /// and a hostile human neighbor should autonomously build and launch a
    /// real `NukeExecution` at them within a couple of ticks. Uses "Hard" +
    /// exactly 2 alive players so `findBestNukeTarget`'s very first branch
    /// (target the only other player) fires deterministically, without
    /// needing to fake relations or dice rolls.
    #[test]
    fn maybe_send_nuke_builds_and_launches_a_real_nuke_end_to_end() {
        let size = 70u32;
        let mut game = tiny_game(size, size, "Hard", "Free For All");
        let attacker = add_player(&mut game, "attacker", PlayerType::Nation);
        let target = add_player(&mut game, "target", PlayerType::Human);

        // Target owns essentially the whole map; attacker owns only its own
        // silo's tile, far from the target's structure so every bounding-box
        // tile checked by `isValidNukeTile` stays inside the target's territory.
        for x in 0..size {
            for y in 0..size {
                let t = game.ref_xy(x, y);
                if x == 0 && y == 0 {
                    continue;
                }
                game.conquer(target, t);
            }
        }
        let silo_tile = game.ref_xy(0, 0);
        game.conquer(attacker, silo_tile);
        game.build_unit(attacker, unit_type::MISSILE_SILO, silo_tile);
        if let Some(p) = game.player_by_small_id_mut(attacker) {
            p.gold = 1_000_000; // enough for an atom bomb (750k), not a hydrogen bomb (5M).
        }

        let target_structure_tile = game.ref_xy(35, 35);
        game.build_unit(target, unit_type::CITY, target_structure_tile);

        // Nukes can't be built while the global spawn-immunity window is
        // active (see `nuke_spawn`'s `is_spawn_immunity_active` guard).
        for _ in 0..game.wire.spawn_immunity_duration() + 1 {
            game.execute_next_tick();
        }

        let mut random = PseudoRandom::new(42);
        let mut nuke_state = NationNukeState {
            atom_bomb_perceived_cost: game.structure_cost(attacker, unit_type::ATOM_BOMB),
            hydrogen_bomb_perceived_cost: game.structure_cost(attacker, unit_type::HYDROGEN_BOMB),
            ..Default::default()
        };
        let mut emoji_state = NationEmojiState::default();

        maybe_send_nuke(
            &mut game,
            &mut random,
            attacker,
            false,
            &mut nuke_state,
            &mut emoji_state,
        );

        assert_eq!(
            nuke_state.recently_sent_nukes.len(),
            1,
            "maybe_send_nuke should have queued exactly one nuke"
        );
        assert_eq!(nuke_state.atom_bombs_launched, 1);

        // Let the queued `NukeExecution` actually init (tick 1) and spawn (tick 2).
        game.execute_next_tick();
        game.execute_next_tick();

        assert_eq!(
            game.unit_count(attacker, unit_type::ATOM_BOMB),
            1,
            "NukeExecution should have spawned a real in-flight Atom Bomb unit"
        );
        assert!(
            game.player_by_small_id(attacker).unwrap().gold < 1_000_000,
            "building the nuke should have spent gold"
        );
    }

    /// TS `NationNukeSamOverwhelm.test.ts` - "nation overwhelms enemy SAM with
    /// atom bomb salvo on Impossible difficulty". Calls `maybe_send_nuke`
    /// directly (like the smoke test above) instead of driving a full
    /// `NationExecution` through many ticks with multiple game-id retries -
    /// the TS test's retry loop exists only to work around its own
    /// attack-tick RNG alignment; `should_attack` is unconditionally `true`
    /// on Impossible (no RNG draw involved at all), so one direct call is
    /// fully deterministic and exercises the exact same mechanism.
    #[test]
    fn maybe_send_nuke_overwhelms_enemy_sam_with_atom_bomb_salvo_on_impossible() {
        let size = 100u32;
        let mut game = tiny_game(size, size, "Impossible", "Free For All");
        let nation = add_player(&mut game, "nation", PlayerType::Nation);
        let human = add_player(&mut game, "human", PlayerType::Human);

        for x in 10..40 {
            for y in 10..40 {
                game.conquer(nation, game.ref_xy(x, y));
            }
        }
        for x in 60..90 {
            for y in 60..90 {
                game.conquer(human, game.ref_xy(x, y));
            }
        }

        // Level-1 SAM at the exact center of human's 30x30 block: real
        // production `sam_range(1)` (70) comfortably covers the whole block
        // (max corner distance ~21) from there, so every direct nuke attempt
        // into human territory is judged interceptable, forcing
        // `maybe_send_nuke`'s Impossible-only fallback.
        let sam_tile = game.ref_xy(75, 75);
        game.build_unit(human, unit_type::SAM_LAUNCHER, sam_tile);

        // 3 level-1 missile silos (1 slot each). Overwhelming a level-1 SAM
        // needs 2 bombs (1 intercepted + 1 that gets through).
        for &(x, y) in &[(20u32, 20u32), (25, 25), (30, 30)] {
            game.build_unit(nation, unit_type::MISSILE_SILO, game.ref_xy(x, y));
        }

        if let Some(p) = game.player_by_small_id_mut(nation) {
            p.gold = 1_000_000_000;
            p.troops = 100_000;
        }
        if let Some(p) = game.player_by_small_id_mut(human) {
            p.troops = 100_000;
        }

        // Nukes can't be built while the global spawn-immunity window is
        // active (see `nuke_spawn`'s `is_spawn_immunity_active` guard).
        for _ in 0..game.wire.spawn_immunity_duration() + 1 {
            game.execute_next_tick();
        }

        let mut random = PseudoRandom::new(42);
        let mut nuke_state = NationNukeState {
            atom_bomb_perceived_cost: game.structure_cost(nation, unit_type::ATOM_BOMB),
            hydrogen_bomb_perceived_cost: game.structure_cost(nation, unit_type::HYDROGEN_BOMB),
            ..Default::default()
        };
        let mut emoji_state = NationEmojiState::default();

        maybe_send_nuke(
            &mut game,
            &mut random,
            nation,
            false,
            &mut nuke_state,
            &mut emoji_state,
        );

        assert_eq!(
            nuke_state.atom_bombs_launched, 2,
            "overwhelming a level-1 SAM needs exactly bombsNeeded (1 intercepted + 1 through)"
        );

        // Let the queued `NukeExecution`s actually init and spawn.
        game.execute_next_tick();
        game.execute_next_tick();

        let atom_bomb_count = game.unit_count(nation, unit_type::ATOM_BOMB);
        assert!(
            atom_bomb_count >= 2,
            "expected at least 2 atom bombs in flight, got {atom_bomb_count}"
        );

        let targets: Vec<Option<TileRef>> = game
            .player_by_small_id(nation)
            .unwrap()
            .units
            .iter()
            .filter(|u| u.unit_type == unit_type::ATOM_BOMB)
            .map(|u| u.target_tile)
            .collect();
        for target in targets {
            assert_eq!(target, Some(sam_tile), "every bomb should target the SAM tile");
        }
    }
}
