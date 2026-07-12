//! Nation warship AI decision layer (`NationWarshipBehavior.ts`'s
//! `trackShipsAndRetaliate`/`counterWarshipInfestation` subset; `maybeSpawnWarship` also
//! lives here now, moved from `nation_tick.rs` - see its doc comment below for why).
//!
//! `warship.rs`'s `WarshipExecution` is the warship *unit mechanic* (patrol, shells, hunting
//! trade ships); this module is the *nation AI* layer that decides when to build or redirect
//! one, the same split as `attack.rs`'s `AttackExecution` (mechanic) vs
//! `ai_attack.rs`'s `AiAttackBehavior`-equivalent (AI decision layer).

use super::nation_emoji::{maybe_send_emoji, send_emoji, NationEmojiState};
use super::ordered_units::OrderedUnitSet;
use super::warship::{can_build_warship, warship_random_water_tile_near};
use super::{ConstructionExecution, ExecEnum, Execution};
use crate::core::schemas::unit_type::{PORT, TRADE_SHIP, TRANSPORT, WARSHIP};
use crate::game::{Game, PlayerType};
use crate::map::TileRef;
use crate::prng::PseudoRandom;
use std::collections::HashSet;

/// TS `EMOJI_WARSHIP_RETALIATION` is a single-emoji list (`["⛵"]`), so every
/// `randElement` draw over it is a no-op single choice - named purely for readability at
/// call sites, matching `nation_emoji.rs`'s existing `EMOJI_*_LEN` convention.
const EMOJI_WARSHIP_RETALIATION_LEN: i32 = 1;
/// TS `trackIncomingTransportsAndRetaliate`'s early-exit distance: a transport closer than
/// this to its own landing target is "too close to deal with".
const INCOMING_TRANSPORT_TOO_CLOSE_DIST: u32 = 20;
/// TS same function's `hasUnitNearby(target, 90, Warship, ...)` / patrol-tile proximity
/// range used to decide whether the target area already has a defender warship.
const DEFENSIVE_WARSHIP_RANGE: u32 = 90;
/// TS `maybeMoveWarship`'s "don't send ships which are already traveling" cutoff.
const WARSHIP_ALREADY_TRAVELING_DIST: u32 = 130;
/// TS `warshipSpawnTile`'s search radius as called from `trackIncomingTransportsAndRetaliate`
/// (distinct from `maybeSpawnWarship`'s own call with radius 250).
const INCOMING_TRANSPORT_OCEAN_SEARCH_RADIUS: i32 = 30;
/// TS `maybeSpawnWarship`'s own `warshipSpawnTile` search radius.
const SPAWN_PATROL_SEARCH_RADIUS: i32 = 250;
/// TS `maybeRetaliateWithWarship`/`shouldCounterWarshipInfestation`'s cap on simultaneous
/// nation-built warships (raw unit count, i.e. `player.units(Warship).length`, not the
/// level-sum `unitCount` used for the *global* threshold in `shouldCounterWarshipInfestation`).
const MAX_NATION_WARSHIPS: usize = 10;

/// TS `NationWarshipBehavior`'s four `Set<Unit>` instance fields. Kept by unit id since
/// native units don't have TS's stable object identity - see `ordered_units.rs`'s module doc
/// for why the three *iterated* sets need insertion order (`OrderedUnitSet`) while
/// `dealt_with_transport_ship` (only ever `.contains`/`.insert`/`.remove`, never iterated)
/// is a plain `HashSet<i32>`.
#[derive(Debug, Default)]
pub struct NationWarshipState {
    tracked_transport_ships: OrderedUnitSet,
    tracked_trade_ships: OrderedUnitSet,
    tracked_incoming_transport_ships: OrderedUnitSet,
    dealt_with_transport_ship: HashSet<i32>,
}

#[derive(Clone, Copy)]
enum RetaliationReason {
    Trade,
    Transport,
}

impl RetaliationReason {
    /// TS `maybeRetaliateWithWarship`'s `player.updateRelation(enemy, reason === "trade" ?
    /// -7.5 : -15)` - the trade-ship case is why `Game::update_relation` needed an `f64`
    /// variant (`update_relation_f64`); see that function's doc comment in `game.rs`.
    fn relation_delta(self) -> f64 {
        match self {
            RetaliationReason::Trade => -7.5,
            RetaliationReason::Transport => -15.0,
        }
    }
}

/// TS `NationWarshipBehavior.maybeSpawnWarship` - moved here (from a `nation_tick.rs` stub)
/// because it shares `can_build_warship`/`warship_random_water_tile_near` with the rest of
/// this module and the task brief's claim that it was "already fully implemented" didn't
/// hold up: the stub only ever consumed the `chance(50)` PRNG draw and then unconditionally
/// returned `false`, silently skipping the `randElement(ports)` + up-to-50×2 `warshipSpawnTile`
/// draws (and the actual `ConstructionExecution`) TS performs whenever gold/ports/warship-count
/// allow a spawn - a real PRNG-consumption/behavior bug, not just an unported feature.
pub fn maybe_spawn_warship(game: &mut Game, random: &mut PseudoRandom, small_id: u16) -> bool {
    if game.wire.is_unit_disabled(WARSHIP) {
        return false;
    }
    if !random.chance(50) {
        return false;
    }
    let ports: Vec<TileRef> = game
        .player_by_small_id(small_id)
        .map(|p| {
            p.units
                .iter()
                .filter(|u| u.unit_type == PORT)
                .map(|u| u.tile as TileRef)
                .collect()
        })
        .unwrap_or_default();
    let ships = game.unit_count(small_id, WARSHIP);
    // TS: `this.player.gold() > this.cost(...)` - strictly greater, evaluated (and
    // short-circuited on failure) before any further PRNG draw, same as TS's `&&` chain.
    let gold = game.player_by_small_id(small_id).map(|p| p.gold).unwrap_or(0);
    let cost = game.structure_cost(small_id, WARSHIP);
    if ports.is_empty() || ships != 0 || gold <= cost {
        return false;
    }
    let Some(port_tile) = random.rand_element(&ports) else {
        return false;
    };
    let Some(target_tile) =
        warship_random_water_tile_near(game, random, port_tile, SPAWN_PATROL_SEARCH_RADIUS)
    else {
        return false;
    };
    if !can_build_warship(game, small_id, target_tile) {
        return false;
    }
    game.add_execution(ExecEnum::Construction(ConstructionExecution::new(
        small_id, WARSHIP, target_tile, true,
    )));
    true
}

/// TS `NationWarshipBehavior.trackShipsAndRetaliate`.
pub fn track_ships_and_retaliate(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    state: &mut NationWarshipState,
    emoji: &mut NationEmojiState,
) {
    track_owned_transport_ships_and_retaliate(game, random, small_id, state, emoji);
    track_owned_trade_ships_and_retaliate(game, random, small_id, state, emoji);
    track_incoming_transports_and_retaliate(game, random, small_id, state, emoji);
}

/// TS `trackTransportShipsAndRetaliate`. Native has no `Unit.delete(displayMessage,
/// destroyer)` call to distinguish "destroyed by an enemy" from "arrived/retreated" after
/// the fact (units are just removed) - `Game::record_transport_kill`/`take_transport_kill`
/// (recorded by `ShellExecution`/`NukeExecution` right before their own `remove_unit` calls)
/// stand in for TS's `ship.wasDestroyedByEnemy() && ship.destroyer()`.
fn track_owned_transport_ships_and_retaliate(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    state: &mut NationWarshipState,
    emoji: &mut NationEmojiState,
) {
    if game.wire.is_unit_disabled(TRANSPORT) {
        return;
    }
    if let Some(p) = game.player_by_small_id(small_id) {
        let ids: Vec<i32> = p
            .units
            .iter()
            .filter(|u| u.unit_type == TRANSPORT)
            .map(|u| u.id)
            .collect();
        for id in ids {
            state.tracked_transport_ships.insert(id);
        }
    }
    for ship_id in state.tracked_transport_ships.to_vec() {
        if game.unit_exists(small_id, ship_id) {
            continue;
        }
        if let Some((destroyer, tile)) = game.take_transport_kill(ship_id) {
            maybe_retaliate_with_warship(
                game,
                random,
                small_id,
                tile,
                destroyer,
                RetaliationReason::Transport,
                emoji,
            );
        }
        state.tracked_transport_ships.remove(ship_id);
    }
}

/// TS `trackTradeShipsAndRetaliate`. Trade ships change owner in place (`capture_unit` keeps
/// the same unit id), so - unlike transports - "no longer active" and "captured" are
/// distinguished by `Game::find_unit_owner` rather than a recorded-kill lookup.
fn track_owned_trade_ships_and_retaliate(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    state: &mut NationWarshipState,
    emoji: &mut NationEmojiState,
) {
    if let Some(p) = game.player_by_small_id(small_id) {
        let ids: Vec<i32> = p
            .units
            .iter()
            .filter(|u| u.unit_type == TRADE_SHIP)
            .map(|u| u.id)
            .collect();
        for id in ids {
            state.tracked_trade_ships.insert(id);
        }
    }
    for ship_id in state.tracked_trade_ships.to_vec() {
        let Some(owner) = game.find_unit_owner(ship_id) else {
            state.tracked_trade_ships.remove(ship_id);
            continue;
        };
        if owner != small_id {
            if let Some(tile) = game.unit_tile_of(owner, ship_id) {
                maybe_retaliate_with_warship(
                    game,
                    random,
                    small_id,
                    tile,
                    owner,
                    RetaliationReason::Trade,
                    emoji,
                );
            }
            state.tracked_trade_ships.remove(ship_id);
        }
    }
}

/// TS `trackIncomingTransportsAndRetaliate`. Note the `!transport.owner().isAlliedWith(this.player)`
/// gate below is deliberately `is_allied_with`, not `is_friendly`/`isFriendly` (which
/// `findFreeForAllWarshipTarget` and the emoji behaviors elsewhere in this file/crate use) -
/// TS's own choice here, `isAlliedWith` alone excludes teammates without a formal alliance,
/// unlike every other friendliness check in this module.
fn track_incoming_transports_and_retaliate(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    state: &mut NationWarshipState,
    emoji: &mut NationEmojiState,
) {
    let incoming_ids: Vec<i32> = game
        .live_transports()
        .filter(|t| t.is_active())
        .filter_map(|t| {
            let target = t.target_tile()?;
            let unit_id = t.unit_id()?;
            if !t.is_retreating()
                && game.map.owner_id(target) == small_id
                && t.owner_small_id() != small_id
            {
                Some(unit_id)
            } else {
                None
            }
        })
        .collect();
    for unit_id in incoming_ids {
        state.tracked_incoming_transport_ships.insert(unit_id);
    }

    for transport_id in state.tracked_incoming_transport_ships.to_vec() {
        let snapshot = game
            .live_transports()
            .filter(|t| t.is_active())
            .find(|t| t.unit_id() == Some(transport_id))
            .map(|t| (t.owner_small_id(), t.target_tile(), t.is_retreating()));
        let Some((owner, target, is_retreating)) = snapshot else {
            state.tracked_incoming_transport_ships.remove(transport_id);
            state.dealt_with_transport_ship.remove(&transport_id);
            continue;
        };
        let Some(target) = target else {
            state.tracked_incoming_transport_ships.remove(transport_id);
            state.dealt_with_transport_ship.remove(&transport_id);
            continue;
        };
        if is_retreating {
            state.tracked_incoming_transport_ships.remove(transport_id);
            state.dealt_with_transport_ship.remove(&transport_id);
            continue;
        }
        if state.dealt_with_transport_ship.contains(&transport_id) {
            continue;
        }

        let Some(transport_tile) = game.unit_tile_of(owner, transport_id) else {
            continue;
        };
        let distance_to_target = game.manhattan_dist(transport_tile, target);
        if distance_to_target < INCOMING_TRANSPORT_TOO_CLOSE_DIST {
            state.dealt_with_transport_ship.insert(transport_id);
            continue;
        }

        if !game.is_allied_with(owner, small_id) {
            let already_defended = game.has_own_unit_nearby(
                small_id,
                target,
                DEFENSIVE_WARSHIP_RANGE,
                WARSHIP,
            ) || game
                .warship_patrol_candidates(small_id)
                .iter()
                .any(|&(_, _, patrol)| game.manhattan_dist(target, patrol) < DEFENSIVE_WARSHIP_RANGE);
            if already_defended {
                state.dealt_with_transport_ship.insert(transport_id);
                continue;
            }
            let Some(ocean_tile) = warship_random_water_tile_near(
                game,
                random,
                target,
                INCOMING_TRANSPORT_OCEAN_SEARCH_RADIUS,
            ) else {
                continue;
            };
            maybe_retaliate_with_warship(
                game,
                random,
                small_id,
                ocean_tile,
                owner,
                RetaliationReason::Transport,
                emoji,
            );
            state.dealt_with_transport_ship.insert(transport_id);
            break;
        }
    }
}

/// TS `maybeRetaliateWithWarship`.
fn maybe_retaliate_with_warship(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    tile: TileRef,
    enemy_small_id: u16,
    reason: RetaliationReason,
    emoji: &mut NationEmojiState,
) {
    // TS: an own-nuke-destroyed-own-transport edge case where `destroyer === this.player`.
    if enemy_small_id == small_id {
        return;
    }
    if game.unit_count(small_id, WARSHIP) >= MAX_NATION_WARSHIPS {
        maybe_move_warship(game, small_id, tile);
        return;
    }

    let difficulty = game.wire.game_config().difficulty.clone();
    // TS's `&&`-chained `difficulty === X && nextInt(...) < N` only ever evaluates (and
    // therefore only ever draws) the `nextInt` for the single matching difficulty - this
    // `match` reproduces that same one-draw-or-zero-draws PRNG discipline (Easy: no draw at all).
    let should_retaliate = match difficulty.as_str() {
        "Medium" => random.next_int(0, 100) < 15,
        "Hard" => random.next_int(0, 100) < 50,
        "Impossible" => random.next_int(0, 100) < 80,
        _ => false,
    };
    if !should_retaliate {
        return;
    }

    if !can_build_warship(game, small_id, tile) {
        maybe_move_warship(game, small_id, tile);
        return;
    }
    game.add_execution(ExecEnum::Construction(ConstructionExecution::new(
        small_id, WARSHIP, tile, true,
    )));
    maybe_send_emoji(
        random,
        game,
        small_id,
        emoji,
        Some(enemy_small_id),
        EMOJI_WARSHIP_RETALIATION_LEN,
    );
    game.update_relation_f64(small_id, enemy_small_id, reason.relation_delta());
}

/// TS `maybeMoveWarship` - redirect the least-busy existing warship instead of building a
/// new one, used both when at the `MAX_NATION_WARSHIPS` cap and as `canBuild`'s failure
/// fallback.
fn maybe_move_warship(game: &mut Game, small_id: u16, tile: TileRef) {
    if !game.is_water(tile) {
        return;
    }
    let candidate = game
        .warship_patrol_candidates(small_id)
        .into_iter()
        .filter(|&(_, current_tile, patrol_tile)| {
            game.manhattan_dist(current_tile, patrol_tile) < WARSHIP_ALREADY_TRAVELING_DIST
        })
        .min_by_key(|&(_, current_tile, _)| game.manhattan_dist(current_tile, tile));
    if let Some((unit_id, _, _)) = candidate {
        game.set_warship_patrol_tile(small_id, unit_id, tile);
    }
}

/// TS `NationWarshipBehavior.counterWarshipInfestation`.
pub fn counter_warship_infestation(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    emoji: &mut NationEmojiState,
) {
    if !should_counter_warship_infestation(game, small_id) {
        return;
    }
    let is_team_game = game
        .player_by_small_id(small_id)
        .is_some_and(|p| p.team.is_some());
    if !is_rich_player(game, small_id, is_team_game) {
        return;
    }
    if is_team_game {
        // TS `findTeamGameWarshipTarget`: this project's curriculum is FFA-only (see
        // docs/bot-ai-parity-nation-relations/README.md), making team mode dead code here -
        // only `findFreeForAllWarshipTarget` below is ported. `player.team` is always `None`
        // in that curriculum, so this branch is unreachable in practice; it's a deliberate
        // no-op rather than a silently-wrong port of the team-target search.
        return;
    }
    if let Some(target_tile) = find_ffa_warship_infestation_target(game, random, small_id) {
        build_counter_warship(game, random, small_id, target_tile, emoji);
    }
}

/// TS `shouldCounterWarshipInfestation`.
fn should_counter_warship_infestation(game: &Game, small_id: u16) -> bool {
    if game.wire.is_unit_disabled(WARSHIP) {
        return false;
    }
    let difficulty = game.wire.game_config().difficulty.as_str();
    if difficulty != "Hard" && difficulty != "Impossible" {
        return false;
    }
    // TS `this.game.unitCount(Warship)` - the level-sum global count, not a raw unit count
    // (see `MAX_NATION_WARSHIPS`'s doc comment for the distinction).
    if game.unit_level_sum_global(WARSHIP) <= 10 {
        return false;
    }
    let Some(player) = game.player_by_small_id(small_id) else {
        return false;
    };
    if game.structure_cost(small_id, WARSHIP) > player.gold {
        return false;
    }
    if game.unit_count(small_id, PORT) == 0 {
        return false;
    }
    if game.unit_count(small_id, WARSHIP) >= MAX_NATION_WARSHIPS {
        return false;
    }
    true
}

/// TS `isRichPlayer` - top 3 by gold among non-human (Bot/Nation) *alive* players
/// (`game.players()` in TS already filters to alive; `Game::all_players()` doesn't, so that
/// filter is applied here explicitly).
fn is_rich_player(game: &Game, small_id: u16, is_team_game: bool) -> bool {
    let Some(player) = game.player_by_small_id(small_id) else {
        return false;
    };
    let mut candidates: Vec<(u16, i64)> = game
        .all_players()
        .iter()
        .filter(|p| p.alive)
        .filter(|p| p.player_type != PlayerType::Human)
        .filter(|p| !is_team_game || p.team == player.team)
        .map(|p| (p.small_id, p.gold))
        .collect();
    // TS `Array.prototype.sort` is stable; `sort_by` is too, so gold ties keep their
    // `game.players()` relative order same as TS's tie-break.
    candidates.sort_by(|a, b| b.1.cmp(&a.1));
    candidates.truncate(3);
    candidates.iter().any(|&(sid, _)| sid == small_id)
}

/// TS `findFreeForAllWarshipTarget` - first (in `game.players()` order) non-friendly, alive
/// enemy with more than 10 warships; returns one of their warships' tiles chosen via
/// `randElement`. (`findTeamGameWarshipTarget`, the team-mode sibling, is deliberately not
/// ported - see `counter_warship_infestation`'s doc comment.)
fn find_ffa_warship_infestation_target(
    game: &Game,
    random: &mut PseudoRandom,
    small_id: u16,
) -> Option<TileRef> {
    let enemies: Vec<u16> = game
        .all_players()
        .iter()
        .filter(|p| p.alive)
        .filter(|p| p.small_id != small_id && !game.is_friendly(small_id, p.small_id))
        .map(|p| p.small_id)
        .collect();
    for enemy in enemies {
        let warship_ids: Vec<i32> = game
            .player_by_small_id(enemy)
            .map(|p| {
                p.units
                    .iter()
                    .filter(|u| u.unit_type == WARSHIP)
                    .map(|u| u.id)
                    .collect()
            })
            .unwrap_or_default();
        if warship_ids.len() > MAX_NATION_WARSHIPS {
            let chosen = random.rand_element(&warship_ids)?;
            return game.unit_tile_of(enemy, chosen);
        }
    }
    None
}

/// TS `buildCounterWarship`.
fn build_counter_warship(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    target_tile: TileRef,
    emoji: &mut NationEmojiState,
) {
    if !can_build_warship(game, small_id, target_tile) {
        maybe_move_warship(game, small_id, target_tile);
        return;
    }
    game.add_execution(ExecEnum::Construction(ConstructionExecution::new(
        small_id, WARSHIP, target_tile, true,
    )));
    // TS: `this.emojiBehavior.sendEmoji(AllPlayers, EMOJI_WARSHIP_RETALIATION)` - `sendEmoji`
    // (not `maybeSendEmoji`), and `AllPlayers` is `None` here (see `nation_emoji.rs`'s
    // `should_send_emoji`, which treats `recipient: None` as the always-visible broadcast).
    send_emoji(random, game, small_id, emoji, None, EMOJI_WARSHIP_RETALIATION_LEN);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::TransportShipExecution;
    use crate::execution::WarshipExecution;
    use crate::game::PlayerInfo;
    use crate::map::{GameMap, MapMeta};

    fn add_player(game: &mut Game, id: &str, team: Option<&str>) -> u16 {
        game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type: PlayerType::Nation,
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

    // A bigger all-water map so `manhattan_dist`/`is_water`/`map.owner_id` give meaningful,
    // non-degenerate geometry - `Game::default()`'s 1x1 map collapses every distance check in
    // this file to zero. No `mini_water_hpa` is built (this swaps only `game.map`, leaving
    // `mini_map`/`mini_water_hpa` at their `Default` no-op values), so anything routed through
    // `get_water_component` (`can_build_warship`'s port lookup, real pathfinding) still can't
    // work here - see `warship.rs`'s and `transport_ship.rs`'s own test modules for the same
    // documented limitation. Distance-only logic (this file's actual subject matter) doesn't
    // need it.
    fn big_water_game() -> Game {
        let mut game = Game::default();
        game.map = GameMap::from_terrain_bytes(
            &MapMeta {
                width: 200,
                height: 200,
                num_land_tiles: 0,
            },
            &vec![0u8; 200 * 200],
        )
        .expect("all-water test map");
        game.end_spawn_phase();
        game
    }

    #[test]
    fn owned_transport_ships_are_tracked_and_untracked_when_no_longer_owned() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let nation = add_player(&mut game, "nation", None);
        let tile = game.map.ref_xy(0, 0);
        let uid = game.build_unit(nation, TRANSPORT, tile);
        let mut random = PseudoRandom::new(1);
        let mut state = NationWarshipState::default();
        let mut emoji = NationEmojiState::default();

        track_owned_transport_ships_and_retaliate(
            &mut game, &mut random, nation, &mut state, &mut emoji,
        );
        assert!(state.tracked_transport_ships.contains(uid));

        // Ship arrives/retreats (removed with no recorded kill) - untracked, no retaliation.
        game.remove_unit(nation, uid);
        track_owned_transport_ships_and_retaliate(
            &mut game, &mut random, nation, &mut state, &mut emoji,
        );
        assert!(!state.tracked_transport_ships.contains(uid));
    }

    #[test]
    fn a_transport_ship_destroyed_by_an_enemy_consumes_the_retaliation_prng_roll_but_a_peaceful_removal_does_not() {
        // TS `wasDestroyedByEnemy() && destroyer() !== undefined` gates whether
        // `maybeRetaliateWithWarship` (and therefore its difficulty PRNG roll) runs at all;
        // native has no post-hoc way to inspect a removed unit's cause of death, so this
        // exercises `Game::record_transport_kill`/`take_transport_kill` as the substitute by
        // observing PRNG consumption (`PseudoRandom` has no `PartialEq`, so this clones and
        // compares the derived `Debug` string as a stand-in for "did the RNG stream advance").
        let mut game = Game::default();
        game.end_spawn_phase();
        set_difficulty(&mut game, "Medium");
        let nation = add_player(&mut game, "nation", None);
        let enemy = add_player(&mut game, "enemy", None);
        let tile = game.map.ref_xy(0, 0);

        let uid_peaceful = game.build_unit(nation, TRANSPORT, tile);
        let uid_killed = game.build_unit(nation, TRANSPORT, tile);
        let mut random = PseudoRandom::new(1);
        let mut state = NationWarshipState::default();
        let mut emoji = NationEmojiState::default();
        track_owned_transport_ships_and_retaliate(
            &mut game, &mut random, nation, &mut state, &mut emoji,
        );

        game.remove_unit(nation, uid_peaceful);
        let before_peaceful = format!("{random:?}");
        track_owned_transport_ships_and_retaliate(
            &mut game, &mut random, nation, &mut state, &mut emoji,
        );
        assert_eq!(before_peaceful, format!("{random:?}"), "peaceful removal must not draw");

        game.remove_unit(nation, uid_killed);
        game.record_transport_kill(uid_killed, nation, enemy, tile);
        let before_killed = format!("{random:?}");
        track_owned_transport_ships_and_retaliate(
            &mut game, &mut random, nation, &mut state, &mut emoji,
        );
        assert_ne!(before_killed, format!("{random:?}"), "enemy kill must draw the difficulty roll");
    }

    #[test]
    fn a_captured_trade_ship_is_detected_via_owner_change_and_untracked() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let nation = add_player(&mut game, "nation", None);
        let enemy = add_player(&mut game, "enemy", None);
        let tile = game.map.ref_xy(0, 0);
        let uid = game.build_unit(nation, TRADE_SHIP, tile);
        let mut random = PseudoRandom::new(1);
        let mut state = NationWarshipState::default();
        let mut emoji = NationEmojiState::default();

        track_owned_trade_ships_and_retaliate(&mut game, &mut random, nation, &mut state, &mut emoji);
        assert!(state.tracked_trade_ships.contains(uid));

        game.capture_unit(nation, enemy, uid);
        track_owned_trade_ships_and_retaliate(&mut game, &mut random, nation, &mut state, &mut emoji);
        assert!(!state.tracked_trade_ships.contains(uid));
        assert_eq!(game.find_unit_owner(uid), Some(enemy));
    }

    #[test]
    fn is_rich_player_only_counts_alive_non_human_players_ranked_by_gold() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let rich = add_player(&mut game, "rich", None);
        let poor1 = add_player(&mut game, "poor1", None);
        let poor2 = add_player(&mut game, "poor2", None);
        let poor3 = add_player(&mut game, "poor3", None);
        let poor4 = add_player(&mut game, "poor4", None);
        for (sid, gold) in [(rich, 1_000), (poor1, 500), (poor2, 400), (poor3, 300), (poor4, 200)] {
            game.player_by_small_id_mut(sid).unwrap().gold = gold;
        }
        assert!(is_rich_player(&game, rich, false));
        assert!(is_rich_player(&game, poor1, false));
        assert!(is_rich_player(&game, poor2, false));
        assert!(!is_rich_player(&game, poor3, false), "4th richest is not top 3");
        assert!(!is_rich_player(&game, poor4, false));

        // A dead top-3 candidate doesn't occupy a slot: the 4th-richest becomes rich once the
        // 3rd-richest is no longer alive (TS `game.players()` already filters to alive).
        game.player_by_small_id_mut(poor2).unwrap().alive = false;
        assert!(is_rich_player(&game, poor3, false));
    }

    #[test]
    fn is_rich_player_excludes_humans_and_scopes_to_team_in_team_games() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let nation = add_player(&mut game, "nation", Some("ALPHA"));
        let teammate = add_player(&mut game, "teammate", Some("ALPHA"));
        let enemy = add_player(&mut game, "enemy", Some("BETA"));
        for sid in [nation, teammate, enemy] {
            game.player_by_small_id_mut(sid).unwrap().gold = 100;
        }
        game.player_by_small_id_mut(nation).unwrap().player_type = PlayerType::Human;
        // Human `nation` is excluded from the ranking pool entirely, even in FFA mode.
        assert!(!is_rich_player(&game, nation, false));
        assert!(is_rich_player(&game, teammate, false));

        game.player_by_small_id_mut(nation).unwrap().player_type = PlayerType::Nation;
        game.player_by_small_id_mut(enemy).unwrap().gold = 1_000;
        // In team mode, richer out-of-team players don't push `nation` out of its own team's
        // top 3.
        assert!(is_rich_player(&game, nation, true));
    }

    #[test]
    fn should_counter_warship_infestation_gates_on_difficulty_global_count_gold_and_ports() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let nation = add_player(&mut game, "nation", None);
        let enemy = add_player(&mut game, "enemy", None);
        let tile = game.map.ref_xy(0, 0);
        for _ in 0..11 {
            game.build_unit(enemy, WARSHIP, tile);
        }
        game.player_by_small_id_mut(nation).unwrap().gold = 10_000_000;
        game.build_unit(nation, PORT, tile);

        set_difficulty(&mut game, "Medium");
        assert!(!should_counter_warship_infestation(&game, nation), "wrong difficulty");

        set_difficulty(&mut game, "Hard");
        assert!(should_counter_warship_infestation(&game, nation));

        game.player_by_small_id_mut(nation).unwrap().gold = 0;
        assert!(!should_counter_warship_infestation(&game, nation), "can't afford one");
        game.player_by_small_id_mut(nation).unwrap().gold = 10_000_000;

        let port_id = game.player_by_small_id(nation).unwrap().units[0].id;
        game.remove_unit(nation, port_id);
        assert!(!should_counter_warship_infestation(&game, nation), "no port");
        game.build_unit(nation, PORT, tile);

        for _ in 0..MAX_NATION_WARSHIPS {
            game.build_unit(nation, WARSHIP, tile);
        }
        assert!(
            !should_counter_warship_infestation(&game, nation),
            "already at the nation's own warship cap"
        );
    }

    #[test]
    fn find_ffa_warship_infestation_target_skips_friends_and_needs_more_than_ten_warships() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let nation = add_player(&mut game, "nation", None);
        let ally = add_player(&mut game, "ally", None);
        let small_enemy = add_player(&mut game, "small_enemy", None);
        let big_enemy = add_player(&mut game, "big_enemy", None);
        let tile = game.map.ref_xy(0, 0);
        game.create_alliance_request(ally, nation, 0);
        game.accept_alliance_request(ally, nation, 0);
        for _ in 0..5 {
            game.build_unit(ally, WARSHIP, tile);
            game.build_unit(small_enemy, WARSHIP, tile);
        }
        let mut random = PseudoRandom::new(1);
        assert!(
            find_ffa_warship_infestation_target(&game, &mut random, nation).is_none(),
            "no enemy has more than 10 warships yet"
        );

        for _ in 0..11 {
            game.build_unit(big_enemy, WARSHIP, tile);
        }
        let target = find_ffa_warship_infestation_target(&game, &mut random, nation);
        assert_eq!(target, Some(tile), "must target the over-10 enemy, not the allied or small one");
    }

    // Ported from the intent of `openfront/tests/NationCounterWarshipInfestation.test.ts`'s
    // FFA case ("rich nation sends counter-warship in FFA when enemy has too many warships"):
    // a rich nation with a port, facing an enemy with >10 warships, builds a counter-warship.
    // Unlike the TS test (which drives this through many real `NationExecution` ticks with
    // real geometry to land on the attack-tick timing), this calls `counter_warship_infestation`
    // directly and checks the `ConstructionExecution` queue rather than actually spawning the
    // unit (`can_build_warship`'s port/water-component lookup needs real HPA data no synthetic
    // test map provides - see `big_water_game`'s doc comment and `warship.rs`/`transport_ship.rs`'s
    // own tests for the same limitation), which is the deterministic, PRNG-roll-free path this
    // reaches once already at `MAX_NATION_WARSHIPS` is irrelevant here (rich-nation counters
    // aren't capped the same way retaliation is - `shouldCounterWarshipInfestation` only caps at
    // `MAX_NATION_WARSHIPS`, so with zero owned warships this always attempts `canBuild`, which
    // fails here and falls back to `maybe_move_warship` - so what's actually asserted is the
    // *targeting* decision, i.e. that a target was found at all and the right enemy tile chosen).
    #[test]
    fn rich_nation_targets_the_correct_enemy_for_counter_warship_in_ffa() {
        let mut game = Game::default();
        game.end_spawn_phase();
        set_difficulty(&mut game, "Hard");
        let nation = add_player(&mut game, "nation", None);
        let enemy = add_player(&mut game, "enemy", None);
        let tile = game.map.ref_xy(0, 0);
        game.build_unit(nation, PORT, tile);
        game.player_by_small_id_mut(nation).unwrap().gold = 10_000_000_000;
        for _ in 0..12 {
            game.build_unit(enemy, WARSHIP, tile);
        }

        assert!(should_counter_warship_infestation(&game, nation));
        assert!(is_rich_player(&game, nation, false));
        let mut random = PseudoRandom::new(1);
        assert_eq!(
            find_ffa_warship_infestation_target(&game, &mut random, nation),
            Some(tile)
        );
    }

    // TS `openfront/tests/NationCounterWarshipInfestation.test.ts`'s Team-mode case
    // ("... when enemy team has too many warships") exercises `findTeamGameWarshipTarget`,
    // which this port deliberately skips (see `counter_warship_infestation`'s doc comment -
    // team mode is dead code for this project's FFA-only curriculum).
    #[test]
    #[ignore = "findTeamGameWarshipTarget is deliberately unported dead code - see counter_warship_infestation's doc comment"]
    fn team_mode_counter_warship_infestation_is_unported() {}

    #[test]
    fn nation_at_its_warship_cap_redirects_an_existing_patrol_toward_an_incoming_enemy_transport() {
        // The end-to-end smoke test: an active warship reacting to an enemy transport heading
        // toward this nation's territory. Uses the `MAX_NATION_WARSHIPS`-cap branch of
        // `maybeRetaliateWithWarship` (deterministic redirect, no difficulty PRNG roll and no
        // `can_build_warship` port lookup) rather than the "build a brand new warship" branch,
        // since the latter needs real water-component/HPA data (`can_build_warship`'s
        // `warship_build_port_tile` call) that no synthetic test map here provides - see
        // `big_water_game`'s doc comment.
        let mut game = big_water_game();
        let nation = add_player(&mut game, "nation", None);
        let enemy = add_player(&mut game, "enemy", None);

        let home = game.map.ref_xy(5, 5);
        game.map.set_owner_id(home, nation);
        let far_from_home = game.map.ref_xy(150, 150);

        // Fill the nation's warship cap; the last one gets a matching `WarshipExecution` so
        // `warship_patrol_candidates` has something to redirect.
        let mut patrol_unit_id = 0;
        for _ in 0..MAX_NATION_WARSHIPS {
            patrol_unit_id = game.build_unit(nation, WARSHIP, far_from_home);
        }
        game.push_exec_for_test(ExecEnum::Warship(WarshipExecution::new_for_test(
            nation,
            far_from_home,
            patrol_unit_id,
        )));

        let transport_id = game.build_unit(enemy, TRANSPORT, far_from_home);
        game.push_exec_for_test(ExecEnum::TransportShip(TransportShipExecution::new_for_test(
            enemy,
            transport_id,
            home,
            false,
        )));

        let mut random = PseudoRandom::new(1);
        let mut state = NationWarshipState::default();
        let mut emoji = NationEmojiState::default();
        track_ships_and_retaliate(&mut game, &mut random, nation, &mut state, &mut emoji);

        let patrol_tile_after = game
            .warship_patrol_candidates(nation)
            .into_iter()
            .find(|&(uid, _, _)| uid == patrol_unit_id)
            .map(|(_, _, patrol)| patrol)
            .expect("warship still tracked");
        assert_ne!(
            patrol_tile_after, far_from_home,
            "the nation's warship must have been redirected toward the incoming transport"
        );
        assert!(
            game.manhattan_dist(patrol_tile_after, home) < game.manhattan_dist(far_from_home, home),
            "redirected patrol tile should be much closer to the threatened territory"
        );
    }
}
