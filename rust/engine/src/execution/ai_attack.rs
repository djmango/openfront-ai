//! Shared AI attack behavior (TS `AiAttackBehavior.ts` subset for hash parity).

use crate::game::{Game, PlayerType, Relation};
use crate::map::TileRef;
use crate::prng::PseudoRandom;
use crate::spatial::{can_build_transport_ship, closest_two_tiles, shore_border_tiles};
use std::collections::HashSet;

pub fn land_attack_troops(game: &Game, small_id: u16, reserve_or_expand_ratio: f64) -> Option<f64> {
    let attacker = game.player_by_small_id(small_id)?;
    let max_troops = game.max_troops_for(attacker.small_id);
    let target_troops = max_troops * reserve_or_expand_ratio;
    let troops = attacker.troops as f64 - target_troops;
    if troops < 1.0 {
        return None;
    }
    Some(troops)
}

pub fn has_reserve_ratio(game: &Game, small_id: u16, reserve_ratio: f64) -> bool {
    let Some(attacker) = game.player_by_small_id(small_id) else {
        return false;
    };
    let max_troops = game.max_troops_for(attacker.small_id);
    if max_troops <= 0.0 {
        return false;
    }
    attacker.troops as f64 / max_troops >= reserve_ratio
}

pub fn has_trigger_ratio(game: &Game, small_id: u16, trigger_ratio: f64) -> bool {
    let Some(attacker) = game.player_by_small_id(small_id) else {
        return false;
    };
    let max_troops = game.max_troops_for(attacker.small_id);
    if max_troops <= 0.0 {
        return false;
    }
    attacker.troops as f64 / max_troops >= trigger_ratio
}

pub fn has_land_border_tn(game: &Game, small_id: u16) -> bool {
    let Some(border) = game.border_tiles_of(small_id) else {
        return false;
    };
    let mut nbuf = [TileRef::MAX; 4];
    for border_tile in border.iter() {
        let n = game.map.neighbors4_ts(border_tile, &mut nbuf);
        for i in 0..n {
            let neighbor = nbuf[i];
            if game.is_land(neighbor)
                && !game.is_impassable(neighbor)
                && !game.has_owner(neighbor)
                && !game.has_fallout(neighbor)
            {
                return true;
            }
        }
    }
    false
}

fn sampled_shore_tiles(game: &Game, small_id: u16) -> Vec<TileRef> {
    let shores: Vec<TileRef> = game
        .border_tiles_of(small_id)
        .map(|border| border.iter().filter(|t| game.is_shore(*t)).collect())
        .unwrap_or_default();
    shores.into_iter().step_by(10).collect()
}

fn has_shore_reachable_tn(game: &Game, small_id: u16) -> bool {
    let directions: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
    for border in sampled_shore_tiles(game, small_id) {
        let bx = game.x(border) as i32;
        let by = game.y(border) as i32;
        for (dx, dy) in directions {
            let x1 = bx + dx;
            let y1 = by + dy;
            if !game.is_valid_coord(x1, y1) {
                continue;
            }
            let t1 = game.ref_xy(x1 as u32, y1 as u32);
            if !game.is_water(t1) {
                continue;
            }
            let nx = bx + dx * 5;
            let ny = by + dy * 5;
            if !game.is_valid_coord(nx, ny) {
                continue;
            }
            let tile = game.ref_xy(nx as u32, ny as u32);
            if game.is_land(tile)
                && !game.is_impassable(tile)
                && !game.has_owner(tile)
                && !game.has_fallout(tile)
            {
                return true;
            }
        }
    }
    false
}

pub fn has_non_nuked_tn(game: &Game, small_id: u16) -> bool {
    has_land_border_tn(game, small_id) || has_shore_reachable_tn(game, small_id)
}

pub fn nearby_land_player_small_ids(game: &Game, small_id: u16) -> Vec<u16> {
    let mut seen = HashSet::new();
    game.for_each_border_tile(small_id, |tile| {
        game.map.for_each_neighbor4(tile, |neighbor| {
            // TS `PlayerImpl.nearby()`: `map.isLand(n) && !map.isImpassable(n)`.
            if !game.is_land(neighbor) || game.is_impassable(neighbor) {
                return;
            }
            let owner = game.map.owner_id(neighbor);
            if owner != small_id && owner != 0 {
                seen.insert(owner);
            }
        });
    });
    let mut out: Vec<u16> = seen.into_iter().collect();
    out.sort_unstable();
    out
}

pub fn nearby_player_small_ids(game: &Game, small_id: u16) -> Vec<u16> {
    let mut seen = HashSet::new();
    game.for_each_border_tile(small_id, |tile| {
        game.map.for_each_neighbor4(tile, |neighbor| {
            // TS `PlayerImpl.nearby()`: `map.isLand(n) && !map.isImpassable(n)`.
            if !game.is_land(neighbor) || game.is_impassable(neighbor) {
                return;
            }
            let owner = game.map.owner_id(neighbor);
            if owner != small_id && owner != 0 {
                seen.insert(owner);
            }
        });
    });
    // Shore-reachable neighbors (TS `PlayerImpl.shoreReachableNeighbors`).
    let directions: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
    let mut i = 0usize;
    game.for_each_border_tile(small_id, |border| {
        if !game.is_shore(border) {
            return;
        }
        i += 1;
        if i % 10 != 1 {
            return;
        }
        let bx = game.x(border) as i32;
        let by = game.y(border) as i32;
        for (dx, dy) in directions {
            let x1 = bx + dx;
            let y1 = by + dy;
            if !game.is_valid_coord(x1, y1) {
                continue;
            }
            let t1 = game.ref_xy(x1 as u32, y1 as u32);
            if !game.is_water(t1) {
                continue;
            }
            let nx = bx + dx * 5;
            let ny = by + dy * 5;
            if !game.is_valid_coord(nx, ny) {
                continue;
            }
            let tile = game.ref_xy(nx as u32, ny as u32);
            if !game.is_land(tile) || game.is_impassable(tile) || game.has_fallout(tile) {
                continue;
            }
            let owner = game.map.owner_id(tile);
            if owner != small_id && owner != 0 {
                seen.insert(owner);
            }
        }
    });
    let mut out: Vec<u16> = seen.into_iter().collect();
    out.sort_unstable();
    out
}

pub fn send_boat_attack_to_nearby_tn(game: &mut Game, small_id: u16) -> bool {
    if game.wire.is_unit_disabled(crate::core::schemas::unit_type::TRANSPORT) {
        return false;
    }
    if game.unit_count(small_id, crate::core::schemas::unit_type::TRANSPORT) >= game.wire.boat_max_number() {
        return false;
    }

    // TS `sendBoatAttackToNearbyTerraNullius`: on the *first* tile passing the
    // geometric + canBuildTransportShip checks, compute troops and either send
    // the attack or give up entirely (it does not keep searching after that).
    // We can't call `can_build_transport_ship` (needs `&mut Game`) from inside
    // the `for_each_border_tile` closure (needs `&Game`), so first collect the
    // ordered list of geometric candidates, then re-check them in order.
    let directions: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
    let mut shore_i = 0usize;
    let mut candidates: Vec<TileRef> = Vec::new();
    game.for_each_border_tile(small_id, |border| {
        if !game.is_shore(border) {
            return;
        }
        shore_i += 1;
        if shore_i % 10 != 1 {
            return;
        }
        let bx = game.x(border) as i32;
        let by = game.y(border) as i32;
        for (dx, dy) in directions {
            let x1 = bx + dx;
            let y1 = by + dy;
            if !game.is_valid_coord(x1, y1) {
                continue;
            }
            let t1 = game.ref_xy(x1 as u32, y1 as u32);
            if !game.is_water(t1) {
                continue;
            }
            let nx = bx + dx * 5;
            let ny = by + dy * 5;
            if !game.is_valid_coord(nx, ny) {
                continue;
            }
            let tile = game.ref_xy(nx as u32, ny as u32);
            if game.is_land(tile)
                && !game.is_impassable(tile)
                && !game.has_owner(tile)
                && !game.has_fallout(tile)
            {
                candidates.push(tile);
            }
        }
    });

    let Some(&dst) = candidates.iter().find(|&&t| can_build_transport_ship(game, small_id, t).is_some()) else {
        return false;
    };
    // TS `sendBoatAttackToNearbyTerraNullius`: `troops = this.player.troops() / 5`
    // - unlike `Config.boatAttackAmount()`, this is NOT floored.
    let troops = game
        .player_by_small_id(small_id)
        .map(|p| p.troops as f64 / 5.0)
        .unwrap_or(0.0);
    if troops < 1.0 {
        return false;
    }
    game.add_transport_attack(small_id, dst, troops);
    true
}

pub fn try_send_tn_attack(game: &mut Game, small_id: u16, expand_ratio: f64) -> bool {
    if has_land_border_tn(game, small_id) {
        if let Some(troops) = land_attack_troops(game, small_id, expand_ratio) {
            game.add_land_attack(small_id, None, Some(troops));
            return true;
        }
        return false;
    }
    send_boat_attack_to_nearby_tn(game, small_id)
}

pub fn send_tn_attack(game: &mut Game, small_id: u16, expand_ratio: f64) -> bool {
    try_send_tn_attack(game, small_id, expand_ratio)
}

pub fn try_send_player_attack(
    game: &mut Game,
    random: &mut PseudoRandom,
    attacker_small_id: u16,
    target_small_id: u16,
    reserve_ratio: f64,
    expand_ratio: f64,
    bot_attack_troops_sent: &mut f64,
    difficulty: &str,
    emoji: Option<&mut super::nation_emoji::NationEmojiState>,
) -> bool {
    try_send_player_attack_forced(
        game,
        random,
        attacker_small_id,
        target_small_id,
        reserve_ratio,
        expand_ratio,
        bot_attack_troops_sent,
        difficulty,
        emoji,
        false,
    )
}

fn try_send_player_attack_forced(
    game: &mut Game,
    random: &mut PseudoRandom,
    attacker_small_id: u16,
    target_small_id: u16,
    reserve_ratio: f64,
    expand_ratio: f64,
    bot_attack_troops_sent: &mut f64,
    difficulty: &str,
    emoji: Option<&mut super::nation_emoji::NationEmojiState>,
    force: bool,
) -> bool {
    if !should_attack(game, random, attacker_small_id, target_small_id, force) {
        return false;
    }

    // TS `AiAttackBehavior.calculateAttackTroops`: whenever the target is a Bot
    // and the attacker is not itself a Bot, troop sizing always goes through
    // `calculateBotAttackTroops` (target.troops() * 4, capped) with the
    // expand-ratio reserve when the bot owns structures - regardless of which
    // strategy (retaliate/weakest/traitor/hated/afk/betray/island/...)
    // triggered the attack, not just the dedicated `attackBots()` strategy.
    let attacker_is_bot = game
        .player_by_small_id(attacker_small_id)
        .is_some_and(|p| p.player_type == PlayerType::Bot);
    let target_is_bot = game
        .player_by_small_id(target_small_id)
        .is_some_and(|p| p.player_type == PlayerType::Bot);
    if target_is_bot && !attacker_is_bot {
        let target_reserve_ratio = if player_has_structure_units(game, target_small_id) {
            expand_ratio
        } else {
            reserve_ratio
        };
        return try_send_nation_bot_attack(
            game,
            attacker_small_id,
            target_small_id,
            target_reserve_ratio,
            bot_attack_troops_sent,
            difficulty,
        );
    }

    let is_nation = game
        .player_by_small_id(attacker_small_id)
        .is_some_and(|p| p.player_type == PlayerType::Nation);
    let is_human = game
        .player_by_small_id(target_small_id)
        .is_some_and(|p| p.player_type == PlayerType::Human);

    if game.shares_land_border_with(attacker_small_id, target_small_id) {
        let target_id = match game.player_by_small_id(target_small_id) {
            Some(p) => p.id.clone(),
            None => return false,
        };
        let Some(troops) = land_attack_troops(game, attacker_small_id, reserve_ratio) else {
            return false;
        };
        if is_nation && is_human {
            if let Some(state) = emoji {
                super::nation_emoji::maybe_send_attack_emoji(
                    game,
                    random,
                    attacker_small_id,
                    state,
                    target_small_id,
                );
            }
        }
        game.add_land_attack(attacker_small_id, Some(target_id), Some(troops));
        return true;
    }
    send_boat_attack_to_player(
        game,
        attacker_small_id,
        target_small_id,
        game.player_by_small_id(attacker_small_id)
            .map(|p| game.wire.boat_attack_amount(p.troops))
            .unwrap_or(0.0),
    )
}

pub fn send_boat_attack_to_player(
    game: &mut Game,
    attacker_small_id: u16,
    target_small_id: u16,
    troops: f64,
) -> bool {
    if troops < 1.0 {
        return false;
    }
    if game.wire.is_unit_disabled(crate::core::schemas::unit_type::TRANSPORT) {
        return false;
    }
    if game.unit_count(attacker_small_id, crate::core::schemas::unit_type::TRANSPORT)
        >= game.wire.boat_max_number()
    {
        return false;
    }

    let attacker_shores = shore_border_tiles(game, attacker_small_id);
    let target_shores = shore_border_tiles(game, target_small_id);
    let Some((_src_shore, dst_shore)) = closest_two_tiles(game, &attacker_shores, &target_shores) else {
        return false;
    };
    if can_build_transport_ship(game, attacker_small_id, dst_shore).is_none() {
        return false;
    }

    game.add_transport_attack(attacker_small_id, dst_shore, troops);
    true
}

pub fn collect_bordering_players_pub(game: &Game, small_id: u16) -> Vec<u16> {
    collect_bordering_players(game, small_id)
}

fn collect_bordering_players(game: &Game, small_id: u16) -> Vec<u16> {
    let mut seen = HashSet::new();
    let mut ordered: Vec<u16> = Vec::new();
    let mut push = |sid: u16| {
        if sid != small_id && sid != 0 && seen.insert(sid) {
            ordered.push(sid);
        }
    };

    // TS `AiAttackBehavior.maybeAttack`: borderTiles flatMap neighbors() order.
    if let Some(border) = game.border_tiles_of(small_id) {
        let mut nbuf = [TileRef::MAX; 4];
        for border_tile in border.iter() {
            let n = game.map.neighbors4_ts(border_tile, &mut nbuf);
            for i in 0..n {
                let neighbor = nbuf[i];
                if !game.is_land(neighbor) || game.is_impassable(neighbor) {
                    continue;
                }
                let owner = game.map.owner_id(neighbor);
                push(owner);
            }
        }
    }

    // TS `PlayerImpl.nearby()` players only (shore-reachable included).
    for sid in nearby_players_ts_order(game, small_id) {
        push(sid);
    }

    ordered.sort_by_key(|&sid| game.player_by_small_id(sid).map(|p| p.troops).unwrap_or(0));
    ordered
}

/// TS `PlayerImpl.nearby()` small IDs (players AND `TerraNullius`, small ID
/// `0`) in `Set` insertion order. TS's `nearby()` unconditionally
/// `ns.add(this.mg.playerBySmallID(owner))` for every non-self bordering/
/// shore-reachable owner, INCLUDING owner `0` (`TerraNullius`) - callers
/// that need a players-only view filter it themselves (e.g.
/// `getNeighborTraitorToAttack`'s `n.isPlayer()` type guard,
/// `TribeExecution.maybeAttack`'s shuffled loop's `if (!neighbor.isPlayer())
/// continue`). Do NOT filter out `0` here: several callers (notably the
/// tribe random-target shuffle) rely on `TerraNullius` occupying a slot in
/// this list so the shuffle consumes the same number of PRNG draws, and in
/// the same relative order, as TS - dropping it here silently changes every
/// subsequent shuffled pick for this entity. Safe for the other 3
/// call sites: they each already drop small ID `0` via
/// `game.player_by_small_id(0) == None` (`filter_map`/`Option` chains) or an
/// equivalent player-only filter.
fn nearby_players_ts_order(game: &Game, small_id: u16) -> Vec<u16> {
    let mut seen = HashSet::new();
    let mut ordered: Vec<u16> = Vec::new();
    let mut push = |sid: u16| {
        if sid != small_id && seen.insert(sid) {
            ordered.push(sid);
        }
    };

    if let Some(border) = game.border_tiles_of(small_id) {
        let mut nbuf = [TileRef::MAX; 4];
        for border_tile in border.iter() {
            let n = game.map.neighbors4_ts(border_tile, &mut nbuf);
            for i in 0..n {
                let neighbor = nbuf[i];
                if !game.is_land(neighbor) || game.is_impassable(neighbor) {
                    continue;
                }
                push(game.map.owner_id(neighbor));
            }
        }
    }

    let directions: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
    let shores: Vec<TileRef> = game
        .border_tiles_of(small_id)
        .map(|border| border.iter().filter(|&t| game.is_shore(t)).collect())
        .unwrap_or_default();
    for i in (0..shores.len()).step_by(10) {
        let border = shores[i];
        let bx = game.x(border) as i32;
        let by = game.y(border) as i32;
        for (dx, dy) in directions {
            let x1 = bx + dx;
            let y1 = by + dy;
            if !game.is_valid_coord(x1, y1) {
                continue;
            }
            let t1 = game.ref_xy(x1 as u32, y1 as u32);
            if !game.is_water(t1) {
                continue;
            }
            let nx = bx + dx * 5;
            let ny = by + dy * 5;
            if !game.is_valid_coord(nx, ny) {
                continue;
            }
            let tile = game.ref_xy(nx as u32, ny as u32);
            if !game.is_land(tile) || game.is_impassable(tile) || game.has_fallout(tile) {
                continue;
            }
            push(game.map.owner_id(tile));
        }
    }

    ordered
}

fn bordering_enemies_by_troops(game: &Game, small_id: u16) -> Vec<u16> {
    collect_bordering_players(game, small_id)
        .into_iter()
        .filter(|&sid| sid != small_id && !game.is_friendly(small_id, sid))
        .collect()
}

fn bordering_friends_by_troops(game: &Game, small_id: u16) -> Vec<u16> {
    collect_bordering_players(game, small_id)
        .into_iter()
        .filter(|&sid| sid != small_id && game.is_friendly(small_id, sid))
        .collect()
}

fn player_has_structure_units(game: &Game, small_id: u16) -> bool {
    const STRUCTURES: &[&str] = &[
        crate::core::schemas::unit_type::CITY,
        crate::core::schemas::unit_type::PORT,
        crate::core::schemas::unit_type::FACTORY,
        crate::core::schemas::unit_type::DEFENSE_POST,
        crate::core::schemas::unit_type::MISSILE_SILO,
        crate::core::schemas::unit_type::SAM_LAUNCHER,
    ];
    game.player_by_small_id(small_id).is_some_and(|p| {
        p.units.iter().any(|u| STRUCTURES.contains(&u.unit_type.as_str()))
    })
}

fn has_neighboring_bot_with_structures(game: &Game, small_id: u16) -> bool {
    nearby_player_small_ids(game, small_id).into_iter().any(|sid| {
        game.player_by_small_id(sid).is_some_and(|p| {
            p.player_type == PlayerType::Bot && player_has_structure_units(game, sid)
        })
    })
}

fn is_bordering_nuked_territory(game: &Game, small_id: u16) -> bool {
    if game.wire.is_unit_disabled(crate::core::schemas::unit_type::MISSILE_SILO) {
        return false;
    }
    let mut found = false;
    game.for_each_border_tile(small_id, |tile| {
        if found {
            return;
        }
        game.map.for_each_neighbor4(tile, |neighbor| {
            if game.is_land(neighbor) && !game.has_owner(neighbor) && game.has_fallout(neighbor) {
                found = true;
            }
        });
    });
    found
}

fn find_incoming_attacker(game: &Game, small_id: u16) -> Option<u16> {
    let ptype = game
        .player_by_small_id(small_id)
        .map(|p| p.player_type)
        .unwrap_or(PlayerType::Human);
    game.find_incoming_land_attacker(small_id, ptype)
}

fn find_victim(game: &Game, bordering: &[u16]) -> Option<u16> {
    for &sid in bordering {
        let Some(enemy) = game.player_by_small_id(sid) else {
            continue;
        };
        let incoming = game.incoming_land_troops(sid);
        if incoming > enemy.troops as f64 * 0.5 {
            return Some(sid);
        }
    }
    None
}

fn find_very_weak_enemy(game: &Game, bordering: &[u16]) -> Option<u16> {
    for &sid in bordering {
        let Some(enemy) = game.player_by_small_id(sid) else {
            continue;
        };
        let max = game.max_troops_for(enemy.small_id);
        if (enemy.troops as f64) < max * 0.15 {
            return Some(sid);
        }
    }
    None
}

fn player_center_tile(game: &Game, small_id: u16) -> Option<TileRef> {
    let border = game.border_tiles_of(small_id)?;
    let mut tiles = border.iter();
    let first = tiles.next()?;
    let (mut min_x, mut max_x) = (game.x(first), game.x(first));
    let (mut min_y, mut max_y) = (game.y(first), game.y(first));
    for tile in tiles {
        let x = game.x(tile);
        let y = game.y(tile);
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    Some(game.ref_xy(min_x + (max_x - min_x) / 2, min_y + (max_y - min_y) / 2))
}

fn find_nearest_island_enemy(
    game: &mut Game,
    random: &mut PseudoRandom,
    attacker_small_id: u16,
) -> Option<u16> {
    if game.wire.is_unit_disabled(crate::core::schemas::unit_type::TRANSPORT) {
        return None;
    }
    if game.unit_count(attacker_small_id, crate::core::schemas::unit_type::TRANSPORT)
        >= game.wire.boat_max_number()
    {
        return None;
    }
    let has_shore = {
        let mut found = false;
        game.for_each_border_tile(attacker_small_id, |t| {
            if game.is_shore(t) {
                found = true;
            }
        });
        found
    };
    if !has_shore {
        return None;
    }

    let attacker_troops = game.player_by_small_id(attacker_small_id)?.troops;
    let is_ffa = game.wire.game_config().game_mode != "Team";
    let attacker_center = player_center_tile(game, attacker_small_id)?;
    let mut sorted_players: Vec<(u16, u32)> = game
        .players_in_order()
        .iter()
        .filter(|p| {
            p.small_id != attacker_small_id
                && p.alive
                && !game.is_friendly(attacker_small_id, p.small_id)
                && (!is_ffa || p.troops < attacker_troops)
        })
        .filter_map(|p| {
            let center = player_center_tile(game, p.small_id)?;
            Some((p.small_id, game.manhattan_dist(attacker_center, center)))
        })
        .collect();
    sorted_players.sort_by_key(|(_, distance)| *distance);

    let attacker_shores = shore_border_tiles(game, attacker_small_id);
    let mut reachable_players = Vec::with_capacity(2);
    for (target_sid, _) in sorted_players {
        let target_shores = shore_border_tiles(game, target_sid);
        let Some((_src, dst)) = closest_two_tiles(game, &attacker_shores, &target_shores) else {
            continue;
        };
        if can_build_transport_ship(game, attacker_small_id, dst).is_none() {
            continue;
        }
        reachable_players.push(target_sid);
        if reachable_players.len() >= 2 {
            break;
        }
    }
    if reachable_players.is_empty() {
        return None;
    }
    if reachable_players.len() >= 2 && random.chance(3) {
        Some(reachable_players[1])
    } else {
        Some(reachable_players[0])
    }
}

fn nation_try_attack_player(
    game: &mut Game,
    random: &mut PseudoRandom,
    attacker_small_id: u16,
    target_small_id: u16,
    reserve_ratio: f64,
    expand_ratio: f64,
    bot_attack_troops_sent: &mut f64,
    difficulty: &str,
    emoji: Option<&mut super::nation_emoji::NationEmojiState>,
    force: bool,
) -> bool {
    match emoji {
        Some(state) => try_send_player_attack_forced(
            game,
            random,
            attacker_small_id,
            target_small_id,
            reserve_ratio,
            expand_ratio,
            bot_attack_troops_sent,
            difficulty,
            Some(state),
            force,
        ),
        None => try_send_player_attack_forced(
            game,
            random,
            attacker_small_id,
            target_small_id,
            reserve_ratio,
            expand_ratio,
            bot_attack_troops_sent,
            difficulty,
            None,
            force,
        ),
    }
}

fn should_attack(
    game: &Game,
    random: &mut PseudoRandom,
    attacker_small_id: u16,
    target_small_id: u16,
    force: bool,
) -> bool {
    if force {
        return true;
    }
    let Some(target) = game.player_by_small_id(target_small_id) else {
        return true;
    };
    let Some(attacker) = game.player_by_small_id(attacker_small_id) else {
        return false;
    };
    if target.player_type != PlayerType::Human
        || game.is_traitor(target_small_id)
        || attacker.player_type == PlayerType::Bot
    {
        return true;
    }
    let difficulty = game.wire.game_config().difficulty.as_str();
    match difficulty {
        "Easy" => random.next_int(0, 4) == 0,
        "Medium" => !random.chance(4),
        _ => true,
    }
}

fn nation_attack_best_target(
    game: &mut Game,
    random: &mut PseudoRandom,
    attacker_small_id: u16,
    trigger_ratio: f64,
    reserve_ratio: f64,
    expand_ratio: f64,
    bot_attack_troops_sent: &mut f64,
    difficulty: &str,
    bordering: &[u16],
    mut emoji: Option<&mut super::nation_emoji::NationEmojiState>,
) -> bool {
    if has_neighboring_bot_with_structures(game, attacker_small_id) {
        if attack_bots(
            game,
            random,
            attacker_small_id,
            reserve_ratio,
            expand_ratio,
            bot_attack_troops_sent,
            difficulty,
        ) {
            return true;
        }
    }

    if !has_reserve_ratio(game, attacker_small_id, reserve_ratio) {
        return false;
    }
    if !has_trigger_ratio(game, attacker_small_id, trigger_ratio) && !random.chance(10) {
        return false;
    }

    match difficulty {
        "Easy" => {
            if is_bordering_nuked_territory(game, attacker_small_id)
                && send_tn_attack(game, attacker_small_id, expand_ratio)
            {
                return true;
            }
            if attack_bots(
                game,
                random,
                attacker_small_id,
                reserve_ratio,
                expand_ratio,
                bot_attack_troops_sent,
                difficulty,
            ) {
                return true;
            }
            if let Some(attacker) = find_incoming_attacker(game, attacker_small_id) {
                return nation_try_attack_player(
                    game,
                    random,
                    attacker_small_id,
                    attacker,
                    reserve_ratio,
                    expand_ratio,
                    bot_attack_troops_sent,
                    difficulty,
                    emoji.as_deref_mut(),
                    true,
                );
            }
            // TS Easy order is [nuked, bots, retaliate, assist, betray, hated,
            // weakest]. `assist` (TS `AiAttackBehavior.assistAllies`, ally
            // target-following) has no native port yet - see DEVLOG; the
            // remaining strategies were previously missing from this arm
            // entirely (this file already implements `betray`/`hated`, used
            // by the Medium arm below, just never wired in here), which made
            // Easy nations skip straight from `retaliate` to `weakest`
            // whenever TS would have taken a `betray`/`hated` branch instead
            // - a real behavior divergence, not just an unlikely edge case
            // (root-caused via a bots=5 curriculum-parity bisection, see
            // docs/bot-ai-parity-*/ for the methodology).
            if nation_strategy_betray(
                game,
                random,
                attacker_small_id,
                reserve_ratio,
                expand_ratio,
                bot_attack_troops_sent,
                difficulty,
                bordering,
                emoji.as_deref_mut(),
            ) {
                return true;
            }
            if nation_strategy_hated(
                game,
                random,
                attacker_small_id,
                reserve_ratio,
                expand_ratio,
                bot_attack_troops_sent,
                difficulty,
                bordering,
                emoji.as_deref_mut(),
            ) {
                return true;
            }
            nation_strategy_weakest(
                game,
                random,
                attacker_small_id,
                reserve_ratio,
                expand_ratio,
                bot_attack_troops_sent,
                difficulty,
                bordering,
                emoji.as_deref_mut(),
            )
        }
        "Medium" => {
            if attack_bots(
                game,
                random,
                attacker_small_id,
                reserve_ratio,
                expand_ratio,
                bot_attack_troops_sent,
                difficulty,
            ) {
                return true;
            }
            if is_bordering_nuked_territory(game, attacker_small_id)
                && send_tn_attack(game, attacker_small_id, expand_ratio)
            {
                return true;
            }
            if let Some(attacker) = find_incoming_attacker(game, attacker_small_id) {
                return nation_try_attack_player(
                    game,
                    random,
                    attacker_small_id,
                    attacker,
                    reserve_ratio,
                    expand_ratio,
                    bot_attack_troops_sent,
                    difficulty,
                    emoji.as_deref_mut(),
                    true,
                );
            }
            if nation_strategy_betray(
                game,
                random,
                attacker_small_id,
                reserve_ratio,
                expand_ratio,
                bot_attack_troops_sent,
                difficulty,
                bordering,
                emoji.as_deref_mut(),
            ) {
                return true;
            }
            if nation_strategy_hated(
                game,
                random,
                attacker_small_id,
                reserve_ratio,
                expand_ratio,
                bot_attack_troops_sent,
                difficulty,
                bordering,
                emoji.as_deref_mut(),
            ) {
                return true;
            }
            if nation_strategy_afk(
                game,
                random,
                attacker_small_id,
                reserve_ratio,
                expand_ratio,
                bot_attack_troops_sent,
                difficulty,
                bordering,
                emoji.as_deref_mut(),
            ) {
                return true;
            }
            if nation_strategy_traitor(
                game,
                random,
                attacker_small_id,
                reserve_ratio,
                expand_ratio,
                bot_attack_troops_sent,
                difficulty,
                bordering,
                emoji.as_deref_mut(),
            ) {
                return true;
            }
            if nation_strategy_weakest(
                game,
                random,
                attacker_small_id,
                reserve_ratio,
                expand_ratio,
                bot_attack_troops_sent,
                difficulty,
                bordering,
                emoji.as_deref_mut(),
            ) {
                return true;
            }
            nation_strategy_island(
                game,
                random,
                attacker_small_id,
                reserve_ratio,
                expand_ratio,
                bot_attack_troops_sent,
                difficulty,
                bordering,
                emoji.as_deref_mut(),
            )
        }
        _ => {
            if attack_bots(
                game,
                random,
                attacker_small_id,
                reserve_ratio,
                expand_ratio,
                bot_attack_troops_sent,
                difficulty,
            ) {
                return true;
            }
            if let Some(attacker) = find_incoming_attacker(game, attacker_small_id) {
                return nation_try_attack_player(
                    game,
                    random,
                    attacker_small_id,
                    attacker,
                    reserve_ratio,
                    expand_ratio,
                    bot_attack_troops_sent,
                    difficulty,
                    emoji.as_deref_mut(),
                    true,
                );
            }
            nation_strategy_weakest(
                game,
                random,
                attacker_small_id,
                reserve_ratio,
                expand_ratio,
                bot_attack_troops_sent,
                difficulty,
                bordering,
                emoji.as_deref_mut(),
            )
        }
    }
}

fn nation_strategy_weakest(
    game: &mut Game,
    random: &mut PseudoRandom,
    sid: u16,
    reserve_ratio: f64,
    expand_ratio: f64,
    bot_attack_troops_sent: &mut f64,
    difficulty: &str,
    bordering: &[u16],
    emoji: Option<&mut super::nation_emoji::NationEmojiState>,
) -> bool {
    if let Some(&weakest) = bordering.first() {
        let attacker_troops = game.player_by_small_id(sid).map(|p| p.troops).unwrap_or(0);
        let target_troops = game
            .player_by_small_id(weakest)
            .map(|p| p.troops)
            .unwrap_or(0);
        if target_troops < attacker_troops {
            return nation_try_attack_player(
                game,
                random,
                sid,
                weakest,
                reserve_ratio,
                expand_ratio,
                bot_attack_troops_sent,
                difficulty,
                emoji,
                false,
            );
        }
    }
    false
}

fn nation_strategy_betray(
    game: &mut Game,
    random: &mut PseudoRandom,
    sid: u16,
    reserve_ratio: f64,
    expand_ratio: f64,
    bot_attack_troops_sent: &mut f64,
    difficulty: &str,
    bordering_enemies: &[u16],
    emoji: Option<&mut super::nation_emoji::NationEmojiState>,
) -> bool {
    let friends = bordering_friends_by_troops(game, sid);
    let bordering_count = bordering_enemies.len() + friends.len();
    for &friend in &friends {
        if super::nation_alliance::maybe_betray(game, random, sid, friend, bordering_count) {
            return nation_try_attack_player(
                game,
                random,
                sid,
                friend,
                reserve_ratio,
                expand_ratio,
                bot_attack_troops_sent,
                difficulty,
                emoji,
                true,
            );
        }
    }
    false
}

fn nation_strategy_hated(
    game: &mut Game,
    random: &mut PseudoRandom,
    sid: u16,
    reserve_ratio: f64,
    expand_ratio: f64,
    bot_attack_troops_sent: &mut f64,
    difficulty: &str,
    _: &[u16],
    emoji: Option<&mut super::nation_emoji::NationEmojiState>,
) -> bool {
    let Some(attacker) = game.player_by_small_id(sid) else {
        return false;
    };
    let attacker_troops = attacker.troops;
    // TS `hated`: `if (this.isFFA() && other.troops() > this.player.troops() * 3) continue;`
    // - the troop-cap guard only applies in FFA, not team games.
    let is_ffa = game.wire.game_config().game_mode != "Team";
    // TS `PlayerImpl.allRelationsSorted()`: stable sort by relation value, tie-broken by
    // insertion order (and filtered to alive players) - not `HashMap` iteration order.
    for (other, _) in game.all_relations_sorted(sid) {
        if game.relation(sid, other) != Relation::Hostile {
            continue;
        }
        if game.is_friendly(sid, other) {
            continue;
        }
        if is_ffa {
            if let Some(p) = game.player_by_small_id(other) {
                if p.troops > attacker_troops * 3 {
                    continue;
                }
            }
        }
        return nation_try_attack_player(
            game,
            random,
            sid,
            other,
            reserve_ratio,
            expand_ratio,
            bot_attack_troops_sent,
            difficulty,
            emoji,
            false,
        );
    }
    false
}

fn nation_strategy_afk(
    game: &mut Game,
    random: &mut PseudoRandom,
    sid: u16,
    reserve_ratio: f64,
    expand_ratio: f64,
    bot_attack_troops_sent: &mut f64,
    difficulty: &str,
    bordering: &[u16],
    emoji: Option<&mut super::nation_emoji::NationEmojiState>,
) -> bool {
    let attacker_troops = game.player_by_small_id(sid).map(|p| p.troops).unwrap_or(0);
    for &enemy in bordering {
        let Some(p) = game.player_by_small_id(enemy) else {
            continue;
        };
        if !p.is_disconnected {
            continue;
        }
        if p.troops >= attacker_troops * 3 {
            continue;
        }
        return nation_try_attack_player(
            game,
            random,
            sid,
            enemy,
            reserve_ratio,
            expand_ratio,
            bot_attack_troops_sent,
            difficulty,
            emoji,
            false,
        );
    }
    false
}

fn nation_strategy_traitor(
    game: &mut Game,
    random: &mut PseudoRandom,
    sid: u16,
    reserve_ratio: f64,
    expand_ratio: f64,
    bot_attack_troops_sent: &mut f64,
    difficulty: &str,
    bordering: &[u16],
    emoji: Option<&mut super::nation_emoji::NationEmojiState>,
) -> bool {
    let attacker_troops = game.player_by_small_id(sid).map(|p| p.troops).unwrap_or(0);
    for &enemy in bordering {
        if !game.is_traitor(enemy) {
            continue;
        }
        if let Some(p) = game.player_by_small_id(enemy) {
            if p.troops >= (attacker_troops as f64 * 1.2) as i32 {
                continue;
            }
        }
        return nation_try_attack_player(
            game,
            random,
            sid,
            enemy,
            reserve_ratio,
            expand_ratio,
            bot_attack_troops_sent,
            difficulty,
            emoji,
            false,
        );
    }
    false
}

fn nation_strategy_island(
    game: &mut Game,
    random: &mut PseudoRandom,
    sid: u16,
    reserve_ratio: f64,
    expand_ratio: f64,
    bot_attack_troops_sent: &mut f64,
    difficulty: &str,
    bordering: &[u16],
    emoji: Option<&mut super::nation_emoji::NationEmojiState>,
) -> bool {
    if !bordering.is_empty() {
        return false;
    }
    if let Some(enemy) = find_nearest_island_enemy(game, random, sid) {
        return nation_try_attack_player(
            game,
            random,
            sid,
            enemy,
            reserve_ratio,
            expand_ratio,
            bot_attack_troops_sent,
            difficulty,
            emoji,
            false,
        );
    }
    false
}

fn has_nearby_terra_nullius(game: &Game, small_id: u16) -> bool {
    if has_land_border_tn(game, small_id) {
        return true;
    }
    has_shore_reachable_tn(game, small_id)
}

fn random_boat_attack_troops(game: &Game, attacker_small_id: u16, target_small_id: u16) -> f64 {
    let Some(attacker) = game.player_by_small_id(attacker_small_id) else {
        return 0.0;
    };
    let mut troops = attacker.troops as f64 / 5.0;
    if target_small_id == game.terra_nullius_id()
        || attacker.player_type == PlayerType::Bot
        || game.wire.game_config().game_mode == "Team"
    {
        return troops;
    }

    let retain_fraction = match game.wire.game_config().difficulty.as_str() {
        "Hard" => 0.75,
        "Impossible" => 0.9,
        _ => return troops,
    };
    let max_neighbor_troops = nearby_players_ts_order(game, attacker_small_id)
        .into_iter()
        .filter_map(|sid| game.player_by_small_id(sid))
        .filter(|p| {
            p.player_type != PlayerType::Bot && !game.is_friendly(attacker_small_id, p.small_id)
        })
        .map(|p| p.troops)
        .max()
        .unwrap_or(0);
    if max_neighbor_troops > 0 {
        let min_retained = (max_neighbor_troops as f64 * retain_fraction).ceil();
        troops = troops.min((attacker.troops as f64 - min_retained).max(0.0));
    }

    let incoming = game.incoming_land_troops(attacker_small_id);
    if incoming > 0.0 {
        troops = troops.max(incoming.min(attacker.troops as f64 / 5.0));
    } else if game
        .player_by_small_id(target_small_id)
        .is_some_and(|target| troops < target.troops as f64 * 0.2)
    {
        return 0.0;
    }
    troops
}

fn attack_with_random_boat(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    bordering_enemies: &[u16],
) -> bool {
    if game
        .wire
        .is_unit_disabled(crate::core::schemas::unit_type::TRANSPORT)
    {
        return false;
    }
    if game.unit_count(small_id, crate::core::schemas::unit_type::TRANSPORT)
        >= game.wire.boat_max_number()
    {
        return false;
    }

    let shores: Vec<TileRef> = shore_border_tiles(game, small_id);
    if shores.is_empty() {
        return false;
    }
    let src = shores[random.next_int(0, shores.len() as i32) as usize];

    // High-interest: unowned land, then player targets.
    for high_interest in [true, false] {
        let mut unreachable_players = HashSet::new();
        for _ in 0..500 {
            let bx = game.x(src) as i32;
            let by = game.y(src) as i32;
            let rx = random.next_int(bx - 150, bx + 150);
            let ry = random.next_int(by - 150, by + 150);
            if !game.is_valid_coord(rx, ry) {
                continue;
            }
            let tile = game.ref_xy(rx as u32, ry as u32);
            if !game.is_land(tile) {
                continue;
            }
            let owner = game.map.owner_id(tile);
            if owner == small_id {
                continue;
            }
            if owner != game.terra_nullius_id() {
                if unreachable_players.contains(&owner) || bordering_enemies.contains(&owner) {
                    continue;
                }
                if game.wire.game_config().game_mode != "Team"
                    && game.player_by_small_id(owner).is_some_and(|target| {
                        game.player_by_small_id(small_id)
                            .is_some_and(|attacker| target.troops > attacker.troops)
                    })
                {
                    continue;
                }
            }
            if high_interest {
                if owner != 0
                    && game
                        .player_by_small_id(owner)
                        .is_none_or(|p| p.player_type != PlayerType::Bot)
                {
                    continue;
                }
            } else if owner != game.terra_nullius_id() && game.is_friendly(small_id, owner) {
                continue;
            }
            if can_build_transport_ship(game, small_id, tile).is_none() {
                if owner != game.terra_nullius_id() {
                    unreachable_players.insert(owner);
                }
                continue;
            }
            let troops = random_boat_attack_troops(game, small_id, owner);
            if troops < 1.0 {
                return false;
            }
            game.add_transport_attack(small_id, tile, troops);
            return true;
        }
    }

    if !bordering_enemies.is_empty() {
        let idx = random.next_int(0, bordering_enemies.len() as i32) as usize;
        let target = bordering_enemies[idx];
        let troops = game
            .player_by_small_id(small_id)
            .map(|p| game.wire.boat_attack_amount(p.troops))
            .unwrap_or(0.0);
        return send_boat_attack_to_player(game, small_id, target, troops);
    }
    false
}

fn bot_attack_max_parallelism(random: &mut PseudoRandom, difficulty: &str) -> usize {
    match difficulty {
        "Easy" => 1,
        "Medium" => {
            if random.chance(2) {
                1
            } else {
                2
            }
        }
        "Hard" => 3,
        "Impossible" => 100,
        _ => 2,
    }
}

fn nation_bot_attack_troops(
    game: &Game,
    attacker_small_id: u16,
    target_small_id: u16,
    reserve_ratio: f64,
    bot_attack_troops_sent: f64,
    difficulty: &str,
) -> Option<f64> {
    let attacker = game.player_by_small_id(attacker_small_id)?;
    let target = game.player_by_small_id(target_small_id)?;
    let max_troops = game.max_troops_for(attacker.small_id);
    let target_reserve = max_troops * reserve_ratio;
    let max_send = attacker.troops as f64 - target_reserve - bot_attack_troops_sent;
    if max_send < 1.0 {
        return None;
    }
    if difficulty == "Easy" {
        return Some(max_send);
    }
    let mut troops = target.troops as f64 * 4.0;
    if troops > max_send {
        if max_send < target.troops as f64 * 2.0 {
            return None;
        }
        troops = max_send;
    }
    if troops < 1.0 {
        return None;
    }
    Some(troops)
}

fn try_send_nation_bot_attack(
    game: &mut Game,
    attacker_small_id: u16,
    target_small_id: u16,
    reserve_ratio: f64,
    bot_attack_troops_sent: &mut f64,
    difficulty: &str,
) -> bool {
    if game.shares_land_border_with(attacker_small_id, target_small_id) {
        let Some(troops) = nation_bot_attack_troops(
            game,
            attacker_small_id,
            target_small_id,
            reserve_ratio,
            *bot_attack_troops_sent,
            difficulty,
        ) else {
            return false;
        };
        let target_id = game.player_by_small_id(target_small_id).unwrap().id.clone();
        game.add_land_attack(attacker_small_id, Some(target_id), Some(troops));
        *bot_attack_troops_sent += troops;
        return true;
    }
    let Some(troops) = nation_bot_attack_troops(
        game,
        attacker_small_id,
        target_small_id,
        reserve_ratio,
        *bot_attack_troops_sent,
        difficulty,
    ) else {
        return false;
    };
    if send_boat_attack_to_player(
        game,
        attacker_small_id,
        target_small_id,
        troops,
    ) {
        *bot_attack_troops_sent += troops;
        return true;
    }
    false
}

fn attack_bots(
    game: &mut Game,
    random: &mut PseudoRandom,
    attacker_small_id: u16,
    reserve_ratio: f64,
    expand_ratio: f64,
    bot_attack_troops_sent: &mut f64,
    difficulty: &str,
) -> bool {
    // TS `attackBots`: source list is `this.player.nearby()` (Set insertion
    // order), not a numerically-sorted id list - the stable sort below uses
    // that order as its tie-break, so this must use `nearby_players_ts_order`
    // (not `nearby_player_small_ids`, which sorts by raw id) or ties resolve
    // to a different bot than TS picks.
    let bots: Vec<u16> = nearby_players_ts_order(game, attacker_small_id)
        .into_iter()
        .filter(|&sid| {
            game.player_by_small_id(sid).is_some_and(|p| {
                p.player_type == PlayerType::Bot && !game.is_friendly(attacker_small_id, sid)
            })
        })
        .collect();
    if bots.is_empty() {
        return false;
    }

    *bot_attack_troops_sent = 0.0;
    let mut sorted = bots;
    // TS `attackBots`: primary key is `ownsStructures` (structure-owning bots
    // sorted first), density (troops/tiles) ascending only breaks ties within
    // the same hasStructures group. `sort_by` is stable, matching JS's
    // guaranteed-stable `Array.prototype.sort`.
    sorted.sort_by(|&a, &b| {
        let a_has_structures = player_has_structure_units(game, a);
        let b_has_structures = player_has_structure_units(game, b);
        if a_has_structures != b_has_structures {
            return if a_has_structures {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            };
        }
        let da = {
            let p = game.player_by_small_id(a).unwrap();
            p.troops as f64 / p.tiles_owned.max(1) as f64
        };
        let db = {
            let p = game.player_by_small_id(b).unwrap();
            p.troops as f64 / p.tiles_owned.max(1) as f64
        };
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });

    let parallelism = bot_attack_max_parallelism(random, difficulty);
    let mut sent = false;
    for target in sorted.into_iter().take(parallelism) {
        // TS `calculateAttackTroops`: keep less in reserve (use `expandRatio`)
        // when the bot target owns structures, so we recapture them ASAP.
        let target_reserve_ratio = if player_has_structure_units(game, target) {
            expand_ratio
        } else {
            reserve_ratio
        };
        if try_send_nation_bot_attack(
            game,
            attacker_small_id,
            target,
            target_reserve_ratio,
            bot_attack_troops_sent,
            difficulty,
        ) {
            sent = true;
        }
    }
    sent
}

pub fn nation_maybe_attack(
    game: &mut Game,
    random: &mut PseudoRandom,
    attacker_small_id: u16,
    trigger_ratio: f64,
    reserve_ratio: f64,
    expand_ratio: f64,
    bot_attack_troops_sent: &mut f64,
    difficulty: &str,
    mut emoji: Option<&mut super::nation_emoji::NationEmojiState>,
) {
    if has_non_nuked_tn(game, attacker_small_id) {
        if send_tn_attack(game, attacker_small_id, expand_ratio) {
            return;
        }
    }

    let bordering = bordering_enemies_by_troops(game, attacker_small_id);
    let mut skip_attack_best = false;
    if bordering.is_empty() {
        if random.chance(5) {
            attack_with_random_boat(game, random, attacker_small_id, &bordering);
        }
    } else if random.chance(10) {
        attack_with_random_boat(game, random, attacker_small_id, &bordering);
        skip_attack_best = true;
    }

    if !skip_attack_best {
        if let Some(emoji_state) = emoji.as_mut() {
            if !bordering.is_empty() {
                super::nation_alliance::maybe_send_alliance_requests(
                    game,
                    random,
                    attacker_small_id,
                    &bordering,
                    emoji_state,
                );
            }
            if nation_attack_best_target(
                game,
                random,
                attacker_small_id,
                trigger_ratio,
                reserve_ratio,
                expand_ratio,
                bot_attack_troops_sent,
                difficulty,
                &bordering,
                Some(emoji_state),
            ) {
                return;
            }
        } else if nation_attack_best_target(
            game,
            random,
            attacker_small_id,
            trigger_ratio,
            reserve_ratio,
            expand_ratio,
            bot_attack_troops_sent,
            difficulty,
            &bordering,
            None,
        ) {
            return;
        }
    }
}

/// TS `AiAttackBehavior.getNeighborTraitorToAttack()`: a random traitor among
/// non-friendly nearby players (in `PlayerImpl.nearby()`/Set-insertion order).
fn get_neighbor_traitor_to_attack(
    game: &Game,
    random: &mut PseudoRandom,
    small_id: u16,
) -> Option<u16> {
    if game.wire.disable_alliances() {
        return None;
    }
    let traitors: Vec<u16> = nearby_players_ts_order(game, small_id)
        .into_iter()
        .filter(|&sid| !game.is_friendly(small_id, sid) && game.is_traitor(sid))
        .collect();
    random.rand_element(&traitors)
}

pub fn tribe_maybe_attack(
    game: &mut Game,
    random: &mut PseudoRandom,
    attacker_small_id: u16,
    trigger_ratio: f64,
    reserve_ratio: f64,
    expand_ratio: f64,
    neighbors_terra_nullius: &mut bool,
) {
    // Tribe attackers are always Bot-typed, so the target-is-bot special
    // casing in `try_send_player_attack(_forced)` never triggers here (TS
    // `calculateAttackTroops` only takes that branch when the attacker isn't
    // a Bot) - `bot_attack_troops_sent`/`difficulty` are unused placeholders.
    let mut bot_attack_troops_sent = 0.0;
    let difficulty_owned = game.wire.game_config().difficulty.clone();
    let difficulty = difficulty_owned.as_str();

    // TS `TribeExecution.maybeAttack()`: roll a traitor-neighbor attack first.
    // Odds are 1/6 if we're (still) allied with the traitor, 1/3 otherwise; on
    // success the alliance (if any) is broken before the attack is sent.
    if let Some(traitor) = get_neighbor_traitor_to_attack(game, random, attacker_small_id) {
        let odds = if game.is_friendly(attacker_small_id, traitor) { 6 } else { 3 };
        if random.chance(odds) {
            game.break_alliance_between(attacker_small_id, traitor);
            if try_send_player_attack(
                game,
                random,
                attacker_small_id,
                traitor,
                reserve_ratio,
                expand_ratio,
                &mut bot_attack_troops_sent,
                difficulty,
                None,
            ) {
                return;
            }
        }
    }

    if *neighbors_terra_nullius {
        if has_nearby_terra_nullius(game, attacker_small_id) {
            if send_tn_attack(game, attacker_small_id, expand_ratio) {
                return;
            }
        } else {
            *neighbors_terra_nullius = false;
        }
    }

    // TS `AiAttackBehavior.attackRandomTarget()`: trigger-ratio gate first, then
    // retaliation against the largest incoming attacker, then another traitor
    // roll (odds 1/3, unconditional on alliance), then a random shuffled pick.
    if !has_trigger_ratio(game, attacker_small_id, trigger_ratio) {
        return;
    }

    if let Some(attacker) = find_incoming_attacker(game, attacker_small_id) {
        if try_send_player_attack_forced(
            game,
            random,
            attacker_small_id,
            attacker,
            reserve_ratio,
            expand_ratio,
            &mut bot_attack_troops_sent,
            difficulty,
            None,
            true,
        ) {
            return;
        }
    }

    if let Some(traitor) = get_neighbor_traitor_to_attack(game, random, attacker_small_id) {
        if random.chance(3) {
            if try_send_player_attack(
                game,
                random,
                attacker_small_id,
                traitor,
                reserve_ratio,
                expand_ratio,
                &mut bot_attack_troops_sent,
                difficulty,
                None,
            ) {
                return;
            }
        }
    }

    let neighbors = nearby_players_ts_order(game, attacker_small_id);
    let shuffled = random.shuffle_array(&neighbors);
    for target_sid in shuffled {
        let Some(target) = game.player_by_small_id(target_sid) else {
            continue;
        };
        if game.is_friendly(attacker_small_id, target_sid) {
            continue;
        }
        if target.player_type == PlayerType::Nation || target.player_type == PlayerType::Human {
            if random.chance(2) {
                continue;
            }
        }
        if try_send_player_attack(
            game,
            random,
            attacker_small_id,
            target_sid,
            reserve_ratio,
            expand_ratio,
            &mut bot_attack_troops_sent,
            difficulty,
            None,
        ) {
            return;
        }
    }
}
