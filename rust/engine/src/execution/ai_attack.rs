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

/// TS `AiAttackBehavior.troopSendCap()`: for Hard/Impossible-difficulty
/// non-Bot players in FFA (never for Bots or Team games), caps troops sent
/// in ANY attack against a player - land, boat, or bot-target - so at least
/// `retainFraction` (Hard: 75%, Impossible: 90%) of the strongest
/// non-allied, non-Bot neighbor's current troops remain uncommitted. If the
/// player has active incoming (land) attacks, the cap is raised to at least
/// the total incoming troop count so retaliation is never blocked by it.
/// This whole mechanic, plus `is_attack_too_weak` below, was entirely
/// unported natively - `AiAttackBehavior.test.ts`'s "Hard/Impossible troop
/// floor" describe block (8 tests) caught it.
fn troop_send_cap(game: &Game, small_id: u16) -> f64 {
    let Some(attacker) = game.player_by_small_id(small_id) else {
        return f64::INFINITY;
    };
    if attacker.player_type == PlayerType::Bot {
        return f64::INFINITY;
    }
    if game.wire.game_config().game_mode == "Team" {
        return f64::INFINITY;
    }
    let retain_fraction = match game.wire.game_config().difficulty.as_str() {
        "Hard" => 0.75,
        "Impossible" => 0.9,
        _ => return f64::INFINITY,
    };

    let mut max_neighbor_troops: i32 = 0;
    for sid in nearby_players_ts_order(game, small_id) {
        let Some(p) = game.player_by_small_id(sid) else {
            continue;
        };
        if p.player_type == PlayerType::Bot {
            continue;
        }
        if game.is_friendly(small_id, sid) {
            continue;
        }
        if p.troops > max_neighbor_troops {
            max_neighbor_troops = p.troops;
        }
    }

    let mut cap = if max_neighbor_troops == 0 {
        f64::INFINITY
    } else {
        let min_retained = (max_neighbor_troops as f64 * retain_fraction).ceil();
        (attacker.troops as f64 - min_retained).max(0.0)
    };

    let incoming = game.incoming_attacks(small_id, false);
    if !incoming.is_empty() {
        let total_incoming: f64 = incoming.iter().map(|a| a.troops).sum();
        cap = cap.max(total_incoming);
    }
    cap
}

/// TS `AiAttackBehavior.isAttackTooWeak()`: on Hard/Impossible in FFA,
/// blocks a player-targeted attack that would send less than 20% of the
/// target's troops. Bots, Team games, and attackers already under incoming
/// attack (who may retaliate freely regardless of size) are exempt.
fn is_attack_too_weak(game: &Game, small_id: u16, troops: f64, target_small_id: u16) -> bool {
    let Some(attacker) = game.player_by_small_id(small_id) else {
        return false;
    };
    if attacker.player_type == PlayerType::Bot {
        return false;
    }
    if game.wire.game_config().game_mode == "Team" {
        return false;
    }
    if !game.incoming_attacks(small_id, false).is_empty() {
        return false;
    }
    let difficulty = game.wire.game_config().difficulty.as_str();
    if difficulty != "Hard" && difficulty != "Impossible" {
        return false;
    }
    let target_troops = game
        .player_by_small_id(target_small_id)
        .map(|p| p.troops)
        .unwrap_or(0);
    troops < target_troops as f64 * 0.2
}

/// TS `AiAttackBehavior.calculateAttackTroops`'s shared post-processing for
/// every player-targeted attack (land, boat, and bot-target alike): apply
/// `troop_send_cap`, then `is_attack_too_weak`. Returns `None` when the
/// attack must be blocked entirely (capped below 1 troop, or too weak).
fn cap_player_attack_troops(
    game: &Game,
    small_id: u16,
    target_small_id: u16,
    troops: f64,
) -> Option<f64> {
    let capped = troops.min(troop_send_cap(game, small_id));
    if capped < 1.0 {
        return None;
    }
    if is_attack_too_weak(game, small_id, capped, target_small_id) {
        return None;
    }
    Some(capped)
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

/// TS `AiAttackBehavior.hasLandBorderWithTerraNullius()`: deliberately does
/// NOT filter fallout, unlike `has_land_border_tn` above (TS's
/// `hasNonNukedTerraNullius` border check, used only to decide whether to
/// even attempt TN expansion at all). This one decides `sendAttack`'s
/// land-vs-boat branch once we already know we're sending a TerraNullius
/// attack (from either the early gate or the `nuked` strategy), so it must
/// say "yes, land-attack it" for a directly-adjacent NUKED tile too -
/// that's the whole point of the `nuked` strategy. `try_send_tn_attack`
/// was using `has_land_border_tn` (fallout-filtered) here, so on a nation
/// entirely surrounded by nuked land with no water to fall back to
/// (`AiAttackBehaviorNukedTerritory.test.ts`'s "idle capture" scenarios),
/// it always fell through to the boat path and found nothing, silently
/// dropping the `nuked` strategy's attack.
fn has_land_border_with_terra_nullius(game: &Game, small_id: u16) -> bool {
    let Some(border) = game.border_tiles_of(small_id) else {
        return false;
    };
    let mut nbuf = [TileRef::MAX; 4];
    for border_tile in border.iter() {
        let n = game.map.neighbors4_ts(border_tile, &mut nbuf);
        for i in 0..n {
            let neighbor = nbuf[i];
            if game.is_land(neighbor) && !game.is_impassable(neighbor) && !game.has_owner(neighbor) {
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
    if has_land_border_with_terra_nullius(game, small_id) {
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
        let Some(troops) =
            cap_player_attack_troops(game, attacker_small_id, target_small_id, troops)
        else {
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
    // TS `sendBoatAttack`'s `nonBotTroops` lambda is `() => this.player.troops()
    // / 5` - unlike `Config.boatAttackAmount()` (`floor(troops / 5)`, used
    // only by `TransportShipExecution`'s human-intent default), this value is
    // NOT floored. Previously used `wire.boat_attack_amount` here, which is
    // the floored config function - a same-class-of-bug rounding mismatch to
    // the `send_boat_attack_to_nearby_tn`/`random_boat_attack_troops`
    // functions in this same file, which already correctly use the unfloored
    // form.
    let boat_troops = game
        .player_by_small_id(attacker_small_id)
        .map(|p| p.troops as f64 / 5.0)
        .unwrap_or(0.0);
    let Some(boat_troops) =
        cap_player_attack_troops(game, attacker_small_id, target_small_id, boat_troops)
    else {
        return false;
    };
    send_boat_attack_to_player(game, attacker_small_id, target_small_id, boat_troops)
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
                let owner = game.map.owner_id(neighbor);
                // TS `PlayerImpl.nearby()`'s direct-neighbor `visit`: an
                // unowned tile that is nuked (fallout) contributes NO slot at
                // all (`return` before `ns.add(...)`), unlike a plain unowned
                // tile, which still adds a TerraNullius slot. Native was
                // pushing `owner` unconditionally here, so a nuked-TN border
                // tile occupied a slot in this order-sensitive list (feeding
                // e.g. `tribe_maybe_attack`'s neighbor shuffle and
                // `troop_send_cap`'s neighbor scan) that TS's fixed `nearby()`
                // (see `AiAttackBehaviorNukedTerritory.test.ts`) omits.
                if owner == 0 && game.has_fallout(neighbor) {
                    continue;
                }
                push(owner);
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

pub(crate) fn find_incoming_attacker(game: &Game, small_id: u16) -> Option<u16> {
    let ptype = game
        .player_by_small_id(small_id)
        .map(|p| p.player_type)
        .unwrap_or(PlayerType::Human);
    game.find_incoming_land_attacker(small_id, ptype)
}

/// TS `findVictim`: `if (this.isFFA() && enemy.troops() > this.player.troops()
/// * 1.2) return false;` guard was missing entirely (same class of bug as
/// `nation_strategy_hated`'s missing FFA guard - see
/// docs/bot-ai-parity-nation-relations/README.md).
fn find_victim(game: &Game, attacker_small_id: u16, bordering: &[u16]) -> Option<u16> {
    let is_ffa = game.wire.game_config().game_mode != "Team";
    let attacker_troops = game.player_by_small_id(attacker_small_id).map(|p| p.troops).unwrap_or(0);
    for &sid in bordering {
        let Some(enemy) = game.player_by_small_id(sid) else {
            continue;
        };
        if is_ffa && enemy.troops as f64 > attacker_troops as f64 * 1.2 {
            continue;
        }
        let incoming = game.incoming_land_troops(sid);
        if incoming > enemy.troops as f64 * 0.5 {
            return Some(sid);
        }
    }
    None
}

/// TS `findVeryWeakEnemy`: `(!this.isFFA() || enemy.troops() <
/// this.player.troops() * 1.2)` guard was missing entirely (same bug class).
fn find_very_weak_enemy(game: &Game, attacker_small_id: u16, bordering: &[u16]) -> Option<u16> {
    let is_ffa = game.wire.game_config().game_mode != "Team";
    let attacker_troops = game.player_by_small_id(attacker_small_id).map(|p| p.troops).unwrap_or(0);
    for &sid in bordering {
        let Some(enemy) = game.player_by_small_id(sid) else {
            continue;
        };
        let max = game.max_troops_for(enemy.small_id);
        if (enemy.troops as f64) < max * 0.15
            && (!is_ffa || (enemy.troops as f64) < attacker_troops as f64 * 1.2)
        {
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

/// TS `AiAttackBehavior.shouldAttack(other)` merged with `sendAttack`'s
/// `if (!force && !this.shouldAttack(target)) return false;` guard (`force`
/// short-circuits, matching TS's caller-side check rather than a
/// `shouldAttack` parameter, since TS's `shouldAttack` itself takes none).
pub(crate) fn should_attack(
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
    // TS also short-circuits to `true` when `gameConfig().playerTeams ===
    // HumansVsNations` - missing here entirely before this fix, so an
    // HvN-mode nation would incorrectly roll Easy/Medium's "skip attacking
    // humans" dice against a mode where TS always attacks.
    let is_humans_vs_nations = game
        .wire
        .game_config()
        .player_teams
        .as_ref()
        .is_some_and(|c| c.is_humans_vs_nations());
    if target.player_type != PlayerType::Human
        || game.is_traitor(target_small_id)
        || attacker.player_type == PlayerType::Bot
        || is_humans_vs_nations
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
        // TS Hard order (minus `assist`, dead code - see the Easy arm's
        // comment - and `donate`, live only in `GameMode.Team`, not ported,
        // see docs/bot-ai-parity-nation-relations/README.md's follow-up
        // list): [bots, retaliate, betray, nuked, traitor, afk, hated,
        // veryWeak, victim, weakest, island]. Previously this whole branch
        // (shared with Impossible below, as a single `_` catch-all) only
        // implemented [bots, retaliate, weakest] - 8 of 11 strategies were
        // silently missing, found via a systematic audit of this match
        // after the Easy-arm bug (see docs/bot-ai-parity-nation-relations/).
        "Hard" => {
            if attack_bots(game, random, attacker_small_id, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty) {
                return true;
            }
            if let Some(attacker) = find_incoming_attacker(game, attacker_small_id) {
                return nation_try_attack_player(game, random, attacker_small_id, attacker, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty, emoji.as_deref_mut(), true);
            }
            if nation_strategy_betray(game, random, attacker_small_id, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty, bordering, emoji.as_deref_mut()) {
                return true;
            }
            if is_bordering_nuked_territory(game, attacker_small_id) && send_tn_attack(game, attacker_small_id, expand_ratio) {
                return true;
            }
            if nation_strategy_traitor(game, random, attacker_small_id, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty, bordering, emoji.as_deref_mut()) {
                return true;
            }
            if nation_strategy_afk(game, random, attacker_small_id, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty, bordering, emoji.as_deref_mut()) {
                return true;
            }
            if nation_strategy_hated(game, random, attacker_small_id, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty, bordering, emoji.as_deref_mut()) {
                return true;
            }
            if nation_strategy_very_weak(game, random, attacker_small_id, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty, bordering, emoji.as_deref_mut()) {
                return true;
            }
            if nation_strategy_victim(game, random, attacker_small_id, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty, bordering, emoji.as_deref_mut()) {
                return true;
            }
            if nation_strategy_weakest(game, random, attacker_small_id, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty, bordering, emoji.as_deref_mut()) {
                return true;
            }
            nation_strategy_island(game, random, attacker_small_id, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty, bordering, emoji.as_deref_mut())
        }
        // TS Impossible order (same dead-code exclusions as Hard above):
        // [retaliate, bots, veryWeak, traitor, afk, betray, victim, nuked,
        // hated, weakest, island]. Note this order genuinely differs from
        // Hard's (retaliate before bots; veryWeak much earlier) - it is not
        // just Hard with a different set, so it needs its own arm rather
        // than falling back to a shared default.
        _ => {
            if let Some(attacker) = find_incoming_attacker(game, attacker_small_id) {
                return nation_try_attack_player(game, random, attacker_small_id, attacker, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty, emoji.as_deref_mut(), true);
            }
            if attack_bots(game, random, attacker_small_id, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty) {
                return true;
            }
            if nation_strategy_very_weak(game, random, attacker_small_id, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty, bordering, emoji.as_deref_mut()) {
                return true;
            }
            if nation_strategy_traitor(game, random, attacker_small_id, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty, bordering, emoji.as_deref_mut()) {
                return true;
            }
            if nation_strategy_afk(game, random, attacker_small_id, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty, bordering, emoji.as_deref_mut()) {
                return true;
            }
            if nation_strategy_betray(game, random, attacker_small_id, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty, bordering, emoji.as_deref_mut()) {
                return true;
            }
            if nation_strategy_victim(game, random, attacker_small_id, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty, bordering, emoji.as_deref_mut()) {
                return true;
            }
            if is_bordering_nuked_territory(game, attacker_small_id) && send_tn_attack(game, attacker_small_id, expand_ratio) {
                return true;
            }
            if nation_strategy_hated(game, random, attacker_small_id, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty, bordering, emoji.as_deref_mut()) {
                return true;
            }
            if nation_strategy_weakest(game, random, attacker_small_id, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty, bordering, emoji.as_deref_mut()) {
                return true;
            }
            nation_strategy_island(game, random, attacker_small_id, reserve_ratio, expand_ratio, bot_attack_troops_sent, difficulty, bordering, emoji.as_deref_mut())
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
    // TS `afk`: `!this.isFFA() || enemy.troops() < this.player.troops() * 3`
    // - the troop-cap guard only applies in FFA, not team games (same bug
    // class as `nation_strategy_hated`'s previously-missing guard).
    let is_ffa = game.wire.game_config().game_mode != "Team";
    let attacker_troops = game.player_by_small_id(sid).map(|p| p.troops).unwrap_or(0);
    for &enemy in bordering {
        let Some(p) = game.player_by_small_id(enemy) else {
            continue;
        };
        if !p.is_disconnected {
            continue;
        }
        if is_ffa && p.troops >= attacker_troops * 3 {
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
    // TS `findTraitor`: `!this.isFFA() || enemy.troops() < this.player.troops() * 1.2`
    // - FFA-only guard, same bug class as above.
    let is_ffa = game.wire.game_config().game_mode != "Team";
    let attacker_troops = game.player_by_small_id(sid).map(|p| p.troops).unwrap_or(0);
    for &enemy in bordering {
        if !game.is_traitor(enemy) {
            continue;
        }
        if is_ffa {
            if let Some(p) = game.player_by_small_id(enemy) {
                if p.troops >= (attacker_troops as f64 * 1.2) as i32 {
                    continue;
                }
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

/// TS `getAttackStrategies`' `veryWeak` closure (Hard/Impossible only) -
/// wraps `find_very_weak_enemy`, which was defined but never wired into any
/// difficulty arm.
fn nation_strategy_very_weak(
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
    let Some(very_weak) = find_very_weak_enemy(game, sid, bordering) else {
        return false;
    };
    nation_try_attack_player(
        game,
        random,
        sid,
        very_weak,
        reserve_ratio,
        expand_ratio,
        bot_attack_troops_sent,
        difficulty,
        emoji,
        false,
    )
}

/// TS `getAttackStrategies`' `victim` closure (Hard/Impossible only) - wraps
/// `find_victim`, same previously-dead-code situation as `veryWeak` above.
fn nation_strategy_victim(
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
    let Some(victim) = find_victim(game, sid, bordering) else {
        return false;
    };
    nation_try_attack_player(
        game,
        random,
        sid,
        victim,
        reserve_ratio,
        expand_ratio,
        bot_attack_troops_sent,
        difficulty,
        emoji,
        false,
    )
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
        // TS `calculateAttackTroops`: `troopSendCap`/`isAttackTooWeak` apply
        // to EVERY player-targeted attack, including bot targets (the
        // bot-specific `calculateBotAttackTroops` branch feeds straight into
        // the same shared cap/weak-check below it) - not just the plain
        // land/boat-vs-non-bot path.
        let Some(troops) =
            cap_player_attack_troops(game, attacker_small_id, target_small_id, troops)
        else {
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
    let Some(troops) =
        cap_player_attack_troops(game, attacker_small_id, target_small_id, troops)
    else {
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

/// Ports of `openfront/tests/AiAttackBehavior.test.ts`. That file's "Ai Attack
/// Behavior" describe block exercises the alliance-blocks-attack path
/// through `AttackExecution::init` (already covered indirectly by
/// `attack.rs`'s own tests, but ported here too since it's this file's
/// entry point being exercised); the "Hard/Impossible troop floor" describe
/// block covers `troopSendCap`/`isAttackTooWeak` - both entirely unported
/// natively until the fix in this same file (see the doc comments on
/// `troop_send_cap`/`is_attack_too_weak` above).
#[cfg(test)]
mod ai_attack_behavior_tests {
    use super::*;
    use crate::game::PlayerInfo;
    use crate::map::{GameMap, MapMeta};

    /// TS tests load a real 200x200, all-land "big_plains" fixture map via
    /// `tests/util/Setup.ts` to get genuine border/shore geometry - `Game::
    /// default()`'s 1x1 stub map can't support `sharesBorderWith`/`nearby()`
    /// at all. Loads the same fixture (checked into the pinned `openfront`
    /// submodule) directly via the existing `GameMap::from_terrain_bytes`
    /// (already used by `Game::default()` and `core::terrain`'s fresh-terrain
    /// loader) rather than building new map-geometry test infrastructure.
    /// Returns `None` (callers skip/return early) if the submodule checkout
    /// is unavailable, matching `core::terrain::tests`'s `skip_without_maps`.
    pub(super) fn big_plains_map() -> Option<GameMap> {
        let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .ok()?;
        let map_bin = repo_root.join("openfront/tests/testdata/maps/big_plains/map.bin");
        let bytes = std::fs::read(map_bin).ok()?;
        GameMap::from_terrain_bytes(
            &MapMeta {
                width: 200,
                height: 200,
                num_land_tiles: 40_000,
            },
            &bytes,
        )
        .ok()
    }

    fn test_wire_config(difficulty: &str, game_mode: &str) -> crate::core::config::Config {
        crate::core::config::Config::new(
            crate::core::schemas::GameConfig {
                game_map: "big_plains".into(),
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
        )
    }

    /// Builds a game on the real `big_plains` map with spawn phase ended.
    /// Returns `None` (caller should skip the test) if the fixture map isn't
    /// available in this checkout.
    fn new_game(difficulty: &str, game_mode: &str) -> Option<Game> {
        let map = big_plains_map()?;
        let mut game = Game::default();
        game.map = map;
        game.wire = test_wire_config(difficulty, game_mode);
        game.end_spawn_phase();
        Some(game)
    }

    pub(super) fn add_player(game: &mut Game, id: &str, player_type: PlayerType) -> u16 {
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

    /// TS test helper `assignAlternatingLandTiles`/the inline `forEachTile`
    /// round-robin loops: `big_plains` is 100% land, so the first `count`
    /// row-major tile refs (`0..count`, all within row 0 for any `count <=
    /// 200`) are exactly its first `count` land tiles in TS iteration order.
    fn conquer_round_robin(game: &mut Game, owners: &[u16], count: u32) {
        for i in 0..count {
            game.conquer(owners[(i as usize) % owners.len()], i);
        }
    }

    /// TS `setupTroopFloorTest`: attacker (Nation) / neighbor (Human) / bot
    /// (Bot, target) share borders via 90 alternating tiles; bot gets a
    /// nominal 100 troops so it's a valid (non-zero-troop) target.
    fn troop_floor_setup(difficulty: &str, game_mode: &str) -> Option<(Game, u16, u16, u16)> {
        let mut game = new_game(difficulty, game_mode)?;
        let attacker = add_player(&mut game, "attacker_id", PlayerType::Nation);
        let neighbor = add_player(&mut game, "neighbor_id", PlayerType::Human);
        let bot = add_player(&mut game, "bot_id", PlayerType::Bot);
        conquer_round_robin(&mut game, &[attacker, neighbor, bot], 90);
        game.add_troops(bot, 100.0);
        Some((game, attacker, neighbor, bot))
    }

    /// Reads back the troops actually queued for a still-uninitialized attack
    /// (owner, target) by running exactly one tick - enough for `Execution::
    /// init` (which sets the real, post-cap `troops` value) to run, but not
    /// enough for `Execution::tick` (combat) to consume any of it, matching
    /// TS's tests reading `exec.startTroops` from a spied `addExecution` call
    /// synchronously, before the attack has ever ticked.
    fn queued_attack_troops(game: &mut Game, owner: u16, target: u16) -> Option<f64> {
        game.execute_next_tick();
        game.active_attacks_debug()
            .into_iter()
            .find(|a| a.0 == owner && a.1 == target)
            .map(|a| a.2)
    }

    #[test]
    fn bot_cannot_attack_allied_player() {
        let Some(mut game) = new_game("Medium", "Free For All") else {
            return;
        };
        let bot = add_player(&mut game, "bot_test", PlayerType::Bot);
        let human = add_player(&mut game, "human_test", PlayerType::Human);
        conquer_round_robin(&mut game, &[bot, human], 200);
        game.add_troops(bot, 5000.0);
        game.add_troops(human, 5000.0);

        assert!(game.create_alliance_request(bot, human, 0));
        game.accept_alliance_request(bot, human, 1);
        assert!(game.is_friendly(bot, human));

        let attacks_before = game.player_by_small_id(bot).unwrap().outgoing_land_attacks.len();

        let mut random = PseudoRandom::new(42);
        let mut sent = 0.0;
        try_send_player_attack(
            &mut game, &mut random, bot, human, 0.5, 0.5, &mut sent, "Medium", None,
        );

        for _ in 0..5 {
            game.execute_next_tick();
        }

        assert!(game.is_friendly(bot, human));
        assert_eq!(game.incoming_attacks(human, false).len(), 0);
        assert_eq!(
            game.player_by_small_id(bot).unwrap().outgoing_land_attacks.len(),
            attacks_before
        );
    }

    #[test]
    fn nation_cannot_attack_allied_player() {
        let Some(mut game) = new_game("Medium", "Free For All") else {
            return;
        };
        let bot = add_player(&mut game, "bot_test", PlayerType::Bot);
        let human = add_player(&mut game, "human_test", PlayerType::Human);
        let nation = add_player(&mut game, "nation_test", PlayerType::Nation);
        conquer_round_robin(&mut game, &[bot, human, nation], 21);
        game.add_troops(nation, 1000.0);

        assert!(game.create_alliance_request(nation, human, 0));
        game.accept_alliance_request(nation, human, 1);
        assert!(game.is_friendly(nation, human));

        let attacks_before = game
            .player_by_small_id(nation)
            .unwrap()
            .outgoing_land_attacks
            .len();
        game.add_troops(nation, 50_000.0);

        let mut random = PseudoRandom::new(42);
        let mut sent = 0.0;
        // TS: `nationBehavior.sendAttack(human, true)` - force past `shouldAttack`'s
        // dice roll so the alliance check in `AttackExecution::init` is the layer
        // under test, regardless of RNG outcome.
        try_send_player_attack_forced(
            &mut game, &mut random, nation, human, 0.5, 0.5, &mut sent, "Medium", None, true,
        );

        for _ in 0..5 {
            game.execute_next_tick();
        }

        assert!(game.is_friendly(nation, human));
        assert_eq!(
            game.player_by_small_id(nation).unwrap().outgoing_land_attacks.len(),
            attacks_before
        );
    }

    #[test]
    fn hard_caps_attack_troops_to_75pct_of_strongest_neighbor() {
        let Some((mut game, attacker, neighbor, _bot)) = troop_floor_setup("Hard", "Free For All")
        else {
            return;
        };
        game.add_troops(attacker, 100_000.0);
        game.add_troops(neighbor, 90_000.0);

        let attacker_troops = game.player_by_small_id(attacker).unwrap().troops as f64;
        let neighbor_troops = game.player_by_small_id(neighbor).unwrap().troops as f64;
        let min_retained = (neighbor_troops * 0.75).ceil();
        let expected_cap = (attacker_troops - min_retained).max(0.0);

        let mut random = PseudoRandom::new(42);
        let mut sent = 0.0;
        let result = try_send_player_attack(
            &mut game, &mut random, attacker, neighbor, 0.3, 0.2, &mut sent, "Hard", None,
        );
        assert!(result);
        let start_troops =
            queued_attack_troops(&mut game, attacker, neighbor).expect("attack should be queued");
        assert!(
            start_troops <= expected_cap,
            "start_troops={start_troops} expected_cap={expected_cap}"
        );
    }

    #[test]
    fn hard_prevents_attack_below_75pct_troop_floor() {
        let Some((mut game, attacker, neighbor, bot)) = troop_floor_setup("Hard", "Free For All")
        else {
            return;
        };
        game.add_troops(attacker, 3_000.0);
        game.add_troops(neighbor, 5_000.0);

        let mut random = PseudoRandom::new(42);
        let mut sent = 0.0;
        let result = try_send_player_attack(
            &mut game, &mut random, attacker, bot, 0.3, 0.2, &mut sent, "Hard", None,
        );
        assert!(!result);
        assert!(queued_attack_troops(&mut game, attacker, bot).is_none());
    }

    #[test]
    fn hard_skips_attack_when_capped_troops_too_weak() {
        let Some((mut game, attacker, neighbor, _bot)) = troop_floor_setup("Hard", "Free For All")
        else {
            return;
        };
        let target = add_player(&mut game, "target_id", PlayerType::Human);
        // TS: steal 20 of the attacker's own tiles for the new target, so it
        // borders the attacker directly (matches `attacker.tiles()` insertion
        // order - `conquer_round_robin`'s row-major, per-owner assignment).
        let attacker_tiles: Vec<_> = game
            .player_by_small_id(attacker)
            .unwrap()
            .owned_tiles
            .iter()
            .take(20)
            .copied()
            .collect();
        for tile in attacker_tiles {
            game.conquer(target, tile);
        }

        game.add_troops(attacker, 100_000.0);
        game.add_troops(neighbor, 100_000.0);
        game.add_troops(target, 300_000.0);

        let mut random = PseudoRandom::new(42);
        let mut sent = 0.0;
        let result = try_send_player_attack(
            &mut game, &mut random, attacker, target, 0.3, 0.2, &mut sent, "Hard", None,
        );
        assert!(!result);
        assert!(queued_attack_troops(&mut game, attacker, target).is_none());
    }

    #[test]
    fn impossible_caps_attack_troops_to_90pct_of_strongest_neighbor() {
        let Some((mut game, attacker, neighbor, _bot)) =
            troop_floor_setup("Impossible", "Free For All")
        else {
            return;
        };
        game.add_troops(attacker, 100_000.0);
        game.add_troops(neighbor, 90_000.0);

        let attacker_troops = game.player_by_small_id(attacker).unwrap().troops as f64;
        let neighbor_troops = game.player_by_small_id(neighbor).unwrap().troops as f64;
        let min_retained = (neighbor_troops * 0.9).ceil();
        let expected_cap = (attacker_troops - min_retained).max(0.0);

        let mut random = PseudoRandom::new(42);
        let mut sent = 0.0;
        let result = try_send_player_attack(
            &mut game, &mut random, attacker, neighbor, 0.3, 0.2, &mut sent, "Impossible", None,
        );
        assert!(result);
        let start_troops =
            queued_attack_troops(&mut game, attacker, neighbor).expect("attack should be queued");
        assert!(
            start_troops <= expected_cap,
            "start_troops={start_troops} expected_cap={expected_cap}"
        );
    }

    #[test]
    fn easy_has_no_troop_floor() {
        let Some((mut game, attacker, neighbor, bot)) = troop_floor_setup("Easy", "Free For All")
        else {
            return;
        };
        game.add_troops(attacker, 100_000.0);
        game.add_troops(neighbor, 90_000.0);

        let mut random = PseudoRandom::new(42);
        let mut sent = 0.0;
        let result = try_send_player_attack(
            &mut game, &mut random, attacker, bot, 0.3, 0.2, &mut sent, "Easy", None,
        );
        assert!(result);
        let start_troops =
            queued_attack_troops(&mut game, attacker, bot).expect("attack should be queued");
        assert!(start_troops > 0.0);

        let attacker_troops = game.player_by_small_id(attacker).unwrap().troops as f64;
        let neighbor_troops = game.player_by_small_id(neighbor).unwrap().troops as f64;
        let hard_cap = (attacker_troops - (neighbor_troops * 0.75).ceil()).max(0.0);
        assert!(start_troops > hard_cap);
    }

    #[test]
    fn hard_send_attack_uncapped_with_no_player_neighbors() {
        let Some(mut game) = new_game("Hard", "Free For All") else {
            return;
        };
        let bot = add_player(&mut game, "lone_id", PlayerType::Bot);
        // Bot owns every other land tile map-wide, leaving the rest
        // unowned TerraNullius, so it borders only TerraNullius (no player
        // neighbors) - matches TS's checkerboard `assigned % 2 === 0` loop
        // over the whole map (odd tiles are simply never conquered - they're
        // TerraNullius, small ID 0, by default already).
        for tile in (0..40_000u32).step_by(2) {
            game.conquer(bot, tile);
        }
        game.add_troops(bot, 100_000.0);

        assert!(nearby_player_small_ids(&game, bot).is_empty());

        // TS `sendAttack(terraNullius)` dispatches non-player targets to
        // `sendLandAttack`/`sendBoatAttackToNearbyTerraNullius` directly
        // (never through the player-target `calculateAttackTroops` branch,
        // which is what `try_send_player_attack` models) - `try_send_tn_attack`
        // is the matching native entry point.
        let sent_tn = try_send_tn_attack(&mut game, bot, 0.2);
        assert!(sent_tn);
        let tn = game.terra_nullius_id();
        let start_troops =
            queued_attack_troops(&mut game, bot, tn).expect("attack should be queued");
        assert!(start_troops > 40_000.0);
    }

    #[test]
    fn team_mode_troop_send_cap_is_uncapped() {
        let Some((mut game, attacker, neighbor, bot)) = troop_floor_setup("Hard", "Team") else {
            return;
        };
        game.add_troops(attacker, 100_000.0);
        game.add_troops(neighbor, 90_000.0);

        let mut random = PseudoRandom::new(42);
        let mut sent = 0.0;
        let result = try_send_player_attack(
            &mut game, &mut random, attacker, bot, 0.3, 0.2, &mut sent, "Hard", None,
        );
        assert!(result);
        let start_troops =
            queued_attack_troops(&mut game, attacker, bot).expect("attack should be queued");
        // In FFA Hard this would be capped to ~32.5k; Team mode is uncapped.
        assert!(start_troops > 32_500.0);
    }

    #[test]
    fn team_mode_is_attack_too_weak_never_blocks() {
        let Some((mut game, attacker, neighbor, _bot)) = troop_floor_setup("Hard", "Team") else {
            return;
        };
        let target = add_player(&mut game, "target_id", PlayerType::Human);
        conquer_round_robin(&mut game, &[attacker, neighbor, target], 90);

        game.add_troops(attacker, 100_000.0);
        game.add_troops(neighbor, 100_000.0);
        game.add_troops(target, 300_000.0);

        let mut random = PseudoRandom::new(42);
        let mut sent = 0.0;
        let result = try_send_player_attack(
            &mut game, &mut random, attacker, target, 0.3, 0.2, &mut sent, "Hard", None,
        );
        assert!(result);
        let start_troops =
            queued_attack_troops(&mut game, attacker, target).expect("attack should be queued");
        assert!(start_troops > 0.0);
    }

    #[test]
    fn hard_nation_under_attack_bypasses_troop_send_cap() {
        let Some((mut game, attacker, neighbor, _bot)) = troop_floor_setup("Hard", "Free For All")
        else {
            return;
        };
        game.add_troops(attacker, 100_000.0);
        game.add_troops(neighbor, 200_000.0);

        let attacker_troops = game.player_by_small_id(attacker).unwrap().troops as f64;
        let neighbor_troops = game.player_by_small_id(neighbor).unwrap().troops as f64;
        let normal_cap = (attacker_troops - (neighbor_troops * 0.75).ceil()).max(0.0);
        assert_eq!(normal_cap, 0.0);

        // TS's `TestConfig` hardcodes `nationSpawnImmunityDuration()` to 0;
        // native's `Config::nation_spawn_immunity_duration()` doesn't have a
        // test-config override (always the production default, 50 ticks -
        // see its doc comment), so just tick past it instead.
        for _ in 0..60 {
            game.execute_next_tick();
        }

        let attacker_id = game.player_by_small_id(attacker).unwrap().id.clone();
        game.add_land_attack(neighbor, Some(attacker_id), Some(50_000.0));
        game.execute_next_tick();
        assert!(!game.incoming_attacks(attacker, false).is_empty());

        // Directly inspect the pre-execution capped troop value (matching TS's
        // spy on the `AttackExecution` constructor's args, captured BEFORE the
        // new outgoing attack's own `init()` cancels itself out against this
        // same incoming attack - `queued_attack_troops`'s post-init read would
        // observe the post-cancellation value instead, which is a different,
        // unrelated number here since incoming == the newly capped amount).
        let raw = land_attack_troops(&game, attacker, 0.3).expect("reserve check should pass");
        let capped = cap_player_attack_troops(&game, attacker, neighbor, raw)
            .expect("retaliation should bypass the cap");
        assert!(
            capped >= 50_000.0,
            "capped={capped} should retain at least the incoming 50k"
        );
    }
}

/// Ported from `AiAttackBehaviorNukedTerritory.test.ts`. Covers the
/// `nearby()`/`hasNonNukedTerraNullius` nuked-territory early-out gate in
/// `maybeAttack` (native: `nation_maybe_attack`'s `has_non_nuked_tn` gate)
/// and the `nuked` attack strategy (native: `is_bordering_nuked_territory`).
#[cfg(test)]
mod ai_attack_nuked_territory_tests {
    use super::ai_attack_behavior_tests::{add_player, big_plains_map};
    use super::*;

    /// TS `setupBehavior`'s config: `infiniteGold`/`instantBuild`/
    /// `infiniteTroops` all `true` (irrelevant to this file's assertions, but
    /// matched for parity), plus optional `disabledUnits`.
    fn wire_config(difficulty: &str, disabled_units: Option<Vec<String>>) -> crate::core::config::Config {
        crate::core::config::Config::new(
            crate::core::schemas::GameConfig {
                game_map: "big_plains".into(),
                difficulty: difficulty.into(),
                donate_gold: false,
                donate_troops: false,
                game_type: "Singleplayer".into(),
                game_mode: "Free For All".into(),
                game_map_size: "Normal".into(),
                nations: crate::core::schemas::NationsConfig::Mode("default".into()),
                bots: 0,
                infinite_gold: true,
                infinite_troops: true,
                instant_build: true,
                random_spawn: false,
                doomsday_clock: None,
                disabled_units,
                player_teams: None,
                disable_alliances: None,
                spawn_immunity_duration: None,
                starting_gold: None,
                gold_multiplier: None,
                max_timer_value: None,
                ranked_type: None,
            },
            false,
        )
    }

    fn new_game(difficulty: &str, disabled_units: Option<Vec<String>>) -> Option<Game> {
        let map = big_plains_map()?;
        let mut game = Game::default();
        game.map = map;
        game.wire = wire_config(difficulty, disabled_units);
        game.end_spawn_phase();
        Some(game)
    }

    /// Conquer every land tile of `[x0,x1) x [y0,y1)` for `owner` (`big_plains`
    /// is 100% land, so this never has to skip water).
    fn conquer_rect(game: &mut Game, owner: u16, x0: u32, y0: u32, x1: u32, y1: u32) {
        for x in x0..x1 {
            for y in y0..y1 {
                let t = game.ref_xy(x, y);
                if game.is_land(t) {
                    game.conquer(owner, t);
                }
            }
        }
    }

    /// Mark every unowned land tile in `[x0,x1) x [y0,y1)` as nuked
    /// (fallout). Already-conquered tiles are naturally skipped, matching TS
    /// `nukeRect`'s comment that `setFallout` throws on owned tiles.
    fn nuke_rect(game: &mut Game, x0: u32, y0: u32, x1: u32, y1: u32) {
        for x in x0..x1 {
            for y in y0..y1 {
                let t = game.ref_xy(x, y);
                if game.is_land(t) && !game.has_owner(t) {
                    game.map.set_fallout(t, true);
                }
            }
        }
    }

    /// TS `setupBehavior`: a Nation at x∈[60,80), an optional Human "enemy"
    /// at x∈[40,60) sharing a land border with it (both y∈[60,80)), and,
    /// unless `with_nuke` is false, every other unowned tile in
    /// x∈[40,120), y∈[40,100) marked nuked - so the nation's only non-nuked
    /// border is the enemy, when present.
    struct NukedEnv {
        game: Game,
        nation: u16,
        enemy: u16,
    }

    fn setup_behavior(
        difficulty: &str,
        with_enemy: bool,
        with_nuke: bool,
        nation_troops: f64,
        enemy_troops: f64,
        disabled_units: Option<Vec<String>>,
    ) -> Option<NukedEnv> {
        let mut game = new_game(difficulty, disabled_units)?;
        let nation = add_player(&mut game, "nation_id", PlayerType::Nation);
        let enemy = add_player(&mut game, "enemy_id", PlayerType::Human);

        conquer_rect(&mut game, nation, 60, 60, 80, 80);
        if with_enemy {
            conquer_rect(&mut game, enemy, 40, 60, 60, 80);
        }
        if with_nuke {
            nuke_rect(&mut game, 40, 40, 120, 100);
        }

        game.add_troops(nation, nation_troops);
        game.add_troops(enemy, enemy_troops);

        assert!(game.player_by_small_id(nation).unwrap().tiles_owned > 0);
        Some(NukedEnv { game, nation, enemy })
    }

    /// New outgoing attacks (owner == `owner`) not present in `before`.
    fn new_outgoing_attacks(
        game: &Game,
        owner: u16,
        before: &[(u16, u16, f64, bool, bool, usize, usize)],
    ) -> Vec<(u16, u16, f64, bool, bool, usize, usize)> {
        game.active_attacks_debug()
            .into_iter()
            .filter(|a| a.0 == owner && !before.contains(a))
            .collect()
    }

    #[test]
    fn nearby_excludes_directly_adjacent_nuked_terra_nullius() {
        let Some(NukedEnv { game, nation, .. }) =
            setup_behavior("Impossible", false, true, 5_000_000.0, 50_000.0, None)
        else {
            return;
        };

        // Sanity: the nation really borders nuked land.
        assert!(is_bordering_nuked_territory(&game, nation));

        // nearby() must not report TerraNullius (it's all nuked), and with no
        // enemy there are no player neighbours either.
        let nearby = nearby_players_ts_order(&game, nation);
        assert!(nearby.is_empty(), "nearby={nearby:?} should be empty");
    }

    #[test]
    fn maybe_attack_does_not_preempt_retaliation_with_nuked_tn_attack() {
        // Nation borders nuked TN (east/north/south) and an enemy (west). The
        // enemy attacks the nation. On Impossible `retaliate` is the first
        // strategy, but with the bug the early gate fires first and attacks
        // TerraNullius, so retaliation never runs. The nation has far more
        // troops than the enemy so `retaliate`'s attack is not rejected as
        // "too weak".
        let Some(NukedEnv { mut game, nation, enemy }) =
            setup_behavior("Impossible", true, true, 5_000_000.0, 50_000.0, None)
        else {
            return;
        };

        // Native hardcodes `nation_spawn_immunity_duration()` to 50 ticks
        // (TS's test config overrides it to 0) - tick past it so the
        // enemy's (Human) attack on the nation (Nation) isn't self-cancelled
        // by `can_attack_player`'s immunity check.
        for _ in 0..60 {
            game.execute_next_tick();
        }

        let nation_id = game.player_by_small_id(nation).unwrap().id.clone();
        game.add_land_attack(enemy, Some(nation_id), Some(100_000.0));
        game.execute_next_tick();
        assert!(!game.incoming_attacks(nation, false).is_empty());

        let before = game.active_attacks_debug();
        let mut random = PseudoRandom::new(42);
        let mut sent = 0.0;
        nation_maybe_attack(&mut game, &mut random, nation, 0.0, 0.0, 0.0, &mut sent, "Impossible", None);
        game.execute_next_tick();

        let attacks = new_outgoing_attacks(&game, nation, &before);
        assert!(!attacks.is_empty(), "expected at least one new outgoing attack");
        for a in &attacks {
            assert_eq!(a.1, enemy, "expected retaliation against the enemy, not TerraNullius");
        }
    }

    #[test]
    fn maybe_attack_early_gate_bypassed_when_only_nuked_tn_borders() {
        // No enemy, no incoming attack. The early gate must NOT fire (there
        // is no non-nuked TN). `attackBestTarget` falls through to the
        // `nuked` strategy, which dispatches a land attack on TerraNullius.
        let Some(NukedEnv { mut game, nation, .. }) =
            setup_behavior("Impossible", false, true, 5_000_000.0, 50_000.0, None)
        else {
            return;
        };
        assert!(game.incoming_attacks(nation, false).is_empty());

        let before = game.active_attacks_debug();
        let mut random = PseudoRandom::new(42);
        let mut sent = 0.0;
        nation_maybe_attack(&mut game, &mut random, nation, 0.0, 0.0, 0.0, &mut sent, "Impossible", None);
        game.execute_next_tick();

        let attacks = new_outgoing_attacks(&game, nation, &before);
        assert!(!attacks.is_empty(), "expected at least one new outgoing attack");
        let tn = game.terra_nullius_id();
        for a in &attacks {
            assert_eq!(a.1, tn, "expected a TerraNullius attack, got target {}", a.1);
        }
    }

    #[test]
    fn nuked_strategy_captures_tiles_when_idle() {
        let Some(NukedEnv { mut game, nation, .. }) =
            setup_behavior("Impossible", false, true, 5_000_000.0, 50_000.0, None)
        else {
            return;
        };

        let before = game.active_attacks_debug();
        let mut random = PseudoRandom::new(42);
        let mut sent = 0.0;
        nation_maybe_attack(&mut game, &mut random, nation, 0.0, 0.0, 0.0, &mut sent, "Impossible", None);
        game.execute_next_tick();

        let attacks = new_outgoing_attacks(&game, nation, &before);
        assert!(!attacks.is_empty());
        let tn = game.terra_nullius_id();
        for a in &attacks {
            assert_eq!(a.1, tn);
        }

        // Let the AttackExecution make progress. The nation should conquer
        // at least one previously-nuked tile east of its territory (x >= 80).
        for _ in 0..60 {
            game.execute_next_tick();
        }
        let mut conquered_east = 0;
        for x in 80..120 {
            for y in 60..100 {
                let t = game.ref_xy(x, y);
                if game.map.owner_id(t) == nation {
                    conquered_east += 1;
                }
            }
        }
        assert!(conquered_east > 0, "expected the nation to conquer nuked land east of its territory");
    }

    #[test]
    fn easy_difficulty_nuked_strategy_still_fires_when_idle() {
        let Some(NukedEnv { mut game, nation, .. }) =
            setup_behavior("Easy", false, true, 5_000_000.0, 50_000.0, None)
        else {
            return;
        };

        let before = game.active_attacks_debug();
        let mut random = PseudoRandom::new(42);
        let mut sent = 0.0;
        nation_maybe_attack(&mut game, &mut random, nation, 0.0, 0.0, 0.0, &mut sent, "Easy", None);
        game.execute_next_tick();

        // On Easy the `nuked` strategy is first, so it dispatches a TN attack.
        let attacks = new_outgoing_attacks(&game, nation, &before);
        assert!(!attacks.is_empty());
        let tn = game.terra_nullius_id();
        for a in &attacks {
            assert_eq!(a.1, tn);
        }
    }

    #[test]
    fn missile_silo_disabled_disables_nuked_strategy() {
        // `isBorderingNukedTerritory` returns false when MissileSilo is
        // disabled, so even with nuked TN on the border the `nuked` strategy
        // does NOT fire and no attack is created.
        let Some(NukedEnv { mut game, nation, .. }) = setup_behavior(
            "Impossible",
            false,
            true,
            5_000_000.0,
            50_000.0,
            Some(vec![crate::core::schemas::unit_type::MISSILE_SILO.to_string()]),
        ) else {
            return;
        };

        let before = game.active_attacks_debug();
        let mut random = PseudoRandom::new(42);
        let mut sent = 0.0;
        nation_maybe_attack(&mut game, &mut random, nation, 0.0, 0.0, 0.0, &mut sent, "Impossible", None);
        game.execute_next_tick();

        let attacks = new_outgoing_attacks(&game, nation, &before);
        assert!(attacks.is_empty(), "expected no attack, got {attacks:?}");
    }
}

// TS `ImpassableTerrain.test.ts` "Nation AI attack behavior near impassable
// terrain" describe block.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::Player;

    fn bot_player(id: &str, small_id: u16) -> Player {
        Player {
            id: id.to_string(),
            small_id,
            player_type: PlayerType::Nation,
            troops: 200_000,
            ..Default::default()
        }
    }

    fn wall_scenario() -> (crate::game::Game, u32) {
        let wall_x = 30u32;
        let mut game = crate::test_util::walled_game(60, 20, Some((wall_x, 2)));
        game.add_player(bot_player("nation", 1));
        game.add_player(bot_player("enemy", 2));
        // Nation owns the two columns right next to the wall (full height).
        for y in 0..20 {
            game.conquer(1, game.map.ref_xy(wall_x - 1, y));
            game.conquer(1, game.map.ref_xy(wall_x - 2, y));
        }
        // Enemy owns the five columns directly to the left of the nation
        // (full height) - no TerraNullius gap between them.
        for y in 0..20 {
            for x in (wall_x - 7)..=(wall_x - 3) {
                game.conquer(2, game.map.ref_xy(x, y));
            }
        }
        (game, wall_x)
    }

    /// TS `ImpassableTerrain.test.ts` "hasNonNukedTerraNullius does not
    /// falsely detect impassable tiles as TerraNullius": a nation bordering
    /// an impassable wall must not see the wall's TerraNullius (owner 0) as
    /// a nearby neighbor, but must still see the real enemy player across
    /// its other border. Ported directly against `nearby_players_ts_order`
    /// (TS `PlayerImpl.nearby()`'s native equivalent, already
    /// `!is_impassable`-guarded per its own doc comment) rather than the TS
    /// test's `AiAttackBehavior`/`NationAllianceBehavior`/
    /// `NationEmojiBehavior` orchestration wiring, which has no 1:1 native
    /// port to construct (client-facing plumbing, not decision logic).
    #[test]
    fn nearby_players_excludes_terra_nullius_from_an_impassable_wall() {
        let (game, _) = wall_scenario();
        let neighbors = nearby_players_ts_order(&game, 1);
        assert!(
            neighbors.contains(&2),
            "enemy should be a nearby neighbor: {neighbors:?}"
        );
        assert!(
            !neighbors.contains(&0),
            "impassable-adjacent tiles must not surface TerraNullius: {neighbors:?}"
        );
    }

    /// TS `ImpassableTerrain.test.ts`'s follow-up assertion: the nation must
    /// actually be able to construct a real attack against the enemy (not
    /// TerraNullius) despite bordering the impassable wall.
    #[test]
    fn attack_execution_can_target_enemy_across_from_an_impassable_wall() {
        let (mut game, _) = wall_scenario();
        game.add_execution(crate::execution::ExecEnum::Attack(
            crate::execution::attack::AttackExecution::new(
                1,
                Some("enemy".to_string()),
                Some(1000.0),
            ),
        ));
        game.execute_next_tick();
        game.execute_next_tick();

        assert!(!game
            .player_by_small_id(1)
            .unwrap()
            .outgoing_land_attacks
            .is_empty());
    }
}
