//! Atom Bomb / Hydrogen Bomb / MIRV Warhead flight + detonation (`NukeExecution.ts`).
//!
//! MIRV Warheads reuse this same execution (constructed by `MirvExecution.separate()`);
//! their spawn tile always equals `dst` (TS `canSpawnUnitType(MIRVWarhead) -> targetTile`),
//! which degenerates the parabola curve to a single point so movement completes on the
//! first tick after `waitTicks` - no special-casing needed beyond that spawn resolution.

use super::parabola::{self, Curve};
use super::Execution;
use crate::core::schemas::unit_type;
use crate::game::Game;
use crate::map::TileRef;
use crate::prng::PseudoRandom;
use std::collections::HashMap;

const STRUCTURE_TYPES: [&str; 6] = [
    unit_type::CITY,
    unit_type::PORT,
    unit_type::FACTORY,
    unit_type::DEFENSE_POST,
    unit_type::MISSILE_SILO,
    unit_type::SAM_LAUNCHER,
];

const EXCLUDED_FROM_BLAST: [&str; 5] = [
    unit_type::ATOM_BOMB,
    unit_type::HYDROGEN_BOMB,
    unit_type::MIRV,
    unit_type::MIRV_WARHEAD,
    unit_type::SAM_MISSILE,
];

pub struct NukeExecution {
    nuke_type: String,
    owner_small_id: u16,
    dst: TileRef,
    src: Option<TileRef>,
    speed: f64,
    wait_ticks: u32,
    rocket_direction_up: bool,

    active: bool,
    nuke_unit_id: Option<i32>,
    curve: Option<Curve>,
    tiles_to_destroy_cache: Option<Vec<TileRef>>,
}

impl NukeExecution {
    pub fn new(
        nuke_type: &str,
        owner_small_id: u16,
        dst: TileRef,
        src: Option<TileRef>,
        speed: f64,
        wait_ticks: u32,
        rocket_direction_up: bool,
    ) -> Self {
        Self {
            nuke_type: nuke_type.to_string(),
            owner_small_id,
            dst,
            src,
            speed,
            wait_ticks,
            rocket_direction_up,
            active: true,
            nuke_unit_id: None,
            curve: None,
            tiles_to_destroy_cache: None,
        }
    }

    fn distance_based_height(&self) -> bool {
        self.nuke_type != unit_type::MIRV_WARHEAD
    }

    fn spawn(&mut self, game: &mut Game) {
        let Some(spawn_tile) = can_build_nuke(game, self.owner_small_id, &self.nuke_type, self.dst)
        else {
            self.active = false;
            return;
        };
        self.src = Some(spawn_tile);
        let id = game.build_unit(self.owner_small_id, &self.nuke_type, spawn_tile);
        self.nuke_unit_id = Some(id);

        let target_range_sq = game.wire.default_nuke_targetable_range().powi(2);
        let trajectory = parabola::find_path_tiles(
            game,
            spawn_tile,
            self.dst,
            self.speed,
            self.distance_based_height(),
            self.rocket_direction_up,
        );
        let trajectory_targetable: Vec<bool> = trajectory
            .iter()
            .map(|&t| is_targetable(game, self.dst, t, Some(spawn_tile), target_range_sq))
            .collect();
        if let Some(u) = game.unit_mut(self.owner_small_id, id) {
            u.target_tile = Some(self.dst);
            u.trajectory = trajectory;
            u.trajectory_targetable = trajectory_targetable;
            u.targetable = true;
        }

        if self.nuke_type != unit_type::MIRV_WARHEAD {
            maybe_break_alliances(game, self.owner_small_id, self.dst, &self.nuke_type);
        }

        // TS `NukeExecution.tick` - after launch, put the launching silo on cooldown.
        let silo_id = game
            .player_by_small_id(self.owner_small_id)
            .and_then(|p| {
                p.units
                    .iter()
                    .find(|u| u.unit_type == unit_type::MISSILE_SILO && u.tile as TileRef == spawn_tile)
                    .map(|u| u.id)
            });
        if let Some(sid) = silo_id {
            game.unit_launch(self.owner_small_id, sid);
        }
    }

    fn detonate(&mut self, game: &mut Game) {
        let (inner, outer) = game.wire.nuke_magnitudes(&self.nuke_type);
        let inner2 = (inner * inner) as u32;
        let outer2 = (outer * outer) as u32;
        let dst = self.dst;
        let tick = game.ticks();

        let to_destroy = if let Some(c) = self.tiles_to_destroy_cache.take() {
            c
        } else {
            let rand_cell = std::cell::RefCell::new(PseudoRandom::new(tick as i32));
            game.map.bfs(dst, |gm, n| {
                let d2 = gm.euclidean_dist_squared(dst, n);
                d2 <= outer2 && (d2 <= inner2 || rand_cell.borrow_mut().chance(2))
            })
        };

        let mut tiles_per_player: HashMap<u16, u32> = HashMap::new();
        for &t in &to_destroy {
            let owner = game.map.owner_id(t);
            if owner != 0 {
                game.relinquish_tile(t);
                *tiles_per_player.entry(owner).or_insert(0) += 1;
            }
            if game.is_land(t) {
                game.map.set_fallout(t, true);
            }
        }

        for (&owner_sid, &num_impacted) in tiles_per_player.iter() {
            let tiles_owned_now = game
                .player_by_small_id(owner_sid)
                .map(|p| p.tiles_owned)
                .unwrap_or(0);
            let tiles_before_nuke = tiles_owned_now as f64 + num_impacted as f64;
            let max_troops = game.max_troops_for(owner_sid);
            for i in 0..num_impacted {
                let num_tiles_left = tiles_before_nuke - i as f64;
                let current_troops = game
                    .player_by_small_id(owner_sid)
                    .map(|p| p.troops)
                    .unwrap_or(0);
                let death = game.wire.nuke_death_factor(
                    &self.nuke_type,
                    current_troops as f64,
                    num_tiles_left,
                    max_troops,
                );
                if death > 0.0 {
                    let to_remove = current_troops.min(death.floor() as i32);
                    if to_remove > 0 {
                        if let Some(p) = game.player_by_small_id_mut(owner_sid) {
                            p.troops -= to_remove;
                        }
                    }
                }
                // TS `NukeExecution.detonate` also spends this same per-tile death
                // rate against the impacted player's already-launched attacks and
                // in-flight transport ships (see `apply_nuke_deaths_to_deployed_forces`).
                game.apply_nuke_deaths_to_deployed_forces(
                    owner_sid,
                    &self.nuke_type,
                    num_tiles_left,
                    max_troops,
                );
            }
        }

        let mut to_remove_units: Vec<(u16, i32)> = Vec::new();
        for p in game.players_in_order() {
            for u in &p.units {
                if EXCLUDED_FROM_BLAST.contains(&u.unit_type.as_str()) {
                    continue;
                }
                let d2 = game.map.euclidean_dist_squared(dst, u.tile as TileRef);
                if d2 < outer2 {
                    to_remove_units.push((p.small_id, u.id));
                }
            }
        }
        for (sid, uid) in to_remove_units {
            game.remove_unit(sid, uid);
        }

        self.active = false;
        if let Some(id) = self.nuke_unit_id.take() {
            game.remove_unit(self.owner_small_id, id);
        }
    }
}

impl Execution for NukeExecution {
    fn init(&mut self, game: &mut Game, _tick: u32) {
        if self.speed < 0.0 {
            self.speed = game.wire.default_nuke_speed();
        }
    }

    fn tick(&mut self, game: &mut Game, _tick: u32) {
        if !self.active {
            return;
        }
        let Some(nuke_id) = self.nuke_unit_id else {
            self.spawn(game);
            return;
        };
        if !game.unit_exists(self.owner_small_id, nuke_id) {
            self.active = false;
            return;
        }
        if self.wait_ticks > 0 {
            self.wait_ticks -= 1;
            return;
        }

        let src = self.src.expect("src set on spawn");
        if self.curve.is_none() {
            self.curve = Some(parabola::create_curve(
                game,
                src,
                self.dst,
                self.speed,
                self.distance_based_height(),
                self.rocket_direction_up,
            ));
        }
        let next = self.curve.as_mut().unwrap().increment(self.speed);
        match next {
            None => self.detonate(game),
            Some(p) => {
                update_nuke_targetable(game, self.owner_small_id, nuke_id);
                let tile = parabola::point_to_tile(game, p);
                game.move_unit(self.owner_small_id, nuke_id, tile);
                let idx = self.curve.as_ref().unwrap().current_index();
                if let Some(u) = game.unit_mut(self.owner_small_id, nuke_id) {
                    u.trajectory_index = idx as u32;
                }
            }
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        false
    }
}

fn is_targetable(
    game: &Game,
    target_tile: TileRef,
    nuke_tile: TileRef,
    src: Option<TileRef>,
    target_range_sq: f64,
) -> bool {
    let d2 = game.map.euclidean_dist_squared(nuke_tile, target_tile) as f64;
    if d2 < target_range_sq {
        return true;
    }
    if let Some(s) = src {
        let d2b = game.map.euclidean_dist_squared(s, nuke_tile) as f64;
        if d2b < target_range_sq {
            return true;
        }
    }
    false
}

fn update_nuke_targetable(game: &mut Game, owner_small_id: u16, nuke_id: i32) {
    let Some(u) = game.unit(owner_small_id, nuke_id) else {
        return;
    };
    let Some(target_tile) = u.target_tile else {
        return;
    };
    let current_tile = u.tile as TileRef;
    let target_range_sq = game.wire.default_nuke_targetable_range().powi(2);
    let targetable = is_targetable(game, target_tile, current_tile, None, target_range_sq);
    if let Some(u) = game.unit_mut(owner_small_id, nuke_id) {
        u.targetable = targetable;
    }
}

/// TS `PlayerImpl.canBuild` + `canBuildUnitType` + `canSpawnUnitType`, narrowed to the
/// nuke/MIRV-warhead cases.
pub fn can_build_nuke(
    game: &Game,
    owner_small_id: u16,
    nuke_type: &str,
    dst: TileRef,
) -> Option<TileRef> {
    if game.wire.is_unit_disabled(nuke_type) {
        return None;
    }
    let cost = game.structure_cost(owner_small_id, nuke_type);
    let Some(p) = game.player_by_small_id(owner_small_id) else {
        return None;
    };
    if p.gold < cost {
        return None;
    }
    if nuke_type != unit_type::MIRV_WARHEAD && p.tiles_owned <= 0 {
        return None;
    }
    match nuke_type {
        unit_type::MIRV => {
            if game.map.owner_id(dst) == 0 {
                return None;
            }
            nuke_spawn(game, owner_small_id, nuke_type, dst)
        }
        unit_type::ATOM_BOMB | unit_type::HYDROGEN_BOMB => {
            nuke_spawn(game, owner_small_id, nuke_type, dst)
        }
        unit_type::MIRV_WARHEAD => Some(dst),
        _ => None,
    }
}

fn nuke_spawn(game: &Game, owner_small_id: u16, nuke_type: &str, dst: TileRef) -> Option<TileRef> {
    if game.is_spawn_immunity_active() {
        return None;
    }
    let owner_of_tile = game.map.owner_id(dst);
    let game_over = game.winner.is_some();
    if owner_of_tile != 0 && game.players_on_same_team(owner_small_id, owner_of_tile) && !game_over {
        return None;
    }

    if game.wire.game_config().game_mode == "Team" && nuke_type != unit_type::MIRV && !game_over {
        let (_, outer) = game.wire.nuke_magnitudes(nuke_type);
        let would_hit_teammate = game
            .nearby_structures_any(dst, outer as u32, &STRUCTURE_TYPES)
            .iter()
            .any(|&(sid, ..)| sid != 0 && game.players_on_same_team(owner_small_id, sid));
        if would_hit_teammate {
            return None;
        }
    }

    let Some(p) = game.player_by_small_id(owner_small_id) else {
        return None;
    };
    let mut best: Option<(TileRef, u32)> = None;
    for u in &p.units {
        if u.unit_type != unit_type::MISSILE_SILO || u.under_construction {
            continue;
        }
        if game.unit_is_in_cooldown(owner_small_id, u.id) {
            continue;
        }
        let d = game.manhattan_dist(u.tile as TileRef, dst);
        if best.is_none_or(|(_, bd)| d < bd) {
            best = Some((u.tile as TileRef, d));
        }
    }
    best.map(|(t, _)| t)
}

/// TS `NukeExecution.maybeBreakAlliances` + `Util.listNukeBreakAlliance`.
fn maybe_break_alliances(game: &mut Game, nuker_sid: u16, dst: TileRef, nuke_type: &str) {
    if nuke_type == unit_type::MIRV_WARHEAD {
        return;
    }
    let (inner, outer) = game.wire.nuke_magnitudes(nuke_type);
    let threshold = game.wire.nuke_alliance_break_threshold();
    let targets = list_nuke_break_alliance(game, dst, inner, outer, threshold);

    for &sid in &targets {
        if game.pending_alliance_request(sid, nuker_sid).is_some() {
            game.reject_alliance_request(sid, nuker_sid);
        }
    }

    for &attacked_sid in &targets {
        if game.pending_alliance_request(nuker_sid, attacked_sid).is_some() {
            game.reject_alliance_request(nuker_sid, attacked_sid);
            continue;
        }
        game.break_alliance_silently(nuker_sid, attacked_sid);
        if attacked_sid != nuker_sid {
            game.update_relation(attacked_sid, nuker_sid, -100);
        }
    }
}

pub(crate) fn would_nuke_break_alliance(
    game: &Game,
    dst: TileRef,
    nuke_type: &str,
    ally_small_id: u16,
) -> bool {
    let (inner, outer) = game.wire.nuke_magnitudes(nuke_type);
    list_nuke_break_alliance(
        game,
        dst,
        inner,
        outer,
        game.wire.nuke_alliance_break_threshold(),
    )
    .contains(&ally_small_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{AttackExecution, ExecEnum};
    use crate::game::{PlayerInfo, PlayerType};

    // `PlayerType::Bot` (not `Human`) deliberately, so spawn immunity (which only
    // gates Human/Nation attackers/defenders, see `Game::is_player_immune`) doesn't
    // interfere with the attack's `init()` in the very next tick - matching the
    // existing `boat_landed_attack_cancels_opposing_land_attack` test's pattern in
    // `attack.rs`, since these tests aren't about immunity at all.
    fn add_bot(game: &mut Game, id: &str) -> u16 {
        game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type: PlayerType::Bot,
            client_id: Some(id.into()),
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        })
    }

    // TS `NukeExecution.detonate` (`openfront/src/core/execution/NukeExecution.ts`)
    // applies the SAME per-impacted-tile `nukeDeathFactor` rate to a nuked player's
    // home troops *and* to every one of their live outgoing attacks - ported here as
    // a direct call to `Game::apply_nuke_deaths_to_deployed_forces` (the mechanism
    // this test caught missing) rather than a literal port of `Attack.test.ts`'s
    // "Nuke reduce attacking troop counts", whose exact troop-loss numbers depend on
    // the `ocean_and_land` fixture map's real spawn/border geometry (the nuke lands
    // on the attacker's own spawn tile only because that tile has, by then, already
    // been conquered by the attacker's in-progress attack against a neighboring
    // spawn 5 tiles away) - `Game::default()`'s synthetic 1x1 map can't reproduce that.
    #[test]
    fn nuke_reduces_troops_of_a_live_outgoing_attack_owned_by_the_impacted_player() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let owner = add_bot(&mut game, "owner");
        let target = add_bot(&mut game, "target");
        if let Some(p) = game.player_by_small_id_mut(owner) {
            p.troops = 1_000;
            p.tiles_owned = 5;
        }
        if let Some(p) = game.player_by_small_id_mut(target) {
            p.tiles_owned = 5;
        }

        game.add_execution(ExecEnum::Attack(AttackExecution::new(
            owner,
            Some("target".to_string()),
            Some(300.0),
        )));
        game.execute_next_tick();

        let troops_before: f64 = game
            .live_attacks()
            .find(|a| a.owner_small_id() == owner)
            .map(|a| a.troops())
            .expect("attack should be live after init");
        assert_eq!(troops_before, 300.0);

        // A single impacted tile with 100 tiles left of the owner's territory
        // (tilesOwned before the nuke) - matches TS's diminishing-effect loop
        // running once with `numTilesLeft = 100`.
        game.apply_nuke_deaths_to_deployed_forces(owner, unit_type::ATOM_BOMB, 100.0, 10_000.0);

        let troops_after = game
            .live_attacks()
            .find(|a| a.owner_small_id() == owner)
            .map(|a| a.troops())
            .expect("attack should still be live");
        // nukeDeathFactor(ATOM_BOMB, 300, 100, _) = 5 * 300 / 100 = 15.
        assert_eq!(troops_after, 285.0);
    }

    #[test]
    fn nuke_deaths_never_push_deployed_forces_below_zero() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let owner = add_bot(&mut game, "owner");
        let target = add_bot(&mut game, "target");
        if let Some(p) = game.player_by_small_id_mut(owner) {
            p.troops = 1_000;
            p.tiles_owned = 5;
        }
        if let Some(p) = game.player_by_small_id_mut(target) {
            p.tiles_owned = 5;
        }

        game.add_execution(ExecEnum::Attack(AttackExecution::new(
            owner,
            Some("target".to_string()),
            Some(10.0),
        )));
        game.execute_next_tick();

        // nukeDeathFactor(ATOM_BOMB, 10, 1, _) = 5 * 10 / 1 = 50, far exceeding
        // the attack's 10 troops - TS's `AttackImpl.setTroops` clamps at 0.
        game.apply_nuke_deaths_to_deployed_forces(owner, unit_type::ATOM_BOMB, 1.0, 10_000.0);

        let troops_after = game
            .live_attacks()
            .find(|a| a.owner_small_id() == owner)
            .map(|a| a.troops())
            .expect("attack should still be live");
        assert_eq!(troops_after, 0.0);
    }
}

fn list_nuke_break_alliance(
    game: &Game,
    dst: TileRef,
    inner: f64,
    outer: f64,
    threshold: f64,
) -> Vec<u16> {
    let inner2 = (inner * inner) as u32;
    let outer2 = (outer * outer) as u32;
    let mut weight: HashMap<u16, f64> = HashMap::new();

    let cx = game.x(dst) as i64;
    let cy = game.y(dst) as i64;
    let outer_i = outer as i64;
    let min_x = (cx - outer_i).max(0);
    let max_x = (cx + outer_i).min(game.width() as i64 - 1);
    let min_y = (cy - outer_i).max(0);
    let max_y = (cy + outer_i).min(game.height() as i64 - 1);
    for gy in min_y..=max_y {
        for gx in min_x..=max_x {
            let t = game.ref_xy(gx as u32, gy as u32);
            let d2 = game.map.euclidean_dist_squared(dst, t);
            if d2 > outer2 {
                continue;
            }
            let owner = game.map.owner_id(t);
            if owner == 0 {
                continue;
            }
            let w = if d2 <= inner2 { 1.0 } else { 0.5 };
            *weight.entry(owner).or_insert(0.0) += w;
        }
    }

    let mut result: Vec<u16> = Vec::new();
    for (&owner, &w) in weight.iter() {
        if w > threshold {
            result.push(owner);
        }
    }
    for &(owner, ..) in &game.nearby_structures_any(dst, outer as u32, &STRUCTURE_TYPES) {
        if !result.contains(&owner) {
            result.push(owner);
        }
    }
    result
}
