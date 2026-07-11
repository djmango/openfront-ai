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

        let target_range_sq = game.wire.default_nuke_targetable_range().powi(2);
        let trajectory = parabola::find_path_tiles(
            game,
            spawn_tile,
            self.dst,
            self.speed,
            self.distance_based_height(),
            self.rocket_direction_up,
        );
        // TS `NukeExecution.tick`: "Nuke trajectories cannot pass over
        // impassable terrain, just as they cannot exceed the map border" -
        // the full parabola path is checked BEFORE launch (no gold spent, no
        // unit built) and the launch is aborted if any tile is impassable.
        // Native previously never checked this at all, letting nukes fly
        // straight through impassable walls.
        if trajectory.iter().any(|&t| game.is_impassable(t)) {
            self.active = false;
            return;
        }

        self.src = Some(spawn_tile);
        let id = game.build_unit(self.owner_small_id, &self.nuke_type, spawn_tile);
        self.nuke_unit_id = Some(id);

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
                // TS `NukeExecution.tilesToDestroy`: `d2 <= outer2 && (d2 <=
                // inner2 || rand.chance(2)) && !this.mg.isImpassable(n)` -
                // impassable tiles are excluded from the destroy set itself
                // (not just "solid" against later floods), so they never
                // get flagged with fallout. Native was missing the
                // `!isImpassable` term.
                d2 <= outer2 && (d2 <= inner2 || rand_cell.borrow_mut().chance(2)) && !gm.is_impassable(n)
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

// TS `ImpassableTerrain.test.ts` - "Nukes: targeting" / "Nukes: blast
// radius" / "Nukes: trajectory" describe blocks. Found and fixed three
// related real bugs in this file (see each test's doc comment for which
// one it catches): `nuke_spawn` missing an `is_impassable(dst)` guard,
// `NukeExecution::detonate`'s blast BFS missing the `!is_impassable`
// exclusion, and `NukeExecution::spawn` never checking the flight
// trajectory for impassable terrain before launch.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Game, Player, PlayerType};

    const WALL_X: u32 = 30;

    fn wall_game() -> Game {
        crate::test_util::walled_game(60, 20, Some((WALL_X, 2)))
    }

    fn add_bot(game: &mut Game, id: &str, small_id: u16) {
        game.add_player(Player {
            id: id.to_string(),
            small_id,
            player_type: PlayerType::Bot,
            gold: 1_000_000_000,
            ..Default::default()
        });
    }

    fn run_to_completion(nuke: &mut NukeExecution, game: &mut Game, max_ticks: u32) {
        for tick in 0..max_ticks {
            if !nuke.is_active() {
                break;
            }
            nuke.tick(game, tick);
        }
    }

    #[test]
    fn can_build_atom_bomb_returns_none_for_impassable_target() {
        let mut game = wall_game();
        add_bot(&mut game, "player", 1);
        let home = game.map.ref_xy(10, 10);
        game.conquer(1, home);
        game.build_unit(1, unit_type::MISSILE_SILO, home);
        let target = game.map.ref_xy(WALL_X, 10);
        assert!(can_build_nuke(&game, 1, unit_type::ATOM_BOMB, target).is_none());
    }

    #[test]
    fn can_build_mirv_returns_none_for_impassable_target() {
        let mut game = wall_game();
        add_bot(&mut game, "player", 1);
        let home = game.map.ref_xy(10, 10);
        game.conquer(1, home);
        game.build_unit(1, unit_type::MISSILE_SILO, home);
        let target = game.map.ref_xy(WALL_X, 10);
        assert!(can_build_nuke(&game, 1, unit_type::MIRV, target).is_none());
    }

    #[test]
    fn nuke_execution_deactivates_when_targeting_impassable_tile() {
        let mut game = wall_game();
        add_bot(&mut game, "player", 1);
        let home = game.map.ref_xy(10, 10);
        game.conquer(1, home);
        game.build_unit(1, unit_type::MISSILE_SILO, home);

        let target = game.map.ref_xy(WALL_X, 10);
        let mut nuke = NukeExecution::new(unit_type::ATOM_BOMB, 1, target, None, -1.0, 0, true);
        nuke.init(&mut game, 0);
        run_to_completion(&mut nuke, &mut game, 5);

        assert!(!nuke.is_active());
        // No gold spent, no unit built (TS never even attempts the build).
        assert_eq!(
            game.player_by_small_id(1)
                .unwrap()
                .units
                .iter()
                .filter(|u| u.unit_type == unit_type::ATOM_BOMB)
                .count(),
            0
        );
    }

    /// Catches the missing `!is_impassable(n)` term in `detonate`'s blast
    /// BFS filter: before the fix, a wall tile within blast radius got
    /// `set_fallout(true)`, which TS's `tilesToDestroy()` (which excludes
    /// impassable tiles from the set entirely) never allows.
    #[test]
    fn nuke_blast_does_not_set_fallout_on_impassable_tiles() {
        let mut game = wall_game();
        add_bot(&mut game, "player", 1);
        add_bot(&mut game, "other", 2);
        let home = game.map.ref_xy(10, 10);
        game.conquer(1, home);
        game.build_unit(1, unit_type::MISSILE_SILO, home);
        let target = game.map.ref_xy(WALL_X - 1, 10);
        game.conquer(2, target);

        let mut nuke = NukeExecution::new(unit_type::ATOM_BOMB, 1, target, None, -1.0, 0, true);
        nuke.init(&mut game, 0);
        run_to_completion(&mut nuke, &mut game, 60);
        assert!(!nuke.is_active(), "nuke should have detonated");

        for y in 5..=15 {
            let t = game.map.ref_xy(WALL_X, y);
            assert!(game.is_land(t));
            assert!(game.is_impassable(t));
            assert!(
                !game.map.has_fallout(t),
                "impassable tile must never receive fallout from a nuke blast"
            );
        }
    }

    /// Catches `NukeExecution::spawn` never checking the flight path for
    /// impassable terrain: before the fix, a nuke would build and fly
    /// straight through the wall to its target.
    #[test]
    fn nuke_trajectory_blocked_by_impassable_terrain_aborts_launch() {
        let mut game = wall_game();
        add_bot(&mut game, "player", 1);
        let home = game.map.ref_xy(5, 10);
        game.conquer(1, home);
        game.build_unit(1, unit_type::MISSILE_SILO, home);
        // Target is on the right side of the wall - trajectory must cross it.
        let target = game.map.ref_xy(50, 10);
        assert!(!game.is_impassable(target));

        let mut nuke = NukeExecution::new(unit_type::ATOM_BOMB, 1, target, None, -1.0, 0, true);
        nuke.init(&mut game, 0);
        run_to_completion(&mut nuke, &mut game, 10);

        assert!(!nuke.is_active(), "should have been blocked");
        assert_eq!(
            game.player_by_small_id(1)
                .unwrap()
                .units
                .iter()
                .filter(|u| u.unit_type == unit_type::ATOM_BOMB)
                .count(),
            0,
            "a blocked launch must not build a nuke unit"
        );
    }

    #[test]
    fn nuke_can_launch_when_trajectory_does_not_cross_impassable_terrain() {
        let mut game = wall_game();
        add_bot(&mut game, "player", 1);
        let home = game.map.ref_xy(5, 10);
        game.conquer(1, home);
        game.build_unit(1, unit_type::MISSILE_SILO, home);
        // Target is on the same (left) side - no impassable terrain in between.
        let target = game.map.ref_xy(15, 10);
        assert!(!game.is_impassable(target));

        let mut nuke = NukeExecution::new(unit_type::ATOM_BOMB, 1, target, None, -1.0, 0, true);
        nuke.init(&mut game, 0);
        run_to_completion(&mut nuke, &mut game, 60);

        assert!(!nuke.is_active(), "should have detonated and deactivated normally");
    }
}

fn nuke_spawn(game: &Game, owner_small_id: u16, nuke_type: &str, dst: TileRef) -> Option<TileRef> {
    if game.is_spawn_immunity_active() {
        return None;
    }
    // TS `PlayerImpl.nukeSpawn`: "Impassable terrain cannot be nuked."
    // Native was missing this guard entirely, so `canBuild(AtomBomb/MIRV,
    // impassableTile)` would incorrectly succeed.
    if game.is_impassable(dst) {
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
