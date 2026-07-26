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
        } else if game.wire.water_nukes() {
            // TS `NukeExecution.tilesToDestroy`'s `waterNukes()` branch: instead
            // of the BFS coin-flip, sample 16 angular radii, smooth them into a
            // gently-undulating boundary, then scan the `outer`-radius bounding
            // box and keep every tile inside that irregular boundary (land AND
            // water - unlike the BFS branch there is no per-tile filter). This
            // game (Hawaii team mode) sets `waterNukes: true`, so native must
            // use this algorithm or every nuke destroys a different tile set
            // (see jdxWdFCt tick-2980 water-nuke bisection).
            water_nuke_tiles(game, dst, inner, outer, tick)
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
            // TS `NukeExecution.detonate`: land tiles go through
            // `queueWaterConversion` (water when `waterNukes`, else fallout).
            if game.is_land(t) {
                game.queue_water_conversion(t);
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

        let mut to_remove_units: Vec<(u16, i32, Option<TileRef>)> = Vec::new();
        for p in game.players_in_order() {
            for u in &p.units {
                if EXCLUDED_FROM_BLAST.contains(&u.unit_type.as_str()) {
                    continue;
                }
                let d2 = game.map.euclidean_dist_squared(dst, u.tile as TileRef);
                if d2 < outer2 {
                    let transport_tile =
                        (u.unit_type == unit_type::TRANSPORT).then(|| u.tile as TileRef);
                    to_remove_units.push((p.small_id, u.id, transport_tile));
                }
            }
        }
        for (sid, uid, transport_tile) in to_remove_units {
            // TS `NukeExecution.detonate`: `unit.delete(true, destroyer)` where `destroyer =
            // this.player` - see `Game::record_transport_kill`'s doc comment for why native
            // needs this recorded before `remove_unit` rather than queried after.
            if let Some(tile) = transport_tile {
                game.record_transport_kill(uid, sid, self.owner_small_id, tile);
            }
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
            let spawn = nuke_spawn(game, owner_small_id, nuke_type, dst)?;
            // Match NukeExecution::spawn: trajectory over impassable aborts
            // before gold/unit spend. Without this, can_build / waste checks
            // treated those launches as successful no-ops.
            if nuke_trajectory_blocked(game, spawn, dst, nuke_type) {
                return None;
            }
            Some(spawn)
        }
        unit_type::ATOM_BOMB | unit_type::HYDROGEN_BOMB => {
            let spawn = nuke_spawn(game, owner_small_id, nuke_type, dst)?;
            if nuke_trajectory_blocked(game, spawn, dst, nuke_type) {
                return None;
            }
            Some(spawn)
        }
        unit_type::MIRV_WARHEAD => Some(dst),
        _ => None,
    }
}

fn nuke_trajectory_blocked(
    game: &Game,
    spawn_tile: TileRef,
    dst: TileRef,
    nuke_type: &str,
) -> bool {
    if nuke_type == unit_type::MIRV_WARHEAD {
        return false;
    }
    let speed = game.wire.default_nuke_speed();
    let distance_based_height = nuke_type != unit_type::MIRV_WARHEAD;
    let trajectory = parabola::find_path_tiles(
        game,
        spawn_tile,
        dst,
        speed,
        distance_based_height,
        true,
    );
    trajectory.is_empty() || trajectory.iter().any(|&t| game.is_impassable(t))
}

// TS `ImpassableTerrain.test.ts` - "Nukes: targeting" / "Nukes: blast
// radius" / "Nukes: trajectory" describe blocks. Found and fixed three
// related real bugs in this file (see each test's doc comment for which
// one it catches): `nuke_spawn` missing an `is_impassable(dst)` guard,
// `NukeExecution::detonate`'s blast BFS missing the `!is_impassable`
// exclusion, and `NukeExecution::spawn` never checking the flight
// trajectory for impassable terrain before launch.
#[cfg(test)]
mod impassable_terrain_tests {
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
        assert!(
            can_build_nuke(&game, 1, unit_type::ATOM_BOMB, target).is_none(),
            "can_build_nuke must reject trajectory-blocked targets so RL waste counts them"
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

/// Ported from `openfront/tests/nukes/WaterNukes.test.ts`.
///
/// Per `docs/PARITY_PLAYBOOK.md`, prefer porting the TS unit suite for
/// subsystem completeness over chasing full-game bisect %. Archived
/// `waterNukes:true` fixtures from older OpenFront pins (e.g. `f0da4182`)
/// diverge early against current native for neighbor-order / impassable-
/// filter reasons unrelated to water conversion.
///
/// Note: TS `TestConfig.nukeMagnitudes` forces `{inner:1, outer:1}`; native
/// uses production AtomBomb magnitudes `(12, 30)`. Assertions below match
/// the TS behavior under production magnitudes (crater + shoreline + graph
/// rebuild), not the exact TestConfig geometry.
#[cfg(test)]
mod water_nukes_tests {
    use super::*;
    use crate::execution::exec_enum::ExecEnum;
    use crate::game::{Game, Player, PlayerType};
    use crate::map::{GameMap, MapMeta};
    use crate::test_util::plains_game;

    fn enable_water_nukes(game: &mut Game, on: bool) {
        let mut cfg = game.wire.game_config().clone();
        cfg.water_nukes = Some(on);
        cfg.instant_build = true;
        cfg.infinite_gold = true;
        // TS TestConfig defaults spawnImmunityDuration to 0; production is 50.
        cfg.spawn_immunity_duration = Some(0);
        game.wire = crate::core::config::Config::new(cfg, false);
    }

    /// Half-resolution mini-map (TS `TerrainMapLoader`), required for the
    /// `waterGraphVersion` test's mini-tile majority-water flip path.
    fn attach_half_mini_map(game: &mut Game) {
        let mw = game.map.width / 2;
        let mh = game.map.height / 2;
        let n = (mw * mh) as usize;
        let meta = MapMeta {
            width: mw,
            height: mh,
            num_land_tiles: n as u32,
        };
        game.mini_map = GameMap::from_terrain_bytes(&meta, &vec![0x80u8; n]).unwrap();
        game.mini_water_astar = crate::water::WaterAstarScratch::new(n);
        game.mini_water_hpa = Some(crate::water_hpa::WaterHierarchical::new(&game.mini_map, true));
    }

    fn add_human(game: &mut Game, small_id: u16) {
        game.add_player(Player {
            id: format!("p{small_id}"),
            small_id,
            player_type: PlayerType::Human,
            gold: 1_000_000_000,
            ..Default::default()
        });
    }

    fn silo_game(width: u32, height: u32, water_nukes: bool) -> (Game, u16, TileRef) {
        let mut game = plains_game(width, height);
        enable_water_nukes(&mut game, water_nukes);
        add_human(&mut game, 1);
        let home = game.map.ref_xy(1, 1);
        game.conquer(1, home);
        game.build_unit(1, unit_type::MISSILE_SILO, home);
        (game, 1, home)
    }

    /// TS `launchNukeAt` + `tickUntilNukeLands` (and the graph-version test's
    /// 80-tick window so the 20-tick rebuild throttle can fire).
    fn launch_and_detonate(game: &mut Game, owner: u16, target: TileRef) {
        assert!(
            can_build_nuke(game, owner, unit_type::ATOM_BOMB, target).is_some(),
            "can_build_nuke must succeed (spawn immunity must be off)"
        );
        game.add_execution(ExecEnum::Nuke(NukeExecution::new(
            unit_type::ATOM_BOMB,
            owner,
            target,
            None,
            -1.0,
            0,
            true,
        )));
        for _ in 0..80 {
            game.execute_next_tick();
        }
    }

    #[test]
    fn water_nukes_convert_land_to_water_instead_of_fallout() {
        let (mut game, owner, _) = silo_game(64, 64, true);
        let target = game.map.ref_xy(10, 10);
        assert!(game.is_land(target));
        assert!(!game.has_fallout(target));

        launch_and_detonate(&mut game, owner, target);

        assert!(game.is_water(target), "target must become water");
        assert!(!game.is_land(target));
        assert!(
            !game.has_fallout(target),
            "waterNukes path must not paint fallout"
        );
    }

    #[test]
    fn water_nukes_update_shoreline_bits_around_the_crater() {
        let (mut game, owner, _) = silo_game(64, 64, true);
        let target = game.map.ref_xy(10, 10);

        launch_and_detonate(&mut game, owner, target);

        // Production AtomBomb outer=30; land adjacent to converted water
        // must pick up shoreline bits (TS checks dist-2 under TestConfig
        // magnitudes {1,1}).
        let mut shoreline_land = 0u32;
        for y in 0..64u32 {
            for x in 0..64u32 {
                let t = game.map.ref_xy(x, y);
                if game.is_land(t) && game.map.is_shoreline(t) {
                    shoreline_land += 1;
                }
            }
        }
        assert!(
            shoreline_land > 0,
            "converted crater must leave shoreline on surrounding land"
        );
    }

    #[test]
    fn queue_water_conversion_skips_tiles_conquered_before_flush() {
        let mut game = plains_game(32, 32);
        enable_water_nukes(&mut game, true);
        add_human(&mut game, 1);
        let target = game.map.ref_xy(10, 10);
        assert!(game.is_land(target));
        assert!(!game.has_owner(target));

        game.queue_water_conversion(target);
        game.conquer(1, target);
        assert!(game.has_owner(target));

        game.execute_next_tick();

        assert!(game.is_land(target), "owned tile must stay land");
        assert!(game.has_owner(target));
        assert!(!game.is_water(target));
    }

    #[test]
    fn water_graph_version_increments_after_water_conversion() {
        let (mut game, owner, _) = silo_game(64, 64, true);
        attach_half_mini_map(&mut game);
        let target = game.map.ref_xy(30, 30);
        let before = game.water_graph_version();

        launch_and_detonate(&mut game, owner, target);

        assert!(
            game.water_graph_version() > before,
            "mini water-graph rebuild must bump water_graph_version after a water nuke"
        );
    }

    #[test]
    fn without_water_nukes_nuke_applies_fallout_not_water() {
        let (mut game, owner, _) = silo_game(64, 64, false);
        let target = game.map.ref_xy(10, 10);
        let before = game.water_graph_version();

        launch_and_detonate(&mut game, owner, target);

        assert!(game.is_land(target), "default path keeps land");
        assert!(game.has_fallout(target), "default path paints fallout");
        assert_eq!(
            game.water_graph_version(),
            before,
            "fallout path must not rebuild the water graph"
        );
    }
}

/// TS `NukeExecution.tilesToDestroy`'s `waterNukes()` branch (pin f0da4182).
///
/// Samples `NUM_SAMPLES` angular radii uniformly in `[inner2, outer2)`, applies
/// one light smoothing pass, then scans the `outer`-radius bounding box and
/// keeps every tile whose squared distance is within the angularly-interpolated
/// boundary. Tiles are emitted in row-major (`py` then `px`) order, matching the
/// insertion order of TS's result `Set`. `inner`/`outer` are the nuke's
/// magnitudes (not yet squared).
fn water_nuke_tiles(game: &Game, dst: TileRef, inner: f64, outer: f64, tick: u32) -> Vec<TileRef> {
    const NUM_SAMPLES: usize = 16;
    let inner2 = inner * inner;
    let outer2 = outer * outer;

    let mut rand = PseudoRandom::new(tick as i32);
    let mut radii_sq = [0.0f64; NUM_SAMPLES];
    for r in radii_sq.iter_mut() {
        *r = rand.next_float(inner2, outer2);
    }
    let prev = radii_sq;
    for i in 0..NUM_SAMPLES {
        let l = (i + NUM_SAMPLES - 1) % NUM_SAMPLES;
        let r = (i + 1) % NUM_SAMPLES;
        radii_sq[i] = prev[i] * 0.6 + prev[l] * 0.2 + prev[r] * 0.2;
    }

    let cx = game.x(dst) as i32;
    let cy = game.y(dst) as i32;
    let outer_i = outer as i32;
    let width = game.width() as i32;
    let height = game.height() as i32;
    let x0 = (cx - outer_i).max(0);
    let y0 = (cy - outer_i).max(0);
    let x1 = (cx + outer_i).min(width - 1);
    let y1 = (cy + outer_i).min(height - 1);

    let two_pi = std::f64::consts::PI * 2.0;
    let mut result = Vec::new();
    let mut py = y0;
    while py <= y1 {
        let mut px = x0;
        while px <= x1 {
            let dx = px - cx;
            let dy = py - cy;
            let d2 = (dx * dx + dy * dy) as f64;
            if d2 > outer2 {
                px += 1;
                continue;
            }
            if d2 > inner2 {
                let angle = (dy as f64).atan2(dx as f64) + std::f64::consts::PI;
                let t = (angle / two_pi) * NUM_SAMPLES as f64;
                let i0 = (t.floor() as usize) % NUM_SAMPLES;
                let i1 = (i0 + 1) % NUM_SAMPLES;
                let frac = t - t.floor();
                let threshold = radii_sq[i0] * (1.0 - frac) + radii_sq[i1] * frac;
                if d2 > threshold {
                    px += 1;
                    continue;
                }
            }
            result.push(game.ref_xy(px as u32, py as u32));
            px += 1;
        }
        py += 1;
    }
    result
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

    // Ported from AllianceRequestExecution.test.ts "alliance request is revoked
    // immediately if requester launches a nuke" (fix for
    // github.com/openfrontio/OpenFrontIO/issues/2071). The TS test forces this
    // by monkeypatching `nukeAllianceBreakThreshold` to 0 on the live config
    // instance so the effect fires without needing >100 weighted tiles in the
    // blast; native hardcodes this threshold at the same default value TS ships
    // with (100, see `WireConfig::nuke_alliance_break_threshold`) and has no
    // per-instance override, so instead of faking the threshold this exercises
    // the *other*, threshold-independent inclusion path also present in TS's
    // `Util.listNukeBreakAlliance`: any player with a structure inside the
    // blast outer radius is included in `targets` regardless of tile-ownership
    // weight - built here via a City exactly at the nuke's destination tile.
    #[test]
    fn nuke_at_a_players_structure_revokes_the_nukers_pending_alliance_request() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let nuker = add_bot(&mut game, "nuker");
        let target = add_bot(&mut game, "target");

        assert!(game.create_alliance_request(nuker, target, game.ticks()));
        assert_eq!(game.outgoing_alliance_requests(nuker), vec![target]);

        let dst = game.map.ref_xy(0, 0);
        game.build_unit(target, unit_type::CITY, dst);

        maybe_break_alliances(&mut game, nuker, dst, unit_type::ATOM_BOMB);

        assert_eq!(game.outgoing_alliance_requests(nuker).len(), 0);
        assert!(!game.is_allied_with(nuker, target));
        assert!(!game.is_allied_with(target, nuker));
    }

    // Ported from Attack.test.ts's "Can't send nuke during immunity phase":
    // TS `PlayerImpl.nukeSpawn` refuses to spawn any nuke while
    // `mg.isSpawnImmunityActive()` (a global window, not per-player - see
    // `Game::is_spawn_immunity_active`), independent of `canAttackPlayer`'s
    // separate per-defender-type immunity check exercised by `attack.rs`'s
    // `immunity_tests`. Native's `nuke_spawn` (called from `can_build_nuke`)
    // already had this gate; this is new coverage for it, not a fix.
    #[test]
    fn cannot_build_a_nuke_during_spawn_immunity_but_can_after_it_ends() {
        let mut game = Game::default();
        game.end_spawn_phase();
        // `Human`, not `Bot`: bots run autonomous tribe AI on every tick
        // (spending gold on their own builds) that would otherwise interfere
        // with the plain gold-balance check this test cares about.
        let owner = game.add_from_info(&PlayerInfo {
            name: "owner".into(),
            player_type: PlayerType::Human,
            client_id: Some("owner".into()),
            id: "owner".into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        });
        if let Some(p) = game.player_by_small_id_mut(owner) {
            p.gold = 10_000_000;
            p.tiles_owned = 1;
        }
        let dst = game.map.ref_xy(0, 0);
        game.build_unit(owner, unit_type::MISSILE_SILO, dst);

        assert!(can_build_nuke(&game, owner, unit_type::ATOM_BOMB, dst).is_none());

        for _ in 0..game.wire.spawn_immunity_duration() + 1 {
            game.execute_next_tick();
        }
        assert!(can_build_nuke(&game, owner, unit_type::ATOM_BOMB, dst).is_some());
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
