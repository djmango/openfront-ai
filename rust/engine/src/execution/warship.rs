//! Warship spawn and patrol movement (`WarshipExecution.ts` subset).

use super::Execution;
use crate::core::schemas::unit_type::{PORT, TRADE_SHIP, TRANSPORT, WARSHIP};
use crate::execution::{ExecEnum, ShellExecution};
use crate::game::Game;
use crate::map::TileRef;
use crate::prng::PseudoRandom;
use std::collections::HashSet;

/// TS `PlayerImpl.warshipSpawn(tile)` (the `canBuild(Warship, tile)` case): nearest active,
/// not-under-construction port of `small_id` sharing `tile`'s water component, or `None`
/// (`false` in TS) if none exists. Shared by `WarshipExecution::spawn_tile` (this is what a
/// freshly `ConstructionExecution`-delegated warship actually spawns from) and
/// `NationWarshipBehavior`'s several AI-level `canBuild` checks (`warship_ai.rs`),
/// which need the identical check *before* deciding whether to queue a
/// `ConstructionExecution` at all, without an existing `WarshipExecution` to call it on.
pub fn warship_build_port_tile(game: &Game, small_id: u16, tile: TileRef) -> Option<TileRef> {
    if !game.is_water(tile) {
        return None;
    }
    let component = game.get_water_component(tile)?;
    game.player_by_small_id(small_id)?
        .units
        .iter()
        .filter(|unit| unit.unit_type == PORT && !unit.under_construction)
        .filter(|unit| game.has_water_component(unit.tile as TileRef, component))
        .min_by_key(|unit| game.manhattan_dist(unit.tile as TileRef, tile))
        .map(|unit| unit.tile as TileRef)
}

/// TS `PlayerImpl.canBuild(Warship, tile)`: `canBuildUnitType` (disabled/gold/alive) then
/// `canSpawnUnitType` → `warshipSpawn(tile)` (nearest port sharing `tile`'s water component).
///
/// `ConstructionExecution` never gates Warship on cost (it isn't a structure), so this is the
/// *only* gold/buildability gate a warship goes through — both when AI decides to queue one
/// (`warship_ai`) and again in `WarshipExecution::init` (matching TS `WarshipExecution.init`,
/// which re-checks `canBuild` at spawn time after same-tick spends like SAM construction).
pub fn can_build_warship(game: &Game, small_id: u16, tile: TileRef) -> bool {
    if game.wire.is_unit_disabled(WARSHIP) {
        return false;
    }
    let Some(player) = game.player_by_small_id(small_id) else {
        return false;
    };
    if !player.alive {
        return false;
    }
    if player.gold < game.structure_cost(small_id, WARSHIP) {
        return false;
    }
    warship_build_port_tile(game, small_id, tile).is_some()
}

/// TS `NationWarshipBehavior.warshipSpawnTile(portTile, radius)` - a PRNG-consuming random
/// search for *any* water tile within `radius` of `center` (up to 50 attempts, 2 draws
/// each); unrelated to `warship_build_port_tile` above despite the similar TS name (that one
/// searches this nation's own ports, deterministically, with no PRNG draws at all). Used both
/// to pick a rough patrol location near an existing port (`maybeSpawnWarship`) and to pick a
/// rough ocean tile near an enemy transport's landing target (`trackIncomingTransportsAndRetaliate`).
pub fn warship_random_water_tile_near(
    game: &Game,
    random: &mut PseudoRandom,
    center: TileRef,
    radius: i32,
) -> Option<TileRef> {
    let cx = game.x(center) as i32;
    let cy = game.y(center) as i32;
    for _ in 0..50 {
        let rand_x = random.next_int(cx - radius, cx + radius);
        let rand_y = random.next_int(cy - radius, cy + radius);
        if !game.is_valid_coord(rand_x, rand_y) {
            continue;
        }
        let tile = game.ref_xy(rand_x as u32, rand_y as u32);
        if !game.is_water(tile) {
            continue;
        }
        return Some(tile);
    }
    None
}

pub struct WarshipExecution {
    owner_small_id: u16,
    patrol_tile: TileRef,
    unit_id: Option<i32>,
    random: Option<PseudoRandom>,
    target_tile: Option<TileRef>,
    path: Vec<TileRef>,
    path_idx: usize,
    last_shell_attack: u32,
    already_sent_shell: HashSet<(u16, i32)>,
    retreat_port: Option<TileRef>,
    retreating: bool,
    docked: bool,
    hunt_target_tile: Option<TileRef>,
    hunt_path: Vec<TileRef>,
    hunt_path_idx: usize,
    active: bool,
}

impl WarshipExecution {
    pub fn owner_small_id(&self) -> u16 {
        self.owner_small_id
    }

    pub fn unit_id(&self) -> Option<i32> {
        self.unit_id
    }

    pub fn is_docked(&self) -> bool {
        self.docked
    }

    /// TS `Unit.warshipState().patrolTile`, read by `NationWarshipBehavior.maybeMoveWarship`
    /// (via `warship_ai.rs`) to pick the least-busy existing warship to redirect
    /// instead of building a new one.
    pub fn patrol_tile(&self) -> TileRef {
        self.patrol_tile
    }

    /// TS `warship.updateWarshipState({ patrolTile: tile })` - `NationWarshipBehavior`'s
    /// `maybeMoveWarship` redirect. Deliberately does *not* clear `target_tile`/`path`,
    /// matching TS's `patrol()`: an in-progress patrol leg finishes before the next
    /// `randomTile()` call samples around the new patrol tile.
    pub fn set_patrol_tile(&mut self, tile: TileRef) {
        self.patrol_tile = tile;
    }

    /// TS `MoveWarshipExecution.init()`'s per-warship redirect (a manual player move) -
    /// unlike `set_patrol_tile` (`NationWarshipBehavior.maybeMoveWarship`, which lets an
    /// in-progress patrol leg finish), this also clears the in-flight patrol
    /// waypoint/path (TS `warship.setTargetTile(undefined)`) so the ship redirects toward
    /// the new patrol tile immediately instead of finishing its current leg first.
    pub fn retarget_patrol(&mut self, tile: TileRef) {
        self.patrol_tile = tile;
        self.target_tile = None;
        self.path.clear();
        self.path_idx = 0;
    }

    /// Test-only constructor for a warship whose backing `Unit` already exists, bypassing
    /// `init()`'s water-component port lookup (`Game::default()`/synthetic test maps have no
    /// `mini_water_hpa` - see `warship_ai.rs`'s test module doc comment). Mirrors
    /// `Game::push_exec_for_test`'s rationale and naming.
    #[cfg(test)]
    pub(crate) fn new_for_test(owner_small_id: u16, patrol_tile: TileRef, unit_id: i32) -> Self {
        let mut exec = Self::new(owner_small_id, patrol_tile);
        exec.unit_id = Some(unit_id);
        exec
    }

    pub fn new(owner_small_id: u16, patrol_tile: TileRef) -> Self {
        Self {
            owner_small_id,
            patrol_tile,
            unit_id: None,
            random: None,
            target_tile: None,
            path: Vec::with_capacity(128),
            path_idx: 0,
            last_shell_attack: 0,
            already_sent_shell: HashSet::new(),
            retreat_port: None,
            retreating: false,
            docked: false,
            hunt_target_tile: None,
            hunt_path: Vec::new(),
            hunt_path_idx: 0,
            active: true,
        }
    }

    fn spawn_tile(&self, game: &Game) -> Option<TileRef> {
        warship_build_port_tile(game, self.owner_small_id, self.patrol_tile)
    }

    fn random_target(&mut self, game: &Game, from: TileRef) -> Option<TileRef> {
        let component = game.get_water_component(from);
        let random = self.random.as_mut()?;
        let mut patrol_range = 100i32;
        let mut attempts = 0;
        let mut expand_count = 0;

        while expand_count < 3 {
            let x = game.x(self.patrol_tile) as i32
                + random.next_int(-patrol_range / 2, patrol_range / 2);
            let y = game.y(self.patrol_tile) as i32
                + random.next_int(-patrol_range / 2, patrol_range / 2);
            if !game.is_valid_coord(x, y) {
                continue;
            }
            let tile = game.ref_xy(x as u32, y as u32);
            let connected = component.is_none_or(|c| game.has_water_component(tile, c));
            if game.is_water(tile) && !game.map.is_shoreline(tile) && connected {
                return Some(tile);
            }
            attempts += 1;
            if attempts == 500 {
                expand_count += 1;
                attempts = 0;
                patrol_range += patrol_range / 2;
            }
        }
        None
    }

    fn refresh_path(&mut self, game: &mut Game, from: TileRef, to: TileRef) -> bool {
        if !game.plan_water_path(from, to) {
            return false;
        }
        self.path.clear();
        self.path.extend_from_slice(game.planned_water_path());
        if self.path.is_empty() || self.path.first() != Some(&from) {
            self.path.insert(0, from);
        }
        self.path_idx = usize::from(self.path.first() == Some(&from));
        true
    }

    fn target(
        &self,
        game: &Game,
        from: TileRef,
        include_trade_ships: bool,
    ) -> Option<(u16, i32, &'static str)> {
        let types = [TRANSPORT, WARSHIP, TRADE_SHIP];
        let mut best: Option<(u16, i32, &'static str, usize, f64)> = None;
        for (owner, unit_id, unit_tile, dist_squared) in
            game.nearby_structures_any(from, 130, &types)
        {
            // TS `WarshipExecution` filters targets with `canAttackPlayer(owner, true)`
            // (treatAFKFriendly) so a disconnected team mate's ships are never
            // shelled - they only change hands via conquest (`conquer_player`).
            if owner == self.owner_small_id
                || !game.can_attack_player_ex(self.owner_small_id, owner, true)
                || self.already_sent_shell.contains(&(owner, unit_id))
            {
                continue;
            }
            let Some(unit_type) = game.unit_type_of(owner, unit_id) else {
                continue;
            };
            if unit_type == WARSHIP && game.warship_is_docked(owner, unit_id) {
                continue;
            }
            let (unit_type, priority) = if unit_type == TRANSPORT {
                (TRANSPORT, 0)
            } else if unit_type == WARSHIP {
                (WARSHIP, 1)
            } else if unit_type == TRADE_SHIP {
                if !include_trade_ships {
                    continue;
                }
                let owner_has_port = game
                    .player_by_small_id(self.owner_small_id)
                    .is_some_and(|owner| owner.units.iter().any(|unit| unit.unit_type == PORT));
                let destination_owner = game.trade_ship_destination_owner(unit_id);
                let same_water_component = game
                    .get_water_component(from)
                    .is_some_and(|component| game.has_water_component(unit_tile, component));
                if !owner_has_port
                    || game.trade_ship_is_safe_from_pirates(owner, unit_id)
                    || destination_owner.is_none_or(|destination| {
                        destination == self.owner_small_id
                            || game.is_friendly(destination, self.owner_small_id)
                    })
                    || !same_water_component
                    || game.map.euclidean_dist_squared(self.patrol_tile, unit_tile) > 100 * 100
                {
                    continue;
                }
                (TRADE_SHIP, 2)
            } else {
                continue;
            };
            if best.as_ref().is_none_or(|candidate| {
                priority < candidate.3 || (priority == candidate.3 && dist_squared < candidate.4)
            }) {
                best = Some((owner, unit_id, unit_type, priority, dist_squared));
            }
        }
        best.map(|(owner, unit_id, unit_type, _, _)| (owner, unit_id, unit_type))
    }

    fn best_neighbor_toward(&self, game: &Game, from: TileRef, target: TileRef) -> Option<TileRef> {
        let mut best = None;
        let mut best_distance = game.manhattan_dist(from, target);
        game.map.for_each_neighbor4(from, |neighbor| {
            if !game.is_water(neighbor) {
                return;
            }
            let distance = game.manhattan_dist(neighbor, target);
            if distance < best_distance {
                best_distance = distance;
                best = Some(neighbor);
            }
        });
        best
    }

    fn hunt_trade_ship(
        &mut self,
        game: &mut Game,
        unit_id: i32,
        target_owner: u16,
        target_unit_id: i32,
    ) {
        for _ in 0..2 {
            let Some(from) = game.unit_tile_of(self.owner_small_id, unit_id) else {
                return;
            };
            let Some(target_tile) = game.unit_tile_of(target_owner, target_unit_id) else {
                return;
            };
            let distance = game.manhattan_dist(from, target_tile);
            if distance <= 5 {
                game.capture_unit(target_owner, self.owner_small_id, target_unit_id);
                self.hunt_target_tile = None;
                self.hunt_path.clear();
                self.hunt_path_idx = 0;
                return;
            }
            if distance <= 20 {
                if let Some(next) = self.best_neighbor_toward(game, from, target_tile) {
                    game.move_unit(self.owner_small_id, unit_id, next);
                    continue;
                }
            }
            let path_matches_from =
                self.hunt_path_idx > 0 && self.hunt_path.get(self.hunt_path_idx - 1) == Some(&from);
            if self.hunt_target_tile != Some(target_tile) || !path_matches_from {
                if !game.plan_water_path(from, target_tile) {
                    return;
                }
                self.hunt_path.clear();
                self.hunt_path.extend_from_slice(game.planned_water_path());
                if self.hunt_path.first() != Some(&from) {
                    self.hunt_path.insert(0, from);
                }
                self.hunt_path_idx = 1;
                self.hunt_target_tile = Some(target_tile);
            }
            if let Some(&next) = self.hunt_path.get(self.hunt_path_idx) {
                self.hunt_path_idx += 1;
                game.move_unit(self.owner_small_id, unit_id, next);
            }
        }
    }

    fn shoot_target(
        &mut self,
        game: &mut Game,
        tick: u32,
        from: TileRef,
        unit_id: i32,
        target: (u16, i32, &'static str),
    ) {
        if tick - self.last_shell_attack <= 20 {
            return;
        }
        if target.2 != TRANSPORT {
            self.last_shell_attack = tick;
        }
        game.add_execution(ExecEnum::Shell(ShellExecution::new(
            from,
            self.owner_small_id,
            unit_id,
            target.0,
            target.1,
        )));
        if target.2 == TRANSPORT {
            self.already_sent_shell.insert((target.0, target.1));
        }
    }

    /// TS `healWarship()`'s leading `if (owner.inDoomsdayClock()) return;` - a doomed side
    /// can't repair its navy, so `DoomsdayClockExecution`'s decay actually sinks warships
    /// instead of being out-healed at a port. Inert when the mode is off (the mark is never
    /// set), and shared by both `heal_near_port` (passive) and `heal_at_dock` (active
    /// docked healing), matching TS's single early return covering both call sites.
    fn owner_is_doomed(&self, game: &Game) -> bool {
        game.in_doomsday_clock(self.owner_small_id)
    }

    fn heal_near_port(&self, game: &mut Game, from: TileRef, unit_id: i32) {
        if self.owner_is_doomed(game) {
            return;
        }
        let near_port = game
            .player_by_small_id(self.owner_small_id)
            .is_some_and(|owner| {
                owner.units.iter().any(|unit| {
                    unit.unit_type == PORT
                        && game.map.euclidean_dist_squared(from, unit.tile as TileRef) <= 150 * 150
                })
            });
        if near_port {
            let max_health = game.unit_max_health(self.owner_small_id, unit_id);
            if let Some(unit) = game.unit_mut(self.owner_small_id, unit_id) {
                unit.health = (unit.health + 1).min(max_health);
            }
        }
    }

    fn nearest_port(&self, game: &Game, from: TileRef) -> Option<TileRef> {
        let component = game.get_water_component(from)?;
        game.player_by_small_id(self.owner_small_id)?
            .units
            .iter()
            .filter(|unit| {
                unit.unit_type == PORT && game.has_water_component(unit.tile as TileRef, component)
            })
            .min_by_key(|unit| game.map.euclidean_dist_squared(from, unit.tile as TileRef))
            .map(|unit| unit.tile as TileRef)
    }

    fn heal_at_dock(&self, game: &mut Game, unit_id: i32) {
        if self.owner_is_doomed(game) {
            return;
        }
        let healing = self
            .retreat_port
            .and_then(|port| {
                game.player_by_small_id(self.owner_small_id)?
                    .units
                    .iter()
                    .find(|unit| unit.unit_type == PORT && unit.tile as TileRef == port)
                    .map(|unit| unit.level * 5)
            })
            .unwrap_or(0);
        if healing > 0 {
            let max_health = game.unit_max_health(self.owner_small_id, unit_id);
            if let Some(unit) = game.unit_mut(self.owner_small_id, unit_id) {
                unit.health = (unit.health + healing).min(max_health);
            }
        }
    }

    fn retreat(&mut self, game: &mut Game, from: TileRef, unit_id: i32) -> bool {
        let Some(port) = self.retreat_port else {
            self.retreating = false;
            return false;
        };
        let port_exists = game
            .player_by_small_id(self.owner_small_id)
            .is_some_and(|owner| {
                owner
                    .units
                    .iter()
                    .any(|unit| unit.unit_type == PORT && unit.tile as TileRef == port)
            });
        if !port_exists {
            self.retreat_port = self.nearest_port(game, from);
            if self.retreat_port.is_none() {
                self.retreating = false;
                return false;
            }
            return self.retreat(game, from, unit_id);
        }

        if let Some(target) = self.target(game, from, false) {
            self.shoot_target(game, game.ticks(), from, unit_id, target);
        }
        if game.map.euclidean_dist_squared(from, port) <= 25 {
            self.docked = true;
            self.target_tile = None;
            self.path.clear();
            self.path_idx = 0;
            return true;
        }
        if self.target_tile != Some(port) {
            self.target_tile = Some(port);
            if !self.refresh_path(game, from, port) {
                self.retreating = false;
                self.retreat_port = None;
                self.target_tile = None;
                return false;
            }
        }
        if self.path_idx >= self.path.len() {
            return true;
        }
        let next = self.path[self.path_idx];
        self.path_idx += 1;
        game.move_unit(self.owner_small_id, unit_id, next);
        true
    }
}

impl Execution for WarshipExecution {
    fn init(&mut self, game: &mut Game, tick: u32) {
        if !self.active || self.unit_id.is_some() {
            return;
        }
        // TS `WarshipExecution.init`: re-check `canBuild` at spawn time (gold may have been
        // spent earlier this tick by Structure construction / nukes). On failure, warn and
        // leave inactive — do not call `buildUnit` / drive gold negative.
        if !can_build_warship(game, self.owner_small_id, self.patrol_tile) {
            self.active = false;
            return;
        }
        let Some(spawn) = self.spawn_tile(game) else {
            self.active = false;
            return;
        };
        self.random = Some(PseudoRandom::new(tick as i32));
        self.unit_id = Some(game.build_unit(self.owner_small_id, WARSHIP, spawn));
    }

    fn tick(&mut self, game: &mut Game, tick: u32) {
        let Some(unit_id) = self.unit_id else {
            self.active = false;
            return;
        };
        let Some(from) = game.unit_tile_of(self.owner_small_id, unit_id) else {
            self.active = false;
            return;
        };
        let health_before_healing = game
            .unit(self.owner_small_id, unit_id)
            .map(|unit| unit.health)
            .unwrap_or(0);
        self.heal_near_port(game, from, unit_id);

        if self.docked {
            let port_exists = self.retreat_port.is_some_and(|port| {
                game.player_by_small_id(self.owner_small_id)
                    .is_some_and(|owner| {
                        owner
                            .units
                            .iter()
                            .any(|unit| unit.unit_type == PORT && unit.tile as TileRef == port)
                    })
            });
            if !port_exists {
                self.docked = false;
                self.retreating = false;
                self.retreat_port = None;
            } else {
                self.heal_at_dock(game, unit_id);
                let max_health = game.unit_max_health(self.owner_small_id, unit_id);
                let fully_healed = game
                    .unit(self.owner_small_id, unit_id)
                    .is_none_or(|unit| unit.health >= max_health);
                if !fully_healed {
                    return;
                }
                self.docked = false;
                self.retreating = false;
                self.retreat_port = None;
            }
        }

        if self.retreating && self.retreat(game, from, unit_id) {
            return;
        }
        // TS `shouldStartRepairRetreat`: `Math.floor(maxHealth * warshipRetreatHealthPercent() / 100)`.
        let retreat_threshold = game.unit_max_health(self.owner_small_id, unit_id) * 75 / 100;
        if health_before_healing < retreat_threshold {
            if let Some(port) = self.nearest_port(game, from) {
                self.retreating = true;
                self.retreat_port = Some(port);
                self.target_tile = None;
                self.path.clear();
                self.path_idx = 0;
                if self.retreat(game, from, unit_id) {
                    return;
                }
            }
        }
        if let Some(target) = self.target(game, from, true) {
            if target.2 == TRADE_SHIP {
                self.hunt_trade_ship(game, unit_id, target.0, target.1);
                return;
            }
            self.shoot_target(game, tick, from, unit_id, target);
        }

        if self.target_tile.is_none() {
            self.target_tile = self.random_target(game, from);
            let Some(target) = self.target_tile else {
                return;
            };
            if !self.refresh_path(game, from, target) {
                self.target_tile = None;
                return;
            }
        }

        if self.path_idx > 0 && self.path.get(self.path_idx - 1) != Some(&from) {
            let target = self.target_tile.expect("patrol target set above");
            if !self.refresh_path(game, from, target) {
                self.target_tile = None;
                return;
            }
        }

        if self.path_idx >= self.path.len() {
            self.target_tile = None;
            return;
        }
        let next = self.path[self.path_idx];
        self.path_idx += 1;
        game.move_unit(self.owner_small_id, unit_id, next);
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        false
    }
}

    // Ported from Disconnected.test.ts's "Disconnected team member interactions"
    // (the two Warship-vs-teammate-ships cases). Full end-to-end coverage would
    // need a real water map (patrol/pathfinding, ports, spawn tiles) that no
    // native test currently constructs - `target()`'s own filtering is what the
    // TS tests actually assert on, so exercise it directly instead.
    #[cfg(test)]
    mod tests {
    use super::*;
    use crate::core::schemas::unit_type;
    use crate::execution::{Execution, TradeShipExecution};
    use crate::game::{Player, PlayerType, PlayerInfo};
    use crate::map::{GameMap, MapMeta};

    /// All-water `width`x`height` map wrapped in a `Game` with a real `mini_water_hpa`
    /// (unlike `warship_ai.rs`'s `big_water_game`, which only swaps `Game::default()`'s map
    /// field in place and so has no navmesh at all - `get_water_component` always returns
    /// `None` there). Needed for `move_warships` tests, which gate on water-component
    /// membership. Mirrors `test_util::walled_game`'s all-land equivalent.
    fn water_game(width: u32, height: u32) -> Game {
        let n = (width * height) as usize;
        let map = GameMap::from_terrain_bytes(
            &MapMeta {
                width,
                height,
                num_land_tiles: 0,
            },
            &vec![0u8; n],
        )
        .expect("all-water test map");
        let mini_w = width.div_ceil(2);
        let mini_h = height.div_ceil(2);
        let mini_n = (mini_w * mini_h) as usize;
        let mini_map = GameMap::from_terrain_bytes(
            &MapMeta {
                width: mini_w,
                height: mini_h,
                num_land_tiles: 0,
            },
            &vec![0u8; mini_n],
        )
        .expect("all-water test mini map");

        let mut game = Game::default();
        game.map = map.clone();
        game.mini_map = mini_map.clone();
        game.bfs = crate::water::BfsScratch::new(n);
        game.water_astar = crate::water::WaterAstarScratch::new(n);
        game.mini_water_astar = crate::water::WaterAstarScratch::new(mini_n);
        game.mini_water_hpa = Some(crate::water_hpa::WaterHierarchical::new(&mini_map, true));
        game.water_component = crate::water::build_water_components(&map);
        game.end_spawn_phase();
        game
    }

    fn add_human(game: &mut Game, id: &str) -> u16 {
        game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type: PlayerType::Human,
            client_id: Some(id.into()),
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        })
    }

    /// Like `add_human` but `PlayerType::Nation` - sidesteps Human-only spawn immunity
    /// (`can_attack_player_ex`) for combat/target-selection tests that don't care about it.
    fn add_nation(game: &mut Game, id: &str) -> u16 {
        game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type: PlayerType::Nation,
            client_id: Some(id.into()),
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        })
    }

    /// TS `WarshipExecution.init` re-checks `canBuild` at spawn time. Same-tick structure
    /// spends (e.g. SAM construction) can drop gold below warship cost after Construction
    /// already queued the WarshipExecution — native must skip the build (not drive gold
    /// negative / invent an extra unit), matching TS's "Failed to spawn warship" path.
    #[test]
    fn warship_init_skips_build_when_gold_insufficient_after_prior_spend() {
        let mut game = water_game(20, 20);
        let sid = add_nation(&mut game, "mongolia");
        let port = game.ref_xy(5, 5);
        let patrol = game.ref_xy(8, 5);
        // Free port (tests often don't care about port cost).
        let _ = game.build_unit(sid, unit_type::PORT, port);
        let warship_cost = game.structure_cost(sid, unit_type::WARSHIP);
        assert!(warship_cost > 0);
        if let Some(p) = game.player_by_small_id_mut(sid) {
            // Enough at queue time for warship, but after a same-tick SAM-sized spend only
            // half remains — below warship cost.
            p.gold = warship_cost / 2;
            p.alive = true;
        }
        let units_before = game.player_by_small_id(sid).unwrap().units.len();
        let gold_before = game.player_by_small_id(sid).unwrap().gold;

        let mut exec = WarshipExecution::new(sid, patrol);
        exec.init(&mut game, 1);

        assert!(!exec.is_active(), "must deactivate like TS when canBuild fails");
        assert!(exec.unit_id().is_none());
        assert_eq!(
            game.player_by_small_id(sid).unwrap().units.len(),
            units_before,
            "must not spawn a warship unit"
        );
        assert_eq!(
            game.player_by_small_id(sid).unwrap().gold,
            gold_before,
            "must not spend gold on a failed spawn"
        );
        assert!(gold_before >= 0);
    }

    #[test]
    fn warship_init_builds_when_can_build_passes() {
        let mut game = water_game(20, 20);
        let sid = add_nation(&mut game, "mongolia");
        let port = game.ref_xy(5, 5);
        let patrol = game.ref_xy(8, 5);
        let _ = game.build_unit(sid, unit_type::PORT, port);
        let warship_cost = game.structure_cost(sid, unit_type::WARSHIP);
        if let Some(p) = game.player_by_small_id_mut(sid) {
            p.gold = warship_cost;
            p.alive = true;
        }
        let units_before = game.player_by_small_id(sid).unwrap().units.len();

        let mut exec = WarshipExecution::new(sid, patrol);
        exec.init(&mut game, 1);

        assert!(exec.is_active());
        assert!(exec.unit_id().is_some());
        assert_eq!(
            game.player_by_small_id(sid).unwrap().units.len(),
            units_before + 1
        );
        assert_eq!(game.player_by_small_id(sid).unwrap().gold, 0);
    }

    // TS `WarshipMultiSelection.test.ts` (`MoveWarshipExecution`). Ported by calling
    // `Game::move_warships` directly (the native equivalent of constructing a
    // `MoveWarshipExecution` and calling `.init()` on it - TS's own test does the same,
    // never advancing ticks to observe the effect through `WarshipExecution::tick`).
    mod move_warships_tests {
        use super::*;

        fn patrol_tile_of(game: &Game, owner: u16, unit_id: i32) -> TileRef {
            game.warship_patrol_candidates(owner)
                .into_iter()
                .find(|&(id, _, _)| id == unit_id)
                .map(|(_, _, patrol_tile)| patrol_tile)
                .unwrap()
        }

        #[test]
        fn moves_multiple_warships_to_a_shared_target() {
            let mut game = water_game(60, 60);
            let p1 = add_human(&mut game, "p1");
            let tiles = [game.ref_xy(10, 10), game.ref_xy(11, 10), game.ref_xy(12, 10)];
            let ids: Vec<i32> = tiles
                .iter()
                .map(|&t| game.build_unit(p1, unit_type::WARSHIP, t))
                .collect();
            for (&id, &t) in ids.iter().zip(tiles.iter()) {
                game.add_execution(ExecEnum::Warship(WarshipExecution::new_for_test(p1, t, id)));
            }
            game.execute_next_tick(); // moves the freshly-added execs out of the uninit queue.

            let target = game.ref_xy(30, 30);
            game.move_warships(p1, &ids, target);

            for &id in &ids {
                assert_eq!(patrol_tile_of(&game, p1, id), target);
            }
        }

        #[test]
        fn moves_different_warships_to_independent_targets() {
            let mut game = water_game(60, 60);
            let p1 = add_human(&mut game, "p1");
            let t1 = game.ref_xy(10, 10);
            let t2 = game.ref_xy(11, 10);
            let w1 = game.build_unit(p1, unit_type::WARSHIP, t1);
            let w2 = game.build_unit(p1, unit_type::WARSHIP, t2);
            game.add_execution(ExecEnum::Warship(WarshipExecution::new_for_test(p1, t1, w1)));
            game.add_execution(ExecEnum::Warship(WarshipExecution::new_for_test(p1, t2, w2)));
            game.execute_next_tick();

            let target1 = game.ref_xy(20, 25);
            let target2 = game.ref_xy(25, 30);
            game.move_warships(p1, &[w1], target1);
            game.move_warships(p1, &[w2], target2);

            assert_eq!(patrol_tile_of(&game, p1, w1), target1);
            assert_eq!(patrol_tile_of(&game, p1, w2), target2);
        }

        #[test]
        fn enemy_cannot_move_another_players_warship() {
            let mut game = water_game(60, 60);
            let p1 = add_human(&mut game, "p1");
            let p2 = add_human(&mut game, "p2");
            let original_tile = game.ref_xy(10, 10);
            let w1 = game.build_unit(p1, unit_type::WARSHIP, original_tile);
            game.add_execution(ExecEnum::Warship(WarshipExecution::new_for_test(
                p1,
                original_tile,
                w1,
            )));
            game.execute_next_tick();

            // `p2` (not the owner) tries to move `p1`'s warship.
            game.move_warships(p2, &[w1], game.ref_xy(30, 30));

            assert_eq!(patrol_tile_of(&game, p1, w1), original_tile);
        }

        #[test]
        fn does_not_panic_on_a_destroyed_warship() {
            let mut game = water_game(60, 60);
            let p1 = add_human(&mut game, "p1");
            let tile = game.ref_xy(10, 10);
            let w1 = game.build_unit(p1, unit_type::WARSHIP, tile);
            game.add_execution(ExecEnum::Warship(WarshipExecution::new_for_test(p1, tile, w1)));
            game.remove_unit(p1, w1);

            // Must not panic even though the warship no longer exists.
            game.move_warships(p1, &[w1], game.ref_xy(30, 30));
        }

        #[test]
        fn batch_move_does_not_affect_another_players_warship() {
            let mut game = water_game(60, 60);
            let p1 = add_human(&mut game, "p1");
            let p2 = add_human(&mut game, "p2");
            let p1_tile = game.ref_xy(10, 10);
            let p2_tile = game.ref_xy(11, 10);
            let w1 = game.build_unit(p1, unit_type::WARSHIP, p1_tile);
            let w2 = game.build_unit(p2, unit_type::WARSHIP, p2_tile);
            game.add_execution(ExecEnum::Warship(WarshipExecution::new_for_test(p1, p1_tile, w1)));
            game.add_execution(ExecEnum::Warship(WarshipExecution::new_for_test(p2, p2_tile, w2)));
            game.execute_next_tick();

            let target = game.ref_xy(30, 30);
            // p1 tries to move both its own warship and p2's, in one batch.
            game.move_warships(p1, &[w1, w2], target);

            assert_eq!(patrol_tile_of(&game, p1, w1), target);
            assert_eq!(
                patrol_tile_of(&game, p2, w2),
                p2_tile,
                "unchanged - wrong owner"
            );
        }

        #[test]
        fn does_not_move_a_warship_across_disconnected_water_bodies() {
            // Two separate water pools divided by a land strip - `move_warships` should
            // leave a warship's patrol tile untouched if the target lives in a different
            // water component (TS `hasWaterComponent` gate).
            let width = 40u32;
            let height = 20u32;
            let n = (width * height) as usize;
            let mut data = vec![0u8; n]; // all water
            for y in 0..height {
                data[(y * width + (width / 2)) as usize] = 0b1000_0000; // land, plains
            }
            let map = GameMap::from_terrain_bytes(
                &MapMeta {
                    width,
                    height,
                    num_land_tiles: height,
                },
                &data,
            )
            .unwrap();
            let mini_w = width.div_ceil(2);
            let mini_h = height.div_ceil(2);
            let mut mini_data = vec![0u8; (mini_w * mini_h) as usize];
            for y in 0..mini_h {
                mini_data[(y * mini_w + (mini_w / 2)) as usize] = 0b1000_0000;
            }
            let mini_map = GameMap::from_terrain_bytes(
                &MapMeta {
                    width: mini_w,
                    height: mini_h,
                    num_land_tiles: mini_h,
                },
                &mini_data,
            )
            .unwrap();

            let mut game = Game::default();
            game.map = map.clone();
            game.mini_map = mini_map.clone();
            game.bfs = crate::water::BfsScratch::new(n);
            game.water_astar = crate::water::WaterAstarScratch::new(n);
            game.mini_water_astar =
                crate::water::WaterAstarScratch::new((mini_w * mini_h) as usize);
            game.mini_water_hpa = Some(crate::water_hpa::WaterHierarchical::new(&mini_map, true));
            game.water_component = crate::water::build_water_components(&map);
            game.end_spawn_phase();

            let p1 = add_human(&mut game, "p1");
            let west_tile = game.ref_xy(2, 10);
            let east_tile = game.ref_xy(width - 3, 10);
            let w1 = game.build_unit(p1, unit_type::WARSHIP, west_tile);
            game.add_execution(ExecEnum::Warship(WarshipExecution::new_for_test(
                p1, west_tile, w1,
            )));
            game.execute_next_tick();

            game.move_warships(p1, &[w1], east_tile);

            assert_eq!(
                patrol_tile_of(&game, p1, w1),
                west_tile,
                "different water component - no move"
            );
        }

        #[test]
        fn fails_gracefully_if_warship_id_never_existed() {
            let mut game = water_game(60, 60);
            let p1 = add_human(&mut game, "p1");

            // Must not panic even though unit id 123 was never built.
            game.move_warships(p1, &[123], game.ref_xy(30, 30));
        }
    }

    fn team_setup() -> (Game, u16, u16) {
        let mut game = Game::default();
        // `PlayerType::Nation` (rather than `Human`) sidesteps spawn-immunity
        // gating in `can_attack_player_ex` (only Human attackers respect it),
        // which is orthogonal to the AFK-friendly team check under test here.
        game.add_player(Player {
            id: "p1".to_string(),
            small_id: 1,
            player_type: PlayerType::Nation,
            team: Some("CLAN".to_string()),
            ..Default::default()
        });
        game.add_player(Player {
            id: "p2".to_string(),
            small_id: 2,
            player_type: PlayerType::Nation,
            team: Some("CLAN".to_string()),
            ..Default::default()
        });
        (game, 1, 2)
    }

    fn solo_setup() -> (Game, u16) {
        let mut game = Game::default();
        game.add_player(Player {
            id: "p1".to_string(),
            small_id: 1,
            player_type: PlayerType::Human,
            ..Default::default()
        });
        (game, 1)
    }

    /// Like `team_setup` but with no team relation - genuine (attackable) enemies, for
    /// combat/target-selection tests. `PlayerType::Nation` again sidesteps Human-only
    /// spawn immunity, which is orthogonal to what these tests exercise.
    fn two_nations_setup() -> (Game, u16, u16) {
        let mut game = Game::default();
        game.add_player(Player {
            id: "p1".to_string(),
            small_id: 1,
            player_type: PlayerType::Nation,
            ..Default::default()
        });
        game.add_player(Player {
            id: "p2".to_string(),
            small_id: 2,
            player_type: PlayerType::Nation,
            ..Default::default()
        });
        (game, 1, 2)
    }

    fn three_nations_setup() -> (Game, u16, u16, u16) {
        let (mut game, p1, p2) = two_nations_setup();
        game.add_player(Player {
            id: "p3".to_string(),
            small_id: 3,
            player_type: PlayerType::Nation,
            ..Default::default()
        });
        (game, p1, p2, 3)
    }

    // Ported from `Warship.test.ts`'s "Warship heals only if player has port". Drives
    // `WarshipExecution::tick` directly via `new_for_test` (bypasses `init()`'s
    // water-component port lookup - see that constructor's doc comment) since only the
    // passive-healing branch is under test, not spawn/patrol geometry.
    #[test]
    fn warship_heals_only_if_player_has_port() {
        let (mut game, p1) = solo_setup();
        let tile = game.ref_xy(0, 0);
        let port_id = game.build_unit(p1, unit_type::PORT, tile);
        let ship_id = game.build_unit(p1, unit_type::WARSHIP, tile);
        let max_health = game.unit_max_health(p1, ship_id);
        let mut exec = WarshipExecution::new_for_test(p1, tile, ship_id);

        exec.tick(&mut game, 1);
        assert_eq!(game.unit(p1, ship_id).unwrap().health, max_health);

        game.unit_mut(p1, ship_id).unwrap().health -= 10;
        assert_eq!(game.unit(p1, ship_id).unwrap().health, max_health - 10);
        exec.tick(&mut game, 2);
        assert_eq!(game.unit(p1, ship_id).unwrap().health, max_health - 9);

        game.remove_unit(p1, port_id);
        exec.tick(&mut game, 3);
        assert_eq!(
            game.unit(p1, ship_id).unwrap().health,
            max_health - 9,
            "no port nearby means no passive heal"
        );
    }

    // Ported from `Warship.test.ts`'s "Warship does not heal while its owner is doomed
    // (Doomsday Clock)" - pins the bug `owner_is_doomed`'s addition to `heal_near_port`/
    // `heal_at_dock` fixed (native previously had no Doomsday Clock check in either at
    // all, so a doomed side's navy kept out-healing the clock's drain indefinitely).
    #[test]
    fn warship_does_not_heal_while_owner_is_doomed() {
        let (mut game, p1) = solo_setup();
        let tile = game.ref_xy(0, 0);
        game.build_unit(p1, unit_type::PORT, tile);
        let ship_id = game.build_unit(p1, unit_type::WARSHIP, tile);
        let max_health = game.unit_max_health(p1, ship_id);
        let mut exec = WarshipExecution::new_for_test(p1, tile, ship_id);
        exec.tick(&mut game, 1);

        // Damaged next to a port, it heals normally (+1 passive heal per tick).
        game.unit_mut(p1, ship_id).unwrap().health -= 10;
        assert_eq!(game.unit(p1, ship_id).unwrap().health, max_health - 10);
        exec.tick(&mut game, 2);
        assert_eq!(game.unit(p1, ship_id).unwrap().health, max_health - 9);

        // Once the owner is flagged by the clock, healing is suppressed even next to a
        // port, so the decay in `DoomsdayClockExecution` can actually sink the fleet.
        game.enter_doomsday_clock(p1);
        exec.tick(&mut game, 3);
        assert_eq!(game.unit(p1, ship_id).unwrap().health, max_health - 9); // no heal while doomed

        // Climbing back above the bar clears the mark and healing resumes.
        game.clear_doomsday_clock(p1);
        exec.tick(&mut game, 4);
        assert_eq!(game.unit(p1, ship_id).unwrap().health, max_health - 8);
    }

    #[test]
    fn warship_does_not_target_disconnected_team_mates_transport_ship() {
        let (mut game, p1, p2) = team_setup();
        let warship_tile = game.ref_xy(5, 5);
        let transport_tile = game.ref_xy(5, 6);
        game.build_unit(p1, unit_type::WARSHIP, warship_tile);
        let transport_id = game.build_unit(p2, unit_type::TRANSPORT, transport_tile);
        game.player_by_small_id_mut(p2).unwrap().is_disconnected = true;

        let exec = WarshipExecution::new(p1, warship_tile);
        assert!(exec.target(&game, warship_tile, true).is_none());
        // Sanity check: without the team relation, the same transport ship
        // would be a valid target (proves the assertion above is actually
        // exercising the AFK-friendly team check, not e.g. range/type filters).
        game.player_by_small_id_mut(p2).unwrap().team = None;
        assert_eq!(
            exec.target(&game, warship_tile, true),
            Some((p2, transport_id, unit_type::TRANSPORT))
        );
    }

    #[test]
    fn disconnected_team_mates_warship_does_not_target_teams_transport_ship() {
        let (mut game, p1, p2) = team_setup();
        let warship_tile = game.ref_xy(5, 5);
        let transport_tile = game.ref_xy(5, 6);
        game.build_unit(p2, unit_type::WARSHIP, warship_tile);
        let transport_id = game.build_unit(p1, unit_type::TRANSPORT, transport_tile);
        game.player_by_small_id_mut(p2).unwrap().is_disconnected = true;

        let exec = WarshipExecution::new(p2, warship_tile);
        assert!(exec.target(&game, warship_tile, true).is_none());
        game.player_by_small_id_mut(p2).unwrap().team = None;
        assert_eq!(
            exec.target(&game, warship_tile, true),
            Some((p1, transport_id, unit_type::TRANSPORT))
        );
    }

    // Ported from `Warship.test.ts`'s remaining (non-heal, non-`MoveWarshipExecution`)
    // cases. `target()` is private but reachable from this same-file `mod tests` (Rust
    // privacy is module-tree-scoped, not file-scoped) - most of these call it directly
    // instead of driving full `tick()`/shell mechanics, exactly like the two
    // `disconnected_team_mates_*` tests above already do.
    mod target_and_retreat_tests {
        use super::*;

        #[test]
        fn prioritizes_transport_over_warship_target() {
            let (mut game, p1, p2) = two_nations_setup();
            let warship_tile = game.ref_xy(0, 10);
            let enemy_warship_tile = game.ref_xy(0, 11);
            let enemy_transport_tile = game.ref_xy(0, 12);
            game.build_unit(p2, unit_type::WARSHIP, enemy_warship_tile);
            let transport_id = game.build_unit(p2, unit_type::TRANSPORT, enemy_transport_tile);

            let exec = WarshipExecution::new(p1, warship_tile);
            // The transport is farther away than the enemy warship, but type priority
            // (Transport > Warship > TradeShip) always wins over distance.
            assert_eq!(
                exec.target(&game, warship_tile, true),
                Some((p2, transport_id, unit_type::TRANSPORT))
            );
        }

        #[test]
        fn docked_warship_not_targeted_by_enemy_warship() {
            let (mut game, p1, p2) = two_nations_setup();
            game.end_spawn_phase(); // `add_execution` only promotes out of `uninit` post-spawn.
            let tile1 = game.ref_xy(0, 10);
            let tile2 = game.ref_xy(0, 11);
            let w1 = game.build_unit(p1, unit_type::WARSHIP, tile1);
            let mut docked_exec = WarshipExecution::new_for_test(p1, tile1, w1);
            docked_exec.docked = true;
            game.add_execution(ExecEnum::Warship(docked_exec));
            // Promotes the exec out of the uninit queue and into `execs` (its `init()` is
            // a no-op since `unit_id` is already set) - `warship_is_docked` only scans `execs`.
            game.execute_next_tick();

            let exec = WarshipExecution::new(p2, tile2);
            assert!(exec.target(&game, tile2, true).is_none());
        }

        #[test]
        fn does_not_target_trade_ship_if_owner_has_no_port() {
            let (mut game, p1, p2) = two_nations_setup();
            let warship_tile = game.ref_xy(0, 10);
            let trade_ship_tile = game.ref_xy(0, 11);
            game.build_unit(p2, unit_type::TRADE_SHIP, trade_ship_tile);

            let exec = WarshipExecution::new(p1, warship_tile);
            assert!(exec.target(&game, warship_tile, true).is_none());
        }

        #[test]
        fn does_not_target_trade_ship_safe_from_pirates() {
            let (mut game, p1, p2) = two_nations_setup();
            let warship_tile = game.ref_xy(0, 10);
            let trade_ship_tile = game.ref_xy(0, 11);
            game.build_unit(p1, unit_type::PORT, warship_tile);
            // `Game::build_unit` grants every freshly built trade ship a 20-tick pirate
            // immunity window from its own spawn tick (still tick 0 here).
            game.build_unit(p2, unit_type::TRADE_SHIP, trade_ship_tile);

            let exec = WarshipExecution::new(p1, warship_tile);
            assert!(exec.target(&game, warship_tile, true).is_none());
        }

        #[test]
        fn captures_trade_ship_when_conditions_are_met() {
            let mut game = water_game(60, 60);
            let p1 = add_nation(&mut game, "p1"); // sidesteps Human spawn immunity on p2/p3.
            let p2 = add_human(&mut game, "p2");
            let p3 = add_human(&mut game, "p3");

            let warship_tile = game.ref_xy(10, 10);
            let ship_tile = game.ref_xy(11, 10);
            let dst_tile = game.ref_xy(12, 10);

            game.build_unit(p1, unit_type::PORT, warship_tile);
            let ship_id = game.build_unit(p2, unit_type::TRADE_SHIP, ship_tile);
            let dst_port_id = game.build_unit(p3, unit_type::PORT, dst_tile);
            // Past its spawn-tick pirate-immunity window (see the "safe from pirates" test).
            game.unit_mut(p2, ship_id).unwrap().last_safe_from_pirates_tick = -1000;

            game.add_execution(ExecEnum::TradeShip(TradeShipExecution::new_for_test(
                p2,
                dst_port_id,
                ship_id,
            )));
            game.execute_next_tick();

            let exec = WarshipExecution::new(p1, warship_tile);
            assert_eq!(
                exec.target(&game, warship_tile, true),
                Some((p2, ship_id, unit_type::TRADE_SHIP))
            );
        }

        #[test]
        fn does_not_target_trade_ship_outside_patrol_range() {
            // Trade ship is within the general 130-tile targeting radius of the warship's
            // *current* tile, but beyond the 100-tile trade-ship-specific patrol range
            // measured from `patrol_tile` (which is deliberately different from `from` here,
            // to isolate the patrol-range check from the general radius check).
            let mut game = water_game(200, 200);
            let p1 = add_nation(&mut game, "p1");
            let p2 = add_human(&mut game, "p2");
            let p3 = add_human(&mut game, "p3");

            let patrol_tile = game.ref_xy(10, 10);
            let from = game.ref_xy(10, 150);
            let trade_ship_tile = game.ref_xy(10, 155);
            let dst_tile = game.ref_xy(20, 155);

            game.build_unit(p1, unit_type::PORT, from);
            let ship_id = game.build_unit(p2, unit_type::TRADE_SHIP, trade_ship_tile);
            let dst_port_id = game.build_unit(p3, unit_type::PORT, dst_tile);
            game.unit_mut(p2, ship_id).unwrap().last_safe_from_pirates_tick = -1000;
            game.add_execution(ExecEnum::TradeShip(TradeShipExecution::new_for_test(
                p2,
                dst_port_id,
                ship_id,
            )));
            game.execute_next_tick();

            let exec = WarshipExecution::new(p1, patrol_tile);
            assert!(exec.target(&game, from, true).is_none());
        }

        #[test]
        fn does_not_target_trade_ship_in_different_water_component() {
            let (mut game, west_tile, east_tile) = split_water_game();
            let p1 = add_nation(&mut game, "p1");
            let p2 = add_human(&mut game, "p2");
            let p3 = add_human(&mut game, "p3");

            game.build_unit(p1, unit_type::PORT, west_tile);
            let ship_id = game.build_unit(p2, unit_type::TRADE_SHIP, east_tile);
            let dst_port_id = game.build_unit(p3, unit_type::PORT, east_tile);
            game.unit_mut(p2, ship_id).unwrap().last_safe_from_pirates_tick = -1000;
            game.add_execution(ExecEnum::TradeShip(TradeShipExecution::new_for_test(
                p2,
                dst_port_id,
                ship_id,
            )));
            game.execute_next_tick();

            let exec = WarshipExecution::new(p1, west_tile);
            assert!(
                exec.target(&game, west_tile, true).is_none(),
                "trade ship lives in a water component disconnected from the warship's"
            );
        }

        #[test]
        fn hunt_trade_ship_captures_immediately_within_five_tiles() {
            let mut game = water_game(60, 60);
            let p1 = add_nation(&mut game, "p1");
            let p2 = add_human(&mut game, "p2");
            let warship_tile = game.ref_xy(10, 10);
            let trade_tile = game.ref_xy(13, 10); // Manhattan distance 3.
            let ship_id = game.build_unit(p1, unit_type::WARSHIP, warship_tile);
            let trade_id = game.build_unit(p2, unit_type::TRADE_SHIP, trade_tile);

            let mut exec = WarshipExecution::new_for_test(p1, warship_tile, ship_id);
            exec.hunt_trade_ship(&mut game, ship_id, p2, trade_id);

            assert_eq!(game.find_unit_owner(trade_id), Some(p1));
        }

        #[test]
        fn hunt_trade_ship_uses_greedy_pursuit_within_twenty_tiles() {
            let mut game = water_game(60, 60);
            let p1 = add_nation(&mut game, "p1");
            let p2 = add_human(&mut game, "p2");
            let warship_tile = game.ref_xy(10, 10);
            let trade_tile = game.ref_xy(10, 20); // Manhattan distance 10 - greedy range, not instant-capture.
            let ship_id = game.build_unit(p1, unit_type::WARSHIP, warship_tile);
            let trade_id = game.build_unit(p2, unit_type::TRADE_SHIP, trade_tile);

            let mut exec = WarshipExecution::new_for_test(p1, warship_tile, ship_id);
            exec.hunt_trade_ship(&mut game, ship_id, p2, trade_id);

            assert_eq!(
                game.find_unit_owner(trade_id),
                Some(p2),
                "not yet within instant-capture range"
            );
            let new_tile = game.unit_tile_of(p1, ship_id).unwrap();
            let old_dist = game.manhattan_dist(warship_tile, trade_tile);
            let new_dist = game.manhattan_dist(new_tile, trade_tile);
            assert!(
                new_dist < old_dist,
                "greedy neighbor pursuit should close the gap: {old_dist} -> {new_dist}"
            );
        }

        #[test]
        fn active_healing_when_docked_heals_by_port_level_times_five() {
            let (mut game, p1) = solo_setup();
            let port_tile = game.ref_xy(0, 0);
            // Far enough from the port that passive near-port healing (<=150 tiles) is
            // inert, isolating the active docked-healing formula under test.
            let ship_tile = game.ref_xy(0, 200);
            game.build_unit(p1, unit_type::PORT, port_tile);
            let ship_id = game.build_unit(p1, unit_type::WARSHIP, ship_tile);
            game.unit_mut(p1, ship_id).unwrap().health -= 100;
            let health_before = game.unit(p1, ship_id).unwrap().health;

            let mut exec = WarshipExecution::new_for_test(p1, ship_tile, ship_id);
            exec.docked = true;
            exec.retreat_port = Some(port_tile);
            exec.tick(&mut game, 1);

            assert_eq!(
                game.unit(p1, ship_id).unwrap().health,
                health_before + 5,
                "port level (1) * 5"
            );
        }

        #[test]
        fn cancels_docking_if_retreat_port_destroyed() {
            let (mut game, p1) = solo_setup();
            let port_tile = game.ref_xy(0, 0);
            let ship_tile = game.ref_xy(0, 200);
            let port_id = game.build_unit(p1, unit_type::PORT, port_tile);
            let ship_id = game.build_unit(p1, unit_type::WARSHIP, ship_tile);

            let mut exec = WarshipExecution::new_for_test(p1, ship_tile, ship_id);
            exec.docked = true;
            exec.retreating = true;
            exec.retreat_port = Some(port_tile);

            game.remove_unit(p1, port_id);
            exec.tick(&mut game, 1);

            assert!(!exec.docked);
            assert!(!exec.retreating);
            assert!(exec.retreat_port.is_none());
        }

        #[test]
        fn does_not_start_retreat_when_no_port_exists() {
            // A real navmesh (`water_game`, not `solo_setup`'s navmesh-less `Game::default()`)
            // so this isolates "no ports at all" specifically, rather than also hitting
            // `nearest_port`'s `get_water_component(from)?` navmesh-less bailout.
            let mut game = water_game(30, 30);
            let p1 = add_nation(&mut game, "p1");
            let ship_tile = game.ref_xy(10, 10);
            let ship_id = game.build_unit(p1, unit_type::WARSHIP, ship_tile);
            let max_health = game.unit_max_health(p1, ship_id);
            game.unit_mut(p1, ship_id).unwrap().health = max_health * 50 / 100;

            let mut exec = WarshipExecution::new_for_test(p1, ship_tile, ship_id);
            exec.tick(&mut game, 25);

            assert!(!exec.retreating);
            assert!(!exec.docked);
        }

        /// Two water pools divided by a land strip, each wrapped in a real `mini_water_hpa`
        /// (`get_water_component`/`has_water_component` are permissive/no-ops without one -
        /// see `Game::has_water_component`'s "disableNavMesh" fallback - so a real navmesh is
        /// required to genuinely exercise the different-component rejection, not just the
        /// navmesh-less bailout `does_not_start_retreat_when_no_port_exists` already covers).
        /// Mirrors `move_warships_tests`' `does_not_move_a_warship_across_disconnected_water_bodies`.
        fn split_water_game() -> (Game, TileRef, TileRef) {
            let width = 40u32;
            let height = 20u32;
            let n = (width * height) as usize;
            let mut data = vec![0u8; n];
            for y in 0..height {
                data[(y * width + (width / 2)) as usize] = 0b1000_0000; // land, plains
            }
            let map = GameMap::from_terrain_bytes(
                &MapMeta {
                    width,
                    height,
                    num_land_tiles: height,
                },
                &data,
            )
            .expect("split water/land test map");
            let mini_w = width.div_ceil(2);
            let mini_h = height.div_ceil(2);
            let mut mini_data = vec![0u8; (mini_w * mini_h) as usize];
            for y in 0..mini_h {
                mini_data[(y * mini_w + (mini_w / 2)) as usize] = 0b1000_0000;
            }
            let mini_map = GameMap::from_terrain_bytes(
                &MapMeta {
                    width: mini_w,
                    height: mini_h,
                    num_land_tiles: mini_h,
                },
                &mini_data,
            )
            .expect("split water/land test mini map");

            let mut game = Game::default();
            game.map = map.clone();
            game.mini_map = mini_map.clone();
            game.bfs = crate::water::BfsScratch::new(n);
            game.water_astar = crate::water::WaterAstarScratch::new(n);
            game.mini_water_astar =
                crate::water::WaterAstarScratch::new((mini_w * mini_h) as usize);
            game.mini_water_hpa = Some(crate::water_hpa::WaterHierarchical::new(&mini_map, true));
            game.water_component = crate::water::build_water_components(&map);
            game.end_spawn_phase();
            let west = game.ref_xy(2, height / 2);
            let east = game.ref_xy(width - 3, height / 2);
            (game, west, east)
        }

        // Ported from `openfront/tests/ShellRandom.test.ts`'s "Warship shell attacks have
        // random damage" - drives a real `WarshipExecution` through the full engine loop
        // (targeting, firing, shell flight) rather than calling `ShellExecution`'s testing
        // hook directly, to catch the case where automatic firing itself is a silent no-op.
        // The direct PRNG-roll range/distribution/reproducibility scenarios from the same TS
        // file are ported next to `ShellExecution` itself (`shell_execution.rs`'s
        // `shell_random_tests` module) since they don't need a real engine loop.
        #[test]
        fn warship_execution_lands_shells_with_varied_damage_over_real_ticks() {
            let mut game = water_game(30, 30);
            let p1 = add_nation(&mut game, "p1");
            let p2 = add_nation(&mut game, "p2");
            let port_tile = game.ref_xy(10, 10);
            let ship_tile = game.ref_xy(10, 11);
            let enemy_tile = game.ref_xy(10, 12);
            game.build_unit(p1, unit_type::PORT, port_tile);
            game.build_unit(p1, unit_type::WARSHIP, ship_tile);
            let enemy_id = game.build_unit(p2, unit_type::WARSHIP, enemy_tile);
            let max_health = game.unit_max_health(p2, enemy_id);

            game.add_execution(ExecEnum::Warship(WarshipExecution::new(p1, ship_tile)));

            let mut damages = Vec::new();
            for _ in 0..400u32 {
                if damages.len() >= 8 {
                    break;
                }
                let before = game.unit(p2, enemy_id).map(|u| u.health).unwrap_or(0);
                game.execute_next_tick();
                let after = game.unit(p2, enemy_id).map(|u| u.health).unwrap_or(0);
                if after < before {
                    damages.push(before - after);
                    if let Some(u) = game.unit_mut(p2, enemy_id) {
                        u.health = max_health;
                    }
                }
            }

            assert!(!damages.is_empty(), "warship never landed a shell");
            for &d in &damages {
                assert!((200..=300).contains(&d), "d={d}");
            }
            let unique: std::collections::HashSet<_> = damages.iter().collect();
            assert!(
                unique.len() > 1,
                "damage should vary across shots, got {damages:?}"
            );
        }

        #[test]
        fn cancels_retreat_when_port_is_in_different_water_component() {
            let (mut game, west_tile, east_tile) = split_water_game();
            let p1 = add_nation(&mut game, "p1");
            game.build_unit(p1, unit_type::PORT, east_tile);
            let ship_id = game.build_unit(p1, unit_type::WARSHIP, west_tile);
            let max_health = game.unit_max_health(p1, ship_id);
            game.unit_mut(p1, ship_id).unwrap().health = max_health * 50 / 100;

            let mut exec = WarshipExecution::new_for_test(p1, west_tile, ship_id);
            exec.tick(&mut game, 25);

            assert!(
                !exec.retreating,
                "the only port is unreachable by water - retreat never starts"
            );
        }

        #[test]
        fn low_health_warship_retreats_and_fires_at_nearby_enemy_warship() {
            let mut game = water_game(30, 30);
            let p1 = add_nation(&mut game, "p1");
            let p2 = add_nation(&mut game, "p2");
            let base_tile = game.ref_xy(10, 10);
            let enemy_tile = game.ref_xy(10, 11);
            game.build_unit(p1, unit_type::PORT, base_tile);
            let ship_id = game.build_unit(p1, unit_type::WARSHIP, base_tile);
            let enemy_id = game.build_unit(p2, unit_type::WARSHIP, enemy_tile);
            let max_health = game.unit_max_health(p1, ship_id);
            game.unit_mut(p1, ship_id).unwrap().health = max_health * 50 / 100;

            // `shoot_target`'s reload gate compares `game.ticks()` (not the tick argument
            // passed to `exec.tick` below) against `last_shell_attack` - advance the game's
            // own tick counter past the 20-tick reload window first.
            for _ in 0..25 {
                game.execute_next_tick();
            }
            let tick = game.ticks();
            let mut exec = WarshipExecution::new_for_test(p1, base_tile, ship_id);
            exec.tick(&mut game, tick);

            assert!(
                exec.docked,
                "port is at the same tile - retreat docks immediately"
            );
            game.execute_next_tick(); // promotes the queued ShellExecution out of `uninit`.
            game.execute_next_tick(); // shell travels (distance 1) and hits this tick.

            let enemy_health = game.unit(p2, enemy_id).unwrap().health;
            assert!(
                enemy_health < 1000,
                "warship should fire at the nearby enemy while retreating, got {enemy_health}"
            );
        }

        #[test]
        fn retreating_warship_aggroes_nearby_transport() {
            let mut game = water_game(30, 30);
            let p1 = add_nation(&mut game, "p1");
            let p2 = add_nation(&mut game, "p2");
            let base_tile = game.ref_xy(10, 10);
            let enemy_tile = game.ref_xy(10, 11);
            game.build_unit(p1, unit_type::PORT, base_tile);
            let ship_id = game.build_unit(p1, unit_type::WARSHIP, base_tile);
            let transport_id = game.build_unit(p2, unit_type::TRANSPORT, enemy_tile);
            game.unit_mut(p2, transport_id).unwrap().health = 1;
            let max_health = game.unit_max_health(p1, ship_id);
            game.unit_mut(p1, ship_id).unwrap().health = max_health * 50 / 100;

            // `retreat()`'s aggro shot (`self.target(game, from, false)`, run before the
            // docking check) reads `game.ticks()` for `shoot_target`'s reload gate, not
            // the `tick` argument passed to `exec.tick` - advance the real counter first.
            for _ in 0..25 {
                game.execute_next_tick();
            }
            let tick = game.ticks();
            let mut exec = WarshipExecution::new_for_test(p1, base_tile, ship_id);
            exec.tick(&mut game, tick);
            assert!(exec.docked, "port is at the same tile - retreat docks immediately");

            game.execute_next_tick(); // promotes the queued ShellExecution out of `uninit`.
            game.execute_next_tick(); // shell travels (distance 1) and hits this tick.

            assert!(
                game.find_unit_owner(transport_id).is_none(),
                "1-hp transport should be destroyed by the retreat-aggro shell"
            );
        }

        #[test]
        fn retreating_warship_continues_moving_to_port_after_firing_back() {
            // Unlike `low_health_warship_retreats_and_fires_at_nearby_enemy_warship` (which
            // starts already at the port and docks on the very first `retreat()` call), this
            // ship starts outside the docking radius (`euclidean_dist_squared <= 25`), so
            // `retreat()`'s shoot-then-move branches both run in the same call.
            let mut game = water_game(30, 30);
            let p1 = add_nation(&mut game, "p1");
            let p2 = add_nation(&mut game, "p2");
            let port_tile = game.ref_xy(10, 10);
            let ship_tile = game.ref_xy(20, 10);
            let enemy_tile = game.ref_xy(19, 10);
            game.build_unit(p1, unit_type::PORT, port_tile);
            let ship_id = game.build_unit(p1, unit_type::WARSHIP, ship_tile);
            let enemy_id = game.build_unit(p2, unit_type::WARSHIP, enemy_tile);
            let max_health = game.unit_max_health(p1, ship_id);
            game.unit_mut(p1, ship_id).unwrap().health = max_health * 50 / 100;

            for _ in 0..25 {
                game.execute_next_tick();
            }
            let tick = game.ticks();
            let mut exec = WarshipExecution::new_for_test(p1, ship_tile, ship_id);
            exec.tick(&mut game, tick);

            assert!(!exec.docked, "still outside the docking radius");
            assert!(exec.retreating);
            assert_eq!(exec.target_tile, Some(port_tile));
            let moved_tile = game.unit_tile_of(p1, ship_id).unwrap();
            assert_ne!(
                moved_tile, ship_tile,
                "should still move toward port on the same tick it fires back"
            );

            game.execute_next_tick(); // promotes the queued ShellExecution out of `uninit`.
            game.execute_next_tick(); // shell travels (distance 1) and hits this tick.
            let enemy_health = game.unit(p2, enemy_id).unwrap().health;
            assert!(
                enemy_health < 1000,
                "warship should fire at the nearby enemy while still en route, got {enemy_health}"
            );
        }
    }
}
