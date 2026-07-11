//! Warship spawn and patrol movement (`WarshipExecution.ts` subset).

use super::Execution;
use crate::core::schemas::unit_type::{PORT, TRADE_SHIP, TRANSPORT, WARSHIP};
use crate::execution::{ExecEnum, ShellExecution};
use crate::game::Game;
use crate::map::TileRef;
use crate::prng::PseudoRandom;
use std::collections::HashSet;

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
        if !game.is_water(self.patrol_tile) {
            return None;
        }
        let component = game.get_water_component(self.patrol_tile)?;
        game.player_by_small_id(self.owner_small_id)?
            .units
            .iter()
            .filter(|unit| unit.unit_type == PORT && !unit.under_construction)
            .filter(|unit| game.has_water_component(unit.tile as TileRef, component))
            .min_by_key(|unit| game.manhattan_dist(unit.tile as TileRef, self.patrol_tile))
            .map(|unit| unit.tile as TileRef)
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

    fn heal_near_port(&self, game: &mut Game, from: TileRef, unit_id: i32) {
        let near_port = game
            .player_by_small_id(self.owner_small_id)
            .is_some_and(|owner| {
                owner.units.iter().any(|unit| {
                    unit.unit_type == PORT
                        && game.map.euclidean_dist_squared(from, unit.tile as TileRef) <= 150 * 150
                })
            });
        if near_port {
            if let Some(unit) = game.unit_mut(self.owner_small_id, unit_id) {
                unit.health = (unit.health + 1).min(1000);
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
            if let Some(unit) = game.unit_mut(self.owner_small_id, unit_id) {
                unit.health = (unit.health + healing).min(1000);
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
                let fully_healed = game
                    .unit(self.owner_small_id, unit_id)
                    .is_none_or(|unit| unit.health >= 1000);
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
        if health_before_healing < 750 {
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
    use crate::game::{Player, PlayerType};

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
}
