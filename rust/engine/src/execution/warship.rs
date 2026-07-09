//! Warship spawn and patrol movement (`WarshipExecution.ts` subset).

use super::Execution;
use crate::core::schemas::unit_type::{PORT, TRANSPORT, WARSHIP};
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
    active: bool,
}

impl WarshipExecution {
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

    fn target(&self, game: &Game, from: TileRef) -> Option<(u16, i32, &'static str)> {
        let types = [TRANSPORT, WARSHIP];
        let mut best: Option<(u16, i32, &'static str, usize, f64)> = None;
        for (owner, unit_id, _, dist_squared) in game.nearby_structures_any(from, 130, &types) {
            if owner == self.owner_small_id
                || !game.can_attack_player(self.owner_small_id, owner)
                || self.already_sent_shell.contains(&(owner, unit_id))
            {
                continue;
            }
            let Some(unit_type) = game.unit_type_of(owner, unit_id) else {
                continue;
            };
            let (unit_type, priority) = if unit_type == TRANSPORT {
                (TRANSPORT, 0)
            } else if unit_type == WARSHIP {
                (WARSHIP, 1)
            } else {
                continue;
            };
            if best.as_ref().is_none_or(|candidate| {
                priority < candidate.3
                    || (priority == candidate.3 && dist_squared < candidate.4)
            }) {
                best = Some((owner, unit_id, unit_type, priority, dist_squared));
            }
        }
        best.map(|(owner, unit_id, unit_type, _, _)| (owner, unit_id, unit_type))
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

        if let Some(target) = self.target(game, from) {
            self.shoot_target(game, game.ticks(), from, unit_id, target);
        }
        if game.map.euclidean_dist_squared(from, port) <= 25 {
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
        if let Some(target) = self.target(game, from) {
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
