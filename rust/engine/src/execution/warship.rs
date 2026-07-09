//! Warship spawn and patrol movement (`WarshipExecution.ts` subset).

use super::Execution;
use crate::core::schemas::unit_type::{PORT, WARSHIP};
use crate::game::Game;
use crate::map::TileRef;
use crate::prng::PseudoRandom;

pub struct WarshipExecution {
    owner_small_id: u16,
    patrol_tile: TileRef,
    unit_id: Option<i32>,
    random: Option<PseudoRandom>,
    target_tile: Option<TileRef>,
    path: Vec<TileRef>,
    path_idx: usize,
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

    fn tick(&mut self, game: &mut Game, _tick: u32) {
        let Some(unit_id) = self.unit_id else {
            self.active = false;
            return;
        };
        let Some(from) = game.unit_tile_of(self.owner_small_id, unit_id) else {
            self.active = false;
            return;
        };

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
