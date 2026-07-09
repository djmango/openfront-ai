//! Warship shell flight (`ShellExecution.ts`).

use super::Execution;
use crate::core::schemas::unit_type::SHELL;
use crate::game::Game;
use crate::map::TileRef;
use crate::prng::PseudoRandom;

fn air_path(game: &Game, seed: i32, from: TileRef, to: TileRef) -> Vec<TileRef> {
    let mut random = PseudoRandom::new(seed);
    let mut path = vec![from];
    let mut current = from;
    while current != to {
        let x = game.x(current) as i64;
        let y = game.y(current) as i64;
        let dst_x = game.x(to) as i64;
        let dst_y = game.y(to) as i64;
        let mut next_x = x;
        let mut next_y = y;
        let ratio = 1 + (dst_y - y).abs() / ((dst_x - x).abs() + 1);
        if x == dst_x {
            next_y += if y < dst_y { 1 } else { -1 };
        } else if y == dst_y {
            next_x += if x < dst_x { 1 } else { -1 };
        } else if random.chance(ratio as i32) {
            next_x += if x < dst_x { 1 } else { -1 };
        } else {
            next_y += if y < dst_y { 1 } else { -1 };
        }
        current = game.ref_xy(next_x as u32, next_y as u32);
        path.push(current);
    }
    path
}

pub struct ShellExecution {
    spawn: TileRef,
    owner_small_id: u16,
    owner_unit_id: i32,
    target_owner_small_id: u16,
    target_unit_id: i32,
    shell_unit_id: Option<i32>,
    seed: i32,
    damage_random: Option<PseudoRandom>,
    destroy_at_tick: Option<u32>,
    last_target_tile: Option<TileRef>,
    path: Vec<TileRef>,
    path_index: usize,
    active: bool,
}

impl ShellExecution {
    pub fn new(
        spawn: TileRef,
        owner_small_id: u16,
        owner_unit_id: i32,
        target_owner_small_id: u16,
        target_unit_id: i32,
    ) -> Self {
        Self {
            spawn,
            owner_small_id,
            owner_unit_id,
            target_owner_small_id,
            target_unit_id,
            shell_unit_id: None,
            seed: 0,
            damage_random: None,
            destroy_at_tick: None,
            last_target_tile: None,
            path: Vec::new(),
            path_index: 0,
            active: true,
        }
    }

    fn remove_shell(&mut self, game: &mut Game) {
        if let Some(shell_id) = self.shell_unit_id {
            game.remove_unit(self.owner_small_id, shell_id);
        }
        self.active = false;
    }

    fn hit_target(&mut self, game: &mut Game) {
        let roll = self
            .damage_random
            .as_mut()
            .map(|random| random.next_int(1, 6))
            .unwrap_or(1);
        let damage = (roll - 1) * 25 + 200;
        let destroyed =
            if let Some(target) = game.unit_mut(self.target_owner_small_id, self.target_unit_id) {
                target.health = (target.health - damage).max(0);
                target.health == 0
            } else {
                false
            };
        if destroyed {
            game.remove_unit(self.target_owner_small_id, self.target_unit_id);
        }
        self.remove_shell(game);
    }
}

impl Execution for ShellExecution {
    fn init(&mut self, _game: &mut Game, tick: u32) {
        self.seed = tick as i32;
        self.damage_random = Some(PseudoRandom::new(tick as i32));
    }

    fn tick(&mut self, game: &mut Game, tick: u32) {
        if !self.active {
            return;
        }
        if self.shell_unit_id.is_none() {
            self.shell_unit_id = Some(game.build_unit(self.owner_small_id, SHELL, self.spawn));
        }
        let Some(shell_id) = self.shell_unit_id else {
            self.active = false;
            return;
        };
        if !game.unit_exists(self.owner_small_id, shell_id)
            || !game.unit_exists(self.target_owner_small_id, self.target_unit_id)
            || self.owner_small_id == self.target_owner_small_id
            || self.destroy_at_tick.is_some_and(|destroy| tick >= destroy)
        {
            self.remove_shell(game);
            return;
        }

        let owner_alive = game.unit_exists(self.owner_small_id, self.owner_unit_id);
        if !owner_alive && self.destroy_at_tick.is_none() {
            self.destroy_at_tick = Some(tick + 50);
        }
        let target_tile = game
            .unit(self.target_owner_small_id, self.target_unit_id)
            .map(|unit| unit.tile as TileRef)
            .unwrap();
        for _ in 0..3 {
            let current = game
                .unit_tile_of(self.owner_small_id, shell_id)
                .unwrap_or(self.spawn);
            if current == target_tile {
                self.hit_target(game);
                return;
            }
            if self.last_target_tile != Some(target_tile) {
                self.path = air_path(game, self.seed, current, target_tile);
                self.path_index = usize::from(self.path.first() == Some(&current));
                self.last_target_tile = Some(target_tile);
            }
            if self.path_index >= self.path.len() {
                self.hit_target(game);
                return;
            }
            let next = self.path[self.path_index];
            self.path_index += 1;
            game.move_unit(self.owner_small_id, shell_id, next);
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }
}
