//! SAM interceptor missile flight (`SAMMissileExecution.ts`) - straight-line "air" pathing
//! (TS `PathFinder.Air.ts`) toward a precomputed interception tile.

use super::Execution;
use crate::core::schemas::unit_type;
use crate::game::Game;
use crate::map::TileRef;
use crate::prng::PseudoRandom;

/// TS `AirPathFinder.computeNext` - one RNG-driven diagonal/orthogonal step toward `to`.
fn air_compute_next(game: &Game, from: TileRef, to: TileRef, rand: &mut PseudoRandom) -> TileRef {
    let x = game.x(from) as i64;
    let y = game.y(from) as i64;
    let dst_x = game.x(to) as i64;
    let dst_y = game.y(to) as i64;
    if x == dst_x && y == dst_y {
        return to;
    }
    let mut next_x = x;
    let mut next_y = y;
    let ratio = 1 + (dst_y - y).abs() / ((dst_x - x).abs() + 1);
    if x == dst_x {
        next_y += if y < dst_y { 1 } else { -1 };
    } else if y == dst_y {
        next_x += if x < dst_x { 1 } else { -1 };
    } else if rand.chance(ratio as i32) {
        next_x += if x < dst_x { 1 } else { -1 };
    } else {
        next_y += if y < dst_y { 1 } else { -1 };
    }
    game.ref_xy(next_x as u32, next_y as u32)
}

/// TS `AirPathFinder.findPath` - full deterministic walk, computed once and cached (the
/// stepper never rebuilds since the SAM missile's target tile never changes).
fn air_find_path(game: &Game, seed: i32, from: TileRef, to: TileRef) -> Vec<TileRef> {
    let mut rand = PseudoRandom::new(seed);
    let mut path = vec![from];
    let mut current = from;
    while current != to {
        let next = air_compute_next(game, current, to, &mut rand);
        if next == current {
            break;
        }
        current = next;
        path.push(current);
    }
    path
}

pub struct SamMissileExecution {
    spawn: TileRef,
    owner_small_id: u16,
    /// Launching SAM Launcher unit id (must stay alive/active for the missile to keep flying).
    sam_unit_id: i32,
    target_owner_small_id: u16,
    target_unit_id: i32,
    target_tile: TileRef,

    active: bool,
    initialized: bool,
    missile_unit_id: Option<i32>,
    speed: f64,
    path: Vec<TileRef>,
    path_index: usize,
}

impl SamMissileExecution {
    pub fn new(
        spawn: TileRef,
        owner_small_id: u16,
        sam_unit_id: i32,
        target_owner_small_id: u16,
        target_unit_id: i32,
        target_tile: TileRef,
    ) -> Self {
        Self {
            spawn,
            owner_small_id,
            sam_unit_id,
            target_owner_small_id,
            target_unit_id,
            target_tile,
            active: true,
            initialized: false,
            missile_unit_id: None,
            speed: 0.0,
            path: Vec::new(),
            path_index: 0,
        }
    }
}

impl Execution for SamMissileExecution {
    fn init(&mut self, game: &mut Game, tick: u32) {
        self.speed = game.wire.default_sam_missile_speed();
        // TS `AirPathFinder` constructor - seed captured once, at `init()` time.
        self.path = air_find_path(game, tick as i32, self.spawn, self.target_tile);
        self.path_index = if self.path.first() == Some(&self.spawn) { 1 } else { 0 };
    }

    fn tick(&mut self, game: &mut Game, _tick: u32) {
        if !self.active {
            return;
        }
        if !self.initialized {
            self.initialized = true;
            let id = game.build_unit(self.owner_small_id, unit_type::SAM_MISSILE, self.spawn);
            self.missile_unit_id = Some(id);
        }
        let Some(missile_id) = self.missile_unit_id else {
            self.active = false;
            return;
        };
        if !game.unit_exists(self.owner_small_id, missile_id) {
            self.active = false;
            return;
        }

        let target_alive = game.unit_exists(self.target_owner_small_id, self.target_unit_id);
        let target_is_nuke = target_alive
            && game
                .unit(self.target_owner_small_id, self.target_unit_id)
                .is_some_and(|u| {
                    u.unit_type == unit_type::ATOM_BOMB || u.unit_type == unit_type::HYDROGEN_BOMB
                });
        let sam_alive = game.unit_exists(self.owner_small_id, self.sam_unit_id);
        let same_owner = self.target_owner_small_id == self.owner_small_id;

        if !target_alive || !sam_alive || same_owner || !target_is_nuke {
            if target_alive {
                if let Some(u) = game.unit_mut(self.target_owner_small_id, self.target_unit_id) {
                    u.targeted_by_sam = false;
                }
            }
            game.remove_unit(self.owner_small_id, missile_id);
            self.active = false;
            return;
        }

        let speed = self.speed as usize;
        for _ in 0..speed {
            if self.path_index >= self.path.len() {
                // COMPLETE - reached target tile.
                game.remove_unit(self.target_owner_small_id, self.target_unit_id);
                game.remove_unit(self.owner_small_id, missile_id);
                self.active = false;
                return;
            }
            let node = self.path[self.path_index];
            self.path_index += 1;
            game.move_unit(self.owner_small_id, missile_id, node);
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        false
    }
}
