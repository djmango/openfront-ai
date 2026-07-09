//! MIRV missile flight + warhead separation (`MIRVExecution.ts`).

use super::nuke_execution::{can_build_nuke, NukeExecution};
use super::parabola::{self, Curve};
use super::{ExecEnum, Execution};
use crate::core::schemas::unit_type;
use crate::game::Game;
use crate::map::TileRef;
use crate::prng::PseudoRandom;
use crate::util::simple_hash;

const RANGE: f64 = 1500.0;
const MINIMUM_SPREAD: f64 = 55.0;
const WARHEAD_COUNT: usize = 350;

/// TS `Math.round` - floor(x + 0.5); differs from Rust's `f64::round` (round-half-away-
/// from-zero) at negative half-integers, which matter here since coordinates can go
/// negative before the `isValidCoord` bounds check filters them out.
fn js_round(v: f64) -> f64 {
    (v + 0.5).floor()
}

pub struct MirvExecution {
    owner_small_id: u16,
    dst: TileRef,

    active: bool,
    mirv_unit_id: Option<i32>,
    spawn_tile: Option<TileRef>,
    separate_dst: Option<TileRef>,
    target_owner_small_id: u16,
    speed: f64,
    curve: Option<Curve>,
    random: Option<PseudoRandom>,
    base_x: i64,
    base_y: i64,
}

impl MirvExecution {
    pub fn new(owner_small_id: u16, dst: TileRef) -> Self {
        Self {
            owner_small_id,
            dst,
            active: true,
            mirv_unit_id: None,
            spawn_tile: None,
            separate_dst: None,
            target_owner_small_id: 0,
            speed: -1.0,
            curve: None,
            random: None,
            base_x: 0,
            base_y: 0,
        }
    }
}

impl Execution for MirvExecution {
    fn init(&mut self, game: &mut Game, _tick: u32) {
        let owner_id_str = game
            .player_by_small_id(self.owner_small_id)
            .map(|p| p.id.clone())
            .unwrap_or_default();
        let seed = game.ticks() as i32 + simple_hash(&owner_id_str);
        self.random = Some(PseudoRandom::new(seed));
        self.target_owner_small_id = game.map.owner_id(self.dst);
        self.speed = game.wire.default_nuke_speed();

        if self.target_owner_small_id != 0 {
            game.break_alliance_silently(self.owner_small_id, self.target_owner_small_id);
            if self.target_owner_small_id != self.owner_small_id {
                game.update_relation(self.target_owner_small_id, self.owner_small_id, -100);
                game.update_relation(self.owner_small_id, self.target_owner_small_id, -100);
            }
        }
    }

    fn tick(&mut self, game: &mut Game, _tick: u32) {
        if !self.active {
            return;
        }
        let mirv_id = match self.mirv_unit_id {
            Some(id) => id,
            None => {
                let Some(spawn) = can_build_nuke(game, self.owner_small_id, unit_type::MIRV, self.dst)
                else {
                    self.active = false;
                    return;
                };
                self.spawn_tile = Some(spawn);
                let id = game.build_unit(self.owner_small_id, unit_type::MIRV, spawn);
                if let Some(u) = game.unit_mut(self.owner_small_id, id) {
                    u.target_tile = Some(self.dst);
                }
                self.mirv_unit_id = Some(id);
                let x = ((game.x(self.dst) as i64 + game.x(spawn) as i64) / 2) as u32;
                let y = (game.y(self.dst) as i64 - 500).max(0) as u32 + 50;
                self.separate_dst = Some(game.ref_xy(x, y));
                id
            }
        };

        let spawn = self.spawn_tile.unwrap();
        let separate_dst = self.separate_dst.unwrap();
        if self.curve.is_none() {
            self.curve = Some(parabola::create_curve(
                game,
                spawn,
                separate_dst,
                self.speed,
                true,
                true,
            ));
        }
        let next = self.curve.as_mut().unwrap().increment(self.speed);
        match next {
            None => {
                self.separate(game, mirv_id);
                self.active = false;
            }
            Some(p) => {
                let tile = parabola::point_to_tile(game, p);
                game.move_unit(self.owner_small_id, mirv_id, tile);
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

impl MirvExecution {
    fn separate(&mut self, game: &mut Game, mirv_id: i32) {
        self.base_x = game.x(self.dst) as i64;
        self.base_y = game.y(self.dst) as i64;
        let destinations = self.select_destinations(game);
        for (i, &d) in destinations.iter().enumerate() {
            let wait_ticks = self.random.as_mut().unwrap().next_int(0, 15) as u32;
            let speed = 15.0 + ((i as f64 / WARHEAD_COUNT as f64) * 5.0).floor();
            // TS overwrites `src` with `dst` itself on the warhead's own first tick
            // (`canSpawnUnitType(MIRVWarhead) -> targetTile`), so the constructor value
            // passed here is never actually used for pathing.
            game.add_execution(ExecEnum::Nuke(NukeExecution::new(
                unit_type::MIRV_WARHEAD,
                self.owner_small_id,
                d,
                None,
                speed,
                wait_ticks,
                true,
            )));
        }
        game.remove_unit(self.owner_small_id, mirv_id);
    }

    fn select_destinations(&mut self, game: &Game) -> Vec<TileRef> {
        let mut targets = vec![self.dst];
        for _ in 0..1000 {
            if let Some(t) = self.try_generate_target(game, &targets) {
                targets.push(t);
            }
            if targets.len() >= WARHEAD_COUNT {
                break;
            }
        }
        let dst = self.dst;
        targets.sort_by(|a, b| {
            let dist_a = game.manhattan_dist(*a, dst);
            let dist_b = game.manhattan_dist(*b, dst);
            dist_b.cmp(&dist_a)
        });
        targets
    }

    fn try_generate_target(&mut self, game: &Game, taken: &[TileRef]) -> Option<TileRef> {
        let target_owner = self.target_owner_small_id;
        let base_x = self.base_x;
        let base_y = self.base_y;
        for _ in 0..100 {
            let (r1, r2) = {
                let rand = self.random.as_mut().unwrap();
                let r1 = rand.next();
                let r2 = (r1 * 15485863.0) % 1.0;
                (r1, r2)
            };
            let x = js_round(r1 * RANGE * 2.0 - RANGE + base_x as f64) as i64;
            let y = js_round(r2 * RANGE * 2.0 - RANGE + base_y as f64) as i64;

            if !game.is_valid_coord(x as i32, y as i32) {
                continue;
            }
            let tile = game.ref_xy(x as u32, y as u32);
            if !game.is_land(tile) {
                continue;
            }
            let dx = (x - base_x) as f64;
            let dy = (y - base_y) as f64;
            if dx * dx + dy * dy > RANGE * RANGE {
                continue;
            }
            if game.map.owner_id(tile) != target_owner {
                continue;
            }
            if is_overlapping(game, x, y, taken) {
                continue;
            }
            return Some(tile);
        }
        None
    }
}

fn is_overlapping(game: &Game, x: i64, y: i64, taken: &[TileRef]) -> bool {
    for &t in taken {
        let tx = game.x(t) as i64;
        let ty = game.y(t) as i64;
        let manhattan = (x - tx).abs() + (y - ty).abs();
        if (manhattan as f64) < MINIMUM_SPREAD {
            return true;
        }
    }
    false
}
