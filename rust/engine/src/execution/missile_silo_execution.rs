//! Passive Missile Silo reload-cooldown ticking (`MissileSiloExecution.ts`).

use super::Execution;
use crate::game::Game;

pub struct MissileSiloExecution {
    small_id: u16,
    unit_id: i32,
    active: bool,
}

impl MissileSiloExecution {
    pub fn new(small_id: u16, unit_id: i32) -> Self {
        Self {
            small_id,
            unit_id,
            active: true,
        }
    }
}

impl Execution for MissileSiloExecution {
    fn init(&mut self, _game: &mut Game, _tick: u32) {}

    fn tick(&mut self, game: &mut Game, _tick: u32) {
        let Some(u) = game.unit(self.small_id, self.unit_id) else {
            self.active = false;
            return;
        };
        if u.under_construction {
            return;
        }
        let Some(&front_time) = u.missile_timer_queue.first() else {
            return;
        };
        let cooldown = game.wire.silo_cooldown() as i64 - (game.ticks() as i64 - front_time as i64);
        if cooldown <= 0 {
            game.unit_reload_missile(self.small_id, self.unit_id);
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        false
    }
}
