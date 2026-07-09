//! Attack cancel / retreat (`RetreatExecution.ts`).

use super::Execution;
use crate::game::Game;

const CANCEL_DELAY: u32 = 20;

pub struct RetreatExecution {
    owner_small_id: u16,
    attack_id: String,
    start_tick: u32,
    retreat_ordered: bool,
    active: bool,
}

impl RetreatExecution {
    pub fn new(owner_small_id: u16, attack_id: String) -> Self {
        Self {
            owner_small_id,
            attack_id,
            start_tick: 0,
            retreat_ordered: false,
            active: true,
        }
    }
}

impl Execution for RetreatExecution {
    fn init(&mut self, game: &mut Game, tick: u32) {
        self.start_tick = tick;
    }

    fn tick(&mut self, game: &mut Game, tick: u32) {
        if !self.active {
            return;
        }
        if !self.retreat_ordered {
            game.order_retreat(self.owner_small_id, &self.attack_id);
            self.retreat_ordered = true;
        }
        if tick >= self.start_tick.saturating_add(CANCEL_DELAY) {
            game.execute_retreat(self.owner_small_id, &self.attack_id);
            self.active = false;
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

pub struct BoatRetreatExecution {
    owner_small_id: u16,
    unit_id: i32,
    active: bool,
}

impl BoatRetreatExecution {
    pub fn new(owner_small_id: u16, unit_id: i32) -> Self {
        Self {
            owner_small_id,
            unit_id,
            active: true,
        }
    }
}

impl Execution for BoatRetreatExecution {
    fn init(&mut self, _game: &mut Game, _tick: u32) {}

    fn tick(&mut self, game: &mut Game, _tick: u32) {
        game.order_boat_retreat(self.owner_small_id, self.unit_id);
        self.active = false;
    }

    fn is_active(&self) -> bool {
        self.active
    }
}
