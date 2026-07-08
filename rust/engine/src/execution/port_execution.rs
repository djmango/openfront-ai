//! Port structure behavior (`PortExecution.ts` subset - trade ships deferred).

use super::Execution;
use crate::game::Game;

pub struct PortExecution {
    small_id: u16,
    unit_id: i32,
    active: bool,
}

impl PortExecution {
    pub fn new(small_id: u16, unit_id: i32) -> Self {
        Self {
            small_id,
            unit_id,
            active: true,
        }
    }
}

impl Execution for PortExecution {
    fn init(&mut self, _: &mut Game, _: u32) {}

    fn tick(&mut self, game: &mut Game, _: u32) {
        if !self.active {
            return;
        }
        let still_active = game
            .player_by_small_id(self.small_id)
            .and_then(|p| p.units.iter().find(|u| u.id == self.unit_id))
            .is_some_and(|u| !u.under_construction);
        if !still_active {
            self.active = false;
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }
}
