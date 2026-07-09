//! Structure upgrade (`UpgradeStructureExecution.ts`).

use super::Execution;
use crate::core::schemas::unit_type;
use crate::game::Game;

pub struct UpgradeStructureExecution {
    small_id: u16,
    unit_id: i32,
    active: bool,
}

impl UpgradeStructureExecution {
    pub fn new(small_id: u16, unit_id: i32) -> Self {
        Self {
            small_id,
            unit_id,
            active: true,
        }
    }
}

impl Execution for UpgradeStructureExecution {
    fn init(&mut self, game: &mut Game, _: u32) {
        if !self.active {
            return;
        }
        if !game.upgrade_unit(self.small_id, self.unit_id) {
            self.active = false;
        } else {
            self.active = false;
        }
    }

    fn tick(&mut self, _: &mut Game, _: u32) {}

    fn is_active(&self) -> bool {
        self.active
    }
}

pub fn is_upgradable_type(unit_type: &str) -> bool {
    matches!(
        unit_type,
        unit_type::CITY
            | unit_type::PORT
            | unit_type::FACTORY
            | unit_type::SAM_LAUNCHER
            | unit_type::MISSILE_SILO
    )
}
