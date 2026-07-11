//! Structure upgrade (`UpgradeStructureExecution.ts`).
//!
//! `openfront/tests/AutoUpgrade.test.ts` is *not* a test of this execution (or of any
//! engine mechanic) - it only exercises `AutoUpgradeEvent`/`EventBus` plumbing from
//! `src/client/InputHandler.ts`, a pure UI pointer-event class with no server/engine
//! counterpart. The actual "pick the nearest affordable upgradable building" selection
//! logic it feeds into (`ClientGameRunner.findAndUpgradeNearestBuilding`) also runs
//! entirely client-side and only ever emits a plain `upgrade_structure` intent for a
//! unit id the client already chose - this file's `UpgradeStructureExecution` (already
//! ported, see below) is the only part of that flow with real engine behavior. Skipped
//! for the same reason `NukeTrajectory.test.ts` (client-only GL-renderer math) was
//! skipped - see `parabola.rs`.

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
