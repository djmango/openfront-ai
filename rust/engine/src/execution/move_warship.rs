//! Manual (human-issued) warship redirects (TS `MoveWarshipExecution.ts`).

use super::Execution;
use crate::game::Game;
use crate::map::TileRef;

/// TS `MoveWarshipExecution` - a one-shot manual batch move of one or more of
/// the owner's warships to `position`. All the work happens in `init` (matching
/// TS, whose `init()` performs the redirect and whose `isActive()` is always
/// `false`); the actual per-warship validation/dedup/water-component checks live
/// in `Game::move_warships`.
pub struct MoveWarshipExecution {
    owner_small_id: u16,
    unit_ids: Vec<i32>,
    position: TileRef,
}

impl MoveWarshipExecution {
    pub fn new(owner_small_id: u16, unit_ids: Vec<i32>, position: TileRef) -> Self {
        Self {
            owner_small_id,
            unit_ids,
            position,
        }
    }
}

impl Execution for MoveWarshipExecution {
    fn init(&mut self, game: &mut Game, _tick: u32) {
        game.move_warships(self.owner_small_id, &self.unit_ids, self.position);
    }

    fn tick(&mut self, _game: &mut Game, _tick: u32) {}

    fn is_active(&self) -> bool {
        false
    }
}
