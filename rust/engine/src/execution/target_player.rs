//! Target-player marks (TS `TargetPlayerExecution.ts`).

use super::Execution;
use crate::game::Game;

/// TS `TargetPlayerExecution` - resolves the target in `init`, paints the
/// mark (and applies the -40 relation hit on the target toward the
/// requestor) in `tick`, then deactivates.
pub struct TargetPlayerExecution {
    requestor_small_id: u16,
    target_id: String,
    target_small_id: Option<u16>,
    active: bool,
}

impl TargetPlayerExecution {
    pub fn new(requestor_small_id: u16, target_id: String) -> Self {
        Self {
            requestor_small_id,
            target_id,
            target_small_id: None,
            active: true,
        }
    }
}

impl Execution for TargetPlayerExecution {
    fn init(&mut self, game: &mut Game, _tick: u32) {
        match game.player_by_id(&self.target_id) {
            Some(t) => self.target_small_id = Some(t.small_id),
            None => self.active = false,
        }
    }

    fn tick(&mut self, game: &mut Game, _tick: u32) {
        if !self.active {
            return;
        }
        if let Some(target) = self.target_small_id {
            if game.can_target(self.requestor_small_id, target) {
                game.add_target_mark(self.requestor_small_id, target);
                game.update_relation(target, self.requestor_small_id, -40);
            }
        }
        self.active = false;
    }

    fn is_active(&self) -> bool {
        self.active
    }
}
