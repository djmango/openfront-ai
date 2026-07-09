//! Human embargo intents (`EmbargoExecution.ts` + `EmbargoAllExecution.ts`).

use super::Execution;
use crate::game::{Game, PlayerType};

/// TS `EmbargoExecution` - validates the target in `init`, applies the effect in `tick` (i.e.
/// on the tick *after* the one the intent was submitted on, matching how newly-added
/// executions get `init()`'d this tick and `tick()`'d starting next tick).
pub struct EmbargoExecution {
    player_small_id: u16,
    target_id: String,
    action_start: bool,
    active: bool,
}

impl EmbargoExecution {
    pub fn new(player_small_id: u16, target_id: String, action_start: bool) -> Self {
        Self {
            player_small_id,
            target_id,
            action_start,
            active: true,
        }
    }
}

impl Execution for EmbargoExecution {
    fn init(&mut self, game: &mut Game, _tick: u32) {
        if !game.has_player(&self.target_id) {
            self.active = false;
        }
    }

    fn tick(&mut self, game: &mut Game, tick: u32) {
        if !self.active {
            return;
        }
        if let Some(target) = game.player_by_id(&self.target_id) {
            let target_small_id = target.small_id;
            if self.action_start {
                game.add_embargo(self.player_small_id, target_small_id, false, tick);
            } else {
                game.stop_embargo(self.player_small_id, target_small_id);
            }
        }
        self.active = false;
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

/// TS `EmbargoAllExecution` - everything happens in `init` (single-shot, `isActive()` always
/// false), matching TS where the whole effect runs synchronously with no `tick()` step.
pub struct EmbargoAllExecution {
    player_small_id: u16,
    action_start: bool,
}

impl EmbargoAllExecution {
    pub fn new(player_small_id: u16, action_start: bool) -> Self {
        Self {
            player_small_id,
            action_start,
        }
    }
}

impl Execution for EmbargoAllExecution {
    fn init(&mut self, game: &mut Game, tick: u32) {
        let me = self.player_small_id;
        if !game.can_embargo_all(me) {
            return;
        }
        let others: Vec<u16> = game
            .all_players()
            .iter()
            .filter(|p| {
                p.small_id != me
                    && p.player_type != PlayerType::Bot
                    && !game.players_on_same_team(me, p.small_id)
            })
            .map(|p| p.small_id)
            .collect();
        for other in others {
            if self.action_start {
                if !game.has_embargo_against(me, other) {
                    game.add_embargo(me, other, false, tick);
                }
            } else if game.has_embargo_against(me, other) {
                game.stop_embargo(me, other);
            }
        }
        game.record_embargo_all(me);
    }

    fn tick(&mut self, _game: &mut Game, _tick: u32) {}

    fn is_active(&self) -> bool {
        false
    }
}
