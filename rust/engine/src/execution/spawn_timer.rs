use super::Execution;
use crate::game::Game;

pub struct SpawnTimerExecution {
    active: bool,
}

impl SpawnTimerExecution {
    pub fn new() -> Self {
        Self { active: true }
    }
}

impl Execution for SpawnTimerExecution {
    fn init(&mut self, _: &mut Game, _: u32) {}

    fn tick(&mut self, game: &mut Game, _: u32) {
        if game.ticks() > game.config.num_spawn_phase_turns {
            game.end_spawn_phase();
            self.active = false;
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        true
    }

    fn is_spawn_timer(&self) -> bool {
        true
    }
}
