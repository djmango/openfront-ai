//! `RecomputeRailClusterExecution.ts` - added once at game start, ticks forever.

use super::Execution;
use crate::game::Game;
use crate::rail;

pub struct RecomputeRailClusterExecution;

impl RecomputeRailClusterExecution {
    pub fn new() -> Self {
        Self
    }
}

impl Default for RecomputeRailClusterExecution {
    fn default() -> Self {
        Self::new()
    }
}

impl Execution for RecomputeRailClusterExecution {
    fn init(&mut self, _game: &mut Game, _tick: u32) {}

    fn tick(&mut self, game: &mut Game, _tick: u32) {
        rail::recompute_clusters(game);
    }

    fn is_active(&self) -> bool {
        true
    }

    fn active_during_spawn(&self) -> bool {
        false
    }
}
