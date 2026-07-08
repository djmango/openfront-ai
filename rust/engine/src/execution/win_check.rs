use super::Execution;
use crate::game::Game;

pub struct WinCheckExecution {
    active: bool,
}

impl WinCheckExecution {
    pub fn new() -> Self {
        Self { active: true }
    }
}

impl Execution for WinCheckExecution {
    fn init(&mut self, _: &mut Game, _: u32) {}

    fn tick(&mut self, game: &mut Game, _: u32) {
        let alive: Vec<_> = game.players_alive().map(|p| p.id.clone()).collect();
        if alive.len() == 1 && !game.in_spawn_phase() {
            game.winner = Some(alive[0].clone());
            self.active = false;
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }
}
