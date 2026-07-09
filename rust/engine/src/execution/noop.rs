use super::Execution;
use crate::game::Game;

pub struct NoOpExecution;

impl Execution for NoOpExecution {
    fn init(&mut self, _: &mut Game, _: u32) {}
    fn tick(&mut self, _: &mut Game, _: u32) {}
    fn is_active(&self) -> bool {
        false
    }
}
