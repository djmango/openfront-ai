use super::Execution;

pub struct MarkDisconnectedExecution {
    small_id: u16,
    disconnected: bool,
}

impl MarkDisconnectedExecution {
    pub fn new(small_id: u16, disconnected: bool) -> Self {
        Self {
            small_id,
            disconnected,
        }
    }
}

impl Execution for MarkDisconnectedExecution {
    fn init(&mut self, game: &mut crate::game::Game, _: u32) {
        if let Some(p) = game.player_by_small_id_mut(self.small_id) {
            p.is_disconnected = self.disconnected;
        }
    }

    fn tick(&mut self, _: &mut crate::game::Game, _: u32) {}

    fn is_active(&self) -> bool {
        false
    }
}
