//! Troop donations (`DonateTroopExecution.ts` subset).

use super::Execution;
use crate::game::Game;

pub struct DonateTroopsExecution {
    sender_small_id: u16,
    recipient_id: String,
    troops: Option<f64>,
    active: bool,
    initialized: bool,
}

impl DonateTroopsExecution {
    pub fn new(sender_small_id: u16, recipient_id: String, troops: Option<f64>) -> Self {
        Self {
            sender_small_id,
            recipient_id,
            troops,
            active: true,
            initialized: false,
        }
    }
}

impl Execution for DonateTroopsExecution {
    fn init(&mut self, game: &mut Game, _: u32) {
        if !self.active || self.initialized {
            return;
        }
        self.initialized = true;
        let recipient_small_id = game.player_by_id(&self.recipient_id).map(|p| p.small_id);
        let Some(recipient_small_id) = recipient_small_id else {
            self.active = false;
            return;
        };
        let recipient_troops = game
            .player_by_small_id(recipient_small_id)
            .map(|p| p.troops)
            .unwrap_or(0);
        let troops = self.troops.unwrap_or_else(|| {
            game.player_by_small_id(self.sender_small_id)
                .map(|p| (p.troops as f64 / 3.0).floor())
                .unwrap_or(0.0)
        });
        let max_donation = game.max_troops_for(recipient_small_id) - recipient_troops as f64;
        let troops = troops.min(max_donation);
        if troops <= 0.0 {
            self.active = false;
            return;
        }
        self.troops = Some(troops);
    }

    fn tick(&mut self, game: &mut Game, _: u32) {
        if !self.active || !self.initialized {
            return;
        }
        let troops = self.troops.unwrap_or(0.0);
        let recipient_small_id = game.player_by_id(&self.recipient_id).map(|p| p.small_id);
        let Some(recipient_small_id) = recipient_small_id else {
            self.active = false;
            return;
        };
        if !game.can_donate_troops(self.sender_small_id, recipient_small_id) {
            self.active = false;
            return;
        }
        let removed = game.remove_troops(self.sender_small_id, troops);
        if removed > 0 {
            game.add_troops(recipient_small_id, removed as f64);
            if troops >= removed as f64 {
                game.update_relation(recipient_small_id, self.sender_small_id, 50);
            }
        }
        self.active = false;
    }

    fn is_active(&self) -> bool {
        self.active
    }
}
