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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::ExecEnum;
    use crate::game::{PlayerInfo, PlayerType};

    fn game_with_donations_enabled() -> Game {
        let mut game = Game::default();
        let mut cfg = game.wire.game_config().clone();
        cfg.donate_gold = true;
        cfg.donate_troops = true;
        game.wire = crate::core::config::Config::new(cfg, false);
        game
    }

    fn add_human(game: &mut Game, id: &str, tiles_owned: i32) -> u16 {
        let sid = game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type: PlayerType::Human,
            client_id: Some(id.into()),
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        });
        game.player_by_small_id_mut(sid).unwrap().tiles_owned = tiles_owned;
        sid
    }

    // Ported from AllianceDonation.test.ts's "Can donate troops after alliance
    // formed by reply"/"...by mutual request" (both TS cases collapse to one
    // here, same as the equivalent gold test in donate_gold.rs).
    #[test]
    fn donate_troops_succeeds_once_allied_by_counter_request() {
        let mut game = game_with_donations_enabled();
        game.end_spawn_phase();
        let player1 = add_human(&mut game, "player1", 1);
        let player2 = add_human(&mut game, "player2", 1);
        game.player_by_small_id_mut(player1).unwrap().troops = 1_000;
        game.player_by_small_id_mut(player2).unwrap().troops = 100;

        assert!(game.create_alliance_request(player1, player2, game.ticks()));
        assert!(game.create_alliance_request(player2, player1, game.ticks()));
        assert!(game.is_allied_with(player1, player2));

        assert!(game.can_donate_troops(player1, player2));
        let troops_before = game.player_by_small_id(player2).unwrap().troops;

        game.add_execution(ExecEnum::DonateTroops(DonateTroopsExecution::new(
            player1,
            "player2".into(),
            Some(100.0),
        )));
        game.execute_next_tick();
        game.execute_next_tick();

        assert_eq!(
            game.player_by_small_id(player2).unwrap().troops,
            troops_before + 100
        );
    }
}
