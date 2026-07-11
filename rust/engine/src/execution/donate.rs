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

// Ported from Donate.test.ts. TS's DonateGoldExecution shares the same
// alliance-gated donate path as DonateTroopsExecution, so both live here
// rather than duplicating the setup in donate_gold.rs.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schemas::GameConfig as WireGameConfig;
    use crate::execution::{donate_gold::DonateGoldExecution, ExecEnum};
    use crate::game::{Game, PlayerInfo, PlayerType};

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

    // Second scenario family below needs real land tiles (uses game.conquer),
    // not just a tiles_owned counter - its own map/config/add_human builders.
    fn game_with_donations_enabled_and_map() -> Game {
        let mut game = Game::default();
        game.map = crate::map::GameMap::from_terrain_bytes(
            &crate::map::MapMeta {
                width: 5,
                height: 5,
                num_land_tiles: 25,
            },
            &vec![0x80u8; 25],
        )
        .unwrap();
        game.wire = crate::core::config::Config::new(
            WireGameConfig {
                game_map: "test".into(),
                difficulty: "Medium".into(),
                donate_gold: true,
                donate_troops: true,
                game_type: "Singleplayer".into(),
                game_mode: "Free For All".into(),
                game_map_size: "Normal".into(),
                nations: crate::core::schemas::NationsConfig::Mode("default".into()),
                bots: 0,
                infinite_gold: false,
                infinite_troops: false,
                instant_build: false,
                random_spawn: false,
                doomsday_clock: None,
                disabled_units: None,
                player_teams: None,
                disable_alliances: None,
                spawn_immunity_duration: None,
                starting_gold: None,
                gold_multiplier: None,
                max_timer_value: None,
                ranked_type: None,
            },
            false,
        );
        game.end_spawn_phase();
        game
    }

    fn add_human_no_tiles(game: &mut Game, id: &str) -> u16 {
        game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type: PlayerType::Human,
            client_id: Some(id.into()),
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        })
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

    #[test]
    fn troops_are_donated_between_allies() {
        let mut game = game_with_donations_enabled_and_map();
        let donor = add_human_no_tiles(&mut game, "donor");
        let recipient = add_human_no_tiles(&mut game, "recipient");
        // `can_donate_troops`/`can_donate_gold` require `tiles_owned > 0`
        // (mirrors TS `isAlive() == tiles.size > 0`); TS's tests spawn both
        // players via `SpawnExecution` for the same reason.
        game.conquer(donor, game.ref_xy(0, 0));
        game.conquer(recipient, game.ref_xy(4, 4));
        assert!(game.create_alliance_request(donor, recipient, game.ticks()));
        game.accept_alliance_request(donor, recipient, game.ticks());

        game.player_by_small_id_mut(donor).unwrap().troops += 6000;
        let donor_troops_before = game.player_by_small_id(donor).unwrap().troops;
        let recipient_troops_before = game.player_by_small_id(recipient).unwrap().troops;

        game.add_execution(ExecEnum::DonateTroops(DonateTroopsExecution::new(
            donor,
            game.player_by_small_id(recipient).unwrap().id.clone(),
            Some(5000.0),
        )));
        for _ in 0..5 {
            game.execute_next_tick();
        }

        assert!(game.player_by_small_id(donor).unwrap().troops < donor_troops_before);
        assert!(game.player_by_small_id(recipient).unwrap().troops > recipient_troops_before);
    }

    #[test]
    fn gold_is_donated_between_allies() {
        let mut game = game_with_donations_enabled_and_map();
        let donor = add_human_no_tiles(&mut game, "donor");
        let recipient = add_human_no_tiles(&mut game, "recipient");
        game.conquer(donor, game.ref_xy(0, 0));
        game.conquer(recipient, game.ref_xy(4, 4));
        assert!(game.create_alliance_request(donor, recipient, game.ticks()));
        game.accept_alliance_request(donor, recipient, game.ticks());

        game.player_by_small_id_mut(donor).unwrap().gold += 6000;
        let donor_gold_before = game.player_by_small_id(donor).unwrap().gold;
        let recipient_gold_before = game.player_by_small_id(recipient).unwrap().gold;

        game.add_execution(ExecEnum::DonateGold(DonateGoldExecution::new(
            donor,
            game.player_by_small_id(recipient).unwrap().id.clone(),
            Some(5000),
        )));
        for _ in 0..5 {
            game.execute_next_tick();
        }

        assert!(game.player_by_small_id(donor).unwrap().gold < donor_gold_before);
        assert!(game.player_by_small_id(recipient).unwrap().gold > recipient_gold_before);
    }

    #[test]
    fn troops_are_not_donated_to_a_non_ally() {
        let mut game = game_with_donations_enabled_and_map();
        let donor = add_human_no_tiles(&mut game, "donor");
        let recipient = add_human_no_tiles(&mut game, "recipient");
        game.conquer(donor, game.ref_xy(0, 0));
        game.conquer(recipient, game.ref_xy(4, 4));
        assert!(game.create_alliance_request(donor, recipient, game.ticks()));
        game.reject_alliance_request(donor, recipient);

        let donor_troops_before = game.player_by_small_id(donor).unwrap().troops;
        let recipient_troops_before = game.player_by_small_id(recipient).unwrap().troops;

        game.add_execution(ExecEnum::DonateTroops(DonateTroopsExecution::new(
            donor,
            game.player_by_small_id(recipient).unwrap().id.clone(),
            Some(5000.0),
        )));
        game.execute_next_tick();

        assert!(game.player_by_small_id(donor).unwrap().troops >= donor_troops_before);
        assert!(game.player_by_small_id(recipient).unwrap().troops >= recipient_troops_before);
    }

    #[test]
    fn gold_is_not_donated_to_a_non_ally() {
        let mut game = game_with_donations_enabled_and_map();
        let donor = add_human_no_tiles(&mut game, "donor");
        let recipient = add_human_no_tiles(&mut game, "recipient");
        game.conquer(donor, game.ref_xy(0, 0));
        game.conquer(recipient, game.ref_xy(4, 4));
        assert!(game.create_alliance_request(donor, recipient, game.ticks()));
        game.reject_alliance_request(donor, recipient);

        let donor_gold_before = game.player_by_small_id(donor).unwrap().gold;
        let recipient_gold_before = game.player_by_small_id(recipient).unwrap().gold;

        game.add_execution(ExecEnum::DonateGold(DonateGoldExecution::new(
            donor,
            game.player_by_small_id(recipient).unwrap().id.clone(),
            Some(5000),
        )));
        game.execute_next_tick();

        assert!(game.player_by_small_id(donor).unwrap().gold >= donor_gold_before);
        assert!(game.player_by_small_id(recipient).unwrap().gold >= recipient_gold_before);
    }

    #[test]
    fn self_donation_is_disallowed_despite_being_friendly_with_self() {
        let mut game = game_with_donations_enabled_and_map();
        let player = add_human_no_tiles(&mut game, "player_self");
        game.conquer(player, game.ref_xy(0, 0));

        assert!(game.is_friendly(player, player));
        assert!(!game.can_donate_gold(player, player));
        assert!(!game.can_donate_troops(player, player));

        game.player_by_small_id_mut(player).unwrap().gold += 1000;
        game.player_by_small_id_mut(player).unwrap().troops += 1000;
        let gold_before = game.player_by_small_id(player).unwrap().gold;
        let troops_before = game.player_by_small_id(player).unwrap().troops;

        let self_id = game.player_by_small_id(player).unwrap().id.clone();
        game.add_execution(ExecEnum::DonateGold(DonateGoldExecution::new(
            player,
            self_id.clone(),
            Some(500),
        )));
        game.add_execution(ExecEnum::DonateTroops(DonateTroopsExecution::new(
            player,
            self_id,
            Some(500.0),
        )));
        game.execute_next_tick();

        assert!(game.player_by_small_id(player).unwrap().gold >= gold_before);
        assert!(game.player_by_small_id(player).unwrap().troops >= troops_before);
    }
}
