//! Gold donations (`DonateGoldExecution.ts` subset - emoji reaction intentionally
//! omitted, mirroring `DonateTroopsExecution` in `donate.rs`, since emojis are
//! cosmetic and don't affect the hash).

use super::Execution;
use crate::game::Game;

pub struct DonateGoldExecution {
    sender_small_id: u16,
    recipient_id: String,
    gold: Option<i64>,
    active: bool,
    initialized: bool,
}

impl DonateGoldExecution {
    pub fn new(sender_small_id: u16, recipient_id: String, gold: Option<i64>) -> Self {
        Self {
            sender_small_id,
            recipient_id,
            gold,
            active: true,
            initialized: false,
        }
    }
}

fn gold_chunk_size(difficulty: &str) -> f64 {
    match difficulty {
        "Easy" => 2_500.0,
        "Medium" => 5_000.0,
        "Hard" => 12_500.0,
        "Impossible" => 25_000.0,
        _ => 5_000.0,
    }
}

/// TS `DonateGoldExecution.calculateRelationUpdate`.
fn calculate_relation_update(game: &Game, gold_sent: i64, tick: u32) -> i32 {
    let difficulty = game.wire.game_config().difficulty.clone();
    let chunk_size = gold_chunk_size(&difficulty);
    let chunk_size_multiplier = tick as f64 / (3000.0 + game.wire.num_spawn_phase_turns() as f64);
    let adjusted_chunk_size = (chunk_size + chunk_size * chunk_size_multiplier).round() as i64;
    if adjusted_chunk_size <= 0 {
        return 0;
    }
    let chunks = gold_sent / adjusted_chunk_size;
    let relation_update = (chunks * 5) as i32;
    relation_update.min(100)
}

impl Execution for DonateGoldExecution {
    fn init(&mut self, game: &mut Game, _: u32) {
        if !self.active || self.initialized {
            return;
        }
        self.initialized = true;
        let recipient_small_id = game.player_by_id(&self.recipient_id).map(|p| p.small_id);
        if recipient_small_id.is_none() {
            self.active = false;
            return;
        }
        if self.gold.is_none() {
            let sender_gold = game
                .player_by_small_id(self.sender_small_id)
                .map(|p| p.gold)
                .unwrap_or(0);
            self.gold = Some(sender_gold / 3);
        }
    }

    fn tick(&mut self, game: &mut Game, tick: u32) {
        if !self.active || !self.initialized {
            return;
        }
        self.active = false;
        let gold = self.gold.unwrap_or(0);
        let recipient_small_id = game.player_by_id(&self.recipient_id).map(|p| p.small_id);
        let Some(recipient_small_id) = recipient_small_id else {
            return;
        };
        if !game.can_donate_gold(self.sender_small_id, recipient_small_id) {
            return;
        }
        let removed = game.remove_gold(self.sender_small_id, gold);
        if removed <= 0 {
            return;
        }
        game.add_gold(recipient_small_id, removed);
        let relation_update = calculate_relation_update(game, removed, tick);
        if relation_update > 0 {
            game.update_relation(recipient_small_id, self.sender_small_id, relation_update);
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::alliance_exec::AllianceRequestExecution;
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

    // Ported from AllianceDonation.test.ts's "Can donate gold after alliance
    // formed by reply"/"...by mutual request" (native's donate gating doesn't
    // distinguish who replied to whom, so both TS cases collapse to one here).
    #[test]
    fn donate_gold_succeeds_once_allied_by_counter_request() {
        let mut game = game_with_donations_enabled();
        game.end_spawn_phase();
        let player1 = add_human(&mut game, "player1", 1);
        let player2 = add_human(&mut game, "player2", 1);
        game.player_by_small_id_mut(player1).unwrap().gold = 1000;
        game.player_by_small_id_mut(player2).unwrap().gold = 100;

        assert!(game.create_alliance_request(player1, player2, game.ticks()));
        assert!(game.create_alliance_request(player2, player1, game.ticks()));
        assert!(game.is_allied_with(player1, player2));
        assert!(game.is_friendly(player1, player2));

        assert!(game.can_donate_gold(player1, player2));
        let gold_before = game.player_by_small_id(player2).unwrap().gold;

        game.add_execution(ExecEnum::DonateGold(DonateGoldExecution::new(
            player1,
            "player2".into(),
            Some(100),
        )));
        game.execute_next_tick();
        game.execute_next_tick();

        assert_eq!(
            game.player_by_small_id(player2).unwrap().gold,
            gold_before + 100
        );
    }

    // Ported from AllianceDonation.test.ts's "Can donate immediately after
    // accepting alliance (race condition)": TS's `executeNextTick` ticks
    // already-active executions BEFORE initializing newly-added ones (same
    // order in native's `Game::execute_next_tick`), so an alliance formed by a
    // same-tick counter-request's `init()` is already in place by the time a
    // simultaneously-added `DonateGoldExecution`'s `tick()` runs one tick later.
    #[test]
    fn donation_added_the_same_tick_as_the_counter_request_still_succeeds() {
        let mut game = game_with_donations_enabled();
        game.end_spawn_phase();
        let player1 = add_human(&mut game, "player1", 1);
        let player2 = add_human(&mut game, "player2", 1);
        game.player_by_small_id_mut(player1).unwrap().gold = 1000;
        game.player_by_small_id_mut(player2).unwrap().gold = 100;

        game.add_execution(ExecEnum::AllianceRequest(AllianceRequestExecution::new(
            player1,
            "player2".into(),
        )));
        game.execute_next_tick();

        let gold_before = game.player_by_small_id(player2).unwrap().gold;
        game.add_execution(ExecEnum::AllianceRequest(AllianceRequestExecution::new(
            player2,
            "player1".into(),
        )));
        game.add_execution(ExecEnum::DonateGold(DonateGoldExecution::new(
            player1,
            "player2".into(),
            Some(100),
        )));
        game.execute_next_tick();

        assert!(game.is_allied_with(player1, player2));
        assert!(game.is_allied_with(player2, player1));

        game.execute_next_tick();

        assert_eq!(
            game.player_by_small_id(player2).unwrap().gold,
            gold_before + 100
        );
    }
}
