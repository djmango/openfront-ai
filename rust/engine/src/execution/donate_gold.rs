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
