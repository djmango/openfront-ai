//! Batched tribe AI - one tick driver for all bots (avoids 400× `dyn Execution` dispatch).

use crate::execution::ai_attack::{send_tn_attack, tribe_maybe_attack};
use crate::game::Game;
use crate::prng::PseudoRandom;
use crate::util::simple_hash;

struct TribeState {
    small_id: u16,
    random: PseudoRandom,
    attack_rate: i32,
    attack_tick: i32,
    trigger_ratio: f64,
    reserve_ratio: f64,
    expand_ratio: f64,
    attack_behavior_init: bool,
    neighbors_terra_nullius: bool,
    active: bool,
}

pub struct TribeBatch {
    tribes: Vec<TribeState>,
}

impl TribeBatch {
    pub fn new() -> Self {
        Self { tribes: Vec::new() }
    }

    pub fn register(&mut self, small_id: u16, player_id: &str) {
        let mut random = PseudoRandom::new(simple_hash(player_id));
        let attack_rate = random.next_int(40, 80);
        let attack_tick = random.next_int(0, attack_rate);
        let trigger_ratio = random.next_int(50, 60) as f64 / 100.0;
        let reserve_ratio = random.next_int(30, 40) as f64 / 100.0;
        let expand_ratio = random.next_int(10, 20) as f64 / 100.0;
        self.tribes.push(TribeState {
            small_id,
            random,
            attack_rate,
            attack_tick,
            trigger_ratio,
            reserve_ratio,
            expand_ratio,
            attack_behavior_init: false,
            neighbors_terra_nullius: true,
            active: true,
        });
    }

    pub fn tick(&mut self, game: &mut Game, tick: u32) {
        if game.in_spawn_phase() {
            return;
        }
        let tick_i = tick as i32;
        for t in &mut self.tribes {
            if !t.active || tick_i % t.attack_rate != t.attack_tick {
                continue;
            }
            if game
                .player_by_small_id(t.small_id)
                .is_none_or(|p| !p.alive || p.spawn_tile.is_none())
            {
                t.active = false;
                continue;
            }
            if !t.attack_behavior_init {
                t.attack_behavior_init = true;
                send_tn_attack(game, t.small_id, t.expand_ratio);
                continue;
            }
            tribe_maybe_attack(
                game,
                &mut t.random,
                t.small_id,
                t.trigger_ratio,
                t.reserve_ratio,
                t.expand_ratio,
                &mut t.neighbors_terra_nullius,
            );
        }
    }
}
