//! Tribe bot expansion (TS `TribeExecution.ts` subset).

use crate::execution::ai_attack::{send_tn_attack, tribe_maybe_attack};
use crate::execution::Execution;
use crate::game::Game;
use crate::prng::PseudoRandom;
use crate::util::simple_hash;

pub struct TribeExecution {
    small_id: u16,
    player_id: String,
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

impl TribeExecution {
    pub fn new(small_id: u16, player_id: String) -> Self {
        let mut random = PseudoRandom::new(simple_hash(&player_id));
        let attack_rate = random.next_int(40, 80);
        let attack_tick = random.next_int(0, attack_rate);
        let trigger_ratio = random.next_int(50, 60) as f64 / 100.0;
        let reserve_ratio = random.next_int(30, 40) as f64 / 100.0;
        let expand_ratio = random.next_int(10, 20) as f64 / 100.0;
        Self {
            small_id,
            player_id,
            random,
            attack_rate,
            attack_tick,
            trigger_ratio,
            reserve_ratio,
            expand_ratio,
            attack_behavior_init: false,
            neighbors_terra_nullius: true,
            active: true,
        }
    }
}

impl Execution for TribeExecution {
    fn init(&mut self, _: &mut Game, _: u32) {}

    fn tick(&mut self, game: &mut Game, tick: u32) {
        if !self.active || game.in_spawn_phase() {
            return;
        }
        if (tick as i32) % self.attack_rate != self.attack_tick {
            return;
        }

        if game
            .player_by_small_id(self.small_id)
            .is_none_or(|p| !p.alive || p.spawn_tile.is_none())
        {
            self.active = false;
            return;
        }

        if !self.attack_behavior_init {
            self.attack_behavior_init = true;
            send_tn_attack(game, self.small_id, self.expand_ratio);
            return;
        }

        tribe_maybe_attack(
            game,
            &mut self.random,
            self.small_id,
            self.trigger_ratio,
            self.reserve_ratio,
            self.expand_ratio,
            &mut self.neighbors_terra_nullius,
        );
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        false
    }
}
