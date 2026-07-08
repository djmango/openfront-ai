//! Batched initial tribe spawns - all bots in one execution (matches TS tick-1 wave, one dispatch).

use crate::execution::spawn_util::execute_player_spawn;
use crate::execution::Execution;
use crate::game::{Game, PlayerInfo};
use crate::prng::PseudoRandom;
use crate::util::simple_hash;

struct TribeEntry {
    player_info: PlayerInfo,
    random: PseudoRandom,
}

pub struct TribeMassSpawn {
    tribes: Vec<TribeEntry>,
    next: usize,
    active: bool,
}

impl TribeMassSpawn {
    pub fn new(game_id: &str, tribes: Vec<PlayerInfo>) -> Self {
        let entries = tribes
            .into_iter()
            .map(|player_info| {
                let seed =
                    simple_hash(&player_info.id).wrapping_add(simple_hash(game_id));
                TribeEntry {
                    player_info,
                    random: PseudoRandom::new(seed),
                }
            })
            .collect();
        Self {
            tribes: entries,
            next: 0,
            active: true,
        }
    }
}

impl Execution for TribeMassSpawn {
    fn init(&mut self, _: &mut Game, _: u32) {}

    fn tick(&mut self, game: &mut Game, _: u32) {
        if !self.active {
            return;
        }
        while self.next < self.tribes.len() {
            let entry = &mut self.tribes[self.next];
            self.next += 1;
            execute_player_spawn(game, &entry.player_info, None, &mut entry.random);
        }
        self.active = false;
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        true
    }
}
