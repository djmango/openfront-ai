//! Player spawn execution (`SpawnExecution.ts`).

use super::Execution;
use crate::execution::spawn_util::execute_player_spawn;
use crate::game::{Game, PlayerInfo};
use crate::map::TileRef;
use crate::prng::PseudoRandom;
use crate::util::simple_hash;

pub struct SpawnExecution {
    game_id: String,
    player_info: PlayerInfo,
    tile: Option<TileRef>,
    random: PseudoRandom,
    active: bool,
}

impl SpawnExecution {
    pub fn new(game_id: String, player_info: PlayerInfo, tile: Option<TileRef>) -> Self {
        let seed = simple_hash(&player_info.id).wrapping_add(simple_hash(&game_id));
        Self {
            game_id,
            player_info,
            tile,
            random: PseudoRandom::new(seed),
            active: true,
        }
    }
}

impl Execution for SpawnExecution {
    fn init(&mut self, _: &mut Game, _: u32) {}

    fn tick(&mut self, game: &mut Game, _: u32) {
        if !self.active {
            return;
        }
        self.active = false;
        execute_player_spawn(game, &self.player_info, self.tile, &mut self.random);
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        true
    }
}
