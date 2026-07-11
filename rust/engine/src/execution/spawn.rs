//! Player spawn execution (`SpawnExecution.ts`).

use super::Execution;
use crate::execution::spawn_util::execute_player_spawn;
use crate::game::{Game, PlayerInfo, PlayerType};
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
        let spawned = execute_player_spawn(game, &self.player_info, self.tile, &mut self.random);
        // TS `SpawnExecution.tick`: in singleplayer the spawn phase ends
        // the moment the human player picks a spawn (GameRunner never adds
        // a SpawnTimerExecution for singleplayer). Without this the RL
        // path's game stays in the spawn phase forever - empty legality,
        // no PlayerExecutions ticking, no win check.
        if spawned
            && game.wire.game_type() == "Singleplayer"
            && self.player_info.player_type == PlayerType::Human
        {
            game.end_spawn_phase();
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        true
    }
}

// TS `TerritoryCapture.test.ts`.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::ExecEnum;

    #[test]
    fn player_owns_the_tile_it_spawns_on() {
        let mut game = crate::test_util::plains_game(100, 100);
        let info = PlayerInfo {
            name: "test_player".to_string(),
            player_type: PlayerType::Human,
            client_id: Some("test_id".to_string()),
            id: "test_id".to_string(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        };
        let spawn_tile = game.map.ref_xy(50, 50);
        game.add_execution(ExecEnum::Spawn(SpawnExecution::new(
            "game_id".to_string(),
            info,
            Some(spawn_tile),
        )));
        // Init the execution.
        game.execute_next_tick();
        // Execute the execution.
        game.execute_next_tick();

        let owner_id = game.map.owner_id(spawn_tile);
        assert_ne!(owner_id, 0, "tile should have an owner");
        let owner = game.player_by_small_id(owner_id).expect("owner exists");
        assert_eq!(owner.name, "test_player");
    }
}
