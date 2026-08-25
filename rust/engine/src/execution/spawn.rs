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
        // TS `SpawnExecution.tick` anti-teleport gate (OpenFront pin
        // f0da4182, "reject spawn intents after the spawn phase"): once the
        // game is no longer in the spawn phase, an already-spawned player's
        // spawn intent must be a deterministic no-op rather than
        // relinquishing its entire territory and re-conquering it at the new
        // tile (an instant teleport). Without this, native applies a late
        // spawn intent that TS ignores, moving the player's whole starting
        // blob to a different map location and desyncing every subsequent
        // tick. Gated here (not in the shared `execute_player_spawn`) because
        // TS only guards the human `SpawnExecution` path, not bot
        // `TribeExecution` mass spawns.
        if !game.in_spawn_phase() {
            let small_id = game
                .player_by_id(&self.player_info.id)
                .map(|p| p.small_id);
            if let Some(small_id) = small_id {
                if game.has_spawned(small_id) {
                    return;
                }
            }
        }
        let spawned = execute_player_spawn(game, &self.player_info, self.tile, &mut self.random);
        // TS `SpawnExecution.tick`: in singleplayer the spawn phase ends
        // when the (typically one) human picks a spawn. Duo co-training
        // has two humans; ending on the first pick would leave the partner
        // unable to spawn under empty post-spawn legality. Wait until every
        // human has a spawn tile. One-human FFA is unchanged.
        if spawned
            && game.wire.game_type() == "Singleplayer"
            && self.player_info.player_type == PlayerType::Human
        {
            let humans_pending = game.all_players().iter().any(|p| {
                p.player_type == PlayerType::Human && !game.has_spawned(p.small_id)
            });
            if !humans_pending {
                game.end_spawn_phase();
            }
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
