//! Incomplete Rust sim - only for offline unit tests (`OPENFRONT_STUB=1`).

use crate::bot::TribeSpawner;
use crate::core::config::Config;
use crate::execution::{turn_to_executions, ExecEnum, SpawnTimerExecution, WinCheckExecution};
use crate::game::{Game, GameConfig, PlayerInfo, PlayerType};
use crate::map::GameMap;
use crate::obs::{build_obs_head, tile_bytes_le};
use crate::prng::PseudoRandom;
use crate::record::StampedIntent;
use crate::session::{seed_to_game_id, terrain_bytes, AGENT_CLIENT_ID};
use crate::util::simple_hash;
use serde_json::json;
use serde_json::Value;
use std::path::Path;

pub struct StubSession {
    pub game: Game,
}

impl StubSession {
    pub fn reset(
        repo_root: &Path,
        map_key: &str,
        seed: &str,
        bots: u32,
    ) -> Result<(Self, Value, Vec<u8>, Vec<u8>), String> {
        let map_dir = repo_root
            .join("openfront/resources/maps")
            .join(map_key.to_lowercase());
        let (_manifest, map) = GameMap::load_map_dir(&map_dir)?;
        let game_id = seed_to_game_id(seed);
        let wire = Config::from_value(
            &json!({
                "gameMap": map_key,
                "difficulty": "Medium",
                "donateGold": true,
                "donateTroops": true,
                "gameType": "Singleplayer",
                "gameMode": "Free For All",
                "gameMapSize": "Normal",
                "nations": "default",
                "bots": bots,
                "infiniteGold": false,
                "infiniteTroops": false,
                "instantBuild": false,
                "randomSpawn": false,
            }),
            false,
        )
        .map_err(|e| e.to_string())?;
        let config = GameConfig {
            game_map: map_key.into(),
            bots,
            ..Default::default()
        };
        let mut game = Game::new(game_id.clone(), config, wire, map.clone(), map);
        let mut prng = PseudoRandom::new(simple_hash(&format!("rl-{seed}")) + 1);
        game.add_from_info(&PlayerInfo {
            name: "Agent".into(),
            player_type: PlayerType::Human,
            client_id: Some(AGENT_CLIENT_ID.into()),
            id: prng.next_id(),
        });
        game.add_execution(ExecEnum::SpawnTimer(SpawnTimerExecution::new()));
        let mut spawner = TribeSpawner::new(&game_id);
        for spawn in spawner.spawn_tribes(bots) {
            game.add_execution(ExecEnum::Spawn(spawn));
        }
        game.add_execution(ExecEnum::WinCheck(WinCheckExecution::new()));
        let session = Self { game };
        let head = build_obs_head(&session.game, AGENT_CLIENT_ID);
        let terrain = terrain_bytes(&session.game);
        let tiles = tile_bytes_le(&session.game);
        Ok((session, head, terrain, tiles))
    }

    pub fn step(&mut self, intents: Vec<StampedIntent>, ticks: u32) -> (Value, Vec<u8>, u32) {
        let stamped: Vec<StampedIntent> = intents
            .into_iter()
            .map(|mut i| {
                i.client_id = AGENT_CLIENT_ID.to_string();
                i
            })
            .collect();
        let mut wasted = 0u32;
        let gid = self.game.game_id.clone();
        for _ in 0..ticks {
            let execs = turn_to_executions(&mut self.game, &gid, &stamped);
            if execs.is_empty() && !stamped.is_empty() {
                wasted += 1;
            }
            for e in execs {
                self.game.add_execution(e);
            }
            self.game.execute_next_tick();
        }
        let mut head = build_obs_head(&self.game, AGENT_CLIENT_ID);
        if let Some(obj) = head.as_object_mut() {
            obj.insert("wasted".into(), Value::from(wasted));
        }
        (head, tile_bytes_le(&self.game), wasted)
    }
}
