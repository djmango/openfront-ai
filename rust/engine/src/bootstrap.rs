//! Bootstrap native `Game` from a `GameRecord` (mirrors `datagen/replay.ts` init).

use crate::bot::tribe_spawner::TribeSpawner;
use crate::core::config::Config;
use crate::core::nation::create_nations_for_game;
use crate::core::schemas::GameConfig as WireGameConfig;
use crate::core::terrain::{load_fresh_terrain, GameMapSize};
use crate::execution::{
    ExecEnum, NationExecution, NationRuntime, SpawnExecution, SpawnTimerExecution, WinCheckExecution,
};
use crate::game::{Game, GameConfig, PlayerInfo, PlayerType};
use crate::prng::PseudoRandom;
use crate::record::GameRecord;
use crate::util::simple_hash;
use std::path::Path;

pub fn game_from_record(repo_root: &Path, record: &GameRecord) -> Result<Game, String> {
    let wire: WireGameConfig = serde_json::from_value(record.info.config.clone())
        .map_err(|e| format!("parse config: {e}"))?;
    let cfg = Config::from_value(&record.info.config, true)
        .map_err(|e| format!("config: {e}"))?;
    let map_key = wire.game_map.clone();
    let map_size = GameMapSize::parse(&wire.game_map_size).unwrap_or(GameMapSize::Normal);
    let terrain = load_fresh_terrain(repo_root, &map_key, map_size)?;

    let game_id = record.info.game_id.clone();
    let stub_config = GameConfig {
        game_map: map_key,
        bots: cfg.bots(),
        num_spawn_phase_turns: cfg.num_spawn_phase_turns(),
        random_spawn: cfg.is_random_spawn(),
    };

    let mut random = PseudoRandom::new(simple_hash(&game_id));
    let humans: Vec<PlayerInfo> = record
        .info
        .players
        .iter()
        .map(|p| PlayerInfo {
            name: p.username.clone(),
            player_type: PlayerType::Human,
            client_id: Some(p.client_id.clone()),
            id: random.next_id(),
        })
        .collect();

    let nations = create_nations_for_game(
        &wire,
        &terrain.nations,
        &terrain.additional_nations,
        humans.len() as u32,
        &mut random,
    );

    let mut game = Game::new(
        game_id.clone(),
        stub_config,
        cfg.clone(),
        terrain.game_map,
        terrain.mini_game_map,
    );

    for h in &humans {
        game.add_from_info(h);
    }
    for n in &nations {
        game.add_from_info(&PlayerInfo {
            name: n.nation.name.clone(),
            player_type: PlayerType::Nation,
            client_id: None,
            id: n.player_id.clone(),
        });
    }

    if wire.game_type != "Singleplayer" {
        game.add_execution(ExecEnum::SpawnTimer(SpawnTimerExecution::new()));
    }

    if cfg.spawn_nations() {
        for n in &nations {
            game.add_execution(ExecEnum::Nation(NationExecution::new(
                game_id.clone(),
                NationRuntime {
                    spawn_cell: n.nation.coordinates,
                    player_info: PlayerInfo {
                        name: n.nation.name.clone(),
                        player_type: PlayerType::Nation,
                        client_id: None,
                        id: n.player_id.clone(),
                    },
                },
            )));
        }
    }

    if cfg.is_random_spawn() {
        for h in &humans {
            game.add_execution(ExecEnum::Spawn(SpawnExecution::new(
                game_id.clone(),
                h.clone(),
                None,
            )));
        }
    }

    if cfg.bots() > 0 {
        let mut spawner = TribeSpawner::new(&game_id);
        for spawn in spawner.spawn_tribes(cfg.bots()) {
            game.add_execution(ExecEnum::Spawn(spawn));
        }
    }

    game.add_execution(ExecEnum::WinCheck(WinCheckExecution::new()));

    let _ = terrain.team_game_spawn_areas;
    Ok(game)
}
