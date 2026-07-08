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
            clan_tag: p.clan_tag.clone(),
            friends: p.friends.clone().unwrap_or_default(),
            team: None,
        })
        .collect();

    let nations = create_nations_for_game(
        &wire,
        &terrain.nations,
        &terrain.additional_nations,
        humans.len() as u32,
        &mut random,
    );

    let nation_infos: Vec<PlayerInfo> = nations
        .iter()
        .map(|n| PlayerInfo {
            name: n.nation.name.clone(),
            player_type: PlayerType::Nation,
            client_id: None,
            id: n.player_id.clone(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        })
        .collect();

    let mut all_players = humans.clone();
    all_players.extend(nation_infos.clone());

    let mut game = Game::new(
        game_id.clone(),
        stub_config,
        cfg.clone(),
        terrain.game_map,
        terrain.mini_game_map,
        terrain.team_game_spawn_areas,
    );
    game.init_player_teams(humans.len(), nations.len());

    let mut player_infos = all_players;
    if wire.game_mode == "Team" {
        game.assign_teams_for_players(&mut player_infos);
    }

    let add_order: Vec<usize> = if wire.game_mode == "Team" {
        game.assign_teams_insertion_order(&player_infos)
    } else {
        (0..player_infos.len()).collect()
    };

    for &idx in &add_order {
        let info = &player_infos[idx];
        if wire.game_mode == "Team" && info.team.is_none() {
            continue;
        }
        game.add_from_info(info);
    }

    if wire.game_type != "Singleplayer" {
        game.add_execution(ExecEnum::SpawnTimer(SpawnTimerExecution::new()));
    }

    if cfg.spawn_nations() {
        for (n, info) in nations.iter().zip(player_infos.iter().filter(|p| p.player_type == PlayerType::Nation)) {
            if info.team.is_none() && wire.game_mode == "Team" {
                continue;
            }
            game.add_execution(ExecEnum::Nation(NationExecution::new(
                game_id.clone(),
                NationRuntime {
                    spawn_cell: n.nation.coordinates,
                    player_info: info.clone(),
                },
            )));
        }
    }

    if cfg.is_random_spawn() {
        // TS `PlayerSpawner`: `allPlayers()` insertion order (team assignment Map order).
        let human_spawns: Vec<PlayerInfo> = game
            .players_in_order()
            .iter()
            .filter(|p| p.player_type == PlayerType::Human)
            .filter(|p| p.team.is_some() || wire.game_mode != "Team")
            .map(|p| PlayerInfo {
                name: p.name.clone(),
                player_type: p.player_type,
                client_id: if p.client_id.is_empty() {
                    None
                } else {
                    Some(p.client_id.clone())
                },
                id: p.id.clone(),
                clan_tag: None,
                friends: Vec::new(),
                team: p.team.clone(),
            })
            .collect();
        for info in human_spawns {
            game.add_execution(ExecEnum::Spawn(SpawnExecution::new(
                game_id.clone(),
                info,
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

    Ok(game)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::team_assignment::{assign_teams, populate_player_teams};
    use crate::record::GameRecord;

    #[test]
    fn hxwdr5pk_assign_teams_count() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai-rust-fast".into());
        let path = std::path::Path::new(&repo_root).join("records/0c4c7d7993c9/HxWdr5PK.json.gz");
        let bytes = std::fs::read(&path).unwrap();
        let mut dec = flate2::read::GzDecoder::new(bytes.as_slice());
        let mut raw = Vec::new();
        use std::io::Read;
        dec.read_to_end(&mut raw).unwrap();
        let rec = GameRecord::from_json_bytes(&raw).unwrap();
        let wire: crate::core::schemas::GameConfig =
            serde_json::from_value(rec.info.config.clone()).unwrap();
        let mut random = crate::prng::PseudoRandom::new(crate::util::simple_hash(&rec.info.game_id));
        let humans: Vec<crate::game::PlayerInfo> = rec
            .info
            .players
            .iter()
            .map(|p| crate::game::PlayerInfo {
                name: p.username.clone(),
                player_type: crate::game::PlayerType::Human,
                client_id: Some(p.client_id.clone()),
                id: random.next_id(),
                clan_tag: p.clan_tag.clone(),
                friends: p.friends.clone().unwrap_or_default(),
                team: None,
            })
            .collect();
        let teams = populate_player_teams("Team", wire.player_teams.as_ref(), humans.len(), 0);
        let (assigned, _) = assign_teams(&humans, &teams);
        let ok = assigned.values().filter(|t| *t != "kicked").count();
        eprintln!("teams={teams:?} assigned_ok={ok}/{}", humans.len());
        assert!(ok > 100, "expected most humans assigned, got {ok}");
    }

    #[test]
    fn hxwdr5pk_registers_all_humans() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai-rust-fast".into());
        let path = std::path::Path::new(&repo_root).join("records/0c4c7d7993c9/HxWdr5PK.json.gz");
        let bytes = std::fs::read(&path).unwrap();
        let raw = if path.extension().and_then(|e| e.to_str()) == Some("gz") {
            use std::io::Read;
            let mut dec = flate2::read::GzDecoder::new(bytes.as_slice());
            let mut out = Vec::new();
            dec.read_to_end(&mut out).unwrap();
            out
        } else {
            bytes
        };
        let rec = GameRecord::from_json_bytes(&raw).unwrap();
        let game = game_from_record(std::path::Path::new(&repo_root), &rec).unwrap();
        let humans = game
            .players_in_order()
            .iter()
            .filter(|p| p.player_type == PlayerType::Human)
            .count();
        assert_eq!(humans, 125, "expected 125 humans registered at bootstrap");
        assert_eq!(game.player_teams.len(), 5);
    }
}
