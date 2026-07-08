//! Replay driver - TS engine is the parity oracle.

use crate::backend::Backend;
use crate::execution::intent::turn_to_executions;
use crate::execution::{ExecEnum, SpawnTimerExecution, WinCheckExecution};
use crate::game::Game;
use crate::record::GameRecord;

#[derive(Debug, Clone)]
pub struct ReplayOptions {
    pub backend: Backend,
    pub repo_root: std::path::PathBuf,
}

#[derive(Debug, Clone)]
pub struct ReplayResult {
    pub ok: bool,
    pub reason: Option<String>,
    pub ticks: u32,
    pub hashes_checked: u32,
}

pub fn replay_record(record_path: &std::path::Path, opts: &ReplayOptions) -> ReplayResult {
    match opts.backend {
        Backend::Ts => super::backend::node::verify_record_file(&opts.repo_root, record_path),
        Backend::Native | Backend::Stub => match load_record_bytes(record_path) {
            Ok(bytes) => {
                let record = match GameRecord::from_json_bytes(&bytes) {
                    Ok(r) => r,
                    Err(e) => {
                        return ReplayResult {
                            ok: false,
                            reason: Some(format!("parse record: {e}")),
                            ticks: 0,
                            hashes_checked: 0,
                        };
                    }
                };
                if matches!(opts.backend, Backend::Native) {
                    replay_native(&opts.repo_root, &record)
                } else {
                    replay_stub(&record)
                }
            }
            Err(e) => ReplayResult {
                ok: false,
                reason: Some(format!("read record: {e}")),
                ticks: 0,
                hashes_checked: 0,
            },
        },
    }
}

fn load_record_bytes(path: &std::path::Path) -> Result<Vec<u8>, String> {
    let raw = std::fs::read(path).map_err(|e| e.to_string())?;
    if path.extension().and_then(|s| s.to_str()) == Some("gz") {
        use std::io::Read;
        let mut dec = flate2::read::GzDecoder::new(&raw[..]);
        let mut out = Vec::new();
        dec.read_to_end(&mut out).map_err(|e| e.to_string())?;
        Ok(out)
    } else {
        Ok(raw)
    }
}

fn engine_root(fallback: &std::path::Path) -> std::path::PathBuf {
    std::env::var("OPENFRONT_REPO")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| fallback.to_path_buf())
}

fn replay_native(repo_root: &std::path::Path, record: &GameRecord) -> ReplayResult {
    let root = engine_root(repo_root);
    let rec = record.clone().decompress();
    let mut hashes_checked = 0u32;
    let mut game = match crate::bootstrap::game_from_record(&root, &rec) {
        Ok(g) => g,
        Err(e) => {
            return ReplayResult {
                ok: false,
                reason: Some(format!("bootstrap: {e}")),
                ticks: 0,
                hashes_checked: 0,
            };
        }
    };

    for turn in &rec.turns {
        let gid = game.game_id.clone();
        for e in turn_to_executions(&mut game, &gid, &turn.intents) {
            game.add_execution(e);
        }
        let updates = game.execute_next_tick();
        if let Some(h) = updates.hash {
            if let Some(archived) = rec.turns.get(h.tick as usize).and_then(|t| t.hash) {
                hashes_checked += 1;
                if archived != h.hash {
                    return ReplayResult {
                        ok: false,
                        reason: Some(format!(
                            "desync tick {} (native {} != archived {})",
                            h.tick, h.hash, archived
                        )),
                        ticks: game.ticks(),
                        hashes_checked,
                    };
                }
            }
        }
    }

    ReplayResult {
        ok: true,
        reason: None,
        ticks: game.ticks(),
        hashes_checked,
    }
}

fn replay_stub(record: &GameRecord) -> ReplayResult {
    let rec = record.clone().decompress();
    let mut hashes_checked = 0u32;
    let mut game = Game::default();
    game.add_execution(ExecEnum::SpawnTimer(SpawnTimerExecution::new()));
    game.add_execution(ExecEnum::WinCheck(WinCheckExecution::new()));

    let gid = String::new();
    for turn in &rec.turns {
        for e in turn_to_executions(&mut game, &gid, &turn.intents) {
            game.add_execution(e);
        }
        let updates = game.execute_next_tick();
        if let Some(h) = updates.hash {
            if let Some(archived) = rec.turns.get(h.tick as usize).and_then(|t| t.hash) {
                hashes_checked += 1;
                if archived != h.hash {
                    return ReplayResult {
                        ok: false,
                        reason: Some(format!(
                            "desync tick {} (stub {} != archived {})",
                            h.tick, h.hash, archived
                        )),
                        ticks: game.ticks(),
                        hashes_checked,
                    };
                }
            }
        }
    }

    ReplayResult {
        ok: true,
        reason: None,
        ticks: game.ticks(),
        hashes_checked,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::intent::turn_to_executions;
    use crate::hash::game_hash;
    use crate::map::TileRef;

    fn load_record_bytes(path: &std::path::Path) -> Result<Vec<u8>, String> {
        super::load_record_bytes(path)
    }

    fn replay_to_tick(repo: &std::path::Path, path: &std::path::Path, target: u32) -> Game {
        let bytes = load_record_bytes(path).unwrap();
        let rec = GameRecord::from_json_bytes(&bytes).unwrap().decompress();
        let mut game = crate::bootstrap::game_from_record(repo, &rec).unwrap();
        for turn in rec.turns.iter() {
            if turn.turn_number > target {
                break;
            }
            let gid = game.game_id.clone();
            for e in turn_to_executions(&mut game, &gid, &turn.intents) {
                game.add_execution(e);
            }
            game.execute_next_tick();
        }
        game
    }

    #[test]
    fn hash_checkpoints_280_to_320() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai".into());
        let repo = std::path::Path::new(&repo_root);
        let path = repo.join("records/0c4c7d7993c9/jby2gMJF.json.gz");
        let bytes = load_record_bytes(&path).unwrap();
        let rec = GameRecord::from_json_bytes(&bytes).unwrap().decompress();
        let mut game = crate::bootstrap::game_from_record(repo, &rec).unwrap();
        for turn in rec.turns.iter() {
            let gid = game.game_id.clone();
            for e in turn_to_executions(&mut game, &gid, &turn.intents) {
                game.add_execution(e);
            }
            let updates = game.execute_next_tick();
            if let Some(h) = updates.hash {
                if ![280, 290, 300, 310, 320].contains(&h.tick) {
                    continue;
                }
                let archived = rec.turns.get(h.tick as usize).and_then(|t| t.hash);
                let tiles: i32 = game.players_in_order().iter().map(|p| p.tiles_owned).sum();
                let troops: i32 = game.players_in_order().iter().map(|p| p.troops).sum();
                let ok = archived == Some(h.hash);
                eprintln!(
                    "tick {} ok={} tiles {} troops {} native={} archived={:?}",
                    h.tick, ok, tiles, troops, h.hash, archived
                );
            }
        }
    }

    #[test]
    fn export_tick_state_json() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai".into());
        let repo = std::path::Path::new(&repo_root);
        let target: u32 = std::env::var("EXPORT_TICK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(310);
        let path = std::env::var("EXPORT_RECORD")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| repo.join("records/0c4c7d7993c9/jby2gMJF.json.gz"));
        let game = replay_to_tick(repo, &path, target);
        if let Ok(pid) = std::env::var("EXPORT_PLAYER") {
            if let Some(p) = game.players_in_order().iter().find(|p| p.id == pid) {
                let mut tiles: Vec<u32> = p.owned_tiles.iter().copied().collect();
                tiles.sort_unstable();
                eprintln!(
                    "player {} spawn_tile {:?} tiles_owned {} owned_tiles {:?}",
                    pid, p.spawn_tile, p.tiles_owned, tiles
                );
            }
        }
        let include_units = std::env::var("EXPORT_UNITS").is_ok();
        let players: Vec<_> = game
            .players_in_order()
            .iter()
            .map(|p| {
                let mut row = serde_json::json!({
                    "id": p.id,
                    "playerType": format!("{:?}", p.player_type).to_uppercase(),
                    "tilesOwned": p.tiles_owned,
                    "troops": p.troops,
                    "alive": p.alive,
                    "units": p.units.len(),
                    "unitHash": p.units.iter().map(crate::hash::unit_hash_js).sum::<f64>(),
                    "playerHash": crate::hash::player_hash_js(p),
                    "idHash": p.id_hash,
                });
                if include_units && !p.units.is_empty() {
                    row["unitList"] = serde_json::json!(p.units.iter().map(|u| {
                        serde_json::json!({
                            "type": u.unit_type,
                            "tile": u.tile,
                            "id": u.id,
                            "hash": crate::hash::unit_hash_js(u),
                        })
                    }).collect::<Vec<_>>());
                }
                row
            })
            .collect();
        let tiles: i32 = players.iter().map(|p| p["tilesOwned"].as_i64().unwrap_or(0) as i32).sum();
        let troops: i32 = players.iter().map(|p| p["troops"].as_i64().unwrap_or(0) as i32).sum();
        let out = serde_json::json!({
            "tick": game.ticks(),
            "hash": game_hash(&game),
            "totalTiles": tiles,
            "totalTroops": troops,
            "players": players,
        });
        println!("{}", out);
    }

    fn list_record_paths(repo: &std::path::Path) -> Vec<std::path::PathBuf> {
        let mut paths = Vec::new();
        let records_dir = repo.join("records");
        let commit_filter = std::env::var("PARITY_COMMIT").ok();
        let Ok(entries) = std::fs::read_dir(&records_dir) else {
            return paths;
        };
        for engine_dir in entries.flatten() {
            if !engine_dir.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            if let Some(ref want) = commit_filter {
                if engine_dir.file_name().to_string_lossy() != want.as_str() {
                    continue;
                }
            }
            let Ok(files) = std::fs::read_dir(engine_dir.path()) else {
                continue;
            };
            for file in files.flatten() {
                let path = file.path();
                if path.extension().and_then(|s| s.to_str()) == Some("gz") {
                    paths.push(path);
                }
            }
        }
        paths.sort();
        paths
    }

    #[test]
    #[ignore]
    fn multi_record_parity_report() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai".into());
        let repo = std::path::Path::new(&repo_root);
        let paths = list_record_paths(repo);
        eprintln!(
            "multi_record: {} records under {}",
            paths.len(),
            repo.join("records").display()
        );
        let mut pass = 0u32;
        let mut fail = 0u32;
        let mut fail_at: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        for path in &paths {
            let result = replay_record(
                path,
                &ReplayOptions {
                    backend: Backend::Native,
                    repo_root: repo.to_path_buf(),
                },
            );
            if result.ok {
                pass += 1;
            } else {
                fail += 1;
                if let Some(reason) = &result.reason {
                    if let Some(tick) = reason
                        .split("desync tick ")
                        .nth(1)
                        .and_then(|s| s.split_whitespace().next())
                    {
                        if let Ok(t) = tick.parse::<u32>() {
                            *fail_at.entry(t).or_insert(0) += 1;
                        }
                    }
                }
                eprintln!(
                    "FAIL {:?} ticks={} {:?}",
                    path.file_name(),
                    result.ticks,
                    result.reason
                );
            }
        }
        eprintln!("=== multi_record summary ===");
        eprintln!("pass={pass} fail={fail} total={}", pass + fail);
        let mut buckets: Vec<_> = fail_at.into_iter().collect();
        buckets.sort_by_key(|(t, _)| *t);
        for (tick, count) in buckets {
            eprintln!("  tick {tick}: {count}");
        }
    }

    #[test]
    fn bootstrap_nation_ids_3qnu() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai".into());
        let repo = std::path::Path::new(&repo_root);
        let path = repo.join("records/0c4c7d7993c9/3QNU4eJa.json.gz");
        let bytes = std::fs::read(&path).unwrap();
        use std::io::Read;
        let mut dec = flate2::read::GzDecoder::new(&bytes[..]);
        let mut out = Vec::new();
        dec.read_to_end(&mut out).unwrap();
        let record: crate::record::GameRecord = serde_json::from_slice(&out).unwrap();
        let wire: crate::core::schemas::GameConfig =
            serde_json::from_value(record.info.config.clone()).unwrap();
        let terrain = crate::core::terrain::load_fresh_terrain(repo, &wire.game_map, crate::core::terrain::GameMapSize::parse(&wire.game_map_size).unwrap()).unwrap();
        let mut random = crate::prng::PseudoRandom::new(crate::util::simple_hash(&record.info.game_id));
        for _ in 0..record.info.players.len() {
            random.next_id();
        }
        let nations = crate::core::nation::create_nations_for_game(
            &wire,
            &terrain.nations,
            &terrain.additional_nations,
            record.info.players.len() as u32,
            &mut random,
        );
        for (i, n) in nations.iter().enumerate() {
            if i < 3 || n.player_id == "dg6epoyz" {
                eprintln!(
                    "{} {} id={} cell={:?}",
                    i,
                    n.nation.name,
                    n.player_id,
                    n.nation.coordinates
                );
            }
        }
    }

    #[test]
    fn parity_single_record() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai".into());
        let repo = std::path::Path::new(&repo_root);
        let rel = std::env::var("PARITY_RECORD")
            .unwrap_or_else(|_| "records/0c4c7d7993c9/3QNU4eJa.json.gz".into());
        let path = if rel.starts_with('/') {
            std::path::PathBuf::from(rel)
        } else {
            repo.join(rel)
        };
        let result = replay_record(
            &path,
            &ReplayOptions {
                backend: Backend::Native,
                repo_root: repo.to_path_buf(),
            },
        );
        if !result.ok {
            panic!("{:?}", result.reason);
        }
    }

    #[test]
    fn export_transport_path_json() {
        use crate::spatial::{can_build_transport_ship, target_transport_tile};
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai".into());
        let repo = std::path::Path::new(&repo_root);
        let target: u32 = std::env::var("EXPORT_TICK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(329);
        let rel = std::env::var("EXPORT_RECORD")
            .unwrap_or_else(|_| "records/0c4c7d7993c9/fkVh9QtC.json.gz".into());
        let ref_tile: TileRef = std::env::var("EXPORT_REF_TILE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3_515_786);
        let client_id = std::env::var("EXPORT_CLIENT_ID").unwrap_or_else(|_| "8bLLJCWp".into());
        let path = if rel.starts_with('/') {
            std::path::PathBuf::from(rel)
        } else {
            repo.join(rel)
        };
        let mut game = replay_to_tick(repo, &path, target);
        let Some(p) = game.player_by_client_id(&client_id) else {
            panic!("no player {client_id}");
        };
        let small_id = p.small_id;
        let dst = target_transport_tile(&mut game, ref_tile).unwrap_or(ref_tile);
        let src = can_build_transport_ship(&mut game, small_id, dst);
        let mut path_ok = false;
        let mut path_len = 0usize;
        let mut path0 = 0u32;
        let mut path1 = 0u32;
        if let Some(s) = src {
            path_ok = game.plan_water_path(s, dst);
            if path_ok {
                let path = game.planned_water_path();
                path_len = path.len();
                path0 = path.first().copied().unwrap_or(0);
                path1 = path.get(1).copied().unwrap_or(0);
            }
        }
        let mut out = serde_json::json!({
            "ref": ref_tile,
            "dst": dst,
            "src": src,
            "srcXY": src.map(|t| [game.map.x(t), game.map.y(t)]),
            "dstXY": [game.map.x(dst), game.map.y(dst)],
            "pathOk": path_ok,
            "pathLen": path_len,
            "path0": path0,
            "path0XY": if path0 > 0 { Some([game.map.x(path0), game.map.y(path0)]) } else { None },
            "path1": path1,
            "width": game.map.width,
        });
        if std::env::var("EXPORT_FULL_PATH").is_ok() && path_ok {
            let path = game.planned_water_path();
            out["path"] = serde_json::json!(path);
            out["pathXY"] = serde_json::json!(path.iter().map(|&t| [game.map.x(t), game.map.y(t)]).collect::<Vec<_>>());
        }
        println!("{}", out);
    }

    #[test]
    fn export_record_hash_at_turn() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai".into());
        let repo = std::path::Path::new(&repo_root);
        let target: u32 = std::env::var("EXPORT_TICK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(10);
        let rel = std::env::var("EXPORT_RECORD")
            .unwrap_or_else(|_| "records/0c4c7d7993c9/3QNU4eJa.json.gz".into());
        let path = if rel.starts_with('/') {
            std::path::PathBuf::from(rel)
        } else {
            repo.join(rel)
        };
        let bytes = load_record_bytes(&path).unwrap();
        let rec = GameRecord::from_json_bytes(&bytes).unwrap().decompress();
        let mut game = crate::bootstrap::game_from_record(repo, &rec).unwrap();
        for turn in rec.turns.iter() {
            if turn.turn_number > target {
                break;
            }
            let gid = game.game_id.clone();
            for e in turn_to_executions(&mut game, &gid, &turn.intents) {
                game.add_execution(e);
            }
            let updates = game.execute_next_tick();
            if turn.turn_number == target {
                let tiles: i32 = game.players_in_order().iter().map(|p| p.tiles_owned).sum();
                let troops: i32 = game.players_in_order().iter().map(|p| p.troops).sum();
                eprintln!(
                    "turn={} tick={} hash={} tiles={} troops={} archived={:?}",
                    target,
                    game.ticks(),
                    game_hash(&game),
                    tiles,
                    troops,
                    rec.turns.get(target as usize).and_then(|t| t.hash),
                );
                if let Some(h) = updates.hash {
                    eprintln!("updates hash tick={} hash={}", h.tick, h.hash);
                }
            }
        }
    }

    #[test]
    fn sigmoid_matches_ts() {
        use crate::util::sigmoid;
        let decay = std::f64::consts::LN_2 / 50_000.0;
        let mid = 150_000.0;
        let s = sigmoid(71.0, decay, mid);
        assert!(s < 0.01, "small defender sigmoid near 0, got {s}");
        let ldb = 0.7 + 0.3 * (1.0 - s);
        assert!((ldb - 1.0).abs() < 0.01, "large defender attack debuff ~1, got {ldb}");
    }

    #[test]
    fn debug_attack_7mv_turn_353_354() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai".into());
        let repo = std::path::Path::new(&repo_root);
        let path = repo.join("records/0c4c7d7993c9/7MVmc1cR.json.gz");
        let a_sid = replay_to_tick(repo, &path, 352)
            .player_by_id("t00u0ac6")
            .unwrap()
            .small_id;
        let b_sid = replay_to_tick(repo, &path, 352)
            .player_by_id("egjjaed8")
            .unwrap()
            .small_id;
        for target in [353u32, 354u32] {
            let game = replay_to_tick(repo, &path, target);
            eprintln!("turn={target} tick={}", game.ticks());
            for (o, t, troops, active, live, border, heap) in game.active_attacks_debug() {
                if o == a_sid && t == b_sid {
                    eprintln!(
                        "  attack o={o} t={t} troops={troops} active={active} live={live} border={border} heap={heap}"
                    );
                }
            }
        }
    }

    #[test]
    fn compare_owned_tiles_7mv_turn_353() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai".into());
        let repo = std::path::Path::new(&repo_root);
        let path = repo.join("records/0c4c7d7993c9/7MVmc1cR.json.gz");
        let game = replay_to_tick(repo, &path, 353);
        for pid in ["t00u0ac6", "egjjaed8"] {
            let p = game.player_by_id(pid).unwrap();
            let mut tiles: Vec<u32> = p.owned_tiles.iter().copied().collect();
            tiles.sort_unstable();
            eprintln!("native {pid} tiles_owned={} owned={}", p.tiles_owned, tiles.len());
        }
        let a = game.player_by_id("t00u0ac6").unwrap().small_id;
        let b = game.player_by_id("egjjaed8").unwrap().small_id;
        let mut adj = 0u32;
        game.for_each_border_tile(a, |t| {
            game.map.for_each_neighbor4(t, |n| {
                if game.map.owner_id(n) == b {
                    adj += 1;
                }
            });
        });
        eprintln!("native border-adj count {adj}");
    }

    #[test]
    fn export_exec_order() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai".into());
        let repo = std::path::Path::new(&repo_root);
        let target: u32 = std::env::var("EXPORT_TICK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(302);
        let path = std::env::var("EXPORT_RECORD")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| repo.join("records/0c4c7d7993c9/tCFq6nPn.json.gz"));
        let game = replay_to_tick(repo, &path, target);
        let labels = game.exec_labels();
        eprintln!("{target} execs={}", labels.len());
        for (i, label) in labels.iter().enumerate() {
            eprintln!("{i} {label}");
        }
    }

    #[test]
    fn export_player_border_tiles() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai".into());
        let repo = std::path::Path::new(&repo_root);
        let target: u32 = std::env::var("EXPORT_TICK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(302);
        let path = std::env::var("EXPORT_RECORD")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| repo.join("records/0c4c7d7993c9/tCFq6nPn.json.gz"));
        let player = std::env::var("EXPORT_PLAYER").unwrap_or_else(|_| "ti0zc829".into());
        let game = replay_to_tick(repo, &path, target);
        let sid = game.player_by_id(&player).unwrap().small_id;
        let border = game.border_tiles_for(sid);
        let mut line = format!("{target} {player} {sid} {} ", border.len());
        for t in &border {
            line.push_str(&format!("{t} "));
        }
        eprintln!("{line}");
    }

    #[test]
    fn export_map_owner_blob() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai".into());
        let repo = std::path::Path::new(&repo_root);
        let target: u32 = std::env::var("EXPORT_TICK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(302);
        let path = std::env::var("EXPORT_RECORD")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| repo.join("records/0c4c7d7993c9/jby2gMJF.json.gz"));
        let out = std::env::var("EXPORT_MAP_OUT").unwrap_or_else(|_| "/tmp/nat_owners.bin".into());
        let game = replay_to_tick(repo, &path, target);
        let n = (game.map.width * game.map.height) as usize;
        let mut buf = vec![0u8; n * 2];
        for t in 0..n {
            let o = game.map.owner_id(t as crate::map::TileRef);
            buf[t * 2] = (o & 0xff) as u8;
            buf[t * 2 + 1] = (o >> 8) as u8;
        }
        std::fs::write(&out, buf).unwrap();
        eprintln!("wrote {n} owners to {out}");
    }

    #[test]
    fn debug_6k4_tile_690794_turn_237() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai".into());
        let repo = std::path::Path::new(&repo_root);
        let path = repo.join("records/0c4c7d7993c9/6k4SnLrH.json.gz");
        let tile: crate::map::TileRef = 690794;
        let bot = "3rk2r3mx";
        for target in [236u32, 237u32] {
            let game = replay_to_tick(repo, &path, target);
            let sid = game.player_by_id(bot).unwrap().small_id;
            let owner = game.map.owner_id(tile);
            let mut on_border = false;
            game.map.for_each_neighbor4(tile, |n| {
                if game.map.owner_id(n) == sid {
                    on_border = true;
                }
            });
            eprintln!(
                "turn={target} tile={tile} owner={owner} bot_small={sid} bot_tiles={} on_border={on_border} is_land={} target_owner={}",
                game.player_by_small_id(sid).unwrap().tiles_owned,
                game.is_land(tile),
                game.map.owner_id(tile),
            );
            for (o, t, troops, active, live, border, heap) in game.active_attacks_debug() {
                if o == sid {
                    eprintln!(
                        "  attack -> {t} troops={troops} active={active} live={live} border={border} heap={heap}"
                    );
                }
            }
        }
    }

    #[test]
    fn map_tile_accounting_6k4_turn_195() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai".into());
        let repo = std::path::Path::new(&repo_root);
        let path = repo.join("records/0c4c7d7993c9/6k4SnLrH.json.gz");
        for target in [193u32, 194u32, 195u32] {
            let game = replay_to_tick(repo, &path, target);
            let mut map_counts: std::collections::HashMap<u16, u32> =
                std::collections::HashMap::new();
            let mut unowned_land = 0u32;
            let n = game.map.width * game.map.height;
            for t in 0..n {
                let t = t as crate::map::TileRef;
                if game.is_land(t) {
                    let o = game.map.owner_id(t);
                    if o > 0 {
                        *map_counts.entry(o).or_insert(0) += 1;
                    } else {
                        unowned_land += 1;
                    }
                }
            }
            let sid = game.player_by_id("j6f6ax25").unwrap().small_id;
            let p = game.player_by_small_id(sid).unwrap();
            let map_n = map_counts.get(&sid).copied().unwrap_or(0);
            let sum_owned: i32 = game.players_in_order().iter().map(|p| p.tiles_owned).sum();
            let sum_map: u32 = map_counts.values().sum();
            eprintln!(
                "turn={target} j6f6ax25 tiles_owned={} owned_vec={} map_count={} sum_owned={} sum_map={} unowned_land={}",
                p.tiles_owned,
                p.owned_tiles.len(),
                map_n,
                sum_owned,
                sum_map,
                unowned_land
            );
        }
    }

    #[test]
    fn border_adjacency_7mv_turn_353() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai".into());
        let repo = std::path::Path::new(&repo_root);
        let path = repo.join("records/0c4c7d7993c9/7MVmc1cR.json.gz");
        let game = replay_to_tick(repo, &path, 353);
        let a = game.player_by_id("t00u0ac6").unwrap().small_id;
        let b = game.player_by_id("egjjaed8").unwrap().small_id;
        let mut adj = 0u32;
        game.for_each_border_tile(a, |t| {
            game.map.for_each_neighbor4(t, |n| {
                if game.map.owner_id(n) == b {
                    adj += 1;
                }
            });
        });
        eprintln!(
            "native border adj t00u0ac6->egjjaed8: {adj} owner_tiles={} target_tiles={}",
            game.player_by_small_id(a).unwrap().tiles_owned,
            game.player_by_small_id(b).unwrap().tiles_owned,
        );
    }

    #[test]
    fn compare_attack_logic_turn_317() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai".into());
        let repo = std::path::Path::new(&repo_root);
        let path = repo.join("records/0c4c7d7993c9/jby2gMJF.json.gz");
        let game = replay_to_tick(repo, &path, 317);
        let tile = 956641u32;
        let odw = game.player_by_id("odw1azpw").unwrap().small_id;
        let bot = game.player_by_id("iwwln93n").unwrap().small_id;
        let (atk_loss, def_loss, tiles_used) =
            game.attack_logic_at_tile(2724.8, odw, bot, tile, true);
        assert!((atk_loss - 103.782997207264).abs() < 0.01, "atk_loss {atk_loss}");
        assert!(
            (def_loss - 106.66197183098592).abs() < 0.01,
            "def_loss {def_loss}"
        );
        assert!(
            (tiles_used - 10.746250332845003).abs() < 0.01,
            "tiles_used {tiles_used}"
        );
    }
}
