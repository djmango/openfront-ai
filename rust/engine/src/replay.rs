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
        let players: Vec<_> = game
            .players_in_order()
            .iter()
            .filter(|p| p.tiles_owned > 0)
            .map(|p| {
                serde_json::json!({
                    "id": p.id,
                    "playerType": format!("{:?}", p.player_type).to_uppercase(),
                    "tilesOwned": p.tiles_owned,
                    "troops": p.troops,
                    "units": p.units.len(),
                    "unitHash": p.units.iter().map(crate::hash::unit_hash).sum::<i64>(),
                    "idHash": p.id_hash,
                })
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
        let Ok(entries) = std::fs::read_dir(&records_dir) else {
            return paths;
        };
        for engine_dir in entries.flatten() {
            if !engine_dir.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
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
}
