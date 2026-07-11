//! Replay driver - TS engine is the parity oracle.

use crate::backend::Backend;
use crate::execution::intent::turn_to_executions;
use crate::execution::{ExecEnum, SpawnTimerExecution, WinCheckExecution};
use crate::game::{Game, Player, PlayerType};
use crate::record::GameRecord;
use serde::{Deserialize, Serialize};
use serde_json::Value;

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

pub const OUTCOME_SCHEMA_VERSION: u32 = 1;
pub const OUTCOME_REQUIRED_PASSES: usize = 55;
pub const OUTCOME_EXPECTED_RECORDS: usize = 78;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OutcomeRankingEntry {
    pub identity: String,
    pub name: String,
    pub team: Option<String>,
    pub tiles: i32,
    pub alive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct GameOutcome {
    pub schema_version: u32,
    pub game_id: String,
    pub winner: Option<String>,
    pub terminal_tick: Option<u32>,
    pub terminal_reason: Option<String>,
    pub winner_land_share: Option<f64>,
    pub final_tick: u32,
    pub land_tiles_without_fallout: u32,
    pub final_ranking: Vec<OutcomeRankingEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OutcomeOracleCache {
    pub schema_version: u32,
    pub parity_commit: String,
    pub record_set_hash: String,
    pub outcomes: Vec<GameOutcome>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct OutcomeComparison {
    pub pass: bool,
    pub category: String,
    pub diagnostics: Vec<String>,
    pub winner_match: bool,
    pub timing_match: bool,
    pub land_share_match: bool,
    pub tick_delta_ratio: Option<f64>,
    pub land_share_delta: Option<f64>,
}

#[derive(Debug, Clone)]
struct TerminalCandidate {
    winner: String,
    tick: u32,
    reason: &'static str,
}

pub fn normalize_winner_value(winner: &Value) -> Option<String> {
    let parts = winner.as_array()?;
    let kind = parts.first()?.as_str()?;
    let value = parts.get(1)?.as_str()?;
    match kind {
        "player" | "team" | "nation" => Some(format!("{kind}:{value}")),
        _ => None,
    }
}

fn player_identity(player: &Player) -> String {
    if player.client_id.is_empty() {
        format!("nation:{}", player.name)
    } else {
        format!("player:{}", player.client_id)
    }
}

fn land_tiles_without_fallout(game: &Game) -> u32 {
    game.num_land_tiles()
        .saturating_sub(game.num_tiles_with_fallout())
        .max(1)
}

fn terminal_candidate(game: &Game) -> Option<TerminalCandidate> {
    if game.in_spawn_phase() || game.ticks() % 10 != 0 {
        return None;
    }
    let elapsed_seconds = game.ticks_since_start() / 10;
    let max_timer_reached = game
        .wire
        .game_config()
        .max_timer_value
        .is_some_and(|minutes| elapsed_seconds >= minutes.saturating_mul(60));
    let hard_limit_reached = elapsed_seconds >= 170 * 60;
    let denominator = land_tiles_without_fallout(game) as f64;

    if game.wire.game_config().game_mode == "Team" {
        let mut team_tiles: Vec<(String, i32)> = Vec::new();
        for player in game.players_alive() {
            let Some(team) = player.team.as_ref() else {
                continue;
            };
            if let Some((_, tiles)) = team_tiles.iter_mut().find(|(name, _)| name == team) {
                *tiles += player.tiles_owned;
            } else {
                team_tiles.push((team.clone(), player.tiles_owned));
            }
        }
        team_tiles.sort_by(|a, b| b.1.cmp(&a.1));
        let (team, tiles) = team_tiles.first()?;
        if team == crate::core::team_assignment::BOT_TEAM {
            return None;
        }
        let land_reached = (*tiles as f64 / denominator) * 100.0 > 95.0;
        let reason = if land_reached {
            "land_share"
        } else if max_timer_reached {
            "max_timer"
        } else if hard_limit_reached {
            "hard_time_limit"
        } else {
            return None;
        };
        return Some(TerminalCandidate {
            winner: format!("team:{team}"),
            tick: game.ticks(),
            reason,
        });
    }

    let mut players: Vec<&Player> = game.players_alive().collect();
    players.sort_by(|a, b| b.tiles_owned.cmp(&a.tiles_owned));
    let leader = *players.first()?;
    if game.wire.game_config().ranked_type.as_deref() == Some("1v1") {
        let humans: Vec<&Player> = players
            .iter()
            .copied()
            .filter(|p| p.player_type == PlayerType::Human && !p.is_disconnected)
            .collect();
        if humans.len() == 1 {
            return Some(TerminalCandidate {
                winner: player_identity(humans[0]),
                tick: game.ticks(),
                reason: "one_v_one",
            });
        }
    }
    let land_reached = (leader.tiles_owned as f64 / denominator) * 100.0 > 80.0;
    let reason = if land_reached {
        "land_share"
    } else if max_timer_reached {
        "max_timer"
    } else if hard_limit_reached {
        "hard_time_limit"
    } else {
        return None;
    };
    Some(TerminalCandidate {
        winner: player_identity(leader),
        tick: game.ticks(),
        reason,
    })
}

fn winner_land_share(game: &Game, winner: &str) -> Option<f64> {
    let denominator = land_tiles_without_fallout(game) as f64;
    if let Some(team) = winner.strip_prefix("team:") {
        let tiles: i32 = game
            .all_players()
            .iter()
            .filter(|p| p.team.as_deref() == Some(team))
            .map(|p| p.tiles_owned)
            .sum();
        return Some(tiles as f64 / denominator);
    }
    let player = game
        .all_players()
        .iter()
        .find(|p| player_identity(p) == winner)?;
    Some(player.tiles_owned as f64 / denominator)
}

fn outcome_from_game(game: &Game, terminal: Option<TerminalCandidate>) -> GameOutcome {
    let mut final_ranking: Vec<OutcomeRankingEntry> = game
        .all_players()
        .iter()
        .map(|p| OutcomeRankingEntry {
            identity: player_identity(p),
            name: p.name.clone(),
            team: p.team.clone(),
            tiles: p.tiles_owned,
            alive: p.alive,
        })
        .collect();
    final_ranking.sort_by(|a, b| {
        b.tiles
            .cmp(&a.tiles)
            .then_with(|| a.identity.cmp(&b.identity))
    });
    let winner = terminal.as_ref().map(|t| t.winner.clone());
    GameOutcome {
        schema_version: OUTCOME_SCHEMA_VERSION,
        game_id: game.game_id.clone(),
        winner_land_share: winner
            .as_deref()
            .and_then(|identity| winner_land_share(game, identity)),
        winner,
        terminal_tick: terminal.as_ref().map(|t| t.tick),
        terminal_reason: terminal.as_ref().map(|t| t.reason.to_string()),
        final_tick: game.ticks(),
        land_tiles_without_fallout: land_tiles_without_fallout(game),
        final_ranking,
    }
}

pub fn replay_outcome_native(
    repo_root: &std::path::Path,
    record_path: &std::path::Path,
) -> Result<GameOutcome, String> {
    let bytes = load_record_bytes(record_path).map_err(|e| format!("read record: {e}"))?;
    let record = GameRecord::from_json_bytes(&bytes)
        .map_err(|e| format!("parse record: {e}"))?
        .decompress();
    let root = engine_root(repo_root);
    let mut game =
        crate::bootstrap::game_from_record(&root, &record).map_err(|e| format!("bootstrap: {e}"))?;
    let mut terminal = None;
    for turn in &record.turns {
        let gid = game.game_id.clone();
        for execution in turn_to_executions(&mut game, &gid, &turn.intents) {
            game.add_execution(execution);
        }
        game.execute_next_tick();
        if terminal.is_none() {
            terminal = terminal_candidate(&game);
        }
    }
    Ok(outcome_from_game(&game, terminal))
}

pub fn compare_outcomes(expected: &GameOutcome, actual: &GameOutcome) -> OutcomeComparison {
    let mut diagnostics = Vec::new();
    let missing_winner = expected.winner.is_none() || actual.winner.is_none();
    let winner_match = !missing_winner && expected.winner == actual.winner;
    if missing_winner {
        diagnostics.push("missing_winner".to_string());
    } else if !winner_match {
        diagnostics.push("wrong_winner".to_string());
    }

    let tick_delta_ratio = match (expected.terminal_tick, actual.terminal_tick) {
        (Some(expected_tick), Some(actual_tick)) if expected_tick > 0 => {
            Some(expected_tick.abs_diff(actual_tick) as f64 / expected_tick as f64)
        }
        (Some(0), Some(0)) => Some(0.0),
        _ => None,
    };
    let timing_match = tick_delta_ratio.is_some_and(|delta| delta <= 0.20);
    if !missing_winner && winner_match && !timing_match {
        diagnostics.push("timing_mismatch".to_string());
    }

    let land_share_delta = match (expected.winner_land_share, actual.winner_land_share) {
        (Some(expected_share), Some(actual_share)) => Some((expected_share - actual_share).abs()),
        _ => None,
    };
    let land_share_match = land_share_delta.is_some_and(|delta| delta <= 0.10);
    if !missing_winner && winner_match && !land_share_match {
        diagnostics.push("land_share_mismatch".to_string());
    }

    let pass = winner_match && timing_match && land_share_match;
    let category = if pass {
        "pass"
    } else if missing_winner {
        "missing_winner"
    } else if !winner_match {
        "wrong_winner"
    } else if !timing_match {
        "timing_mismatch"
    } else {
        "land_share_mismatch"
    };
    OutcomeComparison {
        pass,
        category: category.to_string(),
        diagnostics,
        winner_match,
        timing_match,
        land_share_match,
        tick_delta_ratio,
        land_share_delta,
    }
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
    use crate::game::Player;
    use crate::hash::game_hash;
    use crate::map::TileRef;

    fn sample_outcome(
        winner: Option<&str>,
        terminal_tick: Option<u32>,
        winner_land_share: Option<f64>,
    ) -> GameOutcome {
        GameOutcome {
            schema_version: OUTCOME_SCHEMA_VERSION,
            game_id: "game".to_string(),
            winner: winner.map(str::to_string),
            terminal_tick,
            terminal_reason: Some("land_share".to_string()),
            winner_land_share,
            final_tick: terminal_tick.unwrap_or(100),
            land_tiles_without_fallout: 100,
            final_ranking: Vec::new(),
        }
    }

    fn tick_to_ten(game: &mut Game) {
        game.end_spawn_phase();
        for _ in 0..10 {
            game.execute_next_tick();
        }
    }

    #[test]
    fn outcome_winner_identity_normalization() {
        assert_eq!(
            normalize_winner_value(&serde_json::json!(["player", "client-1"])),
            Some("player:client-1".to_string())
        );
        assert_eq!(
            normalize_winner_value(&serde_json::json!(["team", "Blue", "client-1"])),
            Some("team:Blue".to_string())
        );
        assert_eq!(
            normalize_winner_value(&serde_json::json!(["nation", "Brazil"])),
            Some("nation:Brazil".to_string())
        );
        assert_eq!(normalize_winner_value(&serde_json::json!(["unknown", "x"])), None);
    }

    #[test]
    fn outcome_tolerances_are_inclusive() {
        let expected = sample_outcome(Some("player:a"), Some(100), Some(0.50));
        let boundary = sample_outcome(Some("player:a"), Some(120), Some(0.60));
        assert!(compare_outcomes(&expected, &boundary).pass);

        let outside = sample_outcome(Some("player:a"), Some(121), Some(0.601));
        let comparison = compare_outcomes(&expected, &outside);
        assert!(!comparison.pass);
        assert_eq!(
            comparison.diagnostics,
            vec!["timing_mismatch", "land_share_mismatch"]
        );
    }

    #[test]
    fn outcome_no_winner_is_not_a_pass() {
        let expected = sample_outcome(None, None, None);
        let actual = sample_outcome(None, None, None);
        let comparison = compare_outcomes(&expected, &actual);
        assert!(!comparison.pass);
        assert_eq!(comparison.category, "missing_winner");
        assert_eq!(comparison.diagnostics, vec!["missing_winner"]);
    }

    #[test]
    fn outcome_detects_ffa_terminal_condition() {
        let mut game = Game::default();
        game.add_player(Player {
            id: "player-id".to_string(),
            client_id: "client-id".to_string(),
            name: "Player".to_string(),
            small_id: 1,
            tiles_owned: 1,
            ..Default::default()
        });
        tick_to_ten(&mut game);
        let terminal = terminal_candidate(&game).expect("FFA winner");
        assert_eq!(terminal.winner, "player:client-id");
        assert_eq!(terminal.tick, 10);
        assert_eq!(terminal.reason, "land_share");
    }

    #[test]
    fn outcome_detects_team_terminal_condition() {
        let mut game = Game::default();
        let wire = serde_json::json!({
            "gameMap": "Onion",
            "difficulty": "Medium",
            "donateGold": false,
            "donateTroops": false,
            "gameType": "Public",
            "gameMode": "Team",
            "gameMapSize": "Normal",
            "nations": "disabled",
            "bots": 0,
            "infiniteGold": false,
            "infiniteTroops": false,
            "instantBuild": false,
            "randomSpawn": false
        });
        game.wire = crate::core::config::Config::from_value(&wire, false).unwrap();
        game.add_player(Player {
            id: "player-id".to_string(),
            client_id: "client-id".to_string(),
            name: "Player".to_string(),
            small_id: 1,
            tiles_owned: 1,
            team: Some("Blue".to_string()),
            ..Default::default()
        });
        tick_to_ten(&mut game);
        let terminal = terminal_candidate(&game).expect("team winner");
        assert_eq!(terminal.winner, "team:Blue");
        assert_eq!(terminal.tick, 10);
        assert_eq!(terminal.reason, "land_share");
    }

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
            .unwrap_or_else(|_| crate::util::default_repo_root());
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

    /// Bisection utility: replays `FIND_DIVERGENCE_RECORD` (default
    /// `rN7wbZ1Y`) tick-by-tick and stops at the first tick whose hash
    /// disagrees with the record's own archived hash, printing every
    /// player's tiles/troops/gold/alive/unit-count at that tick (and,
    /// if `FIND_DIVERGENCE_TRACK_NAME` names a player, that single
    /// player's tiles/troops/gold every tick up to it) - the native side of
    /// the same before/after-a-fix bisection loop used throughout
    /// `docs/bot-ai-parity-*/`. Note the printed tick is `HashUpdate::tick`,
    /// which is the *pre-increment* tick just processed (see
    /// `execute_next_tick`, hash is computed before `self.ticks += 1`) - one
    /// less than `game.ticks()` read afterward.
    ///
    /// Usage:
    ///   FIND_DIVERGENCE_RECORD=fkVh9QtC cargo test --release -p \
    ///     openfront-engine --lib find_first_divergence -- --nocapture
    #[test]
    fn find_first_divergence() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
        let repo = std::path::Path::new(&repo_root);
        let record_name =
            std::env::var("FIND_DIVERGENCE_RECORD").unwrap_or_else(|_| "rN7wbZ1Y".into());
        let path = repo.join(format!("records/0c4c7d7993c9/{record_name}.json.gz"));
        let bytes = load_record_bytes(&path).unwrap();
        let rec = GameRecord::from_json_bytes(&bytes).unwrap().decompress();
        let mut game = crate::bootstrap::game_from_record(repo, &rec).unwrap();
        let track_name = std::env::var("FIND_DIVERGENCE_TRACK_NAME").ok();
        let mut processed: u32 = 0;
        for turn in rec.turns.iter() {
            let gid = game.game_id.clone();
            for e in turn_to_executions(&mut game, &gid, &turn.intents) {
                game.add_execution(e);
            }
            let updates = game.execute_next_tick();
            processed += 1;
            if let Some(track_name) = track_name.as_deref() {
                if processed <= 200 {
                    if let Some(p) = game.players_in_order().iter().find(|p| p.name == track_name) {
                        eprintln!(
                            "processed {} (game.ticks()={}) {} tiles={} troops={} gold={}",
                            processed, game.ticks(), track_name, p.tiles_owned, p.troops, p.gold
                        );
                    }
                }
            }
            if let Some(h) = updates.hash {
                let archived = rec.turns.get(h.tick as usize).and_then(|t| t.hash);
                if archived.is_some() && archived != Some(h.hash) {
                    eprintln!(
                        "FIRST DIVERGENCE at tick {} native={} archived={:?}",
                        h.tick, h.hash, archived
                    );
                    for p in game.players_in_order() {
                        eprintln!(
                            "  player {} ({}): tiles={} troops={} gold={} alive={} units={}",
                            p.id, p.name, p.tiles_owned, p.troops, p.gold, p.alive, p.units.len()
                        );
                    }
                    return;
                }
            }
        }
        eprintln!("no divergence found in {} turns", rec.turns.len());
    }

    #[test]
    fn export_tick_state_json() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
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
            .unwrap_or_else(|_| crate::util::default_repo_root());
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
            .unwrap_or_else(|_| crate::util::default_repo_root());
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

    /// `jby2gMJF` matches the archived (ground-truth, from the original live
    /// TS engine) hash exactly through turn 300 - the last tick of the spawn
    /// phase (`Config::num_spawn_phase_turns()`, default 300) - once replayed
    /// against period-correct map assets (`map_dir_for_commit`; this
    /// record's "Two Lakes" map was rebalanced upstream after it was
    /// captured, which desynced native from tick 50 onward when replayed
    /// against the live `openfront/` submodule's *current* map).
    ///
    /// Past turn 300, Nation-AI `AttackExecution` instances (deferred during
    /// the spawn phase, see `NationExecution`) activate en masse and desync
    /// permanently no matter how period-correct the assets are: native's
    /// `for_each_neighbor4` intentionally visits N,S,W,E to match *current*
    /// upstream TS's `neighbors()`, but at this record's own `gitCommit`
    /// TS's `AttackExecution.addNeighbors` still used the older W,E,N,S
    /// order (unified to N,S,W,E one `openfront` commit later than
    /// `PARITY_COMMIT`, see `docs/bot-ai-parity-rate/README.md`). Each
    /// neighbor visited draws one PRNG value while building the conquest
    /// frontier, so the order mismatch reorders every subsequent draw -
    /// this is expected, documented drift in the frozen archive, not a
    /// native bug, so the bound stays at 300 rather than chasing it further.
    #[test]
    fn parity_single_record() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
        let repo = std::path::Path::new(&repo_root);
        let rel = std::env::var("PARITY_RECORD")
            .unwrap_or_else(|_| "records/0c4c7d7993c9/jby2gMJF.json.gz".into());
        let path = if rel.starts_with('/') {
            std::path::PathBuf::from(rel)
        } else {
            repo.join(rel)
        };
        let bound: u32 = std::env::var("PARITY_TICK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300);
        let bytes = load_record_bytes(&path).unwrap();
        let record = GameRecord::from_json_bytes(&bytes).unwrap().decompress();
        let expected = record
            .turns
            .iter()
            .find(|turn| turn.turn_number == bound)
            .and_then(|turn| turn.hash)
            .expect("archived hash at bound tick");
        let game = replay_to_tick(repo, &path, bound);
        assert_eq!(game_hash(&game), expected);
    }

    #[test]
    fn export_transport_path_json() {
        use crate::spatial::{can_build_transport_ship, target_transport_tile};
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
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
            .unwrap_or_else(|_| crate::util::default_repo_root());
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
        // TS `Util.sigmoid`/`Config.attackLogic`'s `defenseSig`:
        // `1 / (1 + exp(-decayRate * (numTilesOwned - midpoint)))` with
        // `DEFENSE_DEBUFF_DECAY_RATE = LN2/50_000`, `DEFENSE_DEBUFF_MIDPOINT
        // = 150_000` (see `openfront/src/core/configuration/Config.ts`).
        // With these constants the sigmoid's floor at `numTilesOwned == 0`
        // is `1/9 ~= 0.111` - NOT near 0 (a previous version of this test
        // asserted `sigmoid(71.0, ..) < 0.01`, which is unsatisfiable for
        // any `numTilesOwned >= 0` given `midpoint - 0 == 3` decay
        // half-lives away, `2^-3 == 0.125`-ish, not 100+ half-lives - that
        // was a bug in the test's own expectation, not in `sigmoid()` or
        // its constants, both of which already match TS's formula and
        // config values exactly). Pin the three mathematically exact anchor
        // points instead: floor (0 tiles), midpoint (150_000 tiles, defined
        // to be exactly 0.5), and a large defender well past the midpoint
        // (approaching but never reaching 1).
        use crate::util::sigmoid;
        let decay = std::f64::consts::LN_2 / 50_000.0;
        let mid = 150_000.0;

        let floor = sigmoid(0.0, decay, mid);
        assert!(
            (floor - 1.0 / 9.0).abs() < 1e-9,
            "0-tile defender sigmoid should be exactly 1/9, got {floor}"
        );

        let at_midpoint = sigmoid(mid, decay, mid);
        assert!(
            (at_midpoint - 0.5).abs() < 1e-9,
            "sigmoid at the midpoint should be exactly 0.5, got {at_midpoint}"
        );

        let large = sigmoid(500_000.0, decay, mid);
        assert!(
            large > 0.99 && large < 1.0,
            "large (500k-tile) defender sigmoid should approach 1, got {large}"
        );

        // Attack-logic's derived "large defender attack debuff":
        // `0.7 + 0.3 * (1 - defenseSig)`. A large defender (sigmoid near 1)
        // pushes the attacker's effective troops toward the 0.7 floor (the
        // strongest defense bonus); a 0-tile defender sits at the sigmoid
        // floor's own ceiling (~0.967), not exactly 1.0.
        let ldb_large = 0.7 + 0.3 * (1.0 - large);
        assert!(
            (ldb_large - 0.7).abs() < 0.01,
            "large defender attack debuff near 0.7 (max bonus), got {ldb_large}"
        );
        let ldb_floor = 0.7 + 0.3 * (1.0 - floor);
        assert!(
            (ldb_floor - (0.7 + 0.3 * (1.0 - 1.0 / 9.0))).abs() < 1e-9,
            "0-tile defender attack debuff should sit at the sigmoid floor's ceiling, got {ldb_floor}"
        );
    }

    #[test]
    fn debug_attack_7mv_turn_353_354() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
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
            .unwrap_or_else(|_| crate::util::default_repo_root());
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
    fn dbg_border_ocean() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
        let repo = std::path::Path::new(&repo_root);
        let target: u32 = std::env::var("EXPORT_TICK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(210);
        let path = std::env::var("EXPORT_RECORD")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| repo.join("records/0c4c7d7993c9/3QNU4eJa.json.gz"));
        let target_id = std::env::var("EXPORT_PLAYER").unwrap_or_else(|_| "tf6l7nfm".into());
        let game = replay_to_tick(repo, &path, target);
        let p = game
            .all_players()
            .iter()
            .find(|p| p.id == target_id)
            .expect("player not found");
        let small_id = p.small_id;
        let mut border: Vec<(u32, u32, bool, bool)> = Vec::new();
        game.for_each_border_tile(small_id, |t| {
            let shore = game.is_shore(t);
            let mut touches_ocean = false;
            game.map.for_each_neighbor4(t, |n| {
                if game.is_water(n) && game.map.is_ocean(n) {
                    touches_ocean = true;
                }
            });
            border.push((game.map.x(t), game.map.y(t), shore, touches_ocean));
        });
        border.sort();
        eprintln!("player {target_id} small_id={small_id} border_count={}", border.len());
        let ocean_touching: Vec<_> = border.iter().filter(|(_, _, _, o)| *o).collect();
        eprintln!("ocean_touching_border_tiles={ocean_touching:?}");
        eprintln!("all_border_tiles={border:?}");
    }

    #[test]
    fn export_state_json() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
        let repo = std::path::Path::new(&repo_root);
        let target: u32 = std::env::var("EXPORT_TICK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(302);
        let path = std::env::var("EXPORT_RECORD")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| repo.join("records/0c4c7d7993c9/tCFq6nPn.json.gz"));
        let game = replay_to_tick(repo, &path, target);
        let mut players: Vec<serde_json::Value> = game
            .all_players()
            .iter()
            .filter(|p| !p.id.is_empty())
            .map(|p| {
                let unit_hash: f64 = p.units.iter().map(crate::hash::unit_hash_js).sum();
                let unit_list: Vec<serde_json::Value> = p
                    .units
                    .iter()
                    .map(|u| {
                        serde_json::json!({
                            "type": u.unit_type,
                            "tile": u.tile,
                            "id": u.id,
                            "hash": crate::hash::unit_hash_js(u),
                            "level": u.level,
                        })
                    })
                    .collect();
                serde_json::json!({
                    "id": p.id,
                    "playerType": match p.player_type {
                        crate::game::PlayerType::Human => "HUMAN",
                        crate::game::PlayerType::Bot => "BOT",
                        crate::game::PlayerType::Nation => "NATION",
                    },
                    "tilesOwned": p.tiles_owned,
                    "troops": p.troops,
                    "gold": p.gold.to_string(),
                    "units": p.units.len(),
                    "unitHash": unit_hash,
                    "unitList": unit_list,
                    "idHash": p.id_hash,
                    "sharedWater": crate::execution::nation_structures::shared_water_components(&game, p.small_id)
                        .map(|s| s.into_iter().collect::<Vec<_>>()),
                })
            })
            .collect();
        players.sort_by(|a, b| {
            b["tilesOwned"]
                .as_i64()
                .unwrap_or(0)
                .cmp(&a["tilesOwned"].as_i64().unwrap_or(0))
        });
        let total_tiles: i64 = players.iter().map(|p| p["tilesOwned"].as_i64().unwrap_or(0)).sum();
        let total_troops: i64 = players.iter().map(|p| p["troops"].as_i64().unwrap_or(0)).sum();
        let order: Vec<&str> = game
            .all_players()
            .iter()
            .filter(|p| !p.id.is_empty())
            .map(|p| p.id.as_str())
            .collect();
        let out = serde_json::json!({
            "tick": game.ticks(),
            "hash": crate::hash::game_hash(&game),
            "totalTiles": total_tiles,
            "totalTroops": total_troops,
            "order": order,
            "players": players,
        });
        println!("{}", serde_json::to_string(&out).unwrap());
    }

    #[test]
    fn debug_raw_mini_path() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
        let repo = std::path::Path::new(&repo_root);
        let record = std::env::var("EXPORT_RECORD")
            .unwrap_or_else(|_| "records/0c4c7d7993c9/MdPDuVXZ.json.gz".into());
        let path = std::path::PathBuf::from(&record);
        let path = if path.is_absolute() { path } else { repo.join(path) };
        let tick: u32 = std::env::var("EXPORT_TICK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(328);
        let player_id = std::env::var("DEBUG_PLAYER_ID").unwrap_or_else(|_| "1tho292q".into());
        let ref_tile: crate::map::TileRef = std::env::var("DEBUG_REF_TILE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(874208);
        let mut game = replay_to_tick(repo, &path, tick);
        let sid = game.player_by_id(&player_id).unwrap().small_id;
        let dst = crate::spatial::target_transport_tile(&mut game, ref_tile).unwrap();
        let mut shores: Vec<crate::map::TileRef> = Vec::new();
        game.for_each_border_tile(sid, |t| {
            if game.is_shore(t) && game.is_land(t) && game.map.owner_id(t) == sid {
                if game.get_water_component(t) == game.get_water_component(dst) {
                    shores.push(t);
                }
            }
        });
        let mini_shores: Vec<crate::map::TileRef> = shores
            .iter()
            .map(|&s| game.mini_map.ref_xy(game.map.x(s) / 2, game.map.y(s) / 2))
            .collect();
        let mini_dst = game.mini_map.ref_xy(game.map.x(dst) / 2, game.map.y(dst) / 2);
        let mini_map = game.mini_map.clone();
        let hpa = game.mini_water_hpa.as_mut().unwrap();
        eprintln!("=== MARKER before hpa.find_path ===");
        let raw = hpa.find_path(&mini_map, &mini_shores, mini_dst).unwrap_or_default();
        eprintln!("rawMini.len()={}", raw.len());
        for (i, &t) in raw.iter().enumerate() {
            eprintln!("  rawMini[{}]={} x={} y={}", i, t, mini_map.x(t), mini_map.y(t));
        }
        let smoothed = crate::water::smooth_water_path_pub(&mini_map, &raw);
        eprintln!("smoothedMini.len()={}", smoothed.len());
        for (i, &t) in smoothed.iter().enumerate() {
            eprintln!("  smoothedMini[{}]={} x={} y={}", i, t, mini_map.x(t), mini_map.y(t));
        }
        if let Ok(spec) = std::env::var("DUMP_TRACE") {
            let mut parts = spec.split(',');
            let from: crate::map::TileRef = parts.next().unwrap().parse().unwrap();
            let to: crate::map::TileRef = parts.next().unwrap().parse().unwrap();
            let trace = crate::water::water_trace_line_pub(&mini_map, from, to);
            eprintln!("trace.len()={:?}", trace.as_ref().map(|t| t.len()));
            for (i, &t) in trace.unwrap_or_default().iter().enumerate() {
                eprintln!("  trace[{}]={} x={} y={}", i, t, mini_map.x(t), mini_map.y(t));
            }
        }
        for spec in std::env::var("DUMP_MINI_TERRAIN").unwrap_or_default().split(',') {
            if spec.is_empty() {
                continue;
            }
            let mut parts = spec.split(':');
            let tx: u32 = parts.next().unwrap().parse().unwrap();
            let ty: u32 = parts.next().unwrap().parse().unwrap();
            let t = mini_map.ref_xy(tx, ty);
            eprintln!(
                "mini terrain x={} y={} byte={} magnitude={} isWater={}",
                tx, ty, mini_map.terrain_byte(t), mini_map.terrain_byte(t) & 0x1f, mini_map.is_water(t)
            );
        }
    }

    #[test]
    fn debug_player_client_id() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
        let repo = std::path::Path::new(&repo_root);
        let record = std::env::var("EXPORT_RECORD")
            .unwrap_or_else(|_| "records/0c4c7d7993c9/1MFxEdwr.json.gz".into());
        let path = std::path::PathBuf::from(&record);
        let path = if path.is_absolute() { path } else { repo.join(path) };
        let tick: u32 = std::env::var("EXPORT_TICK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(379);
        let game = replay_to_tick(repo, &path, tick);
        for pid in std::env::var("PLAYER_IDS")
            .unwrap_or_default()
            .split(',')
            .filter(|s| !s.is_empty())
        {
            match game.player_by_id(pid) {
                Some(p) => eprintln!("id={pid} -> client_id={} small_id={}", p.client_id, p.small_id),
                None => eprintln!("id={pid} -> NOT FOUND"),
            }
        }
    }

    #[test]
    fn debug_transport_src_pick() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
        let repo = std::path::Path::new(&repo_root);
        let record = std::env::var("EXPORT_RECORD")
            .unwrap_or_else(|_| "records/0c4c7d7993c9/MdPDuVXZ.json.gz".into());
        let path = std::path::PathBuf::from(&record);
        let path = if path.is_absolute() { path } else { repo.join(path) };
        let tick: u32 = std::env::var("EXPORT_TICK")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(328);
        let player_id = std::env::var("DEBUG_PLAYER_ID").unwrap_or_else(|_| "1tho292q".into());
        let ref_tile: crate::map::TileRef = std::env::var("DEBUG_REF_TILE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(874208);
        let mut game = replay_to_tick(repo, &path, tick);
        let sid = game.player_by_id(&player_id).unwrap().small_id;
        let dst = crate::spatial::target_transport_tile(&mut game, ref_tile).unwrap();
        eprintln!(
            "dst tile={} x={} y={}",
            dst,
            game.map.x(dst),
            game.map.y(dst)
        );
        let mut shores: Vec<crate::map::TileRef> = Vec::new();
        game.for_each_border_tile(sid, |t| {
            if game.is_shore(t) && game.is_land(t) && game.map.owner_id(t) == sid {
                if game.get_water_component(t) == game.get_water_component(dst) {
                    shores.push(t);
                }
            }
        });
        eprintln!("shores.len()={}", shores.len());
        for (i, &s) in shores.iter().enumerate() {
            eprintln!("  shore[{}]={} x={} y={}", i, s, game.map.x(s), game.map.y(s));
        }
        eprintln!("=== MARKER before plan_water_path_multi ===");
        game.plan_water_path_multi(&shores, dst);
        let path = game.planned_water_path().to_vec();
        eprintln!("path.len()={}", path.len());
        let full_dump = std::env::var("DUMP_FULL_PATH").ok().as_deref() == Some("1");
        let n = if full_dump { path.len() } else { 10 };
        for (i, &t) in path.iter().take(n).enumerate() {
            eprintln!("  path[{}]={} x={} y={}", i, t, game.map.x(t), game.map.y(t));
        }
        let src = crate::spatial::can_build_transport_ship(&mut game, sid, ref_tile);
        eprintln!("chosen src={:?}", src.map(|t| (t, game.map.x(t), game.map.y(t))));
    }

    #[test]
    fn debug_player86_state() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
        let repo = std::path::Path::new(&repo_root);
        let path = repo.join("records/0c4c7d7993c9/3QNU4eJa.json.gz");
        // Replay to tick 191 (state just before tick 191 runs)
        let game = replay_to_tick(repo, &path, 191);
        let p86 = game.player_by_small_id(86).unwrap();
        eprintln!("Player 86: id={} client_id={} type={:?} troops={} tiles={} disconnected={}", p86.id, p86.client_id, p86.player_type, p86.troops, p86.tiles_owned, p86.is_disconnected);
        // Print alliances
        for al in &game.alliances {
            if al.requestor_small_id == 86 || al.recipient_small_id == 86 {
                let other_id = if al.requestor_small_id == 86 { al.recipient_small_id } else { al.requestor_small_id };
                let other = game.player_by_small_id(other_id).unwrap();
                eprintln!("  Alliance with: id={} disconnected={} troops={}", other.id, other.is_disconnected, other.troops);
            }
        }
        // Print bordering players
        let bordering = crate::execution::ai_attack::collect_bordering_players_pub(&game, 86);
        eprintln!("Bordering players ({}): ", bordering.len());
        for sid in &bordering {
            let p = game.player_by_small_id(*sid).unwrap();
            let friendly = game.is_friendly(86, *sid);
            eprintln!("  sid={} id={} disconnected={} friendly={} troops={}", sid, p.id, p.is_disconnected, friendly, p.troops);
        }
    }

    #[test]
    fn export_exec_order() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
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
            .unwrap_or_else(|_| crate::util::default_repo_root());
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
            .unwrap_or_else(|_| crate::util::default_repo_root());
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
            .unwrap_or_else(|_| crate::util::default_repo_root());
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
            .unwrap_or_else(|_| crate::util::default_repo_root());
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
            .unwrap_or_else(|_| crate::util::default_repo_root());
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

    /// Tile/player state at turn 317 depends on period-correct map assets
    /// (see `map_dir_for_commit`'s doc comment - this record's "Two Lakes"
    /// map was rebalanced upstream after capture) even though turn 317 is
    /// past the point (turn 300, spawn-phase end) where full hash parity
    /// with the archive is expected to hold - see `parity_single_record`'s
    /// doc comment for why native permanently desyncs from the archived
    /// hash past that point. `attack_logic_at_tile` is a pure function of
    /// its inputs (attacker/defender troop and tile counts, terrain), so as
    /// long as *this specific pair's* state happens to be unaffected by the
    /// post-300 neighbor-order desync, the exact expected numbers below
    /// remain meaningful as a formula-correctness regression pin (verified
    /// against TS's `Config.attackLogic()` - see
    /// `docs/bot-ai-parity-nation-relations/README.md`) once replayed
    /// against the right-vintage terrain.
    #[test]
    fn compare_attack_logic_turn_317() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
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

    #[test]
    fn random_boat_target_filters_match_tcf_through_turn_430() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
        let repo = std::path::Path::new(&repo_root);
        let path = repo.join("records/0c4c7d7993c9/tCFq6nPn.json.gz");
        let bytes = load_record_bytes(&path).unwrap();
        let record = GameRecord::from_json_bytes(&bytes).unwrap().decompress();
        let expected = record
            .turns
            .iter()
            .find(|turn| turn.turn_number == 430)
            .and_then(|turn| turn.hash)
            .expect("archived hash at turn 430");
        let game = replay_to_tick(repo, &path, 430);
        assert_eq!(game_hash(&game), expected);
    }

    #[test]
    fn island_target_filters_match_fdh_through_turn_440() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
        let repo = std::path::Path::new(&repo_root);
        let path = repo.join("records/0c4c7d7993c9/fdh3gYAF.json.gz");
        let bytes = load_record_bytes(&path).unwrap();
        let record = GameRecord::from_json_bytes(&bytes).unwrap().decompress();
        let expected = record
            .turns
            .iter()
            .find(|turn| turn.turn_number == 440)
            .and_then(|turn| turn.hash)
            .expect("archived hash at turn 440");
        let game = replay_to_tick(repo, &path, 440);
        assert_eq!(game_hash(&game), expected);
    }

    #[test]
    fn warship_patrol_matches_fk_through_turn_510() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
        let repo = std::path::Path::new(&repo_root);
        let path = repo.join("records/0c4c7d7993c9/fkVh9QtC.json.gz");
        let bytes = load_record_bytes(&path).unwrap();
        let record = GameRecord::from_json_bytes(&bytes).unwrap().decompress();
        let expected = record
            .turns
            .iter()
            .find(|turn| turn.turn_number == 510)
            .and_then(|turn| turn.hash)
            .expect("archived hash at turn 510");
        let game = replay_to_tick(repo, &path, 510);
        assert_eq!(game_hash(&game), expected);
    }

    #[test]
    fn warship_shells_match_transport_and_warship_targets() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
        let repo = std::path::Path::new(&repo_root);
        for (record_id, turn_number) in [("x7pvCXU3", 500), ("rN7wbZ1Y", 620)] {
            let path = repo.join(format!("records/0c4c7d7993c9/{record_id}.json.gz"));
            let bytes = load_record_bytes(&path).unwrap();
            let record = GameRecord::from_json_bytes(&bytes).unwrap().decompress();
            let expected = record
                .turns
                .iter()
                .find(|turn| turn.turn_number == turn_number)
                .and_then(|turn| turn.hash)
                .unwrap_or_else(|| panic!("archived hash at turn {turn_number}"));
            let game = replay_to_tick(repo, &path, turn_number);
            assert_eq!(
                game_hash(&game),
                expected,
                "{record_id} at turn {turn_number}"
            );
        }
    }

    #[test]
    fn damaged_warship_heals_and_retreats_in_rn7wbz1y() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
        let repo = std::path::Path::new(&repo_root);
        let path = repo.join("records/0c4c7d7993c9/rN7wbZ1Y.json.gz");

        let before = replay_to_tick(repo, &path, 659);
        let before_ship = before
            .player_by_id("ommet66i")
            .and_then(|player| player.units.iter().find(|unit| unit.id == 913))
            .expect("warship 913 before shell impact");
        assert_eq!(before_ship.health, 725);
        assert_eq!(before_ship.tile, 2_026_488);

        let after = replay_to_tick(repo, &path, 660);
        let after_ship = after
            .player_by_id("ommet66i")
            .and_then(|player| player.units.iter().find(|unit| unit.id == 913))
            .expect("warship 913 after shell impact");
        assert_eq!(after_ship.health, 726);
        assert_eq!(after_ship.tile, 2_024_288);
    }

    #[test]
    fn docked_warship_gets_active_healing_in_rn7wbz1y() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
        let repo = std::path::Path::new(&repo_root);
        let path = repo.join("records/0c4c7d7993c9/rN7wbZ1Y.json.gz");

        let before = replay_to_tick(repo, &path, 671);
        let before_ship = before
            .player_by_id("xgfz7usc")
            .and_then(|player| player.units.iter().find(|unit| unit.id == 1369))
            .expect("warship 1369 upon docking");
        assert_eq!(before_ship.health, 460);
        assert_eq!(before_ship.tile, 3_820_018);

        let after = replay_to_tick(repo, &path, 672);
        let after_ship = after
            .player_by_id("xgfz7usc")
            .and_then(|player| player.units.iter().find(|unit| unit.id == 1369))
            .expect("warship 1369 after docked healing");
        assert_eq!(after_ship.health, 466);
        assert_eq!(after_ship.tile, 3_820_018);

        let settled = replay_to_tick(repo, &path, 690);
        let settled_player = settled
            .player_by_id("xgfz7usc")
            .expect("docked warship owner at turn 690");
        let settled_ship = settled_player
            .units
            .iter()
            .find(|unit| unit.id == 1369)
            .expect("warship 1369 while docked");
        assert_eq!(settled_ship.health, 174);
        let shell = settled_player
            .units
            .iter()
            .find(|unit| unit.id == 1563)
            .expect("last shell fired before docking");
        assert_eq!(shell.tile, 3_800_207);
        assert!(!settled_player.units.iter().any(|unit| unit.id == 1564));
    }

    #[test]
    fn warship_hunts_and_captures_trade_ship_in_rn7wbz1y() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
        let repo = std::path::Path::new(&repo_root);
        let path = repo.join("records/0c4c7d7993c9/rN7wbZ1Y.json.gz");

        let before = replay_to_tick(repo, &path, 720);
        let before_ship = before
            .player_by_id("xgfz7usc")
            .and_then(|player| player.units.iter().find(|unit| unit.id == 1468))
            .expect("warship 1468 before piracy pursuit");
        assert_eq!(before_ship.tile, 3_811_208);
        assert!(before
            .player_by_id("0s73rd8u")
            .is_some_and(|player| player.units.iter().any(|unit| unit.id == 1187)));

        let pursuing = replay_to_tick(repo, &path, 721);
        let pursuing_ship = pursuing
            .player_by_id("xgfz7usc")
            .and_then(|player| player.units.iter().find(|unit| unit.id == 1468))
            .expect("warship 1468 pursuing trade ship 1187");
        assert_eq!(pursuing_ship.tile, 3_811_206);

        let captured = replay_to_tick(repo, &path, 729);
        let captor = captured
            .player_by_id("xgfz7usc")
            .expect("trade ship captor at turn 729");
        let captured_trade_ship = captor
            .units
            .iter()
            .find(|unit| unit.id == 1187)
            .expect("captured trade ship 1187");
        assert_eq!(captured_trade_ship.tile, 3_797_992);
        assert!(!captured
            .player_by_id("0s73rd8u")
            .is_some_and(|player| player.units.iter().any(|unit| unit.id == 1187)));
    }

    #[test]
    fn warship_replans_patrol_after_trade_hunt_in_rn7wbz1y() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
        let repo = std::path::Path::new(&repo_root);
        let path = repo.join("records/0c4c7d7993c9/rN7wbZ1Y.json.gz");

        let before = replay_to_tick(repo, &path, 746);
        let before_ship = before
            .player_by_id("xgfz7usc")
            .and_then(|player| player.units.iter().find(|unit| unit.id == 1468))
            .expect("warship 1468 before resuming patrol");
        assert_eq!(before_ship.tile, 3_835_404);

        let after = replay_to_tick(repo, &path, 747);
        let after_ship = after
            .player_by_id("xgfz7usc")
            .and_then(|player| player.units.iter().find(|unit| unit.id == 1468))
            .expect("warship 1468 after patrol replanning");
        assert_eq!(after_ship.tile, 3_837_604);
    }

    #[test]
    fn manual_boat_retreat_matches_giq_through_turn_450() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
        let repo = std::path::Path::new(&repo_root);
        let path = repo.join("records/0c4c7d7993c9/GiQovEcP.json.gz");
        let bytes = load_record_bytes(&path).unwrap();
        let record = GameRecord::from_json_bytes(&bytes).unwrap().decompress();
        let expected = record
            .turns
            .iter()
            .find(|turn| turn.turn_number == 450)
            .and_then(|turn| turn.hash)
            .expect("archived hash at turn 450");
        let game = replay_to_tick(repo, &path, 450);
        assert_eq!(game_hash(&game), expected);
    }

    #[test]
    fn boat_attack_cancellation_matches_2dg_through_turn_670() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
        let repo = std::path::Path::new(&repo_root);
        let path = repo.join("records/0c4c7d7993c9/2dG9dxmX.json.gz");
        let bytes = load_record_bytes(&path).unwrap();
        let record = GameRecord::from_json_bytes(&bytes).unwrap().decompress();
        let expected = record
            .turns
            .iter()
            .find(|turn| turn.turn_number == 670)
            .and_then(|turn| turn.hash)
            .expect("archived hash at turn 670");
        let game = replay_to_tick(repo, &path, 670);
        assert_eq!(game_hash(&game), expected);
    }

    #[test]
    fn trace_alliance_exec_86_wnep5pzi() {
        let repo_root = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| crate::util::default_repo_root());
        let repo = std::path::Path::new(&repo_root);
        let path = repo.join("records/0c4c7d7993c9/3QNU4eJa.json.gz");

        let bytes = load_record_bytes(&path).unwrap();
        let rec = GameRecord::from_json_bytes(&bytes).unwrap().decompress();
        let mut game = crate::bootstrap::game_from_record(repo, &rec).unwrap();

        // Replay up to tick 191
        for turn in rec.turns.iter() {
            if turn.turn_number > 191 {
                break;
            }
            let gid = game.game_id.clone();
            for e in turn_to_executions(&mut game, &gid, &turn.intents) {
                game.add_execution(e);
            }
            game.execute_next_tick();
        }

        let wnep5pzi_sid = game.player_by_id("wnep5pzi").map(|p| p.small_id).unwrap_or(9999);
        eprintln!("wnep5pzi small_id={wnep5pzi_sid}");

        // Print state at tick 191 (before tick 192 runs)
        let all_reqs_86: Vec<String> = game.alliance_requests.iter()
            .filter(|r| r.requestor_small_id == 86 || r.recipient_small_id == 86)
            .map(|r| format!("{}->{}:{:?}(at={})", r.requestor_small_id, r.recipient_small_id, r.status, r.created_at))
            .collect();
        eprintln!("tick=191 all alliance_requests involving 86: {all_reqs_86:?}");

        let all_reqs_343: Vec<String> = game.alliance_requests.iter()
            .filter(|r| r.requestor_small_id == wnep5pzi_sid || r.recipient_small_id == wnep5pzi_sid)
            .map(|r| format!("{}->{}:{:?}(at={})", r.requestor_small_id, r.recipient_small_id, r.status, r.created_at))
            .collect();
        eprintln!("tick=191 all alliance_requests involving 343: {all_reqs_343:?}");

        let alliances_86: Vec<String> = game.alliances.iter()
            .filter(|a| a.requestor_small_id == 86 || a.recipient_small_id == 86)
            .map(|a| format!("{}<->{}(expires={})", a.requestor_small_id, a.recipient_small_id, a.expires_at))
            .collect();
        eprintln!("tick=191 alliances involving 86: {alliances_86:?}");

        let can_send = game.can_send_alliance_request(86, wnep5pzi_sid);
        eprintln!("tick=191 can_send_alliance_request(86, {wnep5pzi_sid})={can_send}");

        for turn in rec.turns.iter() {
            if turn.turn_number < 192 || turn.turn_number > 225 {
                continue;
            }
            let tick = turn.turn_number;

            let pre_reqs: Vec<String> = game.alliance_requests.iter()
                .filter(|r| r.requestor_small_id == 86 || r.recipient_small_id == 86
                    || r.requestor_small_id == wnep5pzi_sid || r.recipient_small_id == wnep5pzi_sid)
                .map(|r| format!("{}->{}:{:?}(at={})", r.requestor_small_id, r.recipient_small_id, r.status, r.created_at))
                .collect();
            if !pre_reqs.is_empty() {
                eprintln!("tick={tick} PRE  req_state: {pre_reqs:?}");
            }

            let gid = game.game_id.clone();
            for e in turn_to_executions(&mut game, &gid, &turn.intents) {
                game.add_execution(e);
            }
            game.execute_next_tick();

            let alliance_execs: Vec<(usize, String)> = game.exec_labels().into_iter().enumerate()
                .filter(|(_, l)| l.contains("AllianceRequest")
                    && (l.contains("wnep5pzi") || l.contains(&format!("({}->", 86)) || l.contains(&format!("->{})", 86))))
                .collect();
            if !alliance_execs.is_empty() {
                eprintln!("tick={tick} POST alliance execs: {alliance_execs:?}");
            }

            let post_reqs: Vec<String> = game.alliance_requests.iter()
                .filter(|r| r.requestor_small_id == 86 || r.recipient_small_id == 86
                    || r.requestor_small_id == wnep5pzi_sid || r.recipient_small_id == wnep5pzi_sid)
                .map(|r| format!("{}->{}:{:?}(at={})", r.requestor_small_id, r.recipient_small_id, r.status, r.created_at))
                .collect();
            if !post_reqs.is_empty() {
                eprintln!("tick={tick} POST req_state: {post_reqs:?}");
            }

            let alliances_86_wnep5pzi: Vec<String> = game.alliances.iter()
                .filter(|a| (a.requestor_small_id == 86 && a.recipient_small_id == wnep5pzi_sid)
                    || (a.requestor_small_id == wnep5pzi_sid && a.recipient_small_id == 86))
                .map(|a| format!("alliance {}<->{} expires={}", a.requestor_small_id, a.recipient_small_id, a.expires_at))
                .collect();
            if !alliances_86_wnep5pzi.is_empty() {
                eprintln!("tick={tick} POST alliance: {alliances_86_wnep5pzi:?}");
            }
        }
    }
}
