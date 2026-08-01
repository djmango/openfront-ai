//! Tick-level tile/troop/gold trace dumper for the bot-AI native-vs-TS
//! parity investigation. Replays a `GameRecord` natively and, every
//! `--every` ticks, snapshots per-player (tiles owned, alive, troops,
//! gold) plus totals, so a Python/TS diff script can find the first tick
//! where a bot/nation's territory share diverges from the TS oracle.
//!
//! Usage:
//!   cargo run --release -p openfront-engine --bin tick_dump -- \
//!     --repo <repo_root> --record <record.json[.gz]> --every 50 \
//!     --out /tmp/native_ticks.json [--max-ticks N]
//!
//! Streaming mode (for `scripts/hash_parity.sh` early-stop compare):
//!   --ndjson writes one JSON object per line and flushes each tick so a
//!   parallel TS dump can be compared online without buffering the full game.
//!
//! Daemon / resume mode (for `scripts/hash_bisect.sh` true binary search):
//!   --daemon --repo R --record REC
//!   stdin commands (one per line):
//!     STATUS | RESET | ADVANCE <tick> | DUMP <path> [units] | QUIT
//!   stdout: `OK tick=N` / `ERR ...`
//!   ADVANCE only moves forward (in-memory resume). RESET reboots from tick 0.
//!
//! Live expand control (hash_parity in-place unit expand, no re-replay):
//!   OF_DUMP_CONTROL=/path/to/file — polled each tick; body `EXPAND until=N`
//!   enables unit dumps and continues until tick N then exits.
use clap::Parser;
use openfront_engine::execution::intent::turn_to_executions;
use openfront_engine::game::Game;
use openfront_engine::record::GameRecord;
use serde::Serialize;
use std::collections::HashMap;
use std::io::{BufRead, Read, Write};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "tick_dump")]
struct Args {
    #[arg(long)]
    repo: PathBuf,
    #[arg(long)]
    record: Option<PathBuf>,
    #[arg(long, default_value_t = 50)]
    every: u32,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long)]
    max_ticks: Option<u32>,
    /// Stream one TickSnapshot JSON object per line (plus a header line).
    /// Enables online native-vs-TS compare with early process kill.
    #[arg(long, default_value_t = false)]
    ndjson: bool,
    /// Interactive stdin command mode for mid-game resume / binary search.
    #[arg(long, default_value_t = false)]
    daemon: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct UnitSnapshot {
    id: i32,
    unit_type: String,
    tile: i32,
    hash: i64,
    level: i32,
    under_construction: bool,
    health: i32,
    veterancy: i32,
    veterancy_progress: i32,
    target_tile: Option<i32>,
    patrol_tile: Option<u32>,
    retreat_port: Option<u32>,
    retreating: bool,
    docked: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PlayerSnapshot {
    identity: String,
    id: String,
    small_id: u16,
    name: String,
    player_type: String,
    team: Option<String>,
    tiles: i32,
    troops: i32,
    gold: i64,
    alive: bool,
    hash: i64,
    /// IEEE-754 bits of the pre-truncation player hash float (decimal string).
    hash_bits: String,
    /// Sum of per-unit hashes (cheap; isolates unit-list drift when tiles/troops match).
    units_hash: i64,
    num_units: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    units: Option<Vec<UnitSnapshot>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    owned_tiles: Option<Vec<i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    border_order: Option<Vec<i32>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    owned_order: Option<Vec<i32>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RailroadSnapshot {
    id: u32,
    from: u32,
    to: u32,
    tiles: Vec<u32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StationSnapshot {
    id: u32,
    unit_id: i32,
    unit_type: String,
    tile: Option<u32>,
    railroads: Vec<u32>,
    cluster: Option<u32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AttackSnapshot {
    owner_small_id: u16,
    target_small_id: u16,
    troops: i64,
    active: bool,
    attack_live: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TickSnapshot {
    tick: u32,
    in_spawn_phase: bool,
    total_land_tiles: u32,
    total_owned_tiles: i32,
    /// `GameImpl.hash()`-compatible aggregate (players folded like TS).
    game_hash: i64,
    /// IEEE-754 bits of the pre-truncation game hash float (decimal string).
    game_hash_bits: String,
    players: Vec<PlayerSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    railroads: Option<Vec<RailroadSnapshot>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stations: Option<Vec<StationSnapshot>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    attacks: Option<Vec<AttackSnapshot>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct Dump {
    engine: &'static str,
    game_id: String,
    every: u32,
    final_tick: u32,
    ticks: Vec<TickSnapshot>,
}

fn player_identity(p: &openfront_engine::game::Player) -> String {
    if p.client_id.is_empty() {
        format!("nation:{}", p.name)
    } else {
        format!("player:{}", p.client_id)
    }
}

fn load_record_bytes(path: &std::path::Path) -> Result<Vec<u8>, String> {
    let raw = std::fs::read(path).map_err(|e| e.to_string())?;
    if path.extension().and_then(|s| s.to_str()) == Some("gz") {
        let mut dec = flate2::read::GzDecoder::new(&raw[..]);
        let mut out = Vec::new();
        dec.read_to_end(&mut out).map_err(|e| e.to_string())?;
        Ok(out)
    } else {
        Ok(raw)
    }
}

fn snapshot(
    game: &openfront_engine::game::Game,
    dump_units: bool,
    dump_owned_tiles: bool,
    dump_border_order: bool,
    dump_owned_order: bool,
    dump_rails: bool,
    dump_attacks: bool,
) -> TickSnapshot {
    let warships: HashMap<(u16, i32), &openfront_engine::execution::WarshipExecution> = game
        .live_warships()
        .filter_map(|warship| Some(((warship.owner_small_id(), warship.unit_id()?), warship)))
        .collect();
    let players: Vec<PlayerSnapshot> = game
        .all_players()
        .iter()
        .map(|p| {
            let units = if dump_units {
                Some(
                    p.units
                        .iter()
                        .map(|u| {
                            let warship = warships.get(&(p.small_id, u.id)).copied();
                            UnitSnapshot {
                                id: u.id,
                                unit_type: u.unit_type.clone(),
                                tile: u.tile,
                                hash: openfront_engine::hash::unit_hash(u),
                                level: u.level,
                                under_construction: u.under_construction,
                                health: u.health,
                                veterancy: u.veterancy,
                                veterancy_progress: u.veterancy_progress,
                                target_tile: warship
                                    .and_then(|w| w.target_tile())
                                    .map(|t| t as i32),
                                patrol_tile: warship.map(|w| w.patrol_tile()),
                                retreat_port: warship.and_then(|w| w.retreat_port()),
                                retreating: warship.is_some_and(|w| w.is_retreating()),
                                docked: warship.is_some_and(|w| w.is_docked()),
                            }
                        })
                        .collect(),
                )
            } else {
                None
            };
            let owned_tiles = if dump_owned_tiles {
                let mut v: Vec<i32> = p.owned_tiles.iter().map(|&t| t as i32).collect();
                v.sort_unstable();
                Some(v)
            } else {
                None
            };
            let border_order = if dump_border_order {
                Some(p.border_tiles.iter().map(|t| t as i32).collect())
            } else {
                None
            };
            let owned_order = if dump_owned_order {
                Some(p.owned_tiles.iter().map(|&t| t as i32).collect())
            } else {
                None
            };
            PlayerSnapshot {
                identity: player_identity(p),
                id: p.id.clone(),
                small_id: p.small_id,
                name: p.name.clone(),
                player_type: format!("{:?}", p.player_type),
                team: p.team.clone(),
                tiles: p.tiles_owned,
                troops: p.troops,
                gold: p.gold,
                // TS `PlayerImpl.isAlive()` is `_tiles.size > 0`; report the
                // same value here so early failed-spawn diagnostics are not
                // polluted by native's internal sticky flag.
                alive: p.tiles_owned > 0,
                hash: openfront_engine::hash::player_hash(p),
                hash_bits: openfront_engine::hash::player_hash_js(p).to_bits().to_string(),
                units_hash: p
                    .units
                    .iter()
                    .map(openfront_engine::hash::unit_hash)
                    .sum(),
                num_units: p.units.len(),
                units,
                owned_tiles,
                border_order,
                owned_order,
            }
        })
        .collect();
    let total_owned_tiles: i32 = players.iter().map(|p| p.tiles).sum();
    let mut game_hash_f64 = 1.0_f64;
    for p in game.players_in_order() {
        game_hash_f64 += openfront_engine::hash::player_hash_js(p);
    }
    let game_hash = game_hash_f64 as i64;
    let game_hash_bits = game_hash_f64.to_bits().to_string();
    let (railroads, stations) = if dump_rails {
        let mut rails: Vec<RailroadSnapshot> = game
            .rail_network
            .railroads
            .values()
            .map(|r| RailroadSnapshot {
                id: r.id,
                from: r.from,
                to: r.to,
                tiles: r.tiles.clone(),
            })
            .collect();
        rails.sort_by_key(|r| r.id);
        let mut sts: Vec<StationSnapshot> = game
            .rail_network
            .stations
            .values()
            .map(|s| StationSnapshot {
                id: s.id,
                unit_id: s.unit_id,
                unit_type: s.unit_type.clone(),
                tile: openfront_engine::rail::station_tile(game, &game.rail_network, s.id),
                railroads: s.railroads.clone(),
                cluster: s.cluster,
            })
            .collect();
        sts.sort_by_key(|s| s.id);
        (Some(rails), Some(sts))
    } else {
        (None, None)
    };
    let attacks = if dump_attacks {
        Some(
            game.active_attacks_debug()
                .into_iter()
                .map(
                    |(owner_small_id, target_small_id, troops, active, attack_live, _, _)| {
                        AttackSnapshot {
                            owner_small_id,
                            target_small_id,
                            troops: troops as i64,
                            active,
                            attack_live,
                        }
                    },
                )
                .collect(),
        )
    } else {
        None
    };
    TickSnapshot {
        tick: game.ticks(),
        in_spawn_phase: game.in_spawn_phase(),
        total_land_tiles: game.num_land_tiles(),
        total_owned_tiles,
        game_hash,
        game_hash_bits,
        players,
        railroads,
        stations,
        attacks,
    }
}

fn poll_expand_control(path: &std::path::Path) -> Option<u32> {
    let body = std::fs::read_to_string(path).ok()?;
    let body = body.trim();
    if let Some(rest) = body.strip_prefix("EXPAND") {
        for part in rest.split_whitespace() {
            if let Some(v) = part.strip_prefix("until=") {
                return v.parse().ok();
            }
        }
        // Bare EXPAND — caller should set a default until.
        return Some(u32::MAX);
    }
    None
}

fn advance_to(game: &mut Game, record: &GameRecord, target: u32) -> Result<(), String> {
    if target < game.ticks() {
        return Err(format!(
            "cannot rewind: at tick {}, requested {}",
            game.ticks(),
            target
        ));
    }
    while game.ticks() < target {
        let turn_idx = game.ticks() as usize;
        if turn_idx >= record.turns.len() {
            break;
        }
        let turn = &record.turns[turn_idx];
        let gid = game.game_id.clone();
        for execution in turn_to_executions(game, &gid, &turn.intents) {
            game.add_execution(execution);
        }
        game.execute_next_tick();
    }
    Ok(())
}

fn write_single_tick_dump(path: &std::path::Path, game: &Game, game_id: &str, units: bool) {
    let snap = snapshot(game, units, false, false, false, false, false);
    let dump = Dump {
        engine: "native",
        game_id: game_id.to_string(),
        every: 1,
        final_tick: snap.tick,
        ticks: vec![snap],
    };
    std::fs::write(path, serde_json::to_string(&dump).unwrap()).expect("write dump");
}

fn run_daemon(repo: &std::path::Path, record_path: &std::path::Path) {
    let bytes = load_record_bytes(record_path).expect("read record");
    let record = GameRecord::from_json_bytes(&bytes)
        .expect("parse record")
        .decompress();
    let game_id = record.info.game_id.clone();
    let bootstrap = || {
        openfront_engine::bootstrap::game_from_record(repo, &record).expect("bootstrap")
    };
    let mut game = bootstrap();
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout();
    writeln!(stdout, "OK tick={}", game.ticks()).ok();
    stdout.flush().ok();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                writeln!(stdout, "ERR read {e}").ok();
                break;
            }
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let cmd = parts.next().unwrap_or("").to_ascii_uppercase();
        match cmd.as_str() {
            "STATUS" => {
                writeln!(stdout, "OK tick={}", game.ticks()).ok();
            }
            "RESET" => {
                game = bootstrap();
                writeln!(stdout, "OK tick={}", game.ticks()).ok();
            }
            "ADVANCE" => {
                let Some(t) = parts.next().and_then(|s| s.parse::<u32>().ok()) else {
                    writeln!(stdout, "ERR usage ADVANCE <tick>").ok();
                    stdout.flush().ok();
                    continue;
                };
                match advance_to(&mut game, &record, t) {
                    Ok(()) => writeln!(stdout, "OK tick={}", game.ticks()).ok(),
                    Err(e) => writeln!(stdout, "ERR {e}").ok(),
                };
            }
            "DUMP" => {
                let Some(path) = parts.next() else {
                    writeln!(stdout, "ERR usage DUMP <path> [units]").ok();
                    stdout.flush().ok();
                    continue;
                };
                let units = parts.any(|p| p == "units" || p == "units=1");
                write_single_tick_dump(std::path::Path::new(path), &game, &game_id, units);
                writeln!(stdout, "OK tick={} path={path}", game.ticks()).ok();
            }
            "QUIT" | "EXIT" => {
                writeln!(stdout, "OK bye").ok();
                stdout.flush().ok();
                break;
            }
            _ => {
                writeln!(stdout, "ERR unknown command {cmd}").ok();
            }
        }
        stdout.flush().ok();
    }
}

fn main() {
    let args = Args::parse();
    if args.daemon {
        let record = args.record.expect("--record required for --daemon");
        run_daemon(&args.repo, &record);
        return;
    }
    let record_path = args.record.expect("--record required");
    let out_path = args.out.expect("--out required");

    let mut dump_units = std::env::var_os("OF_DUMP_UNITS").is_some();
    let dump_owned_tiles = std::env::var_os("OF_DUMP_OWNED_TILES").is_some();
    let dump_border_order = std::env::var_os("OF_DUMP_BORDER_ORDER").is_some();
    let dump_owned_order = std::env::var_os("OF_DUMP_OWNED_ORDER").is_some();
    let dump_rails = std::env::var_os("OF_DUMP_RAILS").is_some();
    let dump_attacks = std::env::var_os("OF_DUMP_ATTACKS").is_some();
    let dump_units_from: u32 = std::env::var("OF_DUMP_UNITS_FROM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let dump_ticks_from: u32 = std::env::var("OF_DUMP_TICKS_FROM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let control_path = std::env::var_os("OF_DUMP_CONTROL").map(PathBuf::from);
    let ndjson = args.ndjson
        || out_path.extension().and_then(|s| s.to_str()) == Some("ndjson")
        || std::env::var_os("OF_DUMP_NDJSON").is_some();
    let bytes = load_record_bytes(&record_path).expect("read record");
    let record = GameRecord::from_json_bytes(&bytes)
        .expect("parse record")
        .decompress();
    let mut game =
        openfront_engine::bootstrap::game_from_record(&args.repo, &record).expect("bootstrap");

    let mut ndjson_file = if ndjson {
        let mut f = std::fs::File::create(&out_path).expect("create ndjson out");
        let header = serde_json::json!({
            "type": "header",
            "engine": "native",
            "gameId": record.info.game_id,
            "every": args.every,
        });
        writeln!(f, "{header}").expect("write header");
        f.flush().ok();
        Some(f)
    } else {
        None
    };
    let mut buffered: Vec<TickSnapshot> = Vec::new();
    let mut last_emitted_tick: Option<u32> = None;
    let mut expand_until: Option<u32> = None;

    for turn in &record.turns {
        if let Some(max) = args.max_ticks {
            if turn.turn_number > max {
                break;
            }
        }
        if let Some(until) = expand_until {
            if game.ticks() >= until {
                break;
            }
        }
        let gid = game.game_id.clone();
        for execution in turn_to_executions(&mut game, &gid, &turn.intents) {
            game.add_execution(execution);
        }
        game.execute_next_tick();

        if let Some(ref ctrl) = control_path {
            if expand_until.is_none() {
                if let Some(until) = poll_expand_control(ctrl) {
                    let until = if until == u32::MAX {
                        game.ticks().saturating_add(25)
                    } else {
                        until
                    };
                    expand_until = Some(until);
                    dump_units = true;
                    eprintln!(
                        "[tick_dump] EXPAND control → units until tick {until} (at {})",
                        game.ticks()
                    );
                }
            }
        }

        if game.ticks() < dump_ticks_from {
            continue;
        }
        let every = if expand_until.is_some() { 1 } else { args.every };
        if game.ticks() % every == 0 {
            let include_units = dump_units && game.ticks() >= dump_units_from;
            let snap = snapshot(
                &game,
                include_units,
                dump_owned_tiles,
                dump_border_order,
                dump_owned_order,
                dump_rails,
                dump_attacks,
            );
            last_emitted_tick = Some(snap.tick);
            if let Some(f) = ndjson_file.as_mut() {
                let line = serde_json::to_string(&snap).expect("serialize snap");
                writeln!(f, "{line}").expect("write snap");
                f.flush().ok();
            } else {
                buffered.push(snap);
            }
        }
        if let Some(until) = expand_until {
            if game.ticks() >= until {
                break;
            }
        }
    }
    if game.ticks() >= dump_ticks_from && last_emitted_tick != Some(game.ticks()) {
        let include_units = dump_units && game.ticks() >= dump_units_from;
        let snap = snapshot(
            &game,
            include_units,
            dump_owned_tiles,
            dump_border_order,
            dump_owned_order,
            dump_rails,
            dump_attacks,
        );
        if let Some(f) = ndjson_file.as_mut() {
            let line = serde_json::to_string(&snap).expect("serialize snap");
            writeln!(f, "{line}").expect("write snap");
            f.flush().ok();
        } else {
            buffered.push(snap);
        }
    }

    if ndjson {
        eprintln!(
            "[tick_dump] streamed ndjson to {} (final tick {})",
            out_path.display(),
            game.ticks()
        );
        return;
    }

    let dump = Dump {
        engine: "native",
        game_id: record.info.game_id.clone(),
        every: args.every,
        final_tick: game.ticks(),
        ticks: buffered,
    };
    std::fs::write(&out_path, serde_json::to_string(&dump).unwrap()).expect("write out");
    eprintln!(
        "[tick_dump] wrote {} snapshots to {} (final tick {})",
        dump.ticks.len(),
        out_path.display(),
        dump.final_tick
    );
}
