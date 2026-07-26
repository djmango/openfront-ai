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
use clap::Parser;
use openfront_engine::execution::intent::turn_to_executions;
use openfront_engine::record::GameRecord;
use serde::Serialize;
use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "tick_dump")]
struct Args {
    #[arg(long)]
    repo: PathBuf,
    #[arg(long)]
    record: PathBuf,
    #[arg(long, default_value_t = 50)]
    every: u32,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    max_ticks: Option<u32>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct UnitSnapshot {
    id: i32,
    unit_type: String,
    tile: i32,
    hash: i64,
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
    name: String,
    player_type: String,
    team: Option<String>,
    tiles: i32,
    troops: i32,
    gold: i64,
    alive: bool,
    hash: i64,
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
struct TickSnapshot {
    tick: u32,
    in_spawn_phase: bool,
    total_land_tiles: u32,
    total_owned_tiles: i32,
    players: Vec<PlayerSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    railroads: Option<Vec<RailroadSnapshot>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stations: Option<Vec<StationSnapshot>>,
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
                num_units: p.units.len(),
                units,
                owned_tiles,
                border_order,
                owned_order,
            }
        })
        .collect();
    let total_owned_tiles: i32 = players.iter().map(|p| p.tiles).sum();
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
    TickSnapshot {
        tick: game.ticks(),
        in_spawn_phase: game.in_spawn_phase(),
        total_land_tiles: game.num_land_tiles(),
        total_owned_tiles,
        players,
        railroads,
        stations,
    }
}

fn main() {
    let args = Args::parse();
    let dump_units = std::env::var_os("OF_DUMP_UNITS").is_some();
    let dump_owned_tiles = std::env::var_os("OF_DUMP_OWNED_TILES").is_some();
    let dump_border_order = std::env::var_os("OF_DUMP_BORDER_ORDER").is_some();
    let dump_owned_order = std::env::var_os("OF_DUMP_OWNED_ORDER").is_some();
    let dump_rails = std::env::var_os("OF_DUMP_RAILS").is_some();
    let dump_units_from: u32 = std::env::var("OF_DUMP_UNITS_FROM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let dump_ticks_from: u32 = std::env::var("OF_DUMP_TICKS_FROM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let bytes = load_record_bytes(&args.record).expect("read record");
    let record = GameRecord::from_json_bytes(&bytes)
        .expect("parse record")
        .decompress();
    let mut game =
        openfront_engine::bootstrap::game_from_record(&args.repo, &record).expect("bootstrap");

    let mut out = Vec::new();
    for turn in &record.turns {
        if let Some(max) = args.max_ticks {
            if turn.turn_number > max {
                break;
            }
        }
        let gid = game.game_id.clone();
        for execution in turn_to_executions(&mut game, &gid, &turn.intents) {
            game.add_execution(execution);
        }
        game.execute_next_tick();
        if game.ticks() < dump_ticks_from {
            continue;
        }
        if game.ticks() % args.every == 0 {
            let include_units = dump_units && game.ticks() >= dump_units_from;
            out.push(snapshot(
                &game,
                include_units,
                dump_owned_tiles,
                dump_border_order,
                dump_owned_order,
                dump_rails,
            ));
        }
    }
    // Always capture the true final state even if it doesn't land on an
    // `every`-tick boundary.
    if game.ticks() >= dump_ticks_from && out.last().map(|s| s.tick) != Some(game.ticks()) {
        let include_units = dump_units && game.ticks() >= dump_units_from;
        out.push(snapshot(
            &game,
            include_units,
            dump_owned_tiles,
            dump_border_order,
            dump_owned_order,
            dump_rails,
        ));
    }

    let dump = Dump {
        engine: "native",
        game_id: record.info.game_id.clone(),
        every: args.every,
        final_tick: game.ticks(),
        ticks: out,
    };
    std::fs::write(&args.out, serde_json::to_string(&dump).unwrap()).expect("write out");
    eprintln!(
        "[tick_dump] wrote {} snapshots to {} (final tick {})",
        dump.ticks.len(),
        args.out.display(),
        dump.final_tick
    );
}
