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
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TickSnapshot {
    tick: u32,
    in_spawn_phase: bool,
    total_land_tiles: u32,
    total_owned_tiles: i32,
    players: Vec<PlayerSnapshot>,
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

fn snapshot(game: &openfront_engine::game::Game) -> TickSnapshot {
    let players: Vec<PlayerSnapshot> = game
        .all_players()
        .iter()
        .map(|p| PlayerSnapshot {
            identity: player_identity(p),
            id: p.id.clone(),
            name: p.name.clone(),
            player_type: format!("{:?}", p.player_type),
            team: p.team.clone(),
            tiles: p.tiles_owned,
            troops: p.troops,
            gold: p.gold,
            alive: p.alive,
            hash: openfront_engine::hash::player_hash(p),
            num_units: p.units.len(),
        })
        .collect();
    let total_owned_tiles: i32 = players.iter().map(|p| p.tiles).sum();
    TickSnapshot {
        tick: game.ticks(),
        in_spawn_phase: game.in_spawn_phase(),
        total_land_tiles: game.num_land_tiles(),
        total_owned_tiles,
        players,
    }
}

fn main() {
    let args = Args::parse();
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
        if game.ticks() % args.every == 0 {
            out.push(snapshot(&game));
        }
    }
    // Always capture the true final state even if it doesn't land on an
    // `every`-tick boundary.
    if out.last().map(|s| s.tick) != Some(game.ticks()) {
        out.push(snapshot(&game));
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
