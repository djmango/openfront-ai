//! `ofcuda_spawn` - engine ground truth for the CUDA port's spawn / initial
//! state at **arbitrary N bots**.
//!
//! This is an *oracle*: it drives the real engine (`openfront-engine`, linked
//! as a library) exactly the way `RlSession::reset` does for an RL episode, and
//! dumps everything the port needs to reproduce the spawn layer for N bots plus
//! the nations variants. The ground truth is built from the engine, never from
//! the CUDA code.
//!
//! What is dumped (text reference, one record per line):
//!   * `player` - every registered player: small_id, type, id, spawn tile,
//!     the tick its spawn was placed, and its final `tiles_owned` count.
//!   * `bot`    - the N bots in spawn order: id, spawn seed
//!     (`simple_hash(pid) + simple_hash(game_id)`), spawn tile.
//!   * `owner`  - the owner plane each bot's selection ran against (the
//!     footprint tiles of every earlier bot), as `(tile, small_id)` overrides.
//!   * `prev`   - the already-placed spawn centres each bot's selection ran
//!     against.
//!   * `owned`  - the tiles each bot ends up owning (== its spawn footprint,
//!     because the spawn path uses `require_all_valid = true`).
//!
//! Usage:
//!   cargo run --release -- --agents 22 --nations 0 --seed parity --out ref.txt
//!   cargo run --release -- --agents 18 --nations 1 --seed parity
//!
//! `--agents` is the bot count (`cfg.bots()`), which is the N in the V10
//! curriculum's `V10_BOT_NATION_DENSITY` table.

use clap::Parser;
use openfront_engine::game::{Game, PlayerType};
use openfront_engine::rl::RlSession;
use openfront_engine::util::simple_hash;
use serde_json::{json, Value};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "ofcuda_spawn", version, about)]
struct Args {
    /// OpenFront repo root (dir containing `openfront/resources/maps`).
    #[arg(long, default_value = "/opt/data/workspaces/skg/openfront-ai")]
    repo: PathBuf,
    /// Episode seed (`seed_to_game_id` input).
    #[arg(long, default_value = "parity")]
    seed: String,
    /// Map key (`RlSession::reset`'s `map_key`).
    #[arg(long, default_value = "Pangaea")]
    map: String,
    /// Number of bots (`cfg.bots()`) - the N to support.
    #[arg(long, default_value_t = 2)]
    agents: u32,
    /// Nations config: `disabled`, `default`, or a count (the V10 table's
    /// `Nations::Exact(n)` is a count). Default `0` = no nations but spawning
    /// enabled (what stage 0..7 use).
    #[arg(long, default_value = "0")]
    nations: String,
    /// Number of RL humans (RlSession clamps to 1..=2).
    #[arg(long, default_value_t = 1)]
    human_agents: u32,
    /// Difficulty string (`Easy` matches the V10 early stages).
    #[arg(long, default_value = "Easy")]
    difficulty: String,
    /// Max engine ticks to run while waiting for spawns to land.
    #[arg(long, default_value_t = 12)]
    max_ticks: u32,
    /// Output path (stdout when omitted).
    #[arg(long)]
    out: Option<PathBuf>,
}

fn nations_value(spec: &str) -> Value {
    match spec {
        "disabled" | "default" => Value::String(spec.to_string()),
        n => match n.parse::<u32>() {
            Ok(v) => json!(v),
            Err(_) => {
                eprintln!("--nations must be `disabled`, `default` or an integer, got {spec:?}");
                std::process::exit(2);
            }
        },
    }
}

fn type_tag(t: PlayerType) -> &'static str {
    match t {
        PlayerType::Human => "H",
        PlayerType::Bot => "B",
        PlayerType::Nation => "N",
    }
}

fn main() {
    let args = Args::parse();
    let nvalue = nations_value(&args.nations);

    let (mut session, _, _, _, _, _) = match RlSession::reset(
        &args.repo,
        &args.map,
        &args.seed,
        args.agents,
        &args.difficulty,
        nvalue.clone(),
        args.human_agents,
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("RlSession::reset failed: {e}");
            std::process::exit(2);
        }
    };

    let game_id = session.game_id().to_string();
    let game_hash = simple_hash(&game_id);
    let (width, height) = (session.game.width(), session.game.height());

    // Tick one at a time and record the first tick each player gets a spawn
    // tile. This is the empirical proof of *spawn ordering*: which players
    // exist on the owner plane when each bot selects.
    let mut spawn_tick: std::collections::HashMap<u16, u32> = std::collections::HashMap::new();
    let mut tick = 0u32;
    let n_bots_expected = args.agents as usize;
    let n_nations_expected = {
        let g: &Game = &session.game;
        g.all_players()
            .iter()
            .filter(|p| p.player_type == PlayerType::Nation)
            .count()
    };
    while tick < args.max_ticks {
        let before: Vec<u16> = session
            .game
            .all_players()
            .iter()
            .filter(|p| p.spawn_tile.is_some())
            .map(|p| p.small_id)
            .collect();
        session.game.execute_next_tick();
        tick += 1;
        for p in session.game.all_players() {
            if p.spawn_tile.is_some() && !before.contains(&p.small_id) {
                spawn_tick.insert(p.small_id, tick);
            }
        }
        let spawned_bots = session
            .game
            .all_players()
            .iter()
            .filter(|p| p.player_type == PlayerType::Bot && p.spawn_tile.is_some())
            .count();
        let spawned_nations = session
            .game
            .all_players()
            .iter()
            .filter(|p| p.player_type == PlayerType::Nation && p.spawn_tile.is_some())
            .count();
        if spawned_bots >= n_bots_expected && spawned_nations >= n_nations_expected {
            break;
        }
    }

    let game: &Game = &session.game;

    // ------------------------------------------------------------------ dump
    let mut o = String::new();
    let mut w = |s: String| {
        o.push_str(&s);
        o.push('\n');
    };
    w("# ofcuda_spawn reference v1  (engine ground truth, arbitrary N)".into());
    w(format!("# repo {}", args.repo.display()));
    w(format!("map {}", args.map));
    w(format!("seed {}", args.seed));
    w(format!("game_id {game_id}"));
    w(format!("game_hash {game_hash}"));
    w(format!("agents {}", args.agents));
    w(format!("nations {}", args.nations));
    w(format!("human_agents {}", args.human_agents));
    w(format!("width {width}"));
    w(format!("height {height}"));
    w(format!(
        "min_dist {}",
        game.wire.min_distance_between_players()
    ));

    // Players in insertion order (humans, nations, then bots as they spawn).
    for p in game.all_players() {
        w(format!(
            "player {} {} {} {} {} {}",
            p.small_id,
            type_tag(p.player_type),
            p.id,
            p.spawn_tile.map(|t| t as i64).unwrap_or(-1),
            spawn_tick.get(&p.small_id).copied().unwrap_or(0),
            p.tiles_owned
        ));
    }

    // Bots in spawn order (small_id order == add_from_info order == the order
    // their `SpawnExecution`s ticked).
    let mut bots: Vec<_> = game
        .all_players()
        .iter()
        .filter(|p| p.player_type == PlayerType::Bot)
        .collect();
    bots.sort_by_key(|p| p.small_id);

    for (bi, b) in bots.iter().enumerate() {
        let seed_sum = simple_hash(&b.id).wrapping_add(game_hash);
        let tile = b.spawn_tile.map(|t| t as i64).unwrap_or(-1);
        w(format!(
            "bot {bi} {} {} {seed_sum} {tile} {} {} {}",
            b.small_id,
            b.id,
            tile.max(0) as u32 % width,
            tile.max(0) as u32 / width,
            b.tiles_owned
        ));

        // The owner plane this bot selected against: exactly the union of the
        // footprints of the bots that spawned before it (nations and the human
        // have no spawn tile at that point - proven by the `player` tick
        // column, which is >= this bot's own spawn tick for all of them).
        for prev_bot in bots.iter().take(bi) {
            for t in &prev_bot.owned_tiles {
                w(format!("owner {bi} {t} {}", prev_bot.small_id));
            }
        }
        for prev_bot in bots.iter().take(bi) {
            if let Some(t) = prev_bot.spawn_tile {
                w(format!("prev {bi} {t}"));
            }
        }
        let mut own = b.owned_tiles.clone();
        own.sort_unstable();
        for t in &own {
            w(format!("owned {bi} {t}"));
        }
    }

    match &args.out {
        Some(p) => {
            if let Some(parent) = p.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            std::fs::write(p, &o).expect("write reference");
            eprintln!(
                "wrote {} (agents={} spawned={} players={} tick={})",
                p.display(),
                args.agents,
                bots.iter().filter(|b| b.spawn_tile.is_some()).count(),
                game.all_players().len(),
                tick,
            );
        }
        None => print!("{o}"),
    }
}
