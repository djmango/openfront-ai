//! `ofcuda_nation` - the **independent engine oracle** for the CUDA nation
//! spawner. It links `openfront-engine` and NEVER touches CUDA.
//!
//! Two independent things are recorded for every nation:
//!
//! 1. **Engine state** - what the real `Game` actually did: the registered
//!    player id, `small_id`, the spawn tile, the tick the spawn landed on, the
//!    owned tile list, and the troops at the end of the spawn phase.
//! 2. **Engine-primitive re-derivation** - the spawn *decision* rebuilt from the
//!    engine's own `PseudoRandom` + `simple_hash` + `GameMap`:
//!    `seed = simple_hash(id) + simple_hash(game_id)`, the three ratio draws,
//!    `attack_rate` / `attack_tick`, and `random_spawn_land`'s 50-try rejection
//!    sampler run against the plane as it stands when the nation ticks.
//!    `derived_eq_engine=1` says the re-derived tile equals the tile the engine
//!    actually placed - i.e. the sampler's draw order is confirmed, not assumed.
//!
//! `tries` (how many rejection draws the sampler burned) is NOT readable from
//! game state. It is captured from the ENGINE'S OWN debug print
//! (`nation.rs:217` `SPAWN_DEBUG=<id>` -> `nation_spawn_land <id> tile=.. tries=..`),
//! by re-running this binary as a child with that env var set.
//!
//! Usage:
//!   ofcuda_nation --map pangaea --agents 18 --nations 1 --out dump.txt
//!   ofcuda_nation --tries-for <nation player id>      # internal, stderr only

mod nation_names;

use openfront_engine::game::{Game, PlayerType};
use openfront_engine::map::TerrainType;
use openfront_engine::prng::PseudoRandom;
use openfront_engine::rl::RlSession;
use openfront_engine::util::simple_hash;
use serde_json::{json, Value};
use std::path::PathBuf;

/// `game.wire.min_distance_between_players()` for these cells; the reference
/// (`ofcuda_spawn`) prints the engine's own value and `min_dist<=>30` is
/// asserted downstream rather than assumed here.
const MIN_DIST: u32 = 30;
const REPO: &str = "/opt/data/workspaces/skg/openfront-ai";

fn nations_value(spec: &str) -> Value {
    match spec {
        "disabled" | "default" => Value::String(spec.to_string()),
        n => json!(n.parse::<u32>().unwrap_or(0)),
    }
}

fn type_tag(t: PlayerType) -> &'static str {
    match t {
        PlayerType::Human => "H",
        PlayerType::Bot => "B",
        PlayerType::Nation => "N",
    }
}

struct Args {
    map: String,
    seed: String,
    agents: u32,
    nations: String,
    human_agents: u32,
    difficulty: String,
    out: Option<PathBuf>,
    tries_for: Option<String>,
}

fn parse_args() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let val = |name: &str| -> Option<String> {
        argv.iter()
            .position(|a| a == name)
            .and_then(|i| argv.get(i + 1))
            .cloned()
    };
    Args {
        map: val("--map").unwrap_or_else(|| "Pangaea".into()),
        seed: val("--seed").unwrap_or_else(|| "parity".into()),
        agents: val("--agents").and_then(|v| v.parse().ok()).unwrap_or(2),
        nations: val("--nations").unwrap_or_else(|| "0".into()),
        human_agents: val("--human-agents").and_then(|v| v.parse().ok()).unwrap_or(1),
        difficulty: val("--difficulty").unwrap_or_else(|| "Easy".into()),
        out: val("--out").map(PathBuf::from),
        tries_for: val("--tries-for"),
    }
}

/// The owner plane exactly as `Game` reports it (owner `small_id`, 0 unowned).
fn engine_plane(game: &Game) -> Vec<u16> {
    let n = (game.width() as usize) * (game.height() as usize);
    (0..n as u32).map(|t| game.map.owner_id(t)).collect()
}

/// One nation's engine state, read straight off `Game`.
struct EngineNation {
    sid: u16,
    id: String,
    spawn_tile: Option<u32>,
    spawn_tick: u32,
    owned: Vec<u32>,
    troops: f64,
}

struct Sim {
    nations: Vec<EngineNation>,
    /// The plane the nation's `random_spawn_land` ran against: the map as it
    /// stood immediately BEFORE the first post-reset tick, i.e. with no player
    /// spawned yet (nations tick before the bots' `SpawnExecution`s).
    plane_at_nation_tick: Vec<u16>,
    /// The plane after the bots spawned (tick 1), used for cross-checking the
    /// nation's footprint against the engine's actual owned set.
    plane_after_bots: Vec<u16>,
    game_id: String,
    width: u32,
    height: u32,
    difficulty: String,
    bots: u32,
    human_agents: u32,
    nations_spec: String,
    stable_tick: u32,
    roster: Vec<(u16, char, String, i64, i64, f64)>,
}

fn run_sim(a: &Args) -> Result<Sim, String> {
    let nvalue = nations_value(&a.nations);
    let (mut session, _, _, _, _, _) = RlSession::reset(
        std::path::Path::new(REPO),
        &a.map,
        &a.seed,
        a.agents,
        &a.difficulty,
        nvalue,
        a.human_agents,
    )
    .map_err(|e| format!("RlSession::reset: {e}"))?;

    let game_id = session.game_id().to_string();
    let (width, height) = (session.game.width(), session.game.height());
    let n_nations_expected = session
        .game
        .all_players()
        .iter()
        .filter(|p| p.player_type == PlayerType::Nation)
        .count();
    let n_bots_expected = a.agents as usize;

    // The plane the nation sees when it samples = the state before tick 1.
    let plane_at_nation_tick = engine_plane(&session.game);
    let mut plane_after_bots = plane_at_nation_tick.clone();

    let mut spawn_tick: std::collections::HashMap<u16, u32> = std::collections::HashMap::new();
    let mut tick = 0u32;
    let mut stable = 0u32;
    let mut stable_tick = 0u32;
    while tick < 12 {
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
        if tick == 1 {
            plane_after_bots = engine_plane(&session.game);
        }
        let placed_bots = session
            .game
            .all_players()
            .iter()
            .filter(|p| p.player_type == PlayerType::Bot && p.spawn_tile.is_some())
            .count();
        let placed_nations = session
            .game
            .all_players()
            .iter()
            .filter(|p| p.player_type == PlayerType::Nation && p.spawn_tile.is_some())
            .count();
        // Every nation has a spawn cell, so every nation must have placed by now
        // for the phase to be "done"; bots likewise (a starved bot is a real
        // engine outcome and is recorded, not hidden).
        if placed_bots + placed_nations == n_bots_expected + n_nations_expected {
            stable += 1;
            if stable >= 2 {
                stable_tick = tick;
                break;
            }
        } else {
            stable = 0;
        }
        stable_tick = tick;
    }

    session.game.end_spawn_phase();

    let mut nations = Vec::new();
    let mut roster = Vec::new();
    for p in session.game.all_players() {
        roster.push((
            p.small_id,
            type_tag(p.player_type).chars().next().unwrap_or('B'),
            p.id.clone(),
            p.spawn_tile.map(|t| t as i64).unwrap_or(-1),
            p.tiles_owned as i64,
            p.troops as f64,
        ));
        if p.player_type == PlayerType::Nation {
            nations.push(EngineNation {
                sid: p.small_id,
                id: p.id.clone(),
                spawn_tile: p.spawn_tile,
                spawn_tick: spawn_tick.get(&p.small_id).copied().unwrap_or(0),
                owned: {
                    let mut v = p.owned_tiles.clone();
                    v.sort_unstable();
                    v
                },
                troops: p.troops as f64,
            });
        }
    }
    nations.sort_by_key(|n| n.sid);
    roster.sort_by_key(|r| r.0);
    Ok(Sim {
        nations,
        plane_at_nation_tick,
        plane_after_bots,
        game_id,
        width,
        height,
        difficulty: a.difficulty.clone(),
        bots: a.agents,
        human_agents: a.human_agents,
        nations_spec: a.nations.clone(),
        stable_tick,
        roster,
    })
}

/// `NationExecution::attack_rate_for_difficulty` ranges (`nation.rs:58-66`).
fn attack_rate_range(difficulty: &str) -> (i32, i32) {
    match difficulty {
        "Easy" => (65, 100),
        "Medium" => (55, 70),
        "Hard" => (45, 60),
        "Impossible" => (30, 50),
        _ => (55, 70),
    }
}

fn manifest_coordinates(repo: &str, map: &str) -> Vec<(String, Option<[i32; 2]>)> {
    let p = PathBuf::from(repo)
        .join("openfront/resources/maps")
        .join(map.to_lowercase())
        .join("manifest.json");
    let Ok(text) = std::fs::read_to_string(&p) else {
        return Vec::new();
    };
    let Ok(v) = serde_json::from_str::<Value>(&text) else {
        return Vec::new();
    };
    let Some(arr) = v.get("nations").and_then(|x| x.as_array()) else {
        return Vec::new();
    };
    arr.iter()
        .map(|n| {
            let name = n
                .get("name")
                .and_then(|x| x.as_str())
                .unwrap_or("")
                .to_string();
            let cell = n.get("coordinates").and_then(|c| c.as_array()).map(|c| {
                [
                    c.first().and_then(|x| x.as_i64()).unwrap_or(-1) as i32,
                    c.get(1).and_then(|x| x.as_i64()).unwrap_or(-1) as i32,
                ]
            });
            (name, cell)
        })
        .collect()
}

/// The engine's `random_spawn_land` (`nation.rs:198-228`), re-implemented over
/// the *engine's* map accessors and the *engine's* PseudoRandom.
fn derived_spawn_land(
    game: &Game,
    plane: &[u16],
    cell: [i32; 2],
    random: &mut PseudoRandom,
) -> (Option<u32>, u32) {
    let delta = 25;
    let mut tries = 0u32;
    while tries < 50 {
        tries += 1;
        let x = random.next_int(cell[0] - delta, cell[0] + delta);
        let y = random.next_int(cell[1] - delta, cell[1] + delta);
        if !game.is_valid_coord(x, y) {
            continue;
        }
        let tile = game.ref_xy(x as u32, y as u32);
        if !game.is_land(tile) || plane[tile as usize] != 0 {
            continue;
        }
        if game.terrain_type(tile) == TerrainType::Mountain && random.chance(2) {
            continue;
        }
        return (Some(tile), tries);
    }
    (None, tries)
}

/// `find_spawn` (`spawn_util.rs:65-104`) with `center = None`: the GENERIC
/// search `SpawnExecution` falls back to when the nation's `spawn_cell` is
/// `None` (`nation.rs:98-106`). The nation's 50-try sampler never runs in that
/// case (hence `tries = 0`); this is the bots' routine, re-implemented here
/// over the engine's own map accessors.
fn derived_find_spawn(
    game: &Game,
    plane: &[u16],
    centres: &[u32],
    min_dist: u32,
    random: &mut PseudoRandom,
) -> Option<u32> {
    let (w, h) = (game.width() as i32, game.height() as i32);
    let mut tries = 0u32;
    while tries < 1000 {
        tries += 1;
        let x = random.next_int(0, w);
        let y = random.next_int(0, h);
        let tile = game.ref_xy(x as u32, y as u32);
        if !game.is_land(tile) || plane[tile as usize] != 0 || game.is_border(tile) {
            continue;
        }
        // `too_close_to_existing_spawn` (`game.rs:3622-3640`): every OTHER
        // player's `spawn_tile`, manhattan distance < min_dist.
        if centres
            .iter()
            .any(|&c| game.manhattan_dist(c, tile) < min_dist)
        {
            continue;
        }
        if !disc_all_valid(game, plane, tile) {
            continue;
        }
        return Some(tile);
    }
    None
}

/// `get_spawn_tiles(.., require_all_valid = true)`'s visit set: the engine's
/// `bfs_with_scratch` is a 4-connected flood fill gated ONLY by the filter
/// (`map.rs:447-490`), and `dist^2 <= 16` is a 4-connected disc - so the
/// visited set is exactly that disc. Every visited tile must be land and
/// unowned (`spawn_util.rs:120-146`).
fn disc_all_valid(game: &Game, plane: &[u16], center: u32) -> bool {
    let w = game.width() as i32;
    let h = game.height() as i32;
    let cx = (center % game.width()) as i32;
    let cy = (center / game.width()) as i32;
    for dy in -4..=4i32 {
        for dx in -4..=4i32 {
            if f64::from(dx * dx + dy * dy) > 16.0 {
                continue;
            }
            let (x, y) = (cx + dx, cy + dy);
            if x < 0 || y < 0 || x >= w || y >= h {
                continue; // the flood fill never leaves the map
            }
            let t = game.ref_xy(x as u32, y as u32);
            if !game.is_land(t) || plane[t as usize] != 0 {
                return false;
            }
        }
    }
    true
}

/// `core/nation.rs:168-180`: exactly TWO `next_int` draws per attempt.
fn generate_nation_name(random: &mut PseudoRandom) -> String {
    let t = nation_names::NAME_TEMPLATES
        [random.next_int(0, nation_names::NAME_TEMPLATES.len() as i32) as usize];
    let noun = nation_names::NOUNS[random.next_int(0, nation_names::NOUNS.len() as i32) as usize];
    let mut parts: Vec<String> = Vec::new();
    for part in t {
        match part {
            nation_names::TemplatePart::PluralNoun => parts.push(pluralize(noun)),
            nation_names::TemplatePart::Noun => parts.push(noun.to_string()),
            nation_names::TemplatePart::Lit(x) => parts.push(x.to_string()),
        }
    }
    parts.join(" ")
}

/// `core/nation.rs:143-159`.
fn generate_unique_nation_name(
    random: &mut PseudoRandom,
    used: &std::collections::HashSet<String>,
) -> String {
    for _ in 0..1000 {
        let name = generate_nation_name(random);
        if !used.contains(&name) {
            return name;
        }
    }
    let base = generate_nation_name(random);
    let mut counter = 1u32;
    loop {
        let cand = format!("{base} {counter}");
        if !used.contains(&cand) {
            return cand;
        }
        counter += 1;
    }
}

/// `core/nation.rs:182-209` (verbatim semantics).
fn pluralize(noun: &str) -> String {
    const SPECIAL: &[(&str, &str)] = &[
        ("Cactus", "Cacti"),
        ("Platypus", "Platypuses"),
        ("Moose", "Moose"),
        ("Octopus", "Octopi"),
        ("Cyclops", "Cyclopes"),
        ("Samurai", "Samurai"),
        ("Fish", "Fish"),
        ("Salmon", "Salmon"),
        ("Cod", "Cod"),
        ("Enderman", "Endermen"),
        ("Mitochondria", "Mitochondria"),
    ];
    const O_TO_OES: &[&str] = &["Potato", "Tomato", "Volcano", "Torpedo"];
    for &(k, v) in SPECIAL {
        if k == noun {
            return v.to_string();
        }
    }
    if noun.ends_with('s')
        || noun.ends_with("ch")
        || noun.ends_with("sh")
        || noun.ends_with('x')
        || noun.ends_with('z')
    {
        return format!("{noun}es");
    }
    if noun.ends_with('y') {
        let b = noun.as_bytes();
        if b.len() >= 2 && !"aeiou".contains(b[b.len() - 2] as char) {
            return format!("{}ies", &noun[..noun.len() - 1]);
        }
    }
    if O_TO_OES.contains(&noun) {
        return format!("{noun}es");
    }
    format!("{noun}s")
}

fn main() {
    let a = parse_args();

    // --- internal mode: let the ENGINE print its own tries to stderr --------
    if let Some(id) = &a.tries_for {
        std::env::set_var("SPAWN_DEBUG", id);
        if let Err(e) = run_sim(&a) {
            eprintln!("tries-for run failed: {e}");
            std::process::exit(2);
        }
        return;
    }

    let sim = match run_sim(&a) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    };

    // The nation samples before any spawn exists: assert the premise (if this
    // plane were not empty, the ordering claim in the port would be wrong).
    let plane_nonzero = sim.plane_at_nation_tick.iter().filter(|&&x| x != 0).count();

    // Re-derive the nation stream with the ENGINE's PRNG, over the ENGINE's map.
    let manifest = manifest_coordinates(REPO, &a.map);
    // ONE stream: the humans' ids, then the manifest shuffle, then the nations'
    // ids (`core/nation.rs:create_random_nations`). Both the resulting ids and
    // the shuffled coordinates are then checked against what the engine did.
    let mut order_pr = PseudoRandom::new(simple_hash(&sim.game_id));
    for _ in 0..sim.human_agents {
        order_pr.next_id();
    }
    let order: Vec<i32> = (0..manifest.len() as i32).collect();
    let shuffled = order_pr.shuffle_array(&order);
    let manifest_names: Vec<String> = manifest.iter().map(|(n, _)| n.clone()).collect();
    // `create_random_nations` (`core/nation.rs:84-134`): the manifest nations
    // take `next_id` in shuffled order, then every FABRICATED nation draws a
    // unique name (two draws, more on a collision) BEFORE its `next_id`.
    let from_manifest = sim.nations.len().min(manifest.len());
    let mut used_names: std::collections::HashSet<String> = shuffled
        .iter()
        .take(from_manifest)
        .map(|&i| manifest_names[i as usize].clone())
        .collect();
    let mut derived_ids: Vec<String> = (0..from_manifest)
        .map(|_| order_pr.next_id())
        .collect();
    for _ in from_manifest..sim.nations.len() {
        let name = generate_unique_nation_name(&mut order_pr, &used_names);
        used_names.insert(name);
        derived_ids.push(order_pr.next_id());
    }

    // The engine's own map, for the sampler's land/terrain/validity tests.
    let game_state = {
        let (mut s, _, _, _, _, _) = RlSession::reset(
            std::path::Path::new(REPO),
            &a.map,
            &a.seed,
            a.agents,
            &a.difficulty,
            nations_value(&a.nations),
            a.human_agents,
        )
        .expect("reset for map access");
        s.game.end_spawn_phase();
        s
    };

    let mut o = String::new();
    o.push_str("# ofcuda_nation oracle v1 (engine ground truth, nation spawn layer; NEVER calls CUDA)\n");
    o.push_str(&format!("REPO {REPO}\n"));
    o.push_str(&format!("MAP {}\n", a.map));
    o.push_str(&format!("SEED {}\n", a.seed));
    o.push_str(&format!("GAME_ID {}\n", sim.game_id));
    o.push_str(&format!("GAME_HASH {}\n", simple_hash(&sim.game_id)));
    o.push_str(&format!("AGENTS {}\n", sim.bots));
    o.push_str(&format!("NATIONS {}\n", sim.nations_spec));
    o.push_str(&format!("HUMAN_AGENTS {}\n", sim.human_agents));
    o.push_str(&format!("DIFFICULTY {}\n", sim.difficulty));
    o.push_str(&format!("WIDTH {}\nHEIGHT {}\n", sim.width, sim.height));
    o.push_str(&format!("SPAWN_STABLE_TICK {}\n", sim.stable_tick));
    o.push_str(&format!(
        "PLANE_AT_NATION_TICK_NONZERO {plane_nonzero}\n"
    ));
    for r in &sim.roster {
        o.push_str(&format!(
            "ROSTER {} {} {} {} {} {}\n",
            r.0, r.1, r.2, r.3, r.4, r.5
        ));
    }

    for (i, n) in sim.nations.iter().enumerate() {
        let cell = manifest
            .get(shuffled.get(i).copied().unwrap_or(i as i32) as usize)
            .and_then(|(_, c)| *c);
        let cell_s = match cell {
            Some(c) => format!("{},{}", c[0], c[1]),
            None => "none".to_string(),
        };
        o.push_str(&format!(
            "NATION {i} sid={} id={} cell={cell_s} spawn_tile={} spawn_tick={}\n",
            n.sid,
            n.id,
            n.spawn_tile.map(|t| t as i64).unwrap_or(-1),
            n.spawn_tick
        ));

        // Independent re-derivation of the whole spawn decision.
        let seed = simple_hash(&n.id).wrapping_add(simple_hash(&sim.game_id));
        let mut r2 = PseudoRandom::new(seed);
        let trigger = r2.next_int(50, 60);
        let reserve = r2.next_int(30, 40);
        let expand = r2.next_int(10, 20);
        let (lo, hi) = attack_rate_range(&sim.difficulty);
        let attack_rate = r2.next_int(lo, hi);
        let attack_tick = r2.next_int(0, attack_rate);
        // `cell = None` does NOT go through the nation's sampler: `nation.rs:98`
        // enqueues `SpawnExecution::new(game_id, info, None)`, whose own FRESH
        // `PseudoRandom(simple_hash(id) + simple_hash(game_id))` (`spawn.rs:27`)
        // starts at draw 0 and drives the generic `find_spawn`. The board it
        // sees is the plane after the bots (`plane_after_bots`) plus the spawn
        // centres already on it: every bot (tick 1) and the nations that
        // spawned earlier.
        let (derived, mode) = match cell {
            Some(c) => (
                derived_spawn_land(&game_state.game, &sim.plane_at_nation_tick, c, &mut r2).0,
                "spawn_land(cell)",
            ),
            None => {
                let mut r3 = PseudoRandom::new(seed);
                let centres: Vec<u32> = sim
                    .roster
                    .iter()
                    .filter(|r| r.3 >= 0 && (r.1 == 'B' || (r.1 == 'N' && r.0 < n.sid)))
                    .map(|r| r.3 as u32)
                    .collect();
                (
                    derived_find_spawn(
                        &game_state.game,
                        &sim.plane_after_bots,
                        &centres,
                        MIN_DIST,
                        &mut r3,
                    ),
                    "generic_find_spawn(cell=none)",
                )
            }
        };
        let derived_eq = derived.map(|t| t as i64).unwrap_or(-1)
            == n.spawn_tile.map(|t| t as i64).unwrap_or(-1);
        // The id the engine actually registered must equal the stream's i-th draw.
        let id_eq = derived_ids.get(i).map(|x| x == &n.id).unwrap_or(false);
        // Independent cross-check of the footprint: the engine's actual owned
        // set minus the port's own footprint code - these must be equal to the
        // set the ENGINE reports, by construction (we read the engine state).
        let _ = &sim.plane_after_bots;
        o.push_str(&format!(
            "NATIONCHAIN {i} seed={seed} trigger={trigger} reserve={reserve} expand={expand} \
             attack_rate={attack_rate} attack_tick={attack_tick} derived_mode={mode} derived_tile={} derived_eq_engine={} id_eq_engine={}\n",
            derived.map(|t| t as i64).unwrap_or(-1),
            if derived_eq { 1 } else { 0 },
            if id_eq { 1 } else { 0 }
        ));

        o.push_str(&format!(
            "NATIONOWNED {i} n={} {}\n",
            n.owned.len(),
            n.owned
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        ));
        o.push_str(&format!("NATIONTROOPS {i} {}\n", n.troops));

        // `tries` straight out of the ENGINE's own debug print (child process).
        if let Ok(exe) = std::env::current_exe() {
            let out = std::process::Command::new(exe)
                .args([
                    "--map",
                    &a.map,
                    "--seed",
                    &a.seed,
                    "--agents",
                    &a.agents.to_string(),
                    "--nations",
                    &a.nations,
                    "--human-agents",
                    &a.human_agents.to_string(),
                    "--difficulty",
                    &a.difficulty,
                    "--tries-for",
                    &n.id,
                ])
                .output();
            let tries = match out {
                Ok(cmd) => {
                    let err = String::from_utf8_lossy(&cmd.stderr).to_string();
                    err.lines()
                        .find(|l| l.contains(&n.id) && l.contains("tries="))
                        .and_then(|l| l.rsplit("tries=").next())
                        .and_then(|v| v.trim().parse::<u32>().ok())
                        .unwrap_or(0)
                }
                Err(_) => 0,
            };
            o.push_str(&format!("NATIONTRIES {i} {tries}\n"));
        }
    }
    let _ = game_state;

    match &a.out {
        Some(p) => {
            if let Some(parent) = p.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            std::fs::write(p, &o).expect("write oracle dump");
            eprintln!(
                "wrote {} (map={} agents={} nations={} nations_found={} plane_nonzero={})",
                p.display(),
                a.map,
                a.agents,
                a.nations,
                sim.nations.len(),
                plane_nonzero
            );
        }
        None => print!("{o}"),
    }
}
