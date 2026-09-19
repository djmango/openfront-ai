//! `oracle` - the REFERENCE ENGINE replay that grounds every `ofcuda_matrix`
//! cell.
//!
//! Ground truth for this matrix must never come from the CUDA port. This binary
//! links the real engine (`openfront-engine`) and drives it exactly as an RL
//! episode is driven (`RlSession::reset` + `Game::execute_next_tick`), then dumps
//! per-boundary:
//!
//!   * the engine's own owner plane hash (FNV-1a-64 over the `w*h` `u16` plane
//!     read through `GameMap::owner_id`), the per-player `tiles_owned`, troops
//!     and gold;
//!   * the tiles each player claimed during the tick that just ran, IN THE
//!     ENGINE'S OWN ORDER (`Player::owned_tiles` push order - the same claim
//!     order the Pangaea proof compares against);
//!   * the live land attacks (`Game::live_attacks`) with their exact `f64`
//!     `troops()` - the value a record can only carry truncated to `i64`;
//!   * the initial owned set per player at boundary 0 (the engine's spawn
//!     output).
//!
//! # Spawn phase
//!
//! `AttackExecution::active_during_spawn()` and `PlayerExecution::
//! active_during_spawn()` are both `false`, so a session that never leaves the
//! spawn phase produces a frozen game: nothing but the spawn executions tick.
//! A real RL episode leaves the phase when the human agent submits a spawn
//! intent; the port's init (the `spawnall --dump` state) contains BOT tiles
//! only, so the matching construction here is to let every bot spawn and then
//! call `Game::end_spawn_phase()` - the same call the engine's own tests make
//! (`game.rs:3992` etc.). The bot spawn tiles this produces are the ones
//! `ofcuda_spawn` dumps and `spawnall` verifies, so the init compared here and
//! the init the CUDA side uploads are the same object.
//!
//! # Modes
//!
//! ```text
//! oracle replay  --maps pangaea,africa --ns 2,7,18 --nations 0 --ticks 120 --out-dir DIR
//! oracle ceiling --maps pangaea,onion   --nations 0 --out FILE
//! ```
//!
//! `replay` writes one dump per (map, N). `ceiling` reports, per map, the
//! largest N at which the engine places every bot (the spawn ceiling).

use openfront_engine::game::{Game, PlayerType};
use openfront_engine::map::TileRef;
use openfront_engine::rl::RlSession;
use openfront_engine::util::simple_hash;
use serde_json::json;
use std::path::{Path, PathBuf};

const REPO: &str = "/opt/data/workspaces/skg/openfront-ai";
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a_u16_le(data: &[u16]) -> u64 {
    let mut h = FNV_OFFSET;
    for &w in data {
        for b in w.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(FNV_PRIME);
        }
    }
    h
}

struct Args {
    mode: String,
    maps: Vec<String>,
    ns: Vec<u32>,
    nations: String,
    ticks: u32,
    seed: String,
    out_dir: PathBuf,
    out: Option<PathBuf>,
    max_n: u32,
}

fn parse() -> Result<Args, String> {
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut a = Args {
        mode: String::new(),
        maps: vec!["pangaea".into()],
        ns: vec![18],
        nations: "0".into(),
        ticks: 120,
        seed: "parity".into(),
        out_dir: PathBuf::from("/opt/data/workspaces/skg/ofcuda_matrix/out"),
        out: None,
        max_n: 4096,
    };
    let mut i = 0;
    if v.is_empty() {
        return Err("usage: oracle {replay|ceiling} [...]".into());
    }
    a.mode = v[0].clone();
    i = 1;
    while i < v.len() {
        let val = |i: usize| v.get(i + 1).cloned().ok_or_else(|| format!("{} needs a value", v[i]));
        match v[i].as_str() {
            "--maps" => {
                a.maps = val(i)?.split(',').map(|s| s.trim().to_string()).collect();
                i += 2;
            }
            "--ns" => {
                a.ns = val(i)?
                    .split(',')
                    .map(|s| s.trim().parse::<u32>().map_err(|e| e.to_string()))
                    .collect::<Result<Vec<_>, _>>()?;
                i += 2;
            }
            "--nations" => {
                a.nations = val(i)?;
                i += 2;
            }
            "--ticks" => {
                a.ticks = val(i)?.parse().map_err(|e| format!("ticks: {e}"))?;
                i += 2;
            }
            "--seed" => {
                a.seed = val(i)?;
                i += 2;
            }
            "--out-dir" => {
                a.out_dir = PathBuf::from(val(i)?);
                i += 2;
            }
            "--out" => {
                a.out = Some(PathBuf::from(val(i)?));
                i += 2;
            }
            "--max-n" => {
                a.max_n = val(i)?.parse().map_err(|e| format!("max-n: {e}"))?;
                i += 2;
            }
            other => return Err(format!("unknown arg {other}")),
        }
    }
    Ok(a)
}

fn nations_value(spec: &str) -> Result<serde_json::Value, String> {
    match spec {
        "disabled" | "default" => Ok(serde_json::Value::String(spec.to_string())),
        n => n
            .parse::<u32>()
            .map(|v| json!(v))
            .map_err(|_| format!("--nations must be disabled|default|int, got {spec:?}")),
    }
}

fn type_tag(t: PlayerType) -> &'static str {
    match t {
        PlayerType::Human => "H",
        PlayerType::Bot => "B",
        PlayerType::Nation => "N",
    }
}

fn reset_once(
    map: &str,
    seed: &str,
    bots: u32,
    nations: &serde_json::Value,
) -> Result<RlSession, String> {
    let (session, _, _, _, _, _) = RlSession::reset(
        Path::new(REPO),
        map,
        seed,
        bots,
        "Easy",
        nations.clone(),
        1,
    )?;
    Ok(session)
}

/// Tick until the bot spawn set stops growing (every spawn lands on tick 1 in
/// practice: `ofcuda_spawn` records `spawn_tick = 1` for all of them).
fn run_spawn_phase(session: &mut RlSession, max_ticks: u32) -> (u32, usize) {
    let mut last = 0usize;
    let mut stable = 0u32;
    let mut tick = 0u32;
    while tick < max_ticks {
        let before = spawned_bots(&session.game);
        session.game.execute_next_tick();
        tick += 1;
        let now = spawned_bots(&session.game);
        if now == before {
            stable += 1;
            if stable >= 2 {
                break;
            }
        } else {
            stable = 0;
            last = now;
        }
    }
    (tick, last)
}

fn spawned_bots(game: &Game) -> usize {
    game.all_players()
        .iter()
        .filter(|p| p.player_type == PlayerType::Bot && p.spawn_tile.is_some())
        .count()
}

/// The engine's authoritative owner plane: one `GameMap::owner_id` read per
/// tile, owner `small_id` in the word, 0 = unowned.
fn engine_plane(game: &Game) -> Vec<u16> {
    let n = (game.width() as usize) * (game.height() as usize);
    let mut plane = vec![0u16; n];
    for t in 0..n {
        plane[t] = game.map.owner_id(t as u32);
    }
    plane
}

fn replay(args: &Args) -> Result<(), String> {
    let nations = nations_value(&args.nations)?;
    std::fs::create_dir_all(&args.out_dir).map_err(|e| e.to_string())?;
    for map in &args.maps {
        for &n in &args.ns {
            let t0 = std::time::Instant::now();
            let dump = replay_cell(map, &args.seed, n, &nations, args.ticks)?;
            let name = format!(
                "{}_n{}_nat{}_t{}.dump",
                map.to_lowercase(),
                n,
                args.nations,
                args.ticks
            );
            let path = args.out_dir.join(&name);
            std::fs::write(&path, dump).map_err(|e| e.to_string())?;
            eprintln!(
                "wrote {} ({} bytes, {:.1}s)",
                path.display(),
                std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
                t0.elapsed().as_secs_f64()
            );
        }
    }
    Ok(())
}

fn replay_cell(
    map: &str,
    seed: &str,
    bots: u32,
    nations: &serde_json::Value,
    ticks: u32,
) -> Result<String, String> {
    let mut session = reset_once(map, seed, bots, nations)?;
    let (spawn_ticks, _last) = run_spawn_phase(&mut session, 12);
    session.game.end_spawn_phase();
    let spawn_end_tick = session.game.spawn_end_tick().unwrap_or(0);
    let game_id = session.game_id().to_string();
    let game_hash = simple_hash(&game_id);
    let (w, h) = (session.game.width(), session.game.height());

    let mut o = String::with_capacity(1 << 20);
    o.push_str("# ofcuda_matrix engine oracle v2\n");
    o.push_str(&format!("REPO {REPO}\n"));
    o.push_str(&format!("MAP {map}\n"));
    o.push_str(&format!("SEED {seed}\n"));
    o.push_str(&format!("GAME_ID {game_id}\n"));
    o.push_str(&format!("GAME_HASH {game_hash}\n"));
    o.push_str(&format!("AGENTS {bots}\n"));
    o.push_str(&format!("NATIONS {}\n", nations));
    o.push_str("HUMAN_AGENTS 1\nDIFFICULTY Easy\n");
    o.push_str(&format!("WIDTH {w}\nHEIGHT {h}\n"));
    o.push_str(&format!(
        "MIN_DIST {}\n",
        session.game.wire.min_distance_between_players()
    ));
    o.push_str(&format!("SPAWN_TICKS {spawn_ticks}\n"));
    o.push_str(&format!("SPAWN_END_TICK {spawn_end_tick}\n"));
    o.push_str(&format!("TICKS {ticks}\n"));
    for p in session.game.all_players() {
        o.push_str(&format!(
            "ROSTER {} {} {} {}\n",
            p.small_id,
            type_tag(p.player_type),
            p.id,
            p.spawn_tile.map(|t| t as i64).unwrap_or(-1)
        ));
    }

    // ---- boundary 0: the engine's spawn output -----------------------------
    let mut prev_owned: Vec<Vec<u32>> = Vec::new();
    {
        let game = &session.game;
        let plane = engine_plane(game);
        emit_boundary(&mut o, 0, game.ticks(), &plane);
        let players: Vec<_> = game.all_players().to_vec();
        prev_owned = players.iter().map(|p| p.owned_tiles.clone()).collect();
        for p in &players {
            o.push_str(&format!(
                "OWNED 0 {} {} {}\n",
                p.small_id,
                p.owned_tiles.len(),
                join_tiles(&p.owned_tiles)
            ));
        }
        selfcheck(&mut o, game, &players, &plane);
        emit_players(&mut o, 0, &players, &prev_owned);
        emit_borders(&mut o, 0, &players);
        emit_attacks(&mut o, 0, game);
    }

    // ---- optional single-attack trace -------------------------------------
    // `OF_TRACE_ATK=owner:target` (and optionally `OF_TRACE_TILE=<tile>`)
    // prints, per boundary, the engine's OWN live values for that attack and its
    // target player immediately BEFORE the tick and immediately AFTER it. This
    // is the only way to see what the engine's intra-tick executions actually
    // changed, since the per-boundary dump only exposes the post-tick record.
    let trace: Option<(u16, u16)> =
        std::env::var("OF_TRACE_ATK").ok().and_then(|s| {
            let mut it = s.split(':');
            Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
        });
    let trace_tile: Option<TileRef> = std::env::var("OF_TRACE_TILE")
        .ok()
        .and_then(|s| s.parse().ok());
    let mut trace_out = String::new();
    let mut snap = |game: &Game, b: u32, tag: &str| -> Option<()> {
        let (o_id, t_id) = trace?;
        let a = game
            .live_attacks()
            .find(|a| a.owner_small_id() == o_id && a.target_small_id() == t_id)?;
        let p = game.player_by_small_id(t_id);
        let (dtroops, dtiles) = p.map(|p| (p.troops, p.tiles_owned)).unwrap_or((-1, -1));
        let inc = if p.is_some() {
            game.troop_increase_rate_raw_for(t_id)
        } else {
            f64::NAN
        };
        let atiles = game
            .player_by_small_id(o_id)
            .map(|p| p.tiles_owned)
            .unwrap_or(-1);
        let mut line = format!(
            "TRACE {b} {tag} owner={o_id} target={t_id} atk={:.12} atk_bits={:#018x} atk_tiles={atiles} def_troops={dtroops} def_tiles={dtiles} def_inc_raw={inc:.12}\n",
            a.troops(),
            a.troops().to_bits(),
        );
        if tag == "pre" {
            if let Some(t) = trace_tile {
                let (mag, speed, attacker_loss) =
                    game.attack_logic_at_tile(a.troops(), o_id, t_id, t, true);
                line.push_str(&format!(
                    "TRACELOGIC {b} tile={t} mag={mag} speed={speed} attacker_loss={attacker_loss:.12} loss_bits={:#018x} def_troops={dtroops} def_tiles={dtiles} atk={:.12}\n",
                    attacker_loss.to_bits(),
                    a.troops(),
                ));
            }
        }
        trace_out.push_str(&line);
        Some(())
    };

    // ---- the post-spawn ticks ---------------------------------------------
    for b in 1..=ticks {
        snap(&session.game, b, "pre");
        session.game.execute_next_tick();
        let game = &session.game;
        snap(game, b, "post");
        let plane = engine_plane(game);
        emit_boundary(&mut o, b, game.ticks(), &plane);
        let players: Vec<_> = game.all_players().to_vec();
        let mut churn_bad = String::new();
        for (i, p) in players.iter().enumerate() {
            let prev: &[u32] = prev_owned.get(i).map(|v| v.as_slice()).unwrap_or(&[]);
            let cur = &p.owned_tiles;
            let n0 = prev.len().min(cur.len());
            let churn = (0..n0).filter(|&j| prev[j] != cur[j]).count();
            if churn > 0 {
                churn_bad.push_str(&format!(" {}:{}", p.small_id, churn));
            }
            if cur.len() > n0 {
                let claims = &cur[n0..];
                o.push_str(&format!(
                    "CLAIM {b} {} {} {}\n",
                    p.small_id,
                    claims.len(),
                    join_tiles(claims)
                ));
            }
        }
        if !churn_bad.is_empty() {
            // The `owned_tiles` vector was reordered/pruned, so the tail-diff is
            // NOT a claim set for these players this tick. Recorded, not hidden.
            o.push_str(&format!("CHURN {b}{churn_bad}\n"));
        }
        emit_players(&mut o, b, &players, &prev_owned);
        emit_borders(&mut o, b, &players);
        if b < ticks {
            emit_attacks(&mut o, b, game);
        }
        prev_owned = players.iter().map(|p| p.owned_tiles.clone()).collect();
    }
    o.push_str(&trace_out);
    Ok(o)
}

fn join_tiles(v: &[u32]) -> String {
    let mut s = String::with_capacity(v.len() * 6);
    for (i, t) in v.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        s.push_str(&t.to_string());
    }
    s
}

fn emit_boundary(o: &mut String, b: u32, engine_tick: u32, plane: &[u16]) {
    let owned = plane.iter().filter(|&&x| x != 0).count();
    o.push_str(&format!(
        "BOUNDARY {b} {engine_tick} {:#018x} {owned}\n",
        fnv1a_u16_le(plane)
    ));
}

fn emit_players(
    o: &mut String,
    b: u32,
    players: &[openfront_engine::game::Player],
    prev: &[Vec<u32>],
) {
    for (i, p) in players.iter().enumerate() {
        let n0 = prev.get(i).map(|v| v.len()).unwrap_or(0).min(p.owned_tiles.len());
        o.push_str(&format!(
            "PLAYER {b} {} {} {} {} {} {} {} {}\n",
            p.small_id,
            type_tag(p.player_type),
            p.id,
            p.tiles_owned,
            p.troops,
            p.gold,
            p.owned_tiles.len(),
            n0
        ));
    }
}

/// The owner's `border_tiles` at the boundary, in the engine's insertion order.
/// This is the input to `AttackExecution::refresh_to_conquer` at attack
/// creation and to a mid-tick refresh, so it is the one adjacency input the
/// CUDA driver cannot derive from the plane alone.
fn emit_borders(o: &mut String, b: u32, players: &[openfront_engine::game::Player]) {
    for p in players {
        o.push_str(&format!(
            "BORDER {b} {} {} {}\n",
            p.small_id,
            p.border_tiles.len(),
            join_tiles(p.border_tiles.as_slice())
        ));
    }
}

fn emit_attacks(o: &mut String, b: u32, game: &Game) {
    for a in game.live_attacks() {
        // The engine's `AttackExecution::init` seeds the frontier from
        // `source_tile` when it has one (attack.rs:160-164) and falls back to
        // `refresh_to_conquer` over the owner's border set only when it does not
        // (attack.rs:1265). The port's only attack-creation entry point used to be
        // border-set based, so `source_tile` was the one piece of state the record
        // did not carry and the composition could not express. It is emitted here
        // (tile, or -1 for none) along with the heap and border sizes, so the port
        // can be *checked* against the engine's post-init frontier instead of
        // argued about.
        o.push_str(&format!(
            "ATTACK {b} {} {} {:#018x} {} {} {} {}\n",
            a.owner_small_id(),
            a.target_small_id(),
            a.troops().to_bits(),
            a.source_tile().map(|t| t as i64).unwrap_or(-1),
            a.to_conquer_len(),
            a.border_tile_count(),
            a.attack_id(),
        ));
    }
}

/// Cross-check that the plane rebuilt from `owned_tiles` is the plane the map
/// itself reports. Any nonzero value means the two engine representations
/// disagree and the claim-diff model below is not safe for that cell.
fn selfcheck(
    o: &mut String,
    game: &Game,
    players: &[openfront_engine::game::Player],
    plane: &[u16],
) {
    let mut rebuilt = vec![0u16; plane.len()];
    for p in players {
        for &t in &p.owned_tiles {
            if let Some(slot) = rebuilt.get_mut(t as usize) {
                *slot = p.small_id;
            }
        }
    }
    let diff = rebuilt.iter().zip(plane).filter(|(a, b)| a != b).count();
    let sum_tiles: i64 = players.iter().map(|p| p.tiles_owned as i64).sum();
    let plane_owned = plane.iter().filter(|&&x| x != 0).count() as i64;
    let _ = game;
    o.push_str(&format!("SELFCHECK {diff} {sum_tiles} {plane_owned}\n"));
}

/// The largest N at which the engine places every bot, for one map.
fn ceiling_for(map: &str, seed: &str, nations: &serde_json::Value, max_n: u32) -> Result<u32, String> {
    // Walk up in coarse steps, then bisect inside the last failing bracket. The
    // walk ALWAYS probes `max_n` itself: stepping past it would report the last
    // coarse grid point as a ceiling (449 = 1+4..+64 grid, never 488).
    let mut lo = 0u32; // known all-spawn (0 = trivially)
    let mut hi: Option<u32> = None;
    let mut n = 1u32;
    while n <= max_n {
        let spawned = probe(map, seed, n, nations)?;
        if spawned != n {
            hi = Some(n);
            break;
        }
        lo = n;
        if n == max_n {
            break;
        }
        // Strictly increasing, never overshooting `max_n`. Above 512 the walk
        // DOUBLES: a +1 tail from 512 up to max_n would be thousands of probes
        // at the most expensive N values. The bisect below restores resolution.
        if n >= 512 {
            n = n.saturating_mul(2);
        } else if n >= 64 {
            n += 64;
        } else if n >= 16 {
            n += 16;
        } else {
            n += 4;
        }
        if n > max_n {
            n = max_n;
        }
    }
    let Some(mut bad) = hi else {
        return Ok(lo); // all-spawn all the way to max_n (which was probed)
    };
    let mut good = lo;
    while bad - good > 1 {
        let mid = good + (bad - good) / 2;
        let spawned = probe(map, seed, mid, nations)?;
        if spawned == mid {
            good = mid;
        } else {
            bad = mid;
        }
    }
    Ok(good)
}

fn probe(map: &str, seed: &str, n: u32, nations: &serde_json::Value) -> Result<u32, String> {
    let mut session = reset_once(map, seed, n, nations)?;
    run_spawn_phase(&mut session, 12);
    Ok(spawned_bots(&session.game) as u32)
}

fn ceiling(args: &Args) -> Result<(), String> {
    let nations = nations_value(&args.nations)?;
    let mut out = String::new();
    out.push_str("# ofcuda_matrix spawn ceiling (largest N where EVERY bot places)\n");
    out.push_str(&format!("# seed {} difficulty Easy human_agents 1\n", args.seed));
    out.push_str("map\tceiling\n");
    for map in &args.maps {
        let c = ceiling_for(map, &args.seed, &nations, args.max_n)?;
        eprintln!("{map}: ceiling {c}");
        out.push_str(&format!("{map}\t{c}\n"));
    }
    write_out(args, out)
}

/// The PLATEAU: how many bots the engine actually places once N grows past the
/// ceiling. Same `probe` as the ceiling, so both numbers come from one code path
/// and cannot disagree about what "placed" means.
fn plateau(args: &Args) -> Result<(), String> {
    let nations = nations_value(&args.nations)?;
    let mut out = String::new();
    out.push_str("# ofcuda_matrix spawn plateau (bots actually placed, engine)\n");
    out.push_str(&format!(
        "# seed {} difficulty Easy human_agents 1\n",
        args.seed
    ));
    out.push_str("map\trequested\tplaced\n");
    for map in &args.maps {
        for &n in &args.ns {
            let placed = probe(map, &args.seed, n, &nations)?;
            eprintln!("{map}: N={n} placed {placed}");
            out.push_str(&format!("{map}\t{n}\t{placed}\n"));
        }
    }
    write_out(args, out)
}

fn write_out(args: &Args, out: String) -> Result<(), String> {
    match &args.out {
        Some(p) => {
            std::fs::write(p, out).map_err(|e| e.to_string())?;
            eprintln!("wrote {}", p.display());
        }
        None => print!("{out}"),
    }
    Ok(())
}

fn main() {
    let args = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    let r = match args.mode.as_str() {
        "replay" => replay(&args),
        "ceiling" => ceiling(&args),
        "plateau" => plateau(&args),
        other => Err(format!("unknown mode {other} (replay|ceiling|plateau)")),
    };
    if let Err(e) = r {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
