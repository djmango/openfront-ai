//! Emit a frozen CLUSTER-CAPTURE case from the CURRENT in-tree engine.
//!
//! Usage:
//!   cargo run --release -- \
//!     --repo /opt/data/workspaces/skg/openfront-ai \
//!     --record .../curr-b002-s1-pangaea.json.gz \
//!     --out  ../../cases/cluster-b002-t<T>.json
//!
//! What it does, in one deterministic replay of the record:
//!   * keeps the owner plane at each tick boundary (`game.ticks() == T` is the
//!     state entering exec-tick T, since `execute_next_tick` passes the
//!     pre-increment `ticks` to every execution and increments at the end);
//!   * after each `execute_next_tick`, diffs the plane and looks for the
//!     signature of a cluster capture: one player loses a set of tiles that is
//!     4-connected in the *pre-tick* plane and a single other player receives
//!     all of them;
//!   * on the first such candidate, replays to that tick and calls the
//!     engine's own `player_clusters::maybe_remove_clusters` on the frozen
//!     state, so the recorded outcome is the engine's own slice result, not a
//!     full-tick diff contaminated by attacks;
//!   * writes the case (input state + engine outcome delta) as JSON.
//!
//! The dump is the input to the `ofcuda_cluster` crate; nothing here is
//! hand-synthesised. `--print-candidates` dumps the diff-only signature at
//! every tick without replaying.

use openfront_engine::bootstrap::game_from_record;
use openfront_engine::execution::intent::turn_to_executions;
use openfront_engine::execution::player_clusters::maybe_remove_clusters;
use openfront_engine::game::Game;
use openfront_engine::record::GameRecord;
use serde_json::json;
use std::collections::HashMap;
use std::path::{Path, PathBuf};


fn load_record(path: &Path) -> GameRecord {
    let raw = std::fs::read(path).expect("read record");
    let bytes = if path.extension().and_then(|s| s.to_str()) == Some("gz") {
        let mut dec = flate2::read::GzDecoder::new(&raw[..]);
        let mut out = Vec::new();
        std::io::Read::read_to_end(&mut dec, &mut out).expect("gunzip");
        out
    } else {
        raw
    };
    GameRecord::from_json_bytes(&bytes)
        .expect("parse record")
        .decompress()
}

fn plane(game: &Game) -> Vec<u16> {
    let n = (game.map.width * game.map.height) as usize;
    (0..n).map(|t| game.map.owner_id(t as u32)).collect()
}

/// 4-neighbour flood over `set` (N,S,W,E), returns true if all of `set` is one
/// component.
fn is_connected4(set: &[u32], w: u32, h: u32) -> bool {
    use std::collections::HashSet;
    if set.len() <= 1 {
        return true;
    }
    let hs: HashSet<u32> = set.iter().copied().collect();
    let mut seen: HashSet<u32> = HashSet::new();
    let mut stack = vec![set[0]];
    seen.insert(set[0]);
    while let Some(t) = stack.pop() {
        let x = t % w;
        let y = t / w;
        let mut nb = [u32::MAX; 4];
        let mut n = 0;
        if y > 0 {
            nb[n] = t - w;
            n += 1;
        }
        if y + 1 < h {
            nb[n] = t + w;
            n += 1;
        }
        if x > 0 {
            nb[n] = t - 1;
            n += 1;
        }
        if x + 1 < w {
            nb[n] = t + 1;
            n += 1;
        }
        for i in 0..n {
            if hs.contains(&nb[i]) && seen.insert(nb[i]) {
                stack.push(nb[i]);
            }
        }
    }
    seen.len() == set.len()
}

/// Diff two owner planes; returns (old_owner, new_owner, tiles).
fn diffs(a: &[u16], b: &[u16]) -> Vec<(u16, u16, Vec<u32>)> {
    let mut m: HashMap<(u16, u16), Vec<u32>> = HashMap::new();
    for i in 0..a.len() {
        if a[i] != b[i] {
            m.entry((a[i], b[i])).or_default().push(i as u32);
        }
    }
    let mut v: Vec<(u16, u16, Vec<u32>)> = m.into_iter().map(|((o, n), t)| (o, n, t)).collect();
    v.sort_by_key(|(o, n, _)| (*o, *n));
    v
}

fn base64_u16_le(v: &[u16]) -> String {
    use base64::Engine;
    let mut bytes = Vec::with_capacity(v.len() * 2);
    for w in v {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn fnv1a_u16(v: &[u16]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for w in v {
        for b in w.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

#[derive(Clone)]
struct Snap {
    plane: Vec<u16>,
}

fn main() {
    let mut repo = PathBuf::from("/opt/data/workspaces/skg/openfront-ai");
    let mut record_path = PathBuf::from(
        "/opt/data/workspaces/skg/openfront-ai/records/early-curriculum-parity/curr-b002-s1-pangaea.json.gz",
    );
    let mut out = PathBuf::from("case.json");
    let mut print_candidates = false;
    let mut force_tick: Option<u32> = None;
    let mut force_player: Option<u16> = None;
    let mut min_lost: usize = 8;
    let mut require_clean = false;

    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--repo" => repo = PathBuf::from(args.next().unwrap()),
            "--record" => record_path = PathBuf::from(args.next().unwrap()),
            "--out" => out = PathBuf::from(args.next().unwrap()),
            "--print-candidates" => print_candidates = true,
            "--tick" => force_tick = Some(args.next().unwrap().parse().unwrap()),
            "--player" => force_player = Some(args.next().unwrap().parse().unwrap()),
            "--min-lost" => min_lost = args.next().unwrap().parse().unwrap(),
            "--require-clean" => require_clean = true,
            other => panic!("unknown arg {other}"),
        }
    }

    let record = load_record(&record_path);
    let build = |repo: &Path, record: &GameRecord| -> Game {
        game_from_record(repo, record).expect("bootstrap")
    };

    // ---- Pass 1: scan for the removal signature -------------------------
    let mut candidates: Vec<(u32, u16, usize, u16)> = Vec::new(); // (exec_tick, victim, lost, gainer)
    {
        let mut game = build(&repo, &record);
        let (w, h) = (game.map.width, game.map.height);
        let mut prev = Snap { plane: plane(&game) };
        let mut last_tick = 0u32;
        let mut groups = 0usize;
        let mut best = (0usize, 0u32, 0u16, 0u16);
        for turn in &record.turns {
            let gid = game.game_id.clone();
            for e in turn_to_executions(&mut game, &gid, &turn.intents) {
                game.add_execution(e);
            }
            let exec_tick = game.ticks(); // value the executions will receive
            game.execute_next_tick();
            let cur = Snap { plane: plane(&game) };
            if exec_tick % 500 == 0 {
                let owned = cur.plane.iter().filter(|&&o| o != 0).count();
                eprintln!(
                    "[scan] tick {} owned={} players={} spawn_phase={}",
                    exec_tick,
                    owned,
                    game.all_players().len(),
                    game.in_spawn_phase()
                );
            }
            for (old, new, tiles) in diffs(&prev.plane, &cur.plane) {
                if old == 0 || new == 0 {
                    continue;
                }
                groups += 1;
                if tiles.len() > best.0 {
                    best = (tiles.len(), exec_tick, old, new);
                }
                if is_connected4(&tiles, w, h) {
                    candidates.push((exec_tick, old, tiles.len(), new));
                }
            }
            prev = cur;
            last_tick = exec_tick;
        }
        eprintln!("[scan] replayed to tick {last_tick}");
        eprintln!("[scan] nonempty (old,new) owner-change groups: {groups}");
        eprintln!("[scan] largest group: {} tiles at tick {} ({} -> {})", best.0, best.1, best.2, best.3);
    }
    eprintln!("[scan] {} candidate ticks with a connected single-gainer loss", candidates.len());
    for (t, v, n, g) in candidates.iter().take(40) {
        println!("cand exec_tick={t} victim={v} lost={n} gainer={g}");
    }
    if print_candidates {
        return;
    }
    if candidates.is_empty() {
        panic!("no cluster-capture candidate found in this record");
    }

    // ---- Pass 2: replay to the candidate tick, snapshot, run engine slice -
    let mut ordered: Vec<(u32, u16, usize, u16)> =
        candidates.iter().copied().filter(|c| c.2 >= min_lost).collect();
    ordered.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)));
    let mut chosen = None;
    for (exec_tick, victim, _lost, _gainer) in &ordered {
        let exec_tick = *exec_tick;
        let victim = *victim;
        if let Some(ft) = force_tick {
            if exec_tick != ft {
                continue;
            }
        }
        if let Some(fp) = force_player {
            if victim != fp {
                continue;
            }
        }
        let (delta, has) = replay_and_call(&build, &repo, &record, exec_tick, victim);
        if !has {
            continue;
        }
        let full = tick_diff_group(&build, &repo, &record, exec_tick, victim);
        let mut fs = full.clone();
        fs.sort();
        let mut ss = delta.clone();
        ss.sort();
        let clean = fs == ss;
        eprintln!(
            "[pick] exec_tick={exec_tick} victim={victim}: engine-slice={} real-tick={} clean={clean}",
            ss.len(),
            fs.len()
        );
        if require_clean && !clean {
            continue;
        }
        chosen = Some((exec_tick, victim, delta));
        break;
    }
    let (exec_tick, victim, delta) =
        chosen.expect("no candidate reproduced a removal under the engine's own maybe_remove_clusters");

    integration_check(&build, &repo, &record, exec_tick, victim, &delta);

    // ---- Pass 3: capture the frozen input state + engine slice outcome ---
    let (case, outcome_plane) =
        capture(&build, &repo, &record, &record_path, exec_tick, victim, &delta);
    std::fs::write(&out, serde_json::to_string(&case).unwrap()).expect("write case");
    eprintln!(
        "[dump] case: exec_tick={exec_tick} victim={victim} engine delta tiles={} plane_fnv=0x{:016x}",
        delta.len(),
        fnv1a_u16(&outcome_plane)
    );
    eprintln!("[dump] wrote {}", out.display());
}

/// Tiles that the REAL tick at `exec_tick` moves off `victim`, with their new owners.
fn tick_diff_group(
    build: &dyn Fn(&Path, &GameRecord) -> Game,
    repo: &Path,
    record: &GameRecord,
    exec_tick: u32,
    victim: u16,
) -> Vec<(u32, u16)> {
    let mut game = replay_to(build, repo, record, exec_tick);
    let before = plane(&game);
    let gid = game.game_id.clone();
    if let Some(turn) = record.turns.iter().find(|t| t.turn_number == exec_tick) {
        for e in turn_to_executions(&mut game, &gid, &turn.intents) {
            game.add_execution(e);
        }
    }
    game.execute_next_tick();
    let after = plane(&game);
    let mut full: Vec<(u32, u16)> = Vec::new();
    for i in 0..before.len() {
        if before[i] == victim && after[i] != victim {
            full.push((i as u32, after[i]));
        }
    }
    full.sort();
    full
}

/// Full-tick integration cross-check: run the REAL tick at `exec_tick` and
/// compare the (victim -> gainer) owner-change group against the slice delta
/// the engine's own `maybe_remove_clusters` produced on the frozen state.
fn integration_check(
    build: &dyn Fn(&Path, &GameRecord) -> Game,
    repo: &Path,
    record: &GameRecord,
    exec_tick: u32,
    victim: u16,
    slice_delta: &[(u32, u16)],
) {
    let full = tick_diff_group(build, repo, record, exec_tick, victim);
    let mut slice = slice_delta.to_vec();
    slice.sort();
    println!(
        "[integration] tick {exec_tick}: real-tick tiles lost by {victim} = {}, engine-slice delta = {}, tile-for-tile identical = {}",
        full.len(),
        slice.len(),
        full == slice
    );
    if full != slice {
        let fs: std::collections::HashSet<(u32, u16)> = full.iter().copied().collect();
        let ss: std::collections::HashSet<(u32, u16)> = slice.iter().copied().collect();
        let mut only_full: Vec<_> = fs.difference(&ss).copied().collect();
        let mut only_slice: Vec<_> = ss.difference(&fs).copied().collect();
        only_full.sort();
        only_slice.sort();
        println!("  only in real tick: {:?}", &only_full[..only_full.len().min(8)]);
        println!("  only in slice    : {:?}", &only_slice[..only_slice.len().min(8)]);
    }
}

/// Replay to `exec_tick`, call the engine's slice fn for `victim`, report its delta.
fn replay_and_call(
    build: &dyn Fn(&Path, &GameRecord) -> Game,
    repo: &Path,
    record: &GameRecord,
    exec_tick: u32,
    victim: u16,
) -> (Vec<(u32, u16)>, bool) {
    let mut game = replay_to(build, repo, record, exec_tick);
    let before = plane(&game);
    maybe_remove_clusters(&mut game, victim, exec_tick);
    let after = plane(&game);
    let mut d: Vec<(u32, u16)> = Vec::new();
    for i in 0..before.len() {
        if before[i] != after[i] {
            d.push((i as u32, after[i]));
        }
    }
    d.sort();
    let has = !d.is_empty();
    (d, has)
}

fn replay_to(
    build: &dyn Fn(&Path, &GameRecord) -> Game,
    repo: &Path,
    record: &GameRecord,
    exec_tick: u32,
) -> Game {
    let mut game = build(repo, record);
    for turn in &record.turns {
        if game.ticks() >= exec_tick {
            break;
        }
        let gid = game.game_id.clone();
        for e in turn_to_executions(&mut game, &gid, &turn.intents) {
            game.add_execution(e);
        }
        game.execute_next_tick();
    }
    assert_eq!(game.ticks(), exec_tick, "replay landed on the wrong tick");
    game
}

#[allow(clippy::type_complexity)]
fn capture(
    build: &dyn Fn(&Path, &GameRecord) -> Game,
    repo: &Path,
    record: &GameRecord,
    record_path: &Path,
    exec_tick: u32,
    victim: u16,
    engine_delta: &[(u32, u16)],
) -> (serde_json::Value, Vec<u16>) {
    let mut game = replay_to(build, repo, record, exec_tick);
    let (w, h) = (game.map.width, game.map.height);
    let input_plane = plane(&game);

    // players
    let players: Vec<serde_json::Value> = game
        .all_players()
        .iter()
        .map(|p| {
            json!({
                "small_id": p.small_id,
                "id": p.id,
                "id_hash": p.id_hash,
                "player_type": format!("{:?}", p.player_type),
                "team": p.team,
                "tiles_owned": p.tiles_owned,
                "alive": p.alive,
                "last_cluster_calc": p.last_cluster_calc,
                "last_tile_change": p.last_tile_change,
                "border": p.border_tiles.iter().collect::<Vec<u32>>(),
                "owned_count": p.owned_tiles.len(),
            })
        })
        .collect();

    // directed friendliness pairs among players
    let mut friends: Vec<(u16, u16)> = Vec::new();
    let sids: Vec<u16> = game.all_players().iter().map(|p| p.small_id).collect();
    for &a in &sids {
        for &b in &sids {
            if a == b {
                continue;
            }
            if game.is_friendly(a, b) {
                friends.push((a, b));
            }
        }
    }

    // live / active attacks: execs order is shared, so the k-th attack_live
    // entry of active_attacks_debug corresponds to the k-th live_attacks() item.
    let dbg = game.active_attacks_debug();
    let live: Vec<(&openfront_engine::execution::AttackExecution, u16, u16)> = game
        .live_attacks()
        .map(|a| (a, a.owner_small_id(), a.target_small_id()))
        .collect();
    let mut li = 0usize;
    let attacks: Vec<serde_json::Value> = dbg
        .iter()
        .map(|(o, t, troops, active, attack_live, _, _)| {
            let mut initialized = false;
            if *attack_live && li < live.len() {
                initialized = live[li].0.is_initialized();
                li += 1;
            }
            json!({
                "owner": o, "target": t,
                "troops_bits": troops.to_bits().to_string(),
                "active": active, "attack_live": attack_live,
                "initialized": initialized,
            })
        })
        .collect();

    // terrain hash of the map plane the case depends on
    let terrain: Vec<u8> = (0..(w * h) as usize)
        .map(|t| game.map.terrain_byte(t as u32))
        .collect();
    let mut th: u64 = 0xcbf2_9ce4_8422_2325;
    for b in &terrain {
        th ^= *b as u64;
        th = th.wrapping_mul(0x0000_0100_0000_01b3);
    }

    // ---- the engine's own slice call on this frozen state ----------------
    let before_counts: Vec<(u16, i32)> = game
        .all_players()
        .iter()
        .map(|p| (p.small_id, p.tiles_owned))
        .collect();
    maybe_remove_clusters(&mut game, victim, exec_tick);
    let mut delta: Vec<(u32, u16)> = Vec::new();
    {
        let after = plane(&game);
        for i in 0..input_plane.len() {
            if input_plane[i] != after[i] {
                delta.push((i as u32, after[i]));
            }
        }
        delta.sort();
    }
    assert_eq!(delta, {
        let mut e = engine_delta.to_vec();
        e.sort();
        e
    }, "engine slice delta differs from the pre-computed one");
    let changed_players: Vec<serde_json::Value> = game
        .all_players()
        .iter()
        .filter(|p| {
            before_counts
                .iter()
                .find(|(s, _)| *s == p.small_id)
                .map(|(_, c)| *c != p.tiles_owned)
                .unwrap_or(false)
        })
        .map(|p| {
            json!({
                "small_id": p.small_id,
                "tiles_owned": p.tiles_owned,
                "owned_order": p.owned_tiles.iter().copied().collect::<Vec<u32>>(),
            })
        })
        .collect();

    let case = json!({
        "meta": {
            "record": record_path.display().to_string(),
            "engine_commit": engine_commit(),
            "exec_tick": exec_tick,
            "victim_small_id": victim,
            "width": w, "height": h,
            "map_dir": "openfront/resources/maps/pangaea",
            "terrain_fnv": format!("0x{th:016x}"),
            "note": "input state is the engine's own state at game.ticks()==exec_tick; outcome is the engine's own player_clusters::maybe_remove_clusters on that state",
        },
        "input_plane_b64": base64_u16_le(&input_plane),
        "input_plane_fnv": format!("0x{:016x}", fnv1a_u16(&input_plane)),
        "players": players,
        "friends": friends.iter().map(|(a,b)| json!([a,b])).collect::<Vec<_>>(),
        "attacks": attacks,
        "engine_outcome_delta": delta.iter().map(|(t,o)| json!([t,o])).collect::<Vec<_>>(),
        "engine_outcome_changed_players": changed_players,
    });
    (case, input_plane)
}

fn engine_commit() -> String {
    std::process::Command::new("git")
        .args(["-C", "/opt/data/workspaces/skg/openfront-ai", "rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default()
        .trim()
        .to_string()
}
