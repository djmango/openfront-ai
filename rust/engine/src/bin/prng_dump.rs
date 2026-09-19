//! PRNG + spawn-tile reference dump for the CUDA parity port.
//!
//! Emits the exact values a CUDA reimplementation of `PseudoRandom`
//! (`rust/engine/src/prng.rs`) and of spawn-tile selection
//! (`rust/engine/src/execution/spawn_util.rs`) must reproduce bit-for-bit:
//!
//!   * `stream`  - the first 24 `next()` values of `PseudoRandom::new(seed)`
//!                 printed as **raw u32 bit patterns** (the f64 `next()`
//!                 return is `(t as u32) as f64 / 2^32`, so the u32 is
//!                 recovered exactly: `(next() * 2^32) as u64 as u32`).
//!   * `draws`   - the first 16 `next_int(min,max)` for four ranges, each
//!                 taken from a *fresh* PRNG on the same seed.
//!   * `sem`     - `chance` / `rand_element` (incl. the empty-list no-draw
//!                 rule) / `shuffle_array` (len-1 draws) semantics, plus the
//!                 three following raw u32 draws so a port that consumes the
//!                 wrong number of draws is caught.
//!   * `pstream` - the same 24-value stream seeded by `simple_hash(player_id)`
//!                 (the per-player seed shape the engine actually uses).
//!   * `spawn`   - the real spawn tile the engine picks for each of the two
//!                 stage-0 bots (stage 0 of the V10 ladder is `(2, 0)` -
//!                 two bots, zero nations, `ofcore/src/curriculum.rs:835-838`),
//!                 the engine's own seed for that spawn, and the owner plane
//!                 snapshot (`owner` lines) + already-placed spawn centres
//!                 (`prev` lines) the selection ran against.
//!   * `human`   - the scripted human spawn: an explicit tile, which must
//!                 consume **zero** draws.
//!
//! Usage (from `rust/`):
//!   cargo run --release -p openfront-engine --bin prng_dump -- \
//!       --out /opt/data/workspaces/skg/ofcuda_prng/reference.txt
//!
//! Every engine behaviour quoted above is cited by file:line in the comments.

use openfront_engine::game::{Game, PlayerType};
use openfront_engine::prng::PseudoRandom;
use openfront_engine::rl::RlSession;
use openfront_engine::util::simple_hash;
use serde_json::json;
use std::fmt::Write as _;
use std::path::PathBuf;

/// The four fixed seed strings (RL seeds - `RlSession::reset`'s `seed`).
const SEEDS: [&str; 4] = ["parity", "alpha", "bravo", "charlie"];
/// Stage 0 of the V10 ladder samples from `V10_BRIDGE_MAPS`
/// (`ofcore/src/curriculum.rs:737-746`); Pangaea is the first entry.
const MAP: &str = "Pangaea";
/// `V10_BOT_NATION_DENSITY[0] == (2, 0)` (`ofcore/src/curriculum.rs:837`).
const BOTS: u32 = 2;

/// `(next() * 2^32) as u64 as u32` - exact, because `next()` is
/// `(t as u32) as f64 / 4294967296.0` (`prng.rs:44`) and dividing by a power
/// of two is exact in f64.
fn next_u32(pr: &mut PseudoRandom) -> u32 {
    (pr.next() * 4_294_967_296.0) as u64 as u32
}

/// Re-exported from the engine so the reference dump can never drift from
/// the production id derivation (`rust/engine/src/session.rs`).
use openfront_engine::session::seed_to_game_id;

fn hex_u32s(v: &[u32]) -> String {
    v.iter().map(|x| format!("{x:08x}")).collect::<Vec<_>>().join(" ")
}

fn main() {
    let repo = std::env::var("OPENFRONT_REPO")
        .unwrap_or_else(|_| openfront_engine::util::default_repo_root());
    let repo = PathBuf::from(repo);
    let out_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("prng_dump.txt"));

    let mut o = String::new();
    let mut w = |s: String| {
        o.push_str(&s);
        o.push('\n');
    };

    w("# ofcuda_prng reference dump v1".into());
    w(format!("# engine openfront-engine; prng rust/engine/src/prng.rs"));
    w(format!("# repo {}", repo.display()));
    w(format!("map {MAP}"));
    w(format!("bots {BOTS}"));
    w(format!("seeds {}", SEEDS.len()));

    // ---------------------------------------------------------------- PRNG ---
    // Pure PRNG sections first: these need no game and are the same table the
    // CUDA host compares against.
    for (si, seed) in SEEDS.iter().enumerate() {
        let game_id = seed_to_game_id(seed);
        let game_hash = simple_hash(&game_id);

        // (a) 24 raw u32 values of next().
        let mut pr = PseudoRandom::new(game_hash);
        let stream: Vec<u32> = (0..24).map(|_| next_u32(&mut pr)).collect();

        // (b) 16 draws per range, each from a fresh PRNG (`prng.rs:47-51`).
        let ranges: [(i32, i32); 4] = [(0, 100), (40, 80), (0, 7), (0, 5)];
        let mut draws_lines = Vec::new();
        for (lo, hi) in ranges {
            let mut r = PseudoRandom::new(game_hash);
            let vals: Vec<i32> = (0..16).map(|_| r.next_int(lo, hi)).collect();
            draws_lines.push(format!(
                "draws {si} {lo} {hi} {}",
                vals.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ")
            ));
        }

        // (c) chance / rand_element / shuffle semantics.
        let mut c = PseudoRandom::new(game_hash);
        let chance100: Vec<u8> = (0..16).map(|_| c.chance(100) as u8).collect();
        let chance2: Vec<u8> = (0..16).map(|_| c.chance(2) as u8).collect();
        let chance1: Vec<u8> = (0..16).map(|_| c.chance(1) as u8).collect();
        let chance1_all = chance1.iter().all(|&b| b == 1);

        // rand_element over a 7-element list: the picked index is the value.
        let arr: Vec<i32> = (0..7).collect();
        let mut e = PseudoRandom::new(game_hash);
        let picked: Vec<i32> = (0..3).map(|_| e.rand_element(&arr).unwrap()).collect();
        let after_rand: Vec<u32> = (0..3).map(|_| next_u32(&mut e)).collect();

        // Empty list: `rand_element` must draw NOTHING (`prng.rs:86-92`).
        let empty: [i32; 0] = [];
        let mut z = PseudoRandom::new(game_hash);
        let before = z.debug_state();
        let got = z.rand_element(&empty);
        let after = z.debug_state();
        let consumed = if before == after { 0 } else { 1 };
        let after_empty: Vec<u32> = (0..3).map(|_| next_u32(&mut z)).collect();

        // shuffle_array of 8 elements = 7 draws (`prng.rs:103-110`).
        let mut s = PseudoRandom::new(game_hash);
        let sh = s.shuffle_array(&(0..8).collect::<Vec<i32>>());
        let after_shuffle: Vec<u32> = (0..3).map(|_| next_u32(&mut s)).collect();

        // (d) the per-player seed shape: simple_hash(player_id).
        let player_id = format!("DUMPPLAYER{si}");
        let phash = simple_hash(&player_id);
        let mut pp = PseudoRandom::new(phash);
        let pstream: Vec<u32> = (0..24).map(|_| next_u32(&mut pp)).collect();

        w(format!("seed {si} {seed} {game_id} {game_hash}"));
        w(format!("stream {si} {}", hex_u32s(&stream)));
        for l in draws_lines {
            w(l);
        }
        w(format!(
            "sem {si} chance100 {}",
            chance100.iter().map(|b| b.to_string()).collect::<Vec<_>>().join("")
        ));
        w(format!(
            "sem {si} chance2 {}",
            chance2.iter().map(|b| b.to_string()).collect::<Vec<_>>().join("")
        ));
        w(format!("sem {si} chance1_all {chance1_all}"));
        w(format!(
            "sem {si} rand7 {} {}",
            picked.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "),
            hex_u32s(&after_rand)
        ));
        w(format!(
            "sem {si} rand_empty {got:?} consumed {consumed} next {}",
            hex_u32s(&after_empty)
        ));
        w(format!(
            "sem {si} shuffle8 {} next {}",
            sh.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "),
            hex_u32s(&after_shuffle)
        ));
        w(format!("pstream {si} {player_id} {phash} {}", hex_u32s(&pstream)));
    }

    // -------------------------------------------------------------- spawns ---
    // Drive the real engine: stage 0, 2 bots, 1 human agent, FFA singleplayer.
    for (si, seed) in SEEDS.iter().enumerate() {
        let (mut session, _, _, _, _, _) = RlSession::reset(&repo, MAP, seed, BOTS, "Easy", json!(0), 1)
            .expect("RlSession::reset");
        let (width, height) = {
            let game: &Game = &session.game;
            (game.width(), game.height())
        };
        {
            let game: &Game = &session.game;
            w(format!("dims {si} {width} {height}"));
            w(format!(
                "players {si} {}",
                game.all_players()
                    .iter()
                    .map(|p| format!(
                        "{}:{}:{}",
                        p.small_id,
                        match p.player_type {
                            PlayerType::Human => "H",
                            PlayerType::Bot => "B",
                            PlayerType::Nation => "N",
                        },
                        p.id
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            ));

            // Owner plane before anything spawns: the state bot #1 selects against.
            let plane0 = owner_plane(game);
            w(format!("ownerplane_before_bot1 {si} {}", nonzero_count(&plane0)));
        }

        // Advance until both bots have spawned. The bots' `SpawnExecution`s are
        // queued by `RlSession::reset` (`rl.rs:196-201` -> `TribeSpawner`).
        let mut ticks = 0u32;
        while ticks < 20 && bots_pending(&session.game, BOTS) {
            session.step(&[], 1);
            ticks += 1;
        }

        let game: &Game = &session.game;
        let mut bots: Vec<(u16, String, u32)> = game
            .all_players()
            .iter()
            .filter(|p| p.player_type == PlayerType::Bot)
            .map(|p| (p.small_id, p.id.clone(), p.small_id as u32))
            .collect();
        bots.sort_by_key(|b| b.0);

        let plane1 = owner_plane(game);

        for (bi, (small_id, pid, _)) in bots.iter().enumerate() {
            let tile = game
                .spawn_tile_of(*small_id)
                .unwrap_or_else(|| panic!("bot {bi} did not spawn"));
            let seed_sum = simple_hash(pid).wrapping_add(simple_hash(&seed_to_game_id(seed)));
            // The owner plane the selection ran against: bot #0 sees the empty
            // plane; bot #1 sees bot #0's footprint only. `plane1` is the plane
            // after the spawn tick, so zero out this bot's own tiles (it does
            // not own them yet when it selects) and keep every other owner.
            let pre: Vec<u16> = if bi == 0 {
                vec![0u16; plane1.len()]
            } else {
                plane1
                    .iter()
                    .map(|&v| if v == *small_id { 0 } else { v })
                    .collect()
            };
            let prev: Vec<u32> = bots[..bi].iter().filter_map(|(s, _, _)| game.spawn_tile_of(*s)).collect();
            w(format!(
                "spawn {si} {bi} {pid} {seed_sum} {tile} {} {} {} {}",
                tile % width,
                tile / width,
                nonzero_count(&pre),
                prev.len()
            ));
            for (t, v) in pre.iter().enumerate() {
                if *v != 0 {
                    w(format!("owner {si} {bi} {t} {v}"));
                }
            }
            for t in &prev {
                w(format!("prev {si} {bi} {t}"));
            }
        }

        // Scripted human spawn: an explicit tile -> the spawn path draws
        // NOTHING (`spawn_util.rs:71-77` returns before `rand_tile`).
        let (human_id, human_small, human_tile, human_seed) = {
            let game: &Game = &session.game;
            let human = game
                .all_players()
                .iter()
                .find(|p| p.player_type == PlayerType::Human)
                .expect("human player");
            let human_id = human.id.clone();
            let human_seed =
                simple_hash(&human_id).wrapping_add(simple_hash(&seed_to_game_id(seed)));
            (
                human_id,
                human.small_id,
                first_free_land(game, (width * height) / 3),
                human_seed,
            )
        };
        session.step(&[json!({"type": "spawn", "tile": human_tile})], 1);
        if !session.game.has_spawned(human_small) {
            // Execs added during `step` can be deferred to the following tick.
            session.step(&[], 1);
        }
        let game: &Game = &session.game;
        let spawned = game.has_spawned(human_small);
        let got_tile = game.spawn_tile_of(human_small);
        w(format!(
            "human {si} {human_id} {human_seed} {human_tile} {} {} {spawned} {got_tile:?}",
            human_tile % width,
            human_tile / width
        ));
    }

    std::fs::write(&out_path, &o).expect("write dump");
    eprintln!("wrote {} ({} bytes)", out_path.display(), o.len());
    print!("{o}");
}

fn owner_plane(game: &Game) -> Vec<u16> {
    let n = (game.width() * game.height()) as usize;
    (0..n as u32).map(|t| game.map.owner_id(t)).collect()
}

fn nonzero_count(v: &[u16]) -> usize {
    v.iter().filter(|&&x| x != 0).count()
}

fn bots_pending(game: &Game, n: u32) -> bool {
    game.all_players()
        .iter()
        .filter(|p| p.player_type == PlayerType::Bot)
        .count()
        < n as usize
        || game
            .all_players()
            .iter()
            .filter(|p| p.player_type == PlayerType::Bot && !game.has_spawned(p.small_id))
            .count()
            > 0
}

fn first_free_land(game: &Game, start: u32) -> u32 {
    let n = game.width() * game.height();
    let mut t = start % n;
    for _ in 0..n {
        if game.is_land(t) && !game.has_owner(t) {
            return t;
        }
        t = (t + 1) % n;
    }
    panic!("no free land tile");
}

#[allow(dead_code)]
fn fmt_state(pr: &PseudoRandom) -> String {
    let (a, b, c, d) = pr.debug_state();
    let mut s = String::new();
    let _ = write!(s, "{a},{b},{c},{d}");
    s
}
