//! `ofcuda_matrix` - parsing and comparison for the map x agent-count matrix.
//!
//! Two text formats are read here, both produced by OTHER binaries:
//!
//! * the **engine oracle** dump (`ofcuda_matrix/oracle` - links
//!   `openfront-engine`): the reference replay. Nothing in the CUDA port is
//!   consulted to build it.
//! * the **spawn init** dump (`ofcuda_prng`'s `spawnall --dump`): the port's own
//!   initial state - spawn tiles verified bit-exact against the engine, and
//!   produced by the **CUDA spawn kernel** when `spawnall` runs without
//!   `--no-gpu`.
//!
//! Splitting the parsing out of `main.rs` keeps the driver readable; both
//! formats are line-oriented and are never re-encoded.

use std::collections::HashMap;
use std::path::Path;

// ---------------------------------------------------------------------------
// Oracle dump
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct AttackSnap {
    pub owner: u16,
    pub target: u16,
    pub troops: f64,
    /// `AttackExecution::source_tile()` (`attack.rs:1172`), `None` when the
    /// attack has no source tile. `None` selects `refresh_to_conquer`
    /// (`attack.rs:160-164`) as the init frontier.
    pub source: Option<u32>,
    /// `to_conquer_len()` at the boundary - the ENGINE's own carried heap size,
    /// so a device re-create can be *checked* against it, not assumed.
    pub heap_len: usize,
    /// `border_tile_count()` at the boundary, same purpose.
    pub border_len: usize,
    /// `AttackExecution::attack_id()` (`attack.rs:1139`). `AttackExecution::init`
    /// mints a fresh id (`attack.rs:156`), so a CHANGED id for a live
    /// `(owner, target)` is the engine having RE-CREATED the attack - a new exec
    /// object with a fresh `PseudoRandom::new(123)` and an empty `to_conquer`,
    /// appended to the end of `execs` (`game.rs:3657-3700`). It is the one
    /// unambiguous re-create signal in the record; troop bits only ever *hint*
    /// at it.
    pub attack_id: String,
}

#[derive(Clone, Debug, Default)]
pub struct PlayerSnap {
    pub sid: u16,
    pub ptype: char,
    pub id: String,
    pub tiles: i32,
    /// **f64, not i32**: the engine prints `p.troops` with `{}` on an f64, so the
    /// record round-trips exactly; parsing it as an integer threw away the
    /// fractional part that `attack_logic_at_tile` divides by for a player
    /// target (`defender_troops / defender_tiles`).
    pub troops: f64,
    pub gold: i64,
    pub nvec: usize,
    pub prefix: usize,
}

#[derive(Clone, Debug, Default)]
pub struct Boundary {
    pub b: u32,
    pub engine_tick: u32,
    pub hash: u64,
    pub owned_total: usize,
    pub players: Vec<PlayerSnap>,
    pub borders: HashMap<u16, Vec<u32>>,
    pub attacks: Vec<AttackSnap>,
    /// tiles this player claimed during the tick that ended at this boundary,
    /// in the engine's own order.
    pub claims: HashMap<u16, Vec<u32>>,
    /// boundary 0 only: the engine's full initial owned set per player.
    pub owned0: HashMap<u16, Vec<u32>>,
    /// players whose `owned_tiles` vector was reordered/pruned, so the
    /// tail-diff above is not a claim set for them at this boundary.
    pub churn: Vec<(u16, usize)>,
    /// sid -> `(last_cluster_calc, last_tile_change, is_disconnected)` as the
    /// ENGINE held them at this boundary (`Player::last_cluster_calc` /
    /// `last_tile_change`, `game.rs:96-97`). The device maintains its own copy
    /// and these are the reference it is checked against.
    pub cadence: HashMap<u16, (u32, u32, bool)>,
    /// `(a, b)` ordered pairs with `Game::is_friendly(a, b) == true` at this
    /// boundary. Emitted only when the alliance/team/disconnected signature
    /// changes, so a boundary with no FRIEND row inherits the previous one.
    pub friends: Vec<(u16, u16)>,
    /// The dump carried a FRIEND row for this boundary (`friends` is
    /// authoritative). A boundary without one inherits the previous row.
    pub friends_emitted: bool,
    /// Live transport-ship exec positions for the tick that ENDS at this
    /// boundary: `(exec_index, owner, motion_plan_dst, natk)` where `natk` is
    /// the number of live ATTACK execs ahead of the ship in `execs`.
    /// `Game::execute_next_tick` ticks `execs` in list order, so the ship's
    /// `land()` -> `game.conquer(dst)` is visible only to attacks ticking after
    /// index `exec_index`; `natk` is how many attacks tick BEFORE it.
    pub transports: Vec<(usize, u16, i64, usize)>,
}

#[derive(Clone, Debug, Default)]
pub struct Oracle {
    pub map: String,
    pub seed: String,
    pub game_id: String,
    pub game_hash: i64,
    pub agents: u32,
    pub nations: String,
    pub width: u32,
    pub height: u32,
    pub min_dist: u16,
    pub spawn_end_tick: u32,
    pub ticks: u32,
    /// sid -> (type, id, spawn_tile)
    pub roster: Vec<(u16, char, String, i64)>,
    pub selfcheck_diff: usize,
    pub selfcheck_owned: i64,
    /// The engine's own player-exec order, from `Game::exec_labels()`
    /// (`game.rs:1414-1416`) filtered to the `Player(<sid>)` labels:
    /// `execute_next_tick` (`game.rs:3662-3669`) ticks `execs` in list order,
    /// and the cluster pass walks the players in that order.
    pub pexec: Vec<u16>,
    pub boundaries: Vec<Boundary>,
}

fn hex64(s: &str) -> u64 {
    u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0)
}

/// The record for boundary `b`, created on first mention.
fn boundary_mut(by_b: &mut Vec<(u32, Boundary)>, b: u32) -> &mut Boundary {
    if let Some(i) = by_b.iter().position(|(x, _)| *x == b) {
        return &mut by_b[i].1;
    }
    by_b.push((b, Boundary { b, ..Default::default() }));
    let last = by_b.len() - 1;
    &mut by_b[last].1
}

pub fn parse_oracle(path: &Path) -> Result<Oracle, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut o = Oracle::default();
    let mut by_b: Vec<(u32, Boundary)> = Vec::new();

    macro_rules! get {
        ($b:expr) => {
            boundary_mut(&mut by_b, $b)
        };
    }

    for line in text.lines() {
        let mut it = line.split_whitespace();
        let Some(tag) = it.next() else { continue };
        match tag {
            // Measurement-only trace channels (see oracle/src/main.rs):
            // ignored here, never part of a comparison.
            "TRACE" | "TRACELOGIC" | "ENGINEHEAP" | "EXECS" => {}
            "#" | "REPO" | "DIFFICULTY" => {}
            "MAP" => o.map = it.next().unwrap_or("").to_string(),
            "SEED" => o.seed = it.next().unwrap_or("").to_string(),
            "GAME_ID" => o.game_id = it.next().unwrap_or("").to_string(),
            "GAME_HASH" => o.game_hash = it.next().unwrap_or("0").parse().unwrap_or(0),
            "AGENTS" => o.agents = it.next().unwrap_or("0").parse().unwrap_or(0),
            "NATIONS" => o.nations = it.next().unwrap_or("").to_string(),
            "HUMAN_AGENTS" => {}
            "WIDTH" => o.width = it.next().unwrap_or("0").parse().unwrap_or(0),
            "HEIGHT" => o.height = it.next().unwrap_or("0").parse().unwrap_or(0),
            "MIN_DIST" => o.min_dist = it.next().unwrap_or("0").parse().unwrap_or(0),
            "SPAWN_TICKS" => {}
            "SPAWN_END_TICK" => o.spawn_end_tick = it.next().unwrap_or("0").parse().unwrap_or(0),
            "TICKS" => o.ticks = it.next().unwrap_or("0").parse().unwrap_or(0),
            "ROSTER" => {
                let sid: u16 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let ty = it.next().unwrap_or("B").chars().next().unwrap_or('B');
                let id = it.next().unwrap_or("").to_string();
                let tile: i64 = it.next().unwrap_or("-1").parse().unwrap_or(-1);
                o.roster.push((sid, ty, id, tile));
            }
            "SELFCHECK" => {
                o.selfcheck_diff = it.next().unwrap_or("0").parse().unwrap_or(0);
                o.selfcheck_owned = it.next().unwrap_or("0").parse().unwrap_or(0);
            }
            "BOUNDARY" => {
                let b: u32 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let et: u32 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let h = hex64(it.next().unwrap_or("0"));
                let owned: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
                let e = get!(b);
                e.engine_tick = et;
                e.hash = h;
                e.owned_total = owned;
            }
            "OWNED" => {
                let b: u32 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let sid: u16 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let n: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
                let tiles: Vec<u32> = it.take(n).filter_map(|x| x.parse().ok()).collect();
                get!(b).owned0.insert(sid, tiles);
            }
            "TRANSPORT" => {
                let b: u32 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let i: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
                let owner: u16 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let dst: i64 = it.next().unwrap_or("-1").parse().unwrap_or(-1);
                let natk: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
                get!(b).transports.push((i, owner, dst, natk));
            }
            "BORDER" => {
                let b: u32 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let sid: u16 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let n: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
                let tiles: Vec<u32> = it.take(n).filter_map(|x| x.parse().ok()).collect();
                get!(b).borders.insert(sid, tiles);
            }
            "PLAYER" => {
                let b: u32 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let mut p = PlayerSnap {
                    sid: it.next().unwrap_or("0").parse().unwrap_or(0),
                    ptype: it.next().unwrap_or("B").chars().next().unwrap_or('B'),
                    id: String::new(),
                    tiles: 0,
                    troops: 0.0,
                    gold: 0,
                    nvec: 0,
                    prefix: 0,
                };
                p.id = it.next().unwrap_or("").to_string();
                p.tiles = it.next().unwrap_or("0").parse().unwrap_or(0);
                p.troops = it.next().unwrap_or("0").parse().unwrap_or(0.0);
                p.gold = it.next().unwrap_or("0").parse().unwrap_or(0);
                p.nvec = it.next().unwrap_or("0").parse().unwrap_or(0);
                p.prefix = it.next().unwrap_or("0").parse().unwrap_or(0);
                get!(b).players.push(p);
            }
            "CLAIM" => {
                let b: u32 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let sid: u16 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let n: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
                let tiles: Vec<u32> = it.take(n).filter_map(|x| x.parse().ok()).collect();
                get!(b).claims.insert(sid, tiles);
            }
            "CHURN" => {
                let b: u32 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let mut v = Vec::new();
                for tok in it {
                    if let Some((s, n)) = tok.split_once(':') {
                        if let (Ok(sid), Ok(n)) = (s.parse::<u16>(), n.parse::<usize>()) {
                            v.push((sid, n));
                        }
                    }
                }
                get!(b).churn = v;
            }
            "PEXEC" => {
                let n: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
                o.pexec = it.take(n).filter_map(|x| x.parse().ok()).collect();
            }
            "CADENCE" => {
                let b: u32 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let sid: u16 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let lcc: u32 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let ltc: u32 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let disc = it.next().unwrap_or("0").parse::<u32>().unwrap_or(0) != 0;
                get!(b).cadence.insert(sid, (lcc, ltc, disc));
            }
            "FRIEND" => {
                let b: u32 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let mut v = Vec::new();
                for tok in it {
                    if let Some((a, c)) = tok.split_once(':') {
                        if let (Ok(a), Ok(c)) = (a.parse::<u16>(), c.parse::<u16>()) {
                            v.push((a, c));
                        }
                    }
                }
                let e = get!(b);
                e.friends = v;
                e.friends_emitted = true;
            }
            "ATTACK" => {
                let b: u32 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let owner: u16 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let target: u16 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let bits = hex64(it.next().unwrap_or("0"));
                // v2 record: `... {troops_bits} {source|-1} {heap_len} {border_len} {attack_id}`.
                // A v1 line (`... {bits} 1`) is REJECTED rather than mis-read:
                // the trailing `1` was a literal placeholder, and parsing it as a
                // source tile would silently invent state.
                let missing = || {
                    format!(
                        "{}: ATTACK at boundary {b} is a v1 line (no source/heap/attack_id). \
                         Regenerate the oracle dump with the current `oracle` binary.",
                        path.display()
                    )
                };
                let src_raw: i64 = it.next().ok_or_else(missing)?.parse().unwrap_or(-1);
                let heap_len: usize = it.next().ok_or_else(missing)?.parse().unwrap_or(0);
                let border_len: usize = it.next().ok_or_else(missing)?.parse().unwrap_or(0);
                let attack_id = it.next().ok_or_else(missing)?.to_string();
                get!(b).attacks.push(AttackSnap {
                    owner,
                    target,
                    troops: f64::from_bits(bits),
                    source: if src_raw < 0 { None } else { Some(src_raw as u32) },
                    heap_len,
                    border_len,
                    attack_id,
                });
            }
            other => return Err(format!("unknown oracle tag {other:?} in {}", path.display())),
        }
    }
    by_b.sort_by_key(|(b, _)| *b);
    o.boundaries = by_b.into_iter().map(|(_, v)| v).collect();
    // `FRIEND` rows are emitted only when the friendly-pair SET changes (it is
    // an O(n^2) `is_friendly` sweep in the oracle), so a boundary without one
    // inherits the most recent preceding row. Boundaries before the first
    // `FRIEND` row keep the empty default - the oracle emits boundary 0.
    {
        let mut cur: Vec<(u16, u16)> = Vec::new();
        let mut seen = false;
        for bd in o.boundaries.iter_mut() {
            if bd.friends_emitted {
                cur = bd.friends.clone();
                seen = true;
            } else if seen {
                bd.friends = cur.clone();
            }
        }
    }
    if o.boundaries.is_empty() {
        return Err(format!("{}: no boundaries", path.display()));
    }
    Ok(o)
}

// ---------------------------------------------------------------------------
// spawnall init dump
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct InitDump {
    pub map: String,
    pub seed: String,
    pub game_id: String,
    pub agents: u32,
    pub nations: String,
    pub width: u32,
    pub height: u32,
    pub min_dist: u16,
    /// bot index -> (small_id, id, spawn_tile, n_tiles)
    pub bots: Vec<(u16, String, i64, usize)>,
    /// bot index -> owned tiles (the PORT's spawn footprint)
    pub owned: HashMap<usize, Vec<u32>>,
    /// nation index -> (small_id, id, cell_x, cell_y, spawn_tile, spawn_tick,
    /// n_tiles). The cell is the manifest `coordinates`; spawn_tile is the
    /// PORT's sampled land tile and spawn_tick the PORT's tick for it.
    pub nation_rows: Vec<(u16, String, i64, i64, i64, u32, usize)>,
    /// nation index -> owned tiles (the PORT's spawn footprint)
    pub nation_owned: HashMap<usize, Vec<u32>>,
    /// nation index -> the rejection-draw count the PORT's sampler burned
    pub nation_tries: HashMap<usize, u32>,
    /// nation index -> the PORT's starting troops for the nation
    pub nation_troops: HashMap<usize, f64>,
    /// engine roster from the same file (reference, not port)
    pub players: Vec<(u16, char, String, i64, u32, i32)>,
}

pub fn parse_init(path: &Path) -> Result<InitDump, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut d = InitDump::default();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let Some(tag) = it.next() else { continue };
        match tag {
            "#" => {}
            "map" => d.map = it.next().unwrap_or("").to_string(),
            "seed" => d.seed = it.next().unwrap_or("").to_string(),
            "game_id" => d.game_id = it.next().unwrap_or("").to_string(),
            "game_hash" => {}
            "agents" => d.agents = it.next().unwrap_or("0").parse().unwrap_or(0),
            "nations" => d.nations = it.next().unwrap_or("").to_string(),
            "human_agents" => {}
            "width" => d.width = it.next().unwrap_or("0").parse().unwrap_or(0),
            "height" => d.height = it.next().unwrap_or("0").parse().unwrap_or(0),
            "min_dist" => d.min_dist = it.next().unwrap_or("0").parse().unwrap_or(0),
            "player" => {
                d.players.push((
                    it.next().unwrap_or("0").parse().unwrap_or(0),
                    it.next().unwrap_or("B").chars().next().unwrap_or('B'),
                    it.next().unwrap_or("").to_string(),
                    it.next().unwrap_or("-1").parse().unwrap_or(-1),
                    it.next().unwrap_or("0").parse().unwrap_or(0),
                    it.next().unwrap_or("0").parse().unwrap_or(0),
                ));
            }
            "bot" => {
                let bi: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
                let sid: u16 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let id = it.next().unwrap_or("").to_string();
                let _seed: i64 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let tile: i64 = it.next().unwrap_or("-1").parse().unwrap_or(-1);
                let n: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
                while d.bots.len() <= bi {
                    d.bots.push((0, String::new(), -1, 0));
                }
                d.bots[bi] = (sid, id, tile, n);
            }
            "owned" => {
                let bi: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
                let tiles: Vec<u32> = it.filter_map(|x| x.parse().ok()).collect();
                d.owned.insert(bi, tiles);
            }
            "nation" => {
                let ni: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
                let sid: u16 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let id = it.next().unwrap_or("").to_string();
                let cx: i64 = it.next().unwrap_or("-1").parse().unwrap_or(-1);
                let cy: i64 = it.next().unwrap_or("-1").parse().unwrap_or(-1);
                let tile: i64 = it.next().unwrap_or("-1").parse().unwrap_or(-1);
                let tick: u32 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let n: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
                while d.nation_rows.len() <= ni {
                    d.nation_rows.push((0, String::new(), -1, -1, -1, 0, 0));
                }
                d.nation_rows[ni] = (sid, id, cx, cy, tile, tick, n);
            }
            "nationowned" => {
                let ni: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
                let tiles: Vec<u32> = it.filter_map(|x| x.parse().ok()).collect();
                d.nation_owned.insert(ni, tiles);
            }
            "nationtries" => {
                let ni: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
                let n: u32 = it.next().unwrap_or("0").parse().unwrap_or(0);
                d.nation_tries.insert(ni, n);
            }
            "nationtroops" => {
                let ni: usize = it.next().unwrap_or("0").parse().unwrap_or(0);
                let t: f64 = it.next().unwrap_or("0").parse().unwrap_or(0.0);
                d.nation_troops.insert(ni, t);
            }
            other => return Err(format!("unknown init tag {other:?} in {}", path.display())),
        }
    }
    Ok(d)
}

/// The owner plane the PORT's init implies: `small_id` in the word, 0 unowned.
pub fn plane_from_init(d: &InitDump, w: u32, h: u32) -> Result<Vec<u16>, String> {
    let mut plane = vec![0u16; (w as usize) * (h as usize)];
    for (bi, tiles) in &d.owned {
        let sid = d
            .bots
            .get(*bi)
            .map(|b| b.0)
            .ok_or_else(|| format!("init dump has `owned {bi}` with no `bot {bi}`"))?;
        for &t in tiles {
            let slot = plane
                .get_mut(t as usize)
                .ok_or_else(|| format!("init tile {t} out of range for {w}x{h}"))?;
            *slot = sid;
        }
    }
    // Nations own tiles too - the engine's boundary-0 plane includes them.
    for (ni, tiles) in &d.nation_owned {
        let sid = d
            .nation_rows
            .get(*ni)
            .map(|n| n.0)
            .ok_or_else(|| format!("init dump has `nationowned {ni}` with no `nation {ni}`"))?;
        for &t in tiles {
            let slot = plane
                .get_mut(t as usize)
                .ok_or_else(|| format!("init nation tile {t} out of range for {w}x{h}"))?;
            *slot = sid;
        }
    }
    Ok(plane)
}

/// Per-player owned sets from a plane (ascending tile order).
pub fn owners_of(plane: &[u16]) -> HashMap<u16, Vec<u32>> {
    let mut m: HashMap<u16, Vec<u32>> = HashMap::new();
    for (i, &o) in plane.iter().enumerate() {
        if o != 0 {
            m.entry(o).or_default().push(i as u32);
        }
    }
    m
}

pub fn fnv1a_u16_le(plane: &[u16]) -> u64 {
    ofcuda_hash::fnv1a_u16_le(ofcuda_hash::FNV_OFFSET_BASIS, plane)
}

/// The engine's own `gameHash` formula (`hash.rs:7-22`), as a cross-check that
/// the oracle's roster/troops/tiles agree with the engine's checksum.
pub fn game_hash_from(players: &[PlayerSnap]) -> i64 {
    let mut h = 1.0f64;
    for p in players {
        h += ofcuda_prng::simple_hash(&p.id) as f64 * (p.troops as f64 + p.tiles as f64);
    }
    h as i64
}
