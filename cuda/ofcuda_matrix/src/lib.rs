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

#[derive(Clone, Copy, Debug)]
pub struct AttackSnap {
    pub owner: u16,
    pub target: u16,
    pub troops: f64,
}

#[derive(Clone, Debug, Default)]
pub struct PlayerSnap {
    pub sid: u16,
    pub ptype: char,
    pub id: String,
    pub tiles: i32,
    pub troops: i32,
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
                    troops: 0,
                    gold: 0,
                    nvec: 0,
                    prefix: 0,
                };
                p.id = it.next().unwrap_or("").to_string();
                p.tiles = it.next().unwrap_or("0").parse().unwrap_or(0);
                p.troops = it.next().unwrap_or("0").parse().unwrap_or(0);
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
            "ATTACK" => {
                let b: u32 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let owner: u16 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let target: u16 = it.next().unwrap_or("0").parse().unwrap_or(0);
                let bits = hex64(it.next().unwrap_or("0"));
                get!(b).attacks.push(AttackSnap {
                    owner,
                    target,
                    troops: f64::from_bits(bits),
                });
            }
            other => return Err(format!("unknown oracle tag {other:?} in {}", path.display())),
        }
    }
    by_b.sort_by_key(|(b, _)| *b);
    o.boundaries = by_b.into_iter().map(|(_, v)| v).collect();
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
