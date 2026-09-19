//! `ofcuda_tick` - host-side shared code for the attack/expansion FRONTIER step.
//!
//! Nothing here touches CUDA: the GPU binary (`src/main.rs`) and the CPU
//! companion (`src/bin/cpu.rs`) link the *same* case reconstruction, the same
//! comparison and the same expectations. Only the frontier computation differs
//! (a cuda-oxide kernel vs. a plain sequential Rust loop), which is what makes a
//! mismatch meaningful.
//!
//! See README.md for what the step is, what the recorded cases are, and what is
//! measured vs. assumed.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

// The frontier core, textually shared with the device module in `main.rs`.
mod core_impl {
    include!("core_impl.rs");
}
pub use core_impl::*;

// ---------------------------------------------------------------------------
// Recorded-case dumps
// ---------------------------------------------------------------------------

/// One player's per-tick record in a parity dump.
///
/// `owned_tiles` / `border_order` / `owned_order` are only present in the
/// `ownedOrder`-carrying dumps (the durable copies under `cases/`); the
/// `hash_parity.*` dumps under `parity-diag/` carry hashes only and cannot
/// reconstruct a frontier.
#[derive(Clone, Debug, Default)]
pub struct PlayerRec {
    pub small_id: u32,
    pub name: String,
    pub tiles: u32,
    pub owned_tiles: Vec<u32>,
    pub border_order: Vec<u32>,
    pub owned_order: Vec<u32>,
}

pub type Dump = HashMap<u32, HashMap<String, PlayerRec>>;

pub fn parse_dump(path: &Path) -> Result<Dump, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out: Dump = HashMap::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value =
            serde_json::from_str(line).map_err(|e| format!("{}: {e}", path.display()))?;
        if v.get("type").and_then(|t| t.as_str()) == Some("header") {
            continue;
        }
        let Some(tick) = v.get("tick").and_then(|t| t.as_u64()) else {
            continue;
        };
        let mut players = HashMap::new();
        if let Some(arr) = v.get("players").and_then(|p| p.as_array()) {
            for p in arr {
                let id = p
                    .get("id")
                    .and_then(|x| x.as_str())
                    .unwrap_or("")
                    .to_string();
                let nums = |k: &str| -> Vec<u32> {
                    p.get(k)
                        .and_then(|x| x.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_u64())
                                .map(|v| v as u32)
                                .collect()
                        })
                        .unwrap_or_default()
                };
                players.insert(
                    id,
                    PlayerRec {
                        small_id: p.get("smallId").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
                        name: p
                            .get("name")
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string(),
                        tiles: p.get("tiles").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
                        owned_tiles: nums("ownedTiles"),
                        border_order: nums("borderOrder"),
                        owned_order: nums("ownedOrder"),
                    },
                );
            }
        }
        out.insert(tick as u32, players);
    }
    Ok(out)
}

/// The tiles claimed strictly between `tick-1` and `tick`, in claim order.
pub fn claims_at(dump: &Dump, pid: &str, tick: u32) -> Vec<u32> {
    let Some(now) = dump.get(&tick).and_then(|t| t.get(pid)) else {
        return Vec::new();
    };
    let prev_len = dump
        .get(&(tick - 1))
        .and_then(|t| t.get(pid))
        .map(|p| p.owned_order.len())
        .unwrap_or(0);
    if now.owned_order.len() <= prev_len {
        return Vec::new();
    }
    now.owned_order[prev_len..].to_vec()
}

// ---------------------------------------------------------------------------
// Map plane
// ---------------------------------------------------------------------------

pub struct MapPlane {
    pub width: u32,
    pub height: u32,
    pub terrain: Vec<u8>,
}

pub fn load_map(map_dir: &Path) -> Result<MapPlane, String> {
    let manifest = fs::read_to_string(map_dir.join("manifest.json"))
        .map_err(|e| format!("{}: {e}", map_dir.display()))?;
    let v: serde_json::Value = serde_json::from_str(&manifest).map_err(|e| e.to_string())?;
    let m = v.get("map").ok_or("manifest has no `map`")?;
    let width = m.get("width").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
    let height = m.get("height").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
    let terrain =
        fs::read(map_dir.join("map.bin")).map_err(|e| format!("{}: {e}", map_dir.display()))?;
    if terrain.len() != (width as usize) * (height as usize) {
        return Err(format!(
            "map.bin is {} bytes, expected {width}x{height}",
            terrain.len()
        ));
    }
    Ok(MapPlane {
        width,
        height,
        terrain,
    })
}

// ---------------------------------------------------------------------------
// Case construction
// ---------------------------------------------------------------------------

/// Everything the frontier step consumes, plus the record it must reproduce.
#[derive(Clone, Debug)]
pub struct FrontierCase {
    pub name: String,
    pub player_id: String,
    pub player_name: String,
    /// Tick at which the claimed tiles appear in the record.
    pub claim_tick: u32,
    /// Tick that goes into the priority (`game.ticks()` when the frontier was
    /// built). Measured, not assumed - see README.md.
    pub priority_tick: u32,
    /// Position of the shared sfc32 stream when the step starts.
    pub cursor: u32,
    /// Border visit order (`ORDER_NSWE` / `ORDER_WENS`).
    pub order: u32,
    /// The attacker's border tiles, in the engine's border-set iteration order.
    pub border: Vec<u32>,
    /// Tiles owned by anybody at `claim_tick - 1` (eligibility filter).
    pub owned_any: Vec<u32>,
    /// Tiles owned by the attacker at `claim_tick - 1` (owner count).
    pub owned_mine: Vec<u32>,
    /// Recorded claim order for this tick.
    pub expected_order: Vec<u32>,
    /// Recorded pop priorities for those claims.
    pub expected_priorities: Vec<f32>,
    /// The other engine's claim order for the same tick.
    pub other_engine_order: Vec<u32>,
    /// Priority units `(priority - priority_tick) / f` for the claims, when the
    /// record states them as units rather than as absolute priorities.
    pub expected_units: Vec<u32>,
}

pub fn build_case(
    name: &str,
    native: &Dump,
    ts: &Dump,
    pid: &str,
    claim_tick: u32,
    priority_tick: u32,
    cursor: u32,
    order: u32,
    expected_priorities: Vec<f32>,
    expected_units: Vec<u32>,
) -> Result<FrontierCase, String> {
    let prev = native
        .get(&(claim_tick - 1))
        .and_then(|t| t.get(pid))
        .ok_or_else(|| format!("no record for {pid} at tick {}", claim_tick - 1))?;
    let now = native
        .get(&claim_tick)
        .and_then(|t| t.get(pid))
        .ok_or_else(|| format!("no record for {pid} at tick {claim_tick}"))?;

    let mut owned_any: Vec<u32> = Vec::new();
    for (_, p) in native
        .get(&(claim_tick - 1))
        .ok_or_else(|| format!("no tick {} in dump", claim_tick - 1))?
    {
        owned_any.extend_from_slice(&p.owned_tiles);
    }
    // Deterministic: the dump is a HashMap, so sort.
    owned_any.sort_unstable();
    owned_any.dedup();

    Ok(FrontierCase {
        name: name.to_string(),
        player_id: pid.to_string(),
        player_name: now.name.clone(),
        claim_tick,
        priority_tick,
        cursor,
        order,
        border: prev.border_order.clone(),
        owned_any,
        owned_mine: prev.owned_tiles.clone(),
        expected_order: claims_at(native, pid, claim_tick),
        expected_priorities,
        other_engine_order: claims_at(ts, pid, claim_tick),
        expected_units,
    })
}

/// The two recorded cases, built from the preserved dumps under `cases/`.
///
/// * b002 / tick 319 / Frankish Duchy (`5iznss4u`) - Highland border, f = 0.75.
/// * b004 / tick 309 / Armenian Supremacy (`be0bsg87`) - Mountain border, f = 1.
pub fn recorded_cases(map: &MapPlane) -> Result<(FrontierCase, FrontierCase), String> {
    let _ = map;
    let n1 = parse_dump(&case_path("b002-t319.native.ndjson"))?;
    let t1 = parse_dump(&case_path("b002-t319.ts.ndjson"))?;
    let n2 = parse_dump(&case_path("b004-t309.native.ndjson"))?;
    let t2 = parse_dump(&case_path("b004-t309.ts.ndjson"))?;

    let c1 = build_case(
        "b002-s1-pangaea tick 319 (Frankish Duchy 5iznss4u)",
        &n1,
        &t1,
        "5iznss4u",
        319,
        318,
        0,
        ORDER_WENS,
        vec![325.5, 325.5, 325.5, 326.25, 326.25, 327.0, 327.0, 328.5],
        vec![10, 10, 10, 11, 11, 12, 12, 14],
    )?;
    let c2 = build_case(
        "b004-s2-pangaea tick 309 (Armenian Supremacy be0bsg87)",
        &n2,
        &t2,
        "be0bsg87",
        309,
        308,
        0,
        ORDER_WENS,
        vec![318.0, 318.0, 318.0, 319.0, 319.0, 320.0, 320.0, 322.0],
        vec![10, 10, 10, 11, 11, 12, 12, 14],
    )?;
    Ok((c1, c2))
}

// ---------------------------------------------------------------------------
// The frontier step, host side
// ---------------------------------------------------------------------------

pub fn frontier_cpu(case: &FrontierCase, map: &MapPlane) -> (Vec<(u32, f32)>, u32) {
    let mut heap = Heap::new();
    let n_enq = frontier_enqueue(
        &case.border,
        &case.owned_any,
        &case.owned_mine,
        &map.terrain,
        map.width,
        map.height,
        case.order,
        case.priority_tick,
        case.cursor,
        &mut heap,
    );
    let mut pops = Vec::with_capacity(n_enq as usize);
    while let Some(p) = heap.dequeue() {
        pops.push(p);
    }
    (pops, n_enq)
}

/// First occurrence of each tile in the pop sequence: the tiles that actually
/// get claimed, in claim order. (The engine pops a duplicate later, finds
/// `owner_id(tile) != target_small_id` and `continue`s.)
pub fn claim_order(pops: &[(u32, f32)]) -> Vec<(u32, f32)> {
    let mut seen: Vec<u32> = Vec::new();
    let mut out = Vec::new();
    for (t, p) in pops {
        if !seen.contains(t) {
            seen.push(*t);
            out.push((*t, *p));
        }
    }
    out
}

/// Run the step and reduce it to `(claim tiles, claim priorities)`.
pub fn run_cpu(case: &FrontierCase, map: &MapPlane) -> (Vec<u32>, Vec<f32>, u32) {
    let (pops, n_enq) = frontier_cpu(case, map);
    let claims = claim_order(&pops);
    (
        claims.iter().map(|c| c.0).collect(),
        claims.iter().map(|c| c.1).collect(),
        n_enq,
    )
}

/// `1 - num_owned_by_me * 0.5 + mag / 2` for one tile, recomputed from the
/// case's inputs. This is the `f` in `priority = (r + 10) * f + tick`.
pub fn priority_factor(tile: u32, case: &FrontierCase, map: &MapPlane) -> f32 {
    let mut buf = [0u32; 4];
    let n = neighbors4(case.order, tile, map.width, map.height, &mut buf);
    let mut k = 0u32;
    for i in 0..n as usize {
        if owned_contains(&case.owned_mine, buf[i]) {
            k += 1;
        }
    }
    let mag2 = mag2_from_terrain(map.terrain[tile as usize]);
    1.0f32 - (k as f32) * 0.5f32 + (mag2 as f32) * 0.25f32
}

/// `(priority - priority_tick) / f` as an integer: the "units" a record states
/// a priority in when it does not state the absolute value.
pub fn priority_unit(tile: u32, pri: f32, case: &FrontierCase, map: &MapPlane) -> u32 {
    let f = priority_factor(tile, case, map);
    ((pri - case.priority_tick as f32) / f).round() as u32
}

/// Every enqueue this case performs, with the draw, the owner count, the
/// magnitude and BOTH priority evaluations - so the f32-vs-f64 claim can be
/// measured instead of assumed.
pub fn enqueue_trace(
    case: &FrontierCase,
    map: &MapPlane,
) -> Vec<(u32, i32, u32, u32, f32, f32)> {
    let mut pr = Prng::new(SEED);
    for _ in 0..case.cursor {
        pr.next_u32();
    }
    let mut out = Vec::new();
    for bt in &case.border {
        let mut nbuf = [0u32; 4];
        let n = neighbors4(case.order, *bt, map.width, map.height, &mut nbuf);
        for i in 0..n as usize {
            let nb = nbuf[i];
            if map.terrain[nb as usize] & 0x80 == 0 || owned_contains(&case.owned_any, nb) {
                continue;
            }
            let r = pr.next_int(0, 7);
            let mut ibuf = [0u32; 4];
            let inner = neighbors4(case.order, nb, map.width, map.height, &mut ibuf);
            let mut k = 0u32;
            for j in 0..inner as usize {
                if owned_contains(&case.owned_mine, ibuf[j]) {
                    k += 1;
                }
            }
            let mag2 = mag2_from_terrain(map.terrain[nb as usize]);
            out.push((
                nb,
                r,
                k,
                mag2,
                priority_f32(r, k, mag2, case.priority_tick),
                priority_f64(r, k, mag2, case.priority_tick),
            ));
        }
    }
    out
}

/// The order the record was produced under, and the order the tip uses.
pub const CASE_OTHER_ENGINE_ORDER_NOTE: &str =
    "recorded 2026-09-18 with the pre-fix native engine (W,E,N,S); the current tip uses N,S,W,E";

pub fn map_dir_note() -> String {
    format!("map plane from {REPO_ROOT}/openfront/resources/maps")
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

#[derive(Default, Debug)]
pub struct Tally {
    pub matched: usize,
    pub total: usize,
    pub first_divergence: Option<String>,
}

impl Tally {
    pub fn push<T: std::fmt::Debug + PartialEq>(
        &mut self,
        label: &str,
        idx: usize,
        expect: &T,
        got: &T,
    ) {
        self.total += 1;
        if expect == got {
            self.matched += 1;
        } else if self.first_divergence.is_none() {
            self.first_divergence =
                Some(format!("{label}[{idx}]: expected {expect:?} vs port {got:?}"));
        }
    }
    pub fn line(&self, what: &str) -> String {
        format!(
            "{what}: {}/{} match{}",
            self.matched,
            self.total,
            match &self.first_divergence {
                None => String::new(),
                Some(d) => format!("  FIRST DIVERGENCE {d}"),
            }
        )
    }
}

pub fn fmt_f32(v: f32) -> String {
    if v == v.trunc() {
        format!("{v:.1}")
    } else {
        format!("{v}")
    }
}

pub fn fmt_u32s(v: &[u32]) -> String {
    format!(
        "[{}]",
        v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
    )
}

/// Index of the first differing element (or `None` when the prefixes agree).
pub fn first_diff<T: PartialEq>(a: &[T], b: &[T]) -> Option<usize> {
    for i in 0..a.len().max(b.len()) {
        if a.get(i) != b.get(i) {
            return Some(i);
        }
    }
    None
}

pub fn fmt_prios(v: &[f32]) -> String {
    format!(
        "[{}]",
        v.iter()
            .map(|x| fmt_f32(*x))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

pub const REPO_ROOT: &str = "/opt/data/workspaces/skg/openfront-ai";
pub const CASES_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/cases");

pub fn case_path(file: &str) -> PathBuf {
    PathBuf::from(CASES_DIR).join(file)
}

pub fn pangaea_map_dir() -> PathBuf {
    Path::new(REPO_ROOT).join("openfront/resources/maps/pangaea")
}

/// The FMA hazard, measured instead of argued.
///
/// The CUDA backend contracts `(r + 10) * f + tick` into a single `fma.rn.f32`
/// (confirmed in the PTX of the compiled `frontier_step`), while a Rust f32 host
/// evaluates `a * b + c` as a rounded multiply followed by a rounded add. Those
/// two disagree in general, so the port must show how much room they have here:
/// this enumerates the whole space this arithmetic can see - every terrain
/// magnitude, owner count, draw and a range of ticks - and counts the inputs
/// where the contracted and the two-step results have different bits.
///
/// Returns `(total, differing, examples)`.
pub fn fma_contraction_sensitivity() -> (u64, u64, Vec<String>) {
    let ticks = [0u32, 1, 7, 100, 308, 309, 318, 319, 400, 1000, 100_000];
    let mut total = 64u64;
    let mut differing = 0u64;
    let mut examples: Vec<String> = Vec::new();
    for mag in [80u32, 100, 120] {
        let mag2 = mag / 2;
        for k in 0..=4u32 {
            let f = 1.0f32 - (k as f32) * 0.5f32 + (mag2 as f32) * 0.5f32;
            for r in 0..7i32 {
                let base = (r + 10) as f32;
                for tick in ticks {
                    let two_step = base * f + (tick as f32);
                    let contracted = base.mul_add(f, tick as f32);
                    total += 1;
                    if two_step.to_bits() != contracted.to_bits() {
                        differing += 1;
                        if examples.len() < 8 {
                            examples.push(format!(
                                "r={r} k={k} mag={mag} tick={tick}: mul+add {} vs fma {}",
                                fmt_f32(two_step),
                                fmt_f32(contracted)
                            ));
                        }
                    }
                }
            }
        }
    }
    (total, differing, examples)
}

// ---------------------------------------------------------------------------
// Part 1: the CONTESTED-BORDER case
// ---------------------------------------------------------------------------
//
// `cases/b007-t581-sid2.contested.json` is a fixed-engine slice (dump of
// `records/early-curriculum-parity/curr-b007-s3-pangaea.json.gz`, ticks 581
// and 582, every 1 tick, `OF_DUMP_OWNED_TILES/BORDER_ORDER/OWNED_ORDER=1`).
//
// It is the case the two earlier ones could not be: player 2 (Maori Council)
// has border tiles adjacent to a RIVAL's tiles, so `owned_any` (everybody's
// tiles - what `owner_id(nb) != target_small_id(0)` rejects) and `owned_mine`
// (the attacker's own tiles - what the priority's `num_owned_by_me` counts) are
// genuinely different sets. Conflating them changes the enqueue stream.

#[derive(Clone, Debug)]
pub struct ContestedPlayer {
    pub small_id: u32,
    pub name: String,
    pub owned_tiles: Vec<u32>,
    pub border_order: Vec<u32>,
    pub owned_order: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct ContestedCase {
    pub game: String,
    /// Tick whose state the frontier is built from (`claim_tick - 1`).
    pub state_tick: u32,
    pub claim_tick: u32,
    pub owner_sid: u32,
    pub players: Vec<ContestedPlayer>,
    pub expected_claims: Vec<u32>,
}

pub fn load_contested_case(path: &Path) -> Result<ContestedCase, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let mut players = Vec::new();
    for p in v.get("players").and_then(|x| x.as_array()).ok_or("no players")? {
        let nums = |k: &str| -> Vec<u32> {
            p.get(k)
                .and_then(|x| x.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect())
                .unwrap_or_default()
        };
        players.push(ContestedPlayer {
            small_id: p.get("smallId").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
            name: p.get("name").and_then(|x| x.as_str()).unwrap_or("").to_string(),
            owned_tiles: nums("ownedTiles"),
            border_order: nums("borderOrder"),
            owned_order: nums("ownedOrder"),
        });
    }
    Ok(ContestedCase {
        game: v.get("game").and_then(|x| x.as_str()).unwrap_or("").to_string(),
        state_tick: v.get("tick").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        claim_tick: v.get("claim_tick").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        owner_sid: v.get("owner_sid").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        players,
        expected_claims: v
            .get("expected_claims")
            .and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect())
            .unwrap_or_default(),
    })
}

pub fn contested_case_path() -> PathBuf {
    case_path("b007-t581-sid2.contested.json")
}

/// The frontier step's inputs for the contested case, in `FrontierCase` form so
/// the SAME kernel, heap and priority code run on it.
pub fn contested_frontier_case(c: &ContestedCase) -> Result<FrontierCase, String> {
    let me = c
        .players
        .iter()
        .find(|p| p.small_id == c.owner_sid)
        .ok_or_else(|| format!("no player {}", c.owner_sid))?;
    let mut owned_any: Vec<u32> = Vec::new();
    for p in &c.players {
        owned_any.extend_from_slice(&p.owned_tiles);
    }
    owned_any.sort_unstable();
    owned_any.dedup();
    if owned_any == me.owned_tiles {
        return Err("this case is NOT contested: owned_any == owned_mine".into());
    }
    Ok(FrontierCase {
        name: format!(
            "{} tick {} (contested border, {} {})",
            c.game, c.state_tick, me.name, me.small_id
        ),
        player_id: me.small_id.to_string(),
        player_name: me.name.clone(),
        claim_tick: c.claim_tick,
        priority_tick: c.state_tick,
        cursor: 0,
        // The fixed native tip visits N,S,W,E (b14a77d).
        order: ORDER_NSWE,
        border: me.border_order.clone(),
        owned_any,
        owned_mine: me.owned_tiles.clone(),
        expected_order: c.expected_claims.clone(),
        expected_priorities: Vec::new(),
        other_engine_order: Vec::new(),
        expected_units: Vec::new(),
    })
}

/// The tiles owned by somebody ELSE than `sid` (the rival tiles a contested
/// border has next to it). `owned_any` minus `owned_mine`.
pub fn rival_tiles(case: &FrontierCase) -> Vec<u32> {
    let mine: std::collections::HashSet<u32> = case.owned_mine.iter().copied().collect();
    case.owned_any.iter().copied().filter(|t| !mine.contains(t)).collect()
}

/// Border tiles that have a rival-owned neighbour: the definition of a
/// contested frontier.
pub fn contested_border_tiles(case: &FrontierCase, map: &MapPlane) -> Vec<(u32, Vec<u32>)> {
    let mine: std::collections::HashSet<u32> = case.owned_mine.iter().copied().collect();
    let mut out = Vec::new();
    for &bt in &case.border {
        let mut nbuf = [0u32; 4];
        let n = neighbors4(case.order, bt, map.width, map.height, &mut nbuf);
        let rivals: Vec<u32> = (0..n as usize)
            .map(|i| nbuf[i])
            .filter(|nb| case.owned_any.contains(nb) && !mine.contains(nb))
            .collect();
        if !rivals.is_empty() {
            out.push((bt, rivals));
        }
    }
    out
}

/// Rebuild the enqueue list under an arbitrary ELIGIBILITY set.
///
/// `frontier_enqueue` uses `owned_any`; running this with `owned_mine` instead
/// is exactly the conflation the earlier cases could not see, so the effect is
/// measurable (candidate count and candidate order) rather than argued.
pub fn enqueue_under(
    elig: &[u32],
    mine: &[u32],
    border: &[u32],
    terrain: &[u8],
    width: u32,
    height: u32,
    order: u32,
    tick: u32,
    cursor: u32,
) -> (Vec<u32>, Vec<f32>) {
    let mut pr = Prng::new(SEED);
    for _ in 0..cursor {
        pr.next_u32();
    }
    let mut tiles = Vec::new();
    let mut prios = Vec::new();
    for &bt in border {
        let mut nbuf = [0u32; 4];
        let n = neighbors4(order, bt, width, height, &mut nbuf);
        for i in 0..n as usize {
            let nb = nbuf[i];
            if terrain[nb as usize] & 0x80 == 0 || owned_contains(elig, nb) {
                continue;
            }
            let r = pr.next_int(0, 7);
            let mut ibuf = [0u32; 4];
            let inner = neighbors4(order, nb, width, height, &mut ibuf);
            let mut k = 0u32;
            for j in 0..inner as usize {
                if owned_contains(mine, ibuf[j]) {
                    k += 1;
                }
            }
            let mag2 = mag2_from_terrain(terrain[nb as usize]);
            tiles.push(nb);
            prios.push(priority_f32(r, k, mag2, tick));
        }
    }
    (tiles, prios)
}

// ---------------------------------------------------------------------------
// Part 2: the composed tick over a bounded window
// ---------------------------------------------------------------------------

/// FNV-1a 64 (the fixed contract - do not substitute).
pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

pub fn fnv1a_u16_le(plane: &[u16]) -> u64 {
    let mut h = FNV_OFFSET;
    for w in plane {
        let b = w.to_le_bytes();
        h = (h ^ b[0] as u64).wrapping_mul(FNV_PRIME);
        h = (h ^ b[1] as u64).wrapping_mul(FNV_PRIME);
    }
    h
}

pub fn hex64(h: u64) -> String {
    format!("0x{h:016x}")
}

/// The engine's `state` plane (`map.rs:31,229-247`): owner small id per tile,
/// 0 elsewhere, plus a fallout bit nothing sets in the early regime. Rebuilt
/// here from per-player `ownedTiles` + `smallId`.
pub fn state_plane_from_players(width: u32, height: u32, players: &[(u32, Vec<u32>)]) -> Vec<u16> {
    let mut plane = vec![0u16; (width * height) as usize];
    for (sid, tiles) in players {
        for t in tiles {
            plane[*t as usize] = *sid as u16;
        }
    }
    plane
}

#[derive(Clone, Debug)]
pub struct TickBlock {
    pub tick: u32,
    pub owner_sid: u32,
    pub border: Vec<u32>,
    pub owned_any: Vec<u32>,
    pub owned_mine: Vec<u32>,
    pub budget: u32,
    pub expected_claims: Vec<u32>,
    pub expected_hash: u64,
}

#[derive(Clone, Debug)]
pub struct TickStep {
    pub tick: u32,
    pub expected_hash: u64,
    pub blocks: Vec<TickBlock>,
}

#[derive(Clone, Debug)]
pub struct TickWindow {
    pub game: String,
    pub tick0: u32,
    pub players0: Vec<(u32, Vec<u32>)>,
    pub steps: Vec<TickStep>,
}

impl TickWindow {
    pub fn plane0(&self, width: u32, height: u32) -> Vec<u16> {
        state_plane_from_players(width, height, &self.players0)
    }
}

pub fn load_tick_window(path: &Path) -> Result<TickWindow, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
    let nums = |x: Option<&serde_json::Value>| -> Vec<u32> {
        x.and_then(|x| x.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect())
            .unwrap_or_default()
    };
    let parse_hash = |s: Option<&str>| -> u64 {
        u64::from_str_radix(s.unwrap_or("0x0").trim_start_matches("0x"), 16).unwrap_or(0)
    };
    let players0 = v
        .get("players0")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .map(|p| {
                    (
                        p.get("smallId").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
                        nums(p.get("ownedTiles")),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    let mut steps = Vec::new();
    for s in v.get("steps").and_then(|x| x.as_array()).ok_or("no steps")? {
        let tick = s.get("tick").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
        let expected_hash = parse_hash(s.get("expectedHash").and_then(|x| x.as_str()));
        let mut blocks = Vec::new();
        let empty_blocks: Vec<serde_json::Value> = Vec::new();
        let block_arr = s.get("blocks").and_then(|x| x.as_array()).unwrap_or(&empty_blocks);
        for b in block_arr {
            blocks.push(TickBlock {
                tick: b.get("tick").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
                owner_sid: b.get("ownerSid").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
                border: nums(b.get("border")),
                owned_any: nums(b.get("ownedAny")),
                owned_mine: nums(b.get("ownedMine")),
                budget: b.get("budget").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
                expected_claims: nums(b.get("expectedClaimSet")),
                expected_hash: parse_hash(b.get("expectedHash").and_then(|x| x.as_str())),
            });
        }
        steps.push(TickStep { tick, expected_hash, blocks });
    }
    Ok(TickWindow {
        game: v.get("game").and_then(|x| x.as_str()).unwrap_or("").to_string(),
        tick0: v.get("tick0").and_then(|x| x.as_u64()).unwrap_or(0) as u32,
        players0,
        steps,
    })
}

pub fn tick_window_path() -> PathBuf {
    case_path("b002-t300-331.ticksteps.json")
}

/// The frontier's CLAIM ORDER under an explicit eligibility set - the host
/// twin of the `compose_tick` kernel's pop loop, used to (a) check the device
/// and (b) *measure* what conflating the two sets does.
pub fn frontier_claims_with(
    border: &[u32],
    elig: &[u32],
    owned_mine: &[u32],
    terrain: &[u8],
    width: u32,
    height: u32,
    order: u32,
    tick: u32,
    cursor: u32,
    budget: u32,
) -> (Vec<u32>, u32) {
    let mut heap = Heap::new();
    let n_enq = frontier_enqueue(
        border, elig, owned_mine, terrain, width, height, order, tick, cursor, &mut heap,
    );
    let mut claims: Vec<u32> = Vec::new();
    while (claims.len() as u32) < budget {
        match heap.dequeue() {
            Some((t, _pr)) => {
                if owned_contains(elig, t) || claims.contains(&t) {
                    continue;
                }
                claims.push(t);
            }
            None => break,
        }
    }
    (claims, n_enq)
}

/// `frontier_claims_with` for a case: the shipped configuration (eligibility =
/// `owned_any`, the owner count in the priority = `owned_mine`).
pub fn frontier_claims(case: &FrontierCase, map: &MapPlane, budget: u32) -> (Vec<u32>, u32) {
    frontier_claims_with(
        &case.border,
        &case.owned_any,
        &case.owned_mine,
        &map.terrain,
        map.width,
        map.height,
        case.order,
        case.priority_tick,
        case.cursor,
        budget,
    )
}

/// The same step with the two sets CONFLATED (`owned_mine` used for both the
/// eligibility filter and the owner count): the bug the earlier cases were
/// blind to.
pub fn frontier_claims_conflated(case: &FrontierCase, map: &MapPlane, budget: u32) -> (Vec<u32>, u32) {
    frontier_claims_with(
        &case.border,
        &case.owned_mine,
        &case.owned_mine,
        &map.terrain,
        map.width,
        map.height,
        case.order,
        case.priority_tick,
        case.cursor,
        budget,
    )
}

/// The frontier with IN-TICK RE-ENQUEUE: after a tile is claimed, its four
/// neighbours are offered to the same heap straight away (the shared PRNG
/// stream continues, one draw per enqueued candidate), and the pop loop runs
/// until `budget` claims.
///
/// This is the candidate explanation for the contested case's claim order: the
/// engine's list contains a tile that is the middle of a 3-run whose two outer
/// tiles the single-pass model already claims, i.e. a tile only reachable if
/// the frontier is refreshed *within* the tick.
pub fn frontier_claims_incremental(
    border: &[u32],
    owned_any: &[u32],
    owned_mine: &[u32],
    terrain: &[u8],
    width: u32,
    height: u32,
    order: u32,
    tick: u32,
    cursor: u32,
    budget: u32,
) -> (Vec<u32>, u32) {
    let mut pr = Prng::new(SEED);
    for _ in 0..cursor {
        pr.next_u32();
    }
    let mut heap = Heap::new();
    let mut owned: Vec<u32> = owned_mine.to_vec();
    let mut n_enq = 0u32;

    let offer = |heap: &mut Heap, pr: &mut Prng, owned: &[u32], nb: u32, n_enq: &mut u32| {
        if terrain[nb as usize] & 0x80 == 0 || owned_contains(owned_any, nb) {
            return;
        }
        let r = pr.next_int(0, 7);
        let mut ibuf = [0u32; 4];
        let inner = neighbors4(order, nb, width, height, &mut ibuf);
        let mut k = 0u32;
        for j in 0..inner as usize {
            if owned_contains(owned, ibuf[j as usize]) {
                k += 1;
            }
        }
        let mag2 = mag2_from_terrain(terrain[nb as usize]);
        heap.enqueue(nb, priority_f32(r, k, mag2, tick));
        *n_enq += 1;
    };

    for &bt in border {
        let mut nbuf = [0u32; 4];
        let n = neighbors4(order, bt, width, height, &mut nbuf);
        for i in 0..n as usize {
            let nb = nbuf[i];
            offer(&mut heap, &mut pr, &owned, nb, &mut n_enq);
        }
    }

    let mut claims: Vec<u32> = Vec::new();
    while (claims.len() as u32) < budget {
        match heap.dequeue() {
            Some((t, _pr)) => {
                if owned_contains(owned_any, t) || claims.contains(&t) {
                    continue;
                }
                claims.push(t);
                owned.push(t);
                let mut nbuf = [0u32; 4];
                let n = neighbors4(order, t, width, height, &mut nbuf);
                for i in 0..n as usize {
                    let nb = nbuf[i];
                    offer(&mut heap, &mut pr, &owned, nb, &mut n_enq);
                }
            }
            None => break,
        }
    }
    (claims, n_enq)
}

/// The engine's own tick loop, as far as it can run without troops/gold:
/// `AttackExecution::tick` (`rust/engine/src/execution/attack.rs:239-325`).
///
/// Differences from `frontier_claims_incremental` above, all read off the tip:
///   * ONE extra draw per tick for the budget: `border_size +
///     self.random.next_int(0, 5)` (`attack.rs:254`), taken before the pop loop;
///   * `add_neighbors` is called for the tile being claimed, BEFORE
///     `game.conquer` (`attack.rs:292`), so the claimed tile is *not yet* the
///     attacker's when its neighbours' owner-counts are computed;
///   * a pop is skipped when the tile is no longer terra nullius **or** has no
///     neighbour owned by the attacker (`attack.rs:284`), and a skipped pop
///     consumes nothing;
///   * an empty heap ends the tick (`refresh_to_conquer` + `retreat`,
///     `attack.rs:264-270`);
///   * the budget is a FLOAT (`num_tiles_per_tick -= tiles_used`) whose per-tile
///     cost comes from `attack_logic_at_tile` (troops/terrain), so the claim
///     count is not the integer budget.
pub struct EngineTickRun {
    pub claims: Vec<u32>,
    pub n_enq: u32,
    pub draws: u32,
}

#[allow(clippy::too_many_arguments)]
pub fn engine_tick_model(
    border: &[u32],
    owned_any: &[u32],
    owned_mine: &[u32],
    terrain: &[u8],
    width: u32,
    height: u32,
    order: u32,
    tick: u32,
    cursor: u32,
    budget_claims: u32,
    consume_budget_draw: bool,
) -> EngineTickRun {
    let mut pr = Prng::new(SEED);
    let mut draws = 0u32;
    let mut skip = 0u32;
    while skip < cursor {
        pr.next_u32();
        draws += 1;
        skip += 1;
    }
    if consume_budget_draw {
        pr.next_int(0, 5);
        draws += 1;
    }

    let mut live: Vec<u32> = owned_any.to_vec();
    let mut mine: Vec<u32> = owned_mine.to_vec();
    let mut heap = Heap::new();
    let mut n_enq = 0u32;

    // The frontier as it stands at the start of the tick (`refresh_to_conquer`
    // over the border set, `attack.rs:1265-1272`).
    for &bt in border {
        let mut nbuf = [0u32; 4];
        let n = neighbors4(order, bt, width, height, &mut nbuf);
        for i in 0..n as usize {
            let nb = nbuf[i];
            if terrain[nb as usize] & 0x80 == 0 || owned_contains(&live, nb) {
                continue;
            }
            let r = pr.next_int(0, 7);
            draws += 1;
            let mut ibuf = [0u32; 4];
            let inner = neighbors4(order, nb, width, height, &mut ibuf);
            let mut k = 0u32;
            for j in 0..inner as usize {
                if owned_contains(&mine, ibuf[j as usize]) {
                    k += 1;
                }
            }
            heap.enqueue(nb, priority_f32(r, k, mag2_from_terrain(terrain[nb as usize]), tick));
            n_enq += 1;
        }
    }

    let mut claims: Vec<u32> = Vec::new();
    while (claims.len() as u32) < budget_claims {
        let Some((t, _p)) = heap.dequeue() else {
            break; // refresh_to_conquer + retreat: the tick ends
        };
        // `attack.rs:278-284`
        if owned_contains(&live, t) {
            continue;
        }
        let mut nbuf = [0u32; 4];
        let n = neighbors4(order, t, width, height, &mut nbuf);
        let mut on_border = false;
        for i in 0..n as usize {
            if owned_contains(&mine, nbuf[i]) {
                on_border = true;
            }
        }
        if !on_border {
            continue;
        }
        if terrain[t as usize] & 0x80 == 0 {
            continue;
        }
        // `add_neighbors(game, tile_to_conquer, tick)` - BEFORE the conquer.
        for i in 0..n as usize {
            let nb = nbuf[i];
            if terrain[nb as usize] & 0x80 == 0 || owned_contains(&live, nb) {
                continue;
            }
            let r = pr.next_int(0, 7);
            draws += 1;
            let mut ibuf = [0u32; 4];
            let inner = neighbors4(order, nb, width, height, &mut ibuf);
            let mut k = 0u32;
            for j in 0..inner as usize {
                if owned_contains(&mine, ibuf[j as usize]) {
                    k += 1;
                }
            }
            heap.enqueue(nb, priority_f32(r, k, mag2_from_terrain(terrain[nb as usize]), tick));
            n_enq += 1;
        }
        claims.push(t);
        live.push(t);
        mine.push(t);
    }
    EngineTickRun { claims, n_enq, draws }
}

// ---------------------------------------------------------------------------
// The engine's own loop, on the two recorded grounds
// ---------------------------------------------------------------------------

/// One attack run through the engine's loop (`AttackExecution::init` +
/// `AttackExecution::tick`), with the state carried across ticks.
#[derive(Clone, Debug, Default)]
pub struct EngineRun {
    pub claims: Vec<u32>,
    pub claim_pri_bits: Vec<u32>,
    /// Candidates enqueued over the whole run (one `next_int(0, 7)` each).
    pub n_enq: u32,
    /// `Prng::calls`: every `next_u32` the attack's stream has made,
    /// warm-ups included (12 of them, `prng.rs:30-32`).
    pub draws: u32,
    /// Ticks in which the heap ran dry (`refresh_to_conquer` + `retreat`).
    pub refreshes: u32,
    /// The carried state after the run, `STATE_WORDS` layout - the device
    /// publishes the same words, so heap/PRNG agreement is checked, not assumed.
    pub state: Vec<u32>,
}

/// The ownership plane the contested case's tick starts from: every player's
/// tiles from the preserved dump at `state_tick` (`claim_tick - 1`).
pub fn contested_plane(cc: &ContestedCase, width: u32, height: u32) -> Vec<u16> {
    let players: Vec<(u32, Vec<u32>)> = cc
        .players
        .iter()
        .map(|p| (p.small_id, p.owned_tiles.clone()))
        .collect();
    state_plane_from_players(width, height, &players)
}

/// PART 1 under the engine's loop: `init` at `priority_tick - 1` (the attack's
/// first frontier and first draw), then ONE pop tick at `priority_tick`.
///
/// `priority_tick` is the step parameter the engine passes to `tick()`
/// (`game.rs:3659`), and - with the reference dump's `attacks` field - the
/// attack's birth tick is one step earlier, which is why the init refresh is
/// stamped `priority_tick - 1` and the in-tick `add_neighbors` `priority_tick`.
pub fn contested_engine_run(
    case: &FrontierCase,
    plane: &[u16],
    map: &MapPlane,
    owner_sid: u16,
    budget: u32,
    consume_budget_draw: bool,
    in_tick_add_neighbors: bool,
) -> EngineRun {
    let mut a = EngineAttack::init(
        &case.border,
        plane,
        owner_sid,
        &map.terrain,
        map.width,
        map.height,
        case.order,
        case.priority_tick - 1,
    );
    a.tick(
        plane,
        owner_sid,
        &case.border,
        &map.terrain,
        map.width,
        map.height,
        case.order,
        case.priority_tick,
        budget,
        consume_budget_draw,
        in_tick_add_neighbors,
    );
    let mut state = [0u32; STATE_WORDS];
    a.state_words(&mut state);
    EngineRun {
        claims: a.claims().to_vec(),
        claim_pri_bits: a.claim_pri_bits[..a.claimed as usize].to_vec(),
        n_enq: a.n_enq,
        draws: a.pr.calls,
        refreshes: a.refreshes,
        state: state.to_vec(),
    }
}

/// One tick of the composed window, as the engine loop produced it.
#[derive(Clone, Debug)]
pub struct WindowTick {
    pub tick: u32,
    pub budget: u32,
    pub claims: Vec<u32>,
    pub expected: Vec<u32>,
    /// Same tiles, any order.
    pub set_ok: bool,
    /// Same tiles in the engine's recorded order.
    pub order_ok: bool,
    /// The whole-plane FNV-1a-64 hash equals the engine's.
    pub hash_ok: bool,
}

#[derive(Clone, Debug, Default)]
pub struct EngineWindowRun {
    pub ticks: Vec<WindowTick>,
    /// Ticks whose claims AND per-tick hash match the engine.
    pub matched: u32,
    /// First tick that does not.
    pub first_bad: Option<u32>,
    pub total_claims: u32,
    pub draws: u32,
    pub refreshes: u32,
    /// The carried state after the last tick, `STATE_WORDS` layout.
    pub final_state: Vec<u32>,
}

/// Hash a window from a claim LIST rather than from a plane: the device kernel
/// returns the engine-loop claim order, and this splits it per tick by the
/// record's per-block budget and checks the FNV-1a-64 hash of the plane it
/// implies. Returns (ticks matching the engine hash, first mismatching tick,
/// the matching ticks) - counted evidence, not "it looks right".
pub fn window_hashes_from_claims(
    map: &MapPlane,
    w: &TickWindow,
    claims: &[u32],
) -> (u32, Option<u32>, Vec<u32>) {
    let mut plane = w.plane0(map.width, map.height);
    let mut at = 0usize;
    let mut ok = 0u32;
    let mut first_bad = None;
    let mut matched = Vec::new();
    for step in &w.steps {
        for b in &step.blocks {
            let n = b.budget as usize;
            let end = (at + n).min(claims.len());
            for t in &claims[at..end] {
                if (*t as usize) < plane.len() {
                    plane[*t as usize] = b.owner_sid as u16;
                }
            }
            at += n;
        }
        if fnv1a_u16_le(&plane) == step.expected_hash {
            ok += 1;
            matched.push(step.tick);
        } else if first_bad.is_none() {
            first_bad = Some(step.tick);
        }
    }
    (ok, first_bad, matched)
}

/// PART 2 under the engine's loop: one attack for the claiming player, its
/// `to_conquer` and its `random` stream carried from tick to tick
/// (`attack.rs:17,21`), the per-tick claim budget taken from the record
/// (`attack.rs:239-255` - the float budget is a different worker's slice; see
/// the README's measured/assumed split), the ONE budget draw per tick consumed
/// before the pop loop, `add_neighbors(tile_to_conquer, tick)` inside the loop
/// (`attack.rs:292`) and the pop-skip guards (`attack.rs:284`).
///
/// The attack is born one step before its first claims: `init` refresh at
/// `first_claim_tick - 2`, first pop tick `first_claim_tick - 1`. The record
/// agrees: the reference dump's `attacks` list is empty at tick 317 and holds
/// owner 2's land attack at tick 318, whose claims land in the 319 record.
pub fn window_engine_run(
    map: &MapPlane,
    w: &TickWindow,
    consume_budget_draw: bool,
    in_tick_add_neighbors: bool,
) -> EngineWindowRun {
    let mut out = EngineWindowRun::default();
    let mut plane = w.plane0(map.width, map.height);
    let mut attacks: Vec<(u32, EngineAttack)> = Vec::new();

    for step in &w.steps {
        let tick = step.tick;
        let pop_tick = tick - 1; // the step parameter that produces this record
        let mut claims_this_tick: Vec<u32> = Vec::new();
        let mut expected: Vec<u32> = Vec::new();
        let mut budget = 0u32;

        // A live attack still spends its budget draw and pops on a tick where
        // no claim is recorded, so the stream stays where the engine left it.
        for (sid, a) in attacks.iter_mut() {
            if step.blocks.iter().all(|b| b.owner_sid != *sid) {
                a.tick(
                    &plane,
                    *sid as u16,
                    &[],
                    &map.terrain,
                    map.width,
                    map.height,
                    ORDER_NSWE,
                    pop_tick,
                    0,
                    consume_budget_draw,
                    in_tick_add_neighbors,
                );
            }
        }

        for b in &step.blocks {
            let owner = b.owner_sid as u16;
            budget = b.budget;
            expected = b.expected_claims.clone();
            let refresh_tick = pop_tick - 1;
            let idx = attacks.iter().position(|(s, _)| *s == b.owner_sid);
            let a = match idx {
                Some(i) => &mut attacks[i].1,
                None => {
                    // The attack's birth: `init` one step before its first pop.
                    attacks.push((
                        b.owner_sid,
                        EngineAttack::init(
                            &b.border,
                            &plane,
                            owner,
                            &map.terrain,
                            map.width,
                            map.height,
                            ORDER_NSWE,
                            refresh_tick,
                        ),
                    ));
                    &mut attacks.last_mut().unwrap().1
                }
            };
            let before = a.claimed;
            a.tick(
                &plane,
                owner,
                &b.border,
                &map.terrain,
                map.width,
                map.height,
                ORDER_NSWE,
                pop_tick,
                b.budget,
                consume_budget_draw,
                in_tick_add_neighbors,
            );
            claims_this_tick.extend_from_slice(&a.claims()[before as usize..a.claimed as usize]);
            for t in &claims_this_tick {
                plane[*t as usize] = owner;
            }
        }

        let set_ok = {
            let mut c = claims_this_tick.clone();
            let mut e = expected.clone();
            c.sort_unstable();
            e.sort_unstable();
            c == e
        };
        let order_ok = claims_this_tick == expected;
        let hash_ok = fnv1a_u16_le(&plane) == step.expected_hash;
        let ok = if step.blocks.is_empty() {
            hash_ok
        } else {
            set_ok && order_ok && hash_ok
        };
        if ok {
            out.matched += 1;
        } else if out.first_bad.is_none() {
            out.first_bad = Some(tick);
        }
        out.ticks.push(WindowTick {
            tick,
            budget,
            claims: claims_this_tick,
            expected,
            set_ok,
            order_ok,
            hash_ok,
        });
    }
    out.total_claims = out.ticks.iter().map(|t| t.claims.len() as u32).sum();
    out.draws = attacks.first().map(|(_, a)| a.pr.calls).unwrap_or(0);
    out.refreshes = attacks.first().map(|(_, a)| a.refreshes).unwrap_or(0);
    out.final_state = match attacks.first() {
        Some((_, a)) => {
            let mut w = [0u32; STATE_WORDS];
            a.state_words(&mut w);
            w.to_vec()
        }
        None => vec![0u32; STATE_WORDS],
    };
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core_impl::{Heap, ORDER_WENS};
    use crate::{enqueue_trace, load_map, pangaea_map_dir, recorded_cases};
    /// The device module in `src/main.rs` carries a literal copy of this crate's
    /// `core_impl.rs` (cuda-oxide's `#[cuda_module]` macro cannot see through an
    /// `include!` inside the module). A copy that silently drifts turns every
    /// GPU-vs-CPU agreement into a comparison of two different programs, so the
    /// copy is checked mechanically: the whole file must appear verbatim inside
    /// `main.rs`. Fix a failure with `bash sync_device_core.sh`.
    #[test]
    fn device_core_is_a_verbatim_copy() {
        let core = include_str!("core_impl.rs").trim_end();
        let dev = include_str!("main.rs");
        assert!(
            dev.contains(core),
            "the device copy inside src/main.rs diverged from src/core_impl.rs - \
             run `bash sync_device_core.sh`"
        );
    }

    /// The heap's documented tie-breaks (flat_heap.rs's own test vector).
    #[test]
    fn heap_tie_break_matches_the_engine() {
        let mut h = Heap::new();
        for tile in [10u32, 20, 30, 40] {
            h.enqueue(tile, 1.0);
        }
        let mut out = Vec::new();
        while let Some((t, _)) = h.dequeue() {
            out.push(t);
        }
        assert_eq!(out, vec![10, 40, 30, 20]);
    }

    /// The f32 priority must equal the engine's f64 evaluation on every enqueue
    /// of both recorded cases - measured, not assumed.
    #[test]
    fn priority_f32_equals_engine_f64_on_the_recorded_cases() {
        let map = load_map(&pangaea_map_dir()).expect("map");
        let (c1, c2) = recorded_cases(&map).expect("cases");
        for case in [&c1, &c2] {
            let trace = enqueue_trace(case, &map);
            assert!(!trace.is_empty(), "no enqueues for {}", case.name);
            for (tile, r, k, mag2, pf32, pf64) in trace {
                assert_eq!(
                    pf32, pf64,
                    "{}: tile {tile} r={r} k={k} mag2={mag2}: f32 {pf32} vs f64 {pf64}",
                    case.name
                );
            }
        }
    }

    /// PART 1's finding, locked in: the contested case really is contested, and
    /// on it the single-pass frontier does NOT reproduce the engine.
    #[test]
    fn contested_case_separates_the_two_owned_sets() {
        let map = load_map(&pangaea_map_dir()).unwrap();
        let cc = load_contested_case(&contested_case_path()).unwrap();
        let case = contested_frontier_case(&cc).unwrap();
        assert!(
            case.owned_any.len() > case.owned_mine.len(),
            "owned_any {} vs owned_mine {}",
            case.owned_any.len(),
            case.owned_mine.len()
        );
        assert!(
            !contested_border_tiles(&case, &map).is_empty(),
            "the border must have at least one tile with a RIVAL neighbour"
        );
        let (c_any, n_any) = frontier_claims(&case, &map, 64);
        let (c_con, n_con) = frontier_claims_conflated(&case, &map, 64);
        assert_ne!(
            n_any, n_con,
            "conflating owned_any with owned_mine must change the enqueue count here"
        );
        assert_ne!(c_any, c_con, "and must change the claim order");
        // The kernel's model, on the recorded prefix.
        let (c_single, _) = frontier_claims(&case, &map, case.expected_order.len() as u32);
        let i = c_single
            .iter()
            .zip(case.expected_order.iter())
            .position(|(a, b)| a != b)
            .expect("the single-pass model must NOT match the engine here any more");
        assert_eq!(i, 23, "first divergence index (engine inserts tile 838544)");
        assert!(case.expected_order.contains(&838544));
        assert!(!c_single.contains(&838544));
    }

    /// PART 2's ground truth, locked in: the recorded claim sets rebuild every
    /// recorded per-tick hash under the fixed FNV-1a-64 contract, so the window
    /// file is complete and the hash is the one the composed kernel must hit.
    #[test]
    fn tick_window_reconstructs_under_the_fixed_hash() {
        let w = load_tick_window(&tick_window_path()).unwrap();
        let mut p = w.plane0(1000, 1000);
        assert_eq!(p.len(), 1_000_000);
        for step in &w.steps {
            for b in &step.blocks {
                for t in &b.expected_claims {
                    p[*t as usize] = b.owner_sid as u16;
                }
            }
            assert_eq!(fnv1a_u16_le(&p), step.expected_hash, "tick {}", step.tick);
        }
        assert_eq!(
            w.steps.iter().filter(|s| !s.blocks.is_empty()).count(),
            13,
            "13 claim ticks (319..331), the rest are no-op ticks"
        );
    }

    /// PART 1, now under the ENGINE's own loop (persistent `to_conquer`,
    /// in-tick `add_neighbors`, pop-skip guards, one budget draw per tick):
    /// all 27 of the contested case's claims, in order, with the second-ring
    /// tile 838544 at index 23. The single-pass control is printed next to it.
    #[test]
    fn contested_case_engine_loop_reproduces_the_engine() {
        let map = load_map(&pangaea_map_dir()).unwrap();
        let cc = load_contested_case(&contested_case_path()).unwrap();
        let case = contested_frontier_case(&cc).unwrap();
        let plane = contested_plane(&cc, map.width, map.height);
        let budget = cc.expected_claims.len() as u32;
        let run = contested_engine_run(
            &case,
            &plane,
            &map,
            cc.owner_sid as u16,
            budget,
            true,
            true,
        );
        assert_eq!(
            run.claims.len(),
            cc.expected_claims.len(),
            "engine-loop claim count"
        );
        let first = first_diff(&run.claims, &cc.expected_claims);
        assert_eq!(
            first, None,
            "engine-loop claim order must equal the engine's 27; first difference at {first:?}\n\
             gpu-model {}\nengine    {}",
            fmt_u32s(&run.claims),
            fmt_u32s(&cc.expected_claims)
        );
        assert_eq!(
            run.claims.iter().position(|&t| t == 838544),
            Some(23),
            "the second-ring tile 838544 must land at claim index 23"
        );
        // The control: the shipped single-pass frontier (what the crate did
        // before) cannot pop 838544 at all and diverges at index 23.
        let (single, n_single) = frontier_claims(&case, &map, budget);
        assert_eq!(n_single, 211);
        assert_eq!(first_diff(&single, &cc.expected_claims), Some(23));
        assert!(!single.contains(&838544));
        assert!(run.n_enq > 211, "in-tick re-enqueue must enqueue more");
        println!(
            "contested: engine loop {} claims, enqueued {}, draws {} (init {} + pop {} + in-tick {}), refreshes {}",
            run.claims.len(),
            run.n_enq,
            run.draws,
            case.border.len(),
            budget,
            run.n_enq - 211,
            run.refreshes
        );
    }

    /// PART 2, now under the ENGINE's own loop with the state carried from tick
    /// to tick. Measured: with the budget draw consumed and in-tick
    /// `add_neighbors` on, EVERY one of the 32 tick hashes matches. The two
    /// controls show each rule is load-bearing (counted, not asserted).
    #[test]
    fn tick_window_engine_loop_reproduces_the_engine() {
        let map = load_map(&pangaea_map_dir()).unwrap();
        let w = load_tick_window(&tick_window_path()).unwrap();
        let run = window_engine_run(&map, &w, true, true);
        assert_eq!(w.steps.len(), 32);
        assert_eq!(run.total_claims, 133, "the window's 13 claim ticks");
        assert_eq!(run.matched, 32, "all 32 per-tick hashes");
        assert_eq!(run.first_bad, None, "first mismatching tick");
        for t in run.ticks.iter().filter(|t| !t.expected.is_empty()) {
            assert!(
                t.set_ok && t.order_ok && t.hash_ok,
                "tick {}: set_ok={} order_ok={} hash_ok={}\nclaims   {}\nexpected {}",
                t.tick,
                t.set_ok,
                t.order_ok,
                t.hash_ok,
                fmt_u32s(&t.claims),
                fmt_u32s(&t.expected)
            );
        }
        // Controls on the same code path.
        let no_in_tick = window_engine_run(&map, &w, true, false);
        assert_eq!(no_in_tick.matched, 20, "in-tick add_neighbors off");
        assert_eq!(no_in_tick.first_bad, Some(320));
        let no_budget_draw = window_engine_run(&map, &w, false, true);
        assert_eq!(no_budget_draw.matched, 20, "budget draw skipped");
        assert_eq!(no_budget_draw.first_bad, Some(320));
        println!(
            "window: engine loop {}/32 ticks, {} claims, {} draws, {} refreshes; \
             controls in_tick=off {}/32 (first bad {:?}), budget draw skipped {}/32 (first bad {:?})",
            run.matched,
            run.total_claims,
            run.draws,
            run.refreshes,
            no_in_tick.matched,
            no_in_tick.first_bad,
            no_budget_draw.matched,
            no_budget_draw.first_bad
        );
    }

    /// The draw account the engine loop implies, locked in: exactly one
    /// `next_int(0, 5)` per tick plus one `next_int(0, 7)` per enqueued
    /// candidate, on top of the stream's 12 warm-ups (`prng.rs:30-32`).
    #[test]
    fn engine_loop_draw_accounting_is_exact() {
        let map = load_map(&pangaea_map_dir()).unwrap();
        let cc = load_contested_case(&contested_case_path()).unwrap();
        let case = contested_frontier_case(&cc).unwrap();
        let plane = contested_plane(&cc, map.width, map.height);
        let budget = cc.expected_claims.len() as u32;
        let run = contested_engine_run(
            &case,
            &plane,
            &map,
            cc.owner_sid as u16,
            budget,
            true,
            true,
        );
        // The init refresh's draw is the first in-tick add_neighbors count:
        // 211 candidates from the border refresh + 45 from the in-tick pass.
        assert_eq!(run.n_enq, 256);
        assert_eq!(
            run.draws,
            12 + 1 + run.n_enq,
            "12 warm-ups + 1 budget draw + one per enqueued candidate"
        );
    }
}
