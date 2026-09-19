// ===========================================================================
// `ofcuda_cluster` - GPU (cuda-oxide) side of the CLUSTER CAPTURE step.
//
// The step is `player_clusters::maybe_remove_clusters` (Rust engine) /
// `PlayerExecution.removeClusters` (TS authority): after a tick's territory
// change, work out which 8-connected components of the victim's border set are
// surrounded, and hand each one to the neighbour with the largest incoming
// land attack (else the enemy with the most bordering tiles).
//
// Kernel shape:
//   * `flood_clusters`  - one thread per **border index**. Thread `j` floods the
//     component containing `border_tiles[j]` with the engine's exact DFS
//     (mark-on-discovery, N,S,W,E only via `neighbors8` for this phase) and
//     writes the discovery order into slot `j` of a flattened per-thread slot
//     buffer. It writes a non-zero length only when `j` is the smallest border
//     index in its component - which is exactly the index the engine's
//     `for start in border.iter()` loop would have started that cluster at. That
//     keeps the engine's cluster list (and its order) bit-exact while the flood
//     itself runs fully parallel.
//   * `decide_remove`   - one thread. Everything after the flood is inherently
//     serial and state-dependent: the largest-cluster test runs on the input
//     plane, then each later cluster is tested on the plane *as already
//     modified* by earlier removals (TS re-checks ownership for exactly this
//     reason), and the captor of each cluster is picked by a strictly-greater
//     scan over first-seen neighbour order. It replays `decide_and_remove`,
//     which is the same function the CPU reference calls.
//
// The device core is a **verbatim copy** of `src/core_impl.rs` (cuda-oxide's
// `#[cuda_module]` cannot see through an `include!`), injected by
// `./sync_device_core.sh` and pinned by `lib.rs::device_core_verbatim_check`,
// which this binary runs first: a diverged copy aborts the run.
// ===========================================================================

use std::path::PathBuf;

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{kernel, launch_bounds, thread};
use cuda_host::cuda_module;
use ofcuda_cluster::{
    Case, Check, Prepared, ReportInput, Run, compare_runs, compare_to_engine, decide_cpu,
    device_core_verbatim_check, diagnose, format_report, load_case, load_terrain, pangaea_map_dir,
    prepare,
};

#[cuda_module]
mod kernels {
    // Bring `kernel`, `launch_bounds`, `thread` (imported at the crate root)
    // into this module's scope: cuda-oxide's `#[cuda_module]` does not inject
    // the device attribute macros, so without this the inner `#[kernel]` /
    // `#[launch_bounds]` attributes are unresolved and no launcher is generated.
    use super::*;

    // ===== BEGIN VERBATIM COPY of src/core_impl.rs =====
// ===========================================================================
// Canonical core of the CLUSTER CAPTURE step.
//
// This file is the single implementation shared by the CPU reference
// (`src/lib.rs`, `mod core_impl`) and the CUDA device module (`src/main.rs`,
//  `mod kernels` - copied VERBATIM between the BEGIN/END VERBATIM COPY markers,
//  re-copied by `sync_device_core.sh`, and pinned by the
//  `device_core_is_a_verbatim_copy` test in `lib.rs`). So the flood, the
//  surround tests, the captor selection and the conquest have exactly one
//  implementation that can disagree with the engine - not one per side.
//
//  Deliberately `#[no_std]`-clean: fixed-size arrays, no allocator, no
//  HashMap/HashSet, no `Vec`, no strings.
//
//  1:1 ports, with citations into `openfront-ai/rust/engine`:
//    * `maybe_remove_clusters`            - src/execution/player_clusters.rs:349-420
//    * `calculate_clusters`               - src/execution/player_clusters.rs:121-141
//    * `flood_border_cluster`             - src/execution/player_clusters.rs:93-119
//    * `surrounded_by_same_enemy`         - src/execution/player_clusters.rs:143-208
//    * `is_surrounded`                    - src/execution/player_clusters.rs:210-256
//    * `get_capturing_player`             - src/execution/player_clusters.rs:258-298
//    * `flood_owned`                      - src/execution/player_clusters.rs:300-323
//    * `remove_cluster`                   - src/execution/player_clusters.rs:325-347
//    * `GameMap::for_each_neighbor8`      - src/map.rs:251-280   (dx-major 8-neighbour)
//    * `GameMap::for_each_neighbor_nswe`  - src/map.rs:352-367   (N,S,W,E)
//    * `GameMap::is_ocean_shore`          - src/map.rs:282-301
//    * `GameMap::is_on_edge_of_map`       - src/map.rs:303-307
//    * `GameMap::is_land/is_ocean/is_shore` - src/map.rs:128-145 (bits 7/5/6)
//    * `Game::conquer_one`                - src/game.rs:1233-1273 (land guard, owner write)
//    * `Game::conquer_player`             - src/game.rs:1163-1220 (no owner-plane effect)
//    * TS authority: openfront/src/core/execution/PlayerExecution.ts:99-155
//      (tick gate), :327-355 (calculateClusters), :376-416 (floodFillWithGen),
//      :211-244 (isSurrounded), :157-209 (surroundedBySamePlayer),
//      :246-280 (removeCluster), :282-325 (getCapturingPlayer).

pub type TileRef = u32;

/// `PlayerExecution.ts:19` `ticksPerClusterCalc = 20`; engine
/// `player_clusters.rs:8`.
pub const TICKS_PER_CLUSTER_CALC: u32 = 20;

/// GPU-path caps. Both recorded cases have |border| = 158 and 659, and a
/// cluster can never exceed the border it is flooded from.
pub const MAX_BORDER: usize = 1024;
pub const MAX_CLUSTER: usize = 1024;
pub const MAX_STACK: usize = 1024;
pub const VIS_WORDS: usize = MAX_BORDER / 32;
pub const MAX_NEIGH: usize = 16;

pub const NONE: u32 = u32::MAX;

// Terrain format bits - src/map.rs:8-10.
const IS_LAND_BIT: u8 = 7;
const OCEAN_BIT: u8 = 5;
const SHORELINE_BIT: u8 = 6;

/// Caller-supplied step parameters (the victim's own player record plus the
/// tick, exactly the fields `maybe_remove_clusters` reads).
#[derive(Clone, Copy, Debug, Default)]
pub struct Params {
    pub width: u32,
    pub height: u32,
    pub tick: u32,
    pub victim: u16,
    /// `simple_hash(&p.id)` - `player_clusters.rs:361`; engine stores it as
    /// `Player::id_hash` (`game.rs:934,945`).
    pub id_hash: i32,
    pub tiles_owned: i32,
    pub alive: bool,
    pub last_cluster_calc: u32,
    pub last_tile_change: u32,
    pub border_len: u32,
}

/// What the step did - for reporting and for the host's own cross-checks.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepOut {
    /// The tick gate + `last_change >= last_calc` gate passed.
    pub fired: bool,
    /// Non-empty clusters found by `calculate_clusters`.
    pub cluster_count: u32,
    pub largest_index: u32,
    pub largest_size: u32,
    /// Clusters that reached `remove_cluster` (largest-surrounded or
    /// is_surrounded).
    pub removed_clusters: u32,
    /// `(tile, new_owner)` writes applied, in application order.
    pub changes: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BBox {
    pub min_x: u32,
    pub min_y: u32,
    pub max_x: u32,
    pub max_y: u32,
}

// ---------------------------------------------------------------------------
// map predicates - src/map.rs
// ---------------------------------------------------------------------------

#[inline]
pub fn is_land(terrain: &[u8], t: u32) -> bool {
    terrain[t as usize] & (1 << IS_LAND_BIT) != 0
}

#[inline]
pub fn is_ocean(terrain: &[u8], t: u32) -> bool {
    terrain[t as usize] & (1 << OCEAN_BIT) != 0
}

/// `Game::is_shore` == `map.is_shore` (`map.rs:144`): land AND shoreline bit.
#[inline]
pub fn is_shore(terrain: &[u8], t: u32) -> bool {
    is_land(terrain, t) && terrain[t as usize] & (1 << SHORELINE_BIT) != 0
}

/// `map.rs:282-301` - true when the (land) tile has an ocean 4-neighbour.
/// Boolean OR, so the probe order does not matter.
#[inline]
pub fn is_ocean_shore(terrain: &[u8], w: u32, h: u32, t: u32) -> bool {
    if !is_land(terrain, t) {
        return false;
    }
    let x = t % w;
    if x > 0 && is_ocean(terrain, t - 1) {
        return true;
    }
    if x + 1 < w && is_ocean(terrain, t + 1) {
        return true;
    }
    if t >= w && is_ocean(terrain, t - w) {
        return true;
    }
    if t < (h - 1) * w && is_ocean(terrain, t + w) {
        return true;
    }
    false
}

/// `map.rs:303-307`.
#[inline]
pub fn is_on_edge_of_map(w: u32, h: u32, t: u32) -> bool {
    let x = t % w;
    let y = t / w;
    x == 0 || x + 1 == w || y == 0 || y + 1 == h
}

/// `map.rs:251-280` - the 8-neighbour probe used by `calculateClusters`
/// (`forEachNeighborWithDiag`). dx-major: NW, W, SW, N, S, NE, E, SE.
#[inline]
pub fn neighbors8(w: u32, h: u32, t: u32, out: &mut [u32; 8]) -> usize {
    let x = t % w;
    let has_n = t >= w;
    let has_s = t < (h - 1) * w;
    let mut n = 0usize;
    if x > 0 {
        if has_n {
            out[n] = t - 1 - w;
            n += 1;
        }
        out[n] = t - 1;
        n += 1;
        if has_s {
            out[n] = t - 1 + w;
            n += 1;
        }
    }
    if has_n {
        out[n] = t - w;
        n += 1;
    }
    if has_s {
        out[n] = t + w;
        n += 1;
    }
    if x + 1 < w {
        if has_n {
            out[n] = t + 1 - w;
            n += 1;
        }
        out[n] = t + 1;
        n += 1;
        if has_s {
            out[n] = t + 1 + w;
            n += 1;
        }
    }
    n
}

/// `map.rs:352-367` - **N, S, W, E** on the current tip. This order is
/// load-bearing (it is what the engine's `for_each_neighbor4` now does, and
/// what TS `GameMap.neighbors4` does); the pre-fix W,E,N,S inversion was one
/// of the two bugs that broke native-vs-TS parity for months.
#[inline]
pub fn neighbors4(w: u32, h: u32, t: u32, out: &mut [u32; 4]) -> usize {
    let x = t % w;
    let mut n = 0usize;
    if t >= w {
        out[n] = t - w;
        n += 1;
    }
    if t < (h - 1) * w {
        out[n] = t + w;
        n += 1;
    }
    if x > 0 {
        out[n] = t - 1;
        n += 1;
    }
    if x + 1 < w {
        out[n] = t + 1;
        n += 1;
    }
    n
}

// ---------------------------------------------------------------------------
// bitsets (replace the engine's HashSet<TileRef> membership tests)
// ---------------------------------------------------------------------------

#[inline]
pub fn bit_get(words: &[u32], i: u32) -> bool {
    (words[(i / 32) as usize] >> (i % 32)) & 1 == 1
}

#[inline]
pub fn bit_set(words: &mut [u32], i: u32) {
    words[(i / 32) as usize] |= 1 << (i % 32);
}

// ---------------------------------------------------------------------------
// friendliness (engine `Game::is_friendly`, `game.rs:3296-3329`)
// ---------------------------------------------------------------------------

/// `friends` holds directed pairs packed as `(a << 16) | b` for every
/// `is_friendly(a, b) == true`. Direction matters: `is_friendly_ex` returns
/// false when **b** is disconnected, so the pair is not symmetric.
#[inline]
pub fn friendly(friends: &[u32], a: u16, b: u16) -> bool {
    if a == b {
        return true;
    }
    let key = ((a as u32) << 16) | b as u32;
    let mut i = 0usize;
    while i < friends.len() {
        if friends[i] == key {
            return true;
        }
        i += 1;
    }
    false
}

// ---------------------------------------------------------------------------
// calculateClusters: the 8-neighbour flood over the victim's border set
// ---------------------------------------------------------------------------

/// One thread's worth of `flood_border_cluster` (`player_clusters.rs:93-119`).
///
/// Thread `j` floods the component containing `border_tiles[j]` with the exact
/// DFS the engine runs (mark on *discovery*, push, pop, probe neighbours in
/// `forEachNeighborWithDiag` order), and writes the discovery order into
/// `slot[0..n]`. It returns `n` only when `j` is the smallest border index in
/// that component - i.e. only when `j` is the index the engine's
/// `for border.iter()` loop would have started this cluster at. Otherwise it
/// returns 0 and `slot` must be ignored. That makes the parallel flood produce
/// exactly the engine's cluster list, in the engine's order.
pub fn flood_cluster_slot(
    w: u32,
    h: u32,
    border_tiles: &[u32],
    tile_to_border: &[u32],
    j: u32,
    slot: &mut [u32],
    border_len: u32,
) -> u32 {
    if border_len as usize > MAX_BORDER || j >= border_len {
        return 0;
    }
    let mut vis = [0u32; VIS_WORDS];
    let mut stack = [0u32; MAX_STACK];
    let start = border_tiles[j as usize];
    bit_set(&mut vis, j);
    slot[0] = start;
    let mut n = 1u32;
    let mut min_idx = j;
    stack[0] = start;
    let mut top = 1usize;
    let mut nb = [0u32; 8];
    while top > 0 {
        top -= 1;
        let t = stack[top];
        let k = neighbors8(w, h, t, &mut nb);
        let mut i = 0usize;
        while i < k {
            let nt = nb[i];
            let bi = tile_to_border[nt as usize];
            if bi != NONE && !bit_get(&vis, bi) {
                if n as usize >= slot.len() || top >= MAX_STACK {
                    // Cannot happen for border_len <= MAX_BORDER; bail out
                    // rather than write out of bounds.
                    return 0;
                }
                bit_set(&mut vis, bi);
                stack[top] = nt;
                top += 1;
                slot[n as usize] = nt;
                n += 1;
                if bi < min_idx {
                    min_idx = bi;
                }
            }
            i += 1;
        }
    }
    if min_idx != j {
        return 0;
    }
    n
}

// ---------------------------------------------------------------------------
// bounding boxes
// ---------------------------------------------------------------------------

pub fn bbox_of(w: u32, cluster: &[u32]) -> BBox {
    let mut bb = BBox {
        min_x: u32::MAX,
        min_y: u32::MAX,
        max_x: 0,
        max_y: 0,
    };
    let mut i = 0usize;
    while i < cluster.len() {
        let t = cluster[i];
        let x = t % w;
        let y = t / w;
        if x < bb.min_x {
            bb.min_x = x;
        }
        if y < bb.min_y {
            bb.min_y = y;
        }
        if x > bb.max_x {
            bb.max_x = x;
        }
        if y > bb.max_y {
            bb.max_y = y;
        }
        i += 1;
    }
    bb
}

#[inline]
pub fn inscribed(outer: BBox, inner: BBox) -> bool {
    outer.min_x <= inner.min_x
        && outer.min_y <= inner.min_y
        && outer.max_x >= inner.max_x
        && outer.max_y >= inner.max_y
}

#[inline]
fn bb_expand(bb: &mut BBox, x: u32, y: u32, init: &mut bool) {
    if !*init {
        bb.min_x = x;
        bb.min_y = y;
        bb.max_x = x;
        bb.max_y = y;
        *init = true;
    } else {
        if x < bb.min_x {
            bb.min_x = x;
        }
        if y < bb.min_y {
            bb.min_y = y;
        }
        if x > bb.max_x {
            bb.max_x = x;
        }
        if y > bb.max_y {
            bb.max_y = y;
        }
    }
}

// ---------------------------------------------------------------------------
// the surround tests
// ---------------------------------------------------------------------------

/// `player_clusters.rs:143-208` (TS `surroundedBySamePlayer`). Returns the
/// single enemy that encloses the cluster, or `NONE`.
pub fn surrounded_by_same_enemy(
    p: &Params,
    terrain: &[u8],
    owners: &[u16],
    cluster: &[u32],
    cluster_bb: BBox,
    friends: &[u32],
) -> u32 {
    let mut enemy: u32 = NONE;
    let mut enemy_bb = BBox::default();
    let mut enemy_init = false;
    let mut n4 = [0u32; 4];
    let mut i = 0usize;
    while i < cluster.len() {
        let tile = cluster[i];
        if is_ocean_shore(terrain, p.width, p.height, tile) || is_on_edge_of_map(p.width, p.height, tile) {
            return NONE;
        }
        let k = neighbors4(p.width, p.height, tile, &mut n4);
        let mut q = 0usize;
        while q < k {
            let o = owners[n4[q] as usize] & 0x0fff;
            if o == 0 {
                return NONE;
            }
            if o != p.victim {
                if enemy != NONE && enemy != o as u32 {
                    return NONE;
                }
                enemy = o as u32;
                let x = n4[q] % p.width;
                let y = n4[q] / p.width;
                bb_expand(&mut enemy_bb, x, y, &mut enemy_init);
            }
            q += 1;
        }
        if enemy == NONE {
            return NONE;
        }
        i += 1;
    }
    if enemy == NONE {
        return NONE;
    }
    if friendly(friends, enemy as u16, p.victim) {
        return NONE;
    }
    if inscribed(enemy_bb, cluster_bb) {
        enemy
    } else {
        NONE
    }
}

/// `player_clusters.rs:210-256` (TS `isSurrounded`).
pub fn is_surrounded(
    p: &Params,
    terrain: &[u8],
    owners: &[u16],
    cluster: &[u32],
) -> bool {
    let mut has_enemy = false;
    let mut enemy_bb = BBox::default();
    let mut enemy_init = false;
    let mut n4 = [0u32; 4];
    let mut i = 0usize;
    while i < cluster.len() {
        let tr = cluster[i];
        if is_shore(terrain, tr) || is_on_edge_of_map(p.width, p.height, tr) {
            return false;
        }
        let k = neighbors4(p.width, p.height, tr, &mut n4);
        let mut q = 0usize;
        while q < k {
            let o = owners[n4[q] as usize] & 0x0fff;
            if o != 0 && o != p.victim {
                has_enemy = true;
                let x = n4[q] % p.width;
                let y = n4[q] / p.width;
                bb_expand(&mut enemy_bb, x, y, &mut enemy_init);
            }
            q += 1;
        }
        i += 1;
    }
    if !has_enemy {
        return false;
    }
    let cluster_bb = bbox_of(p.width, cluster);
    inscribed(enemy_bb, cluster_bb)
}

// ---------------------------------------------------------------------------
// captor selection
// ---------------------------------------------------------------------------

/// `player_clusters.rs:258-298` (TS `getCapturingPlayer`).
///
/// `neighbors` keeps first-seen order (the engine's `Vec<(u16,u32)>` stands in
/// for TS's insertion-ordered `Map<Player, number>`), and `getMode` then takes
/// the first entry with a *strictly* greatest count - so first-seen order is
/// part of the specification, not an implementation detail.
///
/// Attack records are three packed slices: `atk_pt[i] = (owner << 16) | target`,
/// `atk_troops_bits[i]` = IEEE-754 bits of the engine's `f64` troop count, and
/// `atk_flags[i]` bits 0/1/2 = the engine's `is_active` / `attack_live` /
/// `is_initialized`. Array order is `Game.execs` order, which
/// `largest_incoming_land_attack_from_neighbors` (`game.rs:2190-2219`) scans
/// with the neighbour list as the outer loop.
pub fn get_capturing_player(
    p: &Params,
    owners: &[u16],
    cluster: &[u32],
    friends: &[u32],
    atk_pt: &[u32],
    atk_troops_bits: &[u64],
    atk_flags: &[u32],
) -> u32 {
    let mut nb_owner = [0u16; MAX_NEIGH];
    let mut nb_count = [0u32; MAX_NEIGH];
    let mut n_nb = 0usize;
    let mut n4 = [0u32; 4];
    let mut i = 0usize;
    while i < cluster.len() {
        let k = neighbors4(p.width, p.height, cluster[i], &mut n4);
        let mut q = 0usize;
        while q < k {
            let o = owners[n4[q] as usize] & 0x0fff;
            if o != 0 && o != p.victim && !friendly(friends, o, p.victim) {
                let mut found = false;
                let mut z = 0usize;
                while z < n_nb {
                    if nb_owner[z] == o {
                        nb_count[z] += 1;
                        found = true;
                        break;
                    }
                    z += 1;
                }
                if !found {
                    if n_nb < MAX_NEIGH {
                        nb_owner[n_nb] = o;
                        nb_count[n_nb] = 1;
                        n_nb += 1;
                    }
                }
            }
            q += 1;
        }
        i += 1;
    }
    if n_nb == 0 {
        return NONE;
    }

    // largest_incoming_land_attack_from_neighbors: outer loop over the
    // first-seen neighbour list, inner loop over exec order; strictly-greater
    // wins, so the first attacker to reach a given troop count keeps it.
    let mut largest = 0.0f64;
    let mut largest_attacker: u32 = NONE;
    let mut z = 0usize;
    while z < n_nb {
        let attacker = nb_owner[z];
        let mut a = 0usize;
        while a < atk_pt.len() {
            let pt = atk_pt[a];
            let owner = (pt >> 16) as u16;
            let target = (pt & 0xffff) as u16;
            let flags = atk_flags[a];
            let ok = (flags & 1) != 0 && (flags & 2) != 0 && (flags & 4) != 0;
            if ok && owner == attacker && target == p.victim {
                let troops = f64::from_bits(atk_troops_bits[a]);
                if troops > largest {
                    largest = troops;
                    largest_attacker = attacker as u32;
                }
            }
            a += 1;
        }
        z += 1;
    }
    if largest_attacker != NONE {
        return largest_attacker;
    }

    // TS `getMode`: first neighbour with strictly greatest count wins.
    let mut best: u32 = NONE;
    let mut best_count = 0u32;
    let mut s = 0usize;
    while s < n_nb {
        if best == NONE || nb_count[s] > best_count {
            best = nb_owner[s] as u32;
            best_count = nb_count[s];
        }
        s += 1;
    }
    best
}

// ---------------------------------------------------------------------------
// flood_owned (two passes: count, then replay the identical DFS to conquer)
// ---------------------------------------------------------------------------

/// `player_clusters.rs:300-323` (TS `removeCluster`'s
/// `floodFillWithGen(forEachNeighbor)`). Pass 1 marks the 4-connected region of
/// victim-owned tiles reachable from `start` with generation `g` and returns
/// its size; pass 2 (below) walks the *identical* DFS order and emits the
/// tiles. Two passes keep the result list out of local memory without changing
/// the order - the engine's `result` array is in that same discovery order.
pub fn flood_owned_count(
    p: &Params,
    owners: &[u16],
    start: u32,
    marks: &mut [u32],
    g: u32,
    stack: &mut [u32],
) -> u32 {
    let mut count = 0u32;
    let mut top = 0usize;
    if (owners[start as usize] & 0x0fff) == p.victim {
        marks[start as usize] = g;
        stack[top] = start;
        top += 1;
        count += 1;
    } else {
        return 0;
    }
    let mut n4 = [0u32; 4];
    while top > 0 {
        top -= 1;
        let t = stack[top];
        let k = neighbors4(p.width, p.height, t, &mut n4);
        let mut q = 0usize;
        while q < k {
            let nt = n4[q];
            if marks[nt as usize] < g && (owners[nt as usize] & 0x0fff) == p.victim {
                marks[nt as usize] = g;
                if top < stack.len() {
                    stack[top] = nt;
                    top += 1;
                }
                count += 1;
            }
            q += 1;
        }
    }
    count
}

/// Pass 2 of `flood_owned`: the same DFS in the same order, writing each tile
/// as it is discovered (`marks == g` -> `marks = g + 1`) into `out[0..n]`.
/// `out` is the engine's `result` vector (`player_clusters.rs:301-322`); the
/// caller conquers it in this order (`:344-346`), so the engine's
/// `OrderedTiles` insertion sequence is reproduced exactly.
///
/// Fills a buffer rather than taking a callback so that the caller can hold a
/// mutable borrow of the owner plane while it conquers - and so the device
/// copy stays free of closures.
pub fn flood_owned_emit(
    p: &Params,
    owners: &[u16],
    start: u32,
    marks: &mut [u32],
    g: u32,
    stack: &mut [u32],
    out: &mut [u32],
) -> u32 {
    let mut top = 0usize;
    let mut n_out: u32;
    if marks[start as usize] == g {
        marks[start as usize] = g + 1;
        out[0] = start;
        n_out = 1u32;
        stack[top] = start;
        top += 1;
    } else {
        return 0;
    }
    let mut n4 = [0u32; 4];
    while top > 0 {
        top -= 1;
        let t = stack[top];
        let k = neighbors4(p.width, p.height, t, &mut n4);
        let mut q = 0usize;
        while q < k {
            let nt = n4[q];
            // `marks == g` already implies "victim-owned when marked"; the
            // owner re-read mirrors the engine's flood_owned line exactly.
            if marks[nt as usize] == g && (owners[nt as usize] & 0x0fff) == p.victim {
                marks[nt as usize] = g + 1;
                if (n_out as usize) < out.len() {
                    out[n_out as usize] = nt;
                    n_out += 1;
                }
                if top < stack.len() {
                    stack[top] = nt;
                    top += 1;
                }
            }
            q += 1;
        }
    }
    n_out
}

// ---------------------------------------------------------------------------
// remove_cluster
// ---------------------------------------------------------------------------

/// `player_clusters.rs:325-347` (TS `removeCluster`). `victim_tiles_owned` is
/// the live counter the engine keeps on the player (only the victim's is read;
/// `conquer_one` decrements the ousted owner and `wipe_all` compares against
/// it). Changes are appended to `changes` as packed `(tile << 16) | new_owner`
/// in application order. `conquer_player` (`game.rs:1163-1220`) touches ships
/// and gold only, so it contributes no owner-plane write.
#[allow(clippy::too_many_arguments)]
pub fn remove_cluster(
    p: &Params,
    terrain: &[u8],
    owners: &mut [u16],
    cluster: &[u32],
    friends: &[u32],
    atk_pt: &[u32],
    atk_troops_bits: &[u64],
    atk_flags: &[u32],
    marks: &mut [u32],
    g: u32,
    stack: &mut [u32],
    tiles_out: &mut [u32],
    victim_tiles_owned: &mut i32,
    changes: &mut [u64],
    n_changes: &mut u32,
) -> bool {
    let mut i = 0usize;
    while i < cluster.len() {
        if (owners[cluster[i] as usize] & 0x0fff) != p.victim {
            return false;
        }
        i += 1;
    }
    let captor = get_capturing_player(p, owners, cluster, friends, atk_pt, atk_troops_bits, atk_flags);
    if captor == NONE {
        return false;
    }
    let first = cluster[0];
    let n_tiles = flood_owned_count(p, owners, first, marks, g, stack);
    // Engine: `if wipe_all { conquer_player(captor, victim) }` - ships and gold
    // only, no owner-plane write (`game.rs:1163-1220`) - then
    // `for t in tiles { conquer(captor, t) }`.
    let _wipe_all = *victim_tiles_owned == n_tiles as i32;
    let captor = captor as u16;
    let emitted = flood_owned_emit(p, owners, first, marks, g, stack, tiles_out);
    let mut i = 0u32;
    while i < emitted {
        let t = tiles_out[i as usize];
        // `Game::conquer_one` land guard (`game.rs:1237-1240`).
        if is_land(terrain, t) {
            owners[t as usize] = (owners[t as usize] & !0x0fff) | (captor & 0x0fff);
            if (*n_changes as usize) < changes.len() {
                changes[*n_changes as usize] = ((t as u64) << 16) | captor as u64;
                *n_changes += 1;
            }
            *victim_tiles_owned -= 1;
        }
        i += 1;
    }
    true
}

// ---------------------------------------------------------------------------
// the whole step
// ---------------------------------------------------------------------------

/// `player_clusters.rs:349-420` (TS `PlayerExecution.tick` gate :99-113 +
/// `removeClusters` :115-155), with the cluster list already computed.
///
/// `slots` is the flattened per-start-tile cluster storage laid out exactly as
/// `flood_cluster_slot` writes it (`slots[j * MAX_CLUSTER + k]`), and
/// `slot_len[j]` is its length (0 when tile `j` is not its component's
/// minimum border index).
#[allow(clippy::too_many_arguments)]
pub fn decide_and_remove(
    p: &Params,
    terrain: &[u8],
    owners: &mut [u16],
    slots: &[u32],
    slot_len: &[u32],
    friends: &[u32],
    atk_pt: &[u32],
    atk_troops_bits: &[u64],
    atk_flags: &[u32],
    marks: &mut [u32],
    stack: &mut [u32],
    tiles_out: &mut [u32],
    changes: &mut [u64],
) -> StepOut {
    let mut out = StepOut::default();
    if !p.alive || p.tiles_owned == 0 {
        return out;
    }
    let mut last_calc = p.last_cluster_calc;
    if last_calc == 0 {
        last_calc = p.tick + (p.id_hash as u32 % TICKS_PER_CLUSTER_CALC);
    }
    if p.tick.saturating_sub(last_calc) <= TICKS_PER_CLUSTER_CALC && p.tiles_owned >= 100 {
        return out;
    }
    if p.last_tile_change < last_calc {
        return out;
    }
    out.fired = true;

    let border_len = p.border_len;
    if border_len == 0 || border_len as usize > MAX_BORDER {
        return out;
    }

    // The engine's cluster list order == ascending start index.
    let mut starts = [0u32; MAX_BORDER];
    let mut n_clusters = 0usize;
    let mut j = 0u32;
    while j < border_len {
        if slot_len[j as usize] > 0 {
            starts[n_clusters] = j;
            n_clusters += 1;
        }
        j += 1;
    }
    if n_clusters == 0 {
        return out;
    }
    out.cluster_count = n_clusters as u32;

    // Largest: first index with a strictly greater size.
    let mut largest_idx = 0usize;
    let mut largest_size = slot_len[starts[0] as usize];
    let mut i = 1usize;
    while i < n_clusters {
        let s = slot_len[starts[i] as usize];
        if s > largest_size {
            largest_size = s;
            largest_idx = i;
        }
        i += 1;
    }
    out.largest_index = largest_idx as u32;
    out.largest_size = largest_size;

    let cluster_of = |idx: usize| -> &[u32] {
        let j = starts[idx] as usize;
        let len = slot_len[j] as usize;
        &slots[j * MAX_CLUSTER..j * MAX_CLUSTER + len]
    };

    let mut victim_tiles_owned = p.tiles_owned;
    let mut n_changes = 0u32;
    let mut g: u32 = 1;

    let largest = cluster_of(largest_idx);
    let largest_bb = bbox_of(p.width, largest);
    if surrounded_by_same_enemy(p, terrain, owners, largest, largest_bb, friends) != NONE {
        if remove_cluster(
            p,
            terrain,
            owners,
            largest,
            friends,
            atk_pt,
            atk_troops_bits,
            atk_flags,
            marks,
            g,
            stack,
            tiles_out,
            &mut victim_tiles_owned,
            changes,
            &mut n_changes,
        ) {
            out.removed_clusters += 1;
        }
        g += 2;
    }

    let mut idx = 0usize;
    while idx < n_clusters {
        if idx != largest_idx {
            let cluster = cluster_of(idx);
            if is_surrounded(p, terrain, owners, cluster) {
                if remove_cluster(
                    p,
                    terrain,
                    owners,
                    cluster,
                    friends,
                    atk_pt,
                    atk_troops_bits,
                    atk_flags,
                    marks,
                    g,
                    stack,
                    tiles_out,
                    &mut victim_tiles_owned,
                    changes,
                    &mut n_changes,
                ) {
                    out.removed_clusters += 1;
                }
                g += 2;
            }
        }
        idx += 1;
    }
    out.changes = n_changes;
    out
}
    // ===== END VERBATIM COPY of src/core_impl.rs =====

    /// Thread `j` floods the component containing `border_tiles[j]` and writes
    /// its discovery order into slot `j`; `slot_len[j] == 0` marks a thread
    /// whose component is owned by a lower border index.
    #[kernel]
    #[launch_bounds(128)]
    pub fn flood_clusters(
        border_tiles: &[u32],
        tile_to_border: &[u32],
        width: u32,
        height: u32,
        border_len: u32,
        mut slot_len: &mut [u32],
        mut slots: &mut [u32],
    ) {
        let j = thread::index_1d().get() as u32;
        if j >= border_len {
            return;
        }
        let base = (j as usize) * MAX_CLUSTER;
        let n = flood_cluster_slot(
            width,
            height,
            border_tiles,
            tile_to_border,
            j,
            &mut slots[base..base + MAX_CLUSTER],
            border_len,
        );
        slot_len[j as usize] = n;
    }

    /// One thread replays `decide_and_remove` (largest-cluster test, then the
    /// per-cluster `is_surrounded` sweep, then the conquer flood) on the owner
    /// plane, and reports the step outcome in `out_meta`:
    /// `[fired, cluster_count, largest_index, largest_size, removed_clusters, changes]`.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    #[launch_bounds(1)]
    pub fn decide_remove(
        terrain: &[u8],
        slots: &[u32],
        slot_len: &[u32],
        friends: &[u32],
        atk_pt: &[u32],
        atk_troops_bits: &[u64],
        atk_flags: &[u32],
        width: u32,
        height: u32,
        tick: u32,
        victim: u32,
        id_hash_bits: u32,
        tiles_owned_bits: u32,
        alive: u32,
        last_cluster_calc: u32,
        last_tile_change: u32,
        border_len: u32,
        mut owners: &mut [u16],
        mut marks: &mut [u32],
        mut stack: &mut [u32],
        mut tiles_out: &mut [u32],
        mut changes: &mut [u64],
        mut out_meta: &mut [u32],
    ) {
        let p = Params {
            width,
            height,
            tick,
            victim: victim as u16,
            id_hash: id_hash_bits as i32,
            tiles_owned: tiles_owned_bits as i32,
            alive: alive != 0,
            last_cluster_calc,
            last_tile_change,
            border_len,
        };
        let out = decide_and_remove(
            &p,
            terrain,
            owners,
            slots,
            slot_len,
            friends,
            atk_pt,
            atk_troops_bits,
            atk_flags,
            marks,
            stack,
            tiles_out,
            changes,
        );
        out_meta[0] = out.fired as u32;
        out_meta[1] = out.cluster_count;
        out_meta[2] = out.largest_index;
        out_meta[3] = out.largest_size;
        out_meta[4] = out.removed_clusters;
        out_meta[5] = out.changes;
    }
}

const DEFAULT_CASE: &str = "cases/cluster-b030-victim33-t1614.json";

fn repo_root() -> PathBuf {
    std::env::var("OFCUDA_REPO")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/opt/data/workspaces/skg"))
}

/// Run the two kernels and read everything back.
#[allow(clippy::too_many_arguments)]
fn run_gpu(
    ctx: &std::sync::Arc<CudaContext>,
    case: &Case,
    terrain: &[u8],
    prep: &Prepared,
) -> Result<Run, Box<dyn std::error::Error>> {
    let stream = ctx.default_stream();
    let module = unsafe { kernels::load(ctx)? };

    let border_len = prep.border_len;
    // Raw (uncontracted) launch path: `#[launch_contract]` forbids `&mut [T]`
    // parameters and requires `DisjointSlice`, whose Index1D only lets a thread
    // write its own flat index - which cannot express kernel A's per-thread
    // contiguous slot block. The host therefore owns the bounds, and every
    // kernel re-checks its own indices (`j >= border_len`).
    let cfg_a = LaunchConfig {
        grid_dim: (border_len.div_ceil(128), 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    let d_border = DeviceBuffer::from_host(&stream, &prep.border)?;
    let d_t2b = DeviceBuffer::from_host(&stream, &prep.tile_to_border)?;
    let mut d_slot_len = DeviceBuffer::<u32>::zeroed(&stream, border_len as usize)?;
    let mut d_slots = DeviceBuffer::<u32>::zeroed(&stream, prep.slots.len())?;
    unsafe {
        module.flood_clusters(
            &stream,
            cfg_a,
            &d_border,
            &d_t2b,
            case.width,
            case.height,
            border_len,
            &mut d_slot_len,
            &mut d_slots,
        )?;
    }

    let d_terrain = DeviceBuffer::from_host(&stream, terrain)?;
    let d_friends = DeviceBuffer::from_host(&stream, &prep.friends_packed)?;
    let d_pt = DeviceBuffer::from_host(&stream, &prep.atk_pt)?;
    let d_troops = DeviceBuffer::from_host(&stream, &prep.atk_troops)?;
    let d_flags = DeviceBuffer::from_host(&stream, &prep.atk_flags)?;
    let mut d_owners = DeviceBuffer::from_host(&stream, &case.plane)?;
    let mut d_gen = DeviceBuffer::<u32>::zeroed(&stream, case.tiles())?;
    let mut d_stack = DeviceBuffer::<u32>::zeroed(&stream, prep.changes_cap + 1024 + 1)?;
    let mut d_tiles_out = DeviceBuffer::<u32>::zeroed(&stream, prep.changes_cap)?;
    let mut d_changes = DeviceBuffer::<u64>::zeroed(&stream, prep.changes_cap)?;
    let mut d_meta = DeviceBuffer::<u32>::zeroed(&stream, 8)?;

    let cfg_b = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe {
        module.decide_remove(
        &stream,
        cfg_b,
        &d_terrain,
        &d_slots,
        &d_slot_len,
        &d_friends,
        &d_pt,
        &d_troops,
        &d_flags,
        case.width,
        case.height,
        case.exec_tick,
        prep.victim as u32,
        prep.params.id_hash as u32,
        prep.params.tiles_owned as u32,
        prep.params.alive as u32,
        prep.params.last_cluster_calc,
        prep.params.last_tile_change,
        border_len,
        &mut d_owners,
        &mut d_gen,
        &mut d_stack,
        &mut d_tiles_out,
        &mut d_changes,
        &mut d_meta,
    )?;
    }

    let meta = d_meta.to_host_vec(&stream)?;
    let plane = d_owners.to_host_vec(&stream)?;
    let raw = d_changes.to_host_vec(&stream)?;
    let n = meta[5] as usize;
    let changes: Vec<(u32, u16)> = raw
        .iter()
        .take(n.min(raw.len()))
        .map(|c| ((*c >> 16) as u32, (*c & 0xffff) as u16))
        .collect();
    let out = ofcuda_cluster::StepOut {
        fired: meta[0] != 0,
        cluster_count: meta[1],
        largest_index: meta[2],
        largest_size: meta[3],
        removed_clusters: meta[4],
        changes: meta[5],
    };
    let mut fatal = None;
    if n > raw.len() {
        fatal = Some(format!("GPU reported {n} changes but the buffer holds {}", raw.len()));
    }
    Ok(Run {
        out,
        changes,
        plane_after: plane,
        fatal,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if let Err(e) = device_core_verbatim_check() {
        eprintln!("FATAL: device core copy check failed:\n{e}");
        std::process::exit(2);
    }

    let case_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CASE));
    let root = repo_root();
    let case = load_case(&case_path)?;
    let map_dir = pangaea_map_dir(&root);
    let (terrain, terrain_hash) = load_terrain(&map_dir)?;

    let prep = prepare(&case, case.victim)?;
    let diags = diagnose(&case, &terrain, &prep);
    let cpu = decide_cpu(&case, &terrain, &prep);

    let ctx = CudaContext::new(0)?;
    let dev = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,driver_version", "--format=csv,noheader"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let gpu = run_gpu(&ctx, &case, &terrain, &prep)?;

    let gpu_checks: Vec<Check> = compare_runs(&cpu, &gpu);
    let engine_checks = compare_to_engine(&case, &cpu);
    let engine_checks_gpu = compare_to_engine(&case, &gpu);

    let (report, ok) = format_report(&ReportInput {
        case: &case,
        device: &dev,
        terrain_path: &map_dir.display().to_string(),
        terrain_hash,
        prep: &prep,
        diags: &diags,
        cpu: &cpu,
        gpu: Some(&gpu),
        gpu_checks: &gpu_checks,
        engine_checks: &engine_checks,
        engine_checks_gpu: &engine_checks_gpu,
    });
    print!("{report}");
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
