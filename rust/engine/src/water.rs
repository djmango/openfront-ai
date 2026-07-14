//! Water components + stamp-based BFS (no per-query full-map allocations).

use crate::map::{GameMap, TileRef};
use std::collections::VecDeque;

pub struct BfsScratch {
    stamp: u32,
    pub(crate) seen: Vec<u32>,
    pub(crate) parent: Vec<TileRef>,
}

impl BfsScratch {
    pub fn new(tile_count: usize) -> Self {
        Self {
            stamp: 0,
            seen: vec![0; tile_count],
            parent: vec![TileRef::MAX; tile_count],
        }
    }

    pub(crate) fn next_stamp(&mut self) -> u32 {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.seen.fill(0);
            self.stamp = 1;
        }
        self.stamp
    }
}

pub fn build_water_components(map: &GameMap) -> Vec<u32> {
    let n = (map.width * map.height) as usize;
    let mut comp = vec![0u32; n];
    let mut next = 1u32;
    for t in 0..n {
        let t = t as TileRef;
        if !map.is_water(t) || comp[t as usize] != 0 {
            continue;
        }
        let id = next;
        next += 1;
        let mut q = VecDeque::from([t]);
        comp[t as usize] = id;
        while let Some(cur) = q.pop_front() {
            map.for_each_neighbor4(cur, |n| {
                if map.is_water(n) && comp[n as usize] == 0 {
                    comp[n as usize] = id;
                    q.push_back(n);
                }
            });
        }
    }
    comp
}

pub fn water_component_at(map: &GameMap, comp: &[u32], t: TileRef) -> Option<u32> {
    if map.is_water(t) {
        let c = comp[t as usize];
        return if c == 0 { None } else { Some(c) };
    }
    let mut found = None;
    map.for_each_neighbor4(t, |n| {
        if map.is_water(n) {
            let c = comp[n as usize];
            if c != 0 {
                found = Some(c);
            }
        }
    });
    found
}

fn water_endpoint(map: &GameMap, comp: &[u32], t: TileRef, want_comp: u32) -> Option<TileRef> {
    if map.is_water(t) {
        return Some(t);
    }
    let mut adj = None;
    map.for_each_neighbor4(t, |n| {
        if adj.is_none() && map.is_water(n) && comp[n as usize] == want_comp {
            adj = Some(n);
        }
    });
    adj
}

/// Stamp BFS on land tiles. Returns nearest tile matching `owner` shore predicate.
pub fn land_bfs_nearest_shore(
    map: &GameMap,
    scratch: &mut BfsScratch,
    from: TileRef,
    max_dist: u32,
    owner: u16,
) -> Option<TileRef> {
    land_bfs_nearest(
        map,
        scratch,
        from,
        max_dist,
        |t| map.is_shore(t) && map.owner_id(t) == owner,
    )
}

/// TS `SpatialQuery.bfsNearest` at engine commit `0c4c7d7993c9`: full-range BFS + sort.
pub fn land_bfs_nearest(
    map: &GameMap,
    scratch: &mut BfsScratch,
    from: TileRef,
    max_dist: u32,
    mut pred: impl FnMut(TileRef) -> bool,
) -> Option<TileRef> {
    let tiles = map.bfs_with_scratch(scratch, from, |gm, t| {
        gm.manhattan_dist(from, t) <= max_dist
    });
    let mut candidates: Vec<TileRef> = tiles.into_iter().filter(|&t| pred(t)).collect();
    if candidates.is_empty() {
        return None;
    }
    candidates.sort_by_key(|&t| map.manhattan_dist(from, t));
    Some(candidates[0])
}

/// Stamp BFS water path; writes into `path_out` and returns true on success.
pub fn water_bfs_path_into(
    map: &GameMap,
    comp: &[u32],
    scratch: &mut BfsScratch,
    from: TileRef,
    to: TileRef,
    path_out: &mut Vec<TileRef>,
) -> bool {
    path_out.clear();
    let from_comp = match water_component_at(map, comp, from) {
        Some(c) => c,
        None => return false,
    };
    let to_comp = match water_component_at(map, comp, to) {
        Some(c) => c,
        None => return false,
    };
    if from_comp != to_comp {
        return false;
    }

    let Some(start) = water_endpoint(map, comp, from, from_comp) else {
        return false;
    };
    let Some(goal) = water_endpoint(map, comp, to, to_comp) else {
        return false;
    };

    if start == goal {
        path_out.push(start);
        return true;
    }

    let stamp = scratch.next_stamp();
    let mut q = VecDeque::from([start]);
    scratch.seen[start as usize] = stamp;
    scratch.parent[start as usize] = TileRef::MAX;

    let mut found = false;
    while let Some(cur) = q.pop_front() {
        if cur == goal {
            found = true;
            break;
        }
        map.for_each_neighbor4(cur, |nb| {
            if scratch.seen[nb as usize] != stamp
                && map.is_water(nb)
                && comp[nb as usize] == from_comp
            {
                scratch.seen[nb as usize] = stamp;
                scratch.parent[nb as usize] = cur;
                q.push_back(nb);
            }
        });
    }

    if !found {
        return false;
    }

    let mut p = goal;
    while p != TileRef::MAX {
        path_out.push(p);
        if p == start {
            break;
        }
        p = scratch.parent[p as usize];
    }
    path_out.reverse();
    true
}

/// Quick reachability check without building the full path.
pub fn water_reachable(
    map: &GameMap,
    comp: &[u32],
    scratch: &mut BfsScratch,
    from: TileRef,
    to: TileRef,
) -> bool {
    let from_comp = match water_component_at(map, comp, from) {
        Some(c) => c,
        None => return false,
    };
    let to_comp = match water_component_at(map, comp, to) {
        Some(c) => c,
        None => return false,
    };
    if from_comp != to_comp {
        return false;
    }
    let Some(start) = water_endpoint(map, comp, from, from_comp) else {
        return false;
    };
    let Some(goal) = water_endpoint(map, comp, to, to_comp) else {
        return false;
    };
    if start == goal {
        return true;
    }

    let stamp = scratch.next_stamp();
    let mut q = VecDeque::from([start]);
    scratch.seen[start as usize] = stamp;
    while let Some(cur) = q.pop_front() {
        if cur == goal {
            return true;
        }
        map.for_each_neighbor4(cur, |nb| {
            if scratch.seen[nb as usize] != stamp
                && map.is_water(nb)
                && comp[nb as usize] == from_comp
            {
                scratch.seen[nb as usize] = stamp;
                q.push_back(nb);
            }
        });
    }
    false
}

const LAND_BIT: u8 = 7;
const COST_SCALE: u32 = 100;
const BASE_COST: u32 = 100;

fn magnitude_penalty(mag: u8) -> u32 {
    if mag < 3 {
        10 * COST_SCALE
    } else if mag <= 10 {
        0
    } else {
        COST_SCALE
    }
}

fn count_water_neighbors_ts(map: &GameMap, tile: TileRef) -> u32 {
    let mut buf = [0u32; 4];
    let n = map.neighbors4_ts(tile, &mut buf);
    let mut count = 0u32;
    for i in 0..n {
        if map.is_water(buf[i]) {
            count += 1;
        }
    }
    count
}

/// TS `ShoreCoercingTransformer.coerceToWater` - pick highest-connectivity adjacent water.
pub fn coerce_shore_to_water(map: &GameMap, tile: TileRef) -> Option<TileRef> {
    if map.is_water(tile) {
        return Some(tile);
    }
    let mut buf = [0u32; 4];
    let n = map.neighbors4_ts(tile, &mut buf);
    let mut best: Option<TileRef> = None;
    let mut max_score = -1i32;
    for i in 0..n {
        let neighbor = buf[i];
        if !map.is_water(neighbor) {
            continue;
        }
        let score = count_water_neighbors_ts(map, neighbor) as i32;
        if score > max_score {
            max_score = score;
            best = Some(neighbor);
        }
    }
    best
}

pub(crate) struct AstarHeap {
    f: Vec<f32>,
    tile: Vec<TileRef>,
}

impl AstarHeap {
    pub(crate) fn new(cap: usize) -> Self {
        Self {
            f: Vec::with_capacity(cap),
            tile: Vec::with_capacity(cap),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.f.clear();
        self.tile.clear();
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.f.is_empty()
    }

    pub(crate) fn push(&mut self, tile: TileRef, priority: u32) {
        // TS `MinHeap` stores priorities in a Float32Array, so comparisons use
        // f32-rounded values even when callers compute integer f-scores.
        let priority = priority as f32;
        let mut i = self.f.len();
        self.f.push(priority);
        self.tile.push(tile);
        while i > 0 {
            let parent = (i - 1) >> 1;
            if self.f[parent] <= self.f[i] {
                break;
            }
            self.f.swap(parent, i);
            self.tile.swap(parent, i);
            i = parent;
        }
    }

    pub(crate) fn pop(&mut self) -> TileRef {
        let top = self.tile[0];
        let last_f = self.f.pop().unwrap();
        let last_tile = self.tile.pop().unwrap();
        if self.f.is_empty() {
            return top;
        }
        self.f[0] = last_f;
        self.tile[0] = last_tile;
        let mut i = 0usize;
        loop {
            let left = (i << 1) + 1;
            if left >= self.f.len() {
                break;
            }
            let right = left + 1;
            let mut smallest = i;
            if self.f[left] < self.f[smallest] {
                smallest = left;
            }
            if right < self.f.len() && self.f[right] < self.f[smallest] {
                smallest = right;
            }
            if smallest == i {
                break;
            }
            self.f.swap(smallest, i);
            self.tile.swap(smallest, i);
            i = smallest;
        }
        top
    }
}

#[cfg(test)]
mod astar_heap_tests {
    use super::AstarHeap;

    #[test]
    fn equal_priorities_match_ts_minheap_structure_order() {
        let mut heap = AstarHeap::new(8);
        for tile in [10u32, 20, 30, 40] {
            heap.push(tile, 1);
        }

        let mut out = Vec::new();
        while !heap.is_empty() {
            out.push(heap.pop());
        }

        // Equal priorities do not form a FIFO queue; this is the exact order
        // produced by TS MinHeap's strict bubble-down comparisons.
        assert_eq!(out, vec![10, 40, 30, 20]);
    }

    #[test]
    fn priorities_are_compared_at_ts_float32_precision() {
        let mut heap = AstarHeap::new(4);

        heap.push(1, 16_777_217);
        heap.push(2, 16_777_216);

        // Float32Array rounds both priorities to 16_777_216. With exact u32
        // storage the second push would bubble above the first.
        assert_eq!(heap.pop(), 1);
        assert_eq!(heap.pop(), 2);
    }
}

pub struct WaterAstarScratch {
    stamp: u32,
    closed: Vec<u32>,
    g_stamp: Vec<u32>,
    g: Vec<u32>,
    came_from: Vec<TileRef>,
    heap: AstarHeap,
}

impl WaterAstarScratch {
    pub fn new(tile_count: usize) -> Self {
        Self {
            stamp: 0,
            closed: vec![0; tile_count],
            g_stamp: vec![0; tile_count],
            g: vec![0; tile_count],
            came_from: vec![TileRef::MAX; tile_count],
            heap: AstarHeap::new(tile_count),
        }
    }

    fn next_stamp(&mut self) -> u32 {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.closed.fill(0);
            self.g_stamp.fill(0);
            self.stamp = 1;
        }
        self.stamp
    }
}

/// TS `AStar.Water` on water tiles (magnitude-weighted).
pub fn astar_water_path_into(
    map: &GameMap,
    scratch: &mut WaterAstarScratch,
    starts: &[TileRef],
    goal: TileRef,
    path_out: &mut Vec<TileRef>,
) -> bool {
    path_out.clear();
    if starts.is_empty() {
        return false;
    }

    let stamp = scratch.next_stamp();
    let width = map.width;
    let num_nodes = (width * map.height) as usize;
    let heuristic_weight = 5u32;
    let land_mask = 1u8 << LAND_BIT;

    let goal_x = goal % width;
    let goal_y = goal / width;

    let s0 = starts[0];
    let start_x = s0 % width;
    let start_y = s0 / width;
    let dx_goal = goal_x as i32 - start_x as i32;
    let dy_goal = goal_y as i32 - start_y as i32;
    let cross_norm = (dx_goal.abs() + dy_goal.abs()).max(1) as u32;

    let cross_tie = |nx: u32, ny: u32| -> u32 {
        let dx_n = nx as i32 - goal_x as i32;
        let dy_n = ny as i32 - goal_y as i32;
        let cross = (dx_goal * dy_n - dy_goal * dx_n).unsigned_abs();
        ((cross as f64 * (COST_SCALE - 1) as f64) / cross_norm as f64 / cross_norm as f64).floor()
            as u32
    };

    scratch.heap.clear();
    for &s in starts {
        scratch.g[s as usize] = 0;
        scratch.g_stamp[s as usize] = stamp;
        scratch.came_from[s as usize] = TileRef::MAX;
        let sx = s % width;
        let sy = s / width;
        let h = heuristic_weight * BASE_COST * (sx.abs_diff(goal_x) + sy.abs_diff(goal_y));
        scratch.heap.push(s, h);
    }

    let mut iterations = 1_000_000u32;
    while !scratch.heap.is_empty() {
        iterations -= 1;
        if iterations == 0 {
            return false;
        }

        let current = scratch.heap.pop();
        if scratch.closed[current as usize] == stamp {
            continue;
        }
        scratch.closed[current as usize] = stamp;

        if current == goal {
            let mut p = goal;
            while p != TileRef::MAX {
                path_out.push(p);
                if scratch.came_from[p as usize] == TileRef::MAX {
                    break;
                }
                p = scratch.came_from[p as usize];
            }
            path_out.reverse();
            return true;
        }

        let current_g = scratch.g[current as usize];
        let current_x = current % width;
        let current_y = current / width;

        let relax = |scratch: &mut WaterAstarScratch, neighbor: TileRef, nx: u32, ny: u32| {
            let terrain = map.terrain_byte(neighbor);
            if scratch.closed[neighbor as usize] == stamp {
                return;
            }
            if neighbor != goal && terrain & land_mask != 0 {
                return;
            }
            let magnitude = terrain & 0x1f;
            let cost = BASE_COST + magnitude_penalty(magnitude);
            let tentative_g = current_g.saturating_add(cost);
            if scratch.g_stamp[neighbor as usize] != stamp || tentative_g < scratch.g[neighbor as usize] {
                scratch.came_from[neighbor as usize] = current;
                scratch.g[neighbor as usize] = tentative_g;
                scratch.g_stamp[neighbor as usize] = stamp;
                let h = heuristic_weight * BASE_COST * (nx.abs_diff(goal_x) + ny.abs_diff(goal_y));
                let f = tentative_g + h + cross_tie(nx, ny);
                scratch.heap.push(neighbor, f);
            }
        };

        if current_y > 0 {
            let neighbor = current - width;
            relax(scratch, neighbor, current_x, current_y - 1);
        }
        if (current_y as usize) < num_nodes / width as usize - 1 {
            let neighbor = current + width;
            relax(scratch, neighbor, current_x, current_y + 1);
        }
        if current_x > 0 {
            let neighbor = current - 1;
            relax(scratch, neighbor, current_x - 1, current_y);
        }
        if current_x + 1 < width {
            let neighbor = current + 1;
            relax(scratch, neighbor, current_x + 1, current_y);
        }
    }

    false
}

/// TS `AStar.WaterBounded.searchBounded` for `refineStartTile`.
pub fn bounded_water_path(
    map: &GameMap,
    starts: &[TileRef],
    goal: TileRef,
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
) -> Option<Vec<TileRef>> {
    if starts.is_empty() {
        return None;
    }
    if std::env::var("DEBUG_LOCAL_BOUNDED").is_ok() {
        eprintln!(
            "[native local2] starts={:?} goal={} (x={},y={}) bounds=[{},{}]x[{},{}]",
            starts.iter().map(|&s| (s, map.x(s), map.y(s))).collect::<Vec<_>>(),
            goal, map.x(goal), map.y(goal), min_x, max_x, min_y, max_y
        );
    }
    let width = map.width;
    let bounds_w = max_x - min_x + 1;
    let bounds_h = max_y - min_y + 1;
    let num_local = (bounds_w * bounds_h) as usize;
    if num_local == 0 || num_local > 10_000 {
        return None;
    }

    let mut closed = vec![0u32; num_local];
    let mut g_stamp = vec![0u32; num_local];
    let mut g = vec![0u32; num_local];
    let mut came_from = vec![i32::MAX; num_local];
    let mut heap = AstarHeap::new(num_local * 4);
    let stamp = 1u32;
    let land_mask = 1u8 << LAND_BIT;
    let heuristic_weight = 3u32;
    let goal_x = map.x(goal);
    let goal_y = map.y(goal);

    let to_local = |tile: TileRef| -> Option<usize> {
        let x = map.x(tile);
        let y = map.y(tile);
        if x < min_x || x > max_x || y < min_y || y > max_y {
            return None;
        }
        Some(((y - min_y) * bounds_w + (x - min_x)) as usize)
    };
    let to_global = |local: usize| -> TileRef {
        let lx = (local as u32) % bounds_w;
        let ly = (local as u32) / bounds_w;
        map.ref_xy(lx + min_x, ly + min_y)
    };

    let to_local_clamped = |tile: TileRef| -> Option<usize> {
        let mut x = map.x(tile);
        let mut y = map.y(tile);
        x = x.clamp(min_x, max_x);
        y = y.clamp(min_y, max_y);
        Some(((y - min_y) * bounds_w + (x - min_x)) as usize)
    };

    let goal_local = to_local_clamped(goal)?;
    if goal_local >= num_local {
        return None;
    }
    let s0 = starts[0];
    let start_x = map.x(s0);
    let start_y = map.y(s0);
    let dx_goal = goal_x as i32 - start_x as i32;
    let dy_goal = goal_y as i32 - start_y as i32;
    let cross_norm = (dx_goal.abs() + dy_goal.abs()).max(1) as u32;
    let cross_tie = |nx: u32, ny: u32| -> u32 {
        let dx_n = nx as i32 - goal_x as i32;
        let dy_n = ny as i32 - goal_y as i32;
        let cross = (dx_goal * dy_n - dy_goal * dx_n).unsigned_abs();
        ((cross as f64 * (COST_SCALE - 1) as f64) / cross_norm as f64 / cross_norm as f64).floor()
            as u32
    };

    heap.clear();
    for &s in starts {
        let Some(start_local) = to_local_clamped(s) else {
            continue;
        };
        g[start_local] = 0;
        g_stamp[start_local] = stamp;
        came_from[start_local] = -1;
        let h = heuristic_weight
            * BASE_COST
            * (map.x(s).abs_diff(goal_x) + map.y(s).abs_diff(goal_y));
        heap.push(start_local as TileRef, h);
    }

    let mut iterations = 100_000u32;
    while !heap.is_empty() {
        iterations -= 1;
        if iterations == 0 {
            return None;
        }
        let current_local = heap.pop() as usize;
        if closed[current_local] == stamp {
            continue;
        }
        closed[current_local] = stamp;
        if current_local == goal_local {
            let mut path = Vec::new();
            let mut p = goal_local as i32;
            while p >= 0 {
                path.push(to_global(p as usize));
                if came_from[p as usize] < 0 {
                    break;
                }
                p = came_from[p as usize];
            }
            path.reverse();
            return Some(path);
        }
        let current = to_global(current_local);
        let current_g = g[current_local];
        let cx = map.x(current);
        let cy = map.y(current);

        let try_relax = |nbr: TileRef, nlx: u32, nly: u32, n_local: usize,
                         closed: &mut [u32],
                         g_stamp: &mut [u32],
                         g: &mut [u32],
                         came_from: &mut [i32],
                         heap: &mut AstarHeap| {
            if closed[n_local] == stamp {
                return;
            }
            let terrain = map.terrain_byte(nbr);
            if nbr != goal && terrain & land_mask != 0 {
                return;
            }
            let magnitude = terrain & 0x1f;
            let cost = BASE_COST + bounded_magnitude_penalty(magnitude);
            let tentative = current_g.saturating_add(cost);
            if g_stamp[n_local] != stamp || tentative < g[n_local] {
                came_from[n_local] = current_local as i32;
                g[n_local] = tentative;
                g_stamp[n_local] = stamp;
                let h = heuristic_weight
                    * BASE_COST
                    * (nlx.abs_diff(goal_x) + nly.abs_diff(goal_y));
                let f = tentative + h + cross_tie(nlx, nly);
                heap.push(n_local as TileRef, f);
            }
        };

        if cy > min_y {
            let nbr = current - width;
            if let Some(n_local) = to_local(nbr) {
                try_relax(
                    nbr,
                    cx,
                    cy - 1,
                    n_local,
                    &mut closed,
                    &mut g_stamp,
                    &mut g,
                    &mut came_from,
                    &mut heap,
                );
            }
        }
        if cy < max_y {
            let nbr = current + width;
            if let Some(n_local) = to_local(nbr) {
                try_relax(
                    nbr,
                    cx,
                    cy + 1,
                    n_local,
                    &mut closed,
                    &mut g_stamp,
                    &mut g,
                    &mut came_from,
                    &mut heap,
                );
            }
        }
        if cx > min_x {
            let nbr = current - 1;
            if let Some(n_local) = to_local(nbr) {
                try_relax(
                    nbr,
                    cx - 1,
                    cy,
                    n_local,
                    &mut closed,
                    &mut g_stamp,
                    &mut g,
                    &mut came_from,
                    &mut heap,
                );
            }
        }
        if cx < max_x {
            let nbr = current + 1;
            if let Some(n_local) = to_local(nbr) {
                try_relax(
                    nbr,
                    cx + 1,
                    cy,
                    n_local,
                    &mut closed,
                    &mut g_stamp,
                    &mut g,
                    &mut came_from,
                    &mut heap,
                );
            }
        }
    }
    None
}

fn bounded_magnitude_penalty(magnitude: u8) -> u32 {
    if magnitude < 3 {
        3 * BASE_COST
    } else if magnitude <= 10 {
        0
    } else {
        BASE_COST
    }
}

/// TS `SmoothingWaterTransformer` on a mini-map water path.
pub fn smooth_water_path_pub(map: &GameMap, path: &[TileRef]) -> Vec<TileRef> {
    smooth_water_path(map, path)
}

pub fn water_trace_line_pub(map: &GameMap, from: TileRef, to: TileRef) -> Option<Vec<TileRef>> {
    water_trace_line(map, from, to)
}

fn smooth_water_path(map: &GameMap, path: &[TileRef]) -> Vec<TileRef> {
    if path.len() <= 2 {
        return path.to_vec();
    }
    let mut smoothed = los_smooth_water(map, path, 2);
    smoothed = refine_water_endpoints(map, &smoothed);
    if std::env::var("DEBUG_REFINE_ENDPOINTS").is_ok() {
        eprintln!("[native pass2 output] len={}", smoothed.len());
        for (i, &t) in smoothed.iter().enumerate() {
            eprintln!("  pass2[{}]={} x={} y={}", i, t, map.x(t), map.y(t));
        }
    }
    los_smooth_water(map, &smoothed, 3)
}

fn los_smooth_water(map: &GameMap, path: &[TileRef], min_magnitude: u8) -> Vec<TileRef> {
    let debug = std::env::var("DEBUG_LOS_SMOOTH").is_ok();
    let mut result = vec![path[0]];
    let mut current = 0usize;
    while current < path.len().saturating_sub(1) {
        let mut lo = current + 1;
        let mut hi = path.len() - 1;
        let mut farthest = lo;
        while lo <= hi {
            let mid = (lo + hi) / 2;
            if water_can_see(map, path[current], path[mid], min_magnitude) {
                farthest = mid;
                lo = mid + 1;
            } else {
                hi = mid.saturating_sub(1);
            }
        }
        if debug {
            eprintln!(
                "[native los minMag={}] current={} farthest={} from={} (x={},y={}) to={} (x={},y={})",
                min_magnitude, current, farthest, path[current], map.x(path[current]), map.y(path[current]),
                path[farthest], map.x(path[farthest]), map.y(path[farthest])
            );
        }
        if farthest > current + 1 {
            if let Some(trace) = water_trace_line(map, path[current], path[farthest]) {
                for &t in trace.iter().skip(1).take(trace.len().saturating_sub(2)) {
                    result.push(t);
                }
            }
        }
        current = farthest;
        if current < path.len().saturating_sub(1) {
            result.push(path[current]);
        }
    }
    if let Some(&last) = path.last() {
        result.push(last);
    }
    result
}

fn refine_water_endpoints(map: &GameMap, path: &[TileRef]) -> Vec<TileRef> {
    if path.len() <= 2 {
        return path.to_vec();
    }
    const REFINE_DIST: u32 = 50;
    const PADDING: u32 = 10;
    let mut result = path.to_vec();

    let start_end = tile_at_path_distance(map, path, 0, REFINE_DIST, true);
    if start_end > 1 {
        if let Some(seg) = bounded_segment_path(map, path[0], path[start_end], PADDING) {
            if !seg.is_empty() {
                result = seg[..seg.len().saturating_sub(1)]
                    .iter()
                    .copied()
                    .chain(result[start_end..].iter().copied())
                    .collect();
            }
        }
    }

    let end_start = tile_at_path_distance(map, &result, result.len() - 1, REFINE_DIST, false);
    if end_start + 2 < result.len() {
        if let Some(mut seg) =
            bounded_segment_path(map, *result.last().unwrap(), result[end_start], PADDING)
        {
            if !seg.is_empty() {
                seg.reverse();
                result = result[..end_start]
                    .iter()
                    .copied()
                    .chain(seg.iter().copied())
                    .collect();
            }
        }
    }
    result
}

fn tile_at_path_distance(map: &GameMap, path: &[TileRef], start: usize, dist: u32, forward: bool) -> usize {
    let mut cum = 0u32;
    let mut idx = start;
    if forward {
        while idx + 1 < path.len() && cum < dist {
            cum += map.manhattan_dist(path[idx], path[idx + 1]);
            idx += 1;
        }
    } else {
        while idx > 0 && cum < dist {
            cum += map.manhattan_dist(path[idx], path[idx - 1]);
            idx -= 1;
        }
    }
    idx
}

fn bounded_segment_path(map: &GameMap, from: TileRef, to: TileRef, padding: u32) -> Option<Vec<TileRef>> {
    let x0 = map.x(from);
    let y0 = map.y(from);
    let x1 = map.x(to);
    let y1 = map.y(to);
    let min_x = x0.min(x1).saturating_sub(padding);
    let max_x = (x0.max(x1) + padding).min(map.width - 1);
    let min_y = y0.min(y1).saturating_sub(padding);
    let max_y = (y0.max(y1) + padding).min(map.height - 1);
    bounded_water_path(map, &[from], to, min_x, max_x, min_y, max_y)
}

// TS `SmoothingWaterTransformer.canSee` - inline Bresenham with magnitude-aware
// diagonal handling: when the x-first intermediate tile has magnitude < min_magnitude,
// try the y-first alternative rather than failing immediately. This matches TS exactly.
fn water_can_see(map: &GameMap, from: TileRef, to: TileRef, min_magnitude: u8) -> bool {
    let mut x0 = map.x(from);
    let mut y0 = map.y(from);
    let x1 = map.x(to);
    let y1 = map.y(to);
    let dx = (x1 as i32 - x0 as i32).unsigned_abs();
    let dy = (y1 as i32 - y0 as i32).unsigned_abs();
    let sx = if x0 < x1 { 1i32 } else { -1i32 };
    let sy = if y0 < y1 { 1i32 } else { -1i32 };
    let mut err = dx as i32 - dy as i32;
    for _ in 0..100_000 {
        let tile = map.ref_xy(x0, y0);
        if !map.is_water(tile) {
            return false;
        }
        if (map.terrain_byte(tile) & 0x1f) < min_magnitude {
            return false;
        }
        if x0 == x1 && y0 == y1 {
            return true;
        }
        let e2 = 2 * err;
        let should_move_x = e2 > -(dy as i32);
        let should_move_y = e2 < dx as i32;
        if should_move_x && should_move_y {
            let nx = (x0 as i32 + sx) as u32;
            let intermediate = map.ref_xy(nx, y0);
            let int_mag = map.terrain_byte(intermediate) & 0x1f;
            if !map.is_water(intermediate) || int_mag < min_magnitude {
                // x-first fails magnitude check, try y-first alternative (matches TS)
                let ny = (y0 as i32 + sy) as u32;
                err += dx as i32;
                let alt = map.ref_xy(x0, ny);
                let alt_mag = map.terrain_byte(alt) & 0x1f;
                if !map.is_water(alt) || alt_mag < min_magnitude {
                    return false;
                }
                x0 = nx;
                err -= dy as i32;
                y0 = ny;
            } else {
                x0 = nx;
                err -= dy as i32;
                y0 = (y0 as i32 + sy) as u32;
                err += dx as i32;
            }
        } else {
            if should_move_x {
                x0 = (x0 as i32 + sx) as u32;
                err -= dy as i32;
            }
            if should_move_y {
                y0 = (y0 as i32 + sy) as u32;
                err += dx as i32;
            }
        }
    }
    false
}

fn water_trace_line(map: &GameMap, from: TileRef, to: TileRef) -> Option<Vec<TileRef>> {
    let w = map.width;
    let mut x0 = map.x(from);
    let mut y0 = map.y(from);
    let x1 = map.x(to);
    let y1 = map.y(to);
    let dx = (x1 as i32 - x0 as i32).unsigned_abs();
    let dy = (y1 as i32 - y0 as i32).unsigned_abs();
    let sx = if x0 < x1 { 1i32 } else { -1i32 };
    let sy = if y0 < y1 { 1i32 } else { -1i32 };
    let mut err = dx as i32 - dy as i32;
    let mut tiles = Vec::new();
    for _ in 0..100_000 {
        let tile = map.ref_xy(x0, y0);
        if !map.is_water(tile) {
            return None;
        }
        tiles.push(tile);
        if x0 == x1 && y0 == y1 {
            return Some(tiles);
        }
        let e2 = 2 * err;
        let should_move_x = e2 > -(dy as i32);
        let should_move_y = e2 < dx as i32;
        if should_move_x && should_move_y {
            x0 = (x0 as i32 + sx) as u32;
            err -= dy as i32;
            let intermediate = map.ref_xy(x0, y0);
            if !map.is_water(intermediate) {
                x0 = (x0 as i32 - sx) as u32;
                err += dy as i32;
                y0 = (y0 as i32 + sy) as u32;
                err += dx as i32;
                let alt = map.ref_xy(x0, y0);
                if !map.is_water(alt) {
                    return None;
                }
                tiles.push(alt);
                x0 = (x0 as i32 + sx) as u32;
                err -= dy as i32;
            } else {
                tiles.push(intermediate);
                y0 = (y0 as i32 + sy) as u32;
                err += dx as i32;
            }
        } else {
            if should_move_x {
                x0 = (x0 as i32 + sx) as u32;
                err -= dy as i32;
            }
            if should_move_y {
                y0 = (y0 as i32 + sy) as u32;
                err += dx as i32;
            }
        }
    }
    None
}

/// TS `MiniMapTransformer` upscale (scale factor 2).
fn upscale_cells(cells: &[(u32, u32)], scale: u32) -> Vec<(u32, u32)> {
    if cells.is_empty() {
        return Vec::new();
    }
    let scaled: Vec<(u32, u32)> = cells
        .iter()
        .map(|&(x, y)| (x * scale, y * scale))
        .collect();
    let mut smooth = Vec::new();
    for i in 0..scaled.len().saturating_sub(1) {
        let current = scaled[i];
        let next = scaled[i + 1];
        smooth.push(current);
        let dx = next.0 as i32 - current.0 as i32;
        let dy = next.1 as i32 - current.1 as i32;
        let steps = dx.abs().max(dy.abs()) as u32;
        if steps <= 1 {
            continue;
        }
        for step in 1..steps {
            smooth.push((
                (current.0 as f64 + (dx as f64 * step as f64) / steps as f64).round() as u32,
                (current.1 as f64 + (dy as f64 * step as f64) / steps as f64).round() as u32,
            ));
        }
    }
    if let Some(last) = scaled.last() {
        smooth.push(*last);
    }
    smooth
}

fn to_mini_ref(full: &GameMap, mini: &GameMap, tile: TileRef) -> TileRef {
    mini.ref_xy(full.x(tile) / 2, full.y(tile) / 2)
}

fn fix_path_extremes(
    full: &GameMap,
    mut path: Vec<TileRef>,
    cell_src: Option<TileRef>,
    cell_dst: TileRef,
) -> Vec<TileRef> {
    // TS `MiniMapTransformer.fixExtremes` uses `indexOf` (tile-ref equality).
    if let Some(src) = cell_src {
        if let Some(idx) = path.iter().position(|&t| t == src) {
            if idx != 0 {
                path = path[idx..].to_vec();
            }
        } else {
            path.insert(0, src);
        }
    }
    if let Some(idx) = path.iter().position(|&t| t == cell_dst) {
        if idx + 1 != path.len() {
            path.truncate(idx + 1);
        }
    } else {
        path.push(cell_dst);
    }
    // TS `MiniMapTransformer.fixExtremes` does not dedupe consecutive nodes;
    // a duplicate tile makes `PathFinderStepper.next` stall the ship for one
    // tick. Collapsing those dupes made native transports arrive one tick
    // early (seen on curriculum-parity-v4 `curr-b050-s3-world` boat 385:
    // native reversed off a duplicated shore cell immediately while TS spent
    // the stall tick on the duplicate). Match TS and keep the stall.
    path
}

/// Ensure each consecutive pair is 4-neighbor adjacent (TS stepper assumption).
fn densify_path_adjacent(map: &GameMap, path: Vec<TileRef>) -> Vec<TileRef> {
    if path.len() <= 1 {
        return path;
    }
    let mut out = vec![path[0]];
    for &to in &path[1..] {
        let from = *out.last().unwrap();
        if from == to {
            continue;
        }
        if map.manhattan_dist(from, to) == 1 {
            out.push(to);
            continue;
        }
        if let Some(trace) = water_trace_line(map, from, to) {
            for &t in trace.iter().skip(1) {
                if out.last() != Some(&t) {
                    out.push(t);
                }
            }
        } else {
            out.push(to);
        }
    }
    out
}

fn closest_full_source(full: &GameMap, froms: &[TileRef], path_start: TileRef) -> Option<TileRef> {
    if froms.len() == 1 {
        return Some(froms[0]);
    }
    let px = full.x(path_start);
    let py = full.y(path_start);
    let mut best: Option<TileRef> = None;
    let mut best_dist = u32::MAX;
    for &f in froms {
        let dist = full.x(f).abs_diff(px) + full.y(f).abs_diff(py);
        if dist < best_dist {
            best_dist = dist;
            best = Some(f);
        }
    }
    best
}

fn try_short_water_path_multi(
    map: &GameMap,
    starts: &[TileRef],
    goal: TileRef,
    out: &mut Vec<TileRef>,
) -> bool {
    const SHORT_PATH_THRESHOLD: u32 = 120;
    const PADDING: u32 = 10;
    let candidates: Vec<TileRef> = starts
        .iter()
        .copied()
        .filter(|&s| map.manhattan_dist(s, goal) <= SHORT_PATH_THRESHOLD)
        .collect();
    if candidates.is_empty() {
        return false;
    }
    let goal_x = map.x(goal);
    let goal_y = map.y(goal);
    let mut min_x = goal_x;
    let mut max_x = goal_x;
    let mut min_y = goal_y;
    let mut max_y = goal_y;
    for s in &candidates {
        let sx = map.x(*s);
        let sy = map.y(*s);
        min_x = min_x.min(sx);
        max_x = max_x.max(sx);
        min_y = min_y.min(sy);
        max_y = max_y.max(sy);
    }
    if let Some(path) = bounded_water_path(
        map,
        &candidates,
        goal,
        min_x.saturating_sub(PADDING),
        (max_x + PADDING).min(map.width - 1),
        min_y.saturating_sub(PADDING),
        (max_y + PADDING).min(map.height - 1),
    ) {
        *out = path;
        true
    } else {
        false
    }
}

/// Shore coercing + mini-map path (simple A* or HPA) + upscale chain.
fn minimap_inner_path(
    mini_map: &GameMap,
    mini_astar: &mut WaterAstarScratch,
    mut hpa: Option<&mut crate::water_hpa::WaterHierarchical>,
    mini_froms: &[TileRef],
    mini_to: TileRef,
    mini_path_out: &mut Vec<TileRef>,
) -> bool {
    mini_path_out.clear();
    let mut water_starts = Vec::with_capacity(mini_froms.len());
    let mut water_to_orig: std::collections::HashMap<TileRef, Option<TileRef>> =
        std::collections::HashMap::new();
    for &mf in mini_froms {
        let Some(water) = coerce_shore_to_water(mini_map, mf) else {
            continue;
        };
        let orig = if mini_map.is_water(mf) {
            None
        } else {
            Some(mf)
        };
        // TS ShoreCoercingTransformer Map always overwrites, including null originals.
        water_to_orig.insert(water, orig);
        water_starts.push(water);
    }
    if water_starts.is_empty() {
        return false;
    }
    let goal_water = match coerce_shore_to_water(mini_map, mini_to) {
        Some(w) => w,
        None => return false,
    };
    let to_original = if mini_map.is_water(mini_to) {
        None
    } else {
        Some(mini_to)
    };

    // TS `ComponentCheckTransformer`  -  filter sources to destination component.
    if let Some(hpa) = hpa.as_ref() {
        let to_comp = hpa.graph.get_component_id(goal_water);
        water_starts.retain(|&s| hpa.graph.get_component_id(s) == to_comp);
        if water_starts.is_empty() {
            return false;
        }
    }

    let graph_large = hpa
        .as_ref()
        .is_some_and(|h| h.graph.node_count() >= 100);

    // Match TS sharedWaterChain: use HPA only when graph has 100+ nodes.
    let use_hpa = graph_large;

    let found = if water_starts.len() == 1 && water_starts[0] == goal_water {
        // TS AStar.Water(W,W) → [water], then ShoreCoercing restores shores below.
        mini_path_out.push(goal_water);
        true
    } else if use_hpa {
        let hpa = hpa.as_mut().unwrap();
        match hpa.find_path(mini_map, &water_starts, goal_water) {
            Some(p) => {
                *mini_path_out = p;
                true
            }
            None => try_short_water_path_multi(mini_map, &water_starts, goal_water, mini_path_out)
                || astar_water_path_into(
                    mini_map,
                    mini_astar,
                    &water_starts,
                    goal_water,
                    mini_path_out,
                ),
        }
    } else if !astar_water_path_into(mini_map, mini_astar, &water_starts, goal_water, mini_path_out) {
        false
    } else {
        true
    };
    if !found {
        return false;
    }

    if graph_large {
        let smoothed = smooth_water_path(mini_map, mini_path_out);
        *mini_path_out = smoothed;
    }

    if let Some(&first_water) = mini_path_out.first() {
        if let Some(&Some(orig)) = water_to_orig.get(&first_water) {
            mini_path_out.insert(0, orig);
        }
    }
    if let Some(orig) = to_original {
        if mini_path_out.last() != Some(&orig) {
            mini_path_out.push(orig);
        }
    }
    true
}

fn upscale_mini_path(full: &GameMap, mini: &GameMap, mini_path: &[TileRef]) -> Vec<TileRef> {
    let cells: Vec<(u32, u32)> = mini_path
        .iter()
        .map(|&r| (mini.x(r), mini.y(r)))
        .collect();
    upscale_cells(&cells, 2)
        .into_iter()
        .map(|(cx, cy)| full.ref_xy(cx, cy))
        .collect()
}

/// TS `PathFinding.Water` transport path on full-map tile refs.
pub fn transport_path_multi_into(
    full: &GameMap,
    mini: &GameMap,
    mini_astar: &mut WaterAstarScratch,
    mut hpa: Option<&mut crate::water_hpa::WaterHierarchical>,
    froms: &[TileRef],
    to: TileRef,
    path_out: &mut Vec<TileRef>,
) -> bool {
    path_out.clear();
    if froms.is_empty() {
        return false;
    }
    let mini_froms: Vec<TileRef> = froms
        .iter()
        .map(|&f| to_mini_ref(full, mini, f))
        .collect();
    let mini_to = to_mini_ref(full, mini, to);
    let mut mini_path = Vec::with_capacity(64);
    if !minimap_inner_path(mini, mini_astar, hpa, &mini_froms, mini_to, &mut mini_path) {
        return false;
    }
    let upscaled = upscale_mini_path(full, mini, &mini_path);
    if upscaled.is_empty() {
        return false;
    }
    // TS `MiniMapTransformer.findPath`  -  closest source to upscaled[0], no mini-shore filter.
    let cell_src = closest_full_source(full, froms, upscaled[0]);
    // TS `MiniMapTransformer` does NOT densify to 4-adjacency; the upscaled cells are
    // already interpolated to be adjacent for cardinal minimap steps, and the source/
    // destination fix can add a diagonal step (when both x and y are odd) that TS
    // traverses in one move. Densifying that diagonal into two cardinal steps puts
    // native one tick behind TS, explaining the recurring hash diffs of exactly
    // one map-width (e.g. 2200). Use the path as-is to match TS behaviour.
    *path_out = fix_path_extremes(full, upscaled, cell_src, to);
    !path_out.is_empty()
}

/// TS `ShoreCoercingTransformer` + `MiniMapTransformer` + `AStar.Water` for transport ships.
pub fn transport_path_into(
    full: &GameMap,
    mini: &GameMap,
    mini_astar: &mut WaterAstarScratch,
    mut hpa: Option<&mut crate::water_hpa::WaterHierarchical>,
    from: TileRef,
    to: TileRef,
    path_out: &mut Vec<TileRef>,
) -> bool {
    transport_path_multi_into(full, mini, mini_astar, hpa, &[from], to, path_out)
}

/// Back-compat alias for tests.
pub fn shore_transport_path_into(
    full: &GameMap,
    _astar: &mut WaterAstarScratch,
    from_shore: TileRef,
    to_shore: TileRef,
    path_out: &mut Vec<TileRef>,
) -> bool {
    path_out.clear();
    let Some(start_water) = coerce_shore_to_water(full, from_shore) else {
        return false;
    };
    let Some(goal_water) = coerce_shore_to_water(full, to_shore) else {
        return false;
    };
    if start_water == goal_water {
        path_out.push(to_shore);
        return true;
    }
    if !astar_water_path_into(full, _astar, &[start_water], goal_water, path_out) {
        return false;
    }
    if path_out.last() != Some(&to_shore) {
        path_out.push(to_shore);
    }
    true
}
