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

/// Stamp BFS on land tiles. Returns true if a matching tile was found.
pub fn land_bfs_nearest(
    map: &GameMap,
    scratch: &mut BfsScratch,
    from: TileRef,
    max_dist: u32,
    mut pred: impl FnMut(TileRef) -> bool,
) -> Option<TileRef> {
    let stamp = scratch.next_stamp();
    let mut best: Option<(TileRef, u32)> = None;
    let mut q = VecDeque::from([(from, 0u32)]);
    scratch.seen[from as usize] = stamp;

    while let Some((t, _)) = q.pop_front() {
        if pred(t) {
            let dist = map.manhattan_dist(from, t);
            if best.map(|(_, bd)| dist < bd).unwrap_or(true) {
                best = Some((t, dist));
            }
        }
        map.for_each_neighbor4(t, |n| {
            if scratch.seen[n as usize] != stamp
                && map.is_land(n)
                && map.manhattan_dist(from, n) <= max_dist
            {
                scratch.seen[n as usize] = stamp;
                q.push_back((n, 0));
            }
        });
    }
    best.map(|(t, _)| t)
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

fn count_water_neighbors(map: &GameMap, tile: TileRef) -> u32 {
    let mut count = 0u32;
    map.for_each_neighbor4(tile, |n| {
        if map.is_water(n) {
            count += 1;
        }
    });
    count
}

/// TS `ShoreCoercingTransformer.coerceToWater` - pick highest-connectivity adjacent water.
pub fn coerce_shore_to_water(map: &GameMap, tile: TileRef) -> Option<TileRef> {
    if map.is_water(tile) {
        return Some(tile);
    }
    let mut best: Option<TileRef> = None;
    let mut max_score = -1i32;
    map.for_each_neighbor4(tile, |n| {
        if !map.is_water(n) {
            return;
        }
        let score = count_water_neighbors(map, n) as i32;
        if score > max_score {
            max_score = score;
            best = Some(n);
        }
    });
    best
}

struct AstarHeap {
    f: Vec<u32>,
    tile: Vec<TileRef>,
}

impl AstarHeap {
    fn new(cap: usize) -> Self {
        Self {
            f: Vec::with_capacity(cap),
            tile: Vec::with_capacity(cap),
        }
    }

    fn clear(&mut self) {
        self.f.clear();
        self.tile.clear();
    }

    fn is_empty(&self) -> bool {
        self.f.is_empty()
    }

    fn push(&mut self, tile: TileRef, priority: u32) {
        let mut i = self.f.len();
        self.f.push(0);
        self.tile.push(0);
        while i > 0 {
            let parent = (i - 1) >> 1;
            if priority >= self.f[parent] {
                break;
            }
            self.f[i] = self.f[parent];
            self.tile[i] = self.tile[parent];
            i = parent;
        }
        self.f[i] = priority;
        self.tile[i] = tile;
    }

    fn pop(&mut self) -> TileRef {
        let top = self.tile[0];
        let last_f = self.f.pop().unwrap();
        let last_tile = self.tile.pop().unwrap();
        if self.f.is_empty() {
            return top;
        }
        let mut i = 0usize;
        loop {
            let left = (i << 1) + 1;
            if left >= self.f.len() {
                break;
            }
            let right = left + 1;
            let mut smallest = left;
            if right < self.f.len() && self.f[right] < self.f[left] {
                smallest = right;
            }
            if last_f <= self.f[smallest] {
                break;
            }
            self.f[i] = self.f[smallest];
            self.tile[i] = self.tile[smallest];
            i = smallest;
        }
        self.f[i] = last_f;
        self.tile[i] = last_tile;
        top
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
        ((cross * (COST_SCALE - 1)) / cross_norm / cross_norm) as u32
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

    let goal_local = to_local(goal)?;
    heap.clear();
    for &s in starts {
        let Some(start_local) = to_local(s) else {
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
                heap.push(n_local as TileRef, tentative + h);
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
                (current.0 as i32 + (dx * step as i32) / steps as i32) as u32,
                (current.1 as i32 + (dy * step as i32) / steps as i32) as u32,
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
    if let Some(src) = cell_src {
        let sx = full.x(src);
        let sy = full.y(src);
        if let Some(idx) = path
            .iter()
            .position(|&t| full.x(t) == sx && full.y(t) == sy)
        {
            if idx != 0 {
                path = path[idx..].to_vec();
            }
        } else {
            path.insert(0, src);
        }
    }
    let dx = full.x(cell_dst);
    let dy = full.y(cell_dst);
    if let Some(idx) = path
        .iter()
        .position(|&t| full.x(t) == dx && full.y(t) == dy)
    {
        if idx + 1 != path.len() {
            path.truncate(idx + 1);
        }
    } else {
        path.push(cell_dst);
    }
    path
}

fn closest_full_source(full: &GameMap, froms: &[TileRef], path_start: TileRef) -> Option<TileRef> {
    if froms.len() == 1 {
        return Some(froms[0]);
    }
    let px = full.x(path_start);
    let py = full.y(path_start);
    froms
        .iter()
        .min_by_key(|&&f| {
            let fx = full.x(f);
            let fy = full.y(f);
            fx.abs_diff(px) + fy.abs_diff(py)
        })
        .copied()
}

/// Shore coercing + mini-map A* + upscale (TS simple `PathFinding.Water` chain).
fn minimap_inner_path(
    mini_map: &GameMap,
    mini_astar: &mut WaterAstarScratch,
    mini_froms: &[TileRef],
    mini_to: TileRef,
    mini_path_out: &mut Vec<TileRef>,
) -> bool {
    mini_path_out.clear();
    let mut water_starts = Vec::with_capacity(mini_froms.len());
    let mut water_to_orig: std::collections::HashMap<TileRef, TileRef> =
        std::collections::HashMap::new();
    for &mf in mini_froms {
        let Some(water) = coerce_shore_to_water(mini_map, mf) else {
            continue;
        };
        if !mini_map.is_water(mf) {
            water_to_orig.insert(water, mf);
        }
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

    if water_starts.len() == 1 && water_starts[0] == goal_water {
        mini_path_out.push(mini_to);
        return true;
    }

    if !astar_water_path_into(mini_map, mini_astar, &water_starts, goal_water, mini_path_out) {
        return false;
    }

    if let Some(&first_water) = mini_path_out.first() {
        if let Some(&orig) = water_to_orig.get(&first_water) {
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
    if !minimap_inner_path(mini, mini_astar, &mini_froms, mini_to, &mut mini_path) {
        return false;
    }
    let upscaled = upscale_mini_path(full, mini, &mini_path);
    if upscaled.is_empty() {
        return false;
    }
    let cell_src = closest_full_source(full, froms, upscaled[0]);
    *path_out = fix_path_extremes(full, upscaled, cell_src, to);
    !path_out.is_empty()
}

/// TS `ShoreCoercingTransformer` + `MiniMapTransformer` + `AStar.Water` for transport ships.
pub fn transport_path_into(
    full: &GameMap,
    mini: &GameMap,
    mini_astar: &mut WaterAstarScratch,
    from: TileRef,
    to: TileRef,
    path_out: &mut Vec<TileRef>,
) -> bool {
    transport_path_multi_into(full, mini, mini_astar, &[from], to, path_out)
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
