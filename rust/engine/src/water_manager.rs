//! TS `WaterManager.ts` - batched land→water conversion for `waterNukes`.
//!
//! When `Config.waterNukes()` is true, nuke detonation queues land tiles here
//! instead of painting fallout. `tick` flushes the queue: convert terrain,
//! propagate ocean bits, recompute water magnitude near the crater, fix
//! shoreline bits, and update the mini-map (majority-water 2×2 cells → water).
//! Mini water-graph rebuild is throttled to once every 20 ticks (TS).

use crate::map::{GameMap, TileRef};
use std::collections::HashSet;

const WATER_GRAPH_REBUILD_INTERVAL: u32 = 20;
const MAX_MAG_DIST: i32 = 62;

#[derive(Debug, Default)]
pub struct WaterManager {
    pending: Vec<TileRef>,
    pending_set: HashSet<TileRef>,
    water_dist: Vec<u16>,
    water_stamp_arr: Vec<u16>,
    water_stamp: u16,
    dirty_mini: HashSet<TileRef>,
    water_graph_dirty: bool,
    water_graph_last_rebuild_tick: u32,
}

impl WaterManager {
    pub fn queue_tile(&mut self, tile: TileRef) {
        if self.pending_set.insert(tile) {
            self.pending.push(tile);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty() && !self.water_graph_dirty
    }

    /// Returns tiles whose terrain bytes changed (for client tile updates; unused by hash).
    /// `rebuild_hpa` is set when the throttled mini-graph rebuild should run.
    pub fn tick(
        &mut self,
        map: &mut GameMap,
        mini_map: &mut GameMap,
        current_tick: u32,
    ) -> (Vec<TileRef>, bool) {
        let mut changed_tiles = Vec::new();
        if !self.pending.is_empty() {
            let pending = std::mem::take(&mut self.pending);
            self.pending_set.clear();
            let mut converted = Vec::new();
            for tile in pending {
                if map.is_land(tile) && map.owner_id(tile) == 0 {
                    if map.has_fallout(tile) {
                        map.set_fallout(tile, false);
                    }
                    map.set_water(tile);
                    converted.push(tile);
                }
            }
            if !converted.is_empty() {
                self.finalize_water_changes(map, mini_map, &converted, &mut changed_tiles);
            }
        }

        let mut rebuild_hpa = false;
        if self.water_graph_dirty
            && current_tick.saturating_sub(self.water_graph_last_rebuild_tick)
                >= WATER_GRAPH_REBUILD_INTERVAL
        {
            self.water_graph_dirty = false;
            self.water_graph_last_rebuild_tick = current_tick;
            self.dirty_mini.clear();
            rebuild_hpa = true;
        }
        (changed_tiles, rebuild_hpa)
    }

    fn finalize_water_changes(
        &mut self,
        map: &mut GameMap,
        mini_map: &mut GameMap,
        converted_tiles: &[TileRef],
        changed_tiles: &mut Vec<TileRef>,
    ) {
        let converted: HashSet<TileRef> = converted_tiles.iter().copied().collect();
        if converted.is_empty() {
            return;
        }
        let mut changed: HashSet<TileRef> = converted_tiles.iter().copied().collect();
        let w = map.width;
        let h = map.height;
        let total = (w * h) as usize;

        // ── 1. Propagate ocean bit ─────────────────────────────────────
        let mut ocean_queue = Vec::new();
        for &tile in converted_tiles {
            let mut nb = [0u32; 4];
            let n = push_neighbors_nswe(map, tile, &mut nb);
            for i in 0..n {
                if !converted.contains(&nb[i]) && map.is_ocean(nb[i]) {
                    map.set_ocean(tile);
                    ocean_queue.push(tile);
                    break;
                }
            }
        }
        let mut o_head = 0;
        while o_head < ocean_queue.len() {
            let tile = ocean_queue[o_head];
            o_head += 1;
            let mut nb = [0u32; 4];
            let n = push_neighbors_nswe(map, tile, &mut nb);
            for i in 0..n {
                let ntile = nb[i];
                if map.is_water(ntile) && !map.is_ocean(ntile) {
                    map.set_ocean(ntile);
                    changed.insert(ntile);
                    ocean_queue.push(ntile);
                }
            }
        }

        // ── 2. Recompute magnitude via BFS ─────────────────────────────
        if self.water_dist.len() != total {
            self.water_dist = vec![0u16; total];
            self.water_stamp_arr = vec![0u16; total];
            self.water_stamp = 0;
        }
        self.water_stamp = self.water_stamp.wrapping_add(1);
        if self.water_stamp == 0 {
            self.water_stamp_arr.fill(0);
            self.water_stamp = 1;
        }
        let stamp = self.water_stamp;

        let mut c_min_x = w as i32;
        let mut c_max_x = 0i32;
        let mut c_min_y = h as i32;
        let mut c_max_y = 0i32;
        for &tile in converted_tiles {
            let tx = (tile % w) as i32;
            let ty = (tile / w) as i32;
            c_min_x = c_min_x.min(tx);
            c_max_x = c_max_x.max(tx);
            c_min_y = c_min_y.min(ty);
            c_max_y = c_max_y.max(ty);
        }
        let d_min_x = (c_min_x - MAX_MAG_DIST).max(0);
        let d_max_x = (c_max_x + MAX_MAG_DIST).min(w as i32 - 1);
        let d_min_y = (c_min_y - MAX_MAG_DIST).max(0);
        let d_max_y = (c_max_y + MAX_MAG_DIST).min(h as i32 - 1);
        let s_min_x = (c_min_x - MAX_MAG_DIST * 2).max(0);
        let s_max_x = (c_max_x + MAX_MAG_DIST * 2).min(w as i32 - 1);
        let s_min_y = (c_min_y - MAX_MAG_DIST * 2).max(0);
        let s_max_y = (c_max_y + MAX_MAG_DIST * 2).min(h as i32 - 1);

        let mut mag_queue = Vec::new();
        for by in s_min_y..=s_max_y {
            for bx in s_min_x..=s_max_x {
                let tile = (by as u32) * w + (bx as u32);
                if !map.is_water(tile) || self.water_stamp_arr[tile as usize] == stamp {
                    continue;
                }
                let mut nb = [0u32; 4];
                let n = push_neighbors_nswe(map, tile, &mut nb);
                let touches_land = (0..n).any(|i| map.is_land(nb[i]));
                if touches_land {
                    self.water_stamp_arr[tile as usize] = stamp;
                    self.water_dist[tile as usize] = 0;
                    mag_queue.push(tile);
                }
            }
        }

        let mut mag_head = 0;
        while mag_head < mag_queue.len() {
            let tile = mag_queue[mag_head];
            mag_head += 1;
            let dist = self.water_dist[tile as usize];
            let next_dist = dist + 1;
            let mut nb = [0u32; 4];
            let n = push_neighbors_nswe(map, tile, &mut nb);
            for i in 0..n {
                let ntile = nb[i];
                if !map.is_water(ntile) || self.water_stamp_arr[ntile as usize] == stamp {
                    continue;
                }
                let nx = (ntile % w) as i32;
                let ny = (ntile / w) as i32;
                if nx < s_min_x || nx > s_max_x || ny < s_min_y || ny > s_max_y {
                    continue;
                }
                self.water_stamp_arr[ntile as usize] = stamp;
                self.water_dist[ntile as usize] = next_dist;
                mag_queue.push(ntile);
            }
        }

        for dy in d_min_y..=d_max_y {
            for dx in d_min_x..=d_max_x {
                let tile = (dy as u32) * w + (dx as u32);
                if !map.is_water(tile) {
                    continue;
                }
                let old_mag = map.magnitude(tile);
                let new_mag = if self.water_stamp_arr[tile as usize] == stamp {
                    ((self.water_dist[tile as usize] as f64) / 2.0)
                        .ceil()
                        .min(31.0) as u8
                } else {
                    31
                };
                if old_mag != new_mag {
                    map.set_magnitude(tile, new_mag);
                    changed.insert(tile);
                }
            }
        }

        // ── 3. Fix shoreline bits ──────────────────────────────────────
        let mut tiles_to_check: HashSet<TileRef> = HashSet::new();
        for &tile in converted_tiles {
            tiles_to_check.insert(tile);
            let mut nb = [0u32; 4];
            let n = push_neighbors_nswe(map, tile, &mut nb);
            for i in 0..n {
                tiles_to_check.insert(nb[i]);
                let mut nb2 = [0u32; 4];
                let n2 = push_neighbors_nswe(map, nb[i], &mut nb2);
                for j in 0..n2 {
                    tiles_to_check.insert(nb2[j]);
                }
            }
        }
        for &tile in &tiles_to_check {
            let tile_is_land = map.is_land(tile);
            let mut nb = [0u32; 4];
            let n = push_neighbors_nswe(map, tile, &mut nb);
            let has_opposite = (0..n).any(|i| map.is_land(nb[i]) != tile_is_land);
            let old_shoreline = map.is_shoreline(tile);
            if has_opposite {
                if !old_shoreline {
                    map.set_shoreline_bit(tile);
                    changed.insert(tile);
                }
            } else if old_shoreline {
                map.clear_shoreline_bit(tile);
                changed.insert(tile);
            }
        }

        // ── 4. Update minimap terrain ──────────────────────────────────
        let mut mini_tiles_to_check: HashSet<TileRef> = HashSet::new();
        for &tile in converted_tiles {
            let mini_x = map.x(tile) / 2;
            let mini_y = map.y(tile) / 2;
            if mini_map.is_valid_coord(mini_x as i32, mini_y as i32) {
                mini_tiles_to_check.insert(mini_map.ref_xy(mini_x, mini_y));
            }
        }
        let mut converted_mini = HashSet::new();
        for &mini_tile in &mini_tiles_to_check {
            if !mini_map.is_land(mini_tile) {
                continue;
            }
            let fx = mini_map.x(mini_tile) * 2;
            let fy = mini_map.y(mini_tile) * 2;
            let mut water_count = 0u32;
            let mut total_count = 0u32;
            for dy in 0..2u32 {
                for dx in 0..2u32 {
                    if map.is_valid_coord((fx + dx) as i32, (fy + dy) as i32) {
                        total_count += 1;
                        if map.is_water(map.ref_xy(fx + dx, fy + dy)) {
                            water_count += 1;
                        }
                    }
                }
            }
            if water_count >= total_count.min(3) {
                mini_map.set_water(mini_tile);
                converted_mini.insert(mini_tile);
            }
        }

        if !converted_mini.is_empty() {
            self.water_graph_dirty = true;
            for mt in converted_mini {
                self.dirty_mini.insert(mt);
            }
        }

        changed_tiles.extend(changed);
    }
}

/// TS WaterManager `pushNeighbors` - N, S, W, E (same as `GameMap.neighbors()`).
fn push_neighbors_nswe(map: &GameMap, tile: TileRef, out: &mut [TileRef; 4]) -> usize {
    let w = map.width;
    let total = w * map.height;
    let mut n = 0;
    if tile >= w {
        out[n] = tile - w;
        n += 1;
    }
    if tile < total - w {
        out[n] = tile + w;
        n += 1;
    }
    let x = tile % w;
    if x > 0 {
        out[n] = tile - 1;
        n += 1;
    }
    if x < w - 1 {
        out[n] = tile + 1;
        n += 1;
    }
    n
}
