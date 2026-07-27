//! Terrain + per-tile state (`GameMapImpl` subset).

use serde::Deserialize;
use std::path::Path;

pub type TileRef = u32;

const IS_LAND_BIT: u8 = 7;
const OCEAN_BIT: u8 = 5;
const SHORELINE_BIT: u8 = 6;
const MAGNITUDE_MASK: u8 = 0x1f;
const IMPASSABLE_MAGNITUDE: u8 = 31;
const FALLOUT_BIT: u16 = 13;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainType {
    Plains,
    Highland,
    Mountain,
    Impassable,
    Ocean,
}

#[derive(Debug, Clone)]
pub struct GameMap {
    pub width: u32,
    pub height: u32,
    pub num_land_tiles: u32,
    terrain: Vec<u8>,
    /// Packed uint16 per tile: low 12 bits owner, bit 13 fallout, etc.
    state: Vec<u16>,
    num_tiles_with_fallout: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MapManifest {
    pub name: String,
    pub map: MapMeta,
    pub map4x: MapMeta,
    pub map16x: MapMeta,
    pub nations: Vec<Nation>,
    #[serde(default, rename = "additionalNations")]
    pub additional_nations: Vec<Nation>,
    #[serde(default, rename = "teamGameSpawnAreas")]
    pub team_game_spawn_areas: Option<std::collections::HashMap<String, Vec<SpawnArea>>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MapMeta {
    pub width: u32,
    pub height: u32,
    pub num_land_tiles: u32,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct Nation {
    pub name: String,
    #[serde(default)]
    pub flag: Option<String>,
    #[serde(default)]
    pub coordinates: Option<[i32; 2]>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SpawnArea {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl GameMap {
    pub fn from_terrain_bytes(meta: &MapMeta, data: &[u8]) -> Result<Self, String> {
        let n = (meta.width * meta.height) as usize;
        if data.len() != n {
            return Err(format!(
                "terrain {} bytes != {}x{}",
                data.len(),
                meta.width,
                meta.height
            ));
        }
        Ok(Self {
            width: meta.width,
            height: meta.height,
            num_land_tiles: meta.num_land_tiles,
            terrain: data.to_vec(),
            state: vec![0; n],
            num_tiles_with_fallout: 0,
        })
    }

    pub fn load_map_dir(map_dir: &Path) -> Result<(MapManifest, Self), String> {
        let manifest = read_manifest(map_dir)?;
        let data =
            std::fs::read(map_dir.join("map.bin")).map_err(|e| format!("map.bin: {e}"))?;
        let gm = Self::from_terrain_bytes(&manifest.map, &data)?;
        Ok((manifest, gm))
    }

    pub fn ref_xy(&self, x: u32, y: u32) -> TileRef {
        y * self.width + x
    }

    pub fn x(&self, t: TileRef) -> u32 {
        t % self.width
    }

    pub fn y(&self, t: TileRef) -> u32 {
        t / self.width
    }

    pub fn terrain_byte(&self, t: TileRef) -> u8 {
        self.terrain[t as usize]
    }

    /// Raw terrain byte plane, one entry per tile. Lets callers that want
    /// every tile (e.g. `session::terrain_bytes`) copy the whole buffer at
    /// memcpy speed instead of looping `terrain_byte(i)` one tile at a time.
    pub fn terrain_bytes(&self) -> &[u8] {
        &self.terrain
    }

    pub fn is_valid_coord(&self, x: i32, y: i32) -> bool {
        x >= 0 && y >= 0 && (x as u32) < self.width && (y as u32) < self.height
    }

    pub fn is_land(&self, t: TileRef) -> bool {
        self.terrain_byte(t) & (1 << IS_LAND_BIT) != 0
    }

    pub fn is_water(&self, t: TileRef) -> bool {
        !self.is_land(t)
    }

    pub fn is_ocean(&self, t: TileRef) -> bool {
        self.terrain_byte(t) & (1 << OCEAN_BIT) != 0
    }

    pub fn is_shoreline(&self, t: TileRef) -> bool {
        self.terrain_byte(t) & (1 << SHORELINE_BIT) != 0
    }

    pub fn is_shore(&self, t: TileRef) -> bool {
        self.is_land(t) && self.is_shoreline(t)
    }

    pub fn has_fallout(&self, t: TileRef) -> bool {
        self.tile_state(t) & (1 << FALLOUT_BIT) != 0
    }

    pub fn num_tiles_with_fallout(&self) -> u32 {
        self.num_tiles_with_fallout
    }

    pub fn set_fallout(&mut self, t: TileRef, has: bool) {
        let had = self.has_fallout(t);
        if had == has {
            return;
        }
        let mut s = self.tile_state(t);
        if has {
            s |= 1 << FALLOUT_BIT;
            self.num_tiles_with_fallout += 1;
        } else {
            s &= !(1 << FALLOUT_BIT);
            self.num_tiles_with_fallout = self.num_tiles_with_fallout.saturating_sub(1);
        }
        self.set_tile_state(t, s);
    }

    /// TS `GameMapImpl.setWater` - lake water (terrain byte 0), decrement land count.
    pub fn set_water(&mut self, t: TileRef) {
        if !self.is_land(t) {
            return;
        }
        self.terrain[t as usize] = 0;
        self.num_land_tiles = self.num_land_tiles.saturating_sub(1);
    }

    /// TS `GameMapImpl.setOcean`.
    pub fn set_ocean(&mut self, t: TileRef) {
        self.terrain[t as usize] |= 1 << OCEAN_BIT;
    }

    /// TS `GameMapImpl.setShorelineBit`.
    pub fn set_shoreline_bit(&mut self, t: TileRef) {
        self.terrain[t as usize] |= 1 << SHORELINE_BIT;
    }

    /// TS `GameMapImpl.clearShorelineBit`.
    pub fn clear_shoreline_bit(&mut self, t: TileRef) {
        self.terrain[t as usize] &= !(1 << SHORELINE_BIT);
    }

    /// TS `GameMapImpl.setMagnitude`.
    pub fn set_magnitude(&mut self, t: TileRef, value: u8) {
        let tbyte = &mut self.terrain[t as usize];
        *tbyte = (*tbyte & !MAGNITUDE_MASK) | (value & MAGNITUDE_MASK);
    }

    pub fn magnitude(&self, t: TileRef) -> u8 {
        self.terrain_byte(t) & MAGNITUDE_MASK
    }

    /// Live tip (`dd1277e245b5`) removed impassable terrain as a gameplay
    /// concept: magnitude 31 is just deep-inland Mountain (see tip
    /// `GameMap.terrainType`). Always return false so attack/spawn/nuke/
    /// conquer paths match tip TS (which has no `isImpassable` API).
    pub fn is_impassable(&self, _t: TileRef) -> bool {
        false
    }

    pub fn terrain_type(&self, t: TileRef) -> TerrainType {
        if self.is_land(t) {
            let mag = self.magnitude(t);
            // Tip: mag >= 20 is Mountain (including former impassable mag 31).
            if mag < 10 {
                return TerrainType::Plains;
            }
            if mag < 20 {
                return TerrainType::Highland;
            }
            return TerrainType::Mountain;
        }
        TerrainType::Ocean
    }

    pub fn tile_state(&self, t: TileRef) -> u16 {
        self.state[t as usize]
    }

    pub fn set_tile_state(&mut self, t: TileRef, v: u16) {
        self.state[t as usize] = v;
    }

    pub fn owner_id(&self, t: TileRef) -> u16 {
        self.tile_state(t) & 0x0fff
    }

    pub fn set_owner_id(&mut self, t: TileRef, pid: u16) {
        let mut s = self.tile_state(t);
        s = (s & !0x0fff) | (pid & 0x0fff);
        self.set_tile_state(t, s);
    }

    pub fn tile_state_buffer(&self) -> &[u16] {
        &self.state
    }

    pub fn for_each_neighbor8(&self, t: TileRef, mut f: impl FnMut(TileRef)) {
        let w = self.width;
        let x = self.x(t);
        let has_n = t >= w;
        let has_s = t < (self.height - 1) * w;
        if x > 0 {
            if has_n {
                f(t - 1 - w);
            }
            f(t - 1);
            if has_s {
                f(t - 1 + w);
            }
        }
        if has_n {
            f(t - w);
        }
        if has_s {
            f(t + w);
        }
        if x + 1 < w {
            if has_n {
                f(t + 1 - w);
            }
            f(t + 1);
            if has_s {
                f(t + 1 + w);
            }
        }
    }

    pub fn is_ocean_shore(&self, t: TileRef) -> bool {
        if !self.is_land(t) {
            return false;
        }
        let w = self.width;
        let x = self.x(t);
        if x > 0 && self.is_ocean(t - 1) {
            return true;
        }
        if x + 1 < w && self.is_ocean(t + 1) {
            return true;
        }
        if t >= w && self.is_ocean(t - w) {
            return true;
        }
        if t < (self.height - 1) * w && self.is_ocean(t + w) {
            return true;
        }
        false
    }

    pub fn is_on_edge_of_map(&self, t: TileRef) -> bool {
        let x = self.x(t);
        let y = self.y(t);
        x == 0 || x + 1 == self.width || y == 0 || y + 1 == self.height
    }

    pub fn for_each_neighbor4(&self, t: TileRef, mut f: impl FnMut(TileRef)) {
        let w = self.width;
        let x = self.x(t);
        // TS `GameMap.forEachNeighbor` / `neighbors4` order on the live
        // production tip (`dd1277e245b5`): west, east, north, south.
        // Note this intentionally differs from `neighbors()` (N,S,W,E) on
        // that same tip - AttackExecution / cluster capture use neighbors4,
        // while shore-coerce and WaterManager mini-map walks use neighbors().
        // Native previously tracked a post-unification pin where both APIs
        // were N,S,W,E; that desynced every live-tip human game at ~tick 310.
        if x > 0 {
            f(t - 1);
        }
        if x + 1 < w {
            f(t + 1);
        }
        if t >= w {
            f(t - w);
        }
        if t < (self.height - 1) * w {
            f(t + w);
        }
    }

    /// TS `GameMap.neighbors4` / `forEachNeighbor` order: west, east, north, south.
    pub fn neighbors4_ts(&self, t: TileRef, buf: &mut [TileRef; 4]) -> usize {
        let w = self.width;
        let x = self.x(t);
        let mut n = 0usize;
        if x > 0 {
            buf[n] = t - 1;
            n += 1;
        }
        if x + 1 < w {
            buf[n] = t + 1;
            n += 1;
        }
        if t >= w {
            buf[n] = t - w;
            n += 1;
        }
        if t < (self.height - 1) * w {
            buf[n] = t + w;
            n += 1;
        }
        n
    }

    /// TS `GameMap.neighbors()` order: north, south, west, east (live tip
    /// `dd1277e245b5`). Use this when the TS call site iterates `neighbors()`,
    /// not `neighbors4` / `forEachNeighbor`.
    pub fn for_each_neighbor_nswe(&self, t: TileRef, mut f: impl FnMut(TileRef)) {
        let w = self.width;
        let x = self.x(t);
        if t >= w {
            f(t - w);
        }
        if t < (self.height - 1) * w {
            f(t + w);
        }
        if x > 0 {
            f(t - 1);
        }
        if x + 1 < w {
            f(t + 1);
        }
    }

    /// Buffer form of [`Self::for_each_neighbor_nswe`].
    pub fn neighbors_nswe(&self, t: TileRef, buf: &mut [TileRef; 4]) -> usize {
        let w = self.width;
        let x = self.x(t);
        let mut n = 0usize;
        if t >= w {
            buf[n] = t - w;
            n += 1;
        }
        if t < (self.height - 1) * w {
            buf[n] = t + w;
            n += 1;
        }
        if x > 0 {
            buf[n] = t - 1;
            n += 1;
        }
        if x + 1 < w {
            buf[n] = t + 1;
            n += 1;
        }
        n
    }

    pub fn has_owner(&self, t: TileRef) -> bool {
        self.owner_id(t) > 0
    }

    pub fn is_border(&self, t: TileRef) -> bool {
        let owner = self.owner_id(t);
        let x = self.x(t);
        let w = self.width;
        let h = self.height;
        if x > 0 && self.owner_id(t - 1) != owner {
            return true;
        }
        if x + 1 < w && self.owner_id(t + 1) != owner {
            return true;
        }
        if t >= w && self.owner_id(t - w) != owner {
            return true;
        }
        if t < (h - 1) * w && self.owner_id(t + w) != owner {
            return true;
        }
        false
    }

    pub fn euclidean_dist_squared(&self, a: TileRef, b: TileRef) -> u32 {
        let dx = self.x(a) as i32 - self.x(b) as i32;
        let dy = self.y(a) as i32 - self.y(b) as i32;
        (dx * dx + dy * dy) as u32
    }

    pub fn euclidean_dist_squared_center(&self, root: TileRef, n: TileRef) -> f64 {
        let root_x = self.x(root) as f64 - 0.5;
        let root_y = self.y(root) as f64 - 0.5;
        let dx = self.x(n) as f64 - root_x;
        let dy = self.y(n) as f64 - root_y;
        dx * dx + dy * dy
    }

    pub fn manhattan_dist(&self, a: TileRef, b: TileRef) -> u32 {
        let dx = self.x(a).abs_diff(self.x(b));
        let dy = self.y(a).abs_diff(self.y(b));
        dx + dy
    }

    /// Cardinal BFS; returns all tiles matching `filter` (including start if valid).
    pub fn bfs<F>(&self, start: TileRef, filter: F) -> Vec<TileRef>
    where
        F: Fn(&GameMap, TileRef) -> bool,
    {
        let mut scratch = crate::water::BfsScratch::new((self.width * self.height) as usize);
        self.bfs_with_scratch(&mut scratch, start, filter)
    }

    /// Stamp BFS - reuses `scratch` instead of allocating a full-map visited bitset.
    pub fn bfs_with_scratch<F>(
        &self,
        scratch: &mut crate::water::BfsScratch,
        start: TileRef,
        filter: F,
    ) -> Vec<TileRef>
    where
        F: Fn(&GameMap, TileRef) -> bool,
    {
        let stamp = scratch.next_stamp();
        let mut q = Vec::with_capacity(64);
        let mut out = Vec::with_capacity(64);
        if filter(self, start) {
            scratch.seen[start as usize] = stamp;
            q.push(start);
            out.push(start);
        }
        let w = self.width;
        let south_limit = (self.height - 1) * w;
        while let Some(curr) = q.pop() {
            let x = curr % w;
            let visit =
                |n: TileRef, seen: &mut [u32], stamp: u32, q: &mut Vec<TileRef>, out: &mut Vec<TileRef>| {
                    if seen[n as usize] != stamp && filter(self, n) {
                        seen[n as usize] = stamp;
                        q.push(n);
                        out.push(n);
                    }
                };
            if curr >= w {
                visit(curr - w, &mut scratch.seen, stamp, &mut q, &mut out);
            }
            if curr < south_limit {
                visit(curr + w, &mut scratch.seen, stamp, &mut q, &mut out);
            }
            if x > 0 {
                visit(curr - 1, &mut scratch.seen, stamp, &mut q, &mut out);
            }
            if x + 1 < w {
                visit(curr + 1, &mut scratch.seen, stamp, &mut q, &mut out);
            }
        }
        out
    }
}

// TS `NeighborIteration.test.ts` + live tip (`dd1277e245b5`) GameMap:
// `forEachNeighbor`/`neighbors4` are W,E,N,S while `neighbors()` is N,S,W,E.
#[cfg(test)]
mod neighbor_order_tests {
    use super::{GameMap, MapMeta, TileRef};

    fn map16() -> GameMap {
        let n = 16 * 16;
        GameMap::from_terrain_bytes(
            &MapMeta {
                width: 16,
                height: 16,
                num_land_tiles: n as u32,
            },
            &vec![0b1000_0000u8; n as usize],
        )
        .unwrap()
    }

    fn collect_neighbors4(map: &GameMap, t: TileRef) -> Vec<TileRef> {
        let mut out = Vec::new();
        map.for_each_neighbor4(t, |n| out.push(n));
        out
    }

    fn collect_neighbors_nswe(map: &GameMap, t: TileRef) -> Vec<TileRef> {
        let mut out = Vec::new();
        map.for_each_neighbor_nswe(t, |n| out.push(n));
        out
    }

    fn collect_neighbors8(map: &GameMap, t: TileRef) -> Vec<TileRef> {
        let mut out = Vec::new();
        map.for_each_neighbor8(t, |n| out.push(n));
        out
    }

    #[test]
    fn for_each_neighbor4_visits_w_e_n_s_for_interior_tiles() {
        let map = map16();
        let tile = map.ref_xy(5, 7);
        assert_eq!(
            collect_neighbors4(&map, tile),
            vec![
                map.ref_xy(4, 7), // W
                map.ref_xy(6, 7), // E
                map.ref_xy(5, 6), // N
                map.ref_xy(5, 8), // S
            ]
        );
    }

    #[test]
    fn for_each_neighbor_nswe_visits_n_s_w_e_for_interior_tiles() {
        let map = map16();
        let tile = map.ref_xy(5, 7);
        assert_eq!(
            collect_neighbors_nswe(&map, tile),
            vec![
                map.ref_xy(5, 6),
                map.ref_xy(5, 8),
                map.ref_xy(4, 7),
                map.ref_xy(6, 7),
            ]
        );
    }

    #[test]
    fn for_each_neighbor4_clips_at_corners_and_edges() {
        let map = map16();
        let w = map.width;
        let h = map.height;
        // top-left corner: E, S only (W,E,N,S with missing W/N).
        assert_eq!(
            collect_neighbors4(&map, map.ref_xy(0, 0)),
            vec![map.ref_xy(1, 0), map.ref_xy(0, 1)]
        );
        // bottom-right corner: W, N only.
        assert_eq!(
            collect_neighbors4(&map, map.ref_xy(w - 1, h - 1)),
            vec![map.ref_xy(w - 2, h - 1), map.ref_xy(w - 1, h - 2)]
        );
        // left edge: E, N, S.
        assert_eq!(
            collect_neighbors4(&map, map.ref_xy(0, 5)),
            vec![map.ref_xy(1, 5), map.ref_xy(0, 4), map.ref_xy(0, 6)]
        );
        // bottom edge: W, E, N.
        assert_eq!(
            collect_neighbors4(&map, map.ref_xy(5, h - 1)),
            vec![
                map.ref_xy(4, h - 1),
                map.ref_xy(6, h - 1),
                map.ref_xy(5, h - 2),
            ]
        );
    }

    #[test]
    fn for_each_neighbor4_matches_neighbors4_ts_for_every_tile() {
        let map = map16();
        for t in 0..(map.width * map.height) {
            let via_callback = collect_neighbors4(&map, t);
            let mut buf = [TileRef::MAX; 4];
            let n = map.neighbors4_ts(t, &mut buf);
            assert_eq!(via_callback, buf[..n].to_vec(), "tile {t}");
        }
    }

    #[test]
    fn for_each_neighbor8_visits_all_8_neighbors_in_dx_major_order() {
        let map = map16();
        let tile = map.ref_xy(5, 7);
        assert_eq!(
            collect_neighbors8(&map, tile),
            vec![
                map.ref_xy(4, 6),
                map.ref_xy(4, 7),
                map.ref_xy(4, 8),
                map.ref_xy(5, 6),
                map.ref_xy(5, 8),
                map.ref_xy(6, 6),
                map.ref_xy(6, 7),
                map.ref_xy(6, 8),
            ]
        );
    }

    #[test]
    fn for_each_neighbor8_clips_at_corners_and_edges() {
        let map = map16();
        let w = map.width;
        let h = map.height;
        assert_eq!(
            collect_neighbors8(&map, map.ref_xy(0, 0)),
            vec![map.ref_xy(0, 1), map.ref_xy(1, 0), map.ref_xy(1, 1)]
        );
        assert_eq!(
            collect_neighbors8(&map, map.ref_xy(w - 1, h - 1)),
            vec![
                map.ref_xy(w - 2, h - 2),
                map.ref_xy(w - 2, h - 1),
                map.ref_xy(w - 1, h - 2),
            ]
        );
        assert_eq!(
            collect_neighbors8(&map, map.ref_xy(5, 0)),
            vec![
                map.ref_xy(4, 0),
                map.ref_xy(4, 1),
                map.ref_xy(5, 1),
                map.ref_xy(6, 0),
                map.ref_xy(6, 1),
            ]
        );
    }
}

pub fn read_manifest(map_dir: &Path) -> Result<MapManifest, String> {
    let path = map_dir.join("manifest.json");
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn read_terrain_bin(map_dir: &Path, filename: &str, meta: &MapMeta) -> Result<Vec<u8>, String> {
    let path = map_dir.join(filename);
    let data = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let n = (meta.width * meta.height) as usize;
    if data.len() != n {
        return Err(format!(
            "{}: {} bytes != {}x{}",
            path.display(),
            data.len(),
            meta.width,
            meta.height
        ));
    }
    Ok(data)
}
