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

    pub fn magnitude(&self, t: TileRef) -> u8 {
        self.terrain_byte(t) & MAGNITUDE_MASK
    }

    pub fn is_impassable(&self, t: TileRef) -> bool {
        self.is_land(t) && self.magnitude(t) == IMPASSABLE_MAGNITUDE
    }

    pub fn terrain_type(&self, t: TileRef) -> TerrainType {
        if self.is_land(t) {
            let mag = self.magnitude(t);
            if mag >= IMPASSABLE_MAGNITUDE {
                return TerrainType::Impassable;
            }
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
        // TS `GameMap.neighbors4` order: west, east, north, south.
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

    /// TS `GameMap.neighbors4` / `forEachNeighbor` order: north, south, west, east.
    pub fn neighbors4_ts(&self, t: TileRef, buf: &mut [TileRef; 4]) -> usize {
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
