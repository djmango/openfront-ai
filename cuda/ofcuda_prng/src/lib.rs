//! Host-side shared code for the `ofcuda_prng` parity harness.
//!
//! Nothing in here touches CUDA, so the GPU binary and the CPU companion binary
//! link *literally the same* parsing, PRNG and spawn-selection code. Only the
//! computation differs (a cuda-oxide kernel vs. a plain sequential Rust loop),
//! which is what makes a mismatch meaningful.
//!
//! Everything is a 1:1 port of:
//!   * `rust/engine/src/prng.rs`               - sfc32 `PseudoRandom`
//!   * `rust/engine/src/util.rs:4-13`          - `simple_hash`
//!   * `rust/engine/src/session.rs:78-88`      - `seed_to_game_id`
//!   * `rust/engine/src/execution/spawn_util.rs:65-146` - spawn tile selection
//!   * `rust/engine/src/map.rs:401-443`        - `is_border` / `too_close` distance
//!
//! The reference values come from a dump produced by the *real* engine
//! (`rust/engine/src/bin/prng_dump.rs`, which drives an actual `RlSession`), so
//! the comparison is against the engine, not against a second copy of this port.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// simple_hash - `rust/engine/src/util.rs:4-13`
// ---------------------------------------------------------------------------

/// djb2-style hash, exactly `Util.simpleHash`. Note the final `.abs()`.
pub fn simple_hash(s: &str) -> i32 {
    let mut hash: i32 = 0;
    for ch in s.chars() {
        let c = ch as i32;
        hash = hash.wrapping_shl(5).wrapping_sub(hash).wrapping_add(c);
        hash = hash.wrapping_mul(1);
    }
    hash.abs()
}

/// `rust/engine/src/session.rs:78-88`.
pub fn seed_to_game_id(seed: &str) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut h = simple_hash(&format!("rl-{seed}")) as u32;
    let mut out = String::with_capacity(8);
    for _ in 0..8 {
        h = h.wrapping_mul(1_103_515_245).wrapping_add(12_345) & 0x7fff_ffff;
        out.push(ALPHABET[(h as usize) % ALPHABET.len()] as char);
    }
    out
}

// ---------------------------------------------------------------------------
// PseudoRandom - `rust/engine/src/prng.rs`
// ---------------------------------------------------------------------------

/// sfc32. `new` does four splitmix-style splits and then **12 warm-up draws**
/// (`prng.rs:30-32`); `next()` returns `(t as u32) as f64 / 2^32` (`prng.rs:44`)
/// - a u32 division, not an f64 one.
#[derive(Clone, Copy, Debug)]
pub struct PseudoRandom {
    pub s0: i32,
    pub s1: i32,
    pub s2: i32,
    pub s3: i32,
    /// Total `next()` calls made, including the 12 warm-ups.
    pub calls: u64,
}

impl PseudoRandom {
    pub fn new(seed: i32) -> Self {
        let mut h = seed;
        let mut split = || {
            h = h.wrapping_add(0x9e37_79b9u32 as i32);
            let mut t = h ^ ((h as u32) >> 16) as i32;
            t = t.wrapping_mul(0x21f0_aaad);
            t ^= ((t as u32) >> 15) as i32;
            t = t.wrapping_mul(0x735a_2d97);
            t ^ ((t as u32) >> 15) as i32
        };
        let mut pr = Self {
            s0: split(),
            s1: split(),
            s2: split(),
            s3: split(),
            calls: 0,
        };
        for _ in 0..12 {
            pr.next_u32();
        }
        pr
    }

    /// The raw `t` of `prng.rs:37-44`. `next()` is exactly `t as u32 / 2^32`.
    pub fn next_u32(&mut self) -> u32 {
        let t = (self.s0.wrapping_add(self.s1)).wrapping_add(self.s3);
        self.s3 = self.s3.wrapping_add(1);
        self.s0 = self.s1 ^ ((self.s1 as u32) >> 9) as i32;
        self.s1 = self.s2.wrapping_add(self.s2.wrapping_shl(3));
        let s2_bits = self.s2 as u32;
        self.s2 = ((s2_bits << 21) | (s2_bits >> 11)) as i32;
        self.s2 = self.s2.wrapping_add(t);
        self.calls += 1;
        t as u32
    }

    pub fn next(&mut self) -> f64 {
        self.next_u32() as f64 / 4_294_967_296.0
    }

    pub fn next_int(&mut self, min: i32, max: i32) -> i32 {
        let lo = min;
        let hi = max;
        (self.next() * (hi - lo) as f64).floor() as i32 + lo
    }

    /// `prng.rs:59-61`.
    pub fn chance(&mut self, odds: i32) -> bool {
        self.next_int(0, odds) == 0
    }

    /// `prng.rs:86-92`: draws NOTHING when the list is empty.
    pub fn rand_element(&mut self, len: usize) -> Option<usize> {
        if len == 0 {
            return None;
        }
        Some(self.next_int(0, len as i32) as usize)
    }

    /// `prng.rs:103-110`: Fisher-Yates, exactly `len - 1` draws.
    pub fn shuffle_array(&mut self, array: &mut [i32]) {
        for i in (1..array.len()).rev() {
            let j = self.next_int(0, (i + 1) as i32) as usize;
            array.swap(i, j);
        }
    }

    /// Draws since construction, excluding the 12 warm-ups.
    pub fn draws(&self) -> u64 {
        self.calls - 12
    }

    /// `prng.rs:75-90` `next_id`: one draw, mapped into 36^8, then 8 base-36
    /// digits most-significant-first.
    pub fn next_id(&mut self) -> String {
        const POW36_8: f64 = 2_821_109_907_456.0; // 36^8
        let mut v = (self.next() * POW36_8).floor() as u64;
        let mut out = ['0'; 8];
        for slot in out.iter_mut().rev() {
            let digit = (v % 36) as u8;
            *slot = b"0123456789abcdefghijklmnopqrstuvwxyz"[digit as usize] as char;
            v /= 36;
        }
        out.iter().collect()
    }
}

// ---------------------------------------------------------------------------
// Map loading - same pair of files `GameMapSize::Normal` reads
// ---------------------------------------------------------------------------

pub const IS_LAND_BIT: u8 = 0x80;

pub struct MapPlane {
    pub width: u32,
    pub height: u32,
    pub terrain: Vec<u8>,
}

pub fn load_map_normal(map_dir: &Path) -> Result<MapPlane, String> {
    let manifest_path = map_dir.join("manifest.json");
    let manifest_bytes =
        fs::read(&manifest_path).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let map = manifest
        .get("map")
        .ok_or_else(|| format!("{}: no \"map\" entry", manifest_path.display()))?;
    let num = |key: &str| -> Result<u32, String> {
        map.get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .ok_or_else(|| format!("{}: map.{key} missing", manifest_path.display()))
    };
    let (width, height) = (num("width")?, num("height")?);
    let bin_path = map_dir.join("map.bin");
    let terrain = fs::read(&bin_path).map_err(|e| format!("{}: {e}", bin_path.display()))?;
    if terrain.len() != (width as usize) * (height as usize) {
        return Err(format!(
            "{}: {} bytes != {width}x{height}",
            bin_path.display(),
            terrain.len()
        ));
    }
    Ok(MapPlane {
        width,
        height,
        terrain,
    })
}

pub fn map_dir(repo_root: &Path, map_key: &str) -> PathBuf {
    repo_root
        .join("openfront/resources/maps")
        .join(map_key.to_lowercase().replace(' ', ""))
}

// ---------------------------------------------------------------------------
// Spawn tile selection - `rust/engine/src/execution/spawn_util.rs:65-146`
// ---------------------------------------------------------------------------

/// One spawn job: a PRNG seed plus an optional explicit tile (the human path,
/// which draws nothing).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpawnJob {
    pub seed: i32,
    /// `u32::MAX` = no explicit tile (`None`).
    pub explicit: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpawnResult {
    pub tile: u32,
    pub x: u32,
    pub y: u32,
    /// `tries` in `spawn_util.rs:87`; 0 for the explicit-tile path.
    pub attempts: u32,
    /// `next()` calls consumed after the 12 warm-ups.
    pub draws: u64,
    pub spawned: bool,
    /// The next three raw u32 draws after selection - a draw-count tripwire.
    pub after: [u32; 3],
}

/// Shared state for one spawn selection: the terrain plane, the owner
/// *overrides* at selection time and the already-placed spawn centres.
///
/// The owner plane is carried as an override list (`tile -> small_id`) rather
/// than a full `u16` plane so that the CPU path and the CUDA kernel take
/// literally the same representation and the same lookup code.
pub struct SpawnCtx<'a> {
    pub terrain: &'a [u8],
    pub owner_tiles: &'a [u32],
    pub owner_ids: &'a [u16],
    pub prev: &'a [u32],
    pub width: u32,
    pub height: u32,
    /// `game.wire.min_distance_between_players()` = 30 (`core/config.rs:358-360`).
    pub min_dist: u32,
    /// Optional O(1) owner lookup over a full `width*height` plane.
    ///
    /// The engine keeps ownership in a real tile plane, so its lookups are
    /// O(1); the port's recorded-window path instead carries the engine's
    /// owner-override *list*. Scanning that list is O(owned) per query, which
    /// makes a generic N-bot run O(N^2) in the owned-tile count. The 2-bot
    /// reference case is tiny so the list is used there (it is the engine's own
    /// overlay, so parity is unchanged); the generic driver feeds a plane.
    pub owner_plane: Option<&'a [u16]>,
}

impl<'a> SpawnCtx<'a> {
    #[inline]
    pub fn is_land(&self, t: u32) -> bool {
        self.terrain[t as usize] & IS_LAND_BIT != 0
    }
    #[inline]
    pub fn owner(&self, t: u32) -> u16 {
        if let Some(p) = self.owner_plane {
            return p[t as usize];
        }
        let mut i = 0usize;
        while i < self.owner_tiles.len() {
            if self.owner_tiles[i] == t {
                return self.owner_ids[i];
            }
            i += 1;
        }
        0
    }
    #[inline]
    pub fn has_owner(&self, t: u32) -> bool {
        self.owner(t) > 0
    }
    /// `map.rs:405-425`.
    #[inline]
    pub fn is_border(&self, t: u32) -> bool {
        let owner = self.owner(t);
        let x = t % self.width;
        if x > 0 && self.owner(t - 1) != owner {
            return true;
        }
        if x + 1 < self.width && self.owner(t + 1) != owner {
            return true;
        }
        if t >= self.width && self.owner(t - self.width) != owner {
            return true;
        }
        if t < (self.height - 1) * self.width && self.owner(t + self.width) != owner {
            return true;
        }
        false
    }
    /// `game.rs:3622-3639` + `map.rs:439-443`.
    #[inline]
    pub fn too_close(&self, center: u32) -> bool {
        let cx = center % self.width;
        let cy = center / self.width;
        for &p in self.prev {
            let px = p % self.width;
            let py = p / self.width;
            let dx = px.abs_diff(cx);
            let dy = py.abs_diff(cy);
            if dx + dy < self.min_dist {
                return true;
            }
        }
        false
    }

    /// `map.rs:431-437` - note the half-tile offset (`x - 0.5`).
    #[inline]
    pub fn dist2_center(&self, root: u32, n: u32) -> f64 {
        let root_x = (root % self.width) as f64 - 0.5;
        let root_y = (root / self.width) as f64 - 0.5;
        let dx = (n % self.width) as f64 - root_x;
        let dy = (n / self.width) as f64 - root_y;
        dx * dx + dy * dy
    }

    /// `get_spawn_tiles` (`spawn_util.rs:118-146`), BFS included.
    ///
    /// The filter is `dist2_center <= 16.0`; the engine's BFS explores *cardinal*
    /// neighbours and only enqueues tiles passing that filter, so the returned
    /// set is the cardinal-connected component of the centre inside the radius-4
    /// disk. That is NOT the whole disk - the disk is not cardinally connected at
    /// its corners - so porting the BFS (rather than a naive disk test) is
    /// required for exactness.
    ///
    /// Returns `(n_tiles, n_invalid)` where `invalid = has_owner || !is_land`.
    pub fn footprint(&self, center: u32) -> (u32, u32) {
        // The engine measures distance to `(x - 0.5, y - 0.5)`, so the offsets
        // that pass `dist2 <= 16.0` are `-4..=3` in each axis (offset -4 gives
        // -3.5, 12.25 <= 16; offset +4 gives +4.5, 20.25 > 16). Wrap-around
        // around the 8x8 window `[-4..3]^2` covers every reachable tile.
        const SIDE: usize = 8;
        let w = self.width as i32;
        let h = self.height as i32;
        let cx = (center % self.width) as i32;
        let cy = (center / self.width) as i32;

        let mut seen = [false; SIDE * SIDE];
        let mut stack: [i32; SIDE * SIDE] = [-1; SIDE * SIDE];
        let mut sp = 0usize;

        // Start tile: the engine only enqueues it if the filter passes (it does).
        seen[36] = true; // idx_of(0,0) = (0+4)*8 + (0+4) = 36
        stack[0] = 36;
        sp = 1;

        let mut n_tiles = 0u32;
        let mut n_invalid = 0u32;
        while sp > 0 {
            sp -= 1;
            let d = stack[sp];
            let dx = d % SIDE as i32 - 4;
            let dy = d / SIDE as i32 - 4;
            let t = ((cy + dy) as u32) * self.width + (cx + dx) as u32;
            n_tiles += 1;
            if self.has_owner(t) || !self.is_land(t) {
                n_invalid += 1;
            }
            // Same neighbour order/guards as `map.rs:474-499` (N, S, W, E).
            if cy + dy - 1 >= 0 {
                bfs_push(&mut seen, &mut stack, &mut sp, dx, dy - 1);
            }
            if cy + dy + 1 < h {
                bfs_push(&mut seen, &mut stack, &mut sp, dx, dy + 1);
            }
            if cx + dx - 1 >= 0 {
                bfs_push(&mut seen, &mut stack, &mut sp, dx - 1, dy);
            }
            if cx + dx + 1 < w {
                bfs_push(&mut seen, &mut stack, &mut sp, dx + 1, dy);
            }
        }
        (n_tiles, n_invalid)
    }
}

/// One BFS enqueue: pass the `dist2_center <= 16` filter (with the engine's
/// half-tile offset), then the seen/stamp check (`map.rs:466-472`).
#[inline]
fn bfs_push(seen: &mut [bool; 64], stack: &mut [i32; 64], sp: &mut usize, ndx: i32, ndy: i32) {
    const SIDE: i32 = 8;
    let fx = ndx as f64 + 0.5;
    let fy = ndy as f64 + 0.5;
    if fx * fx + fy * fy > 16.0 {
        return;
    }
    let k = ((ndy + 4) * SIDE + (ndx + 4)) as usize;
    if !seen[k] {
        seen[k] = true;
        stack[*sp] = k as i32;
        *sp += 1;
    }
}

/// Build the result and append the three post-selection raw draws, which act as
/// a draw-count tripwire (`draws` is sampled *before* them).
fn finish(
    pr: &mut PseudoRandom,
    tile: u32,
    x: u32,
    y: u32,
    attempts: u32,
    spawned: bool,
) -> SpawnResult {
    let draws = pr.draws();
    let mut after = [0u32; 3];
    for a in after.iter_mut() {
        *a = pr.next_u32();
    }
    SpawnResult {
        tile,
        x,
        y,
        attempts,
        draws,
        spawned,
        after,
    }
}

/// The whole selection: `find_spawn` (`spawn_util.rs:65-104`) + `rand_tile`
/// (`spawn_util.rs:106-116`). `MAX_SPAWN_TRIES = 1000` (`spawn_util.rs:84`).
pub fn select_spawn(ctx: &SpawnCtx, job: &SpawnJob) -> SpawnResult {
    const MAX_SPAWN_TRIES: i32 = 1_000;
    let mut pr = PseudoRandom::new(job.seed);

    if job.explicit != u32::MAX {
        // Explicit tile: `find_spawn` returns before `rand_tile` is ever
        // called, so this path consumes ZERO draws (`spawn_util.rs:71-77`).
        let (n_tiles, n_invalid) = ctx.footprint(job.explicit);
        let spawned = n_tiles - n_invalid > 0;
        return finish(
            &mut pr,
            job.explicit,
            job.explicit % ctx.width,
            job.explicit / ctx.width,
            0,
            spawned,
        );
    }

    let mut tries = 0;
    while tries < MAX_SPAWN_TRIES {
        tries += 1;
        // `rand_tile` with `spawn_area == None` (FFA: no team spawn area) =
        // exactly two `next_int` draws per attempt (`spawn_util.rs:111-114`).
        let x = pr.next_int(0, ctx.width as i32);
        let y = pr.next_int(0, ctx.height as i32);
        let center = (y as u32) * ctx.width + (x as u32);

        if !ctx.is_land(center) || ctx.has_owner(center) || ctx.is_border(center) {
            continue;
        }
        if ctx.too_close(center) {
            continue;
        }
        // `require_all_valid = true` (`spawn_util.rs:98`): any owned or
        // non-land tile in the footprint rejects the attempt.
        let (_n_tiles, n_invalid) = ctx.footprint(center);
        if n_invalid > 0 {
            continue;
        }
        return finish(
            &mut pr,
            center,
            center % ctx.width,
            center / ctx.width,
            tries as u32,
            true,
        );
    }
    finish(&mut pr, u32::MAX, u32::MAX, u32::MAX, MAX_SPAWN_TRIES as u32, false)
}

// ---------------------------------------------------------------------------
// Reference dump parsing
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct RefSeed {
    pub seed_str: String,
    pub game_id: String,
    pub game_hash: i32,
    pub stream: Vec<u32>,
    pub draws: Vec<(i32, i32, Vec<i32>)>,
    pub chance100: Vec<u8>,
    pub chance2: Vec<u8>,
    pub chance1_all: bool,
    pub rand7: Vec<i32>,
    pub rand7_next: Vec<u32>,
    pub rand_empty_consumed: i32,
    pub rand_empty_next: Vec<u32>,
    pub shuffle8: Vec<i32>,
    pub shuffle8_next: Vec<u32>,
    pub pstream_id: String,
    pub pstream_hash: i32,
    pub pstream: Vec<u32>,
    pub dims: (u32, u32),
    pub spawns: Vec<RefSpawn>,
    pub human: Option<RefHuman>,
}

#[derive(Clone, Debug, Default)]
pub struct RefSpawn {
    pub bot: usize,
    pub player_id: String,
    pub seed: i32,
    pub tile: u32,
    pub x: u32,
    pub y: u32,
    pub owners: Vec<(u32, u16)>,
    pub prev: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct RefHuman {
    pub player_id: String,
    pub seed: i32,
    pub tile: u32,
    pub x: u32,
    pub y: u32,
    pub spawned: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Reference {
    pub map: String,
    pub bots: u32,
    pub seeds: Vec<RefSeed>,
}

pub fn parse_reference(text: &str) -> Result<Reference, String> {
    let mut r = Reference::default();
    let mut owner_buf: HashMap<(usize, usize), Vec<(u32, u16)>> = HashMap::new();
    let mut prev_buf: HashMap<(usize, usize), Vec<u32>> = HashMap::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let err = |what: &str| format!("line {}: bad {what}: {line}", lineno + 1);
        match f[0] {
            "map" => r.map = f[1].to_string(),
            "bots" => r.bots = f[1].parse().map_err(|_| err("bots"))?,
            "seeds" => {
                let n: usize = f[1].parse().map_err(|_| err("seeds"))?;
                r.seeds.resize(n, RefSeed::default());
            }
            "seed" => {
                let si: usize = f[1].parse().map_err(|_| err("seed"))?;
                r.seeds[si].seed_str = f[2].to_string();
                r.seeds[si].game_id = f[3].to_string();
                r.seeds[si].game_hash = f[4].parse().map_err(|_| err("game_hash"))?;
            }
            "stream" => {
                let si: usize = f[1].parse().map_err(|_| err("stream"))?;
                r.seeds[si].stream = f[2..]
                    .iter()
                    .map(|v| u32::from_str_radix(v, 16).map_err(|_| err("stream hex")))
                    .collect::<Result<_, _>>()?;
            }
            "draws" => {
                let si: usize = f[1].parse().map_err(|_| err("draws"))?;
                let lo: i32 = f[2].parse().map_err(|_| err("draws lo"))?;
                let hi: i32 = f[3].parse().map_err(|_| err("draws hi"))?;
                let vals: Vec<i32> = f[4..]
                    .iter()
                    .map(|v| v.parse().map_err(|_| err("draws val")))
                    .collect::<Result<_, _>>()?;
                r.seeds[si].draws.push((lo, hi, vals));
            }
            "sem" => {
                let si: usize = f[1].parse().map_err(|_| err("sem"))?;
                match f[2] {
                    "chance100" => {
                        r.seeds[si].chance100 =
                            f[3].bytes().map(|b| b - b'0').collect::<Vec<u8>>()
                    }
                    "chance2" => {
                        r.seeds[si].chance2 = f[3].bytes().map(|b| b - b'0').collect::<Vec<u8>>()
                    }
                    "chance1_all" => r.seeds[si].chance1_all = f[3] == "true",
                    "rand7" => {
                        r.seeds[si].rand7 = f[3..6]
                            .iter()
                            .map(|v| v.parse().map_err(|_| err("rand7")))
                            .collect::<Result<_, _>>()?;
                        r.seeds[si].rand7_next = f[6..9]
                            .iter()
                            .map(|v| u32::from_str_radix(v, 16).map_err(|_| err("rand7 next")))
                            .collect::<Result<_, _>>()?;
                    }
                    "rand_empty" => {
                        r.seeds[si].rand_empty_consumed = f[5].parse().map_err(|_| err("consumed"))?;
                        r.seeds[si].rand_empty_next = f[7..10]
                            .iter()
                            .map(|v| u32::from_str_radix(v, 16).map_err(|_| err("empty next")))
                            .collect::<Result<_, _>>()?;
                    }
                    "shuffle8" => {
                        r.seeds[si].shuffle8 = f[3..11]
                            .iter()
                            .map(|v| v.parse().map_err(|_| err("shuffle8")))
                            .collect::<Result<_, _>>()?;
                        r.seeds[si].shuffle8_next = f[12..15]
                            .iter()
                            .map(|v| u32::from_str_radix(v, 16).map_err(|_| err("shuffle next")))
                            .collect::<Result<_, _>>()?;
                    }
                    other => return Err(err(other)),
                }
            }
            "pstream" => {
                let si: usize = f[1].parse().map_err(|_| err("pstream"))?;
                r.seeds[si].pstream_id = f[2].to_string();
                r.seeds[si].pstream_hash = f[3].parse().map_err(|_| err("pstream hash"))?;
                r.seeds[si].pstream = f[4..]
                    .iter()
                    .map(|v| u32::from_str_radix(v, 16).map_err(|_| err("pstream hex")))
                    .collect::<Result<_, _>>()?;
            }
            "dims" => {
                let si: usize = f[1].parse().map_err(|_| err("dims"))?;
                r.seeds[si].dims = (
                    f[2].parse().map_err(|_| err("dims w"))?,
                    f[3].parse().map_err(|_| err("dims h"))?,
                );
            }
            "spawn" => {
                let si: usize = f[1].parse().map_err(|_| err("spawn"))?;
                r.seeds[si].spawns.push(RefSpawn {
                    bot: f[2].parse().map_err(|_| err("spawn bot"))?,
                    player_id: f[3].to_string(),
                    seed: f[4].parse().map_err(|_| err("spawn seed"))?,
                    tile: f[5].parse().map_err(|_| err("spawn tile"))?,
                    x: f[6].parse().map_err(|_| err("spawn x"))?,
                    y: f[7].parse().map_err(|_| err("spawn y"))?,
                    owners: Vec::new(),
                    prev: Vec::new(),
                });
            }
            "owner" => {
                let si: usize = f[1].parse().map_err(|_| err("owner"))?;
                let bi: usize = f[2].parse().map_err(|_| err("owner bot"))?;
                owner_buf
                    .entry((si, bi))
                    .or_default()
                    .push((f[3].parse().map_err(|_| err("owner tile"))?, f[4].parse().map_err(|_| err("owner id"))?));
            }
            "prev" => {
                let si: usize = f[1].parse().map_err(|_| err("prev"))?;
                let bi: usize = f[2].parse().map_err(|_| err("prev bot"))?;
                prev_buf
                    .entry((si, bi))
                    .or_default()
                    .push(f[3].parse().map_err(|_| err("prev tile"))?);
            }
            "human" => {
                let si: usize = f[1].parse().map_err(|_| err("human"))?;
                r.seeds[si].human = Some(RefHuman {
                    player_id: f[2].to_string(),
                    seed: f[3].parse().map_err(|_| err("human seed"))?,
                    tile: f[4].parse().map_err(|_| err("human tile"))?,
                    x: f[5].parse().map_err(|_| err("human x"))?,
                    y: f[6].parse().map_err(|_| err("human y"))?,
                    spawned: f[7] == "true",
                });
            }
            "players" | "ownerplane_before_bot1" => {}
            other => return Err(err(other)),
        }
    }
    for (si, seed) in r.seeds.iter_mut().enumerate() {
        for (bi, sp) in seed.spawns.iter_mut().enumerate() {
            if let Some(v) = owner_buf.remove(&(si, bi)) {
                sp.owners = v;
            }
            if let Some(v) = prev_buf.remove(&(si, bi)) {
                sp.prev = v;
            }
        }
    }
    Ok(r)
}

// ---------------------------------------------------------------------------
// Comparison helpers
// ---------------------------------------------------------------------------

/// Element-by-element comparison tally with the first divergence kept.
#[derive(Default, Debug)]
pub struct Tally {
    pub matched: usize,
    pub total: usize,
    pub first_divergence: Option<String>,
}

impl Tally {
    pub fn push<T: std::fmt::Debug + PartialEq>(&mut self, label: &str, idx: usize, expect: &T, got: &T) {
        self.total += 1;
        if expect == got {
            self.matched += 1;
        } else if self.first_divergence.is_none() {
            self.first_divergence =
                Some(format!("{label}[{idx}]: engine {expect:?} vs port {got:?}"));
        }
    }
    pub fn merge(&mut self, other: &Tally) {
        self.matched += other.matched;
        self.total += other.total;
        if self.first_divergence.is_none() {
            self.first_divergence = other.first_divergence.clone();
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

pub fn hex(v: &[u32]) -> String {
    v.iter().map(|x| format!("{x:08x}")).collect::<Vec<_>>().join(" ")
}

// ---------------------------------------------------------------------------
// Port output (filled by either the CPU reference or the CUDA kernels)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct PortOutput {
    pub streams: Vec<Vec<u32>>,
    pub draws: Vec<Vec<Vec<i32>>>,
    pub chance100: Vec<Vec<u8>>,
    pub chance2: Vec<Vec<u8>>,
    pub chance1_all: Vec<bool>,
    pub rand7: Vec<Vec<i32>>,
    pub rand7_next: Vec<Vec<u32>>,
    pub rand_empty_consumed: Vec<i32>,
    pub rand_empty_next: Vec<Vec<u32>>,
    pub shuffle8: Vec<Vec<i32>>,
    pub shuffle8_next: Vec<Vec<u32>>,
    pub pstream: Vec<Vec<u32>>,
    /// `(string, simple_hash(string))` in `hash_strings` order.
    pub hashes: Vec<(String, i32)>,
    /// `[seed][job]`, jobs = the two bots then the scripted human.
    pub spawns: Vec<Vec<SpawnResult>>,
}

pub const DRAW_RANGES: [(i32, i32); 4] = [(0, 100), (40, 80), (0, 7), (0, 5)];

/// The strings whose `simple_hash` the port must reproduce, in a fixed order:
/// per seed the game id, the two bot ids, the human id and the `DUMPPLAYER`
/// stream id.
pub fn hash_strings(r: &Reference) -> Vec<String> {
    let mut v = Vec::new();
    for s in &r.seeds {
        v.push(s.game_id.clone());
        for sp in &s.spawns {
            v.push(sp.player_id.clone());
        }
        if let Some(h) = &s.human {
            v.push(h.player_id.clone());
        }
        v.push(s.pstream_id.clone());
    }
    v
}

/// The spawn jobs, in a fixed order: for each seed the two bots (random tile,
/// `explicit = u32::MAX`) then the scripted human (explicit tile).
pub fn spawn_jobs(r: &Reference) -> Vec<SpawnJob> {
    let mut v = Vec::new();
    for s in &r.seeds {
        for sp in &s.spawns {
            v.push(SpawnJob {
                seed: sp.seed,
                explicit: u32::MAX,
            });
        }
        if let Some(h) = &s.human {
            v.push(SpawnJob {
                seed: h.seed,
                explicit: h.tile,
            });
        }
    }
    v
}

/// Plain sequential Rust reference for every value the CUDA port produces.
pub fn cpu_port_output(r: &Reference, map: &MapPlane) -> PortOutput {
    let mut o = PortOutput::default();
    for s in &r.seeds {
        // (a) 24 raw u32 of next().
        let mut pr = PseudoRandom::new(s.game_hash);
        o.streams
            .push((0..24).map(|_| pr.next_u32()).collect());

        // (b) 16 draws per range, each from a fresh PRNG.
        let mut per_range = Vec::new();
        for (lo, hi) in DRAW_RANGES {
            let mut p = PseudoRandom::new(s.game_hash);
            per_range.push((0..16).map(|_| p.next_int(lo, hi)).collect::<Vec<i32>>());
        }
        o.draws.push(per_range);

        // (c) chance / rand_element / shuffle.
        let mut c = PseudoRandom::new(s.game_hash);
        o.chance100
            .push((0..16).map(|_| c.chance(100) as u8).collect());
        o.chance2.push((0..16).map(|_| c.chance(2) as u8).collect());
        o.chance1_all.push((0..16).all(|_| c.chance(1)));

        let mut e = PseudoRandom::new(s.game_hash);
        o.rand7
            .push((0..3).map(|_| e.rand_element(7).unwrap() as i32).collect());
        o.rand7_next
            .push((0..3).map(|_| e.next_u32()).collect());

        let mut z = PseudoRandom::new(s.game_hash);
        let before = (z.s0, z.s1, z.s2, z.s3);
        let got = z.rand_element(0);
        let after = (z.s0, z.s1, z.s2, z.s3);
        assert!(got.is_none(), "empty rand_element must be None");
        o.rand_empty_consumed
            .push(if before == after { 0 } else { 1 });
        o.rand_empty_next
            .push((0..3).map(|_| z.next_u32()).collect());

        let mut sh = PseudoRandom::new(s.game_hash);
        let mut arr: Vec<i32> = (0..8).collect();
        sh.shuffle_array(&mut arr);
        o.shuffle8.push(arr);
        o.shuffle8_next
            .push((0..3).map(|_| sh.next_u32()).collect());

        // (d) simple_hash(player_id) stream.
        let mut pp = PseudoRandom::new(s.pstream_hash);
        o.pstream.push((0..24).map(|_| pp.next_u32()).collect());

        // (e) spawn selection against the real engine state.
        let mut jobs = Vec::new();
        for sp in &s.spawns {
            let owner_tiles: Vec<u32> = sp.owners.iter().map(|(t, _)| *t).collect();
            let owner_ids: Vec<u16> = sp.owners.iter().map(|(_, v)| *v).collect();
            let ctx = SpawnCtx {
                terrain: &map.terrain,
                owner_tiles: &owner_tiles,
                owner_ids: &owner_ids,
                prev: &sp.prev,
                width: map.width,
                height: map.height,
                min_dist: 30,
                owner_plane: None,
            };
            jobs.push(select_spawn(&ctx, &SpawnJob { seed: sp.seed, explicit: u32::MAX }));
        }
        if let Some(h) = &s.human {
            let ctx = SpawnCtx {
                terrain: &map.terrain,
                owner_tiles: &[],
                owner_ids: &[],
                prev: &[],
                width: map.width,
                height: map.height,
                min_dist: 30,
                owner_plane: None,
            };
            jobs.push(select_spawn(
                &ctx,
                &SpawnJob {
                    seed: h.seed,
                    explicit: h.tile,
                },
            ));
        }
        o.spawns.push(jobs);
    }

    for (i, s) in hash_strings(r).iter().enumerate() {
        let _ = i;
        o.hashes.push((s.clone(), simple_hash(s)));
    }
    o
}

// ---------------------------------------------------------------------------
// Comparison + report
// ---------------------------------------------------------------------------

#[derive(Default, Debug)]
pub struct Report {
    pub stream: Tally,
    pub draws: Tally,
    pub chance100: Tally,
    pub chance2: Tally,
    pub chance1: Tally,
    pub rand7: Tally,
    pub rand7_next: Tally,
    pub rand_empty: Tally,
    pub shuffle8: Tally,
    pub shuffle8_next: Tally,
    pub pstream: Tally,
    pub hashes: Tally,
    /// `simple_hash` checked against values the **engine itself** dumped
    /// (`game_hash` and the `simple_hash(pid) + simple_hash(game_id)` seed
    /// derivation in `prng_dump.rs:240`).
    pub hash_engine: Tally,
    pub spawn_tile: Tally,
    pub spawn_xy: Tally,
    pub spawn_after: Tally,
    /// GPU-vs-CPU cross check (values the engine dump does not carry).
    pub cross: Tally,
    pub spawn_lines: Vec<String>,
}

impl Report {
    pub fn first_divergence(&self) -> Option<String> {
        for t in [
            &self.stream,
            &self.draws,
            &self.chance100,
            &self.chance2,
            &self.chance1,
            &self.rand7,
            &self.rand7_next,
            &self.rand_empty,
            &self.shuffle8,
            &self.shuffle8_next,
            &self.pstream,
            &self.hashes,
            &self.hash_engine,
            &self.spawn_tile,
            &self.spawn_xy,
            &self.spawn_after,
            &self.cross,
        ] {
            if let Some(d) = &t.first_divergence {
                return Some(d.clone());
            }
        }
        None
    }
    pub fn all_match(&self) -> bool {
        self.first_divergence().is_none()
    }
}

/// Element-by-element comparison of a port's output against the engine dump.
pub fn compare(r: &Reference, o: &PortOutput) -> Report {
    let mut rep = Report::default();
    for (si, s) in r.seeds.iter().enumerate() {
        for i in 0..24 {
            rep.stream.push(
                &format!("seed{si}.stream"),
                i,
                &s.stream[i],
                &o.streams[si][i],
            );
        }
        for (ri, (lo, hi, want)) in s.draws.iter().enumerate() {
            for (i, w) in want.iter().enumerate() {
                rep.draws.push(
                    &format!("seed{si}.next_int({lo},{hi})"),
                    i,
                    w,
                    &o.draws[si][ri][i],
                );
            }
        }
        for i in 0..16 {
            rep.chance100.push(&format!("seed{si}.chance100"), i, &s.chance100[i], &o.chance100[si][i]);
            rep.chance2.push(&format!("seed{si}.chance2"), i, &s.chance2[i], &o.chance2[si][i]);
        }
        rep.chance1.push(&format!("seed{si}.chance1_all"), 0, &s.chance1_all, &o.chance1_all[si]);
        for i in 0..3 {
            rep.rand7.push(&format!("seed{si}.rand7"), i, &s.rand7[i], &o.rand7[si][i]);
            rep.rand7_next.push(&format!("seed{si}.rand7.next"), i, &s.rand7_next[i], &o.rand7_next[si][i]);
            rep.rand_empty.push(&format!("seed{si}.rand_empty"), i, &s.rand_empty_next[i], &o.rand_empty_next[si][i]);
            rep.shuffle8_next.push(&format!("seed{si}.shuffle8.next"), i, &s.shuffle8_next[i], &o.shuffle8_next[si][i]);
        }
        rep.rand_empty.push(&format!("seed{si}.rand_empty.consumed"), 0, &s.rand_empty_consumed, &o.rand_empty_consumed[si]);
        for i in 0..8 {
            rep.shuffle8.push(&format!("seed{si}.shuffle8"), i, &s.shuffle8[i], &o.shuffle8[si][i]);
        }
        for i in 0..24 {
            rep.pstream.push(&format!("seed{si}.pstream"), i, &s.pstream[i], &o.pstream[si][i]);
        }

        // Spawn jobs: bots then the scripted human.
        for (bi, sp) in s.spawns.iter().enumerate() {
            let got = &o.spawns[si][bi];
            rep.spawn_tile.push(&format!("seed{si}.bot{bi}.tile"), 0, &sp.tile, &got.tile);
            rep.spawn_xy.push(&format!("seed{si}.bot{bi}.x"), 0, &sp.x, &got.x);
            rep.spawn_xy.push(&format!("seed{si}.bot{bi}.y"), 0, &sp.y, &got.y);
            rep.spawn_lines.push(format!(
                "seed{si} bot{bi} id={} seed={} engine_tile={} ({},{}) port_tile={} ({},{}) attempts={} draws={} after=[{}] {}",
                sp.player_id, sp.seed, sp.tile, sp.x, sp.y, got.tile, got.x, got.y,
                got.attempts, got.draws, hex(&got.after),
                if sp.tile == got.tile { "MATCH" } else { "MISMATCH" }
            ));
        }
        if let Some(h) = &s.human {
            let bi = s.spawns.len();
            let got = &o.spawns[si][bi];
            rep.spawn_tile.push(&format!("seed{si}.human.tile"), 0, &h.tile, &got.tile);
            rep.spawn_xy.push(&format!("seed{si}.human.x"), 0, &h.x, &got.x);
            rep.spawn_xy.push(&format!("seed{si}.human.y"), 0, &h.y, &got.y);
            rep.spawn_lines.push(format!(
                "seed{si} human id={} seed={} engine_tile={} ({},{}) port_tile={} ({},{}) attempts={} draws={} after=[{}] {}",
                h.player_id, h.seed, h.tile, h.x, h.y, got.tile, got.x, got.y,
                got.attempts, got.draws, hex(&got.after),
                if h.tile == got.tile { "MATCH" } else { "MISMATCH" }
            ));
        }
    }
    // Engine-anchored `simple_hash` checks: these compare against integers the
    // *engine* printed, not against this crate's own copy of the function.
    for (si, s) in r.seeds.iter().enumerate() {
        rep.hash_engine.push(
            &format!("simple_hash(game_id[seed{si}]={})", s.game_id),
            0,
            &s.game_hash,
            &simple_hash(&s.game_id),
        );
        for (bi, sp) in s.spawns.iter().enumerate() {
            rep.hash_engine.push(
                &format!("simple_hash({})[seed{si}.bot{bi}]", sp.player_id),
                0,
                &sp.seed,
                &simple_hash(&sp.player_id).wrapping_add(s.game_hash),
            );
        }
        if let Some(h) = &s.human {
            rep.hash_engine.push(
                &format!("simple_hash({})[seed{si}.human]", h.player_id),
                0,
                &h.seed,
                &simple_hash(&h.player_id).wrapping_add(s.game_hash),
            );
        }
    }
    let hs = hash_strings(r);
    for (i, (s, want)) in o.hashes.iter().enumerate() {
        let _ = i;
        let idx = hs.iter().position(|x| x == s).unwrap_or(i);
        rep.hashes.push(&format!("hash[{idx}]"), idx, &simple_hash(s), want);
    }
    rep
}

pub fn format_report(r: &Reference, rep: &Report, side: &str) -> String {
    let mut out = String::new();
    out.push_str(&format!("=== {side} vs engine reference ===\n"));
    out.push_str(&format!("map {}  bots {}\n", r.map, r.bots));
    out.push_str(&format!(
        "seeds {}\n",
        r.seeds
            .iter()
            .map(|s| format!("{}={}", s.seed_str, s.game_id))
            .collect::<Vec<_>>()
            .join(" ")
    ));
    out.push_str(&format!("{}\n", rep.stream.line("next() 24 raw u32 per seed")));
    out.push_str(&format!("{}\n", rep.draws.line("next_int 16 draws x 4 ranges x 4 seeds")));
    out.push_str(&format!("{}\n", rep.pstream.line("simple_hash(player_id)-seeded 24 u32")));
    out.push_str(&format!(
        "semantics: {} | {} | {} | {} | {} | {} | {}\n",
        rep.chance100.line("chance(100)x16"),
        rep.chance2.line("chance(2)x16"),
        rep.chance1.line("chance(1)x16"),
        rep.rand7.line("rand_element(7)x3"),
        rep.rand7_next.line("  +next3"),
        rep.rand_empty.line("rand_element(empty) consumed/next3"),
        rep.shuffle8.line("shuffle_array(8)"),
    ));
    out.push_str(&format!("{}\n", rep.shuffle8_next.line("shuffle_array(8) +next3")));
    out.push_str(&format!("{}\n", rep.hashes.line("simple_hash(strings)")));
    out.push_str(&format!(
        "{}\n",
        rep.hash_engine.line("simple_hash vs engine-dumped hashes/seed sums")
    ));
    out.push_str("--- spawn tiles ---\n");
    for l in &rep.spawn_lines {
        out.push_str(l);
        out.push('\n');
    }
    out.push_str(&format!("{}\n", rep.spawn_tile.line("spawn tiles")));
    out.push_str(&format!("{}\n", rep.spawn_xy.line("spawn x/y")));
    out.push_str(&format!("{}\n", rep.spawn_after.line("post-spawn next3 draws")));
    out.push_str(&format!("{}\n", rep.cross.line("GPU vs CPU cross check")));
    match rep.first_divergence() {
        None => out.push_str("FIRST DIVERGENCE: none\n"),
        Some(d) => out.push_str(&format!("FIRST DIVERGENCE: {d}\n")),
    }
    out.push_str(&format!("bit_exact {}\n", rep.all_match()));
    out
}

/// Cross-check two port outputs against each other. The engine dump does not
/// carry the post-selection draws, so those are the only values a GPU-vs-CPU
/// comparison can add - and they are exactly the draw-count tripwire: a port
/// that consumed the wrong number of draws for a spawn job lands on a different
/// `after` triple.
pub fn compare_ports(gpu: &PortOutput, cpu: &PortOutput) -> Tally {
    let mut t = Tally::default();
    for (si, (g, c)) in gpu.spawns.iter().zip(cpu.spawns.iter()).enumerate() {
        for (ji, (gj, cj)) in g.iter().zip(c.iter()).enumerate() {
            for k in 0..3 {
                t.push(
                    &format!("seed{si}.job{ji}.after"),
                    k,
                    &cj.after[k],
                    &gj.after[k],
                );
            }
            t.push(
                &format!("seed{si}.job{ji}.draws"),
                0,
                &cj.draws,
                &gj.draws,
            );
            t.push(
                &format!("seed{si}.job{ji}.attempts"),
                0,
                &cj.attempts,
                &gj.attempts,
            );
        }
    }
    t
}

// ---------------------------------------------------------------------------
// Generic N-bot spawn / initial-state path
// ---------------------------------------------------------------------------
// The engine spawns every bot in a single tick, in `TribeSpawner::spawn_tribes`
// order. Each bot's `SpawnExecution` selects a tile against the owner plane
// left by the bots before it and the centres already placed
// (`spawn_util.rs:65-104` -> `too_close_to_existing_spawn`, `game.rs:3622`).
// The driver below is that path generalised to arbitrary N: it accumulates the
// owner overrides and the placed centres between bots, so `select_spawn` (and
// its device twin `spawn_select`) sees exactly the state the engine saw for
// that bot. Nothing here is N-specific.

impl<'a> SpawnCtx<'a> {
    /// The tile list `get_spawn_tiles(..., require_all_valid = true)` returns:
    /// the cardinal BFS component of `center` inside the radius-4 disk. Same
    /// BFS/settle as [`SpawnCtx::footprint`], but it keeps the tiles - a
    /// successful spawn conquers all of them (`conquer_spawn_tiles`,
    /// `game.rs:1227`), so this is the bot's initial owned set.
    pub fn spawn_tiles(&self, center: u32) -> Vec<u32> {
        const SIDE: usize = 8;
        const SIDE_I: i32 = 8;
        let w = self.width as i32;
        let h = self.height as i32;
        let cx = (center % self.width) as i32;
        let cy = (center / self.width) as i32;
        let mut seen = [false; SIDE * SIDE];
        let mut stack: [i32; SIDE * SIDE] = [-1; SIDE * SIDE];
        seen[36] = true;
        stack[0] = 36;
        let mut sp = 1usize;
        let mut out = Vec::new();
        while sp > 0 {
            sp -= 1;
            let d = stack[sp];
            let dx = d % SIDE_I - 4;
            let dy = d / SIDE_I - 4;
            let t = ((cy + dy) as u32) * self.width + (cx + dx) as u32;
            out.push(t);
            // Same neighbour order/guards as `map.rs:474-499` (N, S, W, E).
            if cy + dy - 1 >= 0 {
                bfs_push(&mut seen, &mut stack, &mut sp, dx, dy - 1);
            }
            if cy + dy + 1 < h {
                bfs_push(&mut seen, &mut stack, &mut sp, dx, dy + 1);
            }
            if cx + dx - 1 >= 0 {
                bfs_push(&mut seen, &mut stack, &mut sp, dx - 1, dy);
            }
            if cx + dx + 1 < w {
                bfs_push(&mut seen, &mut stack, &mut sp, dx + 1, dy);
            }
        }
        out
    }
}

/// `TribeSpawner` (`bot/tribe_spawner.rs:266-292`): a fresh
/// `PseudoRandom(simple_hash(game_id) + 2)`, then per bot two draws for the
/// tribe name and one `next_id()` draw. The name tables' *lengths* only
/// scale the draw result, never the draw count, so the id stream depends on
/// `game_id` alone.
pub fn tribe_bot_ids(game_id: &str, n: u32) -> Vec<String> {
    let mut pr = PseudoRandom::new(simple_hash(game_id).wrapping_add(2));
    (0..n)
        .map(|_| {
            pr.next_int(0, 1); // prefix pick (TRIBE_NAME_PREFIXES)
            pr.next_int(0, 1); // suffix pick (TRIBE_NAME_SUFFIXES)
            pr.next_id()
        })
        .collect()
}

/// One bot's port-side spawn, plus the initial owned set when it succeeded.
#[derive(Clone, Debug)]
pub struct BotSpawn {
    pub result: SpawnResult,
    pub tiles: Vec<u32>,
}

/// The whole engine ground-truth reference for one (N, nations) case, parsed
/// from an `ofcuda_spawn` dump.
#[derive(Clone, Debug, Default)]
pub struct SpawnReference {
    pub map: String,
    pub seed: String,
    pub game_id: String,
    pub game_hash: i32,
    pub agents: u32,
    pub nations: String,
    pub human_agents: u32,
    pub width: u32,
    pub height: u32,
    /// `game.wire.min_distance_between_players()`, read from the reference file
    /// (engine-sourced), never assumed.
    pub min_dist: u32,
    pub players: Vec<RefPlayer>,
    pub bots: Vec<RefBot>,
    /// `[bot]` -> `(tile, owner_small_id)` overrides the engine had in place
    /// before that bot selected.
    pub owner_before: Vec<Vec<(u32, u16)>>,
    /// `[bot]` -> already-placed spawn centres before that bot selected.
    pub prev_before: Vec<Vec<u32>>,
    /// `[bot]` -> the tiles the bot ends up owning.
    pub owned: Vec<Vec<u32>>,
}

#[derive(Clone, Debug, Default)]
pub struct RefPlayer {
    pub small_id: u16,
    pub ptype: char,
    pub id: String,
    pub spawn_tile: i64,
    pub spawn_tick: u32,
    pub tiles_owned: i64,
}

#[derive(Clone, Debug, Default)]
pub struct RefBot {
    pub small_id: u16,
    pub id: String,
    pub seed: i32,
    pub tile: i64,
    pub x: u32,
    pub y: u32,
    pub tiles_owned: i64,
}

pub fn parse_spawn_reference(text: &str) -> Result<SpawnReference, String> {
    let mut r = SpawnReference::default();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split_whitespace().collect();
        let err = |what: &str| format!("line {}: bad {what}: {line}", lineno + 1);
        let num_i64 = |s: &str| s.parse::<i64>().map_err(|_| err("int"));
        let num_u32 = |s: &str| s.parse::<u32>().map_err(|_| err("u32"));
        match f[0] {
            "map" => r.map = f[1].to_string(),
            "seed" => r.seed = f[1].to_string(),
            "game_id" => r.game_id = f[1].to_string(),
            "game_hash" => r.game_hash = f[1].parse().map_err(|_| err("game_hash"))?,
            "agents" => r.agents = num_u32(f[1])?,
            "nations" => r.nations = f[1].to_string(),
            "human_agents" => r.human_agents = num_u32(f[1])?,
            "width" => r.width = num_u32(f[1])?,
            "height" => r.height = num_u32(f[1])?,
            "min_dist" => r.min_dist = num_u32(f[1])?,
            "player" => r.players.push(RefPlayer {
                small_id: f[1].parse().map_err(|_| err("player small_id"))?,
                ptype: f[2].chars().next().ok_or_else(|| err("player type"))?,
                id: f[3].to_string(),
                spawn_tile: num_i64(f[4])?,
                spawn_tick: num_u32(f[5])?,
                tiles_owned: num_i64(f[6])?,
            }),
            "bot" => {
                let bi: usize = f[1].parse().map_err(|_| err("bot index"))?;
                if r.bots.len() <= bi {
                    r.bots.resize(bi + 1, RefBot::default());
                }
                r.bots[bi] = RefBot {
                    small_id: f[2].parse().map_err(|_| err("bot small_id"))?,
                    id: f[3].to_string(),
                    seed: f[4].parse().map_err(|_| err("bot seed"))?,
                    tile: num_i64(f[5])?,
                    x: num_u32(f[6])?,
                    y: num_u32(f[7])?,
                    tiles_owned: num_i64(f[8])?,
                };
            }
            "owner" => {
                let bi: usize = f[1].parse().map_err(|_| err("owner bot"))?;
                if r.owner_before.len() <= bi {
                    r.owner_before.resize(bi + 1, Vec::new());
                }
                r.owner_before[bi].push((
                    num_u32(f[2])?,
                    f[3].parse().map_err(|_| err("owner id"))?,
                ));
            }
            "prev" => {
                let bi: usize = f[1].parse().map_err(|_| err("prev bot"))?;
                if r.prev_before.len() <= bi {
                    r.prev_before.resize(bi + 1, Vec::new());
                }
                r.prev_before[bi].push(num_u32(f[2])?);
            }
            "owned" => {
                let bi: usize = f[1].parse().map_err(|_| err("owned bot"))?;
                if r.owned.len() <= bi {
                    r.owned.resize(bi + 1, Vec::new());
                }
                r.owned[bi].push(num_u32(f[2])?);
            }
            other => return Err(err(other)),
        }
    }
    if r.owner_before.len() < r.bots.len() {
        r.owner_before.resize(r.bots.len(), Vec::new());
    }
    if r.prev_before.len() < r.bots.len() {
        r.prev_before.resize(r.bots.len(), Vec::new());
    }
    if r.owned.len() < r.bots.len() {
        r.owned.resize(r.bots.len(), Vec::new());
    }
    if r.min_dist == 0 {
        // v1 references written before the field existed: the config for this
        // map is 30 (`core/config.rs:358-360`), and `spawn_bots_cpu` used 30.
        r.min_dist = 30;
    }
    Ok(r)
}

/// The generic N-bot spawn path (host): for each bot in engine order, select a
/// spawn against the accumulated state, then fold its footprint into the owner
/// plane and its centre into `prev`. Uses the same `select_spawn` the CUDA
/// kernel's `spawn_select` implements field-by-field.
pub fn spawn_bots_cpu(map: &MapPlane, r: &SpawnReference) -> Vec<BotSpawn> {
    let mut owner_plane: Vec<u16> = vec![0u16; (map.width as usize) * (map.height as usize)];
    let mut prev: Vec<u32> = Vec::new();
    let mut out = Vec::with_capacity(r.bots.len());
    for b in &r.bots {
        let ctx = SpawnCtx {
            terrain: &map.terrain,
            owner_tiles: &[],
            owner_ids: &[],
            prev: &prev,
            width: map.width,
            height: map.height,
            min_dist: 30,
            owner_plane: Some(&owner_plane),
        };
        let result = select_spawn(
            &ctx,
            &SpawnJob {
                seed: b.seed,
                explicit: u32::MAX,
            },
        );
        let tiles = if result.spawned {
            ctx.spawn_tiles(result.tile)
        } else {
            Vec::new()
        };
        if result.spawned {
            for t in &tiles {
                owner_plane[*t as usize] = b.small_id;
            }
            prev.push(result.tile);
        }
        out.push(BotSpawn { result, tiles });
    }
    out
}

/// Why a bot failed to spawn, measured rather than assumed.
///
/// Replays the first `upto` bots exactly (same `select_spawn`), then sweeps the
/// whole map for the *next* bot's chances and classifies every unowned land tile
/// with the very predicates `select_spawn` uses:
/// * `land_unowned`   - land tiles with no owner,
/// * `valid_footprint`- of those, tiles whose footprint is entirely
///   land+unowned (a spawn *could* be placed on that centre),
/// * `too_close`      - of those, centres rejected by the min-distance rule,
/// * `far_enough`     - of those, centres that survive everything.
///
/// `far_enough == 0` while `valid_footprint > 0` means no try in the 1000-draw
/// loop can succeed: retries are exhausted by the *min-distance* rule, not by
/// land capacity (the two constraints are counted separately on purpose).
pub fn starvation_diagnostic(map: &MapPlane, r: &SpawnReference, upto: usize) -> Starvation {
    let w = map.width;
    let h = map.height;
    let md = r.min_dist.max(1);
    let mut seat: Vec<u16> = vec![0u16; (w as usize) * (h as usize)];
    let mut prev: Vec<u32> = Vec::new();
    for b in r.bots.iter().take(upto) {
        let ctx = SpawnCtx {
            terrain: &map.terrain,
            owner_tiles: &[],
            owner_ids: &[],
            prev: &prev,
            width: w,
            height: h,
            min_dist: md,
            owner_plane: Some(&seat),
        };
        let res = select_spawn(
            &ctx,
            &SpawnJob {
                seed: b.seed,
                explicit: u32::MAX,
            },
        );
        if res.spawned {
            for t in ctx.spawn_tiles(res.tile) {
                seat[t as usize] = b.small_id;
            }
            prev.push(res.tile);
        }
    }
    let ctx = SpawnCtx {
        terrain: &map.terrain,
        owner_tiles: &[],
        owner_ids: &[],
        prev: &prev,
        width: w,
        height: h,
        min_dist: md,
        owner_plane: Some(&seat),
    };
    let mut st = Starvation::default();
    for t in 0..(w * h) {
        if !ctx.is_land(t) || ctx.has_owner(t) {
            continue;
        }
        st.land_unowned += 1;
        let (_, invalid) = ctx.footprint(t);
        if invalid > 0 {
            st.footprint_bad += 1;
            continue;
        }
        st.valid_footprint += 1;
        if ctx.is_border(t) {
            st.border += 1;
        } else if ctx.too_close(t) {
            st.too_close += 1;
        } else {
            st.far_enough += 1;
        }
    }
    st
}

/// Counts from [`starvation_diagnostic`].
#[derive(Default, Debug, Clone)]
pub struct Starvation {
    pub land_unowned: u64,
    pub valid_footprint: u64,
    pub border: u64,
    pub too_close: u64,
    pub far_enough: u64,
    pub footprint_bad: u64,
}

/// The result of one (N, nations) case.
#[derive(Default, Debug)]
pub struct SpawnReport {
    /// engine vs port tribe ids, from the shared `TribeSpawner` stream.
    pub ids: Tally,
    /// engine vs port spawn tile (only meaningful when both spawned).
    pub tile: Tally,
    /// engine `spawn_tile.is_some()` vs port `spawned`.
    pub spawned: Tally,
    /// engine vs port initial owned tile SET per bot.
    pub owned: Tally,
    /// engine vs port initial tile count per bot.
    pub count: Tally,
    pub lines: Vec<String>,
}

impl SpawnReport {
    pub fn first_divergence(&self) -> Option<String> {
        for t in [&self.ids, &self.spawned, &self.tile, &self.owned, &self.count] {
            if let Some(d) = &t.first_divergence {
                return Some(d.clone());
            }
        }
        None
    }
    pub fn all_match(&self) -> bool {
        self.first_divergence().is_none()
    }
    pub fn line(&self) -> String {
        format!(
            "ids {}/{}  spawned {}/{}  tiles {}/{}  owned-set {}/{}  counts {}/{}  {}",
            self.ids.matched,
            self.ids.total,
            self.spawned.matched,
            self.spawned.total,
            self.tile.matched,
            self.tile.total,
            self.owned.matched,
            self.owned.total,
            self.count.matched,
            self.count.total,
            match self.first_divergence() {
                None => String::new(),
                Some(d) => format!("FIRST {d}"),
            }
        )
    }
}

fn sorted_set(v: &[u32]) -> Vec<u32> {
    let mut s = v.to_vec();
    s.sort_unstable();
    s.dedup();
    s
}

/// Compare the port's N-bot plan against the engine reference.
pub fn compare_spawn_bots(r: &SpawnReference, plan: &[BotSpawn], port_ids: &[String]) -> SpawnReport {
    let mut rep = SpawnReport::default();
    for (bi, b) in r.bots.iter().enumerate() {
        let engine_spawned = b.tile >= 0;
        let got = &plan[bi];
        // Bot ids: the port's TribeSpawner stream must reproduce the engine's.
        if let Some(pid) = port_ids.get(bi) {
            rep.ids.push(&format!("bot{bi}.id"), bi, &b.id, pid);
        }
        rep.spawned
            .push(&format!("bot{bi}.spawned"), bi, &engine_spawned, &got.result.spawned);
        if engine_spawned && got.result.spawned {
            rep.tile
                .push(&format!("bot{bi}.tile"), bi, &(b.tile as u32), &got.result.tile);
        } else if engine_spawned {
            rep.tile
                .push(&format!("bot{bi}.tile"), bi, &(b.tile as u32), &u32::MAX);
        } else if got.result.spawned {
            rep.tile.push(&format!("bot{bi}.tile"), bi, &u32::MAX, &got.result.tile);
        }
        let want = sorted_set(&r.owned[bi]);
        let have = sorted_set(&got.tiles);
        rep.owned.push(&format!("bot{bi}.owned"), bi, &want, &have);
        rep.count.push(
            &format!("bot{bi}.count"),
            bi,
            &b.tiles_owned,
            &(got.tiles.len() as i64),
        );
    }
    rep
}

pub fn format_spawn_report(r: &SpawnReference, rep: &SpawnReport, side: &str) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "=== {side} vs engine (spawn/initial-state) ===\n"
    ));
    out.push_str(&format!(
        "map {}  seed {}  game_id {}  agents {}  nations {}  humans {}\n",
        r.map, r.seed, r.game_id, r.agents, r.nations, r.human_agents
    ));
    out.push_str(&format!("{}\n", rep.line()));
    for l in &rep.lines {
        out.push_str(l);
        out.push('\n');
    }
    out.push_str(&format!("bit_exact {}\n", rep.all_match()));
    out
}
