#[cuda_module]
mod kernels {
    use super::*;

    /// sfc32, 1:1 with `rust/engine/src/prng.rs`.
    struct Prng {
        s0: i32,
        s1: i32,
        s2: i32,
        s3: i32,
        /// `next()` calls made, including the 12 warm-ups.
        calls: u32,
    }

    impl Prng {
        fn new(seed: i32) -> Self {
            let mut h = seed;
            let mut split = || {
                h = h.wrapping_add(0x9e37_79b9u32 as i32);
                let mut t = h ^ ((h as u32) >> 16) as i32;
                t = t.wrapping_mul(0x21f0_aaad);
                t ^= ((t as u32) >> 15) as i32;
                t = t.wrapping_mul(0x735a_2d97);
                t ^ ((t as u32) >> 15) as i32
            };
            let mut pr = Prng {
                s0: split(),
                s1: split(),
                s2: split(),
                s3: split(),
                calls: 0,
            };
            // TRAP 1: the 12 warm-up draws.
            let mut i = 0;
            while i < 12 {
                pr.next_u32();
                i += 1;
            }
            pr
        }

        /// `prng.rs:37-44`. The raw `t`, i.e. `next() * 2^32`.
        fn next_u32(&mut self) -> u32 {
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

        /// TRAP 2: `next()` is `(t as u32) as f64 / 2^32` - a u32 division.
        fn next(&mut self) -> f64 {
            self.next_u32() as f64 / 4_294_967_296.0
        }

        /// `prng.rs:47-51`. `hi > lo` always here, so `next() * (hi - lo)` is
        /// non-negative and the engine's `floor` is the `as i32` truncation.
        fn next_int(&mut self, lo: i32, hi: i32) -> i32 {
            let v = self.next() * (hi - lo) as f64;
            v as i32 + lo
        }

        /// `prng.rs:59-61`.
        fn chance(&mut self, odds: i32) -> bool {
            self.next_int(0, odds) == 0
        }

        /// TRAP 3: `prng.rs:86-92` - an empty list draws nothing.
        fn rand_element(&mut self, len: usize) -> i32 {
            if len == 0 {
                return -1;
            }
            self.next_int(0, len as i32)
        }

        /// TRAP 4: `prng.rs:103-110` - Fisher-Yates, exactly `len - 1` draws.
        fn shuffle8(&mut self) -> [i32; 8] {
            let mut arr: [i32; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
            let mut i = arr.len();
            while i > 1 {
                i -= 1;
                let j = self.next_int(0, (i + 1) as i32) as usize;
                let tmp = arr[i];
                arr[i] = arr[j];
                arr[j] = tmp;
            }
            arr
        }

        fn draws(&self) -> u32 {
            self.calls - 12
        }
    }

    /// `rust/engine/src/util.rs:4-13`.
    fn simple_hash(bytes: &[u8]) -> i32 {
        let mut hash: i32 = 0;
        let mut i = 0;
        while i < bytes.len() {
            let c = bytes[i] as i32;
            hash = hash.wrapping_shl(5).wrapping_sub(hash).wrapping_add(c);
            hash = hash.wrapping_mul(1);
            i += 1;
        }
        if hash < 0 { -hash } else { hash }
    }

    /// (a) `out[si * 24 + j]` = the (j+1)-th raw `next()` of `Prng::new(seed)`.
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1))]
    pub fn stream24(seeds: &[i32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let g = idx.get();
        let si = g / 24;
        if si >= seeds.len() {
            return;
        }
        let j = g % 24;
        let mut pr = Prng::new(seeds[si]);
        let mut v = 0u32;
        let mut k = 0;
        while k <= j {
            v = pr.next_u32();
            k += 1;
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = v;
        }
    }

    /// (b) `out[job * 16 + k]`, job = si * 4 + ri.
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1))]
    pub fn draws16(seeds: &[i32], ranges: &[i32], mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        let g = idx.get();
        let job = g / 16;
        let si = job / 4;
        if si >= seeds.len() {
            return;
        }
        let ri = job % 4;
        let lo = ranges[ri * 2];
        let hi = ranges[ri * 2 + 1];
        let j = g % 16;
        let mut pr = Prng::new(seeds[si]);
        let mut v = 0i32;
        let mut k = 0;
        while k <= j {
            v = pr.next_int(lo, hi);
            k += 1;
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = v;
        }
    }

    /// One word of the semantics table. 54 words per seed:
    ///   [0,16)   chance(100) flags
    ///   [16,32)  chance(2) flags
    ///   [32]     chance(1) all-true
    ///   [33,36)  rand_element(7) picks
    ///   [36,39)  the 3 draws after them
    ///   [39]     draws consumed by rand_element(empty)
    ///   [40,43)  the 3 draws after the empty call
    ///   [43,51)  shuffle_array(8) result
    ///   [51,54)  the 3 draws after the shuffle
    fn sem_word(seed: i32, w: usize) -> u32 {
        if w < 16 {
            let mut c = Prng::new(seed);
            let mut v = 0u32;
            let mut i = 0;
            while i <= w {
                v = c.chance(100) as u32;
                i += 1;
            }
            return v;
        }
        if w < 32 {
            // The engine's `chance(2)` run continues on the SAME prng after the
            // 16 `chance(100)` calls, so the prefix must be replayed.
            let mut c = Prng::new(seed);
            let mut i = 0;
            while i < 16 {
                c.chance(100);
                i += 1;
            }
            let mut v = 0u32;
            i = 0;
            while i <= w - 16 {
                v = c.chance(2) as u32;
                i += 1;
            }
            return v;
        }
        if w == 32 {
            // ... and `chance(1)` continues after the 32 earlier calls.
            let mut c = Prng::new(seed);
            let mut i = 0;
            while i < 16 {
                c.chance(100);
                i += 1;
            }
            i = 0;
            while i < 16 {
                c.chance(2);
                i += 1;
            }
            let mut all = 1u32;
            i = 0;
            while i < 16 {
                if !c.chance(1) {
                    all = 0;
                }
                i += 1;
            }
            return all;
        }
        if w < 36 {
            let mut e = Prng::new(seed);
            let mut v = 0i32;
            let mut i = 0;
            while i <= w - 33 {
                v = e.rand_element(7);
                i += 1;
            }
            return v as u32;
        }
        if w < 39 {
            let mut e = Prng::new(seed);
            let mut i = 0;
            while i < 3 {
                e.rand_element(7);
                i += 1;
            }
            let mut v = 0u32;
            i = 0;
            while i <= w - 36 {
                v = e.next_u32();
                i += 1;
            }
            return v;
        }
        if w == 39 {
            // TRAP 3: an empty `rand_element` must consume nothing.
            let mut z = Prng::new(seed);
            let before = (z.s0, z.s1, z.s2, z.s3);
            let got = z.rand_element(0);
            let after = (z.s0, z.s1, z.s2, z.s3);
            return if before == after && got == -1 { 0 } else { 1 };
        }
        if w < 43 {
            let mut z = Prng::new(seed);
            z.rand_element(0);
            let mut v = 0u32;
            let mut i = 0;
            while i <= w - 40 {
                v = z.next_u32();
                i += 1;
            }
            return v;
        }
        if w < 51 {
            let mut s = Prng::new(seed);
            let arr = s.shuffle8();
            return arr[w - 43] as u32;
        }
        let mut s = Prng::new(seed);
        s.shuffle8();
        let mut v = 0u32;
        let mut i = 0;
        while i <= w - 51 {
            v = s.next_u32();
            i += 1;
        }
        v
    }

    /// (c) the semantics table, one thread per word.
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1))]
    pub fn sem(seeds: &[i32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let g = idx.get();
        let si = g / 54;
        if si >= seeds.len() {
            return;
        }
        let v = sem_word(seeds[si], g % 54);
        if let Some(slot) = out.get_mut(idx) {
            *slot = v;
        }
    }

    /// (d) the `simple_hash(player_id)`-seeded stream.
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1))]
    pub fn pstream24(hashes: &[i32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let g = idx.get();
        let si = g / 24;
        if si >= hashes.len() {
            return;
        }
        let j = g % 24;
        let mut pr = Prng::new(hashes[si]);
        let mut v = 0u32;
        let mut k = 0;
        while k <= j {
            v = pr.next_u32();
            k += 1;
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = v;
        }
    }

    /// `simple_hash` over a byte arena; element i hashes `bytes[off[i]..off[i+1]]`.
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1))]
    pub fn hash_strings(bytes: &[u8], off: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let t = idx.get();
        if t + 1 >= off.len() {
            return;
        }
        let a = off[t] as usize;
        let b = off[t + 1] as usize;
        let h = simple_hash(&bytes[a..b]);
        if let Some(slot) = out.get_mut(idx) {
            *slot = h as u32;
        }
    }

    /// BFS enqueue test: the engine's filter is
    /// `(dx + 0.5)^2 + (dy + 0.5)^2 <= 16.0` (`map.rs:431-437`), which is
    /// exactly `(2dx+1)^2 + (2dy+1)^2 <= 64` in integers. Returns the bit to
    /// set, or 0.
    #[inline]
    fn bfs_bit(dx: i32, dy: i32, cx: i32, cy: i32, w: i32, h: i32) -> u64 {
        let a = 2 * dx + 1;
        let b = 2 * dy + 1;
        if a * a + b * b > 64 {
            return 0;
        }
        if cx + dx < 0 || cx + dx >= w || cy + dy < 0 || cy + dy >= h {
            return 0;
        }
        // Offsets live in -4..=3 (offset -4 -> -3.5, 12.25 <= 16; +4 -> 20.25).
        1u64 << (((dy + 4) * 8 + (dx + 4)) as u32)
    }

    #[inline]
    fn owner_of(owner_tiles: &[u32], owner_ids: &[u16], t: u32) -> u16 {
        let mut k = 0usize;
        while k < owner_tiles.len() {
            if owner_tiles[k] == t {
                return owner_ids[k];
            }
            k += 1;
        }
        0
    }

    /// `get_spawn_tiles` (`spawn_util.rs:118-146`): the cardinal BFS component
    /// of the centre inside the radius-4 disk, then the validity test. Returns
    /// `(n_tiles, n_invalid)`. The BFS is a bitmask fixed point over the 8x8
    /// window - no queue, no scratch memory, same reachable set as the engine's
    /// `seen`-stamped stack.
    #[inline]
    fn footprint(
        terrain: &[u8],
        owner_tiles: &[u32],
        owner_ids: &[u16],
        center: u32,
        w: u32,
        h: u32,
    ) -> (u32, u32) {
        let cx = (center % w) as i32;
        let cy = (center / w) as i32;
        let wi = w as i32;
        let hi = h as i32;

        let mut cur: u64 = 1u64 << 36; // offset (0,0)
        let mut round = 0;
        while round < 64 {
            let mut next = cur;
            let mut i = 0usize;
            while i < 64 {
                if (cur >> i) & 1 == 1 {
                    let dx = (i % 8) as i32 - 4;
                    let dy = (i / 8) as i32 - 4;
                    // Same neighbour set as `map.rs:474-499` (N, S, W, E).
                    next |= bfs_bit(dx, dy - 1, cx, cy, wi, hi);
                    next |= bfs_bit(dx, dy + 1, cx, cy, wi, hi);
                    next |= bfs_bit(dx - 1, dy, cx, cy, wi, hi);
                    next |= bfs_bit(dx + 1, dy, cx, cy, wi, hi);
                }
                i += 1;
            }
            if next == cur {
                break;
            }
            cur = next;
            round += 1;
        }

        let mut n_tiles = 0u32;
        let mut n_invalid = 0u32;
        let mut i = 0usize;
        while i < 64 {
            if (cur >> i) & 1 == 1 {
                let dx = (i % 8) as i32 - 4;
                let dy = (i / 8) as i32 - 4;
                let t = ((cy + dy) as u32) * w + (cx + dx) as u32;
                n_tiles += 1;
                if owner_of(owner_tiles, owner_ids, t) > 0 || (terrain[t as usize] & 0x80) == 0 {
                    n_invalid += 1;
                }
            }
            i += 1;
        }
        (n_tiles, n_invalid)
    }

    /// `map.rs:405-425`.
    #[inline]
    fn is_border(owner_tiles: &[u32], owner_ids: &[u16], t: u32, w: u32, h: u32) -> bool {
        let owner = owner_of(owner_tiles, owner_ids, t);
        let x = t % w;
        if x > 0 && owner_of(owner_tiles, owner_ids, t - 1) != owner {
            return true;
        }
        if x + 1 < w && owner_of(owner_tiles, owner_ids, t + 1) != owner {
            return true;
        }
        if t >= w && owner_of(owner_tiles, owner_ids, t - w) != owner {
            return true;
        }
        if t < (h - 1) * w && owner_of(owner_tiles, owner_ids, t + w) != owner {
            return true;
        }
        false
    }

    /// The whole spawn selection for one job, returning one field of
    /// `[tile, x, y, attempts, draws, spawned, after0, after1, after2, 0]`.
    /// `explicit == u32::MAX` means no explicit tile (the random path).
    fn spawn_field(
        terrain: &[u8],
        owner_tiles: &[u32],
        owner_ids: &[u16],
        prev: &[u32],
        width: u32,
        height: u32,
        min_dist: u32,
        seed: i32,
        exp: u32,
        field: usize,
    ) -> u32 {
        let mut pr = Prng::new(seed);
        let mut tile = u32::MAX;
        let mut tx = u32::MAX;
        let mut ty = u32::MAX;
        let mut attempts = 0u32;
        let mut spawned = 0u32;

        if exp != u32::MAX {
            // Explicit tile: `find_spawn` returns before `rand_tile`
            // (`spawn_util.rs:71-77`) - ZERO draws.
            let (n_tiles, n_invalid) = footprint(terrain, owner_tiles, owner_ids, exp, width, height);
            tile = exp;
            tx = exp % width;
            ty = exp / width;
            if n_tiles - n_invalid > 0 {
                spawned = 1;
            }
        } else {
            let mut tries = 0u32;
            while tries < 1_000 {
                tries += 1;
                // `rand_tile` with `spawn_area == None`: exactly two draws.
                let x = pr.next_int(0, width as i32);
                let y = pr.next_int(0, height as i32);
                let center = (y as u32) * width + (x as u32);

                if (terrain[center as usize] & 0x80) == 0
                    || owner_of(owner_tiles, owner_ids, center) > 0
                    || is_border(owner_tiles, owner_ids, center, width, height)
                {
                    continue;
                }
                let mut too_close = false;
                let cxx = center % width;
                let cyy = center / width;
                let mut k = 0usize;
                while k < prev.len() {
                    let dx = (prev[k] % width).abs_diff(cxx);
                    let dy = (prev[k] / width).abs_diff(cyy);
                    if dx + dy < min_dist {
                        too_close = true;
                        break;
                    }
                    k += 1;
                }
                if too_close {
                    continue;
                }
                let (_n, n_invalid) = footprint(terrain, owner_tiles, owner_ids, center, width, height);
                if n_invalid > 0 {
                    continue;
                }
                tile = center;
                tx = center % width;
                ty = center / width;
                attempts = tries;
                spawned = 1;
                break;
            }
            if spawned == 0 {
                attempts = 1_000;
            }
        }

        if field == 4 {
            return pr.draws();
        }
        if field >= 6 && field <= 8 {
            // The three post-selection draws: a draw-count tripwire.
            let mut v = 0u32;
            let mut i = 0;
            while i <= field - 6 {
                v = pr.next_u32();
                i += 1;
            }
            return v;
        }
        match field {
            0 => tile,
            1 => tx,
            2 => ty,
            3 => attempts,
            5 => spawned,
            _ => 0,
        }
    }

    /// The whole spawn selection, one thread per output word (10 per job).
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1))]
    pub fn spawn_select(
        terrain: &[u8],
        owner_tiles: &[u32],
        owner_ids: &[u16],
        owner_off: &[u32],
        prev: &[u32],
        prev_off: &[u32],
        width: u32,
        height: u32,
        min_dist: u32,
        seeds: &[i32],
        explicit: &[u32],
        mut out: DisjointSlice<u32>,
    ) {
        let idx = thread::index_1d();
        let g = idx.get();
        let job = g / 10;
        if job >= seeds.len() {
            return;
        }
        let field = g % 10;
        let o0 = owner_off[job] as usize;
        let o1 = owner_off[job + 1] as usize;
        let p0 = prev_off[job] as usize;
        let p1 = prev_off[job + 1] as usize;
        let v = spawn_field(
            terrain,
            &owner_tiles[o0..o1],
            &owner_ids[o0..o1],
            &prev[p0..p1],
            width,
            height,
            min_dist,
            seeds[job],
            explicit[job],
            field,
        );
        if let Some(slot) = out.get_mut(idx) {
            *slot = v;
        }
    }
}
