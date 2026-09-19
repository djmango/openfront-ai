// Shared core of the attack/expansion FRONTIER step.
//
// This file is the canonical source of the frontier core. It is included by the
// host CPU reference (`src/lib.rs`, `mod core_impl`) and copied VERBATIM into
// cuda-oxide's device module (`src/main.rs`, `mod kernels` - see the BEGIN/END
// VERBATIM COPY markers there, and the `device_core_is_a_verbatim_copy` test in
// lib.rs that fails the moment the two drift apart). So the PRNG, the priority
// arithmetic, the candidate filter and the min-heap have exactly one
// implementation that can disagree with the engine -- not one per side.
// (It is `#[no_std]`-clean on purpose: fixed-size arrays only.)
//
// 1:1 ports, with citations:
//   * sfc32 `PseudoRandom`                - rust/engine/src/prng.rs:23-51
//   * `FlatBinaryHeap`                    - rust/engine/src/execution/flat_heap.rs:31-80
//   * `AttackExecution::add_neighbors`    - rust/engine/src/execution/attack.rs:1340-1387
//   * `GameMap::for_each_neighbor_nswe`   - rust/engine/src/map.rs:347-368
//   * `GameMap::terrain_type`             - rust/engine/src/map.rs:214-227
//   * `TerrainType -> mag`                - rust/engine/src/execution/attack.rs:1372-1377
//                                           (same switch in TS AttackExecution.ts:361-374)
//   * `AttackExecution::tick` (the loop)  - rust/engine/src/execution/attack.rs:206-324
//     - the persistent `to_conquer` heap  - attack.rs:17, carried across ticks; only
//                                           `refresh_to_conquer` (attack.rs:1265) clears it
//     - refill ends the tick              - attack.rs:264-268
//     - the pop-skip rule                 - attack.rs:284
//     - in-tick `add_neighbors`           - attack.rs:292 (BEFORE `conquer`, 320)
//     - the one budget draw per tick      - attack.rs:239-255
//   * `AttackExecution::init` refresh     - attack.rs:160-164 (no budget draw before it)
//   * `Game::execute_next_tick` scheduling - game.rs:3657-3700 (a new exec is appended at
//                                           the END of a tick, so its first `tick()` is the
//                                           NEXT tick: `init` refresh at T, pops from T+1)

/// Enqueue capacity of the host-side flat heap.
///
/// The engine's `FlatBinaryHeap` is a growable `Vec` (`Vec::with_capacity(1024)`
/// is only a hint - `execution/flat_heap.rs:8,26-27`), i.e. unbounded. A fixed
/// 256 silently DROPPED 13180 candidates in the composed b002 window and its
/// high-water mark sat exactly on the cap, which is what first desynced the
/// claim order (tick 558). 8192 is a bound, not the engine's behaviour, so the
/// run reports `Heap::drops` and `Heap::peak` instead of assuming it is not
/// binding - and `STATE_HEAP_CAP` below keeps the *serialized* width at the
/// device module's own 256.
pub const HEAP_CAP: usize = 1024;

/// Width of the heap in the *serialized* attack state (`STATE_WORDS`). This is
/// deliberately separate from [`HEAP_CAP`]: the device module (`src/main.rs`)
/// has its own fixed 256-entry heap and reads `STATE_WORDS` from here, so the
/// serialization stride must stay 256 even if the host-side heap grows to match
/// the engine's (unbounded, growable) `FlatBinaryHeap`.
pub const STATE_HEAP_CAP: usize = 256;

/// The border-tile iteration order. It is an *input*, not a constant, because
/// the two engines disagree about it and which one a record was produced under
/// is exactly what has to be shown (see README.md).
pub const ORDER_NSWE: u32 = 0; // N, S, W, E - TS GameMap.neighbors4, and the native tip
pub const ORDER_WENS: u32 = 1; // W, E, N, S - native `for_each_neighbor4` before the fix

pub const SEED: i32 = 123; // `AttackExecution.random = PseudoRandom::new(123)`

// ---------------------------------------------------------------------------
// sfc32 - rust/engine/src/prng.rs:23-51
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct Prng {
    pub s0: i32,
    pub s1: i32,
    pub s2: i32,
    pub s3: i32,
    /// `next()` calls made, including the 12 warm-ups.
    pub calls: u32,
}

impl Prng {
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
        let mut pr = Prng {
            s0: split(),
            s1: split(),
            s2: split(),
            s3: split(),
            calls: 0,
        };
        // The 12 warm-up draws (`prng.rs:30-32`). Missing them desynchronises
        // every value downstream, so they are drawn here and nowhere else.
        let mut i = 0;
        while i < 12 {
            pr.next_u32();
            i += 1;
        }
        pr
    }

    /// `prng.rs:37-44`: the raw `t`, i.e. `next() * 2^32`.
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

    /// `prng.rs:44`: `(t as u32) as f64 / 2^32` - a u32 division, not an f64 one.
    pub fn next(&mut self) -> f64 {
        self.next_u32() as f64 / 4_294_967_296.0
    }

    /// `prng.rs:47-51`. `hi > lo` here, so `floor` is the `as i32` truncation.
    pub fn next_int(&mut self, lo: i32, hi: i32) -> i32 {
        (self.next() * (hi - lo) as f64) as i32 + lo
    }
}

// ---------------------------------------------------------------------------
// Terrain magnitude - rust/engine/src/map.rs:214-227 + attack.rs:1372-1377
// ---------------------------------------------------------------------------

/// `mag * 2` as an integer so the value crosses the host/device boundary
/// without a float: 0 = Ocean/Impassable, 2 = Plains(1.0), 3 = Highland(1.5),
/// 4 = Mountain(2.0).
#[inline]
pub fn mag2_from_terrain(terrain: u8) -> u32 {
    if terrain & 0x80 == 0 {
        return 0; // not land -> the `_ => 0.0` arm
    }
    let m = (terrain & 0x1f) as u32;
    if m < 10 {
        2 // Plains -> 1.0
    } else if m < 20 {
        3 // Highland -> 1.5
    } else {
        4 // Mountain -> 2.0 (mag >= 20, including the old impassable mag 31)
    }
}

// ---------------------------------------------------------------------------
// 4-neighbourhood in an explicit visit order
// ---------------------------------------------------------------------------

/// Fills `out` with the cardinal neighbours of `t` in the requested order and
/// returns how many were written. Off-map neighbours are dropped exactly as the
/// engine's guards do (`map.rs:309-331`).
#[inline]
pub fn neighbors4(order: u32, t: u32, w: u32, h: u32, out: &mut [u32; 4]) -> u32 {
    let x = t % w;
    let mut n = 0usize;
    if order == ORDER_WENS {
        if x != 0 {
            out[n] = t - 1;
            n += 1;
        }
        if x != w - 1 {
            out[n] = t + 1;
            n += 1;
        }
        if t >= w {
            out[n] = t - w;
            n += 1;
        }
        if t < (h - 1) * w {
            out[n] = t + w;
            n += 1;
        }
    } else {
        if t >= w {
            out[n] = t - w;
            n += 1;
        }
        if t < (h - 1) * w {
            out[n] = t + w;
            n += 1;
        }
        if x != 0 {
            out[n] = t - 1;
            n += 1;
        }
        if x != w - 1 {
            out[n] = t + 1;
            n += 1;
        }
    }
    n as u32
}

#[inline]
pub fn owned_contains(owned: &[u32], t: u32) -> bool {
    let mut i = 0usize;
    while i < owned.len() {
        if owned[i] == t {
            return true;
        }
        i += 1;
    }
    false
}

// ---------------------------------------------------------------------------
// Priority - attack.rs:1379-1384
// ---------------------------------------------------------------------------

/// `(r + 10) * (1 - num_owned_by_me * 0.5 + mag / 2) + tick`, as f32.
///
/// The engine evaluates this in f64 and casts the result to f32. Every factor
/// here is exact in binary (r + 10 is a small integer, the bracket is a
/// multiple of 0.25, tick is an integer), so the f32 and f64 evaluations are
/// bit-identical; `priority_f64` exists so the CPU side can *measure* that
/// rather than assume it. `fma` contraction cannot change the result either,
/// for the same reason (the product is exact), but `main.rs` also greps the
/// generated PTX for it.
#[inline]
pub fn priority_f32(r: i32, num_owned: u32, mag2: u32, tick: u32) -> f32 {
    let a = (r + 10) as f32;
    // mag / 2 == (mag * 2) / 4
    let b = 1.0f32 - (num_owned as f32) * 0.5f32 + (mag2 as f32) * 0.25f32;
    a * b + tick as f32
}

/// The engine's own f64 evaluation, for the equivalence measurement.
#[inline]
pub fn priority_f64(r: i32, num_owned: u32, mag2: u32, tick: u32) -> f32 {
    let p = (r as f64 + 10.0)
        * (1.0 - num_owned as f64 * 0.5 + (mag2 as f64 / 2.0) / 2.0)
        + tick as f64;
    p as f32
}

// ---------------------------------------------------------------------------
// FlatBinaryHeap - execution/flat_heap.rs:31-80, tie-breaks included
// ---------------------------------------------------------------------------

/// Sift-up breaks on `priority >= parent`; sift-down picks the RIGHT child only
/// when `pri[right] < pri[left]` and breaks on `last_pri <= pri[child]`. Those
/// ties are part of the frontier specification: with equal priorities the
/// resulting tile order is heap-structural, not insertion order.
pub struct Heap {
    pub pri: [f32; HEAP_CAP],
    pub tiles: [u32; HEAP_CAP],
    pub len: usize,
    /// Candidates refused because `len == HEAP_CAP`. The engine's
    /// `FlatBinaryHeap` is a growable `Vec`, so a *nonzero* value here is a
    /// divergence, not a detail: the refused tile would have been claimed later.
    pub drops: u64,
    /// High-water mark of `len` over the heap's life, so a run can say whether
    /// `HEAP_CAP` is anywhere near binding instead of assuming it is not.
    pub peak: usize,
}

impl Heap {
    pub fn new() -> Self {
        Heap {
            pri: [0.0; HEAP_CAP],
            tiles: [0; HEAP_CAP],
            len: 0,
            drops: 0,
            peak: 0,
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn enqueue(&mut self, tile: u32, priority: f32) {
        // Bound guard: the engine's FlatBinaryHeap is a growable Vec, this one
        // is a fixed array. Overflowing it would be UB in release, so a full
        // heap drops the candidate instead - and a dropped candidate is a tile
        // the engine would have claimed later, so `drops` is reported rather
        // than assumed away (`HEAP_CAP` is sized from the measured `peak`).
        if self.len >= HEAP_CAP {
            self.drops += 1;
            return;
        }
        let mut i = self.len;
        self.len += 1;
        if self.len > self.peak {
            self.peak = self.len;
        }
        while i > 0 {
            let parent = (i - 1) >> 1;
            if priority >= self.pri[parent] {
                break;
            }
            self.pri[i] = self.pri[parent];
            self.tiles[i] = self.tiles[parent];
            i = parent;
        }
        self.pri[i] = priority;
        self.tiles[i] = tile;
    }

    /// Returns the tile **and** the priority it was popped at (the engine
    /// returns the tile only; the priority is needed for the proof).
    pub fn dequeue(&mut self) -> Option<(u32, f32)> {
        if self.len == 0 {
            return None;
        }
        let top = self.tiles[0];
        let top_pri = self.pri[0];
        self.len -= 1;
        let last_pri = self.pri[self.len];
        let last_tile = self.tiles[self.len];
        if self.len == 0 {
            return Some((top, top_pri));
        }
        let mut i = 0usize;
        let n = self.len;
        loop {
            let left = (i << 1) + 1;
            if left >= n {
                break;
            }
            let right = left + 1;
            let child = if right < n && self.pri[right] < self.pri[left] {
                right
            } else {
                left
            };
            if last_pri <= self.pri[child] {
                break;
            }
            self.pri[i] = self.pri[child];
            self.tiles[i] = self.tiles[child];
            i = child;
        }
        self.pri[i] = last_pri;
        self.tiles[i] = last_tile;
        Some((top, top_pri))
    }
}

// ---------------------------------------------------------------------------
// The frontier step itself - attack.rs:1265-1274 + 1340-1387
// ---------------------------------------------------------------------------

/// Enqueue every border tile's eligible 4-neighbours, then hand back the heap.
///
/// Eligibility is `add_neighbors`' filter: the neighbour must be land and must
/// not be owned (the engine compares `owner_id(neighbor) != target_small_id`,
/// and this frontier's target is terra nullius, owner 0), so `owned_any` is
/// the set of tiles owned by *anybody* at the moment the frontier is built.
/// The owner count that feeds the priority is a different set: it counts the
/// **attacker's** tiles only (`owner_id(inner) == owner_small_id`), hence
/// `owned_mine`. In the recorded cases the other players are far away, so the
/// two sets give the same counts -- but conflating them would be a real bug on
/// a contested border, so they are kept separate.
///
/// THE invariant that silently breaks everything: exactly ONE `next_int(0,7)`
/// draw per ENQUEUED neighbour, in visit order. The draw happens after both
/// filters, so a rejected neighbour consumes nothing.
pub fn frontier_enqueue(
    border: &[u32],
    owned_any: &[u32],
    owned_mine: &[u32],
    terrain: &[u8],
    width: u32,
    height: u32,
    order: u32,
    tick: u32,
    cursor: u32,
    heap: &mut Heap,
) -> u32 {
    let mut pr = Prng::new(SEED);
    // The shared stream is not necessarily at position 0 when the step starts:
    // its cursor is an explicit input (see README.md "the stream cursor").
    let mut skip = 0u32;
    while skip < cursor {
        pr.next_u32();
        skip += 1;
    }

    let mut n_enq = 0u32;
    let mut bi = 0usize;
    while bi < border.len() {
        let bt = border[bi];
        let mut nbuf = [0u32; 4];
        let n = neighbors4(order, bt, width, height, &mut nbuf);
        let mut k = 0u32;
        while k < n {
            let nb = nbuf[k as usize];
            k += 1;
            if terrain[nb as usize] & 0x80 == 0 {
                continue; // water
            }
            if owned_contains(owned_any, nb) {
                continue; // wrong owner (`owner_id(nb) != target_small_id`)
            }

            // --- exactly one draw, and only now ---
            let r = pr.next_int(0, 7);

            let mut ibuf = [0u32; 4];
            let inner = neighbors4(order, nb, width, height, &mut ibuf);
            let mut num_owned_by_me = 0u32;
            let mut j = 0u32;
            while j < inner {
                if owned_contains(owned_mine, ibuf[j as usize]) {
                    num_owned_by_me += 1;
                }
                j += 1;
            }

            let mag2 = mag2_from_terrain(terrain[nb as usize]);
            heap.enqueue(nb, priority_f32(r, num_owned_by_me, mag2, tick));
            n_enq += 1;
        }
        bi += 1;
    }
    n_enq
}

// ---------------------------------------------------------------------------
// The ENGINE's attack loop, state carried BETWEEN ticks - attack.rs:206-324
// ---------------------------------------------------------------------------

/// Claim-list capacity of one ATTACK's whole life, not one tick: the engine
/// keeps `to_conquer` between ticks (`attack.rs:17`) and `conquer` writes the
/// map (`attack.rs:320`), so every ownership test in a later tick sees every
/// earlier claim. The measured b002 window claims 133 tiles over 13 ticks; the
/// contested case claims 27 in one. 256 covers both (2 KiB of local memory).
pub const CLAIM_CAP: usize = 256;

/// Words of one attack's carried state (`EngineAttack::state_words`):
/// `[s0,s1,s2,s3,calls, heap_len, heap tiles[HEAP_CAP], heap priority bits[HEAP_CAP]]`.
pub const STATE_WORDS: usize = 6 + 2 * STATE_HEAP_CAP;

impl Heap {
    /// Copy a carried heap back in (`to_conquer` persisting across ticks,
    /// attack.rs:17). The caller's buffers are the device's global memory.
    pub fn load(&mut self, tiles: &[u32], pri: &[f32]) {
        self.len = 0;
        let mut i = 0usize;
        while i < tiles.len() && i < pri.len() && i < HEAP_CAP {
            self.tiles[i] = tiles[i];
            self.pri[i] = pri[i];
            self.len += 1;
            i += 1;
        }
    }

    /// The carried content, for the write-back to device memory.
    pub fn tiles_slice(&self) -> &[u32] {
        &self.tiles[..self.len]
    }

    pub fn pri_slice(&self) -> &[f32] {
        &self.pri[..self.len]
    }
}

impl Prng {
    /// Rebuild the stream from a carried state (`AttackExecution.random` is a
    /// field of the attack, so the stream survives the tick boundary).
    pub fn from_state(words: &[u32]) -> Self {
        Prng {
            s0: words[0] as i32,
            s1: words[1] as i32,
            s2: words[2] as i32,
            s3: words[3] as i32,
            calls: if words.len() > 4 { words[4] } else { 0 },
        }
    }

    /// `[s0, s1, s2, s3, calls]` - five words, so `calls` (the draw counter) is
    /// observable and the account can be reported rather than argued.
    pub fn state_words(&self, out: &mut [u32; 5]) {
        out[0] = self.s0 as u32;
        out[1] = self.s1 as u32;
        out[2] = self.s2 as u32;
        out[3] = self.s3 as u32;
        out[4] = self.calls;
    }

    /// attack.rs:253 - `border_size + self.random.next_int(0, 5)`, the ONE draw
    /// a tick spends on its budget before the pop loop. It is consumed even
    /// though this port takes the resulting claim count from the record: the
    /// draw is what the *next* enqueue in the same tick sees, so skipping it
    /// shifts every in-tick draw by one (measured: 20/32 -> 32/32 on the b002
    /// window, first bad consumer tick 320).
    pub fn budget_draw(&mut self) -> i32 {
        self.next_int(0, 5)
    }
}

/// attack.rs:284 - the tile is still terra nullius (`owner_id(t) ==
/// target_small_id`, and a land attack's target is `terra_nullius_id() == 0`,
/// game.rs:1092) AND at least one of its 4-neighbours is owned by the attacker
/// (`owner_id(n) == owner_small_id`, attack.rs:278-282). `claimed` is the list
/// of tiles THIS tick has already conquered, which the engine sees in the map
/// because `conquer` (attack.rs:320) runs per claim.
#[inline]
pub fn is_terra_nullius(plane: &[u16], claimed: &[u32], t: u32) -> bool {
    if plane[t as usize] != 0 {
        return false;
    }
    let mut i = 0usize;
    while i < claimed.len() {
        if claimed[i] == t {
            return false;
        }
        i += 1;
    }
    true
}

#[inline]
pub fn has_attacker_neighbor(
    plane: &[u16],
    claimed: &[u32],
    owner_sid: u16,
    t: u32,
    width: u32,
    height: u32,
    order: u32,
) -> bool {
    let mut nbuf = [0u32; 4];
    let n = neighbors4(order, t, width, height, &mut nbuf);
    let mut i = 0usize;
    while i < n as usize {
        let nb = nbuf[i];
        i += 1;
        if plane[nb as usize] == owner_sid {
            return true;
        }
        let mut j = 0usize;
        while j < claimed.len() {
            if claimed[j] == nb {
                return true;
            }
            j += 1;
        }
    }
    false
}

/// The owner count that feeds the priority: `num_owned_by_me` over the tile's
/// 4-neighbours (attack.rs:1365-1370), attacker-owned only, including this
/// tick's claims.
#[inline]
pub fn attacker_neighbor_count(
    plane: &[u16],
    claimed: &[u32],
    owner_sid: u16,
    t: u32,
    width: u32,
    height: u32,
    order: u32,
) -> u32 {
    let mut nbuf = [0u32; 4];
    let n = neighbors4(order, t, width, height, &mut nbuf);
    let mut k = 0u32;
    let mut i = 0usize;
    while i < n as usize {
        let nb = nbuf[i];
        i += 1;
        if plane[nb as usize] == owner_sid {
            k += 1;
            continue;
        }
        let mut j = 0usize;
        while j < claimed.len() {
            if claimed[j] == nb {
                k += 1;
                break;
            }
            j += 1;
        }
    }
    k
}

/// attack.rs:1340-1386 - `add_neighbors`. Visits N,S,W,E (`for_each_neighbor_nswe`,
/// map.rs:347-368; `for_each_neighbor4` is an alias of it, map.rs:331-333), skips
/// water (attack.rs:1354) and anything not owned by the target (attack.rs:1359,
/// target = terra nullius here), and draws exactly ONE `next_int(0, 7)` per
/// ENQUEUED neighbour (attack.rs:1380) - the draw that binds jitter to tile in
/// visit order.
pub fn add_neighbors(
    heap: &mut Heap,
    pr: &mut Prng,
    tile: u32,
    plane: &[u16],
    claimed: &[u32],
    owner_sid: u16,
    terrain: &[u8],
    width: u32,
    height: u32,
    order: u32,
    tick: u32,
) -> u32 {
    let mut nbuf = [0u32; 4];
    let n = neighbors4(order, tile, width, height, &mut nbuf);
    let mut enq = 0u32;
    let mut i = 0usize;
    while i < n as usize {
        let nb = nbuf[i];
        i += 1;
        if terrain[nb as usize] & 0x80 == 0 {
            continue; // water (attack.rs:1354)
        }
        if !is_terra_nullius(plane, claimed, nb) {
            continue; // wrong owner (attack.rs:1359)
        }
        let k = attacker_neighbor_count(plane, claimed, owner_sid, nb, width, height, order);
        let r = pr.next_int(0, 7);
        let mag2 = mag2_from_terrain(terrain[nb as usize]);
        heap.enqueue(nb, priority_f32(r, k, mag2, tick));
        enq += 1;
    }
    enq
}

/// attack.rs:1265-1274 - `refresh_to_conquer`: clear the heap, then offer every
/// neighbour of the (game's) border set. The border set is an INPUT here (the
/// attack's own `border_tiles` is what the budget reads, and the budget is
/// exogenous - see the README's measured/assumed split).
pub fn refresh_to_conquer(
    heap: &mut Heap,
    pr: &mut Prng,
    border: &[u32],
    plane: &[u16],
    claimed: &[u32],
    owner_sid: u16,
    terrain: &[u8],
    width: u32,
    height: u32,
    order: u32,
    tick: u32,
) -> u32 {
    heap.clear();
    let mut n_enq = 0u32;
    let mut i = 0usize;
    while i < border.len() {
        let bt = border[i];
        i += 1;
        n_enq += add_neighbors(
            heap, pr, bt, plane, claimed, owner_sid, terrain, width, height, order, tick,
        );
    }
    n_enq
}

pub struct EngineTickOut {
    /// Claims made by THIS tick.
    pub claims: u32,
    pub n_enq: u32,
    /// `to_conquer` ran dry mid-tick: the engine called `refresh_to_conquer`
    /// and then `retreat` (attack.rs:264-268), so the tick ENDED and the
    /// refilled heap is carried into the next one.
    pub refilled: bool,
}

/// `AttackExecution::tick` - attack.rs:206-324, in the order the engine runs it.
///
/// `plane` is the ownership plane at the START of the tick (0 = terra nullius).
/// The tick's own claims are the only mutations the engine's loop makes to the
/// map before the tick is over, so they are carried as `claims` and folded into
/// every ownership test instead of being written back per pop - which also
/// makes the function usable read-only from a device thread.
///
/// `claims` is the attack's **whole-life** claim list: `claimed_so_far` entries
/// from earlier ticks (the engine's `conquer` at attack.rs:320 wrote them into
/// the map, so `owner_id(t) != target_small_id` for them) followed by the tiles
/// this tick claims, appended from index `claimed_so_far`. Keep it across ticks
/// and ownership stays right without a second plane.
///
/// `budget` is the integer number of claims the tick is allowed (the float
/// `num_tiles_per_tick` decremented by `tiles_used`, attack.rs:313, is taken
/// from the record - see the README).
#[allow(clippy::too_many_arguments)]
pub fn engine_tick(
    heap: &mut Heap,
    pr: &mut Prng,
    plane: &[u16],
    owner_sid: u16,
    border: &[u32],
    terrain: &[u8],
    width: u32,
    height: u32,
    order: u32,
    tick: u32,
    budget: u32,
    consume_budget_draw: bool,
    in_tick_add_neighbors: bool,
    claimed_so_far: u32,
    claims: &mut [u32; CLAIM_CAP],
    claim_pri_bits: &mut [u32; CLAIM_CAP],
) -> EngineTickOut {
    // attack.rs:239-255: the budget draw, taken before the pop loop.
    if consume_budget_draw {
        pr.budget_draw();
    }

    let base = claimed_so_far as usize;
    let mut n = base;
    let mut n_enq = 0u32;
    let mut refilled = false;

    // attack.rs:257 `while num_tiles_per_tick > 0.0`
    while ((n - base) as u32) < budget {
        // attack.rs:258-262 `troop_count < 1.0 -> kill_attack` is not ported:
        // troops are not modelled (see the README).
        if heap.is_empty() {
            // attack.rs:264-268: refill, then retreat -> the tick ends here and
            // the refilled heap is what the NEXT tick pops from. `tick` here is
            // `game.ticks()` at the moment of the refill (`refresh_to_conquer`,
            // attack.rs:1269), i.e. the current step's own parameter.
            n_enq += refresh_to_conquer(
                heap,
                pr,
                border,
                plane,
                &claims[..n],
                owner_sid,
                terrain,
                width,
                height,
                order,
                tick,
            );
            refilled = true;
            break;
        }

        let Some((tile_to_conquer, pri)) = heap.dequeue() else {
            break; // attack.rs:271 `?` - unreachable after the emptiness check
        };

        // attack.rs:284: still terra nullius AND on the attacker's border.
        if !is_terra_nullius(plane, &claims[..n], tile_to_conquer) {
            continue;
        }
        if !has_attacker_neighbor(
            plane,
            &claims[..n],
            owner_sid,
            tile_to_conquer,
            width,
            height,
            order,
        ) {
            continue;
        }
        // attack.rs:288 `is_land` guard.
        if terrain[tile_to_conquer as usize] & 0x80 == 0 {
            continue;
        }

        // attack.rs:292 - add_neighbors BEFORE the conquer, so the tile being
        // claimed is still terra nullius for its own neighbours' owner counts.
        // `in_tick_add_neighbors == false` is the control that measures how much
        // this rule alone moves the window.
        if in_tick_add_neighbors {
            n_enq += add_neighbors(
                heap,
                pr,
                tile_to_conquer,
                plane,
                &claims[..n],
                owner_sid,
                terrain,
                width,
                height,
                order,
                tick,
            );
        }

        // attack.rs:313-320: budget -= tiles_used, conquer.
        claims[n] = tile_to_conquer;
        claim_pri_bits[n] = pri.to_bits();
        n += 1;
    }

    EngineTickOut {
        claims: (n - base) as u32,
        n_enq,
        refilled,
    }
}

// ---------------------------------------------------------------------------
// One attack's whole life: the state that survives the tick boundary
// ---------------------------------------------------------------------------

/// `AttackExecution`'s carried state, as far as the frontier step needs it:
/// `to_conquer` (`attack.rs:17`), the attack's own `random` stream
/// (`attack.rs:21`), and the claims already written into the map by `conquer`
/// (`attack.rs:320`).
pub struct EngineAttack {
    pub heap: Heap,
    pub pr: Prng,
    /// Every tile this attack has conquered, in claim order, across all ticks.
    pub claims: [u32; CLAIM_CAP],
    /// The priority each claim was popped at, as f32 bits.
    pub claim_pri_bits: [u32; CLAIM_CAP],
    pub claimed: u32,
    pub n_enq: u32,
    /// Ticks in which `to_conquer` ran dry (`refresh_to_conquer` + `retreat`,
    /// attack.rs:264-268).
    pub refreshes: u32,
}

impl EngineAttack {
    /// `AttackExecution::init` (attack.rs:160-164) - and NOTHING else: the
    /// init refresh is the attack's first frontier and its first draw, so
    /// `init_tick` is the step that runs ONE BEFORE the first pop tick.
    /// Measured, not assumed: on the b002 window the dump shows the attack for
    /// owner 2 first present in the tick-318 record (its init step is 317, its
    /// first pops 318, the claims land in the 319 record - `gen_case_files.py`
    /// derives the tap ticks the same way).
    #[allow(clippy::too_many_arguments)]
    pub fn init(
        border: &[u32],
        plane: &[u16],
        owner_sid: u16,
        terrain: &[u8],
        width: u32,
        height: u32,
        order: u32,
        init_tick: u32,
    ) -> Self {
        let mut a = EngineAttack {
            heap: Heap::new(),
            pr: Prng::new(SEED),
            claims: [0; CLAIM_CAP],
            claim_pri_bits: [0; CLAIM_CAP],
            claimed: 0,
            n_enq: 0,
            refreshes: 0,
        };
        a.n_enq += refresh_to_conquer(
            &mut a.heap,
            &mut a.pr,
            border,
            plane,
            &[],
            owner_sid,
            terrain,
            width,
            height,
            order,
            init_tick,
        );
        a.refreshes = 1;
        a
    }

    /// One `AttackExecution::tick` (attack.rs:206-324) on the carried state.
    /// `tick` is the step parameter the engine passes in (`game.rs:3659`
    /// `let tick = self.ticks`), which is also the tick the in-tick
    /// `add_neighbors` stamps its priorities with.
    #[allow(clippy::too_many_arguments)]
    pub fn tick(
        &mut self,
        plane: &[u16],
        owner_sid: u16,
        border: &[u32],
        terrain: &[u8],
        width: u32,
        height: u32,
        order: u32,
        tick: u32,
        budget: u32,
        consume_budget_draw: bool,
        in_tick_add_neighbors: bool,
    ) -> EngineTickOut {
        let out = engine_tick(
            &mut self.heap,
            &mut self.pr,
            plane,
            owner_sid,
            border,
            terrain,
            width,
            height,
            order,
            tick,
            budget,
            consume_budget_draw,
            in_tick_add_neighbors,
            self.claimed,
            &mut self.claims,
            &mut self.claim_pri_bits,
        );
        self.claimed += out.claims;
        self.n_enq += out.n_enq;
        if out.refilled {
            self.refreshes += 1;
        }
        out
    }

    pub fn claims(&self) -> &[u32] {
        &self.claims[..self.claimed as usize]
    }

    /// The carried state as words, so a device thread can publish it and the
    /// host can compare it against its own run instead of trusting it:
    /// `[s0, s1, s2, s3, calls, heap_len, heap tiles.., heap priority bits..]`.
    pub fn state_words(&self, out: &mut [u32; STATE_WORDS]) {
        let mut pr = [0u32; 5];
        self.pr.state_words(&mut pr);
        let mut i = 0usize;
        while i < 5 {
            out[i] = pr[i];
            i += 1;
        }
        out[5] = self.heap.len as u32;
        let mut j = 0usize;
        while j < STATE_HEAP_CAP {
            out[6 + j] = if j < self.heap.len { self.heap.tiles[j] } else { u32::MAX };
            out[6 + STATE_HEAP_CAP + j] = self.heap.pri[j].to_bits();
            j += 1;
        }
    }

    /// Rebuild from `state_words` (`state_words`' layout) so a device thread
    /// can pick a carried attack up from global memory. `claims_so_far` and
    /// `claims` carry the attack's conquest map, which is a separate array.
    pub fn from_state_words(words: &[u32], claims_so_far: u32, claims: &mut [u32; CLAIM_CAP]) -> Self {
        let mut len = words[5] as usize;
        if len > STATE_HEAP_CAP {
            len = STATE_HEAP_CAP;
        }
        let mut a = EngineAttack {
            heap: Heap::new(),
            pr: Prng::from_state(&words[0..5]),
            claims: [0; CLAIM_CAP],
            claim_pri_bits: [0; CLAIM_CAP],
            claimed: claims_so_far,
            n_enq: 0,
            refreshes: 0,
        };
        let mut i = 0usize;
        while i < len {
            a.heap.tiles[i] = words[6 + i];
            a.heap.pri[i] = f32::from_bits(words[6 + STATE_HEAP_CAP + i]);
            i += 1;
        }
        a.heap.len = len;
        a.heap.peak = len;
        a.heap.drops = 0;
        let mut k = 0usize;
        while k < claims_so_far as usize && k < CLAIM_CAP {
            a.claims[k] = claims[k];
            k += 1;
        }
        a
    }
}
