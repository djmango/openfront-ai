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
/// is only a hint - `execution/flat_heap.rs:8,26-27`), i.e. its frontier is
/// unbounded. A fixed 256 silently DROPPED 13180 candidates in the composed
/// b002 window and its high-water mark sat exactly on the cap, which is what
/// first desynced the claim order (tick 558). 1024 was raised from that, and it
/// saturated AGAIN in the 1000-tick pangaea n18 window: the ENGINE's own
/// recorded `to_conquer` reached 1062 tiles (boundary 873 / engine tick 876,
/// owner 2) and the device refused 17 candidates with its high-water mark
/// exactly on 1024 - which is the missing-tile divergence at boundary 806
/// (device heap 1009 vs engine 1014, engine border 338).
///
/// INVARIANT - checked, never assumed: the device must never refuse a
/// candidate, because a refused candidate is a silently dropped tile and
/// therefore a parity break. A refusal is now FATAL (the device module fails the
/// cell and prints both the refusal count and the measured high-water mark), so
/// this constant must be raised from a MEASURED peak with real headroom, never
/// by guesswork: 8192 is 7.7x the largest frontier the engine has recorded in a
/// 1000-tick window (1062). If a refusal ever fires, take `max heap peak` from
/// the run's `### DEVICE RESOURCES` block and raise this from that number.
pub const HEAP_CAP: usize = 8192;

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

    // =====================================================================
    // player-clusters pass (`engine/src/execution/player_clusters.rs`)
    // =====================================================================
    //
    // `PlayerExecution::tick` (`execution/player.rs:92`) calls
    // `maybe_remove_clusters` (`player_clusters.rs:358`) for EVERY player exec,
    // and the player execs are ticked BEFORE the attack execs in
    // `execute_next_tick`, so a cluster handed to a captor is already owned by
    // the captor when that tick's attacks run. Without this pass the device's
    // claim set is the engine's MINUS those tiles - the boundary-1667 stop at
    // N=18/2000t (one removal in 1671 ticks, tile 141384) and the
    // boundary-302 stop at N=488/500t (64 tiles to player 369).
    //
    // Run by ONE thread: the pass walks the engine's exec order and each
    // player's border set in ORDER, and every `remove_cluster` mutates the
    // plane in place for the players after it, exactly as the engine's
    // `game.conquer` does. A parallel kernel would need a different order
    // guarantee than the engine has.

    /// `player_clusters.rs:8`.
    pub const TICKS_PER_CLUSTER_CALC: u32 = 20;
    /// Capacity of one player's border set in the cluster pass. Measured: the
    /// engine's own `border_tiles` peaks around a few thousand on these maps.
    pub const CB: usize = 8192;
    /// Cluster tile storage for ONE player's pass: the clusters partition the
    /// border set, so their tile counts sum to at most the border length.
    pub const CS: usize = 8192;
    /// Flood stack for the border-cluster flood (`<= CB` entries by
    /// construction: a tile is pushed at most once).
    pub const CSTACK: usize = 8192;
    /// Flood stack / result buffer for `flood_owned`. Capacity is counted, not
    /// assumed: `owned_overflow` in `CSTAT` is non-zero if any component was
    /// larger.
    pub const COWN: usize = 8192;
    /// `u32` words of cluster state per player id.
    pub const CSTATE: usize = 4;
    /// `u32` words of cluster-pass statistics (`out`).
    pub const CSTAT: usize = 8;
    /// FIXED (header) words per removal record in the `rem` stream: `victim`,
    /// `captor`, `count`, then `count` tile words (see `cluster_pass_core`'s
    /// doc). The driver walks the stream with that same 3-word header
    /// (`ofcuda_matrix/src/main.rs`: `victim = rem[woff]`, `captor =
    /// rem[woff+1]`, `n = rem[woff+2]`, `woff += 3`, then `woff += n`), so the
    /// per-record advance must be `CWR + count`. Advancing by 4 instead left a
    /// word of STALE data between records: the first record of a tick still
    /// parsed, but the second was read one word late - its `victim` was the
    /// stale word and its `count` was the real `captor`, which overran the
    /// stream and made the driver `break`. The removal then vanished from the
    /// captor's claim list (engine tick 366 / boundary 363 was the first pass
    /// with two removals, so the first to show it).
    pub const CWR: usize = 3;
    /// Distinct bordering enemies tracked by `get_capturing_player`.
    pub const CNB: usize = 4096;
    /// Capacity of the `rem` stream.
    pub const CREM: usize = 16384;

    #[inline]
    fn cl_bit_get(v: &[u32], i: usize) -> u32 {
        (v[i >> 5] >> (i & 31)) & 1
    }

    #[inline]
    fn cl_bit_set(v: &mut [u32], i: usize) {
        v[i >> 5] |= 1u32 << (i & 31);
    }

    #[inline]
    fn cl_bit_clear(v: &mut [u32], nbits: usize) {
        let mut i = 0usize;
        let words = (nbits + 31) / 32;
        while i < words {
            v[i] = 0;
            i += 1;
        }
    }

    #[inline]
    fn cl_is_land(terrain: &[u8], t: u32) -> bool {
        terrain[t as usize] & 0x80 != 0
    }

    #[inline]
    fn cl_is_ocean(terrain: &[u8], t: u32) -> bool {
        terrain[t as usize] & (1 << 5) != 0
    }

    /// `map.rs:144` `is_shore` = land and shoreline.
    #[inline]
    fn cl_is_shore(terrain: &[u8], t: u32) -> bool {
        let b = terrain[t as usize];
        b & 0x80 != 0 && b & (1 << 6) != 0
    }

    /// `map.rs:282-301`. NOTE the order dependence is only on the *result*
    /// (any ocean 4-neighbour), so W,E,N,S is equivalent here.
    #[inline]
    fn cl_is_ocean_shore(terrain: &[u8], w: u32, h: u32, t: u32) -> bool {
        if !cl_is_land(terrain, t) {
            return false;
        }
        let x = t % w;
        if x > 0 && cl_is_ocean(terrain, t - 1) {
            return true;
        }
        if x + 1 < w && cl_is_ocean(terrain, t + 1) {
            return true;
        }
        if t >= w && cl_is_ocean(terrain, t - w) {
            return true;
        }
        if t < (h - 1) * w && cl_is_ocean(terrain, t + w) {
            return true;
        }
        false
    }

    /// `map.rs:303-307`.
    #[inline]
    fn cl_is_edge(w: u32, h: u32, t: u32) -> bool {
        let x = t % w;
        let y = t / w;
        x == 0 || x + 1 == w || y == 0 || y + 1 == h
    }

    /// `map.rs:363-380` `neighbors_nswe`: north, south, west, east.
    #[inline]
    fn cl_neighbors4(w: u32, h: u32, t: u32, buf: &mut [u32; 4]) -> usize {
        let x = t % w;
        let mut n = 0usize;
        if t >= w {
            buf[n] = t - w;
            n += 1;
        }
        if t < (h - 1) * w {
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

    /// `map.rs:251-281` `for_each_neighbor8`: NW, W, SW, N, S, NE, E, SE.
    #[inline]
    fn cl_neighbors8(w: u32, h: u32, t: u32, buf: &mut [u32; 8]) -> usize {
        let x = t % w;
        let has_n = t >= w;
        let has_s = t < (h - 1) * w;
        let mut n = 0usize;
        if x > 0 {
            if has_n {
                buf[n] = t - 1 - w;
                n += 1;
            }
            buf[n] = t - 1;
            n += 1;
            if has_s {
                buf[n] = t - 1 + w;
                n += 1;
            }
        }
        if has_n {
            buf[n] = t - w;
            n += 1;
        }
        if has_s {
            buf[n] = t + w;
            n += 1;
        }
        if x + 1 < w {
            if has_n {
                buf[n] = t + 1 - w;
                n += 1;
            }
            buf[n] = t + 1;
            n += 1;
            if has_s {
                buf[n] = t + 1 + w;
                n += 1;
            }
        }
        n
    }

    /// `Game::is_friendly(a, b)` (`game.rs:3341` -> `is_friendly_ex(a,b,false)`):
    /// `a == b`, else NOT friendly if `b` is disconnected, else same team or
    /// allied. The pairs are PRE-COMPUTED on the host from the engine's own
    /// `alliances`/`team`/`is_disconnected` state (the oracle's `FRIEND` rows),
    /// so the disconnected rule and the alliance list live in exactly one place.
    #[inline]
    fn cl_friendly(friends: &[u32], nfriends: u32, a: u16, b: u16) -> bool {
        if a == b {
            return true;
        }
        let key = ((a as u32) << 16) | b as u32;
        let mut i = 0u32;
        while i < nfriends {
            if friends[i as usize] == key {
                return true;
            }
            i += 1;
        }
        false
    }

    fn cl_bbox_of(w: u32, cluster: &[u32], bb: &mut [u32; 4]) {
        let mut min_x = u32::MAX;
        let mut min_y = u32::MAX;
        let mut max_x = 0u32;
        let mut max_y = 0u32;
        let mut i = 0usize;
        while i < cluster.len() {
            let t = cluster[i];
            i += 1;
            let x = t % w;
            let y = t / w;
            if x < min_x {
                min_x = x;
            }
            if y < min_y {
                min_y = y;
            }
            if x > max_x {
                max_x = x;
            }
            if y > max_y {
                max_y = y;
            }
        }
        bb[0] = min_x;
        bb[1] = min_y;
        bb[2] = max_x;
        bb[3] = max_y;
    }

    #[inline]
    fn cl_inscribed(o: &[u32; 4], i: &[u32; 4]) -> bool {
        o[0] <= i[0] && o[1] <= i[1] && o[2] >= i[2] && o[3] >= i[3]
    }

    /// `player_clusters.rs:93-119` `flood_border_cluster`. Marked on
    /// DISCOVERY, not on pop (`player_clusters.rs:86-92`), and the resulting
    /// tile order is the engine's `OrderedTiles` insertion order - it feeds
    /// `cluster.first()` in `remove_cluster` (:334) and `get_capturing_player`'s
    /// first-seen neighbour order (:264-280), so it is load-bearing.
    /// `visited` is indexed by the tile's position in `border` (`t2b`), which is
    /// equivalent to the engine's `HashSet<TileRef> visited` because every
    /// candidate neighbour is tested with `border.contains(&n)` first.
    #[allow(clippy::too_many_arguments)]
    fn cl_flood_cluster(
        w: u32,
        h: u32,
        border: &[u32],
        t2b: &[u32],
        start_pos: u32,
        vis: &mut [u32],
        stack: &mut [u32],
        cbuf: &mut [u32],
        cbase: usize,
        slen: &mut usize,
    ) -> usize {
        let mut n = 0usize;
        if cl_bit_get(vis, start_pos as usize) == 0 {
            cl_bit_set(vis, start_pos as usize);
            cbuf[cbase] = border[start_pos as usize];
            n = 1;
            stack[0] = start_pos;
            *slen = 1;
        }
        while *slen > 0 {
            *slen -= 1;
            let p = stack[*slen];
            let t = border[p as usize];
            let mut nb = [0u32; 8];
            let cnt = cl_neighbors8(w, h, t, &mut nb);
            let mut i = 0usize;
            while i < cnt {
                let q = t2b[nb[i] as usize];
                i += 1;
                if q == u32::MAX || cl_bit_get(vis, q as usize) != 0 {
                    continue;
                }
                cl_bit_set(vis, q as usize);
                if cbase + n < CS {
                    cbuf[cbase + n] = border[q as usize];
                    n += 1;
                }
                if *slen < CSTACK {
                    stack[*slen] = q;
                    *slen += 1;
                }
            }
        }
        n
    }

    /// `player_clusters.rs:121-141` `calculate_clusters`. Starts are walked in
    /// the border set's own order; `visited` is shared across starts.
    #[allow(clippy::too_many_arguments)]
    fn cl_calculate_clusters(
        w: u32,
        h: u32,
        border: &[u32],
        blen: usize,
        t2b: &[u32],
        vis: &mut [u32],
        stack: &mut [u32],
        cbuf: &mut [u32],
        clen: &mut [u32],
        coff: &mut [u32],
        overflow: &mut u32,
    ) -> usize {
        cl_bit_clear(vis, blen);
        let mut ncl = 0usize;
        let mut cbase = 0usize;
        let mut slen = 0usize;
        let mut j = 0usize;
        while j < blen {
            if cl_bit_get(vis, j) != 0 {
                j += 1;
                continue;
            }
            let n = cl_flood_cluster(
                w,
                h,
                border,
                t2b,
                j as u32,
                vis,
                stack,
                cbuf,
                cbase,
                &mut slen,
            );
            if n > 0 {
                if ncl >= CB || cbase + n > CS {
                    *overflow += 1;
                } else {
                    clen[ncl] = n as u32;
                    coff[ncl] = cbase as u32;
                    ncl += 1;
                    cbase += n;
                }
            }
            j += 1;
        }
        ncl
    }

    /// `player_clusters.rs:143-208` `surrounded_by_same_enemy`. Returns the
    /// enemy small id, or `u16::MAX` for the engine's `None`.
    #[allow(clippy::too_many_arguments)]
    fn cl_surrounded_by_same_enemy(
        terrain: &[u8],
        plane: &[u16],
        w: u32,
        h: u32,
        friends: &[u32],
        nfriends: u32,
        sid: u16,
        cluster: &[u32],
        cbb: &[u32; 4],
    ) -> u16 {
        let mut enemy: u16 = u16::MAX;
        let mut ebb = [0u32; 4];
        let mut init = false;
        let mut i = 0usize;
        while i < cluster.len() {
            let tile = cluster[i];
            i += 1;
            if cl_is_ocean_shore(terrain, w, h, tile) || cl_is_edge(w, h, tile) {
                return u16::MAX;
            }
            let mut nb = [0u32; 4];
            let n = cl_neighbors4(w, h, tile, &mut nb);
            let mut j = 0usize;
            while j < n {
                let nt = nb[j];
                j += 1;
                let owner = plane[nt as usize];
                if owner == 0 {
                    return u16::MAX;
                }
                if owner == sid {
                    continue;
                }
                if enemy == u16::MAX {
                    enemy = owner;
                } else if enemy != owner {
                    return u16::MAX;
                }
                let x = nt % w;
                let y = nt / w;
                if !init {
                    ebb[0] = x;
                    ebb[1] = y;
                    ebb[2] = x;
                    ebb[3] = y;
                    init = true;
                } else {
                    if x < ebb[0] {
                        ebb[0] = x;
                    }
                    if y < ebb[1] {
                        ebb[1] = y;
                    }
                    if x > ebb[2] {
                        ebb[2] = x;
                    }
                    if y > ebb[3] {
                        ebb[3] = y;
                    }
                }
            }
            if enemy == u16::MAX {
                return u16::MAX;
            }
        }
        if enemy == u16::MAX {
            return u16::MAX;
        }
        if cl_friendly(friends, nfriends, enemy, sid) {
            return u16::MAX;
        }
        if cl_inscribed(&ebb, cbb) {
            enemy
        } else {
            u16::MAX
        }
    }

    /// `player_clusters.rs:210-256` `is_surrounded`.
    #[allow(clippy::too_many_arguments)]
    fn cl_is_surrounded(
        terrain: &[u8],
        plane: &[u16],
        w: u32,
        h: u32,
        sid: u16,
        cluster: &[u32],
    ) -> bool {
        let mut has_enemy = false;
        let mut ebb = [0u32; 4];
        let mut init = false;
        let mut i = 0usize;
        while i < cluster.len() {
            let tile = cluster[i];
            i += 1;
            if cl_is_shore(terrain, tile) || cl_is_edge(w, h, tile) {
                return false;
            }
            let mut nb = [0u32; 4];
            let n = cl_neighbors4(w, h, tile, &mut nb);
            let mut j = 0usize;
            while j < n {
                let nt = nb[j];
                j += 1;
                let owner = plane[nt as usize];
                if owner == 0 || owner == sid {
                    continue;
                }
                has_enemy = true;
                let x = nt % w;
                let y = nt / w;
                if !init {
                    ebb[0] = x;
                    ebb[1] = y;
                    ebb[2] = x;
                    ebb[3] = y;
                    init = true;
                } else {
                    if x < ebb[0] {
                        ebb[0] = x;
                    }
                    if y < ebb[1] {
                        ebb[1] = y;
                    }
                    if x > ebb[2] {
                        ebb[2] = x;
                    }
                    if y > ebb[3] {
                        ebb[3] = y;
                    }
                }
            }
        }
        if !has_enemy {
            return false;
        }
        let mut cbb = [0u32; 4];
        cl_bbox_of(w, cluster, &mut cbb);
        cl_inscribed(&ebb, &cbb)
    }

    fn cl_flood_owned(
        plane: &[u16],
        w: u32,
        h: u32,
        sid: u16,
        start: u32,
        marks: &mut [u32],
        fgen: u32,
        stack: &mut [u32],
        res: &mut [u32],
    ) -> usize {
        let mut n = 0usize;
        let mut slen = 0usize;
        if plane[start as usize] == sid {
            marks[start as usize] = fgen;
            res[0] = start;
            n = 1;
            stack[0] = start;
            slen = 1;
        }
        while slen > 0 {
            slen -= 1;
            let t = stack[slen];
            let mut nb = [0u32; 4];
            let cnt = cl_neighbors4(w, h, t, &mut nb);
            let mut i = 0usize;
            while i < cnt {
                let nt = nb[i];
                i += 1;
                if marks[nt as usize] == fgen || plane[nt as usize] != sid {
                    continue;
                }
                marks[nt as usize] = fgen;
                if n < COWN {
                    res[n] = nt;
                    n += 1;
                }
                if slen < COWN {
                    stack[slen] = nt;
                    slen += 1;
                }
            }
        }
        n
    }

    /// The whole player-clusters pass for ONE engine tick, in the engine's exec
    /// order. `out` = `[fires, removals, border_overflow, cluster_overflow,
    /// own_overflow, nb_overflow, rem_words, 0]`. `rem` is a flat stream of
    /// `victim, captor, count, tile...` records the driver replays into the
    /// captor's claim list (the engine's `owned_tiles` push order).
    #[allow(clippy::too_many_arguments)]
    pub fn cluster_pass_core(
        terrain: &[u8],
        w: u32,
        h: u32,
        tick: u32,
        npl: u32,
        order: &[u32],
        idhash: &[u32],
        mut cad: &mut [u32],
        mut plane: &mut [u16],
        mut t2b: &mut [u32],
        mut marks: &mut [u32],
        mut cbuf: &mut [u32],
        mut clen: &mut [u32],
        mut coff: &mut [u32],
        mut vis: &mut [u32],
        mut stack: &mut [u32],
        mut owned: &mut [u32],
        mut rem: &mut [u32],
        mut out: &mut [u32],
        oborder: &[u32],
        obmeta: &[u32],
        friends: &[u32],
        nfriends: u32,
        atk_owner: &[u16],
        atk_target: &[u16],
        atk_troops: &[f64],
        natk: u32,
        mut pst: &mut [f64],
    ) {
        let mut fires = 0u32;
        let mut removals = 0u32;
        let mut border_ovf = 0u32;
        let mut cluster_ovf = 0u32;
        let mut own_ovf = 0u32;
        let mut nb_ovf = 0u32;
        let mut rem_words = 0usize;
        // Flood generation marks: unique per `flood_owned` call, so no clearing
        // is needed between calls or between ticks.
        let mut fgen: u32 = tick.wrapping_mul(1 << 16);

        let mut pi = 0u32;
        while pi < npl {
            let sid = order[pi as usize] as u16;
            pi += 1;
            let cs = sid as usize * CSTATE;
            if cad[cs + 3] == 0 {
                continue; // `!p.alive`
            }
            let tiles_owned = cad[cs + 2] as i32;
            if tiles_owned == 0 {
                continue;
            }
            let mut last_calc = cad[cs];
            if last_calc == 0 {
                // `player_clusters.rs:375-382`.
                last_calc = tick.wrapping_add(idhash[sid as usize] % TICKS_PER_CLUSTER_CALC);
                cad[cs] = last_calc;
            }
            if tick.saturating_sub(last_calc) <= TICKS_PER_CLUSTER_CALC && tiles_owned >= 100 {
                continue;
            }
            if cad[cs + 1] < last_calc {
                continue; // `last_change < last_calc` (:387)
            }
            cad[cs] = tick;
            fires += 1;

            let boff = obmeta[sid as usize * 2] as usize;
            let blen = obmeta[sid as usize * 2 + 1] as usize;
            if blen == 0 {
                continue;
            }
            if blen > CB {
                border_ovf += 1;
                continue;
            }
            let border: &[u32] = &oborder[boff..boff + blen];

            // `t2b`: border tile -> position in `border` (u32::MAX = absent).
            // Only this player's border entries are touched, and they are
            // restored before the next player, so the array stays `u32::MAX`
            // everywhere outside the current player's border set.
            let mut k = 0usize;
            while k < blen {
                t2b[border[k] as usize] = k as u32;
                k += 1;
            }
            let mut slen = 0usize;
            let ncl = cl_calculate_clusters(
                w,
                h,
                border,
                blen,
                t2b,
                vis,
                stack,
                cbuf,
                clen,
                coff,
                &mut cluster_ovf,
            );
            let _ = slen;

            if ncl > 0 {
                let mut largest_idx = 0usize;
                let mut largest_size = clen[0] as usize;
                let mut idx = 1usize;
                while idx < ncl {
                    let sz = clen[idx] as usize;
                    if sz > largest_size {
                        largest_size = sz;
                        largest_idx = idx;
                    }
                    idx += 1;
                }

                let loff = coff[largest_idx] as usize;
                let llen = clen[largest_idx] as usize;
                let mut lbb = [0u32; 4];
                cl_bbox_of(w, &cbuf[loff..loff + llen], &mut lbb);
                let hit = cl_surrounded_by_same_enemy(
                    terrain,
                    plane,
                    w,
                    h,
                    friends,
                    nfriends,
                    sid,
                    &cbuf[loff..loff + llen],
                    &lbb,
                );
                if hit != u16::MAX {
                    // --- remove_cluster (:325-356) for the largest cluster ---
                    let cluster = &cbuf[loff..loff + llen];
                    // `:326-330`: every cluster tile must still be the victim's.
                    let mut ok = true;
                    let mut q = 0usize;
                    while q < llen {
                        if plane[cluster[q] as usize] != sid {
                            ok = false;
                            break;
                        }
                        q += 1;
                    }
                    if ok {
                        // --- get_capturing_player (:258-298) ---
                        let mut nbsid = [0u16; CNB];
                        let mut nbcount = [0u32; CNB];
                        let mut nnb = 0usize;
                        let mut q = 0usize;
                        while q < llen {
                            let mut nb = [0u32; 4];
                            let nn = cl_neighbors4(w, h, cluster[q], &mut nb);
                            q += 1;
                            let mut j = 0usize;
                            while j < nn {
                                let owner = plane[nb[j] as usize];
                                j += 1;
                                if owner == 0 || owner == sid {
                                    continue;
                                }
                                if cl_friendly(friends, nfriends, owner, sid) {
                                    continue;
                                }
                                let mut f = 0usize;
                                while f < nnb {
                                    if nbsid[f] == owner {
                                        nbcount[f] += 1;
                                        break;
                                    }
                                    f += 1;
                                }
                                if f == nnb {
                                    if nnb < CNB {
                                        nbsid[nnb] = owner;
                                        nbcount[nnb] = 1;
                                        nnb += 1;
                                    } else {
                                        nb_ovf += 1;
                                    }
                                }
                            }
                        }
                        if nnb > 0 {
                            // `largest_incoming_land_attack_from_neighbors`
                            // (`game.rs:2235-2264`): outer loop over the
                            // first-seen neighbour order, inner over `execs`;
                            // strictly-greater keeps the EARLIEST on ties.
                            let mut largest = 0.0f64;
                            let mut captor: u16 = u16::MAX;
                            let mut a = 0usize;
                            while a < nnb {
                                let who = nbsid[a];
                                a += 1;
                                let mut e = 0u32;
                                while e < natk {
                                    if atk_owner[e as usize] == who
                                        && atk_target[e as usize] == sid
                                        && atk_troops[e as usize] > largest
                                    {
                                        largest = atk_troops[e as usize];
                                        captor = who;
                                    }
                                    e += 1;
                                }
                            }
                            if captor == u16::MAX {
                                // TS `getMode`: first neighbour with strictly
                                // greatest count.
                                let mut best: u16 = u16::MAX;
                                let mut bestc = 0u32;
                                let mut a = 0usize;
                                while a < nnb {
                                    if best == u16::MAX || nbcount[a] > bestc {
                                        best = nbsid[a];
                                        bestc = nbcount[a];
                                    }
                                    a += 1;
                                }
                                captor = best;
                            }
                            if captor != u16::MAX {
                                fgen = fgen.wrapping_add(1);
                                let first = cluster[0];
                                let nown = cl_flood_owned(
                                    plane,
                                    w,
                                    h,
                                    sid,
                                    first,
                                    marks,
                                    fgen,
                                    stack,
                                    owned,
                                );
                                if nown >= COWN {
                                    own_ovf += 1;
                                }
                                // Wipe-all reaches `Game::conquer_player`
                                // (`game.rs:1177`): ship transfer only for a
                                // disconnected SAME-TEAM conqueror, plus a gold
                                // transfer. The device models no units and the
                                // harness compares no gold, so the plane effect
                                // (none) is what is ported.
                                if rem_words + CWR + nown <= CREM {
                                    rem[rem_words] = sid as u32;
                                    rem[rem_words + 1] = captor as u32;
                                    rem[rem_words + 2] = nown as u32;
                                    let mut q = 0usize;
                                    while q < nown {
                                        rem[rem_words + 3 + q] = owned[q];
                                        q += 1;
                                    }
                                    rem_words += CWR + nown;
                                    removals += 1;
                                }
                                // `game.conquer(captor, t)` for each tile, in
                                // the flood's insertion order.
                                let mut q = 0usize;
                                while q < nown {
                                    let t = owned[q];
                                    q += 1;
                                    plane[t as usize] = captor;
                                    marks[t as usize] = 0; // no longer the victim's
                                    if (sid as usize) * 3 + 1 < pst.len() {
                                        pst[sid as usize * 3 + 1] -= 1.0;
                                    }
                                    if (captor as usize) * 3 + 1 < pst.len() {
                                        pst[captor as usize * 3 + 1] += 1.0;
                                    }
                                }
                                cad[sid as usize * CSTATE + 1] = tick;
                                cad[sid as usize * CSTATE + 2] = (tiles_owned - nown as i32).max(0) as u32;
                                cad[captor as usize * CSTATE + 1] = tick;
                                cad[captor as usize * CSTATE + 2] += nown as u32;
                            }
                        }
                    }
                }

                // --- every other cluster, in order (:421-428) ---
                let mut idx = 0usize;
                while idx < ncl {
                    if idx != largest_idx {
                        let o0 = coff[idx] as usize;
                        let n0 = clen[idx] as usize;
                        let cluster = &cbuf[o0..o0 + n0];
                        if cl_is_surrounded(terrain, plane, w, h, sid, cluster) {
                            let mut ok = true;
                            let mut q = 0usize;
                            while q < n0 {
                                if plane[cluster[q] as usize] != sid {
                                    ok = false;
                                    break;
                                }
                                q += 1;
                            }
                            if ok {
                                let mut nbsid = [0u16; CNB];
                                let mut nbcount = [0u32; CNB];
                                let mut nnb = 0usize;
                                let mut q = 0usize;
                                while q < n0 {
                                    let mut nb = [0u32; 4];
                                    let nn = cl_neighbors4(w, h, cluster[q], &mut nb);
                                    q += 1;
                                    let mut j = 0usize;
                                    while j < nn {
                                        let owner = plane[nb[j] as usize];
                                        j += 1;
                                        if owner == 0 || owner == sid {
                                            continue;
                                        }
                                        if cl_friendly(friends, nfriends, owner, sid) {
                                            continue;
                                        }
                                        let mut f = 0usize;
                                        while f < nnb {
                                            if nbsid[f] == owner {
                                                nbcount[f] += 1;
                                                break;
                                            }
                                            f += 1;
                                        }
                                        if f == nnb {
                                            if nnb < CNB {
                                                nbsid[nnb] = owner;
                                                nbcount[nnb] = 1;
                                                nnb += 1;
                                            } else {
                                                nb_ovf += 1;
                                            }
                                        }
                                    }
                                }
                                if nnb > 0 {
                                    let mut largest = 0.0f64;
                                    let mut captor: u16 = u16::MAX;
                                    let mut a = 0usize;
                                    while a < nnb {
                                        let who = nbsid[a];
                                        a += 1;
                                        let mut e = 0u32;
                                        while e < natk {
                                            if atk_owner[e as usize] == who
                                                && atk_target[e as usize] == sid
                                                && atk_troops[e as usize] > largest
                                            {
                                                largest = atk_troops[e as usize];
                                                captor = who;
                                            }
                                            e += 1;
                                        }
                                    }
                                    if captor == u16::MAX {
                                        let mut best: u16 = u16::MAX;
                                        let mut bestc = 0u32;
                                        let mut a = 0usize;
                                        while a < nnb {
                                            if best == u16::MAX || nbcount[a] > bestc {
                                                best = nbsid[a];
                                                bestc = nbcount[a];
                                            }
                                            a += 1;
                                        }
                                        captor = best;
                                    }
                                    if captor != u16::MAX {
                                        fgen = fgen.wrapping_add(1);
                                        let nown = cl_flood_owned(
                                            plane,
                                            w,
                                            h,
                                            sid,
                                            cluster[0],
                                            marks,
                                            fgen,
                                            stack,
                                            owned,
                                        );
                                        if nown >= COWN {
                                            own_ovf += 1;
                                        }
                                        if rem_words + CWR + nown <= CREM {
                                            rem[rem_words] = sid as u32;
                                            rem[rem_words + 1] = captor as u32;
                                            rem[rem_words + 2] = nown as u32;
                                            let mut q = 0usize;
                                            while q < nown {
                                                rem[rem_words + 3 + q] = owned[q];
                                                q += 1;
                                            }
                                            rem_words += CWR + nown;
                                            removals += 1;
                                        }
                                        let mut q = 0usize;
                                        while q < nown {
                                            let t = owned[q];
                                            q += 1;
                                            plane[t as usize] = captor;
                                            marks[t as usize] = 0;
                                            if (sid as usize) * 3 + 1 < pst.len() {
                                                pst[sid as usize * 3 + 1] -= 1.0;
                                            }
                                            if (captor as usize) * 3 + 1 < pst.len() {
                                                pst[captor as usize * 3 + 1] += 1.0;
                                            }
                                        }
                                        cad[sid as usize * CSTATE + 1] = tick;
                                        cad[sid as usize * CSTATE + 2] = (tiles_owned - nown as i32).max(0) as u32;
                                        cad[captor as usize * CSTATE + 1] = tick;
                                        cad[captor as usize * CSTATE + 2] += nown as u32;
                                    }
                                }
                            }
                        }
                    }
                    idx += 1;
                }
            }

            let mut k = 0usize;
            while k < blen {
                t2b[border[k] as usize] = u32::MAX;
                k += 1;
            }
        }

        out[0] = fires;
        out[1] = removals;
        out[2] = border_ovf;
        out[3] = cluster_ovf;
        out[4] = own_ovf;
        out[5] = nb_ovf;
        out[6] = rem_words as u32;
        out[7] = 0;
    }


// ================= BOT ATTACK AI (moved from ofcuda_matrix/src/kernels.rs) =================

    fn orig_border_insert(b: &mut [u32], blen: &mut usize, t: u32) {
        let mut i = 0usize;
        while i < *blen {
            if b[i] == t {
                return;
            }
            i += 1;
        }
        if *blen < b.len() {
            b[*blen] = t;
            *blen += 1;
        }
    }


    fn orig_border_remove(b: &mut [u32], blen: &mut usize, t: u32) {
        let mut i = 0usize;
        while i < *blen {
            if b[i] == t {
                let mut j = i;
                while j + 1 < *blen {
                    b[j] = b[j + 1];
                    j += 1;
                }
                *blen -= 1;
                return;
            }
            i += 1;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn orig_offer_dev(
        heap: &mut Heap,
        pr: &mut Prng,
        border: &mut [u32],
        blen: &mut usize,
        plane: &[u16],
        owner_col: u16,
        target_col: u16,
        terrain: &[u8],
        tile: u32,
        w: u32,
        h: u32,
        tick: u32,
    ) -> u32 {
        let mut nbuf = [0u32; 4];
        let n = neighbors4(ORDER_NSWE, tile, w, h, &mut nbuf);
        let mut i = 0usize;
        while i < n as usize {
            let nb = nbuf[i];
            i += 1;
            if terrain[nb as usize] & 0x80 == 0 {
                continue; // water (attack.rs:1354)
            }
            if plane[nb as usize] != target_col {
                continue; // not owned by the TARGET (attack.rs:1359)
            }
            orig_border_insert(border, blen, nb); // attack.rs:1363
        }
        orig_add_neighbors_t(
            heap, pr, tile, plane, owner_col, target_col, terrain, w, h, tick,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn orig_add_neighbors_t(
        heap: &mut Heap,
        pr: &mut Prng,
        tile: u32,
        plane: &[u16],
        owner_col: u16,
        target_col: u16,
        terrain: &[u8],
        w: u32,
        h: u32,
        tick: u32,
    ) -> u32 {
        let mut nbuf = [0u32; 4];
        let n = neighbors4(ORDER_NSWE, tile, w, h, &mut nbuf);
        let mut enq = 0u32;
        let mut i = 0usize;
        while i < n as usize {
            let nb = nbuf[i];
            i += 1;
            if terrain[nb as usize] & 0x80 == 0 {
                continue; // water (attack.rs:1354)
            }
            if plane[nb as usize] != target_col {
                continue; // wrong owner (attack.rs:1359)
            }
            let k = attacker_neighbor_count(plane, &[], owner_col, nb, w, h, ORDER_NSWE);
            let r = pr.next_int(0, 7);
            let mag2 = mag2_from_terrain(terrain[nb as usize]);
            heap.enqueue(nb, priority_f32(r, k, mag2, tick));
            enq += 1;
        }
        enq
    }

    #[allow(clippy::too_many_arguments)]
    fn orig_refresh_dev(
        heap: &mut Heap,
        pr: &mut Prng,
        border: &mut [u32],
        blen: &mut usize,
        plane: &[u16],
        owner_col: u16,
        target_col: u16,
        terrain: &[u8],
        w: u32,
        h: u32,
        tick: u32,
        oborder: &[u32],
        oboff: usize,
        obn: usize,
    ) {
        heap.clear();
        *blen = 0;
        let mut j = 0usize;
        while j < obn {
            let bt = oborder[oboff + j];
            j += 1;
            orig_offer_dev(
                heap, pr, border, blen, plane, owner_col, target_col, terrain, bt, w, h, tick,
            );
        }
    }

    #[inline]
    fn bot_is_land(terrain: &[u8], t: u32) -> bool {
        terrain[t as usize] & 0x80 != 0
    }

    #[inline]
    fn bot_is_shore(terrain: &[u8], t: u32) -> bool {
        terrain[t as usize] & 0xc0 == 0xc0
    }

    fn bot_land_border_tn(
        terrain: &[u8],
        w: u32,
        h: u32,
        plane: &[u16],
        oborder: &[u32],
        ob_meta: &[u32],
        sid: u16,
    ) -> bool {
        let om = sid as usize * 2;
        let off = ob_meta[om] as usize;
        let n = ob_meta[om + 1] as usize;
        let mut i = 0usize;
        while i < n {
            let t = oborder[off + i];
            let mut buf = [0u32; 4];
            let c = neighbors4(ORDER_NSWE, t, w, h, &mut buf) as usize;
            let mut j = 0usize;
            while j < c {
                let nb = buf[j];
                if bot_is_land(terrain, nb) && plane[nb as usize] == 0 {
                    return true;
                }
                j += 1;
            }
            i += 1;
        }
        false
    }

    fn bot_shore_reachable_tn(
        terrain: &[u8],
        w: u32,
        h: u32,
        plane: &[u16],
        oborder: &[u32],
        ob_meta: &[u32],
        sid: u16,
    ) -> bool {
        let om = sid as usize * 2;
        let off = ob_meta[om] as usize;
        let n = ob_meta[om + 1] as usize;
        const DIRS: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
        let mut shore_i = 0usize;
        let mut i = 0usize;
        while i < n {
            let t = oborder[off + i];
            if bot_is_shore(terrain, t) {
                if shore_i % 10 == 0 {
                    let x = (t % w) as i32;
                    let y = (t / w) as i32;
                    let mut d = 0usize;
                    while d < 4 {
                        let (dx, dy) = DIRS[d];
                        let x1 = x + dx;
                        let y1 = y + dy;
                        let nx = x + dx * 5;
                        let ny = y + dy * 5;
                        if x1 >= 0
                            && y1 >= 0
                            && (x1 as u32) < w
                            && (y1 as u32) < h
                            && nx >= 0
                            && ny >= 0
                            && (nx as u32) < w
                            && (ny as u32) < h
                        {
                            let t1 = (y1 as u32) * w + (x1 as u32);
                            let tn = (ny as u32) * w + (nx as u32);
                            if !bot_is_land(terrain, t1)
                                && bot_is_land(terrain, tn)
                                && plane[tn as usize] == 0
                            {
                                return true;
                            }
                        }
                        d += 1;
                    }
                }
                shore_i += 1;
            }
            i += 1;
        }
        false
    }

        const DIRS: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];

    #[allow(clippy::too_many_arguments)]
    pub fn bot_ai_core(
        terrain: &[u8],
        w: u32,
        h: u32,
        tick: u32,
        spawn_end_tick: u32,
        plane: &[u16],
        pst: &[f64],
        oborder: &[u32],
        ob_meta: &[u32],
        bot_sid: &[u32],
        bot_rate: &[u32],
        bot_at: &[u32],
        bot_trigger: &[f64],
        bot_trow: &[u32],
        // `reserve_ratio` row of the same pre-multiplied `max_troops * ratio`
        // table (`bot/tribe.rs:34-56` draws it 4th, between `trigger_ratio` and
        // `expand_ratio`). Used ONLY for the player-attack amount the engine's
        // retaliate branch would size this firing with (`land_attack_troops(..,
        // reserve_ratio)`, `ai_attack.rs:515-520`), which the cancel-opposing
        // model needs.
        bot_rrow: &[u32],
        bot_ff: &[u32],
        bot_state: &mut [u32],
        // Per-bot mask, set by the host ONLY in self-drive mode: 1 = the port's
        // own `send_boat_attack_to_nearby_tn` port (`dev_send_boat_attack_to_nearby_tn`,
        // main.rs) decided this bot sends a TransportShip at THIS firing, so the
        // engine would have returned from `send_tn_attack` and never reached the
        // trigger-ratio / retaliate branch. 0 everywhere else, which leaves the
        // kernel's behaviour byte-identical to before.
        bot_boat: &[u32],
        maxtroops: &[f64],
        tgt: &[f64],
        out: &mut [f64],
        nbots: u32,
    ) {
        let n = nbots as usize;
        let mcap = maxtroops.len();
        let mut k = 0usize;
        while k < n {
            out[k * 3] = 0.0;
            out[k * 3 + 1] = 0.0;
            out[k * 3 + 2] = 0.0;
            let sid = bot_sid[k] as u16;
            let rate = bot_rate[k];
            // `bot/tribe.rs:114-122`: inactive while `game.in_spawn_phase()`
            // (ticks <= spawn_end_tick), then only on `tick % attack_rate ==
            // attack_tick`.
            if tick > spawn_end_tick && rate > 0 && tick % rate == bot_at[k] {
                let ti = sid as usize * 3;
                // `tick` first reads `p.troops`/`p.tiles_owned` after the player
                // execs have ticked, so this is the post-income, post-cluster
                // state the device holds right now.
                let tiles = pst[ti + 1];
                let troops = pst[ti];
                if tiles >= 1.0 {
                    let st = bot_state[k];
                    // `attack_behavior_init` is set at the bot's FIRST ever
                    // scheduled firing; that firing calls `send_tn_attack` and
                    // RETURNS.
                    let first = tick <= bot_ff[k];
                    // `attack_behavior_init` is set on the FIRST firing whatever
                    // the outcome (`bot/tribe.rs:127-131`) and that firing calls
                    // `send_tn_attack` and RETURNS - it never reaches
                    // `tribe_maybe_attack`.
                    if first {
                        bot_state[k] = st | 1;
                    }
                    let idx = if (tiles as usize) < mcap {
                        tiles as usize
                    } else {
                        mcap - 1
                    };
                    // `has_land_border_with_terra_nullius` for the first firing
                    // (`try_send_tn_attack`, `ai_attack.rs:409-422`); for later
                    // firings `tribe_maybe_attack` first gates on
                    // `neighbors_terra_nullius` and `has_nearby_terra_nullius` =
                    // land border OR shore-reachable (`ai_attack.rs:1743-1748`).
                    let land = bot_land_border_tn(terrain, w, h, plane, oborder, ob_meta, sid);
                    let nearby = if first {
                        land
                    } else if st & 2 != 0 {
                        land
                            || bot_shore_reachable_tn(terrain, w, h, plane, oborder, ob_meta, sid)
                    } else {
                        false
                    };
                    let mut sent = false;
                    let boat = bot_boat[k] != 0;
                    if boat {
                        // The engine's `send_tn_attack` -> `try_send_tn_attack` ->
                        // `send_boat_attack_to_nearby_tn` succeeded (the host
                        // resolved the candidate against the device plane), so
                        // `tribe_maybe_attack` RETURNS: no trigger-ratio gate, no
                        // retaliate. The ship itself is sailed and landed by the
                        // host's ship model.
                        out[k * 3 + 2] = 7.0;
                        sent = true;
                    } else if nearby {
                        // `land_attack_troops` (`ai_attack.rs:9-17`). `max_troops
                        // * expand_ratio` is looked up PRE-MULTIPLIED: CUDA
                        // contracts `a - b * c` into one fma (one rounding) while
                        // the engine rounds the product and the subtraction
                        // separately, and that is measurably 1 ulp. The product
                        // is therefore evaluated by the host with the engine's
                        // own expression.
                        let amount =
                            troops - tgt[bot_trow[k] as usize * mcap + idx];
                        if land && amount >= 1.0 {
                            out[k * 3] = 1.0;
                            out[k * 3 + 1] = amount;
                            out[k * 3 + 2] = 1.0;
                            sent = true;
                        } else if land {
                            // fired, `land_attack_troops` < 1.
                            out[k * 3 + 2] = 4.0;
                        } else {
                            // no LAND border: the engine's BOAT path
                            // (`send_boat_attack_to_nearby_tn`) and the port
                            // models no boats.
                            out[k * 3 + 2] = 3.0;
                        }
                    } else if !first {
                        // `ai_attack.rs:2189-2191`.
                        bot_state[k] = bot_state[k] & !2;
                    }
                    // A failed `send_tn_attack` does NOT return: the engine falls
                    // through to the trigger-ratio gate (`ai_attack.rs:2194-2199`).
                    if !sent && !first {
                        if maxtroops[idx] <= 0.0 || !(troops / maxtroops[idx] >= bot_trigger[k]) {
                            // `!has_trigger_ratio` (`ai_attack.rs:42-50`).
                            out[k * 3 + 2] = 6.0;
                        } else {
                            out[k * 3 + 2] = 5.0;
                            // `has_trigger_ratio` passed, so the engine now runs the
                            // RETALIATE branch (`ai_attack.rs:2211-2230`):
                            // `find_incoming_attacker` then
                            // `try_send_player_attack_forced(.., force=true)`, whose
                            // LAND path sizes the new attack with
                            // `land_attack_troops(attacker, reserve_ratio)` and then
                            // `cap_player_attack_troops` (`ai_attack.rs:509-527`).
                            // Compute that amount HERE, off the device's own
                            // post-cluster player state, so the cancel-opposing model
                            // has the engine's exact `start` without reading the
                            // oracle record. `land_ratio` is `reserve_ratio` because
                            // every player on this map is a Bot and no Bot owns
                            // structure units (`ai_attack.rs:511-515`).
                            let pstart = troops - tgt[bot_rrow[k] as usize * mcap + idx];
                            out[k * 3 + 1] = if pstart >= 1.0 { pstart } else { 0.0 };
                        }
                    }
                }
            }
            k += 1;
        }
    }