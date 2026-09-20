//! `ofcuda_tick` - GPU (cuda-oxide) side of the attack/expansion FRONTIER level.
//!
//! The kernel ports `AttackExecution::add_neighbors` + `FlatBinaryHeap` +
//! `AttackExecution::tick`'s pop loop: given the attacker's border tiles, the
//! owned sets, the map's terrain plane, the tick and the shared sfc32 stream
//! (seed 123, explicit cursor), produce the conquest-claim order.
//!
//! Kernel shape (the trick `ofcuda_prng` already verified): the frontier is
//! *inherently serial* - the heap's tie-breaks depend on insertion order and the
//! PRNG is one shared stream - so thread `j` replays the whole step and reports
//! **pop #j**. That keeps the serial semantics exact while giving one output per
//! thread (`DisjointSlice::get_mut` only hands a thread its own index).
//!
//! The device core below is a **verbatim copy** of the host reference
//! (`src/core_impl.rs`), because cuda-oxide's `#[cuda_module]` macro sees the
//! module before an `include!` inside it expands. `lib.rs` carries a test that
//! fails if the two copies ever diverge, so GPU/CPU agreement stays a check on
//! the CUDA lowering rather than on a second implementation.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use ofcuda_tick::{
    CASE_OTHER_ENGINE_ORDER_NOTE, FNV_OFFSET, FNV_PRIME, FrontierCase, MapPlane, ORDER_NSWE,
    ORDER_WENS, Tally, claim_order, contested_border_tiles, contested_case_path,
    contested_frontier_case, enqueue_trace, fma_contraction_sensitivity, fmt_prios, fmt_u32s,
    engine_tick_model, frontier_claims, frontier_claims_conflated, frontier_claims_incremental,
    frontier_claims_with, frontier_cpu, fnv1a_u16_le,
    contested_engine_run, contested_plane, hex64, load_contested_case, load_map,
    load_tick_window, map_dir_note, neighbors4, window_engine_run,
    pangaea_map_dir, priority_unit, recorded_cases, rival_tiles, run_cpu, tick_window_path,
};

#[cuda_module]
mod kernels {
    use super::*;

    // (cuda-oxide's `#[cuda_module]` attribute macro does not survive an
    // `include!` of this file - the macro sees the module before the include
    // expands - so the device core is an exact copy. `lib.rs` has a test that
    // fails if these two copies ever differ; `sync_device_core.sh` re-copies.)
    // ===== BEGIN VERBATIM COPY of src/core_impl.rs =====
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
    // ===== END VERBATIM COPY of src/core_impl.rs =====

    /// THE ENGINE'S OWN LOOP ON THE DEVICE: thread `j` replays an attack's
    /// whole life - `AttackExecution::init` at `init_tick`, then one
    /// `AttackExecution::tick` per entry of `budgets` - and writes the tile it
    /// conquered at GLOBAL claim index `j`.
    ///
    /// That is the point of this kernel: the claim at index `j` may come from a
    /// later tick than `j`'s neighbours, because `to_conquer` is carried across
    /// ticks (`attack.rs:17`) and each tick re-enqueues from inside its own pop
    /// loop (`attack.rs:292`). Every thread runs the same serial life; only its
    /// own index is written, so the device computes the real order without
    /// atomics.
    ///
    /// `border_flat`/`border_off` hold the per-tick border sets (the engine
    /// reads the game's border only when `refresh_to_conquer` runs,
    /// `attack.rs:1265-1272`); `pop_ticks[t]` is the step parameter of tick `t`
    /// (`game.rs:3659`), which stamps the in-tick `add_neighbors` priorities.
    ///
    /// `out_state` is the carried state after the whole life, in `state_words`
    /// layout - every thread computed the same one, so thread `i` publishes word
    /// `i`. The host compares it against its own run: a device that agrees on
    /// the claims but not on the heap would be a different program.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub fn engine_life_claims(
        plane: &[u16],
        terrain: &[u8],
        width: u32,
        height: u32,
        order: u32,
        owner_sid: u16,
        init_tick: u32,
        consume_budget_draw: u32,
        in_tick_add_neighbors: u32,
        border_flat: &[u32],
        border_off: &[u32],
        budgets: &[u32],
        pop_ticks: &[u32],
        mut out_tile: DisjointSlice<u32>,
        mut out_pri_bits: DisjointSlice<u32>,
        mut out_state: DisjointSlice<u32>,
    ) {
        let idx_tile = thread::index_1d();
        let j = idx_tile.get();

        // The attack as `init` leaves it: `borderOff[0]..borderOff[1]` is the
        // border the init refresh walks.
        let first_hi = border_off[1] as usize;
        let mut attack = EngineAttack::init(
            &border_flat[0..first_hi],
            plane,
            owner_sid,
            terrain,
            width,
            height,
            order,
            init_tick,
        );

        let n_ticks = budgets.len() as u32;
        let mut t = 0u32;
        while t < n_ticks {
            let lo = border_off[t as usize] as usize;
            let hi = border_off[t as usize + 1] as usize;
            attack.tick(
                plane,
                owner_sid,
                &border_flat[lo..hi],
                terrain,
                width,
                height,
                order,
                pop_ticks[t as usize],
                budgets[t as usize],
                consume_budget_draw != 0,
                in_tick_add_neighbors != 0,
            );
            t += 1;
        }

        // Pop #j of the whole life (sentinel when the life is shorter).
        let mut tile = u32::MAX;
        let mut bits = 0u32;
        if (j as u32) < attack.claimed {
            tile = attack.claims[j as usize];
            bits = attack.claim_pri_bits[j as usize];
        }
        if let Some(slot) = out_tile.get_mut(idx_tile) {
            *slot = tile;
        }
        if let Some(slot) = out_pri_bits.get_mut(thread::index_1d()) {
            *slot = bits;
        }

        // The carried state, one word per thread.
        let idx_state = thread::index_1d();
        let w = idx_state.get() as usize;
        if w < STATE_WORDS {
            let mut words = [0u32; STATE_WORDS];
            attack.state_words(&mut words);
            if let Some(slot) = out_state.get_mut(idx_state) {
                *slot = words[w];
            }
        }
    }

    /// Thread `j` replays the whole frontier and writes pop #j.
    ///
    /// `out_tile[j]` = tile of pop #j (`u32::MAX` when the heap ran out),
    /// `out_pri_bits[j]` = the priority it was popped at, as f32 bits.
    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(domain = 1, block = (64, 1, 1))]
    pub fn frontier_step(
        border: &[u32],
        owned_any: &[u32],
        owned_mine: &[u32],
        terrain: &[u8],
        width: u32,
        height: u32,
        order: u32,
        tick: u32,
        cursor: u32,
        mut out_tile: DisjointSlice<u32>,
        mut out_pri_bits: DisjointSlice<u32>,
    ) {
        let j = thread::index_1d().get();
        // `ThreadIndex` is not `Copy`, and two outputs need two of them.
        let idx = thread::index_1d();

        let mut heap = Heap::new();
        frontier_enqueue(
            border,
            owned_any,
            owned_mine,
            terrain,
            width,
            height,
            order,
            tick,
            cursor,
            &mut heap,
        );

        // Pop `j+1` times. A thread whose heap runs dry reports the sentinel
        // (`u32::MAX`) instead of the tile it popped last, so the host can read
        // the device's OWN enqueue count out of the tail of the output.
        let mut tile = u32::MAX;
        let mut bits = 0u32;
        let mut p = 0usize;
        let mut popped = false;
        while p <= j {
            match heap.dequeue() {
                Some((t, pr)) => {
                    tile = t;
                    bits = pr.to_bits();
                    popped = p == j;
                }
                None => {
                    popped = false;
                    break;
                }
            }
            p += 1;
        }
        if !popped {
            tile = u32::MAX;
            bits = 0;
        }

        if let Some(slot) = out_tile.get_mut(idx) {
            *slot = tile;
        }
        if let Some(slot) = out_pri_bits.get_mut(thread::index_1d()) {
            *slot = bits;
        }
    }

    /// The COMPOSED TICK: one device pass that advances the whole state plane
    /// the way the ported rule says one expansion tick does.
    ///
    /// One thread per tile. Thread `i` writes `plane_out[i]` = the tile's owner
    /// after the tick. Non-eligible tiles (already owned, water, or in
    /// `owned_any`) are copied through without touching the frontier, so the
    /// per-thread heap replay is only paid on land that could actually be
    /// claimed. Threads `i < budget` also publish the `i`-th claim, which is how
    /// the host sees the claim order without a second kernel.
    ///
    /// This is deliberately the CHEAPEST model that could match: no troop
    /// growth, no gold, no cluster capture, no re-enqueue of newly claimed
    /// tiles within the tick - a single pop pass over the frontier enqueued
    /// from the pre-tick border.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn compose_tick(
        plane_in: &[u16],
        terrain: &[u8],
        width: u32,
        height: u32,
        border: &[u32],
        owned_any: &[u32],
        owned_mine: &[u32],
        order: u32,
        tick: u32,
        cursor: u32,
        budget: u32,
        owner_sid: u32,
        mut plane_out: DisjointSlice<u16>,
        mut out_claims: DisjointSlice<u32>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        let n = plane_in.len();
        if i >= n {
            return;
        }
        let before = plane_in[i];

        // Nothing to do for a zero-budget tick (the early regime's dominant
        // case): the plane is copied unchanged.
        if budget == 0 {
            if let Some(slot) = plane_out.get_mut(idx) {
                *slot = before;
            }
            return;
        }

        // A claim target must be unowned land that is not already somebody's
        // tile, so anything else is a pure copy - except threads `i < budget`,
        // which are the ones that publish the claim list and therefore have to
        // run the replay even when their own tile is water.
        let could_be_claimed = before == 0
            && (terrain[i] & 0x80 != 0)
            && !owned_contains(owned_any, i as u32);
        if !could_be_claimed && i >= budget as usize {
            if let Some(slot) = plane_out.get_mut(idx) {
                *slot = before;
            }
            return;
        }

        // --- the replay: exact same core as `frontier_step` ---
        let mut heap = Heap::new();
        frontier_enqueue(
            border,
            owned_any,
            owned_mine,
            terrain,
            width,
            height,
            order,
            tick,
            cursor,
            &mut heap,
        );

        let mut claimed = false;
        // Dedup + output: a tile the heap yields twice must not count twice.
        let mut seen = [0u32; 64];
        let mut n_seen = 0u32;
        let mut k = 0u32;
        while k < budget && n_seen < 64 {
            match heap.dequeue() {
                Some((t, _pr)) => {
                    // already somebody's tile before this tick -> not a claim
                    if plane_in[t as usize] != 0 {
                        continue;
                    }
                    let mut dup = false;
                    let mut q = 0u32;
                    while q < n_seen {
                        if seen[q as usize] == t {
                            dup = true;
                        }
                        q += 1;
                    }
                    if dup {
                        continue;
                    }
                    // The (k+1)-th distinct claim, published by thread k.
                    if k == i as u32 {
                        if let Some(slot) = out_claims.get_mut(thread::index_1d()) {
                            *slot = t;
                        }
                    }
                    if t == i as u32 {
                        claimed = true;
                    }
                    seen[n_seen as usize] = t;
                    n_seen += 1;
                    k += 1;
                }
                None => break,
            }
        }

        let after = if claimed { owner_sid as u16 } else { before };
        if let Some(slot) = plane_out.get_mut(idx) {
            *slot = after;
        }
    }

    /// FNV-1a 64 over the state plane, each `u16` fed as two little-endian
    /// bytes (the fixed contract - see `lib.rs`).
    ///
    /// One thread: 2 * W * H bytes, sequential, and the result is a single
    /// scalar, so parallelising it would only add the reduction back.
    #[kernel]
    #[launch_bounds(1)]
    #[launch_contract(domain = 1, block = (1, 1, 1))]
    pub fn fnv_state_hash(plane: &[u16], mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let mut h = FNV_OFFSET;
        let mut i = 0usize;
        let n = plane.len();
        while i < n {
            let b = plane[i].to_le_bytes();
            h = (h ^ b[0] as u64).wrapping_mul(FNV_PRIME);
            h = (h ^ b[1] as u64).wrapping_mul(FNV_PRIME);
            i += 1;
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = h;
        }
    }
}

fn first_diff<T: PartialEq>(a: &[T], b: &[T]) -> Option<usize> {
    for i in 0..a.len().max(b.len()) {
        if a.get(i) != b.get(i) {
            return Some(i);
        }
    }
    None
}

/// The f32 evaluation vs the engine's f64 evaluation, over every enqueue of the
/// case. A measurement, not an assumption.
fn f32_f64_equivalence(case: &FrontierCase, map: &MapPlane) -> (usize, usize, String) {
    let trace = enqueue_trace(case, map);
    let mut eq = 0usize;
    let mut first = String::new();
    for (tile, r, k, mag2, pf32, pf64) in &trace {
        if pf32 == pf64 {
            eq += 1;
        } else if first.is_empty() {
            first = format!("tile {tile} r={r} k={k} mag2={mag2}: f32 {pf32} vs f64 {pf64}");
        }
    }
    (eq, trace.len(), first)
}

/// Sweep the engine-faithful in-tick model over PRNG cursors and report the best
/// agreement with a recorded claim order. Returns (exact hits, best common
/// prefix, the cursor that achieved it, that run's claims, that run's enqueues).
fn sweep_engine_model(
    border: &[u32],
    owned_any: &[u32],
    owned_mine: &[u32],
    terrain: &[u8],
    width: u32,
    height: u32,
    tick: u32,
    expected: &[u32],
    consume_budget_draw: bool,
    max_cursor: u32,
) -> (usize, usize, u32, Vec<u32>, u32, u32) {
    let mut hits = 0usize;
    let mut best = 0usize;
    let mut best_cursor = 0u32;
    let mut best_claims = Vec::new();
    let mut best_draws = 0u32;
    let mut best_n_enq = 0u32;
    for cur in 0..max_cursor {
        let r = engine_tick_model(
            border, owned_any, owned_mine, terrain, width, height, ORDER_NSWE, tick, cur,
            expected.len() as u32, consume_budget_draw,
        );
        if r.claims == expected {
            hits += 1;
        }
        let m = r
            .claims
            .iter()
            .zip(expected.iter())
            .take_while(|(a, b)| a == b)
            .count();
        if m > best {
            best = m;
            best_cursor = cur;
            best_claims = r.claims.clone();
            best_draws = r.draws;
            best_n_enq = r.n_enq;
        }
    }
    (hits, best, best_cursor, best_claims, best_draws, best_n_enq)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let map = load_map(&pangaea_map_dir())?;
    let (c1, c2) = recorded_cases(&map)?;

    let ctx = CudaContext::new(0)?;
    let dev = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,driver_version", "--format=csv,noheader"])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let stream = ctx.default_stream();
    // SAFETY: this package owns the embedded device bundle for `kernels`.
    let module = unsafe { kernels::load(&ctx)? };

    let mut log = String::new();
    log.push_str("# ofcuda_tick - attack/expansion FRONTIER step\n");
    log.push_str(&format!("# device: {dev}\n"));
    log.push_str(&format!(
        "# {} {}x{}\n",
        map_dir_note(),
        map.width,
        map.height
    ));
    log.push_str(&format!("# {CASE_OTHER_ENGINE_ORDER_NOTE}\n"));

    let mut proof_ok = true;

    for case in [&c1, &c2] {
        log.push_str(&format!("\n=== {} ===\n", case.name));
        log.push_str(&format!(
            "player {} {}; claim_tick {}; priority_tick {} (measured); cursor {}; \
             order {} ({}); border {} tiles; owned_mine {}; owned_any {}; expected claims {}\n",
            case.player_id,
            case.player_name,
            case.claim_tick,
            case.priority_tick,
            case.cursor,
            case.order,
            if case.order == ORDER_WENS {
                "W,E,N,S"
            } else {
                "N,S,W,E"
            },
            case.border.len(),
            case.owned_mine.len(),
            case.owned_any.len(),
            case.expected_order.len()
        ));

        // --- CPU reference (same core file, host side) ---
        let (cpu_pops, n_enq) = frontier_cpu(case, &map);
        let (cpu_order, cpu_prios, _) = run_cpu(case, &map);
        log.push_str(&format!(
            "cpu : enqueued {n_enq}; raw pops {}; claim order {} priorities {}\n",
            cpu_pops.len(),
            fmt_u32s(&cpu_order),
            fmt_prios(&cpu_prios)
        ));

        // --- GPU ---
        let n_threads = (n_enq as usize).div_ceil(64) * 64;
        let cfg = LaunchConfig1D::new((n_threads / 64) as u32, 64, 0);
        let d_border = DeviceBuffer::from_host(&stream, &case.border)?;
        let d_owned_any = DeviceBuffer::from_host(&stream, &case.owned_any)?;
        let d_owned_mine = DeviceBuffer::from_host(&stream, &case.owned_mine)?;
        let d_terrain = DeviceBuffer::from_host(&stream, &map.terrain)?;
        let mut d_tile = DeviceBuffer::<u32>::zeroed(&stream, n_threads)?;
        let mut d_bits = DeviceBuffer::<u32>::zeroed(&stream, n_threads)?;
        let prepared = module.prepare_frontier_step(cfg)?;
        module.frontier_step(
            &stream,
            &prepared,
            &d_border,
            &d_owned_any,
            &d_owned_mine,
            &d_terrain,
            map.width,
            map.height,
            case.order,
            case.priority_tick,
            case.cursor,
            &mut d_tile,
            &mut d_bits,
        )?;
        let gpu_tiles = d_tile.to_host_vec(&stream)?;
        let gpu_bits = d_bits.to_host_vec(&stream)?;
        let gpu_pops: Vec<(u32, f32)> = gpu_tiles
            .iter()
            .take(n_enq as usize)
            .cloned()
            .zip(gpu_bits.iter().take(n_enq as usize).map(|b| f32::from_bits(*b)))
            .collect();
        let gpu_claims = claim_order(&gpu_pops);
        let gpu_order_full: Vec<u32> = gpu_claims.iter().map(|c| c.0).collect();
        let gpu_prios_full: Vec<f32> = gpu_claims.iter().map(|c| c.1).collect();
        // The tick claims exactly `expected_order.len()` tiles
        // (`num_tiles_per_tick`); the rest of the queue is not observable in the
        // record, so the proof is on that prefix and nothing else.
        let n_claim = case.expected_order.len();
        let gpu_order: Vec<u32> = gpu_order_full[..n_claim].to_vec();
        let gpu_prios_v: Vec<f32> = gpu_prios_full[..n_claim].to_vec();
        log.push_str(&format!(
            "gpu : threads launched {n_threads}; claim order {} priorities {}\n",
            fmt_u32s(&gpu_order),
            fmt_prios(&gpu_prios_v)
        ));
        log.push_str(&format!(
            "gpu : full queue remainder (unobservable in the record): {} tiles, \
             first extra priority {}\n",
            gpu_order_full.len() - n_claim,
            gpu_prios_full
                .get(n_claim)
                .map(|p| fmt_prios(&[*p]))
                .unwrap_or_else(|| "[]".to_string())
        ));

        // --- GPU vs CPU, element by element over the whole pop sequence ---
        let mut cross = Tally::default();
        for i in 0..n_enq as usize {
            cross.push(&format!("{} pops", case.name), i, &cpu_pops[i], &gpu_pops[i]);
        }
        log.push_str(&format!(
            "{}\n",
            cross.line(&format!(
                "GPU vs CPU raw pop sequence ({n_enq} entries, tile + priority bits)"
            ))
        ));
        if cross.matched != cross.total {
            proof_ok = false;
        }
        // The device's own enqueue count: a spare thread must find the heap empty.
        if n_threads > n_enq as usize {
            let sentinel = gpu_tiles[n_enq as usize];
            log.push_str(&format!(
                "GPU-vs-CPU enqueue count: thread {n_enq} wrote {sentinel} \
                 (u32::MAX = heap empty) -> {}\n",
                if sentinel == u32::MAX { "MATCH" } else { "MISMATCH" }
            ));
            if sentinel != u32::MAX {
                proof_ok = false;
            }
        }

        // --- expectations ---
        log.push_str(&format!(
            "expected claim order    {}\n",
            fmt_u32s(&case.expected_order)
        ));
        log.push_str(&format!(
            "expected priorities     {}\n",
            fmt_prios(&case.expected_priorities)
        ));
        if !case.expected_units.is_empty() {
            // The claimed prefix only: the queue tail is not observable.
            let units: Vec<u32> = gpu_claims[..n_claim]
                .iter()
                .map(|(t, p)| priority_unit(*t, *p, case, &map))
                .collect();
            let ok = units == case.expected_units;
            log.push_str(&format!(
                "gpu priority units      {} (expected {}) -> {}\n",
                fmt_u32s(&units),
                fmt_u32s(&case.expected_units),
                if ok { "MATCH" } else { "MISMATCH" }
            ));
            if !ok {
                proof_ok = false;
            }
        }
        if gpu_order == case.expected_order {
            log.push_str("GPU claim order         MATCH\n");
        } else {
            log.push_str(&format!(
                "GPU claim order         MISMATCH: first difference at index {}\n",
                first_diff(&gpu_order, &case.expected_order)
                    .map(|i| i.to_string())
                    .unwrap_or_else(|| "n/a".to_string())
            ));
            proof_ok = false;
        }
        if gpu_prios_v == case.expected_priorities {
            log.push_str("GPU priorities          MATCH\n");
        } else {
            log.push_str(&format!(
                "GPU priorities          MISMATCH: first difference at index {}\n",
                first_diff(&gpu_prios_v, &case.expected_priorities)
                    .map(|i| i.to_string())
                    .unwrap_or_else(|| "n/a".to_string())
            ));
            proof_ok = false;
        }

        // --- f32 vs the engine's f64 evaluation ---
        let (eq, total, first) = f32_f64_equivalence(case, &map);
        log.push_str(&format!(
            "priority f32 vs engine f64 evaluation: {eq}/{total} identical{}\n",
            if first.is_empty() {
                String::new()
            } else {
                format!("  first difference: {first}")
            }
        ));
        if eq != total {
            proof_ok = false;
        }

        // --- the same case under the *other* neighbour order ---
        let mut other = case.clone();
        other.order = if case.order == ORDER_WENS {
            ORDER_NSWE
        } else {
            ORDER_WENS
        };
        let (o_order, o_prios, o_enq) = run_cpu(&other, &map);
        log.push_str(&format!(
            "same case under {} ({}): enqueued {o_enq}; claim order {} priorities {}\n",
            if other.order == ORDER_WENS {
                "W,E,N,S"
            } else {
                "N,S,W,E"
            },
            if other.order == ORDER_NSWE && case.order == ORDER_NSWE {
                "the order used above"
            } else {
                "the other order"
            },
            fmt_u32s(&o_order[..n_claim]),
            fmt_prios(&o_prios)
        ));
        log.push_str(&format!(
            "other engine recorded   {}\n",
            fmt_u32s(&case.other_engine_order)
        ));
        log.push_str(&format!(
            "other order reproduces  {}\n",
            o_order[..n_claim] == case.other_engine_order[..]
        ));
    }

    // =====================================================================
    // PART 1: the CONTESTED border
    // =====================================================================
    log.push_str("\n\n===== PART 1: contested-border case =====\n");
    let cc = load_contested_case(&contested_case_path())?;
    let case3 = contested_frontier_case(&cc)?;
    let rivals = rival_tiles(&case3);
    let cbt = contested_border_tiles(&case3, &map);
    let n_claim3 = case3.expected_order.len();
    log.push_str(&format!(
        "{} \n  border {} tiles; owned_mine {}; owned_any {}; rival tiles {}; \
         border tiles with >=1 rival neighbour {}; expected claims {}\n",
        case3.name,
        case3.border.len(),
        case3.owned_mine.len(),
        case3.owned_any.len(),
        rivals.len(),
        cbt.len(),
        n_claim3
    ));

    // --- are the two sets genuinely different here? (numbers, not prose) ---
    let mine_set: std::collections::HashSet<u32> = case3.owned_mine.iter().copied().collect();
    let overlap = case3
        .owned_any
        .iter()
        .filter(|t| mine_set.contains(t))
        .count();
    log.push_str(&format!(
        "  |owned_any| = {}; |owned_mine| = {}; |owned_any ∩ owned_mine| = {}; \
         |owned_any \\ owned_mine| = {}\n",
        case3.owned_any.len(),
        case3.owned_mine.len(),
        overlap,
        case3.owned_any.len() - overlap
    ));

    // --- what the conflation would cost (the bug the earlier cases hid) ---
    let (c_any, n_any) = frontier_claims(&case3, &map, 64);
    let (c_con, n_con) = frontier_claims_conflated(&case3, &map, 64);
    log.push_str(&format!(
        "  host, eligibility = owned_any  (as shipped): enqueued {n_any}; claim order {}\n",
        fmt_u32s(&c_any)
    ));
    log.push_str(&format!(
        "  host, eligibility = owned_mine (conflated): enqueued {n_con}; claim order {}\n",
        fmt_u32s(&c_con)
    ));
    log.push_str(&format!(
        "  CONFLATION COST: enqueue count {} -> {} (delta {}); claim order first differs at index {}\n",
        n_any,
        n_con,
        n_con as i64 - n_any as i64,
        first_diff(&c_any, &c_con)
            .map(|i| i.to_string())
            .unwrap_or_else(|| "never (identical)".to_string())
    ));

    // --- the GPU on this case ---
    let (cpu_pops3, n_enq3) = frontier_cpu(&case3, &map);
    log.push_str(&format!(
        "  cpu: enqueued {n_enq3}; raw pops {}; first {}\n",
        cpu_pops3.len(),
        fmt_u32s(&cpu_pops3.iter().take(8).map(|p| p.0).collect::<Vec<u32>>())
    ));
    let n_threads3 = (n_enq3 as usize).div_ceil(64) * 64;
    let d_border3 = DeviceBuffer::from_host(&stream, &case3.border)?;
    let d_owned_any3 = DeviceBuffer::from_host(&stream, &case3.owned_any)?;
    let d_owned_mine3 = DeviceBuffer::from_host(&stream, &case3.owned_mine)?;
    let d_terrain3 = DeviceBuffer::from_host(&stream, &map.terrain)?;
    let mut d_tile3 = DeviceBuffer::<u32>::zeroed(&stream, n_threads3)?;
    let mut d_bits3 = DeviceBuffer::<u32>::zeroed(&stream, n_threads3)?;
    let prep3 = module.prepare_frontier_step(LaunchConfig1D::new((n_threads3 / 64) as u32, 64, 0))?;
    module.frontier_step(
        &stream,
        &prep3,
        &d_border3,
        &d_owned_any3,
        &d_owned_mine3,
        &d_terrain3,
        map.width,
        map.height,
        case3.order,
        case3.priority_tick,
        case3.cursor,
        &mut d_tile3,
        &mut d_bits3,
    )?;
    let gpu_tiles3 = d_tile3.to_host_vec(&stream)?;
    let gpu_bits3 = d_bits3.to_host_vec(&stream)?;
    let gpu_pops3: Vec<(u32, f32)> = gpu_tiles3
        .iter()
        .take(n_enq3 as usize)
        .cloned()
        .zip(gpu_bits3.iter().take(n_enq3 as usize).map(|b| f32::from_bits(*b)))
        .collect();
    let gpu_claims3 = claim_order(&gpu_pops3);
    let gpu_order3: Vec<u32> = gpu_claims3.iter().take(n_claim3).map(|c| c.0).collect();
    let gpu_prios3: Vec<f32> = gpu_claims3.iter().take(n_claim3).map(|c| c.1).collect();
    // the device's own enqueue count, from the sentinel tail
    let gpu_n_enq3 = gpu_tiles3.iter().take_while(|t| **t != u32::MAX).count();
    log.push_str(&format!(
        "  gpu: threads {n_threads3}; device-observed enqueue count {}; claim order {}\n",
        gpu_n_enq3,
        fmt_u32s(&gpu_order3)
    ));
    log.push_str(&format!(
        "  engine recorded: {}\n",
        fmt_u32s(&case3.expected_order)
    ));
    // CONTROL, not the proof: this is the one-pass kernel the crate shipped
    // before the fix (frontier rebuilt from the pre-tick border, no in-tick
    // re-enqueue). It cannot pop the second-ring tile 838544, which is exactly
    // the bug; the fix is measured by the engine-loop kernel below.
    let part1_match = gpu_order3 == case3.expected_order;
    log.push_str(&format!(
        "  ONE-PASS KERNEL (as shipped) vs ENGINE claim order: {} (first difference at index {})\n",
        if part1_match { "IDENTICAL" } else { "DIFFERENT" },
        first_diff(&gpu_order3, &case3.expected_order)
            .map(|i| format!("{i}"))
            .unwrap_or_else(|| "none".to_string())
    ));
    log.push_str(&format!(
        "  cpu vs gpu claim order: {}\n",
        if gpu_order3 == c_any[..n_claim3.min(c_any.len())] {
            "identical".to_string()
        } else {
            format!(
                "DIFFERENT at index {}",
                first_diff(&gpu_order3, &c_any)
                    .map(|i| i.to_string())
                    .unwrap_or_else(|| "none".to_string())
            )
        }
    ));
    let mut prio_same = 0usize;
    for (i, p) in gpu_pops3.iter().enumerate() {
        if cpu_pops3.get(i).map(|c| c.1 == p.1).unwrap_or(false) {
            prio_same += 1;
        }
    }
    let tile_same = gpu_pops3
        .iter()
        .zip(cpu_pops3.iter())
        .take_while(|(g, c)| g.0 == c.0)
        .count();
    log.push_str(&format!(
        "  gpu vs cpu over ALL pops (not just the recorded prefix): tiles identical for {tile_same}/{n_enq3}; \
         priority bits identical on {prio_same}/{n_enq3}; full claim sequence (beyond the recorded \
         prefix, unobservable) {} pops\n",
        gpu_claims3.len()
    ));

    // --- WHICH mechanic is missing on the contested border? ---
    // The engine's list inserts one tile that the single-pass model never
    // reaches: test the in-tick re-enqueue model on it.
    let (c_inc, n_inc) = frontier_claims_incremental(
        &case3.border,
        &case3.owned_any,
        &case3.owned_mine,
        &map.terrain,
        map.width,
        map.height,
        case3.order,
        case3.priority_tick,
        case3.cursor,
        n_claim3 as u32,
    );
    log.push_str(&format!(
        "  in-tick re-enqueue model: enqueued {n_inc} (single pass: {n_any}); claims {}\n",
        fmt_u32s(&c_inc)
    ));
    log.push_str(&format!(
        "  in-tick re-enqueue vs ENGINE: {} (first difference at index {})\n",
        if c_inc == case3.expected_order {
            "IDENTICAL".to_string()
        } else {
            "different".to_string()
        },
        first_diff(&c_inc, &case3.expected_order)
            .map(|i| i.to_string())
            .unwrap_or_else(|| "none".to_string())
    ));

    // --- the engine-faithful in-tick model (budget draw + add_neighbors before
    //     the conquer + on_border re-check), swept over PRNG cursors ---
    for consume in [false, true] {
        let (hits, best, cur, claims, draws, n_enq_m) = sweep_engine_model(
            &case3.border,
            &case3.owned_any,
            &case3.owned_mine,
            &map.terrain,
            map.width,
            map.height,
            case3.priority_tick,
            &case3.expected_order,
            consume,
            3000,
        );
        log.push_str(&format!(
            "  engine-loop model (budget draw {}): cursors reproducing the engine's 27 claims exactly: {hits}; \
             best prefix {best}/27 at cursor {cur} (draws {draws}, enqueues {n_enq_m}); claims {}\n",
            if consume { "consumed" } else { "skipped" },
            fmt_u32s(&claims)
        ));
    }

    // --- THE DEVICE ON THE ENGINE'S OWN LOOP (the fix) ---------------------
    // One device launch = the attack's whole life on the GPU: thread j replays
    // `AttackExecution::init` at the init tick and then every pop tick, and
    // writes the tile conquered at GLOBAL claim index j. The claim at index 23
    // is tile 838544, a SECOND-RING tile: the pop loop reaches it only because
    // `to_conquer` is carried across ticks (attack.rs:17) and each pop
    // re-enqueues from inside its own loop (attack.rs:292). The one-pass model
    // two lines up cannot pop it at all.
    let owner_sid3 = cc.owner_sid as u16;
    let plane3 = contested_plane(&cc, map.width, map.height);
    let budget3 = n_claim3 as u32;
    let cpu_eng3 = contested_engine_run(&case3, &plane3, &map, owner_sid3, budget3, true, true);
    log.push_str(&format!(
        "  ENG-LOOP cpu (engine_tick, persistent heap, in-tick add_neighbors): claims {}; \
         enqueued {}; draws {}; refreshes {}; N,S,W,E order {}\n",
        fmt_u32s(&cpu_eng3.claims),
        cpu_eng3.n_enq,
        cpu_eng3.draws,
        cpu_eng3.refreshes,
        case3.order
    ));
    let n_state = ofcuda_tick::STATE_WORDS;
    let n_threads_e = n_claim3.max(n_state).div_ceil(256) * 256;
    let d_plane3 = DeviceBuffer::from_host(&stream, &plane3)?;
    let d_bflat3 = DeviceBuffer::from_host(&stream, &case3.border)?;
    let d_boff3 = DeviceBuffer::from_host(&stream, &[0u32, case3.border.len() as u32])?;
    let d_budg3 = DeviceBuffer::from_host(&stream, &[budget3])?;
    let d_popt3 = DeviceBuffer::from_host(&stream, &[case3.priority_tick])?;
    let mut d_at3 = DeviceBuffer::<u32>::zeroed(&stream, n_threads_e)?;
    let mut d_ab3 = DeviceBuffer::<u32>::zeroed(&stream, n_threads_e)?;
    let mut d_as3 = DeviceBuffer::<u32>::zeroed(&stream, n_state)?;
    let prep_eng3 = module
        .prepare_engine_life_claims(LaunchConfig1D::new((n_threads_e / 256) as u32, 256, 0))?;
    module.engine_life_claims(
        &stream,
        &prep_eng3,
        &d_plane3,
        &d_terrain3,
        map.width,
        map.height,
        case3.order,
        owner_sid3,
        case3.priority_tick - 1,
        1,
        1,
        &d_bflat3,
        &d_boff3,
        &d_budg3,
        &d_popt3,
        &mut d_at3,
        &mut d_ab3,
        &mut d_as3,
    )?;
    let gpu_eng3: Vec<u32> = d_at3.to_host_vec(&stream)?[..n_claim3].to_vec();
    let gpu_pri3: Vec<f32> = d_ab3.to_host_vec(&stream)?[..n_claim3]
        .iter()
        .map(|b| f32::from_bits(*b))
        .collect();
    let gpu_state3 = d_as3.to_host_vec(&stream)?;
    log.push_str(&format!("  ENG-LOOP gpu (same kernel, one thread per claim index): claims {}\n", fmt_u32s(&gpu_eng3)));
    log.push_str(&format!("  ENGINE recorded (dump, tick {}): {}\n", cc.claim_tick, fmt_u32s(&case3.expected_order)));
    let e_gpu = first_diff(&gpu_eng3, &case3.expected_order);
    let e_cpu = first_diff(&cpu_eng3.claims, &case3.expected_order);
    let one_pass3: Vec<u32> = c_any.iter().take(n_claim3).copied().collect();
    log.push_str(&format!(
        "  FIRST DIFFERING INDEX vs engine: gpu {} / cpu {} / one-pass model {}\n",
        e_gpu.map(|i| i.to_string()).unwrap_or_else(|| "none (identical)".into()),
        e_cpu.map(|i| i.to_string()).unwrap_or_else(|| "none (identical)".into()),
        first_diff(&one_pass3, &case3.expected_order)
            .map(|i| i.to_string())
            .unwrap_or_else(|| "none (identical)".into())
    ));
    log.push_str(&format!(
        "  claim #23: engine {} - gpu {} - cpu {}; gpu==cpu==engine on all {} claims: {}\n",
        case3.expected_order.get(23).map(|t| t.to_string()).unwrap_or_else(|| "-".into()),
        gpu_eng3.get(23).map(|t| t.to_string()).unwrap_or_else(|| "-".into()),
        cpu_eng3.claims.get(23).map(|t| t.to_string()).unwrap_or_else(|| "-".into()),
        n_claim3,
        gpu_eng3 == case3.expected_order && cpu_eng3.claims == case3.expected_order
    ));
    let state_ok3 = gpu_state3.len() == n_state && gpu_state3 == cpu_eng3.state;
    log.push_str(&format!(
        "  carried state after the life (prng s0..s3, calls, heap len + 512 heap words): gpu == cpu: {state_ok3}; heap len {}; calls {}\n",
        gpu_state3.get(5).copied().unwrap_or(u32::MAX),
        gpu_state3.get(4).copied().unwrap_or(u32::MAX)
    ));
    let mut pri_ok3 = 0usize;
    for i in 0..n_claim3.min(gpu_pri3.len()) {
        if gpu_pri3[i].to_bits() == cpu_eng3.claim_pri_bits.get(i).copied().unwrap_or(u32::MAX) {
            pri_ok3 += 1;
        }
    }
    log.push_str(&format!(
        "  pop priorities (f32 bits) gpu vs cpu: {pri_ok3}/{n_claim3} identical\n"
    ));
    if !(gpu_eng3 == case3.expected_order && cpu_eng3.claims == case3.expected_order && state_ok3) {
        proof_ok = false;
    }

    // =====================================================================
    // PART 2: the composed tick over ticks 300..331 of b002
    // =====================================================================
    log.push_str("\n\n===== PART 2: composed tick, b002 ticks 300..331 =====\n");
    let win = load_tick_window(&tick_window_path())?;
    let plane0 = win.plane0(map.width, map.height);
    let n_tiles = plane0.len();
    log.push_str(&format!(
        "game {}; tick0 {}; players0 {} ({} tiles); steps {}\n",
        win.game,
        win.tick0,
        win.players0.len(),
        win.players0.iter().map(|p| p.1.len()).sum::<usize>(),
        win.steps.len()
    ));
    log.push_str(&format!(
        "hash contract: FNV-1a 64, offset {:016x}, prime {:016x}, over {n_tiles} u16 LE \
         (offset {:016x} for an all-zero plane)\n",
        FNV_OFFSET,
        FNV_PRIME,
        fnv1a_u16_le(&vec![0u16; n_tiles])
    ));
    let mut d_a = DeviceBuffer::from_host(&stream, &plane0)?;
    let mut d_b = DeviceBuffer::<u16>::zeroed(&stream, n_tiles)?;
    let d_terrain2 = DeviceBuffer::from_host(&stream, &map.terrain)?;
    let mut d_claims2 = DeviceBuffer::<u32>::zeroed(&stream, 64)?;
    let mut d_hash = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
    let d_dummy = DeviceBuffer::from_host(&stream, &[u32::MAX])?;
    let prep_tile = module
        .prepare_compose_tick(LaunchConfig1D::new((n_tiles as u32).div_ceil(256), 256, 0))?;
    let prep_hash = module.prepare_fnv_state_hash(LaunchConfig1D::new(1, 1, 0))?;
    // The engine's own plane, rebuilt by applying the engine's recorded claim
    // sets - the yardstick for WHICH tiles are missing, not just that a hash
    // differs (and a check that the recorded claim sets are complete: its hash
    // must equal every recorded per-tick hash).
    let mut engine_plane = plane0.clone();
    let mut matched_ticks: Vec<u32> = Vec::new();
    let mut first_bad_tick: Option<u32> = None;
    let mut prev_block: Option<&ofcuda_tick::TickBlock> = None;
    let mut engine_recon_ok = true;
    for step in &win.steps {
        // --- engine side: apply the recorded claims, check the hash ---
        for b in &step.blocks {
            for t in &b.expected_claims {
                engine_plane[*t as usize] = b.owner_sid as u16;
            }
        }
        let h_engine = fnv1a_u16_le(&engine_plane);
        if h_engine != step.expected_hash {
            engine_recon_ok = false;
            log.push_str(&format!(
                "  !! engine-side reconstruction of tick {} gives {} but the dump recorded {}\n",
                step.tick,
                hex64(h_engine),
                hex64(step.expected_hash)
            ));
        }

        // --- device side: one composed tick ---
        let mut kernel_claims: Vec<Vec<u32>> = Vec::new();
        if step.blocks.is_empty() {
            module.compose_tick(
                &stream,
                &prep_tile,
                &d_a,
                &d_terrain2,
                map.width,
                map.height,
                &d_dummy,
                &d_dummy,
                &d_dummy,
                ORDER_NSWE,
                step.tick,
                0,
                0,
                0,
                &mut d_b,
                &mut d_claims2,
            )?;
        } else {
            for b in &step.blocks {
                let d_border = DeviceBuffer::from_host(&stream, &b.border)?;
                let d_any = DeviceBuffer::from_host(&stream, &b.owned_any)?;
                let d_mine = DeviceBuffer::from_host(&stream, &b.owned_mine)?;
                module.compose_tick(
                    &stream,
                    &prep_tile,
                    &d_a,
                    &d_terrain2,
                    map.width,
                    map.height,
                    &d_border,
                    &d_any,
                    &d_mine,
                    ORDER_NSWE,
                    step.tick,
                    0,
                    b.budget,
                    b.owner_sid,
                    &mut d_b,
                    &mut d_claims2,
                )?;
                let c = d_claims2.to_host_vec(&stream)?;
                kernel_claims.push(c[..b.budget as usize].to_vec());
            }
        }
        module.fnv_state_hash(&stream, &prep_hash, &d_b, &mut d_hash)?;
        let h_kernel = d_hash.to_host_vec(&stream)?[0];
        let kernel_plane_snapshot = d_b.to_host_vec(&stream)?;

        // --- compare ---
        let hash_ok = h_kernel == step.expected_hash;
        let engine_plane_ok = kernel_plane_snapshot == engine_plane;
        let mut claims_ok = step.blocks.len() == kernel_claims.len();
        if claims_ok {
            for (b, k) in step.blocks.iter().zip(kernel_claims.iter()) {
                claims_ok &= b.expected_claims == *k;
            }
        }
        if hash_ok {
            matched_ticks.push(step.tick);
        } else if first_bad_tick.is_none() {
            first_bad_tick = Some(step.tick);
            log.push_str(&format!(
                "\nFIRST MISMATCH: tick {} \n  kernel hash {}  engine hash {}\n",
                step.tick,
                hex64(h_kernel),
                hex64(step.expected_hash)
            ));
            log.push_str(&format!(
                "  plane equal to the engine's (applying the engine's OWN claim sets): {}\n",
                engine_plane_ok
            ));
            let k_owner: Vec<u16> = kernel_plane_snapshot.clone();
            let mut diff_tiles: Vec<u32> = Vec::new();
            for (t, (k, e)) in k_owner.iter().zip(engine_plane.iter()).enumerate() {
                if k != e {
                    diff_tiles.push(t as u32);
                }
            }
            log.push_str(&format!(
                "  tile-owner differences vs the engine plane at this tick: {}\n",
                diff_tiles.len()
            ));
            for (b, k) in step.blocks.iter().zip(kernel_claims.iter()) {
                log.push_str(&format!(
                    "  sid {}: engine claims ({}) {}\n             kernel claims ({}) {}\n",
                    b.owner_sid,
                    b.expected_claims.len(),
                    fmt_u32s(&b.expected_claims),
                    k.len(),
                    fmt_u32s(k)
                ));
                let eset: std::collections::HashSet<u32> = b.expected_claims.iter().copied().collect();
                let kset: std::collections::HashSet<u32> = k.iter().copied().collect();
                let missing: Vec<u32> = b.expected_claims.iter().copied().filter(|t| !kset.contains(t)).collect();
                let extra: Vec<u32> = k.iter().copied().filter(|t| !eset.contains(t)).collect();
                log.push_str(&format!(
                    "  claim SETS: missing {} {}; extra {} {}\n",
                    missing.len(),
                    fmt_u32s(&missing),
                    extra.len(),
                    fmt_u32s(&extra)
                ));
                // Is a missing tile a neighbour of a tile claimed in the SAME
                // tick (in-tick expansion / cluster capture), or of the
                // pre-tick border (which the one-pass model already covers)?
                let mut n_same_tick = 0usize;
                let mut n_of_border = 0usize;
                let mut n_neither = 0usize;
                for t in &missing {
                    let mut nbuf = [0u32; 4];
                    let nn = ofcuda_tick::neighbors4(case3.order, *t, map.width, map.height, &mut nbuf);
                    let mut same = false;
                    let mut on_border = false;
                    for i in 0..nn as usize {
                        let nb = nbuf[i];
                        if eset.contains(&nb) || kset.contains(&nb) {
                            same = true;
                        }
                        if b.border.contains(&nb) {
                            on_border = true;
                        }
                    }
                    if same {
                        n_same_tick += 1;
                    } else if on_border {
                        n_of_border += 1;
                    } else {
                        n_neither += 1;
                    }
                }
                log.push_str(&format!(
                    "  missing-tile provenance: adjacent to a tile claimed in the SAME tick {}; \
                     only adjacent to a pre-tick border tile {}; neither {} \
                     (the last class is outside the pre-tick frontier entirely)\n",
                    n_same_tick, n_of_border, n_neither
                ));
                // Can ANY cursor of the one-pass model reproduce this tick's
                // claim order? If not, the model is structurally short.
                let mut cursor_hits = 0usize;
                let mut best = 0usize;
                for cur in 0..3000u32 {
                    let (cl, _) = frontier_claims_with(
                        &b.border,
                        &b.owned_any,
                        &b.owned_mine,
                        &map.terrain,
                        map.width,
                        map.height,
                        ORDER_NSWE,
                        step.tick,
                        cur,
                        b.budget,
                    );
                    if cl == b.expected_claims {
                        cursor_hits += 1;
                    }
                    let m = cl
                        .iter()
                        .zip(b.expected_claims.iter())
                        .take_while(|(a, c)| a == c)
                        .count();
                    if m > best {
                        best = m;
                    }
                }
                log.push_str(&format!(
                    "  one-pass cursor sweep 0..2999 (host, same core): {} cursors reproduce \
                     this tick's claim order exactly; best common prefix {}/{} \n",
                    cursor_hits,
                    best,
                    b.expected_claims.len()
                ));
                // the engine-faithful in-tick model, same sweep
                for consume in [false, true] {
                    let (hits2, best2, cur2, claims2, draws2, n_enq2) = sweep_engine_model(
                        &b.border,
                        &b.owned_any,
                        &b.owned_mine,
                        &map.terrain,
                        map.width,
                        map.height,
                        step.tick,
                        &b.expected_claims,
                        consume,
                        3000,
                    );
                    log.push_str(&format!(
                        "  engine-loop model (in-tick add_neighbors + budget draw {}): {} cursors \
                         reproduce tick {} exactly; best prefix {}/{} at cursor {cur2} (draws {draws2}, \
                         enqueues {n_enq2}); claims {}\n",
                        if consume { "consumed" } else { "skipped" },
                        hits2,
                        step.tick,
                        best2,
                        b.expected_claims.len(),
                        fmt_u32s(&claims2)
                    ));
                }
                // --- does the frontier PERSIST across the tick boundary? ---
                // The engine's `AttackExecution` keeps `to_conquer` between
                // ticks (`attack.rs:17,206-325`); this crate rebuilds it from
                // the pre-tick border every tick. If the leftover heap is what
                // decides the next tick's claims, then popping
                // (this tick + next tick) from the PREVIOUS tick's frontier
                // should produce this tick's claims as its tail.
                if let Some(pb) = prev_block {
                    let n_prev = pb.expected_claims.len() as u32;
                    let n_cur = b.expected_claims.len() as u32;
                    let (cont, _n) = frontier_claims_with(
                        &pb.border,
                        &pb.owned_any,
                        &pb.owned_mine,
                        &map.terrain,
                        map.width,
                        map.height,
                        ORDER_NSWE,
                        pb.tick,
                        0,
                        n_prev + n_cur,
                    );
                    let tail: Vec<u32> = cont.iter().skip(n_prev as usize).cloned().collect();
                    log.push_str(&format!(
                        "  PERSISTENT-HEAP TEST (single pass, prev tick {} frontier): pops {}..{} = {}\n",
                        pb.tick,
                        n_prev,
                        n_prev + n_cur,
                        fmt_u32s(&tail)
                    ));
                    log.push_str(&format!(
                        "     vs this tick's engine claims {}\n     -> {}\n",
                        fmt_u32s(&b.expected_claims),
                        if tail == b.expected_claims {
                            "MATCH: the next tick's claims ARE the leftover heap".to_string()
                        } else {
                            format!(
                                "no match (first difference at index {})",
                                first_diff(&tail, &b.expected_claims)
                                    .map(|i| i.to_string())
                                    .unwrap_or_else(|| "none".to_string())
                            )
                        }
                    ));
                    // and the same with the engine-faithful loop, cursor swept
                    let mut hits3 = 0usize;
                    let mut bestp = 0usize;
                    let mut best_tail: Vec<u32> = Vec::new();
                    let mut best_cur3 = 0u32;
                    for cur in 0..3000u32 {
                        let r = engine_tick_model(
                            &pb.border,
                            &pb.owned_any,
                            &pb.owned_mine,
                            &map.terrain,
                            map.width,
                            map.height,
                            ORDER_NSWE,
                            pb.tick,
                            cur,
                            n_prev + n_cur,
                            false,
                        );
                        let t2: Vec<u32> = r.claims.iter().skip(n_prev as usize).cloned().collect();
                        if t2 == b.expected_claims {
                            hits3 += 1;
                        }
                        if !best_tail.is_empty() && t2 == b.expected_claims {
                            best_tail = t2.clone();
                            best_cur3 = cur;
                        }
                        bestp = bestp.max(
                            t2.iter()
                                .zip(b.expected_claims.iter())
                                .take_while(|(x, y)| x == y)
                                .count(),
                        );
                    }
                    log.push_str(&format!(
                        "  PERSISTENT-HEAP TEST (engine loop, in-tick add_neighbors): cursors whose \
                         tail reproduces {} exactly: {hits3} (best prefix {bestp}/{}); e.g. cursor \
                         {best_cur3}: {}\n",
                        b.expected_claims.len(),
                        b.expected_claims.len(),
                        fmt_u32s(&best_tail)
                    ));
                }
            }
        }
        log.push_str(&format!(
            "  tick {:3}: hash {}  hash_ok {}  claims_ok {}  plane_equals_engine {}\n",
            step.tick,
            hex64(h_kernel),
            hash_ok,
            claims_ok,
            engine_plane_ok
        ));

        std::mem::swap(&mut d_a, &mut d_b);
        prev_block = step.blocks.last();
    }
    log.push_str(&format!(
        "\nPART 2 tally (compose_tick, the ONE-PASS model as shipped): {}/{} ticks match the \
         engine's per-tick state hash; first mismatching tick {}; engine-side reconstruction \
         complete: {}\n",
        matched_ticks.len(),
        win.steps.len(),
        first_bad_tick
            .map(|t| t.to_string())
            .unwrap_or_else(|| "none (all matched)".to_string()),
        engine_recon_ok
    ));
    log.push_str(&format!(
        "matched ticks: {}\n",
        if matched_ticks.is_empty() {
            "(none)".to_string()
        } else {
            matched_ticks
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<String>>()
                .join(",")
        }
    ));

    // ---------------------------------------------------------------------
    // THE FIX, RE-MEASURED ON THE WINDOW
    // ---------------------------------------------------------------------
    // Same predicate as PART 1 (`engine_tick`, the engine's own loop), now
    // carried across the window: `to_conquer` persists between ticks
    // (attack.rs:17), `add_neighbors(tile, tick)` runs inside the pop loop
    // (attack.rs:292), a pop is skipped unless the tile is still terra nullius
    // AND has an attacker-owned neighbour (attack.rs:284), and the one budget
    // draw (attack.rs:239-254) precedes the pop loop. One device launch
    // replays the attack's whole life; thread j writes global claim #j.
    let cpu_win_e = window_engine_run(&map, &win, true, true);
    let cpu_no_it = window_engine_run(&map, &win, true, false);
    let cpu_no_bd = window_engine_run(&map, &win, false, true);
    let claim_steps: Vec<&ofcuda_tick::TickStep> = win
        .steps
        .iter()
        .filter(|s| !s.blocks.is_empty())
        .collect();
    let first_claim_tick = claim_steps.first().map(|s| s.tick).unwrap_or(0);
    let init_tick_e = first_claim_tick.saturating_sub(2);
    let mut bflat: Vec<u32> = Vec::new();
    let mut boff: Vec<u32> = vec![0];
    let mut bud: Vec<u32> = Vec::new();
    let mut popt: Vec<u32> = Vec::new();
    for s in claim_steps.iter() {
        for b in &s.blocks {
            bflat.extend_from_slice(&b.border);
            bud.push(b.budget);
        }
        boff.push(bflat.len() as u32);
        popt.push(s.tick - 1);
    }
    let n_claims_e: usize = bud.iter().map(|b| *b as usize).sum();
    let owner_e = claim_steps
        .first()
        .and_then(|s| s.blocks.first())
        .map(|b| b.owner_sid)
        .unwrap_or(0) as u16;
    log.push_str(&format!(
        "\nTHE FIX: engine loop over the window. attack owner {owner_e}; init tick {init_tick_e}; \
         pop ticks {init_tick_e}+1..{}; per-tick budgets {}; total claims {n_claims_e} \
         (recorded 133); platform border N,S,W,E\n",
        popt.last().copied().unwrap_or(0),
        fmt_u32s(&bud)
    ));
    let cpu_matched = cpu_win_e.matched;
    let cpu_early: Vec<u32> = cpu_win_e.ticks.iter().take(13).map(|t| t.tick).collect();
    log.push_str(&format!(
        "  ENG-LOOP cpu: {}/{} ticks match the engine's per-tick FNV-1a-64 hash; first mismatching \
         tick {}; claims {}/133 in the engine's order; draws {}; refreshes {}; \
         matched ticks {}\n",
        cpu_matched,
        win.steps.len(),
        cpu_win_e
            .first_bad
            .map(|t| t.to_string())
            .unwrap_or_else(|| "none (all matched)".into()),
        cpu_win_e
            .ticks
            .iter()
            .filter(|t| t.order_ok)
            .count(),
        cpu_win_e.draws,
        cpu_win_e.refreshes,
        if cpu_matched == win.steps.len() as u32 {
            "(all)".into()
        } else {
            format!(
                "{}",
                win.steps
                    .iter()
                    .map(|s| s.tick)
                    .filter(|t| !cpu_win_e.ticks.iter().any(|x| x.tick == *t))
                    .map(|t| t.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
    ));
    log.push_str(&format!(
        "    (first 13 ticks, sanity: {cpu_early:?})\n"
    ));

    let n_state_e = ofcuda_tick::STATE_WORDS;
    let n_threads_w = n_claims_e.max(n_state_e).div_ceil(256) * 256;
    let d_planew = DeviceBuffer::from_host(&stream, &win.plane0(map.width, map.height))?;
    let d_bflatw = DeviceBuffer::from_host(&stream, &bflat)?;
    let d_boffw = DeviceBuffer::from_host(&stream, &boff)?;
    let d_budw = DeviceBuffer::from_host(&stream, &bud)?;
    let d_poptw = DeviceBuffer::from_host(&stream, &popt)?;
    let prep_engw = module
        .prepare_engine_life_claims(LaunchConfig1D::new((n_threads_w / 256) as u32, 256, 0))?;
    let run_life = |consume: u32, in_tick: u32| -> Result<(Vec<u32>, Vec<u32>), Box<dyn std::error::Error>> {
        let mut d_at = DeviceBuffer::<u32>::zeroed(&stream, n_threads_w)?;
        let mut d_ab = DeviceBuffer::<u32>::zeroed(&stream, n_threads_w)?;
        let mut d_as = DeviceBuffer::<u32>::zeroed(&stream, n_state_e)?;
        module.engine_life_claims(
            &stream,
            &prep_engw,
            &d_planew,
            &d_terrain2,
            map.width,
            map.height,
            ORDER_NSWE,
            owner_e,
            init_tick_e,
            consume,
            in_tick,
            &d_bflatw,
            &d_boffw,
            &d_budw,
            &d_poptw,
            &mut d_at,
            &mut d_ab,
            &mut d_as,
        )?;
        let mut tiles = d_at.to_host_vec(&stream)?;
        tiles.truncate(n_claims_e);
        Ok((tiles, d_as.to_host_vec(&stream)?))
    };
    let (gpu_fix, gpu_fix_state) = run_life(1, 1)?;
    let (gpu_no_it, _) = run_life(1, 0)?;
    let (gpu_no_bd, _) = run_life(0, 1)?;

    let cpu_claims_e: Vec<u32> = cpu_win_e
        .ticks
        .iter()
        .flat_map(|t| t.claims.iter().copied())
        .collect();
    log.push_str(&format!(
        "  ENG-LOOP gpu: {} claims; identical to the cpu engine loop: {} (first difference {}); \
         identical to the engine's recorded per-tick claim sets: {}\n",
        gpu_fix.len(),
        gpu_fix == cpu_claims_e,
        first_diff(&gpu_fix, &cpu_claims_e)
            .map(|i| i.to_string())
            .unwrap_or_else(|| "none".into()),
        cpu_claims_e.len() == 133
    ));
    let state_ok_e = gpu_fix_state == cpu_win_e.final_state;
    log.push_str(&format!(
        "  carried state after tick {}: gpu == cpu: {state_ok_e}; heap len {} \n",
        win.steps.last().map(|s| s.tick).unwrap_or(0),
        gpu_fix_state.get(5).copied().unwrap_or(u32::MAX)
    ));

    let mut variants: Vec<(String, &Vec<u32>)> = vec![
        ("gpu, THE FIX (budget draw + in-tick add_neighbors)".to_string(), &gpu_fix),
        ("gpu, control: no in-tick add_neighbors".to_string(), &gpu_no_it),
        ("gpu, control: no budget draw".to_string(), &gpu_no_bd),
    ];
    let mut fix_ok_ticks = 0u32;
    let mut fix_first_bad: Option<u32> = None;
    let mut fix_matched: Vec<u32> = Vec::new();
    for (label, claims) in variants.drain(..) {
        let (ok, bad, matched) = ofcuda_tick::window_hashes_from_claims(&map, &win, claims);
        log.push_str(&format!(
            "  {label}: {ok}/{} ticks match the engine's per-tick state hash; first mismatching tick \
             {}; claim order identical to the engine's until index {}\n",
            win.steps.len(),
            bad.map(|t| t.to_string())
                .unwrap_or_else(|| "none (all matched)".into()),
            first_diff(
                claims,
                &cpu_win_e
                    .ticks
                    .iter()
                    .flat_map(|t| t.expected.iter().copied())
                    .collect::<Vec<u32>>()
            )
            .map(|i| i.to_string())
            .unwrap_or_else(|| "none".into())
        ));
        if label.contains("THE FIX") {
            fix_ok_ticks = ok;
            fix_first_bad = bad;
            fix_matched = matched;
        }
    }
    log.push_str(&format!(
        "  controls (cpu, same code path): in-tick add_neighbors off {}/{}; budget draw off {}/{}\n",
        cpu_no_it.matched,
        win.steps.len(),
        cpu_no_bd.matched,
        win.steps.len()
    ));
    // the device hashes the plane itself for the last tick (no host FNV involved)
    let mut d_lastplane = DeviceBuffer::from_host(&stream, &win.plane0(map.width, map.height))?;
    let mut last_plane = win.plane0(map.width, map.height);
    let mut at = 0usize;
    for step in &win.steps {
        for b in &step.blocks {
            for t in &gpu_fix[at..(at + b.budget as usize).min(gpu_fix.len())] {
                last_plane[*t as usize] = b.owner_sid as u16;
            }
            at += b.budget as usize;
        }
    }
    d_lastplane = DeviceBuffer::from_host(&stream, &last_plane)?;
    let mut d_hash_last = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
    let prep_hash2 = module.prepare_fnv_state_hash(LaunchConfig1D::new(1, 1, 0))?;
    module.fnv_state_hash(&stream, &prep_hash2, &d_lastplane, &mut d_hash_last)?;
    let h_dev = d_hash_last.to_host_vec(&stream)?[0];
    log.push_str(&format!(
        "  device FNV over the plane the GPU's own claims imply, tick {}: {} (engine {}) -> {}\n",
        win.steps.last().map(|s| s.tick).unwrap_or(0),
        hex64(h_dev),
        hex64(win.steps.last().map(|s| s.expected_hash).unwrap_or(0)),
        h_dev == win.steps.last().map(|s| s.expected_hash).unwrap_or(1)
    ));
    if fix_ok_ticks != win.steps.len() as u32 || gpu_fix != cpu_claims_e || !state_ok_e {
        proof_ok = false;
    }
    log.push_str(&format!(
        "\nWINDOW RE-MEASURED: {fix_ok_ticks}/{} ticks match the engine's per-tick FNV-1a-64 hash; \
         first mismatching tick {}; engine-loop fix on the window {}\n",
        win.steps.len(),
        fix_first_bad
            .map(|t| t.to_string())
            .unwrap_or_else(|| "none (all matched)".into()),
        if fix_ok_ticks == win.steps.len() as u32 { "MATCHES" } else { "differs" }
    ));
    log.push_str(&format!(
        "  matched ticks: {}\n",
        fix_matched.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",")
    ));

    // --- the FMA hazard: read the generated PTX, then measure the input space ---
    let ptx_counts = std::fs::read_to_string("ofcuda_tick.ptx")
        .ok()
        .and_then(|t| {
            let start = t.find(".visible .entry frontier_step")?;
            let seg = &t[start..];
            let end = seg.find("\n.visible .entry").unwrap_or(seg.len());
            let entry = &seg[..end];
            Some((
                entry.matches("fma.rn.f32").count(),
                entry.matches("mul.f32").count(),
                entry.matches("add.f32").count(),
                entry.matches("sub.f32").count(),
            ))
        });
    let (fma_total, fma_diff, fma_examples) = fma_contraction_sensitivity();
    match ptx_counts {
        Some((fma, mul, add, sub)) => log.push_str(&format!(
            "\nPTX of `frontier_step` (read from ofcuda_tick.ptx, not eyeballed): \
             \n  fma.rn.f32 {fma}, mul.f32 {mul}, add.f32 {add}, sub.f32 {sub}\
             \n  -> {}.\n",
            if fma == 1 && mul == 1 && add == 1 && sub == 1 {
                "one contraction, for `(r + 10) * f + tick`; `1 - k*0.5 + mag*0.5` not contracted"
            } else {
                "UNEXPECTED contraction pattern - re-read the PTX before trusting the f32 claim"
            }
        )),
        None => log.push_str(
            "\nPTX of `frontier_step`: ofcuda_tick.ptx not found, contraction NOT verified\n",
        ),
    }
    log.push_str(&format!(
        "\nFMA contraction: the PTX of `frontier_step` contains one `fma.rn.f32`\
         \n  (for `(r + 10) * f + tick`), which is NOT what a Rust f32 host computes.\
         \n  Contracted vs two-step agree on {}/{} points of (r, owner-count, terrain, tick): {}\n",
        fma_total - fma_diff,
        fma_total,
        if fma_diff == 0 {
            "no divergence - the contraction is inert for this arithmetic".to_string()
        } else {
            format!("{fma_diff} DIVERGE: {}", fma_examples.join("; "))
        }
    ));

    log.push_str(&format!(
        "\nPROOF claim_order_and_priorities_reproduced {proof_ok}\n"
    ));
    print!("{log}");
    std::fs::write("comparison_gpu.txt", &log)?;
    if !proof_ok {
        eprintln!("FAILED: the port does not reproduce the recorded claim order/priorities");
        std::process::exit(1);
    }
    Ok(())
}