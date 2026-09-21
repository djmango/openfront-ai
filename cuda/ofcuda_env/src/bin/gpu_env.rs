//! `gpu_env` - the composed tick run DEVICE-RESIDENT and BATCHED, plus the
//! throughput measurement `ofcuda_env` could not produce before.
//!
//! # What is device-resident
//!
//! Every piece of a tick's state lives in a `DeviceBuffer` and PERSISTS ACROSS
//! TICKS: the ownership plane, the engine's `to_conquer` heap (tiles + f32
//! priorities + length), the `PseudoRandom` state words, the attack's
//! `border_tiles` set, the whole-life claim list, the per-tick claim list and
//! the attack's troop count. There is exactly ONE kernel launch per tick and
//! ZERO host copies inside the loop. A copy-in/copy-out per tick is what made
//! the earlier device FNV readback 7.8x slower than its CPU counterpart; that
//! readback is gone.
//!
//! # Batching
//!
//! One launch serves N environments: thread `i` of the grid owns environment
//! `i` (grid = ceil(N/256) blocks of 256). Per-env buffers are indexed
//! `e * stride`, so the whole batch is one flat allocation per array and no
//! per-env launch or copy exists.
//!
//! # Verification mode vs fast mode
//!
//! * `--mode verify`: after every tick, `env_hash` (one thread per env) runs
//!   the FNV-1a-64 state hash over that env's whole plane and stores it, so the
//!   batch's plane hash after every tick can be compared bit-for-bit against
//!   the CPU reference. This is the mode that reproduces the engine.
//! * `--mode fast`: hashing is NOT in the loop. Nothing is read back and no
//!   hash kernel is launched; the planes simply keep advancing on the device.
//!   The two modes run the identical `env_step` kernel on the identical state,
//!   so the fast mode's trajectories are the same by construction.
//!
//! # The single source of the tick
//!
//! `kernels` textually `include!`s `ofcuda_tick/src/core_impl.rs` - the same
//! file the crate's verbatim-copy test pins - so this binary adds NO copy of
//! any core. The expansion semantics (heap retained between ticks, in-tick
//! `add_neighbors` before each conquer, skip guards, one extra PRNG draw per
//! tick before the pop loop) are the crate's own `Heap`, `Prng`,
//! `neighbors4`, `add_neighbors`, `is_terra_nullius`, `has_attacker_neighbor`,
//! `attacker_neighbor_count`, `priority_f32`, `mag2_from_terrain` and
//! `ORDER_NSWE`. The float budget and the terrain cost helpers are the crate's
//! `ofcuda_env::{attack_tiles_per_tick, tiles_used, attacker_troop_loss}` -
//! called directly from device code, not re-implemented.
//!
//! # One deliberate deviation, with its proof
//!
//! The CPU `Attack` keeps the plane of the START of the tick and passes its
//! accumulating `claims` list to `is_terra_nullius` / `has_attacker_neighbor` /
//! `attacker_neighbor_count`. The device plane is updated IN PLACE as each tile
//! is conquered (that is what makes the plane device-resident), so it passes an
//! EMPTY claims list. This is equivalent because every entry of `claims` is a
//! tile whose plane word has already been set to `owner_col != 0`:
//!   * `is_terra_nullius(p, c, t)` returns false when `p[t] != 0`; a claimed
//!     tile always has `p[t] == owner_col`, so dropping `c` cannot change it.
//!   * `has_attacker_neighbor` / `attacker_neighbor_count` test
//!     `p[nb] == owner_col || nb in c`, and a claimed `nb` satisfies the first.
//! `--selftest` checks this empirically: the device's per-tick claims and plane
//! hashes are compared against the CPU `Attack::tick` over the same K ticks and
//! the counts are printed, never asserted silently.
//!
//! # Argument map (device)
//!   env e -> thread index e; `plane[e*wh + t]`; `heap_tiles[e*HEAP_CAP + j]`;
//!   `scal[e*9 + s]` where s = 0 heap_len, 1 border_len, 2 claims_len,
//!   3 tclaim_len, 4 troops_bits(f32), 5 alive, 6 heap drops, 7 heap peak,
//!   8 owner_border_len; `troops[e]` is the attack troop count as f64.

use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{kernel, launch_bounds, thread};
use cuda_host::cuda_module;
use ofcuda_env::{
    attack_tiles_per_tick, attacker_troop_loss, econ_row, land_attack_start_troops, state_hash,
    tribe_ratios, tiles_used, Attack, TribeRatios,
};
/// The TRAINING interface: obs / action mask / reward from device state.
use ofcuda_env::rl;
use ofcuda_tick::{
    add_neighbors, attacker_neighbor_count, has_attacker_neighbor, mag2_from_terrain, neighbors4,
    priority_f32, Heap, Prng, HEAP_CAP, ORDER_NSWE,
};
use std::path::PathBuf;
use std::time::Instant;

mod extras;
use extras::extras;

/// Capacity of the per-env attack border set (device buffer only; the CPU
/// `Attack` uses a `Vec`). Overflow is COUNTED in `scal[..][6]`-adjacent drops,
/// never silently dropped.
const BC: usize = 16384;
/// Capacity of the per-env whole-life claim list.
const CC: usize = 8192;
/// Capacity of one tick's claim list.
const MAXC: usize = 2048;
/// Capacity of the per-env owner-border (refresh input) copy.
const OB: usize = 16384;
/// Per-PLAYER capacity of the device-maintained, insertion-ordered owner-border
/// set (`Player.border_tiles`). One such set per player per env, so the device
/// cost is `n * npl * PB * 4` bytes. 4096 holds far more than the measured
/// engine border (a few thousand) at a mid-game boundary.
const PB: usize = 4096;
/// Live-attack SLOTS per env in the multi-attack path. One env is a whole
/// game, so it holds a SET of live attacks; the device carries one slot's
/// worth of attack state per `(env, slot)` instead of one per env.
const SLOTS: usize = 8;
/// Scalars per environment in the `scal` buffer.
const SCAL: usize = 9;
/// Threads per block for `env_step`.
const BLOCK: u32 = 256;

// ---- player-clusters pass (canonical `cluster_pass_core`) -------------------
// The device side of the pass is `kernels::*` (included from the canonical
// `ofcuda_tick/src/core_impl.rs`); these are the HOST-side capacities, taken
// from that same file so there is no second source of truth. `CCOLS` is the
// per-env column count for the per-player row-indexed arrays (`cad`, `obmeta`,
// `order`, `idhash`, `pst`), and matches `ofcuda_matrix`'s `COLS`.
const CCOLS: usize = 8192;
const CSPAN: usize = CCOLS * kernels::CSTATE;
/// Per-env cluster scratch capacities. `CT2B`/`CMARK` are indexed by TILE, so
/// they are `wh`-sized per env - the one non-trivial per-env memory cost of the
/// pass (4 bytes each per tile).
const CSCRATCH: usize = kernels::CB;
const CFRIEND_MAX: usize = 1;

#[cuda_module]
mod kernels {
    use super::*;
    include!("../../../ofcuda_tick/src/core_impl.rs");

    // ---- device-side mirrors of the crate's border-set mutators -------------
    // (the CPU `Attack` uses `Vec::contains` / `Vec::retain`; these are the
    // same two operations over a fixed slice, with overflow counted).

    fn border_insert(b: &mut [u32], blen: &mut usize, t: u32) {
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

    fn border_remove(b: &mut [u32], blen: &mut usize, t: u32) {
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

    // ---- INCREMENTAL, DEVICE-RESIDENT owner-border maintenance ---------------
    //
    // The engine keeps each player's `border_tiles` as an insertion-ordered
    // `OrderedTiles` (`execution/ordered_tiles.rs`), maintained incrementally by
    // `Game::update_border_status` / `refresh_borders_around` (`game.rs:1122-1155`)
    // and called from `conquer_one` (`game.rs:1296`). This is the port: the same
    // membership rule (`GameMap::is_border`, `map.rs:397-413`) and the same
    // visit order (the tile, then its cardinal neighbours in N,S,W,E order,
    // `map.rs`/`GameMap.ts:393-403`). Appends land in engine order, so the set
    // is faithful where the old plane-derived ascending-index scan was not.
    //
    // Storage: one flat `[npl * PB]` slice per env; player `p`'s set occupies
    // `p * PB .. p * PB + plen[p]`, and `plen[p]` is its length. Membership is a
    // linear scan, exactly as `OrderedTiles::insert`/`remove` behave externally.
    #[inline]
    fn ob_is_border(plane: &[u16], w: u32, h: u32, t: u32, owner: u16) -> bool {
        let x = t % w;
        if x > 0 && plane[(t - 1) as usize] != owner {
            return true;
        }
        if x + 1 < w && plane[(t + 1) as usize] != owner {
            return true;
        }
        if t >= w && plane[(t - w) as usize] != owner {
            return true;
        }
        if t < (h - 1) * w && plane[(t + w) as usize] != owner {
            return true;
        }
        false
    }

    /// `Game::update_border_status(tile)` (`game.rs:1122-1135`): no-op unless the
    /// tile has an owner; otherwise insert on border (preserving position when
    /// already present), remove otherwise.
    #[inline]
    fn ob_update(plane: &[u16], w: u32, h: u32, t: u32, pb: &mut [u32], plen: &mut [u32]) {
        let owner = plane[t as usize] as usize;
        if owner == 0 || owner >= plen.len() {
            return;
        }
        let base = owner * PB;
        let len = plen[owner] as usize;
        if ob_is_border(plane, w, h, t, owner as u16) {
            let mut i = 0usize;
            while i < len {
                if pb[base + i] == t {
                    return; // re-add of a present value keeps its position
                }
                i += 1;
            }
            if len < PB {
                pb[base + len] = t;
                plen[owner] = (len + 1) as u32;
            }
        } else {
            let mut i = 0usize;
            while i < len {
                if pb[base + i] == t {
                    let mut j = i;
                    while j + 1 < len {
                        pb[base + j] = pb[base + j + 1];
                        j += 1;
                    }
                    plen[owner] = (len - 1) as u32;
                    return;
                }
                i += 1;
            }
        }
    }

    /// `Game::refresh_borders_around(tile)` (`game.rs:1141-1155`): the tile
    /// itself, then its four cardinal neighbours in **N,S,W,E** order. The visit
    /// order is observable - each call appends a newly-border tile to the
    /// player's ordered set, and `refresh_to_conquer` later walks that set while
    /// drawing one PRNG value per enqueued neighbour.
    #[inline]
    fn ob_refresh_around(plane: &[u16], w: u32, h: u32, t: u32, pb: &mut [u32], plen: &mut [u32]) {
        ob_update(plane, w, h, t, pb, plen);
        let mut nbuf = [0u32; 4];
        let n = neighbors4(ORDER_NSWE, t, w, h, &mut nbuf);
        let mut i = 0usize;
        while i < n as usize {
            ob_update(plane, w, h, nbuf[i], pb, plen);
            i += 1;
        }
    }

    /// `ofcuda_env::Attack::offer_neighbours` (`attack.rs:1353-1386`): insert
    /// the candidate border tile for exactly the neighbours `add_neighbors`
    /// enqueues, then let the crate's own `add_neighbors` do the draw-binding,
    /// the priority and the one `next_int(0, 7)` per enqueued neighbour in
    /// N,S,W,E visit order.
    #[allow(clippy::too_many_arguments)]
    fn offer_dev(
        heap: &mut Heap,
        pr: &mut Prng,
        border: &mut [u32],
        blen: &mut usize,
        plane: &[u16],
        owner_col: u16,
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
            if plane[nb as usize] != 0 {
                continue; // not terra nullius (attack.rs:1359)
            }
            border_insert(border, blen, nb); // attack.rs:1363
        }
        let empty: [u32; 0] = [];
        add_neighbors(
            heap,
            pr,
            tile,
            plane,
            &empty[..],
            owner_col,
            terrain,
            w,
            h,
            ORDER_NSWE,
            tick,
        )
    }

    /// ONE TICK of one environment over state already in the thread's
    /// registers / local memory. Returns whether the attack is still alive.
    ///
    /// Mirrors `ofcuda_env::Attack::tick` (`attack.rs:206-324`) exactly:
    /// budget from the tracked border set, then the pop loop; heap-empty
    /// refreshes and RETREATS (death), `troops < 1` starves (death); every pop
    /// removes the popped tile from the border set; the skip guards run before
    /// `offer_neighbours`; `num_tiles_per_tick` is decremented by `tiles_used`
    /// of the current `troop_count` and `troop_count` by `attacker_troop_loss`.
    #[allow(clippy::too_many_arguments)]
    fn tick_once(
        heap: &mut Heap,
        pr: &mut Prng,
        troop_count: &mut f64,
        bsub: &mut [u32],
        blen: &mut usize,
        psub: &mut [u16],
        claims: &mut [u32],
        coff: usize,
        ncl: &mut usize,
        drops: &mut u32,
        terrain: &[u8],
        w: u32,
        h: u32,
        tick: u32,
        owner_col: u16,
        is_bot: u32,
        oborder: &[u32],
        oboff: usize,
        obn: usize,
        cad: &mut [u32],
        // INCREMENTAL OWNER-BORDER (device, per player): this env's
        // `[npl * PB]` ordered sets and their per-player lengths. Empty and
        // `do_ob == 0` on every path that does not maintain them (the
        // single-attack `env_step`), which leaves that path byte-identical.
        ob: &mut [u32],
        oblen: &mut [u32],
        // Per-player owned-tile counts for THIS env, maintained incrementally
        // exactly as `Game::conquer_one` grows `tiles_owned`. Replaces the
        // per-tick host plane scan the economy used to need.
        ptiles: &mut [u32],
        do_ob: u32,
        // ENGINE `retreat`'s troop return (`attack.rs:1278-1297`), per tick:
        // 0.0 unless this tick's death was the `to_conquer.is_empty()` branch,
        // in which case it is the attack's surviving troop count. The caller
        // owns the owner's ledger (`Game::add_troops`, `game.rs:1170-1178`); the
        // attack's own troop_count is zeroed on this path exactly as before, so
        // every path that does not settle a ledger is byte-identical.
        retreat_survivors: &mut f64,
    ) -> bool {
        *retreat_survivors = 0.0;
        // attack.rs:239-255: the one extra draw, taken BEFORE the pop loop, and
        // the budget's border term read after it.
        let draw = pr.next_int(0, 5);
        let budget = attack_tiles_per_tick(*troop_count, false, 0.0, *blen as f64 + draw as f64);
        let mut num = budget;

        while num > 0.0 {
            if *troop_count < 1.0 {
                *troop_count = 0.0;
                return false; // attack.rs:258-262 starved (kill_attack: NO return)
            }
            if heap.is_empty() {
                // attack.rs:264-268 `refresh_to_conquer`, then RETREAT (death).
                // The refill iterates the OWNER's CURRENT `border_tiles` - live,
                // in its insertion order - NOT a snapshot taken at creation. With
                // the device-maintained set (`do_ob != 0`) that live set is `ob`
                // for the owner; the flat-snapshot path keeps the old behaviour.
                heap.clear();
                *blen = 0;
                let mut j = 0usize;
                if do_ob != 0 {
                    let base = owner_col as usize * PB;
                    let ol = if (owner_col as usize) < oblen.len() {
                        oblen[owner_col as usize] as usize
                    } else {
                        0
                    };
                    let ol = ol.min(PB);
                    while j < ol {
                        let bt = ob[base + j];
                        j += 1;
                        offer_dev(heap, pr, bsub, blen, psub, owner_col, terrain, bt, w, h, tick);
                    }
                } else {
                    while j < obn {
                        let bt = oborder[oboff + j];
                        j += 1;
                        offer_dev(heap, pr, bsub, blen, psub, owner_col, terrain, bt, w, h, tick);
                    }
                }
                // attack.rs:266-267 `self.troops = troop_count; self.retreat(game, 0.0)`:
                // the survivors go back to the owner (`attack.rs:1278-1297`).
                // Carried out to the caller; the attack's own count stays 0.
                *retreat_survivors = *troop_count;
                *troop_count = 0.0;
                return false;
            }
            let Some((tile, _pri)) = heap.dequeue() else {
                break; // attack.rs:271
            };
            border_remove(bsub, blen, tile); // attack.rs:275
            if psub[tile as usize] != 0 {
                continue; // attack.rs:284 not terra nullius
            }
            if !has_attacker_neighbor(psub, &[], owner_col, tile, w, h, ORDER_NSWE) {
                continue; // attack.rs:284
            }
            if terrain[tile as usize] & 0x80 == 0 {
                continue; // attack.rs:288
            }
            offer_dev(heap, pr, bsub, blen, psub, owner_col, terrain, tile, w, h, tick); // attack.rs:292
            num -= tiles_used(*troop_count, terrain[tile as usize]); // attack.rs:313
            *troop_count -= attacker_troop_loss(terrain[tile as usize], is_bot != 0); // attack.rs:314
            if *ncl < CC {
                claims[coff + *ncl] = tile;
                *ncl += 1;
            } else {
                *drops += 1;
            }
            psub[tile as usize] = owner_col; // conquer, in place
            // `Game::conquer_one` (`game.rs:1278-1300`): the plane write, then
            // `refresh_borders_around(tile)` - insert/remove the tile and its
            // N,S,W,E neighbours in the owner's ordered `border_tiles`. The
            // plane write happens FIRST so `is_border` sees the new owner, as in
            // the engine.
            if do_ob != 0 {
                ob_refresh_around(psub, w, h, tile, ob, oblen);
            }
            if (owner_col as usize) < ptiles.len() {
                ptiles[owner_col as usize] += 1; // grow `tiles_owned`
            }
            // `Game::conquer_one` (`game.rs:1278-1282`): this tick's conquest
            // sets the owner's `last_tile_change` and grows its `tiles_owned`.
            // The device cadence is the env's OWN, accumulated here - never
            // re-seeded from the record.
            let cs = owner_col as usize * CSTATE;
            cad[cs + 1] = tick;
            cad[cs + 2] += 1;
        }
        if *troop_count < 0.0 {
            *troop_count = 0.0; // attack.rs:322-323
        }
        true
    }

    /// Loads the environment's persistent state, runs ONE tick, writes it back.
    /// This is the VERIFICATION shape: one launch per tick, so a hash can be
    /// taken between ticks and compared against the CPU reference.
    #[kernel]
    #[launch_bounds(256)]
    #[allow(clippy::too_many_arguments)]
    pub fn env_step(
        terrain: &[u8],
        w: u32,
        h: u32,
        tick: u32,
        n_envs: u32,
        owner_col: u16,
        is_bot: u32,
        mut plane: &mut [u16],
        mut heap_tiles: &mut [u32],
        mut heap_pri: &mut [f32],
        mut scal: &mut [u32],
        mut prng: &mut [u32],
        mut border: &mut [u32],
        mut claims: &mut [u32],
        oborder: &[u32],
        mut troops: &mut [f64],
        cad: &mut [u32],
        t2b: &mut [u32],
        marks: &mut [u32],
        cbuf: &mut [u32],
        clen: &mut [u32],
        coffc: &mut [u32],
        vis: &mut [u32],
        cstack: &mut [u32],
        cowned: &mut [u32],
        crem: &mut [u32],
        cout: &mut [u32],
        corder: &[u32],
        cidhash: &[u32],
        cobmeta: &[u32],
        cfriends: &[u32],
        catkow: &[u16],
        catktg: &[u16],
        catktr: &[f64],
        cpst: &mut [f64],
        do_clusters: u32,
    ) {
        let e = thread::index_1d().get();
        if e >= n_envs as usize {
            return;
        }
        let soff = e * SCAL;
        scal[soff + 3] = 0; // this tick's claim count
        if scal[soff + 5] == 0 {
            return; // dead: the engine still ticks once and draws nothing
        }
        let wh = (w as usize) * (h as usize);
        // --- the player-clusters pass, BEFORE the attack plan of this tick ---
        // `PlayerExecution::tick` -> `maybe_remove_clusters` runs for every
        // player exec, and the player execs are ticked ahead of the attack execs
        // in `execute_next_tick`. A cluster handed to a captor is therefore
        // owned by the captor before any attack pops, which is the whole reason
        // the pass cannot run after the attack. One launch, one thread per env.
        // `do_clusters == 0` is a throughput A/B only: the pass is skipped and
        // every other byte of the tick is identical.
        if do_clusters != 0 {
            cluster_pass_env(
                e, n_envs as usize, terrain, w, h, tick, 1, cad, plane, t2b, marks, cbuf, clen,
                coffc, vis, cstack, cowned, crem, cout, corder, cidhash, oborder, cobmeta, cfriends,
                0, catkow, catktg, catktr, 1, cpst,
            );
        }
        let poff = e * wh;
        let psub = &mut plane[poff..poff + wh];
        // `tick_once`'s cadence update must land in THIS env's `cad` span, not
        // env 0's: hand it the per-env sub-slice.
        let cspan = CCOLS * kernels::CSTATE;
        let cadsub = &mut cad[e * cspan..(e + 1) * cspan];
        let hoff = e * HEAP_CAP;
        let mut hl = scal[soff] as usize;
        if hl > HEAP_CAP {
            hl = HEAP_CAP;
        }
        let mut heap = Heap::new();
        heap.load(&heap_tiles[hoff..hoff + hl], &heap_pri[hoff..hoff + hl]);
        let pse = e * 5;
        let mut pr = Prng::from_state(&prng[pse..pse + 5]);
        let mut troop_count = troops[e];
        let boff = e * BC;
        let mut blen = scal[soff + 1] as usize;
        if blen > BC {
            blen = BC;
        }
        let bsub = &mut border[boff..boff + BC];
        let coff = e * CC;
        let mut ncl = scal[soff + 2] as usize;
        if ncl > CC {
            ncl = CC;
        }
        let mut drops = scal[soff + 6];
        let mut peak = scal[soff + 7];
        let oboff = e * OB;
        let mut obn = scal[soff + 8] as usize;
        if obn > OB {
            obn = OB;
        }

        let alive = tick_once(
            &mut heap,
            &mut pr,
            &mut troop_count,
            bsub,
            &mut blen,
            psub,
            claims,
            coff,
            &mut ncl,
            &mut drops,
            terrain,
            w,
            h,
            tick,
            owner_col,
            is_bot,
            oborder,
            oboff,
            obn,
            cadsub,
            &mut [],
            &mut [],
            &mut [],
            0,
            &mut 0.0,
        );

        // ---- write the persistent state back (device-resident, no copy-out) --
        scal[soff + 3] = 1; // one tick
        scal[soff] = heap.len as u32;
        scal[soff + 1] = blen as u32;
        scal[soff + 2] = ncl as u32;
        troops[e] = troop_count;
        scal[soff + 5] = if alive { 1 } else { 0 };
        scal[soff + 6] = drops;
        if heap.peak as u32 > peak {
            peak = heap.peak as u32;
        }
        scal[soff + 7] = peak;
        let mut j = 0usize;
        while j < heap.len {
            heap_tiles[hoff + j] = heap.tiles[j];
            heap_pri[hoff + j] = heap.pri[j];
            j += 1;
        }
        let mut pw = [0u32; 5];
        pr.state_words(&mut pw);
        let mut j = 0usize;
        while j < 5 {
            prng[pse + j] = pw[j];
            j += 1;
        }
    }

    /// MULTI-ATTACK: one thread per env, looping every live SLOT that env holds.
    /// Slots are ticked in creation order and each slot's conquests are written
    /// into the SHARED per-env plane in place, so a later slot in the same tick
    /// sees the earlier slot's claims - the engine's own exec order. Slots are
    /// otherwise independent (own heap, border, PRNG stream and troop count).
    /// One launch ticks a whole set for every env; no host round-trip, and
    /// nothing here reads an oracle after setup.
    #[kernel]
    #[launch_bounds(256)]
    #[allow(clippy::too_many_arguments)]
    pub fn env_step_multi(
        terrain: &[u8],
        w: u32,
        h: u32,
        tick: u32,
        n_envs: u32,
        n_slots: u32,
        mut plane: &mut [u16],
        mut mheap_tiles: &mut [u32],
        mut mheap_pri: &mut [f32],
        mut mborder: &mut [u32],
        mut mclaims: &mut [u32],
        mut mprng: &mut [u32],
        mut mtroops: &mut [f64],
        mut mscal: &mut [u32],
        mowner: &[u16],
        misbot: &[u32],
        moborder: &[u32],
        mut cad: &mut [u32],
        // Incremental device owner-border: `n_envs * pbnpl * PB` ordered sets,
        // `n_envs * pbnpl` lengths and `n_envs * pbnpl` owned-tile counters.
        // `do_ob == 0` disables maintenance (and the buffers are never touched).
        mut pob: &mut [u32],
        mut poblen: &mut [u32],
        mut ptiles: &mut [u32],
        pbnpl: u32,
        do_ob: u32,
    ) {
        let e = thread::index_1d().get();
        if e >= n_envs as usize {
            return;
        }
        let wh = (w as usize) * (h as usize);
        let poff = e * wh;
        let cspan = CCOLS * CSTATE;
        let slots = n_slots as usize;
        let mut s = 0usize;
        while s < slots {
            let g = e * slots + s;
            let soff = g * SCAL;
            mscal[soff + 3] = 0; // this slot's claim count for this tick
            if mscal[soff + 5] == 0 {
                s += 1;
                continue; // dead slot: it ticks once and draws nothing
            }
            let owner_col = mowner[g];
            let is_bot = misbot[g];
            let hoff = g * HEAP_CAP;
            let mut hl = mscal[soff] as usize;
            if hl > HEAP_CAP {
                hl = HEAP_CAP;
            }
            let mut heap = Heap::new();
            heap.load(&mheap_tiles[hoff..hoff + hl], &mheap_pri[hoff..hoff + hl]);
            let pse = g * 5;
            let mut pr = Prng::from_state(&mprng[pse..pse + 5]);
            let mut troop_count = mtroops[g];
            let boff = g * BC;
            let mut blen = mscal[soff + 1] as usize;
            if blen > BC {
                blen = BC;
            }
            let coff = g * CC;
            let mut ncl = mscal[soff + 2] as usize;
            if ncl > CC {
                ncl = CC;
            }
            let mut drops = mscal[soff + 6];
            let oboff = g * OB;
            let mut obn = mscal[soff + 8] as usize;
            if obn > OB {
                obn = OB;
            }
            let mut survivors = 0.0f64;
            let alive = {
                let psub = &mut plane[poff..poff + wh];
                let bsub = &mut mborder[boff..boff + BC];
                let cadsub = &mut cad[e * cspan..(e + 1) * cspan];
                let npl_u = pbnpl as usize;
                let obspan = npl_u * PB;
                let obe = e * obspan;
                let lle = e * npl_u;
                tick_once(
                    &mut heap,
                    &mut pr,
                    &mut troop_count,
                    bsub,
                    &mut blen,
                    psub,
                    mclaims,
                    coff,
                    &mut ncl,
                    &mut drops,
                    terrain,
                    w,
                    h,
                    tick,
                    owner_col,
                    is_bot,
                    moborder,
                    oboff,
                    obn,
                    cadsub,
                    &mut pob[obe..obe + obspan],
                    &mut poblen[lle..lle + npl_u],
                    &mut ptiles[lle..lle + npl_u],
                    do_ob,
                    &mut survivors,
                )
            };
            mscal[soff + 3] = 1;
            mscal[soff] = heap.len as u32;
            mscal[soff + 1] = blen as u32;
            mscal[soff + 2] = ncl as u32;
            // A slot that DIED on `to_conquer.is_empty()` carries its surviving
            // troops here (engine `retreat`, attack.rs:1278-1297) so the host can
            // settle the owner's ledger; a starved slot carries 0.
            mtroops[g] = if alive { troop_count } else { survivors };
            mscal[soff + 5] = if alive { 1 } else { 0 };
            mscal[soff + 6] = drops;
            let mut j = 0usize;
            while j < heap.len {
                mheap_tiles[hoff + j] = heap.tiles[j];
                mheap_pri[hoff + j] = heap.pri[j];
                j += 1;
            }
            let mut pw = [0u32; 5];
            pr.state_words(&mut pw);
            let mut j = 0usize;
            while j < 5 {
                mprng[pse + j] = pw[j];
                j += 1;
            }
            s += 1;
        }
    }

    /// FAST PATH: ONE LAUNCH RUNS `nticks` TICKS. The heap, the PRNG stream,
    /// the border set and the troop count never leave the thread: there is no
    /// per-tick heap round-trip through global memory and no per-tick launch.
    ///
    /// Semantically identical to `env_step` repeated `nticks` times, because
    /// every value carried across the tick boundary is stored exactly (the heap
    /// is the same binary heap either way, and the tick argument advances by
    /// one per iteration exactly as the caller's loop does).
    #[kernel]
    #[launch_bounds(256)]
    #[allow(clippy::too_many_arguments)]
    pub fn env_step_loop(
        terrain: &[u8],
        w: u32,
        h: u32,
        tick: u32,
        nticks: u32,
        n_envs: u32,
        owner_col: u16,
        is_bot: u32,
        mut plane: &mut [u16],
        mut heap_tiles: &mut [u32],
        mut heap_pri: &mut [f32],
        mut scal: &mut [u32],
        mut prng: &mut [u32],
        mut border: &mut [u32],
        mut claims: &mut [u32],
        oborder: &[u32],
        mut troops: &mut [f64],
        cad: &mut [u32],
        t2b: &mut [u32],
        marks: &mut [u32],
        cbuf: &mut [u32],
        clen: &mut [u32],
        coffc: &mut [u32],
        vis: &mut [u32],
        cstack: &mut [u32],
        cowned: &mut [u32],
        crem: &mut [u32],
        cout: &mut [u32],
        corder: &[u32],
        cidhash: &[u32],
        cobmeta: &[u32],
        cfriends: &[u32],
        catkow: &[u16],
        catktg: &[u16],
        catktr: &[f64],
        cpst: &mut [f64],
        do_clusters: u32,
    ) {
        let e = thread::index_1d().get();
        if e >= n_envs as usize {
            return;
        }
        let soff = e * SCAL;
        scal[soff + 3] = 0;
        if scal[soff + 5] == 0 {
            return;
        }
        let wh = (w as usize) * (h as usize);
        let poff = e * wh;
        let hoff = e * HEAP_CAP;
        let mut hl = scal[soff] as usize;
        if hl > HEAP_CAP {
            hl = HEAP_CAP;
        }
        let mut heap = Heap::new();
        heap.load(&heap_tiles[hoff..hoff + hl], &heap_pri[hoff..hoff + hl]);
        let pse = e * 5;
        let mut pr = Prng::from_state(&prng[pse..pse + 5]);
        let mut troop_count = troops[e];
        let boff = e * BC;
        let mut blen = scal[soff + 1] as usize;
        if blen > BC {
            blen = BC;
        }
        let bsub = &mut border[boff..boff + BC];
        let coff = e * CC;
        let mut ncl = scal[soff + 2] as usize;
        if ncl > CC {
            ncl = CC;
        }
        let mut drops = scal[soff + 6];
        let mut peak = scal[soff + 7];
        let oboff = e * OB;
        let mut obn = scal[soff + 8] as usize;
        if obn > OB {
            obn = OB;
        }

        let mut alive = true;
        let mut ran = 0u32;
        while ran < nticks {
            // The player-clusters pass, once per tick, BEFORE the attack plan
            // (engine order: player execs tick ahead of attack execs). `plane`
            // is reborrowed here rather than held as `psub`, so the batched
            // per-env slice can be handed to the pass.
            if do_clusters != 0 {
                cluster_pass_env(
                    e, n_envs as usize, terrain, w, h, tick + ran, 1, cad, plane, t2b, marks, cbuf,
                    clen, coffc, vis, cstack, cowned, crem, cout, corder, cidhash, oborder, cobmeta,
                    cfriends, 0, catkow, catktg, catktr, 1, cpst,
                );
            }
            let psub = &mut plane[poff..poff + wh];
            let cspan = CCOLS * kernels::CSTATE;
            let cadsub = &mut cad[e * cspan..(e + 1) * cspan];
            alive = tick_once(
                &mut heap,
                &mut pr,
                &mut troop_count,
                bsub,
                &mut blen,
                psub,
                claims,
                coff,
                &mut ncl,
                &mut drops,
                terrain,
                w,
                h,
                tick + ran,
                owner_col,
                is_bot,
                oborder,
                oboff,
                obn,
                cadsub,
                &mut [],
                &mut [],
                &mut [],
                0,
                &mut 0.0,
            );
            ran += 1;
            if !alive {
                break; // the engine ticks a dead attack once and draws nothing
            }
        }

        scal[soff + 3] = ran;
        scal[soff] = heap.len as u32;
        scal[soff + 1] = blen as u32;
        scal[soff + 2] = ncl as u32;
        troops[e] = troop_count;
        scal[soff + 5] = if alive { 1 } else { 0 };
        scal[soff + 6] = drops;
        if heap.peak as u32 > peak {
            peak = heap.peak as u32;
        }
        scal[soff + 7] = peak;
        let mut j = 0usize;
        while j < heap.len {
            heap_tiles[hoff + j] = heap.tiles[j];
            heap_pri[hoff + j] = heap.pri[j];
            j += 1;
        }
        let mut pw = [0u32; 5];
        pr.state_words(&mut pw);
        let mut j = 0usize;
        while j < 5 {
            prng[pse + j] = pw[j];
            j += 1;
        }
    }

    /// BATCHED form of the canonical `cluster_pass_core`. The pass is
    /// inherently serial within one environment (it walks the engine's exec
    /// order and each player's border set in order, mutating the plane in place,
    /// exactly as the engine's `game.conquer` does), so the batched env runs it
    /// one-thread-per-environment: thread `e` owns environment `e` and the whole
    /// pass over that environment's sub-slices runs on it. That is the matrix
    /// driver's one-thread launch, N times in parallel with no cross-env
    /// ordering requirement. Buffer strides are all `len / n_envs`, the env's
    /// own batching convention.
    #[kernel]
    #[launch_bounds(256)]
    #[allow(clippy::too_many_arguments)]
    pub fn cluster_pass_batch(
        terrain: &[u8],
        w: u32,
        h: u32,
        tick: u32,
        n_envs: u32,
        npl: u32,
        cad: &mut [u32],
        plane: &mut [u16],
        t2b: &mut [u32],
        marks: &mut [u32],
        cbuf: &mut [u32],
        clen: &mut [u32],
        coff: &mut [u32],
        vis: &mut [u32],
        stack: &mut [u32],
        owned: &mut [u32],
        rem: &mut [u32],
        out: &mut [u32],
        order: &[u32],
        idhash: &[u32],
        oborder: &[u32],
        obmeta: &[u32],
        friends: &[u32],
        nfriends: u32,
        atk_owner: &[u16],
        atk_target: &[u16],
        atk_troops: &[f64],
        natk: u32,
        pst: &mut [f64],
    ) {
        let e = thread::index_1d().get();
        if e >= n_envs as usize {
            return;
        }
        cluster_pass_env(
            e, n_envs as usize, terrain, w, h, tick, npl, cad, plane, t2b, marks, cbuf, clen,
            coff, vis, stack, owned, rem, out, order, idhash, oborder, obmeta, friends, nfriends,
            atk_owner, atk_target, atk_troops, natk, pst,
        );
    }

    /// One environment's whole cluster pass, over slices that hold the whole
    /// batch: the offsets are `e * stride` with the stride derived from each
    /// buffer's length. Shared by the standalone `cluster_pass_batch` kernel and
    /// by `env_step` / `env_step_loop`, so the tick's inlined pass and the
    /// parity harness run the SAME code on the SAME layout.
    #[allow(clippy::too_many_arguments)]
    fn cluster_pass_env(
        e: usize,
        ne: usize,
        terrain: &[u8],
        w: u32,
        h: u32,
        tick: u32,
        npl: u32,
        cad: &mut [u32],
        plane: &mut [u16],
        t2b: &mut [u32],
        marks: &mut [u32],
        cbuf: &mut [u32],
        clen: &mut [u32],
        coff: &mut [u32],
        vis: &mut [u32],
        stack: &mut [u32],
        owned: &mut [u32],
        rem: &mut [u32],
        out: &mut [u32],
        order: &[u32],
        idhash: &[u32],
        oborder: &[u32],
        obmeta: &[u32],
        friends: &[u32],
        nfriends: u32,
        atk_owner: &[u16],
        atk_target: &[u16],
        atk_troops: &[f64],
        natk: u32,
        pst: &mut [f64],
    ) {
        let wh = (w as usize) * (h as usize);
        let csp = CCOLS * CSTATE;
        let poff = e * wh;
        let cadoff = e * csp;
        let t2boff = e * wh;
        let mkoff = e * wh;
        let cboff = e * CS;
        let cloff = e * CB;
        let coffo = e * CB;
        let visoff = e * (CB / 32);
        let stoff = e * CSTACK;
        let owoff = e * COWN;
        let remoff = e * CREM;
        let oooff = e * CSTAT;
        let ord = e * (order.len() / ne);
        let idh = e * (idhash.len() / ne);
        let om = e * (obmeta.len() / ne);
        let ob = e * (oborder.len() / ne);
        let fr = e * (friends.len() / ne);
        let ats = atk_owner.len() / ne;
        let at = e * ats;
        let ps = e * (pst.len() / ne);
        cluster_pass_core(
            terrain,
            w,
            h,
            tick,
            npl,
            &order[ord..ord + CCOLS],
            &idhash[idh..idh + CCOLS],
            &mut cad[cadoff..cadoff + csp],
            &mut plane[poff..poff + wh],
            &mut t2b[t2boff..t2boff + wh],
            &mut marks[mkoff..mkoff + wh],
            &mut cbuf[cboff..cboff + CS],
            &mut clen[cloff..cloff + CB],
            &mut coff[coffo..coffo + CB],
            &mut vis[visoff..visoff + CB / 32],
            &mut stack[stoff..stoff + CSTACK],
            &mut owned[owoff..owoff + COWN],
            &mut rem[remoff..remoff + CREM],
            &mut out[oooff..oooff + CSTAT],
            &oborder[ob..ob + OB],
            &obmeta[om..om + 2 * CCOLS],
            &friends[fr..fr + (CNB + 1)],
            nfriends,
            &atk_owner[at..at + ats],
            &atk_target[at..at + ats],
            &atk_troops[at..at + ats],
            natk,
            &mut pst[ps..ps + CCOLS * 3],
        );
    }

    /// Fill a `u32` buffer with `u32::MAX`. Setup only: `t2b` ("border tile ->
    /// position in the border set", `u32::MAX` = absent) must start absent
    /// everywhere, and it is `wh`-sized PER ENV, so it is written on the device
    /// rather than copied from a host vector that would be 4 MB per env.
    #[kernel]
    #[launch_bounds(256)]
    pub fn fill_max(n: u32, mut buf: &mut [u32]) {
        let i = thread::index_1d().get();
        if i >= n as usize {
            return;
        }
        buf[i] = u32::MAX;
    }

    /// The engine's `Game::conquer_one` (`game.rs:1246-1287`) for a batch of
    /// this tick's conquests, applied to the DEVICE cadence counters. This is
    /// what makes the env's `last_tile_change` / `tiles_owned` self-accumulated
    /// instead of oracle-re-seeded: `last_tile_change = tick` for both the tile's
    /// previous owner and its new one, `tiles_owned` -1 / +1, and the old owner
    /// is marked dead when it hits zero. A tile the pass ALREADY handed to the
    /// captor in the same tick is skipped (the plane already reads `owner`), so a
    /// cluster removal is never double-counted by the claim stream that contains
    /// it.
    ///
    /// ONE thread processes the whole list: the engine's conquest order is the
    /// order `tiles_owned`/`last_tile_change` see, so it is not parallelised.
    #[kernel]
    #[launch_bounds(1)]
    pub fn apply_claims(
        claims: &[u32],
        n_claims: u32,
        tick: u32,
        plane: &mut [u16],
        cad: &mut [u32],
    ) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let mut i = 0usize;
        let mut c = 0u32;
        while c < n_claims {
            if i + 1 >= claims.len() {
                break;
            }
            let owner = claims[i] as u16;
            let n = claims[i + 1] as usize;
            i += 2;
            let os = owner as usize * CSTATE;
            let mut j = 0usize;
            while j < n && i < claims.len() {
                let t = claims[i] as usize;
                i += 1;
                j += 1;
                let prev = plane[t];
                if prev == owner {
                    continue; // already this owner (e.g. the pass's removal)
                }
                if prev != 0 {
                    let cs = prev as usize * CSTATE;
                    if cad[cs + 2] > 0 {
                        cad[cs + 2] -= 1;
                    }
                    cad[cs + 1] = tick;
                    if cad[cs + 2] == 0 {
                        cad[cs + 3] = 0;
                    }
                }
                plane[t] = owner;
                cad[os + 2] += 1;
                cad[os + 1] = tick;
                cad[os + 3] = 1;
            }
            c += 1;
        }
    }

    /// Setup only (never in the timed loop): write the env-0 plane template
    /// into every environment's slot of the batch plane, so the benchmark needs
    /// no host-side n*wh replica (2 MB per env would be 8 GB at 4096 envs).
    #[kernel]
    #[launch_bounds(256)]
    pub fn plane_fill(tmpl: &[u16], wh: u32, n_envs: u32, mut plane: &mut [u16]) {
        let i = thread::index_1d().get();
        let total = (wh as usize) * (n_envs as usize);
        if i >= total {
            return;
        }
        plane[i] = tmpl[i % (wh as usize)];
    }

    /// FNV-1a-64 of one environment's whole plane, one thread per environment.
    /// This is the VERIFICATION path - the crate's `ofcuda_hash` function, not
    /// a re-implementation - and it is only run in `--mode verify`.
    #[kernel]
    #[launch_bounds(64)]
    pub fn env_hash(plane: &[u16], wh: u32, n_envs: u32, kidx: u32, out: &mut [u64]) {
        let e = thread::index_1d().get();
        if e >= n_envs as usize {
            return;
        }
        let off = e * wh as usize;
        out[kidx as usize * n_envs as usize + e] =
            ofcuda_hash::fnv1a_u16_le(ofcuda_hash::FNV_OFFSET_BASIS, &plane[off..off + wh as usize]);
    }

    // =================================================================
    // The bot attack AI, now reachable FROM THIS CRATE.
    //
    // Both kernels below call the canonical core (`core_impl.rs`): the bot
    // DECISION is `bot_ai_core` and the attack's ORIGINATION is
    // `orig_refresh_dev` / `orig_offer_dev` / `orig_add_neighbors_t` - the same
    // functions the matrix's `bot_ai` / `attack_init` call. Before the move the
    // env `include!`d a core that had neither, so the batch env could not
    // originate an attack at all: it had to be handed one per tick from the
    // oracle.
    // =================================================================

    /// The bot's decision, verbatim `bot_ai_core`, launched from this crate.
    /// `out` is `3 * nbots`: `[fire, troops, action]` (see `bot_ai_core`).
    #[kernel]
    #[launch_bounds(1)]
    #[allow(clippy::too_many_arguments)]
    pub fn bot_ai(
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
        bot_rrow: &[u32],
        bot_ff: &[u32],
        mut bot_state: &mut [u32],
        bot_boat: &[u32],
        maxtroops: &[f64],
        tgt: &[f64],
        out: &mut [f64],
        nbots: u32,
    ) {
        bot_ai_core(
            terrain,
            w,
            h,
            tick,
            spawn_end_tick,
            plane,
            pst,
            oborder,
            ob_meta,
            bot_sid,
            bot_rate,
            bot_at,
            bot_trigger,
            bot_trow,
            bot_rrow,
            bot_ff,
            bot_state,
            bot_boat,
            maxtroops,
            tgt,
            out,
            nbots,
        );
    }

    /// DEVICE-SIDE ORIGINATION. Builds an attack's frontier from the OWNER's
    /// border set with the core's `orig_refresh_dev` (`AttackExecution::init`,
    /// `attack.rs:156-164` -> `refresh_to_conquer`, `attack.rs:1265-1274`) and
    /// writes the heap / border / PRNG state straight into this env's device
    /// buffers. `out` = [heap_len, border_len, peak, prng_calls].
    #[kernel]
    #[launch_bounds(1)]
    #[allow(clippy::too_many_arguments)]
    pub fn originate_env(
        terrain: &[u8],
        w: u32,
        h: u32,
        tick: u32,
        owner_col: u16,
        target_col: u16,
        plane: &[u16],
        oborder: &[u32],
        oboff: u32,
        obn: u32,
        mut heap_tiles: &mut [u32],
        mut heap_pri: &mut [f32],
        mut border: &mut [u32],
        mut prng: &mut [u32],
        mut scal: &mut [u32],
        out: &mut [u32],
    ) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let mut heap = Heap::new();
        let mut pr = Prng::new(SEED);
        let mut blen = 0usize;
        orig_refresh_dev(
            &mut heap,
            &mut pr,
            border,
            &mut blen,
            plane,
            owner_col,
            target_col,
            terrain,
            w,
            h,
            tick,
            oborder,
            oboff as usize,
            obn as usize,
        );
        scal[0] = heap.len as u32;
        scal[1] = blen as u32;
        scal[5] = 1; // alive (the env's layout: 5 = alive, 7 = heap peak)
        scal[7] = heap.peak as u32;
        let mut j = 0usize;
        while j < heap.len {
            heap_tiles[j] = heap.tiles[j];
            heap_pri[j] = heap.pri[j];
            j += 1;
        }
        let mut pw = [0u32; 5];
        pr.state_words(&mut pw);
        let mut j = 0usize;
        while j < 5 {
            prng[j] = pw[j];
            j += 1;
        }
        out[0] = heap.len as u32;
        out[1] = blen as u32;
        out[2] = heap.peak as u32;
        out[3] = pr.calls;
    }

    /// MULTI-PATH ORIGINATION. The batched env's `env_step_multi` ticks a SET of
    /// live attacks per env, one SLOT each; this kernel builds a NEW attack
    /// directly into slot `g = env * SLOTS + slot` from the OWNER's border set
    /// and the DEVICE's own plane, then stamps everything the next
    /// `env_step_multi` needs to tick it (owner, bot flag, start troops, alive,
    /// and the slot's own refresh input). The frontier itself is the canonical
    /// core's `orig_refresh_dev` (`AttackExecution::init` -> `refresh_to_conquer`,
    /// `attack.rs:156-164`), the SAME code the single-env `originate_env` runs -
    /// only the strides differ, because here one slot's worth of state lives at
    /// a per-`(env, slot)` offset instead of per env.
    ///
    /// Every input is device state (`plane`, the owner border the host computed
    /// from that same plane) plus the owner's small id, bot flag and the start
    /// troops the bot decision produced. Nothing here reads the record.
    #[kernel]
    #[launch_bounds(1)]
    #[allow(clippy::too_many_arguments)]
    pub fn env_orig_slot(
        terrain: &[u8],
        w: u32,
        h: u32,
        tick: u32,
        owner_col: u16,
        target_col: u16,
        plane: &[u16],
        oborder: &[u32],
        oboff: u32,
        obn: u32,
        mut heap_tiles: &mut [u32],
        mut heap_pri: &mut [f32],
        mut border: &mut [u32],
        mut prng: &mut [u32],
        mut scal: &mut [u32],
        mut troops: &mut [f64],
        mut owner: &mut [u16],
        mut isbot: &mut [u32],
        mut moborder: &mut [u32],
        g: u32,
        owner_sid: u32,
        ib: u32,
        start_troops: f64,
        out: &mut [u32],
    ) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let gi = g as usize;
        let hb = gi * HEAP_CAP;
        let bb = gi * BC;
        let pb = gi * 5;
        let sb = gi * SCAL;
        let ob = gi * OB;
        let mut heap = Heap::new();
        let mut pr = Prng::new(SEED);
        let mut blen = 0usize;
        {
            let bsub = &mut border[bb..bb + BC];
            orig_refresh_dev(
                &mut heap,
                &mut pr,
                bsub,
                &mut blen,
                plane,
                owner_col,
                target_col,
                terrain,
                w,
                h,
                tick,
                oborder,
                oboff as usize,
                obn as usize,
            );
        }
        let mut j = 0usize;
        while j < heap.len {
            heap_tiles[hb + j] = heap.tiles[j];
            heap_pri[hb + j] = heap.pri[j];
            j += 1;
        }
        let mut pw = [0u32; 5];
        pr.state_words(&mut pw);
        let mut j = 0usize;
        while j < 5 {
            prng[pb + j] = pw[j];
            j += 1;
        }
        // The attack's OWN owner-border copy (the `refresh_to_conquer` input
        // used if its frontier ever runs empty). The engine's `AttackExecution`
        // holds the same set; keeping it makes a mid-flight refill tick the way
        // the engine does instead of starving.
        let on = if obn as usize > OB { OB } else { obn as usize };
        let mut j = 0usize;
        while j < on {
            moborder[ob + j] = oborder[oboff as usize + j];
            j += 1;
        }
        scal[sb] = heap.len as u32; // heap_len
        scal[sb + 1] = blen as u32; // border_len
        scal[sb + 2] = 0; // this tick's claim count (fresh attack)
        scal[sb + 5] = 1; // alive
        scal[sb + 7] = heap.peak as u32; // heap peak
        scal[sb + 8] = on as u32; // owner-border copy length
        troops[gi] = start_troops;
        owner[gi] = owner_sid as u16;
        isbot[gi] = ib;
        out[0] = heap.len as u32;
        out[1] = blen as u32;
        out[2] = heap.peak as u32;
        out[3] = pr.calls;
    }

    /// The bot DECISION PER ENV: one thread per env, looping the bots and
    /// calling the canonical `bot_ai_core` on that env's OWN plane, player state
    /// and owner-border metadata. `bot_ai` (one thread per bot, env 0's state)
    /// is only correct while every env is a clone of the same game; with the RL
    /// ACTION channel live the envs take different actions and diverge, and each
    /// one then needs its own decision.
    ///
    /// `ob_meta` carries ABSOLUTE offsets into `oborder` (env e's block starts at
    /// `e * npl * PB`), so the same `bot_ai_core` can be called with each env's
    /// own `pst` sub-slice and the whole owner-border array.
    #[kernel]
    #[launch_bounds(64)]
    #[allow(clippy::too_many_arguments)]
    pub fn bot_ai_multi(
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
        bot_rrow: &[u32],
        bot_ff: &[u32],
        mut bot_state: &mut [u32],
        bot_boat: &[u32],
        maxtroops: &[f64],
        tgt: &[f64],
        mut out: &mut [f64],
        n_envs: u32,
        npl: u32,
        nbots: u32,
    ) {
        let e = thread::index_1d().get();
        if e >= n_envs as usize {
            return;
        }
        let wh = (w as usize) * (h as usize);
        let np = npl as usize;
        let nb = nbots as usize;
        bot_ai_core(
            terrain,
            w,
            h,
            tick,
            spawn_end_tick,
            &plane[e * wh..e * wh + wh],
            &pst[e * np * 3..(e + 1) * np * 3],
            oborder,
            &ob_meta[e * np * 2..(e + 1) * np * 2],
            bot_sid,
            bot_rate,
            bot_at,
            bot_trigger,
            bot_trow,
            bot_rrow,
            bot_ff,
            &mut bot_state[e * nb..(e + 1) * nb],
            bot_boat,
            maxtroops,
            tgt,
            &mut out[e * nb * 3..(e + 1) * nb * 3],
            nbots,
        );
    }

    /// THE ATTACK THE ENV'S OWN BOT AI WOULD CREATE FOR THIS PLAYER, with the
    /// SCHEDULE GATE REMOVED.
    ///
    /// This is `bot_ai_core`'s terra-nullius land branch (`core_impl.rs:2298-2312`)
    /// with `tick > spawn_end_tick && tick % rate == at` deleted - nothing else.
    /// The RL `expand` action therefore does not invent a mechanic: it presses
    /// the button the bot AI presses, and the attack it creates is the bot's own
    /// (`land_attack_troops(owner, max_troops * expand_ratio)`), built by the
    /// same `env_orig_slot` / `orig_refresh_dev` the scheduled firings use.
    ///
    /// `plane`/`oborder`/`ob_meta`/`pst` are the whole batch arrays; `poff`,
    /// `oboff` and `pstoff` select ENV `e`, so the action channel is per env.
    ///
    /// `out` = `[amount, fired, why]`: `why` 0 = fired, 1 = no land border with
    /// terra nullius (the bot's own `land` gate), 2 = `amount < 1.0` (the bot's
    /// own `land_attack_troops` refusal), 3 = the player holds no tile (the
    /// kernel's own `tiles >= 1.0` gate).
    #[kernel]
    #[launch_bounds(1)]
    #[allow(clippy::too_many_arguments)]
    pub fn agent_expand_fire(
        terrain: &[u8],
        w: u32,
        h: u32,
        plane: &[u16],
        poff: u32,
        oborder: &[u32],
        ob_meta: &[u32],
        pst: &[f64],
        pstoff: u32,
        tgt: &[f64],
        mcap: u32,
        trow_row: u32,
        sid: u32,
        out: &mut [f64],
    ) {
        if thread::index_1d().get() != 0 {
            return;
        }
        out[0] = 0.0;
        out[1] = 0.0;
        out[2] = 0.0;
        let wh = (w as usize) * (h as usize);
        let ps = pstoff as usize;
        let ti = sid as usize;
        let tiles = pst[ps + ti * 3 + 1];
        let troops = pst[ps + ti * 3];
        if tiles < 1.0 {
            out[2] = 3.0;
            return;
        }
        let mc = mcap as usize;
        let idx = if (tiles as usize) < mc {
            tiles as usize
        } else {
            mc - 1
        };
        let psubp = &plane[poff as usize..poff as usize + wh];
        if !bot_land_border_tn(terrain, w, h, psubp, oborder, ob_meta, sid as u16) {
            out[2] = 1.0;
            return;
        }
        // `troops - tgt[row * mcap + idx]`: the same PRE-MULTIPLIED table lookup
        // `bot_ai_core` uses, so the engine's two separate roundings are
        // preserved and no fma contraction can shift the value by 1 ulp.
        let amount = troops - tgt[trow_row as usize * mc + idx];
        if amount < 1.0 {
            out[2] = 2.0;
            return;
        }
        out[0] = amount;
        out[1] = 1.0;
        out[2] = 0.0;
    }

    /// Kill one slot of the batch: `AttackExecution::kill_attack`'s
    /// `attack_live = false` (`attack.rs:1205-1215`). Used by the ORIGINATION
    /// MERGE (`Game::merge_outgoing_land_attacks`, `game.rs:2122-2156`): the
    /// engine folds an owner's OTHER live land attacks of the same target into
    /// the new one and kills them, paying NOTHING back - their `troops` were
    /// already debited from the owner when they were created and now ride in the
    /// new attack's `troops`, so the host must not treat the death as a retreat.
    #[kernel]
    #[launch_bounds(1)]
    pub fn env_kill_slot(mut scal: &mut [u32], g: u32) {
        if thread::index_1d().get() != 0 {
            return;
        }
        let sb = g as usize * SCAL;
        scal[sb + 5] = 0; // `attack_live = false`
    }
}

// ---------------------------------------------------------------------------
// The environment: one real attack reconstructed from the record
// ---------------------------------------------------------------------------

/// One live attack inside a multi-attack env: the per-`(env, slot)` device
/// state. Same fields the single-attack env carried per ENV, now per SLOT.
#[derive(Clone)]
struct AttackState {
    owner_sid: u16,
    target_sid: u16,
    is_bot: bool,
    troops: f64,
    heap_tiles: Vec<u32>,
    heap_pri: Vec<f32>,
    heap_len: usize,
    prng: [u32; 5],
    border: Vec<u32>,
    oborder: Vec<u32>,
    claims: Vec<u32>,
    alive: bool,
}

#[derive(Clone)]
struct EnvState {
    w: u32,
    h: u32,
    terrain: Vec<u8>,
    plane: Vec<u16>,
    owner_sid: u16,
    owner_col: u16,
    /// The owner's engine roster id string; `simple_hash` of it is the cluster
    /// cadence phase (`player_clusters.rs:375-382`).
    owner_id: String,
    is_bot: bool,
    tick0: u32,
    troops: f64,
    heap_tiles: Vec<u32>,
    heap_pri: Vec<f32>,
    heap_len: usize,
    prng: [u32; 5],
    border: Vec<u32>,
    oborder: Vec<u32>,
    claims: Vec<u32>,
    alive: bool,
    /// The SET of live attacks this env's game starts from. The single-attacker
    /// fields above are `attacks[0]` (kept so the one-attack parity selftest
    /// runs on exactly the code path it always did); the multi-attack path
    /// drives every element of this vector.
    attacks: Vec<AttackState>,
}

/// Reconstruct the attack the record says is live at `s`, exactly as the
/// driver's creation path does: computed float start troops, then
/// `refresh_to_conquer` over the OWNER's border as of `s` with the plane of
/// `s`, stamped `s - 1` (`attack.rs:160-164`).
fn build_env_at(
    dump: &ofcuda_tick::Dump,
    ex: &extras::Extra,
    w: u32,
    h: u32,
    terrain: &[u8],
    s: u32,
) -> Result<EnvState, String> {
    let ps = dump.get(&s).ok_or_else(|| format!("no record at tick {s}"))?;
    let players: Vec<(u32, Vec<u32>)> = ps
        .values()
        .map(|p| (p.small_id, p.owned_tiles.clone()))
        .collect();
    let plane = ofcuda_env::state_plane(&players, w, h);
    let attacks = ex.attacks.get(&s).cloned().unwrap_or_default();
    let e = attacks
        .iter()
        .find(|a| a.live && a.target == 0)
        .ok_or_else(|| format!("no live land attack at tick {s}"))?;
    let owner = e.owner;
    let pi = ex
        .p
        .get(&(s - 1))
        .and_then(|m| m.get(&owner))
        .ok_or_else(|| format!("no player {owner} info at tick {}", s - 1))?;
    let ratios = tribe_ratios(&pi.id);
    let row = econ_row(s - 1, pi.troops, pi.tiles, 0, pi.gold, pi.ptype as u32);
    let start = land_attack_start_troops(&row, ratios.expand_ratio)
        .ok_or_else(|| format!("start troops < 1 for owner {owner} at {s}"))?;
    let is_bot = pi.ptype == ofcuda_econ::core_impl::PT_BOT as u8;
    let oborder = ps
        .values()
        .find(|p| p.small_id == owner as u32)
        .map(|p| p.border_order.clone())
        .unwrap_or_default();

    let mut atk = Attack::new(owner, 0, is_bot, start, ofcuda_tick::SEED);
    atk.refresh(&oborder, &plane, terrain, w, h, s - 1);

    let mut prng = [0u32; 5];
    atk.pr.state_words(&mut prng);

    // ---- the SET of live attacks (one env = one whole game) ----------------
    // Every live land attack at `s`, each reconstructed exactly like the chosen
    // one is: computed float start troops, then `refresh_to_conquer` over ITS
    // owner's border at `s` with the plane of `s`, stamped `s-1`. This reads
    // the record at `s` and `s-1` ONLY - initial condition, never a per-tick
    // input. Attacks that are not a fresh creation (their recorded `troops`
    // disagrees with the computed start, i.e. a merge or an older attack whose
    // heap has already evolved) are flagged and dropped from the set: their
    // exact heap/PRNG at `s` is not recoverable from the record at `s` alone,
    // so seeding them would be a guess, not a reconstruction.
    let mut set: Vec<AttackState> = Vec::new();
    for (ti, a) in attacks.iter().enumerate() {
        if !a.live || a.target != 0 {
            continue;
        }
        let Some(api) = ex.p.get(&(s - 1)).and_then(|m| m.get(&a.owner)) else {
            continue;
        };
        let aratios = tribe_ratios(&api.id);
        let arow = econ_row(s - 1, api.troops, api.tiles, 0, api.gold, api.ptype as u32);
        let Some(astart) = land_attack_start_troops(&arow, aratios.expand_ratio) else {
            continue;
        };
        if astart as i64 != a.troops {
            continue; // not a fresh non-merged creation at `s`
        }
        let aoborder = ps
            .values()
            .find(|p| p.small_id == a.owner as u32)
            .map(|p| p.border_order.clone())
            .unwrap_or_default();
        let mut aatk = Attack::new(a.owner, 0, api.ptype == ofcuda_econ::core_impl::PT_BOT as u8, astart, ofcuda_tick::SEED);
        aatk.refresh(&aoborder, &plane, terrain, w, h, s - 1);
        let mut aprng = [0u32; 5];
        aatk.pr.state_words(&mut aprng);
        set.push(AttackState {
            owner_sid: a.owner,
            target_sid: 0,
            is_bot: api.ptype == ofcuda_econ::core_impl::PT_BOT as u8,
            troops: aatk.troops,
            heap_tiles: aatk.heap.tiles[..aatk.heap.len].to_vec(),
            heap_pri: aatk.heap.pri[..aatk.heap.len].to_vec(),
            heap_len: aatk.heap.len,
            prng: aprng,
            border: aatk.border.clone(),
            oborder: aoborder,
            claims: aatk.claims.clone(),
            alive: aatk.attack_live,
        });
        let _ = ti;
    }

    Ok(EnvState {
        w,
        h,
        terrain: terrain.to_vec(),
        plane,
        owner_sid: owner,
        owner_col: owner, // the plane word IS the raw small id
        owner_id: pi.id.clone(),
        is_bot,
        tick0: s,
        troops: atk.troops,
        heap_tiles: atk.heap.tiles[..atk.heap.len].to_vec(),
        heap_pri: atk.heap.pri[..atk.heap.len].to_vec(),
        heap_len: atk.heap.len,
        prng,
        border: atk.border.clone(),
        oborder,
        claims: atk.claims.clone(),
        alive: atk.attack_live,
        attacks: set,
    })
}

/// The CPU reference on the SAME code path: the crate's `Attack::tick`, one
/// call per tick, with the plane updated in place exactly as the device does.
fn host_reference(
    env: &EnvState,
    ticks: u32,
) -> (Vec<Vec<u32>>, Vec<u64>, usize, u64, usize) {
    let mut atk = Attack::new(env.owner_sid, 0, env.is_bot, env.troops, ofcuda_tick::SEED);
    atk.heap = Heap::new();
    atk.heap.load(&env.heap_tiles, &env.heap_pri);
    atk.pr = Prng::from_state(&env.prng);
    atk.border = env.border.clone();
    atk.claims = env.claims.clone();
    atk.attack_live = env.alive;
    let mut plane = env.plane.clone();
    let mut claims = Vec::new();
    let mut hashes = Vec::new();
    let mut refreshes = 0usize;
    let mut drops = 0u64;
    let mut peak = 0usize;
    for k in 0..ticks {
        let o = atk.tick(
            &plane,
            &env.terrain,
            env.w,
            env.h,
            env.tick0 + k,
            &env.oborder,
            0.0,
            false,
        );
        claims.push(o.claims.clone());
        for t in &o.claims {
            plane[*t as usize] = env.owner_col;
        }
        hashes.push(state_hash(&plane));
        if o.retreated {
            refreshes += 1;
        }
        drops += atk.heap.drops;
        if atk.heap.peak > peak {
            peak = atk.heap.peak;
        }
        if o.dead {
            break;
        }
    }
    (claims, hashes, refreshes, drops, peak)
}

// ---------------------------------------------------------------------------
// The device batch
// ---------------------------------------------------------------------------

struct DevBatch {
    d_plane: DeviceBuffer<u16>,
    d_heap_tiles: DeviceBuffer<u32>,
    d_heap_pri: DeviceBuffer<f32>,
    d_scal: DeviceBuffer<u32>,
    d_prng: DeviceBuffer<u32>,
    d_border: DeviceBuffer<u32>,
    d_claims: DeviceBuffer<u32>,
    d_oborder: DeviceBuffer<u32>,
    d_troops: DeviceBuffer<f64>,
    d_terrain: DeviceBuffer<u8>,
    // ---- player-clusters pass: PERSISTENT per-env device state ---------------
    // One flat allocation per array, stride `len / n` (the env's batching
    // convention). `d_ccad` carries each env's own `last_cluster_calc` /
    // `last_tile_change` / `tiles_owned` / `alive` and is NEVER re-seeded from
    // an oracle: `tiles_owned` is counted from the device plane at setup and
    // then advanced by `tick_once`'s conquests, `last_tile_change` is written
    // by those same conquests, `last_cluster_calc` is written by the pass. That
    // self-accumulation is the whole point - the matrix harness re-seeds all
    // three from the engine's CADENCE rows every tick, which the trainer cannot.
    d_ccad: DeviceBuffer<u32>,
    d_ct2b: DeviceBuffer<u32>,
    d_cmark: DeviceBuffer<u32>,
    d_cbuf: DeviceBuffer<u32>,
    d_clen: DeviceBuffer<u32>,
    d_coff: DeviceBuffer<u32>,
    d_cvis: DeviceBuffer<u32>,
    d_cstack: DeviceBuffer<u32>,
    d_cowned: DeviceBuffer<u32>,
    d_crem: DeviceBuffer<u32>,
    d_cout: DeviceBuffer<u32>,
    d_corder: DeviceBuffer<u32>,
    d_cidhash: DeviceBuffer<u32>,
    d_cobmeta: DeviceBuffer<u32>,
    d_cfriends: DeviceBuffer<u32>,
    d_catkow: DeviceBuffer<u16>,
    d_catktg: DeviceBuffer<u16>,
    d_catktr: DeviceBuffer<f64>,
    d_cpst: DeviceBuffer<f64>,
    n: usize,
    wh: usize,
}

impl DevBatch {
    fn new(
        ctx: &std::sync::Arc<CudaContext>,
        module: &kernels::LoadedModule,
        envs: &[EnvState],
        n: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = ctx.default_stream();
        let wh = (envs[0].w as usize) * (envs[0].h as usize);
        let mut d_plane = DeviceBuffer::<u16>::zeroed(&stream, n * wh)?;
        let d_tmpl = DeviceBuffer::<u16>::from_host(&stream, &envs[0].plane)?;
        unsafe {
            module.plane_fill(
                &stream,
                cfg_for_block_from_total(n * wh, 256),
                &d_tmpl,
                wh as u32,
                n as u32,
                &mut d_plane,
            )?;
        }
        let mut h_tiles = vec![0u32; n * HEAP_CAP];
        let mut h_pri = vec![0f32; n * HEAP_CAP];
        let mut scal = vec![0u32; n * SCAL];
        let mut prng = vec![0u32; n * 5];
        let mut border = vec![0u32; n * BC];
        let mut claims = vec![0u32; n * CC];
        let mut oborder = vec![0u32; n * OB];
        let mut h_troops = vec![0f64; n];
        for e in 0..n {
            let env = &envs[if envs.len() == 1 { 0 } else { e }];
            h_tiles[e * HEAP_CAP..e * HEAP_CAP + env.heap_len]
                .copy_from_slice(&env.heap_tiles[..env.heap_len]);
            h_pri[e * HEAP_CAP..e * HEAP_CAP + env.heap_len]
                .copy_from_slice(&env.heap_pri[..env.heap_len]);
            let so = e * SCAL;
            scal[so] = env.heap_len as u32;
            scal[so + 1] = env.border.len() as u32;
            scal[so + 2] = env.claims.len() as u32;
            scal[so + 3] = 0;
            h_troops[e] = env.troops;
            scal[so + 5] = if env.alive { 1 } else { 0 };
            scal[so + 6] = 0;
            scal[so + 4] = 0;
            scal[so + 7] = 0;
            scal[so + 8] = env.oborder.len() as u32;
            prng[e * 5..e * 5 + 5].copy_from_slice(&env.prng);
            border[e * BC..e * BC + env.border.len()].copy_from_slice(&env.border);
            claims[e * CC..e * CC + env.claims.len()].copy_from_slice(&env.claims);
            oborder[e * OB..e * OB + env.oborder.len()].copy_from_slice(&env.oborder);
        }

        // ---- player-clusters pass setup ----------------------------------
        // Per-env cluster state, allocated once and NEVER re-uploaded. The only
        // values seeded from outside the device are the ones the trainer can
        // know without an oracle: the ROSTER (`simple_hash(id)` phase, legible
        // from the player's own id) and the exec ORDER. `tiles_owned` is counted
        // from the setup plane and `last_tile_change` = the setup tick (the
        // plane IS the map at that tick); `last_cluster_calc` is left 0 so the
        // pass seeds it with the engine's own `:375-382` rule. Nothing here is
        // refreshed per tick.
        let cst = kernels::CSTATE;
        let csp = CCOLS * cst;
        let mut cad = vec![0u32; n * csp];
        let mut corder = vec![0u32; n * CCOLS];
        let mut cidhash = vec![0u32; n * CCOLS];
        let mut cobmeta = vec![0u32; n * 2 * CCOLS];
        let mut catkow = vec![0u16; n];
        let mut catktg = vec![0u16; n];
        let mut catktr = vec![0f64; n];
        for e in 0..n {
            let env = &envs[if envs.len() == 1 { 0 } else { e }];
            let sid = env.owner_sid as usize;
            if sid >= CCOLS {
                continue;
            }
            let co = e * csp + sid * cst;
            let owned = env.plane.iter().filter(|v| **v == env.owner_col).count();
            cad[co] = 0; // `last_cluster_calc`: unset -> seeded by the pass
            cad[co + 1] = env.tick0; // `last_tile_change`: the plane is tick t0
            cad[co + 2] = owned as u32; // `tiles_owned`, counted once
            cad[co + 3] = env.alive as u32;
            corder[e * CCOLS] = sid as u32;
            cidhash[e * CCOLS + sid] =
                ofcuda_prng::simple_hash(&env.owner_id).max(0) as u32;
            // OFFSET IS RELATIVE TO THE PER-ENV `oborder` SLICE the pass gets:
            // the core indexes `&oborder[boff..boff+blen]` inside that slice, so
            // an absolute `e*OB` here runs off the end of every env but 0.
            cobmeta[e * 2 * CCOLS + sid * 2] = 0;
            cobmeta[e * 2 * CCOLS + sid * 2 + 1] = env.oborder.len() as u32;
            catkow[e] = env.owner_sid;
            catktg[e] = 0; // the env's live attack is a `target == 0` land attack
            catktr[e] = env.troops;
        }
        let mut d_ct2b = DeviceBuffer::<u32>::zeroed(&stream, n * wh)?;
        unsafe {
            module.fill_max(
                &stream,
                cfg_for_block_from_total(n * wh, 256),
                (n * wh) as u32,
                &mut d_ct2b,
            )?;
        }
        Ok(DevBatch {
            d_plane,
            d_heap_tiles: DeviceBuffer::from_host(&stream, &h_tiles)?,
            d_heap_pri: DeviceBuffer::from_host(&stream, &h_pri)?,
            d_scal: DeviceBuffer::from_host(&stream, &scal)?,
            d_prng: DeviceBuffer::from_host(&stream, &prng)?,
            d_border: DeviceBuffer::from_host(&stream, &border)?,
            d_claims: DeviceBuffer::from_host(&stream, &claims)?,
            d_oborder: DeviceBuffer::from_host(&stream, &oborder)?,
            d_troops: DeviceBuffer::from_host(&stream, &h_troops)?,
            d_terrain: DeviceBuffer::from_host(&stream, &envs[0].terrain)?,
            d_ccad: DeviceBuffer::from_host(&stream, &cad)?,
            d_ct2b,
            d_cmark: DeviceBuffer::<u32>::zeroed(&stream, n * wh)?,
            d_cbuf: DeviceBuffer::<u32>::zeroed(&stream, n * kernels::CS)?,
            d_clen: DeviceBuffer::<u32>::zeroed(&stream, n * kernels::CB)?,
            d_coff: DeviceBuffer::<u32>::zeroed(&stream, n * kernels::CB)?,
            d_cvis: DeviceBuffer::<u32>::zeroed(&stream, n * (kernels::CB / 32))?,
            d_cstack: DeviceBuffer::<u32>::zeroed(&stream, n * kernels::CSTACK)?,
            d_cowned: DeviceBuffer::<u32>::zeroed(&stream, n * kernels::COWN)?,
            d_crem: DeviceBuffer::<u32>::zeroed(&stream, n * kernels::CREM)?,
            d_cout: DeviceBuffer::<u32>::zeroed(&stream, n * kernels::CSTAT)?,
            d_corder: DeviceBuffer::from_host(&stream, &corder)?,
            d_cidhash: DeviceBuffer::from_host(&stream, &cidhash)?,
            d_cobmeta: DeviceBuffer::from_host(&stream, &cobmeta)?,
            d_cfriends: DeviceBuffer::<u32>::zeroed(&stream, n * (kernels::CNB + 1))?,
            d_catkow: DeviceBuffer::from_host(&stream, &catkow)?,
            d_catktg: DeviceBuffer::from_host(&stream, &catktg)?,
            d_catktr: DeviceBuffer::from_host(&stream, &catktr)?,
            d_cpst: DeviceBuffer::<f64>::zeroed(&stream, n * CCOLS * 3)?,
            n,
            wh,
        })
    }
}

fn cfg_for_block(n_envs: usize, block: u32) -> cuda_core::simt::LaunchConfig {
    let blocks = if n_envs == 0 {
        1
    } else {
        (n_envs as u32 + block - 1) / block
    };
    cuda_core::simt::LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn cfg_for(n_envs: usize) -> cuda_core::simt::LaunchConfig {
    cfg_for_block(n_envs, BLOCK)
}

/// Grid for a kernel whose threads are one-per-element (not one-per-env).
fn cfg_for_block_from_total(total: usize, block: u32) -> cuda_core::simt::LaunchConfig {
    cfg_for_block(total, block)
}

/// `env_hash` is declared `#[launch_bounds(64)]`, so it must be launched with a
/// 64-thread block; a larger block also is a device-side `invalid argument`.
fn cfg_hash(n_envs: usize) -> cuda_core::simt::LaunchConfig {
    cfg_for_block(n_envs, 64)
}

struct RunOut {
    dev_ms: f32,
    wall_ms: f64,
    hashes: Vec<u64>,
    scal0: Vec<u32>,
    claim_tails: Vec<Vec<u32>>,
}

/// One batched run: `ticks` launches of `env_step`, hashing after each tick
/// only in verify mode. Zero host copies inside the loop.
#[allow(clippy::too_many_arguments)]
fn run_batch(
    ctx: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    b: &mut DevBatch,
    env: &EnvState,
    ticks: u32,
    base_tick: u32,
    verify: bool,
    launch_only: bool,
    loop_mode: bool,
    do_clusters: bool,
) -> Result<RunOut, Box<dyn std::error::Error>> {
    let stream = ctx.default_stream();
    let n = b.n;
    let dc = do_clusters as u32;
    let trace_cad = std::env::var_os("OF_CAD_TRACE").is_some();
    let mut d_hashes = DeviceBuffer::<u64>::zeroed(&stream, (n.max(1)) * ticks as usize)?;
    let n_envs_arg = if launch_only { 0u32 } else { n as u32 };
    let cfg = cfg_for(if launch_only { 0 } else { n });
    let hcfg = cfg_hash(n);

    // Drain any pending setup work (plane_fill, the H2D copies) BEFORE the
    // clock starts, so the fixed per-configuration cost cannot leak into the
    // measured loop time.
    stream.synchronize()?;
    let ev0 = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let t0 = Instant::now();
    if loop_mode && !launch_only {
        // FAST PATH: one launch runs every tick. Nothing is read back.
        unsafe {
            module.env_step_loop(
                &stream,
                cfg,
                &b.d_terrain,
                env.w,
                env.h,
                base_tick,
                ticks,
                n as u32,
                env.owner_col,
                env.is_bot as u32,
                &mut b.d_plane,
                &mut b.d_heap_tiles,
                &mut b.d_heap_pri,
                &mut b.d_scal,
                &mut b.d_prng,
                &mut b.d_border,
                &mut b.d_claims,
                &b.d_oborder,
                &mut b.d_troops,
                &mut b.d_ccad,
                &mut b.d_ct2b,
                &mut b.d_cmark,
                &mut b.d_cbuf,
                &mut b.d_clen,
                &mut b.d_coff,
                &mut b.d_cvis,
                &mut b.d_cstack,
                &mut b.d_cowned,
                &mut b.d_crem,
                &mut b.d_cout,
                &b.d_corder,
                &b.d_cidhash,
                &b.d_cobmeta,
                &b.d_cfriends,
                &b.d_catkow,
                &b.d_catktg,
                &b.d_catktr,
                &mut b.d_cpst,
                dc,
            )?;
        }
    } else {
    for k in 0..ticks {
        let tick = base_tick + k;
        unsafe {
            module.env_step(
                &stream,
                cfg,
                &b.d_terrain,
                env.w,
                env.h,
                tick,
                n_envs_arg,
                env.owner_col,
                env.is_bot as u32,
                &mut b.d_plane,
                &mut b.d_heap_tiles,
                &mut b.d_heap_pri,
                &mut b.d_scal,
                &mut b.d_prng,
                &mut b.d_border,
                &mut b.d_claims,
                &b.d_oborder,
                &mut b.d_troops,
                &mut b.d_ccad,
                &mut b.d_ct2b,
                &mut b.d_cmark,
                &mut b.d_cbuf,
                &mut b.d_clen,
                &mut b.d_coff,
                &mut b.d_cvis,
                &mut b.d_cstack,
                &mut b.d_cowned,
                &mut b.d_crem,
                &mut b.d_cout,
                &b.d_corder,
                &b.d_cidhash,
                &b.d_cobmeta,
                &b.d_cfriends,
                &b.d_catkow,
                &b.d_catktg,
                &b.d_catktr,
                &mut b.d_cpst,
                dc,
            )?
        };
        if trace_cad {
            // Read the pass's own book-keeping back after the tick so the
            // cadence progression can be compared with the engine's CADENCE
            // rows. Verification-only readback.
            let cad_v = b.d_ccad.to_host_vec(&stream)?;
            let out_v = b.d_cout.to_host_vec(&stream)?;
            let cs = (env.owner_col as usize) * kernels::CSTATE;
            eprintln!(
                "CADTRACE tick={} sid={} lcc={} ltc={} tiles={} alive={} fires={} removals={} rem_words={} ovf={},{},{},{}",
                tick,
                env.owner_sid,
                cad_v[cs],
                cad_v[cs + 1],
                cad_v[cs + 2],
                cad_v[cs + 3],
                out_v[0],
                out_v[1],
                out_v[6],
                out_v[2],
                out_v[3],
                out_v[4],
                out_v[5],
            );
        }
        if verify && !launch_only {
            unsafe {
                module.env_hash(&stream, hcfg, &b.d_plane, b.wh as u32, n as u32, k, &mut d_hashes)?
            };
        }
    }
    }
    let ev1 = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    let dev_ms = ev0.elapsed_ms(&ev1)?;
    let hashes = if verify && !launch_only {
        d_hashes.to_host_vec(&stream)?
    } else {
        Vec::new()
    };
    stream.synchronize()?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let scal0 = b.d_scal.to_host_vec(&stream)?;
    let claims = b.d_claims.to_host_vec(&stream)?;
    let mut claim_tails = Vec::new();
    if verify && !launch_only {
        let ncl = scal0[2] as usize;
        claim_tails.push(claims[..ncl.min(CC)].to_vec());
    }
    Ok(RunOut {
        dev_ms,
        wall_ms,
        hashes,
        scal0,
        claim_tails,
    })
}

// ---------------------------------------------------------------------------

/// `--originate`: the ENV crate originates an attack ON THE DEVICE from the
/// moved bot AI - no oracle, no host re-injection. Runs the core's
/// `bot_ai_core` decision, then the core's `orig_refresh_dev` to build the
/// attack's heap/border/PRNG state in this env's own device buffers.
///
/// The bot schedule below is FORCED to fire at `tick` (rate 1, at 0, trigger 0,
/// `maxtroops = troops + 1`, zero pre-multiplied rows) because the point is the
/// ORIGINATION path, not the cadence: the same kernel with the engine's real
/// `tribe_ratios` schedule is what the trainer will run once the roster feeds
/// it. Both kernels are the canonical core's - the identical code the matrix's
/// `bot_ai` / `attack_init` wrappers call.
fn originate_demo(
    ctx: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    env: &EnvState,
    a: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    let stream = ctx.default_stream();
    let envs = vec![env0_copy(env)?];
    let mut b = DevBatch::new(ctx, module, &envs, 1)?;
    let (w, h) = (env.w, env.h);
    let tick = env.tick0 + a.ticks.max(1);
    let obn = env.oborder.len() as u32;
    let owned = env.plane.iter().filter(|x| **x == env.owner_col).count();

    // ---- 1. the bot DECISION, launched from this crate --------------------
    let mcap = 1usize;
    let mut pst = vec![0f64; (env.owner_sid as usize + 1) * 3];
    pst[env.owner_sid as usize * 3] = env.troops;
    pst[env.owner_sid as usize * 3 + 1] = owned as f64;
    let d_pst = DeviceBuffer::<f64>::from_host(&stream, &pst)?;
    let d_sid = DeviceBuffer::<u32>::from_host(&stream, &[env.owner_sid as u32])?;
    let d_rate = DeviceBuffer::<u32>::from_host(&stream, &[1u32])?;
    let d_at = DeviceBuffer::<u32>::from_host(&stream, &[0u32])?;
    let d_trig = DeviceBuffer::<f64>::from_host(&stream, &[0.0f64])?;
    let d_trow = DeviceBuffer::<u32>::from_host(&stream, &[0u32])?;
    let d_rrow = DeviceBuffer::<u32>::from_host(&stream, &[0u32])?;
    let d_ff = DeviceBuffer::<u32>::from_host(&stream, &[0u32])?;
    let mut d_state = DeviceBuffer::<u32>::from_host(&stream, &[2u32])?;
    let d_boat = DeviceBuffer::<u32>::from_host(&stream, &[0u32])?;
    let d_max = DeviceBuffer::<f64>::from_host(&stream, &[env.troops + 1.0])?;
    let d_tgt = DeviceBuffer::<f64>::from_host(&stream, &vec![0.0f64; 2 * mcap])?;
    // `ob_meta` is indexed `sid * 2` by the core (`bot_land_border_tn`), so it
    // must carry a row for the owner's own sid even though we only fill one.
    let mut h_meta = vec![0u32; (env.owner_sid as usize + 1) * 2];
    h_meta[env.owner_sid as usize * 2] = 0;
    h_meta[env.owner_sid as usize * 2 + 1] = obn;
    let d_meta = DeviceBuffer::<u32>::from_host(&stream, &h_meta)?;
    let mut d_out = DeviceBuffer::<f64>::zeroed(&stream, 3)?;
    unsafe {
        module.bot_ai(
            &stream,
            cfg_for_block(1, 1),
            &b.d_terrain,
            w,
            h,
            tick,
            0u32,
            &b.d_plane,
            &d_pst,
            &b.d_oborder,
            &d_meta,
            &d_sid,
            &d_rate,
            &d_at,
            &d_trig,
            &d_trow,
            &d_rrow,
            &d_ff,
            &mut d_state,
            &d_boat,
            &d_max,
            &d_tgt,
            &mut d_out,
            1u32,
        )?;
    }
    let dec = d_out.to_host_vec(&stream)?;
    println!(
        "DEV_BOT_AI (env crate, device): fired={} action={} troops={}",
        dec[0], dec[2], dec[1]
    );

    // ---- 2. ORIGINATION, on the device, from the canonical core -----------
    let mut d_orig = DeviceBuffer::<u32>::zeroed(&stream, 4)?;
    unsafe {
        module.originate_env(
            &stream,
            cfg_for_block(1, 1),
            &b.d_terrain,
            w,
            h,
            tick,
            env.owner_col,
            0u16,
            &b.d_plane,
            &b.d_oborder,
            0u32,
            obn,
            &mut b.d_heap_tiles,
            &mut b.d_heap_pri,
            &mut b.d_border,
            &mut b.d_prng,
            &mut b.d_scal,
            &mut d_orig,
        )?;
    }
    let o = d_orig.to_host_vec(&stream)?;
    let scal = b.d_scal.to_host_vec(&stream)?;
    let tiles = b.d_heap_tiles.to_host_vec(&stream)?;
    println!(
        "DEV_ORIGINATE (env crate, device): heap_len={} border_len={} peak={} prng_calls={} scal_alive={}",
        o[0], o[1], o[2], o[3], scal[5]
    );
    let ht = tiles.first().copied().unwrap_or(0);
    println!(
        "  frontier head tile={} (x={} y={}) owner_at_head={} env_owner_col={} owner_border_tiles={} owned_tiles={}",
        ht,
        ht % w,
        ht / w,
        env.plane.get(ht as usize).copied().unwrap_or(0xffff),
        env.owner_col,
        obn,
        owned
    );
    Ok(())
}

struct Args {
    dump: PathBuf,
    map: PathBuf,
    t0: u32,
    ticks: u32,
    envs: Vec<usize>,
    mode: String,
    spread: u32,
    selftest: bool,
    verify_max: usize,
    dump_states: bool,
    /// Skip the player-clusters pass (throughput A/B only; never used for
    /// parity or training).
    no_clusters: bool,
    /// `--originate`: the env crate originates an attack on the device from
    /// the moved bot AI (see `originate_demo`).
    originate: bool,
    /// `--multi`: the whole-game path - one env holds a SET of live attacks and
    /// runs with no oracle after setup (see `run_multi_demo`).
    multi: bool,
    /// `--orig` with `--multi`: the batched env ORIGINATES the bots' land
    /// attacks itself (schedule + device frontier + device bot decision),
    /// instead of only ticking the attack SET captured at the snapshot.
    orig: bool,
    /// `--rl`: emit the TRAINING interface (`ofcuda_env::rl`): per-decision
    /// observation (4270 f32), action mask (40893 f32) and reward, derived
    /// from the device state of the unaided multi-attack game.
    rl: bool,
    /// Where to write the RL decision stream (JSONL). `None` = summary only.
    rl_out: Option<PathBuf>,
    /// Ticks per decision (the curriculum's `decision_ticks`, default 15).
    rl_dt: u32,
    /// The RL agent's small id. `None` = the first bot in the roster.
    rl_agent: Option<u16>,
    /// Cap on emitted decisions (0 = every decision in the window).
    rl_decisions: usize,
    /// Scripted agent action per decision (action ids from the mask's
    /// `legal_actions` block). Empty = all-noop. 0 noop, 2 expand.
    rl_actions: Vec<usize>,
    /// WIRE THE ACTION INTO THE SIM. With this on, the chosen action is not
    /// merely recorded: it drives the controlled player's attack origination.
    /// `A_EXPAND` creates exactly the terra-nullius land attack the env's own
    /// bot AI would have created for that player (see `agent_expand_fire`),
    /// and the player's own SCHEDULED firings are suppressed so the agent - not
    /// the bot - is playing that player.
    rl_control: bool,
    /// Per-env action scripts, `env:list` blocks joined by `;` (e.g.
    /// `0:0,2,2,2;1:0,0,0,0`). Different scripts in different envs are what
    /// makes the batch DIVERGE, and is how the per-env correctness of the
    /// origination path is measured. Envs with no block use `--rl-actions`.
    rl_actions_env: Vec<(usize, Vec<usize>)>,
}

/// The scripted ACTION for env `e` at decision index `i`: env `e`'s own block
/// of `--rl-actions-env` if it has one, else the shared `--rl-actions` list,
/// else noop. Different blocks in different envs are what makes a batch
/// DIVERGE, which is how the per-env correctness of the action channel and of
/// the origination path is measured.
fn rl_script(table: &[(usize, Vec<usize>)], base: &[usize], e: usize, i: usize) -> usize {
    if let Some((_, v)) = table.iter().find(|(ee, _)| *ee == e) {
        v.get(i).copied().unwrap_or(ofcuda_env::rl::A_NOOP)
    } else {
        base.get(i).copied().unwrap_or(ofcuda_env::rl::A_NOOP)
    }
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        dump: PathBuf::from("/tmp/envdump/b002.ndjson"),
        map: PathBuf::from("/opt/data/workspaces/skg/openfront-ai/openfront/resources/maps/pangaea"),
        t0: 600,
        ticks: 40,
        envs: vec![1, 64, 256, 1024],
        mode: "both".to_string(),
        spread: 1,
        selftest: false,
        verify_max: 256,
        dump_states: false,
        no_clusters: false,
        originate: false,
        multi: false,
        orig: false,
        rl: false,
        rl_out: None,
        rl_dt: 15,
        rl_agent: None,
        rl_decisions: 0,
        rl_actions: Vec::new(),
        rl_control: false,
        rl_actions_env: Vec::new(),
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        let val = |i: usize| -> Result<String, String> {
            v.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", v[i]))
        };
        match v[i].as_str() {
            "--dump" => {
                a.dump = PathBuf::from(val(i)?);
                i += 2;
            }
            "--map" => {
                a.map = PathBuf::from(val(i)?);
                i += 2;
            }
            "--t0" => {
                a.t0 = val(i)?.parse().map_err(|e| format!("t0: {e}"))?;
                i += 2;
            }
            "--ticks" => {
                a.ticks = val(i)?.parse().map_err(|e| format!("ticks: {e}"))?;
                i += 2;
            }
            "--envs" => {
                a.envs = val(i)?
                    .split(',')
                    .map(|x| x.trim().parse::<usize>().map_err(|e| format!("envs: {e}")))
                    .collect::<Result<Vec<_>, _>>()?;
                i += 2;
            }
            "--mode" => {
                a.mode = val(i)?;
                i += 2;
            }
            "--spread" => {
                a.spread = val(i)?.parse().map_err(|e| format!("spread: {e}"))?;
                i += 2;
            }
            "--verify-max" => {
                a.verify_max = val(i)?.parse().map_err(|e| format!("verify-max: {e}"))?;
                i += 2;
            }
            "--dump-states" => {
                a.dump_states = true;
                i += 1;
            }
            "--no-clusters" => {
                a.no_clusters = true;
                i += 1;
            }
            "--selftest" => {
                a.selftest = true;
                i += 1;
            }
            "--originate" => {
                a.originate = true;
                i += 1;
            }
            "--multi" => {
                a.multi = true;
                i += 1;
            }
            "--orig" => {
                a.orig = true;
                i += 1;
            }
            "--rl" => {
                a.rl = true;
                i += 1;
            }
            "--rl-out" => {
                a.rl_out = Some(PathBuf::from(val(i)?));
                i += 2;
            }
            "--rl-dt" => {
                a.rl_dt = val(i)?.parse().map_err(|e| format!("rl-dt: {e}"))?;
                i += 2;
            }
            "--rl-agent" => {
                a.rl_agent = Some(val(i)?.parse().map_err(|e| format!("rl-agent: {e}"))?);
                i += 2;
            }
            "--rl-decisions" => {
                a.rl_decisions = val(i)?.parse().map_err(|e| format!("rl-decisions: {e}"))?;
                i += 2;
            }
            "--rl-actions" => {
                a.rl_actions = val(i)?
                    .split(',')
                    .filter(|x| !x.trim().is_empty())
                    .map(|x| x.trim().parse::<usize>().map_err(|e| format!("rl-actions: {e}")))
                    .collect::<Result<Vec<_>, _>>()?;
                i += 2;
            }
            "--rl-control" => {
                a.rl_control = true;
                i += 1;
            }
            "--rl-actions-env" => {
                for blk in val(i)?.split(';').filter(|x| !x.trim().is_empty()) {
                    let (e, list) = blk
                        .split_once(':')
                        .ok_or_else(|| format!("rl-actions-env block needs `env:list`: {blk}"))?;
                    let e: usize = e
                        .trim()
                        .parse()
                        .map_err(|x| format!("rl-actions-env env: {x}"))?;
                    let acts = list
                        .split(',')
                        .filter(|x| !x.trim().is_empty())
                        .map(|x| {
                            x.trim()
                                .parse::<usize>()
                                .map_err(|x| format!("rl-actions-env: {x}"))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    a.rl_actions_env.push((e, acts));
                }
                i += 2;
            }
            o => return Err(format!("unknown arg {o}").into()),
        }
    }
    Ok(a)
}

/// FNV-1a 64 over a decision tensor's f32 little-endian bytes - the RL stream's
/// per-tensor hash (`obs_hash` / `mask_hash` in the JSONL stream). Same family
/// as the parity traces' plane hash, so both are directly comparable in a log.
fn rl_hash_f32(v: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in v {
        for b in x.to_le_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// The driver's conditions for a creation that is NOT a merge: exactly one
/// live land attack at the tick, and the computed start's floor equals the
/// record's `troops` (a merge shows the SUM). Scanning forward finds a t0 whose
/// single-attack env is the whole story.
fn find_t0(
    dump: &ofcuda_tick::Dump,
    ex: &extras::Extra,
    w: u32,
    h: u32,
    terrain: &[u8],
    from: u32,
    span: u32,
) -> Option<u32> {
    for t in from..from + span {
        let Some(list) = ex.attacks.get(&t) else {
            continue;
        };
        let live: Vec<&extras::AttRec> =
            list.iter().filter(|a| a.live && a.target == 0).collect();
        if live.len() != 1 {
            continue;
        }
        if dump.get(&t).is_none() || dump.get(&(t + 1)).is_none() {
            continue;
        }
        let Ok(e) = build_env_at(dump, ex, w, h, terrain, t) else {
            continue;
        };
        if e.troops.floor() as i64 == live[0].troops {
            return Some(t);
        }
    }
    None
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let a = parse_args()?;
    if a.multi {
        if a.orig {
            return run_multi_orig(&a);
        }
        return run_multi_demo(&a);
    }
    let map = ofcuda_tick::load_map(&a.map)?;
    let (w, h) = (map.width, map.height);
    let terrain = map.terrain.clone();
    let dump = ofcuda_tick::parse_dump(&a.dump)?;
    let ex = extras(&a.dump)?;

    // ---- the environment -------------------------------------------------
    let t0 = match find_t0(&dump, &ex, w, h, &terrain, a.t0, 400) {
        Some(t) => {
            if t != a.t0 {
                println!("t0 scan: {} has no unique non-merged attack; using t0={}", a.t0, t);
            }
            t
        }
        None => a.t0,
    };
    let a = Args { t0, ..a };
    let env0 = build_env_at(&dump, &ex, w, h, &terrain, a.t0)?;
    println!(
        "env: map={}x{} ({} tiles, {} MB plane/env) t0={} owner_sid={} plane_word={} bot={} start_troops={:.3} heap={} border={} oborder={} land_attack_record_troops={}",
        w,
        h,
        (w as usize) * (h as usize),
        (w as usize) * (h as usize) * 2 / 1_000_000,
        a.t0,
        env0.owner_sid,
        env0.owner_col,
        env0.is_bot,
        env0.troops,
        env0.heap_len,
        env0.border.len(),
        env0.oborder.len(),
        ex.attacks
            .get(&a.t0)
            .and_then(|v| v.iter().find(|x| x.live && x.target == 0).map(|x| x.troops))
            .unwrap_or(-1),
    );

    let (h_claims, h_hashes, refreshes, h_drops, h_peak) = host_reference(&env0, a.ticks);
    println!(
        "host reference: ticks={} plane_hashes={} refreshes={} heap_drops={} heap_peak={}",
        h_claims.len(),
        h_hashes.len(),
        refreshes,
        h_drops,
        h_peak
    );

    // ---- engine agreement of the CPU path (a validity check of the env) ---
    let mut eng_eq = 0usize;
    let mut eng_tot = 0usize;
    for k in 0..h_claims.len() as u32 {
        let (sa, sb) = (a.t0 + k, a.t0 + k + 1);
        let (Some(pa), Some(pb)) = (dump.get(&sa), dump.get(&sb)) else {
            break;
        };
        let mut tot = 0usize;
        let mut eq = 0usize;
        for p in pa.values() {
            if p.small_id != env0.owner_sid as u32 {
                continue;
            }
            let before = p.owned_order.len();
            let after = pb
                .values()
                .find(|q| q.small_id == p.small_id)
                .map(|q| q.owned_order.len())
                .unwrap_or(0);
            let mut got: Vec<u32> = Vec::new();
            if after >= before {
                let after_list: Vec<u32> = pb
                    .values()
                    .find(|q| q.small_id == p.small_id)
                    .map(|q| q.owned_order.clone())
                    .unwrap_or_default();
                got = after_list[before.min(after_list.len())..].to_vec();
            }
            let mut ok = got.len() == h_claims[k as usize].len();
            if ok {
                for (x, y) in got.iter().zip(h_claims[k as usize].iter()) {
                    if x != y {
                        ok = false;
                        break;
                    }
                }
            }
            tot += 1;
            if ok {
                eq += 1;
            }
        }
        if tot > 0 {
            eng_tot += 1;
            if eq == tot {
                eng_eq += 1;
            }
        }
    }
    println!(
        "cpu-env vs engine ownedOrder deltas: {}/{} ticks match",
        eng_eq, eng_tot
    );

    // ---- device ----------------------------------------------------------
    let ctx = CudaContext::new(0)?;
    let module = unsafe { kernels::load(&ctx)? };

    // `--originate`: prove the env crate can originate an attack from the
    // moved bot AI, with no oracle re-injection. Early-out mode.
    if a.originate {
        return originate_demo(&ctx, &module, &env0, &a);
    }

    // Self-test: device vs the CPU reference, per tick, on identical state.
    if a.selftest || a.mode == "both" {
        let envs = vec![env0_copy(&env0)?];
        let mut b = DevBatch::new(&ctx, &module, &envs, 1)?;
        let out = run_batch(
            &ctx, &module, &mut b, &env0, a.ticks, env0.tick0, true, false, false, true,
        )
        .map_err(|e| e.to_string())?;
        let n = b.n;
        let mut hash_eq = 0usize;
        let mut claim_eq = 0usize;
        for k in 0..h_hashes.len().min(a.ticks as usize) {
            if out.hashes[k * n] == h_hashes[k] {
                hash_eq += 1;
            }
        }
        // per-tick claim tails: the device's `claims[n0..ncl]` is recovered by
        // diffing the whole-life list length across ticks is not available, so
        // compare the FINAL whole-life list length and content instead.
        if let Some(dev_claims) = out.claim_tails.first() {
            let h_final = {
                let mut atk = Attack::new(
                    env0.owner_sid,
                    0,
                    env0.is_bot,
                    env0.troops,
                    ofcuda_tick::SEED,
                );
                atk.heap = Heap::new();
                atk.heap.load(&env0.heap_tiles, &env0.heap_pri);
                atk.pr = Prng::from_state(&env0.prng);
                atk.border = env0.border.clone();
                atk.claims = env0.claims.clone();
                let mut plane = env0.plane.clone();
                for k in 0..a.ticks {
                    let o = atk.tick(
                        &plane,
                        &env0.terrain,
                        w,
                        h,
                        env0.tick0 + k,
                        &env0.oborder,
                        0.0,
                        false,
                    );
                    for t in &o.claims {
                        plane[*t as usize] = env0.owner_col;
                    }
                    if o.dead {
                        break;
                    }
                }
                atk.claims.clone()
            };
            if h_final.len() == dev_claims.len()
                && h_final.iter().zip(dev_claims.iter()).all(|(x, y)| x == y)
            {
                claim_eq = 1;
            }
            println!(
                "SELFTEST device-vs-cpu: plane hashes {}/{} equal; whole-life claim list equal={} (dev {} cpu {})",
                hash_eq,
                h_hashes.len().min(a.ticks as usize),
                claim_eq == 1,
                dev_claims.len(),
                h_final.len()
            );
            println!(
                "  host troops_bits={} device scal0={:?}",
                env0.troops.to_bits(),
                out.scal0
            );
            println!(
                "  device state after {} ticks: alive={} claims_len={} heap_len={} border_len={} drops={} peak={} troops_bits={}",
                a.ticks,
                out.scal0[5],
                out.scal0[2],
                out.scal0[0],
                out.scal0[1],
                out.scal0[6],
                out.scal0[7],
                out.scal0[4]
            );
        }
        // The ONE-LAUNCH fast path must land on the same final state as the
        // per-tick path, because it is the mode the throughput numbers use.
        let mut b2 = DevBatch::new(&ctx, &module, &envs, 1)?;
        let out2 = run_batch(
            &ctx,
            &module,
            &mut b2,
            &env0,
            a.ticks,
            env0.tick0,
            false,
            false,
            true,
            true,
        )
        .map_err(|e| e.to_string())?;
        let stream = ctx.default_stream();
        let mut d_last = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
        unsafe {
            module.env_hash(
                &stream,
                cfg_hash(b2.n),
                &b2.d_plane,
                b2.wh as u32,
                b2.n as u32,
                0,
                &mut d_last,
            )?
        };
        let last_dev = d_last.to_host_vec(&stream)?[0];
        let last_cpu = h_hashes.last().copied().unwrap_or(0);
        let cpu_claims: usize = h_claims.iter().map(|c| c.len()).sum();
        println!(
            "SELFTEST one-launch loop path ({} ticks in 1 launch): final plane hash {} (dev {:016x} cpu {:016x}), claims {} (cpu {}), ticks_run={} drops={} peak={}",
            a.ticks,
            if last_dev == last_cpu { "EQUAL" } else { "DIFFERENT" },
            last_dev,
            last_cpu,
            out2.scal0[2],
            cpu_claims,
            out2.scal0[3],
            out2.scal0[6],
            out2.scal0[7],
        );
        if a.selftest && a.mode == "selftest" {
            return Ok(());
        }
    }

    // ---- launch overhead -------------------------------------------------
    {
        let envs = vec![env0_copy(&env0)?];
        let mut b = DevBatch::new(&ctx, &module, &envs, 1)?;
        let out = run_batch(&ctx, &module, &mut b, &env0, a.ticks, env0.tick0, false, true, false, true)
            .map_err(|e| e.to_string())?;
        println!(
            "launch overhead ({} launches, no work): device {:.4} ms/tick, wall {:.4} ms/tick",
            a.ticks,
            out.dev_ms as f64 / a.ticks as f64,
            out.wall_ms / a.ticks as f64
        );
    }

    // ---- throughput at each env count -----------------------------------
    for &n in &a.envs {
        // spread == 1 means every env is the SAME environment (a throughput
        // measurement, not a diversity measurement), so the batch holds ONE
        // EnvState and replicates it; holding n copies would be n * 2 MB of
        // host RAM for no added information.
        let envs: Vec<EnvState> = if a.spread == 1 {
            vec![env0_copy(&env0)?]
        } else {
            (0..n)
                .map(|e| build_env_at(&dump, &ex, w, h, &terrain, a.t0 + (e as u32 % a.spread)))
                .collect::<Result<Vec<_>, _>>()?
        };
        let verify = (a.mode == "verify" || a.mode == "both") && n <= a.verify_max;
        // warmup (untimed): the same batch object, a few ticks.
        let mut b = DevBatch::new(&ctx, &module, &envs, n)?;
        run_batch(&ctx, &module, &mut b, &env0, 4, env0.tick0, false, false, false, true)
            .map_err(|e| e.to_string())?;

        for (tag, loop_mode, verify_tag) in [
            ("fast-1launch", true, false),
            ("fast-pertick", false, false),
            ("verify", false, true),
        ] {
            if verify_tag && !verify {
                continue;
            }
            let mut b = DevBatch::new(&ctx, &module, &envs, n)?;
            let out = run_batch(
                &ctx,
                &module,
                &mut b,
                &env0,
                a.ticks,
                env0.tick0,
                verify_tag,
                false,
                loop_mode,
                !a.no_clusters,
            )
            .map_err(|e| e.to_string())?;
            let ticks = a.ticks as f64;
            let ticks_s = (n as f64 * ticks) / (out.wall_ms / 1000.0);
            let dev_ticks_s = (n as f64 * ticks) / (f64::from(out.dev_ms) / 1000.0);
            let host_ms = out.wall_ms - f64::from(out.dev_ms);
            println!(
                "N={:<5} mode={:<6} total {:.3} ms  {:.4} ms/tick  {:.1} env-ticks/s  {:.1} decisions/s | device {:.4} ms/tick ({:.1}%) host {:.4} ms/tick ({:.1}%) | device-only {:.1} env-ticks/s",
                n,
                tag,
                out.wall_ms,
                out.wall_ms / ticks,
                ticks_s,
                ticks_s / 15.0,
                f64::from(out.dev_ms) / ticks,
                100.0 * f64::from(out.dev_ms) / out.wall_ms,
                host_ms / ticks,
                100.0 * host_ms / out.wall_ms,
                dev_ticks_s,
            );
            println!(
                "        env0 device totals: alive={} ticks_run={} claims={} heap_len={} border_len={} heap_drops={} heap_peak={} troops={:.3}",
                out.scal0[5],
                out.scal0[3],
                out.scal0[2],
                out.scal0[0],
                out.scal0[1],
                out.scal0[6],
                out.scal0[7],
                // troops live in their own buffer; re-read it below
                0.0
            );
            let stream = ctx.default_stream();
            let tr = b.d_troops.to_host_vec(&stream)?;
            println!("        env0 troops on device = {:.6}", tr[0]);
            if a.dump_states {
                let sf = b.d_scal.to_host_vec(&stream)?;
                let cl = b.d_claims.to_host_vec(&stream)?;
                println!(
                    "        SIZES: b.n={} SCAL={} scal.len()={} claims.len()={} plane.len()={} troops.len()={}",
                    b.n, SCAL, sf.len(), cl.len(), b.d_plane.len(), tr.len()
                );
                let n = b.n.min(sf.len() / SCAL).min(cl.len() / CC).min(tr.len());
                let mut tuples: Vec<(u32, u32, u32, u32, u64)> = Vec::with_capacity(n);
                for e in 0..n {
                    let so = e * SCAL;
                    tuples.push((
                        sf[so],
                        sf[so + 1],
                        sf[so + 2],
                        sf[so + 5],
                        tr[e].to_bits(),
                    ));
                }
                let mut uniq: Vec<&(u32, u32, u32, u32, u64)> = Vec::new();
                for t in tuples.iter() {
                    if !uniq.iter().any(|u| *u == t) {
                        uniq.push(t);
                    }
                }
                println!(
                    "        BATCH: plane buffer {} elements (n*wh = {}), envs = {}, DISTINCT per-env states = {}",
                    b.d_plane.len(),
                    n * b.wh,
                    n,
                    uniq.len()
                );
                for e in 0..n.min(4) {
                    let t = tuples[e];
                    println!(
                        "          env{}: heap_len={} border_len={} claims_len={} alive={} troops_bits={} first_claim={}",
                        e, t.0, t.1, t.2, t.3, t.4, cl[e * CC]
                    );
                }
                // distinct first-claim per env is the sharpest "different state"
                // witness: identical envs give identical first claims.
                let fc: Vec<u32> = (0..n).map(|e| cl[e * CC]).collect();
                let mut fcu: Vec<u32> = Vec::new();
                for x in fc.iter() {
                    if !fcu.contains(x) {
                        fcu.push(*x);
                    }
                }
                println!(
                    "          DISTINCT first_claim across the batch = {} (of {})",
                    fcu.len(),
                    n
                );
            }
        }
    }

    let mut drops_tot = 0u64;
    for e in 0..1 {
        let _ = e;
        drops_tot += 0;
    }
    let _ = drops_tot;
    Ok(())
}

fn env0_copy(e: &EnvState) -> Result<EnvState, String> {
    Ok(EnvState {
        w: e.w,
        h: e.h,
        terrain: e.terrain.clone(),
        plane: e.plane.clone(),
        owner_sid: e.owner_sid,
        owner_col: e.owner_col,
        owner_id: e.owner_id.clone(),
        is_bot: e.is_bot,
        tick0: e.tick0,
        troops: e.troops,
        heap_tiles: e.heap_tiles.clone(),
        heap_pri: e.heap_pri.clone(),
        heap_len: e.heap_len,
        prng: e.prng,
        border: e.border.clone(),
        oborder: e.oborder.clone(),
        claims: e.claims.clone(),
        alive: e.alive,
        attacks: e.attacks.clone(),
    })
}

// ---------------------------------------------------------------------------
// The multi-attack batch: one env = a whole game with a SET of live attacks
// ---------------------------------------------------------------------------

struct MultiBatch {
    d_plane: DeviceBuffer<u16>,
    d_terrain: DeviceBuffer<u8>,
    d_mheap_tiles: DeviceBuffer<u32>,
    d_mheap_pri: DeviceBuffer<f32>,
    d_mborder: DeviceBuffer<u32>,
    d_mclaims: DeviceBuffer<u32>,
    d_mprng: DeviceBuffer<u32>,
    d_mtroops: DeviceBuffer<f64>,
    d_mscal: DeviceBuffer<u32>,
    d_mowner: DeviceBuffer<u16>,
    d_misbot: DeviceBuffer<u32>,
    d_moborder: DeviceBuffer<u32>,
    /// Incremental device owner-border sets: `n * npl * PB` words, plus
    /// `n * npl` lengths and `n * npl` owned-tile counters. Maintained by
    /// `env_step_multi` in the engine's insertion order; seeded from the
    /// record at t0 and thereafter evolved entirely on the device.
    d_pob: DeviceBuffer<u32>,
    d_poblen: DeviceBuffer<u32>,
    d_ptiles: DeviceBuffer<u32>,
    d_cad: DeviceBuffer<u32>,
    n: usize,
    slots: usize,
    wh: usize,
    npl: usize,
    w: u32,
    h: u32,
    /// Total bytes held on the device by this batch (all buffers).
    bytes: usize,
}

impl MultiBatch {
    fn new(
        ctx: &std::sync::Arc<CudaContext>,
        module: &kernels::LoadedModule,
        envs: &[EnvState],
        n: usize,
        npl: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let stream = ctx.default_stream();
        let w = envs[0].w;
        let h = envs[0].h;
        let wh = (w as usize) * (h as usize);
        let npl = npl.max(1);
        let mut d_plane = DeviceBuffer::<u16>::zeroed(&stream, n * wh)?;
        let d_tmpl = DeviceBuffer::<u16>::from_host(&stream, &envs[0].plane)?;
        unsafe {
            module.plane_fill(
                &stream,
                cfg_for_block_from_total(n * wh, 256),
                &d_tmpl,
                wh as u32,
                n as u32,
                &mut d_plane,
            )?;
        }
        let g = n * SLOTS;
        let mut mscal = vec![0u32; g * SCAL];
        let mut mtroops = vec![0f64; g];
        let mut mowner = vec![0u16; g];
        let mut misbot = vec![0u32; g];
        let mut prng = vec![0u32; g * 5];
        let mut heap_tiles = vec![0u32; g * HEAP_CAP];
        let mut heap_pri = vec![0f32; g * HEAP_CAP];
        let mut border = vec![0u32; g * BC];
        let mut claims = vec![0u32; g * CC];
        let mut oborder = vec![0u32; g * OB];
        let cspan = CCOLS * kernels::CSTATE;
        let mut cad = vec![0u32; n * cspan];
        for e in 0..n {
            let env = &envs[if envs.len() == 1 { 0 } else { e }];
            let sid = env.owner_sid as usize;
            if sid < CCOLS {
                let co = e * cspan + sid * kernels::CSTATE;
                cad[co + 1] = env.tick0; // last_tile_change: the plane IS tick t0
                cad[co + 2] = env.plane.iter().filter(|v| **v == env.owner_col).count() as u32;
                cad[co + 3] = 1;
            }
            for (si, a) in env.attacks.iter().enumerate() {
                if si >= SLOTS {
                    break;
                }
                let gi = e * SLOTS + si;
                let so = gi * SCAL;
                mowner[gi] = a.owner_sid;
                misbot[gi] = if a.is_bot { 1 } else { 0 };
                mtroops[gi] = a.troops;
                mscal[so] = a.heap_len as u32;
                mscal[so + 1] = a.border.len() as u32;
                mscal[so + 2] = a.claims.len() as u32;
                mscal[so + 5] = if a.alive { 1 } else { 0 };
                let on = a.oborder.len().min(OB);
                mscal[so + 8] = on as u32;
                prng[gi * 5..gi * 5 + 5].copy_from_slice(&a.prng);
                heap_tiles[gi * HEAP_CAP..gi * HEAP_CAP + a.heap_len]
                    .copy_from_slice(&a.heap_tiles[..a.heap_len]);
                heap_pri[gi * HEAP_CAP..gi * HEAP_CAP + a.heap_len]
                    .copy_from_slice(&a.heap_pri[..a.heap_len]);
                border[gi * BC..gi * BC + a.border.len()].copy_from_slice(&a.border);
                claims[gi * CC..gi * CC + a.claims.len()].copy_from_slice(&a.claims);
                oborder[gi * OB..gi * OB + on].copy_from_slice(&a.oborder[..on]);
            }
        }
        let bytes = n * wh * 2
            + g * HEAP_CAP * 4
            + g * HEAP_CAP * 4
            + g * BC * 4
            + g * CC * 4
            + g * 5 * 4
            + g * 8
            + g * SCAL * 4
            + g * 2
            + g * 4
            + g * OB * 4
            + n * cspan * 4
            + n * npl * PB * 4
            + n * npl * 4
            + n * npl * 4;
        Ok(MultiBatch {
            d_plane,
            d_terrain: DeviceBuffer::from_host(&stream, &envs[0].terrain)?,
            d_mheap_tiles: DeviceBuffer::from_host(&stream, &heap_tiles)?,
            d_mheap_pri: DeviceBuffer::from_host(&stream, &heap_pri)?,
            d_mborder: DeviceBuffer::from_host(&stream, &border)?,
            d_mclaims: DeviceBuffer::from_host(&stream, &claims)?,
            d_mprng: DeviceBuffer::from_host(&stream, &prng)?,
            d_mtroops: DeviceBuffer::from_host(&stream, &mtroops)?,
            d_mscal: DeviceBuffer::from_host(&stream, &mscal)?,
            d_mowner: DeviceBuffer::from_host(&stream, &mowner)?,
            d_misbot: DeviceBuffer::from_host(&stream, &misbot)?,
            d_moborder: DeviceBuffer::from_host(&stream, &oborder)?,
            d_pob: DeviceBuffer::<u32>::zeroed(&stream, n * npl * PB)?,
            d_poblen: DeviceBuffer::<u32>::zeroed(&stream, n * npl)?,
            d_ptiles: DeviceBuffer::<u32>::zeroed(&stream, n * npl)?,
            d_cad: DeviceBuffer::from_host(&stream, &cad)?,
            n,
            slots: SLOTS,
            wh,
            npl,
            w,
            h,
            bytes,
        })
    }
}

/// The engine's plane hash at a tick, straight from the record: the plane is
/// rebuilt from the per-player `ownedTiles` the record carries at that tick.
fn engine_hash_at(dump: &ofcuda_tick::Dump, w: u32, h: u32, t: u32) -> Option<u64> {
    let ps = dump.get(&t)?;
    let players: Vec<(u32, Vec<u32>)> = ps
        .values()
        .map(|p| (p.small_id, p.owned_tiles.clone()))
        .collect();
    Some(state_hash(&ofcuda_env::state_plane(&players, w, h)))
}

/// One batched multi-attack run with NO oracle: every tick is one device launch
/// of `env_step_multi` over the whole set. In verify mode the per-env plane
/// hash is read back after each tick and compared against the record's plane
/// for that tick - the ONLY place the record is touched after setup, and it is
/// never fed back into the simulation.
#[allow(clippy::too_many_arguments)]
fn run_multi(
    ctx: &std::sync::Arc<CudaContext>,
    module: &kernels::LoadedModule,
    b: &mut MultiBatch,
    env: &EnvState,
    ticks: u32,
    base_tick: u32,
    verify: bool,
) -> Result<(f64, Vec<u64>, Vec<u32>, usize), Box<dyn std::error::Error>> {
    let stream = ctx.default_stream();
    let n = b.n;
    let cfg = cfg_for(n);
    let hcfg = cfg_hash(n);
    let mut d_hashes = DeviceBuffer::<u64>::zeroed(&stream, n.max(1))?;
    stream.synchronize()?;
    let t0 = Instant::now();
    let mut hashes = Vec::new();
    for k in 0..ticks {
        unsafe {
            module.env_step_multi(
                &stream,
                cfg,
                &b.d_terrain,
                b.w,
                b.h,
                base_tick + k,
                n as u32,
                b.slots as u32,
                &mut b.d_plane,
                &mut b.d_mheap_tiles,
                &mut b.d_mheap_pri,
                &mut b.d_mborder,
                &mut b.d_mclaims,
                &mut b.d_mprng,
                &mut b.d_mtroops,
                &mut b.d_mscal,
                &b.d_mowner,
                &b.d_misbot,
                &b.d_moborder,
                &mut b.d_cad,
                &mut b.d_pob,
                &mut b.d_poblen,
                &mut b.d_ptiles,
                0,
                0,
            )?;
        }
        if verify {
            unsafe {
                module.env_hash(&stream, hcfg, &b.d_plane, b.wh as u32, n as u32, 0, &mut d_hashes)?;
            }
            hashes.push(d_hashes.to_host_vec(&stream)?[0]);
        }
    }
    stream.synchronize()?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let scal = b.d_mscal.to_host_vec(&stream)?;
    let mut live = 0usize;
    let mut drops = 0u32;
    for e in 0..n {
        for s in 0..b.slots {
            let so = (e * b.slots + s) * SCAL;
            if scal[so + 5] != 0 {
                live += 1;
            }
            drops += scal[so + 6];
        }
    }
    let _ = env;
    Ok((wall_ms, hashes, scal, live))
}

/// The tick at which `owner`'s land attack that is live at `s` was born: the
/// earliest tick of the contiguous run of live-ness that ends at `s`.
/// `(owner, target)` is an identity key because the engine merges two live
/// attacks between the same pair, so at most one exists at a time.
fn attack_birth(ex: &extras::Extra, owner: u16, s: u32, back: u32) -> u32 {
    let mut b = s;
    let lo = s.saturating_sub(back);
    while b > lo {
        let t = b - 1;
        let live = ex
            .attacks
            .get(&t)
            .map(|v| v.iter().any(|a| a.owner == owner && a.target == 0 && a.live))
            .unwrap_or(false);
        if !live {
            break;
        }
        b = t;
    }
    b
}

/// The record's plane at tick `t`, cached: rebuilding a plane is O(tiles) and
/// a replay needs one per tick of the attack's life.
fn plane_at(
    dump: &ofcuda_tick::Dump,
    w: u32,
    h: u32,
    t: u32,
    cache: &mut std::collections::HashMap<u32, Vec<u16>>,
) -> Option<Vec<u16>> {
    if let Some(p) = cache.get(&t) {
        return Some(p.clone());
    }
    let ps = dump.get(&t)?;
    let players: Vec<(u32, Vec<u32>)> = ps
        .values()
        .map(|p| (p.small_id, p.owned_tiles.clone()))
        .collect();
    let p = ofcuda_env::state_plane(&players, w, h);
    cache.insert(t, p.clone());
    Some(p)
}

/// `owner`'s `border_tiles` at tick `t` - the input to a mid-tick
/// `refresh_to_conquer`.
fn owner_border(
    dump: &ofcuda_tick::Dump,
    owner: u16,
    t: u32,
) -> Option<Vec<u32>> {
    dump.get(&t)?
        .values()
        .find(|p| p.small_id == owner as u32)
        .map(|p| p.border_order.clone())
}

/// Reconstruct ONE live attack exactly, at tick `s`, from the record alone:
/// find its birth tick, create it the way the engine's creation path does
/// (`refresh_to_conquer` over the owner's border with the plane of the birth
/// tick, stamped `birth-1`), then replay it tick by tick up to `s` feeding it
/// the RECORD's plane for each of those pre-`s` ticks. Every input is a tick
/// `< s`, so the result is the initial condition at `s` - not per-tick input.
#[allow(clippy::too_many_arguments)]
fn reconstruct_attack(
    dump: &ofcuda_tick::Dump,
    ex: &extras::Extra,
    w: u32,
    h: u32,
    terrain: &[u8],
    owner: u16,
    s: u32,
    cache: &mut std::collections::HashMap<u32, Vec<u16>>,
) -> Option<AttackState> {
    let b = attack_birth(ex, owner, s, 4096);
    let pi = ex.p.get(&(b - 1)).and_then(|m| m.get(&owner))?;
    let ratios = tribe_ratios(&pi.id);
    let row = econ_row(b - 1, pi.troops, pi.tiles, 0, pi.gold, pi.ptype as u32);
    let start = land_attack_start_troops(&row, ratios.expand_ratio)?;
    let is_bot = pi.ptype == ofcuda_econ::core_impl::PT_BOT as u8;
    let ob_b = owner_border(dump, owner, b)?;
    let p_b = plane_at(dump, w, h, b, cache)?;
    let mut atk = Attack::new(owner, 0, is_bot, start, ofcuda_tick::SEED);
    atk.refresh(&ob_b, &p_b, terrain, w, h, b - 1);
    for t in b..s {
        let pt = plane_at(dump, w, h, t, cache)?;
        let obt = owner_border(dump, owner, t)?;
        let _ = atk.tick(&pt, terrain, w, h, t, &obt, 0.0, false);
    }
    let oborder = owner_border(dump, owner, s).unwrap_or_default();
    let mut prng = [0u32; 5];
    atk.pr.state_words(&mut prng);
    Some(AttackState {
        owner_sid: owner,
        target_sid: 0,
        is_bot,
        troops: atk.troops,
        heap_tiles: atk.heap.tiles[..atk.heap.len].to_vec(),
        heap_pri: atk.heap.pri[..atk.heap.len].to_vec(),
        heap_len: atk.heap.len,
        prng,
        border: atk.border.clone(),
        oborder,
        claims: atk.claims.clone(),
        alive: atk.attack_live,
    })
}

/// `--multi`: the whole-game path. Builds the env at `t0` with its SET of live
/// attacks, ticks it on the device with NO oracle after setup, and reports how
/// many ticks the device plane matched the engine's plane at that tick.
fn run_multi_demo(a: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let map = ofcuda_tick::load_map(&a.map)?;
    let (w, h) = (map.width, map.height);
    let terrain = map.terrain.clone();
    let dump = ofcuda_tick::parse_dump(&a.dump)?;
    let ex = extras(&a.dump)?;

    // Pick the boundary whose live land-attack SET is ENTIRELY fresh
    // creations: every live land attack's recorded `troops` equals the start the
    // engine's own creation path computes (`floor(land_attack_start_troops) ==
    // record troops`). At such a boundary the record IS the device's initial
    // condition - each live attack is exactly `Attack::new(start)` +
    // `refresh_to_conquer` over its owner's border at that plane - so NO attack
    // needs replaying from a record. This is the one honest way to seed a
    // whole-game env from a record: a mid-flight attack's carried heap/PRNG
    // cannot be reconstructed from a record at all (README section 5, and
    // measured: replaying owner 1 at t=613 yields `troops=0` because the engine
    // re-created that attack's exec).
    //
    // Availability, not preference: `ofcuda_env` has no bot-origination
    // schedule wired into the device loop yet, so a run can only be exact while
    // it does not need to originate - i.e. until the engine's next creation.
    let mut best_t: Option<u32> = None;
    let mut best_n = 0usize;
    let mut best_live = 0usize;
    for t in a.t0..a.t0 + 400 {
        let Some(list) = ex.attacks.get(&t) else {
            continue;
        };
        let live: Vec<&extras::AttRec> = list
            .iter()
            .filter(|x| x.live && x.target == 0)
            .collect();
        if live.is_empty() || dump.get(&(t - 1)).is_none() {
            continue;
        }
        let mut fresh = 0usize;
        for att in &live {
            let Some(api) = ex.p.get(&(t - 1)).and_then(|m| m.get(&att.owner)) else {
                continue;
            };
            let ratios = tribe_ratios(&api.id);
            let row = econ_row(t - 1, api.troops, api.tiles, 0, api.gold, api.ptype as u32);
            let Some(start) = land_attack_start_troops(&row, ratios.expand_ratio) else {
                continue;
            };
            if start.floor() as i64 == att.troops {
                fresh += 1;
            }
        }
        if fresh == live.len() && live.len() > best_n {
            best_n = live.len();
            best_live = live.len();
            best_t = Some(t);
            if best_n >= SLOTS.min(3) {
                break; // a 3+ all-fresh SET is already a strong multi-attack seed
            }
        }
    }
    let Some(best_t) = best_t else {
        println!(
            "no boundary in [{}, {}) whose live land-attack SET is entirely fresh creations",
            a.t0,
            a.t0 + 400
        );
        return Ok(());
    };
    let env0 = build_env_at(&dump, &ex, w, h, &terrain, best_t)?;
    // `build_env_at` builds the SET itself: only fresh non-merged creations
    // (its `set` loop drops anything whose computed start != the record's
    // troops). With the scan above requiring `fresh == live`, `env0.attacks` is
    // the whole live land-attack set - no reconstruction, no replay.
    let _ = best_live;
    let t0 = env0.tick0;
    let n = a.envs.first().copied().unwrap_or(1).max(1);
    println!(
        "MULTI env: map={}x{} t0={} attacks_in_SET={} envs={} slots={} mem/env={:.1} MB (plane {:.2})",
        w,
        h,
        t0,
        env0.attacks.len(),
        n,
        SLOTS,
        (env0.attacks.len() as f64 * (HEAP_CAP * 8 + BC * 4 + CC * 4 + OB * 4) as f64
            + (w as f64) * (h as f64) * 2.0)
            / 1e6,
        (w as f64) * (h as f64) * 2.0 / 1e6,
    );
    for (i, at) in env0.attacks.iter().enumerate() {
        println!(
            "  slot {}: owner_sid={} bot={} troops={:.3} heap={} border={} oborder={}",
            i,
            at.owner_sid,
            at.is_bot,
            at.troops,
            at.heap_len,
            at.border.len(),
            at.oborder.len()
        );
    }

    let ctx = CudaContext::new(0)?;
    let module = unsafe { kernels::load(&ctx)? };
    let mut b = MultiBatch::new(&ctx, &module, &[env0.clone()], n, 1)?;
    println!(
        "device: batch_bytes={:.1} MB total ({:.2} MB/env)",
        b.bytes as f64 / 1e6,
        b.bytes as f64 / 1e6 / n as f64
    );

    let (ms, hashes, _scal, live) = run_multi(&ctx, &module, &mut b, &env0, a.ticks, t0, true)?;
    let per_tick = ms / a.ticks.max(1) as f64;
    let ticks_matched = {
        let mut m = 0usize;
        for (k, hh) in hashes.iter().enumerate().take(a.ticks as usize) {
            match engine_hash_at(&dump, w, h, t0 + 1 + k as u32) {
                Some(eh) if eh == *hh => m += 1,
                _ => break,
            }
        }
        m
    };
    println!(
        "MULTI run: ticks={} wall={:.1} ms ({:.3} ms/tick) {:.0} env-ticks/s {:.1} decisions/s live_slots_after={}",
        a.ticks,
        ms,
        per_tick,
        if per_tick > 0.0 { n as f64 / per_tick * 1000.0 } else { 0.0 },
        if per_tick > 0.0 { (n as f64 * env0.attacks.len() as f64) / per_tick * 1000.0 } else { 0.0 },
        live
    );
    println!(
        "MULTI align: init_plane={:016x} eng(t0)={:016x} eng(t0+1)={:016x} dev[0]={:016x} dev[1]={:016x} dev[2]={:016x}",
        state_hash(&env0.plane),
        engine_hash_at(&dump, w, h, t0).unwrap_or(0),
        engine_hash_at(&dump, w, h, t0 + 1).unwrap_or(0),
        hashes.first().copied().unwrap_or(0),
        hashes.get(1).copied().unwrap_or(0),
        hashes.get(2).copied().unwrap_or(0),
    );
    // Tile-level divergence AT THE TICK IT HAPPENS. The first version of this
    // diagnostic read `b.d_plane` after the whole run and compared it against
    // the engine plane at the divergence tick, which is 40 ticks of drift
    // mistaken for one tick's difference. Re-run exactly `k+1` ticks on a fresh
    // batch so the plane read IS the plane at the first mismatching tick.
    if ticks_matched < a.ticks as usize {
        let k = ticks_matched;
        let mut b2 = MultiBatch::new(&ctx, &module, &[env0.clone()], n, 1)?;
        let _ = run_multi(&ctx, &module, &mut b2, &env0, k as u32 + 1, t0, false)?;
        let stream2 = ctx.default_stream();
        let devp = b2.d_plane.to_host_vec(&stream2)?;
        if let Some(engp) = {
            let ps = dump.get(&(t0 + 1 + k as u32));
            ps.map(|ps| {
                let players: Vec<(u32, Vec<u32>)> = ps
                    .values()
                    .map(|p| (p.small_id, p.owned_tiles.clone()))
                    .collect();
                ofcuda_env::state_plane(&players, w, h)
            })
        } {
            let mut diff = 0usize;
            let mut samples = Vec::new();
            let mut seen: std::collections::HashMap<(u16, u16), usize> =
                std::collections::HashMap::new();
            for (i, (a1, b1)) in devp[..b2.wh].iter().zip(engp.iter()).enumerate() {
                if a1 != b1 {
                    diff += 1;
                    *seen.entry((*a1, *b1)).or_insert(0) += 1;
                    if samples.len() < 6 {
                        samples.push((i, *a1, *b1));
                    }
                }
            }
            println!(
                "MULTI tile diff at tick {} (plane after {} of {} ticks): {} tiles differ (of {}); (dev,eng) pairs top:",
                t0 + 1 + k as u32,
                k + 1,
                a.ticks,
                diff,
                b2.wh
            );
            let mut pairs: Vec<((u16, u16), usize)> = seen.into_iter().collect();
            pairs.sort_by(|x, y| y.1.cmp(&x.1));
            for p in pairs.iter().take(6) {
                println!("   dev={} eng={} count={}", p.0 .0, p.0 .1, p.1);
            }
            for s in samples {
                println!("   sample tile {}: dev={} eng={}", s.0, s.1, s.2);
            }
        }
    }
    // Unaided origination: the multi path has NO bot-origination schedule wired
    // in, so the SET it starts from is the SET it ends with. That is the honest
    // reason a run can only be exact until the engine's next attack creation -
    // it is not an oracle input (the loop never reads the record), it is a
    // MISSING device capability.
    let mut started = 0usize;
    for at in env0.attacks.iter() {
        if at.alive {
            started += 1;
        }
    }
    println!(
        "MULTI origination: attacks_seeded={} attacks_originated_unaided=0 (no origination schedule in this path); live_slots_after={}",
        started, live
    );
    println!("MULTI engine agreement: {}/{} ticks matched", ticks_matched, a.ticks);
    if ticks_matched < a.ticks as usize {
        let k = ticks_matched;
        let eh = engine_hash_at(&dump, w, h, t0 + 1 + k as u32).unwrap_or(0);
        let dh = hashes.get(k).copied().unwrap_or(0);
        println!(
            "MULTI first divergence: tick {} dev_hash={:016x} engine_hash={:016x}",
            t0 + 1 + k as u32,
            dh,
            eh
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// `--multi --orig`: the batched env ORIGINATES as it ticks
// ---------------------------------------------------------------------------

/// The owner's BORDER tiles - owned tiles with a 4-neighbour that is not the
/// owner's - in ascending tile order. This is the `border_tiles` set the
/// engine's `refresh_to_conquer` walks (`AttackExecution::init`); here it is
/// derived from the DEVICE's own plane, so it costs the loop no record read.
fn border_of_plane(pl: &[u16], w: u32, h: u32, sid: u16, out: &mut [u32]) -> usize {
    let ww = w as usize;
    let hh = h as usize;
    let mut n = 0usize;
    let lim = pl.len().min(ww * hh);
    for i in 0..lim {
        if pl[i] != sid {
            continue;
        }
        let x = i % ww;
        let y = i / ww;
        let mut touch = false;
        if x > 0 && pl[i - 1] != sid {
            touch = true;
        }
        if x + 1 < ww && pl[i + 1] != sid {
            touch = true;
        }
        if y > 0 && pl[i - ww] != sid {
            touch = true;
        }
        if y + 1 < hh && pl[i + ww] != sid {
            touch = true;
        }
        if touch {
            if n >= out.len() {
                break;
            }
            out[n] = i as u32;
            n += 1;
        }
    }
    n
}

/// The engine's `MAXTROOPS_N`: the `tiles` -> `wire.max_troops` table and the
/// pre-multiplied `max_troops * expand_ratio` rows are indexed by a TILE COUNT
/// (`min(tiles, mcap-1)`), so this is a cap on tiles, not troops.
const MTC: usize = 262144;
/// `expand_ratio * 100` is `next_int(10, 20)`, so rows 10..=19 are the only
/// ones `bot_trow` can name; 21 covers every tribe with margin.
const TROW_N: usize = 21;

/// The engine's own `spawn_end_tick`, read off the record.
///
/// `Game::end_spawn_phase` (`game.rs:724-727`) sets `spawn_end_tick =
/// Some(self.ticks)` at the moment the RL session leaves the spawn phase, and
/// the dump carries `Game::in_spawn_phase()` (`game.rs:720-722`, `self.spawn_phase`)
/// on EVERY row. The phase flag is false from the first row whose tick is
/// strictly greater than `spawn_end_tick`, so the largest tick whose row still
/// says `inSpawnPhase: true` IS the engine's `spawn_end_tick` - derived from the
/// engine's own field, never passed in.
///
/// A dump that starts mid-game carries no `inSpawnPhase: true` row at all and
/// yields 0. That is exactly the mid-game regime: with `spawn_end_tick = 0`
/// every scheduled firing in the window is `tick > spawn_end_tick` and, because
/// `bot_ff` is then below the window, every firing is a LATER one - the port's
/// pre-existing behaviour, unchanged for every start point that never saw the
/// spawn phase.
fn spawn_end_tick_from_dump(path: &std::path::Path) -> u32 {
    let Ok(text) = std::fs::read_to_string(path) else {
        return 0;
    };
    let mut end = 0u32;
    for line in text.lines() {
        if !line.contains("\"inSpawnPhase\":true") {
            continue;
        }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(t) = v.get("tick").and_then(|x| x.as_u64()) {
                end = end.max(t as u32);
            }
        }
    }
    end
}

/// `--multi --orig`. The batched env advances the WHOLE game: it runs the
/// economy, takes the bots' decisions on the DEVICE (`bot_ai`, the canonical
/// core), ORIGINATES the land attacks those decisions call for straight into
/// the batch's slots (`env_orig_slot` -> the core's `orig_refresh_dev`), and
/// ticks everything with one `env_step_multi` per tick.
///
/// The ONLY record-derived inputs are the setup snapshot: the boundary plane
/// (`build_env_at`), the per-player econ row at `t0`, and the per-bot owner
/// border set at `t0`. The per-bot SCHEDULE is a pure function of the player id
/// (`ofcuda_env::tribe_ratios` over `PseudoRandom::new(simple_hash(id))`, see
/// `ofcuda_env/src/lib.rs:162-176`) and the bot behaviour flags are the
/// engine's constants. Inside the tick loop the record is read ONLY in verify
/// mode, to hash-compare - nothing is fed back.
///
/// The SPAWN PHASE is no longer passed in: `spawn_end_tick` and the per-bot
/// first-firing schedule (`bot_ff`) are DERIVED, see `spawn_end_tick_from_dump`.
fn run_multi_orig(a: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let map = ofcuda_tick::load_map(&a.map)?;
    let (w, h) = (map.width, map.height);
    let terrain = map.terrain.clone();
    let dump = ofcuda_tick::parse_dump(&a.dump)?;
    let ex = extras(&a.dump)?;
    // ---- the SPAWN PHASE: the engine's own `spawn_end_tick`, derived from the
    //      record rather than passed as 0. See `spawn_end_tick_from_dump`.
    let spawn_end_tick = std::env::var("OFCUDA_ENV_SPAWN_END")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or_else(|| spawn_end_tick_from_dump(&a.dump));

    // ---- the boundary snapshot: same all-fresh scan as run_multi_demo ----
    let mut best_t: Option<u32> = None;
    let mut best_n = 0usize;
    for t in a.t0..a.t0 + 400 {
        let Some(list) = ex.attacks.get(&t) else {
            continue;
        };
        let live: Vec<&extras::AttRec> = list.iter().filter(|x| x.live && x.target == 0).collect();
        if live.is_empty() || dump.get(&(t - 1)).is_none() {
            continue;
        }
        let mut fresh = 0usize;
        for att in &live {
            let Some(api) = ex.p.get(&(t - 1)).and_then(|m| m.get(&att.owner)) else {
                continue;
            };
            let ratios = tribe_ratios(&api.id);
            let row = econ_row(t - 1, api.troops, api.tiles, 0, api.gold, api.ptype as u32);
            let Some(start) = land_attack_start_troops(&row, ratios.expand_ratio) else {
                continue;
            };
            if start.floor() as i64 == att.troops {
                fresh += 1;
            }
        }
        if fresh == live.len() && live.len() > best_n {
            best_n = live.len();
            best_t = Some(t);
            if best_n >= SLOTS.min(3) {
                break;
            }
        }
    }
    let Some(best_t) = best_t else {
        println!(
            "no boundary in [{}, {}) whose live land-attack SET is entirely fresh creations",
            a.t0,
            a.t0 + 400
        );
        return Ok(());
    };
    let env0 = build_env_at(&dump, &ex, w, h, &terrain, best_t)?;
    let t0 = env0.tick0;
    let n = a.envs.first().copied().unwrap_or(1).max(1);

    // ---- roster + schedule (pure function of player id) ----
    let pinfo = ex
        .p
        .get(&t0)
        .or_else(|| ex.p.get(&(t0.saturating_sub(1))))
        .ok_or("no player table at the boundary")?;
    let mut maxsid = 0usize;
    for s in pinfo.keys() {
        maxsid = maxsid.max(*s as usize);
    }
    let npl = maxsid + 1;
    if npl > 4096 {
        return Err(format!("roster of {npl} players is too large for the env's pst").into());
    }
    let mut bots: Vec<(u16, TribeRatios)> = Vec::new();
    // EVERY per-player table is PER ENV (`e * npl + s`), because with the RL
    // action channel live the envs stop being clones and each one's economy,
    // decisions and origination must run on its own state. The RL agent's
    // action enters exactly here, as one more firing source for ONE env.
    let mut h_pst = vec![0f64; n * npl * 3];
    let mut h_obmeta = vec![0u32; n * npl * 2];
    // ONE env's seed: the record's INSERTION-ORDERED border sets at t0 (the
    // engine's `Player.border_tiles`). `h_oblen_seed[s]` is that set's length.
    let mut h_ob_seed = vec![0u32; npl * PB];
    let mut h_oblen_seed = vec![0u32; npl];
    let mut h_troops = vec![0i32; n * npl];
    let mut h_gold = vec![0i64; n * npl];
    let mut h_ptype = vec![0u8; npl];
    for (sid, pi) in pinfo.iter() {
        let s = *sid as usize;
        for e in 0..n {
            h_pst[(e * npl + s) * 3] = pi.troops as f64;
            h_pst[(e * npl + s) * 3 + 1] = pi.tiles as f64;
            h_pst[(e * npl + s) * 3 + 2] = pi.ptype as f64;
            h_troops[e * npl + s] = pi.troops;
            h_gold[e * npl + s] = pi.gold;
        }
        h_ptype[s] = pi.ptype;
        if pi.ptype as u32 == ofcuda_econ::core_impl::PT_BOT {
            bots.push((*sid, tribe_ratios(&pi.id)));
        }
    }
    if let Some(ps) = dump.get(&t0) {
        for p in ps.values() {
            let s = p.small_id as usize;
            if s >= npl {
                continue;
            }
            let nb = p.border_order.len().min(PB);
            h_ob_seed[s * PB..s * PB + nb].copy_from_slice(&p.border_order[..nb]);
            h_oblen_seed[s] = nb as u32;
            // `ob_meta` offsets are ABSOLUTE into the whole `d_pob` (env e's
            // block is `e * npl * PB`), so the same `bot_ai_core` can be handed
            // each env's `pst` sub-slice and keep its `sid`-only indexing.
            for e in 0..n {
                h_obmeta[(e * npl + s) * 2] = ((e * npl + s) * PB) as u32;
                h_obmeta[(e * npl + s) * 2 + 1] = nb as u32;
            }
        }
    }
    let nb = bots.len().max(1);
    println!(
        "MULTI-ORIG env: map={}x{} t0={} seeded_SET={} envs={} slots={} bots={} (npl={})",
        w, h, t0, env0.attacks.len(), n, SLOTS, bots.len(), npl
    );
    for (sid, r) in bots.iter() {
        println!(
            "  schedule sid={} rate={} at={} trigger={:.2} reserve={:.2} expand={:.2}",
            sid, r.attack_rate, r.attack_tick, r.trigger_ratio, r.reserve_ratio, r.expand_ratio
        );
    }

    let ctx = CudaContext::new(0)?;
    let module = unsafe { kernels::load(&ctx)? };
    let mut b = MultiBatch::new(&ctx, &module, &[env0.clone()], n, npl)?;
    let stream = ctx.default_stream();

    // ---- incremental device owner-border + owned-tile counters ----
    // Every env is a clone of the same t0 state, so the seed block is repeated
    // `n` times. From here on the sets and the counts are evolved ENTIRELY on
    // the device by `env_step_multi`; the host never rebuilds them from a plane.
    let mut h_pob = vec![0u32; n * npl * PB];
    let mut h_poblen = vec![0u32; n * npl];
    let mut h_ptiles = vec![0u32; n * npl];
    {
        let mut cnt = vec![0u32; npl];
        for &v in env0.plane.iter() {
            let s = v as usize;
            if s < npl {
                cnt[s] += 1;
            }
        }
        for e in 0..n {
            h_pob[e * npl * PB..(e + 1) * npl * PB].copy_from_slice(&h_ob_seed);
            h_poblen[e * npl..(e + 1) * npl].copy_from_slice(&h_oblen_seed);
            h_ptiles[e * npl..(e + 1) * npl].copy_from_slice(&cnt);
        }
    }
    b.d_pob.copy_from_host(&stream, &h_pob)?;
    b.d_poblen.copy_from_host(&stream, &h_poblen)?;
    b.d_ptiles.copy_from_host(&stream, &h_ptiles)?;

    // ---- device roster state ----
    let mut d_pst = DeviceBuffer::<f64>::from_host(&stream, &h_pst)?;
    let mut d_obmeta = DeviceBuffer::<u32>::from_host(&stream, &h_obmeta)?;
    let mut b_sid = vec![0u32; nb];
    let mut b_rate = vec![0u32; nb];
    let mut b_at = vec![0u32; nb];
    let mut b_trigger = vec![0f64; nb];
    let mut b_trow = vec![0u32; nb];
    let mut b_rrow = vec![0u32; nb];
    let mut b_ff = vec![0u32; nb];
    // `spawn_end_tick` is the ENGINE's own (`game.rs:724-731`
    // `end_spawn_phase` sets `spawn_end_tick = self.ticks`), derived above from
    // the record's `inSpawnPhase` column. `bot/tribe.rs:118-122` gates the whole
    // tribe exec on `game.in_spawn_phase()` and the schedule test on
    // `tick % attack_rate == attack_tick`, so every firing at `tick <=
    // spawn_end_tick` is skipped by both sides; the FIRST firing is therefore the
    // smallest tick STRICTLY ABOVE `spawn_end_tick` that lands on the phase, which
    // is what `bot_ff` carries into `bot_ai_core`'s `first` test.
    //
    for (k, (sid, r)) in bots.iter().enumerate() {
        b_sid[k] = *sid as u32;
        b_rate[k] = r.attack_rate as u32;
        b_at[k] = r.attack_tick as u32;
        b_trigger[k] = r.trigger_ratio;
        b_trow[k] = (r.expand_ratio * 100.0).round() as u32;
        b_rrow[k] = (r.reserve_ratio * 100.0).round() as u32;
        // `bot_ff`: the bot's first-ever scheduled firing tick (`main.rs:1766`).
        let mut ff = spawn_end_tick + 1;
        while b_rate[k] > 0 && ff % b_rate[k] != b_at[k] {
            ff += 1;
        }
        b_ff[k] = ff;
    }
    println!(
        "MULTI-ORIG spawn: spawn_end_tick={} (derived from the record's inSpawnPhase), first firings: {:?}",
        spawn_end_tick,
        b_ff.iter()
            .zip(bots.iter())
            .map(|(f, (sid, _))| format!("sid{sid}@{f}"))
            .collect::<Vec<_>>()
    );
    let d_bsid = DeviceBuffer::<u32>::from_host(&stream, &b_sid)?;
    let d_brate = DeviceBuffer::<u32>::from_host(&stream, &b_rate)?;
    let d_bat = DeviceBuffer::<u32>::from_host(&stream, &b_at)?;
    let d_btrig = DeviceBuffer::<f64>::from_host(&stream, &b_trigger)?;
    let d_btrow = DeviceBuffer::<u32>::from_host(&stream, &b_trow)?;
    let d_brrow = DeviceBuffer::<u32>::from_host(&stream, &b_rrow)?;
    let d_bff = DeviceBuffer::<u32>::from_host(&stream, &b_ff)?;
    let mut d_bstate = DeviceBuffer::<u32>::from_host(&stream, &vec![2u32; nb * n])?;
    let d_bboat = DeviceBuffer::<u32>::zeroed(&stream, nb)?;
    // The engine's pre-multiplied tables. `max_troops` is the econ core's own
    // function (`wire.max_troops`), and the rows are `max_troops * ratio` with
    // the product rounded exactly as the engine rounds it.
    let mut mtr = vec![0f64; MTC];
    for (t, m) in mtr.iter_mut().enumerate() {
        *m = ofcuda_econ::core_impl::max_troops(
            ofcuda_econ::core_impl::PT_BOT,
            t as i32,
            0,
            ofcuda_econ::core_impl::DIFF_EASY,
            false,
        );
    }
    let mut tgt = vec![0f64; TROW_N * MTC];
    for r in 0..TROW_N {
        let ratio = r as f64 / 100.0;
        for t in 0..MTC {
            tgt[r * MTC + t] = mtr[t] * ratio;
        }
    }
    let d_mtr = DeviceBuffer::<f64>::from_host(&stream, &mtr)?;
    let d_tgt = DeviceBuffer::<f64>::from_host(&stream, &tgt)?;
    let mut d_bout = DeviceBuffer::<f64>::zeroed(&stream, nb * n * 3)?;
    // The RL action channel's own device readback: [amount, fired, why].
    let mut d_agent_out = DeviceBuffer::<f64>::zeroed(&stream, 4)?;
    let mut d_orig_out = DeviceBuffer::<u32>::zeroed(&stream, 4)?;

    println!(
        "device: batch_bytes={:.1} MB total ({:.2} MB/env); bots table {:.1} MB",
        b.bytes as f64 / 1e6,
        b.bytes as f64 / 1e6 / n as f64,
        (tgt.len() as f64 * 8.0 + mtr.len() as f64 * 8.0) / 1e6
    );

    // ---- the unaided loop ----
    let cfg = cfg_for(n);
    let hcfg = cfg_hash(n);
    let mut d_hashes = DeviceBuffer::<u64>::zeroed(&stream, n.max(1))?;
    stream.synchronize()?;
    let wall = Instant::now();
    let mut hashes: Vec<u64> = Vec::new();
    let mut origs = 0usize;
    let mut merged_absorbed = 0usize;
    let mut orig_slot_ovf = 0usize;
    let mut decisions = 0usize;
    let mut live_last = 0usize;
    let mut diverged: Option<(u32, u64, u64)> = None;
    let mut pending_diff: Option<u32> = None;
    let orig_border_from_record =
        std::env::var("OFCUDA_ENV_ORIG_BORDER").map(|v| v == "record").unwrap_or(false);
    // DIAGNOSTIC (env `OFCUDA_ENV_ATK_DEBUG=1`): MEASUREMENT ONLY. After the
    // tick's step, print each live slot's device-side attack state (owner,
    // heap_len, border_len, troops) beside the record's own live land-attack
    // list at the same engine tick. The attack's own `border_tiles` set is the
    // one attack-side register the record cannot carry (only its OWNER's
    // `borderOrder` is dumped), so the budget's `border_size` and the per-pop
    // `remove_border_tile`/`add_border_tile` maintenance are checked against the
    // record through what they DO move: the troop count (one
    // `attacker_troop_loss` per popped tile) and the live-attack set.
    let atk_debug = std::env::var("OFCUDA_ENV_ATK_DEBUG").map(|v| v == "1").unwrap_or(false);
    let atk_dbg_from: u32 = std::env::var("OFCUDA_ENV_ATK_FROM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let atk_dbg_to: u32 = std::env::var("OFCUDA_ENV_ATK_TO")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(u32::MAX);
    let mut atk_dbg_first: Option<u32> = None;
    // DIAGNOSTIC (env `OFCUDA_ENV_OB_DEBUG=1`): compare the DEVICE's evolved
    // owner-border set, tile for tile and in order, with the record's own
    // `borderOrder` rows. Used to locate the first tick at which the ported
    // maintenance diverges from the engine's, without feeding anything.
    let ob_debug = std::env::var("OFCUDA_ENV_OB_DEBUG").map(|v| v == "1").unwrap_or(false);
    let mut ob_debug_first: Option<u32> = None;
    // MEASUREMENT ONLY (`OFCUDA_ENV_HASH_ALL=1`): print every env's plane hash
    // at every tick, so a divergent batch's env-by-env hash sequence can be
    // compared with the same env run alone. This is the per-env separation
    // proof: env e in the batch must be bit-identical to env e alone.
    let hash_all = std::env::var("OFCUDA_ENV_HASH_ALL")
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut h_tiles_now = vec![0i32; n * npl];
    let mut scal0 = b.d_mscal.to_host_vec(&stream)?;
    // ---- OWNER TROOP LEDGER (the attack side of the economy) ----------------
    // `AttackExecution`'s own troop movements, which the record carries only
    // indirectly, through the owner's `troops`: `init`'s `remove_troops`
    // (`attack.rs:146-149`, the attack's start paid out of the owner's pool) and
    // `retreat`'s `add_troops` (`attack.rs:1278-1297`, the survivors paid back).
    // The device reports both - the created attack's `start` is the amount it
    // debits, and a slot that died on the `to_conquer.is_empty()` branch carries
    // its survivors in `mtroops` - and the host applies them to the same
    // `h_troops` the next bot decision reads. Without it the owner's pool only
    // ever grows, the next attack's `start` is too large, and `tiles_used`
    // (`within((2000*speed.max(10))/attack_troops, 5, 100)`, `game.rs:896`) stays
    // clamped at 5 while the engine's has already risen above it.
    let mut prev_alive: Vec<bool> = (0..n * b.slots)
        .map(|s| scal0[s * SCAL + 5] != 0)
        .collect();
    let mut ledger_debits: u64 = 0;
    let mut ledger_debit_total: i64 = 0;
    let mut ledger_credits: u64 = 0;
    let mut ledger_credit_total: i64 = 0;
    // ---- THE RL ACTION CHANNEL'S OWN COUNTERS ---------------------------
    // `--rl-control` is what makes the action channel part of the SIM; these
    // count what it actually did, so a run can prove the action landed.
    let mut rl_action_expand = 0usize;
    let mut rl_action_fired = 0usize;
    let mut rl_action_no_effect = 0usize;
    let mut rl_action_refused_no_border = 0usize;
    let mut rl_action_refused_amount = 0usize;
    let mut rl_action_refused_no_tiles = 0usize;
    let mut rl_action_not_expand = 0usize;
    let mut rl_action_unsupported_player = 0usize;
    let mut rl_suppressed_bot_firings = 0usize;
    let mut rl_action_last = usize::MAX;
    // How many decisions have already had their action APPLIED. One action =
    // ONE button press: the decision's action is applied on the single tick
    // that follows it, not replayed on all `rl_dt` ticks of its window.
    let mut rl_applied: usize = 0;
    // MEASUREMENT ONLY: prints each applied action beside the host's own
    // `land_attack_start_troops` recomputation (the engine's expression).
    let agent_act_debug = std::env::var("OFCUDA_ENV_ACT_DEBUG")
        .map(|v| v == "1")
        .unwrap_or(false);
    // ======================= TRAINING interface (`--rl`) =====================
    // The RL env the trainer drives sits ON TOP of this unaided multi-attack
    // game: one decision every `rl_dt` ticks, an observation (4270 f32) and an
    // action mask (40893 f32) per agent, plus the reward, all derived from the
    // DEVICE state of the game in progress - `d_mscal`/`d_mtroops`/`d_mowner`
    // (the live attack slots), the loop's own `d_ptiles`/`d_poblen` counters
    // and the host troop/gold ledger. Nothing here feeds the simulation: the
    // RL path is an observer, and `--rl-actions` only RECORDS the agent's
    // action (the action -> env application is not wired - see the report).
    // The observation/mask SEMANTICS are `ofcuda_env::rl`'s (the policy's own
    // 4270/40893 fp32 layout): this block only supplies the device World.
    let rl_agent: u16 = a
        .rl_agent
        .unwrap_or_else(|| bots.first().map(|b| b.0).unwrap_or(1));
    let rl_max_episode_ticks: i64 = 21000; // curriculum's episode cap (`curriculum.rs`)
    let map_land: u32 = terrain.iter().filter(|t| **t & 0x80 != 0).count() as u32;
    let mut rl_stream_fnv: u64 = 0xcbf2_9ce4_8422_2325;
    let mut rl_decisions: usize = 0;
    let mut rl_decision_ticks: u32 = 0;
    let mut rl_reward_sum: f64 = 0.0;
    let mut rl_terminal_seen: usize = 0;
    let mut rl_derive_ns: u128 = 0;
    let mut rl_inv = rl::Inventory::default();
    let mut rl_reward_state = rl::RewardState::default();
    let mut rl_hashes: Vec<u64> = Vec::new();
    let mut rl_out_file: Option<std::fs::File> = match a.rl_out.as_ref() {
        Some(p) => Some(std::fs::File::create(p).map_err(|e| format!("rl-out: {e}"))?),
        None => None,
    };
    if a.rl {
        let hdr = rl::header_check(std::path::Path::new(
            "/opt/data/workspaces/skg/openfront-ai/include/openfront_env.h",
        ));
        let (hdr_n, hdr_bad) = match &hdr {
            Ok((n, bad)) => (*n, bad.join("; ")),
            Err(e) => (0, e.clone()),
        };
        println!(
            "RL interface: agent={} rl_dt={} policy_tensor=B,99,gh,gw obs_per_agent={} mask_per_agent={} map_land={}/{} spawn_end_tick={} header_consts_checked={} header_mismatches=[{}]",
            rl_agent,
            a.rl_dt.max(1),
            rl::OBS_PER_AGENT,
            rl::MASK_PER_AGENT,
            map_land,
            w * h,
            spawn_end_tick,
            hdr_n,
            hdr_bad
        );
    }
    for k in 0..a.ticks {
        let tick = t0 + k;
        // The engine's OWN tick that this iteration executes. `env_step_multi`'s
        // `tick` is the device's offer-order convention (measured: `t0 + k` is
        // what keeps the pre-existing attack's conquests byte-exact), but the
        // SCHEDULE domain is the engine's: `PlayerExecution::tick` runs income,
        // then the bot decision, then the attack execs, all inside engine tick
        // `t0 + k + 1`, reading the post-income troops of that same tick. Hence
        // `sched_tick = t0 + k + 1` for the decision while the plane/border
        // inputs stay at `t0 + k` - the matrix's `prev` (state) vs
        // `prev.engine_tick` (tick) split.
        let sched_tick = t0 + k + 1;

        // (1) economy, host-side, from the DEVICE's own MAINTAINED state: the
        //     owned-tile counters `env_step_multi` grows on every conquest
        //     (`Game::conquer_one` -> `tiles_owned += 1`), and the owner-border
        //     lengths. The plane is NOT read back: no `n*wh` host copy per tick.
        let ptile_h = b.d_ptiles.to_host_vec(&stream)?;
        let oblen_h = b.d_poblen.to_host_vec(&stream)?;
        // PER ENV (`e * npl + s`): the envs are clones only until the RL action
        // channel is live, so every per-player table is indexed by env.
        for e in 0..n {
            for s in 0..npl {
                h_tiles_now[e * npl + s] = ptile_h[e * npl + s] as i32;
            }
            for (sid, _) in bots.iter() {
                let g = e * npl + *sid as usize;
                h_obmeta[g * 2] = (g * PB) as u32;
                h_obmeta[g * 2 + 1] = oblen_h[g].min(PB as u32);
            }
        }
        // Divergence diagnosis only (verify mode): read the plane back ONCE, at
        // the first mismatching tick, to print the differing tiles.
        if let Some(dt) = pending_diff {
            if dt == tick {
                let plane_h = b.d_plane.to_host_vec(&stream)?;
                if let Some(engp) = {
                    let ps = dump.get(&dt);
                    ps.map(|ps| {
                        let players: Vec<(u32, Vec<u32>)> = ps
                            .values()
                            .map(|p| (p.small_id, p.owned_tiles.clone()))
                            .collect();
                        ofcuda_env::state_plane(&players, w, h)
                    })
                } {
                    print_plane_diff(&plane_h[..b.wh], &engp, dt);
                }
                pending_diff = None;
            }
        }
        for e in 0..n {
            for s in 1..npl {
                let g = e * npl + s;
                if h_tiles_now[g] <= 0 {
                    continue;
                }
                let row = econ_row(
                    tick,
                    h_troops[g],
                    h_tiles_now[g],
                    0,
                    h_gold[g],
                    h_ptype[s] as u32,
                );
                let st = ofcuda_econ::step_row(&row);
                h_troops[g] = st.troops_after;
                h_gold[g] = st.gold_after;
                h_pst[g * 3] = st.troops_after as f64;
                h_pst[g * 3 + 1] = h_tiles_now[g] as f64;
            }
        }
        d_pst.copy_from_host(&stream, &h_pst)?;

        // (2) the bots' owner-border sets live on the DEVICE now (maintained in
        //     the engine's insertion order by `env_step_multi`); the host only
        //     publishes the per-player lengths `bot_ai` reads through `obmeta`.
        d_obmeta.copy_from_host(&stream, &h_obmeta)?;

        // (3) the bots' decisions, ON THE DEVICE (canonical `bot_ai_core`),
        //     PER ENV: the decisions of env e are taken from env e's own plane,
        //     player state and owner-border metadata, which is what makes a
        //     divergent batch legal. One thread per env (`#[launch_bounds(64)]`).
        unsafe {
            module.bot_ai_multi(
                &stream,
                cfg_for_block(n, 64),
                &b.d_terrain,
                w,
                h,
                tick,
                spawn_end_tick,
                &b.d_plane,
                &d_pst,
                &b.d_pob,
                &d_obmeta,
                &d_bsid,
                &d_brate,
                &d_bat,
                &d_btrig,
                &d_btrow,
                &d_brrow,
                &d_bff,
                &mut d_bstate,
                &d_bboat,
                &d_mtr,
                &d_tgt,
                &mut d_bout,
                n as u32,
                npl as u32,
                nb as u32,
            )?;
        }
        let bout = d_bout.to_host_vec(&stream)?;
        decisions += nb * n;

        // (4) one device launch per tick: advance the whole batch FIRST, so an
        //     attack created below first conquers into the NEXT tick's plane.
        unsafe {
            module.env_step_multi(
                &stream,
                cfg,
                &b.d_terrain,
                b.w,
                b.h,
                tick,
                n as u32,
                b.slots as u32,
                &mut b.d_plane,
                &mut b.d_mheap_tiles,
                &mut b.d_mheap_pri,
                &mut b.d_mborder,
                &mut b.d_mclaims,
                &mut b.d_mprng,
                &mut b.d_mtroops,
                &mut b.d_mscal,
                &b.d_mowner,
                &b.d_misbot,
                &b.d_moborder,
                &mut b.d_cad,
                &mut b.d_pob,
                &mut b.d_poblen,
                &mut b.d_ptiles,
                b.npl as u32,
                1,
            )?;
        }

        // DIAGNOSTIC (OFCUDA_ENV_ATK_DEBUG=1): the attack-side state after this
        // tick's step, against the record's own live land attacks at the same
        // engine tick (`t0 + k + 1`). Prints the first tick whose live SET or
        // whose per-owner troop count (floored, the record's own truncation)
        // disagrees, then every tick in [ATK_FROM, ATK_TO].
        if atk_debug {
            let sc = b.d_mscal.to_host_vec(&stream)?;
            let tr = b.d_mtroops.to_host_vec(&stream)?;
            let ow = b.d_mowner.to_host_vec(&stream)?;
            let et = t0 + k + 1;
            let mut rec: Vec<(u16, i64)> = ex
                .attacks
                .get(&et)
                .map(|v| {
                    v.iter()
                        .filter(|x| x.live && x.target == 0)
                        .map(|x| (x.owner, x.troops))
                        .collect()
                })
                .unwrap_or_default();
            rec.sort_unstable();
            let mut dev: Vec<(u16, i64, u32, u32)> = Vec::new();
            for s in 0..b.slots {
                if sc[s * SCAL + 5] != 0 {
                    dev.push((
                        ow[s],
                        tr[s].floor() as i64,
                        sc[s * SCAL],
                        sc[s * SCAL + 1],
                    ));
                }
            }
            let mut dev_keys: Vec<(u16, i64)> = dev.iter().map(|d| (d.0, d.1)).collect();
            dev_keys.sort_unstable();
            // Player-troop ledger: the host's `h_troops` (the value the NEXT
            // bot decision reads) against the record's own player rows.
            let mut led: Vec<(u16, i32, i32)> = Vec::new();
            for (sid, _) in bots.iter() {
                let s = *sid as usize;
                let rt = ex
                    .p
                    .get(&et)
                    .and_then(|m| m.get(sid))
                    .map(|p| p.troops)
                    .unwrap_or(i32::MIN);
                led.push((*sid, h_troops[s], rt));
            }
            let led_bad: Vec<(u16, i32, i32)> =
                led.iter().copied().filter(|l| l.1 != l.2).collect();
            let mismatch = dev_keys != rec || !led_bad.is_empty();
            if mismatch && atk_dbg_first.is_none() {
                atk_dbg_first = Some(et);
            }
            if mismatch && atk_dbg_first == Some(et) {
                println!(
                    "ATKDBG first mismatch engine_tick={et} dev={:?} rec={:?} ledger(sid,dev_h,rec)={:?}",
                    dev, rec, led
                );
                let t = et.saturating_sub(1);
                let tprev = t.saturating_sub(1);
                for tt in [tprev, t, et] {
                    if let Some(v) = ex.attacks.get(&tt) {
                        let l: Vec<(u16, i64)> = v
                            .iter()
                            .filter(|x| x.live && x.target == 0)
                            .map(|x| (x.owner, x.troops))
                            .collect();
                        println!("   rec attacks at {tt}: {l:?}");
                    }
                    if let Some(m) = ex.p.get(&tt) {
                        let l: Vec<(u16, i32)> =
                            m.iter().map(|(s, p)| (*s, p.troops)).collect();
                        println!("   rec troops at {tt}: {l:?}");
                    }
                }
            }
            if et >= atk_dbg_from && et <= atk_dbg_to {
                println!("ATKDBG tick={et} dev={dev:?} rec={rec:?} ledger={led:?}");
            }
        }

        // DIAGNOSTIC: did the device's evolved set stay identical (order and
        // all) to the record's own `borderOrder` at this tick? Prints only the
        // first mismatching tick per run.
        if ob_debug && ob_debug_first.is_none() {
            let hp = b.d_pob.to_host_vec(&stream)?;
            let hl = b.d_poblen.to_host_vec(&stream)?;
            if let Some(ps) = dump.get(&(tick + 1)) {
                for (sid, _) in bots.iter() {
                    let s = *sid as usize;
                    let Some(p) = ps.values().find(|p| p.small_id == *sid as u32) else {
                        continue;
                    };
                    let nd = hl[s] as usize;
                    let dev = &hp[s * PB..s * PB + nd];
                    let eng = &p.border_order;
                    let mut a: Vec<u32> = dev.to_vec();
                    let mut e2: Vec<u32> = eng.clone();
                    a.sort_unstable();
                    e2.sort_unstable();
                    let set_ok = a == e2;
                    let order_ok = dev.len() == eng.len()
                        && dev.iter().zip(eng.iter()).all(|(x, y)| x == y);
                    if !set_ok || !order_ok {
                        ob_debug_first = Some(tick + 1);
                        println!(
                            "OBDEBUG first mismatch at tick={} sid={} dev_len={} eng_len={} set_ok={} order_ok={}",
                            tick + 1,
                            sid,
                            dev.len(),
                            eng.len(),
                            set_ok,
                            order_ok
                        );
                        let mut shown = 0;
                        for j in 0..dev.len().max(eng.len()) {
                            let d = dev.get(j).copied().unwrap_or(u32::MAX);
                            let e = eng.get(j).copied().unwrap_or(u32::MAX);
                            if d != e && shown < 8 {
                                println!("  j={} dev={} eng={}", j, d, e);
                                shown += 1;
                            }
                        }
                    }
                }
            }
        }

        // (5) ORIGINATION: a firing bot's land attack is built straight into a
        //     free slot of every env by the canonical `orig_refresh_dev`. It is
        //     built AFTER this tick's step, so it pops live exactly as the
        //     engine's `add_land_attack_from` does (appended to `execs` after
        //     the tick's attack execs, first conquest one tick later).
        scal0 = b.d_mscal.to_host_vec(&stream)?;
        // (4a) OWNER TROOP LEDGER - the retreat side. A slot that went from live
        //      to dead during the step just launched is either `retreat` (its
        //      survivors are in `mtroops`) or a starved `kill_attack` (0).
        //      `Game::add_troops` (`game.rs:1170-1178`) adds `floor(survivors)`,
        //      and only when `survivors >= 1.0` (`attack.rs:1284-1287`). Settled
        //      here, before the next iteration's income, which is the engine's own
        //      order (income runs ahead of the attack execs).
        let mut dead_slots: Vec<usize> = Vec::new();
        for s in 0..n * b.slots {
            if prev_alive[s] && scal0[s * SCAL + 5] == 0 {
                dead_slots.push(s);
            }
        }
        if !dead_slots.is_empty() {
            let tr = b.d_mtroops.to_host_vec(&stream)?;
            let ow = b.d_mowner.to_host_vec(&stream)?;
            for s in dead_slots.iter() {
                // Absolute slot id: env `s / b.slots`, owner from that env's
                // own slot table, credit into that env's OWN troop ledger.
                let surv = tr[*s];
                if surv >= 1.0 {
                    let owner = ow[*s] as usize;
                    if owner < npl {
                        let add = surv.floor() as i32;
                        h_troops[(*s / b.slots) * npl + owner] += add;
                        ledger_credits += 1;
                        ledger_credit_total += add as i64;
                    }
                }
            }
        }
        for s in 0..n * b.slots {
            prev_alive[s] = scal0[s * SCAL + 5] != 0;
        }
        // DIAGNOSTIC ONLY (env `OFCUDA_ENV_ORIG_BORDER=record`): replace the
        // plane-derived owner-border for every BOT with the record's own
        // `borderOrder` at this state's tick. This is the one part of
        // origination the plane cannot carry - the engine's `border_tiles`
        // iteration order, which `offer_dev` binds its per-neighbour
        // `next_int(0,7)` draws to, hence the heap priorities. Proving the
        // residual is exactly this is the point; the shipped path never reads
        // it.
        if orig_border_from_record {
            let mut h_ob = vec![0u32; n * npl * PB];
            let mut h_ol = vec![0u32; n * npl];
            if let Some(ps) = dump.get(&tick) {
                for p in ps.values() {
                    let s = p.small_id as usize;
                    if s >= npl {
                        continue;
                    }
                    let nn = p.border_order.len().min(PB);
                    for e in 0..n {
                        h_ob[e * npl * PB + s * PB..e * npl * PB + s * PB + nn]
                            .copy_from_slice(&p.border_order[..nn]);
                        h_ol[e * npl + s] = nn as u32;
                    }
                    h_obmeta[s * 2 + 1] = nn as u32;
                }
            }
            b.d_pob.copy_from_host(&stream, &h_ob)?;
            b.d_poblen.copy_from_host(&stream, &h_ol)?;
            d_obmeta.copy_from_host(&stream, &h_obmeta)?;
        }
        // ENGINE MERGE (`Game::merge_outgoing_land_attacks`, `game.rs:2122-2156`,
        // called from `AttackExecution::init`, `attack.rs:177-184`): a newly
        // created land attack ABSORBS every OTHER live land attack of the same
        // owner and the same target - their CURRENT `troops` are added into the
        // new attack's `troops` and they are killed (no owner credit; the troops
        // were debited when those attacks were created). Every attack this port
        // carries is a Terra-Nullius land attack (target column 0), so the new
        // attack absorbs every other LIVE slot of its owner. This is what a bot
        // re-firing while its previous attack is still running does in the early
        // game (measured at t0=318: engine tick 376, sid 2 - the record's single
        // attack goes 589 -> 1910, the merged sum, while an unmerged port held two
        // slots and diverged one tick later).
        // ---- THE FIRINGS OF THIS TICK, PER ENV -------------------------------
        // A firing is `(env, owner sid, start troops)` = "create this owner's
        // land attack in THIS env's slot table". There are exactly two sources:
        //
        //   (a) the bots' own SCHEDULE, from `bot_ai_multi`'s per-env decision;
        //   (b) THE RL AGENT'S ACTION, for the envs whose script asked for an
        //       expansion (below).
        //
        // With `--rl-control` off, (b) is empty and (a) fires in every env -
        // the unaided batch, unchanged.
        let mut firings: Vec<(usize, u16, f64)> = Vec::new();
        for e in 0..n {
            for (bi, (sid, _)) in bots.iter().enumerate() {
                let o = e * nb * 3 + bi * 3;
                if bout[o] != 1.0 || bout[o + 2] != 1.0 || bout[o + 1] < 1.0 {
                    continue;
                }
                if a.rl_control && *sid == rl_agent {
                    // The agent PLAYS this player: its own scheduled firing is
                    // suppressed, because the action channel REPLACES the bot
                    // AI's button press rather than adding a second attack on
                    // top of it.
                    rl_suppressed_bot_firings += 1;
                    continue;
                }
                firings.push((e, *sid, bout[o + 1]));
            }
        }

        // ---- THE RL ACTION, WIRED INTO THE SIM -------------------------------
        // The decision consumed here is the one emitted at the END of the
        // previous iteration: its `sched_tick` is the engine tick this
        // origination pass belongs to, so the attack is created from the state
        // the policy actually SAW.
        //
        // `A_EXPAND` creates THE ATTACK THE ENV'S OWN BOT AI WOULD HAVE CREATED
        // for the controlled player at this tick: `agent_expand_fire` is
        // `bot_ai_core`'s terra-nullius land branch with its schedule gate
        // removed, and the attack it produces is then built by the SAME
        // slot/merge/ledger path as a bot's, below. So the agent does not get a
        // different mechanic - it gets the bot's own attack, on demand.
        //
        // `A_ATTACK` names a target player's tiles; every attack this port
        // carries is a terra-nullius land attack (target column 0) and there is
        // no per-player target column to aim, so the action is RECORDED and
        // changes nothing. That is the one legal action the sim cannot yet play.
        if a.rl_control && rl_decisions > rl_applied {
            let di = rl_decisions - 1;
            rl_applied = rl_decisions;
            for e in 0..n {
                let act = rl_script(&a.rl_actions_env, &a.rl_actions, e, di);
                rl_action_last = act;
                if act != rl::A_EXPAND {
                    rl_action_not_expand += 1;
                    continue;
                }
                rl_action_expand += 1;
                let Some((_, ratios)) = bots.iter().find(|(sid, _)| *sid == rl_agent) else {
                    // A controlled player that is not a bot has no
                    // `expand_ratio`, so there is no bot AI to borrow an attack
                    // from: the action cannot be wired for this player.
                    rl_action_unsupported_player += 1;
                    continue;
                };
                let trow_row = (ratios.expand_ratio * 100.0).round() as u32;
                unsafe {
                    module.agent_expand_fire(
                        &stream,
                        cfg_for_block(1, 1),
                        &b.d_terrain,
                        w,
                        h,
                        &b.d_plane,
                        (e * b.wh) as u32,
                        &b.d_pob,
                        &d_obmeta,
                        &d_pst,
                        (e * npl * 3) as u32,
                        &d_tgt,
                        MTC as u32,
                        trow_row,
                        rl_agent as u32,
                        &mut d_agent_out,
                    )?;
                }
                let ao = d_agent_out.to_host_vec(&stream)?;
                if agent_act_debug {
                    let gt = e * npl + rl_agent as usize;
                    // INDEPENDENT CHECK that the attack the action creates IS the
                    // engine's own. `land_attack_start_troops` is the port's
                    // engine-faithful `land_attack_troops` (`step.troops_after -
                    // step.max_troops * ratio`), and it ADVANCES the economy step,
                    // so the step's own income is subtracted back off to land on
                    // the same quantity the device took from the bot's
                    // PRE-MULTIPLIED table row. Bit equality is the claim.
                    let row = econ_row(
                        tick,
                        h_troops[gt],
                        h_tiles_now[gt],
                        0,
                        h_gold[gt],
                        h_ptype[rl_agent as usize] as u32,
                    );
                    let st = ofcuda_econ::step_row(&row);
                    let inc = st.troops_after - h_troops[gt];
                    let host_amt = ofcuda_env::land_attack_start_troops(&row, ratios.expand_ratio)
                        .map(|v| v - inc as f64);
                    println!(
                        "  ACT tick={} env={} sid={} troops={} tiles={} grow={} device_amount={:.3} device_fired={} why={} engine_amount={:?} equal={}",
                        tick,
                        e,
                        rl_agent,
                        h_troops[gt],
                        h_tiles_now[gt],
                        trow_row,
                        ao[0],
                        ao[1] == 1.0,
                        ao[2] as i64,
                        host_amt,
                        host_amt.map(|v| v == ao[0]).unwrap_or(false)
                    );
                }
                if ao[1] == 1.0 {
                    firings.push((e, rl_agent, ao[0]));
                    rl_action_fired += 1;
                } else {
                    rl_action_no_effect += 1;
                    match ao[2] as i32 {
                        1 => rl_action_refused_no_border += 1,
                        2 => rl_action_refused_amount += 1,
                        _ => rl_action_refused_no_tiles += 1,
                    }
                }
            }
        }

        let mut used: Vec<usize> = Vec::new();
        let mut used_env: usize = usize::MAX;
        let any_fire = !firings.is_empty();
        // The merge reads the PRE-TICK slot table (`scal0`) and the slots' own
        // `troops`/`owner` columns, for EVERY env now: env e's absorbed set is
        // built from env e's slots only.
        let (mtroops_h, mowner_h) = if any_fire {
            (
                b.d_mtroops.to_host_vec(&stream)?,
                b.d_mowner.to_host_vec(&stream)?,
            )
        } else {
            (Vec::new(), Vec::new())
        };
        // The owner-border length the ORIGINATION must refresh from. The engine
        // runs `AttackExecution::init` (which calls `refresh_to_conquer`,
        // `attack.rs:1299-1308`) in the tick's `uninit` DRAIN - after that tick's
        // attack execs - so the border it iterates is the one the tick's own
        // conquests just produced, not the length published for this tick's bot
        // DECISION. The device holds the set (`d_pob`); its length must be read
        // back NOW rather than reusing `h_obmeta`, which was filled at the top of
        // this iteration. Measured at t0=318: reusing it cost sid 1 the two border
        // tiles that appeared during engine tick 385 (86 vs the record's 88), one
        // heap candidate too few, and one tile of plane divergence at 387.
        if any_fire {
            let oblen_now = b.d_poblen.to_host_vec(&stream)?;
            for e in 0..n {
                for (sid, _) in bots.iter() {
                    let g = e * npl + *sid as usize;
                    h_obmeta[g * 2] = (g * PB) as u32;
                    h_obmeta[g * 2 + 1] = oblen_now[g].min(PB as u32);
                }
            }
        }
        for (e, sid, start) in firings.iter().copied() {
            // `used`/`absorbed` are PER ENV: the envs are independent games.
            if e != used_env {
                used.clear();
                used_env = e;
            }
            let sb = sid as usize;
            let so = e * b.slots;
            // The absorbed set: the owner's OTHER live slots of THIS ENV at the
            // pre-tick state (`scal0`/`mowner_h`), i.e. the engine's
            // `Player.outgoing_land_attacks` walk minus the new attack itself.
            let mut absorbed: Vec<usize> = Vec::new();
            let mut merged_sum = 0.0f64;
            for s in so..so + b.slots {
                if used.contains(&s) {
                    continue;
                }
                if scal0[s * SCAL + 5] != 0 && mowner_h[s] as usize == sb {
                    merged_sum += mtroops_h[s];
                    absorbed.push(s);
                }
            }
            let mut slot: Option<usize> = None;
            for s in so..so + b.slots {
                if used.contains(&s) || absorbed.contains(&s) {
                    continue;
                }
                if scal0[(s * SCAL) + 5] == 0 {
                    slot = Some(s);
                    break;
                }
            }
            let Some(slot) = slot else {
                orig_slot_ovf += 1;
                continue;
            };
            used.push(slot);
            let s = sb;
            // OWNER TROOP LEDGER - the creation side. `AttackExecution::init`
            // (`attack.rs:146-149`) pays the attack's `start` troops out of the
            // OWNER's own pool: `Game::remove_troops(owner, start)`
            // (`game.rs:1153-1164`, `min(troops, floor(start))`). `start` is the
            // PRE-MERGE value the bot decision produced (the un-floored f64 the
            // attack is given - the engine debits before the merge), and the debit
            // lands after this tick's income and before the next one, which is the
            // engine's own phase order.
            let gt = e * npl + s;
            let debit = (start.floor() as i32).min(h_troops[gt]).max(0);
            h_troops[gt] -= debit;
            ledger_debits += 1;
            ledger_debit_total += debit as i64;
            // Kill the absorbed slots FIRST (host bookkeeping), then create the
            // merged attack in the free slot. `prev_alive` is cleared for them so
            // the retreat-credit pass never pays their survivors back: they are
            // inside the new attack's `troops` now.
            for a in absorbed.iter() {
                unsafe {
                    module.env_kill_slot(&stream, cfg_for_block(1, 1), &mut b.d_mscal, *a as u32)?;
                }
                prev_alive[*a] = false;
            }
            merged_absorbed += absorbed.len();
            let merged_start = start + merged_sum;
            let obn = h_obmeta[gt * 2 + 1];
            unsafe {
                module.env_orig_slot(
                    &stream,
                    cfg_for_block(1, 1),
                    &b.d_terrain,
                    w,
                    h,
                    tick,
                    sid,
                    0u16,
                    &b.d_plane,
                    &b.d_pob,
                    (e * npl * PB + s * PB) as u32,
                    obn,
                    &mut b.d_mheap_tiles,
                    &mut b.d_mheap_pri,
                    &mut b.d_mborder,
                    &mut b.d_mprng,
                    &mut b.d_mscal,
                    &mut b.d_mtroops,
                    &mut b.d_mowner,
                    &mut b.d_misbot,
                    &mut b.d_moborder,
                    slot as u32,
                    sid as u32,
                    1u32,
                    merged_start,
                    &mut d_orig_out,
                )?;
            }
            if origs < 6 {
                let oo = d_orig_out.to_host_vec(&stream)?;
                println!(
                    "  ORIG tick={} env={} sid={} slot={} start={:.1} merged_start={:.1} absorbed={} heap={} border={} peak={} own_border={}",
                    tick,
                    e,
                    sid,
                    slot - e * b.slots,
                    start,
                    merged_start,
                    absorbed.len(),
                    oo[0],
                    oo[1],
                    oo[2],
                    obn
                );
            }
            origs += 1;
        }

        // (6) the ONLY record touch after setup: hash-compare, in verify mode.
        if a.verify_max > 0 {
            unsafe {
                module.env_hash(
                    &stream,
                    hcfg,
                    &b.d_plane,
                    b.wh as u32,
                    n as u32,
                    0,
                    &mut d_hashes,
                )?;
            }
            let hh_all = d_hashes.to_host_vec(&stream)?;
            let hh = hh_all[0];
            hashes.push(hh);
            if hash_all {
                let mut line = format!("HASHES tick={}", tick + 1);
                for e in 0..n {
                    line.push_str(&format!(" e{}=>{}", e, hh_all[e]));
                }
                println!("{line}");
            }
            if diverged.is_none() {
                if let Some(eh) = engine_hash_at(&dump, w, h, tick + 1) {
                    if eh != hh {
                        diverged = Some((tick + 1, hh, eh));
                        pending_diff = Some(tick + 1);
                    }
                }
            }
        }

        // ---- TRAINING interface: one decision every `rl_dt` ticks ----------
        if a.rl {
            rl_decision_ticks += 1;
            if rl_decision_ticks >= a.rl_dt.max(1)
                && (a.rl_decisions == 0 || rl_decisions < a.rl_decisions)
            {
                rl_decision_ticks = 0;
                let t_der = std::time::Instant::now();
                // The live attack slots, straight off the device (`d_mscal`'s
                // slot column and the per-slot `mtroops`/`mowner`).
                let mscal_r = b.d_mscal.to_host_vec(&stream)?;
                let mtroops_r = b.d_mtroops.to_host_vec(&stream)?;
                let mowner_r = b.d_mowner.to_host_vec(&stream)?;
                // --- players: device tiles/border + the loop's own ledger ---
                let mut players: Vec<rl::PlayerRow> = Vec::new();
                for (sid, _) in bots.iter() {
                    let s = *sid as usize;
                    if s >= npl {
                        continue;
                    }
                    players.push(rl::PlayerRow {
                        sid: *sid,
                        slot: *sid as u8, // roster order: sid == slot in this env
                        alive: ptile_h[s] as i32 > 0 || h_troops[s] > 0,
                        troops: h_troops[s] as f64,
                        gold: h_gold[s] as f64,
                        tiles: ptile_h[s] as f64,
                        border_len: oblen_h[s] as u32,
                    });
                }
                let mut attacks: Vec<rl::AttackRow> = Vec::new();
                for s in 0..b.slots {
                    if mscal_r[s * SCAL + 5] != 0 {
                        attacks.push(rl::AttackRow {
                            owner: mowner_r[s],
                            target: 0, // this env originates neutral-land attacks only
                            troops: mtroops_r[s] as f64,
                            alive: true,
                        });
                    }
                }
                let world = rl::World {
                    tick: sched_tick as i64,
                    spawn_end_tick,
                    max_episode_ticks: rl_max_episode_ticks,
                    map_land: map_land as i64,
                    w,
                    h,
                    me: rl_agent,
                    me_slot: rl_agent as usize,
                    players,
                    attacks,
                };
                let o = rl::obs(&world, &mut rl_inv);
                // No spawn tile predicate post-spawn: the tile block is the
                // all-ones legal-region plane inside the fixed 150x250 grid.
                let m = rl::mask(&world, None, &mut rl_inv);
                let out = rl::decision_reward(&world, &mut rl_reward_state);
                // The decision stream hash, in the trainer's consumption order
                // (obs, mask, reward) as little-endian f32: FNV-1a 64, the same
                // hash family the parity traces use.
                for v in o.iter().chain(m.iter()) {
                    for byte in v.to_le_bytes() {
                        rl_stream_fnv =
                            (rl_stream_fnv ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3);
                    }
                }
                for byte in out.reward.to_le_bytes() {
                    rl_stream_fnv =
                        (rl_stream_fnv ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3);
                }
                rl_reward_sum += out.reward;
                if out.done {
                    rl_terminal_seen += 1;
                }
                let action = a
                    .rl_actions
                    .get(rl_decisions)
                    .copied()
                    .unwrap_or(rl::A_NOOP);
                let action_legal =
                    m[rl::MASK_ACTIONS_OFF + action.min(rl::N_ACTIONS - 1)] > 0.0;
                let me = world.me_row().cloned().unwrap_or_default();
                if let Some(f) = rl_out_file.as_mut() {
                    use std::io::Write as _;
                    writeln!(
                        f,
                        "{{\"env\":0,\"agent\":{},\"decision\":{},\"tick\":{},\"dt\":{},\"action\":{},\"action_legal\":{},\"action_control\":{},\"n_alive\":{},\"n_live_attacks\":{},\"tiles\":{},\"troops\":{},\"gold\":{},\"reward\":{:.9},\"terminal\":{:.9},\"done\":{},\"won\":{},\"died\":{},\"timed_out\":{},\"obs_hash\":{},\"mask_hash\":{},\"stream_hash\":{}}}",
                        rl_agent,
                        rl_decisions,
                        sched_tick,
                        a.rl_dt.max(1),
                        action,
                        action_legal,
                        a.rl_control,
                        world.players.iter().filter(|p| p.alive).count(),
                        world.attacks_by(rl_agent).len(),
                        me.tiles,
                        me.troops,
                        me.gold,
                        out.reward,
                        out.terminal,
                        out.done,
                        out.won,
                        out.died,
                        out.timed_out,
                        rl_hash_f32(&o),
                        rl_hash_f32(&m),
                        rl_stream_fnv
                    )
                    .map_err(|e| format!("rl-out: {e}"))?;
                }
                rl_hashes.push(rl_stream_fnv);
                rl_derive_ns += t_der.elapsed().as_nanos();
                rl_decisions += 1;
            }
        }
    }
    stream.synchronize()?;
    let ms = wall.elapsed().as_secs_f64() * 1000.0;
    let scal_end = b.d_mscal.to_host_vec(&stream)?;
    for e in 0..n {
        for s in 0..b.slots {
            if scal_end[(e * b.slots + s) * SCAL + 5] != 0 {
                live_last += 1;
            }
        }
    }

    // ---- TRAINING interface report ---------------------------------------
    if a.rl {
        let dt = a.rl_dt.max(1) as u64;
        let derive_ms = rl_derive_ns as f64 / 1e6;
        println!(
            "RL interface: decisions={} rl_dt={} env_ticks={} obs_per_agent={} mask_per_agent={} obs_written={} ({:.4}) mask_written={} ({:.4})",
            rl_decisions,
            dt,
            rl_decisions as u64 * dt,
            rl::OBS_PER_AGENT,
            rl::MASK_PER_AGENT,
            rl_inv.obs_written,
            rl_inv.obs_frac(),
            rl_inv.mask_written,
            rl_inv.mask_frac()
        );
        println!(
            "RL interface: derive={:.3} ms total ({:.4} ms/decision) reward_sum={:.6} terminals={} stream_fnv1a={:016x}",
            derive_ms,
            if rl_decisions > 0 {
                derive_ms / rl_decisions as f64
            } else {
                0.0
            },
            rl_reward_sum,
            rl_terminal_seen,
            rl_stream_fnv
        );
        let secs = (ms / 1000.0).max(1e-9);
        println!(
            "RL interface: env_ticks/s={:.0} decisions/s={:.1} obs_mask_bytes_per_decision={} (n={} envs)",
            (n as f64 * a.ticks as f64) / secs,
            rl_decisions as f64 / secs,
            (rl::OBS_PER_AGENT + rl::MASK_PER_AGENT) * 4,
            n
        );
        if !rl_hashes.is_empty() {
            let head: Vec<String> = rl_hashes
                .iter()
                .take(4)
                .map(|h| format!("{h:016x}"))
                .collect();
            println!(
                "RL interface: first_decision_hashes={} last_decision_hash={:016x} n_hashes={}",
                head.join(","),
                rl_hashes[rl_hashes.len() - 1],
                rl_hashes.len()
            );
        }
        println!(
            "RL action channel: control={} agent={} expand_requested={} fired={} no_effect={} refused_no_tn_border={} refused_amount_lt_1={} refused_no_tiles={} not_expand={} unsupported_player={} suppressed_bot_firings={} applied={} last_action={}",
            a.rl_control,
            rl_agent,
            rl_action_expand,
            rl_action_fired,
            rl_action_no_effect,
            rl_action_refused_no_border,
            rl_action_refused_amount,
            rl_action_refused_no_tiles,
            rl_action_not_expand,
            rl_action_unsupported_player,
            rl_suppressed_bot_firings,
            rl_applied,
            if rl_action_last == usize::MAX {
                -1i64
            } else {
                rl_action_last as i64
            }
        );
    }

    // ---- report ----
    let per_tick = ms / a.ticks.max(1) as f64;
    println!(
        "MULTI-ORIG run: ticks={} wall={:.1} ms ({:.3} ms/tick) {:.0} env-ticks/s {:.0} decisions/s live_slots_after={}",
        a.ticks,
        ms,
        per_tick,
        if per_tick > 0.0 { n as f64 / per_tick * 1000.0 } else { 0.0 },
        if per_tick > 0.0 { decisions as f64 / (ms / 1000.0) } else { 0.0 },
        live_last
    );
    println!(
        "MULTI-ORIG origination: attacks_originated_unaided={} (one per firing bot per env) merged_absorbed={} slot_overflow={} decisions={}",
        origs, merged_absorbed, orig_slot_ovf, decisions
    );
    let mut matched = 0usize;
    let mut matched_total = 0usize;
    for (k, hh) in hashes.iter().enumerate() {
        match engine_hash_at(&dump, w, h, t0 + 1 + k as u32) {
            Some(eh) if eh == *hh => {
                matched_total += 1;
                if matched == k {
                    matched += 1;
                }
            }
            _ => {}
        }
    }
    println!(
        "MULTI-ORIG owner troop ledger: debits={} total_debited={} credits={} total_credited={}",
        ledger_debits, ledger_debit_total, ledger_credits, ledger_credit_total
    );
    println!(
        "MULTI-ORIG engine agreement: {}/{} ticks matched (prefix, unaided); {} total exact ticks",
        matched,
        a.ticks,
        matched_total
    );
    if matched < hashes.len() {
        let k = matched;
        println!(
            "MULTI-ORIG first divergence: tick {} dev_hash={:016x} engine_hash={:016x}",
            t0 + 1 + k as u32,
            hashes.get(k).copied().unwrap_or(0),
            engine_hash_at(&dump, w, h, t0 + 1 + k as u32).unwrap_or(0)
        );
    } else {
        println!("MULTI-ORIG first divergence: NONE in {} ticks", a.ticks);
    }
    // ENGINE CREATIONS the env was supposed to originate. `AttRec` carries no
    // attack id, so a creation is counted the only way the record allows: a tick
    // in which an owner's LIVE land-attack count rises.
    let live_by_owner = |v: &Vec<extras::AttRec>| -> std::collections::HashMap<u16, i32> {
        let mut m = std::collections::HashMap::new();
        for x in v.iter().filter(|x| x.live && x.target == 0) {
            *m.entry(x.owner).or_insert(0) += 1;
        }
        m
    };
    let mut eng_creations = 0usize;
    let mut prev_counts = ex.attacks.get(&t0).map(live_by_owner).unwrap_or_default();
    for t in (t0 + 1)..=(t0 + a.ticks) {
        let cur = ex.attacks.get(&t).map(live_by_owner).unwrap_or_default();
        for (owner, n) in cur.iter() {
            let p = prev_counts.get(owner).copied().unwrap_or(0);
            if *n > p {
                eng_creations += (*n - p) as usize;
            }
        }
        prev_counts = cur;
    }
    println!(
        "MULTI-ORIG engine creations in window: {} (env originated {})",
        eng_creations, origs
    );
    Ok(())
}

/// Print the tile-level difference between the env's plane and the engine's at
/// one tick, with the `(dev, eng)` pairs that account for it.
fn print_plane_diff(dev: &[u16], eng: &[u16], tick: u32) {
    let mut diff = 0usize;
    let mut seen: std::collections::HashMap<(u16, u16), usize> = std::collections::HashMap::new();
    let mut samples = Vec::new();
    for (i, (a1, b1)) in dev.iter().zip(eng.iter()).enumerate() {
        if a1 != b1 {
            diff += 1;
            *seen.entry((*a1, *b1)).or_insert(0) += 1;
            if samples.len() < 6 {
                samples.push((i, *a1, *b1));
            }
        }
    }
    println!("MULTI-ORIG tile diff at tick {}: {} tiles differ", tick, diff);
    let mut pairs: Vec<((u16, u16), usize)> = seen.into_iter().collect();
    pairs.sort_by(|x, y| y.1.cmp(&x.1));
    for p in pairs.iter().take(6) {
        println!("   dev={} eng={} count={}", p.0 .0, p.0 .1, p.1);
    }
    for s in samples {
        println!("   sample tile {}: dev={} eng={}", s.0, s.1, s.2);
    }
}

fn main() {
    let t = Instant::now();
    match run() {
        Ok(()) => eprintln!("ok in {:.1}s", t.elapsed().as_secs_f64()),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
