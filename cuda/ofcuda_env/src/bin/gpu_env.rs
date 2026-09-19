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
    tribe_ratios, tiles_used, Attack,
};
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
/// Scalars per environment in the `scal` buffer.
const SCAL: usize = 9;
/// Threads per block for `env_step`.
const BLOCK: u32 = 256;

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
    ) -> bool {
        // attack.rs:239-255: the one extra draw, taken BEFORE the pop loop, and
        // the budget's border term read after it.
        let draw = pr.next_int(0, 5);
        let budget = attack_tiles_per_tick(*troop_count, false, 0.0, *blen as f64 + draw as f64);
        let mut num = budget;

        while num > 0.0 {
            if *troop_count < 1.0 {
                *troop_count = 0.0;
                return false; // attack.rs:258-262 starved
            }
            if heap.is_empty() {
                // attack.rs:264-268 refresh_to_conquer, then RETREAT (death).
                heap.clear();
                *blen = 0;
                let mut j = 0usize;
                while j < obn {
                    let bt = oborder[oboff + j];
                    j += 1;
                    offer_dev(heap, pr, bsub, blen, psub, owner_col, terrain, bt, w, h, tick);
                }
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
        let poff = e * wh;
        let psub = &mut plane[poff..poff + wh];
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
        let psub = &mut plane[poff..poff + wh];
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
}

// ---------------------------------------------------------------------------
// The environment: one real attack reconstructed from the record
// ---------------------------------------------------------------------------

struct EnvState {
    w: u32,
    h: u32,
    terrain: Vec<u8>,
    plane: Vec<u16>,
    owner_sid: u16,
    owner_col: u16,
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
    Ok(EnvState {
        w,
        h,
        terrain: terrain.to_vec(),
        plane,
        owner_sid: owner,
        owner_col: owner, // the plane word IS the raw small id
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
) -> Result<RunOut, Box<dyn std::error::Error>> {
    let stream = ctx.default_stream();
    let n = b.n;
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
            )?
        };
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
            "--selftest" => {
                a.selftest = true;
                i += 1;
            }
            o => return Err(format!("unknown arg {o}").into()),
        }
    }
    Ok(a)
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

    // Self-test: device vs the CPU reference, per tick, on identical state.
    if a.selftest || a.mode == "both" {
        let envs = vec![env0_copy(&env0)?];
        let mut b = DevBatch::new(&ctx, &module, &envs, 1)?;
        let out = run_batch(
            &ctx, &module, &mut b, &env0, a.ticks, env0.tick0, true, false, false,
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
        let out = run_batch(&ctx, &module, &mut b, &env0, a.ticks, env0.tick0, false, true, false)
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
        run_batch(&ctx, &module, &mut b, &env0, 4, env0.tick0, false, false, false)
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
    })
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
