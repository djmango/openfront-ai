//! Device module for `ofcuda_matrix`.
//!
//! The device core is NOT copied here: `include!` pulls in the canonical
//! `ofcuda_tick/src/core_impl.rs` (the same file `ofcuda_tick`'s host reference
//! and `ofcuda_env`'s `gpu_env` use), so the PRNG, the flat heap, the
//! `add_neighbors` draw-binding and the pop loop have exactly one
//! implementation. The float budget and the terrain cost helpers are
//! `ofcuda_env`'s, called from device code - the same two the composed CPU tick
//! calls.
//!
//! # Shape
//!
//! One launch = ONE attack tick, run by thread 0. The engine ticks its `execs`
//! in one global order and mutates the owner plane IN PLACE as tiles are
//! conquered (`map.set_owner`), so two attacks in the same tick see each other
//! - the plane therefore lives on the device across the whole cell and is never
//! re-uploaded per tick. (The CPU `ofcuda_env::Attack` passes a tick-start plane
//! plus its own accumulating claims list, which is equivalent only for a single
//! live attack per contested tile; the in-place device plane is the engine's own
//! semantics.)
//!
//! # Two buffer classes
//!
//! * PERSISTENT, indexed `slot * STRIDE` and never read back per tick:
//!   `heap_tiles`/`heap_pri` (`HEAP_CAP`, the carried `to_conquer` heap),
//!   `border` (`BC`, the attack's `border_tiles` set = the budget's only
//!   adjacency input, `attack.rs:240`), `prng` (5 sfc32 words,
//!   `PseudoRandom::new(123)`), `troops` (the carried `f64`), `scal`
//!   (`SCAL`: heap_len, border_len, claims_len, alive, drops, peak, rsvd,
//!   rsvd).
//! * SCRATCH, one small buffer shared by every launch, copied back after each
//!   launch: `out` (`SCAL_OUT` words: this tick's claim count, alive, drops,
//!   peak) and `claims` (`MAXC`, THIS tick's claims in pop order).
//!
//! `ob_meta[owner_col*2]` / `[owner_col*2+1]` = (offset, length) of that owner's
//! `border_tiles` in the flat `oborder` array for THIS tick - the input to
//! `refresh_to_conquer` (`attack.rs:1265-1274`), which fires at attack creation
//! and on a mid-tick heap-empty retreat.

use cuda_device::{kernel, launch_bounds, thread};
use cuda_host::cuda_module;
use ofcuda_env::{attack_tiles_per_tick, attacker_troop_loss, terrain_mag, terrain_speed, tiles_used};
use ofcuda_tick::{
    attacker_neighbor_count, has_attacker_neighbor, mag2_from_terrain, neighbors4, priority_f32, Heap, Prng,
    HEAP_CAP, ORDER_NSWE,
};

/// `Util.within(value, min, max)` (`util.rs:16-18`) = `value.max(min).min(max)`,
/// spelled with branches so it lowers to the same pair of selects in device code.
#[inline]
fn within(v: f64, lo: f64, hi: f64) -> f64 {
    let a = if v > lo { v } else { lo };
    if a < hi { a } else { hi }
}

/// A multiply the NVPTX device backend is not allowed to contract into an FMA.
///
/// The engine is host Rust, and there `a * b + c` is TWO roundings: the product
/// is rounded, then the sum is rounded. This backend marks every `fmul`/`fadd`
/// with LLVM's `contract` fast-math flag (the generated IR carries
/// `fadd contract` and `fmul contract`), so it fuses the pattern into a single
/// `fma.rn.f64` (ONE rounding) - a 1-ULP difference, and 1 ULP of a troop count
/// is a comparator-visible divergence.
///
/// This was not hypothetical. The generated PTX carried exactly two such
/// contracts, both in the player-target branch of `attack_tick` - a path the port
/// never used to exercise, because a player-targeted land attack arriving by boat
/// landing did not exist in the device until `land()`'s branches were ported:
///
///   `fma.rn.f64 %rd286, %rd285, 0d3FD3333333333333, 0d3FE6666666666666`
///       = fma(0.3, dsig, 0.7)                  <- `dbuf`
///   `fma.rn.f64 %rd418, %rd296, 0d3FD999999999999A, %rd297`
///       = fma(0.4, alt, 0.6 * cur)             <- `attack_loss`
///
/// `#[inline(never)]` cannot stop it: this backend's IR carries only
/// `attributes #0 = { convergent }`, so the attribute is dropped and the callee
/// is inlined anyway. A VOLATILE read is the barrier that survives: the product
/// has to be materialised in memory, so the backend cannot fold it into the
/// following add. No value is changed - this restores the engine's own double
/// rounding, it is not a correction towards it.
#[inline]
fn mul_round(a: f64, b: f64) -> f64 {
    let p = a * b;
    unsafe { core::ptr::read_volatile(&p) }
}

/// Capacity of one attack's border set (device only; the CPU `Attack` uses a
/// `Vec`). Overflow is COUNTED in `scal[slot*8+4]`, never silently dropped.
pub const BC: usize = 8192;
/// Capacity of one attack's per-tick claim list. Overflow is counted too.
pub const MAXC: usize = 512;
/// Persistent scalars per slot.
pub const SCAL: usize = 8;
/// Scratch scalars written per launch.
pub const SCAL_OUT: usize = 4;
/// Threads per block. Thread 0 runs the tick; the rest return immediately.
pub const BLOCK: u32 = 256;

#[cuda_module]
pub mod device {
    use super::*;

    include!("../../ofcuda_tick/src/core_impl.rs");

    /// `#[kernel]` entry point for the canonical `cluster_pass_core`
    /// (`ofcuda_tick/src/core_impl.rs`). It has to live here rather than in the
    /// canonical file because both `#[kernel]`/`#[launch_bounds]` and
    /// `thread::index_1d()` need the `#[cuda_module]` scope that this module
    /// (and `ofcuda_env`'s) provides, while `ofcuda_tick`'s host `lib.rs`
    /// includes the same file with neither.
    ///
    /// ONE thread runs the whole pass (`launch_bounds(1)`): the pass walks the
    /// engine's exec order and each player's border set in ORDER, and every
    /// `remove_cluster` mutates the plane in place for the players after it,
    /// exactly as the engine's `game.conquer` does. A parallel kernel would need
    /// a different order guarantee than the engine has.
    #[kernel]
    #[launch_bounds(1)]
    #[allow(clippy::too_many_arguments)]
    pub fn cluster_pass(
        terrain: &[u8],
        w: u32,
        h: u32,
        tick: u32,
        npl: u32,
        order: &[u32],
        idhash: &[u32],
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
        if thread::index_1d().get() != 0 {
            return;
        }
        cluster_pass_core(
            terrain, w, h, tick, npl, order, idhash, cad, plane, t2b, marks, cbuf, clen, coff,
            vis, stack, owned, rem, out, oborder, obmeta, friends, nfriends, atk_owner,
            atk_target, atk_troops, natk, pst,
        );
    }

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
    /// the border tile for exactly the neighbours `add_neighbors` enqueues, then
    /// let the crate's own `add_neighbors` do the draw-binding and priority.
    #[allow(clippy::too_many_arguments)]
    fn offer_dev(
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
            border_insert(border, blen, nb); // attack.rs:1363
        }
        add_neighbors_t(
            heap, pr, tile, plane, owner_col, target_col, terrain, w, h, tick,
        )
    }

    /// `AttackExecution::add_neighbors` (`attack.rs:1340-1386`) with the target
    /// as a parameter instead of the crate's hardcoded terra nullius.
    ///
    /// `attack.rs:1359` skips any neighbour whose `owner_id` is not
    /// `self.target_small_id`; the crate's `ofcuda_tick::add_neighbors` spells
    /// that as `is_terra_nullius(...)`, which is the same predicate only while
    /// the attack targets terra nullius (owner 0). The body below is otherwise
    /// the crate's function verbatim - same N,S,W,E visit order, same
    /// ONE draw per ENQUEUED neighbour, the draw-binding site.
    #[allow(clippy::too_many_arguments)]
    fn add_neighbors_t(
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

    /// `AttackExecution::refresh_to_conquer` (`attack.rs:1265-1274`): clear the
    /// heap AND the border set, then offer every neighbour of the owner's
    /// current border.
    #[allow(clippy::too_many_arguments)]
    fn refresh_dev(
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
            offer_dev(
                heap, pr, border, blen, plane, owner_col, target_col, terrain, bt, w, h, tick,
            );
        }
    }

    /// `TransportShipExecution::land` (`transport_ship.rs:374-401`): a boat that
    /// reaches shore CONQUERS its destination tile and only then creates an
    /// attack whose `source_tile` is that destination
    /// (`game.add_land_attack_from(owner, None, Some(troops), Some(dst))`,
    /// `transport_ship.rs:383-386`).
    ///
    /// The device has no units, so the record's own attack list is where a
    /// landing is observable: an attack that carries a `source_tile` and was not
    /// there at the previous boundary is a boat's landing, and its source tile is
    /// the tile `land()` conquered in that tick. This kernel applies exactly that
    /// one `game.conquer(owner, dst)` - the plane cell and the tick's claim list,
    /// the same two effects a `tick_once` conquest has.
    #[kernel]
    #[launch_bounds(256)]
    pub fn land_dev(
        tile: u32,
        owner_col: u16,
        mut plane: &mut [u16],
        mut claims: &mut [u32],
        mut out: &mut [u32],
    ) {
        let e = thread::index_1d().get();
        if e != 0 {
            return;
        }
        plane[tile as usize] = owner_col;
        claims[0] = tile;
        out[0] = 1;
    }

    /// ONE TICK of one attack (`AttackExecution::tick`, `attack.rs:206-324`).
    /// Returns whether the attack is still alive. `tick` is the ENGINE's tick
    /// index (`game.ticks()`), because it enters every enqueued tile's priority.
    #[allow(clippy::too_many_arguments)]
    fn tick_once(
        heap: &mut Heap,
        pr: &mut Prng,
        troop_count: &mut f64,
        bsub: &mut [u32],
        blen: &mut usize,
        psub: &mut [u16],
        claims: &mut [u32],
        ncl: &mut usize,
        drops: &mut u32,
        terrain: &[u8],
        w: u32,
        h: u32,
        tick: u32,
        owner_col: u16,
        target_col: u16,
        is_bot: u32,
        pst: &mut [f64],
        defsig: &[f64],
        oborder: &[u32],
        oboff: usize,
        obn: usize,
    ) -> bool {
        // attack.rs:239-255 - the one extra draw, taken BEFORE the pop loop.
        // The budget's branch (`attack.rs:246-252`) is selected by
        // `target_is_player` and reads the TARGET's live troop count, not 0.
        let player_target = target_col != 0;
        let draw = pr.next_int(0, 5);
        let defender_troops = if player_target {
            pst[target_col as usize * 3]
        } else {
            0.0
        };
        let budget = attack_tiles_per_tick(
            *troop_count,
            player_target,
            defender_troops,
            *blen as f64 + draw as f64,
        );
        let mut num = budget;

        while num > 0.0 {
            if *troop_count < 1.0 {
                *troop_count = 0.0;
                return false; // attack.rs:258-262 starved
            }
            if heap.is_empty() {
                // attack.rs:264-268: refresh_to_conquer, then RETREAT (death).
                refresh_dev(
                    heap, pr, bsub, blen, psub, owner_col, target_col, terrain, w, h, tick, oborder,
                    oboff, obn,
                );
                // `attack.rs:264-268` calls `refresh_to_conquer` and then
                // `retreat(game, 0.0)` (`attack.rs:1244-1263`), which BEFORE
                // zeroing `self.troops` runs
                // `game.add_troops(self.owner_small_id, (troops - deaths).max(0.0))`,
                // `deaths = troops * malus / 100` with `malus = 0` on this path.
                //
                // This is the THIRD intra-tick mutation of a player's live pool
                // (after `PlayerExecution` income and `conquer_one`'s
                // `tiles_owned += 1`): an attack that DIES returns its survivors
                // to its OWNER, and any LATER attack in exec order that targets
                // that owner reads the inflated `defender_troops`. Traced on
                // pangaea N=488 nat=0, boundary 172: player 269's own attack
                // (269 -> 0) died in the same tick and returned 3955.93 -> +3955,
                // so 441's attack read
                // `defender_troops = 14126 (record) + 111 (income) + 3955 = 18192`
                // and `defender_tiles = 822 + 1 = 823`, which reproduces the
                // engine's `attacker_loss` = 104.261740669 to 5.7e-13.
                //
                // `retreat` subtracts `start_troops` first only when
                // `!self.remove_troops && self.source_tile.is_none()`. The
                // measured bot attack has no usable `start_troops` (its survivors
                // came out exactly `troops - pop_loss`), so no subtraction is
                // modelled. A cell whose dying attacks carry pending troops would
                // need `start_troops` threaded through `Slot`.
                let ocol = owner_col as usize;
                if ocol * 3 < pst.len() && *troop_count >= 1.0 {
                    pst[ocol * 3] += (*troop_count).floor();
                }
                *troop_count = 0.0;
                return false;
            }
            let Some((tile, _pri)) = heap.dequeue() else {
                break; // attack.rs:271
            };
            border_remove(bsub, blen, tile); // attack.rs:275
            if psub[tile as usize] != target_col {
                continue; // attack.rs:284 wrong owner for this target
            }
            if !has_attacker_neighbor(psub, &[], owner_col, tile, w, h, ORDER_NSWE) {
                continue; // attack.rs:284
            }
            if terrain[tile as usize] & 0x80 == 0 {
                continue; // attack.rs:288
            }
            offer_dev(
                heap, pr, bsub, blen, psub, owner_col, target_col, terrain, tile, w, h, tick,
            ); // attack.rs:292
            if player_target {
                // `game.attack_logic_at_tile(.., defender_is_player = true)`
                // (`game.rs:784-878`), the branch a player-target attack takes.
                let tidx = target_col as usize * 3;
                let dtroops = pst[tidx];
                let raw_tiles = pst[tidx + 1];
                let dtiles = if raw_tiles < 1.0 { 1.0 } else { raw_tiles };
                let mut mag = terrain_mag(terrain[tile as usize]);
                let speed = terrain_speed(terrain[tile as usize]);
                // `matches!(attacker_type, Human | Nation) && defender == Bot`.
                let attacker_bot = pst[owner_col as usize * 3 + 2] == 1.0;
                let defender_bot = pst[tidx + 2] == 1.0;
                if !attacker_bot && defender_bot {
                    mag *= 0.7;
                }
                // `1.0 - sigmoid(defender_tiles, LN_2/50_000, 150_000)`: a pure
                // function of the integer tile count, so the host evaluates it
                // with the engine's own expression and the device reads it.
                let di = if raw_tiles < 1.0 { 1usize } else { raw_tiles as usize };
                let dsig = if di < defsig.len() {
                    defsig[di]
                } else {
                    defsig[defsig.len() - 1]
                };
                let dbuf = 0.7 + mul_round(0.3, dsig);
                // `defender_troop_loss` stays the EXACT fraction - it feeds
                // `alt_attacker_loss` below, and the engine computes it that way
                // (`attack.rs`: `defender_troops as f64 / defender_tiles as f64`).
                let def_loss = dtroops / dtiles;
                let cur = within(dtroops / *troop_count, 0.6, 2.0) * mag * 0.8 * dbuf;
                let alt = 1.3 * def_loss * (mag / 100.0);
                let attack_loss = mul_round(0.6, cur) + mul_round(0.4, alt);
                let t_used = within(dtroops / (5.0 * *troop_count), 0.2, 1.5) * speed * dbuf;
                num -= t_used; // attack.rs:311
                *troop_count -= attack_loss; // attack.rs:312
                // `game.remove_troops(target, defender_loss)` (`game.rs:1130`):
                // `to_remove = min(p.troops, to_int(defender_loss))` and `to_int`
                // is `floor` (`util.rs:26`). A `Player`'s troops are an `i32`, so
                // the defender's pool steps by WHOLE troops. Subtracting the exact
                // fraction here shed a fraction MORE than the engine on every pop,
                // leaving the device's defender lower; `attacker_loss` is
                // (approximately) proportional to the defender's troops, so the
                // device's attacker then lost slightly LESS per pop and drifted
                // ABOVE the engine - the measured `device >= engine` troop drift.
                let to_remove = if def_loss > 0.0 {
                    def_loss.floor().min(dtroops)
                } else {
                    0.0
                };
                pst[tidx] = dtroops - to_remove; // game.remove_troops(target, ..)
                pst[tidx + 1] = raw_tiles - 1.0; // game.conquer(owner, tile)
            } else {
                num -= tiles_used(*troop_count, terrain[tile as usize]); // attack.rs:313
                *troop_count -= attacker_troop_loss(terrain[tile as usize], is_bot != 0); // attack.rs:314
            }
            if *ncl < claims.len() {
                claims[*ncl] = tile;
                *ncl += 1;
            } else {
                *drops += 1;
            }
            // `game.conquer(owner, tile)` -> `conquer_one` (`game.rs:1233-1267`)
            // also does `p.tiles_owned += 1` for the CONQUEROR. The loser's side
            // is the `pst[tidx + 1]` decrement above (reachable only when the
            // target is a player, matching `conquer_one`'s `if prev > 0`).
            //
            // This is the second half of the same class of intra-tick mutation as
            // the income: a player's OWN claims grow its `tiles_owned` mid-tick,
            // and any LATER attack (exec order) that targets that player reads
            // `defender_tiles` from the grown value, which moves
            // `large_defender_attack_debuff`, `alt_attacker_loss` and
            // `defender_troop_loss`. Traced on pangaea N=488 nat=0 boundary 166:
            // the engine used `defender_tiles = 815` for player 269 while the
            // per-boundary record still said 812 (269's own attack into terra
            // nullius had already claimed 3 tiles earlier in the same tick), and
            // `(defender_troops = 13677, defender_tiles = 815)` reproduces the
            // engine's `attacker_loss` to 5.7e-13.
            let ocol = owner_col as usize;
            if ocol * 3 + 1 < pst.len() {
                pst[ocol * 3 + 1] += 1.0;
            }
            psub[tile as usize] = owner_col; // conquer, IN PLACE (map.set_owner)
        }
        if *troop_count < 0.0 {
            *troop_count = 0.0; // attack.rs:322-323
        }
        true
    }

    /// Load the slot's persistent state, run ONE tick, write it back. The
    /// scratch `out`/`claims` describe exactly this launch's tick.
    #[kernel]
    #[launch_bounds(256)]
    #[allow(clippy::too_many_arguments)]
    pub fn attack_tick(
        terrain: &[u8],
        w: u32,
        h: u32,
        tick: u32,
        slot: u32,
        owner_col: u16,
        target_col: u16,
        is_bot: u32,
        mut pst: &mut [f64],
        defsig: &[f64],
        mut plane: &mut [u16],
        mut heap_tiles: &mut [u32],
        mut heap_pri: &mut [f32],
        mut border: &mut [u32],
        mut prng: &mut [u32],
        mut troops: &mut [f64],
        mut scal: &mut [u32],
        mut out: &mut [u32],
        mut claims: &mut [u32],
        oborder: &[u32],
        ob_meta: &[u32],
        // `Game::is_friendly(a, b)` for THIS boundary: the oracle's own `FRIEND`
        // pairs, the same table `cluster_pass` already consumes. Needed by the
        // alliance retreat at the top of the attack's own tick
        // (`attack.rs:222-229`).
        friends: &[u32],
        nfriends: u32,
    ) {
        let e = thread::index_1d().get();
        if e != 0 {
            return;
        }
        let s = slot as usize;
        let soff = s * SCAL;
        let hoff = s * HEAP_CAP;
        let boff = s * BC;
        let pse = s * 5;
        out[0] = 0; // a fresh tick: no claims yet
        out[1] = 0;
        out[2] = scal[soff + 4];
        out[3] = scal[soff + 5];
        if scal[soff + 3] == 0 {
            return; // dead: the engine ticks it once and it draws nothing
        }

        // `AttackExecution::tick` (`attack.rs:222-229`):
        //
        // ```ignore
        // if self.target_is_player {
        //     if game.is_friendly(self.owner_small_id, self.target_small_id) {
        //         self.retreat(game, 0.0); // every remaining troop returned
        //         return;
        //     }
        // }
        // ```
        //
        // The retreat is at the TOP of the attack's OWN tick, i.e. BEFORE it pops
        // a single tile, and `retreat` ends in `kill_attack` + `active = false`
        // (`attack.rs:1279-1295`). So an alliance (or same team) that holds by
        // this boundary costs the attack this tick's whole claim set AND its
        // life. This is the one engine eviction the record cannot report in time:
        // the attack is still in `prev.ATTACK` and gone from `cur.ATTACK`, so a
        // listener that only diffs the two lists kills it one boundary LATE - and
        // the tick it claims in the meantime is exactly what made the device's
        // per-player claim list a SUPERSET of the engine's (pangaea N=18 nat=1
        // boundary 1709: engine 40 tiles, device 112). Computing the rule here
        // from the engine's own friendship pairs for this boundary kills it on
        // the tick the engine killed it.
        if target_col != 0
            && target_col != owner_col
            && cl_friendly(friends, nfriends, owner_col, target_col)
        {
            scal[soff + 3] = 0; // `kill_attack`: dead from here on
            scal[soff] = 0;
            out[1] = 0;
            return;
        }
        let mut hl = scal[soff] as usize;
        if hl > HEAP_CAP {
            hl = HEAP_CAP;
        }
        let mut heap = Heap::new();
        heap.load(&heap_tiles[hoff..hoff + hl], &heap_pri[hoff..hoff + hl]);
        let mut pr = Prng::from_state(&prng[pse..pse + 5]);
        let mut troop_count = troops[s];
        let mut blen = scal[soff + 1] as usize;
        if blen > BC {
            blen = BC;
        }
        let mut ncl = 0usize;
        let mut drops = scal[soff + 4];
        let om = owner_col as usize * 2;
        let oboff = ob_meta[om] as usize;
        let obn = ob_meta[om + 1] as usize;

        let alive = tick_once(
            &mut heap,
            &mut pr,
            &mut troop_count,
            &mut border[boff..boff + BC],
            &mut blen,
            plane,
            claims,
            &mut ncl,
            &mut drops,
            terrain,
            w,
            h,
            tick,
            owner_col,
            target_col,
            is_bot,
            pst,
            defsig,
            oborder,
            oboff,
            obn,
        );

        scal[soff] = heap.len as u32;
        scal[soff + 1] = blen as u32;
        scal[soff + 2] = ncl as u32;
        troops[s] = troop_count;
        scal[soff + 3] = if alive { 1 } else { 0 };
        scal[soff + 4] = drops;
        // DIAGNOSTIC: `Heap::drops` (candidates REFUSED because the fixed
        // HEAP_CAP was reached). The engine's `FlatBinaryHeap` is a growable
        // `Vec` (`flat_heap.rs:11-13`), so a refusal here is a tile the engine
        // would have enqueued and claimed later - it is the one frontier loss
        // that leaves no trace in the plane until it is popped. Accumulated per
        // slot so the driver can report it; never used as a pass condition.
        scal[soff + 6] += heap.drops as u32;
        scal[soff + 5] = if heap.peak as u32 > scal[soff + 5] {
            heap.peak as u32
        } else {
            scal[soff + 5]
        };
        out[0] = ncl as u32;
        out[1] = if alive { 1 } else { 0 };
        out[2] = drops;
        out[3] = scal[soff + 5];
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

    /// `AttackExecution::init`'s frontier seed (`attack.rs:156-164`), run on the
    /// device: the attack's heap and border set are built from the owner's
    /// border tiles (`refresh_to_conquer`, `attack.rs:1265-1274`) and every
    /// enqueue draws from the attack's OWN stream.
    ///
    /// This is BOTH device entry points the engine has, because they are the
    /// same call in the engine:
    ///
    /// * **creation** - a `(owner, target)` the device has no slot for. The
    ///   engine constructed a new `AttackExecution` and `execute_next_tick`
    ///   initialised it AT THE END of the tick that produced the boundary it
    ///   first appears at (`game.rs:3657-3700`), then appended it to `execs`,
    ///   so its first pop is the following tick.
    /// * **RE-CREATE** - a live `(owner, target)` whose `attack_id` CHANGED
    ///   (`attack.rs:156`). The bot AI re-issued the attack
    ///   (`game.add_land_attack_from`, `game.rs:1469-1484`), the new exec was
    ///   coalesced/merged in `init` (`merge_outgoing_land_attacks`,
    ///   `game.rs:2122-2156`, which adds the old attack's troops and kills it)
    ///   and the OLD attack still ticks in that same tick, first, in its old
    ///   `execs` position. So the re-create lands at the END of the tick:
    ///   `fresh_prng = 1` restores `PseudoRandom::new(123)`
    ///   (`attack.rs:47-57`) and the frontier is rebuilt from scratch.
    ///
    /// `fresh_prng = 0` rebuilds from the slot's carried stream instead, which
    /// is what a caller that has already written a fresh state into `prng`
    /// wants (the creation path).
    #[kernel]
    #[launch_bounds(256)]
    #[allow(clippy::too_many_arguments)]
    pub fn attack_init(
        terrain: &[u8],
        w: u32,
        h: u32,
        tick: u32,
        slot: u32,
        owner_col: u16,
        target_col: u16,
        // The attack's `source_tile` (`attack.rs:19`), `u32::MAX` for none.
        // `init` seeds the frontier from it when it is set (`attack.rs:160-164`),
        // and the ONLY engine call sites that pass one are transport-ship
        // landings (`transport_ship.rs:386,391`) and nation-structure attacks
        // (`nation_structures.rs:1892`).
        source: u32,
        fresh_prng: u32,
        plane: &[u16],
        mut heap_tiles: &mut [u32],
        mut heap_pri: &mut [f32],
        mut border: &mut [u32],
        mut prng: &mut [u32],
        mut scal: &mut [u32],
        mut out: &mut [u32],
        oborder: &[u32],
        ob_meta: &[u32],
    ) {
        let e = thread::index_1d().get();
        if e != 0 {
            return;
        }
        let s = slot as usize;
        let soff = s * SCAL;
        let boff = s * BC;
        let pse = s * 5;
        scal[soff] = 0;
        scal[soff + 1] = 0;
        scal[soff + 2] = 0;
        scal[soff + 4] = 0;
        scal[soff + 6] = 0;
        scal[soff + 5] = 0;
        scal[soff + 3] = 1;
        let mut heap = Heap::new();
        let mut pr = if fresh_prng != 0 {
            Prng::new(SEED)
        } else {
            Prng::from_state(&prng[pse..pse + 5])
        };
        let mut blen = 0usize;
        let om = owner_col as usize * 2;
        let oboff = ob_meta[om] as usize;
        let obn = ob_meta[om + 1] as usize;
        if source != u32::MAX {
            // `attack.rs:160-164`: with a `source_tile` the frontier is seeded
            // from the SOURCE TILE's own neighbours (`add_neighbors(src)`), not
            // from the owner's border set. `offer_dev` is that call plus the
            // attack-border insert (`offer_neighbours`, `attack.rs:1353-1386`),
            // in the same N,S,W,E order with the same one draw per enqueued
            // neighbour.
            offer_dev(
                &mut heap,
                &mut pr,
                &mut border[boff..boff + BC],
                &mut blen,
                plane,
                owner_col,
                target_col,
                terrain,
                source,
                w,
                h,
                tick,
            );
        } else {
            refresh_dev(
                &mut heap,
                &mut pr,
                &mut border[boff..boff + BC],
                &mut blen,
                plane,
                owner_col,
                target_col,
                terrain,
                w,
                h,
                tick,
                oborder,
                oboff,
                obn,
            );
        }
        scal[soff] = heap.len as u32;
        scal[soff + 1] = blen as u32;
        scal[soff + 5] = heap.peak as u32;
        out[0] = 0;
        out[1] = 1;
        out[2] = 0;
        out[3] = heap.peak as u32;
        let mut j = 0usize;
        while j < heap.len {
            heap_tiles[s * HEAP_CAP + j] = heap.tiles[j];
            heap_pri[s * HEAP_CAP + j] = heap.pri[j];
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

    // -----------------------------------------------------------------------
    // BOT ATTACK ORIGINATION (self-drive)
    //
    // `TribeExecution::tick` (`openfront-ai/rust/engine/src/bot/tribe.rs:114-157`)
    // is the ONLY origin of a new attack in a game with no human input: a bot's
    // periodic expansion attack (`send_tn_attack` -> `Game::add_land_attack_from`,
    // `game.rs:1514`). Everything below reproduces the part of that policy that
    // ORIGINATES A LAND ATTACK against Terra Nullius; the parts that attack
    // PLAYERS (`tribe_maybe_attack`'s retaliate/traitor/weakest/random-target
    // ladder, `ai_attack.rs:2194-2263`) are named in `out[k*3+2] == 5` but not
    // reproduced.
    // -----------------------------------------------------------------------

    /// `GameMap::is_land` (`map.rs:128-130`): terrain bit 7.
    #[inline]
    fn dev_is_land(terrain: &[u8], t: u32) -> bool {
        terrain[t as usize] & 0x80 != 0
    }

    /// `GameMap::is_shore` (`map.rs:144-146`): `is_land && is_shoreline`, i.e.
    /// terrain bits 7 and 6 (the shoreline bit is precomputed in the map bytes).
    #[inline]
    fn dev_is_shore(terrain: &[u8], t: u32) -> bool {
        terrain[t as usize] & 0xc0 == 0xc0
    }

    /// `has_land_border_with_terra_nullius` (`ai_attack.rs:187-204`) and
    /// `has_land_border_tn` (`ai_attack.rs:147-168`, the TN probe inside
    /// `tribe_maybe_attack`): any tile in `sid`'s own `border_tiles` with an
    /// N,S,W,E neighbour that is land and unowned. The two engine predicates
    /// differ only by a `has_fallout` guard on the neighbour, and this port
    /// models no fallout, so they collapse into one scan.
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
                if dev_is_land(terrain, nb) && plane[nb as usize] == 0 {
                    return true;
                }
                j += 1;
            }
            i += 1;
        }
        false
    }

    /// `has_shore_reachable_tn` (`ai_attack.rs:214-248`): over
    /// `sampled_shore_tiles` (`ai_attack.rs:206-212` - the owner's border tiles
    /// filtered to shore, then `step_by(10)`), for each of the four directions
    /// (0,-1),(0,1),(-1,0),(1,0), the near tile must be water and the tile five
    /// steps out must be valid, land and unowned.
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
            if dev_is_shore(terrain, t) {
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
                            if !dev_is_land(terrain, t1)
                                && dev_is_land(terrain, tn)
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

    /// ONE bot's attack decision, on the device, for one tick.
    ///
    /// `out` is `3 * nbots`: `[fire, troops, action]`.
    ///   action 1 = a TN land attack is originated (`send_tn_attack` ->
    ///              `add_land_attack_from(sid, None, Some(troops))`);
    ///   action 3 = fired, no Terra Nullius in reach (the engine's BOAT path,
    ///              `send_boat_attack_to_nearby_tn` - no attack object is created
    ///              by the engine either, and the port models no boats);
    ///   action 4 = fired, but `land_attack_troops` < 1 (`ai_attack.rs:9-17`)
    ///              so the engine sends nothing;
    ///   action 5 = reached the PLAYER-target part of `tribe_maybe_attack`
    ///              (`ai_attack.rs:2194-2263`) - NOT reproduced, nothing sent;
    ///   action 6 = `has_trigger_ratio` false (`ai_attack.rs:35-43`) so the
    ///              engine also sends nothing;
    ///   action 0 = did not fire this tick.
    ///
    /// `bot_state` bit 0 = `attack_behavior_init`, bit 1 = `neighbors_terra_nullius`.
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
        if thread::index_1d().get() != 0 {
            return;
        }
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

    /// FNV-1a-64 of the whole owner plane, one thread. `ofcuda_hash`'s function,
    /// not a re-implementation.
    #[kernel]
    #[launch_bounds(64)]
    pub fn plane_hash(plane: &[u16], wh: u32, out: &mut [u64]) {
        let i = thread::index_1d().get();
        if i != 0 {
            return;
        }
        out[0] = ofcuda_hash::fnv1a_u16_le(ofcuda_hash::FNV_OFFSET_BASIS, &plane[..wh as usize]);
    }

    /// Owned-tile count per owner column, straight off the device plane.
    #[kernel]
    #[launch_bounds(256)]
    pub fn plane_counts(
        plane: &[u16],
        wh: u32,
        mut counts: &mut [cuda_device::atomic::DeviceAtomicU32],
    ) {
        let i = thread::index_1d().get();
        if i >= wh as usize {
            return;
        }
        let o = plane[i] as usize;
        if o != 0 && o < counts.len() {
            counts[o].fetch_add(1, cuda_device::atomic::AtomicOrdering::Relaxed);
        }
    }

    /// Copy a slice of the device plane out, for the renderer.
    #[kernel]
    #[launch_bounds(256)]
    pub fn plane_copy(plane: &[u16], wh: u32, mut out: &mut [u16]) {
        let i = thread::index_1d().get();
        if i >= wh as usize {
            return;
        }
        out[i] = plane[i];
    }
}

// `kernels::device::{load, LoadedModule, *kernels}` is the module the
// `#[cuda_module]` attribute generates.
