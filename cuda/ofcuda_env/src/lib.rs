//! `ofcuda_env` - ONE per-tick environment step composed from the four verified
//! CUDA slices.
//!
//! Until this crate existed, four slices were each bit-exact against the
//! engine but none of them talked to the others, and the per-tick claim budget
//! was taken from the record (exogenous) instead of computed. This crate runs
//! the whole tick in the engine's order and computes the budget.
//!
//! ## The composed order, with citations
//!
//! 1. **Per-tick economy** - `execution/player.rs:82-88` `PlayerExecution::tick`
//!    -> `troop_increase_rate_raw_for` + `wire.gold_addition_rate` +
//!    `add_troops`. Reused from `ofcuda_econ` (`step_row`), which reproduces the
//!    device-vs-host path 7791/7791. There is **no per-tile economy** in stages
//!    0-7 (measured negative, not re-litigated here).
//! 2. **The float claim budget** - `execution/attack.rs:239-255` calls
//!    `wire.attack_tiles_per_tick(troop_count, owner_type, target_is_player,
//!    defender_troops, border_size + random.next_int(0, 5))` with
//!    `border_size = self.border_tiles.len()` (`attack.rs:240`). The draw is
//!    taken **before** the pop loop (`attack.rs:253`), exactly once per tick.
//!    For a terra-nullius target the body is `num_adjacent * 2.0`
//!    (`core/config.rs:496-501`) - the budget does **not** read the troop count
//!    there; the troop count only enters the per-claim decrement.
//! 3. **The persistent-heap expansion** - `attack.rs:257-322`: the carried
//!    `to_conquer` heap (`attack.rs:17`), refilled only when empty
//!    (`attack.rs:264-268` -> `refresh_to_conquer`, `attack.rs:1265-1274`),
//!    `remove_border_tile` on **every** pop including skipped ones
//!    (`attack.rs:275`), the still-terra-nullius + attacker-neighbour guards
//!    (`attack.rs:284`), the land guard (`attack.rs:288`), and in-tick
//!    `add_neighbors` **before** the conquer (`attack.rs:292`).
//! 4. **Cluster capture** - `ofcuda_cluster`'s flood-fill (mark-on-discovery,
//!    neighbours8 dx-major, decide-and-remove replayed serially). Wired here
//!    through [`cluster_capture`]; it fires only on a player loss, which does
//!    not happen inside the measured window.
//! 5. **State-plane update** - the claims folded into the `u16` ownership plane.
//! 6. **FNV-1a-64** - `ofcuda_hash`: offset `0xcbf29ce484222325`, prime
//!    `0x100000001b3`, `w*h` `u16` as little-endian bytes, lowercase `0x` + 16
//!    hex digits. The contract is fixed and is never relaxed here.
//!
//! ## What this crate had to add that no slice carried
//!
//! The engine's budget reads `self.border_tiles` (`attack.rs:240`), and
//! `add_border_tile` is called for exactly the neighbours that pass the water +
//! owner==target filters (`attack.rs:1353-1363`), i.e. the enqueued set, with
//! `remove_border_tile` on every pop (`attack.rs:275`). `ofcuda_tick`'s
//! `add_neighbors`/`refresh_to_conquer` deliberately do **not** carry that set
//! (its README says the budget is exogenous), so [`Attack`] maintains it here -
//! that is what turns the budget from assumed into computed. No file in
//! `ofcuda_tick`, `ofcuda_cluster`, `ofcuda_econ` or `ofcuda_hash` was changed.

use ofcuda_tick::{
    add_neighbors, has_attacker_neighbor, is_terra_nullius, mag2_from_terrain, neighbors4, Heap,
    Prng, ORDER_NSWE,
};

/// `util.rs` `within(value, min, max)`.
#[inline]
pub fn within(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// `core/config.rs:484-501` `GameConfig::attack_tiles_per_tick`.
///
/// `num_adjacent` is the engine's `border_size + random.next_int(0, 5)`
/// (`attack.rs:253`). For a player target the body is the defender-troop
/// formula; for terra nullius it is `num_adjacent * 2.0`.
pub fn attack_tiles_per_tick(
    attack_troops: f64,
    defender_is_player: bool,
    defender_troops: f64,
    num_adjacent: f64,
) -> f64 {
    if defender_is_player {
        within(
            ((5.0 * attack_troops) / defender_troops) * 2.0,
            0.01,
            0.5,
        ) * num_adjacent
            * 3.0
    } else {
        num_adjacent * 2.0
    }
}

/// `game.rs:890-896`: the terrain cost magnitudes `mag` (Plains 80, Highland
/// 100, Mountain 120) and the attacker's per-claim troop loss, `mag / 10` for a
/// Bot attacker and `mag / 5` otherwise.
pub fn terrain_mag(terrain: u8) -> f64 {
    match mag2_from_terrain(terrain) {
        2 => 80.0,
        3 => 100.0,
        4 => 120.0,
        _ => 80.0,
    }
}

pub fn terrain_speed(terrain: u8) -> f64 {
    match mag2_from_terrain(terrain) {
        2 => 16.5,
        3 => 20.0,
        4 => 25.0,
        _ => 16.5,
    }
}

pub fn attacker_troop_loss(terrain: u8, attacker_is_bot: bool) -> f64 {
    let mag = terrain_mag(terrain);
    if attacker_is_bot {
        mag / 10.0
    } else {
        mag / 5.0
    }
}

/// `game.rs:896` `tiles_used = within((2000.0 * speed.max(10.0)) /
/// attack_troops, 5.0, 100.0)` - the value decremented off the budget at
/// `attack.rs:313`.
pub fn tiles_used(attack_troops: f64, terrain: u8) -> f64 {
    let speed = terrain_speed(terrain);
    within((2000.0 * speed.max(10.0)) / attack_troops, 5.0, 100.0)
}

/// `execution/ai_attack.rs:9-18` `land_attack_troops` - the engine's own
/// construction of a land attack's starting troops, and therefore the exact
/// `f64` the budget decrement divides by.
///
/// This is what the record cannot carry: `tick_dump.rs` writes
/// `AttackSnapshot.troops = troops as i64` (`bin/tick_dump.rs:326`), which
/// truncates. Feeding the truncated integer to `tiles_used` shifts every
/// per-claim cost by up to ~1/1363, which flips claims on knife-edge ticks.
pub fn land_attack_troops(owner_troops: f64, max_troops: f64, expand_ratio: f64) -> Option<f64> {
    let troops = owner_troops - max_troops * expand_ratio;
    if troops < 1.0 { None } else { Some(troops) }
}

/// One land attack: the persistent heap plus the engine's `border_tiles` set.
pub struct Attack {
    pub owner_sid: u16,
    pub target_sid: u16,
    pub is_bot: bool,
    pub troops: f64,
    pub heap: Heap,
    pub pr: Prng,
    /// `self.border_tiles` (`attack.rs:18`) - the budget's only adjacency input.
    pub border: Vec<u32>,
    /// The whole-life claim list (`claimed_so_far` + this tick's claims).
    pub claims: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct TickOut {
    pub budget: f64,
    pub budget_draw: i32,
    pub border_size: u32,
    pub claims: Vec<u32>,
    pub refilled: bool,
    pub killed: bool,
    pub troops_after: f64,
}

impl Attack {
    pub fn new(owner_sid: u16, target_sid: u16, is_bot: bool, troops: f64, seed: i32) -> Self {
        Attack {
            owner_sid,
            target_sid,
            is_bot,
            troops,
            heap: Heap::new(),
            pr: Prng::new(seed),
            border: Vec::new(),
            claims: Vec::new(),
        }
    }

    fn border_insert(&mut self, t: u32) {
        if !self.border.contains(&t) {
            self.border.push(t);
        }
    }

    fn border_remove(&mut self, t: u32) {
        self.border.retain(|x| *x != t);
    }

    /// `refresh_to_conquer` (`attack.rs:1265-1274`): clear the heap **and** the
    /// attack's border set, then offer every neighbour of the owner's current
    /// border.
    pub fn refresh(
        &mut self,
        owner_border: &[u32],
        plane: &[u16],
        terrain: &[u8],
        w: u32,
        h: u32,
        tick: u32,
    ) -> u32 {
        self.heap.clear();
        self.border.clear();
        let mut n = 0;
        for &bt in owner_border {
            n += self.offer_neighbours(bt, plane, terrain, w, h, tick);
        }
        n
    }

    /// `attack.rs:1353-1386`: insert the candidate border tile for exactly the
    /// neighbours `add_neighbors` enqueues, then let the crate's own
    /// `add_neighbors` do the draw-binding and priority (it draws exactly one
    /// `next_int(0, 7)` per enqueued neighbour, in N,S,W,E visit order).
    fn offer_neighbours(
        &mut self,
        tile: u32,
        plane: &[u16],
        terrain: &[u8],
        w: u32,
        h: u32,
        tick: u32,
    ) -> u32 {
        let mut nbuf = [0u32; 4];
        let n = neighbors4(ORDER_NSWE, tile, w, h, &mut nbuf);
        for i in 0..n as usize {
            let nb = nbuf[i];
            if terrain[nb as usize] & 0x80 == 0 {
                continue; // water (attack.rs:1354)
            }
            if !is_terra_nullius(plane, &self.claims, nb) {
                continue; // wrong owner (attack.rs:1359)
            }
            self.border_insert(nb); // attack.rs:1363
        }
        add_neighbors(
            &mut self.heap,
            &mut self.pr,
            tile,
            plane,
            &self.claims,
            self.owner_sid,
            terrain,
            w,
            h,
            ORDER_NSWE,
            tick,
        )
    }

    /// `AttackExecution::tick` (`attack.rs:206-324`) with a **computed** float
    /// budget. `plane` is the ownership plane at the start of the tick;
    /// `owner_border` is the owner's `border_tiles` at the start of the tick
    /// (the input to a mid-tick `refresh_to_conquer`).
    #[allow(clippy::too_many_arguments)]
    pub fn tick(
        &mut self,
        plane: &[u16],
        terrain: &[u8],
        w: u32,
        h: u32,
        tick: u32,
        owner_border: &[u32],
        defender_troops: f64,
        defender_is_player: bool,
    ) -> TickOut {
        // attack.rs:233-237
        let mut troop_count = self.troops;
        // attack.rs:239-255 - the one extra draw, taken BEFORE the pop loop.
        let draw = self.pr.next_int(0, 5);
        let border_size = self.border.len() as u32;
        let budget = attack_tiles_per_tick(
            troop_count,
            defender_is_player,
            defender_troops,
            border_size as f64 + draw as f64,
        );
        let mut num = budget;
        let n0 = self.claims.len();
        let mut refilled = false;
        let mut killed = false;

        // attack.rs:257 `while num_tiles_per_tick > 0.0`
        while num > 0.0 {
            // attack.rs:258-262
            if troop_count < 1.0 {
                self.troops = 0.0;
                killed = true;
                return TickOut {
                    budget,
                    budget_draw: draw,
                    border_size,
                    claims: self.claims[n0..].to_vec(),
                    refilled,
                    killed,
                    troops_after: 0.0,
                };
            }
            // attack.rs:264-268
            if self.heap.is_empty() {
                self.refresh(owner_border, plane, terrain, w, h, tick);
                refilled = true;
                break;
            }
            let Some((tile, _pri)) = self.heap.dequeue() else {
                break; // attack.rs:271
            };
            self.border_remove(tile); // attack.rs:275 (also for skipped pops)
            if !is_terra_nullius(plane, &self.claims, tile) {
                continue; // attack.rs:284
            }
            if !has_attacker_neighbor(plane, &self.claims, self.owner_sid, tile, w, h, ORDER_NSWE) {
                continue; // attack.rs:284
            }
            if terrain[tile as usize] & 0x80 == 0 {
                continue; // attack.rs:288
            }
            self.offer_neighbours(tile, plane, terrain, w, h, tick); // attack.rs:292
            num -= tiles_used(troop_count, terrain[tile as usize]); // attack.rs:313
            troop_count -= attacker_troop_loss(terrain[tile as usize], self.is_bot); // attack.rs:314
            self.claims.push(tile); // attack.rs:320 conquer
        }
        self.troops = if troop_count > 0.0 { troop_count } else { 0.0 };
        TickOut {
            budget,
            budget_draw: draw,
            border_size,
            claims: self.claims[n0..].to_vec(),
            refilled,
            killed,
            troops_after: self.troops,
        }
    }
}

// ---------------------------------------------------------------------------
// The other three stages, as libraries
// ---------------------------------------------------------------------------

/// Stage 1: `execution/player.rs:82-88` per-player troops/gold step, from
/// `ofcuda_econ`.
pub fn economy_step(row: &ofcuda_econ::Row) -> ofcuda_econ::StepOut {
    ofcuda_econ::step_row(row)
}

/// Stage 4: the cluster capture, from `ofcuda_cluster` - `prepare` builds the
/// border/shape tables, `decide_cpu` replays the engine's decide-and-remove
/// serially, `clusters_from` returns the captured `(cluster, tile)` pairs.
pub fn cluster_capture(
    case: &ofcuda_cluster::Case,
    terrain: &[u8],
    victim: u16,
) -> Result<Vec<(u32, u32)>, String> {
    let prep = ofcuda_cluster::prepare(case, victim)?;
    let _run = ofcuda_cluster::decide_cpu(case, terrain, &prep);
    Ok(ofcuda_cluster::clusters_from(&prep))
}

/// Stage 5: the `u16` ownership plane, via `ofcuda_tick`'s plane builder.
pub fn state_plane(players: &[(u32, Vec<u32>)], w: u32, h: u32) -> Vec<u16> {
    ofcuda_tick::state_plane_from_players(w, h, players)
}

/// Stage 6: the FNV-1a-64 state hash, via `ofcuda_hash`.
pub fn state_hash(plane: &[u16]) -> u64 {
    ofcuda_hash::fnv1a_u16_le(ofcuda_hash::FNV_OFFSET_BASIS, plane)
}

/// The whole chain as one named pipeline, for the README's order table.
pub const PIPELINE: [&str; 6] = [
    "economy (player.rs:82-88)",
    "float budget (attack.rs:239-255, config.rs:484-501)",
    "expansion (attack.rs:257-322)",
    "cluster capture (ofcuda_cluster)",
    "plane update",
    "FNV-1a-64 (ofcuda_hash)",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_is_num_adjacent_times_two_for_terra_nullius() {
        // config.rs:496-501
        assert_eq!(attack_tiles_per_tick(1363.0, false, 0.0, 87.0), 174.0);
        assert_eq!(attack_tiles_per_tick(10.0, false, 0.0, 0.0), 0.0);
    }

    #[test]
    fn budget_diverges_when_the_defender_is_a_player() {
        // config.rs:488-496: within((5*1000/5000)*2,0.01,0.5) * 10 * 3
        let b = attack_tiles_per_tick(1000.0, true, 5000.0, 10.0);
        assert_eq!(b, 15.0);
    }

    #[test]
    fn tiles_used_matches_the_engine_magnitudes() {
        // game.rs:896: 2000*speed/attack_troops, clamped to [5,100].
        // Terrain bytes carry the land bit 0x80: 0x80|15 = 143 -> magnitude 15
        // -> Highland -> speed 20, mag 100.
        assert_eq!(terrain_mag(143), 100.0);
        assert_eq!(terrain_mag(0x80 | 15), terrain_mag(143));
        assert!((tiles_used(1363.0, 143) - 40000.0 / 1363.0).abs() < 1e-12);
        // 0x80|8 = 136 -> magnitude 8 -> Plains -> speed 16.5, mag 80.
        assert_eq!(terrain_mag(136), 80.0);
        assert!((tiles_used(100.0, 136) - 100.0).abs() < 1e-12); // clamped
        assert_eq!(terrain_mag(0x80 | 24), 120.0); // Mountain
        assert_eq!(tiles_used(100000.0, 0x80 | 24), 5.0); // clamped low
    }

    #[test]
    fn attacker_loss_is_bot_tenth_and_other_fifth() {
        assert_eq!(attacker_troop_loss(143, true), 10.0); // Highland, Bot
        assert_eq!(attacker_troop_loss(143, false), 20.0); // Highland, other
        assert_eq!(attacker_troop_loss(136, true), 8.0); // Plains
        assert_eq!(attacker_troop_loss(0x80 | 24, true), 12.0); // Mountain
    }

    #[test]
    fn land_attack_troops_is_the_engine_formula() {
        // ai_attack.rs:9-18
        assert_eq!(land_attack_troops(11334.0, 10000.0, 0.14), Some(9934.0));
        assert_eq!(land_attack_troops(10.0, 100.0, 0.5), None);
    }

    #[test]
    fn pipeline_order_is_the_engine_order() {
        assert_eq!(PIPELINE.len(), 6);
        assert!(PIPELINE[1].contains("budget"));
    }
}
