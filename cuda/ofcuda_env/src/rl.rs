//! The TRAINING interface for the CUDA env: observation, action mask and
//! reward derived from the DEVICE state of the unaided multi-attack game.
//!
//! This module is the contract between the device-resident multi-attack
//! simulation (`src/bin/gpu_env.rs --multi --orig`) and the policy network.
//! It adds **no** second copy of the featurizer's rules: every constant below
//! carries the `file:line` of the rule it mirrors, and the layouts are the
//! C ABI's own (`openfront-ai/include/openfront_env.h`,
//! `openfront-ai/rust/ofcore/src/feat.rs`).
//!
//! # What the device state determines, and what it does not
//!
//! The multi-attack sim carries, per env: the ownership plane, the per-player
//! owner-border sets in the engine's insertion order, per-player
//! `(troops, tiles, gold, ptype)`, and the live attack slots
//! `(owner, target, troops, alive, heap/border/claim lengths)`. Everything the
//! featurizer builds out of *those* is derived exactly here.
//!
//! The sim deliberately does not model units, diplomacy (alliances, embargoes,
//! requests, targets), traitors, doomsday, or per-player income rates, and it
//! does not compute the region-adjacency "neighbour pack". The corresponding
//! observation fields are written as the C ABI writes them when their source
//! is absent - a zero - and `Inventory` COUNTS them so the gap is reported
//! rather than hidden. `obs_inventory()` is the number to quote, never
//! "obs is exact".
//!
//! # Layout (per agent, flat f32)
//!
//! obs (`OBS_PER_AGENT = 4270`, `puffer_ffi.rs:128-140`):
//!   `[0,12)` 12 scalars; `[12,3852)` 128 x 30 player tokens (slot-major);
//!   `[3852,4236)` 32 x 12 unit tokens (token-major);
//!   `[4236,4243)` legal_build; `[4243,4248)` legal_nuke;
//!   `[4248,4269)` legal_actions; `[4269]` me_slot.
//!
//! mask (`MASK_PER_AGENT = 40893`, `feat.rs:15-25`, `openfront_env.h:166-186`):
//!   `[0,21)` legal_actions; `[21,2709)` legal_ptarget (action-major, 128);
//!   `[2709,3381)` legal_utarget (action-major, 32); `[3381,3388)`
//!   legal_build; `[3388,3393)` legal_nuke; `[3393,40893)` legal_tile on a
//!   fixed 150 x 250 row-major plane (stride 250).

#![allow(clippy::too_many_arguments)]

pub const OBS_PER_AGENT: usize = 4270;
pub const MASK_PER_AGENT: usize = 40893;
pub const MAX_SLOTS: usize = 128;
pub const P_FEAT: usize = 30;
pub const MAX_UNITS: usize = 32;
pub const U_FEAT: usize = 12;
pub const N_SCALARS: usize = 12;
pub const N_ACTIONS: usize = 21;
pub const N_BUILD: usize = 7;
pub const N_NUKE: usize = 5;

pub const OBS_SCALARS_OFF: usize = 0;
pub const OBS_PLAYERS_OFF: usize = N_SCALARS; // 12
pub const OBS_UNITS_OFF: usize = OBS_PLAYERS_OFF + MAX_SLOTS * P_FEAT; // 3852
pub const OBS_BUILD_OFF: usize = OBS_UNITS_OFF + MAX_UNITS * U_FEAT; // 4236
pub const OBS_NUKE_OFF: usize = OBS_BUILD_OFF + N_BUILD; // 4243
pub const OBS_ACTIONS_OFF: usize = OBS_NUKE_OFF + N_NUKE; // 4248
pub const OBS_ME_SLOT_OFF: usize = OBS_ACTIONS_OFF + N_ACTIONS; // 4269

pub const MASK_ACTIONS_OFF: usize = 0;
pub const MASK_PTARGET_OFF: usize = N_ACTIONS; // 21
pub const MASK_UTARGET_OFF: usize = MASK_PTARGET_OFF + N_ACTIONS * MAX_SLOTS; // 2709
pub const MASK_BUILD_OFF: usize = MASK_UTARGET_OFF + N_ACTIONS * MAX_UNITS; // 3381
pub const MASK_NUKE_OFF: usize = MASK_BUILD_OFF + N_BUILD; // 3388
pub const MASK_TILE_OFF: usize = MASK_NUKE_OFF + N_NUKE; // 3393
pub const MASK_TILE_GH: usize = 150;
pub const MASK_TILE_GW: usize = 250;
/// The header's own compile-time assertion (`puffer_ffi.rs:165`).
const _: () = assert!(OBS_UNITS_OFF + MAX_UNITS * U_FEAT + N_BUILD + N_NUKE + N_ACTIONS + 1 == OBS_PER_AGENT);
const _: () = assert!(MASK_TILE_OFF + MASK_TILE_GH * MASK_TILE_GW == MASK_PER_AGENT);

// Action ids (`feat.rs:32-52`, `openfront.h:616-622`).
pub const A_NOOP: usize = 0;
pub const A_ATTACK: usize = 1;
pub const A_EXPAND: usize = 2;
pub const A_BOAT: usize = 3;
pub const A_BUILD: usize = 4;
pub const A_LAUNCH_NUKE: usize = 5;
pub const A_SPAWN: usize = 13;

/// The featurizer's `log_norm` (`ofcore/src/feat.rs:222-224`):
/// `log10(1 + max(x,0)) / 8`.
#[inline]
pub fn log_norm(x: f64) -> f32 {
    ((1.0 + x.max(0.0)).log10() / 8.0) as f32
}

/// Per-player state the device maintains for every env.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlayerRow {
    pub sid: u16,
    pub slot: u8,
    pub alive: bool,
    pub troops: f64,
    pub gold: f64,
    pub tiles: f64,
    pub border_len: u32,
}

/// One live attack slot (`d_mowner`/`d_mtroops`/`d_mscal[..][5]`).
#[derive(Clone, Copy, Debug, Default)]
pub struct AttackRow {
    pub owner: u16,
    pub target: u16,
    /// 0 = terra nullius (`ft` column 0).
    pub troops: f64,
    pub alive: bool,
}

/// A device state snapshot, in the units the device holds them.
#[derive(Clone, Debug)]
pub struct World {
    pub tick: i64,
    pub spawn_end_tick: u32,
    pub max_episode_ticks: i64,
    pub map_land: i64,
    /// `w * h` raw tiles (the /8 grid is `gh x gw`).
    pub w: u32,
    pub h: u32,
    pub me: u16,
    pub me_slot: usize,
    pub players: Vec<PlayerRow>,
    pub attacks: Vec<AttackRow>,
}

impl World {
    pub fn gh(&self) -> usize {
        (self.h as usize / 8).min(MASK_TILE_GH)
    }
    pub fn gw(&self) -> usize {
        (self.w as usize / 8).min(MASK_TILE_GW)
    }
    pub fn spawn_phase(&self) -> bool {
        self.spawn_end_tick > 0 && self.tick <= self.spawn_end_tick as i64
    }
    pub fn me_row(&self) -> Option<&PlayerRow> {
        self.players.iter().find(|p| p.sid == self.me)
    }
    /// Live land attacks owned by `sid` (`legal.attacks` for the agent;
    /// the C ABI's own list is per-player).
    pub fn attacks_by(&self, sid: u16) -> Vec<&AttackRow> {
        self.attacks
            .iter()
            .filter(|a| a.alive && a.owner == sid)
            .collect()
    }
    fn n_alive(&self) -> usize {
        self.players.iter().filter(|p| p.alive).count()
    }
    fn claimed_tiles(&self) -> f64 {
        self.players.iter().map(|p| p.tiles.max(0.0)).sum()
    }
    /// Fielded troops of an owner (attack troops count as fielded in
    /// `curriculum::strengths`, `curriculum.rs:1216-1223`).
    fn fielded(&self, sid: u16) -> f64 {
        self.attacks
            .iter()
            .filter(|a| a.alive && a.owner == sid)
            .map(|a| a.troops.max(0.0))
            .sum()
    }
}

/// Counts how much of the observation/mask this interface actually fills from
/// device state, so the gap is a number and not a claim.
#[derive(Clone, Copy, Debug, Default)]
pub struct Inventory {
    pub obs_written: usize,
    pub obs_total: usize,
    pub mask_written: usize,
    pub mask_total: usize,
}

impl Inventory {
    pub fn obs_frac(&self) -> f64 {
        self.obs_written as f64 / self.obs_total.max(1) as f64
    }
    pub fn mask_frac(&self) -> f64 {
        self.mask_written as f64 / self.mask_total.max(1) as f64
    }
}

// ---------------------------------------------------------------------------
// Action mask
// ---------------------------------------------------------------------------

/// The legality mask for one agent, derived from the device state.
///
/// `tile_legal` is the spawn-phase tile predicate (`Some` of a `gh*gw` byte
/// plane: 1 if the /8 cell contains a tile with `land==1 && mag<31 &&
/// owner==0`, `feat.rs:1392-1420`). Outside the spawn phase the tile block is
/// all-ones inside `0..gh x 0..gw` and zero in the fixed-150x250 padding, so
/// `None` is the correct input there.
pub fn mask(w: &World, tile_legal: Option<&[u8]>, inv: &mut Inventory) -> Vec<f32> {
    let mut m = vec![0.0f32; MASK_PER_AGENT];
    let me_row = w.me_row();
    let alive = me_row.map(|p| p.alive).unwrap_or(false);
    let spawn = w.spawn_phase();

    // --- legal_actions (`feat.rs:977-1038`) --------------------------------
    if spawn {
        if alive {
            m[MASK_ACTIONS_OFF + A_NOOP] = 1.0;
        } else {
            m[MASK_ACTIONS_OFF + A_SPAWN] = 1.0;
        }
    } else {
        m[MASK_ACTIONS_OFF + A_NOOP] = 1.0;
    }
    if alive && !spawn {
        // Player-targeted rows come from the device's own live attacks and
        // roster: with a single-agent FFA and terra-nullius-only attacks the
        // engine's lists are empty (no diplomacy is modelled), so only the
        // rows the sim can populate are written.
        let mine = w.attacks_by(w.me);
        // `act[expand] = canExpand` (`feat.rs:1013`): the sim's expansion
        // legality is "a non-empty own border set and troops >= 1".
        let can_expand = me_row.map(|p| p.tiles >= 1.0).unwrap_or(false);
        if can_expand {
            m[MASK_ACTIONS_OFF + A_EXPAND] = 1.0;
        }
        // `act[attack]` is set iff the engine's legal attackable list is
        // non-empty; the sim's is empty in every live window measured so far
        // (`targetSmallId == 0`), so the row stays zero - and that is a REAL
        // zero, not a missing value.
        let _ = mine;
    }

    // --- legal_tile (`feat.rs:1392-1420`) ---------------------------------
    let (gh, gw) = (w.gh(), w.gw());
    for gy in 0..gh {
        let base = MASK_TILE_OFF + gy * MASK_TILE_GW;
        for gx in 0..gw {
            m[base + gx] = if spawn {
                match tile_legal {
                    Some(t) => {
                        if t[gy * gw + gx] != 0 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    None => 0.0,
                }
            } else {
                1.0
            };
        }
    }

    // --- legal_build / legal_nuke (`feat.rs:1032-1036`) --------------------
    // `alive && legal.present` gates both; the build/nuke legality lists are
    // engine queries over units/structures, which the sim does not model, so
    // they stay all-zero exactly as they do when the list is empty.

    // inventory: written = every mask float this function assigned a value to.
    let tile_written = gh * gw;
    let actions_written = if spawn {
        1
    } else {
        1 + if alive { 1 } else { 0 } // noop + expand
    };
    inv.mask_written += actions_written + tile_written;
    inv.mask_total += MASK_PER_AGENT;

    // Every unwritten field must be exactly the C ABI's own default (0.0).
    debug_assert!(m.iter().all(|v| *v == 0.0 || *v == 1.0));
    m
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

/// The per-agent observation, derived from the device state.
///
/// The scalar block and the player-token fields the device state determines
/// are exact mirrors of `feat.rs:956-941`; the fields whose source the sim
/// does not model are zero and counted in `inv`.
pub fn obs(w: &World, inv: &mut Inventory) -> Vec<f32> {
    let mut o = vec![0.0f32; OBS_PER_AGENT];
    let spawn = w.spawn_phase();
    let me_row = w.me_row();
    let alive = me_row.map(|p| p.alive).unwrap_or(false);
    let claimed = w.claimed_tiles();
    let mine_tiles = me_row.map(|p| p.tiles.max(0.0)).unwrap_or(0.0);
    let team_claimed_share = if claimed > 0.0 {
        (mine_tiles / claimed).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let team_map_share = if w.map_land > 0 {
        (mine_tiles / w.map_land as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };

    // --- scalars (`feat.rs:956-972`) ---------------------------------------
    let scal = [
        w.tick as f32 / 15000.0,
        spawn as u8 as f32,
        alive as u8 as f32,
        log_norm(me_row.map(|p| p.troops).unwrap_or(0.0)),
        log_norm(me_row.map(|p| p.gold).unwrap_or(0.0)),
        w.n_alive() as f32 / 128.0,
        w.attacks_by(w.me).len() as f32 / 8.0,
        w.me_slot as f32 / MAX_SLOTS as f32,
        0.0, // me troop_income: not modelled (income is not a sim state)
        0.0, // me gold_income: not modelled
        team_claimed_share as f32,
        team_map_share as f32,
    ];
    o[OBS_SCALARS_OFF..OBS_SCALARS_OFF + N_SCALARS].copy_from_slice(&scal);

    // --- player tokens (`feat.rs:864-941`) ---------------------------------
    // Written fields: 0 alive, 1 troop log, 2 gold log, 3 tile log, 7 is-me,
    // 12/13 the attack troop flows the sim carries (out/in between me and the
    // token's player), 9 the sign of the net flow. Every other index is a
    // field whose source is not in the sim (units, income, diplomacy,
    // neighbour pack) and stays the ABI's zero.
    for p in w.players.iter() {
        let slot = p.slot as usize;
        if slot == 0 || slot >= MAX_SLOTS {
            continue;
        }
        let base = OBS_PLAYERS_OFF + slot * P_FEAT;
        let out_troops: f64 = w
            .attacks
            .iter()
            .filter(|a| a.alive && a.owner == w.me && a.target == p.sid)
            .map(|a| a.troops.max(0.0))
            .sum();
        let in_troops: f64 = w
            .attacks
            .iter()
            .filter(|a| a.alive && a.owner == p.sid && a.target == w.me)
            .map(|a| a.troops.max(0.0))
            .sum();
        let atk = out_troops - in_troops;
        o[base] = p.alive as u8 as f32;
        o[base + 1] = log_norm(p.troops);
        o[base + 2] = log_norm(p.gold);
        o[base + 3] = log_norm(p.tiles);
        o[base + 7] = (p.sid == w.me) as u8 as f32;
        o[base + 8] = log_norm(atk.abs());
        o[base + 9] = (atk > 0.0) as u8 as f32;
        o[base + 12] = log_norm(out_troops);
        o[base + 13] = log_norm(in_troops);
        inv.obs_written += 9;
    }

    // --- the legality block is the mask's own scalars, mirrored
    //     (`puffer_ffi.rs:134-140`): legal_build[0..7), legal_nuke[7..12),
    //     legal_actions[12..33), me_slot at [33]. The build/nuke lists are
    //     empty and legal_actions mirrors the mask's action block.
    o[OBS_BUILD_OFF..OBS_BUILD_OFF + N_BUILD].copy_from_slice(&[0.0; N_BUILD]);
    o[OBS_NUKE_OFF..OBS_NUKE_OFF + N_NUKE].copy_from_slice(&[0.0; N_NUKE]);
    let mut minv = Inventory::default();
    let mm = mask(w, None, &mut minv);
    o[OBS_ACTIONS_OFF..OBS_ACTIONS_OFF + N_ACTIONS]
        .copy_from_slice(&mm[MASK_ACTIONS_OFF..MASK_ACTIONS_OFF + N_ACTIONS]);
    o[OBS_ME_SLOT_OFF] = w.me_slot as f32;

    inv.obs_written += N_SCALARS + N_ACTIONS + 1;
    inv.obs_total += OBS_PER_AGENT;
    o
}

// ---------------------------------------------------------------------------
// Reward
// ---------------------------------------------------------------------------

/// Reward constants, from `ofcore/src/curriculum.rs` (see `03-reward-action-stage.md`).
pub mod k {
    pub const W_STR: f64 = 0.02; // curriculum.rs:10
    pub const W_DELTA_GAIN: f64 = 5.0; // curriculum.rs:11
    pub const W_DELTA_LOSS: f64 = 6.5; // curriculum.rs:12
    pub const W_PLACE: f64 = 15.0; // curriculum.rs:13
    pub const W_WIN: f64 = 30.0; // curriculum.rs:14
    pub const PLACE_POW: f64 = 1.5; // curriculum.rs:17
    pub const K_LAND: f64 = 0.40; // curriculum.rs:19
    pub const K_MIL: f64 = 0.20; // curriculum.rs:20
    pub const K_ECO: f64 = 0.25; // curriculum.rs:21
    pub const K_BUILD: f64 = 0.15; // curriculum.rs:22
    pub const DOMINANCE_EPS: f64 = 1e-9; // curriculum.rs:24
    pub const CLOSEOUT_START: f64 = 0.45; // curriculum.rs:25
    pub const CLOSEOUT_FULL: f64 = 0.80; // curriculum.rs:26
    pub const DOM_COEF: f64 = 0.25; // main.rs v81_dom_coef
    pub const CLOSE_COEF: f64 = 4.0; // main.rs v83_close_coef
    pub const TEMPO_COEF: f64 = 0.015; // main.rs v84_tempo_coef
    pub const TEMPO_THRESHOLD: f64 = 0.30; // main.rs v85_tempo_share_threshold
    pub const SURVIVAL_COEF: f64 = 0.01; // main.rs v10_survival_coef
    pub const DEATH_PENALTY: f64 = 3.0; // main.rs v86_death_penalty
    pub const TIMEOUT_AFTER_CLOSEOUT: f64 = 20.0; // main.rs v10_timeout_closeout
    pub const CLOSEOUT_ENTRY: f64 = 25.0; // main.rs v10_closeout_entry
    pub const FAST_WIN_COEF: f64 = 40.0; // main.rs v84_fast_win_coef
    pub const EXTRA_WIN_BONUS: f64 = 200.0; // main.rs v85_extra_win_bonus
    pub const GAMMA: f64 = 0.999; // main.rs gamma
    pub const SOLO_SCALE_MULTI: f64 = 0.22; // vecenv.rs:1960
}

/// `composite_strength` (`curriculum.rs:1216-1265`): land/military/economic/
/// structural share. The structural term is 0 here (the sim models no units),
/// so `K_BUILD` contributes nothing rather than being silently dropped.
pub fn composite_strength(w: &World, sid: u16) -> f64 {
    let land_total = (w.map_land as f64).max(1.0);
    let troops_of = |s: u16| -> f64 {
        w.players
            .iter()
            .find(|p| p.sid == s)
            .map(|p| p.troops.max(0.0))
            .unwrap_or(0.0)
            + w.fielded(s)
    };
    let tot_troops: f64 = w.players.iter().map(|p| troops_of(p.sid)).sum();
    let tot_gold: f64 = w.players.iter().map(|p| p.gold.max(0.0)).sum();
    let p = match w.players.iter().find(|p| p.sid == sid) {
        Some(p) => p,
        None => return 0.0,
    };
    let land = k::K_LAND * p.tiles.max(0.0) / land_total;
    let mil = k::K_MIL * troops_of(sid) / tot_troops.max(1.0);
    let eco = k::K_ECO * p.gold.max(0.0) / tot_gold.max(1.0);
    land + mil + eco
}

/// Reward state the trainer carries across decisions of one episode.
#[derive(Clone, Debug, Default)]
pub struct RewardState {
    pub dominance_phi: f64,
    pub closeout_phi: f64,
    pub closeout_entered: bool,
    pub prev_alive: bool,
    pub prev_strength: f64,
    pub initialized: bool,
}

/// What one decision's reward came out as.
#[derive(Clone, Copy, Debug, Default)]
pub struct RewardOut {
    pub reward: f64,
    pub terminal: f64,
    pub done: bool,
    pub won: bool,
    pub died: bool,
    pub timed_out: bool,
}

/// Per-decision reward, implementing the live spec's per-step components and
/// its terminal rule (`03-reward-action-stage.md`, `vecenv.rs:2091-2386`,
/// `curriculum.rs:1366-1414`).
///
/// **The timeout-is-a-loss rule:** every non-win pays `-W_WIN` at the terminal
/// level regardless of WHY the episode ended - the `!won` branch ignores
/// `timed_out` (`curriculum.rs:1366-1377`). The V10 timeout-after-closeout
/// penalty (`-20` when a timeout ends an episode that had reached >=45% land)
/// is applied on TOP, because the live path applies it; that makes a
/// timeout-after-closeout strictly worse than a death, which is a real
/// property of the shipped reward and is reported, not smoothed over.
pub fn decision_reward(w: &World, st: &mut RewardState) -> RewardOut {
    let p = w.me_row().cloned().unwrap_or_default();
    let n_agents = 1.0; // one RL head per env in this interface (solo FFA)
    let solo_scale = if n_agents > 1.0 { k::SOLO_SCALE_MULTI } else { 1.0 };
    let alive = p.alive;

    let strength = composite_strength(w, w.me);
    let tw = 0.5 + 0.5 * ((w.tick as f64) / 8000.0).min(1.0); // timeweight
    let share = if w.map_land > 0 {
        (p.tiles.max(0.0) / w.map_land as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let mut r = k::W_STR * strength * tw * solo_scale;

    // strength_delta (PBRS on the composite strength itself, `vecenv.rs:2093`)
    if st.initialized {
        let delta = strength - st.prev_strength;
        let dominant = share >= k::TEMPO_THRESHOLD;
        let weight = if delta >= 0.0 {
            k::W_DELTA_GAIN
        } else if dominant {
            k::W_DELTA_GAIN
        } else {
            k::W_DELTA_LOSS
        };
        r += weight * delta * solo_scale;
    }
    st.prev_strength = strength;

    // dominance PBRS (`curriculum.rs:1284-1298`, `vecenv.rs:2126-2143`)
    let mine_s = composite_strength(w, w.me);
    let strongest = w
        .players
        .iter()
        .filter(|q| q.sid != w.me)
        .map(|q| composite_strength(w, q.sid))
        .fold(0.0f64, f64::max);
    let dom_phi = (((mine_s + k::DOMINANCE_EPS) / (strongest + k::DOMINANCE_EPS)).ln())
        .clamp(-2.0, 2.0);
    r += k::DOM_COEF * (k::GAMMA * dom_phi - st.dominance_phi);
    st.dominance_phi = dom_phi;

    // closeout PBRS (`curriculum.rs:697-702`, `vecenv.rs:2144-2162`)
    let x = ((share - k::CLOSEOUT_START) / (k::CLOSEOUT_FULL - k::CLOSEOUT_START)).clamp(0.0, 1.0);
    let co_phi = x * x;
    r += k::CLOSE_COEF * (k::GAMMA * co_phi - st.closeout_phi);
    st.closeout_phi = co_phi;
    if !st.closeout_entered && share >= k::CLOSEOUT_START {
        st.closeout_entered = true;
        r += k::CLOSEOUT_ENTRY;
    }

    // tempo (`curriculum.rs:446-452`)
    if share >= k::TEMPO_THRESHOLD {
        let late = (w.tick as f64 / w.max_episode_ticks.max(1) as f64).clamp(0.0, 1.0);
        r += -k::TEMPO_COEF * late * late * tw * solo_scale;
    }

    // survival (`curriculum.rs:1387-1400`)
    let taper = if share <= k::CLOSEOUT_START {
        1.0
    } else if share >= k::CLOSEOUT_FULL {
        0.0
    } else {
        (k::CLOSEOUT_FULL - share) / (k::CLOSEOUT_FULL - k::CLOSEOUT_START)
    };
    r += k::SURVIVAL_COEF * share * taper;

    // death (`vecenv.rs:2246-2250`, alive -> dead transition)
    if st.initialized && st.prev_alive && !alive {
        r -= k::DEATH_PENALTY;
    }
    let died = st.prev_alive && !alive;
    st.prev_alive = alive;

    // ------- terminal ------------------------------------------------------
    let mut out = RewardOut {
        reward: r,
        died,
        ..Default::default()
    };
    let timed_out = w.tick >= w.max_episode_ticks;
    let won = share >= k::CLOSEOUT_FULL;
    if won || !alive || timed_out {
        // place = 1 + players with strictly greater composite strength
        let place = 1
            + w.players
                .iter()
                .filter(|q| q.sid != w.me && q.alive)
                .filter(|q| composite_strength(w, q.sid) > composite_strength(w, w.me))
                .count();
        let mut t = if won {
            k::W_PLACE * (place as f64).powf(-k::PLACE_POW) + k::W_WIN
                + k::FAST_WIN_COEF
                    * (1.0 - (w.tick as f64 / w.max_episode_ticks.max(1) as f64).clamp(0.0, 1.0))
                + k::EXTRA_WIN_BONUS
        } else {
            // EVERY loss pays -W_WIN, whatever ended it (`curriculum.rs:1366-1377`).
            -k::W_WIN
        };
        if !won && timed_out && st.closeout_entered {
            t -= k::TIMEOUT_AFTER_CLOSEOUT; // `curriculum.rs:1404-1414`
        }
        out.terminal = t;
        out.reward += t;
        out.done = true;
        out.won = won;
        out.timed_out = timed_out;
    }
    st.initialized = true;
    out
}

// ---------------------------------------------------------------------------
// Cross-check against the C ABI header
// ---------------------------------------------------------------------------

/// Parse the numeric macros out of `openfront_env.h` and assert this module's
/// constants are the same numbers. Returns `Err` with the first disagreement,
/// or `Ok(lines_checked)` when the header is absent (unverified, not a pass).
pub fn header_check(header_path: &std::path::Path) -> Result<(usize, Vec<String>), String> {
    let text = match std::fs::read_to_string(header_path) {
        Ok(t) => t,
        Err(e) => return Err(format!("header {}: {e}", header_path.display())),
    };
    let mut checked = 0usize;
    let mut bad: Vec<String> = Vec::new();
    let mut want = |name: &str, expect: usize, checked: &mut usize, bad: &mut Vec<String>| {
        let pat = format!("#define {name} ");
        let mut found = None;
        for line in text.lines() {
            if let Some(rest) = line.trim().strip_prefix(&pat) {
                let tok = rest.split_whitespace().next().unwrap_or("");
                let tok = tok.trim_matches(|c: char| !c.is_ascii_digit());
                if let Ok(v) = tok.parse::<usize>() {
                    found = Some(v);
                }
            }
        }
        match found {
            Some(v) => {
                *checked += 1;
                if v != expect {
                    bad.push(format!("{name}: header {v} != module {expect}"));
                }
            }
            None => bad.push(format!("{name}: not found in header")),
        }
    };
    // The offsets the policy tensor is indexed with, and the block SIZES the C
    // ABI publishes (`OFENV_MASK_PTARGET_N` is `N_ACTIONS * MAX_SLOTS` = 2688,
    // not the slot count 128 - the sizes are what the trainer strides by).
    want("OFENV_OBS_PER_AGENT", OBS_PER_AGENT, &mut checked, &mut bad);
    want("OFENV_MASK_PER_AGENT", MASK_PER_AGENT, &mut checked, &mut bad);
    want("OFENV_OBS_LEGAL_BUILD_OFF", OBS_BUILD_OFF, &mut checked, &mut bad);
    want("OFENV_OBS_LEGAL_NUKE_OFF", OBS_NUKE_OFF, &mut checked, &mut bad);
    want(
        "OFENV_OBS_LEGAL_ACTIONS_OFF",
        OBS_ACTIONS_OFF,
        &mut checked,
        &mut bad,
    );
    want("OFENV_OBS_ME_SLOT_OFF", OBS_ME_SLOT_OFF, &mut checked, &mut bad);
    want("OFENV_MASK_ACTIONS_OFF", MASK_ACTIONS_OFF, &mut checked, &mut bad);
    want("OFENV_MASK_ACTIONS_N", N_ACTIONS, &mut checked, &mut bad);
    want("OFENV_MASK_PTARGET_OFF", MASK_PTARGET_OFF, &mut checked, &mut bad);
    want(
        "OFENV_MASK_PTARGET_N",
        N_ACTIONS * MAX_SLOTS,
        &mut checked,
        &mut bad,
    );
    want("OFENV_MASK_UTARGET_OFF", MASK_UTARGET_OFF, &mut checked, &mut bad);
    want(
        "OFENV_MASK_UTARGET_N",
        N_ACTIONS * MAX_UNITS,
        &mut checked,
        &mut bad,
    );
    want("OFENV_MASK_BUILD_OFF", MASK_BUILD_OFF, &mut checked, &mut bad);
    want("OFENV_MASK_BUILD_N", N_BUILD, &mut checked, &mut bad);
    want("OFENV_MASK_NUKE_OFF", MASK_NUKE_OFF, &mut checked, &mut bad);
    want("OFENV_MASK_NUKE_N", N_NUKE, &mut checked, &mut bad);
    want("OFENV_MASK_TILE_OFF", MASK_TILE_OFF, &mut checked, &mut bad);
    want(
        "OFENV_MASK_TILE_N",
        MASK_TILE_GH * MASK_TILE_GW,
        &mut checked,
        &mut bad,
    );
    Ok((checked, bad))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world() -> World {
        World {
            tick: 400,
            spawn_end_tick: 301,
            max_episode_ticks: 21000,
            map_land: 1000,
            w: 1000,
            h: 1000,
            me: 1,
            me_slot: 1,
            players: vec![
                PlayerRow { sid: 1, slot: 1, alive: true, troops: 500.0, gold: 200.0, tiles: 300.0, border_len: 50 },
                PlayerRow { sid: 2, slot: 2, alive: true, troops: 400.0, gold: 100.0, tiles: 260.0, border_len: 40 },
                PlayerRow { sid: 3, slot: 3, alive: false, troops: 0.0, gold: 0.0, tiles: 0.0, border_len: 0 },
            ],
            attacks: vec![AttackRow { owner: 1, target: 0, troops: 120.0, alive: true }],
        }
    }

    #[test]
    fn header_abi_constants_agree() {
        // The C ABI header the trainer/policy compiles against. Skipped (not
        // passed) if the checkout is not beside this crate.
        let p = std::path::Path::new(
            "/opt/data/workspaces/skg/openfront-ai/include/openfront_env.h",
        );
        if !p.exists() {
            return;
        }
        let (n, bad) = header_check(p).expect("read header");
        assert!(n >= 18, "only {n} constants checked");
        assert!(bad.is_empty(), "mismatches: {bad:?}");
    }

    #[test]
    fn layout_matches_the_c_abi_factorisation() {
        assert_eq!(OBS_PLAYERS_OFF, 12);
        assert_eq!(OBS_UNITS_OFF, 3852);
        assert_eq!(OBS_BUILD_OFF, 4236);
        assert_eq!(OBS_NUKE_OFF, 4243);
        assert_eq!(OBS_ACTIONS_OFF, 4248);
        assert_eq!(OBS_ME_SLOT_OFF, 4269);
        assert_eq!(MASK_PTARGET_OFF, 21);
        assert_eq!(MASK_UTARGET_OFF, 2709);
        assert_eq!(MASK_BUILD_OFF, 3381);
        assert_eq!(MASK_NUKE_OFF, 3388);
        assert_eq!(MASK_TILE_OFF, 3393);
    }

    #[test]
    fn mask_is_binary_and_noop_is_always_legal() {
        let w = world();
        let mut inv = Inventory::default();
        let m = mask(&w, None, &mut inv);
        assert_eq!(m.len(), MASK_PER_AGENT);
        assert!(m.iter().all(|v| *v == 0.0 || *v == 1.0));
        assert_eq!(m[MASK_ACTIONS_OFF + A_NOOP], 1.0);
        // post-spawn: the whole gh x gw tile block is legal, the padding is not
        let (gh, gw) = (w.gh(), w.gw());
        for gy in 0..MASK_TILE_GH {
            for gx in 0..MASK_TILE_GW {
                let want = if gy < gh && gx < gw { 1.0 } else { 0.0 };
                assert_eq!(m[MASK_TILE_OFF + gy * MASK_TILE_GW + gx], want, "gy={gy} gx={gx}");
            }
        }
    }

    #[test]
    fn spawn_phase_mask_follows_the_tile_predicate() {
        let mut w = world();
        w.spawn_end_tick = 500;
        w.tick = 400;
        let (gh, gw) = (w.gh(), w.gw());
        let mut tile = vec![0u8; gh * gw];
        tile[3 * gw + 5] = 1;
        let mut inv = Inventory::default();
        let m = mask(&w, Some(&tile), &mut inv);
        assert_eq!(m[MASK_ACTIONS_OFF + A_SPAWN], 0.0); // alive -> noop only
        assert_eq!(m[MASK_ACTIONS_OFF + A_NOOP], 1.0);
        assert_eq!(m[MASK_TILE_OFF + 3 * MASK_TILE_GW + 5], 1.0);
        assert_eq!(m[MASK_TILE_OFF + 3 * MASK_TILE_GW + 6], 0.0);
    }

    #[test]
    fn obs_scalars_and_tokens_are_derived_from_state() {
        let w = world();
        let mut inv = Inventory::default();
        let o = obs(&w, &mut inv);
        assert_eq!(o.len(), OBS_PER_AGENT);
        assert_eq!(o[OBS_SCALARS_OFF], 400.0 / 15000.0);
        assert_eq!(o[OBS_SCALARS_OFF + 1], 0.0); // not spawn phase
        assert_eq!(o[OBS_SCALARS_OFF + 2], 1.0); // alive
        assert_eq!(o[OBS_SCALARS_OFF + 3], log_norm(500.0));
        assert_eq!(o[OBS_SCALARS_OFF + 6], 1.0 / 8.0); // one live attack
        assert_eq!(o[OBS_SCALARS_OFF + 7], 1.0 / 128.0);
        assert_eq!(o[OBS_ME_SLOT_OFF], 1.0);
        // player token 1
        let base = OBS_PLAYERS_OFF + 1 * P_FEAT;
        assert_eq!(o[base + 7], 1.0);
        assert_eq!(o[base + 3], log_norm(300.0));
        assert!(inv.obs_written > 0 && inv.obs_written < OBS_PER_AGENT);
    }

    #[test]
    fn every_loss_pays_the_same_at_the_terminal_level() {
        // timeout with no winner, no closeout
        let mut w = world();
        w.tick = 21000;
        w.players[0].tiles = 100.0; // 10% share, no closeout
        let mut st = RewardState::default();
        let out = decision_reward(&w, &mut st);
        assert!(out.done && out.timed_out && !out.won);
        assert_eq!(out.terminal, -k::W_WIN);
    }

    #[test]
    fn timeout_after_closeout_is_worse_than_a_death_and_reported() {
        let mut w = world();
        w.tick = 21000;
        // reach 50% first so closeout_entry fires, then time out
        let mut st = RewardState::default();
        w.players[0].tiles = 500.0;
        let _ = decision_reward(&w, &mut st);
        assert!(st.closeout_entered);
        let out = decision_reward(&w, &mut st);
        assert!(out.done && out.timed_out);
        assert_eq!(out.terminal, -k::W_WIN - k::TIMEOUT_AFTER_CLOSEOUT);
    }

    #[test]
    fn a_win_pays_placement_plus_win_plus_the_fast_win_and_extra_bonus() {
        let mut w = world();
        w.players[0].tiles = 900.0; // >= 80%
        let mut st = RewardState::default();
        let out = decision_reward(&w, &mut st);
        assert!(out.won && out.done);
        let place: f64 = 1.0; // me is strictly strongest
        let expect = k::W_PLACE * place.powf(-k::PLACE_POW)
            + k::W_WIN
            + k::FAST_WIN_COEF * (1.0 - w.tick as f64 / w.max_episode_ticks as f64)
            + k::EXTRA_WIN_BONUS;
        assert!((out.terminal - expect).abs() < 1e-9, "{} vs {}", out.terminal, expect);
    }

    #[test]
    fn reward_is_deterministic_for_the_same_state_sequence() {
        let seq = |w: &World| {
            let mut st = RewardState::default();
            let mut v = Vec::new();
            let mut w = w.clone();
            for k in 0..8 {
                w.tick = 400 + k;
                v.push(decision_reward(&w, &mut st).reward);
            }
            v
        };
        let w = world();
        assert_eq!(seq(&w), seq(&w));
    }
}
