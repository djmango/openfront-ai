//! Single-env worker: bridge session + curriculum episode bookkeeping +
//! reward shaping. Port of `rl/vec.py::EnvWorker`. `VecEnv` fans this out
//! over one OS thread per env (see module-level doc below); unlike the
//! Python side there's no GIL, so no multiprocessing/pickle framing is
//! needed to keep JSON decode + featurization off the main thread.

use anyhow::{ensure, Result};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::sync::Mutex;

use ofcore::curriculum::{
    self, action_churn_penalty, boat_commit_potential, boat_outcome_reward,
    classify_boat_resolution, closeout_potential, combat_outcome_reward, continent_span_potential,
    dominance_potential, duo_first_structure_bonus, duo_pact_success_bonus, duo_potential,
    duo_structure_delete_penalty, economy_potential, embargo_stop_outcome_reward, fast_win_bonus,
    formally_allied, label_continents, land_share, leftover_continent_counts,
    leftover_continent_potential, normalized_strength_share, occupied_continent_count, placement,
    placement_score, player_gold_income, port_stand_potential, sample_episode, stages_for_schedule,
    strength_delta_weight, team_completed_structures, team_owner_ids, team_transport_ships,
    tempo_pressure, terminal_reward, timeweight, v10_closeout_entry_bonus, v10_combat_action_bonus,
    v10_diplo_panic_penalty, v10_survival_reward, v10_timeout_after_closeout_penalty,
    v83_action_churn_penalty, ActionChurnTracker, ActionPairCounts, ActionTarget,
    BoatOutcomeCounts, ChosenAction, CombatOutcome, CurriculumSchedule, DominanceShaper,
    InverseActionPair, RewardComponents, RewardConfig, Stage, CITY_UNIT_CLASS, DUO_SOLO_SCALE,
    PORT_UNIT_CLASS, TRANSPORT_UNIT_CLASS, V83_CLOSEOUT_SHARE_START, W_STR, W_WASTE,
};
use ofcore::feat::{
    self, ACTIONS, A_ALLIANCE_REQUEST, A_ATTACK, A_BOAT, A_BREAK_ALLIANCE, A_BUILD, A_CANCEL_BOAT,
    A_DONATE_GOLD, A_DONATE_TROOPS, A_EMBARGO, A_EMBARGO_STOP, A_RETREAT, IS_LAND_BIT, MAG_MASK,
    REGION,
};
use ofcore::translate::{translate, Choice, IntentTranslator};

use crate::ae::{self, AeRaw, StaticTerrain, TerrainCacheKey};
use crate::engine::{self, EngineKind, GameEngine, RawObs};

/// Per-env layout of [`CompactHostBuffers::extras`] (all f32):
/// `players | units | umask | legal_utarget | local | legal_ptarget |
/// pmask | scalars | legal_actions | legal_build | legal_nuke |
/// partner_players | partner_pmask | partner_scalars | partner_context`.
/// Partner tensors are appended so the local prefix stays aligned with
/// older compact payloads; they are zeros when `n_agents==1`.
/// `partner_context` is the sibling's previous [`ActionOutcome`] (14 floats).
pub(crate) fn compact_extras_per_env() -> usize {
    compact_extras_core_n()
        + compact_extras_players_n()
        + feat::MAX_SLOTS
        + feat::N_SCALARS
        + crate::recurrent::CONTEXT_FLOATS
}

/// Bytes before the MAPPO partner block (local actor extras).
pub(crate) fn compact_extras_core_n() -> usize {
    compact_extras_players_n()
        + compact_extras_units_n()
        + compact_extras_umask_n()
        + compact_extras_legal_utarget_n()
        + compact_extras_local_n()
        + compact_extras_legal_ptarget_n()
        + feat::MAX_SLOTS
        + feat::N_SCALARS
        + feat::N_ACTIONS
        + feat::N_BUILD
        + feat::N_NUKE
}

pub(crate) fn compact_extras_players_n() -> usize {
    feat::MAX_SLOTS * feat::P_FEAT
}
pub(crate) fn compact_extras_units_n() -> usize {
    feat::MAX_UNITS * feat::U_FEAT
}
pub(crate) fn compact_extras_umask_n() -> usize {
    feat::MAX_UNITS
}
pub(crate) fn compact_extras_legal_utarget_n() -> usize {
    feat::N_ACTIONS * feat::MAX_UNITS
}
pub(crate) fn compact_extras_local_n() -> usize {
    use crate::policy::LOCAL;
    5 * LOCAL as usize * LOCAL as usize
}
pub(crate) fn compact_extras_legal_ptarget_n() -> usize {
    feat::N_ACTIONS * feat::MAX_SLOTS
}

const AGENT_CLIENT_IDS: [&str; 2] = ["AGENTRL1", "AGENTRL2"];

/// CPU-owned foveated rollout payload. Grid samples cross the actor/learner
/// boundary as fp16 values; masks and crop metadata stay explicit so the
/// learner never has to reconstruct a full fine grid or infer coordinates.
/// Non-grid tensors (`players` / `local` / …) live in `extras` so
/// `PreparedObs` can drop its per-step Vec copies after compact.
#[derive(Default)]
pub(crate) struct CompactHostBuffers {
    pub grids: Vec<half::f16>,
    pub masks: Vec<f32>,
    pub origins: Vec<i64>,
    pub extras: Vec<f32>,
}

/// Actor-created pool for compact D2H payloads. A payload is returned only
/// when the last `CompactGrid` range into it is dropped (normally after the
/// learner has finished with that rollout), so current observations can never
/// alias or mutate an older `Step`.
#[derive(Default)]
pub(crate) struct CompactHostArena {
    free: Mutex<Vec<CompactHostBuffers>>,
}

impl CompactHostArena {
    pub fn lease(self: &Arc<Self>) -> CompactHostLease {
        let buffers = self
            .free
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pop()
            .unwrap_or_default();
        CompactHostLease {
            arena: Arc::clone(self),
            buffers: Some(buffers),
        }
    }

    #[cfg(test)]
    pub fn free_len(&self) -> usize {
        self.free
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }
}

pub(crate) struct CompactHostLease {
    arena: Arc<CompactHostArena>,
    buffers: Option<CompactHostBuffers>,
}

impl CompactHostLease {
    pub fn buffers_mut(&mut self) -> &mut CompactHostBuffers {
        self.buffers.as_mut().expect("compact host lease consumed")
    }

    pub fn publish(mut self) -> Arc<CompactHostPayload> {
        Arc::new(CompactHostPayload {
            arena: Arc::clone(&self.arena),
            buffers: self.buffers.take().expect("compact host lease consumed"),
        })
    }
}

impl Drop for CompactHostLease {
    fn drop(&mut self) {
        if let Some(buffers) = self.buffers.take() {
            self.arena
                .free
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(buffers);
        }
    }
}

pub(crate) struct CompactHostPayload {
    arena: Arc<CompactHostArena>,
    pub buffers: CompactHostBuffers,
}

impl Drop for CompactHostPayload {
    fn drop(&mut self) {
        let buffers = std::mem::take(&mut self.buffers);
        self.arena
            .free
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(buffers);
    }
}

/// An immutable per-environment view into one batch-contiguous host payload.
/// Cloning this type clones only the `Arc` and ranges, never the fp16/mask
/// bytes. Exact-shape buckets therefore need three host allocations/transfers
/// per bucket rather than six allocations and six slice copies per env.
#[derive(Clone)]
pub struct CompactGrid {
    payload: Arc<CompactHostPayload>,
    fine: Range<usize>,       // (C_GRID, fine_h, fine_w)
    fine_valid: Range<usize>, // (fine_h, fine_w)
    fine_legal: Range<usize>, // (fine_h, fine_w)
    pub fine_h: usize,
    pub fine_w: usize,
    pub origin_y: i64,
    pub origin_x: i64,
    coarse: Range<usize>,       // (C_GRID, coarse_h, coarse_w)
    coarse_valid: Range<usize>, // (coarse_h, coarse_w)
    coarse_legal: Range<usize>, // (coarse_h, coarse_w)
    pub coarse_h: usize,
    pub coarse_w: usize,
    /// Range into [`CompactHostBuffers::extras`] for this env.
    extras: Range<usize>,
}

impl CompactGrid {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        payload: Arc<CompactHostPayload>,
        fine: Range<usize>,
        fine_valid: Range<usize>,
        fine_legal: Range<usize>,
        fine_h: usize,
        fine_w: usize,
        origin_y: i64,
        origin_x: i64,
        coarse: Range<usize>,
        coarse_valid: Range<usize>,
        coarse_legal: Range<usize>,
        coarse_h: usize,
        coarse_w: usize,
        extras: Range<usize>,
    ) -> Self {
        Self {
            payload,
            fine,
            fine_valid,
            fine_legal,
            fine_h,
            fine_w,
            origin_y,
            origin_x,
            coarse,
            coarse_valid,
            coarse_legal,
            coarse_h,
            coarse_w,
            extras,
        }
    }

    pub fn fine(&self) -> &[half::f16] {
        &self.payload.buffers.grids[self.fine.clone()]
    }
    pub fn fine_valid(&self) -> &[f32] {
        &self.payload.buffers.masks[self.fine_valid.clone()]
    }
    pub fn fine_legal(&self) -> &[f32] {
        &self.payload.buffers.masks[self.fine_legal.clone()]
    }
    pub fn coarse(&self) -> &[half::f16] {
        &self.payload.buffers.grids[self.coarse.clone()]
    }
    pub fn coarse_valid(&self) -> &[f32] {
        &self.payload.buffers.masks[self.coarse_valid.clone()]
    }
    pub fn coarse_legal(&self) -> &[f32] {
        &self.payload.buffers.masks[self.coarse_legal.clone()]
    }

    fn extras_slice(&self) -> &[f32] {
        &self.payload.buffers.extras[self.extras.clone()]
    }

    pub fn players(&self) -> &[f32] {
        let n = compact_extras_players_n();
        &self.extras_slice()[..n]
    }
    pub fn units(&self) -> &[f32] {
        let start = compact_extras_players_n();
        let n = compact_extras_units_n();
        &self.extras_slice()[start..start + n]
    }
    pub fn umask(&self) -> &[f32] {
        let start = compact_extras_players_n() + compact_extras_units_n();
        &self.extras_slice()[start..start + compact_extras_umask_n()]
    }
    pub fn legal_utarget(&self) -> &[f32] {
        let start =
            compact_extras_players_n() + compact_extras_units_n() + compact_extras_umask_n();
        let n = compact_extras_legal_utarget_n();
        &self.extras_slice()[start..start + n]
    }
    pub fn local(&self) -> &[f32] {
        let start = compact_extras_players_n()
            + compact_extras_units_n()
            + compact_extras_umask_n()
            + compact_extras_legal_utarget_n();
        let n = compact_extras_local_n();
        &self.extras_slice()[start..start + n]
    }
    pub fn legal_ptarget(&self) -> &[f32] {
        let start = compact_extras_players_n()
            + compact_extras_units_n()
            + compact_extras_umask_n()
            + compact_extras_legal_utarget_n()
            + compact_extras_local_n();
        let n = compact_extras_legal_ptarget_n();
        &self.extras_slice()[start..start + n]
    }
    pub fn pmask(&self) -> &[f32] {
        let start = compact_extras_players_n()
            + compact_extras_units_n()
            + compact_extras_umask_n()
            + compact_extras_legal_utarget_n()
            + compact_extras_local_n()
            + compact_extras_legal_ptarget_n();
        &self.extras_slice()[start..start + feat::MAX_SLOTS]
    }
    pub fn scalars(&self) -> &[f32] {
        let start = compact_extras_players_n()
            + compact_extras_units_n()
            + compact_extras_umask_n()
            + compact_extras_legal_utarget_n()
            + compact_extras_local_n()
            + compact_extras_legal_ptarget_n()
            + feat::MAX_SLOTS;
        &self.extras_slice()[start..start + feat::N_SCALARS]
    }
    pub fn legal_actions(&self) -> &[f32] {
        let start = compact_extras_players_n()
            + compact_extras_units_n()
            + compact_extras_umask_n()
            + compact_extras_legal_utarget_n()
            + compact_extras_local_n()
            + compact_extras_legal_ptarget_n()
            + feat::MAX_SLOTS
            + feat::N_SCALARS;
        &self.extras_slice()[start..start + feat::N_ACTIONS]
    }
    pub fn legal_build(&self) -> &[f32] {
        let start = compact_extras_players_n()
            + compact_extras_units_n()
            + compact_extras_umask_n()
            + compact_extras_legal_utarget_n()
            + compact_extras_local_n()
            + compact_extras_legal_ptarget_n()
            + feat::MAX_SLOTS
            + feat::N_SCALARS
            + feat::N_ACTIONS;
        &self.extras_slice()[start..start + feat::N_BUILD]
    }
    pub fn legal_nuke(&self) -> &[f32] {
        let start = compact_extras_players_n()
            + compact_extras_units_n()
            + compact_extras_umask_n()
            + compact_extras_legal_utarget_n()
            + compact_extras_local_n()
            + compact_extras_legal_ptarget_n()
            + feat::MAX_SLOTS
            + feat::N_SCALARS
            + feat::N_ACTIONS
            + feat::N_BUILD;
        &self.extras_slice()[start..start + feat::N_NUKE]
    }
    pub fn partner_players(&self) -> &[f32] {
        let start = compact_extras_core_n();
        let n = compact_extras_players_n();
        &self.extras_slice()[start..start + n]
    }
    pub fn partner_pmask(&self) -> &[f32] {
        let start = compact_extras_core_n() + compact_extras_players_n();
        &self.extras_slice()[start..start + feat::MAX_SLOTS]
    }
    pub fn partner_scalars(&self) -> &[f32] {
        let start = compact_extras_core_n() + compact_extras_players_n() + feat::MAX_SLOTS;
        &self.extras_slice()[start..start + feat::N_SCALARS]
    }
    pub fn partner_context(&self) -> &[f32] {
        let start = compact_extras_core_n()
            + compact_extras_players_n()
            + feat::MAX_SLOTS
            + feat::N_SCALARS;
        &self.extras_slice()[start..start + crate::recurrent::CONTEXT_FLOATS]
    }

    #[cfg(test)]
    pub(crate) fn grid_storage_ptr(&self) -> *const half::f16 {
        self.payload.buffers.grids.as_ptr()
    }

    #[cfg(test)]
    pub(crate) fn mask_storage_ptr(&self) -> *const f32 {
        self.payload.buffers.masks.as_ptr()
    }

    #[cfg(test)]
    pub(crate) fn storage_capacities(&self) -> (usize, usize, usize, usize) {
        (
            self.payload.buffers.grids.capacity(),
            self.payload.buffers.masks.capacity(),
            self.payload.buffers.origins.capacity(),
            self.payload.buffers.extras.capacity(),
        )
    }
}

#[derive(Clone)]
pub struct EpisodeInfo {
    pub reward: f64,
    pub length: i64,
    pub final_tiles: f64,
    pub final_land_share: f64,
    pub max_land_share: f64,
    pub closeout_reached: bool,
    pub closeout_entry_tick: Option<i64>,
    pub decisions_after_closeout: u64,
    pub converted: bool,
    pub timeout_after_closeout: bool,
    pub post_closeout_churn_pairs: u64,
    pub final_tick: i64,
    pub place: i64,
    pub n_players: i64,
    pub score: f64,
    pub won: bool,
    /// True when the episode ended because the agent died or never spawned
    /// (`tiles == 0` / TS `!isAlive()`). A no-show is a death, not a timeout.
    pub died: bool,
    pub wasted: i64,
    pub stage: usize,
    pub map: String,
    pub rehearsal: bool,
    pub reward_components: RewardComponents,
    pub action_pair_counts: ActionPairCounts,
    pub boat_outcome_counts: BoatOutcomeCounts,
    pub embargo_bad_stops: u64,
    pub embargo_good_stops: u64,
    pub premature_retreats: u64,
    pub thrash_reengages: u64,
}

#[derive(Clone, Debug, Default)]
struct PendingBoat {
    troops: f64,
    cancel_requested: bool,
}

#[derive(Clone, Debug, Default)]
struct PendingBoatTracker {
    pending: HashMap<usize, PendingBoat>,
    counts: BoatOutcomeCounts,
}

impl PendingBoatTracker {
    fn reset(&mut self) {
        self.pending.clear();
        self.counts = BoatOutcomeCounts::default();
    }

    fn counts(&self) -> BoatOutcomeCounts {
        self.counts
    }

    fn note_launch(&mut self, unit_id: usize, troops: f64) {
        self.pending.insert(
            unit_id,
            PendingBoat {
                troops,
                cancel_requested: false,
            },
        );
    }

    fn note_cancel(&mut self, unit_id: usize) {
        if let Some(boat) = self.pending.get_mut(&unit_id) {
            boat.cancel_requested = true;
        }
    }

    fn resolve_missing(
        &mut self,
        alive_ids: &HashSet<usize>,
        troops_before: f64,
        troops_after: f64,
        new_sourced_attack: bool,
        has_sourced_attack: bool,
        config: RewardConfig,
        stage: usize,
    ) -> f64 {
        if !config.boat_outcome_active(stage) {
            self.pending
                .retain(|unit_id, _| alive_ids.contains(unit_id));
            return 0.0;
        }
        let mut reward = 0.0;
        let finished: Vec<usize> = self
            .pending
            .keys()
            .copied()
            .filter(|id| !alive_ids.contains(id))
            .collect();
        for unit_id in finished {
            let Some(boat) = self.pending.remove(&unit_id) else {
                continue;
            };
            let outcome = classify_boat_resolution(
                boat.cancel_requested,
                boat.troops,
                troops_before,
                troops_after,
                new_sourced_attack,
                has_sourced_attack,
            );
            self.counts.record(outcome);
            reward += boat_outcome_reward(outcome, config);
        }
        reward
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct CloseoutTracker {
    max_land_share: f64,
    entry_tick: Option<i64>,
    decisions_after_entry: u64,
    post_entry_churn_pairs: u64,
}

impl CloseoutTracker {
    fn reset(&mut self, share: f64, tick: i64) {
        self.max_land_share = share;
        self.entry_tick = (share >= V83_CLOSEOUT_SHARE_START).then_some(tick);
        self.decisions_after_entry = 0;
        self.post_entry_churn_pairs = 0;
    }

    /// Returns true the first time land share crosses closeout entry this episode.
    fn observe(&mut self, share: f64, tick: i64, inverse_pair: bool) -> bool {
        self.max_land_share = self.max_land_share.max(share);
        let mut just_entered = false;
        if self.entry_tick.is_some() {
            self.decisions_after_entry += 1;
        } else if share >= V83_CLOSEOUT_SHARE_START {
            self.entry_tick = Some(tick);
            just_entered = true;
        }
        if self.entry_tick.is_some() && inverse_pair {
            self.post_entry_churn_pairs += 1;
        }
        just_entered
    }

    fn reached(self) -> bool {
        self.entry_tick.is_some()
    }
}

/// Outcome of the action immediately preceding an observation. This stays on
/// the host and is supplied separately to recurrent policies; legacy
/// observation tensors are unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionOutcome {
    pub action: i64,
    pub player_slot: i64,
    pub tile_region: i64,
    pub build_type: i64,
    pub nuke_type: i64,
    pub success: bool,
    pub wasted: bool,
    /// Stable engine player/unit identity, or -1 when no identity applies.
    pub target_identity: i64,
    /// Region-center coordinates normalized to [0, 1], or -1 when unused.
    pub target_y: f32,
    pub target_x: f32,
    pub quantity: f32,
    /// Number of consecutive decisions with the same semantic commitment.
    pub commitment_age: u32,
    pub had_action: bool,
    /// 0 = none, 1 = player, 2 = unit.
    pub target_kind: u8,
}

impl Default for ActionOutcome {
    fn default() -> Self {
        Self {
            action: -1,
            player_slot: -1,
            tile_region: -1,
            build_type: -1,
            nuke_type: -1,
            success: false,
            wasted: false,
            target_identity: -1,
            target_y: -1.0,
            target_x: -1.0,
            quantity: -1.0,
            commitment_age: 0,
            had_action: false,
            target_kind: 0,
        }
    }
}

impl ActionOutcome {
    pub fn as_floats(&self) -> [f32; crate::recurrent::CONTEXT_FLOATS] {
        [
            self.action as f32,
            self.player_slot as f32,
            self.tile_region as f32,
            self.build_type as f32,
            self.nuke_type as f32,
            self.success as u8 as f32,
            self.wasted as u8 as f32,
            self.target_identity as f32,
            self.target_y,
            self.target_x,
            self.quantity,
            self.commitment_age as f32,
            self.had_action as u8 as f32,
            self.target_kind as f32,
        ]
    }
}

fn normalized_tile_target(tile: i64, gh: usize, gw: usize) -> (f32, f32) {
    // Policy tile ids use the fixed GW_MAX global stride, not this map's
    // compact width (see policy::fine_local_to_global).
    let y = tile.div_euclid(feat::GW_MAX as i64);
    let x = tile.rem_euclid(feat::GW_MAX as i64);
    (
        (y as f32 + 0.5) / gh.max(1) as f32,
        (x as f32 + 0.5) / gw.max(1) as f32,
    )
}

pub struct EnvTransition {
    pub next_obs: PreparedObs,
    pub reward: f64,
    pub done: bool,
    pub info: Option<EpisodeInfo>,
    pub outcome: ActionOutcome,
}

/// Per-env observation ready to batch into `policy::Obs`.
///
/// Production path (`batch::build_obs` with an `AePair`): GPU AE encode
/// replaces the old 6ch `stat` placeholder with a 32ch latent, yielding
/// `C_GRID = 99 = latent(32) + static(6) + ego(3) + db(1) + transient(57)`.
///
/// `grid` is only filled for the no-AE test/legacy path (63ch
/// stat+ego+db+transient); training always passes an AE and rebuilds
/// `grid` inside `build_obs`.
#[derive(Clone)]
pub struct PreparedObs {
    /// Previous action/result context for a recurrent policy. It is not
    /// included by any legacy `batch::build_obs*` path.
    pub prev_action: ActionOutcome,
    /// Compact host ownership boundary used by `--compact-rollout`.
    /// Contains no device handles and is consumed directly by policy/update.
    pub compact: Option<CompactGrid>,
    /// Optional pre-assembled fine grid (C_GRID, gh, gw). Filled by the
    /// actor encode path so the learner can rebuild Obs without holding an
    /// AE (tch Optimizer/Tensor are !Sync across shard batch-build threads).
    pub grid: Option<Vec<f32>>,
    /// Optional native /16 coarse grid (C_GRID, cgh, cgw).
    pub grid_coarse: Option<Vec<f32>>,
    pub cgh: usize,
    pub cgw: usize,
    /// Full-res AE inputs for batched GPU encode.
    pub ae_raw: AeRaw,
    /// Pooled ego fractions at /8: (3, gh, gw).
    pub ego: Vec<f32>,
    /// Pooled defense bonus at /8: (1, gh, gw).
    pub db: Vec<f32>,
    /// Transient planes at /8: (57, gh, gw).
    pub transient: Vec<f32>,
    pub legal_tile: Vec<f32>, // (gh, gw)
    pub gh: usize,
    pub gw: usize,
    pub players: Vec<f32>, // (MAX_SLOTS, P_FEAT)
    pub pmask: [f32; feat::MAX_SLOTS],
    pub units: Vec<f32>, // (MAX_UNITS, U_FEAT)
    pub umask: [f32; feat::MAX_UNITS],
    pub legal_utarget: Vec<f32>, // (N_ACTIONS, MAX_UNITS)
    pub scalars: [f32; feat::N_SCALARS],
    pub me_slot: i64,
    pub legal_actions: [f32; feat::N_ACTIONS],
    pub legal_ptarget: Vec<f32>, // (N_ACTIONS, MAX_SLOTS)
    pub legal_build: [f32; feat::N_BUILD],
    pub legal_nuke: [f32; feat::N_NUKE],
    pub local: Vec<f32>, // (5, LOCAL, LOCAL)
    /// Teammate compact extras for the MAPPO critic. Zeros when `n_agents==1`.
    pub partner_players: Vec<f32>, // (MAX_SLOTS, P_FEAT)
    pub partner_pmask: [f32; feat::MAX_SLOTS],
    pub partner_scalars: [f32; feat::N_SCALARS],
    /// Partner's previous action (simultaneous-step delay). Zeros when
    /// `n_agents==1` or before the first action of an episode.
    pub partner_context: [f32; crate::recurrent::CONTEXT_FLOATS],
}

impl PreparedObs {
    /// Drop host-owned rollout tensors after the learner has uploaded a
    /// `ShardBatch` (or after compact has moved them into the arena).
    /// Keeps only tiny metadata the Step struct still needs.
    pub fn release_rollout_payload(&mut self) {
        self.compact = None;
        self.grid = None;
        self.grid_coarse = None;
        self.ae_raw.owners = Vec::new();
        self.ae_raw.static_terrain.land_mag = Vec::<f32>::new().into();
        self.ae_raw.fallout = Vec::new();
        self.ae_raw.stat = Vec::new();
        self.ego = Vec::new();
        self.db = Vec::new();
        self.transient = Vec::new();
        self.legal_tile = Vec::new();
        self.players = Vec::new();
        self.units = Vec::new();
        self.umask = [0.0; feat::MAX_UNITS];
        self.legal_utarget = Vec::new();
        self.local = Vec::new();
        self.legal_ptarget = Vec::new();
        self.pmask = [0.0; feat::MAX_SLOTS];
        self.scalars = [0.0; feat::N_SCALARS];
        self.legal_actions = [0.0; feat::N_ACTIONS];
        self.legal_build = [0.0; feat::N_BUILD];
        self.legal_nuke = [0.0; feat::N_NUKE];
        self.partner_players = Vec::new();
        self.partner_pmask = [0.0; feat::MAX_SLOTS];
        self.partner_scalars = [0.0; feat::N_SCALARS];
        self.partner_context = [0.0; crate::recurrent::CONTEXT_FLOATS];
    }
}

fn selected_player_id(choice: &Choice, lut: &[u8], ents: &feat::EntsData) -> Option<usize> {
    choice.player_slot.and_then(|slot| {
        ents.players
            .iter()
            .find(|player| {
                lut.get(player.id)
                    .is_some_and(|&mapped| i64::from(mapped) == slot)
            })
            .map(|player| player.id)
    })
}

fn player_troops(ents: &feat::EntsData, me: usize) -> f64 {
    ents.players
        .iter()
        .find(|p| p.id == me)
        .map(|p| p.troops)
        .unwrap_or(0.0)
}

#[derive(Clone, Debug, Default)]
struct CombatStickyTracker {
    last_attack_decision: HashMap<usize, i64>,
    last_retreat_decision: HashMap<usize, i64>,
    decision_i: i64,
    premature_retreats: u64,
    thrash_reengages: u64,
    embargo_bad_stops: u64,
    embargo_good_stops: u64,
}

impl CombatStickyTracker {
    fn reset(&mut self) {
        self.last_attack_decision.clear();
        self.last_retreat_decision.clear();
        self.decision_i = 0;
        self.premature_retreats = 0;
        self.thrash_reengages = 0;
        self.embargo_bad_stops = 0;
        self.embargo_good_stops = 0;
    }

    fn observe_combat(
        &mut self,
        action: i64,
        target: Option<usize>,
        window: usize,
        config: RewardConfig,
        stage: usize,
    ) -> f64 {
        self.decision_i = self.decision_i.saturating_add(1);
        let Some(player) = target else {
            return 0.0;
        };
        if !config.combat_outcome_active(stage) || window == 0 {
            if action == A_ATTACK {
                self.last_attack_decision.insert(player, self.decision_i);
            } else if action == A_RETREAT {
                self.last_retreat_decision.insert(player, self.decision_i);
            }
            return 0.0;
        }
        let mut reward = 0.0;
        if action == A_RETREAT {
            if let Some(atk_at) = self.last_attack_decision.get(&player).copied() {
                if self.decision_i - atk_at <= window as i64 {
                    self.premature_retreats += 1;
                    reward += combat_outcome_reward(CombatOutcome::PrematureRetreat, config);
                }
            }
            self.last_retreat_decision.insert(player, self.decision_i);
            // Clear sticky open so a later re-engage starts a fresh window;
            // reinforce must not keep refreshing the clock (see observe ATTACK).
            self.last_attack_decision.remove(&player);
        } else if action == A_ATTACK {
            if let Some(ret_at) = self.last_retreat_decision.get(&player).copied() {
                if self.decision_i - ret_at <= window as i64 {
                    self.thrash_reengages += 1;
                    reward += combat_outcome_reward(CombatOutcome::ThrashReengage, config);
                }
            }
            // First open only - reinforcing the same target must not refresh
            // the premature-retreat clock (otherwise every retreat is "premature").
            self.last_attack_decision
                .entry(player)
                .or_insert(self.decision_i);
        }
        reward
    }

    fn observe_embargo_stop(
        &mut self,
        relation_value: f64,
        config: RewardConfig,
        stage: usize,
    ) -> f64 {
        if !config.embargo_outcome_active(stage) {
            return 0.0;
        }
        let reward = embargo_stop_outcome_reward(relation_value, config);
        if reward < 0.0 {
            self.embargo_bad_stops += 1;
        } else if reward > 0.0 {
            self.embargo_good_stops += 1;
        }
        reward
    }
}

fn transport_unit_ids(ents: &feat::EntsData, me: usize) -> HashSet<usize> {
    ents.units
        .iter()
        .filter(|u| u.owner == me && u.class == TRANSPORT_UNIT_CLASS && u.uid >= 0)
        .map(|u| u.uid as usize)
        .collect()
}

fn churn_action(
    choice: &Choice,
    lut: &[u8],
    ents: &feat::EntsData,
    intents: &[Value],
    boats_before: &[usize],
    boats_after: &[usize],
) -> ChosenAction {
    let target = match choice.action {
        A_ATTACK | A_EMBARGO | A_EMBARGO_STOP | A_ALLIANCE_REQUEST | A_BREAK_ALLIANCE
        | A_DONATE_GOLD | A_DONATE_TROOPS
            if !intents.is_empty() =>
        {
            selected_player_id(choice, lut, ents).map(ActionTarget::Player)
        }
        A_RETREAT => intents
            .first()
            .and_then(|intent| intent["attackID"].as_str())
            .and_then(|attack_id| {
                ents.attacks
                    .iter()
                    .find(|attack| attack.aid == attack_id && attack.to != 0)
                    .map(|attack| ActionTarget::Player(attack.to))
            }),
        A_BOAT if !intents.is_empty() => {
            let mut created = boats_after
                .iter()
                .copied()
                .filter(|unit| !boats_before.contains(unit));
            let first = created.next();
            if created.next().is_none() {
                first.map(ActionTarget::Unit)
            } else {
                None
            }
        }
        A_CANCEL_BOAT => intents
            .first()
            .and_then(|intent| intent["unitID"].as_u64())
            .and_then(|unit| usize::try_from(unit).ok())
            .map(ActionTarget::Unit),
        _ => None,
    };
    ChosenAction::new(choice.action, target)
}

fn humans_won(winner: &Value, n_agents: usize) -> bool {
    let Some(a) = winner.as_array() else {
        return false;
    };
    match a.first().and_then(|v| v.as_str()) {
        Some("player") => n_agents == 1 && a.get(1).and_then(|v| v.as_str()) == Some("AGENTRL1"),
        Some("team") => {
            a.get(1).and_then(|v| v.as_str()) == Some("Humans")
                || a.iter()
                    .any(|v| matches!(v.as_str(), Some("AGENTRL1") | Some("AGENTRL2")))
        }
        _ => false,
    }
}

pub struct EnvWorker {
    pub idx: usize,
    bridge: Box<dyn GameEngine>,
    stages: Vec<Stage>,
    curriculum_schedule: CurriculumSchedule,
    stage: usize,
    episode_stage: usize,
    max_episode_ticks: i64,
    reward_config: RewardConfig,
    decision_ticks: u32,
    rng: SmallRng,
    episode: u64,
    ep_reward: f64,
    ep_len: i64,
    ep_wasted: i64,
    obs: Option<RawObs>,
    /// Cached featurizer view of `obs`. Filled from native `structured`
    /// side-channels or JSON parse; reused across apply→prepare so collect
    /// never re-walks the same entities/legal twice.
    cached_ents: Option<feat::EntsData>,
    cached_legal: Option<feat::Legal>,
    /// AGENTRL2 head + legality when `n_agents == 2`.
    n_agents: usize,
    duo_head: Option<Value>,
    cached_legal_duo: Option<feat::Legal>,
    lut: Vec<u8>,
    translator: Option<IntentTranslator>,
    land_total: i64,
    prev_strength: [f64; 2],
    dominance_shaper: [DominanceShaper; 2],
    closeout_shaper: [DominanceShaper; 2],
    duo_shaper: [DominanceShaper; 2],
    eco_shaper: [DominanceShaper; 2],
    boat_commit_shaper: [DominanceShaper; 2],
    leftover_continent_shaper: [DominanceShaper; 2],
    port_stand_shaper: [DominanceShaper; 2],
    continent_span_shaper: [DominanceShaper; 2],
    closeout_tracker: [CloseoutTracker; 2],
    action_churn_tracker: [ActionChurnTracker; 2],
    boat_tracker: [PendingBoatTracker; 2],
    combat_tracker: [CombatStickyTracker; 2],
    prev_action: [ActionOutcome; 2],
    last_commitment: [Option<(i64, i64, i64, i64, i64, u64)>; 2],
    was_alive: [bool; 2],
    /// True if this human ever had `alive` this episode. Spawn-miss is a
    /// death/loss, not a placement gift (ghost humans with 0 tiles).
    ever_alive: [bool; 2],
    /// Duo: already paid the one-shot formal-pact bonus this episode.
    pact_bonus_paid: bool,
    /// Duo: already paid the first completed-City one-shot this episode.
    city_bonus_paid: bool,
    /// Duo: already paid the first completed-Port one-shot this episode.
    port_bonus_paid: bool,
    /// Last step's completed City/Port counts (for delete-penalty drops).
    prev_n_cities: usize,
    prev_n_ports: usize,
    ep_reward_components: RewardComponents,
    spawn_steps: i64,
    map_name: String,
    rehearsal: bool,
    hr: usize,
    wr: usize,
    land: Vec<u8>,
    /// 4-connected land-component ids for leftover-continent Φ (0 = water).
    continent_of: Vec<u16>,
    mag: Vec<u8>,
    ae_static: StaticTerrain,
    engine_kind: EngineKind,
}

static NEXT_TERRAIN_ID: AtomicU64 = AtomicU64::new(1);

impl EnvWorker {
    pub fn new(
        idx: usize,
        stage: usize,
        max_episode_ticks: i64,
        engine: EngineKind,
        reward_config: RewardConfig,
        curriculum_schedule: CurriculumSchedule,
        n_agents: u32,
    ) -> Result<Self> {
        let n_agents = n_agents.clamp(1, 2) as usize;
        let mut bridge = engine::create(engine)?;
        bridge.set_agent_count(n_agents as u32);
        let mut w = EnvWorker {
            idx,
            bridge,
            stages: stages_for_schedule(curriculum_schedule),
            curriculum_schedule,
            stage,
            episode_stage: stage,
            max_episode_ticks,
            reward_config,
            decision_ticks: 15,
            rng: SmallRng::seed_from_u64(1000 + idx as u64),
            episode: 0,
            ep_reward: 0.0,
            ep_len: 0,
            ep_wasted: 0,
            obs: None,
            cached_ents: None,
            cached_legal: None,
            n_agents,
            duo_head: None,
            cached_legal_duo: None,
            lut: Vec::new(),
            translator: None,
            land_total: 1,
            prev_strength: [0.0; 2],
            dominance_shaper: [DominanceShaper::default(), DominanceShaper::default()],
            closeout_shaper: [DominanceShaper::default(), DominanceShaper::default()],
            duo_shaper: [DominanceShaper::default(), DominanceShaper::default()],
            eco_shaper: [DominanceShaper::default(), DominanceShaper::default()],
            boat_commit_shaper: [DominanceShaper::default(), DominanceShaper::default()],
            leftover_continent_shaper: [DominanceShaper::default(), DominanceShaper::default()],
            port_stand_shaper: [DominanceShaper::default(), DominanceShaper::default()],
            continent_span_shaper: [DominanceShaper::default(), DominanceShaper::default()],
            closeout_tracker: [CloseoutTracker::default(), CloseoutTracker::default()],
            action_churn_tracker: [ActionChurnTracker::default(), ActionChurnTracker::default()],
            boat_tracker: [PendingBoatTracker::default(), PendingBoatTracker::default()],
            combat_tracker: [
                CombatStickyTracker::default(),
                CombatStickyTracker::default(),
            ],
            prev_action: [ActionOutcome::default(), ActionOutcome::default()],
            last_commitment: [None, None],
            was_alive: [false; 2],
            ever_alive: [false; 2],
            pact_bonus_paid: false,
            city_bonus_paid: false,
            port_bonus_paid: false,
            prev_n_cities: 0,
            prev_n_ports: 0,
            ep_reward_components: RewardComponents::default(),
            spawn_steps: 0,
            map_name: String::new(),
            rehearsal: false,
            hr: 0,
            wr: 0,
            land: Vec::new(),
            continent_of: Vec::new(),
            mag: Vec::new(),
            ae_static: StaticTerrain {
                key: TerrainCacheKey {
                    env_id: idx as u64,
                    episode: 0,
                    static_id: 0,
                    hr: 0,
                    wr: 0,
                },
                map: Arc::from(""),
                land_mag: Vec::<f32>::new().into(),
            },
            engine_kind: engine,
        };
        w.reset_episode()?;
        Ok(w)
    }

    /// Fixed map/seed/bots episode for showcase watch (Node bridge + GameRecord).
    pub fn reset_watch(
        &mut self,
        map_name: &str,
        seed: &str,
        bots: u32,
        difficulty: &str,
        nations: Value,
    ) -> Result<()> {
        self.episode_stage = self.stage;
        // Always use the stage table's decision_ticks (V10 is uniformly 15).
        self.decision_ticks = self
            .stages
            .get(self.stage)
            .map(|s| s.decision_ticks.max(1))
            .unwrap_or(10);
        self.map_name = map_name.to_string();
        self.rehearsal = false;
        let obs = self
            .bridge
            .reset(map_name, seed, bots, difficulty, nations)?;

        let width = self.bridge.width();
        let height = self.bridge.height();
        let hr = height - height % REGION;
        let wr = width - width % REGION;
        self.hr = hr;
        self.wr = wr;
        let terrain = self.bridge.terrain();
        let mut land = vec![0u8; hr * wr];
        let mut mag = vec![0u8; hr * wr];
        for y in 0..hr {
            for x in 0..wr {
                let t = terrain[y * width + x];
                land[y * wr + x] = (t >> IS_LAND_BIT) & 1;
                mag[y * wr + x] = t & MAG_MASK;
            }
        }
        self.land = land;
        self.continent_of = label_continents(&self.land, wr, hr);
        self.mag = mag;
        self.ae_static = StaticTerrain {
            key: TerrainCacheKey {
                env_id: self.idx as u64,
                episode: self.episode,
                static_id: NEXT_TERRAIN_ID.fetch_add(1, Ordering::Relaxed),
                hr,
                wr,
            },
            map: Arc::from(map_name),
            land_mag: ae::pack_static_terrain(&self.land, &self.mag, hr, wr),
        };
        self.land_total = (self.land.iter().map(|&l| l as i64).sum::<i64>()).max(1);
        self.translator = Some(IntentTranslator::new(self.bridge.terrain(), width, hr, wr));
        self.lut.clear();
        self.set_obs(obs);
        self.ever_alive = [false; 2];
        self.pact_bonus_paid = false;
        self.city_bonus_paid = false;
        self.port_bonus_paid = false;
        self.prev_n_cities = 0;
        self.prev_n_ports = 0;
        self.seed_agent_trackers();
        self.spawn_steps = 0;
        self.ep_reward = 0.0;
        self.ep_reward_components = RewardComponents::default();
        self.ep_len = 0;
        self.ep_wasted = 0;
        self.episode += 1;
        Ok(())
    }

    pub fn save_record(&mut self, path: &str) -> Result<serde_json::Value> {
        self.bridge.save_record(path)
    }

    /// Best-effort GameRecord spool for HF parquet upload (never fails the episode).
    fn spool_finished_episode(&mut self, won: bool, timed_out: bool) {
        let engine = match self.engine_kind {
            EngineKind::Native => "native",
            EngineKind::Node => "node",
        };
        let meta = serde_json::json!({
            "map": self.map_name,
            "stage": self.stage,
            "episode_stage": self.episode_stage,
            "engine": engine,
            "n_agents": self.n_agents,
            "duo": self.n_agents > 1,
            "won": won,
            "timed_out": timed_out,
            "run_name": std::env::var("HF_RUN_PREFIX")
                .or_else(|_| std::env::var("RUN_NAME"))
                .unwrap_or_default(),
            "policy_update": std::env::var("POLICY_UPDATE").ok(),
            "policy_repo": std::env::var("HF_REPO_ID").unwrap_or_else(|_| "djmango/openfront-rl".into()),
            "policy_revision": std::env::var("POLICY_REVISION").ok(),
        });
        if let Err(e) = crate::replay_spool::spool_episode(self.bridge.as_mut(), &meta) {
            eprintln!("[replay-spool] {e}");
        }
    }

    pub fn current_obs(&self) -> Option<&RawObs> {
        self.obs.as_ref()
    }

    pub fn spawn_randomly_public(&mut self) -> Result<()> {
        self.spawn_randomly()
    }

    /// Translate + step without auto-reset (for watch/record episodes).
    pub fn apply_watch(&mut self, choice: &Choice) -> Result<()> {
        let lut = self.current_lut();
        let width = self.obs.as_ref().unwrap().head["width"].as_u64().unwrap() as usize;
        let mut owners_trim = vec![0i64; self.hr * self.wr];
        {
            let obs = self.obs.as_ref().unwrap();
            for y in 0..self.hr {
                for x in 0..self.wr {
                    owners_trim[y * self.wr + x] = obs.owner_at(y * width + x) as i64;
                }
            }
        }
        let me = self.obs.as_ref().unwrap().me();
        let ents = self.ents().clone();
        let legal = self.legal().clone();
        let intents = translate(
            choice,
            self.translator.as_mut().unwrap(),
            &owners_trim,
            me,
            &ents,
            &legal,
            &lut,
        );
        let new_obs = self.bridge.step(&intents, self.decision_ticks)?;
        if new_obs.spawn_phase() {
            self.spawn_steps += 1;
        }
        self.set_obs(new_obs);
        Ok(())
    }

    pub fn reset_episode(&mut self) -> Result<()> {
        self.episode_stage = self.stage;
        let stg = &self.stages[self.stage];
        self.decision_ticks = stg.decision_ticks;
        // Sticky same-map resets (~70%) keep (hr,wr) stable across episodes so
        // work-conserving actor batches stay same-shape and the AE path
        // avoids churning unique map geometries every episode. Does not
        // change reward / legal actions - only map resampling bias.
        const STICKY_MAP_P: f64 = 0.70;
        let sticky = !self.map_name.is_empty()
            && stg.maps.iter().any(|m| *m == self.map_name.as_str())
            && self.rng.r#gen::<f64>() < STICKY_MAP_P;
        let (map_name, bots, difficulty, nations, rehearsal) = if sticky {
            (
                self.map_name.clone(),
                stg.bots,
                stg.difficulty,
                stg.nations,
                false,
            )
        } else {
            sample_episode(&self.stages, self.stage, &mut self.rng)
        };
        self.map_name = map_name.clone();
        self.rehearsal = rehearsal;
        let nations_val = match nations {
            curriculum::Nations::Default => Value::String("default".into()),
            curriculum::Nations::Exact(n) => Value::from(n),
        };
        let seed = format!("w{}-ep{}", self.idx, self.episode);
        let obs = self
            .bridge
            .reset(&map_name, &seed, bots, difficulty, nations_val)?;

        let width = self.bridge.width();
        let height = self.bridge.height();
        let hr = height - height % REGION;
        let wr = width - width % REGION;
        self.hr = hr;
        self.wr = wr;
        let terrain = self.bridge.terrain();
        let mut land = vec![0u8; hr * wr];
        let mut mag = vec![0u8; hr * wr];
        for y in 0..hr {
            for x in 0..wr {
                let t = terrain[y * width + x];
                land[y * wr + x] = (t >> IS_LAND_BIT) & 1;
                mag[y * wr + x] = t & MAG_MASK;
            }
        }
        self.land = land;
        self.continent_of = label_continents(&self.land, wr, hr);
        self.mag = mag;
        self.ae_static = StaticTerrain {
            key: TerrainCacheKey {
                env_id: self.idx as u64,
                episode: self.episode,
                static_id: NEXT_TERRAIN_ID.fetch_add(1, Ordering::Relaxed),
                hr,
                wr,
            },
            map: Arc::from(map_name.as_str()),
            land_mag: ae::pack_static_terrain(&self.land, &self.mag, hr, wr),
        };
        self.land_total = (self.land.iter().map(|&l| l as i64).sum::<i64>()).max(1);
        self.translator = Some(IntentTranslator::new(self.bridge.terrain(), width, hr, wr));
        self.lut.clear();
        self.set_obs(obs);
        self.ever_alive = [false; 2];
        self.pact_bonus_paid = false;
        self.city_bonus_paid = false;
        self.port_bonus_paid = false;
        self.prev_n_cities = 0;
        self.prev_n_ports = 0;
        self.seed_agent_trackers();
        self.spawn_steps = 0;
        self.ep_reward = 0.0;
        self.ep_reward_components = RewardComponents::default();
        self.ep_len = 0;
        self.ep_wasted = 0;
        self.episode += 1;
        Ok(())
    }

    /// Install a new observation and refresh the ents/legal cache from the
    /// native structured side-channel when present, otherwise JSON parse.
    fn set_obs(&mut self, mut obs: RawObs) {
        let duo = obs.duo.take();
        if let Some((ents, legal)) = obs.structured.take() {
            self.cached_ents = Some(ents);
            self.cached_legal = Some(legal);
        } else {
            self.cached_ents = Some(feat::parse_ents(obs.entities()));
            self.cached_legal = Some(feat::parse_legal(obs.legal_actions()));
        }
        if let Some((head_b, legal_b)) = duo {
            self.duo_head = Some(head_b);
            self.cached_legal_duo = Some(legal_b);
        } else {
            self.duo_head = None;
            self.cached_legal_duo = None;
        }
        self.obs = Some(obs);
    }

    fn agent_head(&self, i: usize) -> &Value {
        if i == 0 {
            &self.obs.as_ref().unwrap().head
        } else {
            self.duo_head
                .as_ref()
                .unwrap_or(&self.obs.as_ref().unwrap().head)
        }
    }

    fn agent_legal(&self, i: usize) -> &feat::Legal {
        if i == 0 {
            self.legal()
        } else {
            self.cached_legal_duo.as_ref().unwrap_or(self.legal())
        }
    }

    fn agent_me(&self, i: usize) -> i64 {
        self.agent_head(i)["me"].as_i64().unwrap_or(-1)
    }

    fn agent_alive(&self, i: usize) -> bool {
        // TS `isAlive()` is tiles on the map, not the sticky `Player.alive`
        // flag. Unspawned humans used to report alive=true with 0 tiles, so
        // a no-show was a timeout instead of a death.
        self.agent_on_map(i)
    }

    fn agent_on_map(&self, i: usize) -> bool {
        let me = self.agent_me(i);
        if me < 0 {
            return false;
        }
        self.ents()
            .players
            .iter()
            .any(|p| p.id as i64 == me && p.tiles > 0.0)
    }

    fn seed_agent_trackers(&mut self) {
        let initial_strengths = curriculum::strengths(self.ents(), self.land_total);
        let tick = self.obs.as_ref().unwrap().tick();
        for i in 0..self.n_agents {
            let me = self.agent_me(i).max(0) as usize;
            self.prev_strength[i] = initial_strengths.get(&me).copied().unwrap_or(0.0);
            self.dominance_shaper[i].reset(dominance_potential(
                &initial_strengths,
                me,
                self.reward_config.v81_potential_clamp,
            ));
            let share = land_share(
                ofcore::translate::my_tiles(self.ents(), self.agent_me(i)),
                self.land_total,
            );
            self.closeout_shaper[i].reset(closeout_potential(share));
            if self.n_agents > 1 {
                let partner = self.agent_me(1 - i).max(0) as usize;
                let s_me = initial_strengths.get(&me).copied().unwrap_or(0.0);
                let s_partner = initial_strengths.get(&partner).copied().unwrap_or(0.0);
                let both = self.agent_alive(0) && self.agent_alive(1);
                let allied = formally_allied(
                    self.ents(),
                    self.agent_me(0).max(0) as usize,
                    self.agent_me(1).max(0) as usize,
                );
                self.duo_shaper[i].reset(duo_potential(s_me, s_partner, both, allied));
                self.eco_shaper[i].reset(economy_potential(
                    player_gold_income(self.ents(), me),
                    player_gold_income(self.ents(), partner),
                    self.reward_config.duo_eco_coef,
                ));
                let owners = [me, partner];
                self.boat_commit_shaper[i].reset(boat_commit_potential(
                    team_transport_ships(self.ents(), &owners),
                    self.reward_config.duo_boat_commit,
                ));
                self.leftover_continent_shaper[i].reset(self.leftover_continent_phi());
                self.port_stand_shaper[i].reset(port_stand_potential(
                    team_completed_structures(self.ents(), &owners, PORT_UNIT_CLASS),
                    self.reward_config.duo_port_stand,
                ));
                self.continent_span_shaper[i].reset(self.continent_span_phi());
            }
            self.closeout_tracker[i].reset(share, tick);
            self.action_churn_tracker[i].reset();
            self.boat_tracker[i].reset();
            self.combat_tracker[i].reset();
            self.prev_action[i] = ActionOutcome::default();
            self.last_commitment[i] = None;
            self.was_alive[i] = self.agent_alive(i);
            self.ever_alive[i] |= self.was_alive[i];
        }
    }

    pub fn ents(&self) -> &feat::EntsData {
        self.cached_ents
            .as_ref()
            .expect("obs cache missing; call set_obs first")
    }

    /// Leftover-opponent purity on continents the team occupies.
    /// Disabled when `duo_leftover_continent` is 0 or obs is missing.
    fn leftover_continent_phi(&self) -> f64 {
        let coef = self.reward_config.duo_leftover_continent;
        if coef == 0.0 || self.n_agents <= 1 {
            return 0.0;
        }
        let Some(obs) = self.obs.as_ref() else {
            return 0.0;
        };
        let map_width = obs.head["width"].as_u64().unwrap_or(self.wr as u64) as usize;
        if map_width == 0 {
            return 0.0;
        }
        let agents = [
            self.agent_me(0).max(0) as usize,
            self.agent_me(1).max(0) as usize,
        ];
        let team = team_owner_ids(self.ents(), &agents);
        let (team_n, leftover_n) = leftover_continent_counts(
            &self.continent_of,
            self.wr,
            self.hr,
            map_width,
            |src| obs.owner_at(src),
            &team,
        );
        leftover_continent_potential(team_n, leftover_n, coef)
    }

    /// Occupied-landmass count Φ. Disabled when `duo_continent_span` is 0
    /// or obs is missing. Complementary to leftover-continent: first tile
    /// on a new island raises this even when leftover red is still there.
    fn continent_span_phi(&self) -> f64 {
        let coef = self.reward_config.duo_continent_span;
        if coef == 0.0 || self.n_agents <= 1 {
            return 0.0;
        }
        let Some(obs) = self.obs.as_ref() else {
            return 0.0;
        };
        let map_width = obs.head["width"].as_u64().unwrap_or(self.wr as u64) as usize;
        if map_width == 0 {
            return 0.0;
        }
        let agents = [
            self.agent_me(0).max(0) as usize,
            self.agent_me(1).max(0) as usize,
        ];
        let team = team_owner_ids(self.ents(), &agents);
        let n = occupied_continent_count(
            &self.continent_of,
            self.wr,
            self.hr,
            map_width,
            |src| obs.owner_at(src),
            &team,
        );
        continent_span_potential(n, coef)
    }

    pub fn legal(&self) -> &feat::Legal {
        self.cached_legal
            .as_ref()
            .expect("obs cache missing; call set_obs first")
    }

    fn current_lut(&mut self) -> Vec<u8> {
        let spawn_phase = self.obs.as_ref().unwrap().spawn_phase();
        // Mirrors ObsBuilder._slot_lut: rebuild every tick during spawn
        // (roster still filling in), freeze on first post-spawn obs.
        if spawn_phase || self.lut.is_empty() {
            let ids: Vec<usize> = self.ents().players.iter().map(|p| p.id).collect();
            let lut = feat::make_lut(&ids);
            if !spawn_phase {
                self.lut = lut.clone();
            }
            lut
        } else {
            self.lut.clone()
        }
    }

    pub fn prepare(&mut self) -> PreparedObs {
        self.prepare_agent(0)
    }

    pub fn prepare_all(&mut self) -> Vec<PreparedObs> {
        let mut outs: Vec<PreparedObs> =
            (0..self.n_agents).map(|i| self.prepare_agent(i)).collect();
        Self::fill_partner_features(&mut outs);
        outs
    }

    /// Copy sibling player tokens / scalars into each row's partner extras.
    /// Must run at prepare time: shape-bucketing and minibatch shuffle later
    /// break adjacent-row pairing, so V cannot gather `i^1` on the GPU.
    pub(crate) fn fill_partner_features(outs: &mut [PreparedObs]) {
        if outs.len() != 2 {
            return;
        }
        let a_players = outs[0].players.clone();
        let a_pmask = outs[0].pmask;
        let a_scalars = outs[0].scalars;
        let a_context = outs[0].prev_action.as_floats();
        let b_players = outs[1].players.clone();
        let b_pmask = outs[1].pmask;
        let b_scalars = outs[1].scalars;
        let b_context = outs[1].prev_action.as_floats();
        outs[0].partner_players = b_players;
        outs[0].partner_pmask = b_pmask;
        outs[0].partner_scalars = b_scalars;
        outs[0].partner_context = b_context;
        outs[1].partner_players = a_players;
        outs[1].partner_pmask = a_pmask;
        outs[1].partner_scalars = a_scalars;
        outs[1].partner_context = a_context;
    }

    fn prepare_agent(&mut self, agent_i: usize) -> PreparedObs {
        let profile = std::env::var_os("OF_COLLECT_PROFILE").is_some();
        let t0 = std::time::Instant::now();
        let lut = self.current_lut();
        let me = self.agent_me(agent_i);
        let clut = feat::make_clut(&lut, me, self.ents());
        let (hr, wr) = (self.hr, self.wr);
        let (gh, gw) = (hr / REGION, wr / REGION);
        let width = self.obs.as_ref().unwrap().head["width"]
            .as_u64()
            .unwrap_or(wr as u64) as usize;
        let tiles = self.obs.as_ref().unwrap().prepare_tiles_with_ego(
            &lut,
            width,
            hr,
            wr,
            REGION,
            Some(&clut),
        );
        let owners_slotted = tiles.owners_slotted;
        let ego = tiles.ego;
        let center = tiles.center;

        let tick = self.obs.as_ref().unwrap().tick();
        let spawn_phase = self.obs.as_ref().unwrap().spawn_phase();
        let alive = self.agent_alive(agent_i);
        let legal = self.agent_legal(agent_i).clone();
        let f = feat::featurize(
            gh,
            gw,
            &lut,
            &self.land,
            &self.mag,
            &owners_slotted,
            tick,
            spawn_phase,
            alive,
            me,
            self.ents(),
            &legal,
        );
        debug_assert_eq!(f.clut, clut);

        let local = {
            let obs = self.obs.as_ref().unwrap();
            feat::local_crop_at_with_defense(
                &owners_slotted,
                &f.clut,
                &self.land,
                hr,
                wr,
                crate::policy::LOCAL as usize,
                center,
                |i| {
                    let y = i / wr;
                    let x = i % wr;
                    obs.defense_bonus_at(y * width + x)
                },
            )
        };
        let ae_raw = AeRaw {
            owners: owners_slotted,
            static_terrain: self.ae_static.clone(),
            fallout: tiles.fallout_packed,
            stat: f.stat,
            hr,
            wr,
        };

        let out = PreparedObs {
            prev_action: self.prev_action[agent_i].clone(),
            compact: None,
            grid: None,
            grid_coarse: None,
            cgh: 0,
            cgw: 0,
            ae_raw,
            ego,
            db: tiles.db,
            transient: f.transient,
            legal_tile: f.legal_tile,
            gh,
            gw,
            players: f.players,
            pmask: f.pmask,
            units: f.units,
            umask: f.umask,
            legal_utarget: f.legal_utarget,
            scalars: f.scalars,
            me_slot: f.me_slot,
            legal_actions: f.legal_actions,
            legal_ptarget: f.legal_ptarget,
            legal_build: f.legal_build,
            legal_nuke: f.legal_nuke,
            local,
            partner_players: vec![0.0; feat::MAX_SLOTS * feat::P_FEAT],
            partner_pmask: [0.0; feat::MAX_SLOTS],
            partner_scalars: [0.0; feat::N_SCALARS],
            partner_context: [0.0; crate::recurrent::CONTEXT_FLOATS],
        };
        if profile {
            static PREPARE_N: AtomicU64 = AtomicU64::new(0);
            let n = PREPARE_N.fetch_add(1, Ordering::Relaxed);
            if n < 8 || n % 64 == 0 {
                eprintln!(
                    "[collect-profile] prepare env={} n={} prepare_us={} hr={} wr={}",
                    self.idx,
                    n,
                    t0.elapsed().as_micros(),
                    hr,
                    wr
                );
            }
        }
        out
    }

    /// Combined apply-then-prepare, matching a Gym-style `env.step()`:
    /// returns the NEXT observation alongside the reward/done/info from
    /// applying `choice` to the current one. Drives the threaded rollout
    /// loop in `train.rs`.
    pub fn step(&mut self, choice: &Choice) -> Result<EnvTransition> {
        let mut outs = self.step_agents(std::slice::from_ref(choice))?;
        Ok(outs.remove(0))
    }

    pub fn step_agents(&mut self, choices: &[Choice]) -> Result<Vec<EnvTransition>> {
        let profile = std::env::var_os("OF_COLLECT_PROFILE").is_some();
        let t0 = std::time::Instant::now();
        let scored = self.apply_agents(choices)?;
        let apply_us = t0.elapsed().as_micros();
        let t1 = std::time::Instant::now();
        let prepared = self.prepare_all();
        let prepare_us = t1.elapsed().as_micros();
        if profile {
            static STEP_N: AtomicU64 = AtomicU64::new(0);
            let n = STEP_N.fetch_add(1, Ordering::Relaxed);
            if n < 8 || n % 64 == 0 {
                eprintln!(
                    "[collect-profile] env={} step_n={} apply_us={} prepare_us={} hr={} wr={}",
                    self.idx, n, apply_us, prepare_us, self.hr, self.wr
                );
            }
        }
        Ok(scored
            .into_iter()
            .zip(prepared)
            .map(|((reward, done, info, outcome), next_obs)| EnvTransition {
                next_obs,
                reward,
                done,
                info,
                outcome,
            })
            .collect())
    }

    /// Translate + step. Auto-resets on episode end.
    pub fn apply(
        &mut self,
        choice: &Choice,
    ) -> Result<(f64, bool, Option<EpisodeInfo>, ActionOutcome)> {
        let mut outs = self.apply_agents(std::slice::from_ref(choice))?;
        Ok(outs.remove(0))
    }

    pub fn apply_agents(
        &mut self,
        choices: &[Choice],
    ) -> Result<Vec<(f64, bool, Option<EpisodeInfo>, ActionOutcome)>> {
        let n = self.n_agents.max(1);
        ensure!(
            choices.len() == n,
            "apply_agents expected {n} choices, got {}",
            choices.len()
        );
        let lut = self.current_lut();
        let width = self.obs.as_ref().unwrap().head["width"].as_u64().unwrap() as usize;
        let mut owners_trim = vec![0i64; self.hr * self.wr];
        {
            let obs = self.obs.as_ref().unwrap();
            for y in 0..self.hr {
                for x in 0..self.wr {
                    owners_trim[y * self.wr + x] = obs.owner_at(y * width + x) as i64;
                }
            }
        }
        let ents = self.ents().clone();
        let mut all_intents: Vec<Value> = Vec::new();
        let mut agent_intents: Vec<Vec<Value>> = Vec::with_capacity(n);
        let mut boats_before: Vec<Vec<usize>> = Vec::with_capacity(n);
        let mut me_pre: Vec<usize> = Vec::with_capacity(n);
        let mut troops_before: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let legal = self.agent_legal(i).clone();
            boats_before.push(legal.boats.clone());
            let me = self.agent_me(i);
            me_pre.push(me.max(0) as usize);
            troops_before.push(player_troops(&ents, me.max(0) as usize));
            let mut intents = translate(
                &choices[i],
                self.translator.as_mut().unwrap(),
                &owners_trim,
                me,
                &ents,
                &legal,
                &lut,
            );
            for intent in &mut intents {
                if let Some(map) = intent.as_object_mut() {
                    map.insert("clientID".into(), Value::String(AGENT_CLIENT_IDS[i].into()));
                }
            }
            all_intents.extend(intents.iter().cloned());
            agent_intents.push(intents);
        }
        if let Some(staggered) = stagger_simultaneous_duo_spawn(
            self.obs.as_ref().unwrap().spawn_phase(),
            &(0..n).map(|i| !self.agent_alive(i)).collect::<Vec<_>>(),
            &agent_intents,
        ) {
            all_intents = staggered;
        }
        let pre_attack_ids: HashSet<String> = ents.attacks.iter().map(|a| a.aid.clone()).collect();

        let new_obs = self.bridge.step(&all_intents, self.decision_ticks)?;
        let boats_after_global = if let Some((_, legal)) = new_obs.structured.as_ref() {
            legal.boats.clone()
        } else {
            feat::parse_legal(new_obs.legal_actions()).boats
        };
        self.set_obs(new_obs);

        if self.obs.as_ref().unwrap().spawn_phase() {
            self.spawn_steps += 1;
            if self.spawn_steps >= 8 {
                self.spawn_randomly()?;
            }
            self.seed_agent_trackers();
            let still_spawning = self.obs.as_ref().unwrap().spawn_phase() && self.spawn_steps < 16;
            if still_spawning {
                self.ep_len += 1;
                let dummy = ActionOutcome::default();
                return Ok((0..n).map(|_| (0.0, false, None, dummy.clone())).collect());
            }
            // Spawn never completed: fall through so never-alive is a death.
        }

        let winner_val = self.obs.as_ref().unwrap().winner().clone();
        let obs_tick = self.obs.as_ref().unwrap().tick();
        let alives: Vec<bool> = (0..n).map(|i| self.agent_alive(i)).collect();
        let all_dead = alives.iter().all(|a| !*a);
        let won = humans_won(&winner_val, n);
        let mut timed_out = false;
        let mut done = false;
        let mut died = false;
        if all_dead {
            done = true;
            died = true;
        } else if !winner_val.is_null() {
            done = true;
        } else if obs_tick >= self.max_episode_ticks {
            done = true;
            timed_out = true;
        }
        if done && self.ever_alive.iter().take(n).all(|alive| !*alive) {
            died = true;
        }

        let partner_me = if n > 1 {
            self.agent_me(1).max(0) as usize
        } else {
            usize::MAX
        };
        let composite = curriculum::strengths(self.ents(), self.land_total);
        let s0 = composite
            .get(&(self.agent_me(0).max(0) as usize))
            .copied()
            .unwrap_or(0.0);
        let s1 = if n > 1 {
            composite.get(&partner_me).copied().unwrap_or(0.0)
        } else {
            0.0
        };
        let both_alive = n > 1 && alives[0] && alives[1];
        let allied =
            n > 1 && formally_allied(self.ents(), self.agent_me(0).max(0) as usize, partner_me);
        let next_duo_phi = if n > 1 {
            duo_potential(s0, s1, both_alive, allied)
        } else {
            0.0
        };
        let next_eco_phi = if n > 1 {
            economy_potential(
                player_gold_income(self.ents(), self.agent_me(0).max(0) as usize),
                player_gold_income(self.ents(), partner_me),
                self.reward_config.duo_eco_coef,
            )
        } else {
            0.0
        };
        let team_owners: [usize; 2] = [
            self.agent_me(0).max(0) as usize,
            if n > 1 { partner_me } else { usize::MAX },
        ];
        let next_boat_phi = if n > 1 {
            boat_commit_potential(
                team_transport_ships(self.ents(), &team_owners),
                self.reward_config.duo_boat_commit,
            )
        } else {
            0.0
        };
        let next_leftover_phi = if n > 1 {
            self.leftover_continent_phi()
        } else {
            0.0
        };
        let n_cities = if n > 1 {
            team_completed_structures(self.ents(), &team_owners, CITY_UNIT_CLASS)
        } else {
            0
        };
        let n_ports = if n > 1 {
            team_completed_structures(self.ents(), &team_owners, PORT_UNIT_CLASS)
        } else {
            0
        };
        let next_port_phi = if n > 1 {
            port_stand_potential(n_ports, self.reward_config.duo_port_stand)
        } else {
            0.0
        };
        let next_span_phi = if n > 1 {
            self.continent_span_phi()
        } else {
            0.0
        };
        let solo_scale = if n > 1 { DUO_SOLO_SCALE } else { 1.0 };

        let mut results = Vec::with_capacity(n);
        let mut episode_info: Option<EpisodeInfo> = None;
        for i in 0..n {
            let choice = &choices[i];
            let intents = &agent_intents[i];
            let name = ACTIONS[choice.action as usize];
            let boats_after = if choice.action == A_BOAT {
                boats_after_global.clone()
            } else {
                Vec::new()
            };
            let chosen_action =
                churn_action(choice, &lut, &ents, intents, &boats_before[i], &boats_after);
            let mut embargo_outcome_r = 0.0;
            if choice.action == A_EMBARGO_STOP {
                if let Some(ActionTarget::Player(target)) = chosen_action.target {
                    let rel = ents
                        .players
                        .iter()
                        .find(|p| p.id == me_pre[i])
                        .map(|p| p.relation_to(target))
                        .unwrap_or(0.0);
                    embargo_outcome_r = self.combat_tracker[i].observe_embargo_stop(
                        rel,
                        self.reward_config,
                        self.episode_stage,
                    );
                }
            }
            let combat_target = match chosen_action.target {
                Some(ActionTarget::Player(id))
                    if choice.action == A_ATTACK || choice.action == A_RETREAT =>
                {
                    Some(id)
                }
                _ => None,
            };
            let combat_outcome_r = self.combat_tracker[i].observe_combat(
                choice.action,
                combat_target,
                self.reward_config.v81_churn_window,
                self.reward_config,
                self.episode_stage,
            );
            if let Some(ActionTarget::Unit(unit_id)) = chosen_action.target {
                if choice.action == A_BOAT && !intents.is_empty() {
                    let troops = self
                        .ents()
                        .units
                        .iter()
                        .find(|u| u.uid == unit_id as i64 && u.class == TRANSPORT_UNIT_CLASS)
                        .map(|u| u.troops)
                        .unwrap_or(0.0);
                    self.boat_tracker[i].note_launch(unit_id, troops);
                } else if choice.action == A_CANCEL_BOAT {
                    self.boat_tracker[i].note_cancel(unit_id);
                }
            }
            let inverse_pair = self.action_churn_tracker[i]
                .observe(chosen_action, self.reward_config.v81_churn_window);
            let engine_wasted = self.obs.as_ref().unwrap().wasted();
            let mut wasted = if i == 0 { engine_wasted } else { 0 };
            if intents.is_empty() && name != "noop" && name != "spawn" {
                wasted += 1;
            }
            let (target_kind, target_identity) = match chosen_action.target {
                Some(ActionTarget::Player(id)) => (1, id as i64),
                Some(ActionTarget::Unit(id)) => (2, id as i64),
                None => (0, -1),
            };
            let (target_y, target_x) = choice
                .tile_region
                .map(|tile| normalized_tile_target(tile, self.hr / REGION, self.wr / REGION))
                .unwrap_or((-1.0, -1.0));
            let commitment = (
                choice.action,
                choice.player_slot.unwrap_or(-1),
                choice.tile_region.unwrap_or(-1),
                choice.build_type.unwrap_or(-1),
                choice.nuke_type.unwrap_or(-1),
                choice.quantity_frac.unwrap_or(-1.0).to_bits(),
            );
            let commitment_age = if self.last_commitment[i] == Some(commitment) {
                self.prev_action[i].commitment_age.saturating_add(1)
            } else {
                0
            };
            self.last_commitment[i] = Some(commitment);
            let outcome = ActionOutcome {
                action: choice.action,
                player_slot: choice.player_slot.unwrap_or(-1),
                tile_region: choice.tile_region.unwrap_or(-1),
                build_type: choice.build_type.unwrap_or(-1),
                nuke_type: choice.nuke_type.unwrap_or(-1),
                success: name == "noop" || intents.len() as i64 > engine_wasted,
                wasted: wasted > 0,
                target_identity,
                target_y,
                target_x,
                quantity: choice.quantity_frac.unwrap_or(-1.0) as f32,
                commitment_age,
                had_action: true,
                target_kind,
            };
            self.prev_action[i] = outcome.clone();

            let obs_me = self.agent_me(i);
            let obs_alive = alives[i];
            let tiles = ofcore::translate::my_tiles(self.ents(), obs_me);
            let share = land_share(tiles, self.land_total);
            let closeout_just_entered =
                self.closeout_tracker[i].observe(share, obs_tick, inverse_pair.is_some());
            let me = obs_me.max(0) as usize;
            let mine = composite.get(&me).copied().unwrap_or(0.0);
            let tw = timeweight(obs_tick);
            let delta = mine - self.prev_strength[i];
            let normalized_share = if self.reward_config.dominant_loss_active(self.episode_stage) {
                normalized_strength_share(&composite, me)
            } else {
                0.0
            };
            let has_active_attack = self.ents().attacks.iter().any(|a| a.from == me);
            let delta_weight = strength_delta_weight(
                delta,
                normalized_share,
                self.episode_stage,
                self.reward_config,
                has_active_attack,
            );
            let mut components = RewardComponents {
                strength: W_STR * mine * tw * solo_scale,
                strength_delta: delta_weight * delta * solo_scale,
                ..RewardComponents::default()
            };
            let mut reward = components.strength + components.strength_delta;
            components.action_churn = if self.curriculum_schedule.uses_v83_closeout() {
                v83_action_churn_penalty(
                    inverse_pair,
                    self.episode_stage,
                    share,
                    self.reward_config,
                )
            } else {
                action_churn_penalty(inverse_pair, self.episode_stage, self.reward_config)
            } * solo_scale;
            if self.reward_config.v86_skip_combat_churn
                && matches!(
                    inverse_pair,
                    Some(InverseActionPair::AttackRetreat | InverseActionPair::RetreatAttack)
                )
            {
                components.action_churn = 0.0;
            }
            if components.action_churn != 0.0 {
                reward += components.action_churn;
            }
            components.embargo_outcome = embargo_outcome_r * solo_scale;
            if components.embargo_outcome != 0.0 {
                reward += components.embargo_outcome;
            }
            components.combat_outcome = combat_outcome_r * solo_scale;
            if components.combat_outcome != 0.0 {
                reward += components.combat_outcome;
            }
            let next_potential = if self.curriculum_schedule.uses_v83_closeout() && done {
                0.0
            } else {
                dominance_potential(&composite, me, self.reward_config.v81_potential_clamp)
            };
            if self
                .reward_config
                .dominance_shaping_active(self.episode_stage)
            {
                components.dominance = self.dominance_shaper[i].transition(
                    next_potential,
                    self.reward_config.gamma,
                    self.reward_config.v81_dom_coef,
                ) * solo_scale;
                reward += components.dominance;
            } else {
                self.dominance_shaper[i].reset(next_potential);
            }
            let next_closeout_potential = if done { 0.0 } else { closeout_potential(share) };
            if self.curriculum_schedule.uses_v83_closeout()
                && self.reward_config.v83_close_coef != 0.0
            {
                components.closeout = self.closeout_shaper[i].transition(
                    next_closeout_potential,
                    self.reward_config.gamma,
                    self.reward_config.v83_close_coef,
                ) * solo_scale;
                reward += components.closeout;
            } else {
                self.closeout_shaper[i].reset(next_closeout_potential);
            }
            let entry_bonus =
                v10_closeout_entry_bonus(closeout_just_entered, self.reward_config) * solo_scale;
            if entry_bonus != 0.0 {
                components.closeout += entry_bonus;
                reward += entry_bonus;
            }

            let troops_after = player_troops(self.ents(), me);
            let alive_transports = transport_unit_ids(self.ents(), me);
            let new_sourced_attack = self.ents().attacks.iter().any(|a| {
                a.from == me
                    && a.src_x.is_some()
                    && a.src_y.is_some()
                    && !pre_attack_ids.contains(&a.aid)
            });
            let has_sourced_attack = self
                .ents()
                .attacks
                .iter()
                .any(|a| a.from == me && a.src_x.is_some() && a.src_y.is_some());
            components.boat_outcome = self.boat_tracker[i].resolve_missing(
                &alive_transports,
                troops_before[i],
                troops_after,
                new_sourced_attack,
                has_sourced_attack,
                self.reward_config,
                self.episode_stage,
            ) * solo_scale;
            if components.boat_outcome != 0.0 {
                reward += components.boat_outcome;
            }

            let tempo_share = if self.reward_config.tempo_active(self.episode_stage) {
                normalized_strength_share(&composite, me)
            } else {
                0.0
            };
            if self.reward_config.tempo_active(self.episode_stage) {
                components.tempo = -self.reward_config.v84_tempo_coef
                    * tempo_pressure(
                        obs_tick,
                        self.max_episode_ticks,
                        tempo_share,
                        self.reward_config.tempo_share_threshold(),
                    )
                    * tw
                    * solo_scale;
                if components.tempo != 0.0 {
                    reward += components.tempo;
                }
            }

            components.survival =
                v10_survival_reward(obs_alive, share, self.reward_config) * solo_scale;
            if components.survival != 0.0 {
                reward += components.survival;
            }
            components.diplo_panic = v10_diplo_panic_penalty(
                choice.action,
                share,
                obs_tick,
                self.max_episode_ticks,
                self.reward_config,
            ) * solo_scale;
            if components.diplo_panic != 0.0 {
                reward += components.diplo_panic;
            }
            let emitted_ok = !intents.is_empty() && (intents.len() as i64) > engine_wasted;
            let has_action_target = match choice.action {
                A_ATTACK | A_BOAT | A_BUILD => emitted_ok,
                _ => {
                    choice.player_slot.is_some()
                        || choice.tile_region.is_some()
                        || matches!(chosen_action.target, Some(_))
                }
            };
            components.combat_action =
                v10_combat_action_bonus(choice.action, has_action_target, self.reward_config)
                    * solo_scale;
            if components.combat_action != 0.0 {
                reward += components.combat_action;
            }

            reward -= W_WASTE * wasted as f64;
            components.waste = -W_WASTE * wasted as f64;
            self.ep_wasted += wasted;
            self.prev_strength[i] = mine;

            if !obs_alive && self.was_alive[i] {
                let death = self.reward_config.death_penalty();
                reward -= death;
                components.death = -death;
            }
            self.was_alive[i] = obs_alive;
            self.ever_alive[i] |= obs_alive;

            if n > 1 {
                // Ng 1999 PBRS on team Φ. Do not pay alliance_request /
                // donate actions, and do not pay alive/allied as a wage
                // (those were the timeout farm). Absorbing terminal Φ=0.
                let phi = if done { 0.0 } else { next_duo_phi };
                let mut duo_r = self.duo_shaper[i].transition(phi, self.reward_config.gamma, 1.0);
                // Ng 1999 PBRS on log team gold-income. Cities/ports raise
                // income; spending gold stock on a city does not drop Φ
                // (unlike K_ECO). Absorbing terminal Φ=0.
                let eco_phi = if done { 0.0 } else { next_eco_phi };
                duo_r += self.eco_shaper[i].transition(eco_phi, self.reward_config.gamma, 1.0);
                // Ng 1999 PBRS on team in-flight TransportShip count.
                // Launch raises Φ; land/cancel/destroy drops it. Never the
                // `boat` action. Absorbing terminal Φ=0.
                let boat_phi = if done { 0.0 } else { next_boat_phi };
                duo_r +=
                    self.boat_commit_shaper[i].transition(boat_phi, self.reward_config.gamma, 1.0);
                // Ng 1999 PBRS on leftover opponent tiles on continents the
                // team occupies. Mop leftover red raises Φ; flee a dirty
                // continent drops it. Never an attack/boat action.
                // Absorbing terminal Φ=0.
                let leftover_phi = if done { 0.0 } else { next_leftover_phi };
                duo_r += self.leftover_continent_shaper[i].transition(
                    leftover_phi,
                    self.reward_config.gamma,
                    1.0,
                );
                // Ng 1999 PBRS on team completed Port count. Completing a
                // Port raises Φ; losing one drops it. Never the `build`
                // action. Absorbing terminal Φ=0.
                let port_phi = if done { 0.0 } else { next_port_phi };
                duo_r +=
                    self.port_stand_shaper[i].transition(port_phi, self.reward_config.gamma, 1.0);
                // Ng 1999 PBRS on occupied landmass count. First tile on a
                // new continent raises Φ even if leftover red is still
                // there. Never a boat/attack action. Absorbing terminal Φ=0.
                let span_phi = if done { 0.0 } else { next_span_phi };
                duo_r += self.continent_span_shaper[i].transition(
                    span_phi,
                    self.reward_config.gamma,
                    1.0,
                );
                // Outcome-only one-shot: first formal pact this episode.
                if allied && !self.pact_bonus_paid {
                    duo_r += duo_pact_success_bonus(true, self.reward_config);
                }
                // Outcome-only one-shots: first completed City / Port.
                // Never the `build` action; never per-tick while standing.
                if n_cities > 0 && !self.city_bonus_paid {
                    duo_r += duo_first_structure_bonus(true, self.reward_config.duo_first_city);
                }
                if n_ports > 0 && !self.port_bonus_paid {
                    duo_r += duo_first_structure_bonus(true, self.reward_config.duo_first_port);
                }
                // Outcome-only: completed City/Port count dropped. Never
                // the `delete_unit` action.
                let dropped_cities = self.prev_n_cities.saturating_sub(n_cities);
                let dropped_ports = self.prev_n_ports.saturating_sub(n_ports);
                duo_r += duo_structure_delete_penalty(
                    dropped_cities,
                    self.reward_config.duo_city_delete,
                );
                duo_r +=
                    duo_structure_delete_penalty(dropped_ports, self.reward_config.duo_port_delete);
                components.duo = duo_r;
                reward += duo_r;
            }

            if done {
                let (place, _pn) = placement(self.ents(), obs_me, obs_alive, self.land_total);
                // Never-spawned is a death/loss, not a placement gift.
                if !self.ever_alive[i] && !won && components.death == 0.0 {
                    let death = self.reward_config.death_penalty();
                    reward -= death;
                    components.death = -death;
                }
                let no_play = timed_out || !self.ever_alive[i];
                components.terminal = (terminal_reward(place, won, no_play)
                    + fast_win_bonus(
                        won,
                        obs_tick,
                        self.max_episode_ticks,
                        self.reward_config.v84_fast_win_coef,
                    ))
                    * if n > 1 { 1.0 } else { 1.0 };
                if won {
                    components.terminal += self.reward_config.v85_extra_win_bonus;
                }
                components.terminal += v10_timeout_after_closeout_penalty(
                    timed_out,
                    self.closeout_tracker[i].reached(),
                    self.reward_config,
                );
                reward += components.terminal;
            }
            self.ep_reward_components.add_assign(components);
            self.ep_reward += reward;

            if done && i == 0 {
                let (place, pn) = placement(self.ents(), obs_me, obs_alive, self.land_total);
                episode_info = Some(EpisodeInfo {
                    reward: self.ep_reward,
                    length: self.ep_len + 1,
                    final_tiles: tiles,
                    final_land_share: share,
                    max_land_share: self.closeout_tracker[0].max_land_share,
                    closeout_reached: self.closeout_tracker[0].reached(),
                    closeout_entry_tick: self.closeout_tracker[0].entry_tick,
                    decisions_after_closeout: self.closeout_tracker[0].decisions_after_entry,
                    converted: self.closeout_tracker[0].reached() && won,
                    timeout_after_closeout: timed_out && self.closeout_tracker[0].reached(),
                    post_closeout_churn_pairs: self.closeout_tracker[0].post_entry_churn_pairs,
                    final_tick: obs_tick,
                    place,
                    n_players: pn,
                    score: placement_score(place, pn),
                    won,
                    died,
                    wasted: self.ep_wasted,
                    stage: self.stage,
                    map: self.map_name.clone(),
                    rehearsal: self.rehearsal,
                    reward_components: self.ep_reward_components,
                    action_pair_counts: self.action_churn_tracker[0].counts(),
                    boat_outcome_counts: self.boat_tracker[0].counts(),
                    embargo_bad_stops: self.combat_tracker[0].embargo_bad_stops,
                    embargo_good_stops: self.combat_tracker[0].embargo_good_stops,
                    premature_retreats: self.combat_tracker[0].premature_retreats,
                    thrash_reengages: self.combat_tracker[0].thrash_reengages,
                });
            }
            results.push((
                reward,
                done,
                if i == 0 { episode_info.clone() } else { None },
                outcome,
            ));
        }
        if n > 1 && allied {
            self.pact_bonus_paid = true;
        }
        if n > 1 && n_cities > 0 {
            self.city_bonus_paid = true;
        }
        if n > 1 && n_ports > 0 {
            self.port_bonus_paid = true;
        }
        self.prev_n_cities = n_cities;
        self.prev_n_ports = n_ports;
        self.ep_len += 1;
        if done {
            self.spool_finished_episode(won, timed_out);
            self.reset_episode()?;
        }
        Ok(results)
    }

    /// Emergency fallback matching `rl/ppo_translate.py::spawn_randomly`:
    /// stalled spawn snapping picks a uniformly random legal tile instead.
    fn spawn_randomly(&mut self) -> Result<()> {
        let obs = self.obs.as_ref().unwrap();
        let width = obs.head["width"].as_u64().unwrap() as usize;
        let mut candidates = Vec::new();
        for y in 0..self.hr {
            for x in 0..self.wr {
                let i = y * self.wr + x;
                let src = y * width + x;
                if self.land[i] == 1
                    && self.mag[i] < feat::IMPASSABLE_MAGNITUDE
                    && obs.owner_at(src) == 0
                {
                    candidates.push((y as i64, x as i64));
                }
            }
        }
        if candidates.is_empty() {
            return Ok(());
        }
        let mut intents = Vec::new();
        for i in 0..self.n_agents {
            if self.agent_alive(i) {
                continue;
            }
            if candidates.is_empty() {
                break;
            }
            let idx = self.rng.gen_range(0..candidates.len());
            let (y, x) = candidates.swap_remove(idx);
            let tile = y * width as i64 + x;
            intents.push(serde_json::json!({
                "type": "spawn",
                "tile": tile,
                "clientID": AGENT_CLIENT_IDS[i],
            }));
        }
        if intents.is_empty() {
            return Ok(());
        }
        let new_obs = self.bridge.step(&intents, self.decision_ticks)?;
        self.set_obs(new_obs);
        Ok(())
    }

    pub fn set_stage(&mut self, stage: usize) {
        self.stage = stage;
    }

    pub fn close(&mut self) {
        self.bridge.close();
    }
}

/// During spawn, if every agent is still unplaced and all emit spawn in
/// the same step, keep only agent 0's intents so agent 1's next obs sees
/// the partner blob (class-2 clut) instead of a simultaneous empty-map
/// commit. Without this, both heads still land on tick 1 and sequential
/// spawn is a no-op.
fn stagger_simultaneous_duo_spawn(
    spawn_phase: bool,
    unspawned: &[bool],
    agent_intents: &[Vec<Value>],
) -> Option<Vec<Value>> {
    if !spawn_phase || agent_intents.len() < 2 {
        return None;
    }
    if unspawned.len() != agent_intents.len() || !unspawned.iter().all(|&u| u) {
        return None;
    }
    let is_spawn = |intents: &[Value]| {
        intents
            .iter()
            .any(|v| v.get("type").and_then(|t| t.as_str()) == Some("spawn"))
    };
    if agent_intents.iter().all(|intents| is_spawn(intents)) {
        Some(agent_intents[0].clone())
    } else {
        None
    }
}

#[cfg(test)]
mod churn_action_tests {
    use super::*;
    use serde_json::json;

    fn choice(action: i64, player_slot: Option<i64>, tile_region: Option<i64>) -> Choice {
        Choice {
            action,
            player_slot,
            tile_region,
            unit_index: None,
            build_type: None,
            nuke_type: None,
            quantity_frac: None,
        }
    }

    #[test]
    fn records_only_resolved_player_and_transport_targets() {
        let ents = feat::parse_ents(&json!({
            "players": [
                {"id": 1, "pid": "me", "alive": true},
                {"id": 5, "pid": "target", "alive": true}
            ],
            "units": [],
            "attacks": [
                {"aid": "attack-5", "from": 1, "to": 5, "retreating": false}
            ],
            "alliances": []
        }));
        let lut = feat::make_lut(&[1, 5]);
        let target_slot = i64::from(lut[5]);

        assert_eq!(
            churn_action(
                &choice(A_ATTACK, Some(target_slot), None),
                &lut,
                &ents,
                &[json!({"type": "attack", "targetID": "target"})],
                &[],
                &[]
            ),
            ChosenAction::new(A_ATTACK, Some(ActionTarget::Player(5)))
        );
        assert_eq!(
            churn_action(
                &choice(A_RETREAT, Some(target_slot), None),
                &lut,
                &ents,
                &[json!({"type": "cancel_attack", "attackID": "attack-5"})],
                &[],
                &[]
            ),
            ChosenAction::new(A_RETREAT, Some(ActionTarget::Player(5)))
        );
        assert_eq!(
            churn_action(
                &choice(A_ALLIANCE_REQUEST, Some(target_slot), None),
                &lut,
                &ents,
                &[json!({"type": "allianceRequest", "recipientID": "target"})],
                &[],
                &[]
            ),
            ChosenAction::new(A_ALLIANCE_REQUEST, Some(ActionTarget::Player(5)))
        );
        assert_eq!(
            churn_action(
                &choice(A_BREAK_ALLIANCE, Some(target_slot), None),
                &lut,
                &ents,
                &[json!({"type": "breakAlliance", "recipientID": "target"})],
                &[],
                &[]
            ),
            ChosenAction::new(A_BREAK_ALLIANCE, Some(ActionTarget::Player(5)))
        );
        assert_eq!(
            churn_action(
                &choice(A_BOAT, None, Some(27)),
                &lut,
                &ents,
                &[json!({"type": "boat", "dst": 27})],
                &[9],
                &[9, 42]
            ),
            ChosenAction::new(A_BOAT, Some(ActionTarget::Unit(42)))
        );
        assert_eq!(
            churn_action(
                &choice(A_CANCEL_BOAT, None, Some(27)),
                &lut,
                &ents,
                &[json!({"type": "cancel_boat", "unitID": 42})],
                &[42],
                &[]
            ),
            ChosenAction::new(A_CANCEL_BOAT, Some(ActionTarget::Unit(42)))
        );
        assert_eq!(
            churn_action(
                &choice(feat::A_DONATE_GOLD, Some(target_slot), None),
                &lut,
                &ents,
                &[json!({"type": "donate_gold"})],
                &[],
                &[]
            ),
            ChosenAction::new(feat::A_DONATE_GOLD, None)
        );
        assert_eq!(
            churn_action(
                &choice(A_ATTACK, Some(target_slot), None),
                &lut,
                &ents,
                &[],
                &[],
                &[]
            ),
            ChosenAction::new(A_ATTACK, None),
            "an untranslated choice is not a clear committed action"
        );
        assert_eq!(
            churn_action(
                &choice(A_BOAT, None, Some(27)),
                &lut,
                &ents,
                &[json!({"type": "boat", "dst": 27})],
                &[],
                &[41, 42]
            ),
            ChosenAction::new(A_BOAT, None),
            "ambiguous transport creation must not create a false match"
        );
    }

    #[test]
    fn recurrent_tile_context_uses_global_policy_stride() {
        let tile = 3 * feat::GW_MAX as i64 + 7;
        let (y, x) = normalized_tile_target(tile, 10, 20);
        assert!((y - 0.35).abs() < 1e-6);
        assert!((x - 0.375).abs() < 1e-6);
    }

    #[test]
    fn closeout_tracker_records_entry_max_decisions_conversion_inputs_and_reset() {
        let mut tracker = CloseoutTracker::default();
        tracker.reset(0.10, 100);
        assert!(!tracker.observe(0.44, 200, true));
        assert!(!tracker.reached());
        assert_eq!(tracker.post_entry_churn_pairs, 0);

        assert!(tracker.observe(0.45, 300, false));
        assert!(!tracker.observe(0.62, 400, true));
        assert!(!tracker.observe(0.50, 500, false));
        assert!(tracker.reached());
        assert_eq!(tracker.entry_tick, Some(300));
        assert_eq!(tracker.max_land_share, 0.62);
        assert_eq!(tracker.decisions_after_entry, 2);
        assert_eq!(tracker.post_entry_churn_pairs, 1);

        tracker.reset(0.20, 600);
        assert_eq!(
            tracker,
            CloseoutTracker {
                max_land_share: 0.20,
                ..CloseoutTracker::default()
            }
        );
    }
}

#[cfg(test)]
mod compact_extras_tests {
    use super::*;

    #[test]
    fn partner_block_appends_after_local_extras() {
        let partner_n = compact_extras_players_n()
            + feat::MAX_SLOTS
            + feat::N_SCALARS
            + crate::recurrent::CONTEXT_FLOATS;
        assert_eq!(
            compact_extras_per_env(),
            compact_extras_core_n() + partner_n
        );
        assert!(partner_n > 0);
    }

    #[test]
    fn fill_partner_features_swaps_sibling_player_rows() {
        let a = PreparedObs {
            prev_action: ActionOutcome::default(),
            compact: None,
            grid: None,
            grid_coarse: None,
            cgh: 0,
            cgw: 0,
            ae_raw: crate::ae::AeRaw {
                owners: Vec::new(),
                static_terrain: crate::ae::StaticTerrain {
                    key: crate::ae::TerrainCacheKey {
                        env_id: 0,
                        episode: 0,
                        static_id: 0,
                        hr: 8,
                        wr: 8,
                    },
                    map: std::sync::Arc::from("t"),
                    land_mag: Vec::<f32>::new().into(),
                },
                fallout: Vec::new(),
                stat: Vec::new(),
                hr: 8,
                wr: 8,
            },
            ego: Vec::new(),
            db: Vec::new(),
            transient: Vec::new(),
            legal_tile: Vec::new(),
            gh: 1,
            gw: 1,
            players: vec![1.0; feat::MAX_SLOTS * feat::P_FEAT],
            pmask: [1.0; feat::MAX_SLOTS],
            units: Vec::new(),
            umask: [0.0; feat::MAX_UNITS],
            legal_utarget: Vec::new(),
            scalars: [3.0; feat::N_SCALARS],
            me_slot: 0,
            legal_actions: [0.0; feat::N_ACTIONS],
            legal_ptarget: Vec::new(),
            legal_build: [0.0; feat::N_BUILD],
            legal_nuke: [0.0; feat::N_NUKE],
            local: Vec::new(),
            partner_players: vec![0.0; feat::MAX_SLOTS * feat::P_FEAT],
            partner_pmask: [0.0; feat::MAX_SLOTS],
            partner_scalars: [0.0; feat::N_SCALARS],
            partner_context: [0.0; crate::recurrent::CONTEXT_FLOATS],
        };
        let mut b = a.clone();
        b.players = vec![2.0; feat::MAX_SLOTS * feat::P_FEAT];
        b.pmask = [0.5; feat::MAX_SLOTS];
        b.scalars = [4.0; feat::N_SCALARS];
        b.prev_action.action = 7;
        b.prev_action.had_action = true;
        let mut outs = vec![a, b];
        EnvWorker::fill_partner_features(&mut outs);
        assert_eq!(outs[0].partner_players[0], 2.0);
        assert_eq!(outs[1].partner_players[0], 1.0);
        assert_eq!(outs[0].partner_pmask[0], 0.5);
        assert_eq!(outs[1].partner_pmask[0], 1.0);
        assert_eq!(outs[0].partner_scalars[0], 4.0);
        assert_eq!(outs[1].partner_scalars[0], 3.0);
        assert_eq!(outs[0].partner_context[0], 7.0);
        assert_eq!(outs[0].partner_context[12], 1.0);
        assert_eq!(outs[1].partner_context[0], -1.0);
        EnvWorker::fill_partner_features(&mut outs[..1]);
        assert_eq!(outs[0].partner_players[0], 2.0, "solo slice is a no-op");
    }
}

#[cfg(test)]
mod sequential_spawn_tests {
    use super::*;
    use serde_json::json;

    fn spawn_intent(tile: i64) -> Vec<Value> {
        vec![json!({"type": "spawn", "tile": tile, "clientID": "AGENTRL1"})]
    }

    #[test]
    fn drops_agent1_spawn_when_both_unplaced() {
        let kept = stagger_simultaneous_duo_spawn(
            true,
            &[true, true],
            &[spawn_intent(10), spawn_intent(20)],
        )
        .expect("should stagger");
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0]["tile"], 10);
    }

    #[test]
    fn does_not_stagger_once_a_partner_is_on_the_map() {
        assert!(stagger_simultaneous_duo_spawn(
            true,
            &[false, true],
            &[spawn_intent(10), spawn_intent(20)],
        )
        .is_none());
    }

    #[test]
    fn does_not_stagger_outside_spawn_phase() {
        assert!(stagger_simultaneous_duo_spawn(
            false,
            &[true, true],
            &[spawn_intent(10), spawn_intent(20)],
        )
        .is_none());
    }
}
