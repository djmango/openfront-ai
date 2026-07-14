//! Single-env worker: bridge session + curriculum episode bookkeeping +
//! reward shaping. Port of `rl/vec.py::EnvWorker`. `VecEnv` fans this out
//! over one OS thread per env (see module-level doc below); unlike the
//! Python side there's no GIL, so no multiprocessing/pickle framing is
//! needed to keep JSON decode + featurization off the main thread.

use anyhow::Result;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use ofcore::curriculum::{
    self, ActionChurnTracker, ActionPairCounts, ActionTarget, BoatOutcomeCounts, ChosenAction,
    CombatOutcome, CurriculumSchedule, DominanceShaper, InverseActionPair, RewardComponents,
    RewardConfig, Stage, TRANSPORT_UNIT_CLASS, V83_CLOSEOUT_SHARE_START, W_STR, W_WASTE,
    action_churn_penalty, boat_outcome_reward, classify_boat_resolution, closeout_potential,
    combat_outcome_reward, dominance_potential, embargo_stop_outcome_reward, fast_win_bonus,
    land_share, normalized_strength_share, placement, placement_score, sample_episode,
    sparse_terminal_reward, stages_for_schedule, strength_delta_weight, tempo_pressure,
    terminal_reward, timeweight, v83_action_churn_penalty,
};
use ofcore::feat::{
    self, A_ATTACK, A_BOAT, A_CANCEL_BOAT, A_EMBARGO, A_EMBARGO_STOP, A_RETREAT, ACTIONS,
    IS_LAND_BIT, MAG_MASK, REGION,
};
use ofcore::translate::{Choice, IntentTranslator, translate};

use crate::ae::{self, AeRaw, StaticTerrain, TerrainCacheKey};
use crate::engine::{self, EngineKind, GameEngine, RawObs};

/// CPU-owned foveated rollout payload. Grid samples cross the actor/learner
/// boundary as fp16 values; masks and crop metadata stay explicit so the
/// learner never has to reconstruct a full fine grid or infer coordinates.
#[derive(Default)]
pub(crate) struct CompactHostBuffers {
    pub grids: Vec<half::f16>,
    pub masks: Vec<f32>,
    pub origins: Vec<i64>,
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

    #[cfg(test)]
    pub(crate) fn grid_storage_ptr(&self) -> *const half::f16 {
        self.payload.buffers.grids.as_ptr()
    }

    #[cfg(test)]
    pub(crate) fn mask_storage_ptr(&self) -> *const f32 {
        self.payload.buffers.masks.as_ptr()
    }

    #[cfg(test)]
    pub(crate) fn storage_capacities(&self) -> (usize, usize, usize) {
        (
            self.payload.buffers.grids.capacity(),
            self.payload.buffers.masks.capacity(),
            self.payload.buffers.origins.capacity(),
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

    fn observe(&mut self, share: f64, tick: i64, inverse_pair: bool) {
        self.max_land_share = self.max_land_share.max(share);
        if self.entry_tick.is_some() {
            self.decisions_after_entry += 1;
        } else if share >= V83_CLOSEOUT_SHARE_START {
            self.entry_tick = Some(tick);
        }
        if self.entry_tick.is_some() && inverse_pair {
            self.post_entry_churn_pairs += 1;
        }
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
/// `C_GRID = 89 = latent(32) + ego(3) + db(1) + transient(53)`.
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
    /// Transient planes at /8: (53, gh, gw).
    pub transient: Vec<f32>,
    pub legal_tile: Vec<f32>, // (gh, gw)
    pub gh: usize,
    pub gw: usize,
    pub players: Vec<f32>, // (MAX_SLOTS, P_FEAT)
    pub pmask: [f32; feat::MAX_SLOTS],
    pub scalars: [f32; feat::N_SCALARS],
    pub me_slot: i64,
    pub legal_actions: [f32; feat::N_ACTIONS],
    pub legal_ptarget: Vec<f32>, // (N_ACTIONS, MAX_SLOTS)
    pub legal_build: [f32; feat::N_BUILD],
    pub legal_nuke: [f32; feat::N_NUKE],
    pub local: Vec<f32>, // (5, LOCAL, LOCAL)
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
            // First open only — reinforcing the same target must not refresh
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
        A_ATTACK | A_EMBARGO | A_EMBARGO_STOP if !intents.is_empty() => {
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
    lut: Vec<u8>,
    translator: Option<IntentTranslator>,
    land_total: i64,
    prev_strength: f64,
    dominance_shaper: DominanceShaper,
    closeout_shaper: DominanceShaper,
    closeout_tracker: CloseoutTracker,
    action_churn_tracker: ActionChurnTracker,
    boat_tracker: PendingBoatTracker,
    combat_tracker: CombatStickyTracker,
    prev_action: ActionOutcome,
    last_commitment: Option<(i64, i64, i64, i64, i64, u64)>,
    ep_reward_components: RewardComponents,
    spawn_steps: i64,
    map_name: String,
    rehearsal: bool,
    hr: usize,
    wr: usize,
    land: Vec<u8>,
    mag: Vec<u8>,
    ae_static: StaticTerrain,
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
    ) -> Result<Self> {
        let bridge = engine::create(engine)?;
        let mut w = EnvWorker {
            idx,
            bridge,
            stages: stages_for_schedule(curriculum_schedule),
            curriculum_schedule,
            stage,
            episode_stage: stage,
            max_episode_ticks,
            reward_config,
            decision_ticks: 10,
            rng: SmallRng::seed_from_u64(1000 + idx as u64),
            episode: 0,
            ep_reward: 0.0,
            ep_len: 0,
            ep_wasted: 0,
            obs: None,
            lut: Vec::new(),
            translator: None,
            land_total: 1,
            prev_strength: 0.0,
            dominance_shaper: DominanceShaper::default(),
            closeout_shaper: DominanceShaper::default(),
            closeout_tracker: CloseoutTracker::default(),
            action_churn_tracker: ActionChurnTracker::default(),
            boat_tracker: PendingBoatTracker::default(),
            combat_tracker: CombatStickyTracker::default(),
            prev_action: ActionOutcome::default(),
            last_commitment: None,
            ep_reward_components: RewardComponents::default(),
            spawn_steps: 0,
            map_name: String::new(),
            rehearsal: false,
            hr: 0,
            wr: 0,
            land: Vec::new(),
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
        // rl.watch always steps 10 ticks per decision, not stage.decision_ticks.
        self.decision_ticks = 10;
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
        let initial_strengths =
            curriculum::strengths(&feat::parse_ents(obs.entities()), self.land_total);
        let me0 = obs.me().max(0) as usize;
        // Seed prev_strength so the first post-spawn delta isn't a free
        // +W_DELTA_GAIN * share windfall from 0.0.
        self.prev_strength = initial_strengths.get(&me0).copied().unwrap_or(0.0);
        self.dominance_shaper.reset(dominance_potential(
            &initial_strengths,
            me0,
            self.reward_config.v81_potential_clamp,
        ));
        let initial_share = land_share(
            ofcore::translate::my_tiles(&feat::parse_ents(obs.entities()), obs.me()),
            self.land_total,
        );
        self.closeout_shaper
            .reset(closeout_potential(initial_share));
        self.closeout_tracker.reset(initial_share, obs.tick());
        self.spawn_steps = 0;
        self.ep_reward = 0.0;
        self.action_churn_tracker.reset();
        self.boat_tracker.reset();
        self.combat_tracker.reset();
        self.ep_reward_components = RewardComponents::default();
        self.ep_len = 0;
        self.ep_wasted = 0;
        self.episode += 1;
        self.obs = Some(obs);
        Ok(())
    }

    pub fn save_record(&mut self, path: &str) -> Result<serde_json::Value> {
        self.bridge.save_record(path)
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
        let obs = self.obs.as_ref().unwrap();
        let ents = feat::parse_ents(obs.entities());
        let legal = feat::parse_legal(obs.legal_actions());
        let width = obs.head["width"].as_u64().unwrap() as usize;
        let mut owners_trim = vec![0i64; self.hr * self.wr];
        for y in 0..self.hr {
            for x in 0..self.wr {
                owners_trim[y * self.wr + x] = obs.owner_at(y * width + x) as i64;
            }
        }
        let me = obs.me();
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
        self.obs = Some(new_obs);
        Ok(())
    }

    pub fn reset_episode(&mut self) -> Result<()> {
        self.episode_stage = self.stage;
        let stg = &self.stages[self.stage];
        self.decision_ticks = stg.decision_ticks;
        let (map_name, bots, difficulty, nations, rehearsal) =
            sample_episode(&self.stages, self.stage, &mut self.rng);
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
        let initial_strengths =
            curriculum::strengths(&feat::parse_ents(obs.entities()), self.land_total);
        let me0 = obs.me().max(0) as usize;
        self.prev_strength = initial_strengths.get(&me0).copied().unwrap_or(0.0);
        self.dominance_shaper.reset(dominance_potential(
            &initial_strengths,
            me0,
            self.reward_config.v81_potential_clamp,
        ));
        let initial_share = land_share(
            ofcore::translate::my_tiles(&feat::parse_ents(obs.entities()), obs.me()),
            self.land_total,
        );
        self.closeout_shaper
            .reset(closeout_potential(initial_share));
        self.closeout_tracker.reset(initial_share, obs.tick());
        self.spawn_steps = 0;
        self.ep_reward = 0.0;
        self.action_churn_tracker.reset();
        self.boat_tracker.reset();
        self.combat_tracker.reset();
        self.prev_action = ActionOutcome::default();
        self.last_commitment = None;
        self.ep_reward_components = RewardComponents::default();
        self.ep_len = 0;
        self.ep_wasted = 0;
        self.episode += 1;
        self.obs = Some(obs);
        Ok(())
    }

    fn current_lut(&mut self) -> Vec<u8> {
        let ents = feat::parse_ents(self.obs.as_ref().unwrap().entities());
        let spawn_phase = self.obs.as_ref().unwrap().spawn_phase();
        // Mirrors ObsBuilder._slot_lut: rebuild every tick during spawn
        // (roster still filling in), freeze on first post-spawn obs.
        if spawn_phase || self.lut.is_empty() {
            let ids: Vec<usize> = ents.players.iter().map(|p| p.id).collect();
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
        let lut = self.current_lut();
        let obs = self.obs.as_ref().unwrap();
        let ents = feat::parse_ents(obs.entities());
        let legal = feat::parse_legal(obs.legal_actions());
        let (hr, wr) = (self.hr, self.wr);
        let (gh, gw) = (hr / REGION, wr / REGION);

        let width = obs.head["width"].as_u64().unwrap_or(wr as u64) as usize;
        let tiles = obs.prepare_tiles(&lut, width, hr, wr, REGION);
        let owners_slotted = tiles.owners_slotted;

        let f = feat::featurize(
            gh,
            gw,
            &lut,
            &self.land,
            &self.mag,
            &owners_slotted,
            obs.tick(),
            obs.spawn_phase(),
            obs.alive(),
            obs.me(),
            &ents,
            &legal,
        );

        let (ego, center) = feat::pool_ego_and_center(&owners_slotted, &f.clut, hr, wr);
        let local = feat::local_crop_at_with_defense(
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
        );
        let ae_raw = AeRaw {
            owners: owners_slotted,
            static_terrain: self.ae_static.clone(),
            fallout: tiles.fallout_packed,
            stat: f.stat,
            hr,
            wr,
        };

        PreparedObs {
            prev_action: self.prev_action.clone(),
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
            scalars: f.scalars,
            me_slot: f.me_slot,
            legal_actions: f.legal_actions,
            legal_ptarget: f.legal_ptarget,
            legal_build: f.legal_build,
            legal_nuke: f.legal_nuke,
            local,
        }
    }

    /// Combined apply-then-prepare, matching a Gym-style `env.step()`:
    /// returns the NEXT observation alongside the reward/done/info from
    /// applying `choice` to the current one. Drives the threaded rollout
    /// loop in `train.rs`.
    pub fn step(&mut self, choice: &Choice) -> Result<EnvTransition> {
        let (reward, done, info, outcome) = self.apply(choice)?;
        let prepared = self.prepare();
        Ok(EnvTransition {
            next_obs: prepared,
            reward,
            done,
            info,
            outcome,
        })
    }

    /// Translate + step. Auto-resets on episode end.
    pub fn apply(
        &mut self,
        choice: &Choice,
    ) -> Result<(f64, bool, Option<EpisodeInfo>, ActionOutcome)> {
        let name = ACTIONS[choice.action as usize];
        let lut = self.current_lut();
        let obs = self.obs.as_ref().unwrap();
        let ents = feat::parse_ents(obs.entities());
        let legal = feat::parse_legal(obs.legal_actions());
        let boats_before = legal.boats.clone();
        let pre_attack_ids: HashSet<String> = ents.attacks.iter().map(|a| a.aid.clone()).collect();
        let me_pre = obs.me().max(0) as usize;
        let troops_before = player_troops(&ents, me_pre);
        // Raw tiles are full (untrimmed width) resolution; translate wants
        // owner ids trimmed to (hr, wr) matching the translator's grids.
        let width = obs.head["width"].as_u64().unwrap() as usize;
        let mut owners_trim = vec![0i64; self.hr * self.wr];
        for y in 0..self.hr {
            for x in 0..self.wr {
                owners_trim[y * self.wr + x] = obs.owner_at(y * width + x) as i64;
            }
        }
        let me = obs.me();
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
        let boats_after = if choice.action == A_BOAT {
            feat::parse_legal(new_obs.legal_actions()).boats
        } else {
            Vec::new()
        };
        let chosen_action =
            churn_action(choice, &lut, &ents, &intents, &boats_before, &boats_after);

        // Snapshot relation before the intent lands (embargo_stop outcome).
        let mut embargo_outcome_r = 0.0;
        if choice.action == A_EMBARGO_STOP {
            if let Some(ActionTarget::Player(target)) = chosen_action.target {
                let rel = ents
                    .players
                    .iter()
                    .find(|p| p.id == me_pre)
                    .map(|p| p.relation_to(target))
                    .unwrap_or(0.0);
                embargo_outcome_r = self.combat_tracker.observe_embargo_stop(
                    rel,
                    self.reward_config,
                    self.episode_stage,
                );
            }
        }
        let combat_target = match chosen_action.target {
            Some(ActionTarget::Player(id)) if choice.action == A_ATTACK || choice.action == A_RETREAT => {
                Some(id)
            }
            _ => None,
        };
        let combat_outcome_r = self.combat_tracker.observe_combat(
            choice.action,
            combat_target,
            self.reward_config.v81_churn_window,
            self.reward_config,
            self.episode_stage,
        );

        if let Some(ActionTarget::Unit(unit_id)) = chosen_action.target {
            if choice.action == A_BOAT && !intents.is_empty() {
                let after_ents = feat::parse_ents(new_obs.entities());
                let troops = after_ents
                    .units
                    .iter()
                    .find(|u| u.uid == unit_id as i64 && u.class == TRANSPORT_UNIT_CLASS)
                    .map(|u| u.troops)
                    .unwrap_or(0.0);
                self.boat_tracker.note_launch(unit_id, troops);
            } else if choice.action == A_CANCEL_BOAT {
                self.boat_tracker.note_cancel(unit_id);
            }
        }
        let inverse_pair = self
            .action_churn_tracker
            .observe(chosen_action, self.reward_config.v81_churn_window);
        let engine_wasted = new_obs.wasted();
        let mut wasted = engine_wasted;
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
        let commitment_age = if self.last_commitment == Some(commitment) {
            self.prev_action.commitment_age.saturating_add(1)
        } else {
            0
        };
        self.last_commitment = Some(commitment);
        let outcome = ActionOutcome {
            action: choice.action,
            player_slot: choice.player_slot.unwrap_or(-1),
            tile_region: choice.tile_region.unwrap_or(-1),
            build_type: choice.build_type.unwrap_or(-1),
            nuke_type: choice.nuke_type.unwrap_or(-1),
            // The engine reports a count, so a multi-intent action can have
            // both accepted and wasted components.
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
        self.prev_action = outcome.clone();
        self.obs = Some(new_obs);
        let obs = self.obs.as_ref().unwrap();

        if obs.spawn_phase() {
            self.spawn_steps += 1;
            if self.spawn_steps >= 8 {
                // Fallback: pick a uniformly random legal spawn tile.
                self.spawn_randomly()?;
            } else {
                let next_ents = feat::parse_ents(obs.entities());
                let composite = curriculum::strengths(&next_ents, self.land_total);
                let me0 = obs.me().max(0) as usize;
                self.prev_strength = composite.get(&me0).copied().unwrap_or(0.0);
                self.dominance_shaper.reset(dominance_potential(
                    &composite,
                    me0,
                    self.reward_config.v81_potential_clamp,
                ));
                let share = land_share(
                    ofcore::translate::my_tiles(&next_ents, obs.me()),
                    self.land_total,
                );
                self.closeout_shaper.reset(closeout_potential(share));
                self.ep_len += 1;
                return Ok((0.0, false, None, outcome));
            }
        }

        let obs = self.obs.as_ref().unwrap();
        let ents = feat::parse_ents(obs.entities());
        let tiles = ofcore::translate::my_tiles(&ents, obs.me());
        let share = land_share(tiles, self.land_total);
        self.closeout_tracker
            .observe(share, obs.tick(), inverse_pair.is_some());
        let composite = curriculum::strengths(&ents, self.land_total);
        let me = obs.me().max(0) as usize;
        let mut done = false;
        let mut won = false;
        let mut timed_out = false;
        if !obs.alive() {
            done = true;
        } else if !obs.winner().is_null() {
            let w = obs.winner();
            won = w
                .as_array()
                .map(|a| a.len() > 1 && a[1] == "AGENTRL1")
                .unwrap_or(false);
            done = true;
        } else if obs.tick() >= self.max_episode_ticks {
            done = true;
            timed_out = true;
        }
        let mine = composite
            .get(&(obs.me().max(0) as usize))
            .copied()
            .unwrap_or(0.0);

        // V9 sparse path: only terminal ±1. No strength/delta/waste/death/place
        // or V8.x shaping: environment curriculum carries the learning signal.
        if self.reward_config.v9_sparse_win {
            self.prev_strength = mine;
            let mut components = RewardComponents::default();
            let mut reward = 0.0;
            let mut info = None;
            if done {
                reward = sparse_terminal_reward(won);
                components.terminal = reward;
                let (place, n) = placement(&ents, obs.me(), obs.alive(), self.land_total);
                self.ep_reward_components.add_assign(components);
                self.ep_reward += reward;
                self.ep_len += 1;
                info = Some(EpisodeInfo {
                    reward: self.ep_reward,
                    length: self.ep_len,
                    final_tiles: tiles,
                    final_land_share: share,
                    max_land_share: self.closeout_tracker.max_land_share,
                    closeout_reached: self.closeout_tracker.reached(),
                    closeout_entry_tick: self.closeout_tracker.entry_tick,
                    decisions_after_closeout: self.closeout_tracker.decisions_after_entry,
                    converted: self.closeout_tracker.reached() && won,
                    timeout_after_closeout: timed_out && self.closeout_tracker.reached(),
                    post_closeout_churn_pairs: self.closeout_tracker.post_entry_churn_pairs,
                    final_tick: obs.tick(),
                    place,
                    n_players: n,
                    score: placement_score(place, n),
                    won,
                    wasted: self.ep_wasted,
                    stage: self.stage,
                    map: self.map_name.clone(),
                    rehearsal: self.rehearsal,
                    reward_components: self.ep_reward_components,
                    action_pair_counts: self.action_churn_tracker.counts(),
                    boat_outcome_counts: self.boat_tracker.counts(),
                    embargo_bad_stops: self.combat_tracker.embargo_bad_stops,
                    embargo_good_stops: self.combat_tracker.embargo_good_stops,
                    premature_retreats: self.combat_tracker.premature_retreats,
                    thrash_reengages: self.combat_tracker.thrash_reengages,
                });
                self.reset_episode()?;
            } else {
                self.ep_len += 1;
            }
            return Ok((reward, done, info, outcome));
        }

        let tw = timeweight(obs.tick());
        let delta = mine - self.prev_strength;
        let normalized_share = if self.reward_config.dominant_loss_active(self.episode_stage) {
            normalized_strength_share(&composite, me)
        } else {
            0.0
        };
        let has_active_attack = ents.attacks.iter().any(|a| a.from == me);
        let delta_weight = strength_delta_weight(
            delta,
            normalized_share,
            self.episode_stage,
            self.reward_config,
            has_active_attack,
        );
        let mut components = RewardComponents {
            strength: W_STR * mine * tw,
            strength_delta: delta_weight * delta,
            ..RewardComponents::default()
        };
        let mut reward = components.strength + components.strength_delta;
        components.action_churn = if self.curriculum_schedule == CurriculumSchedule::V83 {
            v83_action_churn_penalty(inverse_pair, self.episode_stage, share, self.reward_config)
        } else {
            action_churn_penalty(inverse_pair, self.episode_stage, self.reward_config)
        };
        // V8.6: combat sticky outcomes already price attack↔retreat thrash;
        // stacking flat churn makes attacking a trap.
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
        components.embargo_outcome = embargo_outcome_r;
        if components.embargo_outcome != 0.0 {
            reward += components.embargo_outcome;
        }
        components.combat_outcome = combat_outcome_r;
        if components.combat_outcome != 0.0 {
            reward += components.combat_outcome;
        }
        let next_potential = if self.curriculum_schedule == CurriculumSchedule::V83 && done {
            0.0
        } else {
            dominance_potential(&composite, me, self.reward_config.v81_potential_clamp)
        };
        if self
            .reward_config
            .dominance_shaping_active(self.episode_stage)
        {
            components.dominance = self.dominance_shaper.transition(
                next_potential,
                self.reward_config.gamma,
                self.reward_config.v81_dom_coef,
            );
            reward += components.dominance;
        } else {
            // Avoid even adding zero in the disabled legacy-parity path.
            self.dominance_shaper.reset(next_potential);
        }
        let next_closeout_potential = if done { 0.0 } else { closeout_potential(share) };
        if self.curriculum_schedule == CurriculumSchedule::V83 && self.episode_stage >= 5 {
            components.closeout = self.closeout_shaper.transition(
                next_closeout_potential,
                self.reward_config.gamma,
                self.reward_config.v83_close_coef,
            );
            reward += components.closeout;
        } else {
            self.closeout_shaper.reset(next_closeout_potential);
        }

        let troops_after = player_troops(&ents, me);
        let alive_transports = transport_unit_ids(&ents, me);
        let new_sourced_attack = ents.attacks.iter().any(|a| {
            a.from == me
                && a.src_x.is_some()
                && a.src_y.is_some()
                && !pre_attack_ids.contains(&a.aid)
        });
        let has_sourced_attack = ents
            .attacks
            .iter()
            .any(|a| a.from == me && a.src_x.is_some() && a.src_y.is_some());
        components.boat_outcome = self.boat_tracker.resolve_missing(
            &alive_transports,
            troops_before,
            troops_after,
            new_sourced_attack,
            has_sourced_attack,
            self.reward_config,
            self.episode_stage,
        );
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
                    obs.tick(),
                    self.max_episode_ticks,
                    tempo_share,
                    self.reward_config.tempo_share_threshold(),
                )
                * tw;
            if components.tempo != 0.0 {
                reward += components.tempo;
            }
        }

        reward -= W_WASTE * wasted as f64;
        components.waste = -W_WASTE * wasted as f64;
        self.ep_wasted += wasted;
        self.prev_strength = mine;

        if !obs.alive() {
            let death = self.reward_config.death_penalty();
            reward -= death;
            components.death = -death;
        }

        let mut info = None;
        if done {
            let (place, n) = placement(&ents, obs.me(), obs.alive(), self.land_total);
            components.terminal = terminal_reward(place, won)
                + fast_win_bonus(
                    won,
                    obs.tick(),
                    self.max_episode_ticks,
                    self.reward_config.v84_fast_win_coef,
                );
            if won {
                components.terminal += self.reward_config.v85_extra_win_bonus;
            }
            reward += components.terminal;
            self.ep_reward_components.add_assign(components);
            self.ep_reward += reward;
            self.ep_len += 1;
            info = Some(EpisodeInfo {
                reward: self.ep_reward,
                length: self.ep_len,
                final_tiles: tiles,
                final_land_share: share,
                max_land_share: self.closeout_tracker.max_land_share,
                closeout_reached: self.closeout_tracker.reached(),
                closeout_entry_tick: self.closeout_tracker.entry_tick,
                decisions_after_closeout: self.closeout_tracker.decisions_after_entry,
                converted: self.closeout_tracker.reached() && won,
                timeout_after_closeout: timed_out && self.closeout_tracker.reached(),
                post_closeout_churn_pairs: self.closeout_tracker.post_entry_churn_pairs,
                final_tick: obs.tick(),
                place,
                n_players: n,
                score: placement_score(place, n),
                won,
                wasted: self.ep_wasted,
                stage: self.stage,
                map: self.map_name.clone(),
                rehearsal: self.rehearsal,
                reward_components: self.ep_reward_components,
                action_pair_counts: self.action_churn_tracker.counts(),
                boat_outcome_counts: self.boat_tracker.counts(),
                embargo_bad_stops: self.combat_tracker.embargo_bad_stops,
                embargo_good_stops: self.combat_tracker.embargo_good_stops,
                premature_retreats: self.combat_tracker.premature_retreats,
                thrash_reengages: self.combat_tracker.thrash_reengages,
            });
            self.reset_episode()?;
        } else {
            self.ep_reward_components.add_assign(components);
            self.ep_reward += reward;
            self.ep_len += 1;
        }
        Ok((reward, done, info, outcome))
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
        let (y, x) = candidates[self.rng.gen_range(0..candidates.len())];
        let tile = y * width as i64 + x;
        let new_obs = self.bridge.step(
            &[serde_json::json!({"type": "spawn", "tile": tile})],
            self.decision_ticks,
        )?;
        self.obs = Some(new_obs);
        Ok(())
    }

    pub fn set_stage(&mut self, stage: usize) {
        self.stage = stage;
    }

    pub fn close(&mut self) {
        self.bridge.close();
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
        tracker.observe(0.44, 200, true);
        assert!(!tracker.reached());
        assert_eq!(tracker.post_entry_churn_pairs, 0);

        tracker.observe(0.45, 300, false);
        tracker.observe(0.62, 400, true);
        tracker.observe(0.50, 500, false);
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
