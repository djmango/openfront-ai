//! Curriculum stages and the strength-based reward. Port of
//! `rl/curriculum.py`; see that file for the design rationale in comments.

use crate::feat::{
    A_ALLIANCE_REQUEST, A_ATTACK, A_BOAT, A_BREAK_ALLIANCE, A_BUILD, A_CANCEL_BOAT,
    A_DONATE_GOLD, A_DONATE_TROOPS, A_EMBARGO, A_EMBARGO_STOP, A_RETREAT, EntsData,
};
use std::collections::{HashMap, VecDeque};

pub const W_STR: f64 = 0.02;
pub const W_DELTA_GAIN: f64 = 5.0;
pub const W_DELTA_LOSS: f64 = 6.5;
pub const W_PLACE: f64 = 15.0;
pub const W_WIN: f64 = 30.0;
pub const W_DEATH: f64 = 1.0;
pub const W_WASTE: f64 = 0.01;
pub const PLACE_POW: f64 = 1.5;

pub const K_LAND: f64 = 0.40;
pub const K_MIL: f64 = 0.20;
pub const K_ECO: f64 = 0.25;
pub const K_BUILD: f64 = 0.15;

pub const DOMINANCE_EPS: f64 = 1e-9;
pub const V83_CLOSEOUT_SHARE_START: f64 = 0.45;
pub const V83_CLOSEOUT_SHARE_FULL: f64 = 0.80;
pub const V83_REWARD_PROFILE: &str = "v8.3-closeout-v1";
pub const V84_REWARD_PROFILE: &str = "v8.4-boat-tempo-v1";
pub const V85_REWARD_PROFILE: &str = "v8.5-win-urgency-v1";
pub const V86_REWARD_PROFILE: &str = "v8.6-attack-fair-v1";
/// Parallel sparse experiment: terminal win/loss only (±1), no dense shaping.
pub const V9_REWARD_PROFILE: &str = "v9-sparse-win-v1";
pub const V9_SPARSE_WIN: f64 = 1.0;
pub const V9_SPARSE_LOSS: f64 = -1.0;
/// V10 anti-death-spiral: dense V8.6-like reward + survival / anti-diplo / combat priors.
pub const V10_REWARD_PROFILE: &str = "v10-anti-spiral-v1";
/// Soft death default for V10 launches (override via `--v86-death-penalty`).
pub const V10_DEFAULT_DEATH_PENALTY: f64 = 3.0;
/// Rolling death-rate ceiling required before a V10 stage advance.
pub const V10_ADVANCE_MAX_DEATH_RATE: f64 = 0.55;
/// Demote when window win-rate is below this and death-rate is above
/// [`V10_DEMOTE_MIN_DEATH_RATE`].
pub const V10_DEMOTE_MAX_WIN_RATE: f64 = 0.10;
pub const V10_DEMOTE_MIN_DEATH_RATE: f64 = 0.85;
/// Longer LR warmup after V10 stage changes (advance or demote).
pub const V10_LR_WARMUP_UPDATES: u64 = 200;
/// Relation score bands mirror engine `Relation` / TS `PlayerImpl.relation()`.
pub const RELATION_HOSTILE_LT: f64 = -50.0;
pub const RELATION_DISTRUSTFUL_LT: f64 = 0.0;
pub const RELATION_NEUTRAL_LT: f64 = 50.0;
/// `feat::unit_class("Transport")`.
pub const TRANSPORT_UNIT_CLASS: usize = 7;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RewardConfig {
    pub gamma: f64,
    pub v81_dom_coef: f64,
    pub v81_min_stage: usize,
    pub v81_potential_clamp: f64,
    pub v81_dominant_loss: bool,
    pub v81_dominance_threshold: f64,
    pub v81_delta_loss_dominant: f64,
    pub v81_churn_coef: f64,
    pub v81_churn_window: usize,
    pub v81_churn_min_stage: usize,
    pub v83_close_coef: f64,
    pub v83_churn_coef: f64,
    /// Reward when a boat resolves into a sourced land attack (enemy or TN).
    pub v84_boat_useful: f64,
    /// Penalty when a boat is destroyed without landing an attack.
    pub v84_boat_destroyed: f64,
    /// Mild penalty when a boat is cancelled (churn already covers the pair).
    pub v84_boat_cancelled: f64,
    /// Penalty when a boat returns to own shore without invading.
    pub v84_boat_own_shore: f64,
    pub v84_boat_min_stage: usize,
    /// Late-game tempo pressure while dominant (finish the win).
    pub v84_tempo_coef: f64,
    pub v84_tempo_min_stage: usize,
    /// Extra terminal bonus for faster wins: coef * (1 - tick/max_ticks).
    pub v84_fast_win_coef: f64,
    /// V8.5: land/strength share threshold for tempo (0 = use v81_dominance_threshold).
    pub v85_tempo_share_threshold: f64,
    /// Extra terminal win bonus on top of W_WIN (win must dominate shaping).
    pub v85_extra_win_bonus: f64,
    /// Penalty for embargo_stop while still Hostile/Distrustful.
    pub v85_embargo_bad_stop: f64,
    /// Small reward for embargo_stop after relation recovered (Neutral+).
    pub v85_embargo_good_stop: f64,
    pub v85_embargo_min_stage: usize,
    /// Penalty for retreating an attack that was just opened on the same target.
    pub v85_premature_retreat: f64,
    /// Penalty for re-attacking a target just after retreating.
    pub v85_thrash_reengage: f64,
    pub v85_combat_min_stage: usize,
    /// V8.6: override `W_DELTA_LOSS` when > 0 (soften attack variance tax).
    pub v86_delta_loss: f64,
    /// V8.6: while the agent has an open attack, price losses like gains.
    pub v86_attack_symmetric_loss: bool,
    /// V8.6: do not stack flat attack↔retreat churn on top of combat outcomes.
    pub v86_skip_combat_churn: bool,
    /// V8.6: override `W_DEATH` when > 0 (death must hurt more than mid-place).
    pub v86_death_penalty: f64,
    /// V9: terminal win/loss only (`+1` / `-1`); disables all dense shaping.
    pub v9_sparse_win: bool,
    /// V10: per-decision survival shaping while alive (`coef * land_share`).
    pub v10_survival_coef: f64,
    /// V10: penalty magnitude for late/dominant diplo/donate panic actions.
    pub v10_diplo_panic: f64,
    /// V10: land-share threshold that arms diplo-panic shaping.
    pub v10_diplo_panic_share: f64,
    /// V10: tick/max_ticks fraction that arms diplo-panic shaping.
    pub v10_diplo_panic_tick_frac: f64,
    /// V10: bonus for productive combat/build/boat actions.
    pub v10_combat_action: f64,
}

impl RewardConfig {
    pub fn v81_active(self, stage: usize) -> bool {
        stage >= self.v81_min_stage
    }

    pub fn dominance_shaping_active(self, stage: usize) -> bool {
        self.v81_active(stage) && self.v81_dom_coef != 0.0
    }

    pub fn dominant_loss_active(self, stage: usize) -> bool {
        self.v81_active(stage) && self.v81_dominant_loss
    }

    pub fn churn_penalty_active(self, stage: usize) -> bool {
        stage >= self.v81_churn_min_stage
            && self.v81_churn_coef != 0.0
            && self.v81_churn_window != 0
    }

    pub fn boat_outcome_active(self, stage: usize) -> bool {
        stage >= self.v84_boat_min_stage
            && (self.v84_boat_useful != 0.0
                || self.v84_boat_destroyed != 0.0
                || self.v84_boat_cancelled != 0.0
                || self.v84_boat_own_shore != 0.0)
    }

    pub fn tempo_active(self, stage: usize) -> bool {
        stage >= self.v84_tempo_min_stage && self.v84_tempo_coef != 0.0
    }

    pub fn v84_reward_active(self) -> bool {
        self.v84_boat_useful != 0.0
            || self.v84_boat_destroyed != 0.0
            || self.v84_boat_cancelled != 0.0
            || self.v84_boat_own_shore != 0.0
            || self.v84_tempo_coef != 0.0
            || self.v84_fast_win_coef != 0.0
    }

    pub fn tempo_share_threshold(self) -> f64 {
        if self.v85_tempo_share_threshold > 0.0 {
            self.v85_tempo_share_threshold
        } else {
            self.v81_dominance_threshold
        }
    }

    pub fn embargo_outcome_active(self, stage: usize) -> bool {
        stage >= self.v85_embargo_min_stage
            && (self.v85_embargo_bad_stop != 0.0 || self.v85_embargo_good_stop != 0.0)
    }

    pub fn combat_outcome_active(self, stage: usize) -> bool {
        stage >= self.v85_combat_min_stage
            && (self.v85_premature_retreat != 0.0 || self.v85_thrash_reengage != 0.0)
    }

    pub fn v85_reward_active(self) -> bool {
        self.v85_tempo_share_threshold > 0.0
            || self.v85_extra_win_bonus != 0.0
            || self.v85_embargo_bad_stop != 0.0
            || self.v85_embargo_good_stop != 0.0
            || self.v85_premature_retreat != 0.0
            || self.v85_thrash_reengage != 0.0
    }

    pub fn v86_reward_active(self) -> bool {
        self.v86_delta_loss > 0.0
            || self.v86_attack_symmetric_loss
            || self.v86_skip_combat_churn
            || self.v86_death_penalty > 0.0
    }

    pub fn v10_reward_active(self) -> bool {
        self.v10_survival_coef != 0.0
            || self.v10_diplo_panic != 0.0
            || self.v10_combat_action != 0.0
    }

    /// V8.4 boat/tempo knobs and/or V8.5 win-urgency knobs.
    pub fn v84_or_v85_reward_active(self) -> bool {
        self.v84_reward_active() || self.v85_reward_active()
    }

    pub fn delta_loss(self) -> f64 {
        if self.v86_delta_loss > 0.0 {
            self.v86_delta_loss
        } else {
            W_DELTA_LOSS
        }
    }

    pub fn death_penalty(self) -> f64 {
        if self.v86_death_penalty > 0.0 {
            self.v86_death_penalty
        } else {
            W_DEATH
        }
    }

    /// Active V8.3+ / V9 / V10 reward profile id for TrainState sidecars.
    pub fn reward_profile_id(self) -> &'static str {
        if self.v9_sparse_win {
            V9_REWARD_PROFILE
        } else if self.v10_reward_active() {
            V10_REWARD_PROFILE
        } else if self.v86_reward_active() {
            V86_REWARD_PROFILE
        } else if self.v85_reward_active() {
            V85_REWARD_PROFILE
        } else if self.v84_reward_active() {
            V84_REWARD_PROFILE
        } else {
            V83_REWARD_PROFILE
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RewardComponents {
    pub strength: f64,
    pub strength_delta: f64,
    pub dominance: f64,
    pub closeout: f64,
    pub action_churn: f64,
    pub boat_outcome: f64,
    pub tempo: f64,
    pub embargo_outcome: f64,
    pub combat_outcome: f64,
    pub survival: f64,
    pub diplo_panic: f64,
    pub combat_action: f64,
    pub waste: f64,
    pub death: f64,
    pub terminal: f64,
}

impl RewardComponents {
    pub fn add_assign(&mut self, other: Self) {
        self.strength += other.strength;
        self.strength_delta += other.strength_delta;
        self.dominance += other.dominance;
        self.closeout += other.closeout;
        self.action_churn += other.action_churn;
        self.boat_outcome += other.boat_outcome;
        self.tempo += other.tempo;
        self.embargo_outcome += other.embargo_outcome;
        self.combat_outcome += other.combat_outcome;
        self.survival += other.survival;
        self.diplo_panic += other.diplo_panic;
        self.combat_action += other.combat_action;
        self.waste += other.waste;
        self.death += other.death;
        self.terminal += other.terminal;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoatOutcome {
    UsefulLanding,
    OwnShoreReturn,
    Cancelled,
    Destroyed,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BoatOutcomeCounts {
    pub useful_landing: u64,
    pub own_shore_return: u64,
    pub cancelled: u64,
    pub destroyed: u64,
}

impl BoatOutcomeCounts {
    pub fn record(&mut self, outcome: BoatOutcome) {
        match outcome {
            BoatOutcome::UsefulLanding => self.useful_landing += 1,
            BoatOutcome::OwnShoreReturn => self.own_shore_return += 1,
            BoatOutcome::Cancelled => self.cancelled += 1,
            BoatOutcome::Destroyed => self.destroyed += 1,
        }
    }

    pub fn total(self) -> u64 {
        self.useful_landing + self.own_shore_return + self.cancelled + self.destroyed
    }
}

/// Classify a resolved transport. Strength already counts fielded troops, so
/// this is a small categorical signal for the tile/action heads — not a
/// re-pricing of troop deltas.
///
/// `has_sourced_attack` covers landings that merge into an already-open
/// sourced attack (no *new* attack id): without it those resolve as
/// Destroyed because refund ≈ 0.
pub fn classify_boat_resolution(
    cancel_requested: bool,
    committed_troops: f64,
    troops_before: f64,
    troops_after: f64,
    new_sourced_attack: bool,
    has_sourced_attack: bool,
) -> BoatOutcome {
    if cancel_requested {
        return BoatOutcome::Cancelled;
    }
    let refund = troops_after - troops_before;
    if new_sourced_attack
        || (has_sourced_attack
            && !(committed_troops > 0.0 && refund >= 0.5 * committed_troops))
    {
        return BoatOutcome::UsefulLanding;
    }
    if committed_troops > 0.0 && refund >= 0.5 * committed_troops {
        return BoatOutcome::OwnShoreReturn;
    }
    BoatOutcome::Destroyed
}

pub fn boat_outcome_reward(outcome: BoatOutcome, config: RewardConfig) -> f64 {
    match outcome {
        BoatOutcome::UsefulLanding => config.v84_boat_useful,
        BoatOutcome::OwnShoreReturn => config.v84_boat_own_shore,
        BoatOutcome::Cancelled => config.v84_boat_cancelled,
        BoatOutcome::Destroyed => config.v84_boat_destroyed,
    }
}

/// Quadratic late-game pressure while already dominant: finish the win.
pub fn tempo_pressure(
    tick: i64,
    max_ticks: i64,
    normalized_share: f64,
    threshold: f64,
) -> f64 {
    if max_ticks <= 0 || normalized_share < threshold {
        return 0.0;
    }
    let late = (tick as f64 / max_ticks as f64).clamp(0.0, 1.0);
    late * late
}

/// Terminal bonus for winning earlier in the episode budget.
pub fn fast_win_bonus(won: bool, tick: i64, max_ticks: i64, coef: f64) -> f64 {
    if !won || coef == 0.0 || max_ticks <= 0 {
        return 0.0;
    }
    coef * (1.0 - (tick as f64 / max_ticks as f64).clamp(0.0, 1.0))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelationBand {
    Hostile,
    Distrustful,
    Neutral,
    Friendly,
}

pub fn relation_band(value: f64) -> RelationBand {
    if value < RELATION_HOSTILE_LT {
        RelationBand::Hostile
    } else if value < RELATION_DISTRUSTFUL_LT {
        RelationBand::Distrustful
    } else if value < RELATION_NEUTRAL_LT {
        RelationBand::Neutral
    } else {
        RelationBand::Friendly
    }
}

pub fn relation_is_hostileish(band: RelationBand) -> bool {
    matches!(band, RelationBand::Hostile | RelationBand::Distrustful)
}

/// Embargo-stop outcome vs current relation to the target (expert sticky Hostile).
pub fn embargo_stop_outcome_reward(relation_value: f64, config: RewardConfig) -> f64 {
    if relation_is_hostileish(relation_band(relation_value)) {
        config.v85_embargo_bad_stop
    } else {
        config.v85_embargo_good_stop
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CombatOutcome {
    PrematureRetreat,
    ThrashReengage,
}

pub fn combat_outcome_reward(outcome: CombatOutcome, config: RewardConfig) -> f64 {
    match outcome {
        CombatOutcome::PrematureRetreat => config.v85_premature_retreat,
        CombatOutcome::ThrashReengage => config.v85_thrash_reengage,
    }
}

/// Stable identifier relevant to deciding whether two chosen actions undo one
/// another. Player slots are converted to player ids and boat actions to the
/// newly created transport unit id by the environment before recording.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActionTarget {
    Player(usize),
    Unit(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChosenAction {
    pub action: i64,
    pub target: Option<ActionTarget>,
}

impl ChosenAction {
    pub const fn new(action: i64, target: Option<ActionTarget>) -> Self {
        Self { action, target }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InverseActionPair {
    BoatCancelBoat,
    EmbargoEmbargoStop,
    AttackRetreat,
    RetreatAttack,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ActionPairCounts {
    pub boat_cancel_boat: u64,
    pub embargo_embargo_stop: u64,
    pub attack_retreat: u64,
    pub retreat_attack: u64,
}

impl ActionPairCounts {
    pub fn record(&mut self, pair: InverseActionPair) {
        match pair {
            InverseActionPair::BoatCancelBoat => self.boat_cancel_boat += 1,
            InverseActionPair::EmbargoEmbargoStop => self.embargo_embargo_stop += 1,
            InverseActionPair::AttackRetreat => self.attack_retreat += 1,
            InverseActionPair::RetreatAttack => self.retreat_attack += 1,
        }
    }

    pub fn total(self) -> u64 {
        self.boat_cancel_boat
            + self.embargo_embargo_stop
            + self.attack_retreat
            + self.retreat_attack
    }
}

/// Per-environment history of actual policy choices. Every choice consumes one
/// position in the decision window, including noops and unrelated actions.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ActionChurnTracker {
    history: VecDeque<ChosenAction>,
    counts: ActionPairCounts,
}

impl ActionChurnTracker {
    pub fn reset(&mut self) {
        self.history.clear();
        self.counts = ActionPairCounts::default();
    }

    pub fn counts(&self) -> ActionPairCounts {
        self.counts
    }

    /// Record a chosen action and return a clear inverse pair, if any.
    ///
    /// A same-action/same-target record newer than the possible inverse means
    /// the current choice is a repeat, not another reversal.
    pub fn observe(&mut self, current: ChosenAction, window: usize) -> Option<InverseActionPair> {
        if window == 0 {
            self.history.clear();
            return None;
        }

        let mut reversal = None;
        for &previous in self.history.iter().rev() {
            if previous.action == current.action && previous.target == current.target {
                break;
            }
            if let Some(pair) = inverse_action_pair(previous, current) {
                reversal = Some(pair);
                break;
            }
        }

        self.history.push_back(current);
        while self.history.len() > window {
            self.history.pop_front();
        }
        if let Some(pair) = reversal {
            self.counts.record(pair);
        }
        reversal
    }
}

fn inverse_action_pair(previous: ChosenAction, current: ChosenAction) -> Option<InverseActionPair> {
    if previous.target.is_none() || previous.target != current.target {
        return None;
    }
    match (previous.action, current.action) {
        (A_BOAT, A_CANCEL_BOAT) => Some(InverseActionPair::BoatCancelBoat),
        (A_EMBARGO, A_EMBARGO_STOP) => Some(InverseActionPair::EmbargoEmbargoStop),
        (A_ATTACK, A_RETREAT) => Some(InverseActionPair::AttackRetreat),
        (A_RETREAT, A_ATTACK) => Some(InverseActionPair::RetreatAttack),
        _ => None,
    }
}

pub fn action_churn_penalty(
    reversal: Option<InverseActionPair>,
    stage: usize,
    config: RewardConfig,
) -> f64 {
    if reversal.is_some() && config.churn_penalty_active(stage) {
        -config.v81_churn_coef
    } else {
        0.0
    }
}

pub fn v83_action_churn_penalty(
    reversal: Option<InverseActionPair>,
    stage: usize,
    land_share: f64,
    config: RewardConfig,
) -> f64 {
    if reversal.is_some()
        && stage >= 5
        && land_share >= V83_CLOSEOUT_SHARE_START
        && config.v83_churn_coef != 0.0
        && config.v81_churn_window != 0
    {
        -config.v83_churn_coef
    } else {
        action_churn_penalty(reversal, stage, config)
    }
}

/// Per-environment Φ(current), reset at each new episode.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DominanceShaper {
    prior: f64,
}

impl DominanceShaper {
    pub fn reset(&mut self, initial: f64) {
        self.prior = finite_or_zero(initial);
    }

    pub fn prior(self) -> f64 {
        self.prior
    }

    pub fn transition(&mut self, next: f64, gamma: f64, coefficient: f64) -> f64 {
        let next = finite_or_zero(next);
        let increment = coefficient * (gamma * next - self.prior);
        self.prior = next;
        finite_or_zero(increment)
    }
}

/// V8.3 land-share closeout potential. `land_total` is fixed at reset.
pub fn land_share(agent_tiles: f64, land_total: i64) -> f64 {
    if !agent_tiles.is_finite() || land_total <= 0 {
        return 0.0;
    }
    (agent_tiles / land_total as f64).clamp(0.0, 1.0)
}

pub fn closeout_potential(share: f64) -> f64 {
    let x = ((finite_or_zero(share).clamp(0.0, 1.0) - V83_CLOSEOUT_SHARE_START)
        / (V83_CLOSEOUT_SHARE_FULL - V83_CLOSEOUT_SHARE_START))
        .clamp(0.0, 1.0);
    x * x
}

pub const WINDOW: usize = 40;
pub const REHEARSAL_P: f64 = 0.25;

pub fn struct_value(unit_type: &str) -> Option<f64> {
    Some(match unit_type {
        "City" => 1.0,
        "Port" => 1.0,
        "Factory" => 1.0,
        "Missile Silo" => 1.0,
        "Defense Post" => 0.25,
        "SAM Launcher" => 3.0,
        _ => return None,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nations {
    Default,
    Exact(u32),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage {
    pub maps: &'static [&'static str],
    pub bots: u32,
    pub difficulty: &'static str,
    pub nations: Nations,
    pub decision_ticks: u32,
    pub win_at: f64,
}

pub const ALL_MAPS: [&str; 7] = [
    "Onion",
    "Pangaea",
    "Caucasus",
    "BlackSea",
    "BetweenTwoSeas",
    "World",
    "Asia",
];

/// Smaller V8.2 bridge pool: production maps with varied terrain and naval
/// pressure, without introducing the largest world-scale maps all at once.
pub const V82_STAGE5_MAPS: [&str; 8] = [
    "Onion",
    "Pangaea",
    "Caucasus",
    "BlackSea",
    "BetweenTwoSeas",
    "Europe",
    "Scandinavia",
    "GreatLakes",
];

/// Broad V8.2 pool. These are `GameMapType` enum keys (the values accepted
/// by the Node bridge and normalized to asset-directory names by Rust).
pub const V82_MAPS: [&str; 16] = [
    "Onion",
    "Pangaea",
    "Caucasus",
    "BlackSea",
    "BetweenTwoSeas",
    "Europe",
    "Asia",
    "World",
    "NorthAmerica",
    "SouthAmerica",
    "Africa",
    "Australia",
    "EastAsia",
    "MiddleEast",
    "Scandinavia",
    "GreatLakes",
];

/// Stable curriculum identities persisted in trainer checkpoints.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CurriculumSchedule {
    #[default]
    Legacy,
    V81,
    V811,
    V82,
    V83,
    /// Parallel sparse-win ladder: many tiny steps, high graduation gates.
    V9,
    /// Anti-death-spiral: V8.3 closeout ladder + demote / death gates / softer density.
    V10,
}

impl CurriculumSchedule {
    pub const fn id(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::V81 => "v8.1",
            Self::V811 => "v8.1.1",
            Self::V82 => "v8.2",
            Self::V83 => "v8.3",
            Self::V9 => "v9",
            Self::V10 => "v10",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "legacy" => Some(Self::Legacy),
            "v8.1" => Some(Self::V81),
            "v8.1.1" => Some(Self::V811),
            "v8.2" => Some(Self::V82),
            "v8.3" => Some(Self::V83),
            "v9" => Some(Self::V9),
            "v10" => Some(Self::V10),
            _ => None,
        }
    }

    /// V8.3 closeout insert + potential/churn behavior (shared by V10).
    pub const fn uses_v83_closeout(self) -> bool {
        matches!(self, Self::V83 | Self::V10)
    }
}

/// V8.1 env workers per GPU/shard. Early maps are cheap enough to keep the
/// actor full; later, larger maps use fewer workers to bound host memory and
/// engine tail latency.
pub const V81_ENV_TARGETS: [usize; 11] = [24, 24, 24, 24, 24, 24, 12, 10, 8, 8, 8];

/// V8.1.1 keeps the new same-map Medium bridge at the small-map target, then
/// uses at most 12 envs/GPU once World/Asia enter the pool.
pub const V811_ENV_TARGETS: [usize; 12] = [24, 24, 24, 24, 24, 24, 24, 12, 10, 8, 8, 8];

/// V8.2 envs per GPU/shard. The eight-map bridge can sustain 16; the broad
/// pool uses 12/10/8 as player count and simulation cost increase. Stage 8
/// safely grows back to 12 when the player count resets from 80 to 30.
pub const V82_ENV_TARGETS: [usize; 14] = [24, 24, 24, 24, 24, 16, 12, 10, 12, 10, 8, 8, 8, 8];
pub const V83_ENV_TARGETS: [usize; 15] = [24, 24, 24, 24, 24, 24, 16, 12, 10, 12, 10, 8, 8, 8, 8];

/// `(bots, nations)` for each V8.3 stage. Ratios stay ~6–8× bots over
/// nations so maps with large manifests (World/Europe) never invert the
/// intended OpenFront mix under `Nations::Default`.
pub const V83_BOT_NATION_DENSITY: [(u32, u32); 15] = [
    (30, 5),   // 0 Onion Easy — normal small lobby
    (30, 5),   // 1 Onion Easy
    (40, 5),   // 2 Onion/Pangaea Easy
    (50, 6),   // 3 Pangaea/Caucasus Easy
    (80, 10),  // 4 three-map Easy
    (50, 6),   // 5 closeout Easy
    (80, 10),  // 6 eight-map Easy bridge
    (100, 12), // 7 broad Easy
    (120, 15), // 8 broad Easy
    (100, 12), // 9 Medium (was 30 bots + Default — inverted on World)
    (120, 15), // 10 Medium
    (150, 20), // 11 Medium
    (150, 20), // 12 Hard
    (180, 25), // 13 Hard
    (200, 30), // 14 Impossible — normal large lobby
];

fn apply_v83_bot_heavy_density(stages: &mut [Stage]) {
    debug_assert_eq!(stages.len(), V83_BOT_NATION_DENSITY.len());
    for (stage, &(bots, nations)) in stages.iter_mut().zip(V83_BOT_NATION_DENSITY.iter()) {
        stage.bots = bots;
        stage.nations = Nations::Exact(nations);
    }
}

/// V10 softens the mid-ladder density cliffs that triggered stage-8 slaughter
/// under V8.3 (especially 100/12 → 120/15), while keeping bots ≫ nations.
pub const V10_BOT_NATION_DENSITY: [(u32, u32); 15] = [
    (30, 5),   // 0 Onion Easy
    (30, 5),   // 1 Onion Easy
    (40, 5),   // 2 Onion/Pangaea Easy
    (50, 6),   // 3 Pangaea/Caucasus Easy
    (70, 9),   // 4 three-map Easy (was 80/10)
    (50, 6),   // 5 closeout Easy
    (70, 9),   // 6 eight-map Easy bridge (was 80/10)
    (90, 11),  // 7 broad Easy (was 100/12)
    (100, 12), // 8 broad Easy (was 120/15 — main cliff)
    (110, 14), // 9 Medium — do not drop bots when difficulty rises
    (120, 15), // 10 Medium
    (140, 18), // 11 Medium
    (150, 20), // 12 Hard
    (170, 24), // 13 Hard
    (200, 30), // 14 Impossible
];

fn apply_v10_bot_heavy_density(stages: &mut [Stage]) {
    debug_assert_eq!(stages.len(), V10_BOT_NATION_DENSITY.len());
    for (stage, &(bots, nations)) in stages.iter_mut().zip(V10_BOT_NATION_DENSITY.iter()) {
        stage.bots = bots;
        stage.nations = Nations::Exact(nations);
    }
}

/// V10 reuses V8.3 env floors (same 15-stage closeout ladder geometry).
pub const V10_ENV_TARGETS: [usize; 15] = V83_ENV_TARGETS;

/// V9 envs/GPU. Early Onion bot-food maps stay saturated; later 200–350-bot
/// broad-pool stages drop toward 8 like V8.3 as sim cost grows.
pub const V9_ENV_TARGETS: [usize; 25] = [
    24, 24, 24, 24, 24, // 0-4 bots-only → 30/5
    24, 24, 24, 20, 20, // 5-9 map bridge (50→100 bots)
    16, 16, 12, 12, 10, // 10-14 climb toward 200/30
    10, 10, 8, 8, // 15-18 Easy 200→250 bots
    10, 8, 8, // 19-21 Medium
    8, 8, 8, // 22-24 Hard / Impossible
];

pub fn stages() -> Vec<Stage> {
    stages_for_schedule(CurriculumSchedule::Legacy)
}

/// Return the selected gate schedule. V8.1 deliberately changes only
/// `win_at`: stage indices, maps, opponents, difficulty, and decision cadence
/// remain stable so checkpoints and episode metadata retain their identity.
pub fn stages_for(v81: bool) -> Vec<Stage> {
    stages_for_schedule(if v81 {
        CurriculumSchedule::V81
    } else {
        CurriculumSchedule::Legacy
    })
}

pub fn stages_for_schedule(schedule: CurriculumSchedule) -> Vec<Stage> {
    use Nations::{Default as ND, Exact as NE};
    let mut stages = vec![
        Stage {
            maps: &["Onion"],
            bots: 0,
            difficulty: "Easy",
            nations: NE(1),
            decision_ticks: 15,
            win_at: 0.9,
        },
        Stage {
            maps: &["Onion"],
            bots: 0,
            difficulty: "Easy",
            nations: NE(3),
            decision_ticks: 15,
            win_at: 0.8,
        },
        Stage {
            maps: &["Onion", "Pangaea"],
            bots: 5,
            difficulty: "Easy",
            nations: NE(3),
            decision_ticks: 15,
            win_at: 0.75,
        },
        Stage {
            maps: &["Pangaea", "Caucasus"],
            bots: 10,
            difficulty: "Easy",
            nations: NE(6),
            decision_ticks: 15,
            win_at: 0.65,
        },
        Stage {
            maps: &["Pangaea", "Caucasus", "BlackSea"],
            bots: 30,
            difficulty: "Easy",
            nations: ND,
            decision_ticks: 10,
            win_at: 0.55,
        },
        Stage {
            maps: &["BlackSea", "BetweenTwoSeas", "Caucasus"],
            bots: 30,
            difficulty: "Medium",
            nations: ND,
            decision_ticks: 10,
            win_at: 0.5,
        },
        Stage {
            maps: &["World", "Asia", "BlackSea"],
            bots: 50,
            difficulty: "Medium",
            nations: ND,
            decision_ticks: 10,
            win_at: 0.5,
        },
        Stage {
            maps: &["World", "Asia", "BetweenTwoSeas", "Caucasus"],
            bots: 80,
            difficulty: "Medium",
            nations: ND,
            decision_ticks: 10,
            win_at: 0.45,
        },
        Stage {
            maps: &ALL_MAPS,
            bots: 80,
            difficulty: "Hard",
            nations: ND,
            decision_ticks: 10,
            win_at: 0.4,
        },
        Stage {
            maps: &ALL_MAPS,
            bots: 120,
            difficulty: "Hard",
            nations: ND,
            decision_ticks: 10,
            win_at: 0.35,
        },
        Stage {
            maps: &ALL_MAPS,
            bots: 150,
            difficulty: "Impossible",
            nations: ND,
            decision_ticks: 10,
            win_at: 0.3,
        },
    ];
    match schedule {
        CurriculumSchedule::Legacy => {}
        CurriculumSchedule::V81 => {
            // Stages 0-3 are intentionally unchanged. Crowded-map wins are
            // much rarer than placement success, so require repeatable wins
            // without making the old 0.5 gate a permanent lock.
            let gates = [0.35, 0.30, 0.25, 0.22, 0.20, 0.18, 0.15];
            for (stage, gate) in stages[4..].iter_mut().zip(gates) {
                stage.win_at = gate;
            }
        }
        CurriculumSchedule::V811 => {
            // Stage 5 is the opt-in Easy bridge. Stage 6 repeats its exact
            // map/bot/nation setup at Medium before World/Asia first appear
            // at stage 7. The old stages 6+ retain their challenge and gate
            // after shifting by one.
            stages[5].difficulty = "Easy";
            let medium_bridge = Stage {
                maps: stages[5].maps,
                bots: stages[5].bots,
                difficulty: "Medium",
                nations: stages[5].nations,
                decision_ticks: stages[5].decision_ticks,
                win_at: 0.25,
            };
            stages.insert(6, medium_bridge);
            let gates = [0.35, 0.30, 0.25, 0.25, 0.22, 0.20, 0.18, 0.15];
            for (stage, gate) in stages[4..].iter_mut().zip(gates) {
                stage.win_at = gate;
            }
        }
        CurriculumSchedule::V82 | CurriculumSchedule::V83 | CurriculumSchedule::V10 => {
            // Stages 0-4 retain the V8.1/V8.1.1 identities and gate. The
            // 30-player/eight-map Easy bridge precedes broad-pool Easy at
            // 50 and 80 players. Medium only begins after all three Easy
            // loads, then repeats 30/50/80 before Hard and Impossible.
            stages.truncate(5);
            stages[4].win_at = 0.35;
            stages.extend([
                Stage {
                    maps: &V82_STAGE5_MAPS,
                    bots: 30,
                    difficulty: "Easy",
                    nations: ND,
                    decision_ticks: 10,
                    win_at: 0.30,
                },
                Stage {
                    maps: &V82_MAPS,
                    bots: 50,
                    difficulty: "Easy",
                    nations: ND,
                    decision_ticks: 10,
                    win_at: 0.25,
                },
                Stage {
                    maps: &V82_MAPS,
                    bots: 80,
                    difficulty: "Easy",
                    nations: ND,
                    decision_ticks: 10,
                    win_at: 0.20,
                },
                Stage {
                    maps: &V82_MAPS,
                    bots: 30,
                    difficulty: "Medium",
                    nations: ND,
                    decision_ticks: 10,
                    win_at: 0.25,
                },
                Stage {
                    maps: &V82_MAPS,
                    bots: 50,
                    difficulty: "Medium",
                    nations: ND,
                    decision_ticks: 10,
                    win_at: 0.22,
                },
                Stage {
                    maps: &V82_MAPS,
                    bots: 80,
                    difficulty: "Medium",
                    nations: ND,
                    decision_ticks: 10,
                    win_at: 0.20,
                },
                Stage {
                    maps: &V82_MAPS,
                    bots: 80,
                    difficulty: "Hard",
                    nations: ND,
                    decision_ticks: 10,
                    win_at: 0.18,
                },
                Stage {
                    maps: &V82_MAPS,
                    bots: 120,
                    difficulty: "Hard",
                    nations: ND,
                    decision_ticks: 10,
                    win_at: 0.15,
                },
                Stage {
                    maps: &V82_MAPS,
                    bots: 150,
                    difficulty: "Impossible",
                    nations: ND,
                    decision_ticks: 10,
                    win_at: 0.12,
                },
            ]);
            if schedule == CurriculumSchedule::V83 || schedule == CurriculumSchedule::V10 {
                stages.insert(
                    5,
                    Stage {
                        maps: &["Onion", "Pangaea", "Caucasus"],
                        bots: 10,
                        difficulty: "Easy",
                        nations: NE(6),
                        decision_ticks: 10,
                        win_at: 0.45,
                    },
                );
                // OpenFront games are bot-heavy: ~30 bots / ~5 nations on
                // small maps, ~200 bots / ~30 nations at world scale. Never
                // start nation-only (bots=0), and never let Nations::Default
                // outnumber bots on World/Europe (50–72 nations).
                if schedule == CurriculumSchedule::V10 {
                    apply_v10_bot_heavy_density(&mut stages);
                } else {
                    apply_v83_bot_heavy_density(&mut stages);
                }
            }
        }
        CurriculumSchedule::V9 => {
            // Parallel sparse-win curriculum: extremely gradual difficulty,
            // high gates (0.90–0.975). win_at never hits 1.0 because
            // should_advance uses strict `mean > gate` over WINDOW=40.
            //
            // Bots are always present and always far outnumber nations
            // (~6:1+, e.g. 30 bots / 5 nations, 200 bots / 30 nations).
            // Never use Nations::Default — that dumps the full map manifest
            // (World ~72 nations) and inverts the bot:nation ratio.
            // Early stages are bots-only, then nations are layered on top.
            stages = vec![
                Stage {
                    maps: &["Onion"],
                    bots: 15,
                    difficulty: "Easy",
                    nations: NE(0),
                    decision_ticks: 15,
                    win_at: 0.975,
                },
                Stage {
                    maps: &["Onion"],
                    bots: 30,
                    difficulty: "Easy",
                    nations: NE(0),
                    decision_ticks: 15,
                    win_at: 0.95,
                },
                Stage {
                    maps: &["Onion"],
                    bots: 30,
                    difficulty: "Easy",
                    nations: NE(2),
                    decision_ticks: 15,
                    win_at: 0.95,
                },
                Stage {
                    maps: &["Onion"],
                    bots: 30,
                    difficulty: "Easy",
                    nations: NE(5),
                    decision_ticks: 15,
                    win_at: 0.925,
                },
                Stage {
                    maps: &["Onion"],
                    bots: 40,
                    difficulty: "Easy",
                    nations: NE(5),
                    decision_ticks: 15,
                    win_at: 0.925,
                },
                Stage {
                    maps: &["Onion", "Pangaea"],
                    bots: 50,
                    difficulty: "Easy",
                    nations: NE(6),
                    decision_ticks: 15,
                    win_at: 0.90,
                },
                Stage {
                    maps: &["Pangaea"],
                    bots: 60,
                    difficulty: "Easy",
                    nations: NE(8),
                    decision_ticks: 15,
                    win_at: 0.90,
                },
                Stage {
                    maps: &["Pangaea"],
                    bots: 80,
                    difficulty: "Easy",
                    nations: NE(10),
                    decision_ticks: 15,
                    win_at: 0.90,
                },
                Stage {
                    maps: &["Pangaea", "Caucasus"],
                    bots: 100,
                    difficulty: "Easy",
                    nations: NE(12),
                    decision_ticks: 15,
                    win_at: 0.90,
                },
                Stage {
                    maps: &["Pangaea", "Caucasus"],
                    bots: 100,
                    difficulty: "Easy",
                    nations: NE(15),
                    decision_ticks: 10,
                    win_at: 0.90,
                },
                Stage {
                    maps: &["Pangaea", "Caucasus", "BlackSea"],
                    bots: 120,
                    difficulty: "Easy",
                    nations: NE(15),
                    decision_ticks: 10,
                    win_at: 0.90,
                },
                Stage {
                    maps: &["Pangaea", "Caucasus", "BlackSea"],
                    bots: 150,
                    difficulty: "Easy",
                    nations: NE(20),
                    decision_ticks: 10,
                    win_at: 0.90,
                },
                Stage {
                    maps: &["Pangaea", "Caucasus", "BlackSea"],
                    bots: 150,
                    difficulty: "Easy",
                    nations: NE(20),
                    decision_ticks: 10,
                    win_at: 0.90,
                },
                Stage {
                    maps: &V82_STAGE5_MAPS,
                    bots: 180,
                    difficulty: "Easy",
                    nations: NE(25),
                    decision_ticks: 10,
                    win_at: 0.90,
                },
                Stage {
                    maps: &V82_MAPS,
                    bots: 200,
                    difficulty: "Easy",
                    nations: NE(30),
                    decision_ticks: 10,
                    win_at: 0.90,
                },
                Stage {
                    maps: &V82_MAPS,
                    bots: 200,
                    difficulty: "Easy",
                    nations: NE(30),
                    decision_ticks: 10,
                    win_at: 0.90,
                },
                Stage {
                    maps: &V82_MAPS,
                    bots: 220,
                    difficulty: "Easy",
                    nations: NE(30),
                    decision_ticks: 10,
                    win_at: 0.90,
                },
                Stage {
                    maps: &V82_MAPS,
                    bots: 250,
                    difficulty: "Easy",
                    nations: NE(35),
                    decision_ticks: 10,
                    win_at: 0.90,
                },
                Stage {
                    maps: &V82_MAPS,
                    bots: 250,
                    difficulty: "Easy",
                    nations: NE(35),
                    decision_ticks: 10,
                    win_at: 0.90,
                },
                Stage {
                    maps: &V82_MAPS,
                    bots: 200,
                    difficulty: "Medium",
                    nations: NE(30),
                    decision_ticks: 10,
                    win_at: 0.90,
                },
                Stage {
                    maps: &V82_MAPS,
                    bots: 250,
                    difficulty: "Medium",
                    nations: NE(35),
                    decision_ticks: 10,
                    win_at: 0.90,
                },
                Stage {
                    maps: &V82_MAPS,
                    bots: 300,
                    difficulty: "Medium",
                    nations: NE(40),
                    decision_ticks: 10,
                    win_at: 0.90,
                },
                Stage {
                    maps: &V82_MAPS,
                    bots: 250,
                    difficulty: "Hard",
                    nations: NE(35),
                    decision_ticks: 10,
                    win_at: 0.90,
                },
                Stage {
                    maps: &V82_MAPS,
                    bots: 300,
                    difficulty: "Hard",
                    nations: NE(40),
                    decision_ticks: 10,
                    win_at: 0.90,
                },
                Stage {
                    maps: &V82_MAPS,
                    bots: 350,
                    difficulty: "Impossible",
                    nations: NE(50),
                    decision_ticks: 10,
                    win_at: 0.90,
                },
            ];
        }
    }
    stages
}

pub const GH_MAX: i64 = 150;
pub const GW_MAX: i64 = 250;

pub fn sample_episode(
    stg: &[Stage],
    stage: usize,
    rng: &mut impl rand::Rng,
) -> (String, u32, &'static str, Nations, bool) {
    let cur = &stg[stage];
    if stage > 0 && rng.r#gen::<f64>() < REHEARSAL_P {
        let past_i = rng.gen_range(0..stage);
        let past = &stg[past_i];
        let m = past.maps[rng.gen_range(0..past.maps.len())];
        // Rehearse the past lobby setup (bots/difficulty/nations), not just
        // the past map with current-stage pressure. Current-stage bots on an
        // old map was a fake rehearsal that never relieved density cliffs.
        return (
            m.to_string(),
            past.bots,
            past.difficulty,
            past.nations,
            true,
        );
    }
    let m = cur.maps[rng.gen_range(0..cur.maps.len())];
    (m.to_string(), cur.bots, cur.difficulty, cur.nations, false)
}

pub fn timeweight(tick: i64) -> f64 {
    0.5 + 0.5 * (tick as f64 / 8000.0).min(1.0)
}

/// Composite strength per living player: land / military / economic /
/// structural share. See rl/curriculum.py::strengths for the boat-churn
/// rationale behind counting fielded troops.
pub fn strengths(ents: &EntsData, land_total: i64) -> HashMap<usize, f64> {
    let alive: Vec<&crate::feat::PlayerE> = ents.players.iter().filter(|p| p.alive).collect();
    let mut fielded: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    let mut sv: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    for u in &ents.units {
        if u.troops > 0.0 {
            *fielded.entry(u.owner).or_insert(0.0) += u.troops;
        }
    }
    // Struct value needs the engine type string, which parse_ents doesn't
    // keep (only the class index); recompute from class since STATIC
    // classes 0..6 map 1:1 to BUILD_TYPES-minus-Warship order.
    for u in &ents.units {
        let ty = match u.class {
            0 => "City",
            1 => "Port",
            2 => "Defense Post",
            3 => "Missile Silo",
            4 => "SAM Launcher",
            5 => "Factory",
            _ => continue,
        };
        if let Some(v) = struct_value(ty) {
            if !u.constructing {
                *sv.entry(u.owner).or_insert(0.0) += v * u.level.max(1.0);
            }
        }
    }
    for a in &ents.attacks {
        *fielded.entry(a.from).or_insert(0.0) += a.troops;
    }
    let troops = |p: &crate::feat::PlayerE| p.troops + fielded.get(&p.id).copied().unwrap_or(0.0);
    let tot_troops: f64 = alive.iter().map(|p| troops(p)).sum::<f64>() + 1e-9;
    let tot_gold: f64 = alive.iter().map(|p| p.gold).sum::<f64>() + 1e-9;
    let tot_sv: f64 = alive
        .iter()
        .map(|p| sv.get(&p.id).copied().unwrap_or(0.0))
        .sum::<f64>()
        + 1e-9;
    alive
        .iter()
        .map(|p| {
            let s = K_LAND * (p.tiles / land_total as f64)
                + K_MIL * (troops(p) / tot_troops)
                + K_ECO * (p.gold / tot_gold)
                + K_BUILD * (sv.get(&p.id).copied().unwrap_or(0.0) / tot_sv);
            (p.id, s)
        })
        .collect()
}

fn finite_nonnegative(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 {
        value
    } else {
        0.0
    }
}

fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() { value } else { 0.0 }
}

/// V8.1 Φ(s), derived from the exact composite map used by placement.
pub fn dominance_potential(composite: &HashMap<usize, f64>, me: usize, clamp: f64) -> f64 {
    let clamp = finite_nonnegative(clamp);
    if clamp == 0.0 {
        return 0.0;
    }
    let mine = finite_nonnegative(composite.get(&me).copied().unwrap_or(0.0));
    let strongest_opponent = composite
        .iter()
        .filter(|&(&pid, _)| pid != me)
        .map(|(_, &strength)| finite_nonnegative(strength))
        .fold(0.0, f64::max);
    ((mine + DOMINANCE_EPS) / (strongest_opponent + DOMINANCE_EPS))
        .ln()
        .clamp(-clamp, clamp)
}

/// Agent share after normalizing the placement composite over living players.
pub fn normalized_strength_share(composite: &HashMap<usize, f64>, me: usize) -> f64 {
    let mine = finite_nonnegative(composite.get(&me).copied().unwrap_or(0.0));
    let total: f64 = composite
        .values()
        .map(|&value| finite_nonnegative(value))
        .sum();
    if total > 0.0 {
        (mine / total).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

pub fn strength_delta_weight(
    delta: f64,
    normalized_share: f64,
    stage: usize,
    config: RewardConfig,
    has_active_attack: bool,
) -> f64 {
    if delta >= 0.0 {
        W_DELTA_GAIN
    } else if config.v86_attack_symmetric_loss && has_active_attack {
        // Attack burns troops before land accrues; don't asymmetrically tax
        // the intentional dip while an attack the agent opened is in flight.
        W_DELTA_GAIN
    } else if config.dominant_loss_active(stage)
        && normalized_share >= config.v81_dominance_threshold
    {
        config.v81_delta_loss_dominant
    } else {
        config.delta_loss()
    }
}

/// (place, n_players).
pub fn placement(ents: &EntsData, me: i64, agent_alive: bool, land_total: i64) -> (i64, i64) {
    let ids: std::collections::HashSet<usize> = ents.players.iter().map(|p| p.id).collect();
    let n = ids.len() as i64 + if ids.contains(&(me as usize)) { 0 } else { 1 };
    let s = strengths(ents, land_total);
    let me_u = me as usize;
    if !agent_alive || !s.contains_key(&me_u) {
        let others_alive = s.keys().filter(|&&pid| pid != me_u).count() as i64;
        return ((1 + others_alive).min(n), n);
    }
    let mine = s[&me_u];
    let better = s
        .iter()
        .filter(|&(&pid, &v)| pid != me_u && v > mine)
        .count() as i64;
    (1 + better, n)
}

pub fn placement_score(place: i64, n: i64) -> f64 {
    (n - place) as f64 / (n - 1).max(1) as f64
}

pub fn terminal_reward(place: i64, won: bool) -> f64 {
    let mut r = W_PLACE * (place as f64).powf(-PLACE_POW);
    if won {
        r += W_WIN;
    }
    r
}

/// V9 sparse terminal: win = +1, anything else (loss / death / timeout) = -1.
pub fn sparse_terminal_reward(won: bool) -> f64 {
    if won {
        V9_SPARSE_WIN
    } else {
        V9_SPARSE_LOSS
    }
}

/// Alive+land survival shaping: small positive signal so death is not the only
/// non-win terminal contrast under a softer death penalty.
pub fn v10_survival_reward(alive: bool, land_share: f64, config: RewardConfig) -> f64 {
    if !alive || config.v10_survival_coef == 0.0 {
        0.0
    } else {
        config.v10_survival_coef * finite_or_zero(land_share).clamp(0.0, 1.0)
    }
}

fn v10_diplo_panic_armed(land_share: f64, tick: i64, max_ticks: i64, config: RewardConfig) -> bool {
    let share_armed = finite_or_zero(land_share) >= config.v10_diplo_panic_share;
    let tick_frac = tick as f64 / max_ticks.max(1) as f64;
    let late_armed = tick_frac >= config.v10_diplo_panic_tick_frac;
    share_armed || late_armed
}

/// Penalize late/dominant diplomacy spam (donate / alliance / embargo thrash)
/// that GameRecord analysis showed as the death-spiral failure mode.
pub fn v10_diplo_panic_penalty(
    action: i64,
    land_share: f64,
    tick: i64,
    max_ticks: i64,
    config: RewardConfig,
) -> f64 {
    if config.v10_diplo_panic == 0.0 {
        return 0.0;
    }
    if !v10_diplo_panic_armed(land_share, tick, max_ticks, config) {
        return 0.0;
    }
    match action {
        A_DONATE_GOLD
        | A_DONATE_TROOPS
        | A_ALLIANCE_REQUEST
        | A_BREAK_ALLIANCE
        | A_EMBARGO
        | A_EMBARGO_STOP => -config.v10_diplo_panic.abs(),
        _ => 0.0,
    }
}

/// Small prior for productive combat/build actions (targeted attack/boat, build).
pub fn v10_combat_action_bonus(action: i64, has_target: bool, config: RewardConfig) -> f64 {
    if config.v10_combat_action == 0.0 {
        return 0.0;
    }
    let coef = config.v10_combat_action.abs();
    match action {
        A_ATTACK if has_target => coef,
        A_BOAT if has_target => coef * 0.75,
        A_BUILD => coef * 0.5,
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feat::parse_ents;
    use serde_json::json;

    fn config() -> RewardConfig {
        RewardConfig {
            gamma: 0.9,
            v81_dom_coef: 0.25,
            v81_min_stage: 4,
            v81_potential_clamp: 2.0,
            v81_dominant_loss: true,
            v81_dominance_threshold: 0.55,
            v81_delta_loss_dominant: 5.25,
            v81_churn_coef: 0.05,
            v81_churn_window: 2,
            v81_churn_min_stage: 4,
            v83_close_coef: 4.0,
            v83_churn_coef: 0.06,
            v84_boat_useful: 0.15,
            v84_boat_destroyed: -0.20,
            v84_boat_cancelled: -0.03,
            v84_boat_own_shore: -0.05,
            v84_boat_min_stage: 4,
            v84_tempo_coef: 0.005,
            v84_tempo_min_stage: 4,
            v84_fast_win_coef: 8.0,
            v85_tempo_share_threshold: 0.0,
            v85_extra_win_bonus: 0.0,
            v85_embargo_bad_stop: -0.15,
            v85_embargo_good_stop: 0.02,
            v85_embargo_min_stage: 4,
            v85_premature_retreat: -0.10,
            v85_thrash_reengage: -0.10,
            v85_combat_min_stage: 4,
            v86_delta_loss: 0.0,
            v86_attack_symmetric_loss: false,
            v86_skip_combat_churn: false,
            v86_death_penalty: 0.0,
            v9_sparse_win: false,
            v10_survival_coef: 0.0,
            v10_diplo_panic: 0.0,
            v10_diplo_panic_share: 0.35,
            v10_diplo_panic_tick_frac: 0.55,
            v10_combat_action: 0.0,
        }
    }

    fn composite(values: &[(usize, f64)]) -> HashMap<usize, f64> {
        values.iter().copied().collect()
    }

    #[test]
    fn potential_shaping_telescopes_and_cannot_reward_churn() {
        let phi = [0.2, 1.1, -0.4, 0.2];
        let (gamma, coefficient) = (0.9f64, 0.25);
        let mut shaper = DominanceShaper::default();
        shaper.reset(phi[0]);
        let increments: Vec<f64> = phi[1..]
            .iter()
            .map(|&next| shaper.transition(next, gamma, coefficient))
            .collect();
        let discounted: f64 = increments
            .iter()
            .enumerate()
            .map(|(t, value)| gamma.powi(t as i32) * value)
            .sum();
        let expected =
            coefficient * (-phi[0] + gamma.powi((phi.len() - 1) as i32) * phi[phi.len() - 1]);
        assert!((discounted - expected).abs() < 1e-12);
        assert_eq!(shaper.prior().to_bits(), phi[phi.len() - 1].to_bits());
    }

    #[test]
    fn closeout_potential_telescopes_and_terminal_zero_prevents_positive_cycles() {
        assert_eq!(closeout_potential(0.45), 0.0);
        assert_eq!(closeout_potential(0.80), 1.0);
        let shares = [0.45, 0.60, 0.52, 0.60];
        let mut shaper = DominanceShaper::default();
        shaper.reset(closeout_potential(shares[0]));
        let shaped: Vec<_> = shares[1..]
            .iter()
            .map(|&q| shaper.transition(closeout_potential(q), 0.9, 4.0))
            .collect();
        let discounted: f64 = shaped
            .iter()
            .enumerate()
            .map(|(t, value)| 0.9f64.powi(t as i32) * value)
            .sum();
        assert!(discounted >= 0.0);
        let terminal = shaper.transition(0.0, 0.9, 4.0);
        let full_return = discounted + 0.9f64.powi(shaped.len() as i32) * terminal;
        assert!(full_return.abs() < 1e-12);
        assert!(
            terminal < 0.0,
            "timeout after closeout must repay potential"
        );
    }

    #[test]
    fn v83_churn_increase_is_closeout_only_and_v82_path_is_unchanged() {
        let cfg = config();
        let pair = Some(InverseActionPair::AttackRetreat);
        assert_eq!(action_churn_penalty(pair, 5, cfg), -0.05);
        assert_eq!(v83_action_churn_penalty(pair, 4, 0.9, cfg), -0.05);
        assert_eq!(v83_action_churn_penalty(pair, 5, 0.449, cfg), -0.05);
        assert_eq!(v83_action_churn_penalty(pair, 5, 0.45, cfg), -0.06);
    }

    #[test]
    fn potential_uses_the_strongest_alive_opponent() {
        let base = dominance_potential(&composite(&[(1, 0.6), (2, 0.2), (3, 0.4)]), 1, 10.0);
        let weaker_changed =
            dominance_potential(&composite(&[(1, 0.6), (2, 0.3), (3, 0.4)]), 1, 10.0);
        let strongest_changed =
            dominance_potential(&composite(&[(1, 0.6), (2, 0.2), (3, 0.5)]), 1, 10.0);
        assert_eq!(base.to_bits(), weaker_changed.to_bits());
        assert!(strongest_changed < base);
    }

    #[test]
    fn shaper_reset_drops_prior_episode_potential() {
        let mut shaper = DominanceShaper::default();
        shaper.reset(1.5);
        let _ = shaper.transition(-0.7, 0.9, 0.25);
        shaper.reset(0.3);
        assert_eq!(shaper.prior().to_bits(), 0.3f64.to_bits());
        assert!((shaper.transition(0.5, 0.9, 0.25) - 0.25 * (0.9 * 0.5 - 0.3)).abs() < 1e-15);
    }

    #[test]
    fn stage_gate_controls_both_v81_behaviors() {
        let cfg = config();
        assert!(!cfg.dominance_shaping_active(3));
        assert!(!cfg.dominant_loss_active(3));
        assert!(cfg.dominance_shaping_active(4));
        assert!(cfg.dominant_loss_active(4));
        assert_eq!(strength_delta_weight(-0.1, 0.9, 3, cfg, false), W_DELTA_LOSS);
    }

    #[test]
    fn dominant_threshold_relaxes_only_losses_at_or_above_threshold() {
        let cfg = config();
        assert_eq!(strength_delta_weight(-0.1, 0.549, 4, cfg, false), W_DELTA_LOSS);
        assert_eq!(
            strength_delta_weight(-0.1, 0.55, 4, cfg, false),
            cfg.v81_delta_loss_dominant
        );
        assert_eq!(strength_delta_weight(0.1, 0.99, 4, cfg, false), W_DELTA_GAIN);
    }

    #[test]
    fn v86_softens_loss_and_symmetrizes_during_active_attack() {
        let mut cfg = config();
        cfg.v86_delta_loss = 5.5;
        cfg.v86_attack_symmetric_loss = true;
        assert_eq!(strength_delta_weight(-0.1, 0.1, 4, cfg, false), 5.5);
        assert_eq!(
            strength_delta_weight(-0.1, 0.1, 4, cfg, true),
            W_DELTA_GAIN
        );
        assert_eq!(cfg.death_penalty(), W_DEATH);
        cfg.v86_death_penalty = 10.0;
        assert_eq!(cfg.death_penalty(), 10.0);
        assert_eq!(cfg.reward_profile_id(), V86_REWARD_PROFILE);
    }

    #[test]
    fn v9_sparse_win_selects_sparse_reward_profile() {
        let mut cfg = config();
        cfg.v9_sparse_win = true;
        assert_eq!(cfg.reward_profile_id(), V9_REWARD_PROFILE);
        // Sparse wins over V8.6 knobs if both were somehow set.
        cfg.v86_death_penalty = 10.0;
        assert_eq!(cfg.reward_profile_id(), V9_REWARD_PROFILE);
    }

    #[test]
    fn v10_reward_profile_and_shaping_helpers() {
        let mut cfg = config();
        cfg.v10_survival_coef = 0.01;
        cfg.v10_diplo_panic = 0.08;
        cfg.v10_diplo_panic_share = 0.35;
        cfg.v10_diplo_panic_tick_frac = 0.55;
        cfg.v10_combat_action = 0.02;
        cfg.v86_death_penalty = 3.0;
        assert_eq!(cfg.reward_profile_id(), V10_REWARD_PROFILE);
        assert_eq!(cfg.death_penalty(), 3.0);
        assert_eq!(v10_survival_reward(true, 0.5, cfg), 0.005);
        assert_eq!(v10_survival_reward(false, 0.5, cfg), 0.0);
        assert_eq!(
            v10_diplo_panic_penalty(A_DONATE_GOLD, 0.40, 100, 1000, cfg),
            -0.08
        );
        assert_eq!(
            v10_diplo_panic_penalty(A_ATTACK, 0.40, 100, 1000, cfg),
            0.0
        );
        assert_eq!(
            v10_diplo_panic_penalty(A_DONATE_GOLD, 0.10, 100, 1000, cfg),
            0.0
        );
        assert_eq!(
            v10_diplo_panic_penalty(A_EMBARGO, 0.10, 600, 1000, cfg),
            -0.08
        );
        assert_eq!(v10_combat_action_bonus(A_ATTACK, true, cfg), 0.02);
        assert_eq!(v10_combat_action_bonus(A_ATTACK, false, cfg), 0.0);
        assert_eq!(v10_combat_action_bonus(A_BOAT, true, cfg), 0.015);
        assert_eq!(v10_combat_action_bonus(A_BUILD, false, cfg), 0.01);
        // Sparse still wins if somehow both are set.
        cfg.v9_sparse_win = true;
        assert_eq!(cfg.reward_profile_id(), V9_REWARD_PROFILE);
    }

    #[test]
    fn disabled_configuration_preserves_legacy_reward_bits() {
        let mut cfg = config();
        cfg.v81_dom_coef = 0.0;
        cfg.v81_dominant_loss = false;
        let (mine, previous, tw) = (0.3125, 0.375, 0.78125);
        let delta = mine - previous;
        let legacy = W_STR * mine * tw
            + (if delta >= 0.0 {
                W_DELTA_GAIN
            } else {
                W_DELTA_LOSS
            }) * delta;
        let current =
            W_STR * mine * tw + strength_delta_weight(delta, 0.99, 10, cfg, false) * delta;
        assert_eq!(legacy.to_bits(), current.to_bits());
        assert!(!cfg.dominance_shaping_active(10));
    }

    #[test]
    fn potential_and_share_are_finite_at_all_edges() {
        for values in [
            composite(&[]),
            composite(&[(1, 0.0)]),
            composite(&[(1, f64::NAN), (2, f64::INFINITY)]),
            composite(&[(1, 1e300), (2, 1e-300)]),
        ] {
            let phi = dominance_potential(&values, 1, 2.0);
            let share = normalized_strength_share(&values, 1);
            assert!(phi.is_finite() && (-2.0..=2.0).contains(&phi));
            assert!(share.is_finite() && (0.0..=1.0).contains(&share));
        }
    }

    #[test]
    fn dominance_uses_exact_placement_composite_strength() {
        let ents = parse_ents(&json!({
            "players": [
                {"id": 1, "pid": "me", "troops": 100, "gold": 80, "tiles": 60, "alive": true},
                {"id": 2, "pid": "opp", "troops": 50, "gold": 20, "tiles": 40, "alive": true}
            ],
            "units": [], "attacks": [], "alliances": []
        }));
        let exact = strengths(&ents, 100);
        let expected = ((exact[&1] + DOMINANCE_EPS) / (exact[&2] + DOMINANCE_EPS)).ln();
        assert!((dominance_potential(&exact, 1, 10.0) - expected).abs() < 1e-15);
        assert_eq!(placement(&ents, 1, true, 100), (1, 2));
    }

    fn player_action(action: i64, player: usize) -> ChosenAction {
        ChosenAction::new(action, Some(ActionTarget::Player(player)))
    }

    fn unit_action(action: i64, unit: usize) -> ChosenAction {
        ChosenAction::new(action, Some(ActionTarget::Unit(unit)))
    }

    #[test]
    fn churn_detects_every_supported_inverse_pair() {
        let cases = [
            (
                unit_action(A_BOAT, 17),
                unit_action(A_CANCEL_BOAT, 17),
                InverseActionPair::BoatCancelBoat,
            ),
            (
                player_action(A_EMBARGO, 3),
                player_action(A_EMBARGO_STOP, 3),
                InverseActionPair::EmbargoEmbargoStop,
            ),
            (
                player_action(A_ATTACK, 3),
                player_action(A_RETREAT, 3),
                InverseActionPair::AttackRetreat,
            ),
            (
                player_action(A_RETREAT, 3),
                player_action(A_ATTACK, 3),
                InverseActionPair::RetreatAttack,
            ),
        ];
        for (first, second, expected) in cases {
            let mut tracker = ActionChurnTracker::default();
            assert_eq!(tracker.observe(first, 2), None);
            assert_eq!(tracker.observe(second, 2), Some(expected));
            assert_eq!(tracker.counts().total(), 1);
        }
    }

    #[test]
    fn churn_does_not_treat_one_way_pairs_as_symmetric() {
        for (first, second) in [
            (unit_action(A_CANCEL_BOAT, 17), unit_action(A_BOAT, 17)),
            (
                player_action(A_EMBARGO_STOP, 3),
                player_action(A_EMBARGO, 3),
            ),
        ] {
            let mut tracker = ActionChurnTracker::default();
            tracker.observe(first, 2);
            assert_eq!(tracker.observe(second, 2), None);
            assert_eq!(tracker.counts().total(), 0);
        }
    }

    #[test]
    fn churn_requires_the_same_relevant_target_and_target_kind() {
        let cases = [
            (player_action(A_ATTACK, 3), player_action(A_RETREAT, 4)),
            (
                player_action(A_EMBARGO, 3),
                player_action(A_EMBARGO_STOP, 4),
            ),
            (unit_action(A_BOAT, 17), unit_action(A_CANCEL_BOAT, 18)),
            (
                ChosenAction::new(A_ATTACK, None),
                ChosenAction::new(A_RETREAT, None),
            ),
            (
                ChosenAction::new(A_ATTACK, Some(ActionTarget::Player(17))),
                ChosenAction::new(A_RETREAT, Some(ActionTarget::Unit(17))),
            ),
        ];
        for (first, second) in cases {
            let mut tracker = ActionChurnTracker::default();
            tracker.observe(first, 2);
            assert_eq!(tracker.observe(second, 2), None);
            assert_eq!(tracker.counts().total(), 0);
        }
    }

    #[test]
    fn churn_window_counts_decisions_and_expires_old_actions() {
        let attack = player_action(A_ATTACK, 3);
        let retreat = player_action(A_RETREAT, 3);
        let noop = ChosenAction::new(crate::feat::A_NOOP, None);

        let mut within = ActionChurnTracker::default();
        within.observe(attack, 2);
        within.observe(noop, 2);
        assert_eq!(
            within.observe(retreat, 2),
            Some(InverseActionPair::AttackRetreat)
        );

        let mut expired = ActionChurnTracker::default();
        expired.observe(attack, 2);
        expired.observe(noop, 2);
        expired.observe(noop, 2);
        assert_eq!(expired.observe(retreat, 2), None);
        assert_eq!(expired.counts().total(), 0);
    }

    #[test]
    fn churn_ignores_repeats_noops_and_unrelated_actions() {
        let attack = player_action(A_ATTACK, 3);
        let retreat = player_action(A_RETREAT, 3);
        let noop = ChosenAction::new(crate::feat::A_NOOP, None);
        let unrelated = player_action(crate::feat::A_DONATE_GOLD, 3);
        let mut tracker = ActionChurnTracker::default();

        tracker.observe(attack, 4);
        assert_eq!(tracker.observe(attack, 4), None);
        assert_eq!(tracker.observe(noop, 4), None);
        assert_eq!(tracker.observe(unrelated, 4), None);
        assert_eq!(
            tracker.observe(retreat, 4),
            Some(InverseActionPair::AttackRetreat)
        );
        assert_eq!(tracker.observe(noop, 4), None);
        assert_eq!(tracker.observe(retreat, 4), None);
        assert_eq!(tracker.counts().attack_retreat, 1);
    }

    #[test]
    fn churn_reset_clears_history_and_episode_counters() {
        let mut tracker = ActionChurnTracker::default();
        tracker.observe(player_action(A_ATTACK, 3), 2);
        tracker.reset();
        assert_eq!(tracker.observe(player_action(A_RETREAT, 3), 2), None);
        assert_eq!(tracker.counts(), ActionPairCounts::default());
    }

    #[test]
    fn churn_zero_window_disables_detection_and_drops_history() {
        let mut tracker = ActionChurnTracker::default();
        tracker.observe(player_action(A_ATTACK, 3), 2);
        assert_eq!(tracker.observe(player_action(A_RETREAT, 3), 0), None);
        assert_eq!(
            tracker.observe(player_action(A_RETREAT, 3), 2),
            None,
            "zero window must not leave an old action to match later"
        );
    }

    #[test]
    fn churn_penalty_gate_is_opt_in_and_stage_specific() {
        let mut cfg = config();
        assert!(!cfg.churn_penalty_active(3));
        assert!(cfg.churn_penalty_active(4));
        assert_eq!(
            action_churn_penalty(Some(InverseActionPair::AttackRetreat), 3, cfg),
            0.0
        );
        assert_eq!(
            action_churn_penalty(Some(InverseActionPair::AttackRetreat), 4, cfg),
            -0.05
        );
        assert_eq!(action_churn_penalty(None, 4, cfg), 0.0);
        cfg.v81_churn_coef = 0.0;
        assert!(!cfg.churn_penalty_active(10));
        let legacy_reward = 1.25f64;
        let current =
            legacy_reward + action_churn_penalty(Some(InverseActionPair::AttackRetreat), 10, cfg);
        assert_eq!(current.to_bits(), legacy_reward.to_bits());
        cfg.v81_churn_coef = 0.05;
        cfg.v81_churn_window = 0;
        assert!(!cfg.churn_penalty_active(10));
    }

    #[test]
    fn boat_outcome_classifies_landing_cancel_return_and_destroy() {
        assert_eq!(
            classify_boat_resolution(false, 100.0, 500.0, 500.0, true, false),
            BoatOutcome::UsefulLanding
        );
        assert_eq!(
            classify_boat_resolution(true, 100.0, 500.0, 575.0, false, false),
            BoatOutcome::Cancelled
        );
        assert_eq!(
            classify_boat_resolution(false, 100.0, 500.0, 575.0, false, false),
            BoatOutcome::OwnShoreReturn
        );
        assert_eq!(
            classify_boat_resolution(false, 100.0, 500.0, 500.0, false, false),
            BoatOutcome::Destroyed
        );
        // Landing merges into an already-open sourced attack: no new attack id,
        // no troop refund → used to misclassify as Destroyed.
        assert_eq!(
            classify_boat_resolution(false, 100.0, 500.0, 500.0, false, true),
            BoatOutcome::UsefulLanding
        );
        // Own-shore refund still wins over a concurrent land attack.
        assert_eq!(
            classify_boat_resolution(false, 100.0, 500.0, 575.0, false, true),
            BoatOutcome::OwnShoreReturn
        );
    }

    #[test]
    fn boat_outcome_rewards_are_opt_in_and_stage_gated() {
        let mut cfg = config();
        assert!(cfg.boat_outcome_active(4));
        assert!(!cfg.boat_outcome_active(3));
        assert_eq!(
            boat_outcome_reward(BoatOutcome::UsefulLanding, cfg),
            0.15
        );
        assert_eq!(
            boat_outcome_reward(BoatOutcome::Destroyed, cfg),
            -0.20
        );
        cfg.v84_boat_useful = 0.0;
        cfg.v84_boat_destroyed = 0.0;
        cfg.v84_boat_cancelled = 0.0;
        cfg.v84_boat_own_shore = 0.0;
        assert!(!cfg.boat_outcome_active(10));
    }

    #[test]
    fn tempo_pressure_is_zero_until_dominant_and_grows_late() {
        assert_eq!(tempo_pressure(9000, 10000, 0.40, 0.55), 0.0);
        let early = tempo_pressure(1000, 10000, 0.70, 0.55);
        let late = tempo_pressure(9000, 10000, 0.70, 0.55);
        assert!(early < late);
        assert!((late - 0.81).abs() < 1e-9);
    }

    #[test]
    fn fast_win_bonus_scales_with_remaining_budget() {
        assert_eq!(fast_win_bonus(false, 100, 1000, 8.0), 0.0);
        assert!((fast_win_bonus(true, 0, 1000, 8.0) - 8.0).abs() < 1e-12);
        assert!((fast_win_bonus(true, 500, 1000, 8.0) - 4.0).abs() < 1e-12);
        assert!((fast_win_bonus(true, 1000, 1000, 8.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn relation_bands_match_engine_thresholds() {
        assert_eq!(relation_band(-51.0), RelationBand::Hostile);
        assert_eq!(relation_band(-1.0), RelationBand::Distrustful);
        assert_eq!(relation_band(0.0), RelationBand::Neutral);
        assert_eq!(relation_band(49.0), RelationBand::Neutral);
        assert_eq!(relation_band(50.0), RelationBand::Friendly);
    }

    #[test]
    fn embargo_stop_prices_hostile_vs_recovered() {
        let mut cfg = config();
        cfg.v85_embargo_bad_stop = -0.15;
        cfg.v85_embargo_good_stop = 0.02;
        assert!((embargo_stop_outcome_reward(-80.0, cfg) - (-0.15)).abs() < 1e-12);
        assert!((embargo_stop_outcome_reward(-10.0, cfg) - (-0.15)).abs() < 1e-12);
        assert!((embargo_stop_outcome_reward(10.0, cfg) - 0.02).abs() < 1e-12);
        assert!((embargo_stop_outcome_reward(80.0, cfg) - 0.02).abs() < 1e-12);
    }

    #[test]
    fn tempo_share_threshold_prefers_v85_when_set() {
        let mut cfg = config();
        cfg.v81_dominance_threshold = 0.55;
        assert!((cfg.tempo_share_threshold() - 0.55).abs() < 1e-12);
        cfg.v85_tempo_share_threshold = 0.30;
        assert!((cfg.tempo_share_threshold() - 0.30).abs() < 1e-12);
    }
}

#[cfg(test)]
mod curriculum_v81_tests {
    use super::*;

    #[test]
    fn v81_preserves_stage_identity_and_only_recalibrates_late_gates() {
        let legacy = stages();
        let v81 = stages_for(true);
        assert_eq!(legacy.len(), v81.len());
        for (index, (old, new)) in legacy.iter().zip(&v81).enumerate() {
            assert_eq!(old.maps, new.maps, "stage {index} maps changed");
            assert_eq!(old.bots, new.bots, "stage {index} bots changed");
            assert_eq!(
                old.difficulty, new.difficulty,
                "stage {index} difficulty changed"
            );
            assert_eq!(old.nations, new.nations, "stage {index} nations changed");
            assert_eq!(
                old.decision_ticks, new.decision_ticks,
                "stage {index} cadence changed"
            );
        }
        assert_eq!(
            v81.iter().map(|stage| stage.win_at).collect::<Vec<_>>(),
            vec![
                0.9, 0.8, 0.75, 0.65, 0.35, 0.30, 0.25, 0.22, 0.20, 0.18, 0.15
            ]
        );
        for index in 0..4 {
            assert_eq!(legacy[index].win_at, v81[index].win_at);
        }
    }

    #[test]
    fn v81_env_defaults_match_map_scale_plan() {
        assert_eq!(&V81_ENV_TARGETS[..6], &[24; 6]);
        assert_eq!(&V81_ENV_TARGETS[6..], &[12, 10, 8, 8, 8]);
        assert_eq!(V81_ENV_TARGETS.len(), stages().len());
    }

    #[test]
    fn v811_adds_easy_then_medium_bridge_before_world_scale() {
        let v81 = stages_for_schedule(CurriculumSchedule::V81);
        let v811 = stages_for_schedule(CurriculumSchedule::V811);
        assert_eq!(v811.len(), v81.len() + 1);
        assert_eq!(v811[5].maps, v81[5].maps);
        assert_eq!(v811[5].bots, v81[5].bots);
        assert_eq!(v811[5].nations, v81[5].nations);
        assert_eq!(v811[5].difficulty, "Easy");
        assert_eq!(v811[6].maps, v81[5].maps);
        assert_eq!(v811[6].bots, 30);
        assert_eq!(v811[6].nations, v81[5].nations);
        assert_eq!(v811[6].difficulty, "Medium");
        for (old, shifted) in v81[6..].iter().zip(&v811[7..]) {
            assert_eq!(old.maps, shifted.maps);
            assert_eq!(old.bots, shifted.bots);
            assert_eq!(old.difficulty, shifted.difficulty);
            assert_eq!(old.nations, shifted.nations);
        }
        assert_eq!(&v811[7].maps[..2], &["World", "Asia"]);
        assert_eq!(v811[7].bots, 50);
    }

    #[test]
    fn v811_gates_and_env_targets_follow_shifted_progression() {
        let v811 = stages_for_schedule(CurriculumSchedule::V811);
        assert_eq!(
            v811.iter().map(|stage| stage.win_at).collect::<Vec<_>>(),
            vec![
                0.9, 0.8, 0.75, 0.65, 0.35, 0.30, 0.25, 0.25, 0.22, 0.20, 0.18, 0.15
            ]
        );
        assert_eq!(V811_ENV_TARGETS.len(), v811.len());
        assert_eq!(&V811_ENV_TARGETS[..7], &[24; 7]);
        assert!(V811_ENV_TARGETS[7..].iter().all(|&target| target <= 12));
    }

    #[test]
    fn v82_preserves_stages_zero_through_four_and_uses_broad_progression() {
        let v811 = stages_for_schedule(CurriculumSchedule::V811);
        let v82 = stages_for_schedule(CurriculumSchedule::V82);
        assert_eq!(&v82[..5], &v811[..5]);
        assert_eq!(v82.len(), 14);

        let expected = [
            (30, "Easy", 0.30),
            (50, "Easy", 0.25),
            (80, "Easy", 0.20),
            (30, "Medium", 0.25),
            (50, "Medium", 0.22),
            (80, "Medium", 0.20),
            (80, "Hard", 0.18),
            (120, "Hard", 0.15),
            (150, "Impossible", 0.12),
        ];
        for (index, (stage, &(bots, difficulty, gate))) in
            v82[5..].iter().zip(&expected).enumerate()
        {
            let expected_maps: &[&str] = if index == 0 {
                &V82_STAGE5_MAPS
            } else {
                &V82_MAPS
            };
            assert_eq!(stage.maps, expected_maps, "stage {} maps", index + 5);
            assert_eq!(stage.bots, bots, "stage {} bots", index + 5);
            assert_eq!(
                stage.difficulty,
                difficulty,
                "stage {} difficulty",
                index + 5
            );
            assert_eq!(stage.win_at, gate, "stage {} gate", index + 5);
            assert_eq!(stage.nations, Nations::Default);
            assert_eq!(stage.decision_ticks, 10);
        }
    }

    #[test]
    fn v82_map_pool_and_env_targets_are_exact_and_versioned() {
        assert_eq!(
            V82_MAPS,
            [
                "Onion",
                "Pangaea",
                "Caucasus",
                "BlackSea",
                "BetweenTwoSeas",
                "Europe",
                "Asia",
                "World",
                "NorthAmerica",
                "SouthAmerica",
                "Africa",
                "Australia",
                "EastAsia",
                "MiddleEast",
                "Scandinavia",
                "GreatLakes",
            ]
        );
        let unique: std::collections::HashSet<_> = V82_MAPS.into_iter().collect();
        assert_eq!(unique.len(), V82_MAPS.len());
        assert!(V82_STAGE5_MAPS.iter().all(|map| V82_MAPS.contains(map)));
        assert_eq!(
            V82_ENV_TARGETS,
            [24, 24, 24, 24, 24, 16, 12, 10, 12, 10, 8, 8, 8, 8]
        );
        assert_eq!(
            V82_ENV_TARGETS.len(),
            stages_for_schedule(CurriculumSchedule::V82).len()
        );
        assert_eq!(CurriculumSchedule::V82.id(), "v8.2");
        assert_eq!(
            CurriculumSchedule::from_id("v8.2"),
            Some(CurriculumSchedule::V82)
        );
    }

    #[test]
    fn v83_inserts_exact_closeout_stage_and_shifts_v82_tail() {
        let v82 = stages_for_schedule(CurriculumSchedule::V82);
        let v83 = stages_for_schedule(CurriculumSchedule::V83);
        assert_eq!(v83.len(), 15);
        // Maps / difficulty / gates still follow the V8.2 ladder + closeout
        // insert; only bot/nation density is rewritten for OpenFront realism.
        for index in 0..5 {
            assert_eq!(v83[index].maps, v82[index].maps, "stage {index} maps");
            assert_eq!(
                v83[index].difficulty, v82[index].difficulty,
                "stage {index} difficulty"
            );
            assert_eq!(v83[index].win_at, v82[index].win_at, "stage {index} win_at");
            assert_eq!(
                v83[index].decision_ticks, v82[index].decision_ticks,
                "stage {index} decision_ticks"
            );
        }
        assert_eq!(v83[5].maps, &["Onion", "Pangaea", "Caucasus"]);
        assert_eq!(v83[5].difficulty, "Easy");
        assert_eq!(v83[5].decision_ticks, 10);
        assert_eq!(v83[5].win_at, 0.45);
        for (index, stage) in v83[6..].iter().enumerate() {
            let v82_stage = &v82[index + 5];
            assert_eq!(stage.maps, v82_stage.maps, "stage {} maps", index + 6);
            assert_eq!(
                stage.difficulty, v82_stage.difficulty,
                "stage {} difficulty",
                index + 6
            );
            assert_eq!(stage.win_at, v82_stage.win_at, "stage {} win_at", index + 6);
        }
        for (index, (stage, &(bots, nations))) in
            v83.iter().zip(V83_BOT_NATION_DENSITY.iter()).enumerate()
        {
            assert_eq!(stage.bots, bots, "stage {index} bots");
            assert_eq!(
                stage.nations,
                Nations::Exact(nations),
                "stage {index} nations"
            );
            assert!(
                stage.bots > nations * 5,
                "stage {index}: bots {} should stay >> nations {}",
                stage.bots,
                nations
            );
        }
        assert_eq!(v83[0].bots, 30);
        assert_eq!(v83[0].nations, Nations::Exact(5));
        assert_eq!(v83[14].bots, 200);
        assert_eq!(v83[14].nations, Nations::Exact(30));
        assert_eq!(
            V83_ENV_TARGETS,
            [24, 24, 24, 24, 24, 24, 16, 12, 10, 12, 10, 8, 8, 8, 8]
        );
        assert_eq!(V83_ENV_TARGETS.len(), v83.len());
        assert_eq!(CurriculumSchedule::V83.id(), "v8.3");
        assert_eq!(
            CurriculumSchedule::from_id("v8.3"),
            Some(CurriculumSchedule::V83)
        );
    }

    #[test]
    fn v10_softens_mid_ladder_density_and_keeps_closeout() {
        let v83 = stages_for_schedule(CurriculumSchedule::V83);
        let v10 = stages_for_schedule(CurriculumSchedule::V10);
        assert_eq!(v10.len(), 15);
        assert_eq!(v10[5].maps, &["Onion", "Pangaea", "Caucasus"]);
        assert_eq!(v10[5].win_at, 0.45);
        assert!(CurriculumSchedule::V10.uses_v83_closeout());
        assert_eq!(CurriculumSchedule::V10.id(), "v10");
        assert_eq!(
            CurriculumSchedule::from_id("v10"),
            Some(CurriculumSchedule::V10)
        );
        assert_eq!(V10_ENV_TARGETS.len(), v10.len());
        for (index, (stage, &(bots, nations))) in
            v10.iter().zip(V10_BOT_NATION_DENSITY.iter()).enumerate()
        {
            assert_eq!(stage.bots, bots, "stage {index} bots");
            assert_eq!(
                stage.nations,
                Nations::Exact(nations),
                "stage {index} nations"
            );
            assert!(
                stage.bots > nations * 5,
                "stage {index}: bots {} should stay >> nations {}",
                stage.bots,
                nations
            );
        }
        // Softened cliff vs V8.3 stage 8.
        assert!(v10[8].bots < v83[8].bots);
        assert_eq!(v10[8].bots, 100);
        // Medium does not drop bots below the prior Easy stage.
        assert!(v10[9].bots >= v10[8].bots);
        for index in 0..5 {
            assert_eq!(v10[index].maps, v83[index].maps);
            assert_eq!(v10[index].win_at, v83[index].win_at);
        }
    }

    #[test]
    fn sample_episode_rehearsal_uses_past_lobby_setup() {
        use rand::SeedableRng;
        use rand::rngs::SmallRng;
        let stages = stages_for_schedule(CurriculumSchedule::V10);
        let mut rng = SmallRng::seed_from_u64(42);
        let mut found = false;
        for _ in 0..200 {
            let (_map, bots, difficulty, nations, rehearsal) =
                sample_episode(&stages, 8, &mut rng);
            if !rehearsal {
                continue;
            }
            found = true;
            let matches_past = stages[..8].iter().any(|s| {
                s.bots == bots && s.difficulty == difficulty && s.nations == nations
            });
            assert!(
                matches_past,
                "rehearsal lobby must come from a past stage, got bots={bots} difficulty={difficulty:?} nations={nations:?}"
            );
            break;
        }
        assert!(found, "expected a rehearsal sample within 200 draws");
    }

    #[test]
    fn v9_sparse_ladder_is_gradual_with_high_gates() {
        let v9 = stages_for_schedule(CurriculumSchedule::V9);
        assert_eq!(v9.len(), 25);
        assert_eq!(V9_ENV_TARGETS.len(), v9.len());
        assert_eq!(CurriculumSchedule::V9.id(), "v9");
        assert_eq!(
            CurriculumSchedule::from_id("v9"),
            Some(CurriculumSchedule::V9)
        );
        // Bots always on, and always far outnumber nations (~6:1+). Never
        // Nations::Default (full map manifests invert the ratio on World/etc).
        assert!(v9.iter().all(|s| s.bots > 0));
        assert!(v9.iter().all(|s| !matches!(s.nations, Nations::Default)));
        for stage in &v9 {
            if let Nations::Exact(n) = stage.nations {
                if n > 0 {
                    assert!(
                        stage.bots >= 5 * n,
                        "bots {} must be >= 5x nations {} ({:?}/{})",
                        stage.bots,
                        n,
                        stage.maps,
                        stage.difficulty
                    );
                }
            }
        }
        assert_eq!(v9[0].bots, 15);
        assert_eq!(v9[0].nations, Nations::Exact(0));
        assert_eq!(v9[0].win_at, 0.975);
        assert_eq!(v9[1].bots, 30);
        assert_eq!(v9[1].nations, Nations::Exact(0));
        assert_eq!(v9[3].bots, 30);
        assert_eq!(v9[3].nations, Nations::Exact(5));
        assert_eq!(v9[14].bots, 200);
        assert_eq!(v9[14].nations, Nations::Exact(30));
        assert_eq!(v9[4].win_at, 0.925);
        assert!(v9.iter().all(|s| s.win_at >= 0.90 && s.win_at < 1.0));
        // Difficulty only steps after Easy player-count ladder completes.
        assert!(v9[..19].iter().all(|s| s.difficulty == "Easy"));
        assert_eq!(v9[19].difficulty, "Medium");
        assert_eq!(v9[22].difficulty, "Hard");
        assert_eq!(v9[24].difficulty, "Impossible");
        assert_eq!(v9[24].bots, 350);
        assert_eq!(v9[24].nations, Nations::Exact(50));
        assert_eq!(v9[24].maps, &V82_MAPS);
        // Bot counts are non-decreasing within each difficulty band.
        for window in v9[..19].windows(2) {
            assert!(window[1].bots >= window[0].bots);
        }
        assert_eq!(sparse_terminal_reward(true), V9_SPARSE_WIN);
        assert_eq!(sparse_terminal_reward(false), V9_SPARSE_LOSS);
    }

}
