//! Curriculum stages and the strength-based reward. Port of
//! `rl/curriculum.py`; see that file for the design rationale in comments.

use crate::feat::{
    A_ALLIANCE_REQUEST, A_ATTACK, A_BOAT, A_BREAK_ALLIANCE, A_BUILD, A_CANCEL_BOAT, A_DONATE_GOLD,
    A_DONATE_TROOPS, A_EMBARGO, A_EMBARGO_STOP, A_RETREAT, EntsData,
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
pub const LEGACY_V83_SCHEDULE_ID: &str = "v8.3";
pub const V86_REWARD_PROFILE: &str = "v8.6-attack-fair-v1";
/// V10 anti-death-spiral: dense V8.6-like reward + survival / anti-diplo / combat priors.
pub const V10_REWARD_PROFILE: &str = "v10-anti-spiral-v1";
/// Soft death default for V10 launches (override via `--v86-death-penalty`).
pub const V10_DEFAULT_DEATH_PENALTY: f64 = 3.0;
/// Reference mid-ladder gate (Medium/Hard band). Prefer [`v10_win_at_for_stage`].
pub const V10_WIN_AT: f64 = 0.70;
/// Bots-only early-ramp gate (stages `0 .. V10_NATION_INTRO_STAGE`).
/// 0.90 (need >36/40 under the strict `>` compare).
pub const V10_RAMP_WIN_AT: f64 = 0.90;
/// First stage that introduces a nation (see [`V10_BOT_NATION_DENSITY`]).
pub const V10_NATION_INTRO_STAGE: usize = 8;
/// Softened gate for the 1-nation band
/// (`V10_NATION_INTRO_STAGE .. V10_MULTI_NATION_STAGE`). Holding 0.90 here
/// pinned mid-ladder runs for thousands of updates.
pub const V10_ONE_NATION_WIN_AT: f64 = 0.80;
/// First stage with 2+ nations (end of the 1-nation win-gate band).
pub const V10_MULTI_NATION_STAGE: usize = 12;
/// Softened gate for 2+ nation Easy density ramp and the starting point of
/// the post-ramp smooth decay.
pub const V10_NATION_RAMP_WIN_AT: f64 = 0.75;
/// Terminal Impossible-stage gate after the smooth decay from
/// [`V10_NATION_RAMP_WIN_AT`].
pub const V10_WIN_AT_END: f64 = 0.65;
/// Floor for `lr * stage_lr_decay ^ stage`. Without this, mid-ladder stages
/// decay to ~1e-6 and learning stalls even when the win gate is reachable.
pub const V10_STAGE_LR_FLOOR: f64 = 1e-5;
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
    /// V10: per-decision survival shaping while alive (`coef * land_share`),
    /// tapered to zero across the closeout band so camping is not paid.
    pub v10_survival_coef: f64,
    /// V10: penalty magnitude for late/dominant diplo/donate panic actions.
    pub v10_diplo_panic: f64,
    /// V10: land-share threshold that arms diplo-panic shaping.
    pub v10_diplo_panic_share: f64,
    /// V10: tick/max_ticks fraction that arms diplo-panic shaping.
    pub v10_diplo_panic_tick_frac: f64,
    /// V10: bonus for productive combat/build/boat actions.
    pub v10_combat_action: f64,
    /// V10: terminal penalty magnitude when timing out after closeout entry.
    pub v10_timeout_closeout: f64,
    /// V10: one-shot bonus the first time land share crosses closeout entry (45%).
    pub v10_closeout_entry: f64,
    /// Duo: one-shot bonus the first time the two humans form a *formal*
    /// alliance this episode. Outcome-only (pact formed), never the
    /// `alliance_request` / `donate_*` actions. 0 disables.
    pub duo_pact_success: f64,
    /// Duo: Ng 1999 PBRS coefficient on log team gold-*income* (not gold
    /// stock). Cities/ports raise income; spending gold on a city does not
    /// drop this the way `K_ECO` gold-share does. 0 disables.
    pub duo_eco_coef: f64,
    /// Duo: one-shot when the team first owns a completed City. Outcome-only
    /// (structure exists), never the `build` action. 0 disables.
    pub duo_first_city: f64,
    /// Duo: one-shot when the team first owns a completed Port. Outcome-only.
    /// 0 disables.
    pub duo_first_port: f64,
    /// Duo: per completed City lost (count drop). Outcome-only, never the
    /// `delete_unit` action. 0 disables. Positive magnitude; callers subtract.
    pub duo_city_delete: f64,
    /// Duo: per completed Port lost (count drop). Outcome-only. 0 disables.
    pub duo_port_delete: f64,
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
            || self.v10_timeout_closeout != 0.0
            || self.v10_closeout_entry != 0.0
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

    /// Active V10 reward profile id for TrainState sidecars.
    pub fn reward_profile_id(self) -> &'static str {
        if self.v10_reward_active() {
            V10_REWARD_PROFILE
        } else if self.v86_reward_active() {
            V86_REWARD_PROFILE
        } else {
            V10_REWARD_PROFILE
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
    /// Team-mode duo: Ng-1999 PBRS on [`duo_potential`] (not per-tick wages).
    pub duo: f64,
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
        self.duo += other.duo;
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
/// this is a small categorical signal for the tile/action heads - not a
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
        || (has_sourced_attack && !(committed_troops > 0.0 && refund >= 0.5 * committed_troops))
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
pub fn tempo_pressure(tick: i64, max_ticks: i64, normalized_share: f64, threshold: f64) -> f64 {
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
    AllianceRequestBreak,
    BreakAllianceRequest,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ActionPairCounts {
    pub boat_cancel_boat: u64,
    pub embargo_embargo_stop: u64,
    pub attack_retreat: u64,
    pub retreat_attack: u64,
    pub alliance_request_break: u64,
    pub break_alliance_request: u64,
}

impl ActionPairCounts {
    pub fn record(&mut self, pair: InverseActionPair) {
        match pair {
            InverseActionPair::BoatCancelBoat => self.boat_cancel_boat += 1,
            InverseActionPair::EmbargoEmbargoStop => self.embargo_embargo_stop += 1,
            InverseActionPair::AttackRetreat => self.attack_retreat += 1,
            InverseActionPair::RetreatAttack => self.retreat_attack += 1,
            InverseActionPair::AllianceRequestBreak => self.alliance_request_break += 1,
            InverseActionPair::BreakAllianceRequest => self.break_alliance_request += 1,
        }
    }

    pub fn total(self) -> u64 {
        self.boat_cancel_boat
            + self.embargo_embargo_stop
            + self.attack_retreat
            + self.retreat_attack
            + self.alliance_request_break
            + self.break_alliance_request
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
        (A_ALLIANCE_REQUEST, A_BREAK_ALLIANCE) => Some(InverseActionPair::AllianceRequestBreak),
        (A_BREAK_ALLIANCE, A_ALLIANCE_REQUEST) => Some(InverseActionPair::BreakAllianceRequest),
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
    // Closeout-band churn is land-share gated only (no curriculum stage gate).
    if reversal.is_some()
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

/// V10 bridge pool: production maps with varied terrain and naval
/// pressure, without introducing the largest world-scale maps all at once.
pub const V10_BRIDGE_MAPS: [&str; 8] = [
    "Pangaea",
    "Europe",
    "Caucasus",
    "BlackSea",
    "BetweenTwoSeas",
    "Britannia",
    "GreatLakes",
    "Onion",
];

/// Broad V10 pool. These are `GameMapType` enum keys (the values accepted
/// by the Node bridge and normalized to asset-directory names by Rust).
/// Every entry must exist under `openfront/resources/maps/<lowercase>/`.
/// Order is not a preference - watch/showcase sample the pool; Onion is not first.
pub const V10_BROAD_MAPS: [&str; 16] = [
    "Pangaea",
    "Europe",
    "Asia",
    "World",
    "NorthAmerica",
    "SouthAmerica",
    "Africa",
    "Australia",
    "Caucasus",
    "BlackSea",
    "BetweenTwoSeas",
    "EastAsia",
    "MiddleEast",
    "Britannia",
    "GreatLakes",
    "Onion",
];

/// Stable curriculum identities persisted in trainer checkpoints.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CurriculumSchedule {
    #[default]
    /// Anti-death-spiral: V8.3 closeout ladder + demote / death gates / softer density.
    V10,
}

impl CurriculumSchedule {
    pub const fn id(self) -> &'static str {
        match self {
            Self::V10 => "v10",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "v10" => Some(Self::V10),
            _ => None,
        }
    }

    /// V10 always uses the closeout potential/churn behavior.
    pub const fn uses_v83_closeout(self) -> bool {
        true
    }
}

/// Early Easy density-ramp length (stages `0 .. V10_EASY_RAMP_LEN`): bots-only
/// then 1→4 nations. Maps are already mixed (bridge → broad); density is the
/// hard axis, not map identity.
pub const V10_EASY_RAMP_LEN: usize = 22;
/// Stages `0 .. V10_MAP_WARMUP_LEN` use the 8-map bridge pool before broad-16.
pub const V10_MAP_WARMUP_LEN: usize = 8;
/// First stage that samples the full 16-map broad pool.
pub const V10_BROAD_STAGE: usize = V10_MAP_WARMUP_LEN;
/// Oldest V10 sidecar length (pre-ramp 15-stage dense ladder).
pub const V10_PRE_RAMP_SIDECAR_LEN: usize = 15;
/// Short V10 sidecar length (20 Easy ramp + 15 dense = 35).
pub const V10_SHORT_SIDECAR_LEN: usize = 35;
/// Prior long V10 sidecar length (100-stage Easy micro-ramp table).
pub const V10_PRIOR_SIDECAR_LEN: usize = 100;
/// Full V10 ladder: faster Easy → Medium → Hard → Impossible.
/// Medium/Hard/Impossible lobby rows match the prior 100-stage table; only
/// Easy is compressed (fewer +2-bot micro-stages).
pub const V10_STAGE_COUNT: usize = 68;
/// Easy density conversion-gate milestone (maps already broad by here).
pub const V10_CLOSEOUT_STAGE: usize = 28;
/// Later Easy conversion-gate milestone (maps already broad by here).
pub const V10_BRIDGE_STAGE: usize = 31;
/// First Medium stage.
pub const V10_MEDIUM_START: usize = 36;
/// First Hard stage.
pub const V10_HARD_START: usize = 50;
/// First Impossible stage.
pub const V10_IMPOSSIBLE_START: usize = 60;

/// V10 density: compressed Easy ramp, then densify into Medium+.
///
/// At each difficulty jump (Easy→Medium, Medium→Hard, Hard→Impossible) nation
/// count **resets low** so the policy learns the new bot strength with a sparse
/// lobby, then nations ramp back up inside that band. Keeps bots ≫ nations
/// once nations appear. Map pools are applied separately in [`build_v10_stages`]
/// (bridge→broad).
pub const V10_BOT_NATION_DENSITY: [(u32, u32); V10_STAGE_COUNT] = [
    // --- Easy density ramp (0-21): bots-only → 1n → 2n → 3n → 4n ---
    (2, 0),  // 0
    (4, 0),  // 1
    (7, 0),  // 2
    (10, 0), // 3
    (14, 0), // 4
    (18, 0), // 5
    (22, 0), // 6
    (26, 0), // 7
    (18, 1), // 8 introduce 1 nation
    (22, 1), // 9
    (26, 1), // 10
    (30, 1), // 11
    (22, 2), // 12
    (26, 2), // 13
    (30, 2), // 14
    (34, 2), // 15
    (26, 3), // 16
    (30, 3), // 17
    (34, 3), // 18
    (30, 4), // 19
    (34, 4), // 20
    (38, 4), // 21
    // --- Easy densify (22-27) ---
    (42, 5), // 22
    (48, 5), // 23
    (52, 6), // 24
    (58, 7), // 25
    (66, 8), // 26
    (70, 9), // 27
    // --- closeout Easy (28-30) ---
    (54, 6), // 28 CLOSEOUT
    (60, 7), // 29
    (68, 8), // 30
    // --- bridge-gate Easy (31-33) ---
    (76, 9),  // 31 BRIDGE
    (82, 10), // 32
    (88, 10), // 33
    // --- peak Easy (34-35) ---
    (102, 12), // 34
    (118, 14), // 35 peak Easy nations before Medium reset
    // --- Medium (36-49): same lobbies as prior stages 68-81 ---
    (90, 4),   // 36 MEDIUM_START nation reset
    (94, 5),   // 37
    (98, 6),   // 38
    (102, 7),  // 39
    (106, 8),  // 40
    (110, 9),  // 41
    (114, 10), // 42
    (118, 11), // 43
    (122, 12), // 44
    (126, 13), // 45
    (130, 14), // 46
    (134, 15), // 47
    (138, 16), // 48
    (142, 16), // 49 peak Medium nations before Hard reset
    // --- Hard (50-59): same lobbies as prior stages 82-91 ---
    (130, 6),  // 50 HARD_START nation reset
    (134, 7),  // 51
    (138, 8),  // 52
    (142, 9),  // 53
    (146, 10), // 54
    (150, 12), // 55
    (154, 14), // 56
    (158, 16), // 57
    (162, 18), // 58
    (166, 18), // 59 peak Hard nations before Impossible reset
    // --- Impossible (60-67): same lobbies as prior stages 92-99 ---
    (155, 8),  // 60 IMPOSSIBLE_START nation reset
    (160, 10), // 61
    (165, 12), // 62
    (170, 14), // 63
    (175, 16), // 64
    (180, 18), // 65
    (185, 20), // 66
    (190, 22), // 67
];

/// Prior long-table densities (for sidecar remap onto the current stage table).
pub const V10_PRIOR_BOT_NATION_DENSITY: [(u32, u32); V10_PRIOR_SIDECAR_LEN] = [
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 0),
    (6, 0),
    (7, 0),
    (8, 0),
    (9, 0),
    (10, 0),
    (12, 0),
    (14, 0),
    (16, 0),
    (18, 0),
    (20, 0),
    (22, 0),
    (16, 1),
    (18, 1),
    (20, 1),
    (22, 1),
    (24, 1),
    (20, 2),
    (22, 2),
    (24, 2),
    (26, 2),
    (22, 3),
    (24, 3),
    (26, 3),
    (24, 4),
    (26, 4),
    (28, 4),
    (30, 5),
    (32, 5),
    (34, 5),
    (36, 5),
    (38, 5),
    (40, 5),
    (42, 5),
    (44, 5),
    (40, 5),
    (44, 5),
    (48, 6),
    (52, 6),
    (50, 6),
    (54, 6),
    (58, 7),
    (62, 7),
    (66, 8),
    (70, 9),
    (50, 6),
    (54, 6),
    (58, 7),
    (60, 7),
    (64, 8),
    (68, 8),
    (70, 9),
    (76, 9),
    (82, 10),
    (88, 10),
    (90, 11),
    (94, 11),
    (98, 12),
    (102, 12),
    (106, 13),
    (110, 13),
    (112, 14),
    (114, 14),
    (116, 14),
    (118, 14),
    (90, 4),
    (94, 5),
    (98, 6),
    (102, 7),
    (106, 8),
    (110, 9),
    (114, 10),
    (118, 11),
    (122, 12),
    (126, 13),
    (130, 14),
    (134, 15),
    (138, 16),
    (142, 16),
    (130, 6),
    (134, 7),
    (138, 8),
    (142, 9),
    (146, 10),
    (150, 12),
    (154, 14),
    (158, 16),
    (162, 18),
    (166, 18),
    (155, 8),
    (160, 10),
    (165, 12),
    (170, 14),
    (175, 16),
    (180, 18),
    (185, 20),
    (190, 22),
];

/// Smooth win-rate gate: hold [`V10_RAMP_WIN_AT`] while bots-only, soften once
/// nations appear, then ease from [`V10_NATION_RAMP_WIN_AT`] down to
/// [`V10_WIN_AT_END`] by the final Impossible stage.
pub fn v10_win_at_for_stage(index: usize) -> f64 {
    debug_assert!(index < V10_STAGE_COUNT);
    if index < V10_NATION_INTRO_STAGE {
        return V10_RAMP_WIN_AT;
    }
    if index < V10_MULTI_NATION_STAGE {
        return V10_ONE_NATION_WIN_AT;
    }
    if index < V10_EASY_RAMP_LEN {
        return V10_NATION_RAMP_WIN_AT;
    }
    let span = (V10_STAGE_COUNT - 1 - V10_EASY_RAMP_LEN) as f64;
    let t = (index - V10_EASY_RAMP_LEN) as f64 / span;
    let s = t * t * (3.0 - 2.0 * t);
    V10_NATION_RAMP_WIN_AT - s * (V10_NATION_RAMP_WIN_AT - V10_WIN_AT_END)
}

/// `max(floor, base_lr * decay ^ stage)` - shared by advance/demote/resume.
pub fn stage_learning_rate(base_lr: f64, decay: f64, stage: usize, floor: f64) -> f64 {
    (base_lr * decay.powi(stage as i32)).max(floor)
}

/// Invert uncapped `base * decay ^ stage` to recover an implied stage.
/// Used to detect sidecars whose `stage` was rewritten downward while
/// `lr_now` still reflected a much higher stage (the u11571 28→8 cliff).
pub fn imply_stage_from_learning_rate(lr_now: f64, base_lr: f64, decay: f64) -> Option<usize> {
    if !(base_lr > 0.0 && lr_now > 0.0 && (0.0 < decay && decay < 1.0)) {
        return None;
    }
    let ratio = lr_now / base_lr;
    if ratio > 1.0 + 1e-9 {
        return Some(0);
    }
    let stage = (ratio.ln() / decay.ln()).round();
    if !stage.is_finite() || stage < 0.0 {
        return None;
    }
    Some((stage as usize).min(V10_STAGE_COUNT - 1))
}

/// Remap a short (35-slot) V10 sidecar index onto the current stage table.
pub fn remap_v10_short_sidecar_stage(old: usize) -> usize {
    let old = old.min(V10_SHORT_SIDECAR_LEN - 1);
    if old < 20 {
        (old * V10_EASY_RAMP_LEN) / 20
    } else {
        let legacy = old - 20;
        V10_EASY_RAMP_LEN + (legacy * (V10_STAGE_COUNT - V10_EASY_RAMP_LEN - 1)) / 14
    }
}

/// Remap a prior long-table V10 sidecar index onto the current stage table by
/// matching lobby density (bots, nations). Prefers the same nation count and
/// the smallest bots ≥ the old lobby so progress never moves backwards.
pub fn remap_v10_prior_sidecar_stage(old: usize) -> usize {
    let old = old.min(V10_PRIOR_SIDECAR_LEN - 1);
    let (bots, nations) = V10_PRIOR_BOT_NATION_DENSITY[old];
    remap_density_to_stage(bots, nations)
}

fn remap_density_to_stage(bots: u32, nations: u32) -> usize {
    if let Some((index, _)) = V10_BOT_NATION_DENSITY
        .iter()
        .enumerate()
        .find(|&(_, &(b, n))| b == bots && n == nations)
    {
        return index;
    }
    if let Some((index, _)) = V10_BOT_NATION_DENSITY
        .iter()
        .enumerate()
        .filter(|&(_, &(_, n))| n == nations)
        .filter(|&(_, &(b, _))| b >= bots)
        .min_by_key(|&(_, &(b, _))| b)
    {
        return index;
    }
    V10_BOT_NATION_DENSITY
        .iter()
        .enumerate()
        .min_by_key(|&(_, &(b, n))| (n.abs_diff(nations), b.abs_diff(bots)))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn apply_v10_stage_params(stages: &mut [Stage]) {
    debug_assert_eq!(stages.len(), V10_BOT_NATION_DENSITY.len());
    for (index, (stage, &(bots, nations))) in stages
        .iter_mut()
        .zip(V10_BOT_NATION_DENSITY.iter())
        .enumerate()
    {
        stage.bots = bots;
        stage.nations = Nations::Exact(nations);
        stage.win_at = v10_win_at_for_stage(index);
    }
}

/// Build the full V10 ladder (maps/difficulty/cadence + density/gates).
///
/// Map variety is introduced early so the policy cannot overfit Onion:
/// stages 0-7 sample the 8-map bridge pool, then stages 8+ use the full
/// 16-map broad pool. Opponent density (bots/nations) remains the hard axis.
fn build_v10_stages() -> Vec<Stage> {
    use Nations::Exact as NE;
    let mut stages = Vec::with_capacity(V10_STAGE_COUNT);
    let mut push = |maps: &'static [&'static str],
                    difficulty: &'static str,
                    decision_ticks: u32,
                    count: usize| {
        for _ in 0..count {
            stages.push(Stage {
                maps,
                bots: 0,
                difficulty,
                nations: NE(0),
                decision_ticks,
                win_at: V10_RAMP_WIN_AT,
            });
        }
    };
    // Cadence is permanently 15 ticks/decision for every V10 stage (train,
    // watch, replay). Do not reintroduce a faster late-Easy / Medium cadence —
    // the dt=10 experiment cratered the live policy.
    // 0-7: 8-map warm-up (terrain/naval variety without World/Asia yet)
    push(&V10_BRIDGE_MAPS, "Easy", 15, V10_MAP_WARMUP_LEN);
    // 8-27: full broad-16 through early/mid Easy density
    push(&V10_BROAD_MAPS, "Easy", 15, 20);
    // 28-35: broad Easy closeout → peak
    push(&V10_BROAD_MAPS, "Easy", 15, 8);
    push(&V10_BROAD_MAPS, "Medium", 15, 14); // 36-49
    push(&V10_BROAD_MAPS, "Hard", 15, 10); // 50-59
    push(&V10_BROAD_MAPS, "Impossible", 15, 8); // 60-67
    debug_assert_eq!(stages.len(), V10_STAGE_COUNT);
    apply_v10_stage_params(&mut stages);
    stages
}

/// V10 env floors: saturated early Easy density ramp, taper as bots grow.
///
/// These are *aspirational* per-shard floors for larger GPUs. On current
/// 46 GB A40 pods, `pod_train_v10.sh`'s `MAX_ENVS` (default 14) is the
/// hard VRAM ceiling - oftrain clamps resize requests to `--max-envs` so
/// a floor of 24 does not trigger noop cold restarts. Raising throughput
/// means freeing Obs VRAM (see compact Half-resident grids), not editing
/// this table down to the A40 cap.
pub const V10_ENV_TARGETS: [usize; V10_STAGE_COUNT] = [
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, // 0-9
    24, 24, 24, 24, 24, 24, 24, 24, 24, 24, // 10-19
    24, 24, 24, 24, 24, 24, 24, 24, 20, 20, // 20-29
    20, 16, 16, 16, 12, 12, 10, 10, 10, 10, // 30-39
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, // 40-49
    8, 8, 8, 8, 8, 8, 8, 8, 8, 8, // 50-59
    8, 8, 8, 8, 8, 8, 8, 8, // 60-67
];

pub fn stages_for_schedule(_schedule: CurriculumSchedule) -> Vec<Stage> {
    build_v10_stages()
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

/// Terminal outcome. Timeout without a win is a loss (`−W_WIN`), not a
/// placement gift: paying `W_PLACE * place^{-p}` on the clock made camping
/// until `max_episode_ticks` the PPO optimum (v4 safe-second / duo donate
/// farm). The same rule applies when the caller flags a never-spawned
/// human (`timed_out=true`): they did not play, so this is a death/loss,
/// not a placement. AlphaStar's +1/0/−1 treats timeout as failure; this
/// is the same rule at the existing win-bonus scale so a real win still
/// dominates.
pub fn terminal_reward(place: i64, won: bool, timed_out: bool) -> f64 {
    if timed_out && !won {
        return -W_WIN;
    }
    let mut r = W_PLACE * (place as f64).powf(-PLACE_POW);
    if won {
        r += W_WIN;
    }
    r
}

/// Alive+land survival shaping: small positive signal so death is not the only
/// non-win terminal contrast under a softer death penalty.
///
/// Tapers to zero across the closeout band `[0.45, 0.80]` so once the agent is
/// dominant, camping no longer accrues survival pay - finish the win instead.
pub fn v10_survival_reward(alive: bool, land_share: f64, config: RewardConfig) -> f64 {
    if !alive || config.v10_survival_coef == 0.0 {
        return 0.0;
    }
    let share = finite_or_zero(land_share).clamp(0.0, 1.0);
    let taper = if share <= V83_CLOSEOUT_SHARE_START {
        1.0
    } else if share >= V83_CLOSEOUT_SHARE_FULL {
        0.0
    } else {
        (V83_CLOSEOUT_SHARE_FULL - share) / (V83_CLOSEOUT_SHARE_FULL - V83_CLOSEOUT_SHARE_START)
    };
    config.v10_survival_coef * share * taper
}

/// Terminal stick for timing out after entering closeout (≥45% land) without
/// converting the win. Flat magnitude; disabled when coef is 0.
pub fn v10_timeout_after_closeout_penalty(
    timed_out: bool,
    closeout_reached: bool,
    config: RewardConfig,
) -> f64 {
    if timed_out && closeout_reached && config.v10_timeout_closeout != 0.0 {
        -config.v10_timeout_closeout.abs()
    } else {
        0.0
    }
}

/// One-shot mid-game milestone: pay when land share first crosses closeout
/// entry (45%). Disabled when coef is 0.
pub fn v10_closeout_entry_bonus(just_entered: bool, config: RewardConfig) -> f64 {
    if just_entered && config.v10_closeout_entry != 0.0 {
        config.v10_closeout_entry.abs()
    } else {
        0.0
    }
}

/// One-shot when teammates first form a formal pact this episode.
/// Disabled when `duo_pact_success` is 0. Callers must pass `just_formed`
/// only on the transition into `formally_allied`, never on the
/// `alliance_request` action itself and never per-tick while allied.
pub fn duo_pact_success_bonus(just_formed: bool, config: RewardConfig) -> f64 {
    if just_formed && config.duo_pact_success != 0.0 {
        config.duo_pact_success.abs()
    } else {
        0.0
    }
}

/// One-shot when the team first owns a completed City or Port.
/// Disabled when `amount` is 0. Callers must pass `just_completed` only on
/// the 0→1 transition, never the `build` action and never per-tick while
/// the structure stands (that would be a camping wage).
pub fn duo_first_structure_bonus(just_completed: bool, amount: f64) -> f64 {
    if just_completed && amount != 0.0 {
        amount.abs()
    } else {
        0.0
    }
}

/// Penalty when the team's completed City/Port count drops. `dropped` is
/// `prev.saturating_sub(now)`. Disabled when `amount` is 0. Never keyed off
/// the `delete_unit` action (that would be donate-dirac with extra steps).
pub fn duo_structure_delete_penalty(dropped: usize, amount: f64) -> f64 {
    if dropped == 0 || amount == 0.0 {
        0.0
    } else {
        -amount.abs() * dropped as f64
    }
}

/// City / Port class indices (`feat::unit_class`).
pub const CITY_UNIT_CLASS: usize = 0;
pub const PORT_UNIT_CLASS: usize = 1;
/// Normalize log-income Φ into ~[0, 1] before `duo_eco_coef`. Ally train
/// gold is 35k; a couple of cities sit in this ballpark.
pub const ECO_INCOME_REF: f64 = 100_000.0;

/// Team gold-*income* potential. Buildings raise income; cashing gold into
/// a city does not drop this (unlike `K_ECO` which is gold *stock* share).
/// Ng 1999: apply via [`DominanceShaper`], absorbing Φ=0 at done.
pub fn economy_potential(income_a: f64, income_b: f64, coef: f64) -> f64 {
    if coef == 0.0 {
        return 0.0;
    }
    let a = finite_or_zero(income_a).max(0.0);
    let b = finite_or_zero(income_b).max(0.0);
    let norm = (1.0 + ECO_INCOME_REF).ln();
    coef * ((1.0 + a).ln() + (1.0 + b).ln()) * 0.5 / norm
}

pub fn player_gold_income(ents: &crate::feat::EntsData, pid: usize) -> f64 {
    ents.players
        .iter()
        .find(|p| p.id == pid)
        .map(|p| finite_or_zero(p.gold_income).max(0.0))
        .unwrap_or(0.0)
}

/// Completed (not-under-construction) structures of `class` owned by any
/// of `owners` (agent small-ids).
pub fn team_completed_structures(
    ents: &crate::feat::EntsData,
    owners: &[usize],
    class: usize,
) -> usize {
    ents.units
        .iter()
        .filter(|u| u.class == class && !u.constructing && owners.iter().any(|&o| o == u.owner))
        .count()
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
        A_DONATE_GOLD | A_DONATE_TROOPS | A_ALLIANCE_REQUEST | A_BREAK_ALLIANCE | A_EMBARGO
        | A_EMBARGO_STOP => -config.v10_diplo_panic.abs(),
        _ => 0.0,
    }
}

/// Small prior for productive combat/build actions (targeted attack/boat, build).
///
/// `has_target` must mean the action actually emitted a usable intent
/// (translated tile / player), not merely that the policy sampled a head
/// value. Paying boat/build bonuses on empty translates made empty boats
/// net-positive vs waste and suppressed real builds.
pub fn v10_combat_action_bonus(action: i64, has_target: bool, config: RewardConfig) -> f64 {
    if config.v10_combat_action == 0.0 {
        return 0.0;
    }
    let coef = config.v10_combat_action.abs();
    match action {
        A_ATTACK if has_target => coef,
        A_BOAT if has_target => coef * 0.75,
        A_BUILD if has_target => coef * 0.5,
        _ => 0.0,
    }
}

/// Net shaping an empty boat/build translate used to receive under V10:
/// combat-action bonus counted `tile_region.is_some()` as a target, so a
/// wasted empty boat was `+0.015 - W_WASTE = +0.005` (free EV). Builds
/// were `+0.01 - W_WASTE = 0`. Callers must require a real emitted intent.
pub fn v10_empty_action_net_reward(action: i64, config: RewardConfig) -> f64 {
    let bogus_bonus = v10_combat_action_bonus(action, true, config);
    bogus_bonus - W_WASTE
}

/// Scale applied to the inherited 1vN shaping terms when two co-trained
/// humans share a Team-mode match, so team-win / welfare / real-alliance
/// synergy can dominate.
pub const DUO_SOLO_SCALE: f64 = 0.22;
/// Φ weight while both humans are alive. **Not a per-tick wage** — callers
/// must apply [`duo_potential`] through [`DominanceShaper`] (Ng 1999 PBRS:
/// `γ Φ(s') − Φ(s)`). Paying this every decision was a camping salary
/// (~1400 steps × 0.02) that made timeout EV beat a rare win.
pub const W_DUO_BOTH_ALIVE: f64 = 0.02;
/// `min(s1,s2)` welfare term inside [`duo_potential`].
pub const W_DUO_WELFARE_MIN: f64 = 0.015;
/// Geometric-mean welfare term inside [`duo_potential`].
pub const W_DUO_GEO: f64 = 0.01;
/// `|s1-s2|/(s1+s2)` inequity tax inside [`duo_potential`].
pub const W_DUO_INEQUITY: f64 = 0.02;
/// Φ weight while the pair has a *formal* alliance (not merely the same
/// team). PBRS pays the *transition* into a pact (so they still learn to
/// request for ally train gold) without a per-tick allied wage.
///
/// Outcome-only (devlog boat-churn rule): pay this *state* as a potential,
/// never the `alliance_request` / `donate_*` actions. Do not reintroduce
/// `W_DUO_ALLY_REQUEST` / `W_DUO_DONATE_PARTNER`.
pub const W_DUO_ALLIED: f64 = 0.05;

/// True when `ents.alliances` contains a formal pact between `a` and `b`
/// (small ids). Team membership alone is not an alliance.
pub fn formally_allied(ents: &crate::feat::EntsData, a: usize, b: usize) -> bool {
    if a == b {
        return false;
    }
    ents.alliances
        .iter()
        .any(|al| (al.0 == a && al.1 == b) || (al.0 == b && al.1 == a))
}

/// Min + geo-mean welfare minus inequity. A *state potential* Φ term, not
/// a per-tick wage — see [`duo_potential`].
pub fn duo_welfare_reward(s1: f64, s2: f64) -> f64 {
    let mn = s1.min(s2);
    let geo = (s1.max(0.0) * s2.max(0.0)).sqrt();
    let ineq = (s1 - s2).abs() / (s1 + s2 + 1e-9);
    W_DUO_WELFARE_MIN * mn + W_DUO_GEO * geo - W_DUO_INEQUITY * ineq
}

/// Survival + formal-alliance synergy as a *state potential*. `allied`
/// must be a real pact. Callers apply this via PBRS, not as a wage.
pub fn duo_synergy_reward(both_alive: bool, allied: bool) -> f64 {
    let mut r = 0.0;
    if both_alive {
        r += W_DUO_BOTH_ALIVE;
    }
    if allied && both_alive {
        r += W_DUO_ALLIED;
    }
    r
}

/// Team-mode state potential Φ: welfare + both-alive + formal pact.
/// Ng 1999: the policy-invariant shaping is `F = γ Φ(s') − Φ(s)`
/// ([`DominanceShaper::transition`] with coefficient 1). Camping at a
/// constant Φ (timeout farm) telescopes to ~0; forming a pact or growing
/// together is a one-shot delta.
pub fn duo_potential(s1: f64, s2: f64, both_alive: bool, allied: bool) -> f64 {
    finite_or_zero(duo_welfare_reward(s1, s2) + duo_synergy_reward(both_alive, allied))
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
            v10_survival_coef: 0.0,
            v10_diplo_panic: 0.0,
            v10_diplo_panic_share: 0.35,
            v10_diplo_panic_tick_frac: 0.55,
            v10_combat_action: 0.0,
            v10_timeout_closeout: 0.0,
            v10_closeout_entry: 0.0,
            duo_pact_success: 0.0,
            duo_eco_coef: 0.0,
            duo_first_city: 0.0,
            duo_first_port: 0.0,
            duo_city_delete: 0.0,
            duo_port_delete: 0.0,
        }
    }

    fn composite(values: &[(usize, f64)]) -> HashMap<usize, f64> {
        values.iter().copied().collect()
    }

    #[test]
    fn v10_decision_ticks_are_uniformly_fifteen() {
        let stages = stages_for_schedule(CurriculumSchedule::V10);
        assert_eq!(stages.len(), V10_STAGE_COUNT);
        for (i, stage) in stages.iter().enumerate() {
            assert_eq!(
                stage.decision_ticks, 15,
                "stage {i} decision_ticks={}",
                stage.decision_ticks
            );
        }
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
        // Closeout-band churn is land-share gated only (any curriculum stage).
        assert_eq!(v83_action_churn_penalty(pair, 0, 0.9, cfg), -0.06);
        assert_eq!(v83_action_churn_penalty(pair, 4, 0.9, cfg), -0.06);
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
        assert_eq!(
            strength_delta_weight(-0.1, 0.9, 3, cfg, false),
            W_DELTA_LOSS
        );
    }

    #[test]
    fn dominant_threshold_relaxes_only_losses_at_or_above_threshold() {
        let cfg = config();
        assert_eq!(
            strength_delta_weight(-0.1, 0.549, 4, cfg, false),
            W_DELTA_LOSS
        );
        assert_eq!(
            strength_delta_weight(-0.1, 0.55, 4, cfg, false),
            cfg.v81_delta_loss_dominant
        );
        assert_eq!(
            strength_delta_weight(0.1, 0.99, 4, cfg, false),
            W_DELTA_GAIN
        );
    }

    #[test]
    fn v86_softens_loss_and_symmetrizes_during_active_attack() {
        let mut cfg = config();
        cfg.v86_delta_loss = 5.5;
        cfg.v86_attack_symmetric_loss = true;
        assert_eq!(strength_delta_weight(-0.1, 0.1, 4, cfg, false), 5.5);
        assert_eq!(strength_delta_weight(-0.1, 0.1, 4, cfg, true), W_DELTA_GAIN);
        assert_eq!(cfg.death_penalty(), W_DEATH);
        cfg.v86_death_penalty = 10.0;
        assert_eq!(cfg.death_penalty(), 10.0);
        assert_eq!(cfg.reward_profile_id(), V86_REWARD_PROFILE);
    }

    #[test]
    fn v10_reward_profile_and_shaping_helpers() {
        let mut cfg = config();
        cfg.v10_survival_coef = 0.01;
        cfg.v10_diplo_panic = 0.08;
        cfg.v10_diplo_panic_share = 0.35;
        cfg.v10_diplo_panic_tick_frac = 0.55;
        cfg.v10_combat_action = 0.02;
        cfg.v10_timeout_closeout = 20.0;
        cfg.v10_closeout_entry = 25.0;
        cfg.v86_death_penalty = 3.0;
        assert_eq!(cfg.reward_profile_id(), V10_REWARD_PROFILE);
        assert_eq!(cfg.death_penalty(), 3.0);
        // Below closeout: full survival. Inside band: tapered. At/above full: 0.
        assert!((v10_survival_reward(true, 0.20, cfg) - 0.002).abs() < 1e-12);
        let mid = v10_survival_reward(true, 0.5, cfg);
        assert!(mid > 0.0 && mid < 0.005);
        assert!((mid - 0.01 * 0.5 * ((0.80 - 0.5) / 0.35)).abs() < 1e-12);
        assert_eq!(v10_survival_reward(true, 0.80, cfg), 0.0);
        assert_eq!(v10_survival_reward(false, 0.5, cfg), 0.0);
        assert_eq!(v10_timeout_after_closeout_penalty(true, true, cfg), -20.0);
        assert_eq!(v10_timeout_after_closeout_penalty(true, false, cfg), 0.0);
        assert_eq!(v10_timeout_after_closeout_penalty(false, true, cfg), 0.0);
        assert_eq!(v10_closeout_entry_bonus(true, cfg), 25.0);
        assert_eq!(v10_closeout_entry_bonus(false, cfg), 0.0);
        assert_eq!(
            v10_diplo_panic_penalty(A_DONATE_GOLD, 0.40, 100, 1000, cfg),
            -0.08
        );
        assert_eq!(v10_diplo_panic_penalty(A_ATTACK, 0.40, 100, 1000, cfg), 0.0);
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
        assert_eq!(v10_combat_action_bonus(A_BOAT, false, cfg), 0.0);
        assert_eq!(v10_combat_action_bonus(A_BUILD, true, cfg), 0.01);
        assert_eq!(v10_combat_action_bonus(A_BUILD, false, cfg), 0.0);
        // Historical bug: counting sampled tile heads as targets made empty
        // boats net-positive and empty builds reward-neutral.
        assert!(v10_empty_action_net_reward(A_BOAT, cfg) > 0.0);
        assert_eq!(v10_empty_action_net_reward(A_BUILD, cfg), 0.0);
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
            (
                player_action(A_ALLIANCE_REQUEST, 3),
                player_action(A_BREAK_ALLIANCE, 3),
                InverseActionPair::AllianceRequestBreak,
            ),
            (
                player_action(A_BREAK_ALLIANCE, 3),
                player_action(A_ALLIANCE_REQUEST, 3),
                InverseActionPair::BreakAllianceRequest,
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
        assert_eq!(boat_outcome_reward(BoatOutcome::UsefulLanding, cfg), 0.15);
        assert_eq!(boat_outcome_reward(BoatOutcome::Destroyed, cfg), -0.20);
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

    #[test]
    fn v10_stage_curve_has_one_nation_band_and_smooth_gates() {
        let v10 = stages_for_schedule(CurriculumSchedule::V10);
        assert_eq!(v10.len(), V10_STAGE_COUNT);
        assert_eq!(V10_EASY_RAMP_LEN, 22);
        assert_eq!(V10_MAP_WARMUP_LEN, 8);
        assert_eq!(V10_BROAD_STAGE, 8);
        assert_eq!(V10_NATION_INTRO_STAGE, 8);
        assert_eq!(V10_MULTI_NATION_STAGE, 12);
        assert_eq!(V10_CLOSEOUT_STAGE, 28);
        assert_eq!(V10_BRIDGE_STAGE, 31);
        assert_eq!(V10_MEDIUM_START, 36);
        assert_eq!(V10_HARD_START, 50);
        assert_eq!(V10_IMPOSSIBLE_START, 60);
        assert_eq!(v10[0].bots, 2);
        assert_eq!(v10[0].nations, Nations::Exact(0));
        // Early map variety: bridge-8 warm-up, then broad-16 for the rest.
        // Scandinavia is not a GameMapType / asset dir - pools must stay valid.
        assert!(!V10_BRIDGE_MAPS.contains(&"Scandinavia"));
        assert!(!V10_BROAD_MAPS.contains(&"Scandinavia"));
        assert_eq!(v10[0].maps, &V10_BRIDGE_MAPS);
        assert_eq!(v10[V10_MAP_WARMUP_LEN - 1].maps, &V10_BRIDGE_MAPS);
        assert_eq!(v10[V10_BROAD_STAGE].maps, &V10_BROAD_MAPS);
        assert_eq!(v10[V10_CLOSEOUT_STAGE].maps, &V10_BROAD_MAPS);
        assert_eq!(v10[V10_BRIDGE_STAGE].maps, &V10_BROAD_MAPS);
        assert_eq!(v10[V10_MEDIUM_START].maps, &V10_BROAD_MAPS);
        assert_eq!(v10[V10_MEDIUM_START].difficulty, "Medium");
        assert_eq!(v10[V10_HARD_START].difficulty, "Hard");
        assert_eq!(v10[V10_IMPOSSIBLE_START].difficulty, "Impossible");
        // Medium/Hard/Impossible lobbies unchanged vs the prior 100-stage table.
        assert_eq!(
            &V10_BOT_NATION_DENSITY[V10_MEDIUM_START..],
            &V10_PRIOR_BOT_NATION_DENSITY[68..]
        );
        // Difficulty jumps reset nations low, then each band ramps back up.
        let Nations::Exact(easy_peak_n) = v10[V10_MEDIUM_START - 1].nations else {
            panic!("expected Exact nations");
        };
        let Nations::Exact(med_start_n) = v10[V10_MEDIUM_START].nations else {
            panic!("expected Exact nations");
        };
        let Nations::Exact(med_peak_n) = v10[V10_HARD_START - 1].nations else {
            panic!("expected Exact nations");
        };
        let Nations::Exact(hard_start_n) = v10[V10_HARD_START].nations else {
            panic!("expected Exact nations");
        };
        let Nations::Exact(hard_peak_n) = v10[V10_IMPOSSIBLE_START - 1].nations else {
            panic!("expected Exact nations");
        };
        let Nations::Exact(imp_start_n) = v10[V10_IMPOSSIBLE_START].nations else {
            panic!("expected Exact nations");
        };
        assert!(
            med_start_n * 2 <= easy_peak_n,
            "Medium start nations {med_start_n} should drop hard from Easy peak {easy_peak_n}"
        );
        assert!(
            hard_start_n * 2 <= med_peak_n,
            "Hard start nations {hard_start_n} should drop hard from Medium peak {med_peak_n}"
        );
        assert!(
            imp_start_n * 2 <= hard_peak_n,
            "Impossible start nations {imp_start_n} should drop hard from Hard peak {hard_peak_n}"
        );
        assert!(med_peak_n > med_start_n, "Medium band should ramp nations");
        assert!(hard_peak_n > hard_start_n, "Hard band should ramp nations");
        let Nations::Exact(imp_end_n) = v10[V10_STAGE_COUNT - 1].nations else {
            panic!("expected Exact nations");
        };
        assert!(
            imp_end_n > imp_start_n,
            "Impossible band should ramp nations"
        );
        assert!(CurriculumSchedule::V10.uses_v83_closeout());
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
                nations == 0 || stage.bots > nations * 5,
                "stage {index}: bots {} should stay >> nations {}",
                stage.bots,
                nations
            );
            let expect = v10_win_at_for_stage(index);
            assert!(
                (stage.win_at - expect).abs() < 1e-9,
                "stage {index} win_at {} != {}",
                stage.win_at,
                expect
            );
            if index < V10_BROAD_STAGE {
                assert_eq!(stage.maps, &V10_BRIDGE_MAPS, "stage {index} maps");
            } else {
                assert_eq!(stage.maps, &V10_BROAD_MAPS, "stage {index} maps");
            }
        }
        for index in 0..V10_NATION_INTRO_STAGE {
            assert_eq!(v10[index].nations, Nations::Exact(0), "stage {index}");
        }
        for index in V10_NATION_INTRO_STAGE..V10_MULTI_NATION_STAGE {
            assert_eq!(v10[index].nations, Nations::Exact(1), "stage {index}");
        }
        assert_eq!(v10[V10_MULTI_NATION_STAGE].nations, Nations::Exact(2));
        assert_eq!(v10[0].win_at, V10_RAMP_WIN_AT);
        assert_eq!(v10[V10_NATION_INTRO_STAGE - 1].win_at, V10_RAMP_WIN_AT);
        assert_eq!(v10[V10_NATION_INTRO_STAGE].win_at, V10_ONE_NATION_WIN_AT);
        assert_eq!(v10[V10_MULTI_NATION_STAGE].win_at, V10_NATION_RAMP_WIN_AT);
        assert_eq!(v10[V10_EASY_RAMP_LEN - 1].win_at, V10_NATION_RAMP_WIN_AT);
        assert!(v10[V10_CLOSEOUT_STAGE].win_at < V10_NATION_RAMP_WIN_AT);
        assert!(v10[V10_CLOSEOUT_STAGE].win_at > V10_WIN_AT_END - 1e-9);
        assert!((v10[V10_STAGE_COUNT - 1].win_at - V10_WIN_AT_END).abs() < 1e-9);
        assert!(
            (stage_learning_rate(2.5e-4, 0.85, 28, V10_STAGE_LR_FLOOR) - V10_STAGE_LR_FLOOR).abs()
                < 1e-15
        );
        assert_eq!(
            imply_stage_from_learning_rate(2.5e-4 * 0.85_f64.powi(28), 2.5e-4, 0.85),
            Some(28)
        );
        for index in V10_EASY_RAMP_LEN..(V10_STAGE_COUNT - 1) {
            assert!(
                v10[index + 1].win_at <= v10[index].win_at + 1e-12,
                "win_at rose at stage {}",
                index + 1
            );
        }
        assert!(v10[1].bots > v10[0].bots);
        assert_eq!(remap_v10_short_sidecar_stage(0), 0);
        assert_eq!(remap_v10_short_sidecar_stage(19), 20);
        assert_eq!(remap_v10_short_sidecar_stage(20), 22);
        assert_eq!(remap_v10_short_sidecar_stage(34), 67);
        // Live ppo_v11 was on prior stage 22 (24 bots / 2 nations).
        assert_eq!(remap_v10_prior_sidecar_stage(22), 13);
        assert_eq!(remap_v10_prior_sidecar_stage(68), V10_MEDIUM_START);
        assert_eq!(remap_v10_prior_sidecar_stage(99), V10_STAGE_COUNT - 1);
    }

    #[test]
    fn sample_episode_rehearsal_uses_past_lobby_setup() {
        use rand::SeedableRng;
        use rand::rngs::SmallRng;
        let stages = stages_for_schedule(CurriculumSchedule::V10);
        let mut rng = SmallRng::seed_from_u64(42);
        let mut found = false;
        for _ in 0..200 {
            let (_map, bots, difficulty, nations, rehearsal) = sample_episode(&stages, 8, &mut rng);
            if !rehearsal {
                continue;
            }
            found = true;
            let matches_past = stages[..8]
                .iter()
                .any(|s| s.bots == bots && s.difficulty == difficulty && s.nations == nations);
            assert!(
                matches_past,
                "rehearsal lobby must come from a past stage, got bots={bots} difficulty={difficulty:?} nations={nations:?}"
            );
            break;
        }
        assert!(found, "expected a rehearsal sample within 200 draws");
    }

    #[test]
    fn formally_allied_requires_a_pact_not_just_two_players() {
        let ents = parse_ents(&json!({
            "players": [
                {"id": 1, "pid": "AGENTRL1", "alive": true, "tiles": 10},
                {"id": 2, "pid": "AGENTRL2", "alive": true, "tiles": 10}
            ],
            "units": [],
            "attacks": [],
            "alliances": []
        }));
        assert!(!formally_allied(&ents, 1, 2));
        let allied = parse_ents(&json!({
            "players": [
                {"id": 1, "pid": "AGENTRL1", "alive": true, "tiles": 10},
                {"id": 2, "pid": "AGENTRL2", "alive": true, "tiles": 10}
            ],
            "units": [],
            "attacks": [],
            "alliances": [[1, 2, 500]]
        }));
        assert!(formally_allied(&allied, 1, 2));
        assert!(formally_allied(&allied, 2, 1));
        assert!(!formally_allied(&allied, 1, 1));
    }

    #[test]
    fn duo_synergy_pays_alliance_only_when_formally_allied_and_alive() {
        assert_eq!(duo_synergy_reward(true, false), W_DUO_BOTH_ALIVE);
        assert_eq!(
            duo_synergy_reward(true, true),
            W_DUO_BOTH_ALIVE + W_DUO_ALLIED
        );
        assert_eq!(duo_synergy_reward(false, true), 0.0);
    }

    #[test]
    fn duo_does_not_pay_request_or_donate_actions() {
        // Outcome-only: these used to be W_DUO_ALLY_REQUEST=0.08 and
        // W_DUO_DONATE_PARTNER=0.04. Re-adding them is the donate/pact farm.
        assert_eq!(duo_synergy_reward(true, true), 0.07);
        let even = duo_welfare_reward(0.4, 0.4);
        assert!(even > 0.0);
        assert!(even < 0.02);
    }

    #[test]
    fn duo_pact_success_is_oneshot_outcome_not_an_action_wage() {
        let mut cfg = config();
        cfg.duo_pact_success = 5.0;
        assert_eq!(duo_pact_success_bonus(false, cfg), 0.0);
        assert_eq!(duo_pact_success_bonus(true, cfg), 5.0);
        cfg.duo_pact_success = 0.0;
        assert_eq!(duo_pact_success_bonus(true, cfg), 0.0);
        // Still smaller than a win so timeout-while-allied stays a loss.
        assert!(5.0 < W_WIN);
    }

    #[test]
    fn economy_potential_is_income_not_gold_stock_and_stays_below_a_win() {
        assert_eq!(economy_potential(50_000.0, 50_000.0, 0.0), 0.0);
        let before = economy_potential(25_000.0, 25_000.0, 0.25);
        let after_city = economy_potential(40_000.0, 25_000.0, 0.25);
        assert!(after_city > before);
        // Spending gold stock is invisible to this Φ (that's the K_ECO hole).
        assert_eq!(
            economy_potential(25_000.0, 25_000.0, 0.25),
            economy_potential(25_000.0, 25_000.0, 0.25)
        );
        let saturated = economy_potential(ECO_INCOME_REF, ECO_INCOME_REF, 0.25);
        assert!(saturated > 0.0);
        assert!(saturated < 1.0);
        assert!(saturated < W_WIN);
    }

    #[test]
    fn first_city_and_port_are_oneshot_outcomes_not_build_wages() {
        assert_eq!(duo_first_structure_bonus(false, 3.0), 0.0);
        assert_eq!(duo_first_structure_bonus(true, 3.0), 3.0);
        assert_eq!(duo_first_structure_bonus(true, 5.0), 5.0);
        assert_eq!(duo_first_structure_bonus(true, 0.0), 0.0);
        assert!(3.0 + 5.0 < W_WIN);
    }

    #[test]
    fn structure_delete_penalty_is_count_drop_not_an_action_wage() {
        assert_eq!(duo_structure_delete_penalty(0, 3.0), 0.0);
        assert_eq!(duo_structure_delete_penalty(1, 3.0), -3.0);
        assert_eq!(duo_structure_delete_penalty(2, 5.0), -10.0);
        assert_eq!(duo_structure_delete_penalty(1, 0.0), 0.0);
        assert!(3.0 < W_WIN);
    }

    #[test]
    fn team_completed_structures_ignores_construction_and_foreign_owners() {
        let ents = parse_ents(&json!({
            "players": [
                {"id": 1, "pid": "AGENTRL1", "alive": true, "goldIncome": 25000},
                {"id": 2, "pid": "AGENTRL2", "alive": true, "goldIncome": 25000},
                {"id": 3, "pid": "BOT", "alive": true}
            ],
            "units": [
                {"type": "City", "owner": 1, "constructing": false, "x": 0, "y": 0},
                {"type": "City", "owner": 1, "constructing": true, "x": 1, "y": 0},
                {"type": "Port", "owner": 2, "constructing": false, "x": 2, "y": 0},
                {"type": "City", "owner": 3, "constructing": false, "x": 3, "y": 0}
            ],
            "attacks": [],
            "alliances": []
        }));
        let owners = [1usize, 2];
        assert_eq!(
            team_completed_structures(&ents, &owners, CITY_UNIT_CLASS),
            1
        );
        assert_eq!(
            team_completed_structures(&ents, &owners, PORT_UNIT_CLASS),
            1
        );
        assert_eq!(player_gold_income(&ents, 1), 25000.0);
    }

    #[test]
    fn duo_welfare_penalizes_lopsided_strength() {
        let even = duo_welfare_reward(0.4, 0.4);
        let lopsided = duo_welfare_reward(0.8, 0.05);
        assert!(even > lopsided);
    }

    #[test]
    fn timeout_without_a_win_is_a_loss_not_a_placement_gift() {
        let place_first = terminal_reward(1, false, false);
        assert!(place_first > 0.0);
        assert_eq!(terminal_reward(1, false, true), -W_WIN);
        assert_eq!(terminal_reward(2, false, true), -W_WIN);
        let win = terminal_reward(1, true, false);
        assert!(win > W_WIN);
        // Won-and-timeout should not happen, but a win still pays the win.
        assert_eq!(terminal_reward(1, true, true), win);
    }

    #[test]
    fn never_spawned_is_the_same_loss_as_timeout() {
        // Callers pass timed_out=true when the human never owned tiles.
        assert_eq!(terminal_reward(1, false, true), -W_WIN);
        assert_eq!(terminal_reward(8, false, true), -W_WIN);
    }

    #[test]
    fn duo_pbrs_camping_does_not_accumulate_and_pact_is_a_oneshot() {
        let gamma = 0.999;
        let start = duo_potential(0.1, 0.1, true, false);
        let mut shaper = DominanceShaper::default();
        shaper.reset(start);
        let mut camped = 0.0;
        for _ in 0..1400 {
            camped += shaper.transition(start, gamma, 1.0);
        }
        let wage_farm = 1400.0 * start;
        // Constant Φ: each F = (γ−1)Φ. 1400 wages would be ~35; PBRS is
        // a small negative drift (~1.4 Φ), not a timeout salary.
        assert!(
            camped.abs() < 0.1 * wage_farm,
            "camping PBRS={camped} would-be wage={wage_farm} Φ={start}"
        );
        assert!(camped < 0.0);

        shaper.reset(start);
        let allied = duo_potential(0.1, 0.1, true, true);
        let pact_delta = shaper.transition(allied, gamma, 1.0);
        assert!(
            (pact_delta - (gamma * allied - start)).abs() < 1e-12,
            "pact delta {pact_delta}"
        );
        assert!(pact_delta > 0.0);

        // Absorbing terminal: leftover Φ is charged once, not as a timeout gift.
        let close = shaper.transition(0.0, gamma, 1.0);
        assert!((close + allied).abs() < 1e-12);
    }
}
