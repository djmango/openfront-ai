//! C ABI (FFI) over [`crate::rl::RlSession`] so an out-of-tree C env (a
//! PufferLib-style `template.h` env, or any other C/C++/Python-ctypes
//! driver) can step the native Rust engine in-process.
//!
//! Nothing here invents a new observation or action space: the observation
//! is the raw pre-AE featurization the Rust trainer already consumes
//! (`ofcore::feat::featurize`, see `oftrain/src/vecenv.rs::prepare_agent`),
//! and the mask is that same featurizer's legality output (`legal_actions`,
//! `legal_ptarget`, `legal_utarget`, `legal_build`, `legal_nuke`,
//! `legal_tile`) which is derived from `crate::obs_typed::legality_typed`.
//!
//! # Layouts (all `f32`, flat, per agent, then concatenated over agents)
//!
//! `n_agents` is 1 or 2 (`RlSession::reset` clamps it). The buffers returned
//! by [`ofenv_obs`] / [`ofenv_mask`] hold `n_agents` consecutive per-agent
//! blocks; the per-agent strides are [`OFENV_OBS_PER_AGENT`] and
//! [`OFENV_MASK_PER_AGENT`] (compile-time constants, mirrored in
//! `include/openfront_env.h`).
//!
//! ## Observation block (`OFENV_OBS_PER_AGENT` = 4270 floats)
//!
//! | offset | len  | contents                                              |
//! |--------|------|-------------------------------------------------------|
//! | 0      | 12   | `Feat::scalars` (`N_SCALARS`)                          |
//! | 12     | 3840 | `Feat::players`, `MAX_SLOTS`(128) x `P_FEAT`(30)       |
//! | 3852   | 384  | `Feat::units`, `MAX_UNITS`(32) x `U_FEAT`(12)          |
//! | 4236   | 34   | legality scalars (see below)                           |
//!
//! Legality scalars at 4236: `legal_build[0..7]`, `legal_nuke[7..12]`,
//! `legal_actions[12..33]`, `me_slot[33]` (`Feat::me_slot` as f32).
//!
//! NOT included (the AE-side additions live in `oftrain` and need torch):
//! fine/coarse grid crops `ae_v32_nostatic` at 1/8 (`stat` = 6 planes and
//! `transient` = 57 planes at `REGION=8` resolution) plus the AE latent, the
//! coarse 1/16 crops, the `ego` (3) / `db` (1) / `local` (5xLOCALxLOCAL)
//! pooled planes, and the full tile plane at `GW_MAX=250` x `GH_MAX=150`.
//! Those are all downstream of the fields exported here (`owners_slotted`
//! is reconstructed from [`ofenv_tiles`] with the same LUT the featurizer
//! uses) plus `Feat::stat` / `Feat::transient` / `Feat::legal_tile`.
//!
//! ## Mask block (`OFENV_MASK_PER_AGENT` = 40893 floats)
//!
//! | offset | len   | contents                                            |
//! |--------|-------|-----------------------------------------------------|
//! | 0      | 21    | `Feat::legal_actions` (`N_ACTIONS`, A_NOOP..A_ALLIANCE_EXTENSION) |
//! | 21     | 2688  | `Feat::legal_ptarget`, `N_ACTIONS` x `MAX_SLOTS`    |
//! | 2709   | 672   | `Feat::legal_utarget`, `N_ACTIONS` x `MAX_UNITS`    |
//! | 3381   | 7     | `Feat::legal_build` (`N_BUILD`)                     |
//! | 3388   | 5     | `Feat::legal_nuke` (`N_NUKE`)                       |
//! | 3393   | 37500 | `Feat::legal_tile` on a fixed `GH_MAX`(150) x `GW_MAX`(250) row-major plane; only rows `0..gh`, cols `0..gw` are written |
//!
//! The fixed tile plane is what makes the total a compile-time constant;
//! `gh`/`gw` for the live map are reported in [`ofenv_meta`].
//!
//! # Reward / terminal (the trainer's V10 curriculum, partially wired)
//!
//! `ofenv_reward(env, agent)` returns a *per-decision* shape built from
//! `ofcore::curriculum`'s V10 reward recipe (`RewardConfig` defaults here
//! mirror the `oftrain` CLI defaults, see [`trainer_default_reward_config`]),
//! the same component functions `oftrain/src/vecenv.rs` calls. `ofenv_terminal`
//! is the trainer's episode-done condition: all agents off the map, a winner
//! decided (human win / Humans team / combined-80% duo territory), or the
//! tick cap `ofcore::DEFAULT_MAX_EPISODE_TICKS` reached. During the spawn
//! phase both are 0 (mirrors vecenv's spawn early-return).
//!
//! WIRED (computed from `RlSession` state plus per-decision trackers the FFI
//! keeps in [`OFEnvState`]: `prev_strength`, the dominance/closeout
//! `DominanceShaper`s, `was_alive`, `closeout_max_share`, `closeout_entry_paid`):
//!   - `strength`        `W_STR * composite_strength(me) * timeweight(tick)`
//!   - `strength_delta`  `strength_delta_weight(..) * delta` (dominant-loss aware)
//!   - `dominance`       PBRS `DominanceShaper` on `dominance_potential`
//!   - `closeout`        PBRS on `closeout_potential` + `v10_closeout_entry_bonus`
//!   - `tempo`           `-v84_tempo_coef * tempo_pressure(..) * timeweight`
//!   - `survival`        `v10_survival_reward(alive, land_share, cfg)`
//!   - `waste`           `-W_WASTE * engine_wasted` for agent 0 (vecenv's `i==0`
//!                       rule); the per-agent "+1 for an empty non-noop intent"
//!                       increment is NOT wired
//!   - `death`           `-death_penalty()` on the alive→dead transition
//!   - `terminal`        `terminal_reward(place, won, no_play) +
//!                       fast_win_bonus + v85_extra_win_bonus +
//!                       v10_timeout_after_closeout_penalty`, only on the done
//!                       decision
//!
//! NOT WIRED (needs trainer state that does not exist inside `RlSession` — the
//! FFI never sees the policy's structured `Choice`, so it cannot replay these):
//!   - `action_churn` (`ActionChurnTracker` / `v83_action_churn_penalty`; needs
//!     the chosen action id + resolved target id per decision)
//!   - `embargo_outcome` (`CombatTracker::observe_embargo_stop`; needs the
//!     embargo-stop target's relation and the sticky-window tracker)
//!   - `combat_outcome` (`CombatTracker::observe_combat`; needs attack/retreat
//!     target ids and the just-opened-attack bookkeeping)
//!   - `boat_outcome` (`BoatTracker` launch/resolve windows, pre/post troop
//!     deltas, `classify_boat_resolution`; needs the boat launch decision history)
//!   - `diplo_panic` (`v10_diplo_panic_penalty`; needs the chosen action id)
//!   - `combat_action` (`v10_combat_action_bonus`; needs the chosen action id and
//!     whether it emitted a real intent)
//!   - `duo` (all `duo_*` terms + `DUO_SOLO_SCALE`: team PBRS, pacts, structures,
//!     `W_DUO_*`; these fire only for the 2-agent team mode and need the
//!     one-shot-paid flags the trainer keeps in `EnvWorker`)
//!
//! `ofenv_stage_info` reports the V10 curriculum stage the env is pinned to;
//! `ofenv_set_stage` rebuilds the `RlSession` for that stage's bots / nations /
//! difficulty / decision_ticks.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::PathBuf;

use ofcore::curriculum::{
    self, CurriculumSchedule, DominanceShaper, Nations, RewardComponents, RewardConfig,
    V10_ENV_TARGETS, V10_STAGE_COUNT, W_STR, W_WASTE,
};
use ofcore::feat::{
    self, Legal, A_ALLIANCE_EXTENSION, ACTIONS, BUILD_TYPES, IS_LAND_BIT, MAG_MASK, MAX_SLOTS,
    MAX_UNITS, N_ACTIONS, N_BUILD, N_NUKE, N_SCALARS, P_FEAT, U_FEAT,
};
use serde_json::{json, Value};

use crate::obs_typed::legality_typed;
use crate::rl::RlSession;
use crate::session::{AGENT_CLIENT_IDS, AGENT_CLIENT_ID, AGENT_CLIENT_ID_2};

// ---------------------------------------------------------------------------
// Layout constants (mirrored in include/openfront_env.h)
// ---------------------------------------------------------------------------

/// Scalars / player-token / unit-token offsets, in floats.
pub const OFENV_OBS_SCALARS_OFF: usize = 0;
pub const OFENV_OBS_SCALARS_N: usize = N_SCALARS; // 12
pub const OFENV_OBS_PLAYERS_OFF: usize = OFENV_OBS_SCALARS_OFF + OFENV_OBS_SCALARS_N; // 12
pub const OFENV_OBS_PLAYERS_N: usize = MAX_SLOTS * P_FEAT; // 3840
pub const OFENV_OBS_UNITS_OFF: usize = OFENV_OBS_PLAYERS_OFF + OFENV_OBS_PLAYERS_N; // 3852
pub const OFENV_OBS_UNITS_N: usize = MAX_UNITS * U_FEAT; // 384
pub const OFENV_OBS_LEGAL_OFF: usize = OFENV_OBS_UNITS_OFF + OFENV_OBS_UNITS_N; // 4236
/// `legal_build` (7) + `legal_nuke` (5) + `legal_actions` (21) + `me_slot` (1).
pub const OFENV_OBS_LEGAL_N: usize = N_BUILD + N_NUKE + N_ACTIONS + 1; // 34
pub const OFENV_OBS_LEGAL_BUILD_OFF: usize = OFENV_OBS_LEGAL_OFF;
pub const OFENV_OBS_LEGAL_NUKE_OFF: usize = OFENV_OBS_LEGAL_BUILD_OFF + N_BUILD;
pub const OFENV_OBS_LEGAL_ACTIONS_OFF: usize = OFENV_OBS_LEGAL_NUKE_OFF + N_NUKE;
pub const OFENV_OBS_ME_SLOT_OFF: usize = OFENV_OBS_LEGAL_ACTIONS_OFF + N_ACTIONS;
/// Floats per agent in the observation buffer.
pub const OFENV_OBS_PER_AGENT: usize = OFENV_OBS_LEGAL_OFF + OFENV_OBS_LEGAL_N; // 4270

/// `legal_actions` (N_ACTIONS) - the action bit vector the trainer masks with.
pub const OFENV_MASK_ACTIONS_OFF: usize = 0;
pub const OFENV_MASK_ACTIONS_N: usize = N_ACTIONS; // 21
/// `legal_ptarget` (N_ACTIONS x MAX_SLOTS), row-major by action.
pub const OFENV_MASK_PTARGET_OFF: usize = OFENV_MASK_ACTIONS_OFF + OFENV_MASK_ACTIONS_N; // 21
pub const OFENV_MASK_PTARGET_N: usize = N_ACTIONS * MAX_SLOTS; // 2688
/// `legal_utarget` (N_ACTIONS x MAX_UNITS), row-major by action.
pub const OFENV_MASK_UTARGET_OFF: usize = OFENV_MASK_PTARGET_OFF + OFENV_MASK_PTARGET_N; // 2709
pub const OFENV_MASK_UTARGET_N: usize = N_ACTIONS * MAX_UNITS; // 672
/// `legal_build` (N_BUILD).
pub const OFENV_MASK_BUILD_OFF: usize = OFENV_MASK_UTARGET_OFF + OFENV_MASK_UTARGET_N; // 3381
pub const OFENV_MASK_BUILD_N: usize = N_BUILD; // 7
/// `legal_nuke` (N_NUKE).
pub const OFENV_MASK_NUKE_OFF: usize = OFENV_MASK_BUILD_OFF + OFENV_MASK_BUILD_N; // 3388
pub const OFENV_MASK_NUKE_N: usize = N_NUKE; // 5
/// `legal_tile` on a fixed GH_MAX x GW_MAX plane.
pub const OFENV_MASK_TILE_OFF: usize = OFENV_MASK_NUKE_OFF + OFENV_MASK_NUKE_N; // 3393
pub const OFENV_MASK_TILE_N: usize = (feat::GH_MAX as usize) * (feat::GW_MAX as usize); // 37500
/// Floats per agent in the mask buffer.
pub const OFENV_MASK_PER_AGENT: usize = OFENV_MASK_TILE_OFF + OFENV_MASK_TILE_N; // 40893

const _: () = assert!(OFENV_OBS_PER_AGENT == 4270);
const _: () = assert!(OFENV_MASK_PER_AGENT == 40893);

/// Owner id bits of the packed tile state (`crate::Map::tile_state`).
const OWNER_MASK: u16 = 0x0FFF;

/// Public mirror of [`OWNER_MASK`] for consumers of [`ofenv_tiles`]
/// (same value as `OFENV_TILE_OWNER_MASK` in `include/openfront_env.h`).
pub const OFENV_TILE_OWNER_MASK: u16 = OWNER_MASK;
/// Fallout flag bit in the packed tile state.
pub const OFENV_TILE_FALLOUT_BIT: u16 = 13;
/// Defense-bonus flag bit in the packed tile state.
pub const OFENV_TILE_DEFENSE_BONUS_BIT: u16 = 14;

/// Number of live RL humans the FFI exposes (RlSession clamps to 1..=2).
pub const OFENV_MAX_AGENTS: usize = 2;

/// The `/8`-grid envelope of the fixed `legal_tile` mask plane (mirrors
/// `OFENV_GH_MAX`/`OFENV_GW_MAX` in `include/openfront_env.h`). Maps whose
/// trimmed `/8` dims exceed this cannot be represented (see `new_state`).
pub const OFENV_GH_MAX: usize = feat::GH_MAX as usize;
pub const OFENV_GW_MAX: usize = feat::GW_MAX as usize;

// ---------------------------------------------------------------------------
// last_error (thread-local; valid until the next call on this thread)
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: impl Into<String>) {
    let msg = msg.into().replace('\0', " ");
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(msg).unwrap_or_default();
    });
}

fn clear_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::default();
    });
}

/// `const char* ofenv_last_error(void)` - always non-NULL; empty string when
/// the last call on this thread succeeded. Valid until the next FFI call on
/// this thread.
#[no_mangle]
pub extern "C" fn ofenv_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

// ---------------------------------------------------------------------------
// Config parsing
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Cfg {
    repo_root: PathBuf,
    map: String,
    seed: String,
    bots: u32,
    difficulty: String,
    nations: Value,
    n_agents: u32,
    ticks_per_decision: u32,
    stage: usize,
}

impl Cfg {
    fn from_str(raw: &str) -> Result<Self, String> {
        let trimmed = raw.trim();
        let fields: HashMap<String, String> = if trimmed.starts_with('{') {
            let v: Value = serde_json::from_str(trimmed)
                .map_err(|e| format!("json_cfg is neither `k=v` pairs nor a JSON object: {e}"))?;
            let obj = v
                .as_object()
                .ok_or_else(|| "json_cfg JSON must be an object".to_string())?;
            obj.iter()
                .map(|(k, val)| {
                    let s = match val {
                        Value::String(s) => s.clone(),
                        Value::Null => String::new(),
                        other => other.to_string(),
                    };
                    (k.to_ascii_lowercase(), s)
                })
                .collect()
        } else {
            let mut m = HashMap::new();
            for part in trimmed.split(',') {
                let part = part.trim();
                if part.is_empty() {
                    continue;
                }
                let (k, v) = part
                    .split_once('=')
                    .ok_or_else(|| format!("expected `key=value` in json_cfg field {part:?}"))?;
                m.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
            }
            m
        };
        let get = |k: &str| fields.get(k).map(String::as_str);
        let repo_root = get("repo_root")
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "json_cfg requires repo_root=<openfront-ai repo root>".to_string())?;
        let map = get("map").filter(|s| !s.is_empty()).unwrap_or("plains");
        let seed = get("seed").filter(|s| !s.is_empty()).unwrap_or("s1");
        let bots = match get("bots") {
            Some(s) => s.parse::<u32>().map_err(|e| format!("bots={s}: {e}"))?,
            None => 3,
        };
        let difficulty = get("difficulty").filter(|s| !s.is_empty()).unwrap_or("Easy");
        let n_agents = match get("n_agents") {
            Some(s) => s.parse::<u32>().map_err(|e| format!("n_agents={s}: {e}"))?,
            None => 1,
        }
        .clamp(1, OFENV_MAX_AGENTS as u32);
        let ticks_per_decision = match get("ticks_per_decision") {
            Some(s) => s
                .parse::<u32>()
                .map_err(|e| format!("ticks_per_decision={s}: {e}"))?,
            None => 8,
        }
        .max(1);
        let nations = match get("nations") {
            Some("default") => Value::String("default".into()),
            Some(s) => match s.parse::<i64>() {
                Ok(n) => Value::from(n),
                Err(_) => Value::String(s.to_string()),
            },
            None => Value::from(0),
        };
        let stage = match get("stage") {
            Some(s) => s
                .parse::<usize>()
                .map_err(|e| format!("stage={s}: {e}"))?
                .min(V10_STAGE_COUNT - 1),
            None => 0,
        };
        Ok(Cfg {
            repo_root: PathBuf::from(repo_root),
            map: map.to_string(),
            seed: seed.to_string(),
            bots,
            difficulty: difficulty.to_string(),
            nations,
            n_agents,
            ticks_per_decision,
            stage,
        })
    }
}

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

/// Opaque env state behind the `OFEnv` pointer.
pub struct OFEnvState {
    cfg: Cfg,
    session: RlSession,
    n_agents: usize,
    width: usize,
    height: usize,
    hr: usize,
    wr: usize,
    gh: usize,
    gw: usize,
    land: Vec<u8>,
    mag: Vec<u8>,
    land_total: f64,
    ents: ofcore::feat::EntsData,
    legal: Vec<Legal>,
    winner: Value,
    obs: Vec<f32>,
    mask: Vec<f32>,
    tiles: Vec<u16>,
    reward: Vec<f64>,
    terminal: Vec<c_int>,
    prev_share: Vec<f64>,
    ever_on_map: Vec<bool>,
    meta: CString,
    step_count: u64,
    // --- curriculum reward state (see the module doc's WIRED / NOT WIRED) ---
    /// Pinned V10 curriculum stage (see `ofenv_set_stage`).
    stage: usize,
    /// V10 reward recipe; defaults mirror the `oftrain` CLI defaults.
    reward_config: RewardConfig,
    /// Episode tick cap (`ofcore::DEFAULT_MAX_EPISODE_TICKS`).
    max_episode_ticks: i64,
    /// Composite strength at the previous decision, per agent.
    prev_strength: Vec<f64>,
    /// PBRS shaper for the V8.1 dominance potential, per agent.
    dominance_shaper: Vec<DominanceShaper>,
    /// PBRS shaper for the V8.3 closeout potential, per agent.
    closeout_shaper: Vec<DominanceShaper>,
    /// On-map at the previous decision (death-transition detection).
    was_alive: Vec<bool>,
    /// Max land share seen this episode, per agent (timeout-after-closeout).
    closeout_max_share: Vec<f64>,
    /// Whether the one-shot closeout-entry bonus has been paid, per agent.
    closeout_entry_paid: Vec<bool>,
    /// Debug components of the last reward (mirrored into `ofenv_meta`).
    components: Vec<RewardComponents>,
    /// JSON of the current stage (`ofenv_stage_info`).
    stage_info: CString,
    /// Engine-wasted intents reported by the last `RlSession::step`.
    last_wasted: u32,
}

/// Episode tick cap used by `ofenv_terminal` (mirrors the trainer default).
pub const OFENV_MAX_EPISODE_TICKS: i64 = ofcore::DEFAULT_MAX_EPISODE_TICKS;

/// The V10 reward recipe the `oftrain` CLI runs with by default. Values are
/// the `clap` defaults in `oftrain/src/main.rs` (see its
/// `v10_reward_recipe_is_the_cli_default` test); `duo_*` terms stay 0 by
/// default and are not applied by this FFI anyway (see the module doc).
pub fn trainer_default_reward_config() -> RewardConfig {
    RewardConfig {
        gamma: 0.999,
        v81_dom_coef: 0.25,
        v81_min_stage: 0,
        v81_potential_clamp: 2.0,
        v81_dominant_loss: true,
        v81_dominance_threshold: 0.30,
        v81_delta_loss_dominant: 5.0,
        v81_churn_coef: 0.05,
        v81_churn_window: 16,
        v81_churn_min_stage: 0,
        v83_close_coef: 4.0,
        v83_churn_coef: 0.06,
        v84_boat_useful: 0.15,
        v84_boat_destroyed: -0.20,
        v84_boat_cancelled: -0.03,
        v84_boat_own_shore: -0.05,
        v84_boat_min_stage: 0,
        v84_tempo_coef: 0.015,
        v84_tempo_min_stage: 0,
        v84_fast_win_coef: 40.0,
        v85_tempo_share_threshold: 0.30,
        v85_extra_win_bonus: 200.0,
        v85_embargo_bad_stop: -0.15,
        v85_embargo_good_stop: 0.02,
        v85_embargo_min_stage: 0,
        v85_premature_retreat: -0.03,
        v85_thrash_reengage: -0.03,
        v85_combat_min_stage: 0,
        v86_delta_loss: 5.5,
        v86_attack_symmetric_loss: true,
        v86_skip_combat_churn: true,
        v86_death_penalty: 3.0,
        v10_survival_coef: 0.01,
        v10_diplo_panic: 0.08,
        v10_diplo_panic_share: 0.35,
        v10_diplo_panic_tick_frac: 0.55,
        v10_combat_action: 0.02,
        v10_timeout_closeout: 20.0,
        v10_closeout_entry: 25.0,
        duo_pact_success: 0.0,
        duo_eco_coef: 0.0,
        duo_first_city: 0.0,
        duo_first_port: 0.0,
        duo_city_delete: 0.0,
        duo_port_delete: 0.0,
        duo_boat_commit: 0.0,
        duo_leftover_continent: 0.0,
        duo_port_stand: 0.0,
        duo_continent_span: 0.0,
        duo_boat_land: 0.0,
        duo_city_stand: 0.0,
        duo_defense_stand: 0.0,
        duo_partner_tiles: 0.0,
    }
}

/// A V10 stage table row, resolved for the FFI (maps pool kept for
/// `ofenv_stage_info`). `nations` mirrors `Nations::{Default, Exact}`.
pub struct StageParams {
    pub index: usize,
    pub name: String,
    pub difficulty: String,
    pub bots: u32,
    pub nations: u32,
    pub nations_default: bool,
    pub decision_ticks: u32,
    pub win_at: f64,
    pub env_target: usize,
    pub maps: Vec<String>,
}

/// Resolve stage `index` of the V10 schedule into concrete session knobs.
pub fn stage_params(index: usize) -> Option<StageParams> {
    if index >= V10_STAGE_COUNT {
        return None;
    }
    let stages = curriculum::stages_for_schedule(CurriculumSchedule::V10);
    let st = stages.get(index)?;
    let (nations, nations_default) = match st.nations {
        Nations::Default => (0, true),
        Nations::Exact(n) => (n, false),
    };
    Some(StageParams {
        index,
        // `Stage` has no `name` field in ofcore::curriculum; the session
        // difficulty is the closest stable label.
        name: st.difficulty.to_string(),
        difficulty: st.difficulty.to_string(),
        bots: st.bots,
        nations,
        nations_default,
        decision_ticks: st.decision_ticks,
        win_at: st.win_at,
        env_target: V10_ENV_TARGETS.get(index).copied().unwrap_or(0),
        maps: st.maps.iter().map(|m| m.to_string()).collect(),
    })
}

impl OFEnvState {
    fn client_id(&self, agent: usize) -> &'static str {
        AGENT_CLIENT_IDS[agent.min(OFENV_MAX_AGENTS - 1)]
    }

    fn me(&self, agent: usize) -> i64 {
        self.session
            .game
            .player_by_client_id(self.client_id(agent))
            .map(|p| p.small_id as i64)
            .unwrap_or(-1)
    }

    fn tiles_of(&self, agent: usize) -> f64 {
        let me = self.me(agent);
        if me < 0 {
            return 0.0;
        }
        self.ents
            .players
            .iter()
            .find(|p| p.id as i64 == me)
            .map(|p| p.tiles.max(0.0))
            .unwrap_or(0.0)
    }

    fn on_map(&self, agent: usize) -> bool {
        self.tiles_of(agent) > 0.0
    }

    /// Rebuild the observation + mask blocks for every agent from the current
    /// session/ents/legal state.
    fn rebuild_buffers(&mut self) {
        let tick = self.session.game.ticks() as i64;
        let spawn_phase = self.session.game.in_spawn_phase();
        let n = self.n_agents;
        for agent in 0..n {
            let me = self.me(agent);
            let alive = self.on_map(agent);
            // Same slot LUT rule as the trainer's `current_lut`: rebuild from
            // the live roster every decision (the roster is only stable after
            // the spawn phase, and rebuilding is exactly equivalent then).
            let ids: Vec<usize> = self.ents.players.iter().map(|p| p.id).collect();
            let lut = feat::make_lut(&ids);
            let owners_slotted = self.owners_slotted(&lut);
            let legal = &self.legal[agent];
            let f = feat::featurize(
                self.gh,
                self.gw,
                &lut,
                &self.land,
                &self.mag,
                &owners_slotted,
                tick,
                spawn_phase,
                alive,
                me,
                &self.ents,
                legal,
            );
            self.write_obs(agent, &f);
            self.write_mask(agent, &f);
        }
    }

    fn owners_slotted(&self, lut: &[u8]) -> Vec<u8> {
        let packed = self.session.tile_state();
        let mut out = vec![0u8; self.hr * self.wr];
        for y in 0..self.hr {
            let src_row = y * self.width;
            let dst_row = y * self.wr;
            for x in 0..self.wr {
                let owner = (packed[src_row + x] & OWNER_MASK) as usize;
                out[dst_row + x] = lut.get(owner).copied().unwrap_or(0);
            }
        }
        out
    }

    fn write_obs(&mut self, agent: usize, f: &feat::Feat) {
        let base = agent * OFENV_OBS_PER_AGENT;
        let block = &mut self.obs[base..base + OFENV_OBS_PER_AGENT];
        block[OFENV_OBS_SCALARS_OFF..OFENV_OBS_SCALARS_OFF + N_SCALARS]
            .copy_from_slice(&f.scalars);
        block[OFENV_OBS_PLAYERS_OFF..OFENV_OBS_PLAYERS_OFF + OFENV_OBS_PLAYERS_N]
            .copy_from_slice(&f.players[..OFENV_OBS_PLAYERS_N]);
        block[OFENV_OBS_UNITS_OFF..OFENV_OBS_UNITS_OFF + OFENV_OBS_UNITS_N]
            .copy_from_slice(&f.units[..OFENV_OBS_UNITS_N]);
        block[OFENV_OBS_LEGAL_BUILD_OFF..OFENV_OBS_LEGAL_BUILD_OFF + N_BUILD]
            .copy_from_slice(&f.legal_build);
        block[OFENV_OBS_LEGAL_NUKE_OFF..OFENV_OBS_LEGAL_NUKE_OFF + N_NUKE]
            .copy_from_slice(&f.legal_nuke);
        block[OFENV_OBS_LEGAL_ACTIONS_OFF..OFENV_OBS_LEGAL_ACTIONS_OFF + N_ACTIONS]
            .copy_from_slice(&f.legal_actions);
        block[OFENV_OBS_ME_SLOT_OFF] = f.me_slot as f32;
    }

    fn write_mask(&mut self, agent: usize, f: &feat::Feat) {
        let base = agent * OFENV_MASK_PER_AGENT;
        let block = &mut self.mask[base..base + OFENV_MASK_PER_AGENT];
        // Zero the fixed tile plane's overflow region every step: the live map
        // is smaller than GH_MAX x GW_MAX, and stale rows would otherwise look
        // like legal tiles to a C consumer.
        block[OFENV_MASK_TILE_OFF..OFENV_MASK_TILE_OFF + OFENV_MASK_TILE_N].fill(0.0);
        block[OFENV_MASK_ACTIONS_OFF..OFENV_MASK_ACTIONS_OFF + N_ACTIONS]
            .copy_from_slice(&f.legal_actions);
        block[OFENV_MASK_PTARGET_OFF..OFENV_MASK_PTARGET_OFF + OFENV_MASK_PTARGET_N]
            .copy_from_slice(&f.legal_ptarget[..OFENV_MASK_PTARGET_N]);
        block[OFENV_MASK_UTARGET_OFF..OFENV_MASK_UTARGET_OFF + OFENV_MASK_UTARGET_N]
            .copy_from_slice(&f.legal_utarget[..OFENV_MASK_UTARGET_N]);
        block[OFENV_MASK_BUILD_OFF..OFENV_MASK_BUILD_OFF + N_BUILD]
            .copy_from_slice(&f.legal_build);
        block[OFENV_MASK_NUKE_OFF..OFENV_MASK_NUKE_OFF + N_NUKE]
            .copy_from_slice(&f.legal_nuke);
        let gw_max = feat::GW_MAX as usize;
        for y in 0..self.gh {
            for x in 0..self.gw {
                let v = f.legal_tile[y * self.gw + x];
                if v != 0.0 {
                    block[OFENV_MASK_TILE_OFF + y * gw_max + x] = v;
                }
            }
        }
    }

    /// 1 when the episode is done for every agent: all agents off the map, a
    /// winner decided, or the tick cap reached. Mirrors `vecenv`'s `done`.
    fn episode_done(&self) -> (bool, bool, bool) {
        let n = self.n_agents;
        let tick = self.session.game.ticks() as i64;
        let alive: Vec<bool> = (0..n).map(|a| self.on_map(a)).collect();
        let all_dead = alive.iter().all(|&a| !a);
        let won = winner_includes_agent(&self.winner) || self.duo_territory_win();
        let winner_decided = !self.winner.is_null();
        let timed_out = !won && !winner_decided && !all_dead && tick >= self.max_episode_ticks;
        let done = all_dead || won || winner_decided || tick >= self.max_episode_ticks;
        (done, won, timed_out)
    }

    /// Combined human team land >= `DUO_TEAM_WIN_MAP_SHARE` (Team mode only).
    fn duo_territory_win(&self) -> bool {
        if self.n_agents < 2 {
            return false;
        }
        let team_tiles: f64 = (0..self.n_agents).map(|a| self.tiles_of(a)).sum();
        curriculum::team_territory_win(team_tiles, self.land_total.max(0.0) as i64)
    }

    fn stage_info_json(&self) -> CString {
        let p = match stage_params(self.stage) {
            Some(p) => p,
            None => return CString::default(),
        };
        let v = json!({
            "schedule": CurriculumSchedule::V10.id(),
            "index": p.index,
            "name": p.name,
            "difficulty": p.difficulty,
            "decision_ticks": p.decision_ticks,
            "bots": p.bots,
            "nations": p.nations,
            "nations_default": p.nations_default,
            "win_at": p.win_at,
            "env_target": p.env_target,
            "maps": p.maps,
            "stage_count": V10_STAGE_COUNT,
            "uses_v83_closeout": CurriculumSchedule::V10.uses_v83_closeout(),
            "reward_profile": self.reward_config.reward_profile_id(),
        });
        CString::new(v.to_string()).unwrap_or_default()
    }

    /// Seed / reset the per-decision trackers from the current state. Mirrors
    /// `vecenv::seed_agent_trackers` (called every spawn-phase decision) so the
    /// first post-spawn delta is not an artificial jump from zero.
    fn seed_trackers(&mut self) {
        let land_total = self.land_total.max(1.0) as i64;
        let composite = curriculum::strengths(&self.ents, land_total);
        for agent in 0..self.n_agents {
            let me = self.me(agent).max(0) as usize;
            let mine = composite.get(&me).copied().unwrap_or(0.0);
            let share = curriculum::land_share(self.tiles_of(agent), land_total);
            self.prev_strength[agent] = mine;
            self.was_alive[agent] = self.on_map(agent);
            self.closeout_max_share[agent] = share;
            self.closeout_entry_paid[agent] = share >= curriculum::V83_CLOSEOUT_SHARE_START;
            let potential = curriculum::dominance_potential(
                &composite,
                me,
                self.reward_config.v81_potential_clamp,
            );
            self.dominance_shaper[agent].reset(potential);
            self.closeout_shaper[agent].reset(curriculum::closeout_potential(share));
        }
    }

    /// Compute the per-decision V10 reward and done flags for every agent from
    /// the current post-step session state. See the module doc for the wired
    /// versus not-wired component list.
    fn update_curriculum_rewards(&mut self) {
        let n = self.n_agents;
        let spawn_phase = self.session.game.in_spawn_phase();
        if spawn_phase {
            // vecenv returns (0.0, false) during the spawn phase and re-seeds
            // its trackers every spawn decision.
            self.seed_trackers();
            for a in 0..n {
                self.reward[a] = 0.0;
                self.terminal[a] = 0;
                self.components[a] = RewardComponents::default();
                self.prev_share[a] = 0.0;
            }
            return;
        }

        let tick = self.session.game.ticks() as i64;
        let max_ticks = self.max_episode_ticks.max(1);
        let land_total = self.land_total.max(1.0) as i64;
        let composite = curriculum::strengths(&self.ents, land_total);
        let (done, won, timed_out) = self.episode_done();

        for agent in 0..n {
            let me_i = self.me(agent);
            let me = me_i.max(0) as usize;
            let tiles = self.tiles_of(agent);
            let alive = tiles > 0.0;
            let share = curriculum::land_share(tiles, land_total);
            let mine = composite.get(&me).copied().unwrap_or(0.0);
            let tw = curriculum::timeweight(tick);
            let delta = mine - self.prev_strength[agent];
            let normalized_share = if self.reward_config.dominant_loss_active(self.stage) {
                curriculum::normalized_strength_share(&composite, me)
            } else {
                0.0
            };
            let has_active_attack = self.ents.attacks.iter().any(|a| a.from == me);
            let delta_weight = curriculum::strength_delta_weight(
                delta,
                normalized_share,
                self.stage,
                self.reward_config,
                has_active_attack,
            );
            let mut c = RewardComponents {
                strength: W_STR * mine * tw,
                strength_delta: delta_weight * delta,
                ..RewardComponents::default()
            };
            let mut reward = c.strength + c.strength_delta;

            // V8.1 dominance PBRS.
            let next_potential = if done {
                0.0
            } else {
                curriculum::dominance_potential(
                    &composite,
                    me,
                    self.reward_config.v81_potential_clamp,
                )
            };
            if self.reward_config.dominance_shaping_active(self.stage) {
                c.dominance = self.dominance_shaper[agent].transition(
                    next_potential,
                    self.reward_config.gamma,
                    self.reward_config.v81_dom_coef,
                );
                reward += c.dominance;
            } else {
                self.dominance_shaper[agent].reset(next_potential);
            }

            // V8.3 closeout PBRS + V10 closeout-entry one-shot.
            let next_closeout = if done {
                0.0
            } else {
                curriculum::closeout_potential(share)
            };
            if CurriculumSchedule::V10.uses_v83_closeout()
                && self.reward_config.v83_close_coef != 0.0
            {
                c.closeout = self.closeout_shaper[agent].transition(
                    next_closeout,
                    self.reward_config.gamma,
                    self.reward_config.v83_close_coef,
                );
                reward += c.closeout;
            } else {
                self.closeout_shaper[agent].reset(next_closeout);
            }
            if share > self.closeout_max_share[agent] {
                self.closeout_max_share[agent] = share;
            }
            let just_entered = !done
                && share >= curriculum::V83_CLOSEOUT_SHARE_START
                && !self.closeout_entry_paid[agent];
            if just_entered {
                self.closeout_entry_paid[agent] = true;
            }
            let entry_bonus =
                curriculum::v10_closeout_entry_bonus(just_entered, self.reward_config);
            if entry_bonus != 0.0 {
                c.closeout += entry_bonus;
                reward += entry_bonus;
            }
            let closeout_reached =
                self.closeout_max_share[agent] >= curriculum::V83_CLOSEOUT_SHARE_START;

            // V8.4 tempo pressure while dominant.
            if self.reward_config.tempo_active(self.stage) {
                let tempo_share = curriculum::normalized_strength_share(&composite, me);
                let t = -self.reward_config.v84_tempo_coef
                    * curriculum::tempo_pressure(
                        tick,
                        max_ticks,
                        tempo_share,
                        self.reward_config.tempo_share_threshold(),
                    )
                    * tw;
                c.tempo = t;
                if t != 0.0 {
                    reward += t;
                }
            }

            // V10 survival shaping.
            c.survival = curriculum::v10_survival_reward(alive, share, self.reward_config);
            if c.survival != 0.0 {
                reward += c.survival;
            }

            // Waste: engine count for agent 0 only (vecenv's `i == 0` rule).
            let wasted = if agent == 0 { self.last_wasted as f64 } else { 0.0 };
            c.waste = -W_WASTE * wasted;
            if wasted != 0.0 {
                reward += c.waste;
            }

            // Death on the alive -> off-map transition.
            if !alive && self.was_alive[agent] {
                let death = self.reward_config.death_penalty();
                c.death = -death;
                reward -= death;
            }
            self.was_alive[agent] = alive;
            if alive {
                self.ever_on_map[agent] = true;
            }
            self.prev_share[agent] = share;
            self.prev_strength[agent] = mine;

            // Terminal on the done decision.
            if done {
                if !self.ever_on_map[agent] && !won && c.death == 0.0 {
                    let death = self.reward_config.death_penalty();
                    c.death = -death;
                    reward -= death;
                }
                let (place, _pn) = curriculum::placement(&self.ents, me_i, alive, land_total);
                let no_play = timed_out || !self.ever_on_map[agent];
                c.terminal = curriculum::terminal_reward(place, won, no_play)
                    + curriculum::fast_win_bonus(
                        won,
                        tick,
                        max_ticks,
                        self.reward_config.v84_fast_win_coef,
                    );
                if won {
                    c.terminal += self.reward_config.v85_extra_win_bonus;
                }
                c.terminal += curriculum::v10_timeout_after_closeout_penalty(
                    timed_out,
                    closeout_reached,
                    self.reward_config,
                );
                reward += c.terminal;
            }

            self.reward[agent] = reward;
            self.terminal[agent] = if done { 1 } else { 0 };
            self.components[agent] = c;
        }
    }

    fn build_meta(&mut self) {
        let mut agents = Vec::with_capacity(self.n_agents);
        for agent in 0..self.n_agents {
            agents.push(json!({
                "client_id": self.client_id(agent),
                "me": self.me(agent),
                "on_map": self.on_map(agent),
                "tiles": self.tiles_of(agent),
                "share": self.prev_share[agent],
                "reward": self.reward[agent],
                "terminal": self.terminal[agent],
                "reward_components": {
                    "strength": self.components[agent].strength,
                    "strength_delta": self.components[agent].strength_delta,
                    "dominance": self.components[agent].dominance,
                    "closeout": self.components[agent].closeout,
                    "action_churn": self.components[agent].action_churn,
                    "boat_outcome": self.components[agent].boat_outcome,
                    "tempo": self.components[agent].tempo,
                    "embargo_outcome": self.components[agent].embargo_outcome,
                    "combat_outcome": self.components[agent].combat_outcome,
                    "survival": self.components[agent].survival,
                    "diplo_panic": self.components[agent].diplo_panic,
                    "combat_action": self.components[agent].combat_action,
                    "waste": self.components[agent].waste,
                    "death": self.components[agent].death,
                    "terminal": self.components[agent].terminal,
                    "duo": self.components[agent].duo,
                },
            }));
        }
        let stage = stage_params(self.stage);
        let v = json!({
            "step": self.step_count,
            "tick": self.session.game.ticks(),
            "width": self.width,
            "height": self.height,
            "hr": self.hr,
            "wr": self.wr,
            "gh": self.gh,
            "gw": self.gw,
            "spawn_phase": self.session.game.in_spawn_phase(),
            "winner": self.winner,
            "game_id": self.session.game_id(),
            "repo_root": self.cfg.repo_root.display().to_string(),
            "map": self.cfg.map,
            "seed": self.cfg.seed,
            "bots": self.cfg.bots,
            "difficulty": self.cfg.difficulty,
            "n_agents": self.n_agents,
            "ticks_per_decision": self.cfg.ticks_per_decision,
            "stage": self.stage,
            "stage_name": stage.as_ref().map(|s| s.name.clone()).unwrap_or_default(),
            "win_at": stage.as_ref().map(|s| s.win_at).unwrap_or(0.0),
            "env_target": stage.as_ref().map(|s| s.env_target).unwrap_or(0),
            "max_episode_ticks": self.max_episode_ticks,
            "reward_profile": self.reward_config.reward_profile_id(),
            "obs_per_agent": OFENV_OBS_PER_AGENT,
            "mask_per_agent": OFENV_MASK_PER_AGENT,
            "obs_size": self.obs.len(),
            "mask_size": self.mask.len(),
            "actions": ACTIONS,
            "build_types": BUILD_TYPES,
            "agents": agents,
        });
        let s = v.to_string();
        self.meta = CString::new(s).unwrap_or_default();
    }
}

fn winner_includes_agent(winner: &Value) -> bool {
    let Some(a) = winner.as_array() else {
        return false;
    };
    match a.first().and_then(Value::as_str) {
        Some("player") => matches!(
            a.get(1).and_then(Value::as_str),
            Some(AGENT_CLIENT_ID) | Some(AGENT_CLIENT_ID_2)
        ),
        Some("team") => a.iter().any(|v| {
            matches!(v.as_str(), Some(AGENT_CLIENT_ID) | Some(AGENT_CLIENT_ID_2))
        }),
        _ => false,
    }
}

fn new_state(cfg: Cfg) -> Result<OFEnvState, String> {
    let (session, _head, ents, legal, terrain, duo) = RlSession::reset(
        &cfg.repo_root,
        &cfg.map,
        &cfg.seed,
        cfg.bots,
        &cfg.difficulty,
        cfg.nations.clone(),
        cfg.n_agents,
    )?;
    let width = session.game.width() as usize;
    let height = session.game.height() as usize;
    let hr = height - height % feat::REGION;
    let wr = width - width % feat::REGION;
    if hr == 0 || wr == 0 {
        return Err(format!("map {width}x{height} is smaller than one REGION cell"));
    }
    if width * height > 4096 * 4096 {
        return Err(format!("map {width}x{height} exceeds supported width/height"));
    }
    let (gh, gw) = (hr / feat::REGION, wr / feat::REGION);
    // The mask's legal_tile plane is a fixed GH_MAX x GW_MAX region and the
    // trainer's tile-pointer action ids use a fixed GW_MAX stride, so maps
    // coarser than that envelope cannot be represented (see
    // `oftrain/src/vecenv.rs::tile_to_grid_xy` and `policy.rs::FOVEATE_SIZE`).
    if gh > feat::GH_MAX as usize || gw > feat::GW_MAX as usize {
        return Err(format!(
            "map {width}x{height} is {gw}x{gh} /8-grid cells, outside the trainer \
             envelope GH_MAX={} x GW_MAX={}",
            feat::GH_MAX,
            feat::GW_MAX
        ));
    }
    let mut land = vec![0u8; hr * wr];
    let mut mag = vec![0u8; hr * wr];
    for y in 0..hr {
        for x in 0..wr {
            let t = terrain[y * width + x];
            land[y * wr + x] = (t >> IS_LAND_BIT) & 1;
            mag[y * wr + x] = t & MAG_MASK;
        }
    }
    let land_total = (land.iter().map(|&l| l as i64).sum::<i64>()).max(1) as f64;
    let n_agents = cfg.n_agents.clamp(1, OFENV_MAX_AGENTS as u32) as usize;
    let mut legal_by_agent = Vec::with_capacity(n_agents);
    legal_by_agent.push(legal);
    if n_agents > 1 {
        let (_head_b, legal_b) =
            duo.ok_or_else(|| "n_agents=2 but the session returned no second agent".to_string())?;
        legal_by_agent.push(legal_b);
    }
    let tiles = session.tile_state().to_vec();
    let mut state = OFEnvState {
        cfg,
        session,
        n_agents,
        width,
        height,
        hr,
        wr,
        gh: hr / feat::REGION,
        gw: wr / feat::REGION,
        land,
        mag,
        land_total,
        ents,
        legal: legal_by_agent,
        winner: Value::Null,
        obs: vec![0.0; n_agents * OFENV_OBS_PER_AGENT],
        mask: vec![0.0; n_agents * OFENV_MASK_PER_AGENT],
        tiles,
        reward: vec![0.0; n_agents],
        terminal: vec![0; n_agents],
        prev_share: vec![0.0; n_agents],
        ever_on_map: vec![false; n_agents],
        meta: CString::default(),
        step_count: 0,
        stage: 0,
        reward_config: trainer_default_reward_config(),
        max_episode_ticks: OFENV_MAX_EPISODE_TICKS,
        prev_strength: vec![0.0; n_agents],
        dominance_shaper: vec![DominanceShaper::default(); n_agents],
        closeout_shaper: vec![DominanceShaper::default(); n_agents],
        was_alive: vec![false; n_agents],
        closeout_max_share: vec![0.0; n_agents],
        closeout_entry_paid: vec![false; n_agents],
        components: vec![RewardComponents::default(); n_agents],
        stage_info: CString::default(),
        last_wasted: 0,
    };
    // The pinned stage comes from the config (`stage=NN`, default 0). Session
    // knobs stay exactly as parsed so `ofenv_create` semantics are unchanged;
    // `ofenv_set_stage` is what applies a stage's bots/nations/difficulty.
    state.stage = state.cfg.stage.min(V10_STAGE_COUNT - 1);
    state.stage_info = state.stage_info_json();
    state.rebuild_buffers();
    state.update_curriculum_rewards();
    // reset() is not a decision: report zero reward/terminal for it.
    state.reward.iter_mut().for_each(|r| *r = 0.0);
    state.terminal.iter_mut().for_each(|t| *t = 0);
    state.build_meta();
    Ok(state)
}

// ---------------------------------------------------------------------------
// extern "C" surface
// ---------------------------------------------------------------------------

/// `OFEnv ofenv_create(const char* json_cfg)` - NULL on failure (see
/// [`ofenv_last_error`]).
#[no_mangle]
pub extern "C" fn ofenv_create(json_cfg: *const c_char) -> *mut c_void {
    clear_error();
    if json_cfg.is_null() {
        set_error("ofenv_create: json_cfg is NULL");
        return std::ptr::null_mut();
    }
    let raw = match unsafe { CStr::from_ptr(json_cfg) }.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_error(format!("ofenv_create: json_cfg is not UTF-8: {e}"));
            return std::ptr::null_mut();
        }
    };
    let cfg = match Cfg::from_str(raw) {
        Ok(c) => c,
        Err(e) => {
            set_error(format!("ofenv_create: {e}"));
            return std::ptr::null_mut();
        }
    };
    match new_state(cfg) {
        Ok(state) => Box::into_raw(Box::new(state)) as *mut c_void,
        Err(e) => {
            set_error(format!("ofenv_create: {e}"));
            std::ptr::null_mut()
        }
    }
}

/// `void ofenv_destroy(OFEnv)` - NULL is a no-op.
#[no_mangle]
pub extern "C" fn ofenv_destroy(env: *mut c_void) {
    if env.is_null() {
        return;
    }
    drop(unsafe { Box::from_raw(env as *mut OFEnvState) });
}

unsafe fn state_mut<'a>(env: *mut c_void) -> Result<&'a mut OFEnvState, ()> {
    if env.is_null() {
        set_error("NULL env handle");
        return Err(());
    }
    Ok(&mut *(env as *mut OFEnvState))
}

/// `int ofenv_reset(OFEnv)` - 0 on success, -1 on error. Starts a fresh
/// `RlSession` with the config's seed/bots/difficulty.
#[no_mangle]
pub extern "C" fn ofenv_reset(env: *mut c_void) -> c_int {
    clear_error();
    let Ok(state) = (unsafe { state_mut(env) }) else {
        return -1;
    };
    let cfg = state.cfg.clone();
    match new_state(cfg) {
        Ok(fresh) => {
            *state = fresh;
            0
        }
        Err(e) => {
            set_error(format!("ofenv_reset: {e}"));
            -1
        }
    }
}

/// `int ofenv_step(OFEnv, const char* intents_json)` - 0 on success, -1 on
/// error. `intents_json` NULL, empty, or `[]` means no intents this decision.
#[no_mangle]
pub extern "C" fn ofenv_step(env: *mut c_void, intents_json: *const c_char) -> c_int {
    clear_error();
    let Ok(state) = (unsafe { state_mut(env) }) else {
        return -1;
    };
    let intents: Vec<Value> = if intents_json.is_null() {
        Vec::new()
    } else {
        let raw = match unsafe { CStr::from_ptr(intents_json) }.to_str() {
            Ok(s) => s,
            Err(e) => {
                set_error(format!("ofenv_step: intents_json is not UTF-8: {e}"));
                return -1;
            }
        };
        if raw.trim().is_empty() {
            Vec::new()
        } else {
            match serde_json::from_str::<Value>(raw) {
                Ok(Value::Array(a)) => a,
                Ok(Value::Null) => Vec::new(),
                Ok(other) => {
                    set_error(format!(
                        "ofenv_step: intents_json must be a JSON array, got {other}"
                    ));
                    return -1;
                }
                Err(e) => {
                    set_error(format!("ofenv_step: intents_json parse: {e}"));
                    return -1;
                }
            }
        }
    };
    let ticks = state.cfg.ticks_per_decision;
    let (head, _ents, _legal, _duo) = state.session.step(&intents, ticks);
    state.last_wasted = head
        .get("wasted")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    state.step_count += 1;
    // Refresh per-agent caches straight from the engine (the session's step
    // return is already the post-step state).
    state.ents = crate::obs_typed::entities_typed(&state.session.game);
    let n = state.n_agents;
    for agent in 0..n {
        state.legal[agent] = legality_typed(&state.session.game, AGENT_CLIENT_IDS[agent]);
    }
    state.tiles = state.session.tile_state().to_vec();
    state.winner = crate::obs::winner_value(&state.session.game);
    state.rebuild_buffers();
    state.update_curriculum_rewards();
    state.build_meta();
    0
}

/// `const float* ofenv_obs(OFEnv, int* n)` - pointer to the agent-concatenated
/// observation buffer; `*n` (when non-NULL) receives the total float count.
/// Valid until the next `ofenv_step`/`ofenv_reset`/`ofenv_destroy`.
#[no_mangle]
pub extern "C" fn ofenv_obs(env: *mut c_void, n: *mut c_int) -> *const f32 {
    let Ok(state) = (unsafe { state_mut(env) }) else {
        if !n.is_null() {
            unsafe { *n = 0 };
        }
        return std::ptr::null();
    };
    if !n.is_null() {
        unsafe { *n = state.obs.len() as c_int };
    }
    state.obs.as_ptr()
}

/// `const float* ofenv_mask(OFEnv, int* n)` - same contract as [`ofenv_obs`].
#[no_mangle]
pub extern "C" fn ofenv_mask(env: *mut c_void, n: *mut c_int) -> *const f32 {
    let Ok(state) = (unsafe { state_mut(env) }) else {
        if !n.is_null() {
            unsafe { *n = 0 };
        }
        return std::ptr::null();
    };
    if !n.is_null() {
        unsafe { *n = state.mask.len() as c_int };
    }
    state.mask.as_ptr()
}

/// `const unsigned short* ofenv_tiles(OFEnv, int* n)` - packed tile state
/// (`owner_id & 0x0FFF`, bit 13 fallout, bit 14 defense bonus), width x height,
/// row-major.
#[no_mangle]
pub extern "C" fn ofenv_tiles(env: *mut c_void, n: *mut c_int) -> *const u16 {
    let Ok(state) = (unsafe { state_mut(env) }) else {
        if !n.is_null() {
            unsafe { *n = 0 };
        }
        return std::ptr::null();
    };
    if !n.is_null() {
        unsafe { *n = state.tiles.len() as c_int };
    }
    state.tiles.as_ptr()
}

/// `const char* ofenv_meta(OFEnv)` - NUL-terminated JSON (never NULL for a
/// live env). Valid until the next call on this handle.
#[no_mangle]
pub extern "C" fn ofenv_meta(env: *mut c_void) -> *const c_char {
    let Ok(state) = (unsafe { state_mut(env) }) else {
        return LAST_ERROR.with(|e| e.borrow().as_ptr());
    };
    state.meta.as_ptr()
}

/// `double ofenv_reward(OFEnv, int agent)` - the agent's curriculum-shaped
/// per-decision reward (see the module doc for the wired / not-wired
/// components). 0.0 during the spawn phase and for an out-of-range agent.
#[no_mangle]
pub extern "C" fn ofenv_reward(env: *mut c_void, agent: c_int) -> f64 {
    let Ok(state) = (unsafe { state_mut(env) }) else {
        return 0.0;
    };
    if agent < 0 || agent as usize >= state.n_agents {
        return 0.0;
    }
    state.reward[agent as usize]
}

/// `int ofenv_terminal(OFEnv, int agent)` - the trainer's episode-done flag
/// for this decision (win, elimination, or the tick cap). 1 for an
/// out-of-range agent.
#[no_mangle]
pub extern "C" fn ofenv_terminal(env: *mut c_void, agent: c_int) -> c_int {
    let Ok(state) = (unsafe { state_mut(env) }) else {
        return 1;
    };
    if agent < 0 || agent as usize >= state.n_agents {
        return 1;
    }
    state.terminal[agent as usize]
}

/// `int ofenv_stage_count(void)` - number of V10 curriculum stages
/// (`ofcore::curriculum::V10_STAGE_COUNT`).
#[no_mangle]
pub extern "C" fn ofenv_stage_count() -> c_int {
    V10_STAGE_COUNT as c_int
}

/// `int ofenv_set_stage(OFEnv, int stage)` - rebuild the session for that
/// stage's bots / nations / difficulty / decision_ticks (the map is kept when
/// it is in the stage's map pool, else the pool's first map is used). 0 on
/// success, -1 on error (out-of-range stage or a session build failure; see
/// [`ofenv_last_error`]).
#[no_mangle]
pub extern "C" fn ofenv_set_stage(env: *mut c_void, stage: c_int) -> c_int {
    clear_error();
    let Ok(state) = (unsafe { state_mut(env) }) else {
        return -1;
    };
    if stage < 0 || stage as usize >= V10_STAGE_COUNT {
        set_error(format!(
            "ofenv_set_stage: stage {stage} is outside 0..{}",
            V10_STAGE_COUNT - 1
        ));
        return -1;
    }
    let stage = stage as usize;
    let params = match stage_params(stage) {
        Some(p) => p,
        None => {
            set_error(format!("ofenv_set_stage: stage {stage} has no params"));
            return -1;
        }
    };
    let mut cfg = state.cfg.clone();
    cfg.bots = params.bots;
    cfg.difficulty = params.difficulty.clone();
    cfg.nations = if params.nations_default {
        Value::String("default".into())
    } else {
        Value::from(params.nations as i64)
    };
    cfg.ticks_per_decision = params.decision_ticks.max(1);
    if !params
        .maps
        .iter()
        .any(|m| m.eq_ignore_ascii_case(&cfg.map))
    {
        if let Some(first) = params.maps.first() {
            cfg.map = first.clone();
        }
    }
    cfg.stage = stage;
    match new_state(cfg) {
        Ok(fresh) => {
            *state = fresh;
            0
        }
        Err(e) => {
            set_error(format!("ofenv_set_stage({stage}): {e}"));
            -1
        }
    }
}

/// `const char* ofenv_stage_info(OFEnv)` - NUL-terminated JSON for the pinned
/// stage: schedule id, index, name, difficulty, decision_ticks, bots, nations,
/// win_at (win gate), env_target (env floor), maps, stage_count,
/// reward_profile. Never NULL for a live env.
#[no_mangle]
pub extern "C" fn ofenv_stage_info(env: *mut c_void) -> *const c_char {
    let Ok(state) = (unsafe { state_mut(env) }) else {
        return LAST_ERROR.with(|e| e.borrow().as_ptr());
    };
    state.stage_info.as_ptr()
}

/// `int ofenv_obs_size(OFEnv)` - total floats in the observation buffer
/// (`n_agents * OFENV_OBS_PER_AGENT`).
#[no_mangle]
pub extern "C" fn ofenv_obs_size(env: *mut c_void) -> c_int {
    let Ok(state) = (unsafe { state_mut(env) }) else {
        return 0;
    };
    state.obs.len() as c_int
}

/// `int ofenv_mask_size(OFEnv)` - total floats in the mask buffer.
#[no_mangle]
pub extern "C" fn ofenv_mask_size(env: *mut c_void) -> c_int {
    let Ok(state) = (unsafe { state_mut(env) }) else {
        return 0;
    };
    state.mask.len() as c_int
}

/// Ignore: keeps the A_ALLIANCE_EXTENSION bound check honest when N_ACTIONS
/// changes upstream.
const _: () = assert!(A_ALLIANCE_EXTENSION as usize == N_ACTIONS - 1);