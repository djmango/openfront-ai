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
//! # Reward / terminal (no reference implementation exists in the trainer)
//!
//! `ofenv_reward` is a documented host-side default, not the trainer's
//! shaping:
//! `share_delta + 1.0*won + (-1.0)*newly_dead`, where `share_delta` is the
//! change in this agent's tiles / map-land this decision. `ofenv_terminal`
//! is 1 once the match has a winner or once an agent that had been on the
//! map is no longer on it (spawn phase itself is never terminal).

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::PathBuf;

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
        Ok(Cfg {
            repo_root: PathBuf::from(repo_root),
            map: map.to_string(),
            seed: seed.to_string(),
            bots,
            difficulty: difficulty.to_string(),
            nations,
            n_agents,
            ticks_per_decision,
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

    fn update_rewards(&mut self) {
        let won = winner_includes_agent(&self.winner);
        let n = self.n_agents;
        for agent in 0..n {
            let on_map = self.on_map(agent);
            let share = if self.land_total > 0.0 {
                (self.tiles_of(agent) / self.land_total).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut r = share - self.prev_share[agent];
            if won {
                r += 1.0;
            }
            if self.ever_on_map[agent] && !on_map {
                r -= 1.0;
            }
            self.reward[agent] = r;
            self.prev_share[agent] = share;
            if on_map {
                self.ever_on_map[agent] = true;
            }
            let dead = self.ever_on_map[agent] && !on_map;
            let spawn_phase = self.session.game.in_spawn_phase();
            self.terminal[agent] = if !self.winner.is_null() || (dead && !spawn_phase) {
                1
            } else {
                0
            };
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
            }));
        }
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
    };
    state.rebuild_buffers();
    state.update_rewards();
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
    state.session.step(&intents, ticks);
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
    state.update_rewards();
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

/// `double ofenv_reward(OFEnv, int agent)` - see the module doc; 0.0 for an
/// out-of-range agent.
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

/// `int ofenv_terminal(OFEnv, int agent)` - 1 when the episode is over for
/// this agent (winner decided, or an agent that had been on the map is gone).
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