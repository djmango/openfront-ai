/*
 * openfront_env.h - C ABI over the OpenFront native Rust RL session
 * (`openfront-engine` crate, `src/puffer_ffi.rs`).
 *
 * Drives `RlSession` in-process so a PufferLib-style C env (or any
 * C/C++/ctypes driver) can reset/step the engine and read the same
 * observation and legality masks the Rust trainer consumes:
 * `ofcore::feat::featurize` (raw pre-AE featurization) and the
 * `crate::obs_typed::legality_typed` action space (A_NOOP=0 ..
 * A_ALLIANCE_EXTENSION=20, `N_ACTIONS = 21`).
 *
 * Link: rust/target/release/libopenfront_engine.so (crate-type
 * ["lib", "cdylib"], lib name `openfront_engine`).
 *
 * Threading: one OFEnv per thread. `ofenv_last_error()` is thread-local and
 * valid until the next FFI call on that thread. Buffer pointers returned by
 * ofenv_obs/ofenv_mask/ofenv_tiles/ofenv_meta are owned by the env and stay
 * valid until the next step/reset/destroy of that env.
 *
 * Config string (`ofenv_create`):
 *   repo_root=/path/to/openfront-ai, map=plains, seed=s1, bots=3,
 *   difficulty=Easy, n_agents=1, ticks_per_decision=8, stage=0
 * (a JSON object with the same keys is also accepted). `repo_root` is
 * required; `map` defaults to `plains`, `seed` to `s1`, `bots` to 3,
 * `difficulty` to `Easy`, `n_agents` to 1 (clamped to 1..=2),
 * `ticks_per_decision` to 8, `nations` to 0, `stage` to 0 (the pinned V10
 * curriculum stage; see ofenv_set_stage / ofenv_stage_info).
 *
 * intents_json for ofenv_step is a JSON array of engine intent objects, e.g.
 *   [{"type":"spawn","tile":12345}]
 * NULL, "", or "[]" means no intents. `clientID` defaults to AGENTRL1.
 *
 * Reward / terminal (see rust/engine/src/puffer_ffi.rs for the full list):
 *
 *   ofenv_reward(env, agent) returns the agent's per-decision reward from the
 *   trainer's V10 curriculum recipe (`ofcore::curriculum`, the component
 *   functions `oftrain/src/vecenv.rs` calls). ofenv_terminal(env, agent) is
 *   the trainer's episode-done flag: a winner decided, all agents off the map,
 *   or the tick cap ofcore::DEFAULT_MAX_EPISODE_TICKS (21000) reached. During
 *   the spawn phase both are 0.
 *
 *   WIRED components (from engine/session state plus per-decision trackers the
 *   env keeps internally):
 *     strength         W_STR * composite_strength(me) * timeweight(tick)
 *     strength_delta   strength_delta_weight(..) * delta (dominant-loss aware)
 *     dominance        V8.1 log strength-ratio potential (PBRS), stage-gated
 *     closeout         V8.3 land-share closeout potential (PBRS) + the V10
 *                      closeout-entry one-shot bonus
 *     tempo            -v84_tempo_coef * tempo_pressure(..) * timeweight
 *     survival         v10_survival_reward(alive, land_share)
 *     waste            -W_WASTE * engine-wasted intents (agent 0 only, the
 *                      trainer's i==0 rule)
 *     death            -death_penalty() on the alive->off-map transition
 *     terminal         terminal_reward(place, won, no_play) + fast-win bonus
 *                      + extra win bonus + timeout-after-closeout penalty
 *
 *   NOT WIRED (these need the policy's structured action choice, which the FFI
 *   never sees; they are documented, not faked, and read as 0.0 here):
 *     action_churn       (ActionChurnTracker / v83_action_churn_penalty)
 *     embargo_outcome    (CombatTracker embargo-stop relation window)
 *     combat_outcome     (CombatTracker attack/retreat window)
 *     boat_outcome       (BoatTracker launch/resolve windows)
 *     diplo_panic        (v10_diplo_panic_penalty; needs the chosen action id)
 *     combat_action      (v10_combat_action_bonus; needs the chosen action id)
 *     duo                (all duo_* team terms + DUO_SOLO_SCALE, 2-agent mode)
 *
 *   ofenv_set_stage(env, stage) rebuilds the session for that V10 stage's
 *   bots / nations / difficulty / decision_ticks (map kept if it is in the
 *   stage's pool). ofenv_stage_info(env) is JSON: index, name, difficulty,
 *   decision_ticks, bots, nations, win_at (win gate), env_target (env floor),
 *   maps, reward_profile.
 */

#ifndef OPENFRONT_ENV_H
#define OPENFRONT_ENV_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque env handle. */
typedef void *OFEnv;

/*
 * Observation layout (f32), one block per agent, blocks concatenated in
 * agent order (agent 0 = AGENTRL1, agent 1 = AGENTRL2). Per-agent stride is
 * OFENV_OBS_PER_AGENT = 4270 floats; ofenv_obs_size() returns
 * n_agents * 4270.
 *
 *   off     len   contents
 *   -----   ----  -------------------------------------------------------
 *   0       12    Feat::scalars  (N_SCALARS = 12)
 *   12      3840  Feat::players  (MAX_SLOTS 128 x P_FEAT 30, slot-major)
 *   3852    384   Feat::units    (MAX_UNITS 32 x U_FEAT 12, token-major)
 *   4236    34    legality scalars:
 *                   [0..7)   legal_build  (N_BUILD 7)
 *                   [7..12)  legal_nuke   (N_NUKE 5)
 *                   [12..33) legal_actions (N_ACTIONS 21)
 *                   [33]     me_slot (slot index, MAX_SLOTS-normalised unit)
 *
 * Scalars (N_SCALARS = 12), in order:
 *   tick/15000, spawn_phase, alive, log_norm(troops), log_norm(gold),
 *   n_alive/128, n_live_own_attacks/8, me_slot/128,
 *   log_norm(troop_income), log_norm(gold_income),
 *   team_claimed_share, team_map_share
 * Player token (P_FEAT = 30) and unit token (U_FEAT = 12) columns are defined
 * by ofcore/src/feat.rs::featurize / build_unit_tokens - unchanged here.
 *
 * NOT exported (AE-side additions in oftrain, torch-only): the fine (1/8)
 * ae_v32_nostatic stat (6 planes) + transient (57 planes) crops, the coarse
 * (1/16) crops, the AE latent, the pooled ego (3) / db (1) / local
 * (5 x LOCAL x LOCAL) planes, and the full-resolution tile plane at
 * GW_MAX 250 x GH_MAX 150. Their inputs are recoverable here:
 * owners_slotted from ofenv_tiles() with the featurizer LUT, land/mag from
 * the raw terrain bytes returned by reset.
 */

/*
 * Mask layout (f32), one block per agent, blocks concatenated in agent order.
 * Per-agent stride OFENV_MASK_PER_AGENT = 40893 floats; ofenv_mask_size()
 * returns n_agents * 40893. All entries are 0/1.
 *
 *   off     len    contents
 *   -----   -----  -------------------------------------------------------
 *   0       21     Feat::legal_actions (N_ACTIONS 21; index = action id
 *                  A_NOOP=0 .. A_ALLIANCE_EXTENSION=20)
 *   21      2688   Feat::legal_ptarget (N_ACTIONS x MAX_SLOTS 128,
 *                  row-major by action: row a = ptarget[a*128 + slot])
 *   2709    672    Feat::legal_utarget (N_ACTIONS x MAX_UNITS 32,
 *                  row-major by action: row a = utarget[a*32 + unit_token])
 *   3381    7      Feat::legal_build (N_BUILD 7: City, Port, Defense Post,
 *                  Missile Silo, SAM Launcher, Factory, Warship)
 *   3388    5      Feat::legal_nuke (N_NUKE 5: Atom Up, Atom Down,
 *                  Hydrogen Up, Hydrogen Down, MIRV)
 *   3393    37500  Feat::legal_tile on a FIXED GH_MAX(150) x GW_MAX(250)
 *                  row-major plane; only rows 0..gh, cols 0..gw are written
 *                  (gh/gw are reported in ofenv_meta) and the rest is 0.
 *
 * Because the legal_tile plane is fixed (and the trainer's tile-pointer
 * action ids use a fixed GW_MAX stride), ofenv_create/ofenv_reset reject
 * maps whose trimmed /8 dims exceed gh <= 150, gw <= 250; ofenv_last_error
 * names the map and its dims. On this checkout that envelope covers e.g.
 * Pangaea (125x125), World (125x250), Asia (150x250) but not Europe or
 * SouthAmerica.
 *
 * Everything except the tile plane is exactly the featurizer's
 * `legal_*` output for that agent (1.0 = legal). During the spawn phase
 * legal_actions is spawn-only for an unplaced agent and noop-only once it
 * has been placed, matching `rl/obs.py::_masks`.
 */

#define OFENV_N_ACTIONS 21
#define OFENV_MAX_SLOTS 128
#define OFENV_MAX_UNITS 32
#define OFENV_N_BUILD 7
#define OFENV_N_NUKE 5
#define OFENV_GW_MAX 250
#define OFENV_GH_MAX 150

#define OFENV_OBS_SCALARS_OFF 0
#define OFENV_OBS_SCALARS_N 12
#define OFENV_OBS_PLAYERS_OFF 12
#define OFENV_OBS_PLAYERS_N 3840
#define OFENV_OBS_UNITS_OFF 3852
#define OFENV_OBS_UNITS_N 384
#define OFENV_OBS_LEGAL_OFF 4236
#define OFENV_OBS_LEGAL_N 34
#define OFENV_OBS_LEGAL_BUILD_OFF 4236
#define OFENV_OBS_LEGAL_NUKE_OFF 4243
#define OFENV_OBS_LEGAL_ACTIONS_OFF 4248
#define OFENV_OBS_ME_SLOT_OFF 4269
#define OFENV_OBS_PER_AGENT 4270

#define OFENV_MASK_ACTIONS_OFF 0
#define OFENV_MASK_ACTIONS_N 21
#define OFENV_MASK_PTARGET_OFF 21
#define OFENV_MASK_PTARGET_N 2688
#define OFENV_MASK_UTARGET_OFF 2709
#define OFENV_MASK_UTARGET_N 672
#define OFENV_MASK_BUILD_OFF 3381
#define OFENV_MASK_BUILD_N 7
#define OFENV_MASK_NUKE_OFF 3388
#define OFENV_MASK_NUKE_N 5
#define OFENV_MASK_TILE_OFF 3393
#define OFENV_MASK_TILE_N 37500
#define OFENV_MASK_PER_AGENT 40893

#define OFENV_MAX_AGENTS 2

/* Packed tile state returned by ofenv_tiles(): owner id in the low 12 bits,
 * bit 13 fallout, bit 14 defense bonus. Row-major, width x height (width and
 * height are reported in ofenv_meta). */
#define OFENV_TILE_OWNER_MASK 0x0FFF
#define OFENV_TILE_FALLOUT_BIT 13
#define OFENV_TILE_DEFENSE_BONUS_BIT 14

/* Create an env from the config string above. NULL on failure; call
 * ofenv_last_error() for the reason. */
OFEnv ofenv_create(const char *json_cfg);

/* Free an env. NULL is a no-op. */
void ofenv_destroy(OFEnv env);

/* Start a fresh session with the config's seed/bots/difficulty. 0 ok, -1 err. */
int ofenv_reset(OFEnv env);

/* Apply one decision of `ticks_per_decision` engine ticks. `intents_json` is
 * a JSON array of intents; NULL / "" / "[]" means none. 0 ok, -1 err. */
int ofenv_step(OFEnv env, const char *intents_json);

/* Observation buffer, agent blocks concatenated. *n (if non-NULL) gets the
 * total float count. NULL on error. Owned by the env. */
const float *ofenv_obs(OFEnv env, int *n);

/* Mask buffer, agent blocks concatenated. Same contract as ofenv_obs. */
const float *ofenv_mask(OFEnv env, int *n);

/* Packed tile state buffer, width*height unsigned shorts. Same contract. */
const unsigned short *ofenv_tiles(OFEnv env, int *n);

/* NUL-terminated JSON metadata for the current state (tick, width, height,
 * gh, gw, spawn_phase, winner, per-agent me/on_map/tiles/reward/terminal,
 * sizes, action names). Owned by the env; valid until the next call. */
const char *ofenv_meta(OFEnv env);

/* Curriculum-shaped per-decision reward for `agent` (0-based), built from the
 * trainer's V10 recipe (see the "Reward / terminal" list above). 0.0 during
 * the spawn phase and for an out-of-range agent. */
double ofenv_reward(OFEnv env, int agent);

/* 1 when the episode is over for `agent` this decision (a winner was decided,
 * all agents are off the map, or the tick cap was reached), else 0. Returns 1
 * for an out-of-range agent. */
int ofenv_terminal(OFEnv env, int agent);

/* Number of V10 curriculum stages (ofcore::curriculum::V10_STAGE_COUNT). */
int ofenv_stage_count(void);

/* Pin the env to V10 curriculum stage `stage` (0-based): rebuilds the session
 * for that stage's bots / nations / difficulty / decision_ticks (the map is
 * kept when it is in the stage's map pool, else the pool's first map is used).
 * 0 on success, -1 on error (out-of-range stage or session build failure). */
int ofenv_set_stage(OFEnv env, int stage);

/* NUL-terminated JSON for the pinned stage (index, name, difficulty,
 * decision_ticks, bots, nations, win_at, env_target, maps, reward_profile).
 * Never NULL for a live env; valid until the next call on this handle. */
const char *ofenv_stage_info(OFEnv env);

/* Total floats in ofenv_obs() / ofenv_mask() (n_agents * per-agent stride). */
int ofenv_obs_size(OFEnv env);
int ofenv_mask_size(OFEnv env);

/* Message for the most recent failed FFI call on this thread; "" when the
 * last call succeeded. Never NULL. */
const char *ofenv_last_error(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* OPENFRONT_ENV_H */