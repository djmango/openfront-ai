//! Single-env worker: bridge session + curriculum episode bookkeeping +
//! reward shaping. Port of `rl/vec.py::EnvWorker`. `VecEnv` fans this out
//! over one OS thread per env (see module-level doc below); unlike the
//! Python side there's no GIL, so no multiprocessing/pickle framing is
//! needed to keep JSON decode + featurization off the main thread.

use anyhow::Result;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde_json::Value;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ofcore::curriculum::{
    self, Stage, W_DEATH, W_DELTA_GAIN, W_DELTA_LOSS, W_STR, W_WASTE, placement, placement_score,
    sample_episode, stages, terminal_reward, timeweight,
};
use ofcore::feat::{self, ACTIONS, IS_LAND_BIT, MAG_MASK, REGION};
use ofcore::translate::{Choice, IntentTranslator, translate};

use crate::ae::{self, AeRaw, StaticTerrain, TerrainCacheKey};
use crate::engine::{self, EngineKind, GameEngine, RawObs};

/// CPU-owned foveated rollout payload. Grid samples cross the actor/learner
/// boundary as fp16 values; masks and crop metadata stay explicit so the
/// learner never has to reconstruct a full fine grid or infer coordinates.
#[derive(Clone)]
pub struct CompactGrid {
    pub fine: Vec<half::f16>, // (C_GRID, fine_h, fine_w)
    pub fine_valid: Vec<f32>, // (fine_h, fine_w)
    pub fine_legal: Vec<f32>, // (fine_h, fine_w)
    pub fine_h: usize,
    pub fine_w: usize,
    pub origin_y: i64,
    pub origin_x: i64,
    pub coarse: Vec<half::f16>, // (C_GRID, coarse_h, coarse_w)
    pub coarse_valid: Vec<f32>, // (coarse_h, coarse_w)
    pub coarse_legal: Vec<f32>, // (coarse_h, coarse_w)
    pub coarse_h: usize,
    pub coarse_w: usize,
}

#[derive(Clone)]
pub struct EpisodeInfo {
    pub reward: f64,
    pub length: i64,
    pub final_tiles: f64,
    pub final_tick: i64,
    pub place: i64,
    pub n_players: i64,
    pub score: f64,
    pub won: bool,
    pub wasted: i64,
    pub stage: usize,
    pub map: String,
    pub rehearsal: bool,
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

pub struct EnvWorker {
    pub idx: usize,
    bridge: Box<dyn GameEngine>,
    stages: Vec<Stage>,
    stage: usize,
    max_episode_ticks: i64,
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
    ) -> Result<Self> {
        let bridge = engine::create(engine)?;
        let mut w = EnvWorker {
            idx,
            bridge,
            stages: stages(),
            stage,
            max_episode_ticks,
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

    pub fn reset_episode(&mut self) -> Result<()> {
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
        self.prev_strength = 0.0;
        self.spawn_steps = 0;
        self.ep_reward = 0.0;
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

        let mut owners_slotted = vec![0u8; hr * wr];
        let mut fallout = vec![0u8; hr * wr];
        let mut defense_bonus = vec![0u8; hr * wr];
        for y in 0..hr {
            for x in 0..wr {
                let src = y * obs.head["width"].as_u64().unwrap_or(wr as u64) as usize + x;
                let raw_id = obs.owners[src] as usize;
                let dst = y * wr + x;
                owners_slotted[dst] = *lut.get(raw_id).unwrap_or(&0);
                fallout[dst] = obs.fallout[src];
                defense_bonus[dst] = obs.defense_bonus[src];
            }
        }

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

        let (ego, db) = feat::pool_ego_db(&owners_slotted, &f.clut, &defense_bonus, hr, wr);
        let local = feat::local_crop(
            &owners_slotted,
            &f.clut,
            &self.land,
            &defense_bonus,
            hr,
            wr,
            crate::policy::LOCAL as usize,
        );
        let owners_i64: Vec<i64> = owners_slotted.iter().map(|&o| o as i64).collect();
        let ae_raw = AeRaw {
            owners: owners_i64,
            static_terrain: self.ae_static.clone(),
            fallout: ae::pack_fallout(&fallout),
            stat: f.stat,
            hr,
            wr,
        };

        PreparedObs {
            compact: None,
            grid: None,
            grid_coarse: None,
            cgh: 0,
            cgw: 0,
            ae_raw,
            ego,
            db,
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
    pub fn step(
        &mut self,
        choice: &Choice,
    ) -> Result<(PreparedObs, f64, bool, Option<EpisodeInfo>)> {
        let (reward, done, info) = self.apply(choice)?;
        let prepared = self.prepare();
        Ok((prepared, reward, done, info))
    }

    /// Translate + step. Auto-resets on episode end.
    pub fn apply(&mut self, choice: &Choice) -> Result<(f64, bool, Option<EpisodeInfo>)> {
        let name = ACTIONS[choice.action as usize];
        let lut = self.current_lut();
        let obs = self.obs.as_ref().unwrap();
        let ents = feat::parse_ents(obs.entities());
        let legal = feat::parse_legal(obs.legal_actions());
        let owners_raw: Vec<i64> = obs.owners.iter().map(|&o| o as i64).collect();
        // obs.owners is at full (untrimmed width) resolution; translate
        // wants it trimmed to (hr, wr) matching the translator's grids.
        let width = obs.head["width"].as_u64().unwrap() as usize;
        let mut owners_trim = vec![0i64; self.hr * self.wr];
        for y in 0..self.hr {
            for x in 0..self.wr {
                owners_trim[y * self.wr + x] = owners_raw[y * width + x];
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
        let mut wasted = new_obs.wasted();
        if intents.is_empty() && name != "noop" && name != "spawn" {
            wasted += 1;
        }
        self.obs = Some(new_obs);
        let obs = self.obs.as_ref().unwrap();

        if obs.spawn_phase() {
            self.spawn_steps += 1;
            if self.spawn_steps >= 8 {
                // Fallback: pick a uniformly random legal spawn tile.
                self.spawn_randomly()?;
            } else {
                self.ep_len += 1;
                return Ok((0.0, false, None));
            }
        }

        let obs = self.obs.as_ref().unwrap();
        let ents = feat::parse_ents(obs.entities());
        let tiles = ofcore::translate::my_tiles(&ents, obs.me());
        let mine = curriculum::strengths(&ents, self.land_total)
            .get(&(obs.me().max(0) as usize))
            .copied()
            .unwrap_or(0.0);
        let tw = timeweight(obs.tick());
        let delta = mine - self.prev_strength;
        let mut reward = W_STR * mine * tw
            + (if delta >= 0.0 {
                W_DELTA_GAIN
            } else {
                W_DELTA_LOSS
            }) * delta;
        reward -= W_WASTE * wasted as f64;
        self.ep_wasted += wasted;
        self.prev_strength = mine;

        let mut done = false;
        let mut won = false;
        if !obs.alive() {
            reward -= W_DEATH;
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
        }

        let mut info = None;
        if done {
            let (place, n) = placement(&ents, obs.me(), obs.alive(), self.land_total);
            reward += terminal_reward(place, won);
            self.ep_reward += reward;
            self.ep_len += 1;
            info = Some(EpisodeInfo {
                reward: self.ep_reward,
                length: self.ep_len,
                final_tiles: tiles,
                final_tick: obs.tick(),
                place,
                n_players: n,
                score: placement_score(place, n),
                won,
                wasted: self.ep_wasted,
                stage: self.stage,
                map: self.map_name.clone(),
                rehearsal: self.rehearsal,
            });
            self.reset_episode()?;
        } else {
            self.ep_reward += reward;
            self.ep_len += 1;
        }
        Ok((reward, done, info))
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
                    && obs.owners[src] == 0
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
