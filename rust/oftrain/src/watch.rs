//! One greedy Node-engine episode → GameRecord + `.debug.json` sidecar.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use ofcore::curriculum::{self, CurriculumSchedule, Nations, RewardConfig};
use ofcore::feat::{self, ACTIONS, BUILD_TYPES, GW_MAX, NUKE_TYPES};
use ofcore::translate::{my_tiles, Choice};
use serde_json::{json, Value};
use tch::{nn, Device, Kind};

use crate::ae::{AePair, TerrainDeviceCache};
use crate::batch;
use crate::engine::EngineKind;
use crate::policy::{self, PolicyNet};
use crate::vecenv::EnvWorker;

pub struct WatchConfig<'a> {
    pub policy: &'a str,
    pub record: PathBuf,
    pub ae_ckpt: &'a str,
    pub coarse_ckpt: Option<&'a str>,
    pub stage: usize,
    pub seed: String,
    pub map: Option<String>,
    pub bots: Option<u32>,
    pub difficulty: Option<String>,
    pub nations: Option<String>,
    pub max_steps: usize,
    pub debug: bool,
    pub device: Device,
    pub amp: bool,
    pub foveate: bool,
    pub gc: i64,
    pub blocks: i64,
    pub curriculum_schedule: CurriculumSchedule,
    pub reward_config: RewardConfig,
    /// Match training: V8.2+ / V10 policies need recurrent hidden state.
    pub recurrent_policy: bool,
}

fn choice_from_act(
    act: i64,
    player: i64,
    tile: i64,
    build: i64,
    nuke: i64,
    qty: f32,
) -> Choice {
    let name = ACTIONS[act as usize];
    let np = policy::needs_player(name);
    let nt = policy::needs_tile(name);
    let nq = policy::needs_quantity(name);
    let is_build = name == "build";
    let is_nuke = name == "launch_nuke";
    Choice {
        action: act,
        player_slot: np.then_some(player),
        tile_region: nt.then_some(tile),
        build_type: is_build.then_some(build),
        nuke_type: is_nuke.then_some(nuke),
        quantity_frac: nq.then_some(qty as f64),
    }
}

fn describe_choice(choice: &Choice, legal_troops: i64) -> String {
    let name = ACTIONS[choice.action as usize];
    let mut parts = vec![name.to_string()];
    if let Some(slot) = choice.player_slot {
        parts.push(format!("-> P{slot}"));
    }
    if let Some(region) = choice.tile_region {
        let gy = region / GW_MAX;
        let gx = region % GW_MAX;
        parts.push(format!("@({gx},{gy})"));
    }
    if let Some(bt) = choice.build_type {
        if (bt as usize) < BUILD_TYPES.len() {
            parts.push(BUILD_TYPES[bt as usize].to_string());
        }
    }
    if let Some(nt) = choice.nuke_type {
        if (nt as usize) < NUKE_TYPES.len() {
            let (unit, up) = NUKE_TYPES[nt as usize];
            parts.push(match up {
                Some(true) => format!("{unit} up"),
                Some(false) => format!("{unit} down"),
                None => unit.to_string(),
            });
        }
    }
    if let Some(frac) = choice.quantity_frac {
        let troops = (legal_troops as f64 * frac) as i64;
        parts.push(format!("{:.0}% ({troops})", frac * 100.0));
    }
    parts.join(" ")
}

fn resolve_nations(st: &curriculum::Stage, override_n: Option<&str>) -> Value {
    match override_n {
        None => match st.nations {
            Nations::Default => json!("default"),
            Nations::Exact(n) => json!(n),
        },
        Some("default") => json!("default"),
        Some("disabled") => json!("disabled"),
        Some(s) => {
            if let Ok(n) = s.parse::<u32>() {
                json!(n)
            } else {
                json!(s)
            }
        }
    }
}

pub fn run_watch(cfg: WatchConfig<'_>) -> Result<()> {
    let stages = curriculum::stages_for_schedule(cfg.curriculum_schedule);
    if cfg.stage >= stages.len() {
        bail!("--stage {} out of range for schedule", cfg.stage);
    }
    let st = &stages[cfg.stage];
    let map_name = cfg.map.as_deref().unwrap_or(st.maps[0]);
    let bots = cfg.bots.unwrap_or(st.bots);
    let difficulty = cfg.difficulty.as_deref().unwrap_or(st.difficulty);
    let nations = resolve_nations(st, cfg.nations.as_deref());

    println!(
        "stage {}: {map_name}, nations={nations}, {bots} bots, {difficulty}",
        cfg.stage
    );

    if let Some(parent) = cfg.record.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut vs = nn::VarStore::new(cfg.device);
    let policy = PolicyNet::new_with_recurrence(
        &vs.root(),
        cfg.amp,
        cfg.foveate,
        cfg.gc,
        cfg.blocks,
        cfg.recurrent_policy,
    );
    vs.load(cfg.policy)
        .with_context(|| format!("load policy {}", cfg.policy))?;
    let ae = AePair::load(
        Path::new(cfg.ae_ckpt),
        cfg.coarse_ckpt.map(Path::new),
        cfg.device,
        cfg.amp,
        false,
    )?;
    println!(
        "device: {:?} recurrent={}",
        cfg.device, cfg.recurrent_policy
    );
    let mut hidden = if cfg.recurrent_policy {
        Some(policy.initial_hidden(1))
    } else {
        None
    };
    {
        let stem = Path::new(cfg.policy)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("latest");
        let alt = Path::new(cfg.policy).with_file_name(format!("{stem}.state.json"));
        if let Ok(text) = std::fs::read_to_string(&alt) {
            if let Ok(state) = serde_json::from_str::<Value>(&text) {
                println!(
                    "policy from update {:?}, stage {:?}",
                    state.get("update"),
                    state.get("stage")
                );
            }
        }
    }

    let mut worker = EnvWorker::new(
        0,
        cfg.stage,
        15000,
        EngineKind::Node,
        cfg.reward_config,
        cfg.curriculum_schedule,
    )?;
    worker.reset_watch(map_name, &cfg.seed, bots, difficulty, nations)?;

    let mut terrain_cache = TerrainDeviceCache::new(cfg.device);
    let mut debug_log: Vec<Value> = Vec::new();
    // Only set to win/death when the episode actually ends. Hitting
    // `--max-steps` while still alive is a truncation ("timeout"), not a loss —
    // the old default of "death" made strong mid-game clips look like wipeouts.
    let mut episode_outcome = "timeout".to_string();
    let mut end_tick = 0i64;
    let mut finished = false;

    // Spawn phase: up to 8 greedy decisions, not logged.
    for _ in 0..8 {
        let obs = worker.current_obs().unwrap();
        if !obs.spawn_phase() {
            break;
        }
        let prepared = worker.prepare();
        let obs_t = batch::build_obs_with_ae_cached(
            &[&prepared],
            cfg.device,
            false,
            false,
            &ae,
            &mut terrain_cache,
        )?;
        let (a, p, t, b, n, q, _lp, _v) = if let Some(h) = hidden.as_ref() {
            let context = crate::recurrent::context_tensor(&[prepared.prev_action.clone()], cfg.device);
            let (acts, h_out) = policy.act_with_state(&obs_t, h, &context, true);
            hidden = Some(h_out);
            acts
        } else {
            policy.act(&obs_t, true)
        };
        let choice = choice_from_act(
            a.int64_value(&[0]),
            p.int64_value(&[0]),
            t.int64_value(&[0]),
            b.int64_value(&[0]),
            n.int64_value(&[0]),
            q.double_value(&[0]) as f32,
        );
        worker.apply_watch(&choice)?;
    }
    if worker.current_obs().unwrap().spawn_phase() {
        worker.spawn_randomly_public()?;
    }

    for step in 0..cfg.max_steps {
        let prepared = worker.prepare();
        let obs_t = batch::build_obs_with_ae_cached(
            &[&prepared],
            cfg.device,
            false,
            false,
            &ae,
            &mut terrain_cache,
        )?;
        let (a, p, t, b, n, q, _lp, v, probs) = if let Some(h) = hidden.as_ref() {
            let context = crate::recurrent::context_tensor(&[prepared.prev_action.clone()], cfg.device);
            if cfg.debug {
                let (acts, h_out) = policy.act_with_state_debug(&obs_t, h, &context, true);
                hidden = Some(h_out);
                acts
            } else {
                let (acts, h_out) = policy.act_with_state(&obs_t, h, &context, true);
                hidden = Some(h_out);
                let empty =
                    tch::Tensor::zeros(&[1, ACTIONS.len() as i64], (Kind::Float, cfg.device));
                (
                    acts.0, acts.1, acts.2, acts.3, acts.4, acts.5, acts.6, acts.7, empty,
                )
            }
        } else if cfg.debug {
            policy.act_with_debug(&obs_t, true)
        } else {
            let (a, p, t, b, n, q, lp, v) = policy.act(&obs_t, true);
            let empty = tch::Tensor::zeros(&[1, ACTIONS.len() as i64], (Kind::Float, cfg.device));
            (a, p, t, b, n, q, lp, v, empty)
        };
        let choice = choice_from_act(
            a.int64_value(&[0]),
            p.int64_value(&[0]),
            t.int64_value(&[0]),
            b.int64_value(&[0]),
            n.int64_value(&[0]),
            q.double_value(&[0]) as f32,
        );

        let obs = worker.current_obs().unwrap();
        let tick = obs.tick();
        let ents = feat::parse_ents(obs.entities());
        let legal = feat::parse_legal(obs.legal_actions());
        let me = ents.players.iter().find(|p| p.id == obs.me() as usize);
        let tiles = me.map(|p| p.tiles as i64).unwrap_or(0);
        let troops = me.map(|p| p.troops as i64).unwrap_or(0);
        let desc = describe_choice(&choice, legal.troops as i64);

        if cfg.debug {
            let value = (v.double_value(&[0]) * 1000.0).round() / 1000.0;
            let probs_v: Vec<f64> = (0..ACTIONS.len())
                .map(|i| {
                    let p = probs.double_value(&[0, i as i64]);
                    (p * 10000.0).round() / 10000.0
                })
                .collect();
            let mut entry = json!({
                "tick": tick,
                "desc": desc,
                "action": ACTIONS[choice.action as usize],
                "tiles": tiles,
                "troops": troops,
                "value": value,
                "probs": probs_v,
            });
            // World tile for MODEL overlay red X (matches rl/watch.action_tile_xy).
            if let Some(region) = choice.tile_region {
                let gy = region / GW_MAX;
                let gx = region % GW_MAX;
                let r = feat::REGION as i64;
                entry["tile_x"] = json!(gx * r + r / 2);
                entry["tile_y"] = json!(gy * r + r / 2);
            }
            debug_log.push(entry);
        }

        worker.apply_watch(&choice)?;
        let obs = worker.current_obs().unwrap();
        if step % 100 == 0 {
            let ents = feat::parse_ents(obs.entities());
            println!(
                "step {step}, tick {}, my tiles {}, alive {}",
                obs.tick(),
                my_tiles(&ents, obs.me()),
                obs.alive()
            );
        }
        if !obs.alive() || !obs.winner().is_null() {
            end_tick = obs.tick();
            let w = obs.winner();
            let won = w
                .as_array()
                .map(|a| a.len() > 1 && a[1] == "AGENTRL1")
                .unwrap_or(false);
            episode_outcome = if won { "win" } else { "death" }.to_string();
            finished = true;
            println!(
                "episode over at tick {end_tick}: alive={}, winner={w}, outcome={episode_outcome}",
                obs.alive()
            );
            break;
        }
    }
    if !finished {
        let obs = worker.current_obs().unwrap();
        end_tick = obs.tick();
        let ents = feat::parse_ents(obs.entities());
        let tiles = my_tiles(&ents, obs.me());
        println!(
            "watch truncated at --max-steps {}: tick {end_tick}, tiles {tiles}, alive={} \
             (outcome=timeout — not a loss; raise SHOWCASE_MAX_STEPS / --max-steps to play out)",
            cfg.max_steps,
            obs.alive()
        );
    }

    let record_path = cfg.record.canonicalize().unwrap_or(cfg.record.clone());
    let info = worker.save_record(record_path.to_str().unwrap())?;
    println!(
        "game record: {:?} (gameID {:?}, {:?} turns)",
        info.get("saved"),
        info.get("gameID"),
        info.get("turns")
    );
    let spool_meta = json!({
        "map": cfg.map.clone().unwrap_or_default(),
        "stage": cfg.stage,
        "engine": "watch",
        "won": episode_outcome == "win",
        "timed_out": false,
        "run_name": std::env::var("HF_RUN_PREFIX")
            .or_else(|_| std::env::var("RUN_NAME"))
            .unwrap_or_default(),
        "policy_repo": std::env::var("HF_REPO_ID").unwrap_or_else(|_| "djmango/openfront-rl".into()),
        "source": "oftrain-watch",
    });
    if let Err(e) = crate::replay_spool::spool_existing_record(&record_path, &spool_meta) {
        eprintln!("[replay-spool] watch spool: {e}");
    }
    if cfg.debug {
        let sidecar = {
            let s = cfg.record.to_string_lossy();
            if let Some(stem) = s.strip_suffix(".json") {
                PathBuf::from(format!("{stem}.debug.json"))
            } else {
                cfg.record.with_extension("debug.json")
            }
        };
        let payload = json!({
            "actions": ACTIONS,
            "log": debug_log,
            "outcome": episode_outcome,
            "end_tick": end_tick,
        });
        std::fs::write(&sidecar, serde_json::to_string(&payload)?)?;
        println!(
            "debug sidecar: {} ({} decisions)",
            sidecar.display(),
            debug_log.len()
        );
    }
    Ok(())
}
