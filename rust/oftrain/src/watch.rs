//! One watch episode → GameRecord + `.debug.json` + compact `.thinking.json`
//! (few-KB top-3 trace for HF parquet).
//!
//! Defaults match the historical showcase path (Node + greedy). Pass
//! `--engine native` and/or `--watch-stochastic` to A/B against training.

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
    /// Same cap as training (`--max-episode-ticks` /
    /// `ofcore::DEFAULT_MAX_EPISODE_TICKS`). Watch stops on win/death or when
    /// the sim tick reaches this — trainer and watch share one tick budget.
    pub max_episode_ticks: i64,
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
    /// Simulation backend for the watch episode (`native` or `node`).
    pub engine: EngineKind,
    /// When true, sample actions from the policy; when false (default), argmax.
    pub stochastic: bool,
}

fn choice_from_act(
    act: i64,
    player: i64,
    tile: i64,
    unit: i64,
    build: i64,
    nuke: i64,
    qty: f32,
) -> Choice {
    let name = ACTIONS[act as usize];
    let np = policy::needs_player(name);
    let nt = policy::needs_tile(name);
    let nu = policy::needs_unit(name);
    let nq = policy::needs_quantity(name);
    let is_build = name == "build";
    let is_nuke = name == "launch_nuke";
    Choice {
        action: act,
        player_slot: np.then_some(player),
        tile_region: nt.then_some(tile),
        unit_index: nu.then_some(unit),
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
    // No map override ⇒ sample the stage pool (do not default to maps[0]/Onion).
    let map_name = match cfg.map.as_deref() {
        Some(m) => m,
        None if st.maps.is_empty() => bail!("stage {} has empty map pool", cfg.stage),
        None => {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            *st.maps
                .choose(&mut rng)
                .ok_or_else(|| anyhow::anyhow!("stage {} has empty map pool", cfg.stage))?
        }
    };
    let bots = cfg.bots.unwrap_or(st.bots);
    let difficulty = cfg.difficulty.as_deref().unwrap_or(st.difficulty);
    let nations = resolve_nations(st, cfg.nations.as_deref());

    let greedy = !cfg.stochastic;
    let engine_label = match cfg.engine {
        EngineKind::Native => "native",
        EngineKind::Node => "node",
    };
    let sample_label = if greedy { "greedy" } else { "stochastic" };
    println!(
        "stage {}: {map_name}, nations={nations}, {bots} bots, {difficulty} \
         engine={engine_label} sample={sample_label}",
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
    // Keep AE on the non-persistent load path: enabling bf16/shared-pair
    // encode here *increased* peak VRAM for batch-1 watch on large maps.
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
        cfg.max_episode_ticks,
        cfg.engine,
        cfg.reward_config,
        cfg.curriculum_schedule,
    )?;
    worker.reset_watch(map_name, &cfg.seed, bots, difficulty, nations)?;

    let mut terrain_cache = TerrainDeviceCache::new(cfg.device);
    let mut debug_log: Vec<Value> = Vec::new();
    // Only set to win/death when the episode actually ends. Hitting the
    // training tick budget or `--max-steps` while still alive is a truncation
    // ("timeout"), not a loss — the old default of "death" made strong
    // mid-game clips look like wipeouts.
    let mut episode_outcome = "timeout".to_string();
    let mut end_tick = 0i64;
    let mut finished = false;

    // Spawn phase: up to 8 decisions, not logged (same sampling mode as play).
    for _ in 0..8 {
        let obs = worker.current_obs().unwrap();
        if !obs.spawn_phase() {
            break;
        }
        let prepared = worker.prepare();
        // Watch must run under no_grad — otherwise each step retains the
        // autograd graph and VRAM climbs to the full GPU over ~hundreds of steps.
        let choice = tch::no_grad(|| -> Result<Choice> {
            let obs_t = batch::build_obs_with_ae_cached(
                &[&prepared],
                cfg.device,
                false,
                false,
                &ae,
                &mut terrain_cache,
            )?;
            let (a, p, t, u, b, n, q, _lp, _v) = if let Some(h) = hidden.as_ref() {
                let context =
                    crate::recurrent::context_tensor(&[prepared.prev_action.clone()], cfg.device);
                let (acts, h_out) = policy.act_with_state(&obs_t, h, &context, greedy);
                hidden = Some(h_out);
                acts
            } else {
                policy.act(&obs_t, greedy)
            };
            Ok(choice_from_act(
                a.int64_value(&[0]),
                p.int64_value(&[0]),
                t.int64_value(&[0]),
                u.int64_value(&[0]),
                b.int64_value(&[0]),
                n.int64_value(&[0]),
                q.double_value(&[0]) as f32,
            ))
        })?;
        worker.apply_watch(&choice)?;
    }
    if worker.current_obs().unwrap().spawn_phase() {
        worker.spawn_randomly_public()?;
    }

    for step in 0..cfg.max_steps {
        let prepared = worker.prepare();
        let (choice, debug_entry) = tch::no_grad(|| -> Result<(Choice, Option<Value>)> {
            let obs_t = batch::build_obs_with_ae_cached(
                &[&prepared],
                cfg.device,
                false,
                false,
                &ae,
                &mut terrain_cache,
            )?;
            let (a, p, t, u, b, n, q, _lp, v, probs) = if let Some(h) = hidden.as_ref() {
                let context =
                    crate::recurrent::context_tensor(&[prepared.prev_action.clone()], cfg.device);
                if cfg.debug {
                    let (acts, h_out) =
                        policy.act_with_state_debug(&obs_t, h, &context, greedy);
                    hidden = Some(h_out);
                    acts
                } else {
                    let (acts, h_out) = policy.act_with_state(&obs_t, h, &context, greedy);
                    hidden = Some(h_out);
                    let empty =
                        tch::Tensor::zeros(&[1, ACTIONS.len() as i64], (Kind::Float, cfg.device));
                    (
                        acts.0, acts.1, acts.2, acts.3, acts.4, acts.5, acts.6, acts.7, acts.8,
                        empty,
                    )
                }
            } else if cfg.debug {
                policy.act_with_debug(&obs_t, greedy)
            } else {
                let (a, p, t, u, b, n, q, lp, v) = policy.act(&obs_t, greedy);
                let empty =
                    tch::Tensor::zeros(&[1, ACTIONS.len() as i64], (Kind::Float, cfg.device));
                (a, p, t, u, b, n, q, lp, v, empty)
            };
            let choice = choice_from_act(
                a.int64_value(&[0]),
                p.int64_value(&[0]),
                t.int64_value(&[0]),
                u.int64_value(&[0]),
                b.int64_value(&[0]),
                n.int64_value(&[0]),
                q.double_value(&[0]) as f32,
            );

            let tick = worker.current_obs().unwrap().tick();
            let me_id = worker.current_obs().unwrap().me() as usize;
            let ents = worker.ents();
            let legal_troops = worker.legal().troops as i64;
            let me = ents.players.iter().find(|p| p.id == me_id);
            let tiles = me.map(|p| p.tiles as i64).unwrap_or(0);
            let troops = me.map(|p| p.troops as i64).unwrap_or(0);
            let desc = describe_choice(&choice, legal_troops);

            let debug_entry = if cfg.debug {
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
                Some(entry)
            } else {
                None
            };
            Ok((choice, debug_entry))
        })?;
        if let Some(entry) = debug_entry {
            debug_log.push(entry);
        }

        worker.apply_watch(&choice)?;
        let tick = worker.current_obs().unwrap().tick();
        let alive = worker.current_obs().unwrap().alive();
        let winner = worker.current_obs().unwrap().winner().clone();
        let me = worker.current_obs().unwrap().me();
        if step % 100 == 0 {
            println!(
                "step {step}, tick {tick}, my tiles {}, alive {alive}",
                my_tiles(worker.ents(), me),
            );
        }
        if !alive || !winner.is_null() {
            end_tick = tick;
            let won = winner
                .as_array()
                .map(|a| a.len() > 1 && a[1] == "AGENTRL1")
                .unwrap_or(false);
            episode_outcome = if won { "win" } else { "death" }.to_string();
            finished = true;
            println!(
                "episode over at tick {end_tick}: alive={alive}, winner={winner}, outcome={episode_outcome}",
            );
            break;
        }
        // Same tick budget as training (`--max-episode-ticks`).
        if tick >= cfg.max_episode_ticks {
            end_tick = tick;
            episode_outcome = "timeout".to_string();
            finished = true;
            println!(
                "watch hit max-episode-ticks {} at tick {end_tick} (outcome=timeout; same budget as training)",
                cfg.max_episode_ticks
            );
            break;
        }
    }
    if !finished {
        let tick = worker.current_obs().unwrap().tick();
        let alive = worker.current_obs().unwrap().alive();
        let me = worker.current_obs().unwrap().me();
        end_tick = tick;
        let tiles = my_tiles(worker.ents(), me);
        println!(
            "watch truncated at --max-steps {} before tick budget {}: tick {end_tick}, tiles {tiles}, \
             alive={alive} (outcome=timeout — raise --max-steps; tick budget is --max-episode-ticks)",
            cfg.max_steps,
            cfg.max_episode_ticks,
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
    if cfg.debug {
        let stem = {
            let s = record_path.to_string_lossy();
            s.strip_suffix(".json")
                .map(|x| x.to_string())
                .unwrap_or_else(|| record_path.to_string_lossy().to_string())
        };
        let debug_path = PathBuf::from(format!("{stem}.debug.json"));
        let thinking_path = PathBuf::from(format!("{stem}.thinking.json"));
        let payload = json!({
            "actions": ACTIONS,
            "log": debug_log,
            "outcome": episode_outcome,
            "end_tick": end_tick,
        });
        std::fs::write(&debug_path, serde_json::to_string(&payload)?)?;
        let thinking = compact_thinking(&debug_log, &episode_outcome, end_tick, 15);
        std::fs::write(&thinking_path, serde_json::to_string(&thinking)?)?;
        let thinking_bytes = std::fs::metadata(&thinking_path).map(|m| m.len()).unwrap_or(0);
        println!(
            "debug sidecar: {} ({} decisions); thinking: {} ({} bytes, {} kept)",
            debug_path.display(),
            debug_log.len(),
            thinking_path.display(),
            thinking_bytes,
            thinking
                .get("s")
                .and_then(|v| v.as_array())
                .map(|a| a.len())
                .unwrap_or(0)
        );
    }
    // Spool after sidecars exist so parquet upload packs thinking_json too.
    let spool_meta = json!({
        "map": cfg.map.clone().unwrap_or_default(),
        "stage": cfg.stage,
        "engine": engine_label,
        "sample": sample_label,
        "won": episode_outcome == "win",
        "timed_out": episode_outcome == "timeout",
        "run_name": std::env::var("HF_RUN_PREFIX")
            .or_else(|_| std::env::var("RUN_NAME"))
            .unwrap_or_default(),
        "policy_repo": std::env::var("HF_REPO_ID").unwrap_or_else(|_| "djmango/openfront-rl".into()),
        "source": "oftrain-watch",
    });
    if let Err(e) = crate::replay_spool::spool_existing_record(&record_path, &spool_meta) {
        eprintln!("[replay-spool] watch spool: {e}");
    }
    Ok(())
}

/// Stride-sampled top-3 action trace for HF parquet (`thinking_json`).
fn compact_thinking(debug_log: &[Value], outcome: &str, end_tick: i64, stride: usize) -> Value {
    let n = debug_log.len();
    let mut steps = Vec::new();
    let mut prev_a: Option<&str> = None;
    for (i, entry) in debug_log.iter().enumerate() {
        let action = entry.get("action").and_then(|v| v.as_str()).unwrap_or("noop");
        let keep = i < 3
            || i + 5 >= n
            || (stride > 0 && i % stride == 0)
            || (action != "noop" && prev_a != Some(action));
        prev_a = Some(action);
        if !keep {
            continue;
        }
        let tick = entry.get("tick").and_then(|v| v.as_i64()).unwrap_or(0);
        let value = entry.get("value").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let a_idx = ACTIONS
            .iter()
            .position(|&a| a == action)
            .unwrap_or(255) as i64;
        let probs = entry
            .get("probs")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        let mut order: Vec<usize> = (0..probs.len()).collect();
        order.sort_by(|&a, &b| {
            let pa = probs[a].as_f64().unwrap_or(0.0);
            let pb = probs[b].as_f64().unwrap_or(0.0);
            pb.partial_cmp(&pa).unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut row = vec![
            json!(tick),
            json!(a_idx),
            json!((value * 100.0).round() as i64),
        ];
        for &k in order.iter().take(3) {
            let p = probs[k].as_f64().unwrap_or(0.0);
            row.push(json!(k as i64));
            row.push(json!((p * 1000.0).round() as i64));
        }
        if action != "noop" {
            let desc = entry
                .get("desc")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let short: String = desc.chars().take(48).collect();
            row.push(json!(short));
        }
        steps.push(json!(row));
    }
    json!({
        "v": 1,
        "o": outcome,
        "T": end_tick,
        "n": n,
        "stride": stride,
        "a": ACTIONS,
        "s": steps,
    })
}
