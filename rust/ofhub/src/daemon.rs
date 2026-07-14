//! `ofshowcase daemon` - pull policy, run watch, render clips (port of eval_daemon.py).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use crate::hf;
use crate::paths::{clips_dir, data_dir, records_dir, repo_root, revision_path, state_path};
use crate::util::{
    featured_showcase_entry, load_json, map_seed, policy_meta, showcase_maps, utc_now, write_json,
};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn log(msg: &str) {
    println!("[eval_daemon] {msg}");
}

fn oftrain_bin() -> PathBuf {
    if let Ok(p) = std::env::var("OFTRAIN_BIN") {
        return PathBuf::from(p);
    }
    let release = repo_root().join("rust/target/release/oftrain");
    if release.exists() {
        return release;
    }
    repo_root().join("rust/target/debug/oftrain")
}

async fn policy_changed(client: &hf_hub::HFClient, run_name: &str) -> bool {
    if !revision_path().exists() {
        return true;
    }
    match hf::policy_revision(client, run_name).await {
        Ok(remote) => {
            let local = fs::read_to_string(revision_path()).unwrap_or_default();
            local.trim() != remote
        }
        Err(e) => {
            log(&format!("revision check failed ({e}); regenerating"));
            true
        }
    }
}

fn needs_showcase(state: &Value, run_name: &str, watch_stage: i64, policy_changed: bool) -> bool {
    if policy_changed {
        return true;
    }
    let expected: std::collections::HashSet<_> =
        showcase_maps().into_iter().map(|m| map_seed(&m)).collect();
    let have: std::collections::HashSet<_> = state
        .get("maps")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|e| e.get("seed").and_then(|s| s.as_str()).map(str::to_string))
        .collect();
    if !expected.is_subset(&have) {
        return true;
    }
    for map_name in showcase_maps() {
        let seed = map_seed(&map_name);
        if !clips_dir().join(format!("{seed}.webm")).is_file() {
            return true;
        }
        let record = records_dir().join(format!("{run_name}_s{watch_stage}_{seed}.json"));
        if !record.is_file() {
            return true;
        }
    }
    false
}

fn run_watch(
    policy: &Path,
    ae: &Path,
    seed: &str,
    record: &Path,
    stage: i64,
    map_name: &str,
    nations: &str,
    bots: i64,
    difficulty: &str,
) -> Result<()> {
    let bin = oftrain_bin();
    if !bin.exists() {
        bail!(
            "oftrain binary not found at {} (set OFTRAIN_BIN or build oftrain)",
            bin.display()
        );
    }
    if let Some(parent) = record.parent() {
        fs::create_dir_all(parent)?;
    }
    let device = env_or("SHOWCASE_DEVICE", "cuda");
    let max_steps = env_or("SHOWCASE_MAX_STEPS", "600");
    let status = Command::new(&bin)
        .args([
            "--watch",
            "--policy",
            &policy.to_string_lossy(),
            "--ckpt",
            &ae.to_string_lossy(),
            "--stage",
            &stage.to_string(),
            "--seed",
            seed,
            "--record",
            &record.to_string_lossy(),
            "--map",
            map_name,
            "--nations",
            nations,
            "--bots",
            &bots.to_string(),
            "--difficulty",
            difficulty,
            "--device",
            &device,
            "--max-steps",
            &max_steps,
            // Sidecar debug JSON makes CPU/GPU watch much slower.
            "--debug",
            "false",
        ])
        .current_dir(repo_root())
        .status()
        .with_context(|| format!("spawn {}", bin.display()))?;
    if !status.success() {
        bail!("oftrain --watch failed with {status}");
    }
    Ok(())
}

fn render_client_clip(record: &Path, out: &Path) -> Result<()> {
    let py = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
    let max_sec = env_or("CLIP_MAX_SEC", "90");
    let width = env_or("CLIP_WIDTH", "1920");
    let height = env_or("CLIP_HEIGHT", "1080");
    let crf = env_or("CLIP_CRF", "18");
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    let status = Command::new(py)
        .arg("scripts/render_client_replay.py")
        .args([
            "--record",
            &record.to_string_lossy(),
            "--out",
            &out.to_string_lossy(),
            "--reuse-services",
            "--trim-gameplay",
            "--max-duration",
            &max_sec,
            "--width",
            &width,
            "--height",
            &height,
            "--crf",
            &crf,
        ])
        .current_dir(repo_root())
        .status()?;
    if !status.success() {
        bail!("render_client_replay.py failed with {status}");
    }
    Ok(())
}

fn generate_clip(
    policy: &Path,
    ae: &Path,
    map_name: &str,
    run_name: &str,
    watch_stage: i64,
    stage: i64,
    nations: &str,
    bots: i64,
    difficulty: &str,
) -> Result<Value> {
    let seed = map_seed(map_name);
    let base = format!("{run_name}_s{watch_stage}_{seed}");
    let record = records_dir().join(format!("{base}.json"));
    let clip = clips_dir().join(format!("{seed}.webm"));

    // Legacy Pangaea migration from showcase0 naming.
    if !record.exists() && map_name == "Pangaea" {
        let legacy = records_dir().join(format!("{run_name}_s{stage}_showcase0.json"));
        if legacy.is_file() {
            fs::copy(&legacy, &record)?;
            log(&format!("clip {map_name}: migrated legacy {}", legacy.display()));
        }
    }
    if !clip.exists() && map_name == "Pangaea" {
        let legacy_clip = clips_dir().join("showcase0.webm");
        if legacy_clip.is_file() {
            fs::copy(&legacy_clip, &clip)?;
            log(&format!(
                "clip {map_name}: migrated legacy {}",
                legacy_clip.display()
            ));
        }
    }

    if !record.exists() {
        log(&format!(
            "clip {map_name}: oftrain --watch stage {watch_stage} -> {}",
            record.display()
        ));
        run_watch(
            policy, ae, &seed, &record, watch_stage, map_name, nations, bots, difficulty,
        )?;
    } else {
        log(&format!("clip {map_name}: reusing {}", record.display()));
    }
    if !clip.exists() {
        log(&format!(
            "clip {map_name}: render client video -> {}",
            clip.display()
        ));
        // Watch works from the GameRecord alone; don't fail the map if
        // Playwright times out (common while Vite is still warming).
        if let Err(e) = render_client_clip(&record, &clip) {
            log(&format!("clip {map_name}: render failed ({e}); keeping record for Watch"));
        }
    } else {
        log(&format!("clip {map_name}: reusing {}", clip.display()));
    }

    let meta: Value = serde_json::from_str(&fs::read_to_string(&record)?)?;
    let game_id = meta
        .pointer("/info/gameID")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let map = meta
        .pointer("/info/map")
        .and_then(|v| v.as_str())
        .unwrap_or(map_name)
        .to_string();
    let mut info = json!({
        "seed": seed,
        "game_id": game_id,
        "map": map,
        "record": record.display().to_string(),
        "clip": clip.display().to_string(),
    });
    if clip.is_file() {
        info["url"] = json!(format!("/archive/clips/{}.webm", seed));
    }
    Ok(info)
}

fn write_showcase_state(
    clip_infos: &[Value],
    policy: &Path,
    run_name: &str,
    stage: i64,
    watch_stage: i64,
) -> Result<Value> {
    let mut state = json!({
        "maps": clip_infos,
        "run_name": run_name,
        "stage": stage,
        "watch_stage": watch_stage,
        // Legacy field: featured map is now chosen at random, not hourly.
        "rotate_hours": 1,
        "generated_at": utc_now(),
    });
    if let Ok(meta) = policy_meta(policy) {
        if let Some(obj) = state.as_object_mut() {
            if let Some(m) = meta.as_object() {
                for (k, v) in m {
                    obj.insert(k.clone(), v.clone());
                }
            }
        }
    }
    // Keep every clip URL so landing/watch aren't stuck on a single map.
    let hero_urls: Vec<Value> = clip_infos
        .iter()
        .filter_map(|c| c.get("url").cloned())
        .collect();
    if !hero_urls.is_empty() {
        state["hero_clips"] = Value::Array(hero_urls);
    }
    if let Some(featured) = featured_showcase_entry(&state) {
        state["game_id"] = featured.get("game_id").cloned().unwrap_or(Value::Null);
        state["map"] = featured.get("map").cloned().unwrap_or(Value::Null);
        state["record"] = featured.get("record").cloned().unwrap_or(Value::Null);
    }
    write_json(&state_path(), &state)?;
    Ok(state)
}

async fn generate_showcase(
    client: &hf_hub::HFClient,
    policy: &Path,
    ae: &Path,
    run_name: &str,
    stage: i64,
    watch_stage: i64,
    nations: &str,
    bots: i64,
    difficulty: &str,
) -> Result<Value> {
    fs::create_dir_all(records_dir())?;
    fs::create_dir_all(clips_dir())?;

    let maps = showcase_maps();
    let first_map = maps.first().map(String::as_str).unwrap_or("showcase");
    let mut clip_infos = Vec::new();
    let mut state = json!({
        "status": "generating",
        "status_message": format!("Generating {first_map} replay…"),
        "run_name": run_name,
        "stage": stage,
        "watch_stage": watch_stage,
        "maps": [],
        "updated_at": utc_now(),
    });
    write_json(&state_path(), &state)?;

    for map_name in &maps {
        state["status"] = json!("generating");
        state["status_message"] = json!(format!("Generating {map_name} replay…"));
        state["updated_at"] = json!(utc_now());
        let _ = write_json(&state_path(), &state);

        match generate_clip(
            policy,
            ae,
            map_name,
            run_name,
            watch_stage,
            stage,
            nations,
            bots,
            difficulty,
        ) {
            Ok(info) => {
                clip_infos.push(info);
                // Publish after each map so Watch works before the full cycle finishes.
                state = write_showcase_state(&clip_infos, policy, run_name, stage, watch_stage)?;
                if let Some(obj) = state.as_object_mut() {
                    obj.insert("status".into(), json!("partial"));
                    obj.insert(
                        "status_message".into(),
                        json!(format!("Ready: {} / {} maps", clip_infos.len(), maps.len())),
                    );
                }
                write_json(&state_path(), &state)?;
                log(&format!(
                    "showcase partial: {} clip(s) after {map_name}",
                    clip_infos.len()
                ));
            }
            Err(e) => log(&format!("clip {map_name} failed: {e}")),
        }
    }
    if clip_infos.is_empty() {
        bail!("no showcase clips generated");
    }

    let rev = hf::policy_revision(client, run_name).await?;
    fs::write(revision_path(), rev)?;
    if let Some(obj) = state.as_object_mut() {
        obj.insert("status".into(), json!("ready"));
        obj.insert("status_message".into(), Value::Null);
    }
    write_json(&state_path(), &state)?;
    log(&format!(
        "showcase ready: {} clip(s), game_id={} update={:?}",
        state
            .get("maps")
            .and_then(|v| v.as_array())
            .map(|a| a.len())
            .unwrap_or(0),
        state.get("game_id").and_then(|v| v.as_str()).unwrap_or("?"),
        state.get("policy_update")
    ));
    Ok(state)
}

fn resolve_ae_path() -> PathBuf {
    let ae = env_or("AE_CKPT", "weights/ae/ae_v31_d8c32.encoder.safetensors");
    let p = PathBuf::from(&ae);
    if p.is_absolute() {
        p
    } else {
        repo_root().join(p)
    }
}

pub async fn run_daemon() -> Result<()> {
    fs::create_dir_all(data_dir())?;
    let run_name = env_or("RUN_NAME", "ppo_v81");
    let stage: i64 = env_or("STAGE", "4").parse().unwrap_or(4);
    let watch_stage: i64 = env_or("SHOWCASE_WATCH_STAGE", &stage.to_string())
        .parse()
        .unwrap_or(stage);
    let nations = env_or("SHOWCASE_NATIONS", "disabled");
    let bots: i64 = env_or("SHOWCASE_BOTS", "30").parse().unwrap_or(30);
    let difficulty = env_or("SHOWCASE_DIFFICULTY", "Easy");
    let refresh_hours: f64 = env_or("REFRESH_HOURS", "1").parse().unwrap_or(1.0);

    let ae = resolve_ae_path();
    let client = hf::client_with_optional_token()?;
    if !ae.exists() {
        let name = ae
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("ae_v31_d8c32.encoder.safetensors");
        log(&format!("fetching AE encoder {name}"));
        let _ = hf::ensure_ae_encoder(&client, name, &ae).await;
    }

    loop {
        let mut sleep_hours = refresh_hours;
        match (async {
            let changed = policy_changed(&client, &run_name).await;
            let state = load_json(&state_path())?;
            if needs_showcase(&state, &run_name, watch_stage, changed) {
                let policy = hf::ensure_policy(&client, &run_name).await?;
                generate_showcase(
                    &client,
                    &policy,
                    &ae,
                    &run_name,
                    stage,
                    watch_stage,
                    &nations,
                    bots,
                    &difficulty,
                )
                .await?;
                let state2 = load_json(&state_path())?;
                if needs_showcase(&state2, &run_name, watch_stage, false) {
                    sleep_hours = refresh_hours.min(0.25);
                    log(&format!("showcase incomplete; retry in {sleep_hours}h"));
                }
            } else {
                log(&format!(
                    "showcase ready ({run_name}); next check in {refresh_hours}h"
                ));
            }
            Ok::<(), anyhow::Error>(())
        })
        .await
        {
            Ok(()) => {}
            Err(e) => {
                log(&format!("showcase generation failed: {e}"));
                let mut state = load_json(&state_path()).unwrap_or_else(|_| json!({}));
                if let Some(obj) = state.as_object_mut() {
                    obj.insert("error".into(), json!(e.to_string()));
                    obj.insert("failed_at".into(), json!(utc_now()));
                }
                let _ = write_json(&state_path(), &state);
                sleep_hours = refresh_hours.min(0.25);
            }
        }
        tokio::time::sleep(Duration::from_secs_f64(sleep_hours.max(0.25) * 3600.0)).await;
    }
}
