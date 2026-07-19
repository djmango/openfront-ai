//! `ofshowcase daemon` / `ofshowcase clip` - pull policy, run watch, render clips
//! (port of eval_daemon.py).
//!
//! Clip pixels still come from Playwright + the real OpenFront client
//! (`scripts/render_client_replay.py`); Rust owns the reliable lifecycle:
//! SoftGL force, reuse-services with self-contained fallback, and hard-fail
//! when a MODEL-overlay WebM cannot be produced.

use std::fs;
use std::net::{TcpStream, ToSocketAddrs};
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
    // HF can hang; don't block the daemon loop forever when local weights exist.
    let check = tokio::time::timeout(Duration::from_secs(20), hf::policy_revision(client, run_name));
    match check.await {
        Ok(Ok(remote)) => {
            let local = fs::read_to_string(revision_path()).unwrap_or_default();
            local.trim() != remote
        }
        Ok(Err(e)) => {
            log(&format!(
                "revision check failed ({e}); keeping local policy (no force regen)"
            ));
            false
        }
        Err(_) => {
            log("revision check timed out; keeping local policy (no force regen)");
            false
        }
    }
}

fn needs_showcase(state: &Value, run_name: &str, watch_stage: i64, policy_changed: bool) -> bool {
    if policy_changed {
        return true;
    }
    let expected: std::collections::HashSet<_> = showcase_maps().into_iter().collect();
    let have: std::collections::HashSet<_> = state
        .get("maps")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|e| e.get("map").and_then(|s| s.as_str()).map(str::to_string))
        .collect();
    if !expected.is_subset(&have) {
        return true;
    }
    let artifacts_ok = state
        .get("maps")
        .and_then(Value::as_array)
        .is_some_and(|entries| {
            entries.iter().all(|entry| {
                entry
                    .get("artifact_tag")
                    .and_then(Value::as_str)
                    .is_some()
                    && entry
                    .get("clip")
                    .and_then(Value::as_str)
                    .is_some_and(|p| Path::new(p).is_file())
                    && entry
                        .get("record")
                        .and_then(Value::as_str)
                        .is_some_and(|p| {
                            let record = Path::new(p);
                            record.is_file() && debug_sidecar_path(record).is_file()
                        })
            })
        });
    if !artifacts_ok
        || state.get("run_name").and_then(Value::as_str) != Some(run_name)
        || state.get("watch_stage").and_then(Value::as_i64) != Some(watch_stage)
    {
        return true;
    }
    false
}

fn debug_sidecar_path(record: &Path) -> PathBuf {
    let raw = record.to_string_lossy();
    if let Some(stem) = raw.strip_suffix(".json") {
        PathBuf::from(format!("{stem}.debug.json"))
    } else {
        record.with_extension("debug.json")
    }
}

fn policy_artifact_tag(policy: &Path) -> String {
    let meta = policy_meta(policy).unwrap_or_else(|_| json!({}));
    if let Some(update) = meta.get("policy_update").and_then(Value::as_i64) {
        return format!("u{update}");
    }
    meta.get("policy_sha256")
        .and_then(Value::as_str)
        .map(|sha| format!("sha{}", &sha[..sha.len().min(12)]))
        .unwrap_or_else(|| "unknown".into())
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
    // Full Easy/Frenzy games often need well past 600 decisions; 600 was
    // truncating dominant positions and (before the watch fix) labeling them death.
    let max_steps = env_or("SHOWCASE_MAX_STEPS", "2500");
    let coarse = env_or(
        "SHOWCASE_COARSE_CKPT",
        "weights/ae/ae_v31_d16c32.encoder.safetensors",
    );
    let coarse_path = {
        let p = PathBuf::from(&coarse);
        if p.is_absolute() {
            p
        } else {
            repo_root().join(p)
        }
    };
    let use_cuda = device.starts_with("cuda");
    let mut args: Vec<String> = vec![
        "--watch".into(),
        "--policy".into(),
        policy.to_string_lossy().into_owned(),
        "--ckpt".into(),
        ae.to_string_lossy().into_owned(),
        "--stage".into(),
        stage.to_string(),
        "--seed".into(),
        seed.into(),
        "--record".into(),
        record.to_string_lossy().into_owned(),
        "--map".into(),
        map_name.into(),
        "--nations".into(),
        nations.into(),
        "--bots".into(),
        bots.to_string(),
        "--difficulty".into(),
        difficulty.into(),
        "--device".into(),
        device,
        "--max-steps".into(),
        max_steps,
        // MODEL overlay reads <record>.debug.json via /archive/debug/<id>.
        "--debug".into(),
        "true".into(),
        // Match the live training recipe so VarStore load is strict.
        "--foveate".into(),
    ];
    // AMP is CUDA-oriented; skip it for CPU watch (busy-pod clip default).
    if use_cuda {
        args.push("--amp".into());
    }
    if coarse_path.is_file() {
        args.push("--coarse-ckpt".into());
        args.push(coarse_path.to_string_lossy().into_owned());
    }
    // ppo_v10 (and any future v10* run) needs the V10 schedule + a reward knob
    // so `--stage` indices past the legacy 11-stage ladder stay in bounds.
    let run_hint = std::env::var("RUN_NAME").unwrap_or_default();
    let v10 = run_hint.contains("v10")
        || env_or("SHOWCASE_V10", "1") == "1"
        || stage >= 15;
    // Recurrent is required for V8.2+/V10 checkpoints; older runs stay feed-forward.
    let recurrent = match env_or("SHOWCASE_RECURRENT", "auto").as_str() {
        "1" | "true" | "yes" => true,
        "0" | "false" | "no" => false,
        _ => {
            v10
                || run_hint.contains("v9")
                || run_hint.contains("v82")
                || run_hint.contains("v83")
                || run_hint.contains("v84")
                || run_hint.contains("v85")
                || run_hint.contains("v86")
        }
    };
    if recurrent {
        args.push("--persistent-actors".into());
        args.push("--recurrent-policy".into());
    }
    if v10 {
        args.extend([
            "--v10-curriculum".into(),
            "--v10-survival-coef".into(),
            "0.01".into(),
            "--v10-diplo-panic".into(),
            "0.08".into(),
            "--v10-combat-action".into(),
            "0.02".into(),
            "--max-episode-ticks".into(),
            "21000".into(),
        ]);
    }
    let status = Command::new(&bin)
        .args(&args)
        .current_dir(repo_root())
        .status()
        .with_context(|| format!("spawn {}", bin.display()))?;
    if !status.success() {
        bail!("oftrain --watch failed with {status}");
    }
    Ok(())
}

fn port_open(host: &str, port: u16) -> bool {
    let Ok(mut addrs) = (host, port).to_socket_addrs() else {
        return false;
    };
    let Some(addr) = addrs.next() else {
        return false;
    };
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

fn showcase_services_ready() -> bool {
    // archive API + vite client (docker entrypoint / local stack).
    port_open("127.0.0.1", 8987) && port_open("127.0.0.1", 9000)
}

fn render_client_clip_once(record: &Path, out: &Path, reuse_services: bool) -> Result<()> {
    let py = std::env::var("PYTHON").unwrap_or_else(|_| {
        // Prefer a venv with Playwright (Docker image, then repo `.venv`).
        let repo_venv = repo_root().join(".venv/bin/python");
        if Path::new("/app/.venv/bin/python").is_file() {
            "/app/.venv/bin/python".into()
        } else if repo_venv.is_file() {
            repo_venv.to_string_lossy().into_owned()
        } else {
            "python3".into()
        }
    });
    let max_sec = env_or("CLIP_MAX_SEC", "90");
    // Prefer 1080p on GPU hosts; SoftGL path in the renderer downscales itself.
    let width = env_or("CLIP_WIDTH", "1920");
    let height = env_or("CLIP_HEIGHT", "1080");
    let crf = env_or("CLIP_CRF", "18");
    let dpr = env_or("CLIP_DEVICE_SCALE_FACTOR", "1");
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut args = vec![
        "scripts/render_client_replay.py".to_string(),
        "--record".into(),
        record.to_string_lossy().into_owned(),
        "--out".into(),
        out.to_string_lossy().into_owned(),
        "--trim-gameplay".into(),
        "--max-duration".into(),
        max_sec,
        "--width".into(),
        width,
        "--height".into(),
        height,
        "--device-scale-factor".into(),
        dpr,
        "--crf".into(),
        crf,
    ];
    if reuse_services {
        args.push("--reuse-services".into());
    }
    let status = Command::new(py)
        .args(&args)
        // SoftGL is the supported headless path; GPU WebGL still falls back
        // in Chromium and then hits the unpatched WebGL gate without this.
        .env("OF_FORCE_SWIFTSHADER", "1")
        .current_dir(repo_root())
        .status()?;
    if !status.success() {
        bail!(
            "render_client_replay.py failed with {status} (reuse_services={reuse_services})"
        );
    }
    if !out.is_file() {
        bail!(
            "render_client_replay.py exited 0 but {} missing",
            out.display()
        );
    }
    Ok(())
}

/// Render a MODEL-overlay WebM. Prefer live archive+vite when healthy, then
/// fall back to a self-contained patched SoftGL worktree (reliable on pods).
fn render_client_clip(record: &Path, out: &Path) -> Result<()> {
    let prefer_reuse = env_or("CLIP_REUSE_SERVICES", "auto");
    let try_reuse = match prefer_reuse.as_str() {
        "1" | "true" | "yes" => true,
        "0" | "false" | "no" => false,
        _ => showcase_services_ready(),
    };
    if try_reuse {
        match render_client_clip_once(record, out, true) {
            Ok(()) => return Ok(()),
            Err(e) => log(&format!(
                "reuse-services render failed ({e}); retrying self-contained SoftGL worktree"
            )),
        }
    }
    render_client_clip_once(record, out, false)
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
    artifact_tag: &str,
) -> Result<Value> {
    let map_key = map_seed(map_name);
    // Policy version is part of both the filename and game seed. Without it,
    // every update reused stale records and collided on the same gameID.
    let seed = format!("{run_name}-s{watch_stage}-{artifact_tag}-{map_key}");
    let base = format!("{run_name}_s{watch_stage}_{artifact_tag}_{map_key}");
    let record = records_dir().join(format!("{base}.json"));
    let clip = clips_dir().join(format!("{base}.webm"));

    // Legacy Pangaea migration from showcase0 naming.
    if artifact_tag == "unknown" && !record.exists() && map_name == "Pangaea" {
        let legacy = records_dir().join(format!("{run_name}_s{stage}_showcase0.json"));
        if legacy.is_file() {
            fs::copy(&legacy, &record)?;
            log(&format!("clip {map_name}: migrated legacy {}", legacy.display()));
        }
    }
    if artifact_tag == "unknown" && !clip.exists() && map_name == "Pangaea" {
        let legacy_clip = clips_dir().join("showcase0.webm");
        if legacy_clip.is_file() {
            fs::copy(&legacy_clip, &clip)?;
            log(&format!(
                "clip {map_name}: migrated legacy {}",
                legacy_clip.display()
            ));
        }
    }

    let debug_sidecar = debug_sidecar_path(&record);
    if !record.exists() || !debug_sidecar.exists() {
        log(&format!(
            "clip {map_name}: oftrain --watch stage {watch_stage} -> {} (debug={})",
            record.display(),
            debug_sidecar.display()
        ));
        // Same seed => same gameID, so existing clips stay valid when we only
        // backfill a missing MODEL overlay sidecar.
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
        // Record stays on disk for Watch even if this fails; hard-fail the
        // map so showcase never marks "ready" without a MODEL-overlay WebM.
        render_client_clip(&record, &clip).with_context(|| {
            format!(
                "clip {map_name}: MODEL-overlay render failed (record kept at {})",
                record.display()
            )
        })?;
    } else {
        log(&format!("clip {map_name}: reusing {}", clip.display()));
    }
    if !clip.is_file() {
        bail!("clip {map_name}: expected WebM at {}", clip.display());
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
    Ok(json!({
        "seed": seed,
        "map_key": map_key,
        "artifact_tag": artifact_tag,
        "game_id": game_id,
        "map": map,
        "record": record.display().to_string(),
        "clip": clip.display().to_string(),
        "url": format!("/archive/clips/{base}.webm"),
        "generated_at": utc_now(),
    }))
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
        // Legacy field: featured map is the newest entry, not hourly rotation.
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
    let artifact_tag = policy_artifact_tag(policy);
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
            &artifact_tag,
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
    // Best-effort: flush any spooled GameRecords to HF parquet.
    let upload = Command::new(repo_root().join("rust/target/release/ofhf"))
        .args(["replays", "--min-files", "1"])
        .current_dir(repo_root())
        .status();
    match upload {
        Ok(s) if s.success() => log("replay spool flushed to Hugging Face"),
        Ok(s) => log(&format!("replay spool flush skipped ({s})")),
        Err(e) => log(&format!("replay spool flush skipped ({e})")),
    }
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

/// One-shot MODEL-overlay clip generation (no forever loop).
#[derive(Debug, Clone)]
pub struct ClipConfig {
    pub run_name: String,
    pub stage: i64,
    pub watch_stage: i64,
    pub nations: String,
    pub bots: i64,
    pub difficulty: String,
    /// Map keys (`Onion`, `Pangaea`, …). Empty ⇒ full shuffled curriculum pool.
    pub maps: Vec<String>,
    /// Optional local policy `.safetensors` (skips HF pull).
    pub policy: Option<PathBuf>,
    /// Force re-watch + re-render even when artifacts exist.
    pub force: bool,
}

impl Default for ClipConfig {
    fn default() -> Self {
        let stage: i64 = env_or("STAGE", "27").parse().unwrap_or(27);
        Self {
            run_name: env_or("RUN_NAME", "ppo_v10"),
            stage,
            watch_stage: env_or("SHOWCASE_WATCH_STAGE", &stage.to_string())
                .parse()
                .unwrap_or(stage),
            nations: env_or("SHOWCASE_NATIONS", "4"),
            bots: env_or("SHOWCASE_BOTS", "24").parse().unwrap_or(24),
            difficulty: env_or("SHOWCASE_DIFFICULTY", "Easy"),
            maps: Vec::new(),
            policy: None,
            force: false,
        }
    }
}

/// Pull (or use) a policy, run `oftrain --watch` + SoftGL client render for
/// each map, write `state.json`, and exit. Hard-fails if any map cannot produce
/// a MODEL-overlay WebM.
pub async fn run_clip(cfg: ClipConfig) -> Result<Value> {
    fs::create_dir_all(data_dir())?;
    fs::create_dir_all(records_dir())?;
    fs::create_dir_all(clips_dir())?;

    // One-shot clips often run on busy training hosts. Prefer CPU unless the
    // caller explicitly pins a GPU via SHOWCASE_DEVICE (daemon keeps cuda).
    if std::env::var_os("SHOWCASE_DEVICE").is_none() {
        std::env::set_var("SHOWCASE_DEVICE", "cpu");
        log("clip: SHOWCASE_DEVICE defaulting to cpu (set explicitly to use GPU)");
    }
    if std::env::var_os("OF_FORCE_SWIFTSHADER").is_none() {
        std::env::set_var("OF_FORCE_SWIFTSHADER", "1");
    }

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

    let policy = if let Some(p) = cfg.policy {
        if !p.is_file() {
            bail!("policy not found: {}", p.display());
        }
        p
    } else {
        hf::ensure_policy(&client, &cfg.run_name).await?
    };

    let maps = if cfg.maps.is_empty() {
        showcase_maps()
    } else {
        cfg.maps
    };
    if maps.is_empty() {
        bail!("no maps requested");
    }

    let artifact_tag = policy_artifact_tag(&policy);
    log(&format!(
        "clip once: run={} stage={} watch_stage={} bots={} nations={} maps={:?} force={} softgl=1",
        cfg.run_name,
        cfg.stage,
        cfg.watch_stage,
        cfg.bots,
        cfg.nations,
        maps,
        cfg.force
    ));

    let mut clip_infos = Vec::new();
    for map_name in &maps {
        if cfg.force {
            let map_key = map_seed(map_name);
            let base = format!(
                "{}_s{}_{}_{}",
                cfg.run_name, cfg.watch_stage, artifact_tag, map_key
            );
            let _ = fs::remove_file(records_dir().join(format!("{base}.json")));
            let _ = fs::remove_file(records_dir().join(format!("{base}.debug.json")));
            let _ = fs::remove_file(clips_dir().join(format!("{base}.webm")));
        }
        let info = generate_clip(
            &policy,
            &ae,
            map_name,
            &cfg.run_name,
            cfg.watch_stage,
            cfg.stage,
            &cfg.nations,
            cfg.bots,
            &cfg.difficulty,
            &artifact_tag,
        )?;
        clip_infos.push(info);
        let state =
            write_showcase_state(&clip_infos, &policy, &cfg.run_name, cfg.stage, cfg.watch_stage)?;
        let _ = write_json(&state_path(), &state);
        log(&format!(
            "clip ready: {} -> {}",
            map_name,
            clip_infos
                .last()
                .and_then(|c| c.get("url"))
                .and_then(|u| u.as_str())
                .unwrap_or("?")
        ));
    }

    let mut state =
        write_showcase_state(&clip_infos, &policy, &cfg.run_name, cfg.stage, cfg.watch_stage)?;
    if let Some(obj) = state.as_object_mut() {
        obj.insert("status".into(), json!("ready"));
        obj.insert("status_message".into(), Value::Null);
    }
    write_json(&state_path(), &state)?;
    if let Ok(rev) = hf::policy_revision(&client, &cfg.run_name).await {
        let _ = fs::write(revision_path(), rev);
    }
    log(&format!(
        "clip once complete: {} WebM(s) with MODEL overlay",
        clip_infos.len()
    ));
    Ok(state)
}

pub async fn run_daemon() -> Result<()> {
    fs::create_dir_all(data_dir())?;
    // Defaults track the live V10 Easy-ramp run (Onion ~24 bots / 4 nations).
    let run_name = env_or("RUN_NAME", "ppo_v10");
    let stage: i64 = env_or("STAGE", "27").parse().unwrap_or(27);
    let watch_stage: i64 = env_or("SHOWCASE_WATCH_STAGE", &stage.to_string())
        .parse()
        .unwrap_or(stage);
    let nations = env_or("SHOWCASE_NATIONS", "4");
    let bots: i64 = env_or("SHOWCASE_BOTS", "24").parse().unwrap_or(24);
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
