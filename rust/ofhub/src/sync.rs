//! Checkpoint sync against Hugging Face (port of `scripts/hf_checkpoint_sync.py`).

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::hf::{self, default_policy_repo};
use crate::util::utc_now;

const RESTORE_FILES: &[&str] = &["latest.safetensors", "latest.state.json", "manifest.json"];
const EXACT_FILES: &[&str] = &[
    "latest.safetensors",
    "latest.state.json",
    "manifest.json",
    "best_eval.safetensors",
    "best_eval.state.json",
];
const CHUNK: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct SyncState {
    schema: u32,
    repo_id: String,
    run_prefix: String,
    files: HashMap<String, FileState>,
    #[serde(default)]
    updated_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FileState {
    sha256: String,
    bytes: u64,
    source_signature: Vec<i64>,
    uploaded_at: String,
}

fn normalize_prefix(prefix: &str) -> Result<String> {
    let normalized = prefix.trim_matches('/');
    if normalized.is_empty()
        || normalized
            .split('/')
            .any(|p| p.is_empty() || p == "." || p == "..")
    {
        bail!("invalid Hugging Face run prefix: {prefix:?}");
    }
    Ok(normalized.to_string())
}

fn signature(path: &Path) -> Result<[i64; 4]> {
    let meta = fs::metadata(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok([
            meta.dev() as i64,
            meta.ino() as i64,
            meta.len() as i64,
            meta.mtime_nsec(),
        ])
    }
    #[cfg(not(unix))]
    {
        Ok([0, 0, meta.len() as i64, 0])
    }
}

fn discover(checkpoint_dir: &Path) -> Result<Vec<PathBuf>> {
    if !checkpoint_dir.is_dir() {
        return Ok(Vec::new());
    }
    let milestone = regex_lite_milestone();
    let mut out = Vec::new();
    for entry in fs::read_dir(checkpoint_dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if EXACT_FILES.contains(&name) || milestone(name) {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

fn regex_lite_milestone() -> impl Fn(&str) -> bool {
    |name: &str| is_policy_update_milestone(name) || is_curriculum_milestone(name)
}

fn is_policy_update_milestone(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("policy_update") else {
        return false;
    };
    let (digits, suffix) = if let Some(s) = rest.strip_suffix(".safetensors") {
        (s, true)
    } else if let Some(s) = rest.strip_suffix(".state.json") {
        (s, true)
    } else {
        ("", false)
    };
    suffix && !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit())
}

/// Curriculum advance/demote artifacts:
/// `curriculum_{advance|demote}_u{update}_s{from}_to_{to}.{safetensors|state.json|note.json}`
fn is_curriculum_milestone(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("curriculum_") else {
        return false;
    };
    let (stem, ok_suffix) = if let Some(s) = rest.strip_suffix(".safetensors") {
        (s, true)
    } else if let Some(s) = rest.strip_suffix(".state.json") {
        (s, true)
    } else if let Some(s) = rest.strip_suffix(".note.json") {
        (s, true)
    } else {
        ("", false)
    };
    if !ok_suffix {
        return false;
    }
    let (event, after_event) = if let Some(s) = stem.strip_prefix("advance_u") {
        ("advance", s)
    } else if let Some(s) = stem.strip_prefix("demote_u") {
        ("demote", s)
    } else {
        return false;
    };
    let _ = event;
    let Some((update, after_update)) = after_event.split_once("_s") else {
        return false;
    };
    let Some((from, to)) = after_update.split_once("_to_") else {
        return false;
    };
    !update.is_empty()
        && update.chars().all(|c| c.is_ascii_digit())
        && !from.is_empty()
        && from.chars().all(|c| c.is_ascii_digit())
        && !to.is_empty()
        && to.chars().all(|c| c.is_ascii_digit())
}

fn atomic_json(path: &Path, value: &SyncState) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_file_name(format!(
        ".{}.{}.tmp",
        path.file_name().and_then(|n| n.to_str()).unwrap_or("state"),
        std::process::id()
    ));
    {
        let mut f = File::create(&tmp)?;
        serde_json::to_writer_pretty(&mut f, value)?;
        f.write_all(b"\n")?;
        f.sync_all()?;
    }
    fs::rename(tmp, path)?;
    Ok(())
}

fn snapshot(source: &Path, directory: &Path) -> Result<(PathBuf, String, u64, [i64; 4])> {
    let before = signature(source)?;
    fs::create_dir_all(directory)?;
    let tmp = directory.join(format!(
        ".{}.{}.tmp",
        source.file_name().and_then(|n| n.to_str()).unwrap_or("file"),
        std::process::id()
    ));
    let (sha, size) = {
        let mut src = File::open(source)?;
        let mut dst = File::create(&tmp)?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; CHUNK];
        let mut size = 0u64;
        loop {
            let n = src.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            dst.write_all(&buf[..n])?;
            size += n as u64;
        }
        dst.sync_all()?;
        (format!("{:x}", hasher.finalize()), size)
    };
    if signature(source)? != before {
        let _ = fs::remove_file(&tmp);
        bail!("{} changed while being snapshotted", source.display());
    }
    let stable = directory.join(format!(
        "{sha}-{}",
        source.file_name().and_then(|n| n.to_str()).unwrap_or("file")
    ));
    if stable.exists() {
        let _ = fs::remove_file(&tmp);
    } else {
        fs::rename(&tmp, &stable)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&stable)?.permissions();
            perms.set_mode(0o444);
            fs::set_permissions(&stable, perms)?;
        }
    }
    Ok((stable, sha, size, before))
}

pub struct SyncConfig {
    pub checkpoint_dir: PathBuf,
    pub repo_id: String,
    pub run_prefix: String,
    pub interval_secs: f64,
    pub max_retries: u32,
    pub dry_run: bool,
    pub once: bool,
    pub restore_latest: bool,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("checkpoints"),
            repo_id: default_policy_repo(),
            run_prefix: env_or("HF_RUN_PREFIX", "ppo_v10"),
            interval_secs: env_or("HF_SYNC_INTERVAL_SECONDS", "600")
                .parse()
                .unwrap_or(600.0),
            max_retries: 5,
            dry_run: false,
            once: false,
            restore_latest: false,
        }
    }
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

pub async fn restore_latest(
    client: &hf_hub::HFClient,
    repo_id: &str,
    run_prefix: &str,
    destination: &Path,
) -> Result<bool> {
    let prefix = normalize_prefix(run_prefix)?;
    fs::create_dir_all(destination)?;
    let mut staged = HashMap::new();
    for name in &RESTORE_FILES[..2] {
        let remote = format!("{prefix}/{name}");
        let dest = destination.join(name);
        match hf::download_to(client, repo_id, &remote, &dest).await {
            Ok(p) => {
                staged.insert(*name, p);
            }
            Err(e) => {
                eprintln!("[hf-sync] no complete safetensors checkpoint to restore: {e}");
                return Ok(false);
            }
        }
    }
    let _ = staged;
    let manifest_remote = format!("{prefix}/manifest.json");
    let manifest_dest = destination.join("manifest.json");
    let _ = hf::download_to(client, repo_id, &manifest_remote, &manifest_dest).await;
    println!("[hf-sync] restored {prefix}/latest.safetensors and state");
    Ok(true)
}

pub async fn run_sync(cfg: SyncConfig) -> Result<()> {
    if cfg.interval_secs <= 0.0 {
        bail!("interval must be positive");
    }
    let prefix = normalize_prefix(&cfg.run_prefix)?;
    if cfg.restore_latest {
        let client = hf::client_with_optional_token()?;
        let ok = restore_latest(&client, &cfg.repo_id, &prefix, &cfg.checkpoint_dir).await?;
        if ok {
            return Ok(());
        }
        bail!("restore-latest failed");
    }

    let client = if cfg.dry_run {
        hf::client_with_optional_token()?
    } else {
        let client = hf::client_requiring_token()?;
        hf::whoami(&client).await?;
        hf::ensure_repo(&client, &cfg.repo_id).await?;
        client
    };

    let state_dir = cfg.checkpoint_dir.join(".hf-sync");
    let snapshot_dir = state_dir.join("snapshots");
    let state_path = state_dir.join("sync-manifest.json");
    let mut state = if state_path.exists() {
        let raw: SyncState = serde_json::from_str(&fs::read_to_string(&state_path)?)?;
        if raw.repo_id != cfg.repo_id || raw.run_prefix != prefix {
            bail!("sync state targets a different HF run");
        }
        raw
    } else {
        SyncState {
            schema: 1,
            repo_id: cfg.repo_id.clone(),
            run_prefix: prefix.clone(),
            files: HashMap::new(),
            updated_at: None,
        }
    };
    state.updated_at = Some(utc_now());
    atomic_json(&state_path, &state)?;

    loop {
        for source in discover(&cfg.checkpoint_dir)? {
            let name = source
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("file")
                .to_string();
            let remote = format!("{prefix}/{name}");
            let source_sig = match signature(&source) {
                Ok(s) => s.to_vec(),
                Err(e) => {
                    eprintln!("[hf-sync] skip {name}: {e}");
                    continue;
                }
            };
            if let Some(prev) = state.files.get(&remote) {
                if prev.source_signature == source_sig {
                    continue;
                }
            }
            let (stable, sha, size, sig) = match snapshot(&source, &snapshot_dir) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("[hf-sync] skipped unstable source: {e}");
                    continue;
                }
            };
            if let Some(prev) = state.files.get_mut(&remote) {
                if prev.sha256 == sha {
                    prev.source_signature = sig.to_vec();
                    state.updated_at = Some(utc_now());
                    atomic_json(&state_path, &state)?;
                    let _ = fs::remove_file(&stable);
                    continue;
                }
            }
            if !cfg.dry_run {
                let mut attempt = 0u32;
                loop {
                    match hf::upload_file(
                        &client,
                        &cfg.repo_id,
                        &stable,
                        &remote,
                        &format!("Sync {name} ({})", &sha[..12.min(sha.len())]),
                    )
                    .await
                    {
                        Ok(()) => break,
                        Err(e) if attempt < cfg.max_retries => {
                            attempt += 1;
                            let sleep = Duration::from_secs_f64((2f64.powi(attempt as i32)).min(60.0));
                            eprintln!("[hf-sync] upload retry {attempt} for {name}: {e}");
                            tokio::time::sleep(sleep).await;
                        }
                        Err(e) => {
                            eprintln!("[hf-sync] upload failed for {name}: {e}");
                            break;
                        }
                    }
                }
                state.files.insert(
                    remote.clone(),
                    FileState {
                        sha256: sha.clone(),
                        bytes: size,
                        source_signature: sig.to_vec(),
                        uploaded_at: utc_now(),
                    },
                );
                state.updated_at = Some(utc_now());
                atomic_json(&state_path, &state)?;
            }
            println!(
                "[hf-sync] {} {remote}",
                if cfg.dry_run {
                    "would upload"
                } else {
                    "uploaded"
                }
            );
            let _ = fs::remove_file(&stable);
        }
        if cfg.once {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_secs_f64(cfg.interval_secs)).await;
    }
}

pub async fn pull_policy(run_name: &str, dest_dir: &Path) -> Result<PathBuf> {
    let client = hf::client_with_optional_token()?;
    let (weights, state) = crate::util::hf_policy_paths(run_name);
    fs::create_dir_all(dest_dir)?;
    let weights_dest = dest_dir.join(Path::new(&weights).file_name().unwrap());
    match hf::download_to(&client, crate::paths::POLICY_REPO, &weights, &weights_dest).await {
        Ok(p) => {
            if let Some(state_remote) = state {
                let state_dest = dest_dir.join(Path::new(&state_remote).file_name().unwrap());
                let _ = hf::download_to(
                    &client,
                    crate::paths::POLICY_REPO,
                    &state_remote,
                    &state_dest,
                )
                .await;
            }
            Ok(p)
        }
        Err(e) => {
            eprintln!(
                "[ofhf] warning: {weights} unavailable ({e}); trying policy.pt fallback"
            );
            let fallback = format!("{run_name}/policy.pt");
            let dest_pt = dest_dir.join("policy.pt");
            hf::download_to(&client, crate::paths::POLICY_REPO, &fallback, &dest_pt).await
        }
    }
}

/// Download AE encoder safetensors (or full AE then leave for ofexport).
pub async fn pull_ae_encoders(ae_dir: &Path) -> Result<()> {
    let client = hf::client_with_optional_token()?;
    fs::create_dir_all(ae_dir)?;
    for name in [
        "ae_v32_nostatic_d8c32.encoder.safetensors",
        "ae_v32_nostatic_d16c32.encoder.safetensors",
        // Legacy V10 fine/coarse encoders (kept for older recipes).
        "ae_v31_d8c32.encoder.safetensors",
        "ae_v31_d16c32.encoder.safetensors",
    ] {
        let dest = ae_dir.join(name);
        if dest.exists() {
            println!("keep {}", dest.display());
            continue;
        }
        match hf::download_to(&client, crate::paths::AE_REPO, name, &dest).await {
            Ok(_) => println!("fetched {name} -> {}", dest.display()),
            Err(e) => {
                eprintln!(
                    "[ofhf] {name} not on HF ({e}); falling back to .pt + ofexport path is external"
                );
                // Also try downloading the .pt for the Python export fallback path.
                let pt_name = name.replace(".encoder.safetensors", ".pt");
                let pt_dest = ae_dir.join(&pt_name);
                let _ = hf::download_to(&client, crate::paths::AE_REPO, &pt_name, &pt_dest).await;
            }
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn _json_example() -> serde_json::Value {
    json!({})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovers_policy_update_milestones() {
        assert!(is_policy_update_milestone("policy_update120.safetensors"));
        assert!(is_policy_update_milestone("policy_update120.state.json"));
        assert!(!is_policy_update_milestone("policy_update120.note.json"));
        assert!(!is_policy_update_milestone("latest.safetensors"));
    }

    #[test]
    fn discovers_curriculum_milestones_and_notes() {
        assert!(is_curriculum_milestone(
            "curriculum_advance_u8400_s26_to_27.safetensors"
        ));
        assert!(is_curriculum_milestone(
            "curriculum_demote_u8410_s27_to_26.state.json"
        ));
        assert!(is_curriculum_milestone(
            "curriculum_advance_u8400_s26_to_27.note.json"
        ));
        assert!(!is_curriculum_milestone("curriculum_advance_u8400.safetensors"));
        assert!(!is_curriculum_milestone("policy_update120.safetensors"));
    }
}
