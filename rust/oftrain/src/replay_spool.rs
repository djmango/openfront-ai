//! Local GameRecord spool for batch parquet upload to Hugging Face.

use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::engine::GameEngine;

static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// `$DATA_DIR/replay-spool` (or `./replay-spool` when DATA_DIR unset).
/// Train pods should export `DATA_DIR` (see `pod_train_v10.sh`) so oftrain and
/// `ofhf replays` share the same spool.
pub fn spool_dir() -> PathBuf {
    let base = env::var("DATA_DIR").unwrap_or_else(|_| ".".into());
    PathBuf::from(base).join("replay-spool")
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn unique_tmp_path(dir: &Path) -> PathBuf {
    // Many env workers can finish in the same millisecond — include pid+seq
    // so parallel spool_episode calls never clobber each other's temps.
    dir.join(format!(
        "._tmp_{}_{}_{}.json",
        now_ms(),
        std::process::id(),
        TMP_SEQ.fetch_add(1, Ordering::Relaxed)
    ))
}

fn write_sidecar(dest: &Path, meta: &Value, game_id: &str) -> Result<()> {
    let mut sidecar = meta.clone();
    if let Some(obj) = sidecar.as_object_mut() {
        obj.insert("game_id".into(), json!(game_id));
        obj.insert("spooled_at".into(), json!(now_ms()));
        obj.insert(
            "record_path".into(),
            json!(dest.file_name().and_then(|n| n.to_str()).unwrap_or("")),
        );
    }
    let meta_path = dest.with_extension("meta.json");
    fs::write(&meta_path, serde_json::to_string_pretty(&sidecar)?)?;
    Ok(())
}

/// Persist a GameRecord into the spool with a sidecar meta JSON for parquet columns.
/// Failures are logged and ignored so training never blocks on spool I/O.
pub fn spool_episode(
    engine: &mut dyn GameEngine,
    meta: &Value,
) -> Result<Option<PathBuf>> {
    let dir = spool_dir();
    fs::create_dir_all(&dir)?;
    // Temporary path; rename after we know gameID from save_record.
    let tmp = unique_tmp_path(&dir);
    let info = match engine.save_record(tmp.to_str().unwrap_or("/tmp/of_replay.json")) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[replay-spool] save_record skipped: {e}");
            let _ = fs::remove_file(&tmp);
            return Ok(None);
        }
    };
    let game_id = info
        .get("gameID")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let dest = dir.join(format!("{game_id}.json"));
    if dest.exists() {
        let _ = fs::remove_file(&dest);
    }
    fs::rename(&tmp, &dest)
        .or_else(|_| {
            fs::copy(&tmp, &dest)?;
            fs::remove_file(&tmp)
        })
        .with_context(|| format!("spool rename {} -> {}", tmp.display(), dest.display()))?;
    write_sidecar(&dest, meta, game_id)?;
    Ok(Some(dest))
}

/// Copy an already-written GameRecord into the spool (e.g. oftrain --watch output).
pub fn spool_existing_record(record_path: &Path, meta: &Value) -> Result<Option<PathBuf>> {
    if !record_path.is_file() {
        return Ok(None);
    }
    let dir = spool_dir();
    fs::create_dir_all(&dir)?;
    let text = fs::read_to_string(record_path)?;
    let record: Value = serde_json::from_str(&text)?;
    let game_id = record
        .pointer("/info/gameID")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| {
            record_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string()
        });
    let dest = dir.join(format!("{game_id}.json"));
    fs::write(&dest, text)?;
    write_sidecar(&dest, meta, &game_id)?;
    Ok(Some(dest))
}
