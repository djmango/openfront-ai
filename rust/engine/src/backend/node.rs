//! Node subprocess - full TS engine hash oracle.

use crate::replay::ReplayResult;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Repo with `openfront/node_modules` (set `OPENFRONT_REPO` when using a worktree).
fn engine_root(fallback: &Path) -> PathBuf {
    std::env::var("OPENFRONT_REPO")
        .map(PathBuf::from)
        .unwrap_or_else(|_| fallback.to_path_buf())
}

pub fn verify_record_file(repo_root: &Path, record_path: &Path) -> ReplayResult {
    let root = engine_root(repo_root);
    let tsx = root.join("openfront/node_modules/.bin/tsx");
    let script = root.join("scripts/hash_verify.ts");
    if !tsx.exists() || !script.exists() {
        return ReplayResult {
            ok: false,
            reason: Some(format!(
                "TS engine unavailable (need {} and {}; set OPENFRONT_REPO if needed)",
                tsx.display(),
                script.display()
            )),
            ticks: 0,
            hashes_checked: 0,
        };
    }

    let out = match Command::new(&tsx)
        .arg(&script)
        .arg(record_path)
        .current_dir(&root)
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            return ReplayResult {
                ok: false,
                reason: Some(format!("spawn tsx: {e}")),
                ticks: 0,
                hashes_checked: 0,
            };
        }
    };

    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        return ReplayResult {
            ok: false,
            reason: Some(format!("tsx failed: {err}")),
            ticks: 0,
            hashes_checked: 0,
        };
    }

    #[derive(serde::Deserialize)]
    struct Out {
        ok: bool,
        reason: Option<String>,
        ticks: u32,
        hashes_checked: u32,
    }

    match serde_json::from_slice::<Out>(&out.stdout) {
        Ok(r) => ReplayResult {
            ok: r.ok,
            reason: r.reason,
            ticks: r.ticks,
            hashes_checked: r.hashes_checked,
        },
        Err(_) => parse_json_stdout(&out.stdout),
    }
}

fn parse_json_stdout(stdout: &[u8]) -> ReplayResult {
    let text = String::from_utf8_lossy(stdout);
    for line in text.lines().rev() {
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        if let Ok(r) = serde_json::from_str::<serde_json::Value>(line) {
            return ReplayResult {
                ok: r.get("ok").and_then(|v| v.as_bool()).unwrap_or(false),
                reason: r
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string()),
                ticks: r.get("ticks").and_then(|v| v.as_u64()).unwrap_or(0) as u32,
                hashes_checked: r
                    .get("hashes_checked")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32,
            };
        }
    }
    ReplayResult {
        ok: false,
        reason: Some(format!("no json in tsx output: {text}")),
        ticks: 0,
        hashes_checked: 0,
    }
}
