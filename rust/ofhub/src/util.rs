//! Showcase util helpers ported from `rl/showcase_util.py`.

use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};
use rand::seq::SliceRandom;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::paths::clips_dir;

pub const LEGACY_POLICY_RUNS: &[&str] = &["ppo_v5", "ppo_v7"];

pub fn showcase_maps() -> Vec<String> {
    let mut maps = if let Ok(raw) = std::env::var("SHOWCASE_MAPS") {
        let maps: Vec<String> = raw
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect();
        if !maps.is_empty() {
            maps
        } else {
            ofcore::curriculum::ALL_MAPS
                .iter()
                .map(|m| (*m).to_string())
                .collect()
        }
    } else {
        ofcore::curriculum::ALL_MAPS
            .iter()
            .map(|m| (*m).to_string())
            .collect()
    };
    // Shuffle so generation / featured bias isn't fixed to list order.
    maps.shuffle(&mut rand::thread_rng());
    maps
}

pub fn map_seed(map_name: &str) -> String {
    map_name.to_ascii_lowercase().replace(' ', "_")
}

/// Convert a training/showcase map key (`BlackSea`) to the OpenFront
/// `GameMapType` value accepted by adminbot (`"Black Sea"`).
///
/// Training and `oftrain --map` use enum keys; `/api/adminbot/create_game`
/// validates against the string enum values from `Maps.gen.ts`.
pub fn game_map_api_name(map_key: &str) -> String {
    let key = map_key.trim();
    if key.is_empty() || key.contains(' ') {
        return key.to_string();
    }
    // Explicit overrides where camelCase-splitting is wrong.
    match key {
        "ArchipelagoSea" | "MilkyWay" | "SoutheastAsia" => return key.to_string(),
        "GatewayToTheAtlantic" => return "Gateway to the Atlantic".into(),
        "GulfOfStLawrence" => return "Gulf of St. Lawrence".into(),
        "StraitOfGibraltar" => return "Strait of Gibraltar".into(),
        "StraitOfHormuz" => return "Strait of Hormuz".into(),
        "Tourney1" => return "Tourney 2 Teams".into(),
        "Tourney2" => return "Tourney 3 Teams".into(),
        "Tourney3" => return "Tourney 4 Teams".into(),
        "Tourney4" => return "Tourney 8 Teams".into(),
        _ => {}
    }
    let mut out = String::with_capacity(key.len() + 4);
    for (i, ch) in key.chars().enumerate() {
        if i > 0 && ch.is_ascii_uppercase() {
            out.push(' ');
        }
        out.push(ch);
    }
    out
}

/// Pick a random replay entry from `state["maps"]`.
///
/// Falls back to a legacy top-level `game_id` entry when `maps` is empty.
pub fn featured_showcase_entry(state: &Value) -> Option<Value> {
    if let Some(entries) = state.get("maps").and_then(|v| v.as_array()) {
        if !entries.is_empty() {
            return entries.choose(&mut rand::thread_rng()).cloned();
        }
    }
    if state.get("game_id").is_some() {
        return Some(state.clone());
    }
    None
}

pub fn featured_game_id(state: &Value) -> Option<String> {
    featured_showcase_entry(state)?
        .get("game_id")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

pub fn hf_policy_paths(run_name: &str) -> (String, Option<String>) {
    if LEGACY_POLICY_RUNS.contains(&run_name) {
        return (format!("{run_name}/policy.pt"), None);
    }
    (
        format!("{run_name}/latest.safetensors"),
        Some(format!("{run_name}/latest.state.json")),
    )
}

pub fn policy_meta(policy: &Path) -> Result<Value> {
    let sha = {
        let bytes = fs::read(policy).with_context(|| format!("read {}", policy.display()))?;
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        format!("{:x}", hasher.finalize())[..16].to_string()
    };
    let state = if policy.extension().and_then(|e| e.to_str()) == Some("safetensors") {
        let state_path = policy.with_file_name(format!(
            "{}.state.json",
            policy
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("latest")
        ));
        if !state_path.is_file() {
            bail!("missing safetensors state metadata: {}", state_path.display());
        }
        serde_json::from_str(&fs::read_to_string(&state_path)?)?
    } else if policy.extension().and_then(|e| e.to_str()) == Some("pt") {
        let parent = policy
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("");
        if !LEGACY_POLICY_RUNS.contains(&parent) {
            bail!(
                "policy must be a current .safetensors checkpoint or an explicitly \
                 legacy ppo_v5/ppo_v7 policy.pt"
            );
        }
        // Legacy .pt meta is not loaded in Rust (no pickle). Emit sha only.
        json!({})
    } else {
        bail!("unsupported policy artifact: {}", policy.display());
    };
    Ok(json!({
        "policy_update": state.get("update"),
        "policy_stage": state.get("stage"),
        "policy_sha256": sha,
    }))
}

pub fn write_json(path: &Path, state: &Value) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(state)? + "\n";
    fs::write(path, text)?;
    Ok(())
}

pub fn load_json(path: &Path) -> Result<Value> {
    if !path.exists() {
        return Ok(json!({}));
    }
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text).unwrap_or_else(|_| json!({})))
}

pub fn utc_now() -> String {
    chrono::Utc::now().to_rfc3339()
}

pub fn hero_clip_urls(state: &Value) -> Vec<String> {
    let mut urls = Vec::new();
    if let Some(entries) = state.get("hero_clips").and_then(|v| v.as_array()) {
        for entry in entries {
            if let Some(s) = entry.as_str() {
                urls.push(if s.starts_with('/') {
                    s.to_string()
                } else {
                    format!("/archive/clips/{s}")
                });
            } else if let Some(u) = entry.get("url").and_then(|v| v.as_str()) {
                urls.push(u.to_string());
            }
        }
    }
    if !urls.is_empty() {
        return urls;
    }
    let dir = clips_dir();
    if dir.is_dir() {
        let mut names: Vec<_> = fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let p = e.path();
                (p.extension().and_then(|x| x.to_str()) == Some("webm"))
                    .then(|| p.file_name()?.to_str().map(str::to_string))
                    .flatten()
            })
            .collect();
        names.sort();
        for name in names {
            urls.push(format!("/archive/clips/{name}"));
        }
    }
    urls
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashSet;

    #[test]
    fn featured_picks_random_map_entry() {
        let state = json!({
            "maps": [
                {"game_id": "aaaaaaaa", "seed": "onion"},
                {"game_id": "bbbbbbbb", "seed": "pangaea"},
                {"game_id": "cccccccc", "seed": "asia"},
            ]
        });
        let mut seen = HashSet::new();
        for _ in 0..80 {
            let entry = featured_showcase_entry(&state).unwrap();
            seen.insert(entry["game_id"].as_str().unwrap().to_string());
        }
        assert_eq!(seen.len(), 3);
    }

    #[test]
    fn featured_falls_back_to_legacy_game_id() {
        let state = json!({"game_id": "deadbeef"});
        let entry = featured_showcase_entry(&state).unwrap();
        assert_eq!(entry["game_id"], "deadbeef");
    }

    #[test]
    fn game_map_api_name_resolves_showcase_keys() {
        assert_eq!(game_map_api_name("Onion"), "Onion");
        assert_eq!(game_map_api_name("BlackSea"), "Black Sea");
        assert_eq!(game_map_api_name("BetweenTwoSeas"), "Between Two Seas");
        assert_eq!(game_map_api_name("Black Sea"), "Black Sea");
    }
}
