//! Shared filesystem + env paths for showcase / HF tooling.

use std::env;
use std::path::{Path, PathBuf};

pub const POLICY_REPO: &str = "djmango/openfront-rl";
pub const AE_REPO: &str = "djmango/openfront-tile-autoencoder";

pub fn repo_root() -> PathBuf {
    if let Ok(root) = env::var("OFHUB_REPO_ROOT") {
        return PathBuf::from(root);
    }
    // ofhub lives at rust/ofhub; walk up to the monorepo root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap_or(Path::new("."))
        .to_path_buf()
}

pub fn data_dir() -> PathBuf {
    PathBuf::from(env::var("DATA_DIR").unwrap_or_else(|_| "/data".into()))
}

pub fn policy_dir() -> PathBuf {
    data_dir().join("policy")
}

pub fn clips_dir() -> PathBuf {
    data_dir().join("clips")
}

pub fn records_dir() -> PathBuf {
    data_dir().join("records")
}

pub fn revision_path() -> PathBuf {
    data_dir().join("policy_revision.txt")
}

pub fn state_path() -> PathBuf {
    data_dir().join("state.json")
}

pub fn hub_state_path() -> PathBuf {
    data_dir().join("hub_state.json")
}
