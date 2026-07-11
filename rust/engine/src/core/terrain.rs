//! Fresh terrain loading for replay bootstrap (`datagen/common.ts::loadFreshTerrain`).

use crate::map::{read_manifest, read_terrain_bin, GameMap, Nation, SpawnArea};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameMapSize {
    Normal,
    Compact,
}

impl GameMapSize {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "Normal" => Some(Self::Normal),
            "Compact" => Some(Self::Compact),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TerrainMapData {
    pub nations: Vec<Nation>,
    pub additional_nations: Vec<Nation>,
    pub game_map: GameMap,
    pub mini_game_map: GameMap,
    pub team_game_spawn_areas: Option<HashMap<String, Vec<SpawnArea>>>,
}

/// Map type string or directory key → `openfront/resources/maps/<key>/`.
pub fn normalize_map_key(map_type: &str) -> String {
    map_type.to_lowercase().replace(' ', "")
}

pub fn map_dir(repo_root: &Path, map_key: &str) -> PathBuf {
    repo_root
        .join("openfront/resources/maps")
        .join(normalize_map_key(map_key))
}

/// Resolve a map directory for archived-record replay, preferring a frozen
/// per-commit asset snapshot over the live `openfront/` submodule checkout
/// when one has been fetched (`rust/scripts/fetch_test_fixture_maps.sh`).
///
/// Map binaries (`map.bin`/`manifest.json`) are ordinary tracked assets in
/// upstream `openfront` and get rebalanced/redrawn over time just like any
/// other game content - they are **not** pinned to `PARITY_COMMIT` the way
/// TS source/behavior is expected to be for archived-record parity. Once the
/// live submodule advances past whatever commit an archived record's terrain
/// was generated against, replaying that record against the *current* map
/// silently diverges from the very first tick any tile geometry differs
/// (e.g. a coastline tweak shifting which spawn tiles are land) - not a
/// native-vs-TS logic bug, just a stale terrain snapshot. Falls back to the
/// live submodule checkout when no frozen snapshot exists (the common case
/// for everything that isn't an old archived-record fixture).
pub fn map_dir_for_commit(repo_root: &Path, map_key: &str, git_commit: Option<&str>) -> PathBuf {
    if let Some(commit) = git_commit {
        let short: String = commit.chars().take(12).collect();
        let frozen = repo_root
            .join("records/frozen-maps")
            .join(&short)
            .join(normalize_map_key(map_key));
        if frozen.join("manifest.json").exists() {
            return frozen;
        }
    }
    map_dir(repo_root, map_key)
}

pub fn load_fresh_terrain(
    repo_root: &Path,
    map_key: &str,
    map_size: GameMapSize,
) -> Result<TerrainMapData, String> {
    load_fresh_terrain_from_dir(&map_dir(repo_root, map_key), map_size)
}

pub fn load_fresh_terrain_for_commit(
    repo_root: &Path,
    map_key: &str,
    map_size: GameMapSize,
    git_commit: Option<&str>,
) -> Result<TerrainMapData, String> {
    load_fresh_terrain_from_dir(&map_dir_for_commit(repo_root, map_key, git_commit), map_size)
}

pub fn load_fresh_terrain_from_dir(
    map_dir: &Path,
    map_size: GameMapSize,
) -> Result<TerrainMapData, String> {
    let manifest = read_manifest(map_dir)?;

    let (game_meta, game_bin, mini_meta, mini_bin) = match map_size {
        GameMapSize::Normal => (
            &manifest.map,
            "map.bin",
            &manifest.map4x,
            "map4x.bin",
        ),
        GameMapSize::Compact => (
            &manifest.map4x,
            "map4x.bin",
            &manifest.map16x,
            "map16x.bin",
        ),
    };

    let game_map = GameMap::from_terrain_bytes(
        game_meta,
        &read_terrain_bin(map_dir, game_bin, game_meta)?,
    )?;
    let mini_game_map = GameMap::from_terrain_bytes(
        mini_meta,
        &read_terrain_bin(map_dir, mini_bin, mini_meta)?,
    )?;

    let mut nations = manifest.nations;
    let mut additional_nations = manifest.additional_nations;
    let mut team_game_spawn_areas = manifest.team_game_spawn_areas;

    if map_size == GameMapSize::Compact {
        scale_nations(&mut nations);
        scale_nations(&mut additional_nations);
        team_game_spawn_areas = team_game_spawn_areas.map(|areas| scale_spawn_areas(&areas));
    }

    Ok(TerrainMapData {
        nations,
        additional_nations,
        game_map,
        mini_game_map,
        team_game_spawn_areas,
    })
}

fn scale_nations(nations: &mut [Nation]) {
    for nation in nations.iter_mut() {
        if let Some([x, y]) = nation.coordinates {
            nation.coordinates = Some([x / 2, y / 2]);
        }
    }
}

fn scale_spawn_areas(areas: &HashMap<String, Vec<SpawnArea>>) -> HashMap<String, Vec<SpawnArea>> {
    areas
        .iter()
        .map(|(key, spawn_areas)| {
            let scaled = spawn_areas
                .iter()
                .map(|a| SpawnArea {
                    x: a.x / 2,
                    y: a.y / 2,
                    width: a.width,
                    height: a.height,
                })
                .collect();
            (key.clone(), scaled)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repo root")
    }

    fn skip_without_maps(root: &Path) -> bool {
        !map_dir(root, "World").join("manifest.json").exists()
    }

    #[test]
    fn loads_world_normal() {
        let root = repo_root();
        if skip_without_maps(&root) {
            return;
        }
        let terrain = load_fresh_terrain(&root, "World", GameMapSize::Normal)
            .expect("load world");
        assert_eq!(terrain.game_map.width, 2000);
        assert_eq!(terrain.game_map.height, 1000);
        assert_eq!(terrain.mini_game_map.width, 1000);
        assert_eq!(terrain.mini_game_map.height, 500);
        assert!(!terrain.nations.is_empty());
    }

    #[test]
    fn compact_scales_nation_coordinates() {
        let root = repo_root();
        if skip_without_maps(&root) {
            return;
        }
        let dir = map_dir(&root, "World");
        let normal = load_fresh_terrain_from_dir(&dir, GameMapSize::Normal).expect("normal");
        let compact = load_fresh_terrain_from_dir(&dir, GameMapSize::Compact).expect("compact");
        let n0 = normal.nations[0].coordinates.expect("coords");
        let c0 = compact.nations[0].coordinates.expect("coords");
        assert_eq!(c0[0], n0[0] / 2);
        assert_eq!(c0[1], n0[1] / 2);
        assert_eq!(compact.game_map.width, 1000);
        assert_eq!(compact.mini_game_map.width, 500);
    }
}
