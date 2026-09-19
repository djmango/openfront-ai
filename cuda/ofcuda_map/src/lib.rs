//! Host-side shared code for the `ofcuda_map` parity harness.
//!
//! Nothing here touches CUDA: the GPU binary and the CPU companion binary both
//! link this crate so that map loading, the hash definition and the printed
//! format are literally the same code. Only the *computation* differs (a
//! cuda-oxide kernel vs. a plain sequential Rust loop), which is what makes a
//! mismatch meaningful.

use std::fs;
use std::path::{Path, PathBuf};

/// FNV-1a 64 offset basis.
pub const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64 prime.
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// `IS_LAND` bit in the terrain byte (`rust/engine/src/map.rs:8`).
pub const IS_LAND_BIT: u8 = 0x80;

/// The `map` entry of `manifest.json` - the plane `GameMapSize::Normal` loads
/// (`rust/engine/src/core/terrain.rs:95-108`).
#[derive(Debug, Clone, Copy)]
pub struct MapMeta {
    pub width: u32,
    pub height: u32,
    pub num_land_tiles: u32,
}

impl MapMeta {
    pub fn tiles(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }
}

/// Which plane file to load. `Normal` is `map.bin`, `Compact`'s game plane is
/// `map4x.bin` and its mini plane is `map16x.bin` (`terrain.rs:95-108`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapSize {
    Normal,
    X4,
    X16,
}

impl MapSize {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "normal" | "1x" | "map" => Some(Self::Normal),
            "4x" | "compact" | "map4x" => Some(Self::X4),
            "16x" | "map16x" => Some(Self::X16),
            _ => None,
        }
    }

    /// Manifest key and file name, in the engine's own naming.
    pub fn manifest_key(&self) -> &'static str {
        match self {
            Self::Normal => "map",
            Self::X4 => "map4x",
            Self::X16 => "map16x",
        }
    }

    pub fn file_name(&self) -> &'static str {
        match self {
            Self::Normal => "map.bin",
            Self::X4 => "map4x.bin",
            Self::X16 => "map16x.bin",
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::X4 => "4x",
            Self::X16 => "16x",
        }
    }
}

impl std::str::FromStr for MapSize {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s).ok_or_else(|| format!("unknown map size {s:?} (normal|4x|16x)"))
    }
}

/// Load `manifest.json` + the plane file for `size` from a map directory.
/// Read-only: the files are only ever opened for reading.
pub fn load_map_plane(map_dir: &Path, size: MapSize) -> Result<(MapMeta, Vec<u8>), String> {
    let manifest_path = map_dir.join("manifest.json");
    let manifest_bytes = fs::read(&manifest_path).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&manifest_bytes).map_err(|e| format!("{}: {e}", manifest_path.display()))?;
    let map = manifest
        .get(size.manifest_key())
        .ok_or_else(|| format!("{}: no {:?} entry", manifest_path.display(), size.manifest_key()))?;
    let num = |key: &str| -> Result<u32, String> {
        map.get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
            .ok_or_else(|| format!("{}: {}.{key} missing", manifest_path.display(), size.manifest_key()))
    };
    let meta = MapMeta {
        width: num("width")?,
        height: num("height")?,
        num_land_tiles: num("num_land_tiles")?,
    };

    let bin_path = map_dir.join(size.file_name());
    let data = fs::read(&bin_path).map_err(|e| format!("{}: {e}", bin_path.display()))?;
    if data.len() != meta.tiles() {
        return Err(format!(
            "{}: {} bytes != {}x{}",
            bin_path.display(),
            data.len(),
            meta.width,
            meta.height
        ));
    }
    Ok((meta, data))
}

/// Every map directory under `<repo_root>/openfront/resources/maps`, sorted.
/// Directory names are the canonical map keys the engine lowercases to.
pub fn list_map_dirs(repo_root: &Path) -> Result<Vec<String>, String> {
    list_map_dirs_in(&repo_root.join("openfront/resources/maps"))
}

/// Every subdirectory of `root`, sorted - the map keys in a maps root.
pub fn list_map_dirs_in(root: &Path) -> Result<Vec<String>, String> {
    let mut names: Vec<String> = fs::read_dir(root)
        .map_err(|e| format!("{}: {e}", root.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    Ok(names)
}

/// Load `manifest.json` + `map.bin` from a map directory, exactly the pair
/// `GameMapSize::Normal` selects (`terrain.rs:96-113`, `map.rs:93-99`).
/// Read-only: the files are only ever opened for reading.
pub fn load_map_normal(map_dir: &Path) -> Result<(MapMeta, Vec<u8>), String> {
    load_map_plane(map_dir, MapSize::Normal)
}

/// Canonical on-disk location of the map the RL engine loads
/// (`repo_root/openfront/resources/maps/<key>/`, key lowercased - `terrain.rs:37-41`,
/// `rl.rs:66-68`).
pub fn map_dir(repo_root: &Path, map_key: &str) -> PathBuf {
    repo_root
        .join("openfront/resources/maps")
        .join(map_key.to_lowercase().replace(' ', ""))
}

/// Sequential FNV-1a 64 over a byte slice, starting from `h0`.
pub fn fnv1a_bytes(h0: u64, bytes: &[u8]) -> u64 {
    let mut h = h0;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(FNV_PRIME);
    }
    h
}

/// Sequential FNV-1a 64 over a `u16` plane, each element fed as two
/// little-endian bytes (low byte first).
pub fn fnv1a_u16_le(h0: u64, words: &[u16]) -> u64 {
    let mut h = h0;
    for &v in words {
        h = (h ^ (v & 0xff) as u64).wrapping_mul(FNV_PRIME);
        h = (h ^ (v >> 8) as u64).wrapping_mul(FNV_PRIME);
    }
    h
}

/// Terrain hash of a plane exactly as the reference defines it.
pub fn terrain_hash(terrain: &[u8]) -> u64 {
    fnv1a_bytes(FNV_OFFSET_BASIS, terrain)
}

/// State-plane hash of a plane exactly as the reference defines it.
pub fn state_hash(state: &[u16]) -> u64 {
    fnv1a_u16_le(FNV_OFFSET_BASIS, state)
}

/// Number of tiles with the `IS_LAND` bit set (`map.rs:128-130`).
pub fn land_tiles(terrain: &[u8]) -> u64 {
    terrain.iter().filter(|&&b| b & IS_LAND_BIT != 0).count() as u64
}

/// The three numbers the harness compares, plus the manifest's own land count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Summary {
    pub terrain_hash: u64,
    pub state_hash: u64,
    pub land_tiles: u64,
}

/// Deterministic, non-trivial state plane used as an extra cross-check. The
/// real initial state plane is all zeros, so hashing it alone cannot prove the
/// u16 -> 2xLE-byte path works (a buffer that was never read would hash the
/// same). This pattern does exercise it.
pub fn synthetic_state(tiles: usize) -> Vec<u16> {
    (0..tiles)
        .map(|t| {
            let owner = ((t % 977) as u16) & 0x0fff;
            let fallout = if t % 4093 == 0 { 0x2000u16 } else { 0 };
            owner | fallout
        })
        .collect()
}

/// One canonical line per number so GPU and CPU output can be diffed verbatim.
pub fn format_report(meta: &MapMeta, summary: &Summary, extra_state_hash: Option<u64>) -> String {
    let mut out = String::new();
    out.push_str(&format!("width {}\n", meta.width));
    out.push_str(&format!("height {}\n", meta.height));
    out.push_str(&format!("terrain_hash {:#018x}\n", summary.terrain_hash));
    out.push_str(&format!("state_hash {:#018x}\n", summary.state_hash));
    out.push_str(&format!("land_tiles {}\n", summary.land_tiles));
    out.push_str(&format!(
        "manifest_land_tiles {}\n",
        meta.num_land_tiles
    ));
    if let Some(h) = extra_state_hash {
        out.push_str(&format!("synthetic_state_hash {:#018x}\n", h));
    }
    out
}
