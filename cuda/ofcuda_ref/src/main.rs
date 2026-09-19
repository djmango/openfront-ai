//! Independent reference numbers for the CUDA map/terrain parity sweep.
//!
//! This binary deliberately shares **no code** with `ofcuda_map`: it links the
//! real reference engine crate (`openfront-engine`) and asks it to load the map
//! (`map::read_manifest` + `map::read_terrain_bin` + `GameMap::from_terrain_bytes`,
//! the same pair `core::terrain::load_fresh_terrain_from_dir` selects) and to
//! interpret the land bit (`GameMap::is_land`).  Only the two hash/reduction
//! primitives are re-implemented here, straight from the FNV-1a 64 spec, so a
//! disagreement between this binary and the CUDA harness is a real disagreement
//! about loading/dimensions/land interpretation, not a shared-bug echo.
//!
//! Output is one TSV line per (map, size), columns exactly:
//!
//! ```text
//! name  size  WxH  land  manifest_land  terrain_hash  status
//! ```
//!
//! `status` is `OK` for a loadable map (hash is the FNV-1a 64 of the terrain
//! plane in row-major order), or `ERR:<reason>` when the engine refuses the
//! map.  The CUDA harness prints the same columns with its own values so
//! `diff` of the two outputs is the whole comparison.

use openfront_engine::map::{read_manifest, read_terrain_bin};
use std::path::{Path, PathBuf};

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// The three plane files the engine knows about, in the order `terrain.rs`
/// selects them for `GameMapSize::Normal` / `Compact`.
const SIZES: [(&str, &str); 3] = [("normal", "map.bin"), ("4x", "map4x.bin"), ("16x", "map16x.bin")];

fn meta_key(size: &str) -> &'static str {
    match size {
        "normal" => "map",
        "4x" => "map4x",
        _ => "map16x",
    }
}

fn fnv1a_bytes(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET_BASIS;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(FNV_PRIME);
    }
    h
}

/// Land count taken through the engine's own `is_land` accessor (bit 7), i.e.
/// one call per tile over the engine's terrain plane.
fn land_via_engine(map: &openfront_engine::map::GameMap) -> u64 {
    let n = (map.width as u64) * (map.height as u64);
    let mut land = 0u64;
    for t in 0..n {
        if map.is_land(t as u32) {
            land += 1;
        }
    }
    land
}

/// Print the reference row for one (dir, size).  Never panics: an unloadable
/// map prints `ERR:<reason>` so the sweep still covers all 100 directories.
fn emit(name: &str, size: &str, dir: &Path) {
    let manifest = match read_manifest(dir) {
        Ok(m) => m,
        Err(e) => {
            println!("{name}\t{size}\tERROR\t0\t0\t0\tERR:{e}");
            return;
        }
    };
    // The engine's own struct field, not a hand-rolled JSON walk.
    let meta = match size {
        "normal" => &manifest.map,
        "4x" => &manifest.map4x,
        _ => &manifest.map16x,
    };
    let file = SIZES.iter().find(|(s, _)| *s == size).unwrap().1;
    let data = match read_terrain_bin(dir, file, meta) {
        Ok(d) => d,
        Err(e) => {
            println!(
                "{name}\t{size}\t{}x{}\t0\t{}\t0\tERR:{e}",
                meta.width, meta.height, meta.num_land_tiles
            );
            return;
        }
    };
    let map = match openfront_engine::map::GameMap::from_terrain_bytes(meta, &data) {
        Ok(m) => m,
        Err(e) => {
            println!("{name}\t{size}\t{}x{}\t0\t{}\t0\tERR:{e}", meta.width, meta.height, meta.num_land_tiles);
            return;
        }
    };
    let hash = fnv1a_bytes(map.terrain_bytes());
    let land = land_via_engine(&map);
    println!(
        "{name}\t{size}\t{}x{}\t{}\t{}\t{:#018x}\tOK",
        map.width, map.height, land, meta.num_land_tiles, hash
    );
    let _ = meta_key(size);
}

/// Engine tick-loadability: the exact call `rl::RlSession::reset` makes
/// (`core::terrain::load_fresh_terrain_from_dir(dir, GameMapSize::Normal)`),
/// which needs the *whole* manifest (nations, map4x as the mini plane, ...)
/// to deserialize and both planes to have the byte length their meta claims.
fn emit_loadable(name: &str, dir: &Path) {
    for (label, size) in [
        ("Normal", openfront_engine::core::terrain::GameMapSize::Normal),
        ("Compact", openfront_engine::core::terrain::GameMapSize::Compact),
    ] {
        match openfront_engine::core::terrain::load_fresh_terrain_from_dir(dir, size) {
            Ok(t) => println!(
                "{name}\t{label}\t{}x{}\t{}x{}\t{}\t{}\t{}",
                t.game_map.width,
                t.game_map.height,
                t.mini_game_map.width,
                t.mini_game_map.height,
                t.nations.len(),
                t.additional_nations.len(),
                t.team_game_spawn_areas.as_ref().map(|m| m.len()).unwrap_or(0)
            ),
            Err(e) => println!("{name}\t{label}\tERROR\t0x0\t0\t0\t0\tERR:{e}"),
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let mut maps_root: Option<PathBuf> = None;
    let mut single: Option<String> = None;
    let mut sizes: Vec<String> = Vec::new();
    let mut loadable = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--maps-root" => maps_root = args.next().map(PathBuf::from),
            "--map" => single = args.next(),
            "--size" => sizes.push(args.next().unwrap_or_else(|| "normal".into())),
            "--loadable" => loadable = true,
            other => return Err(format!("unknown arg {other}").into()),
        }
    }
    let root = maps_root.unwrap_or_else(|| {
        PathBuf::from("/opt/data/workspaces/skg/openfront-ai/openfront/resources/maps")
    });
    if sizes.is_empty() {
        sizes = vec!["normal".into(), "4x".into(), "16x".into()];
    }

    if let Some(name) = single {
        let dir = root.join(name.to_lowercase().replace(' ', ""));
        if loadable {
            emit_loadable(&name.to_lowercase(), &dir);
            return Ok(());
        }
        for s in &sizes {
            emit(&name.to_lowercase(), s, &dir);
        }
        return Ok(());
    }

    let mut dirs: Vec<String> = std::fs::read_dir(&root)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    dirs.sort();
    for name in dirs {
        if loadable {
            emit_loadable(&name, &root.join(&name));
            continue;
        }
        for s in &sizes {
            emit(&name, s, &root.join(&name));
        }
    }
    Ok(())
}
