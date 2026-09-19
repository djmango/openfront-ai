//! `ofcuda_map_cpu` - CPU companion binary, the reference tool.
//!
//! Computes the same three numbers as the GPU binary from the same map file,
//! with a plain sequential FNV-1a 64 over the whole plane. It shares only the
//! map loading and the printed format with the GPU binary; the hashing itself
//! is independent, so a GPU mismatch shows up as a differing line instead of
//! being trusted.
use ofcuda_map::{Summary, format_report, load_map_normal, synthetic_state};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo_root = PathBuf::from("/opt/data/workspaces/skg/openfront-ai");
    let map_dir = match std::env::args().nth(1) {
        Some(p) => PathBuf::from(p),
        None => ofcuda_map::map_dir(&repo_root, "World"),
    };

    let (meta, terrain_bytes) = load_map_normal(&map_dir)?;
    let n_tiles = meta.tiles();
    let state_host: Vec<u16> = vec![0u16; n_tiles];
    let synth_host = synthetic_state(n_tiles);

    let summary = Summary {
        terrain_hash: ofcuda_map::terrain_hash(&terrain_bytes),
        state_hash: ofcuda_map::state_hash(&state_host),
        land_tiles: ofcuda_map::land_tiles(&terrain_bytes),
    };
    let synth = ofcuda_map::state_hash(&synth_host);

    print!("{}", format_report(&meta, &summary, Some(synth)));
    Ok(())
}
