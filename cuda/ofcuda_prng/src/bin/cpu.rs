//! CPU-only companion for `ofcuda_prng`: runs the exact same port (from the
//! shared crate, so literally the same code the GPU binary's comparison uses)
//! without touching CUDA. This separates "the model of the engine is wrong"
//! from "the kernel is wrong" when a comparison fails.
//!
//! Usage:
//!   ofcuda_prng_cpu <reference.txt> [map_dir]
use ofcuda_prng::{
    Reference, compare, cpu_port_output, format_report, load_map_normal, map_dir, parse_reference,
};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let ref_path = args.next().unwrap_or_else(|| "reference.txt".to_string());
    let repo_root = PathBuf::from("/opt/data/workspaces/skg/openfront-ai");
    let map_path = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| map_dir(&repo_root, "Pangaea"));

    let text = std::fs::read_to_string(&ref_path)?;
    let r: Reference = parse_reference(&text)?;
    let map = load_map_normal(&map_path)?;
    println!(
        "# cpu reference: map {} {}x{}  terrain_bytes {}",
        map_path.display(),
        map.width,
        map.height,
        map.terrain.len()
    );

    let out = cpu_port_output(&r, &map);
    let rep = compare(&r, &out);
    print!("{}", format_report(&r, &rep, "CPU port"));
    if !rep.all_match() {
        std::process::exit(1);
    }
    Ok(())
}
