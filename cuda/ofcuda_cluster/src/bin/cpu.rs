// ===========================================================================
// `ofcuda_cluster_cpu` - the host reference, with no CUDA in the crate's own
// dependency graph. It runs the *same* `src/core_impl.rs` the kernel compiles,
// through the same `prepare()` / `decide_cpu()` the GPU binary's CPU arm uses,
// and checks the result against the engine's recorded outcome.
//
// A kernel checked only against itself is worthless: this binary is the
// independent side that can disagree with the device.
// ===========================================================================

use std::path::PathBuf;

use ofcuda_cluster::{
    ReportInput, compare_to_engine, decide_cpu, device_core_verbatim_check, diagnose,
    format_report, load_case, load_terrain, pangaea_map_dir, prepare,
};

const DEFAULT_CASE: &str = "cases/cluster-b030-victim33-t1614.json";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if let Err(e) = device_core_verbatim_check() {
        eprintln!("FATAL: device core copy check failed:\n{e}");
        std::process::exit(2);
    }

    let case_path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_CASE));
    let root = std::env::var("OFCUDA_REPO")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/opt/data/workspaces/skg"));
    let case = load_case(&case_path)?;
    let map_dir = pangaea_map_dir(&root);
    let (terrain, terrain_hash) = load_terrain(&map_dir)?;

    let prep = prepare(&case, case.victim)?;
    let diags = diagnose(&case, &terrain, &prep);
    let cpu = decide_cpu(&case, &terrain, &prep);
    let engine_checks = compare_to_engine(&case, &cpu);

    let (report, ok) = format_report(&ReportInput {
        case: &case,
        device: "",
        terrain_path: &map_dir.display().to_string(),
        terrain_hash,
        prep: &prep,
        diags: &diags,
        cpu: &cpu,
        gpu: None,
        gpu_checks: &[],
        engine_checks: &engine_checks,
        engine_checks_gpu: &[],
    });
    print!("{report}");
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
