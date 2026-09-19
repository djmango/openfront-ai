//! CPU-only companion: the reference runnable with no CUDA in the dependency
//! graph at all (`cargo build --bin ofcuda_econ_cpu` skips the cuda crates), so
//! the numbers in the GPU run can be reproduced on a machine without a device.

use ofcuda_econ::{
    Case, engine_agreement, fma_sensitivity, recorded_case_paths, run_cpu, synthetic_case_path,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("# ofcuda_econ_cpu - CPU reference only (no CUDA in this binary)");
    for path in recorded_case_paths() {
        report(&Case::load(&path)?);
    }
    report(&Case::load(&synthetic_case_path())?);
    let (tested, differing, first) = fma_sensitivity();
    println!(
        "FMA sensitivity: {tested} points tested, {differing} differ between two-step and mul_add \
         ({})",
        first.map(|f| format!("first {f:?}")).unwrap_or_else(|| "none".into())
    );
    Ok(())
}

fn report(case: &Case) {
    let cpu = run_cpu(&case.rows);
    let (agree, bad) = engine_agreement(&case.rows, &cpu);
    println!(
        "=== {} ({} rows, source {}) ===",
        case.record,
        case.rows.len(),
        case.source
    );
    println!(
        "  engine agreement: {agree}/{} ({}%)",
        case.rows.len(),
        100.0 * agree as f64 / case.rows.len() as f64
    );
    for &i in bad.iter().take(10) {
        let r = &case.rows[i];
        println!(
            "    tick {} {} type={} pred_delta={} observed_delta={} residual={} attacks={}",
            r.tick,
            r.identity,
            ofcuda_econ::player_type_name(r.player_type),
            cpu[i].troops_after - r.troops,
            r.next_troops - r.troops,
            r.next_troops - cpu[i].troops_after,
            r.attack_activity
        );
    }
}
