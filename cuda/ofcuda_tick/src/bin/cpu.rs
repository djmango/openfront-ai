//! `ofcuda_tick_cpu` - the host reference, with no CUDA anywhere in its
//! dependency graph (`cargo build --bin ofcuda_tick_cpu` does not even build the
//! cuda crates). It runs the *same* `src/core_impl.rs` the kernel compiles.
//!
//! This is the L0/L1 lesson: a kernel checked only against itself is worthless,
//! so the claim order and priorities have to be produced by independent code
//! that can disagree.

use ofcuda_tick::{
    FrontierCase, MapPlane, ORDER_NSWE, ORDER_WENS, fmt_u32s, fmt_prios, load_map, pangaea_map_dir,
    priority_unit, recorded_cases, run_cpu,
};

fn dump_one(case: &FrontierCase, map: &MapPlane, order: u32, label: &str) {
    let mut c = case.clone();
    c.order = order;
    let (o, p, n) = run_cpu(&c, map);
    println!("  {label}: enqueued {n}; claim order {}", fmt_u32s(&o));
    println!("       priorities {}", fmt_prios(&p));
    if !c.expected_units.is_empty() {
        let units: Vec<u32> = o
            .iter()
            .zip(p.iter())
            .map(|(t, pr)| priority_unit(*t, *pr, &c, map))
            .collect();
        println!("       units {}", fmt_u32s(&units));
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let map = load_map(&pangaea_map_dir())?;
    let (c1, c2) = recorded_cases(&map)?;

    println!("# ofcuda_tick_cpu - host reference for the FRONTIER step");
    println!("# map {}x{}", map.width, map.height);

    let mut ok = true;
    for case in [&c1, &c2] {
        println!("\n=== {} ===", case.name);
        println!(
            "  player {} {}; claim_tick {}; priority_tick {}; cursor {}; border {}; \
             owned_mine {}; expected claims {}",
            case.player_id,
            case.player_name,
            case.claim_tick,
            case.priority_tick,
            case.cursor,
            case.border.len(),
            case.owned_mine.len(),
            case.expected_order.len()
        );
        dump_one(case, &map, ORDER_WENS, "W,E,N,S");
        dump_one(case, &map, ORDER_NSWE, "N,S,W,E");
        println!(
            "  expected (record)    {} priorities {}",
            fmt_u32s(&case.expected_order),
            fmt_prios(&case.expected_priorities)
        );
        println!(
            "  other engine (record) {}",
            fmt_u32s(&case.other_engine_order)
        );

        // The tick claims exactly as many tiles as the record shows
        // (`num_tiles_per_tick`); the rest of the queue is not observable in the
        // dump, so the comparison is on that prefix - and only that prefix.
        let n = case.expected_order.len();
        let (o_w, p_w, _) = run_cpu(&case.clone().with_order(ORDER_WENS), &map);
        let (o_n, p_n, _) = run_cpu(&case.clone().with_order(ORDER_NSWE), &map);
        let w_ok = o_w[..n] == case.expected_order[..] && p_w[..n] == case.expected_priorities[..];
        let n_ok = o_n[..n] == case.expected_order[..] && p_n[..n] == case.expected_priorities[..];
        println!("  W,E,N,S reproduces the record: {w_ok}");
        println!("  N,S,W,E reproduces the record: {n_ok}");
        if !w_ok && !n_ok {
            ok = false;
        }
    }
    println!("\nCPU_REFERENCE_OK {ok}");
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}

// Small helper so the CPU bin can pick an order per case without a builder.
trait WithOrder {
    fn with_order(&self, order: u32) -> FrontierCase;
}
impl WithOrder for FrontierCase {
    fn with_order(&self, order: u32) -> FrontierCase {
        let mut c = self.clone();
        c.order = order;
        c
    }
}