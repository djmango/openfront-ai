//! `ofcuda_econ` - the per-tick ECONOMY / STATE-PLANE update on the GPU.
//!
//! Slice "econ": `PlayerExecution.tick`'s income block plus the arithmetic it
//! depends on (`maxTroops`, `troopIncreaseRate`, `goldAdditionRate`,
//! `addTroops`). One thread per recorded (player, tick-transition) row; the
//! kernel writes back the resulting `troops`, `gold` and the *raw*
//! pre-truncation income float so the device's bits can be compared against
//! the host's, not just the truncated integer.
//!
//! The device core below is a **verbatim copy** of the host reference
//! (`src/core_impl.rs`), because cuda-oxide's `#[cuda_module]` macro sees the
//! module before an `include!` inside it expands. `lib.rs` carries a test that
//! fails if the two copies ever diverge, so GPU/CPU agreement stays a check on
//! the CUDA lowering rather than on a second implementation.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use ofcuda_econ::{
    Case, FieldDiff, Row, StepOut, count_exact, diff_steps, engine_agreement, fma_sensitivity,
    recorded_case_paths, run_cpu, synthetic_case_path,
};

const THREADS: u32 = 256;

#[cuda_module]
mod kernels {
    use super::*;

    // (cuda-oxide's `#[cuda_module]` attribute macro does not survive an
    // `include!` of this file - the macro sees the module before the include
    // expands - so the device core is an exact copy. `lib.rs` has a test that
    // fails if these two copies ever differ; `sync_device_core.sh` re-copies.)
    // ===== BEGIN VERBATIM COPY of src/core_impl.rs =====
// Shared core of the per-tick ECONOMY / STATE-PLANE update (slice "econ").
//
// This file is the canonical source. It is included by the host reference
// (`src/lib.rs`, `mod core_impl`) and copied VERBATIM into cuda-oxide's device
// module (`src/main.rs`, `mod kernels` - see the BEGIN/END VERBATIM COPY
// markers there, and the `device_core_is_a_verbatim_copy` test in lib.rs that
// fails the moment the two copies drift apart). So the income arithmetic has
// exactly one implementation that can disagree with the engine - not one per
// side.
//
// 1:1 ports, with citations (TS is the authority; the Rust engine is the
// cross-check that already reached every-tick parity with TS):
//   * `PlayerExecution.tick` income block   - openfront/src/core/execution/PlayerExecution.ts:75-81
//                                             (Rust live path: rust/engine/src/execution/player.rs:82-88)
//   * `Config.maxTroops`                    - openfront/src/core/configuration/Config.ts:791-823
//                                             (rust/engine/src/core/config.rs:363-390)
//   * `Config.troopIncreaseRate`            - openfront/src/core/configuration/Config.ts:825-857
//                                             (rust/engine/src/core/config.rs:433-458)
//   * `Config.goldAdditionRate`             - openfront/src/core/configuration/Config.ts:859-868
//                                             (rust/engine/src/core/config.rs:477-481)
//   * `PlayerImpl.addTroops` / `removeTroops` - openfront/src/core/game/PlayerImpl.ts:1177-1189
//                                             (rust/engine/src/game.rs:1130-1155)
//   * `Util.toInt`                          - openfront/src/core/Util.ts (floor)
//                                             (rust/engine/src/util.rs:26-35)
//
// `tick_player_income` (rust/engine/src/game.rs:1269) is DEAD CODE: nothing
// calls it. The live per-tick income is `execution/player.rs::PlayerExecution::tick`
// (registered per player by the execution manager), which calls
// `Game::troop_increase_rate_raw_for` + `Config::gold_addition_rate` exactly as
// the TS `PlayerExecution.tick` does. `try_merge_land_attack` is likewise not on
// the income path. See README.md "Dead code" for the grep that establishes it.
//
// (This file is `#[no_std]`-clean on purpose: no allocation, no float methods
// beyond `powf`/`floor`/`min`.)

/// `PlayerType` discriminants, as they cross the host/device boundary.
pub const PT_HUMAN: u32 = 0;
pub const PT_BOT: u32 = 1;
pub const PT_NATION: u32 = 2;

/// `Difficulty` discriminants (`Config._gameConfig.difficulty`).
pub const DIFF_EASY: u32 = 0;
pub const DIFF_MEDIUM: u32 = 1;
pub const DIFF_HARD: u32 = 2;
pub const DIFF_IMPOSSIBLE: u32 = 3;

/// `Config.cityTroopIncrease()` (`Config.ts:392-394`; `config.rs:392-394`).
pub const CITY_TROOP_INCREASE: f64 = 250_000.0;

/// `Util.toInt` (`util.rs:26-35`): `floor`, saturating to the i32 range at the
/// infinities (Rust's `as i32` already saturates, so only the explicit inf
/// handling is needed for parity with the crate's Python probe).
#[inline]
pub fn to_int(num: f64) -> i32 {
    if num.is_infinite() {
        return if num.is_sign_positive() { i32::MAX } else { i32::MIN };
    }
    num.floor() as i32
}

/// `a * b + c`, with the multiply and the add kept as two separately rounded
/// IEEE operations. `black_box` is an identity to the language but opaque to the
/// optimiser, which is what stops `a*b+c` from being contracted into a single
/// `fma.rn.f64` on the device. Rust on x86-64 emits the two-step form anyway
/// (measured: this core reproduces the engine's own dump), so on the host this
/// only documents the intent.
#[inline]
pub fn survive_opt(p: f64) -> f64 {
    // A volatile read of the local is an identity that the device MIR lowering
    // cannot see through (it survives as a real local load/store), which is what
    // keeps the caller's `a*b + c` from being rewritten as one `fma.rn.f64`.
    unsafe { core::ptr::read_volatile(&p) }
}

#[inline]
pub fn mul_then_add(a: f64, b: f64, c: f64) -> f64 {
    survive_opt(a * b) + c
}

/// `Config.maxTroops(player)` (`Config.ts:791-823`; `config.rs:363-390`).
///
/// Order of operations is the engine's: `2*(tiles^0.6*1000 + 50000)` first, the
/// city term added, and only then the per-type scale.
/// `tiles ** 0.6`, kept as its own function so the parity harness can compare
/// the device's libdevice `pow` against the host's `powf` bit-for-bit.
#[inline]
pub fn pow_tiles(tiles_owned: f64) -> f64 {
    tiles_owned.powf(0.6)
}

/// `troops ** 0.73`, same purpose as [`pow_tiles`].
#[inline]
pub fn pow_troops(troops: f64) -> f64 {
    troops.powf(0.73)
}

#[inline]
pub fn max_troops(
    player_type: u32,
    tiles_owned: i32,
    city_level_sum: i64,
    difficulty: u32,
    infinite_troops: bool,
) -> f64 {
    // The engine's own order, with the two contractible multiplies forced to
    // stay separate. `mul_then_add` is an identity, but `black_box` is an
    // optimisation barrier: without it the PVT/ltoir lowering fuses
    // `pow*1000 + 50000` and `city*250000 + t` into `fma.rn.f64` (observed in
    // the emitted PTX), which changes `max_troops` on ~10% of the recorded
    // transitions while the host stays two-step. See README "FMA" for the
    // measurement.
    let mut max = 2.0 * mul_then_add(pow_tiles(tiles_owned as f64), 1000.0, 50_000.0)
        + survive_opt(city_level_sum as f64 * CITY_TROOP_INCREASE);
    if player_type == PT_BOT {
        max /= 3.0;
    } else if player_type == PT_HUMAN {
        if infinite_troops {
            return 1_000_000_000.0;
        }
    } else {
        max *= match difficulty {
            DIFF_EASY => 0.5,
            DIFF_MEDIUM => 0.75,
            DIFF_HARD => 1.0,
            DIFF_IMPOSSIBLE => 1.25,
            _ => 0.75,
        };
    }
    max
}

/// `Config.troopIncreaseRate(player)` un-rounded (`Config.ts:825-857`;
/// `config.rs:433-458`). Returns `min(troops + toAdd, max) - troops`, i.e. the
/// raw (possibly fractional, possibly negative) per-tick troop delta.
#[inline]
pub fn troop_increase_rate_raw(
    player_type: u32,
    troops: i32,
    tiles_owned: i32,
    city_level_sum: i64,
    difficulty: u32,
    infinite_troops: bool,
) -> f64 {
    let max = max_troops(player_type, tiles_owned, city_level_sum, difficulty, infinite_troops);
    let mut to_add = 10.0 + pow_troops(troops as f64) / 4.0;
    let ratio = 1.0 - troops as f64 / max;
    to_add *= ratio;
    if player_type == PT_BOT {
        to_add *= 0.5;
    }
    if player_type == PT_NATION {
        to_add *= match difficulty {
            DIFF_EASY => 0.9,
            DIFF_MEDIUM => 0.95,
            DIFF_HARD => 1.0,
            DIFF_IMPOSSIBLE => 1.05,
            _ => 0.95,
        };
    }
    (troops as f64 + to_add).min(max) - troops as f64
}

/// `PlayerImpl.addTroops(amount)` (`PlayerImpl.ts:1177-1182`;
/// `game.rs:1147-1155` + `remove_troops` at `game.rs:1130-1141`).
///
/// A non-negative amount adds `floor(amount)`. A negative amount is routed
/// through `removeTroops(-amount)`, which removes `min(troops, floor(-amount))`
/// - so the pull-back is the floor of the *magnitude*, not the floor of the
/// signed value. That asymmetry is load-bearing: it is the fix for the UK
/// soft-troop fork at tick ~1793 on `curr-b030-s0-pangaea`
/// (`execution/player.rs:130-146`).
#[inline]
pub fn add_troops(troops: i32, amount: f64) -> i32 {
    if amount < 0.0 {
        let to_remove = troops.min(to_int(-amount));
        troops - to_remove.max(0)
    } else {
        troops + to_int(amount)
    }
}

/// `Config.goldAdditionRate(player)` (`Config.ts:859-868`; `config.rs:477-481`).
/// `floor(baseRate * multiplier)`, base 50 for a Bot and 100 otherwise.
#[inline]
pub fn gold_addition_rate(player_type: u32, gold_multiplier: f64) -> i64 {
    let base_rate: f64 = if player_type == PT_BOT { 50.0 } else { 100.0 };
    (base_rate * gold_multiplier).floor() as i64
}

/// The whole per-tick economy step of one player, in one call.
///
/// Returns `(troops_after, gold_after, raw_income)` where `raw_income` is the
/// *pre-truncation* `troopIncreaseRate` float - the same value the RL obs emits
/// (`config.rs:431-432`) - so the GPU's raw float can be compared bit-for-bit
/// against the host's, not just the truncated integer.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn econ_step(
    player_type: u32,
    difficulty: u32,
    troops: i32,
    tiles_owned: i32,
    city_level_sum: i64,
    gold: i64,
    gold_multiplier: f64,
    infinite_troops: bool,
) -> (i32, i64, f64) {
    let raw = troop_increase_rate_raw(
        player_type,
        troops,
        tiles_owned,
        city_level_sum,
        difficulty,
        infinite_troops,
    );
    let troops_after = add_troops(troops, raw);
    let gold_after = gold + gold_addition_rate(player_type, gold_multiplier);
    (troops_after, gold_after, raw)
}
    // ===== END VERBATIM COPY of src/core_impl.rs =====

    /// One thread per (player, tick-transition). All input planes are parallel
    /// arrays; `gold_mult` is bit-compared by the host, not just its value, so
    /// a device-side contraction that only perturbs the last bit shows up.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    #[allow(clippy::too_many_arguments)]
    pub fn econ_step_kernel(
        pt: &[u32],
        difficulty: &[u32],
        troops: &[i32],
        tiles: &[i32],
        city_levels: &[i64],
        gold: &[i64],
        gold_mult: &[f64],
        inf: &[u32],
        mut out_troops: DisjointSlice<i32>,
        mut out_gold: DisjointSlice<i64>,
        mut out_raw: DisjointSlice<f64>,
        mut out_max: DisjointSlice<f64>,
        mut out_pt: DisjointSlice<f64>,
        mut out_tp: DisjointSlice<f64>,
    ) {
        let idx = thread::index_1d();
        let i = idx.get();
        if i >= troops.len() {
            return;
        }
        let (t, g, raw) = econ_step(
            pt[i],
            difficulty[i],
            troops[i],
            tiles[i],
            city_levels[i],
            gold[i],
            gold_mult[i],
            inf[i] != 0,
        );
        if let Some(slot) = out_troops.get_mut(thread::index_1d()) {
            *slot = t;
        }
        if let Some(slot) = out_gold.get_mut(thread::index_1d()) {
            *slot = g;
        }
        if let Some(slot) = out_raw.get_mut(thread::index_1d()) {
            *slot = raw;
        }
        if let Some(slot) = out_max.get_mut(thread::index_1d()) {
            *slot = max_troops(pt[i], tiles[i], city_levels[i], difficulty[i], inf[i] != 0);
        }
        if let Some(slot) = out_pt.get_mut(thread::index_1d()) {
            *slot = pow_tiles(tiles[i] as f64);
        }
        if let Some(slot) = out_tp.get_mut(thread::index_1d()) {
            *slot = pow_troops(troops[i] as f64);
        }
    }
}

/// Split a `Case` into the parallel input planes the kernel takes.
struct Planes {
    pt: Vec<u32>,
    difficulty: Vec<u32>,
    troops: Vec<i32>,
    tiles: Vec<i32>,
    city_levels: Vec<i64>,
    gold: Vec<i64>,
    gold_mult: Vec<f64>,
    inf: Vec<u32>,
}

fn planes(rows: &[Row]) -> Planes {
    Planes {
        pt: rows.iter().map(|r| r.player_type).collect(),
        difficulty: rows.iter().map(|r| r.difficulty).collect(),
        troops: rows.iter().map(|r| r.troops).collect(),
        tiles: rows.iter().map(|r| r.tiles).collect(),
        city_levels: rows.iter().map(|r| r.city_levels).collect(),
        gold: rows.iter().map(|r| r.gold).collect(),
        gold_mult: rows.iter().map(|r| r.gold_multiplier).collect(),
        inf: rows.iter().map(|r| if r.infinite_troops { 1 } else { 0 }).collect(),
    }
}

fn print_diffs(diffs: &[FieldDiff]) {
    if diffs.is_empty() {
        println!("  field diffs GPU-vs-CPU: none");
        return;
    }
    println!("  field diffs GPU-vs-CPU (one row per field, first differing tick):");
    println!("    {:<16} {:<22} {:<24} {:<24} {}", "field", "first tick", "identity", "cpu", "gpu");
    for d in diffs {
        println!(
            "    {:<16} {:<22} {:<24} {:<24} {}",
            d.field, d.tick, d.identity, d.cpu, d.gpu
        );
    }
}

fn run_one(
    launch: &dyn Fn(&Case) -> Result<Vec<StepOut>, String>,
    case: &Case,
    label: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    let cpu = run_cpu(&case.rows);
    let gpu = launch(case).map_err(|e| format!("launch: {e}"))?;
    let (all_ok, raw_ok) = count_exact(&cpu, &gpu);
    let diffs = diff_steps(&case.rows, &cpu, &gpu);

    println!("=== {label} ===");
    println!(
        "  record {} (source {}, {} rows){}",
        case.record,
        case.source,
        case.rows.len(),
        if case.rows.iter().all(|r| r.synthetic) {
            "  [SYNTHESISED]"
        } else {
            ""
        }
    );
    println!(
        "  GPU vs CPU reference (same core, one on the device): rows={} exact_out={} ({}%) raw_income_bits_equal={} ({}%)",
        case.rows.len(),
        all_ok,
        100.0 * all_ok as f64 / case.rows.len() as f64,
        raw_ok,
        100.0 * raw_ok as f64 / case.rows.len() as f64,
    );
    print_diffs(&diffs);
    let counts = ofcuda_econ::field_divergence_counts(&cpu, &gpu);
    println!("  field divergence counts GPU-vs-CPU:");
    for (name, n) in &counts {
        println!("      {:<19} {}/{}", name, n, case.rows.len());
    }
    let ulp = ofcuda_econ::max_raw_ulp(&cpu, &gpu);
    let f32agree = ofcuda_econ::f32_agreement(&cpu, &gpu);
    let abs = ofcuda_econ::max_raw_abs(&cpu, &gpu);
    println!(
        "  raw_income float: worst |delta| {:.3e} (worst |ULP| {} - ULP is relative, and raw_income \
is near-cancelled at maxTroops); narrowed to f32 the device and host agree on {}/{}",
        abs,
        ulp,
        f32agree,
        case.rows.len()
    );

    // Agreement with the engine's own observed next-tick state.
    let (agree, bad) = engine_agreement(&case.rows, &cpu);
    println!(
        "  engine agreement: predicted troops+gold == engine's own next-tick dump on {}/{} rows ({}%)",
        agree,
        case.rows.len(),
        100.0 * agree as f64 / case.rows.len() as f64
    );
    if bad.is_empty() {
        println!("  differing rows: none");
    } else {
        let unlabelled = bad.iter().filter(|&&i| case.rows[i].attack_activity == 0).count();
        println!(
            "  differing rows: {} (first tick {}), of which {} have no engine attack snapshot nearby",
            bad.len(),
            case.rows[bad[0]].tick,
            unlabelled
        );
        println!(
            "    {:<8} {:<24} {:<8} {:<9} {:<9} {:<10} {:<9} {:<9} {}",
            "tick", "identity", "type", "troops", "pred", "observed", "residual", "dTiles", "attacks"
        );
        for &i in bad.iter().take(10) {
            let r = &case.rows[i];
            println!(
                "    {:<8} {:<24} {:<8} {:<9} {:<9} {:<10} {:<9} {:<9} {}",
                r.tick,
                r.identity,
                ofcuda_econ::player_type_name(r.player_type),
                r.troops,
                cpu[i].troops_after - r.troops,
                r.next_troops - r.troops,
                r.next_troops - cpu[i].troops_after,
                r.d_tiles,
                r.attack_activity
            );
        }
        if bad.len() > 10 {
            println!("    ... {} more", bad.len() - 10);
        }
    }
    // Pass criteria, stated separately so a bounded float artefact cannot hide
    // a state divergence (and vice versa):
    //   1. the engine-visible *state* this slice writes - troops (i32) and gold
    //      (i64) - must be bit-identical on the device and on the host;
    //   2. the only float divergence allowed is the <=1-ULP libdevice `pow`
    //      artefact, and it must vanish under the f32 narrowing the obs planes
    //      apply;
    //   3. every disagreement with the engine's own next-tick dump must sit on a
    //      transition where the engine reports an attack (out of this slice).
    let (pow_only, fma_only, dev_ctr, dev_2step, nrows) =
        ofcuda_econ::attribute_max_divergence(&case.rows, &cpu, &gpu);
    println!(
        "  maxTroops attribution: pow-induced divergence {}/{} (device pow vs host pow, same \
two-step form); the contraction the backend emits without `survive_opt` would perturb {}/{}; \
device maxTroops equals the two-step form on {}/{} and the contracted form on {}/{}",
        pow_only, nrows, fma_only, nrows, dev_2step, nrows, dev_ctr, nrows
    );

    let state_ok = counts[0].1 == 0 && counts[1].1 == 0;
    // The obs planes consume these floats after narrowing to f32; require the
    // narrowing to be bit-identical on every row, which is the strongest
    // claim a <=1-ULP f64 source difference can support.
    let float_ok = f32agree == case.rows.len();
    let synth = case.rows.iter().all(|r| r.synthetic);
    let agree_ok = synth || bad.iter().all(|&i| case.rows[i].attack_activity > 0);
    println!(
        "  PASS state_bitexact {} | float_f32_identical {} | engine_diffs_all_attack_labelled {}{}",
        state_ok,
        float_ok,
        agree_ok,
        if synth { " (no engine truth: SYNTHESISED)" } else { "" }
    );
    let _ = all_ok;
    println!();
    Ok(state_ok && float_ok && agree_ok)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("# ofcuda_econ - per-tick ECONOMY / STATE-PLANE update (slice \"econ\")");
    println!("# TS authority: openfront/src/core/execution/PlayerExecution.ts:75-81 +");
    println!("#   openfront/src/core/configuration/Config.ts:791-868");
    println!("# engine cross-check: rust/engine/src/execution/player.rs:82-88,");
    println!("#   rust/engine/src/core/config.rs:363-481, rust/engine/src/game.rs:1130-1155");
    println!();

    // NOTE: the device module's copy of the core is checked for verbatim
    // equality to src/core_impl.rs by `device_core_is_a_verbatim_copy` in
    // lib.rs, so the constants cannot drift; nothing here names `kernels::*`.

    let ctx = CudaContext::new(0)?;
    // SAFETY: this package owns the embedded device bundle for `kernels`.
    let module = unsafe { kernels::load(&ctx)? };

    // One closure, so the generated module's type is never named (cuda-oxide
    // does not emit a nameable `Module` type) while the launch stays in one
    // place for every case.
    let launch = |case: &Case| -> Result<Vec<StepOut>, String> {
        let stream = ctx.default_stream();
        let p = planes(&case.rows);
        let n = case.rows.len();
        let grid = (n as u32).div_ceil(THREADS);
        let d_pt = DeviceBuffer::from_host(&stream, &p.pt).map_err(|e| e.to_string())?;
        let d_diff = DeviceBuffer::from_host(&stream, &p.difficulty).map_err(|e| e.to_string())?;
        let d_troops = DeviceBuffer::from_host(&stream, &p.troops).map_err(|e| e.to_string())?;
        let d_tiles = DeviceBuffer::from_host(&stream, &p.tiles).map_err(|e| e.to_string())?;
        let d_city = DeviceBuffer::from_host(&stream, &p.city_levels).map_err(|e| e.to_string())?;
        let d_gold = DeviceBuffer::from_host(&stream, &p.gold).map_err(|e| e.to_string())?;
        let d_mult = DeviceBuffer::from_host(&stream, &p.gold_mult).map_err(|e| e.to_string())?;
        let d_inf = DeviceBuffer::from_host(&stream, &p.inf).map_err(|e| e.to_string())?;
        let mut o_troops = DeviceBuffer::<i32>::zeroed(&stream, n).map_err(|e| e.to_string())?;
        let mut o_gold = DeviceBuffer::<i64>::zeroed(&stream, n).map_err(|e| e.to_string())?;
        let mut o_raw = DeviceBuffer::<f64>::zeroed(&stream, n).map_err(|e| e.to_string())?;
        let mut o_max = DeviceBuffer::<f64>::zeroed(&stream, n).map_err(|e| e.to_string())?;
        let mut o_pt = DeviceBuffer::<f64>::zeroed(&stream, n).map_err(|e| e.to_string())?;
        let mut o_tp = DeviceBuffer::<f64>::zeroed(&stream, n).map_err(|e| e.to_string())?;
        let cfg = LaunchConfig1D::new(grid, THREADS, 0);
        let prep = module
            .prepare_econ_step_kernel(cfg)
            .map_err(|e| e.to_string())?;
        {
            module
                .econ_step_kernel(
                    &stream,
                    &prep,
                    &d_pt,
                    &d_diff,
                    &d_troops,
                    &d_tiles,
                    &d_city,
                    &d_gold,
                    &d_mult,
                    &d_inf,
                    &mut o_troops,
                    &mut o_gold,
                    &mut o_raw,
                    &mut o_max,
                    &mut o_pt,
                    &mut o_tp,
                )
                .map_err(|e| e.to_string())?;
        }
        let t = o_troops.to_host_vec(&stream).map_err(|e| e.to_string())?;
        let g = o_gold.to_host_vec(&stream).map_err(|e| e.to_string())?;
        let raw = o_raw.to_host_vec(&stream).map_err(|e| e.to_string())?;
        let mx = o_max.to_host_vec(&stream).map_err(|e| e.to_string())?;
        let ptw = o_pt.to_host_vec(&stream).map_err(|e| e.to_string())?;
        let tpw = o_tp.to_host_vec(&stream).map_err(|e| e.to_string())?;
        Ok((0..n)
            .map(|i| StepOut {
                troops_after: t[i],
                gold_after: g[i],
                raw_income: raw[i],
                max_troops: mx[i],
                pow_tiles: ptw[i],
                pow_troops: tpw[i],
            })
            .collect())
    };

    let mut all_pass = true;
    for path in recorded_case_paths() {
        let case = Case::load(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        all_pass &= run_one(&launch, &case, "recorded, bot-only, zero-intent episode")?;
    }

    // Synthetic city-level exercise of the `cityLevelSum * 250000` term.
    let syn = Case::load(&synthetic_case_path())?;
    all_pass &= run_one(&launch, &syn, "SYNTHESISED city-level sweep")?;

    // FMA contraction: does the one contractible expression (`t^0.6*1000 +
    // 50000`, and `a + city*250000`) change bits when contracted?
    let (tested, differing, first) = fma_sensitivity();
    println!("=== FMA contraction sensitivity (host arithmetic) ===");
    println!(
        "  points tested {}; two-step vs mul_add differ on {} ({})",
        tested,
        differing,
        first.map(|(a, b, c)| format!("first ({a}, {b}) -> {c}")).unwrap_or_else(|| "none".into())
    );
    println!();

    println!(
        "PROOF gpu_matches_cpu_on_every_row {}",
        if all_pass { "true" } else { "false" }
    );
    if !all_pass {
        std::process::exit(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_row_matches_econ_step() {
        let r = Row {
            tick: 300,
            troops: 12_500,
            tiles: 38,
            city_levels: 0,
            gold: 0,
            player_type: ofcuda_econ::core_impl::PT_NATION,
            difficulty: ofcuda_econ::core_impl::DIFF_EASY,
            gold_multiplier: 1.0,
            infinite_troops: false,
            next_troops: 0,
            next_gold: 0,
            attack_activity: 0,
            d_tiles: 0,
            identity: "x".into(),
            synthetic: false,
        };
        let s = step_row(&r);
        let (t, g, raw) = ofcuda_econ::core_impl::econ_step(
            r.player_type,
            r.difficulty,
            r.troops,
            r.tiles,
            r.city_levels,
            r.gold,
            r.gold_multiplier,
            r.infinite_troops,
        );
        assert_eq!(s.troops_after, t);
        assert_eq!(s.gold_after, g);
        assert_eq!(s.raw_income.to_bits(), raw.to_bits());
    }
}