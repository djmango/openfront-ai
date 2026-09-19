//! Host-side shared code for the `ofcuda_econ` parity harness.
//!
//! Nothing here touches CUDA: the GPU binary (`src/main.rs`) and the CPU
//! companion binary (`src/bin/cpu.rs`) both link this crate, so the case
//! loading, the income arithmetic (`core_impl`) and the printed format are
//! literally the same code. Only the *execution* differs (a cuda-oxide kernel
//! vs. a plain sequential Rust loop), which is what makes a mismatch
//! meaningful.
//!
//! ## What this slice is, and what it turned out to be
//!
//! The per-tick economy/state-plane update is **per player, not per tile**.
//! `PlayerExecution.tick` (`openfront/src/core/execution/PlayerExecution.ts:44-113`)
//! mutates exactly two per-player scalars each tick outside territory conquest:
//! `_troops` (`PlayerExecution.ts:75-76`) and `_gold` (`:77-78`). There is no
//! per-tile population or per-tile defence field in the engine at all
//! (`grep -rn population openfront/src/core/` is empty), and the tile-state
//! `u16` words the obs and the hash read carry only owner (bits 0..=11),
//! fallout (bit 13) and defense-bonus (bit 14), each written by
//! conquest/nuke/unit code and never by this slice - see README.md
//! "The tile-state buffer" for the citations. So "population growth, gold
//! income, troop growth per player" is the whole slice, and the per-tile half
//! of the deliverable is a negative result, reported rather than assumed.
//!
//! ## Provenance of the inputs
//!
//! `cases/*.json` are **reconstructed** from dumps taken with the CURRENT
//! tree's own `tick_dump` (`CARGO_TARGET_DIR=/tmp/ofecon-target`,
//! `OF_DUMP_UNITS=1 --every 1`), never synthesised: one row per
//! (player, tick-transition) with the engine's own `troops`, `tiles`,
//! `gold`, its own `City` unit levels (for the `cityLevelSum` term) and the
//! *observed* next-tick `troops`/`gold`. `cases/synthetic_city_levels.json`
//! is the one deliberately **synthesised** input, used only to exercise the
//! `cityLevelSum * 250000` term on the GPU/CPU path (no recorded episode in
//! the set builds a city before tick 1200).

pub mod core_impl;

use std::path::{Path, PathBuf};

/// One recorded tick-transition for one player.
#[derive(Debug, Clone)]
pub struct Row {
    pub tick: u32,
    /// `troops` at the start of the transition (the economy step's input).
    pub troops: i32,
    pub tiles: i32,
    pub city_levels: i64,
    pub gold: i64,
    pub player_type: u32,
    pub difficulty: u32,
    pub gold_multiplier: f64,
    pub infinite_troops: bool,
    /// Observed state at the next dump tick (engine ground truth).
    pub next_troops: i32,
    pub next_gold: i64,
    /// Engine's own attack snapshots adjacent to this transition (0 = none).
    pub attack_activity: u32,
    /// Observed tile-count change (diagnostic; the econ slice never changes it).
    pub d_tiles: i32,
    pub identity: String,
    pub synthetic: bool,
}

#[derive(Debug, Clone)]
pub struct Case {
    pub name: String,
    pub record: String,
    pub source: String,
    pub rows: Vec<Row>,
}

impl Case {
    pub fn load(path: &Path) -> Result<Case, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let record = v["record"].as_str().unwrap_or("?").to_string();
        let source = v["source"].as_str().unwrap_or("?").to_string();
        let diff_str = v["difficulty"].as_str().unwrap_or("Easy");
        let difficulty = difficulty_code(diff_str);
        let gold_multiplier = v["goldMultiplier"].as_f64().unwrap_or(1.0);
        let infinite_troops = v["infiniteTroops"].as_bool().unwrap_or(false);
        // A case whose own `source` says it was synthesised has no engine
        // ground truth at all; marking the rows keeps the engine-agreement
        // check from being applied to invented data.
        let synth = source.contains("synthesised");
        let rows_v = v["rows"].as_array().ok_or("no rows")?;
        let mut rows = Vec::with_capacity(rows_v.len());
        for r in rows_v {
            let pt = player_type_code(r["playerType"].as_str().unwrap_or("Human"));
            rows.push(Row {
                tick: r["tick"].as_u64().unwrap_or(0) as u32,
                troops: r["troops"].as_i64().unwrap_or(0) as i32,
                tiles: r["tiles"].as_i64().unwrap_or(0) as i32,
                city_levels: r["cityLevels"].as_i64().unwrap_or(0),
                gold: r["gold"].as_i64().unwrap_or(0),
                player_type: pt,
                difficulty: r["difficulty"]
                    .as_str()
                    .map(difficulty_code)
                    .unwrap_or(difficulty),
                gold_multiplier: r["goldMultiplier"].as_f64().unwrap_or(gold_multiplier),
                infinite_troops: r["infiniteTroops"].as_bool().unwrap_or(infinite_troops),
                next_troops: r["nextTroops"].as_i64().unwrap_or(0) as i32,
                next_gold: r["nextGold"].as_i64().unwrap_or(0),
                attack_activity: r["attackActivity"].as_u64().unwrap_or(0) as u32,
                d_tiles: r["dTiles"].as_i64().unwrap_or(0) as i32,
                identity: r["identity"].as_str().unwrap_or("?").to_string(),
                // A case whose own `source` says it was synthesised has no
                // engine ground truth at all; marking the rows keeps the
                // engine-agreement check from being applied to invented data.
                synthetic: synth,
            });
        }
        Ok(Case {
            name: path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
            record,
            source,
            rows,
        })
    }

    /// Load a file of raw rows (used for the hand-built synthetic city case).
    pub fn load_rows_only(path: &Path) -> Result<Case, String> {
        Case::load(path)
    }
}

pub fn difficulty_code(s: &str) -> u32 {
    match s {
        "Easy" => core_impl::DIFF_EASY,
        "Medium" => core_impl::DIFF_MEDIUM,
        "Hard" => core_impl::DIFF_HARD,
        "Impossible" => core_impl::DIFF_IMPOSSIBLE,
        _ => core_impl::DIFF_EASY,
    }
}

pub fn difficulty_name(code: u32) -> &'static str {
    match code {
        core_impl::DIFF_EASY => "Easy",
        core_impl::DIFF_MEDIUM => "Medium",
        core_impl::DIFF_HARD => "Hard",
        core_impl::DIFF_IMPOSSIBLE => "Impossible",
        _ => "?",
    }
}

pub fn player_type_code(s: &str) -> u32 {
    match s {
        "Bot" => core_impl::PT_BOT,
        "Nation" => core_impl::PT_NATION,
        _ => core_impl::PT_HUMAN,
    }
}

pub fn player_type_name(code: u32) -> &'static str {
    match code {
        core_impl::PT_BOT => "Bot",
        core_impl::PT_NATION => "Nation",
        _ => "Human",
    }
}

/// Output of one economy step, both as the engine would store it and as the
/// raw pre-truncation float whose *bits* must match between GPU and CPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepOut {
    pub troops_after: i32,
    pub gold_after: i64,
    /// `troops + troopIncreaseRateRaw(...)`, i.e. the pre-`addTroops` value.
    /// Compared bit-for-bit between the device and the host.
    pub raw_income: f64,
    /// Intermediate diagnostics, so a divergence can be localised to a single
    /// operation instead of guessed at.
    pub max_troops: f64,
    pub pow_tiles: f64,
    pub pow_troops: f64,
}

pub fn step_row(r: &Row) -> StepOut {
    let (t, g, raw) = core_impl::econ_step(
        r.player_type,
        r.difficulty,
        r.troops,
        r.tiles,
        r.city_levels,
        r.gold,
        r.gold_multiplier,
        r.infinite_troops,
    );
    StepOut {
        troops_after: t,
        gold_after: g,
        raw_income: raw,
        max_troops: core_impl::max_troops(
            r.player_type,
            r.tiles,
            r.city_levels,
            r.difficulty,
            r.infinite_troops,
        ),
        pow_tiles: core_impl::pow_tiles(r.tiles as f64),
        pow_troops: core_impl::pow_troops(r.troops as f64),
    }
}

/// The CPU reference: a plain sequential fold over the rows, on the shared
/// core. This is what the GPU output is compared against.
pub fn run_cpu(rows: &[Row]) -> Vec<StepOut> {
    rows.iter().map(step_row).collect()
}

// ---------------------------------------------------------------------------
// Cases on disk
// ---------------------------------------------------------------------------

pub fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

pub fn cases_dir() -> PathBuf {
    crate_dir().join("cases")
}

/// The recorded cases, in a fixed order.
pub fn recorded_case_paths() -> Vec<PathBuf> {
    vec![
        cases_dir().join("trans-b000s3.json"),
        cases_dir().join("trans-b005s0.json"),
    ]
}

pub fn synthetic_case_path() -> PathBuf {
    cases_dir().join("synthetic_city_levels.json")
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

/// One field-level diff between the GPU's step and the CPU reference.
#[derive(Debug, Clone)]
pub struct FieldDiff {
    pub tick: u32,
    pub identity: String,
    pub field: &'static str,
    pub cpu: String,
    pub gpu: String,
}

/// Compare a GPU output vector against the CPU reference, field by field, and
/// report the first differing tick per field.
/// The per-row output fields compared bit-for-bit between the device and the
/// host reference. Named so a divergence can be attributed to one operation.
pub const STEP_FIELDS: [&str; 6] = [
    "troops(running)",
    "gold(running)",
    "rawIncome(bits)",
    "maxTroops(bits)",
    "tilesPow06(bits)",
    "troopsPow073(bits)",
];

fn field_bits(s: &StepOut, f: usize) -> u64 {
    match f {
        0 => s.troops_after as u64,
        1 => s.gold_after as u64,
        2 => s.raw_income.to_bits(),
        3 => s.max_troops.to_bits(),
        4 => s.pow_tiles.to_bits(),
        5 => s.pow_troops.to_bits(),
        _ => unreachable!(),
    }
}

fn field_str(s: &StepOut, f: usize) -> String {
    let b = field_bits(s, f);
    match f {
        0 => format!("{b}"),
        1 => format!("{b}"),
        _ => format!("0x{b:016x} ({})", f64::from_bits(b)),
    }
}

/// Distance in ULPs between two `f64`, using the usual total ordering over
/// IEEE-754 bit patterns (so `-0.0` and `+0.0` are 1 apart, which is what a
/// bit-level comparison wants).
pub fn ulp_delta(a: f64, b: f64) -> u64 {
    if a == b {
        return 0;
    }
    let key = |x: f64| -> i64 {
        let bits = x.to_bits() as i64;
        if bits < 0 { i64::MIN - bits } else { bits }
    };
    (key(a) - key(b)).unsigned_abs()
}

/// Rows whose `raw_income` still compares equal after the `f64 -> f32` narrowing
/// the observation planes apply. This is the measurement that says whether a
/// sub-ULP `f64` difference can reach an obs plane at all.
pub fn f32_agreement(cpu: &[StepOut], gpu: &[StepOut]) -> usize {
    cpu.iter()
        .zip(gpu.iter())
        .filter(|(c, g)| (c.raw_income as f32).to_bits() == (g.raw_income as f32).to_bits())
        .count()
}

/// Worst-case `raw_income` distance (in ULP) between the device and the host.
///
/// Read with care: `raw_income` is `min(troops + toAdd, max) - troops`, which is
/// near zero for a player pinned at `maxTroops`. ULP distance is a *relative*
/// measure, so a near-cancelled result inflates it; the absolute and f32
/// measures below are the ones that matter.
pub fn max_raw_ulp(cpu: &[StepOut], gpu: &[StepOut]) -> u64 {
    cpu.iter()
        .zip(gpu.iter())
        .map(|(c, g)| ulp_delta(c.raw_income, g.raw_income))
        .max()
        .unwrap_or(0)
}

/// Worst-case absolute `raw_income` difference (device vs host).
pub fn max_raw_abs(cpu: &[StepOut], gpu: &[StepOut]) -> f64 {
    cpu.iter()
        .zip(gpu.iter())
        .map(|(c, g)| (c.raw_income - g.raw_income).abs())
        .fold(0.0f64, f64::max)
}

/// The `maxTroops` expression rebuilt from an *explicit* `pow_tiles` value, in
/// either form, so the two candidates for a device/host divergence - a 1-ULP
/// `pow` and an FMA contraction of `pow*1000 + 50000` / `city*250000 + t` - can
/// be told apart on identical `pow` inputs.
///
/// `contracted = false` is the engine's form (two separately rounded steps).
/// `contracted = true` is the form the emitted PTX actually uses
/// (`fma.rn.f64 %rd82, %rd130, 1000.0, 50000.0` and
/// `fma.rn.f64 %rd22, %rd85, 250000.0, %rd84`).
pub fn max_troops_from_pow(
    player_type: u32,
    powt: f64,
    city_level_sum: i64,
    difficulty: u32,
    infinite_troops: bool,
    contracted: bool,
) -> f64 {
    let mut max = if contracted {
        let t = powt.mul_add(1000.0, 50_000.0);
        (city_level_sum as f64).mul_add(core_impl::CITY_TROOP_INCREASE, 2.0 * t)
    } else {
        2.0 * (powt * 1000.0 + 50_000.0) + city_level_sum as f64 * core_impl::CITY_TROOP_INCREASE
    };
    if player_type == core_impl::PT_BOT {
        max /= 3.0;
    } else if player_type == core_impl::PT_HUMAN {
        if infinite_troops {
            return 1_000_000_000.0;
        }
    } else {
        max *= match difficulty {
            core_impl::DIFF_MEDIUM => 0.75,
            core_impl::DIFF_HARD => 1.0,
            core_impl::DIFF_IMPOSSIBLE => 1.25,
            _ => 0.5,
        };
    }
    max
}

/// Attribution of any `maxTroops` divergence: `(pow_only, fma_only,
/// device_is_contracted_form, device_is_two_step_form, rows)`.
///
/// `pow_only` compares the two forms on the *device's own* `pow` value, so it
/// isolates the contraction. `fma_only` compares the two-step form built from
/// the device's `pow` against the two-step form built from the host's `pow`, so
/// it isolates the `pow` error.
pub fn attribute_max_divergence(
    rows: &[Row],
    cpu: &[StepOut],
    gpu: &[StepOut],
) -> (usize, usize, usize, usize, usize) {
    let (mut pow_only, mut fma_only, mut dev_ctr, mut dev_2step) = (0, 0, 0, 0);
    for ((r, c), g) in rows.iter().zip(cpu.iter()).zip(gpu.iter()) {
        let form = |powt: f64, contracted: bool| {
            max_troops_from_pow(
                r.player_type,
                powt,
                r.city_levels,
                r.difficulty,
                r.infinite_troops,
                contracted,
            )
        };
        let two_host = form(c.pow_tiles, false);
        let two_dev = form(g.pow_tiles, false);
        let ctr_dev = form(g.pow_tiles, true);
        // Sanity: the two-step form rebuilt from the host's own pow must be the
        // host's max_troops (otherwise this diagnostic is not measuring what it
        // claims).
        debug_assert_eq!(two_host.to_bits(), c.max_troops.to_bits());
        if two_host.to_bits() != two_dev.to_bits() {
            pow_only += 1;
        }
        if two_dev.to_bits() != ctr_dev.to_bits() {
            fma_only += 1;
        }
        if g.max_troops.to_bits() == ctr_dev.to_bits() {
            dev_ctr += 1;
        }
        if g.max_troops.to_bits() == two_dev.to_bits() {
            dev_2step += 1;
        }
    }
    (pow_only, fma_only, dev_ctr, dev_2step, rows.len())
}

/// How many rows each field diverges on (device vs host).
pub fn field_divergence_counts(cpu: &[StepOut], gpu: &[StepOut]) -> Vec<(&'static str, usize)> {
    STEP_FIELDS
        .iter()
        .enumerate()
        .map(|(f, name)| {
            let n = cpu
                .iter()
                .zip(gpu.iter())
                .filter(|(c, g)| field_bits(c, f) != field_bits(g, f))
                .count();
            (*name, n)
        })
        .collect()
}

/// One row per (field, first diverging row): the diff table.
pub fn diff_steps(rows: &[Row], cpu: &[StepOut], gpu: &[StepOut]) -> Vec<FieldDiff> {
    let mut out = Vec::new();
    for f in 0..STEP_FIELDS.len() {
        for i in 0..rows.len().min(cpu.len()).min(gpu.len()) {
            let (c, g) = (&cpu[i], &gpu[i]);
            if field_bits(c, f) != field_bits(g, f) {
                out.push(FieldDiff {
                    tick: rows[i].tick,
                    identity: rows[i].identity.clone(),
                    field: STEP_FIELDS[f],
                    cpu: field_str(c, f),
                    gpu: field_str(g, f),
                });
                break;
            }
        }
    }
    out
}

pub fn count_exact(cpu: &[StepOut], gpu: &[StepOut]) -> (usize, usize) {
    let mut raw = 0;
    let mut all = 0;
    for (c, g) in cpu.iter().zip(gpu.iter()) {
        if c.raw_income.to_bits() == g.raw_income.to_bits() {
            raw += 1;
        }
        if c == g {
            all += 1;
        }
    }
    (all, raw)
}

/// Agreement of the *reference* with the engine's observed next-tick state.
pub fn engine_agreement(rows: &[Row], cpu: &[StepOut]) -> (usize, Vec<usize>) {
    let mut ok = 0;
    let mut bad = Vec::new();
    for (i, (r, s)) in rows.iter().zip(cpu.iter()).enumerate() {
        if s.troops_after == r.next_troops && s.gold_after == r.next_gold {
            ok += 1;
        } else {
            bad.push(i);
        }
    }
    (ok, bad)
}

/// A `Row` where the engine's tile-state plane is *not* what this slice writes.
/// (Structural, not measured: the econ core has no tile argument at all.)
pub const TILE_STATE_CLAIM: &str =
    "the econ slice writes no tile-state word; owner/fallout/defense-bonus are \
     written only by conquest/nuke/unit paths (see README.md)";

/// FMA contraction sensitivity of `max_troops`' two contractible expressions:
/// `tiles^0.6 * 1000.0 + 50_000.0` and `2.0 * a + city_levels * 250_000.0`.
///
/// cuda-oxide's lowering has contracted an expression before (see
/// `ofcuda_tick`'s PTX), and the engine's own Rust build does not contract, so
/// if a contraction changes a bit *here* it would change the income's last bit.
/// Returns `(points tested, points where the two forms differ, first example)`.
pub fn fma_sensitivity() -> (u64, u64, Option<(i32, i64, String)>) {
    let mut tested = 0u64;
    let mut differing = 0u64;
    let mut first = None;
    let mut tiles = 0i32;
    // The recorded rows' tile counts plus a broad sweep; city levels up to 4.
    while tiles <= 20_000 {
        let t = tiles as f64;
        let two_step = t.powf(0.6) * 1000.0 + 50_000.0;
        let contracted = t.powf(0.6).mul_add(1000.0, 50_000.0);
        tested += 1;
        if two_step.to_bits() != contracted.to_bits() && first.is_none() {
            differing += 1;
            first = Some((tiles, 0, format!("a: {two_step:?} vs {contracted:?}")));
        } else if two_step.to_bits() != contracted.to_bits() {
            differing += 1;
        }
        for city in 0i64..=4 {
            let two = 2.0 * (t.powf(0.6) * 1000.0 + 50_000.0) + city as f64 * core_impl::CITY_TROOP_INCREASE;
            let con = (2.0 * (t.powf(0.6) * 1000.0 + 50_000.0))
                .mul_add(1.0, city as f64 * core_impl::CITY_TROOP_INCREASE);
            tested += 1;
            if two.to_bits() != con.to_bits() {
                differing += 1;
                if first.is_none() {
                    first = Some((tiles, city, format!("b: {two:?} vs {con:?}")));
                }
            }
        }
        tiles += if tiles < 200 { 1 } else { 37 };
    }
    (tested, differing, first)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The device copy inside `src/main.rs` must be the whole of
    /// `src/core_impl.rs`, verbatim. Fix a failure with `bash sync_device_core.sh`.
    #[test]
    fn device_core_is_a_verbatim_copy() {
        let core = include_str!("core_impl.rs").trim_end();
        let dev = include_str!("main.rs");
        assert!(
            dev.contains(core),
            "the device copy inside src/main.rs diverged from src/core_impl.rs - \
             run `bash sync_device_core.sh`"
        );
    }

    /// The engine's own TS values, re-derived by hand from
    /// `Config.ts:791-857` for a concrete Bot. Guards the port against a
    /// transcription slip in `max_troops` / `troop_increase_rate_raw`.
    #[test]
    fn troop_rate_matches_the_ts_formula_by_hand() {
        let tiles = 1000_i32;
        let troops = 5_000_i32;
        let max = 2.0 * ((tiles as f64).powf(0.6) * 1000.0 + 50_000.0) / 3.0;
        let mut to_add = 10.0 + (troops as f64).powf(0.73) / 4.0;
        to_add *= 1.0 - troops as f64 / max;
        to_add *= 0.5;
        let expect = (troops as f64 + to_add).min(max) - troops as f64;
        assert_eq!(
            core_impl::troop_increase_rate_raw(core_impl::PT_BOT, troops, tiles, 0, core_impl::DIFF_EASY, false),
            expect
        );
    }

    /// The negative-income rule: `addTroops(-6360.28)` removes `floor(6360.28)`
    /// = 6360, not `floor(-6360.28)` = -6361 (`execution/player.rs:130-146`).
    #[test]
    fn negative_income_floors_the_magnitude() {
        assert_eq!(core_impl::add_troops(10_000, -6360.282895866956), 10_000 - 6360);
        assert_eq!(core_impl::add_troops(10_000, 6360.282895866956), 10_000 + 6360);
    }

    /// Gold rate: 50/tick for a Bot, 100/tick otherwise, `floor(base*mult)`.
    #[test]
    fn gold_rates() {
        assert_eq!(core_impl::gold_addition_rate(core_impl::PT_BOT, 1.0), 50);
        assert_eq!(core_impl::gold_addition_rate(core_impl::PT_NATION, 1.0), 100);
        assert_eq!(core_impl::gold_addition_rate(core_impl::PT_HUMAN, 1.0), 100);
        // `goldMultiplier` is a float: floor(50*1.5) = 75.
        assert_eq!(core_impl::gold_addition_rate(core_impl::PT_BOT, 1.5), 75);
    }

    /// The `cityLevelSum * 250000` term is live in `maxTroops`, so it must move
    /// the rate (the recorded cases all have no city before tick 1200, which is
    /// why the synthetic case exists).
    #[test]
    fn city_levels_raise_max_troops() {
        let no_city =
            core_impl::max_troops(core_impl::PT_NATION, 1000, 0, core_impl::DIFF_EASY, false);
        let city =
            core_impl::max_troops(core_impl::PT_NATION, 1000, 1, core_impl::DIFF_EASY, false);
        // The city term is added *before* the per-type scale, so the increase is
        // not `250000 * 0.5` once rounding is taken into account: it is exactly
        // `(base + 250000)*0.5 - base*0.5`. Asserting the exact expression keeps
        // the order of operations load-bearing instead of incidental.
        let base = 2.0 * (1000f64.powf(0.6) * 1000.0 + 50_000.0);
        assert_eq!(no_city, base * 0.5);
        assert_eq!(city - no_city, (base + 250_000.0) * 0.5 - base * 0.5);
        assert!(city > no_city);
    }

    /// Every recorded case must load and produce the row count the dump did.
    #[test]
    fn recorded_cases_load() {
        for p in recorded_case_paths() {
            let c = Case::load(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            assert!(!c.rows.is_empty(), "{} has no rows", p.display());
            // A recorded case must carry engine ground truth.
            assert!(c.rows.iter().any(|r| r.next_troops != r.troops));
        }
    }

    /// The CPU reference must reproduce the engine's own observed next-tick
    /// state on the pure-economy majority, and the only rows it may miss are
    /// the ones the engine's own attack snapshots flag.
    #[test]
    fn cpu_reference_matches_the_engine_except_on_attack_ticks() {
        for p in recorded_case_paths() {
            let c = Case::load(&p).expect("load");
            let cpu = run_cpu(&c.rows);
            let (ok, bad) = engine_agreement(&c.rows, &cpu);
            assert!(ok * 100 / c.rows.len() >= 98, "{}: only {ok}/{}", c.name, c.rows.len());
            for i in bad {
                assert!(
                    c.rows[i].attack_activity > 0,
                    "{}: tick {} differs from the engine with no attack nearby",
                    c.name,
                    c.rows[i].tick
                );
            }
        }
    }
}