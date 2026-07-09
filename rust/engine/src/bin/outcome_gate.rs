use clap::Parser;
use openfront_engine::replay::{
    compare_outcomes, replay_outcome_native, GameOutcome, OutcomeComparison, OutcomeOracleCache,
    OUTCOME_EXPECTED_RECORDS, OUTCOME_REQUIRED_PASSES, OUTCOME_SCHEMA_VERSION,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "openfront-outcome-gate")]
#[command(about = "Compare cached TypeScript and native replay outcomes")]
struct Args {
    #[arg(long)]
    records: PathBuf,
    #[arg(long)]
    oracle: PathBuf,
    #[arg(long, default_value = ".")]
    repo: PathBuf,
    #[arg(long)]
    parity_commit: String,
    #[arg(long, default_value_t = OUTCOME_EXPECTED_RECORDS)]
    expected_records: usize,
    #[arg(long, default_value_t = OUTCOME_REQUIRED_PASSES)]
    required_passes: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct RecordReport {
    game_id: String,
    record: String,
    category: String,
    diagnostics: Vec<String>,
    expected: Option<GameOutcome>,
    actual: Option<GameOutcome>,
    comparison: Option<OutcomeComparison>,
    error: Option<String>,
}

#[derive(Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct CategoryCounts {
    pass: usize,
    wrong_winner: usize,
    missing_winner: usize,
    timing_mismatch: usize,
    land_share_mismatch: usize,
    replay_error: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GateSummary {
    pass: usize,
    total: usize,
    required_passes: usize,
    expected_records: usize,
    record_count_match: bool,
    threshold_met: bool,
    gate_pass: bool,
    categories: CategoryCounts,
    diagnostics: CategoryCounts,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct GateReport {
    schema_version: u32,
    parity_commit: String,
    oracle_record_set_hash: String,
    summary: GateSummary,
    records: Vec<RecordReport>,
}

fn record_paths(records: &Path) -> Result<Vec<PathBuf>, String> {
    let mut paths = std::fs::read_dir(records)
        .map_err(|e| format!("read records directory {}: {e}", records.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with(".json") || name.ends_with(".json.gz"))
        })
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

fn game_id_from_path(path: &Path) -> String {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    name.strip_suffix(".json.gz")
        .or_else(|| name.strip_suffix(".json"))
        .unwrap_or(name)
        .to_string()
}

fn increment(counts: &mut CategoryCounts, category: &str) {
    match category {
        "pass" => counts.pass += 1,
        "wrong_winner" => counts.wrong_winner += 1,
        "missing_winner" => counts.missing_winner += 1,
        "timing_mismatch" => counts.timing_mismatch += 1,
        "land_share_mismatch" => counts.land_share_mismatch += 1,
        "replay_error" => counts.replay_error += 1,
        _ => {}
    }
}

fn run(args: Args) -> Result<GateReport, String> {
    let cache: OutcomeOracleCache = serde_json::from_slice(
        &std::fs::read(&args.oracle)
            .map_err(|e| format!("read oracle {}: {e}", args.oracle.display()))?,
    )
    .map_err(|e| format!("parse oracle {}: {e}", args.oracle.display()))?;
    if cache.schema_version != OUTCOME_SCHEMA_VERSION {
        return Err(format!(
            "oracle schema {} does not match expected {}",
            cache.schema_version, OUTCOME_SCHEMA_VERSION
        ));
    }
    if cache.parity_commit != args.parity_commit {
        return Err(format!(
            "oracle commit {} does not match requested {}",
            cache.parity_commit, args.parity_commit
        ));
    }
    let oracle_by_id: BTreeMap<String, GameOutcome> = cache
        .outcomes
        .into_iter()
        .map(|outcome| (outcome.game_id.clone(), outcome))
        .collect();
    let paths = record_paths(&args.records)?;
    let mut records = Vec::with_capacity(paths.len());
    let mut categories = CategoryCounts::default();
    let mut diagnostics = CategoryCounts::default();

    for path in paths {
        let game_id = game_id_from_path(&path);
        let record = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("")
            .to_string();
        let expected = oracle_by_id.get(&game_id).cloned();
        let actual = replay_outcome_native(&args.repo, &path);
        let report = match (expected, actual) {
            (Some(expected), Ok(actual)) => {
                let comparison = compare_outcomes(&expected, &actual);
                increment(&mut categories, &comparison.category);
                for diagnostic in &comparison.diagnostics {
                    increment(&mut diagnostics, diagnostic);
                }
                if comparison.pass {
                    increment(&mut diagnostics, "pass");
                }
                RecordReport {
                    game_id,
                    record,
                    category: comparison.category.clone(),
                    diagnostics: comparison.diagnostics.clone(),
                    expected: Some(expected),
                    actual: Some(actual),
                    comparison: Some(comparison),
                    error: None,
                }
            }
            (None, _) => {
                increment(&mut categories, "replay_error");
                increment(&mut diagnostics, "replay_error");
                RecordReport {
                    game_id,
                    record,
                    category: "replay_error".to_string(),
                    diagnostics: vec!["replay_error".to_string()],
                    expected: None,
                    actual: None,
                    comparison: None,
                    error: Some("record missing from TypeScript oracle".to_string()),
                }
            }
            (Some(expected), Err(error)) => {
                increment(&mut categories, "replay_error");
                increment(&mut diagnostics, "replay_error");
                RecordReport {
                    game_id,
                    record,
                    category: "replay_error".to_string(),
                    diagnostics: vec!["replay_error".to_string()],
                    expected: Some(expected),
                    actual: None,
                    comparison: None,
                    error: Some(error),
                }
            }
        };
        records.push(report);
    }
    let total = records.len();
    let pass = categories.pass;
    let record_count_match = total == args.expected_records && oracle_by_id.len() == total;
    let threshold_met = pass >= args.required_passes;
    let gate_pass = record_count_match && threshold_met;
    Ok(GateReport {
        schema_version: OUTCOME_SCHEMA_VERSION,
        parity_commit: args.parity_commit,
        oracle_record_set_hash: cache.record_set_hash,
        summary: GateSummary {
            pass,
            total,
            required_passes: args.required_passes,
            expected_records: args.expected_records,
            record_count_match,
            threshold_met,
            gate_pass,
            categories,
            diagnostics,
        },
        records,
    })
}

fn main() {
    let report = match run(Args::parse()) {
        Ok(report) => report,
        Err(error) => {
            eprintln!("openfront-outcome-gate: {error}");
            std::process::exit(2);
        }
    };
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
    if !report.summary.gate_pass {
        std::process::exit(1);
    }
}
