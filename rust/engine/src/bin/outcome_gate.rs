use clap::Parser;
use openfront_engine::replay::{
    compare_outcomes, replay_outcome_native, GameOutcome, OutcomeComparison, OutcomeOracleCache,
    OUTCOME_EXPECTED_RECORDS, OUTCOME_REQUIRED_PASSES, OUTCOME_SCHEMA_VERSION,
};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

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
    #[arg(long)]
    limit: Option<usize>,
    /// Records are fully independent full-game replays (real archived
    /// multiplayer games - dozens of players/bots/nations, thousands of
    /// ticks - a very different compute profile from the tiny RL toy
    /// games the engine's ticks/s benchmark measures), so this is
    /// embarrassingly parallel. Was hardcoded sequential before, which is
    /// why a 78-record run took hours single-threaded with zero progress
    /// output.
    #[arg(long, default_value_t = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4))]
    jobs: usize,
    /// Per-record wall-clock cap so one pathological record can't stall
    /// the whole gate silently (there was no timeout at all before this).
    #[arg(long, default_value_t = 300)]
    record_timeout_seconds: u64,
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

fn record_paths(records: &Path, limit: Option<usize>) -> Result<Vec<PathBuf>, String> {
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
    if let Some(limit) = limit {
        paths.truncate(limit);
    }
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
    let paths = record_paths(&args.records, args.limit)?;
    let total_paths = paths.len();
    let jobs = args.jobs.max(1);
    let timeout = Duration::from_secs(args.record_timeout_seconds);
    eprintln!(
        "[outcome_gate] comparing {total_paths} records across {jobs} worker thread(s), \
         per-record timeout {}s",
        timeout.as_secs()
    );

    let next_idx = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicUsize::new(0));
    let paths = Arc::new(paths);
    let oracle_by_id = Arc::new(oracle_by_id);
    let repo = Arc::new(args.repo.clone());
    let start = Instant::now();

    let mut results: Vec<Option<RecordReport>> = (0..total_paths).map(|_| None).collect();
    std::thread::scope(|scope| {
        let (tx, rx) = std::sync::mpsc::channel::<(usize, RecordReport)>();
        for _ in 0..jobs.min(total_paths.max(1)) {
            let tx = tx.clone();
            let next_idx = Arc::clone(&next_idx);
            let done = Arc::clone(&done);
            let paths = Arc::clone(&paths);
            let oracle_by_id = Arc::clone(&oracle_by_id);
            let repo = Arc::clone(&repo);
            scope.spawn(move || loop {
                let idx = next_idx.fetch_add(1, Ordering::Relaxed);
                if idx >= paths.len() {
                    return;
                }
                let path = paths[idx].clone();
                let game_id = game_id_from_path(&path);
                let record = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("")
                    .to_string();
                let expected = oracle_by_id.get(&game_id).cloned();

                // Run the actual replay on its own thread with a bounded
                // wait, so one pathologically slow/hung record (e.g. a
                // very long real multiplayer game) can't stall the whole
                // gate forever with no signal - it just gets recorded as
                // a timeout and the pool moves on.
                let (replay_tx, replay_rx) = std::sync::mpsc::channel();
                let repo_for_replay = Arc::clone(&repo);
                let path_for_replay = path.clone();
                let record_t0 = Instant::now();
                std::thread::spawn(move || {
                    let actual = replay_outcome_native(&repo_for_replay, &path_for_replay);
                    let _ = replay_tx.send(actual);
                });
                let actual = match replay_rx.recv_timeout(timeout) {
                    Ok(actual) => actual,
                    Err(_) => Err(format!(
                        "replay exceeded {}s timeout (still running when abandoned)",
                        timeout.as_secs()
                    )),
                };
                let record_dt = record_t0.elapsed().as_secs_f64();

                let report = match (expected, actual) {
                    (Some(expected), Ok(actual)) => {
                        let comparison = compare_outcomes(&expected, &actual);
                        let category = comparison.category.clone();
                        RecordReport {
                            game_id: game_id.clone(),
                            record: record.clone(),
                            category,
                            diagnostics: comparison.diagnostics.clone(),
                            expected: Some(expected),
                            actual: Some(actual),
                            comparison: Some(comparison),
                            error: None,
                        }
                    }
                    (None, _) => RecordReport {
                        game_id: game_id.clone(),
                        record: record.clone(),
                        category: "replay_error".to_string(),
                        diagnostics: vec!["replay_error".to_string()],
                        expected: None,
                        actual: None,
                        comparison: None,
                        error: Some("record missing from TypeScript oracle".to_string()),
                    },
                    (Some(expected), Err(error)) => RecordReport {
                        game_id: game_id.clone(),
                        record: record.clone(),
                        category: "replay_error".to_string(),
                        diagnostics: vec!["replay_error".to_string()],
                        expected: Some(expected),
                        actual: None,
                        comparison: None,
                        error: Some(error),
                    },
                };
                let n_done = done.fetch_add(1, Ordering::Relaxed) + 1;
                eprintln!(
                    "[outcome_gate] {n_done}/{total_paths} {record} -> {} ({record_dt:.1}s)",
                    report.category
                );
                if tx.send((idx, report)).is_err() {
                    return;
                }
            });
        }
        drop(tx);
        for (idx, report) in rx {
            results[idx] = Some(report);
        }
    });
    eprintln!(
        "[outcome_gate] all {total_paths} records compared in {:.1}s",
        start.elapsed().as_secs_f64()
    );

    let mut records: Vec<RecordReport> = results.into_iter().flatten().collect();
    records.sort_by(|a, b| a.record.cmp(&b.record));
    let mut categories = CategoryCounts::default();
    let mut diagnostics = CategoryCounts::default();
    for report in &records {
        increment(&mut categories, &report.category);
        for diagnostic in &report.diagnostics {
            increment(&mut diagnostics, diagnostic);
        }
        if report.category == "pass" {
            increment(&mut diagnostics, "pass");
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_limit_is_applied_after_stable_sorting() {
        let dir = std::env::temp_dir().join(format!("outcome-gate-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("z.json.gz"), []).unwrap();
        std::fs::write(dir.join("a.json"), []).unwrap();
        std::fs::write(dir.join("ignored.txt"), []).unwrap();

        let paths = record_paths(&dir, Some(1)).unwrap();

        assert_eq!(paths, vec![dir.join("a.json")]);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
