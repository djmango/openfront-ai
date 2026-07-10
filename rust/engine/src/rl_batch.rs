//! Optional in-process batched stepping for many [`RlSession`]s at once
//! (`--features parallel`, off by default). Steps every session in a
//! batch via `rayon`'s work-stealing thread pool instead of the trainer
//! pinning one OS thread per env for its whole lifetime.
//!
//! This is deliberately *not* wired into `oftrain`'s `VecEnv`/`EnvWorker`
//! (rust/oftrain/src/vecenv.rs) - that already gets full multi-core
//! parallelism from its existing one-thread-per-env design (each
//! `RlSession` is a self-contained, independently-seeded simulation with
//! no shared mutable state, so OS threads and rayon tasks are equally
//! valid ways to run many of them concurrently; `bench_rl_session`
//! benchmarks both and they scale about the same on this workload, since
//! it's memory-bandwidth-bound rather than scheduling-bound). Swapping
//! `VecEnv`'s threading model is a bigger, riskier change than this task
//! calls for. This module exists so that option is available, benchmarked,
//! and one call away if a future workstream wants dynamic load balancing
//! across envs with uneven per-step cost (e.g. episode resets landing on
//! different sessions at different times) instead of a static
//! thread-per-shard split.
//!
//! Determinism/correctness: each `RlSession` only ever touches its own
//! `Game` and `PseudoRandom` state - nothing here shares mutable state
//! across sessions - so stepping them via `par_iter_mut` instead of a
//! sequential loop (or separate OS threads) cannot change any individual
//! session's outcome.

use crate::rl::RlSession;
use rayon::prelude::*;
use serde_json::Value;

/// Steps every session in `sessions` with its corresponding entry in
/// `intents`, in parallel across a rayon thread pool. Returns one obs head
/// per session, same order as the input. Panics if `sessions.len() !=
/// intents.len()` (same contract as zipping two equal-length slices).
pub fn step_many_parallel(
    sessions: &mut [RlSession],
    intents: &[Vec<Value>],
    ticks: u32,
) -> Vec<Value> {
    assert_eq!(
        sessions.len(),
        intents.len(),
        "step_many_parallel: sessions/intents length mismatch"
    );
    sessions
        .par_iter_mut()
        .zip(intents.par_iter())
        .map(|(session, ints)| session.step(ints, ticks))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn repo_root() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .unwrap()
            .to_path_buf()
    }

    /// Stepping the same seeds through `step_many_parallel` vs. a plain
    /// sequential loop must produce byte-identical obs heads: each session
    /// is independent, so parallel scheduling must not perturb any single
    /// session's own deterministic tick/PRNG sequence.
    #[test]
    fn parallel_matches_sequential() {
        let root = repo_root();
        let mut par_sessions: Vec<RlSession> = (0..4)
            .map(|i| {
                let seed = format!("batch-test-{i}");
                RlSession::reset(&root, "Onion", &seed, 0, "Easy", Value::from(3))
                    .unwrap()
                    .0
            })
            .collect();
        let mut seq_sessions: Vec<RlSession> = (0..4)
            .map(|i| {
                let seed = format!("batch-test-{i}");
                RlSession::reset(&root, "Onion", &seed, 0, "Easy", Value::from(3))
                    .unwrap()
                    .0
            })
            .collect();

        let intents: Vec<Vec<Value>> = vec![Vec::new(); 4];
        let par_heads = step_many_parallel(&mut par_sessions, &intents, 5);
        let seq_heads: Vec<Value> = seq_sessions
            .iter_mut()
            .map(|s| s.step(&[], 5))
            .collect();

        assert_eq!(par_heads, seq_heads);
        for (p, s) in par_sessions.iter().zip(seq_sessions.iter()) {
            assert_eq!(p.tile_state(), s.tile_state());
        }
    }
}
