//! Standalone throughput benchmark for `RlSession::reset`/`step` with NO
//! policy network involved. Measures raw engine ticks/s and steps/s,
//! single-threaded and multi-threaded (one OS thread per shard of
//! sessions), plus a coarse component breakdown of `step()`'s cost
//! (`execute_next_tick` vs. obs head construction vs. tile-frame encode).
//!
//! Usage:
//!   cargo run --release -p openfront-engine --example bench_rl_session -- \
//!     --envs 64 --steps 200 --ticks 10 --threads 8 \
//!     --map BlackSea --bots 30 --nations 8 --difficulty Medium
//!
//! `--threads 1` (the default) runs everything on the calling thread.
//! Pass `--profile` to additionally print a per-component time breakdown
//! (averaged over one session) instead of/alongside the throughput run.

use openfront_engine::obs;
use openfront_engine::rl::RlSession;
use openfront_engine::session::AGENT_CLIENT_ID;
use serde_json::Value;
use std::path::PathBuf;
use std::time::{Duration, Instant};

struct Args {
    envs: usize,
    steps: usize,
    ticks: u32,
    threads: usize,
    map: String,
    bots: u32,
    nations: u32,
    difficulty: String,
    profile: bool,
    intents_per_step: usize,
    /// "threads" (default): static shards, one std::thread per shard, each
    /// stepping its sessions sequentially. "rayon": all sessions in one
    /// flat rayon pool, re-scheduled every step (requires building with
    /// `--features parallel`; see `rl_batch::step_many_parallel`).
    mode: String,
    /// Matches the very first (pre-fix) version of this benchmark, which
    /// reset every session inside the timed region. Off by default (the
    /// current, more representative-of-training behavior: reset happens
    /// once up front, outside the clock); pass `--include-reset` to
    /// reproduce numbers comparable to that original methodology.
    include_reset: bool,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            envs: 64,
            steps: 200,
            ticks: 10,
            threads: 1,
            map: "BlackSea".into(),
            bots: 30,
            nations: 8,
            difficulty: "Medium".into(),
            profile: false,
            intents_per_step: 0,
            mode: "threads".into(),
            include_reset: false,
        }
    }
}

fn parse_args() -> Args {
    let mut a = Args::default();
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        let flag = argv[i].as_str();
        let mut next = || {
            i += 1;
            argv.get(i).cloned().unwrap_or_default()
        };
        match flag {
            "--envs" => a.envs = next().parse().expect("--envs"),
            "--steps" => a.steps = next().parse().expect("--steps"),
            "--ticks" => a.ticks = next().parse().expect("--ticks"),
            "--threads" => a.threads = next().parse().expect("--threads"),
            "--map" => a.map = next(),
            "--bots" => a.bots = next().parse().expect("--bots"),
            "--nations" => a.nations = next().parse().expect("--nations"),
            "--difficulty" => a.difficulty = next(),
            "--profile" => a.profile = true,
            "--intents-per-step" => a.intents_per_step = next().parse().expect("--intents-per-step"),
            "--mode" => a.mode = next(),
            "--include-reset" => a.include_reset = true,
            other => {
                eprintln!("unknown flag {other}, ignoring");
            }
        }
        i += 1;
    }
    a
}

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is rust/engine; repo root is two dirs up (same
    // pattern as oftrain's bridge::repo_root, which is one dir further in).
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("engine crate manifest dir has a grandparent")
        .to_path_buf()
}

fn reset_session(root: &PathBuf, args: &Args, idx: usize) -> RlSession {
    let seed = format!("bench{idx}-{}", args.envs);
    let (session, _head, _ents, _legal, _terrain) = RlSession::reset(
        root,
        &args.map,
        &seed,
        args.bots,
        &args.difficulty,
        Value::from(args.nations),
    )
    .expect("RlSession::reset failed");
    session
}

/// No-op/synthetic intents for the raw-throughput loop. Empty by default
/// (pure engine-tick cost); `--intents-per-step N` exercises the
/// `count_wasted` + `turn_to_executions` path with cheap-to-reject
/// `build_unit` intents (never valid this early, so they're all "wasted"
/// but still walk the full per-intent check).
fn synth_intents(n: usize) -> Vec<Value> {
    (0..n)
        .map(|_| {
            serde_json::json!({
                "type": "build_unit",
                "unit": "City",
                "tile": 0,
            })
        })
        .collect()
}

struct RunStats {
    total_ticks: u64,
    total_steps: u64,
    elapsed: Duration,
}

impl RunStats {
    fn ticks_per_sec(&self) -> f64 {
        self.total_ticks as f64 / self.elapsed.as_secs_f64()
    }
    fn steps_per_sec(&self) -> f64 {
        self.total_steps as f64 / self.elapsed.as_secs_f64()
    }
}

/// Steps an already-reset shard of sessions for `args.steps` steps. No
/// resets happen in here - callers reset every session up front, outside
/// the timed region, so this measures steady-state step throughput the
/// way a training loop (which amortizes reset cost over many steps per
/// episode) actually experiences it.
fn run_shard(sessions: &mut [RlSession], args: &Args) -> (u64, u64) {
    let intents = synth_intents(args.intents_per_step);
    let mut total_ticks = 0u64;
    let mut total_steps = 0u64;
    for _ in 0..args.steps {
        for session in sessions.iter_mut() {
            let _ = session.step(&intents, args.ticks);
            // Touch the tile state the way the native trainer backend
            // does post-step, so this loop still pays for exactly the
            // work `NativeEngine::step` does (no more, no less).
            std::hint::black_box(session.tile_state());
            total_ticks += args.ticks as u64;
            total_steps += 1;
        }
    }
    (total_ticks, total_steps)
}

#[cfg(feature = "parallel")]
fn run_rayon(sessions: &mut [RlSession], args: &Args) -> (u64, u64) {
    let intents: Vec<Vec<Value>> =
        (0..sessions.len()).map(|_| synth_intents(args.intents_per_step)).collect();
    let mut total_ticks = 0u64;
    let mut total_steps = 0u64;
    for _ in 0..args.steps {
        let heads = openfront_engine::rl_batch::step_many_parallel(sessions, &intents, args.ticks);
        std::hint::black_box(&heads);
        total_ticks += args.ticks as u64 * sessions.len() as u64;
        total_steps += sessions.len() as u64;
    }
    (total_ticks, total_steps)
}

#[cfg(not(feature = "parallel"))]
fn run_rayon(_sessions: &mut [RlSession], _args: &Args) -> (u64, u64) {
    panic!("--mode rayon requires building with --features parallel");
}

fn run_throughput(args: &Args) -> RunStats {
    let root = repo_root();
    let threads = args.threads.max(1).min(args.envs.max(1));
    let shards: Vec<Vec<usize>> = {
        let mut shards = vec![Vec::new(); threads];
        for i in 0..args.envs {
            shards[i % threads].push(i);
        }
        shards
    };
    let reset_shard = |shard: &[usize]| -> Vec<RlSession> {
        shard.iter().map(|&i| reset_session(&root, args, i)).collect()
    };

    if args.include_reset {
        // Old (pre-fix) methodology: reset happens inside the timed
        // region, so these numbers are directly comparable to the very
        // first baseline runs in this task's report.
        if args.mode == "rayon" {
            let start = Instant::now();
            let mut sessions: Vec<RlSession> =
                (0..args.envs).map(|i| reset_session(&root, args, i)).collect();
            let (total_ticks, total_steps) = run_rayon(&mut sessions, args);
            return RunStats { total_ticks, total_steps, elapsed: start.elapsed() };
        }
        let start = Instant::now();
        let (total_ticks, total_steps) = if threads <= 1 {
            let mut sessions = reset_shard(&shards[0]);
            run_shard(&mut sessions, args)
        } else {
            std::thread::scope(|scope| {
                let handles: Vec<_> = shards
                    .iter()
                    .map(|shard| {
                        scope.spawn(|| {
                            let mut sessions = reset_shard(shard);
                            run_shard(&mut sessions, args)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .fold((0u64, 0u64), |(at, as_), (t, s)| (at + t, as_ + s))
            })
        };
        return RunStats { total_ticks, total_steps, elapsed: start.elapsed() };
    }

    // Default: reset every session (parallelized across the same shards
    // we'll step with) before starting the clock - this is what a
    // training loop actually experiences, since reset cost is amortized
    // over many steps per episode rather than paid every step.
    let mut shard_sessions: Vec<Vec<RlSession>> = if threads <= 1 {
        vec![reset_shard(&shards[0])]
    } else {
        std::thread::scope(|scope| {
            shards
                .iter()
                .map(|shard| scope.spawn(|| reset_shard(shard)))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().unwrap())
                .collect()
        })
    };

    if args.mode == "rayon" {
        let mut all_sessions: Vec<RlSession> = shard_sessions.into_iter().flatten().collect();
        let start = Instant::now();
        let (total_ticks, total_steps) = run_rayon(&mut all_sessions, args);
        return RunStats { total_ticks, total_steps, elapsed: start.elapsed() };
    }

    let start = Instant::now();
    let (total_ticks, total_steps) = if threads <= 1 {
        run_shard(&mut shard_sessions[0], args)
    } else {
        std::thread::scope(|scope| {
            let handles: Vec<_> = shard_sessions
                .iter_mut()
                .map(|sessions| scope.spawn(|| run_shard(sessions, args)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap())
                .fold((0u64, 0u64), |(at, as_), (t, s)| (at + t, as_ + s))
        })
    };
    let elapsed = start.elapsed();
    RunStats { total_ticks, total_steps, elapsed }
}

fn run_profile(args: &Args) {
    let root = repo_root();
    let mut session = reset_session(&root, args, 0);
    let intents = synth_intents(args.intents_per_step);
    let iters = args.steps.max(1);

    let mut t_tick = Duration::ZERO;
    let mut t_entities = Duration::ZERO;
    let mut t_legality = Duration::ZERO;
    let mut t_tiles = Duration::ZERO;
    let mut t_full_step = Duration::ZERO;

    for _ in 0..iters {
        // Full step() timing (what the trainer actually pays per call:
        // count_wasted + turn_to_executions + execute_next_tick*ticks +
        // build_obs_head, i.e. entities()+legality() - tile bytes are
        // NOT part of step() any more, see tile_state()'s doc comment).
        let t0 = Instant::now();
        let _ = session.step(&intents, args.ticks);
        std::hint::black_box(session.tile_state());
        t_full_step += t0.elapsed();

        // Component breakdown re-derived from post-step state (same cost
        // shape as what build_obs_head just did inside step(), so this
        // attributes the step() wall-time above without needing private
        // hooks into RlSession internals).
        let t1 = Instant::now();
        let _ = std::hint::black_box(obs::entities(&session.game));
        t_entities += t1.elapsed();

        let t2 = Instant::now();
        let _ = std::hint::black_box(obs::legality(&session.game, AGENT_CLIENT_ID));
        t_legality += t2.elapsed();

        // obs::tile_bytes_le is no longer called by step()/reset() - the
        // native engine reads tile_state() (a zero-copy &[u16] borrow)
        // directly. Timed here only for comparison against the old
        // architecture and because it's still the encoding the Node
        // bridge/daemon backends need (out-of-process, so they have no
        // choice but to serialize).
        let t3 = Instant::now();
        let _ = std::hint::black_box(obs::tile_bytes_le(&session.game));
        t_tiles += t3.elapsed();
    }
    // Ticking cost = full step - (entities + legality), since step() now
    // does exactly one tick-loop + one head-build per call and we just
    // re-timed the head-build's two components above.
    if t_full_step > t_entities + t_legality {
        t_tick = t_full_step - t_entities - t_legality;
    }

    let per = |d: Duration| d.as_secs_f64() * 1e6 / iters as f64;
    println!("--- profile ({iters} steps, {} ticks/step, {} players) ---",
        args.ticks, session.game.all_players().len() + 1);
    println!("full step() [no tile bytes]:  {:>9.1} us/step (100.0%)", per(t_full_step));
    println!("  execute_next_tick*:          {:>9.1} us/step ({:>5.1}%)", per(t_tick), 100.0 * t_tick.as_secs_f64() / t_full_step.as_secs_f64());
    println!("  obs::entities:               {:>9.1} us/step ({:>5.1}%)", per(t_entities), 100.0 * t_entities.as_secs_f64() / t_full_step.as_secs_f64());
    println!("  obs::legality:               {:>9.1} us/step ({:>5.1}%)", per(t_legality), 100.0 * t_legality.as_secs_f64() / t_full_step.as_secs_f64());
    println!("(* execute_next_tick bucket = full step() minus the two measured components; includes count_wasted + turn_to_executions, both ~0 for {} intents/step)", args.intents_per_step);
    println!(
        "  [for reference only, NOT part of step() any more] obs::tile_bytes_le: {:>9.1} us/step ({:.1}% of old full step cost) - only the Node bridge/daemon backends still pay this",
        per(t_tiles), 100.0 * t_tiles.as_secs_f64() / (t_full_step.as_secs_f64() + t_tiles.as_secs_f64()),
    );
}

fn main() {
    let args = parse_args();
    println!(
        "bench_rl_session: envs={} steps={} ticks/step={} threads={} mode={} map={} bots={} nations={} difficulty={} intents/step={}",
        args.envs, args.steps, args.ticks, args.threads, args.mode, args.map, args.bots, args.nations, args.difficulty, args.intents_per_step,
    );

    if args.profile {
        run_profile(&args);
        println!();
    }

    let stats = run_throughput(&args);
    println!(
        "throughput: {:.0} ticks/s, {:.0} steps/s over {:.2}s ({} total ticks, {} total steps, {} threads)",
        stats.ticks_per_sec(),
        stats.steps_per_sec(),
        stats.elapsed.as_secs_f64(),
        stats.total_ticks,
        stats.total_steps,
        args.threads.max(1).min(args.envs.max(1)),
    );
}
