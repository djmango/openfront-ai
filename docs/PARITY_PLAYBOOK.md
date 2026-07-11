# Parity work playbook (read this before dispatching or doing more parity work)

Why this exists: this project's native-vs-TS parity work went through ~40
real bug fixes and several hundred ported tests across many session rounds.
The methodology converged on a few concrete practices that make each round
fast; this doc exists so every future round starts from those practices
instead of re-deriving them (or re-explaining them in a giant subagent
prompt) each time.

**The actual goal is training, not a leaderboard number.** Native exists to
be a fast, correct RL training environment (`rust/oftrain`). Full-game
bit-exact TS parity is a *diagnostic signal* that native has no remaining
bugs, not the end goal itself - a policy trained in native doesn't need
bit-identical outcomes to TS, it needs mechanics that are actually
implemented and behave correctly (a nation that never nukes, or a warship
AI that's a no-op, teaches a policy a degenerate world). Prioritize
accordingly:

1. **Highest priority: functional completeness.** A subsystem that's a
   silent no-op/stub is the worst kind of bug for training - it's not a
   subtle timing error, it's a whole category of dynamics the policy never
   sees. Find these by porting TS unit test files, one subsystem at a time.
2. **Second priority: fixing bugs the completeness pass surfaces.** A
   failing ported test almost always means a real native bug (see "bug
   classes" below); fix it, don't weaken the assertion.
3. **Lower priority (but still valuable, don't skip entirely): full-game
   outcome-parity bisection.** Useful for catching bugs that only manifest
   under compounding interaction, and worth doing periodically to check the
   `curriculum-parity-v4` gate isn't regressing - but chasing the last few
   points of aggregate pass rate has sharply diminishing returns (see
   "why full-game parity is a different beast" below), and none of it
   changes whether the mechanics themselves are correct.

## Tooling

- **`scripts/bisect_parity.sh <record.json.gz> [--max-ticks N] [--coarse-every N] [--fields ...]`**
  One-command tick-level bisection for any full-game divergence (curriculum-
  parity-v4 self-play records, or any other `GameRecord`). Two-pass: coarse
  dump of both engines (native `tick_dump` bin + TS `dump_ts_tick_state.ts`),
  diff to find the first diverging checkpoint window, then a fine (every-tick)
  re-dump *only* up to that window. This replaced writing a bespoke test/
  script per bisection - do not do that anymore, use this instead. On a
  4000-tick window this takes ~10s; scale `--max-ticks` to what you actually
  need (e.g. from `outcome_gate`'s reported terminal ticks + margin), not the
  full 20000-tick horizon, since the TS side is the slow part and its cost is
  proportional to how far both passes replay.
- **`scripts/diff_tick_dumps.py <native.json> <ts.json>`** - the diff half of
  the above, callable standalone if you already have two dumps. Ignores the
  first few ticks by default (`--skip-before`, default 5) - the two dump
  harnesses' own init paths aren't guaranteed to register nation/bot spawns
  on the identical tick, which is init-order noise specific to these two
  dump tools, not a genuine divergence; don't chase it without checking
  `outcome_gate`'s actual replay path agrees there's a real bug.
- **`rust/engine/src/replay.rs`'s `find_first_divergence` test** - for the
  *archived* fixture records (`records/0c4c7d7993c9/*.json.gz`) specifically,
  which have TS-computed hash checkpoints already embedded per-turn. Cheaper
  than the dump-and-diff approach for those because the ground truth is
  already baked into the record - no live TS replay needed at all. Do NOT
  use this for synthetic self-play records (`records/curriculum-parity-v4/`
  etc.) - they have empty `turns[].hash`, there's nothing to compare against
  in-process; use `bisect_parity.sh` for those.
- **`scripts/run_curriculum_parity_gate.sh`** - the full 48-record aggregate
  gate. Slow (~10-15 min). Run this yourself, periodically, to check the
  aggregate trend - never delegate a subagent to run it as part of a
  porting/bisection task, it's irrelevant to and much slower than what those
  tasks need to validate their own work.

## Dispatch checklist (before launching parity subagents)

- [ ] `git worktree list` and `git branch --list 'agent/*'` first - reuse or
  clean up stragglers from a prior round before creating new ones with
  similar names; duplicate/overlapping dispatches waste real merge time.
- [ ] Give each subagent a **disjoint** scope (distinct TS test files, or a
  distinct record to bisect) so merges are structurally independent.
- [ ] Explicitly tell every subagent: no `git stash` (shared across all
  worktrees of one repo - has corrupted concurrent agents' work more than
  once), targeted `cargo test <substring>` while iterating, full `cargo test
  --lib` only once at the end, never run the curriculum gate themselves.
- [ ] Actually call the dispatch tool in the same turn you describe the
  plan. A described-but-undispatched plan looks identical to being stuck.

## Recurring native bug classes (check for these first in any new bisection/port)

Found multiple times independently across this project's history - check
for these before assuming a divergence needs novel investigation:

- Missing/extra `isFFA()`-style mode guards.
- Swapped function arguments (e.g. `update_relation(a, b, delta)` vs
  `(b, a, delta)` - happened at least twice).
- Missing `!isImpassable`/`hasFallout` exclusions on tile candidates.
- `HashSet`/`HashMap` iteration where TS relies on `Set`/`Map` insertion
  order (affects PRNG-draw order or first-match picks) - use this
  codebase's ordered-collection helpers, not a plain hash-based one, in any
  code path where iteration order can affect a subsequent random draw.
  This is the single most common root cause found this project.
- PRNG draw-count/order mismatches from an extra or skipped branch (a
  fabricated fallback path with no TS counterpart is the worst version of
  this - it doesn't just pick a different target, it draws random numbers
  TS never draws, desyncing every subsequent decision for the rest of the
  game).
- A subsystem that looks implemented but is actually a stub that only
  consumes a PRNG draw and returns early - always check the *return value*
  and *side effects* of a suspicious-looking function, not just that it
  exists and has a plausible-looking signature.

## Why full-game parity is a different beast (context, not a call to stop trying)

Every tick, dozens of PRNG draws happen in a fixed order. The instant any
single draw diverges, every subsequent draw in that game works from a
different random stream, and the two simulations diverge completely - even
when every individual mechanic involved is correct. This is why fixing
several substantial, real bugs in one round can move the aggregate
`curriculum-parity-v4` pass rate by only a couple of percentage points: the
gate rewards "zero remaining divergences across an entire 20k-tick game",
not "how many bugs got fixed". Don't use a flat aggregate number as the
only signal of progress - a fixed bug that measurably improves one record's
tick-delta (even without flipping its pass/fail) is real progress, and the
unit-test pass count is the more reliable trend to watch round over round.
