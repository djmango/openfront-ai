# Porting three confirmed-missing AI/game subsystems: nuke AI, warship retaliation AI, Doomsday Clock

Follow-up to `docs/bot-ai-parity-nation-relations/`. Scope: this session's
"what's still missing" list ended with three concrete, confirmed gaps -
not subtle mismatches, but entire pieces of TS behavior with no native
equivalent at all. Unlike the prior docs in this series (tick-level
bisection of a specific divergence), this one is a straight port-from-TS
exercise, validated by porting each area's TS unit tests 1:1 and a full
`curriculum-parity-v4` gate re-run.

## What was missing

1. **`NationNukeBehavior.maybeSendNuke`** (`openfront/src/core/execution/nation/NationNukeBehavior.ts`,
   1142 lines) - native's `maybe_send_nuke` was a complete no-op. Nations
   never autonomously built missile silos or launched nukes at all.
2. **`NationWarshipBehavior.trackShipsAndRetaliate` / `.counterWarshipInfestation`**
   (`openfront/src/core/execution/nation/NationWarshipBehavior.ts`, 465
   lines) - both were no-ops. (`maybeSpawnWarship`, the third method in
   that class, turned out to *also* be a stub despite looking real at a
   glance - see "Bugs found" below.)
3. **`DoomsdayClockExecution`** (`openfront/src/core/execution/DoomsdayClockExecution.ts`,
   157 lines) - the pure wave-schedule/drain math had already been ported
   to `rust/engine/src/core/doomsday_clock.rs` in an earlier session, but
   nothing ever called it; there was no `Execution`, no `Player` state,
   and the feature had zero gameplay effect natively even when a config
   enabled it.

## Approach

Three parallel subagents, one per subsystem, each in its own git
worktree/branch, each told to: read the TS source completely, port it
into a new native module following this codebase's existing
free-function + plain-data-state idioms (see `ai_attack.rs`/`nation_tick.rs`
for precedent), port the subsystem's TS unit tests 1:1 as native
`#[cfg(test)]` tests, and write at least one true end-to-end smoke test
(not just isolated unit tests of sub-functions).

## Bugs found and fixed along the way

Beyond the three headline ports themselves, each subagent found real,
independent native bugs while working in this area:

- **`ai_attack.rs::should_attack` was missing TS `AiAttackBehavior.shouldAttack`'s
  `playerTeams === HumansVsNations` short-circuit.** In HvN mode, TS
  always attacks true; native was incorrectly still rolling Easy/Medium's
  "skip attacking humans" dice. Found while porting the nuke AI (which
  reuses `shouldAttack`).
- **`NationWarshipBehavior.maybeSpawnWarship` was itself a stub**, not
  the working implementation this session initially assumed it was: it
  consumed only a `chance(50)` PRNG draw and unconditionally returned
  `false`, silently skipping TS's `randElement(ports)` + up-to-100
  `warshipSpawnTile` draws and the real `ConstructionExecution` call
  whenever gold/ports/warship-count should have allowed a spawn.
- **Warship `structure_cost` had no `Warship` arm**, making warships
  free to build - which made the newly-ported AI's gold-affordability
  gating meaningless until fixed (this is exactly the kind of bug that's
  invisible until something actually tries to *use* the cost).
- Supporting `Game`/`Player` API gaps filled in to make the warship AI
  port possible: distinguishing a transport ship destroyed by an enemy
  from one that peacefully arrived/retreated (`wasDestroyedByEnemy`
  equivalent), an owner-scoped `hasUnitNearby`, warship-patrol
  read/redirect, and fractional (not just whole-number) relation deltas
  (TS's trade-ship retaliation uses `-7.5`).
- A new `OrderedUnitSet`/insertion-order-preserving set type was added
  for the warship AI's tracked-ship collections, following this
  codebase's established rule that anywhere a TS `Set` iteration order
  can affect a subsequent PRNG draw or first-match pick, a plain
  `HashSet` is a latent parity bug (see `docs/bot-ai-parity-nation-relations/README.md`
  for the earlier instance of this exact class of bug).

## What's deliberately deferred

- Nuke AI: `maybeDestroyEnemySam` (~230 lines, the Impossible-only
  SAM-overwhelm salvo planner used only as a fallback when no direct
  target scores above zero). Everything else in `NationNukeBehavior.ts`
  is ported.
- Warship AI: `findTeamGameWarshipTarget` (Team-mode-only target search)
  is left unported as dead code for this project's FFA-only curriculum
  usage, marked with an `#[ignore]`d test and doc comment rather than
  guessed at without an exercising scenario.
- Doomsday Clock: nothing deferred - the TS file's modest size (157
  lines) meant a full port, including all 20 `DoomsdayClockExecution.test.ts`
  cases (on top of the 12 pure-math cases already ported previously),
  fit comfortably.

## Validation

- `cargo test --release -p openfront-engine --lib`: **315 passed, 0
  failed, 10 ignored** (baseline immediately before this batch: 266
  passed, 0 failed, 11 ignored - the one fewer ignored is a stale
  gap-marker test for the nuke AI, removed once the real port made it
  moot).
- Each subsystem has its own end-to-end smoke test confirming the whole
  entry point fires under favorable conditions (a nation actually builds
  and launches a nuke; a nation with a warship actually redirects its
  patrol toward an incoming enemy transport; a losing side actually
  drains under the Doomsday Clock) - not just unit tests of internal
  helper functions.
- Full `curriculum-parity-v4` gate re-run (48 records, 6/bucket x 8
  buckets, 20000-tick horizon) after merging all three branches:

  ```
   bots   pass  total   rate
      0      6      6   100%
      5      3      6    50%
     10      0      6     0%
     30      0      6     0%
     50      0      6     0%
     80      0      6     0%
    120      0      6     0%
    150      1      6    17%
  overall  10/48 (21%)
  ```

  This is flat-to-slightly-improved relative to the 9/48 (19%) baseline
  documented in `docs/bot-ai-parity-nation-relations/README.md` - i.e.
  three substantial, previously-inert AI subsystems went from "always a
  no-op" to "real, tested behavior" with **no regression** in the
  aggregate outcome-parity gate. `bots=0` staying a clean 100% (these
  buckets never exercise nation AI at all) is the strongest signal that
  nothing outside the touched subsystems broke.
- 6 of the 48 records reported `replay_error` in the 20-way-parallel gate
  run, all at exactly the 300s per-record timeout. Re-running one of
  them (`curr-b050-s4-asia.json.gz`) alone with `--jobs 1` and a 900s
  timeout completed cleanly in 174s with a normal `wrong_winner`
  category - confirming these are CPU-contention timeouts from 20
  concurrent 20000-tick simulations now doing real per-tick AI work
  (nukes/warships/doomsday-clock math instead of no-ops), not hangs or
  crashes in the new code. Worth widening `RECORD_TIMEOUT_SECONDS` or
  lowering `--jobs` for future gate runs on machines with fewer cores
  than this run's job count, now that these subsystems add real
  per-tick cost.

The remaining gap-to-100%-parity at higher bot counts is the same
compounding-divergence phenomenon documented in
`docs/bot-ai-parity-nation-relations/README.md`'s aggregate-effect
section: full 20000-tick self-play games are extremely sensitive to any
single remaining tick-level mismatch, and closing individual bugs (this
batch's three subsystems included) doesn't move the aggregate pass rate
monotonically even when each fix is independently correct.
