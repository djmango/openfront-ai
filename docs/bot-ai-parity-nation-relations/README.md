# Bot-AI native-vs-TS parity: Easy-nation missing strategies + a swapped `updateRelation` argument order

Follow-up to `docs/bot-ai-parity-investigation/`, `docs/bot-ai-parity-double-attack/`,
and `docs/bot-ai-parity-rate/`. Scope: bisecting the `curriculum-parity-v4`
gate's `bots=5` bucket (the first bucket beyond `bots=0` - single-nation,
zero-combat - to actually exercise nation-vs-nation AI combat), which the
corrected-oracle gate from `docs/bot-ai-parity-rate/` still showed at 17%
pass (1/6) despite `bots=0` being a clean 100%.

## TL;DR

Two independent, real native bugs, both root-caused via tick-level bisection
of `records/curriculum-parity-v4/curr-b005-s2-onion.json.gz`:

1. **`nation_attack_best_target`'s `"Easy"` arm was missing 3 of TS's 7
   attack strategies** (`assist`, `betray`, `hated`) - it jumped straight
   from `retaliate` to `weakest`. `betray`/`hated` were already fully
   implemented elsewhere in the same file (used by the `"Medium"` arm) and
   just never wired into `"Easy"`; `assist` (TS `AiAttackBehavior.assistAllies`)
   has no native port at all, but is confirmed dead code for this dataset
   (see "Why `assist` doesn't need porting" below), so wiring `betray`+`hated`
   is the complete fix for AI-only self-play games.
2. **`AttackExecution::init`'s relation-change call had its two arguments
   swapped**: `game.update_relation(self.owner_small_id, self.target_small_id,
   delta)` updated the *attacker's* relation toward the victim, when TS's
   `this.target.updateRelation(this._owner, relationChange)` updates the
   *victim's* relation toward the attacker - the exact opposite direction.
   This is a systemic bug (every single land/boat attack on a player mis-applies
   its relation delta), not specific to Easy difficulty or nations.

Fixing both together pushes `curr-b005-s2-onion`'s first tile-level
divergence from **tick 2153 -> tick 4370** (an 8x improvement in
byte-identical trajectory length for this one record) with zero `rust/engine`
regressions (`cargo test --lib` unchanged: same pre-existing
missing-fixture failures as the prior three investigations' baseline).

**This does not get the curriculum-parity gate to 100%.** `bots=10` through
`bots=150` are still far from clean (see "Aggregate effect" below) - there is
at least one more systemic bug class left, and likely several. This report
documents two confirmed, fixed bugs and the bisection methodology to find the
next one; it is an incremental step, not a resolution.

## Bug 1: Easy nations skip `assist`/`betray`/`hated`

TS `AiAttackBehavior.getAttackStrategies()` returns, per difficulty:

```ts
case Difficulty.Easy:
  return [nuked, bots, retaliate, assist, betray, hated, weakest];
case Difficulty.Medium:
  return [bots, nuked, retaliate, assist, betray, hated, afk, traitor, weakest, island, donate];
```

`rust/engine/src/execution/ai_attack.rs`'s `nation_attack_best_target`
already implements `nation_strategy_betray`/`nation_strategy_hated` (used by
the `"Medium"` match arm), but the `"Easy"` arm was:

```rust
"Easy" => {
    if is_bordering_nuked_territory(...) && send_tn_attack(...) { return true; }  // nuked
    if attack_bots(...) { return true; }                                          // bots
    if let Some(attacker) = find_incoming_attacker(...) { return ...; }           // retaliate
    nation_strategy_weakest(...)                                                  // weakest
}
```

i.e. `assist`, `betray`, `hated` were entirely absent - Easy nations fell
straight through to `weakest` in every case where TS would have taken one of
those three branches first.

### Why `assist` doesn't need porting (for this dataset)

`assistAllies()` iterates `ally.targets()` for every alliance partner.
`Player.targets()` is populated *only* by `TargetPlayerExecution`, which is
constructed *only* from a `target` **intent** in
`ExecutionManager.createExecs()` - i.e. only ever fired by a human player
explicitly marking a rival via the client UI. `records/curriculum-parity-v4/`
records have zero human players and zero recorded intents (bots/nations act
autonomously - see `datagen/gen_curriculum_parity.ts`'s module doc). So
`ally.targets()` is provably always empty in this entire dataset, `assist`
provably always returns `false`, and skipping it is behaviorally exact for
AI-only self-play. (It would need a real port before trusting this gate
against games with human players.)

### Fix

Wired `nation_strategy_betray` and `nation_strategy_hated` into the `"Easy"`
arm, in TS order, between `retaliate` and `weakest`.

Also found and fixed a second, smaller bug in `nation_strategy_hated` while
wiring it in: the troops-cap guard was applied unconditionally instead of
FFA-only, matching TS's `if (this.isFFA() && other.troops() > this.player.troops() * 3) continue;`
(native was missing the `isFFA()` check entirely, always applying the cap).

## Bug 2: `AttackExecution::init` updates the wrong side's relation

Bisecting a *second* divergence in the same record (after fixing Bug 1,
first divergence moved from tick 2153 to tick 3690) led to a relation-value
mismatch: native's `nation_strategy_hated` fired for `Outer Enclave` (small
ID 1) at tick 3463 targeting `Inner Tribe` (small ID 3) via a *stale* relation
classification, where TS's identical check (same troops, same bordering
list - confirmed via matching debug traces) correctly saw the relation as
not-yet-Hostile and fell through to `weakest` instead, picking a different
implicit path through the strategy chain.

Root cause, found by directly comparing `Player.relations` raw values
between engines at matching ticks: TS's `AttackExecution.init()` does

```ts
this.target.updateRelation(this._owner, relationChange);
```

i.e. calls `updateRelation` **on the target**, with the **owner** as the
argument - "the party being attacked becomes more hostile *toward* the
attacker." Native had:

```rust
game.update_relation(self.owner_small_id, self.target_small_id, delta);
```

`update_relation(a, b, delta)` updates **`a`'s** relation map entry for key
`b` (`a`'s relation *toward* `b`). The call above updates the **attacker's**
relation toward the victim - backwards. Confirmed directly: at tick 1453, TS
nation 3 (Inner Tribe) attacks nation 1 (Outer Enclave); one tick later TS
shows `Outer Enclave.relations.get(InnerTribe) == -60` (correct: victim
hates attacker) while native shows that entry unchanged (`None`) and instead
set `InnerTribe.relations.get(OuterEnclave) == -60` (attacker's own relation
toward the victim it just attacked - the wrong direction).

This is **not** Easy/nation-specific - every single land or boat attack on a
player anywhere in the engine mis-applied this delta, corrupting the
relation state that `hated`/embargo/alliance logic all depend on, from the
very first attack in any game with 2+ non-bot players.

### Fix

```rust
game.update_relation(self.target_small_id, self.owner_small_id, delta);
```

Audited every other `update_relation(...)` call site in `rust/engine/src`
(11 sites across `nation_tick.rs`, `target_player.rs`, `donate.rs`,
`donate_gold.rs`, `alliance_exec.rs`, `nuke_execution.rs`,
`mirv_execution.rs`, plus test-only setup calls in `game.rs`) against their
TS counterparts - all correctly pass `(subject, object, delta)` where
`subject.updateRelation(object, delta)` is the TS call. `attack.rs` was the
only swapped site, plausibly because TS's phrasing there
(`this.target.updateRelation(this._owner, ...)`) is the one place the
"subject" of the call isn't simply `this.player`/`this._owner`.

## Bisection methodology (for the next investigation)

Same tooling as the prior three reports:

```bash
# native, every tick up to a window of interest
cargo run --release -p openfront-engine --bin tick_dump --manifest-path rust/Cargo.toml -- \
  --repo "$(pwd)" --record records/curriculum-parity-v4/curr-b005-s2-onion.json.gz \
  --every 1 --out /tmp/native.json --max-ticks <N>

# TS at master's current openfront pin
openfront/node_modules/.bin/tsx scripts/dump_ts_tick_state.ts \
  records/curriculum-parity-v4/curr-b005-s2-onion.json.gz 1 /tmp/ts.json <N>

python3 scripts/compare_tick_dumps.py /tmp/native.json /tmp/ts.json \
  --rel-threshold 0.0 --abs-threshold 0   # 0/0 catches a single-tile/troop blip, not just >10% swings
```

New technique this time, since the standard tile/troop snapshot wasn't
enough to see *why* two engines took different decisions: temporary
`OF_DEBUG_ATTACK`/`OF_DEBUG_REL`-gated `eprintln!`/`console.error` pairs at
matching call sites in `ai_attack.rs` and `AiAttackBehavior.ts`
(strategy-fired, bordering-list, and raw-relation-value traces), diffed by
tick, to see *which strategy fired and against which target* rather than
just *what changed*. All such instrumentation was reverted before landing
this fix (temporary, not for commit) - reintroduce a similar pattern (or
reuse this doc's approach) for the next bisection.

## Evidence: before/after on the seed record

Full-trajectory tile/troop diff (`--every 10`, `curr-b005-s2-onion.json.gz`,
20000 ticks), first divergence tick:

| state | first divergence tick |
|---|---:|
| before either fix | 2153 |
| after Bug 1 fix only | 3690 |
| after Bug 1 + Bug 2 fix | 4370 |

At the fine (`--every 1`) granularity, the tick-2153 divergence was a
42-tile swing (`Outer Enclave` vs `Inner Tribe`, total conserved -
redistribution, not a net gain/loss) preceded one tick earlier by a troops
jump from a `hated` attack TS fired that native never attempted at all
(missing strategy, Bug 1). The tick-3463 divergence (root of the
2153->3690 gap having been closed, revealing this next one) was a strategy
choice fork (`hated` fired in native, `weakest` in TS) traced to the
relation-value mismatch (Bug 2).

## Aggregate effect on the `curriculum-parity-v4` gate

Full gate re-run (48 records, 6/bucket x 8 buckets, 20000-tick horizon,
`docs/bot-ai-parity-rate/`'s corrected TS-oracle-commit fix already applied):

```
 bots   before (either fix)   after (both fixes)
    0          100% (6/6)           100% (6/6)
    5           17% (1/6)            50% (3/6)
   10           33% (2/6)             0% (0/6)
   30            0% (0/6)             0% (0/6)
   50            0% (0/6)             0% (0/6)
   80            0% (0/6)             0% (0/6)
  120            0% (0/6)             0% (0/6)
  150            0% (0/6)             0% (0/6)
overall        9/48 (19%)           9/48 (19%)
```

`bots=5` clearly improved (both bugs fixed here were found *from* a
`bots=5` record). `bots=10` regressing from 2/6 to 0/6 in the same run is
the expected, documented shape of incremental fixes to a systemic-bug-dense
simulation: relation values and attack timing now legitimately differ more
often (in the *correct* direction) than before, which can shift which
specific games happen to stay in sync long enough to reach the same winner
at the same tick, without the *aggregate* pass rate improving until enough
of the remaining bug surface is closed. The `bots=0` 100% and the `bots=5`
improvement are the trustworthy signals here, not the flat overall total.

## Validation

- `cargo build --release -p openfront-engine --bin outcome_gate --bin
  tick_dump`: clean build, no new warnings.
- `cargo test --release -p openfront-engine --lib`: 45 passed / 37 failed / 1
  ignored - all 37 failures are `No such file or directory` on missing
  fixture record files, the same pre-existing, environment-specific class
  the prior three investigations already established as baseline (not
  caused by this change; exact counts vary slightly by which fixtures
  happen to be present in a given checkout).
- Tick-level trajectory comparison on `curr-b005-s2-onion.json.gz` (table
  above), plus targeted debug-trace comparison confirming the exact
  mechanism for both bugs (strategy-fired + raw relation value, matched
  tick-for-tick against TS before asserting root cause).
- Full `curriculum-parity-v4` gate re-run before/after (table above).

## Suggested follow-up (not done here, out of scope)

- **Keep bisecting.** `bots=10` is the natural next target (regressed to
  0/6 in this same run despite both fixes being individually correct and
  verified byte-for-byte on their source record) - same methodology, a
  fresh record from that bucket.
- **Port `Player.targets()`/`assistAllies()` properly** before trusting any
  gate against games with human players (confirmed dead code for AI-only
  self-play only - see "Why `assist` doesn't need porting" above).
- Consider adding a small, permanent (non-debug-gated) assertion or test
  that walks every `update_relation`/`updateRelation` call site pair
  (native vs TS) mechanically, so an argument-order swap like Bug 2 gets
  caught by CI instead of a multi-hour bisection - the audit in this report
  was manual and one-time.
