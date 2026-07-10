# Bot-AI native-vs-TS parity: the bots=0 "native wins 30-40% faster" rate divergence

Branch: `agent/bot-ai-parity-rate` (isolated worktree off `master` @ `532a321`).
Follow-up to `docs/bot-ai-parity-investigation/` and
`docs/bot-ai-parity-double-attack/`. Scope: a new finding from
`scripts/run_curriculum_parity_gate.sh` (curriculum-representative,
`bots=0`/single-nation/Onion-map bucket): native consistently reaches its
80%-land-share win condition ~25-40% *faster* (fewer ticks) than TS, even in
the simplest possible scenario (one AI nation, zero bots, zero combat,
zero PRNG-target-selection complexity), with a striking "tile count goes
completely flat for ~700 ticks while troop counts oscillate wildly" pattern
in the middle of the game on both sides.

## TL;DR

**Root-caused. Not an engine bug - a stale test-oracle-commit bug in the
parity tooling itself.** `scripts/run_curriculum_parity_gate.sh` calls the
shared `scripts/ensure_parity_openfront.sh`, which pins the `openfront/`
submodule checkout to `PARITY_COMMIT` (default `0c4c7d7993c9`) before
generating/oracling records. That default is correct and necessary for
`run_outcome_gate.sh`'s 78-record **archived real-game** dataset (those
games' recorded human/bot intents are tied to the exact TS commit live when
they were captured), but `run_curriculum_parity_gate.sh`'s records are
**freshly synthesized self-play, with no archival tie to any commit at
all** - reusing the same frozen pin for them was a bug, not a design
requirement.

`0c4c7d7993c9` predates upstream `openfront` commit `22d5aba5a` ("refactor:
standardize cardinal-neighbor iteration on `neighbors()` N,S,W,E order",
PR #4495, merged 2026-07-03), which is the **TS-side counterpart** of the
exact bug `docs/bot-ai-parity-investigation/` found and fixed in native
back in `rust/engine/src/map.rs`'s `for_each_neighbor4`. Before that TS PR,
`neighbors4`/`forEachNeighbor` (used by `AttackExecution.addNeighbors`,
among others) visited in W,E,N,S order while `neighbors()` visited N,S,W,E -
an internal TS inconsistency. Native's `for_each_neighbor4` was fixed to
N,S,W,E to match `neighbors()`/the *current* TS; TS's own `neighbors4` only
got unified to that same order **one bump later** than the commit
`PARITY_COMMIT` is frozen at. Since `AttackExecution.addNeighbors` draws one
PRNG value per neighbor while building the conquest frontier, the
neighbor-order mismatch shifts every subsequent PRNG draw exactly the same
way `docs/bot-ai-parity-investigation/` originally documented - except this
time the "wrong" side is the frozen oracle commit, not native.

Confirmed by bisecting the 107-commit range between `0c4c7d7993c9` (the
frozen `PARITY_COMMIT`) and `58c536b7a` (master's *actual current*
`openfront` gitlink at the time of this investigation) with the exact same
single-seed reproduction: the divergence's terminal tick flips from `8791`
to `5251` (matching native's `5250` almost exactly) at precisely commit
`22d5aba5a`, and nowhere else in the range.

**Fix**: `scripts/run_curriculum_parity_gate.sh` now overrides
`PARITY_COMMIT` (before calling `ensure_parity_openfront.sh`) to whatever
commit the *current superproject checkout* has pinned for `openfront` (via
`git rev-parse HEAD:openfront`), overridable with `CURRICULUM_PARITY_COMMIT`
for a specific pin. `run_outcome_gate.sh`/`run_parity_gate.sh` (the real
78-record archive gate) are untouched and still correctly use the frozen
`PARITY_COMMIT` default - this fix only changes which commit the
**synthetic, non-archival** curriculum gate uses.

No native (Rust) engine code needed to change: native already matches
current upstream TS behavior for this scenario byte-for-byte (tiles
identical at all 2000 sampled ticks across a 20000-tick game; troops
identical until one harmless, non-compounding ±1 rounding blip at tick
7060 - see "Residual" below, the same class of artifact the double-attack
investigation already documented and decided not worth chasing).

Native-vs-TS win-tick ratio across 6 independent seeds:
**before fix: 0.975-1.679 (mean 1.333, i.e. ~33% average gap, matching the
task's reported 30-40%)**; **after fix: 1.0002 for all 6 seeds** (native and
TS agree on the winning tick to within TS's own 1-tick
`WinCheckExecution` sampling granularity) - well inside the outcome gate's
20% timing tolerance.

## The "flat tiles + oscillating troops" stall, explained

The tick~2700-4000 "native's tiles go completely flat for ~700 ticks while
troops oscillate wildly, then TS shows the same shape" pattern reported in
the task write-up is **not a bug and not evidence of a state-machine
mismatch** - it's `AttackExecution`'s troop-reserve/no-reserve toggling
behavior around structure purchases, replaying **byte-identically** on both
engines once compared against the right TS commit:

```
tick   native_tiles  ts(correct-commit)_tiles  native_troops  ts(correct-commit)_troops
2600          69687                    69687         217896                     217896
2700          74099                    74099         235492                     235492
2800          74214                    74214         395326                     395326
2900          75328                    75328         259668                     259668
3000          76248                    76248         272288                     272288
3100          76571                    76571         854269                     854269
3200          76571                    76571         920553                     920553
3300          76571                    76571         962764                     962764
3400          76571                    76571         988724                     988724
3500          76571                    76571        1004355                    1004355
3600          76571                    76571        1013651                    1013651
3700          76571                    76571         830600                     830600
3800          76571                    76571         905029                     905029
3900          79185                    79185         229137                     229137
4000          88379                    88379         249812                     249812
```

Every single number matches exactly, including the flat run of `76571`
tiles from tick 3100 through 3800 and the wild troop swings
(`395326 -> 259668 -> 854269 -> 920553`, exactly the numbers quoted in the
task) - both engines hit the identical "run out of easy attack targets,
switch to husbanding troops for structures, then resume expansion" sequence
at the identical tick, because they're running the identical decision
logic. The task's original observation compared native against the *stale*
`ts_old` snapshot, which drifted onto a completely different, decorrelated
trajectory from PRNG-draw tick ~320 onward (see "Bisection" below) - so the
apparent "stall shape match between native and TS" in the original
write-up was really "native's correct trajectory happens to pass through
a broadly similar shape to the stale reference's, coincidentally, before
diverging on both raw numbers and pacing."

## Reproducing

The `records/curriculum-parity-v3/curr-b000-s*-onion.json.gz` records
already present in the main `openfront-ai` checkout at the time of this
investigation are used directly below (`gen_curriculum_parity.ts`'s
`bots=0` bucket: `nations: 1`, `difficulty: Easy`, map `Onion`,
`gameType: Public`, `donateGold`/`donateTroops: true`,
`maxTimerValue: 40`). Any freshly generated bots=0/Onion/nations=1 record
via `gen_curriculum_parity.ts --buckets 0` reproduces the same thing.

```bash
# from openfront-ai/ (this worktree)

# 1. Confirm current master's ACTUAL openfront pin:
git rev-parse HEAD:openfront   # 58c536b7a3c528c125890b40b1b213109b8b7014 at investigation time

# 2. Native side (deterministic regardless of openfront submodule state -
#    native doesn't read the TS submodule at all):
cargo run --release -p openfront-engine --bin tick_dump --manifest-path rust/Cargo.toml -- \
  --repo "$(pwd)" --record records/curriculum-parity-v3/curr-b000-s0-onion.json.gz \
  --every 10 --out /tmp/native_curr_s0.json --max-ticks 20000

# 3. TS side at master's ACTUAL current pin (openfront/ checked out at
#    58c536b7a, i.e. whatever `git rev-parse HEAD:openfront` says - do NOT
#    run ensure_parity_openfront.sh here, it will pin to the stale
#    PARITY_COMMIT default and reproduce the bug):
openfront/node_modules/.bin/tsx scripts/dump_ts_tick_state.ts \
  records/curriculum-parity-v3/curr-b000-s0-onion.json.gz 10 /tmp/ts_new_s0.json 20000

# 4. TS side at the STALE PARITY_COMMIT (reproduces the divergence):
cd openfront && git checkout 0c4c7d7993c91bd058af2790c5b9f7b48fa8e90b && cd ..
openfront/node_modules/.bin/tsx scripts/dump_ts_tick_state.ts \
  records/curriculum-parity-v3/curr-b000-s0-onion.json.gz 10 /tmp/ts_old_s0.json 20000
cd openfront && git checkout 58c536b7a3c528c125890b40b1b213109b8b7014 && cd ..   # restore

python3 scripts/compare_tick_dumps.py /tmp/native_curr_s0.json /tmp/ts_new_s0.json \
  --rel-threshold 0.001 --abs-threshold 1 --out /tmp/report_after.json
python3 scripts/compare_tick_dumps.py /tmp/native_curr_s0.json /tmp/ts_old_s0.json \
  --rel-threshold 0.10 --abs-threshold 20 --out /tmp/report_before.json
```

Or, using the fixed gate script end-to-end (regenerates everything,
correctly pinned):

```bash
CURRICULUM_BUCKETS=0 CURRICULUM_GAMES_PER_BUCKET=6 CURRICULUM_TICKS=20000 \
CURRICULUM_MAX_TIMER=40 CURRICULUM_RECORDS_DIR="$(pwd)/records/curriculum-parity-b0" \
CURRICULUM_PARITY_LABEL="curriculum-parity-b0" \
bash scripts/run_curriculum_parity_gate.sh --regenerate
```

## Bisection: which upstream TS commit is responsible

The frozen `PARITY_COMMIT=0c4c7d7993c9` (`scripts/parity_env.sh`) and
master's current `openfront` gitlink (`58c536b7a` at investigation time)
are 107 commits apart. Binary-searching that range by checking out each
midpoint commit into `openfront/` and re-running
`datagen/replay.ts`'s `replayOutcome()` directly against
`curr-b000-s0-onion.json.gz` (terminal tick as the bisection signal: `8791`
before the fix commit, `5251` after) converges in ~7 steps:

| commit | terminalTick (seed 0) |
|---|---:|
| `0c4c7d799` (frozen `PARITY_COMMIT`) | 8791 |
| `71d70dfb0` (~50% through the range) | 8791 |
| `6ff202afb` (~75%) | 8791 |
| `78ef7b56f` (doomsday-clock feature added) | 8791 |
| `be77ab4fc` (`22d5aba5a`'s parent) | 8791 |
| **`22d5aba5a` ("standardize cardinal-neighbor iteration...")** | **5251** |
| `66063d617` (~90%) | 5251 |
| `58c536b7a` (master's current pin) | 5251 |

`22d5aba5a`'s own PR description (openfront upstream, PR #4495) confirms
the mechanism directly:

> That PR (#4494) added `forEachNeighborNSWE` as a third neighbor iterator
> because the existing allocation-free helpers (`forEachNeighbor`,
> `neighbors4`) visit in W,E,N,S order while `neighbors()` visits N,S,W,E -
> and substituting one for the other changes simulation behavior at
> order-sensitive call sites. This PR removes that duplication by
> standardizing on one order everywhere: `forEachNeighbor` and `neighbors4`
> now visit in the same N,S,W,E order as `neighbors()`... Callers of the
> flipped helpers that are order-sensitive now make different (equally
> valid) decisions: `AttackExecution.addNeighbors` - PRNG values are drawn
> per neighbor while building the conquest frontier, so attack expansion
> patterns differ... Game outcomes for a given seed differ from previous
> builds (verified: the 12k-tick reference run ends with 31 players alive
> vs 24 before).

This is, almost word for word, the same mechanism
`docs/bot-ai-parity-investigation/`'s root cause #1 already described for
native's `for_each_neighbor4`: a per-neighbor PRNG draw inside
`AttackExecution`/`attack.rs`'s conquest-frontier construction, where the
*order* neighbors are visited in determines which specific tile gets which
PRNG draw, hence a different priority, hence a different heap-dequeue
order, hence a different tile conquered - compounding every tick from the
very first attack onward. Native was fixed to N,S,W,E specifically to match
TS's `neighbors()`/`neighbors4_ts` (see
`rust/engine/src/map.rs`'s `for_each_neighbor4` doc comment, added by the
first investigation). TS's `AttackExecution.addNeighbors` itself only
switched to that same order in `22d5aba5a`, one `openfront` pin-bump *after*
the commit the curriculum gate's oracle was frozen at. Native was already
correct; the oracle just hadn't caught up.

## Fix

`scripts/run_curriculum_parity_gate.sh`: override `PARITY_COMMIT` (before
`ensure_parity_openfront.sh` pins the `openfront/` checkout) to whatever
commit the current superproject checkout has pinned for `openfront`
(`git -C "$ROOT" rev-parse HEAD:openfront`), instead of inheriting
`parity_env.sh`'s frozen default. This makes the curriculum gate always
compare native against "whatever TS commit master currently says is
authoritative," which is the semantically correct target (native has no
other TS reference to mirror), and needs no manual re-pinning as that
commit advances in the future. `CURRICULUM_PARITY_COMMIT` env var added for
pinning to a specific commit when needed (e.g. reproducing a historical
report). Also added `CURRICULUM_BUCKETS` passthrough (to
`gen_curriculum_parity.ts --buckets`) so a single bot-count bucket can be
regenerated/re-gated in isolation instead of paying for all 8 buckets every
time - used throughout this investigation for fast iteration.

`scripts/run_outcome_gate.sh` and `scripts/run_parity_gate.sh` (the real
78-archived-record gate) are **untouched** - they still correctly use the
frozen `PARITY_COMMIT` default, since those records' recorded intents are
tied to that specific historical TS commit and must stay pinned to it.

No `rust/engine` source changes. Native's neighbor-iteration order
(`for_each_neighbor4`, fixed in the first investigation) already matches
current upstream TS; nothing to change there.

## Before/after evidence across seeds

Native win tick (80% land-share threshold, `--every 10` `tick_dump`
sampling) vs TS terminal tick (`replayOutcome`'s own win-check, exact),
same 6 `curr-b000-s{0..5}-onion.json.gz` records, before fix (TS side
pinned to the stale `0c4c7d7993c9`) vs after fix (TS side at master's
actual current pin, `58c536b7a`):

| seed | native win tick | ts win tick (before) | ratio (before) | ts win tick (after) | ratio (after) |
|---:|---:|---:|---:|---:|---:|
| 0 | 5250 | 8791 | 1.674 | 5251 | 1.0002 |
| 1 | 5010 | 6991 | 1.395 | 5011 | 1.0002 |
| 2 | 5670 | 9521 | 1.679 | 5671 | 1.0002 |
| 3 | 5440 | 6161 | 1.133 | 5441 | 1.0002 |
| 4 | 5740 | 6541 | 1.140 | 5741 | 1.0002 |
| 5 | 5900 | 5751 | 0.975 | 5901 | 1.0002 |
| **mean** | | | **1.333** | | **1.0002** |

("ratio" = ts / native; the task's reported examples - TS 8791 vs native
5250, TS 6991 vs native 5010, TS 9521 vs native 5670 - are seeds 0/1/2
above, reproduced exactly.) Seed 5's `before` ratio (0.975, i.e. native
very slightly *slower*, the one exception to the "native always faster"
pattern noted in the task) is also fully explained by the same stale-commit
mechanism - it isn't a second, independent bug, it's the same
PRNG-draw-order shift landing in native's favor by chance for that one
seed's specific tile layout.

Full-trajectory tile/troop parity after the fix (seed 0, `--every 10` for
2000 ticks up to tick 20000, `compare_tick_dumps.py`-equivalent diff):
**tiles are byte-identical at all 2000 sampled ticks** (0 mismatches);
troops match exactly through tick 7050, then a single harmless ±1 blip
(native `1760598` vs TS `1760599` at tick 7060) that does not compound and
never affects tiles - the same class of non-compounding floating-point
rounding artifact `docs/bot-ai-parity-double-attack/`'s "What's left"
section already documented and declined to chase further.

Official `outcome_gate` before/after (bots=0 bucket, 6 records, using the
gate's own pass/fail categorization rather than a custom script):

```
BEFORE (TS oracle at stale PARITY_COMMIT=0c4c7d7993c9):
 bots   pass  total   rate  avg_tiles_ratio
    0      3      6    50%           1.0000
=== non-pass records ===
bots=0 curr-b000-s0-onion.json.gz  category=timing_mismatch  expected_tick=8791 actual_tick=5250
bots=0 curr-b000-s1-onion.json.gz  category=timing_mismatch  expected_tick=6991 actual_tick=5010
bots=0 curr-b000-s2-onion.json.gz  category=timing_mismatch  expected_tick=9521 actual_tick=5670

AFTER (TS oracle at master's current pin, via the fixed script):
 bots   pass  total   rate  avg_tiles_ratio
    0      6      6   100%           1.0000
```

(Note `avg_tiles_ratio=1.0000` in *both* rows: the outcome gate's tile-ratio
metric is computed from each side's own reported *winner's final land
share*, which was already `1.0` on both sides even before the fix - the
divergence was purely about *when*, not *whether/how much*, exactly as the
task's framing described. `timing_mismatch` is the gate's category for
"same winner, same final land share, wrong tick," which is exactly what a
stale-oracle-commit PRNG-order shift produces.)

## Regression check on other bucket sizes

Bumping the curriculum gate's TS reference commit forward by 107 upstream
commits is a bigger step than a single targeted bugfix, so it's worth
checking it didn't newly *introduce* a divergence somewhere else (e.g. via
the `impassable terrain` or `doomsday-clock` features added in that range,
both of which are no-ops for a solo bots=0 nation but are live for
higher-bot-count buckets). Spot-checked `bots=5` (5 records, mixed
Onion/Pangaea maps, `default` nations) with the fix applied:

```
 bots   pass  total   rate  avg_tiles_ratio
    5      4      5    80%           0.9944
bots=5 curr-b005-s3-pangaea.json.gz  category=land_share_mismatch
  expected_tick=3911 actual_tick=3910 (ticks agree - NOT a timing issue)
  expected_share=0.458 actual_share=0.355
```

This is a **different failure category** (`land_share_mismatch`, not
`timing_mismatch`) with matching winner and matching tick - a residual,
independent divergence of the kind the two prior investigations already
catalogued as "real but not yet root-caused," not a new regression from
this fix. No evidence this fix made anything worse; it only removed the
spurious `bots=0` `timing_mismatch` failures.

## Files changed

- `scripts/run_curriculum_parity_gate.sh`: `PARITY_COMMIT` override (root
  cause fix) + `CURRICULUM_BUCKETS` passthrough (tooling convenience used
  throughout this investigation).
- `docs/bot-ai-parity-rate/`: this report.

No `rust/engine` changes - see TL;DR for why.

## Validation

- `cargo test --release -p openfront-engine --lib`: **47 passed / 35
  failed / 1 ignored**, established as the current `master`-@-`532a321`
  baseline *before* any change in this investigation (identical to the
  prior two investigations' reported baseline), and reconfirmed identical
  *after* the fix - expected, since this fix touches no Rust source at
  all. The 35 failures are pre-existing and unrelated (missing fixture
  record files not present in this isolated worktree).
- Primary validation: `outcome_gate` official pass/fail categorization
  before/after (see above), plus tick-level `tick_dump`/`dump_ts_tick_state`
  trajectory comparison across 6 independent seeds (see "Before/after
  evidence" above) and a bisection of the exact upstream TS commit
  responsible (see "Bisection" above).
- Regression spot-check on the `bots=5` bucket (see above): no new
  divergence introduced by advancing the TS reference commit.

## Suggested follow-up (not done here, out of scope)

- The existing `records/curriculum-parity-v3/` + `.manifest.json` cache in
  the main (non-worktree) `openfront-ai` checkout was generated against the
  stale commit and should be regenerated
  (`scripts/run_curriculum_parity_gate.sh --regenerate`) to get accurate
  pass rates for buckets beyond `bots=0` too.
- The `bots=5`/`curr-b005-s3-pangaea` `land_share_mismatch` noted above (and
  presumably similar cases at other bot counts) is a real, independent,
  not-yet-root-caused divergence - a good candidate for a third follow-up
  investigation in this series, using the same tick_dump bisection
  methodology as the first two.
- `scripts/ensure_parity_openfront.sh`'s hardcoded default
  `PARITY_COMMIT="${PARITY_COMMIT:-0c4c7d7993c9}"` will itself go stale
  again if the frozen archived-record dataset is ever regenerated at a
  newer commit; low priority since that dataset changes rarely, but worth
  a comment pointing at this document if it's ever touched.
