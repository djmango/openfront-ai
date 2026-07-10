# Bot-AI native-vs-TS parity investigation (moderate bot count, ~50 nations)

Branch: `agent/bot-ai-parity-investigation` (isolated worktree off `master` @
`ce668d2`). Scope: root-cause the native (`rust/engine`) vs TypeScript
(`openfront/`) territorial-outcome divergence in bot/nation-only self-play, at
a moderate bot count (~50 nations, no humans) chosen for fast iteration
(<40s/replay at 6000 ticks, vs minutes for 400-bot mega-games).

## TL;DR

Found and fixed the dominant root cause: **`GameMap::for_each_neighbor4` in
`rust/engine/src/map.rs` iterated a tile's 4 neighbors in the wrong order**
(west, east, north, south) instead of TS's actual order (north, south, west,
east). This function is the hot-path primitive used by border-tile tracking,
attack-candidate collection, and other neighbor scans across ~10 files. Its
own doc comment even claimed the (wrong) order matched TS.

Because several of those call sites draw from a PRNG *while* iterating
neighbors (e.g. one `random.next_int()` per candidate border tile), the wrong
order didn't just reorder tiles: it silently shifted the PRNG stream
position for every subsequent draw in that entity's decision sequence, from
the very first tick attacks went active. This is exactly the kind of "small
per-tick decision difference that compounds" the task description called out.

Fixing this one function moved the first tick-level divergence in our 50-bot
test game from **tick ~303** (the very first tick TerraNullius expansion
attacks activate, right after spawn) to **tick ~1497-1498**, a ~5x delay,
and made every sampled tick in between **byte-identical** between engines
(not just "close": exactly equal tile/troop counts for all ~59 entities from
tick 50 through tick 1450). Total accumulated tile-attribution error at the
6000-tick mark dropped from 756,227 to 410,340 (-46%).

Also fixed 6 smaller, independently-real `is_impassable` filtering gaps in
`ai_attack.rs` and `attack.rs` that brought native's target-selection and
tile-conquest logic into exact alignment with TS's `AiAttackBehavior.ts` /
`AttackExecution.ts`. These didn't move the divergence point on *this*
particular map/seed (no impassable terrain happened to border the affected
nations early on), but one of them (`attack.rs:285`, the conquer-loop) fixes
a real invariant violation: without it, native could actually *own* an
impassable tile, which should never happen.

A second, smaller-scale residual divergence remains, first appearing at tick
~1498 (troop levels, not tiles) for "Spanish Realm", documented below with
evidence but not root-caused to a specific line, given diminishing returns
after the primary fix. See "Remaining divergence" below.

## Tooling built

All new, in `scripts/` and `rust/engine/src/bin/`:

- `scripts/gen_selfplay_record.ts`: synthesizes a bot/nation-only
  `GameRecord` (zero human players, zero recorded turns beyond
  `info.num_turns`) so the entire trajectory is driven by autonomous AI
  decisions on both engines. `gameType: "Public"` (not Singleplayer) so
  `SpawnTimerExecution` ends the spawn phase on a timer like a real archived
  game. Used to generate `records/selfplay/bs50.json.gz`: BlackSea map, 50
  bots via `nations: "default"` (about 59 total entities after nation-count
  resolution), Medium difficulty, seed `bot-ai-parity-1`, 6000 ticks.
- `rust/engine/src/bin/tick_dump.rs`: native binary replaying a `GameRecord`
  via `bootstrap::game_from_record` + `execute_next_tick()`, snapshotting
  every `--every` ticks (or every tick with `--every 1`): per-entity tiles
  owned, troops, gold, alive status, plus totals and spawn-phase flag.
  `cargo run --release --bin tick_dump -- --repo <root> --record <gz> --every
  50 --out <path> --max-ticks N`.
- `scripts/dump_ts_tick_state.ts`: the TS-side counterpart. Replays the same
  record through the real TS engine (mirroring `createGameRunner`'s init
  order: `SpawnTimerExecution`, nation spawns, bot tribes, `WinCheckExecution`,
  rail clusters) and dumps the identical snapshot schema, tick-for-tick
  comparable to `tick_dump`'s output.
- `scripts/compare_tick_dumps.py`: diffs two tick-dump JSONs, reports the
  first tick where any entity's tile count differs by more than
  `--rel-threshold` (relative) *and* `--abs-threshold` (absolute, to ignore
  noise on tiny territories), plus a full per-tick summary table.

Typical iteration loop for this bot count is well under a minute round-trip
(native: ~35s for 6000 ticks / <1s for 400 ticks; TS: ~2-3s for 1300 ticks),
which is what made bisecting the divergence tick-by-tick tractable.

## Reproducing

```bash
# from openfront-ai/ (this worktree)
openfront/node_modules/.bin/tsx scripts/gen_selfplay_record.ts \
  --map BlackSea --bots 50 --nations default --difficulty Medium \
  --seed bot-ai-parity-1 --max-ticks 6000 --out records/selfplay/bs50.json.gz

cargo run --release -p openfront-engine --bin tick_dump -- \
  --repo "$(pwd)" --record records/selfplay/bs50.json.gz --every 50 \
  --out /tmp/native_ticks.json --max-ticks 6000

openfront/node_modules/.bin/tsx scripts/dump_ts_tick_state.ts \
  records/selfplay/bs50.json.gz 50 /tmp/ts_ticks.json 6000

python3 scripts/compare_tick_dumps.py /tmp/native_ticks.json /tmp/ts_ticks.json \
  --rel-threshold 0.10 --abs-threshold 20 --out /tmp/report.json
```

## Root cause #1 (primary): wrong neighbor iteration order

`rust/engine/src/map.rs:277-293` (before fix), `GameMap::for_each_neighbor4`:

```rust
pub fn for_each_neighbor4(&self, t: TileRef, mut f: impl FnMut(TileRef)) {
    let w = self.width;
    let x = self.x(t);
    // TS `GameMap.neighbors4` order: west, east, north, south.
    if x > 0 { f(t - 1); }
    if x + 1 < w { f(t + 1); }
    if t >= w { f(t - w); }
    if t < (self.height - 1) * w { f(t + w); }
}
```

TS's actual order, `openfront/src/core/game/GameMap.ts:393-403`
(`GameMapImpl.neighbors4`):

```ts
neighbors4(ref: TileRef, out: TileRef[]): number {
  const w = this.width_;
  const x = ref % w;
  let n = 0;
  if (ref >= w) out[n++] = ref - w;                      // north
  if (ref < (this.height_ - 1) * w) out[n++] = ref + w;  // south
  if (x !== 0) out[n++] = ref - 1;                        // west
  if (x !== w - 1) out[n++] = ref + 1;                    // east
  return n;
}
```

North/south/west/east: the *opposite* grouping from what
`for_each_neighbor4`'s own comment claimed. The engine actually already has a
*correct* sibling, `neighbors4_ts` (`rust/engine/src/map.rs:295-316`, added in
commit `1a9ce33` for a specific caller in `ai_attack.rs`), but the original,
far more widely used `for_each_neighbor4` (added in the initial engine port,
commit `745df52`) was never corrected or migrated away from. It's used in 10
files: `replay.rs`, `obs.rs`, `port_execution.rs`, `ai_attack.rs`,
`nation_structures.rs`, `attack.rs`, `player_clusters.rs`, `water.rs`,
`warship.rs`, `game.rs`.

### Why this causes drift, concretely

`rust/engine/src/execution/attack.rs`'s `add_neighbors` (the function that
seeds a fresh attack's border-tile queue and per-tile conquest priority) is a
direct TS mirror of `AttackExecution.addNeighbors`
(`openfront/src/core/execution/AttackExecution.ts:335-380`). Both compute, for
each qualifying neighbor tile, a priority using **one PRNG draw per tile**:

```ts
// TS
const priority =
  (this.random.nextInt(0, 7) + 10) * (1 - numOwnedByMe * 0.5 + mag / 2) + tickNow;
```

```rust
// Rust (unchanged by this fix - it's the ORDER neighbors arrive in that mattered)
let priority = (self.random.next_int(0, 7) as f64 + 10.0)
    * (1.0 - num_owned_by_me as f64 * 0.5 + mag as f64 / 2.0) + tick as f64;
```

`self.random` is `PseudoRandom::new(123)`: a **fixed seed, fresh per
`AttackExecution` instance**. Its output at draw index *N* only depends on
*N*, not on global game state. So as long as the *sequence* of tiles fed into
this loop matches between engines, each specific tile gets the same draw and
the same priority, and the priority-ordered heap dequeues tiles in the same
order.

But the loop iterates neighbors via `game.map.for_each_neighbor4(tile, ...)`
(`attack.rs:629`), and until this fix that order didn't match TS's
`this.map.neighbors4(tile, this.nbuf)` order. Every multi-neighbor border
tile could therefore hand out draws to different specific tiles than TS did,
shifting which "random.next_int(0,7)" value each tile received, hence a
different priority, hence a different heap-dequeue order, hence potentially a
different terrain type conquered first in the very same tick
(`attack.rs:203-320`'s `tick()`, where `tiles_used` depends on
`terrain_type(tile_to_conquer)` via `speed`), hence a different number of
tiles conquered before that tick's troop/tile budget (`num_tiles_per_tick`)
was exhausted.

This is exactly the mechanism observed at the very first attack tick in our
test game (tick 303, one tick after spawn ends at tick 302):

```
tick 303  nation:Armenia    native tiles=63  ts tiles=64   (troops identical: 9552 both)
tick 303  nation:Circassia  native tiles=63  ts tiles=64   (troops identical: 9552 both)
```

Troop counts matching exactly while tile counts differ by 1 rules out a
troop-formula bug and points straight at the conquest-order/rate mechanics,
which is exactly what `for_each_neighbor4`'s wrong order affects, without
necessarily changing total troop expenditure per tick.

### Fix

`rust/engine/src/map.rs`: reordered `for_each_neighbor4` to north, south,
west, east (matching TS and `neighbors4_ts` exactly), and corrected its
misleading comment.

### Before/after evidence (50-bot BlackSea self-play, seed `bot-ai-parity-1`)

First divergence (>2% relative, ignoring entities under 5 tiles), scanning
every tick:

| | first divergence tick | detail |
|---|---|---|
| before fix | **303** | Armenia/Circassia off by 1 tile each; troops identical |
| after fix | **1497-1498** | troop-level divergence, "Spanish Realm" (see below); tiles matched exactly through at least tick 1490 |

Coarse (every-50-tick, >10% relative on entities >=20 tiles) first divergence:

| | tick | worst entity |
|---|---|---|
| before fix | 340 | `nation:Kenyan Caliphate` 103 vs 90 tiles (+13%) |
| after fix | 1750 | `nation:Taino Supremacy` 4338 vs 6564 tiles (34%) |

Total accumulated absolute tile error across all ~59 entities, sampled every
50 ticks (full table in `report_before_fix.json` / `report_after_fix.json` in
this directory; select rows below):

| tick | before fix (abs tile error, summed over all entities) | after fix |
|-----:|-----:|-----:|
| 300  | 0 | 0 |
| 350  | 205 | 0 |
| 500  | 414 | 0 |
| 1000 | 3,392 | 0 |
| 1450 | 26,576 | 0 |
| 1500 | 33,923 | 28 |
| 2000 | 144,050 | 60,911 |
| 3000 | 242,619 | 241,264 |
| 4000 | 387,613 | 224,734 |
| 5000 | 585,121 | 312,593 |
| 6000 | 756,227 | 410,340 |

**Every sampled tick from 50 through 1450 is byte-identical after the fix**
(zero total error): the entire spawn phase plus the first ~1150 ticks of
active expansion now replay identically between native and TS at this bot
count. The fix roughly halves the accumulated end-state error by tick 6000
even though it doesn't eliminate all divergence (see below).

## Root cause #2 (smaller, real, but not the dominant driver here): missing `is_impassable` filters

Found while comparing `rust/engine/src/execution/ai_attack.rs` against
`openfront/src/core/execution/utils/AiAttackBehavior.ts`, and
`rust/engine/src/execution/attack.rs` against
`openfront/src/core/execution/AttackExecution.ts`. TS consistently filters
`isImpassable` tiles out of attack-target-selection and conquest candidate
sets; several native call sites didn't:

1. `ai_attack.rs:51` (`has_land_border_tn`): TerraNullius-adjacency check for
   deciding whether a nation *has* land left to expand into.
2. `ai_attack.rs:89` (`has_shore_reachable_tn`): same, for boat-reachable
   shore expansion.
3. `ai_attack.rs:222` (`send_boat_attack_to_nearby_tn`): candidate tile
   collection for nearby boat-landing expansion.
4. `ai_attack.rs:425` (`collect_bordering_players`) and `ai_attack.rs:459`
   (`nearby_players_ts_order`): border-neighbor scans used to decide which
   players are "nearby" for attack-target selection.
5. `ai_attack.rs:494` (`nearby_players_ts_order`'s shore-search branch).
6. **`attack.rs:629` (`add_neighbors`)**: the border/candidate-tile
   collector for an active attack (same function discussed above). Missing
   `is_impassable` here meant impassable tiles bordering the *target's*
   territory could be added as legitimate candidates when the target was
   TerraNullius (impassable tiles always have owner 0, so they'd pass the
   `owner_id(neighbor) != target_small_id` check unchallenged), consuming an
   extra, TS-incompatible PRNG draw and border-tile slot.
7. **`attack.rs:285` (the `tick()` conquest loop)**: missing
   `is_impassable` here is the most serious of the seven. `game.conquer_one`
   (`game.rs:1133-1157`) only checks `is_land`, not `is_impassable`. Without
   the guard at `attack.rs:285`, a native attack could have actually called
   `game.conquer(...)` on an impassable tile that slipped through gap #6,
   giving it a real owner: a violation of the "impassable tiles are always
   owner 0" invariant the rest of the engine assumes (confirmed by checking
   `conquer_one` and spawn logic; nothing else double-checks this once a
   tile is owned).

### Fix

Added the missing `!game.is_impassable(...)` (or equivalent `||
game.is_impassable(...)` on negated conditions) checks at all 7 sites, each
matching a `!map.isImpassable(...)`/`map.isImpassable(...)` check that
already exists in the corresponding TS.

### Effect on this test game

Byte-identical output before/after (confirmed via `md5sum` on the full
6000-tick dump). This is expected, not a sign the fix is inert: it means no
impassable terrain happened to border any of the ~59 nations' territory
early enough on this specific map/seed/spawn layout to trigger the buggy
paths. The fixes are still correct and necessary: `attack.rs:285` in
particular closes a real invariant violation that would matter on maps with
impassable terrain (mountains/lakes) adjacent to expanding nations, which
BlackSea's spawn layout for this seed apparently didn't exercise in the first
6000 ticks. Recommend keeping these fixes; they cost nothing and remove a
whole class of potential native-only bugs (owned-impassable-tiles) even
though they didn't move this particular game's numbers.

## Remaining divergence (documented, not root-caused)

After both fixes above, the test game still diverges starting around tick
1497-1498. Unlike the primary bug, this one shows up as a **troop**
divergence before a **tile** divergence, and it's a much larger single-tick
jump rather than a gradual 1-2-tile drift:

```
tick 1497  nation:Spanish Realm  native troops=133714  ts troops=133714   (identical)
tick 1498  nation:Spanish Realm  native troops=107206  ts troops=85719    (native lost 26,508; ts lost 47,995)
```

Both engines drop troops sharply at the same tick (consistent with an attack
launching and deducting troops up front), but TS's expenditure (47,995,
about 36% of pre-attack troops) doesn't match the standalone `attackAmount`
formula (`attacker.troops() / 5` for non-Bot, about 26,743, matching
*native's* loss almost exactly) or `/20` for Bot. 47,995 is close to what
you'd get from **two** sequential `/5`-style attacks compounding in the same
tick (133714/5 + (133714-26743)/5 is about 48,137), which suggests TS's
nation AI issued two attacks (e.g. a land attack and a random boat attack,
or two back-to-back target evaluations) in that tick where native issued
only one.

This was not root-caused further given time constraints. Likely candidates
for follow-up (not yet verified): `nation_maybe_attack`
(`ai_attack.rs:1488-1556`) vs `AiAttackBehavior`'s per-tick entry point for
whether more than one attack path can fire per invocation, or a difference in
how "already has an active attack, skip" cooldown state is tracked between
engines. Per the task's guidance to avoid a risky large fix when the
remaining root cause isn't concretely pinned down, this is left as a
documented follow-up rather than a speculative patch. Total accumulated tile
error keeps growing after this point but at a visibly slower/less-explosive
rate than in the pre-fix trajectory (see table above), consistent with this
being a distinct, smaller-magnitude issue than root cause #1.

## Files changed

- `rust/engine/src/map.rs`: `for_each_neighbor4` order fix (root cause #1).
- `rust/engine/src/execution/attack.rs`: 2 `is_impassable` filter fixes
  (root cause #2, sites 6-7 above).
- `rust/engine/src/execution/ai_attack.rs`: 5 `is_impassable` filter fixes
  (root cause #2, sites 1-5 above).
- `rust/engine/src/bin/tick_dump.rs`, `scripts/gen_selfplay_record.ts`,
  `scripts/dump_ts_tick_state.ts`, `scripts/compare_tick_dumps.py`: new
  tooling (see above).
- `docs/bot-ai-parity-investigation/`: this report plus the raw
  before/after `compare_tick_dumps.py` JSON reports.

## Validation

- `cargo test --release -p openfront-engine --lib`: 47 passed / 35 failed,
  identical pass/fail set before and after these changes (confirmed via
  `git stash`). The 35 failures are pre-existing and unrelated: they fail
  with "No such file or directory" looking for fixture record files not
  present in this isolated worktree, on unmodified `master` too.
- Primary validation is the tick-dump comparison methodology described above,
  run at multiple granularities (every 50 ticks for the full 6000-tick game,
  every 1 tick for the tick-303 and tick-1498 bisections).
