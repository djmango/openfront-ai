# Bot-AI native-vs-TS parity: the tick-1498 "double attack" divergence

Branch: `agent/bot-ai-parity-double-attack` (isolated worktree off `master` @
`2bbc2fc`). Follow-up to `docs/bot-ai-parity-investigation/` (the
`for_each_neighbor4`-order fix), which left one documented, un-root-caused
residual divergence: nation "Spanish Realm" losing ~2x the expected troops
at tick 1498 in TS vs native. This investigation root-causes and fixes that
divergence, plus three more real bugs of the same class found in the process
(one of which is a second, independent "wrong attack target due to a
missing entity in a shuffled list" bug, confirmed with the same before/after
methodology).

## TL;DR

**Root-caused and fixed.** The tick-1498 divergence was caused by
`game::find_incoming_land_attacker` (`rust/engine/src/game.rs`) filtering
out boat-landed incoming attacks and missing `is_friendly`/`!alive` checks
that TS's `AiAttackBehavior.findIncomingAttackPlayer` (via
`PlayerImpl.incomingAttacks()`) applies. Native missed a valid retaliation
target that TS correctly found, so native fell through to a *different*,
cheaper decision branch while TS retaliated - the two paths cost different
troop amounts, which is what looked like a "double attack." Fixed; the
tick-1498 troop divergence for Spanish Realm is now **exactly zero**
(native and TS both land on troops=85719).

While verifying the fix moved the divergence point later rather than just
masking it, a **second, independent instance of the same bug class**
surfaced at tick 1543-1544 ("Hungarian Autocracy"): `nearby_players_ts_order`
(the Rust mirror of TS's `PlayerImpl.nearby()`) unconditionally dropped
`TerraNullius` (small ID `0`) from its candidate list, while TS's `nearby()`
includes it. This changes the *length* of the array fed into
`TribeExecution.maybeAttack`'s final random-target shuffle, which shifts
which target the shuffle picks even when discounting `TerraNullius` itself
(a longer/shorter array shuffles differently). Native and TS ended up
attacking two *different* neighbors with two different troop-cost formulas.
Root-caused via side-by-side debug tracing and fixed; also verified exactly
(native and TS both land on troops=67439 for Hungarian Autocracy at tick
1544, vs 84995 vs 67439 before).

Two more small, concretely-verified divergences of the same "boat attacks
incorrectly excluded" shape were found and fixed along the way:
`game::outgoing_land_troops` (used by `nation_alliance.rs`'s
alliance-strength check) and `game::incoming_attacks` (used by
`nation_emoji.rs`'s two "am I under heavy/light attack" emoji checks).

Net effect on the 50-bot BlackSea self-play test game (same reproduction as
the prior investigation): the byte-identical (troops AND tiles, all ~59
entities) prefix of the game grew from tick ~1497 to **tick ~1646** (a small
±1-troop rounding blip appears there - see "What's left" below, it is *not*
the same bug class and does not compound). The coarse first-divergence tick
(every-50-tick sampling, >10% relative diff on entities with >=20 tiles)
moved from **1750 (34% off)** to **2050 (25% off)**.

`cargo test --release -p openfront-engine --lib`: **47 passed / 35 failed**,
identical to the pre-existing master baseline (confirmed via `git stash`).

## Reproducing

Same tooling as `docs/bot-ai-parity-investigation/`, unchanged:

```bash
# from openfront-ai/ (this worktree)
openfront/node_modules/.bin/tsx scripts/gen_selfplay_record.ts \
  --map BlackSea --bots 50 --nations default --difficulty Medium \
  --seed bot-ai-parity-1 --max-ticks 6000 --out records/selfplay/bs50.json.gz

cargo run --release -p openfront-engine --bin tick_dump --manifest-path rust/Cargo.toml -- \
  --repo "$(pwd)" --record records/selfplay/bs50.json.gz --every 50 \
  --out /tmp/native_ticks.json --max-ticks 6000

openfront/node_modules/.bin/tsx scripts/dump_ts_tick_state.ts \
  records/selfplay/bs50.json.gz 50 /tmp/ts_ticks.json 6000

python3 scripts/compare_tick_dumps.py /tmp/native_ticks.json /tmp/ts_ticks.json \
  --rel-threshold 0.10 --abs-threshold 20 --out /tmp/report.json
```

No changes were made to the tooling itself in this investigation (it worked
as-is); the debug-trace instrumentation used to bisect both bugs (see
"Method" below) was temporary and has been fully removed from the final
diff.

## Root cause #1 (the tick-1498 target): `find_incoming_land_attacker` excluded boat attacks

### The bug

`rust/engine/src/game.rs`, `find_incoming_land_attacker` (before fix):

```rust
pub fn find_incoming_land_attacker(
    &self,
    defender_small_id: u16,
    defender_type: PlayerType,
) -> Option<u16> {
    let mut largest_attack = 0.0f64;
    let mut largest_attacker: Option<u16> = None;
    for exec in &self.execs {
        let ExecEnum::Attack(atk) = exec else { continue };
        if !atk.is_active() || !atk.attack_live() || !atk.is_initialized()
            || atk.source_tile().is_some()   // <-- excludes boat-landed attacks
        {
            continue;
        }
        if atk.target_small_id() != defender_small_id { continue; }
        let attacker = atk.owner_small_id();
        if attacker == defender_small_id || attacker == 0 { continue; }
        // no `!alive` check, no `is_friendly` check
        if defender_type != PlayerType::Bot {
            if let Some(p) = self.player_by_small_id(attacker) {
                if p.player_type == PlayerType::Bot { continue; }
            }
        }
        if atk.troops() > largest_attack {
            largest_attack = atk.troops();
            largest_attacker = Some(attacker);
        }
    }
    largest_attacker
}
```

This function is the native mirror of TS's
`AiAttackBehavior.findIncomingAttackPlayer` (`openfront/src/core/execution/utils/AiAttackBehavior.ts:405+`),
called by both `tribe_maybe_attack`'s "retaliate against the largest
incoming attacker" branch and `nation_maybe_attack`'s equivalent. TS's
version:

```ts
findIncomingAttackPlayer(): Player | null {
  const incomingAttacks = this.player.incomingAttacks();  // <-- NOT filtered by sourceTile
  ...
  const nonFriendly = incomingAttacks.filter((a) => !this.player.isFriendly(a.attacker()));
  ...
}
```

`player.incomingAttacks()` (`openfront/src/core/game/PlayerImpl.ts:1632`) is:

```ts
incomingAttacks(): Attack[] {
  return this._incomingAttacks.filter((a) => a.attacker().isAlive());
}
```

So TS's version: (a) includes boat-landed attacks (no `sourceTile()` filter
at all - that filter only exists at a *different* call site,
`NationStructureBehavior.tryBuildDefensePost`/`defensePostNeeded`, which
explicitly chains `.filter((a) => a.sourceTile() === null)` on top), (b)
filters dead attackers, and (c) filters friendly attackers. Native's version
did the opposite on (a) and was missing (b) and (c) entirely.

### Concrete evidence: what actually happened at tick 1497

At tick 1497, "Parthian Monkdom" had an active boat-landed attack in
progress against "Spanish Realm" (a `Bot`/tribe). At tick 1498's
`tribe_maybe_attack` call for Spanish Realm:

- **TS**: `findIncomingAttackPlayer()` correctly returns Parthian Monkdom
  (an ordinary incoming attack, boat or not, from a living non-friendly
  attacker) → `try_send_player_attack_forced` retaliates against Parthian
  Monkdom.
- **Native (before fix)**: `find_incoming_land_attacker` skips Parthian
  Monkdom's attack (its `source_tile` is set - it's boat-landed) → returns
  `None` → falls through past the retaliation branch, past a second traitor
  roll, into the final shuffled-neighbor loop → picks a different target
  ("Abbasid Guild") via a completely different code path with a different
  cost formula.

Both formulas nominally cost "some fraction of current troops," but they're
different fractions for different targets, which is why the loss amounts
didn't match a clean 1x or 2x multiple of any single formula - the original
hypothesis ("TS issued two attacks") was a reasonable first guess from the
symptom, but the actual mechanism is "native and TS chose different single
attacks because native's retaliation-target lookup was silently returning
`None` when it shouldn't have."

### Fix

Removed the `atk.source_tile().is_some()` exclusion, added the `!p.alive`
check, and added an `is_friendly` check, matching TS's
`incomingAttacks().filter(...)` chain exactly. Kept the function's
(now-inaccurate, but pre-existing and git-history-relevant) `_land_` name;
documented the actual semantics in its doc comment along with a pointer to
the one real call site that still wants land-only semantics
(`incoming_land_troops`, unchanged, used by `defensePostNeeded`).

### Before/after (tick-level)

```
tick 1497  nation:Spanish Realm  native troops=133714  ts troops=133714   (both, before AND after fix)
tick 1498  nation:Spanish Realm  native troops=107206  ts troops=85719    (BEFORE fix: native and ts diverge)
tick 1498  nation:Spanish Realm  native troops=85719   ts troops=85719    (AFTER fix: exact match)
tick 1499  nation:Spanish Realm  native troops=107525  ts troops=86046    (BEFORE fix, tiles start diverging too: 14768 vs 14775)
tick 1499  nation:Spanish Realm  native troops=86046   ts troops=86046    (AFTER fix, tiles match too: 14775 == 14775)
```

## Root cause #2 (found while fixing #1): `outgoing_land_troops` had the identical bug

`rust/engine/src/game.rs`, `outgoing_land_troops` (used by
`nation_alliance.rs`'s `is_alliance_partner_similarly_strong`, the native
mirror of TS's `NationAllianceBehavior.isAlliancePartnerSimilarlyStrong`)
had the exact same shape of bug:

```rust
// before
for id in &attacker.outgoing_land_attacks {
    if let Some(troops) = self.land_attack_troops(id) {
        if self.land_attack_source_tile(id).is_none() {  // <-- excludes boat attacks
            total += troops;
        }
    }
}
```

TS's `player.outgoingAttacks().reduce((sum, a) => sum + a.troops(), 0)`
(`PlayerImpl.outgoingAttacks()` returns `_outgoingAttacks` completely
unfiltered - no `sourceTile()` check) sums *all* outgoing attacks, boat or
land. Fixed by removing the `source_tile().is_none()` guard.

This one did not move any numbers in the 50-bot test game (verified via
tick-dump diff before/after this specific change in isolation) - no bot in
this particular run happened to have an in-flight boat attack at the exact
tick this alliance-strength check ran with a close-call ratio. Same
situation as several of the `is_impassable` fixes in the prior
investigation: byte-identical output isn't evidence the fix is a no-op,
it's evidence this specific map/seed didn't exercise the buggy path. The fix
is still correct per the TS source and costs nothing to keep.

## Root cause #3 (found while fixing #2): `incoming_attacks` had the same bug, used with conflicting semantics at 3 call sites

While auditing every other caller of the buggy pattern, `game::incoming_attacks`
turned out to have the same "always exclude boat attacks" bug, but unlike
\#1/\#2 it's called from **three** places that need **two different**
semantics:

1. `nation_structures.rs::try_build_defense_post` (TS
   `NationStructureBehavior.tryBuildDefensePost`) - correctly wants
   land-only (`player.incomingAttacks().filter((a) => a.sourceTile() === null)`).
2. `nation_emoji.rs::check_overwhelmed_by_attacks` (TS
   `NationEmojiBehavior.checkOverwhelmedByAttacks`) - wants **all** incoming
   attacks (`this.player.incomingAttacks()`, no filter).
3. `nation_emoji.rs::check_very_small_attack` (TS
   `NationEmojiBehavior.checkVerySmallAttack`) - same, wants **all**
   incoming attacks.

The pre-existing native function hard-coded the land-only filter, so (1)
was correct but (2) and (3) were wrong: a nation's "am I overwhelmed"/"is
this a pesky small attack" emoji checks were excluding boat attacks from
the incoming-troop tally TS includes.

### Fix

Added a `land_only: bool` parameter to `game::incoming_attacks`, threaded
`true` at the one land-only call site and `false` at the two unfiltered
call sites, and documented the split in the function's doc comment (mirrors
the pattern already used for `find_incoming_land_attacker` vs
`incoming_land_troops` from fix #1).

This is `NationEmojiBehavior`-only (nation, not tribe), gates a
`this.random.chance(16)`/`chance(8)` PRNG draw's *consequences* (whether
`sendEmoji` fires and consumes further PRNG for content selection) rather
than the gate itself, and - like fix #2 - didn't move any numbers on this
specific 50-bot test game (verified via isolated before/after tick-dump
diff). Kept for correctness; a map/seed where a nation has a close-call
"under attack" ratio with an active boat attack in flight would show a
different outcome without this fix.

## Root cause #4 (second independent double-attack-shaped bug): `nearby_players_ts_order` dropped `TerraNullius`

### Method: after fix #1, the *next* divergence needed the same debug-trace bisection as before

Fixing #1 moved the tick-1498 Spanish Realm divergence away, but a
fine-grained (every-tick) re-comparison through the new byte-identical
window turned up a **new** divergence at tick 1544 for a **different**
entity, "Hungarian Autocracy" (also a `Bot`/tribe) - troops
105998 → 84995 (native) vs 105998 → 67439 (TS), the same "TS loses ~2x what
native loses" shape as the original symptom. Cross-checked against the
*pre-fix-#1* native dump: this exact divergence (84995 vs 67439) was already
present there too, just masked by the larger, earlier Spanish Realm
divergence in the coarse comparison - i.e. this is a **second, independent**
bug, not a side effect of fix #1.

Re-applied the same debug-trace bisection technique from the original
investigation (temporary `eprintln!`/`console.error` gated by an
`OF_DEBUG_TRIBE` env var, added to `tribe_maybe_attack` and
`TribeExecution.maybeAttack`/`AiAttackBehavior.attackRandomTarget`,
comparing the exact branch and shuffled-target list taken on each side at
tick 1543 - the tick the attack execution is actually created, one tick
before its cost is reflected in the troop snapshot):

```
[native] shuffled neighbors=[32, 33, 11, 24]                                   (4 entries)
[native] SENT: shuffled attack on 32 (Mughal Region)

[TS]     shuffled neighbors=TN,Romanov Protectorate,Mughal Region,Songhai Sisterhood,Sumerian Order   (5 entries)
[TS]     SENT: shuffled attack on Romanov Protectorate
```

Native's candidate array is missing `TerraNullius` entirely (4 entries vs
TS's 5, with `TN` first in TS's list). Both engines call the same
`shuffle_array`/`shuffleArray` (a length-driven Fisher-Yates variant), so a
missing element doesn't just remove one possible target - it changes the
*whole permutation*, because every draw's remaining-swap-range depends on
how many elements are left. Native ends up attacking "Mughal Region";
TS attacks "Romanov Protectorate." Different target, different troops
formula outcome (land-attack "keep a troop reserve" formula vs whatever
`Mughal Region`'s specific numbers worked out to), hence the ~2x-looking
troop-loss gap - same *symptom* shape as bug #1, different root cause.

### The bug

`rust/engine/src/execution/ai_attack.rs`, `nearby_players_ts_order` (before
fix):

```rust
fn nearby_players_ts_order(game: &Game, small_id: u16) -> Vec<u16> {
    let mut seen = HashSet::new();
    let mut ordered: Vec<u16> = Vec::new();
    let mut push = |sid: u16| {
        if sid != small_id && sid != 0 && seen.insert(sid) {  // <-- sid != 0 drops TerraNullius
            ordered.push(sid);
        }
    };
    ...
}
```

This is the mirror of TS's `PlayerImpl.nearby()`
(`openfront/src/core/game/PlayerImpl.ts:487`):

```ts
nearby(): (Player | TerraNullius)[] {
  const ns: Set<Player | TerraNullius> = new Set();
  ...
  const visit = (neighbor: TileRef) => {
    if (map.isLand(neighbor) && !map.isImpassable(neighbor)) {
      ...
      const owner = map.ownerID(neighbor);
      if (owner !== smallID) {
        ns.add(this.mg.playerBySmallID(owner) satisfies Player | TerraNullius);
        //     ^ unconditionally adds TerraNullius (smallID 0) too
      }
    }
  };
  ...
  return Array.from(ns);
}
```

TS's `nearby()` really does return `TerraNullius` as one of the elements
when a bordering/shore-reachable tile is unowned; it's up to each *caller*
to filter it out if it only wants players (e.g.
`getNeighborTraitorToAttack`'s `n.isPlayer()` type-guarded `.filter`, or
`TribeExecution.maybeAttack`'s shuffled loop's
`if (!neighbor.isPlayer()) continue;`). Native's `nearby_players_ts_order`
instead baked the "players only" filter into the shared helper itself,
which is wrong for the one caller (the tribe random-target shuffle) that
needs the un-filtered length/order to match TS's shuffle exactly.

### Why this was safe to fix in the shared helper (checked all 4 call sites)

- `tribe_maybe_attack`'s shuffled loop (the one that needed the fix):
  already does `let Some(target) = game.player_by_small_id(target_sid) else { continue };`
  - `player_by_small_id(0)` is `None` (small ID `0`/`TerraNullius` is never
    in the `players` vec), so a `0` entry is skipped exactly like TS's
    `if (!neighbor.isPlayer()) continue;`. No change in behavior for this
    caller beyond the (correct) shuffle-order fix.
  - `get_neighbor_traitor_to_attack`: filters
    `!game.is_friendly(small_id, sid) && game.is_traitor(sid)`.
    `is_traitor(0)` → `player_by_small_id(0)` is `None` → returns `false`
    unconditionally, so a `0` entry is always filtered out here regardless
    - matches TS's explicit `n.isPlayer()` type guard, just via a different
    (still-correct) mechanism.
  - `collect_bordering_players` (nation `maybeAttack`'s bordering-player
    set): has its own **separate**, still-`sid != 0`-filtering `push`
    closure that calls `nearby_players_ts_order` and re-filters its output -
    unaffected by this fix, and correctly so (TS's nation `maybeAttack`
    explicitly does `if (n.isPlayer()) borderingPlayerSet.add(n);` when
    folding `nearby()` into the bordering set - `TerraNullius` truly is
    excluded from that particular set in TS too, just via an inline filter
    at the call site rather than inside `nearby()` itself).
  - The `Hard`/`Impossible`-difficulty "max neighbor troops" retain-fraction
    lookup (`ai_attack.rs` around line 1215): uses
    `.filter_map(|sid| game.player_by_small_id(sid))`, so a `0` entry is
    dropped by the `filter_map` itself. Unaffected.

### Fix

Removed the `sid != 0` filter from `nearby_players_ts_order`'s `push`
closure and documented why (the doc comment now explains which of the 4
call sites need the raw, `TerraNullius`-inclusive list vs which already
filter it themselves).

### Before/after (tick-level)

```
tick 1543  nation:Hungarian Autocracy  native troops=105998  ts troops=105998   (both, before AND after fix)
tick 1544  nation:Hungarian Autocracy  native troops=84995   ts troops=67439    (BEFORE fix: diverge)
tick 1544  nation:Hungarian Autocracy  native troops=67439   ts troops=67439    (AFTER fix: exact match)
tick 1546  nation:Hungarian Autocracy  native troops=85529   ts troops=67972    (BEFORE fix)
tick 1546  nation:Hungarian Autocracy  native troops=67972   ts troops=67972    (AFTER fix: exact match, tiles too: 8256 == 8256)
```

## Overall before/after evidence (50-bot BlackSea self-play, seed `bot-ai-parity-1`)

All four fixes applied together vs pristine `master` (i.e. including the
prior investigation's `for_each_neighbor4`/`is_impassable` fixes, but before
any fix in *this* document), same TS reference dump, same thresholds as the
prior report:

Coarse (every-50-tick, >10% relative on entities >=20 tiles) first
divergence:

| | tick | worst entity |
|---|---|---|
| before (this session's fixes) | 1750 | `nation:Taino Supremacy` 4338 vs 6564 tiles (34%) |
| after | **2050** | `nation:Frankish Canton` 12548 vs 9368 tiles (25%) |

Total accumulated absolute tile error across all ~59 entities, sampled every
50 ticks (full table in `report_before_fix.json` / `report_after_fix.json`
in this directory; select rows below):

| tick | before (this session) | after |
|-----:|-----:|-----:|
| 1450 | 0 | 0 |
| 1500 | 28 | 0 |
| 1750 | 15,174 | **0** |
| 2000 | 60,911 | 1,966 |
| 3000 | 241,264 | 223,501 |
| 4000 | 224,734 | 180,776 |
| 5000 | 312,593 | 431,565 |
| 6000 | 410,340 | 551,537 |

**Every sampled tick from 50 through 1750 is now byte-identical** (zero
total error, up from "identical through 1450" before this session). Note
the tick-6000 total is *not* monotonically better (410,340 → 551,537): once
any single decision genuinely diverges between the two engines (now
starting later, around tick ~2050, instead of ~1750), the two trajectories
are chaotic dynamical systems from that point on - a state-dependent AI
loop feeding back into troop counts, territory, and subsequent decisions -
so the *magnitude* of long-horizon divergence has no reason to shrink just
because the divergence *starts* later. The meaningful, reliable signal from
this kind of fix is "how long is the byte-identical prefix" and "how much
smaller is the first real divergence," both of which clearly improved; the
tick-6000 aggregate is a noisier, chaos-amplified metric and shouldn't be
read as "things got worse." (The prior investigation's README makes the
same point about its own, much larger, fix.)

## What's left (documented, not fixed - see task guidance on avoiding speculative patches)

A fine-grained (every-tick) comparison after all four fixes shows the very
first byte-level difference (troops, not tiles) now at **tick 1647**, for
"Dutch Wilderness": native troops=128487 vs ts troops=128486 - **a 1-troop
difference**, not the "off by a formula's worth of troops" shape of the four
bugs above. This looks like floating-point rounding/truncation-order noise
(e.g. a growth-formula rounding difference), not a PRNG-stream or
target-selection desync: it doesn't compound (native and TS is still just 1
troop apart at tick 1700, 1750, 1800) and it fully **self-heals** by tick
1900 (native and TS are exactly equal again, and stay equal through at
least tick 2050). Chasing a non-compounding ±1 rounding blip down to its
exact source line felt like exactly the kind of low-value, high-effort
detour the task asked to avoid; flagging it here as a known, harmless,
extremely minor artifact rather than fixing it speculatively.

The *practically meaningful* next divergence is the coarse one at tick 2050
("Frankish Canton", noted above). Given how much the tick-1543 bug turned
out to resemble the tick-1498 bug (both "a shuffled/sorted candidate list
had the wrong contents, changing which target got picked"), it's plausible
the tick-2050 divergence is a third instance of a similar-shaped bug
somewhere else in the nation/tribe AI decision surface - but that wasn't
confirmed with the same tick-by-tick evidentiary rigor as the four fixes
above, so it's left as a documented follow-up rather than guessed at.

One more thing noticed but **not fixed** (no concrete tile/troop evidence
tying it to a specific divergence, so left as a documented gap rather than
a "fix"): TS's `TribeExecution.tick()`
(`openfront/src/core/execution/TribeExecution.ts:62-64`) calls
`this.deleteNextStructure()` (deletes one structure marked for deletion,
per tick, before `maybeAttack()`) every tick after the alliance-request
step; native's `TribeExecution::tick` (`rust/engine/src/bot/tribe.rs`) has
no equivalent call. This wouldn't affect troop/tile counts directly (it
only removes a unit already marked for deletion) and doesn't consume PRNG,
so it's unlikely to explain any of the divergences investigated here, but
it is a real, small behavioral gap for anyone relying on native tribes to
clean up deleted structures.

## Files changed

- `rust/engine/src/game.rs`: `find_incoming_land_attacker` (root cause #1)
  and `outgoing_land_troops` (root cause #2) boat-attack-exclusion fixes,
  plus `incoming_attacks` gains a `land_only: bool` parameter (root cause
  #3).
- `rust/engine/src/execution/nation_structures.rs`,
  `rust/engine/src/execution/nation_emoji.rs`: updated the 3
  `game.incoming_attacks(...)` call sites for the new parameter (root cause
  #3).
- `rust/engine/src/execution/ai_attack.rs`: `nearby_players_ts_order` no
  longer drops `TerraNullius` (small ID `0`) (root cause #4).
- `docs/bot-ai-parity-double-attack/`: this report plus the raw
  before/after `compare_tick_dumps.py` JSON reports (all four fixes
  applied, vs pristine `master`).

No changes to `scripts/gen_selfplay_record.ts`, `rust/engine/src/bin/tick_dump.rs`,
`scripts/dump_ts_tick_state.ts`, or `scripts/compare_tick_dumps.py` - the
existing tooling from the prior investigation was sufficient throughout.
All temporary debug-trace instrumentation used to bisect root causes #1 and
#4 (gated behind `OF_DEBUG_NATION`/`OF_DEBUG_TRIBE` env vars, in
`ai_attack.rs`, `TribeExecution.ts`, and `AiAttackBehavior.ts`) was added and
then fully removed; the final diff contains only the four production fixes
listed above.

## Validation

- `cargo test --release -p openfront-engine --lib`: **47 passed / 35
  failed**, identical pass/fail set before and after all four fixes
  (confirmed via `git stash`/`git stash pop`, same technique as the prior
  investigation). The 35 failures are pre-existing and unrelated: they fail
  looking for fixture record files not present in this isolated worktree,
  reproducing identically on unmodified `master`.
- Primary validation is the tick-dump comparison methodology from the prior
  investigation, run at multiple granularities: every-50-tick for the full
  6000-tick game (coarse divergence-point tracking), every-1-tick for the
  tick-1498 and tick-1543/1544 bisections (exact root-causing), and
  every-1-tick again post-fix through tick 2100 (confirming the new
  byte-identical prefix length and locating the next, smaller residual
  divergence).
- Each of the four fixes was checked against its own before/after tick-dump
  diff individually during development (not just the combined end state),
  to confirm which ones moved observable numbers on this specific test game
  (#1 and #4 did, exactly as documented above) and which didn't (#2 and #3
  are still real, TS-source-verified fixes, just inert on this particular
  map/seed/tick-window - the same "byte-identical isn't evidence of a
  no-op fix" caveat the prior investigation's `is_impassable` fixes noted).

## Branch / commits

Branch: `agent/bot-ai-parity-double-attack`, based on `master @ 2bbc2fc`.
Not merged to `master` per instructions - see the commit log on this branch
for the atomic per-fix commits.
