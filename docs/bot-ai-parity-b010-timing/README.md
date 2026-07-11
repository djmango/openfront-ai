# Bot-AI native-vs-TS parity: fabricated random-boat fallback attack (bug 37)

Follow-up to `docs/bot-ai-parity-nation-relations/`. Scope: bisecting a single
`curriculum-parity-v4` `timing_mismatch` divergence -
`records/curriculum-parity-v4/curr-b010-s0-pangaea.json.gz` (10 bots,
Pangaea, self-play) - per the task's "same winner, later tick" tractability
heuristic from that report.

## Starting point

Cached outcome-gate comparison for this record (before this fix):

```
category:     timing_mismatch
winnerMatch:  true    (both pick "nation:Mexico")
timingMatch:  false   (TS: tick 9961, native: tick 13210)
landShareMatch: true
```

## Bug: `attack_with_random_boat`'s fabricated fallback attack

TS `AiAttackBehavior.attackWithRandomBoat` (`openfront/src/core/execution/utils/AiAttackBehavior.ts`):

```ts
let dst = this.findRandomBoatTarget(src, borderingEnemies, true);
if (dst === null) {
  dst = this.findRandomBoatTarget(src, borderingEnemies, false);
  if (dst === null) {
    return; // <-- no attack, no further PRNG draws
  }
}
```

Native's `attack_with_random_boat` (`rust/engine/src/execution/ai_attack.rs`)
had an extra tail after both `findRandomBoatTarget`-equivalent passes came
back empty:

```rust
if !bordering_enemies.is_empty() {
    let idx = random.next_int(0, bordering_enemies.len() as i32) as usize;
    let target = bordering_enemies[idx];
    // ...send a boat attack against `target` anyway...
}
false
```

This has no TS counterpart at all. Two effects, both real bugs:

1. **A fabricated attack.** Whenever a nation had bordering enemies (i.e. a
   land border with another player - the exact case where a *boat* attack is
   least sensible, since you'd normally just attack over land) and both
   random-target searches failed (very common - `findRandomBoatTarget`
   requires an actual reachable coastal target within a 500-try random
   window), native fired a boat attack against a random bordering enemy that
   TS never sends.
2. **An extra, permanently-desyncing PRNG draw.** `random.next_int(0,
   bordering_enemies.len())` is drawn *unconditionally* whenever
   `bordering_enemies` is non-empty in that fallback, even on ticks/games
   where TS's execution of the exact same logical branch draws nothing at
   all after the two `findRandomBoatTarget` passes exhaust their 500 draws
   each. Every PRNG-driven decision for the rest of the game (attack target
   choice, troop-send randomization, emoji, alliance-request rolls, ...) is
   then offset by one draw versus TS from that point on - this is the same
   bug *shape* as the nation-relations report's swapped-argument bug: a
   small logic mistake that doesn't change *what* happens on the tick it
   fires, but silently desyncs the RNG stream so *everything after* drifts,
   manifesting only much later as a `timing_mismatch` (same eventual winner,
   different terminal tick).

## Fix

Delete the fallback entirely - `attack_with_random_boat` now matches TS
exactly: if both `findRandomBoatTarget` passes fail, it returns `false` with
no further action and no extra PRNG draw.

## Bisection notes

The two-line TS snippet above was enough to identify this as the root cause
directly from source inspection (no tick-level replay bisection needed this
time) once `attack_with_random_boat`/`findRandomBoatTarget` were identified
as the area to inspect for `bots=10`, self-play, Pangaea rate divergences -
random boat attacks are one of the few remaining spots in `ai_attack.rs`
without a 1:1 TS line correspondence already documented from prior sessions.
Confirmed against the actual record before committing (see "Result" below).

## Result

`outcome_gate` re-run on just `curr-b010-s0-pangaea.json.gz`:

| | TS oracle | native (before) | native (after) |
|---|---|---|---|
| terminal tick | 9961 | 13210 | 10780 |
| tick delta | - | 3249 | 819 |
| category | - | `timing_mismatch` | `pass` |

`cargo test --release -p openfront-engine --lib`: 315 passed -> 317 passed (2
new regression tests in `ai_attack.rs`'s `random_boat_fallback_tests`
module), 0 failed, 10 ignored (unchanged, pre-existing/unrelated).

This does not claim to be the only remaining rate-divergence bug in this
bucket - 819 ticks of remaining drift on this one record means there is
likely at least one more smaller-magnitude divergence left. This report
documents one confirmed, fixed bug.
