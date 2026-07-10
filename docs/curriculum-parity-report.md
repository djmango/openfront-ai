# Curriculum-representative TS/native outcome parity check

**TL;DR:** the existing 78-record `outcome_gate` archive is uniformly
400-bot/125-human mega-games and can't tell us whether TS/native divergence
is a problem at the bot counts the RL curriculum actually trains on
(0/5/10/30/50/80/120/150). This generates a small self-play record set at
those exact bot counts and runs the same comparison. Result: **overall
55% (22/40) pass**, with a qualitatively different and much milder failure
mode than the 400-bot case - never a wildly wrong game, always a close race
where a different (but similarly-weak) player narrowly edges out the lead.
Curriculum-relevant bot counts (≤30, the first ~5 stages) are **not clean**
(60-100% at bots≤10, but **0%** at bots=30) - see caveats below before
reading that as "it's fine."

## Method

1. **Generation** - `datagen/gen_curriculum_parity.ts` builds a `GameRecord`
   per game directly (no live TS game object needs to be driven to
   "record" anything): bots + AI nations only, **zero human players**,
   `gameMode: "Free For All"`, `gameType: "Public"`, and the map/bot/nation
   counts/difficulty for one representative curriculum stage per distinct
   bot count (see table below and `BUCKETS` in the script). Turns are
   entirely empty (`info.num_turns = N`, `turns: []`) - both the TS
   (`decompressGameRecord`) and native (`GameRecord::decompress`) loaders
   zero-pad that to `N` empty-intent turns, and since bots/nations act
   autonomously from engine-internal AI rather than from replayed intents,
   there is nothing else to record. `N = 4500` ticks, `maxTimerValue = 6`
   minutes, for every bucket.
2. **TS ground truth** - the generator calls `replayOutcome()` (imported
   straight from `datagen/replay.ts`) on each record right after building
   it, and separately the standard `datagen/replay.ts --outcome-oracle`
   pipeline is run unmodified against the output directory to produce the
   oracle cache `outcome_gate` expects. Both call the identical function,
   so they're a consistency check on each other, not two different
   methods.
3. **Native side** - `rust/engine/src/bin/outcome_gate` unmodified, pointed
   at the new records dir + oracle cache (`--parity-commit
   curriculum-parity-v1`, an arbitrary label - this record set has no
   corresponding openfront-engine git commit, so it's not tied to one).
4. **Comparison** - `compare_outcomes()` unmodified: winner match, terminal
   tick within 20%, land share within 10% (absolute).

Run it end to end with `scripts/run_curriculum_parity_gate.sh` (mirrors
`scripts/run_outcome_gate.sh`'s structure/env-var conventions).

### A config quirk this had to work around

A `"Singleplayer"`-type game (what `rust/oftrain`'s `RlSession::reset` /
`bridge/env.ts` actually use) only ends its spawn phase when a **human**
player spawns - `SpawnExecution.tick()` (TS) / `spawn.rs` (native) both
gate `endSpawnPhase()` on `playerType === Human`. With zero human players
that condition never fires, and since real bot/nation AI behavior
(`PlayerExecution`, `TribeExecution`, `WinCheckExecution`, and the
"already spawned" branch of `NationExecution`) is gated on
`!inSpawnPhase()`, a 0-human Singleplayer game sits in a perpetual
respawn-in-place loop forever - verified empirically: total tiles owned
stayed flat at ~400 for 2000 ticks in a scratch test with 0 humans / 5
bots / 3 nations. This also means the existing `datagen/generate.ts`
(bot-only data-gen script, `gameType: Singleplayer`, 0 humans) has the
same bug and its bots never actually play - worth a follow-up look
independent of this parity check.

Fix: use `gameType: "Public"` instead (the type real archived multiplayer
games use). That adds a `SpawnTimerExecution`, which force-ends the spawn
phase after `numSpawnPhaseTurns()` ticks (300, for `randomSpawn: false`)
regardless of players - so bots/nations actually start playing at tick
~300. `maxTimerValue` guarantees a terminal condition (via `max_timer`)
within the tick budget even if no one naturally reaches the land-share win
threshold, since most of these games (many similarly-weak AI
nations/bots, no dominant human driving expansion) don't reach an 80%
land-share win in under ~4000 ticks.

## Bucket definitions (one representative stage per distinct bot count)

| bots | nations | difficulty | curriculum stage | maps rotated across 5 seeds |
|---|---|---|---|---|
| 0 | 1 (exact) | Easy | stage 0 | Onion |
| 5 | 3 (exact) | Easy | stage 2 | Onion, Pangaea |
| 10 | 6 (exact) | Easy | stage 3 | Pangaea, Caucasus |
| 30 | default | Easy | stage 4 | Pangaea, Caucasus, BlackSea |
| 50 | default | Medium | stage 6 | World, Asia, BlackSea |
| 80 | default | Medium | stage 7 | World, Asia, BetweenTwoSeas, Caucasus |
| 120 | default | Hard | stage 9 | all 7 maps |
| 150 | default | Impossible | stage 10 | all 7 maps |

(bots=30 and bots=80 each appear in two curriculum stages - Easy/Medium and
Medium/Hard respectively; the lower-difficulty occurrence was used for
both.) 5 games/bucket, 40 records total, ~160KB gzipped.

## Results: pass rate by bot count

| bots | pass/total | rate | avg total-tiles ratio (native/TS) | avg \|land-share delta\| on wrong-winner records |
|---:|---:|---:|---:|---:|
| 0 | 5/5 | **100%** | 1.0000 | n/a (no failures) |
| 5 | 3/5 | **60%** | 0.9834 | 0.034 |
| 10 | 4/5 | **80%** | 0.9970 | 0.033 |
| 30 | 0/5 | **0%** | 0.9997 | 0.016 |
| 50 | 2/5 | **40%** | 0.9993 | 0.034 |
| 80 | 3/5 | **60%** | 1.0009 | 0.030 |
| 120 | 3/5 | **60%** | 1.0012 | 0.038 |
| 150 | 2/5 | **40%** | 1.0001 | 0.079 |
| **all** | **22/40** | **55%** | | |

Every single failure (18/18) is category `wrong_winner`; there were
**zero** `missing_winner`, `timing_mismatch`, or `land_share_mismatch`
failures. Terminal tick always matched almost exactly (3910 native vs.
3911 TS - the 1-tick difference is `%10`-alignment noise, both are the
`max_timer` cutoff firing at the same point).

## Why this looks nothing like the 400-bot result

The 400-bot archive's failure mode (from the prior investigation,
`docs/devlog.html` "Why the gate took 4 hours…") was **`missing_winner`
across the board**, with wildly different final states - one example
had TS's real game end with a team at 95.05% land share while the native
replay of the exact same intents left that team at 14.8%, with 36.4% of
the map going to unaffiliated "Bot" entities. Not a near miss; a
different game.

Nothing like that shows up here:

- **Total conquered land matches almost exactly** at every bot count -
  the "avg total-tiles ratio" column above is 0.98-1.001 across the board.
  Both engines agree closely on *how much* of the map got fought over.
- **Land-share deltas on the winner, even when the winner itself is
  wrong, stay small** - mostly 0.016–0.041, i.e. well inside the 10%
  absolute tolerance that `compare_outcomes` uses for the land-share
  check specifically (that check just isn't reached once the winner
  identity itself disagrees, per `compare_outcomes`'s category-priority
  order). Only one record (`curr-b150-s1-pangaea`, delta 0.128) would
  have also failed land-share on its own.
- **Deep-dive example** (`curr-b030-s0-pangaea.json.gz`, worst bucket):
  59 players on both sides, total tiles conquered `420,335` on both
  (ratio 1.000), TS's top-3 land holdings `[60396, 55856, 42773]` vs.
  native's `[59638, 53616, 51274]` - same order of magnitude, same
  general shape of "no one dominates yet," just a different nation
  (Finland vs. China) narrowly on top at the tick-3910/3911 snapshot.

So what's actually happening: with `nations: "default"` (bots=30 and up)
the map fills with dozens of AI nations of similar starting strength, none
of which has grown dominant by the max_timer cutoff (~tick 3900, land
shares of the leader are typically only 8-20%). "Who is narrowly ahead of
a crowded field of similar competitors at an arbitrary snapshot tick" is
inherently sensitive to small decision differences - and there evidently
*is* a real, small, systemic behavioral difference between native and TS
bot/nation AI (attack targeting, timing, or similar; not scoped further
here - see `docs/devlog.html`'s existing "Investigate the bot-AI
behavioral gap" next-step item), enough to flip who's on top without
changing the overall shape of the game. That's a categorically different
(and far milder) problem than the 400-bot case's outright divergent
simulation.

It's also worth being honest that some of this "wrong winner" signal is
partly an artifact of the test methodology itself, not purely an engine
bug: comparing "who's leading a many-way FFA at one arbitrary cutoff tick"
is a high-variance question when several competitors are close - even two
runs of *the same* engine with a slightly different RNG path could plausibly
disagree on the exact leader in these crowded-default-nations configs.
Since native and TS are meant to be deterministic mirrors of each other
given identical seeds, any disagreement at all does confirm a real
behavioral gap exists - but the practical severity of "which of 20
similarly-weak nations is narrowly in first" is much lower than it would
be if this were a heads-up or small-N game where a wrong winner would mean
something qualitatively different actually happened.

## Is curriculum-relevant parity (bots ≤ 30) fine?

**Mixed, not a clean "yes."** Bots 0/5/10 (stages 0-3, the very first
curriculum stages) are 100%/60%/80% - reasonable, though bots=0 is a
degenerate no-opposition case (exactly one nation, trivially always
"wins," so it doesn't really exercise AI parity at all) and 60-80% at
5-10 bots is not nothing: 2-4 wrong winners out of 10 games is already a
real gap even before bot AI has much surface area to diverge on. Bots=30
(stage 4, still an *early* stage) is the worst bucket in the entire
sweep at **0/5** - worse than every higher-bot-count bucket, which argues
against a simple "parity degrades monotonically with bot count" story;
`nations: "default"` (used from bots=30 up) seems to matter more than raw
bot count for this specific "who narrowly leads a crowded map" failure
mode. Given the 5-games-per-bucket sample size, treat the exact ordering
across 30/50/80/120/150 as noisy, but the *presence* of a non-trivial
wrong-winner rate starting immediately at bots=5 (stage 2) is a
consistent, real signal, not sampling noise.

For RL training validity specifically - the trainer runs entirely on one
engine (native) self-play; it never cross-compares against TS mid-training
- the more relevant reassurance is the total-tiles-ratio ≈ 1.0 and
similar-shape land distributions at every bucket, which suggests native's
overall game dynamics (growth rates, how contested the map gets, how many
players survive) track TS reasonably closely even where the exact winner
disagrees. But this check does not by itself prove policies trained on
native transfer cleanly to real TS gameplay; it only shows the aggregate
simulation shape is similar, not that per-tick strategic decisions
(rewards are placement/strength-based every tick in `curriculum.rs`, not
just a single terminal winner check) are identical.

## Files

- `datagen/gen_curriculum_parity.ts` - record generator (self-contained;
  imports `replayOutcome` from `datagen/replay.ts` for the inline
  cross-check, otherwise no new engine-driving logic).
- `scripts/run_curriculum_parity_gate.sh` - end-to-end driver (generate →
  TS oracle cache → native `outcome_gate` → breakdown), mirrors
  `scripts/run_outcome_gate.sh`.
- `scripts/analyze_curriculum_parity.py` - turns an `outcome_gate` JSON
  report into the by-bot-count table + failure detail above (parses bot
  count from the `curr-b<N>-...` filename convention).
- `records/curriculum-parity-v1/` - the 40 generated `.json.gz` records
  (~160KB) + `records/curriculum-parity-v1.manifest.json` (generation
  metadata + the inline TS cross-check outcome for every record). Force-
  added despite `records/` being gitignored (that ignore rule exists
  because the 78-record human-game archive is fetched at runtime from HF,
  not committed) since this set is small and the task asked for it to be
  checked in.
- `/tmp/curriculum_gate_report.json` (not committed, regenerate via the
  script above) - full per-record `outcome_gate` output.

## Reproducing

```bash
scripts/run_curriculum_parity_gate.sh            # uses existing records/ if present
scripts/run_curriculum_parity_gate.sh --regenerate  # regenerate from scratch (~4 min gen + ~5 min oracle + ~2 min native)
```
