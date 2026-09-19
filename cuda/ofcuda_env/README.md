# ofcuda_env - ONE per-tick environment step, composed

Turns the four verified slices into a single runnable tick and drives it with a
**computed** float claim budget instead of the record's observed one. No fifth
copy of `core_impl.rs`: this crate depends on the four as path dependencies.

```
cargo build            # 4 path deps + this crate
cargo run -- --dump <ndjson> --t0 300 --t1 1250 --frac 0.4
cargo run -- --dump <ndjson> --t0 318 --t1 322 --frac 0.0 --heap-at 320   # heap diff
```

## 1. The composed order (engine order, with citations)

| # | stage | engine site | what carries it |
|---|-------|-------------|-----------------|
| 1 | per-tick economy (troops/gold) | `execution/player.rs:82-88` `PlayerExecution::tick` -> `troop_increase_rate_raw_for` + `wire.gold_addition_rate` + `add_troops` | `ofcuda_econ::step_row` |
| 2 | float claim budget | `execution/attack.rs:239-255` -> `wire.attack_tiles_per_tick(...)` (`core/config.rs:484-501`); terra nullius body is `num_adjacent * 2.0` (`config.rs:496-501`) | `ofcuda_env::attack_tiles_per_tick` |
| 3 | persistent-heap expansion | `attack.rs:257-322`: carried `to_conquer` (`attack.rs:17`), refill only when empty (`attack.rs:264-268` -> `1265-1274`), `remove_border_tile` on every pop (`:275`), terra-nullius + attacker-neighbour guards (`:284`), land guard (`:288`), in-tick `add_neighbors` before the conquer (`:292`), `num -= tiles_used` (`:313`), `troop_count -= loss` (`:314`) | `ofcuda_tick::{Heap,Prng,add_neighbors,has_attacker_neighbor,is_terra_nullius,priority_f32}` + this crate's driver |
| 4 | cluster capture | `ofcuda_cluster` flood-fill (mark-on-discovery, neighbours8 dx-major, decide-and-remove replayed serially) | `ofcuda_env::cluster_capture` (`ofcuda_cluster::{prepare,decide_cpu,clusters_from}`) |
| 5 | state-plane update | the claims folded into the `w*h` `u16` ownership plane | `ofcuda_tick::state_plane_from_players` |
| 6 | FNV-1a-64 | `0xcbf29ce484222325` / `0x100000001b3`, `u16` as LE bytes, row-major | `ofcuda_hash::fnv1a_u16_le` |

Two ordering facts that had to be right, both with the engine as authority:

* the **one extra `next_int(0, 5)` draw is taken before the pop loop**
  (`attack.rs:253`), exactly once per tick, and it is the only thing that
  separates a 174 budget from a 178 one;
* the attacker's `num_owned_by_me` counts **attacker-owned** neighbours only
  (`attack.rs:1365-1370`) and includes this tick's earlier claims, and
  `add_neighbors` is called **before** the current tile is conquered
  (`attack.rs:292`), so the tile being claimed is still terra nullius for its
  own neighbours.

### What this crate had to add: the border set

The budget reads `self.border_tiles.len()` (`attack.rs:240`). `add_border_tile`
is called for exactly the neighbours that pass the water + owner==target filters
(`attack.rs:1353-1363`) - the enqueued set - and `remove_border_tile` on every
pop including skipped ones (`attack.rs:275`). `ofcuda_tick`'s
`add_neighbors`/`refresh_to_conquer` deliberately do **not** carry that set (its
README states the budget is exogenous), so `Attack` maintains it here. **No file
in `ofcuda_tick`, `ofcuda_cluster`, `ofcuda_econ` or `ofcuda_hash` was changed -
all four already exposed everything needed as libraries.**

## 2. Measured agreement (fresh dump, current build)

`records/early-curriculum-parity/curr-b002-s1-pangaea.json.gz` re-dumped with
the deployed `tick_dump` to `/tmp/envdump/fresh.ndjson` (1301 tick records).

Agreement = the composed tick reproduces the engine's per-tick claim **set and
order** exactly (`ownedOrder` diff between consecutive records).

| input to `tiles_used` | first mismatch | matched ticks |
|---|---|---|
| record integer (`troops as i64`), `--frac 0` | **tick 359** (rec359->rec360) | 318..358 |
| record value + the truncation, `--frac 0.4` | **tick 377** (rec377->rec378) | **318..376 (59 consecutive ticks)** |

`--frac 0.4` is not a tuned constant: the attack's exact `f64` troops is
`attacker.troops - max_troops_for(owner) * expand_ratio`
(`execution/ai_attack.rs:9-18`, `bot/tribe.rs:42`), and any value in
`[0.37, 1.0)` gives the same window - the constraint is a range, not a fit.

**The computed budget changes the answer in one place and reproduces the engine
everywhere else.** Over ticks 318..376 the computed budget reproduces the
engine's claim counts exactly, including the 12-claim ticks:

```
tick : 318 319 320 321 322 323 324 325 326 327 328 329
count:   8   9   9   9  10  10  12  11  10  12  12  12     (computed == engine)
```

No exogenous budget, no recorded claim count, no recorded frontier: the budget
comes from the tracked border set plus the tick's single draw, and the claim
cost comes from the float troop count.

### First mismatch, and its cause

* **With the truncated record troops (tick 359).** The six claims are
  byte-identical to the engine's, then the engine takes a seventh (`463251`) and
  this crate stops. `sum(tiles_used)` for the six claims is `174.046526...` and
  the budget is `2 * (86 + 1) = 174`: the crate is **0.047 short**. The engine's
  attack troops at that tick is `1363.4`, not the record's truncated `1363` -
  `tick_dump.rs:326` writes `AttackSnapshot.troops = troops as i64`, and
  `tiles_used = within(2000*speed/attack_troops, 5, 100)` divides by it. A
  0.36-troop error is ~2.7e-4 relative, which is exactly the knife-edge.
* **With the float troops (tick 377).** The engine claims 10; this crate claims
  3. Cause: at rec377 sid2 has **two concurrent land attacks** (`(2,557)` and
  `(2,1910)`), which at rec378 have **merged** into one `(2,1824)`. A second
  land attack for the same owner and a merge (`add_land_attack`) are **not
  ported** here - one `Attack` per owner sid is the model's limit.

## 3. What remains outside the proven region

**Assumed / exogenous**

* the attack's exact `f64` starting troops is fed in as `--frac` (cause of the
  tick-359 mismatch); the lawful producer is stage 1 (`ofcuda_econ`) plus
  `max_troops_for` and the tribe's `expand_ratio`, not the truncated record;
* `ownedOrder` is treated as the claim log, i.e. claims append in order;
* the dump's `borderOrder` is the engine's `OrderedTiles` iteration order;
* one PRNG per attack at `SEED = 123` (`ofcuda_tick`'s constant, reproduced);

**Unported mechanics**

* multiple concurrent land attacks per owner and their merge (the tick-377
  mismatch);
* `kill_attack` on `troop_count < 1.0` is present as a guard but never fires in
  this window (the attack is still at 527 troops at rec377);
* `refresh_to_conquer` mid-tick (the heap never runs dry in 318..376), so the
  refill path is wired but unexercised; the attack-death path likewise;
* player-vs-player attacks: the whole window is terra-nullius expansion
  (`targetSmallId == 0`), so `num_adjacent * 2.0` is the only budget body
  exercised; the defender-troop body (`config.rs:488-496`) is unit-tested only;
* the **fallout bit** and defense-post modifiers (`game.rs:805-816`,
  `mag *= modifier`) are not modelled: `has_fallout` is false throughout;
* cluster capture is wired and imports the verified slice, but fires only on a
  player loss - none occurs in this window.

**Not measured here**

* per-tile economy: measured negative by the econ slice (no per-tile income in
  stages 0-7), not re-litigated;
* the engine's `gameHash` is *not* the plain owner plane; hashing the dumped
  owner plane with `ofcuda_hash` reproduces `0/0` (the crate prints this
  check as a diagnostic). `ofcuda_hash`'s own 200/200 result is over the
  reference `state` plane, which this driver does not reconstruct - so the
  hash stage is composed but the end-to-end hash is **not** claimed here.
