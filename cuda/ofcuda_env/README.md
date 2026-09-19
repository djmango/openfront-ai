# ofcuda_env - ONE per-tick environment step, composed

Turns the four verified slices into a single runnable tick and drives it with a
**computed** float claim budget instead of the record's observed one. No fifth
copy of `core_impl.rs`: this crate depends on the four as path dependencies.

```
cargo build
cargo run -- --dump <ndjson> --t0 318 --t1 390 --verbose     # computed f64 start troops (default)
cargo run -- --dump <ndjson> --t0 318 --t1 400 --start record --frac 0.4   # A/B: truncated record value
cargo run -- --dump <ndjson> --t0 300 --t1 1301              # whole window
cargo run -- --dump <ndjson> --t0 318 --t1 322 --heap-at 320 # heap diff
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

### 1a. The attack *set*: concurrent attacks and the merge

The engine holds one `AttackExecution` per in-flight attack; a player may have
several. The model here is one `Attack` per record entry, per owner, in the
record's own order, and the merge is ported:

* **Live merge path:** `execution/attack.rs:177-184`
  (`AttackExecution::init`) calls
  `game.merge_outgoing_land_attacks(owner_small_id, target_small_id,
  current_attack_id, &mut troops)` - only for land attacks
  (`source_tile.is_none()`).
* **`merge_outgoing_land_attacks`** is `game.rs:2122-2156`. It walks the
  owner's `Player.outgoing_land_attacks` (`game.rs:125`, pushed by
  `register_land_attack` `game.rs:2020`, retained on kill at `game.rs:2040`),
  skips the new attack itself, keeps entries with the same owner **and** the
  same `target_small_id`, adds each one's `troops` into the new attack's
  `*troops`, and calls `kill_attack()` on the absorbed execution -
  which sets `attack_live = false` (`attack.rs:1212`) while `active` stays
  true, so the absorbed attack still occupies one more tick in `execs` and
  ticks a no-op (`attack.rs:215-218`).
* **Dead code, not ported:** `try_merge_land_attack` (`game.rs:1991`) occurs
  **once in the whole tree** - the definition - with no caller. (The only other
  mentions are two comments in `ofcuda_econ`.) It is not the merge.

So the behaviour at rec377 is *not* two attacks racing: the engine absorbs
`(2,557)` into `(2,1910)` and the record shows the absorbed one for exactly one
more tick (`attackLive=false`) before it is gone. That lingering tick is
modelled: `lingering no-op ticks = 12` below is those absorbed attacks.

### 1b. What this crate had to add: the border set

The budget reads `self.border_tiles.len()` (`attack.rs:240`). `add_border_tile`
is called for exactly the neighbours that pass the water + owner==target filters
(`attack.rs:1353-1363`) - the enqueued set - and `remove_border_tile` on every
pop including skipped ones (`attack.rs:275`). `ofcuda_tick`'s
`add_neighbors`/`refresh_to_conquer` deliberately do **not** carry that set (its
README states the budget is exogenous), so `Attack` maintains it here.

### 1c. The f64 start troops (replacing `tick_dump.rs`'s truncation)

`tick_dump.rs:326` writes `AttackSnapshot.troops = troops as i64`, so the record
can only give an integral attack troop count. Stage 1 (`ofcuda_econ`, itself
device-vs-host exact at 7791/7791) plus the tribe's own ratio reconstructs the
engine's value:

```
start = player.troops_after(rec s-1)
        - max_troops_for(owner) * ratio        # ai_attack.rs:9-18, tribe.rs:42
ratio = expand_ratio  (target == 0) | reserve_ratio
```

`--start computed` (the default) uses that; `--start record --frac <x>` is kept
only as the A/B that shows what the truncation costs (below).

## 2. Measured agreement (fresh dump, current build)

`records/early-curriculum-parity/curr-b002-s1-pangaea.json.gz` re-dumped with
the deployed `tick_dump` to `/tmp/envdump/fresh.ndjson` (1301 tick records,
178 MB, written 19:00 from the 15:08 `libopenfront_engine.so`;
`tick_dump.rs` mtime predates it, so the dump is of the current build).

Agreement = the composed tick reproduces the engine's per-tick claim **set and
order** exactly (`ownedOrder` diff between consecutive records), per owner.

| input to `tiles_used` | first mismatch | matched |
|---|---|---|
| record integer + `--frac 0.4` (the old A/B) | **tick 377** (rec377->rec378, engine 10 vs crate 12) | 127/164 pairs, 318..376 |
| **computed f64 start (default)** | **none in 2..1301** | **2598/2598 (tick,player) pairs**, i.e. 1299 transitions |
| computed f64 start, attacks only | **none in 318..1301** | **1966/1966 pairs** (983 consecutive transitions) |

Counts, from the earliest attack (rec318) - engine == composed on every tick,
including the 12-claim ticks:

```
tick : 318 319 320 321 322 323 324 325 326 327 328 329 330 331
count:   8   9   9   9   9  10  10  12  11  10  12  12  12  12
```

Construction accounting over 2..1301:

```
creations 36 (floor(computed start) == record troops on 24/24 non-merge creations)
merges    12 (floor(computed merged troops) == record troops on 12/12)
deaths    starved(troops<1)=24 retreated(heap dry)=0 | lingering no-op ticks 12
heap      candidates dropped on a full heap 0 | peak heap depth 851
evictions merge-killed=12 unexplained=0 | unusable record entries 0
```

`floor(start) == record troops` is the independent check on 1c: the computed
start reproduces the truncated record field exactly whenever no merge happened.
The 12 merges, e.g. the tick-377 one (`new start=1353.332073 + absorbed
557.462201 -> 1910.794274`, record 1910), floor to the record's value too.

### First mismatch, and its causes (both now closed)

* **tick 359, with the truncated record value.** Six claims byte-identical,
  then the engine takes a 7th: `sum(tiles_used) = 174.046526` against a budget
  of `2*(86+1) = 174` - 0.047 short, because `tiles_used` divides by the
  truncated `1363` instead of `1363.4`. Closed by 1c: the budget now divides by
  the computed `f64`.
* **tick 377, with the truncated value: two concurrent attacks.** Closed by 1a
  (the second attack is created, `init` absorbs it, the absorbed attack ticks
  one no-op, then it is gone) *and* by 1c (the absorbed 557.462201 must be
  exact or the merged total misses its floor).
* **tick 557->558, the last one found (single attack, no merge involved).**
  The composed list matched to index 22 and then missed `457284` while the
  engine took it. Cause, measured not guessed: **the host heap was a fixed 256
  array that DROPS a candidate when full, while the engine's
  `FlatBinaryHeap` is a growable `Vec`** (`execution/flat_heap.rs:8,26-27`,
  `Vec::with_capacity(1024)` is only a hint). Our heap peaked at exactly 256
  and the run reported **13180 dropped candidates**; the divergence starts at
  the first tick whose heap saturates. Fixed in `ofcuda_tick` (see 4).

## 3. The hash question: which hash this actually reproduces

There are two different hashes and they are not interchangeable.

* **`gameHash` (the record's) is not a plane hash at all.** It is the engine's
  JS sync checksum, `hash.rs:7-22`: `1.0 + sum_p [ id_hash * (troops +
  tiles_owned) ] + sum_units unit_hash`, with `id_hash = |simple_hash(id)|`
  (`util.rs`'s `simple_hash` ends in `.abs()`), evaluated in `f64`.
  Re-evaluated from the record's own `(id, troops, tiles)` it reproduces
  `1299/1299` - so that *is* what the field is. No serializer over the owner
  plane can match it, because it never reads the plane:
  `ofcuda_hash`'s FNV over the record's owner plane vs `gameHash` = **0/1299**.
  The driver no longer claims `0/1001` as a failure; it says what the two
  objects are.
* **What is reproducible from composed state:** the FNV-1a-64 state hash over
  the composed ownership plane. Because the composed claims equal the engine's
  claims (section 2), the plane rebuilt from them is **word-for-word identical**
  to the engine's next-tick plane (`1299/1299`) and therefore
  `FNV(composed) == FNV(engine plane)` (`1299/1299`). It is a hash of the
  composed state, and it is exact - but its countervalue in this dump is the
  plane, not `gameHash`; there is no `stateHash` field in `fresh.ndjson`.
  `ofcuda_hash`'s own 200/200 is device-vs-host over the same
  `ofcuda_hash/oracle` dump, where the oracle *generates* `state_hash =
  fnv1a64_u16_le(plane)` (`oracle/src/main.rs:246`) - a serializer agreement,
  not a third engine hash.
* Not reproduced: a `gameHash` computed from *composed* state, because its
  inputs are player-level `troops`/`tiles`, and composed player troops are
  still record-seeded (player-level troop flows - `send_attack`'s
  `remove_troops` and the survivors' return - are outside the composed tick).

## 4. Change made outside this crate

`ofcuda_tick/src/core_impl.rs` (+ its verbatim device copy, re-synced with
`bash sync_device_core.sh`, whose invariant test passes):

* `HEAP_CAP 256 -> 1024`. The engine's heap is unbounded; 256 was truncating
  the composed window (13180 dropped candidates, section 2). Measured peak in
  this window is **851** and the run now reports `drops = 0`, so 1024 is not
  binding here. The counter is the point: an overflow is now **reported**
  rather than silent.
* `STATE_HEAP_CAP = 256` added and `STATE_WORDS = 6 + 2*STATE_HEAP_CAP` left
  **unchanged at 518**, so the device module's serialized layout, its thread
  count and its own recorded cases (max 211 enqueues, `heap_tie_break` etc.)
  are untouched. Only the host-side array grew: 2 KiB -> 8 KiB per heap.
* `Heap::drops`/`Heap::peak` added (measurement only).
* `ofcuda_tick`'s own 8 lib tests pass after the change (`cargo test --lib`,
  private `CARGO_TARGET_DIR`). Its **GPU** binary was not rebuilt or re-run
  here, so its 200/200 device verification is unverified after this change -
  only the heap array size moved.

No other crate was changed. `ofcuda_hash`, `ofcuda_econ` and `ofcuda_cluster`
are used as-is.

## 5. What remains outside the proven region

**Measured (counted, in the run above)**
* 2598/2598 claim-order pairs, 1299 transitions, no mismatch;
* 24/24 non-merge creations and 12/12 merges have `floor(f64) ==` the record;
* 24 starved deaths, 0 heap-dry retreats, 12 lingering no-op ticks, 0
  unexplained evictions, 0 unusable record entries, 0 heap drops;
* `gameHash` formula 1299/1299; composed plane 1299/1299 word-for-word.

**Assumed / exogenous**
* `ownedOrder` is treated as the claim log, i.e. claims append in order;
* the dump's `borderOrder` is the engine's `OrderedTiles` iteration order;
* one PRNG per attack at `SEED = 123` (`ofcuda_tick`'s constant, reproduced);
* the record supplies *which* attacks exist and *when* they are created; the
  crate never invents an attack, it aligns to the record's attack list
  (a creation is taken from the record's entry and given the computed start).
  A window may therefore not start mid-attack: `--t0 548` on the same dump
  fails immediately, because the attack's carried heap cannot be reconstructed
  from a record. The proven windows start at or before the first attack (318)
  or at the start of the dump.

**Not measured / unported**
* player-vs-player attacks: the whole window is terra-nullius expansion
  (`targetSmallId == 0`), so `num_adjacent * 2.0` is the only budget body
  exercised; the defender-troop body (`config.rs:488-496`) is unit-tested only;
* `refresh_to_conquer` mid-tick (restocking when the heap runs dry): the heap
  never runs dry in this window (`retreated = 0`), so the path is wired but
  unexercised - note it needs a fast-growing frontier, which is exactly the
  case the 256 heap could not have held;
* the fallout bit and defense-post modifiers (`game.rs:805-816`, `mag *=
  modifier`) are not modelled: `has_fallout` is false throughout;
* cluster capture is wired and imports the verified slice, but fires only on a
  player loss - none occurs in this window;
* attacks with a `source_tile` (boats/expansion) and `cancel_opposing`: not
  seen in this window (the merge only applies to `source_tile.is_none()`, which
  is what is ported);
* per-tile economy: measured negative by the econ slice, not re-litigated.
