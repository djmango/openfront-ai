# ofcuda_tick - the attack/expansion FRONTIER step on the GPU

Slice **L2** of the GPU env port: the frontier step only. `add_neighbors`' enqueue
loop, the priority computation and `FlatBinaryHeap`'s pop order - not the rest of
the tick, not the game loop, not the featurizer, not the reward.

    bash sync_device_core.sh          # re-copy the core into the device module
    ofcuda_env.sh cargo oxide run     # GPU proof (launches frontier_step)
    ofcuda_env.sh cargo test --offline --lib   # CPU-side tests
    ofcuda_env.sh cargo build --release --offline --bin ofcuda_tick_cpu  # no CUDA at all

## Layout

| file | role |
| --- | --- |
| `src/core_impl.rs` | the frontier core: sfc32 `Prng`, `neighbors4`, `mag2_from_terrain`, `priority_f32`/`priority_f64`, `Heap`, `frontier_enqueue`. Canonical source. |
| `src/main.rs` | `#[cuda_module] mod kernels` holds a **verbatim copy** of `core_impl.rs` plus the `frontier_step` kernel; `fn main` builds the inputs, launches, cross-checks. |
| `src/lib.rs` | host side: dump reconstruction (`recorded_cases`), the CPU reference, the f32-vs-f64 and FMA measurements, tests. |
| `src/bin/cpu.rs` | the reference runnable with no CUDA in the dependency graph. |
| `cases/` | the preserved `ownedOrder` dumps, one pair per recorded case. |
| `comparison_gpu.txt` | the full output of the last GPU run. |

`core_impl.rs` is **copied**, not `include!`d, into the device module: cuda-oxide's
`#[cuda_module]` attribute macro runs on the module before an `include!` inside it
expands, so the kernel silently fails to generate. The copy is not on trust - the
`device_core_is_a_verbatim_copy` test asserts the whole file appears verbatim in
`main.rs`, and was checked to FAIL when the copy was deliberately perturbed.

## The kernels

`engine_life_claims` is the one that matters: **one thread per global claim
index**, no atomics. Thread `j` replays the attack's whole life on the device -
`EngineAttack::init` at the init tick (the refresh that fills `to_conquer`), then
one `AttackExecution::tick` per pop tick - and writes the tile that was popped at
global claim index `j`. Nothing is carried in from the host except the case's
own inputs (plane, terrain, per-tick border/budget/pop-tick) and nothing comes
back except the claims, their f32 priority bits and the attack's final state, one
word per thread. That shape is the point: claim #j can come from a later tick
than its neighbours' claims because the heap survives the tick boundary.

`frontier_step` / `compose_tick` remain as the **one-pass control** (frontier
rebuilt from the pre-tick border, no in-tick re-enqueue) - the model this crate
shipped before the fix, kept so its failure is measured, not narrated.

The whole heap and both claim arrays are in local memory (the older PTX line is
`.local .align 8 .b8 __local_depot0[1064]`); `HEAP_CAP = 128` and
`CLAIM_CAP = 256` keep that at a few KiB per thread.

## Input: reconstructed from dumps, not synthesised

* map plane: `openfront/resources/maps/pangaea/map.bin`, 1000x1000, tick-independent.
* border set, `owned_mine`, `owned_any`: reconstructed from each case's preserved
  `ownedOrder` dump (`cases/*.ndjson`, copies of `/tmp/ord325.*` and
  `/tmp/ownb004.*`).
* `priority_tick` = `claim_tick - 1` for both cases (measured from the dumps, not
  assumed from the tick numbers).
* stream cursor = 0 for both cases (measured: the dumps' draws only line up with
  the stream at position 0).

## Measured vs assumed

**Measured** (all reproducible from `comparison_gpu.txt`, which the run writes):

* **The engine's own loop reproduces both grounds bit-exactly, on the GPU and on
  the CPU reference** - same shared `engine_tick`, so the CUDA lowering is what
  the host evaluates:
  * contested case: 27/27 claims, `838544` at index 23, priorities 27/27;
  * tick window: 133/133 claims in engine order and **32/32 per-tick FNV-1a-64
    state hashes**.
* GPU vs CPU on the device kernel: claims identical (first difference: none), the
  pops' f32 priority bits identical (27/27), and the **carried state after the
  life identical word for word** (PRNG `s0..s3` + `calls`, heap length and all
  512 heap words) - a device that agreed on the claims but not on the heap would
  be a different program.
* The device hashes the plane it implies itself for the window's last tick:
  `0x52735b9ab5848067`, equal to the engine's recorded hash.
* Controls, counted, on the same kernel: with in-tick `add_neighbors` off
  **20/32** ticks and first mismatch **320**; with the per-tick budget draw off
  **20/32** and first mismatch **320**. Both controls are the dead worker's
  previous best, so the fix is exactly the delta between 20/32 and 32/32.
* The one-pass model this crate shipped (kept in the run as the control): 27-claim
  contested run differs at index **23** - the second-ring tile `838544` is never
  popped - and the window is 20/32.
* `f32` vs the engine's f64 evaluation and the single FMA contraction in
  `frontier_step`: unchanged from before, 1219/1219 inert.
* Heap tie-breaks: the installed `Heap` reproduces `flat_heap.rs`'s own test
  vector `[10, 20, 30, 40] @ 1.0 -> 10, 40, 30, 20`.
* Draw accounting per tick is exact in DRAW COUNT: the contested run's stream
  makes 269 draws = 12 warm-ups (`prng.rs:30-32`) + 1 budget draw + 256 enqueue
  draws; the window's 341 = 12 + 13 budget draws + 316 enqueue draws. So the
  "one extra PRNG draw per tick before the pop loop" rule is measured, not
  assumed, albeit against the recorded claim counts rather than a recorded
  stream (the decomposition holds because `engine_tick` draws only in
  `Prng::budget_draw` - one per tick - and once per enqueue in
  `add_neighbors`/`refresh_to_conquer`, `core_impl.rs:592`, `:377`, `:468`).

**Assumed** (stated, not proven here):

* **The per-tick claim budget is EXOGENOUS.** The engine's real budget is a
  FLOAT, `attack_tiles_per_tick(troop_count, owner_type, target_is_player,
  defender_troops, border_size + random.next_int(0, 5))`, decremented per claim
  by the terrain/troop-dependent `tiles_used` (`attack.rs:239-254`, `313`).
  Troop growth and gold income are a different worker's slice and are not in this
  crate, so the run takes the **observed per-tick claim count from the case file**
  as the budget (27 for the contested tick; 8,9,9,9,9,10,10,12,11,10,12,12,12 for
  the window) and only spends the draw on the stream. A model that computed the
  budget would need the troop state the record does not carry.
* that the dump's `ownedOrder` at `claim_tick - 1` is exactly the border set the
  engine had when it built this frontier.
* the init tick (`init_tick = first_pop_tick - 1`) - the record supports it (the
  reference dump's `attacks` list is empty at tick 317 and holds owner 2's land
  attack at 318), but the tick that ran `init` itself is not in the record.

**Resolved against the contested case** (was "assumed", now measured):
`owned_any` and `owned_mine` coincide in the two old cases only because no rival
is adjacent. On a border that has one, they are different sets, giving a
different enqueue count (211 vs 220), and the claim order first differs at index
3 - measured, not argued.

## The engine loop, rule by rule

Every rule the device kernel encodes, with its citation in
`openfront-ai/rust/engine/src/execution/attack.rs` (read, not remembered):

| rule | where | how it shows up here |
| --- | --- | --- |
| `to_conquer` **persists across ticks**; the attack object holds it and the PRNG stream | `attack.rs:17`, `:21` | `EngineAttack { heap, pr, .. }` is kept between `tick()` calls (`core_impl.rs`, `EngineAttack`) - the previous worker rebuilt it per tick, and that was the bug |
| `AttackExecution::init` = `refresh_to_conquer` only | `attack.rs:160-164`, `:1265-1272` | `EngineAttack::init`, one refresh, one tick earlier |
| the heap is refilled **only when empty**, then the tick ends (`retreat`) | `attack.rs:264-268` | the `heap.is_empty()` branch in `engine_tick`: refresh, `refilled = true`, `break` |
| `add_neighbors(tile_to_conquer, tick)` runs **inside** the pop loop, **before** the conquer | `attack.rs:292` | `engine_tick`'s in-tick `add_neighbors` call, gated by `in_tick_add_neighbors` (the control) |
| a pop is skipped unless the tile is still **terra nullius** AND has an **attacker-owned neighbour** | `attack.rs:284` | the two `continue` guards in `engine_tick` |
| a skipped pop consumes **nothing** | `attack.rs:284-288` | the guards `continue` before any budget accounting |
| one **extra PRNG draw per tick** before the pop loop, for the float budget | `attack.rs:239-254` | `pr.budget_draw()` at the top of `engine_tick`, gated by `consume_budget_draw` (the control) |
| a claim costs `tiles_used` from the budget, terrain/troop dependent | `attack.rs:313` | taken as the exogenous integer budget (see above) |
| `is_land` guard | `attack.rs:288` | `terrain[t] & 0x80` check |
| the tick parameter stamped into priorities is `game.ticks()` | `game.rs:3659` | `tick` argument of `tick()`; init uses `tick - 1`, the in-tick pass uses `tick` |
| neighbour order | platform `neighbors4` | `ORDER_NSWE` = 0 for both grounds - an inversion here was a months-long parity bug |

## Contested border (`cases/b007-t581-sid2.contested.json`)

Recorded player 2 (Maori Council) of `curr-b007-s3-pangaea` at ticks 581/582
with the **fixed** engine (`gen_case_files.py verify b007 <dump>` re-derives the
file from a fresh tip dump and checks the plane hash, so "fixed engine" is
checkable, not asserted).

| quantity | value |
| --- | --- |
| border tiles | 146 |
| border tiles with >= 1 rival neighbour | 7 |
| `owned_mine` (Maori) | 1832 |
| `owned_any` (everybody) | 14332 |
| `owned_any \ owned_mine` | 12500 |

**Claim order, GPU vs the engine** (verbatim from the run):

    ENGINE recorded (dump, tick 582):
      [838541, 885551, 884539, 868569, 884554, 863570, 873525, 838547, 840553,
       838539, 838548, 841561, 839538, 840531, 844565, 843529, 842563, 848567,
       849568, 838543, 838545, 846566, 839550, 838544, 843564, 840557, 847526]
    ENG-LOOP gpu (engine_life_claims, one thread per claim index):
      [838541, 885551, 884539, 868569, 884554, 863570, 873525, 838547, 840553,
       838539, 838548, 841561, 839538, 840531, 844565, 843529, 842563, 848567,
       849568, 838543, 838545, 846566, 839550, 838544, 843564, 840557, 847526]
    ENG-LOOP cpu (same core):
      [ ... identical, 27/27 ... ]

* **First differing index vs the engine: gpu `none (identical)`, cpu
  `none (identical)`, one-pass model `23`.** Claim #23 is `838544` on all three:
  engine `838544`, gpu `838544`, cpu `838544`.
* So the contested claim is settled: `838544` is a **second-ring** tile whose
  neighbours `838543`/`838545` the engine claims 3 pops earlier; the carried heap
  plus the in-tick re-enqueue is what puts it in the queue inside the same tick.
  The one-pass model never pops it at all.
* The conflated-set bug is still measurable here (enqueue 211 -> 220, first
  difference at index 3), so the eligibility rule is not the explanation.

## Tick window (`cases/b002-t300-331.ticksteps.json`), re-measured

One `engine_life_claims` launch for the whole window (133 claims, 13 claim ticks
319..331, attack born at tick 317), then the per-tick planes hashed under the
**fixed** FNV contract (FNV-1a 64, offset `0xcbf29ce484222325`, prime
`0x100000001b3`, 1000x1000 u16 as little-endian bytes, lowercase `0x` + 16 hex
digits).

* **32/32 ticks match the engine's per-tick state hash** - 300..318 (no-op ticks)
  and **319..331 (every claim tick)** - on the GPU and on the CPU reference.
  First mismatching tick: **none (all matched)**.
* Matched ticks: `300,301,...,331` (all 32).
* The engine's recorded claim sets are reproduced claim-for-claim (133/133, first
  difference none) and the device's own FNV over the plane its claims imply at
  tick 331 is `0x52735b9ab5848067` = the recorded hash.
* **Where the previous 20/32 came from, and the cause of the old first mismatch
  at 320**: with in-tick `add_neighbors` off, the model reproduces 20/32 and
  first fails at 320 (the dead worker's number, reproduced by this run as a
  control). At 320 the one-pass model claims `[460261, 453261, 453254, 460254,
  461260, 457262, 455253, 458262, 461258]` against the engine's `[461257,
  461259, 461258, 457262, 452259, 460261, 453254, 458262, 455253]` - missing 3
  tiles `[461257, 461259, 452259]`, claiming 3 that the engine does not
  `[453261, 460254, 461260]`. The missing tiles' provenance, printed by the run:
  2 of the 3 are adjacent only to a tile claimed **in the same tick**, 1 only to
  a pre-tick border tile, 0 reachable from nothing - i.e. they are in the queue
  only because the pop loop enqueues as it pops. That is the whole cause: the
  missing in-tick `add_neighbors`, on top of the missing persistent heap.
* Old `compose_tick` tally, kept as the control: 20/32, first mismatch 320.

## Result

* `PROOF claim_order_and_priorities_reproduced true`, exit 0. Both grounds are
  reproduced by the engine's own loop, on the GPU, with the CPU reference on the
  same code path - and the controls that fail (20/32) are the previous state, so
  the improvement is counted, not asserted.
* What is still not modelled, deliberately: troop counts
  (`troop_count < 1.0 -> kill_attack`, `attack.rs:258-262`) and the float budget
  computation, both of which need state a different worker owns. They are stated
  as assumptions above, not quietly faked.

## The neighbour-order finding

The port takes the border iteration order as an **explicit input** (`ORDER_WENS`
= 1, `ORDER_NSWE` = 0), because the two recorded dumps disagree:

* `b002-t319.native.ndjson` (the tick-319 case this task specifies) is
  reproduced **only** under `W,E,N,S`.
* `b002-t319.ts.ndjson` / `b004-t309.ts.ndjson` are reproduced **only** under
  `N,S,W,E`.

Both are printed by the run, so whichever engine revision a caller means, the
order it needs is visible and selectable. The required target order in the task
(`[460260, 459254, ...]`) is the `W,E,N,S` one - the order the native dump was
recorded with.

## How to re-run

    bash sync_device_core.sh                       # if core_impl.rs changed
    ofcuda_env.sh cargo oxide run                  # GPU run, writes comparison_gpu.txt
    ofcuda_env.sh cargo test --offline --lib       # 8 tests, both findings locked in
    python3 gen_case_files.py verify b002 <dump>   # case files vs a fresh fixed-engine dump
