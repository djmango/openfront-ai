# MATRIX_REPORT — the composed CUDA tick driven end to end on many maps and many agent counts

Crate: `/opt/data/workspaces/skg/ofcuda_matrix` (own workspace, empty `[workspace]` table,
read-only path deps on `ofcuda_map` / `ofcuda_prng` / `ofcuda_tick` / `ofcuda_env`).
Nothing outside this crate was edited. Ground truth is always `openfront-engine`, linked
directly by `ofcuda_matrix/oracle`.

Binary: `/opt/data/workspaces/skg/.target-ofcuda-matrix/release/ofcuda_matrix`
sha256 prefix `91ad59acd05a3ec5`. Oracle: `/opt/data/workspaces/skg/ofcuda_matrix/oracle/target/release/oracle`
(`ab5c136c73248ca2`, dump format v2).

---

## 1. The pipeline, and what is device-computed vs host-computed

`ofcuda_matrix --map <name> --agents <N> --nations {0|1|2} --ticks <T> --out <dir>`
(also `--cells-file <f>`, `--dump-planes`, `--seed`, `--oracle-dir`, `--refs-dir`).

Init = the map's terrain plane (`ofcuda_map`, bytes read on the host) + the engine-exact
spawn/owned sets for N agents (`ofcuda_spawn` makes the engine reference, `ofcuda_prng
spawnall --dump` computes the port's sets **on the GPU**) + the composed tick
(`ofcuda_tick`'s canonical `core_impl.rs`, `include!`d into this crate's `kernels.rs`).

**DEVICE-computed** (CUDA kernels, this crate's `src/kernels.rs`, which `include!`s
`ofcuda_tick/src/core_impl.rs` — the device core is not copied or rewritten):

* the spawn/initial owned sets (`spawnall`, `ofcuda_prng`),
* every tick: claim selection and priority ordering, heap push/pop, the PRNG budget
  draws, border insert/remove, all plane writes,
* the per-tick owned-tile counts (`plane_counts`),
* the owner plane hash compared against the engine (`plane_hash`); the host recomputes
  the same hash over the copied plane as `HASH_INTERNAL` redundancy, and any disagreement
  is reported.

**HOST-computed / record-driven** (this is the honest limit of the composition):

* the map bytes and the terrain upload (`ofcuda_map`),
* the engine reference dump (spawn output, per-player owned/border sets, per-boundary
  claims, the engine's own plane hash) — the ground truth,
* **the attack lifecycle schedule**: attack *creation*, *re-creation* and *eviction* timing
  is taken from the engine record (`INIT_ATTACK` / `REINIT` / `ENGINE_EVICTION` lines). The
  device has no attack-creation path of its own; it evolves an attack once the driver has
  created it, and a re-issued attack (changed `attack_id`) has its frontier re-built by the
  driver **after** that boundary's tick, in the engine's order (see §4.2),
* **the per-player border tile sets** staged into `oborder`/`ob_meta` from the record
  (`prev.borders`). Border membership is therefore host-supplied, not device-derived —
  the device consumes it in `attack_init` and the refresh path,
* the comparison/verification itself (hash and count comparison, divergence reporting),
* the rendering (`mk1000.py` decimation + the committed render scripts).

---

## 2. Spawn ceiling and plateau, per map

Two different quantities. `ceiling` = the largest N at which **every** bot places
(spawned == N). `plateau` = how many bots are actually placed as N grows past the
ceiling. Both are measured on `openfront-engine` through this crate's oracle, seed
`parity`, difficulty `Easy`, `human_agents 1`, `nations 0`; `min_distance_between_players()`
is 30 (constant, difficulty-independent) and the per-bot attempt budget is
`MAX_SPAWN_TRIES = 1000` (`openfront-ai/rust/engine/src/execution/spawn_util.rs:84`).

| map | dims (tiles) | land (bytes) | **ceiling** | **plateau** (N=5000) |
|---|---|---|---|---|
| pangaea | 1000x1000 (1.0M) | 420,335 | **488** | 569 |
| world | 2000x1000 (2.0M) | 651,569 | **749** | 934 |
| passage | 6000x400 (2.4M) | 803,994 | **988** | 1,216 |
| amazonriver | 5536x276 (1.53M) | 1,150,819 | **1,474** | 1,677 |
| mississippiriver | 400x4200 (1.68M) | 1,500,944 | **1,865** | 2,099 |
| africa | 1948x2032 (3.96M) | 2,183,279 | **2,393** | 2,860 |
| giantworldmap | 4108x1948 (8.0M) | 2,335,403 | **2,582** | 3,180 |
| europe | 2904x1672 (4.86M) | 2,345,907 | **2,687** | 3,155 |
| thebox | 2048x2048 (4.19M) | 4,194,304 | **4,749** | 4,988 |
| onion | 512x512 (0.26M) | 210,555 | **285** | 313 |

* The ceiling is **not flat and not a global cap** — it depends on the map, exactly as
  predicted (thebox highest, onion lowest), because `min_distance_between_players() = 30`
  limits the eligible fraction of the draw pool and the 1000-attempt cap then binds at a
  map-dependent N.
* Ceiling and plateau are different answers to different questions: on pangaea the
  ceiling is 488 while the plateau is 569 (+17%); on giantworldmap the plateau (3,180)
  exceeds europe's (3,155) even though its ceiling (2,582) is *lower*.
* Pristine reproduction of the user's independent pangaea plateau numbers:
  N=700 → 557, N=1200 → 565, N=2000 → 566, N=3000 → 569, N=5000 → 569.

### 2.1 The 449 defect, and the 488 reconciliation

The first ceiling scan reported a flat **449 for nine of ten maps** (onion 285). That was a
scan bug, and the user's reading of it was correct — 449 is the last grid point of the
coarse walk, not a ceiling:

* `ceiling_for` walked 1, 5, 9, 13, 17, 33, … 449 (+64 steps), then stepped to **513**,
  which is `> max_n` (512), so the loop exited with `hi = None` and returned `lo`, i.e.
  the last grid point it had probed. It never probed 450…512, so **488 was never
  measured**. File:line at the time of the defect: `oracle/src/main.rs:441-475`
  (`ceiling_for`); the walk step block is now `oracle/src/main.rs:458-470`.
* Fix: the walk now **always probes `max_n` itself** (the step is clamped to `max_n`) and
  doubles above 512 so the search stays cheap at large N; the bisect inside the last
  failing bracket then restores resolution. Nothing was tuned to make pangaea read 488 —
  the number falls out (proved by running the same fixed scan at `--max-n 8192`, where
  every map lands somewhere else entirely).
* Second trap, worth recording: `CARGO_TARGET_DIR` exported for the CUDA build leaked into
  the oracle's `cargo build`, so the *fixed* oracle binary was written to
  `.target-ofcuda-matrix/release/oracle` while `ofcuda_matrix/oracle/target/release/oracle`
  kept the stale one. A re-run "still giving 449" was that stale binary. Always
  `unset CARGO_TARGET_DIR` before building/running the oracle.
* Reconciliation with the independent `ofcuda_prng spawnall` measurement
  (pangaea, `--nations 0`): **488 = 488**, with the exact starve at 489 — the engine's
  replay in this crate's oracle at N=489 places 488 of 489 bots. The two implementations
  agree bit-exactly; the earlier 449 was mine alone.
* Config comparison the two paths share: `difficulty Easy` does **not** change
  `min_distance_between_players()` (constant 30) and does not change the attempt budget
  (1000/bot), `human_agents 1`, seed `parity`, `nations 0`. There is no player cap in the
  engine (no `max_players`-style constant); the flat 449 was scan resolution.

---

## 3. The matrix

Cells in the file (30): every one of the 10 maps at N=18, and N ∈ {2,7,18,64,488} spread
across maps so each N value is covered on ≥3 maps, plus `nations` Exact(1)/Exact(2) on
pangaea. **28 are driven and all 28 pass**; the 2 `nations > 0` cells are *skipped with a
reason* rather than failed (§5 — the spawn layer ports bots only). `ticks 60`,
`nations 0` unless stated, seed `parity`.

Plus a 5-cell **short-window control** (`tools/cells_short.txt`, `ticks 45`) closing the
window *before* the divergence event, which is what isolates the cause (§4).

Per-cell results, engine-verified (init owned sets, per-tick owned-tile counts, and the
full owner-plane hash every tick):

**28 / 28 driven cells pass** — every value below is the device against the engine
oracle, `ticks 60`. `claims` = per-boundary claimed-tile sets, `plane hash` = the full
owner plane every boundary, `counts` = per-player owned-tile counts, `troops` = per-attack
troop values, `re-creates` = engine-side attack re-creations followed, with the device's
post-init frontier compared against the engine's recorded `to_conquer`/`border_tiles`
sizes (the "(n agreed)" column).

Every cell is `init yes` (init plane hash **and** per-player owned sets), the seed is
`parity`, and `nations 0` unless the row says otherwise.

| map | WxH | N | init | claims | plane hash | counts | troops | re-creates |
|---|---|---|---|---|---|---|---|---|
| pangaea | 1000x1000 | 2 | yes | 14/14 | 60/60 | 180/180 | 12/12 | 0 (0 agreed) |
| pangaea | 1000x1000 | 7 | yes | 134/134 | 60/60 | 480/480 | 128/128 | 0 (0 agreed) |
| pangaea | 1000x1000 | 18 | yes | 447/447 | 60/60 | 1140/1140 | 432/432 | 2 (2 agreed) |
| pangaea | 1000x1000 | 64 | yes | 1601/1601 | 60/60 | 3900/3900 | 1546/1546 | 5 (5 agreed) |
| pangaea | 1000x1000 | 488 | yes | 14069/14069 | 60/60 | 29340/29340 | 13632/13632 | 43 (43 agreed) |
| africa | 1948x2032 | 2 | yes | 14/14 | 60/60 | 180/180 | 12/12 | 0 (0 agreed) |
| africa | 1948x2032 | 18 | yes | 447/447 | 60/60 | 1140/1140 | 432/432 | 2 (2 agreed) |
| africa | 1948x2032 | 64 | yes | 1601/1601 | 60/60 | 3900/3900 | 1546/1546 | 5 (5 agreed) |
| africa | 1948x2032 | 488 | yes | 14069/14069 | 60/60 | 29340/29340 | 13632/13632 | 44 (44 agreed) |
| onion | 512x512 | 7 | yes | 134/134 | 60/60 | 480/480 | 128/128 | 0 (0 agreed) |
| onion | 512x512 | 18 | yes | 447/447 | 60/60 | 1140/1140 | 432/432 | 2 (2 agreed) |
| onion | 512x512 | 64 | yes | 1601/1601 | 60/60 | 3900/3900 | 1546/1546 | 5 (5 agreed) |
| onion | 512x512 | 488 | yes | 9031/9031 | 60/60 | 29340/29340 | 8751/8751 | 32 (32 agreed) |
| europe | 2904x1672 | 2 | yes | 14/14 | 60/60 | 180/180 | 12/12 | 0 (0 agreed) |
| europe | 2904x1672 | 7 | yes | 134/134 | 60/60 | 480/480 | 128/128 | 0 (0 agreed) |
| europe | 2904x1672 | 18 | yes | 447/447 | 60/60 | 1140/1140 | 432/432 | 2 (2 agreed) |
| world | 2000x1000 | 18 | yes | 447/447 | 60/60 | 1140/1140 | 432/432 | 2 (2 agreed) |
| world | 2000x1000 | 64 | yes | 1601/1601 | 60/60 | 3900/3900 | 1546/1546 | 5 (5 agreed) |
| amazonriver | 5536x276 | 18 | yes | 447/447 | 60/60 | 1140/1140 | 432/432 | 1 (1 agreed) |
| amazonriver | 5536x276 | 488 | yes | 14069/14069 | 60/60 | 29340/29340 | 13632/13632 | 34 (34 agreed) |
| thebox | 2048x2048 | 18 | yes | 447/447 | 60/60 | 1140/1140 | 432/432 | 1 (1 agreed) |
| thebox | 2048x2048 | 488 | yes | 14069/14069 | 60/60 | 29340/29340 | 13632/13632 | 33 (33 agreed) |
| giantworldmap | 4108x1948 | 7 | yes | 134/134 | 60/60 | 480/480 | 128/128 | 0 (0 agreed) |
| giantworldmap | 4108x1948 | 18 | yes | 447/447 | 60/60 | 1140/1140 | 432/432 | 2 (2 agreed) |
| passage | 6000x400 | 18 | yes | 447/447 | 60/60 | 1140/1140 | 432/432 | 2 (2 agreed) |
| passage | 6000x400 | 488 | yes | 14069/14069 | 60/60 | 29340/29340 | 13632/13632 | 41 (41 agreed) |
| mississippiriver | 400x4200 | 18 | yes | 447/447 | 60/60 | 1140/1140 | 432/432 | 2 (2 agreed) |
| mississippiriver | 400x4200 | 488 | yes | 14069/14069 | 60/60 | 29340/29340 | 13632/13632 | 41 (41 agreed) |
| pangaea (nations=1) | 1000x1000 | 18 | **skipped** | – | – | – | – | – |
| pangaea (nations=2) | 1000x1000 | 18 | **skipped** | – | – | – | – | – |

Matrix totals over the 28 driven cells: **104,897/104,897** per-boundary claim sets,
**1,680/1,680** full-plane hashes (28 × 60 boundaries), **234,840/234,840** per-player
owned counts, **101,595/101,595** troop values, **306/306** re-created frontiers matching
the engine's recorded sizes, **0** engine evictions and **0** churn-skipped owners.

`onion N=488` is the starve cell (§5): the port does not place every bot, the engine does
not either, and the cells still match tile for tile (9,031 claims) — that is the point of
including it.

The 5-cell `ticks 45` control (window closed before boundary 49) also passes 100%
(`out/short/cells.tsv`: 225/225 plane hashes, 2,479/2,479 claims, 8,415/8,415 counts,
2,360/2,360 troops, 0 re-creations in the window).

---

## 4. The one divergence that was there, and the fix

**No cell diverges any more, in the tile kernels or anywhere else.** Before the fix below,
every cell with N≥18 was bit-exact until **boundary 49 (engine tick 52)**, and then
diverged on `attack owner 9` (player 9, target 0 = terra nullius) — identically on 7
different maps including 5536x276 and 400x4200 strips, so the bug was in the code, not the
map. The cells that never reached that tick (N=2, N=7 on two maps, and the 5-cell
`ticks 45` control) passed **100%**, plane hash included; that is what isolated the cause
to a mid-flight attack event rather than tick drift.

### 4.1 The engine event, as the record shows it

1. At engine tick 52 the engine **re-creates / re-registers player 9's attack**:
   `AttackExecution::init` (`attack.rs:136-150`) sets `self.troops = start`, where `start`
   is `self.start_troops` — the figure the bot AI passed through
   `game.add_land_attack_from(owner, target, start_troops, source_tile)`
   (`game.rs:1469-1484`) — or `game.wire.attack_amount(...)` (`core/config.rs:461-468`:
   5% of the owner's troops for a `Bot`, 20% otherwise). The same `init` mints a **fresh
   `attack_id`** (`attack.rs:156`), re-registers the exec (`attack.rs:158`, visible in the
   record as the attack **moving to the end of `live_attacks()`** at boundary 49 — it is
   first at boundaries 47–48) and builds its frontier from the source tile
   (`attack.rs:160-164`, `add_neighbors`) when it has one, falling back to
   `refresh_to_conquer` (`attack.rs:1265`) over the owner's `border_tiles` when it does not.
2. The engine's recorded troops for that attack jump
   `1279.5697607467282 -> 1507.6145973769035` (`0x4093fe476f5c76f8 -> 0x40978e755903c808`).
   None of the engine's observed attack amounts is a multiple of 0.05, so none came from
   `attack_amount()` with an integer `i32`: the engine's bot AI passed them in explicitly
   (`game.rs:1477`), so the port must **reproduce** them, not recompute them.
3. Before the fix, that tick claimed **7** tiles for player 9 in the engine and **9** on the
   device (`CLAIM_DIVERGENCE boundary 49 (engine tick 52) player 9: engine 7 tiles, device
   9 tiles; first index 7; cause: tile sets differ: engine-only [] device-only [1858269,
   1901828]`), 303 vs 302 tiles across all players, so the plane hash differed from that
   boundary on. Up to and including boundary 48 the device tracked the engine *bit-exactly*
   (europe N=18, owner 9 troops `b47 eng==dev 0x4094de476f5c76f8`, `b48 eng==dev
   0x4093fe476f5c76f8`).

### 4.2 The fix: follow the re-creation, in the engine's order

Two things were missing from the composition, and only the second one moves the plane.

**(i) The record did not carry the identity of the attack object.** Re-creation is exactly
identifiable: a *live* `(owner, target)` whose `attack_id` has **changed** is an engine-side
re-issue, because the id is minted per `AttackExecution::init` and coalesced by
`merge_outgoing_land_attacks` (`game.rs:2122-2156`). Inferring it from the troops (as the
old code did) is a guess; the id is a fact. The oracle dump is now **v2**: the `ATTACK` row
carries `source_tile`, the frontier sizes and the id —
`ATTACK {b} {owner} {target} {troops:#018x} {source_tile} {to_conquer_len}
{border_tile_count} {attack_id}` (`oracle/src/main.rs:405-425`) — and the driver keys
re-creation on the id (`src/main.rs:707-731`).

**(ii) The order was wrong.** `execute_next_tick` (`game.rs:3657-3700`) ticks every
*existing* exec and only then initialises the new ones. So for the tick an attack is
re-issued in:

* the tick still belongs to the **old** exec and the **old** frontier, and
* the new exec's frontier is built from the world as of the **end** of that tick
  (`refresh_to_conquer` over the owner's border set at that boundary — measured over all
  110,044 `ATTACK` rows in the matrix, **every** attack carries `source_tile == -1`, so the
  `add_neighbors` path at `attack.rs:160-164` is not the one taken anywhere here; the
  record carries the tile anyway so the path stays expressible).

The old code did the opposite: it re-seeded the recorded troops **before** the tick
(`TROOP_REALLOC`), handing the stale frontier the new, larger budget — which is where the
2 extra tiles came from. The driver now re-seeds nothing before the tick, and for a
re-created slot re-runs `attack_init` **after** the tick, against the boundary's own border
set and plane (`src/main.rs:897-995`, `pending_reinit`).

The device's re-init is not taken on trust: its post-init frontier sizes are compared
against the engine's recorded `to_conquer` / `border_tile_count`, and **all 306
re-creations in the matrix agree** (`out/matrix/cells.tsv`, columns `reinits` /
`reinit_checked` / `reinit_agree`). Example record (africa N=18):

```
REINIT boundary 49 (engine re-created the attack in the tick stamped 52): owner 9 target 0
id 47l4ldxb -> 75imsa56 troops 0x4096fe476f5c76f8 -> 0x40a19c7a4e81b532 |
device post-init heap 106 vs engine 106 border 78 vs engine 78 -> AGREE
REINIT boundary 52 (engine re-created the attack in the tick stamped 55): owner 13 target 0
id 16ezumi3 -> cgccxjs5 troops 0x409a9fe632226320 -> 0x409e00c8e40ca7f0 |
device post-init heap 116 vs engine 116 border 88 vs engine 88 -> AGREE
```

and the same cell's result block:

```
tick claims matched 447/447     hash matched 60/60
owned counts matched 1140/1140  troops matched 432/432
engine re-creates followed 2    device frontier agreed with the engine's recorded
                                to_conquer/border_tiles 2/2
first divergence: none
```

The cause is reproducible on demand, not asserted. `OFCUDA_MATRIX_PRETICK_REALLOC=1`
re-instates the old ordering (re-seed the re-issued attack's troops **before** the tick) and
the same single cell then fails exactly where it used to, caught by the re-init check:

```
$ OFCUDA_MATRIX_PRETICK_REALLOC=1 ... --map europe --agents 18 --ticks 60
europe 2904x1672 18 60 init yes/yes  claims 290/291  hash 48/49  counts 930/931
  boundary 49: re-created attack owner 9 target 0: device frontier (heap 112 border 88)
               != engine (heap 116 border 90)
cells: 0 passed, 1 failed/not-driven, 0 skipped
```

The pre-tick re-seed makes the **old** attack claim one tile more than the engine at
boundary 49, which changes that boundary's plane and therefore the border set the
re-creation then walks — so the new frontier comes out 112/88 where the engine's record says
116/90. The pre-fix build's "+2 tiles" at that boundary is the same defect one round earlier
(there the post-tick frontier rebuild did not exist either). Both directions are measured.

The kernels themselves were not touched: the composition (record + init ordering) was
wrong, which is why the same fix clears 7 maps at once.

### 4.3 Harness defects found and fixed while chasing this (each one could have faked a result)

* `main.rs:664-673` (old): the slot binder had an **owner-only fallback** (`loose`) that
  bound any same-owner slot and re-stamped the host struct's troop bits without touching
  the device — it hid an engine-side re-init instead of reporting it. Replaced by strict
  `(owner, target)` matching with no owner-only fallback.
* the spawn reference was generated **without `--nations`**, so `nations 1|2` cells
  compared the engine's `Exact(n)` stream against the port's `0` stream.
* the **init dump filename omitted `nations`**, so the `Exact(2)` cell compared against a
  `nations 0` port init (reported as 20 `INIT_MISMATCH` players). Both now carry `nat{n}`.
* 20 cells died with `load module: failed to read .../release/ofcuda_matrix (deleted)` —
  the CUDA module loader reads the **executable path**, and I rebuilt the binary while the
  matrix was running. Infra race, not a port bug; re-run cleanly with no rebuild in flight.
* `tools/render_cells.sh` had a repo-relative path for the committed render scripts (they
  live in `openfront-ai/scripts`, not beside the crate) and did per-cell work before
  validating its inputs; it now uses absolute paths from `REPO_ROOT` and fails once, up
  front, listing every missing input.
* `tools/render_cells.sh` could **produce zero movies and exit 0**. It read its cell list
  from `$1`, so the documented two-argument form (and a call with the render dir first)
  handed `read` a **directory**: `read: 0: read error: Is a directory` on stderr, the
  success banner on stdout, `render_dims.tsv` header-only, exit 0. Fixed: the cells argument
  must be a regular file (a directory is a hard error naming the swapped arguments), a cell
  list that parses to **0 cells** is a failure, the parsed count is printed in one line
  (`rendering N cells from <file>`), and after the loop the script asserts **#MP4 == #cells**,
  **≥1 still per cell** and **render_dims.tsv has data rows** — any of them fails the run.
  Verified both ways: a directory and an empty cell file now exit non-zero, and a real run
  prints `OK: 11 movies, 44 stills, 11 dims rows for 11 cells`.
* the `nations > 0` cells were reported as **failed/not-driven** (an INIT mismatch) when
  they are a spawn-layer gap; they are now classified `skipped` and counted separately
  (§5), so the summary line's failure count means tick failures only.

---

## 5. Cells NOT run, and why

* **N=488 where the map's ceiling is below 488** (onion: 285; world's is 749, so 488 < 749
  is fine): **onion N=488 is driven anyway, and passes** — 9,031/9,031 claims, 60/60 plane
  hashes, 32/32 re-created frontiers. The port reproduces the starve exactly (some bots
  never get a spawn tile, as in the engine) and the tick driver drives the players that did
  place; the cell's `note` reads `PORT FAILED TO PLACE SOME BOT`, so the starve is reported
  and never hidden inside a pass.
* **`nations = 1 | 2` (Exact) through the CUDA tick: SKIPPED, with a reason.** The spawn
  layer ports **bots only** (`ofcuda_spawn` / `spawnall`): with `nations > 0` the engine also
  seeds `PlayerType::Nation` players, whose spawn tick is *after* every bot's (they are
  chosen against the full bot plane) and whose footprint never reaches the reference file
  (`player 2 N 0b4eakcu 610150 2 52` — a count, no tiles). Boundary 0 therefore cannot be
  rebuilt from the port, which is a **spawn-layer** gap, not a tick gap. The driver now
  detects `nations > 0` before launching anything, writes the cell with
  `SKIPPED: nations=N not ported (...)`, counts it under `skipped` (not `failed`), and the
  run's summary line reads
  `cells: 28 passed, 0 failed/not-driven, 2 skipped (nations>0, spawn layer ports bots only)`.
  Before this it was run, failed at INIT and was reported as a **failed cell** — a
  spawn-layer gap wearing a tick failure's clothes, which is exactly the kind of health
  signal that must not be ambiguous.
* **N > 512** on any map: the matrix deliberately stops at 488 (the "last N where every bot
  places" on pangaea is 488; the ceilings above 512 are reported from the oracle, not
  driven through the CUDA tick). Driving N=2687 or N=4749 through the device tick is
  possible in principle (COLS=8192, MAX_SLOTS) but was not part of the requested counts.
* **nations = `disabled` / `default` through the CUDA tick**: only `0/1/2` were driven; the
  spawn-phase proof for disabled/default belongs to the spawn work, not this matrix.
* Cells where **attacks are already live at boundary 0** (spawn-phase nations): the driver
  refuses and reports `not drivable: N attack(s) already live at boundary 0 - the
  spawn-phase schedule is not modelled`. That is a deliberate, reported refusal.
* `manifest.json` land counts are **not** used anywhere: 4 maps (china, tierradelfuego,
  unitedstates, losangeles) undercount their own bytes, so land is always taken from the
  bytes. None of those 4 maps are in this matrix.

---

## 6. Artifacts

All paths relative to `/opt/data/workspaces/skg/ofcuda_matrix`.

**The claim: 28 driven cells, 0 divergences**

* `out/matrix/cells.tsv` — the machine-checkable matrix result (30 rows: 28 driven, 2
  skipped). Columns: `init_sets init_hash tick_match tick_total hash_match hash_total
  count_match count_total troops_match troops_total churn_skipped engine_evictions reinits
  reinit_checked reinit_agree dev_claims first_div note`.
* `out/matrix/<map>_n<N>_nat<n>_t60/result.txt` — per-cell evidence: INIT block (init plane
  hash, per-player owned sets, per-bot spawn tiles), the `REINIT ... -> AGREE` records for
  every re-created attack, `ENGINE_EVICTION` records (0 in this matrix), the `### RESULT`
  block, and the `first divergence: none` line.
* `out/matrix/oracle/*.dump` — the engine oracle dumps (**format v2**: `ATTACK` rows carry
  `source_tile`, `to_conquer_len`, `border_tile_count`, `attack_id`).
* `out/matrix/refs/` — the engine spawn references (`*.spawn.txt`) and the port's GPU spawn
  dump (`*.init.txt`) per cell.
* `out/short/cells.tsv` — the 5-cell `ticks 45` control (window closed before boundary 49).
* `out/renders/<cell>/{planes.bin,terrain.bin}` — the per-tick device owner planes (61
  frames each) plus terrain, dumped by `--dump-planes`.

**The renders and the contact sheet**

* `out/report/contact_sheet.png` — **all 11 render cells in one image, map + N captioned**
  (`magick montage`, 4 columns, 1584x1281, 3.0 MB).
* `out/report/*.mp4` — 11 movies, one per render cell, 61 frames each at 12 fps:
  `pangaea_n488_nat0_t60.mp4`, `pangaea_n2_nat0_t60.mp4`, `africa_n18_nat0_t60.mp4`,
  `europe_n18_nat0_t60.mp4`, `world_n18_nat0_t60.mp4`, `amazonriver_n18_nat0_t60.mp4`,
  `thebox_n64_nat0_t60.mp4`, `onion_n7_nat0_t60.mp4`, `giantworldmap_n7_nat0_t60.mp4`,
  `passage_n18_nat0_t60.mp4`, `mississippiriver_n18_nat0_t60.mp4`.
* `out/report/stills/<cell>_<frame>.png` — 44 stills, 4 per cell (frames 0, 15, 30, 60).
* `out/report/render_dims.tsv` — per cell `map W H aspect frames out_px terrain_ok`, with
  `terrain_ok yes` = the terrain bytes matched `W*H` from the map manifest. **Dims come from
  the manifest's `map.width`/`map.height` and the plane is never squared**:

  | cell | WxH | aspect | out px |
  |---|---|---|---|
  | pangaea_n488 / pangaea_n2 / thebox_n64 / onion_n7 | 1000x1000 / 2048x2048 / 512x512 | 1.0000 | 800x800 |
  | africa_n18 | 1948x2032 | 0.9587 | 767x800 |
  | europe_n18 | 2904x1672 | 1.7368 | 800x461 |
  | world_n18 | 2000x1000 | 2.0000 | 800x400 |
  | giantworldmap_n7 | 4108x1948 | 2.1088 | 800x379 |
  | **amazonriver_n18** | 5536x276 | **20.0580** | **800x39** (wide strip) |
  | **passage_n18** | 6000x400 | **15.0000** | **800x53** (wide strip) |
  | **mississippiriver_n18** | 400x4200 | **0.0952** | **76x800** (tall strip) |

  The three extreme maps — the ones that used to come out squashed — are the three that
  prove it: 20:1 and 15:1 strips come out wide, 1:10.5 comes out tall, and no cell is
  forced square (`stills/*_060.png` measured: 800x40, 800x53, 76x800 — one pixel of rounding
  versus the table's `out_px`).

**The report and the re-runnable commands**

* `MATRIX_REPORT.md` — this file (`<!-- MATRIX_TABLE -->` and `<!-- ARTIFACTS -->` filled).
* `tools/cells.txt` (30 cells), `tools/cells_short.txt` (5), `tools/cells_renders.txt` (11),
  `tools/contact_sheet_manifest.txt` (the 11 captions), `tools/render_cells.sh`,
  `tools/contact_sheet.py`, `tools/mk1000.py`.

## 7. Commands

```bash
# spawn ceiling (fixed walk; probes max_n, doubles above 512)
cd /opt/data/workspaces/skg/ofcuda_matrix && unset CARGO_TARGET_DIR
./oracle/target/release/oracle ceiling --maps pangaea,africa,europe,world,amazonriver,thebox,onion,giantworldmap,passage,mississippiriver \
    --nations 0 --max-n 8192 --out out/report/ceilings_fixed.tsv

# spawn plateau (bots actually placed)
./oracle/target/release/oracle plateau --maps pangaea --ns 700,1200,2000,3000,5000 --nations 0 --out out/report/plateau_pangaea.tsv
./oracle/target/release/oracle plateau --maps thebox,onion,amazonriver,europe,passage,mississippiriver,world,africa,giantworldmap,pangaea \
    --ns 800,2000,5000 --nations 0 --out out/report/plateau_maps.tsv

# the matrix (30 cells) and the short-window control (5 cells)
M=/opt/data/workspaces/skg/ofcuda_matrix
B=/opt/data/workspaces/skg/.target-ofcuda-matrix/release/ofcuda_matrix
bash /opt/data/workspaces/skg/ofcuda_env.sh $B --cells-file $M/tools/cells.txt       --out $M/out/matrix --ticks 60
bash /opt/data/workspaces/skg/ofcuda_env.sh $B --cells-file $M/tools/cells_short.txt --out $M/out/short  --ticks 45 \
     --oracle-dir $M/out/matrix/oracle --refs-dir $M/out/matrix/refs

# A/B control for §4.2: the OLD ordering (re-seed the re-issued attack's troops before the
# tick). Must FAIL on europe N=18 at boundary 49; the matrix is run WITHOUT it.
OFCUDA_MATRIX_PRETICK_REALLOC=1 bash /opt/data/workspaces/skg/ofcuda_env.sh $B \
     --map europe --agents 18 --ticks 60 --out /tmp/ab_control \
     --oracle-dir $M/out/matrix/oracle --refs-dir $M/out/matrix/refs

# the render cells (per-tick device planes dumped) then movies + stills + contact sheet
bash /opt/data/workspaces/skg/ofcuda_env.sh $B --cells-file $M/tools/cells_renders.txt \
     --out $M/out/renders --oracle-dir $M/out/matrix/oracle --refs-dir $M/out/matrix/refs --ticks 60 --dump-planes
# 3-arg form: <cells-file> <out-with-dumps> <report-dir>. It prints the cell count it
# parsed and FAILS (non-zero) if a movie/still is missing or render_dims.tsv is empty.
bash $M/tools/render_cells.sh $M/tools/cells_renders.txt $M/out/renders $M/out/report
python3 $M/tools/contact_sheet.py $M/tools/contact_sheet_manifest.txt $M/out/report/contact_sheet.png 4x
```

Build (CUDA crate, isolated target dir, one rebuild at a time):
```bash
cd /opt/data/workspaces/skg/ofcuda_matrix
export CARGO_TARGET_DIR=/opt/data/workspaces/skg/.target-ofcuda-matrix
nice -n 10 bash /opt/data/workspaces/skg/ofcuda_env.sh cargo oxide build -- --release
```
