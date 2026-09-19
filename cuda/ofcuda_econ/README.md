# ofcuda_econ — per-tick ECONOMY / STATE-PLANE update (slice "econ")

The third slice of the OpenFront tick, after `ofcuda_map` (map load) and
`ofcuda_tick` (frontier / conquest). This one is everything that changes
per-tile / per-player state each tick *outside* territory conquest, with the
income arithmetic on the GPU and a CPU reference on the **same code path**.

Crate: `/opt/data/workspaces/skg/ofcuda_econ` (new crate; nothing in
`rust/engine`, `rust/ofcore` or `rust/oftrain` was touched, and
`libopenfront_engine.so` / `PufferLib5/puffer` were never built or overwritten —
all builds here go to a private `CARGO_TARGET_DIR`).

## 1. What the slice contains, and where it is specified

The TypeScript is the authority; the Rust engine is the cross-check.

| Operation | TS (authority) | Rust engine (cross-check) |
|---|---|---|
| per-tick income block | `openfront/src/core/execution/PlayerExecution.ts:75-78` | `rust/engine/src/execution/player.rs:82-83` |
| `Config.maxTroops` | `openfront/src/core/configuration/Config.ts:791-797` | `rust/engine/src/core/config.rs:363-390` |
| `Config.troopIncreaseRate` | `Config.ts:825-856` | `core/config.rs:433-458` |
| `Config.goldAdditionRate` | `Config.ts:859-868` | `core/config.rs:477-481` |
| `PlayerImpl.addTroops` / `removeTroops` | `openfront/src/core/game/PlayerImpl.ts:1177-1189` | `game.rs:1130-1155` |
| `Util.toInt` (floor) | `Util.ts` | `rust/engine/src/util.rs:26-35` |
| storage types the per-tick hash reads | — | `game.rs:86` `troops: i32`, `game.rs:88` `tiles_owned: i32`; hash at `hash.rs:19-22` |

```ts
// PlayerExecution.ts:75-78
const troopInc = this.config.troopIncreaseRate(this.player);
this.player.addTroops(troopInc);
const goldFromWorkers = this.config.goldAdditionRate(this.player);
this.player.addGold(goldFromWorkers);
```

Per tick, per player, exactly two things change on this slice: `troops` and
`gold` (plus the derived `troops ** 0.73` / `maxTroops` intermediates that go
into the observation planes).

### 1.1 What this slice does *not* contain (and the tile state buffer)

The observation / hash tile-state plane is the packed `u16` per tile (owner bits,
fallout bit, defence-bonus bit) written by `GameMap.setOwner` / `setFallout`
(`openfront/src/core/game/GameMap.ts:287-305`; `rust/engine/src/map.rs`
`set_owner` / `set_fallout`). **This slice writes none of it**, and that is a
structural fact rather than a measurement: `econ_step` has no tile index and no
tile-state argument — its only inputs are
`(player_type, difficulty, troops, tiles_owned, city_level_sum, gold,
gold_multiplier, infinite_troops)`. The claim is deliberately *not* stated as a
measured equality, because a measurement over a dump cannot prove a negative
about a buffer the dump does not carry.

### 1.2 The two "dead code" methods — what the live path actually is

Both flags are confirmed by grep over the whole Rust tree (one occurrence each —
the definition, no caller):

```
$ grep -rn "tick_player_income" --include=*.rs rust/
engine/src/game.rs:1269:    fn tick_player_income(&mut self) {
$ grep -rn "try_merge_land_attack" --include=*.rs rust/
engine/src/game.rs:1991:    fn try_merge_land_attack(
```

The live per-tick income is `execution/player.rs::PlayerExecution::tick`
(`player.rs:82-83`), reached through the execution manager in the tick loop — the
same structure as TS `PlayerExecution.tick`. Nothing in
`tick_player_income` is on the live path, and the observations do **not** call it
either: they read `game.troop_increase_rate_raw_for(sid)` directly
(`obs.rs:129`, `obs_typed.rs:57`). So: a duplicate, already-superseded
implementation, not a second live path; `try_merge_land_attack` is likewise not
on the income path (it is conquest-side, out of this slice).

## 2. Layout

```
ofcuda_econ/
  src/core_impl.rs        canonical implementation (host + device share it)
  src/lib.rs              host reference, case loading, diff tables, tests
  src/main.rs             #[cuda_module] kernels { VERBATIM copy of core_impl + econ_step_kernel }
  src/bin/cpu.rs          CPU-only companion binary (no CUDA in the dependency graph)
  cases/trans-b000s3.json    599 recorded transitions   (ground truth from the engine)
  cases/trans-b005s0.json   7192 recorded transitions   (ground truth from the engine)
  cases/synthetic_city_levels.json  300 SYNTHESISED rows (no engine truth)
  sync_device_core.sh     re-copies core_impl.rs into the device module
  run_output.txt          verbatim output of the final run
  ofcuda_econ.ptx         device PTX emitted by the build (evidence for the FMA section)
```

* **Kernel** — `econ_step_kernel` (`src/main.rs:263`), one thread per
  (player, tick-transition), `256` threads/block, `#[launch_contract(domain = 1)]`.
  It writes `troops_after: i32`, `gold_after: i64`, and three diagnostics
  (`raw_income`, `max_troops`, `pow_tiles`, `pow_troops`, all `f64`).
* **CPU reference** — `run_cpu` (`src/lib.rs`) folds the same rows sequentially
  through **the same `econ_step`** (`core_impl.rs`). There is no second
  implementation to drift.
* **Verbatim discipline** — `src/main.rs` contains a byte-for-byte copy of
  `src/core_impl.rs` between `===== BEGIN/END VERBATIM COPY` markers, because
  `#[cuda_module]` cannot see through `include!`. The test
  `device_core_is_a_verbatim_copy` (`src/lib.rs:538`) compares the copy to the
  canonical file and fails with *"the device copy inside src/main.rs diverged
  from src/core_impl.rs — run `bash sync_device_core.sh`"*. **Demonstrated to
  fail when perturbed**: changing `pow_troops(troops as f64)` to
  `pow_troops(troops as f64 + 1.0)` *inside the device copy only* turned the test
  red (6 passed, 1 failed), then `sync_device_core.sh` restored it.

## 3. Cases: reconstructed vs synthesised

* `cases/trans-b000s3.json` — **reconstructed** from a dump taken here, from the
  current build: `tick_dump` on the recorded zero-intent episode
  `records/curriculum-parity-v1/curr-b000-s3-onion.json.gz` (Onion 512×512,
  bots=0 / nations=1, Easy), `--every 1 --max-ticks 900`, `OF_DUMP_UNITS=1`.
  1 nation, 599 tick-transitions, `playerType = Nation`.
* `cases/trans-b005s0.json` — **reconstructed** the same way from
  `curr-b005-s0-onion.json.gz` (bots=5 / nations=3, Easy), `--max-ticks 1200`.
  5 players, 7192 tick-transitions, covering both `Bot` (×0.5)` and `Nation`
  (×difficulty)` scales.
* `cases/synthetic_city_levels.json` — **SYNTHESISED** in this crate: 300 rows
  sweeping `cityLevels = 0..4` × `Human/Bot/Nation/Hard` × three tile counts.
  No recorded episode builds a city before tick 1200, so the
  `cityLevelSum * 250000` term is not exercised by recorded data at all. These
  rows carry **no engine ground truth** (`nextTroops`/`nextGold` were computed
  from the same model, purely so the loader has a well-formed file); the run
  prints `[SYNTHESISED]` and the engine-agreement check is skipped for them. The
  number they *do* establish is the device-vs-host one (300/300).

Both dumps come from the current build (`rust/engine` at the deployed tip, which
carries the N,S,W,E neighbour-order fix), built privately:

```sh
export PATH="/nix/store/qmdxxa88bgbdx31dav3qlssb4rghr14c-gcc-wrapper-14.4.0/bin:$PATH"
export CARGO_TARGET_DIR=/tmp/ofecon-target           # never the live target/
R=/opt/data/workspaces/skg/openfront-ai
cargo build --release --manifest-path $R/rust/Cargo.toml -p openfront-engine --bin tick_dump
export OF_DUMP_UNITS=1
/tmp/ofecon-target/release/tick_dump --repo $R \
  --record $R/records/curriculum-parity-v1/curr-b005-s0-onion.json.gz \
  --every 1 --max-ticks 1200 --out /tmp/econ/b005s0.ndjson
```

## 4. Measured results

The block below is an excerpt of the real run output, trimmed only by dropping
the `(NN%)` suffixes and the repeating per-row diff tables; the untrimmed text is
`run_output.txt` in this directory (129 lines, captured from the run below).

```
=== recorded, bot-only, zero-intent episode ===
  record curr-b000-s3-onion (source b000s3.ndjson, 599 rows)
  GPU vs CPU reference (same core, one on the device): rows=599 exact_out=313 raw_income_bits_equal=593
      troops(running)     0/599
      gold(running)       0/599
      rawIncome(bits)     6/599
      maxTroops(bits)     80/599
      tilesPow06(bits)    170/599
      troopsPow073(bits)  164/599
  raw_income float: worst |delta| 7.276e-12 ... narrowed to f32 the device and host agree on 599/599
  engine agreement: predicted troops+gold == engine's own next-tick dump on 591/599 rows
  differing rows: 8 (first tick 302), of which 0 have no engine attack snapshot nearby
  PASS state_bitexact true | float_f32_identical true | engine_diffs_all_attack_labelled true

=== recorded, bot-only, zero-intent episode ===
  record curr-b005-s0-onion (source b005s0.ndjson, 7192 rows)
      troops(running)     0/7192
      gold(running)       0/7192
      rawIncome(bits)     29/7192
      maxTroops(bits)     1033/7192
      tilesPow06(bits)    1777/7192
      troopsPow073(bits)  1901/7192
  raw_income float: worst |delta| 1.455e-11 ... narrowed to f32 the device and host agree on 7192/7192
  engine agreement: predicted troops+gold == engine's own next-tick dump on 7102/7192 rows
  differing rows: 90 (first tick 306), of which 0 have no engine attack snapshot nearby
  PASS state_bitexact true | float_f32_identical true | engine_diffs_all_attack_labelled true

=== SYNTHESISED city-level sweep ===
  record SYNTHESISED (300 rows)  [SYNTHESISED]
      troops(running)     0/300
      gold(running)       0/300
      rawIncome(bits)     0/300
      maxTroops(bits)     0/300
      tilesPow06(bits)    0/300
      troopsPow073(bits)  60/300
  PASS state_bitexact true | float_f32_identical true | engine_diffs_all_attack_labelled true (no engine truth: SYNTHESISED)

=== FMA contraction sensitivity (host arithmetic) ===
  points tested 4416; two-step vs mul_add differ on 58

PROOF gpu_matches_cpu_on_every_row true
```

`cargo test --lib`: **7 passed, 0 failed**.

### Agreement counts, and what each one means

| Claim | Count |
|---|---|
| device == host, `troops` (i32, running value) | **7791 / 7791** |
| device == host, `gold` (i64, running value) | **7791 / 7791** |
| CPU reference == the engine's own next-tick `troops`+`gold` | **7693 / 7791** (98.75 %) |
| … of the 98 disagreements, ones **not** on an engine-reported attack transition | **0** |
| `raw_income` identical after the f32 narrowing the obs planes apply | **7791 / 7791** |
| `raw_income` identical as f64 (bit level) | 7762 / 7791 |

The 98 engine disagreements are the per-tick income *plus* the attack escrow
happening on the same transition: the engine removes `startTroops` from the owner
when an attack is created (`AttackExecution`), which is conquest-side and out of
this slice. Every one of them carries an engine attack snapshot at one endpoint
of the transition (in `b000` the same player is under a persistent attack for 451
of its 599 transitions and the income still reproduces exactly on all of them —
only the 8 creation/return events break it). Residual = `observed − predicted` is
negative and its magnitude tracks the attack's escrowed troop count, e.g.
`b000` tick 452: residual −17641 vs an attack holding 17641.

## 5. FMA contraction: measured, real, and neutralised

Not inert here — and the PTX proves it. The first build emitted, in the
`maxTroops` chain:

```
    // ofcuda_econ.ptx, pre-fix
    fma.rn.f64 %rd82, %rd130, 0d408F400000000000, 0d40E86A0000000000;   // pow*1000 + 50000
    add.f64    %rd83, %rd82, %rd82;                                     // *2  (exact)
    fma.rn.f64 %rd22, %rd85, 0d410E848000000000, %rd84;                 // city*250000 + t
```

(`0d408F400000000000` = 1000.0, `0d40E86A0000000000` = 50000.0,
`0d410E848000000000` = 250000.0.) The host is two-step. Measured effect: the
contracted form perturbs `maxTroops` on **734 / 7192** and **74 / 599** rows —
so the trap is real, not hypothetical.

`black_box` is rejected by the device codegen
(`.../core/src/hint.rs:491: invalid input program`), and `#[inline(never)]` is
ignored. A volatile read of the intermediate **is** honoured, and survives as a
real local load/store that the MIR-level contraction cannot see through:

```rust
#[inline]
pub fn survive_opt(p: f64) -> f64 { unsafe { core::ptr::read_volatile(&p) } }

let mut max = 2.0 * mul_then_add(pow_tiles(tiles_owned as f64), 1000.0, 50_000.0)
    + survive_opt(city_level_sum as f64 * CITY_TROOP_INCREASE);
```

After that the same sites are plain multiplies
(`mul.f64 %rd79, %rd137, 0d408F400000000000` at `ofcuda_econ.ptx:216`, and
`mul.f64 %rd21, %rd85, 0d410E848000000000` at `:224`; no `fma.rn.f64` anywhere in
the kernel still carries 1000.0 / 50000.0 / 250000.0), the
device's `maxTroops` equals the two-step form on **7192 / 7192** rows, and the
only remaining device/host arithmetic difference is libdevice `pow` itself (see
§6). The 45 `fma.rn.f64` left in the PTX are inside `__nv_pow`'s own polynomial.
Cost of the barrier: one local store/load per intervening op, negligible at one
row per thread.

## 6. First (and only) remaining point of disagreement

**libdevice `__nv_pow` ≠ glibc `pow` by 1 ULP** on integer inputs:
`tiles ^ 0.6` differs on 1777/7192 rows, `troops ^ 0.73` on 1901/7192 (device vs
host, same expression, same rounding mode). That is the entire residual:

* it propagates into `maxTroops` (1033/7192 rows differ, and that count is
  exactly the `pow`-induced count once the FMA barrier is in);
* it reaches the pre-truncation income float on **29 / 7192** rows (first at
  record tick 355, `nation:Inner Tribe`, worst |delta| 1.455e-11);
* it never changes a floored result in these episodes: `troops` and `gold`
  diverge on **0 / 7791** rows, and after the f32 narrowing the obs planes apply
  (`log_norm` → f32 in `ofcore/src/feat.rs:909`) the device and host agree on
  **7791 / 7791**.

So the residual is: a `f64` difference of ≤1 ULP in the observation entity's
`troop_income` (`feat.rs:294`, `obs_typed.rs:57`), measured invisible at the f32
plane type, but not *provably* invisible to any future consumer that reads that
field as f64. Closing it needs a correctly-rounded `pow` on the device
(integer-input double-double `log2`/`exp2`, or `x^3`-then-5th-root integer
Newton for 0.6 and an arbitrary-precision route for 0.73). Not attempted here.

## 7. Assumptions and limitations (stated, not measured)

1. `cityLevelSum` — 0 on every row of both recorded episodes; the term is
   exercised only by the SYNTHESISED case (`cityLevels 0..4`), which has no
   engine ground truth. The `cityLevelSum * 250000` placement (before the
   per-type scale) rests on the TS/Rust source, not on an observation.
2. `difficulty` — both episodes are `Easy`; the `Medium/Hard/Impossible`
   multipliers (0.75/1.0/1.25 for `maxTroops`, 0.95/1.0/1.05 for the rate) rest
   on source. Unknown difficulty strings default to `Easy` in the loader and are
   never exercised.
3. `infiniteTroops` (the `Human` → 1e9 branch) — false on every row; not
   exercised. `Human` players appear only in the SYNTHESISED case.
4. `goldMultiplier` — 1.0 on every row; only the `floor(50|100 * mult)` shape is
   exercised.
5. The `negative income` path (`add_troops` routing a negative amount through
   `remove_troops`, i.e. `-floor(-x)`) is unit-tested
   (`negative_income_floors_the_magnitude`) but was **never hit** by the recorded
   transitions (every predicted income was positive).
6. The engine's *float* form for `maxTroops`/income is inferred from its source,
   not observed: the dump carries the floored integers, so "the engine is
   two-step" is supported by the two-step form reproducing every observed integer
   on all 7693 economy transitions, and by the TS/Rust source text.
7. `troops_owned`/`tiles_owned` and everything else in `Player` are inputs here,
   not outputs: this slice advances `troops` and `gold` only, so the reproduced
   state is the per-player economy state, not the whole `Player` struct.

## 8. Reproduce

```sh
cd /opt/data/workspaces/skg/ofcuda_econ
bash sync_device_core.sh                       # device copy := core_impl.rs
/opt/data/workspaces/skg/ofcuda_env.sh cargo oxide run     # GPU vs CPU vs engine  -> run_output.txt
export PATH="/nix/store/qmdxxa88bgbdx31dav3qlssb4rghr14c-gcc-wrapper-14.4.0/bin:$PATH"
/opt/data/workspaces/skg/ofcuda_env.sh cargo test --lib    # 7 passed
# CPU-only, no CUDA in the dependency graph at all:
cargo build --no-default-features --bin ofcuda_econ_cpu
cargo run  --no-default-features --bin ofcuda_econ_cpu
```

The GPU run exits non-zero unless all three of its stated criteria hold:
`state_bitexact`, `float_f32_identical`, and
`engine_diffs_all_attack_labelled` (engine agreement is not required for a
SYNTHESISED case, which has no engine truth).

## 9. Independent re-verification (fresh dump, current build)

The recorded cases in §3 were re-derived from scratch against a `tick_dump`
rebuilt from the CURRENT tree (`CARGO_TARGET_DIR=/tmp/ofecon-target`,
2026-09-18 17:46; the deployed `libopenfront_engine.so` and `PufferLib5/puffer`
were **not** touched). Both episodes were re-dumped to `/tmp/econ_fresh/` with
`OF_DUMP_UNITS=1 OF_DUMP_ATTACKS=1`, re-extracted with the same probe, and the
resulting row sets compared field-by-field against the committed cases:

```
b000s3: 599/599 rows identical  (every field, incl. attackActivity)
b005s0: 7192/7192 rows identical (every field, incl. attackActivity)
```

`cargo oxide run` against the freshly derived cases exits 0 with output
identical to `run_output.txt` on every line except the dump-source path:

```
  engine agreement b000s3: 591/599   (98.66%)  differing 8  (first tick 302), 0 unlabelled
  engine agreement b005s0: 7102/7192 (98.75%)  differing 90 (first tick 306), 0 unlabelled
  PROOF gpu_matches_cpu_on_every_row true
```

`cargo test --lib`: 7 passed, 0 failed. `ofcuda_econ_cpu`
(`--no-default-features`) exits 0 with the same engine counts. The emitted
`ofcuda_econ.ptx` carries no `fma.rn.f64` on 1000.0 / 50000.0 / 250000.0 and
keeps the two-step `mul.f64` at `:216` / `:224`; the 45 remaining `fma.rn.f64`
are inside `__nv_pow`. So §5's FMA barrier is still in place in the current
build, and §6's libdevice-`pow` 1-ULP residual is the only substantive
difference (and only at the f64 `rawIncome`/`maxTroops` diagnostics — never at
`troops`/`gold`).
