# MATRIX_REPORT — the composed CUDA tick driven end to end on many maps and many agent counts

Crate: `/opt/data/workspaces/skg/ofcuda_matrix` (own workspace, empty `[workspace]` table,
read-only path deps on `ofcuda_map` / `ofcuda_prng` / `ofcuda_tick` / `ofcuda_env`).
Nothing outside this crate was edited. Ground truth is always `openfront-engine`, linked
directly by `ofcuda_matrix/oracle`.

Binary: `/opt/data/workspaces/skg/.target-ofcuda-matrix/release/ofcuda_matrix`
sha256 prefix `21f3b67884f08541`. Oracle: `/opt/data/workspaces/skg/ofcuda_matrix/oracle/target/release/oracle`.

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
* **the attack lifecycle schedule**: attack *creation* and *eviction* timing is taken from
  the engine record (`INIT_ATTACK` / `ENGINE_EVICTION` lines). The device has no
  attack-creation path of its own; it evolves an attack once the driver has created it,
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

Cells run (30): every one of the 10 maps at N=18, and N ∈ {2,7,18,64,488} spread across
maps so each N value is covered on ≥3 maps, plus `nations` Exact(1)/Exact(2) on pangaea.
`ticks 60`, `nations 0` unless stated, seed `parity`.

Plus a 5-cell **short-window control** (`tools/cells_short.txt`, `ticks 45`) closing the
window *before* the divergence event, which is what isolates the cause (§4).

Per-cell results, engine-verified (init owned sets, per-tick owned-tile counts, and the
full owner-plane hash every tick):

<!-- MATRIX_TABLE -->

---

## 4. First divergence anywhere, with cause

**None of the N≥18/N=64/N=488 cells diverge in the tile kernels.** Every one of them is
bit-exact until **boundary 49 (engine tick 52)**, then diverges on `attack owner 9`
(player 9, target 0 = terra nullius), identically on 7 different maps including 5536x276
and 400x4200 strips — a bug that reproduces across wildly different geometry is in the
code, not the map. The cells that never reach that tick (N=2, N=7 on two maps, and the
5-cell `ticks 45` control set) pass **100%**, plane hash included.

### 4.1 The engine event, as the record shows it

1. At engine tick 52 the engine **re-initialises / re-registers player 9's attack**:
   `AttackExecution::init` (`attack.rs:136-150`) sets `self.troops = start`, where `start`
   is `self.start_troops` — the figure the bot AI passed through
   `game.add_land_attack_from(owner, target, start_troops, source_tile)` (`game.rs:1469-1480`)
   — or `game.wire.attack_amount(...)` (`core/config.rs:461-468`: 5% of the owner's troops
   for a `Bot`, 20% otherwise). The same `init` re-registers the attack
   (`attack.rs:158`, visible in the record as the attack **moving to the end of
   `live_attacks()`** at boundary 49 — it is first at boundaries 47–48) and seeds its
   heap from the source tile (`attack.rs:160-164`, `add_neighbors`) when it has one,
   falling back to `refresh_to_conquer` (`attack.rs:1265`) when it does not.
2. The engine's recorded troops for that attack jump
   `1279.5697607467282 -> 1507.6145973769035` (`0x4093fe476f5c76f8 -> 0x40978e755903c808`).
   None of the engine's observed attack amounts is a multiple of 0.05, so none came from
   `attack_amount()` with an integer `i32`: the engine's bot AI passed them in explicitly
   (`game.rs:1477`), so the port must **reproduce** them, not recompute them.
3. That tick the engine claims **7** tiles for player 9 where the device claims **9**
   (`CLAIM_DIVERGENCE boundary 49 (engine tick 52) player 9: engine 7 tiles, device 9 tiles;
   first index 7; cause: tile sets differ: engine-only [] device-only [1858269, 1901828]`),
   303 vs 302 tiles across all players, so the plane hash differs from that boundary on.
   Before it, the device tracks the engine *bit-exactly*: europe N=18, owner 9 troops
   `b47 eng==dev 0x4094de476f5c76f8`, `b48 eng==dev 0x4093fe476f5c76f8`.

### 4.2 What the port now reproduces, and what it still cannot

Fixed in this pass (record-driven, kernel-untouched): the troop **re-allocation** is now
followed. When the recorded amount for a live `(owner, target)` goes **up** — the device's
own arithmetic only ever drains troops (`kernels.rs:225-236`, `attack.rs:313-314`), so an
increase can only be an engine decision — the device's `troops` word is re-seeded from the
record (`src/main.rs:689-740`, `TROOP_REALLOC`). Measured on europe N=18:

```
TROOP_REALLOC boundary 49 (engine tick 52): owner 9 target 0 troops 1279.5697607467282 ->
1507.6145973769035 bits 0x4093fe476f5c76f8 -> 0x40978e755903c808 (engine decision,
re-seeded from the record; heap/prng untouched)
```

**That did not move the plane by a single tile.** With the troops exactly the engine's, the
same boundary still diverges by the same 2 tiles (`device-only [1858269, 1901828]`), and the
same +2 appears on the build without the fix. So, measured rather than assumed:

* the divergence is **not** the troop value, not a clobber of the seeded amount, and not an
  owner-type branch. `attack_init` (`kernels.rs:358-388`) never touches `troops` — troops
  live in a separate `troops: &mut [f64]` (`kernels.rs:259, 289, 324`) — so the engine's
  number survives init intact; the shipped value is the engine's own, one tick of drain later,
* it is the attack's **heap state**: the engine's re-registration gives the attack a fresh
  heap seeded from its `source_tile` (`attack.rs:160-164`), so in that tick it can only
  reach the tiles that seed offered (7). The device's heap is the continuing one, still
  holding cheaper queued tiles, so it claims 2 more. Re-seeding the device's heap from the
  engine's **border** set instead was tried first and also lands on +2 (`attack_init` is
  border-set based), which is why the troop-only fix above is what shipped,
* the pipeline has no source-tile path at all: the device's only attack-init entry point is
  `attack_init` (`kernels.rs:358`), which seeds from the border set, and **the record does
  not carry the source tile** — the oracle's `ATTACK` line is
  `ATTACK {boundary} {owner} {target} {troops_bits} 1` (`oracle/src/main.rs:406-412`), where
  the trailing `1` is a literal placeholder and `AttackExecution::source_tile()`
  (`attack.rs:1172`) is never read. Closing the gap needs (i) the source tile in the record
  and (ii) a source-tile-seeded init kernel; neither exists today.

So: the device's tile arithmetic is not wrong here — it is doing the right thing for the
attack state it was given. The engine re-registers an attack mid-flight into a state the
composition cannot express, and the gap is in the composition (record + init path), not in
the kernels. Every N≥18/64/488 cell is reported as **failed** on that boundary; nothing is
flattened into a pass.

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

---

## 5. Cells NOT run, and why

* **N=488 on maps whose ceiling is below 488**: onion (285), world (749 — fine, 488 < 749),
  so in practice: **onion N=488** is a starve cell (bots that never place) on purpose is
  *not* run for ticks; the port reproduces the starve exactly (the `spawnall` proof covers
  it), but the tick driver requires every player to have a spawn tile.
* **N > 512** on any map: the matrix deliberately stops at 488 (the "last N where every bot
  places" on pangaea is 488; the ceilings above 512 are reported from the oracle, not
  driven through the CUDA tick). Driving N=2687 or N=4749 through the device tick is
  possible in principle (COLS=8192, MAX_SLOTS) but was not part of the requested counts.
* **nations = `disabled`/`default` through the CUDA tick**: only `0/1/2` were driven; the
  spawn-phase proof for disabled/default belongs to the spawn work, not this matrix.
* Cells where **attacks are already live at boundary 0** (spawn-phase nations): the driver
  refuses and reports `not drivable: N attack(s) already live at boundary 0 - the
  spawn-phase schedule is not modelled`. That is a deliberate, reported refusal.
* `manifest.json` land counts are **not** used anywhere: 4 maps (china, tierradelfuego,
  unitedstates, losangeles) undercount their own bytes, so land is always taken from the
  bytes. None of those 4 maps are in this matrix.

---

## 6. Artifacts

<!-- ARTIFACTS -->

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

# the render cells (per-tick device planes dumped) then movies + stills + contact sheet
bash /opt/data/workspaces/skg/ofcuda_env.sh $B --cells-file $M/tools/cells_renders.txt \
     --out $M/out/renders --oracle-dir $M/out/matrix/oracle --refs-dir $M/out/matrix/refs --ticks 60 --dump-planes
bash $M/tools/render_cells.sh $M/out/renders $M/out/report
python3 $M/tools/contact_sheet.py $M/tools/contact_sheet_manifest.txt $M/out/report/contact_sheet.png 4x
```

Build (CUDA crate, isolated target dir, one rebuild at a time):
```bash
cd /opt/data/workspaces/skg/ofcuda_matrix
export CARGO_TARGET_DIR=/opt/data/workspaces/skg/.target-ofcuda-matrix
nice -n 10 bash /opt/data/workspaces/skg/ofcuda_env.sh cargo oxide build -- --release
```
