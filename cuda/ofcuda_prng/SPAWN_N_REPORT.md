# Spawn / initial-state for a RANGE of agent counts — measured report

Everything below is measured against the **real engine** (`openfront-engine`, driven by the
`ofcuda_spawn` oracle), never against the CUDA port itself. Ground truth for each N is
produced by re-running the engine, then the port is compared to it.

## The CLI (this is the deliverable command)

```bash
# any N, any nations spec; engine reference auto-generated, GPU kernel per bot:
bash /opt/data/workspaces/skg/ofcuda_env.sh \
     /opt/data/workspaces/skg/ofcuda_prng/target/release/spawnall \
     --agents 22 --nations 0

# large N / cause analysis (skips the per-bot device launches):
... spawnall --agents 500 --nations 0 --no-gpu --diag

# write the port's initial state (spawn tiles + owned tile sets) for a tick driver:
... spawnall --agents 22 --nations 0 --dump /tmp/initial_state_a22.txt
```

Flags: `--agents N`, `--nations {0|1|2|disabled|default}`, `--no-gpu`, `--diag`,
`--dump FILE`. A positional `.txt` path may be given instead of `--agents` to check a
previously written engine reference. Batched matrix: `scripts/run_matrix.sh [--gpu] [--diag] N…`
(`NATIONS=1 run_matrix.sh …` for a nations variant).

Exit code 0 iff every compared element matches; 1 otherwise.

## Per-N table (Pangaea, seed `parity`, 15 ticks/decision config, 1 human slot)

| N | bots spawned (engine = port) | ids | spawn tiles | owned sets | per-player counts | GPU vs host CPU | first difference |
|---|---|---|---|---|---|---|---|
| 2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | none |
| 4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | none |
| 7 | 7/7 | 7/7 | 7/7 | 7/7 | 7/7 | 7/7 | none |
| 18 | 18/18 | 18/18 | 18/18 | 18/18 | 18/18 | 18/18 | none |
| 22 | 22/22 | 22/22 | 22/22 | 22/22 | 22/22 | 22/22 | none |
| 26 | 26/26 | 26/26 | 26/26 | 26/26 | 26/26 | 26/26 | none |
| 64 | 64/64 | 64/64 | 64/64 | 64/64 | 64/64 | 64/64 | none |
| 100 | 100/100 | 100/100 | 100/100 | 100/100 | 100/100 | (skipped) | none |
| 200 | 200/200 | 200/200 | 200/200 | 200/200 | 200/200 | (skipped) | none |
| 300 | 300/300 | 300/300 | 300/300 | 300/300 | 300/300 | (skipped) | none |
| 400 | 400/400 | 400/400 | 400/400 | 400/400 | 400/400 | (skipped) | none |
| **488** | **488/488** | 488/488 | 488/488 | 488/488 | 488/488 | (skipped) | none — **last fully-spawning N** |
| **489** | 488 spawned, 1 starved — **engine and port agree** | 489/489 | 488/488 | 489/489 | 489/489 | (skipped) | none — first **engine-side** starvation |
| 490 | 490/490 | 490/490 | 489/489 | 490/490 | 490/490 | (skipped) | none |
| 500 | 497 spawned (engine = port) | 500/500 | 497/497 | 500/500 | 500/500 | (skipped) | none |
| 550 | 527 spawned (engine = port) | 550/550 | 527/527 | 550/550 | 550/550 | (skipped) | none |
| 700 | 557 spawned (engine = port) | 700/700 | 557/557 | 700/700 | 700/700 | (skipped) | none |
| 1200 | 565 spawned (engine = port) | 1200/1200 | 565/565 | 1200/1200 | 1200/1200 | (skipped) | none |
| 2000 | 566 spawned (engine = port) | 2000/2000 | 566/566 | 2000/2000 | 2000/2000 | (skipped) | none |
| 3000 | 569 spawned (engine = port) | 3000/3000 | 569/569 | 3000/3000 | 3000/3000 | (skipped) | none |

"spawn tiles" counts only bots that spawned in *both* engine and port; the `spawned` row is
the authoritative per-bot agreement (it includes the starved bots, which the port reproduces
exactly — `tries` reaches the 1000 cap in both).

### Nations variants (all with the GPU kernel per bot)

`nations = 1` (`Exact(1)`), `2` (`Exact(2)`), `disabled` and `default` × N ∈ {18, 22, 26, 64}:
**all 16 cases bit-exact** — ids, spawn tiles, owned sets and per-player counts.

With nations enabled the engine's player table is `human(1) → nation(s) → bots`, and a nation
occupies a spawn tile (e.g. N=7, nations=1: `player 2 N 0b4eakcu spawn_tile=610150 tick=2`).
The port's bot path is unperturbed by that: same bot ids, same spawn tiles, same owned sets.
Nation spawning itself lives in `execution/nation.rs` (engine-side), not in the ported
`TribeSpawner`/`find_spawn` path, and is reported rather than ported.

## Where it breaks, and why

**The port never diverges** — not at any N in 2…3000, including all the starved bots.
What breaks is the *engine's spawn phase itself*:

* **First starved bot: N = 489** (bot index 488). Engine and port both fail to place it, both
  exhaust exactly 1000 tries, and the port's `spawned` row agrees 489/489.
* Saturation: spawned counts plateau at **566–569** regardless of N (2000 → 566, 3000 → 569).
  Beyond that, extra bots are simply never given a spawn tile.

Cause, measured with `--diag` (whole-map sweep using the same predicates `find_spawn` uses,
at the state where bot 488 runs):

```
land_unowned=394959  footprint_bad=112829  valid_footprint=282130
border=0             too_close=279819      far_enough=2311
```

* **Not land capacity**: 394,959 land tiles (94%) are still unowned; 2,311 centres pass
  *every* test (land, unowned, non-border, footprint entirely land+unowned, ≥30 Manhattan
  from all 488 existing centres).
* **Not nation/human slot handling**: the failure is identical with `nations=0` and the human
  slot is always present and never consumes a spawn attempt.
* **It is the min-distance rule + the fixed 1000-try budget**: `min_distance_between_players()`
  = 30 (engine-sourced, read from the reference file, not assumed) has collapsed the eligible
  fraction of the 1,000,000-tile draw pool to 2,311/1,000,000 ≈ 0.23%. Each attempt is one
  uniform `rand_tile` draw plus exactly 2 `next_int` draws, so a 1000-draw sample misses all
  eligible centres with probability ≈ 10% at that density — which is why the first failure
  appears around N≈489 and the failure count grows continuously with N.

So: **retries exhausted under the min-distance packing constraint**, faithfully reproduced by
the port, not retries exhausted by lack of land and not a slot/ordering bug.

## Changes made

* `ofcuda_prng/src/lib.rs`
  * `SpawnCtx::owner_plane: Option<&[u16]>` — O(1) owner lookup over a full `w*h` plane. The
    recorded-window path still uses the engine's override *list* (`None`); the generic driver
    uses the plane, which is what stops the N-bot run being O(N²) in owned tiles (this is why
    the port can now finish N=3000); parity is unaffected.
  * `PseudoRandom::next_id` (`prng.rs:75-90`) so the port generates the tribe ids itself.
  * Generic N-bot driver `spawn_bots_cpu` (plane + `prev` centres, engine spawn order),
    `tribe_bot_ids`, `compare_spawn_bots`, `format_spawn_report`, `starvation_diagnostic`
    (+`Starvation`), `SpawnReference::min_dist` parsed from the reference.
* `ofcuda_prng/src/kernels.rs` (new) — the kernel source extracted from `main.rs` so both bins
  embed literally the same device code; `main.rs` now `include!`s it.
* `ofcuda_prng/src/bin/spawnall.rs` (new bin) — the N-bot CLI above.
* `ofcuda_spawn/` (new standalone crate, not a workspace member) — the engine oracle; emits
  `width/height/min_dist/player/bot/owned/prev/owner` reference files.
* `scripts/run_matrix.sh` — the per-N matrix driver.

No regression: the original recorded-window 2-owner parity test still reports
`ALL_BIT_EXACT true` (engine vs GPU, engine vs CPU, and GPU-VS-CPU cross 60/60) when run as
`bash /opt/data/workspaces/skg/ofcuda_env.sh ./target/release/ofcuda_prng reference.txt`.
