# Nation spawn parity (CUDA port vs the engine)

Independent-oracle evidence for the nation spawner. Every "engine" number below
is produced by one of two engine-linking binaries; **nothing is validated by the
port's own code**:

* `ofcuda_spawn` - drives the real engine (`RlSession`) and dumps its spawn
  state; this is the reference the port is diffed against.
* `ofcuda_nation` - a SECOND, independent engine wrapper (own crate, never
  calls CUDA) that **re-derives** each nation's id, spawn tile and tick from the
  engine's seed chain and board state, then compares its derivation with what
  the engine actually registered (`derived_eq_engine`, `id_eq_engine`).

Reproduce: `bash /opt/data/workspaces/skg/ofcuda_nation/sweep.sh`
Table:      /opt/data/workspaces/skg/ofcuda_nation/out/nation_parity.tsv

## Result: 45/45 cells OK

| map | N | nations | status | nations count | tiles engine/port | tries engine/port |
|-----|---|---------|--------|---------------|-------------------|-------------------|
| pangaea | 7 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=52, tiles(port)=52, | tries engine=1, port=1, |
| pangaea | 7 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=52,52, tiles(port)=52,52, | tries engine=1,1, port=1,1, |
| pangaea | 18 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=52, tiles(port)=52, | tries engine=1, port=1, |
| pangaea | 18 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=52,52, tiles(port)=52,52, | tries engine=1,1, port=1,1, |
| pangaea | 64 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=52, tiles(port)=52, | tries engine=1, port=1, |
| pangaea | 64 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=52,52, tiles(port)=52,52, | tries engine=1,1, port=1,1, |
| africa | 7 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=24, tiles(port)=24, | tries engine=3, port=3, |
| africa | 7 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=24,52, tiles(port)=24,52, | tries engine=3,1, port=3,1, |
| africa | 18 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=24, tiles(port)=24, | tries engine=3, port=3, |
| africa | 18 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=24,52, tiles(port)=24,52, | tries engine=3,1, port=3,1, |
| africa | 64 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=24, tiles(port)=24, | tries engine=3, port=3, |
| africa | 64 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=24,52, tiles(port)=24,52, | tries engine=3,1, port=3,1, |
| europe | 7 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=52, tiles(port)=52, | tries engine=1, port=1, |
| europe | 7 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=52,52, tiles(port)=52,52, | tries engine=1,1, port=1,1, |
| europe | 18 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=52, tiles(port)=52, | tries engine=1, port=1, |
| europe | 18 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=52,52, tiles(port)=52,52, | tries engine=1,1, port=1,1, |
| europe | 64 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=52, tiles(port)=52, | tries engine=1, port=1, |
| europe | 64 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=52,52, tiles(port)=52,52, | tries engine=1,1, port=1,1, |
| world | 7 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=52, tiles(port)=52, | tries engine=1, port=1, |
| world | 7 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=52,52, tiles(port)=52,52, | tries engine=1,1, port=1,1, |
| world | 18 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=52, tiles(port)=52, | tries engine=1, port=1, |
| world | 18 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=52,52, tiles(port)=52,52, | tries engine=1,1, port=1,1, |
| world | 64 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=52, tiles(port)=52, | tries engine=1, port=1, |
| world | 64 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=52,52, tiles(port)=52,52, | tries engine=1,1, port=1,1, |
| thebox | 7 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=52, tiles(port)=52, | tries engine=1, port=1, |
| thebox | 7 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=52,52, tiles(port)=52,52, | tries engine=1,1, port=1,1, |
| thebox | 18 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=52, tiles(port)=52, | tries engine=1, port=1, |
| thebox | 18 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=52,52, tiles(port)=52,52, | tries engine=1,1, port=1,1, |
| thebox | 64 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=52, tiles(port)=52, | tries engine=1, port=1, |
| thebox | 64 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=52,52, tiles(port)=52,52, | tries engine=1,1, port=1,1, |
| onion | 7 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=52, tiles(port)=52, | tries engine=1, port=1, |
| onion | 7 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=52,52, tiles(port)=52,52, | tries engine=1,1, port=1,1, |
| onion | 18 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=52, tiles(port)=52, | tries engine=1, port=1, |
| onion | 18 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=52,52, tiles(port)=52,52, | tries engine=1,1, port=1,1, |
| onion | 64 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=52, tiles(port)=52, | tries engine=1, port=1, |
| onion | 64 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=52,52, tiles(port)=52,52, | tries engine=1,1, port=1,1, |
| baikalnukewars | 7 | 0 | OK | nations= | tiles(engine)= tiles(port)= | tries engine= port= |
| baikalnukewars | 7 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=52, tiles(port)=52, | tries engine=0, port=0, |
| baikalnukewars | 7 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=52,52, tiles(port)=52,52, | tries engine=0,0, port=0,0, |
| baikalnukewars | 18 | 0 | OK | nations= | tiles(engine)= tiles(port)= | tries engine= port= |
| baikalnukewars | 18 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=52, tiles(port)=52, | tries engine=0, port=0, |
| baikalnukewars | 18 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=52,52, tiles(port)=52,52, | tries engine=0,0, port=0,0, |
| baikalnukewars | 64 | 0 | OK | nations= | tiles(engine)= tiles(port)= | tries engine= port= |
| baikalnukewars | 64 | 1 | OK | nations=engine nations 1 port nations 1 | tiles(engine)=52, tiles(port)=52, | tries engine=0, port=0, |
| baikalnukewars | 64 | 2 | OK | nations=engine nations 2 port nations 2 | tiles(engine)=52,52, tiles(port)=52,52, | tries engine=0,0, port=0,0, |

Bar: for each cell the nation's **player id, small_id, spawn tile, spawn tick,
owned-tile set (exact tile indices) and starting troops** must equal the
engine's. `id_eq=1` means the oracle's own re-derivation of the id stream
(including the fabricated-name draws) equals the id the engine registered;
`derived_eq=1` means the oracle's re-implementation of the placement algorithm
reproduces the engine's tile. `tries` is the engine's own counter
(`SPAWN_DEBUG` print) vs the port's.

## The cell=None case (map `baikalnukewars` declares 0 nations)

`baikalnukewars` ships `nations: []`. Requesting nations on it exercises the
path where the engine FABRICATES the nation (`core/nation.rs:120-131`: a
`generate_unique_nation_name` draw - 2 draws, more on a collision - then a
`next_id`), and `nation.rs:98` enqueues
`SpawnExecution::new(game_id, player_info, None)`: **spawn_cell is None**, so
placement does not go through the nation's 50-try sampler at all (`tries = 0`)
but through the GENERIC `find_spawn` the bots use, against the plane the bots
already filled plus the centres already on the board.

Engine and port agree bit-for-bit on all of it, including the fabricated ids:

```
nation 0 2 qgzjxq2a -1 -1 528507 2 52      <- engine (cell none, 52 tiles, tick 2)
nation 0 2 qgzjxq2a -1 -1 528507 2 52      <- port
NATIONCHAIN 0 derived_mode=generic_find_spawn(cell=none) derived_tile=528507 derived_eq_engine=1 id_eq_engine=1
```

An earlier revision of the port gated nation CREATION on the manifest length
(`min(nation_count, manifest.len())`), which reported `port nations 0` here.
That was wrong: `NationExecution::init` (`nation.rs:71-77`) creates the player
unconditionally; placement is separate. The oracle's derivation for `cell=None`
was wrong in the same revision (it modelled only `random_spawn_land`); it now
re-implements the generic search (`spawn_util.rs:65-104` + the disc predicate of
`get_spawn_tiles(.., require_all_valid = true)`).

## What the driven window exercises, and what is NOT ported

Ported and proven: nation player creation, the `PseudoRandom(simple_hash(id) +
simple_hash(game_id))` seed, the trigger/reserve/expand and
attack_rate/attack_tick draws, spawn ordering (nations' centres sampled against a
plane with no bot on it, footprints cut on tick 2 against the full bot plane),
`random_spawn_land`'s 50-try rejection sampler, the fabricated-nation path, and
the generic `find_spawn` fallback.

NOT ported: `nation_tick.rs`'s post-spawn behaviour layer
(`initialize_nation_behaviors`, `tick_nation_post_spawn`, the
`add_land_attack(small_id, None, troops/2)` ordering). Measured boundary, from
`ofcuda_matrix --map pangaea --agents 18 --nations 1 --ticks 250`:
the plane hash and per-tick counts stay bit-exact for 250/250 ticks, but the
ATTACK-event stream first diverges at engine tick 106 (owner small_id 2 = the
nation, engine emits an attack of troops 0x40cd1d8999026114, the device emits
none). The same divergence class appears **without any nation** at engine tick
118 (`--agents 18`, no nations, owner 14 = a bot), so it is a pre-existing limit
of the port's attack-event model beyond the cell window, not a
nation-specific defect. All matrix nation cells run a 60-tick window, which is
inside the exact region (60/60 ticks, hash and counts identical).
