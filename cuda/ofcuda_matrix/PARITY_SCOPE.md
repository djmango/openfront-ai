# What the matrix's green actually proves

Audit of `ofcuda_matrix/src/main.rs`: every place the harness reads the engine
oracle and writes DEVICE state. Purpose: so a `30/30` is never quoted as if the
port played the game unaided. Line numbers are from main.rs at the commit that
adds this file.

The doc comment at the top of main.rs (lines 22-40) is NARROWER than the code:
it claims only the attack schedule and the attack's creation troop count are
host-supplied. Sections 1, 1b, 3c, 4b and 4e also write host state into the
device every boundary. Treat this file as the accurate inventory.

## Host-supplied (read from the oracle record, written to the device)

| section | line | what is written | cadence |
|---|---|---|---|
| 1 | 1082-1097 | every owner's `border_tiles` array, in the ENGINE's insertion order | every boundary |
| 1b | 1099+ | the per-player state the tick starts from. The file's own comment: "Nothing is carried over: the record is the source, so a device-side drift cannot accumulate silently." | every boundary |
| - | 1452-1464 | the cluster cadence counters (`last_cluster_calc`, `last_tile_change`, tiles_owned) | every boundary |
| 3c | 1465-1472 | the friendship table the attack ticks read, from the oracle's own `FRIEND` pairs | every boundary |
| - | 1473-1484 | the live attacks' owner / target / TROOPS arrays | every boundary |
| 2 | 1150+, 1331-1332 | when an attack is created, and its creation troop count | on creation |
| - | 900, 916 | attack ids and the conquer order | on events |
| 4b | 1876-1910 | the engine's RE-CREATEs, including their troops, applied AFTER the ticks | on re-create |
| 4e | 2296-2371 | a live attack's troops LOWERED to the record's value wherever the record shows the same `attack_id` dropping by >= 1.0 | every boundary |

Note the two that are easy to misquote:

* **Cadence counters are NOT device-accumulated in the matrix.** They are
  re-seeded from the record every boundary (1452-1464). They are only
  device-accumulated in `ofcuda_env` (`gpu_env.rs`), which is why the cluster
  work needed a separate env path.
* **Section 4e is not a port.** Both its trigger and the value it writes are the
  engine's own numbers: it overwrites the device's troop count with the record's.
  It closed the N=488 500-tick stop (312 -> 363 boundaries, plane hash 363/363)
  INSIDE THE HARNESS and proves nothing about `cancel_opposing_land_attacks`. A
  real port computes `incoming - start` from device state alone.

## Device-computed (the port's own work, this is the evidence)

* the heap order and its priorities, the PRNG draws and the troop budget, the
  pop loop, the frontier (`to_conquer`), the per-tick claim SETS and ORDER, the
  owner plane and its FNV-1a-64 hash, the `add_neighbors` priority order;
* the player-clusters pass, the alliance auto-retreat, transport-ship landings,
  and (in self-drive mode only) the bot attack origination itself.

## What a green cell therefore means

"Given the engine's attack schedule, attack troop counts, border insertion
order, per-player state, cadence counters, friendship pairs and re-created
attacks re-injected at every boundary, ONE TICK of the device's computation
reproduces the engine's claim sets, claim order, owned counts and plane hash."

It does NOT mean the port plays the game. It cannot yet: the env has no oracle
to re-inject from. The only measurement of unaided play is
`OFCUDA_MATRIX_FREEZE_ATTACKS=b`, which freezes the attack re-seed and then
measures how many boundaries the device stays exact on its own: currently 40
boundaries, breaking on a self-originated terra-nullius attack's troop amount.

## Work list: what the env must supply for itself

1. attack origination: when, against what, with how many troops (ported, and it
   now self-drives 86 boundaries past the freeze; the remaining cap is a missing
   feature, not a wrong draw). **Where that code lives matters:** the bot AI is
   currently in `ofcuda_matrix/src/main.rs` (the origination pass) and
   `ofcuda_matrix/src/kernels.rs` (the `bot_ai` decision/metric kernel) - i.e.
   device code in the MATRIX crate. `ofcuda_env/src/bin/gpu_env.rs` includes only
   the canonical `ofcuda_tick/src/core_impl.rs`, so the env does NOT have it.
   Move the AI into the canonical core (as was done for the player-clusters pass)
   before the env can originate anything;
2. TransportShip boat landings: PORTED device-side (the decision to send, the
   sailing, the landing and the cargo transfer), and the self-drive ceiling moved
   102 -> 154 on both freeze arms with `0 engine-created attacks not created`.
   The remaining gap is the ROUTE, and it is a fidelity gap rather than a missing
   draw: the engine plans a ship's water path with the half-resolution HPA
   (`water_hpa.rs plan_water_path` + `upscale_cells` + `fix_path_extremes`), while
   the device substitutes an 8-connected Chebyshev water BFS that reproduces the
   engine's path length in only 4 of 15 measured launches (off by 1-4 otherwise,
   and the engine's routes visibly wobble, which a shortest-path search cannot
   produce). So the 154 stop is a MIS-TIMED LANDING - the device lands ~1 tick
   early and creates an attack the engine had not yet created. Port the water HPA;
   also unexercised: `land()`'s non-TerraNullius branches and the
   aircraft-carrier branch (`transport_ship.rs:404-410`, needs FlightDeck);
3. the border insertion order (device-side `refresh_to_conquer` order);
4. per-player state and the friendship table (derivable device-side);
5. re-creates and merges (ported);
6. `cancel_opposing_land_attacks`, both branches: now PORTED device-side
   (main.rs section 4d-bis), REDUCE and KILL, with the engine's own start bits
   reproduced. The oracle-informed 4e correction is still present and should be
   deleted once the device port is proven load-bearing in self-drive;
7. the cluster cadence counters (ported into the env);
8. the nation post-spawn AI (not ported).

## Ceiling ladder (updated as it moves)

Self-drive (`OFCUDA_MATRIX_FREEZE_ATTACKS=b`) first-divergence boundary, and what
the ceiling turned out to be each time. Every step was proven by re-breaking it
with a control, never by assertion.

| divergence | past freeze (b=20) | ceiling was |
|---|---|---|
| 20 (at the freeze) | 0 | no bot AI at all - the port could not originate an attack |
| 60 | 40 | wrong PRNG draw count (`refresh_to_conquer` draws) |
| 102 | 82 | wrong troop amount on self-originated TN attacks |
| 154 | 134 | unmodelled `cancel_opposing_land_attacks`; then unmodelled boats |
| 166 | 146 | wrong ship ROUTES (Chebyshev BFS vs the half-res water HPA) |
| **166 (current)** | **146** | **player-targeted BOAT LANDINGS**: `TransportShipExecution::init` snapshots `target_small_id` from the END-of-tick plane, so a ship landing on a just-conquered tile attacks that player (engine: owner 441 target 269, cargo 3554.6, claiming 337264 at boundary 166). The device's origination gate refuses non-TN land attacks and creates a TN one. |

Ship ROUTES are now exact: a launch-by-launch comparison of the engine's own
`OF_ENG_BOAT=1` path trace against the device's `OFCUDA_MATRIX_BOATDBG=1`
`DEV_BOAT_PLAN` output gave 12/12 exact tile sequences, including a route
(566150,565150,564150,563150,562150,561150,560150,561151) that falls then rises
and so cannot come from a shortest-path search.

The landed-but-unexercised surface: `land()`'s `target != 0 && !friendly` and
`target != owner && friendly` branches, and the aircraft-carrier branch
(`transport_ship.rs:404-410`, needs FlightDeck), which is not modelled at all.
