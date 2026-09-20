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
2. TransportShip boat landings: `send_boat_attack_to_nearby_tn` -> TransportShip
   -> `land` -> `add_land_attack_from`, with troops = the ship's cargo. NOT
   ported, and this is the current hard self-drive ceiling (boundary 102 of both
   freeze arms, on `ATTACK 101 456 0 ... uprp5m7q`);
3. the border insertion order (device-side `refresh_to_conquer` order);
4. per-player state and the friendship table (derivable device-side);
5. re-creates and merges (ported);
6. `cancel_opposing_land_attacks`, both branches: now PORTED device-side
   (main.rs section 4d-bis), REDUCE and KILL, with the engine's own start bits
   reproduced. The oracle-informed 4e correction is still present and should be
   deleted once the device port is proven load-bearing in self-drive;
7. the cluster cadence counters (ported into the env);
8. the nation post-spawn AI (not ported).
