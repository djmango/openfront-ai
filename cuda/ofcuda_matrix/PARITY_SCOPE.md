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

| **227 (current)** | **207** | **a bot-AI/landing origination gap**: the engine creates attack owner 440 target 227 via `add_land_attack_from -> AttackExecution::init` and the device originates nothing, so player 440's 13 tiles flip to player 227. Note the same 13-tile flip is reported from either side depending on which player the comparison loop reaches first (the diagnostic's iteration order varies between arms; the plane hash does not). |

### The `contract` fast-math flag: a device-path 1-ULP trap

The device backend marks IR with LLVM's `contract` and drops `#[inline(never)]`,
so any expression of the form a*b + c*d compiles to `fma.rn.f64` and differs from
host Rust by 1 ULP. This was the true cause of the long-carried "1-ULP troop
drift" (boundary-192 attack 441->269) that capped several cells over three
sessions - not a type width, not a rounding mode, and never to be papered over
with an epsilon. Fix with a volatile-read rounding barrier (`mul_round` in
`ofcuda_matrix/src/kernels.rs`) and verify with
`grep -c 'fma.rn.f64' ofcuda_matrix.ptx` == 0. Treat any newly-exercised device
path that computes a multiply-add as suspect until that count is checked.

| **280 (current)** | **260** | **the `sendBoatAttack` branch of the bot AI**: when the chosen target shares no land border, the engine does not create a land attack - it launches a TransportShip (`try_send_player_attack` -> `boat_attack_destination_to_player` -> `add_transport_attack`). 42 launches over 500 ticks, 2 inside the self-driven range; first at tick 278, owner 398 -> target 68. |

The bot attack ladder is now ported in full apart from that branch: retaliation, the
shuffled-random-target pick (all 13 device picks reproduce the engine's shuffled
neighbour one-for-one), terra-nullius expansion, merges, cancels, boat sends and
landings, and exact ship routes. Everything above is device state only - no oracle
value is written into the device anywhere in the new code.

| **413 (current)** | **393** | **a single 4-tile flip between two players at engine tick 416** (plane hash 0x7bc2de3015ba3247 vs engine 0xb32d2cb08a891e7b; reported as player 90 884/880 or player 486 972/976 depending on scan order - same event, opposite signs), plus 1 engine-created attack the device did not create and 1 eviction it did not apply at the same break. Pre-dates the income fix (a pinned pre-fix control has identical claims/hash/counts at 413). |
| **401 (fixed)** | **381** | **the income second pass was seeded from the record tick-start tile count.** In the engine, `PlayerExecution::tick` applies income BEFORE that player own cluster pass, and player execs run in ascending order, so a captor has already received tiles from a lower-ordered victim when its income runs (tick 404: victim 260 captor 425, tiles 893 -> 988). Fixed by applying income at the player own exec turn after the device cluster pass. |

## Reading the divergence line

The reported player in `first divergence: boundary N ... player P engine A tiles vs device B` is NOT run-stable: the same divergence is reported from whichever side of a tile flip the mismatch scan reaches first, and the two sides differ in sign (e.g. 90: 884/880 vs 486: 972/976 at boundary 413). Compare the plane hash and the counts across runs; do not treat the reported player as an identifier. Making it deterministic is an open cosmetic fix.

| **435 = capacity wall** | **414** | **neither a divergence nor a parity failure: at boundary 435 the device needs a 1025th attack slot while the record holds 149 live attacks there.** Both freeze arms are byte-exact to 434/434 (hash 434/434, claims 76578/76578, owned counts 212226/212226, troops 79274/79274 for b=20 and 78984/78984 for b=60) with self-drive tallies 0/0/0 and first divergence NONE; the 413 divergence is closed. The record-driven plain replay burns 1024 slot indices by 437 while staying claim/hash-exact, so the burn is the allocation scheme never reusing freed slots, NOT self-drive leaking attacks. |

### Two harness facts that make readings lie

- **A fresh dump lists ZERO ATTACK rows at its own final boundary** (t413, t434, t437) while the complete t500 dump lists 149 there, so pc_create/pc_evict/pc_troops at a cell final boundary are an artefact and must not be read as agreement evidence. Genuine extras through boundary 250 are near zero (pc_evict 3).
- **The reported first-mismatching player in the divergence line is a scan artefact, not an identifier.** The scan iterated a HashMap (randomised order), so one 4-tile flip was reported from either side (player 90 engine 884 vs device 880, or player 486 engine 972 vs device 976 - the same flip, opposite signs). Compare hashes and counts, never the printed player. It is now sorted by sid, so at least it is run-stable.

| **500/500 (record end)** | **480** | **no divergence: both freeze arms run the record full window with hash 500/500, claims 83545/83545, owned counts 244500/244500, troops 87798/87798 (b=20) and 87508/87508 (b=60), self-drive totals 0/0/0, first divergence NONE.** The previous stop was never a divergence: the host allocator used slots.len() as the index and never freed a position, so MAX_SLOTS bounded allocations ever made rather than live attacks. With reuse it bounds concurrency - 488 positions against a 1024 ceiling, and 488 is the peak concurrent count in the record itself (boundary 79). The plain replay now also completes 500 boundaries; its only remaining divergence is the boundary-296 engine eviction. |

### Harness fact: the dump final boundary (FIXED 40afc85)

The oracle emitted its ATTACK section only when b < ticks, so every cell carried ZERO attack rows at its own final boundary and pc_create/pc_evict/pc_troops there could not be read as agreement evidence. It is now emitted unconditionally; regenerating the t500 dump adds exactly 131 ATTACK 500 rows and nothing else. Plane hashes, tick claims, owned counts and troops never read that section, so earlier numbers stand unchanged.

### Harness fact: slot ids are not stable identifiers

A position is now reused as soon as its attack dies, so an old log reading slot 560 and a new one reading slot 19 are the same divergence (sid 161). Compare sid/owner/target/boundary, never slot position.
