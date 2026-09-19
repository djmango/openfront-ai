# ofcuda_cluster — CLUSTER CAPTURE port: status and proof

Spec: `openfront-ai/rust/engine/src/execution/player_clusters.rs`
(`maybe_remove_clusters`, `calculate_clusters`, `flood_border_cluster`,
`surrounded_by_same_enemy`, `is_surrounded`, `get_capturing_player`,
`flood_owned`, `remove_cluster`) + `attack.rs` / `game.rs`
(`largest_incoming_land_attack_from_neighbors`, `is_friendly`, `conquer_one`).
TS authority: `openfront/src/core/execution/PlayerExecution.ts`.

Engine commit of all cases: `522c93d5215c584fd08e9026b6d59c95e548dad4`.
Terrain: pangaea, FNV `0xebffa87c2568cc58` (matched by every case).

## Kernel + CPU reference

* `src/core_impl.rs` is the single implementation. It is copied VERBATIM into
  the device module in `src/main.rs` (`mod kernels`) by `sync_device_core.sh`
  and pinned by `device_core_is_a_verbatim_copy`. GPU-vs-CPU agreement is
  therefore a check on the CUDA lowering, not on a second implementation.
* Kernel A `flood_clusters`: one thread per border index, exact engine DFS
  (mark-on-discovery, `neighbors8` dx-major NW,W,SW,N,S,NE,E,SE), writes the
  discovery order into slot `j`, non-zero only for the component's smallest
  border index → reproduces the engine's cluster list and order.
* Kernel B `decide_remove`: one thread replays `decide_and_remove` (largest
  test on the input plane, then per-cluster `is_surrounded` on the plane as
  already modified by earlier removals, captor via strictly-greater scan over
  first-seen neighbour order, then the two-pass `flood_owned` conquer).
* `src/bin/cpu.rs` (`ofcuda_cluster_cpu`) runs the same code path with no CUDA
  linked — the independent side.

### Trap: hash-map iteration order
There is no hash-map iteration in the decision path. The engine's
`Player::border_tiles` is an **`OrderedTiles`** (insertion-ordered), iterated in
insertion order by `calculate_clusters`; `HashSet` is used only for the
`visited` membership test. `get_capturing_player` uses a first-seen
`Vec<(u16,u32)>`, not a map. So "cluster list = ascending start border index"
and "first-seen neighbour order feeds the `getMode` tie-break" are grounded in
the engine, not invented.

## Cases (4 files, 2 distinct inputs)

| file | tick/victim | delta | order oracle |
|---|---|---|---|
| `cluster-b030-victim33-t1614.json` | 1614 / 33 | 269 tiles → owner 1 | present |
| `cluster-b030-clean.json` | 1614 / 33 (identical input) | 269 → 1 | absent (older dump) |
| `cluster-b030-victim1-t1910.json` | 1910 / 1 (contrast) | 1061 → owner 26 | present |
| `cluster-b030-t2518.json` | 1910 / 1 (identical input) | 1061 → 26 | absent (older dump) |

`clean` == `victim33-t1614` and `t2518` == `victim1-t1910` are the same frozen
input and engine delta; the older dumps simply predate the
`engine_outcome_changed_players` field. Note `t2518`'s filename does not match
its content (content is tick 1910, victim 1).

All inputs come from the dumps (plane b64 + players + friends + attacks +
engine delta); terrain comes from the map dir via `ofcuda_map` and its hash is
asserted against `terrain_fnv`. Nothing decision-relevant is synthesised;
zeroed scratch buffers (marks/generation plane, stack, change list) are
allocator state, not inputs.

## Results (verbatim)

`cargo oxide run` (RTX 5080) — every case PASS:

* `cluster-b030-victim33-t1614.json`: **PASS 14/14**
  (GPU vs CPU bit-exact; both vs engine: 269/269 delta, conquer order, victim
  33→413, captor 1→13388)
* `cluster-b030-victim1-t1910.json` (contrast): **PASS 14/14**
  (1061/1061 delta, order, victim 1→8065, captor 26→18504)
* `cluster-b030-clean.json`: **PASS 10/10**
  (order-oracle checks reported "not evaluated: dump predates
  engine_outcome_changed_players"; delta + whole-plane solved)
* `cluster-b030-t2518.json`: **PASS 10/10** (same, 1061/1061)

`cargo oxide test`: 3 passed (verbatim copy, `neighbors4` N,S,W,E,
`neighbors8` dx-major).

## In-flight defects finished here

The previous worker died mid-edit; these were the blockers, all fixed:

1. `core_impl.rs::flood_owned_emit` — `let n_out: u32;` mutated with `+=`
   (E0384). Made `mut`, re-synced the device copy.
2. `lib.rs::device_copy_text` — left the `// ` prefix of the END-marker line in
   the extracted copy, so the verbatim check could never pass. Strip the partial
   END-marker line.
3. `lib.rs::compare_to_engine` — the "plane differs only on the delta" check
   flagged *any* plane difference as FAIL. Now compares the differing set to the
   recorded delta.
4. `main.rs` `mod kernels` — missing `use super::*;`, so `#[kernel]` /
   `#[launch_bounds]` / `thread` did not resolve and no launcher was generated.
5. `main.rs::flood_clusters` — `thread::index_1d().get()` is `usize`; needed
   `as u32`.
