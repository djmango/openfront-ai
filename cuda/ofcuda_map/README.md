# ofcuda_map — CUDA map + terrain parity across all 100 maps

Proves that Rust CUDA kernels (cuda-oxide) load the **same map files the engine
loads** and reproduce the terrain plane, land count and dimensions on **every**
map in the corpus, not just Pangaea — and cross-checks every number against the
reference **engine crate** and a third, independent C implementation.

## What it loads

The engine resolves a map as `repo_root/openfront/resources/maps/<key lowercased>`
(`rust/engine/src/core/terrain.rs:37-41,95-113`; `rust/engine/src/rl.rs:66-68,93`;
`rust/engine/src/map.rs:93-99`). Three planes exist per map directory:

| size | file | manifest key | role |
|---|---|---|---|
| `normal` | `map.bin` | `map` | `GameMapSize::Normal` game plane (what the tick runs) |
| `4x` | `map4x.bin` | `map4x` | `Compact` game plane / `Normal` mini plane |
| `16x` | `map16x.bin` | `map16x` | `Compact` mini plane |

## What it computes

* FNV-1a 64 of the terrain plane (`u8` per tile, walked row-major `y*width+x`),
  offset basis `0xcbf29ce484222325`, prime `0x100000001b3`
* FNV-1a 64 of a `u16` state plane (two little-endian bytes per tile)
* the number of land tiles (terrain bit `0x80`)
* the plane dimensions from the manifest

Reduction choice: **exact sequential FNV-1a chain** for the hashes, plus a
**deterministic chunked strided reduction with an ordered fold** for the land
count and for the parallel per-chunk digests. A chunked "order-independent
combine" is not available for FNV-1a: `h1 * P^len2 ^ h2` only holds when the
state update is GF(2)-linear, and integer multiplication does not distribute
over XOR (`(3^5)*3 = 18` vs `3*3 ^ 5*3 = 6`). FNV-1a is inherently serial, so
parallelism is spent where it is exact — `chunk_digests_u8` / `chunk_digests_u16`
cover the whole grid one chunk per thread and every digest is compared against a
host recomputation.

## CLI

```bash
cd /opt/data/workspaces/skg/ofcuda_map

# one map, normal plane (this is the form a later task drives the tick with)
nice -n 10 bash /opt/data/workspaces/skg/ofcuda_env.sh \
    ./target/release/ofcuda_map --map africa

# variants / sweep
... --map africa --size 4x            # or 16x
... --map africa --size all           # normal + 4x + 16x
... --map africa --state              # also the u16 state-plane kernels
... --all --size all                  # every map dir, TSV on stdout
... --maps-root /tmp/synthmaps --all  # any map root
... /path/to/map/dir                  # positional dir still works
```

Map names are case/space insensitive (`--map "World Inverted"` works).
Sweep mode: TSV on stdout, progress + `sweep_pass/sweep_fail` on stderr, exit 1
if any unit disagrees with the in-process host reference.
The wrapper is required at run time (it sets `LD_LIBRARY_PATH=/run/opengl-driver/lib`).

Output columns (one line per map/size), deliberately identical to the reference
tool so `diff` is the whole comparison:

```
name  size  WxH  land  manifest_land  terrain_hash  status
```

## Proof layout

1. **CUDA kernels vs host reference in-process** — `status OK` means the GPU
   hash and land count equal the plain-Rust host computation and every chunk
   digest matched: 300/300 (map,size) units, plus 100/100 with `--state`.
2. **CUDA vs the engine crate** — `/opt/data/workspaces/skg/ofcuda_ref` links
   `openfront-engine` and loads each map through `map::read_manifest` +
   `map::read_terrain_bin` + `GameMap::from_terrain_bytes`, counting land with
   `GameMap::is_land`. `diff cuda_all.tsv ref_all.tsv` → identical, all 300 rows.
3. **Third opinion** — `third_opinion.c` (+ `third_opinion_sweep.sh`, dims from
   `jq`) is a separate implementation in C: same 300 rows, identical.
4. **Engine tick-loadability** — `ofcuda_ref --loadable` calls
   `load_fresh_terrain_from_dir` (the exact call `RlSession::reset` makes) for
   `Normal` and `Compact`: 200/200 OK, no map fails to load.

Known-good anchors reproduced: Pangaea `0xebffa87c2568cc58`, land 420335;
World `0xf46bd55d2054841a`, land 651569.

## Build

```bash
cd /opt/data/workspaces/skg/ofcuda_map
export PATH="/nix/store/qmdxxa88bgbdx31dav3qlssb4rghr14c-gcc-wrapper-14.4.0/bin:$PATH"
nice -n 10 bash /opt/data/workspaces/skg/ofcuda_env.sh cargo oxide build -- --release
```

`cargo oxide build` (not `cargo build`: plain cargo links `kernels::load`
against an undefined `cuda_oxide_artifact_anchor_*`), rejects `--manifest-path`,
and `rust-toolchain.toml` resolves from the CWD — always `cd` into the crate.
