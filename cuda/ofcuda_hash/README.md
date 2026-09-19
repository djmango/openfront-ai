# ofcuda_hash — the GPU port's verification yardstick

A CUDA implementation of the engine's **per-tick state hash** plus the
**state-plane serializer**, able to consume a GPU-resident state buffer and
produce hashes bit-identical to the authoritative Rust engine (and to the TS
core, which computes the same function over its own `tileStateBuffer`).

```
ofcuda_hash/
  Cargo.toml              cuda-oxide package + rust-toolchain nightly-2026-08-28
  src/lib.rs              host-side shared code: FNV-1a 64, the LE serializer,
                          dump loading, the (deliberately wrong) naive combiner
  src/main.rs             CUDA harness: kernels + the GPU-vs-engine tables
  src/bin/cpu.rs          CPU companion - same dump, same hash, no CUDA at all
  oracle/                 `ofhash_oracle`: standalone crate that links the
                          engine purely as a path dependency and dumps the raw
                          state planes + the engine's own per-tick hashes
  scripts/run_hash_yardstick.sh  end-to-end reproduction
```

## The hash, exactly

```
FNV-1a 64:  h = 0xcbf29ce484222325 (FNV_OFFSET_BASIS), prime 0x100000001b3
            h = (h XOR byte) * prime, mod 2^64, one byte at a time, in order
```

* **state hash** — over the `tileStateBuffer` `Uint16Array` of `w*h` words,
  serialized as 2 little-endian bytes per word (low byte first). A `u16` written
  as 2 LE bytes is *not* the same input as an `f32` holding the same value.
* **terrain hash** — over `w*h` **bytes**, `game.terrainByte(ref)` per tile, i.e.
  the raw terrain plane, *not* the terrain struct.
* The hash is **not associative**: it cannot be combined across chunks. A
  parallel reduction must not XOR or sum per-chunk digests.

## Kernels

| kernel | shape | purpose |
|---|---|---|
| `serialize_state_le` | one thread per output byte | writes the plane as 2 LE bytes/word on the device |
| `fnv_chain_bytes` | 1 thread | the reference order, over the serialized bytes |
| `fnv_chain_bytes_chunked` | 1 thread | same fold, chunk size is a launch parameter — the invariance proof |
| `fnv_chain_u16` | 1 thread | hashes the `u16` plane directly, no serialized staging buffer |
| `chunk_digests_bytes` | 1 thread per chunk | each chunk hashed from seed 0, independently |

Two device paths are checked against each other every tick: the bytes path
(serialize → chain) and the direct `u16` chain.

The ordered fold is single-threaded **by design**: FNV-1a is a strict
sequential dependency. The cost is one dependent `mul` per byte; nothing about
the hash itself is parallelizable, so the yardstick does not pretend otherwise.
`chunk_digests_bytes` is the only parallel piece and it is used purely as
coverage evidence (every byte was read by *some* thread), not to compute the
hash.

## Proof obligations and how they are met

1. **Ordering.** `fnv_chain_bytes_chunked` folds chunk *k* into the running
   state, in ascending order, never restarting. The same 2,000,000-byte buffer
   is hashed at chunk sizes 1, 2, 3, 5, 7, 64, 1024, 4096, 65536, 250000,
   1000000 — one identical value every time (`chunk_count_invariance_gpu true`).
2. **The wrong design is detectably wrong.** `naive_combine` implements the
   concatenation identity that a careless reduction would use
   (`h * P^len ^ digest`); it disagrees with the serial chain on **11/11** chunk
   sizes for every non-degenerate probe. A reduction that combines per-chunk
   digests cannot pass this harness by accident.
3. **byte order / field width.** `serializer_bytes_identical true` — the device
   bytes equal the host serializer's bytes, and the `u16`-direct chain equals
   the bytes chain on every tick.
4. **Non-degenerate input.** A probe must exercise the byte path. The
   post-reset state plane is **2,000,000 zero bytes** (verified:
   `nonzero_bytes 0/2000000`), so its hash is the closed form `h0 * P^n` — a
   length check, *not* evidence that the device read anything. The harness
   therefore runs the ordering/serializer/digest tests on (a) a synthetic plane
   with 1,987,073/2,000,000 nonzero bytes and (b) the densest real plane in the
   dump, and prints the nonzero density of each probe.

## Results

`verdict PASS` in all five runs below; every per-tick row shows
`gpu_serialized == gpu_direct_u16 == engine_expected`, and the CPU companion
reproduces the same values in its own table.

| dump | window | ticks | GPU matches | distinct hashes | verdict |
|---|---|---|---|---|---|
| `/tmp/ofhash_parity` (RL-session, stage 0, 1 tick/decision) | 2..131 | 130 | 130/130 | 130 | PASS |
| `/tmp/ofhash_record_late` (record, ticks 500..699) | 500..699 | 200 | 200/200 | 200 | PASS |
| `/tmp/ofhash_record` (record, ticks 1..220 — spawn phase) | 1..220 | 220 | 220/220 | 2 | PASS |
| terrain, all runs | — | 1 | 1/1 | 1 | PASS |
| post-reset state (Pangaea) | — | 1 | 1/1 | 1 | PASS |

The two published constants are reproduced from scratch on the device:

```
terrain          0xebffa87c2568cc58   (1000000 raw terrain bytes)
post-reset state 0x6334dfb980453d25   (1000000 u16 words = 2000000 zero bytes)
```

**Window choice matters.** `curr-b002-s1-pangaea` spends its first ~300 ticks in
the spawn phase, where the state plane is byte-identical from tick 2 onward: the
ticks 1..220 dump yields only **2** distinct hashes. The primary record evidence
is therefore the ticks **500..699** window (200 distinct planes), and that
window is independently pinned against `hash_parity.sh`'s
`tick_dump` ndjson output (0/200 `gameHashBits` mismatches).

## Independent cross-checks performed

* `ofhash_oracle` output vs the **prebuilt** `rust/target/release/parity_trace`
  binary: 130/130 per-tick state hashes identical.
* `ofhash_oracle` output vs the **TS core** (`scripts/ts_parity_trace.ts
  --decision-ticks 1 --decisions 130`): 130/130 per-tick state hashes identical.
* `ofhash_oracle` record replay vs `hash_parity.sh`'s `tick_dump` ndjson:
  220/220 and 200/200 `gameHashBits` identical.
* The post-reset plane is byte-identical to the record's tick-1 plane, and both
  equal `h0 * P^(2 000 000) mod 2^64 = 0x6334dfb980453d25` (checked in Python),
  so the published constant is reproduced by two unrelated code paths.

## Run it

```bash
cd /opt/data/workspaces/skg/ofcuda_hash
bash scripts/run_hash_yardstick.sh
```

Or by hand (the exec wrapper, not `source`):

```bash
OF=/opt/data/workspaces/skg/ofcuda_hash
/opt/data/workspaces/skg/ofcuda_env.sh cargo oxide build          # cwd = $OF
/opt/data/workspaces/skg/ofcuda_env.sh ./target/release/ofcuda_hash     /tmp/ofhash_record_late
/opt/data/workspaces/skg/ofcuda_env.sh ./target/release/ofcuda_hash_cpu /tmp/ofhash_record_late
```

Full outputs: `/tmp/run_late_gpu.out`, `/tmp/run_parity_gpu.out`,
`/tmp/run_record_gpu.out` (+ `_cpu` companions).

## Known limits

* Ticks 1..220 of the record are a frozen spawn-phase state; only the
  ticks 500..699 window is real evidence of per-tick agreement.
* Measured cost in this harness: 13.45 s wall for the 200-tick dump vs 9.79 s
  for the 130-tick dump, i.e. ~52 ms per tick (70-tick delta / 3.66 s; the
  box is shared with a live trainer, so treat it as an upper-bound-ish figure).
  It is dominated by the two single-threaded ordered chains over 2 000 000
  bytes each and by a 2 MB host readback per tick, not by the device work.
  This is a correctness yardstick, not a performance-optimized path; a batched
  variant (many planes per launch, one readback) is the obvious next step.
* The per-tick "engine_expected" column comes from the engine library linked
  into `ofhash_oracle`. It is not taken on trust: it is cross-checked against
  two prebuilt current-build binaries and the TS core as listed above.
