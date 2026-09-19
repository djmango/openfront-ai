# ofcuda_prng

Parity harness: the engine's `PseudoRandom` (sfc32) + spawn-tile selection, ported to a
Rust CUDA kernel with [cuda-oxide](https://github.com/NVlabs/cuda-oxide), proven
**bit-exact** against the engine.

## What is compared

`reference.txt` is produced by the engine (`openfront-ai/rust/engine/src/bin/prng_dump.rs`)
and holds, for 4 seeds:

| reference line | engine source | port check |
| --- | --- | --- |
| `stream`    | `prng.rs:44` (`(t as u32)/2^32` raw bits) | first 24 `next()` u32 per seed |
| `draws`     | `prng.rs:52-56` `next_int(lo,hi)`        | first 16 draws for `(0,100) (40,80) (0,7) (0,5)` |
| `sem`       | `prng.rs:60-110`                         | `chance`, `rand_element`, `shuffle_array` semantics |
| `pstream`   | `simple_hash(player_id)`-seeded stream    | `simple_hash` + warm-up via a second seed shape |
| `seed`/`human`/`spawn` | `util.rs::simple_hash`        | `simple_hash(game_id)` and the engine's `simple_hash(pid)+simple_hash(game_id)` spawn seed |
| `spawn`/`human` tile,x,y | `execution/spawn_util.rs::rand_tile` | tile selected by the full attempt loop |

## Traps the port had to get right

1. `PseudoRandom::new` does 4 splitmix-style splits and then **12 warm-up draws** (`prng.rs:20-40`).
2. `next()` is `(t as u32) / 2^32` - a **u32 division**, not `t as f64 / 2^32` (`prng.rs:44`).
3. `rand_element` on an **empty list draws nothing** (`prng.rs:88-95`).
4. `shuffle_array` does **len - 1** draws (`prng.rs:103-110`).
5. The spawn footprint disk is `dist2_center(...) <= 16.0` with the root at `x - 0.5`
   (`spawn_util.rs` + `map.rs`), i.e. tile offsets **-4..=3** per axis, and it is a BFS over
   cardinal neighbours, so unreachable corner tiles must not count as invalid.
6. Every attempt is exactly **2 `next_int` draws**, max `MAX_SPAWN_TRIES = 1000` attempts.
7. An explicit tile (the human path) returns **before** `rand_tile` -> zero draws.

## Build / run

```
cd /opt/data/workspaces/skg/ofcuda_prng
/opt/data/workspaces/skg/ofcuda_env.sh cargo oxide run --bin ofcuda_prng
```

`ofcuda_env.sh` is an exec wrapper (never `source` it); it sets
`LD_LIBRARY_PATH=/run/opengl-driver/lib` so the CUDA driver is reachable.

* `src/lib.rs`  - CUDA-free shared code: reference parser, `simple_hash`, `PseudoRandom`,
  spawn selection, report/comparison. The CPU bin and the GPU host use *literally the same*
  code, so a GPU/CPU difference isolates the kernel.
* `src/main.rs` - the CUDA kernels plus the host driver. Every compared value is read back
  out of a `DeviceBuffer` that only a kernel launch wrote.
* `src/bin/cpu.rs` - CPU companion (`cargo run --release --bin ofcuda_prng_cpu reference.txt`).

Outputs: `comparison_gpu.txt` (written by the program), `comparison_gpu_raw.txt` (full
`cargo oxide run` log, including the sm_120a build).

## Result

On an RTX 5080 (sm_120a): engine-vs-GPU and engine-vs-CPU are element-for-element equal on
every row of the table, `ALL_BIT_EXACT true`.
