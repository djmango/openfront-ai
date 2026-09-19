# Measured speed of the CUDA env vs the CPU env — 2026-09-18, this box

Machine: 8-core, RTX 5080 (16 GB), driver 610.57.04, CUDA UMD 13.3. A **live
training run shares the box** for every number below: `./puffer train
--vec.total_agents=64 --vec.num_threads=8 --train.horizon=128` (PID 2524207,
405 % CPU). Measured contention during the runs: GPU `utilization.gpu` sampled
0–7 %, 51.5 W, 2429 MiB held by the trainer; CPU `loadavg` 7.0–8.1 on 8 cores
(saturated). Every CPU number below is therefore a **contended lower bound**, and
every GPU number was taken while the training process held the device.

## Headline

**There is no GPU ticks/s to report, and k for the tick is unmeasurable: the
composed tick has no device path.** `ofcuda_env` launches no kernel — grep for
`cuda_core|cuda_device|cuda_host|kernel|CudaContext|DeviceBuffer` in
`src/main.rs` and `src/lib.rs` matches nothing (the only matches are comments in
the new `src/bin/bench.rs`). `Attack::tick`, the budget, the heap, the plane
update and the hash all execute on the HOST. The honest answer is a host number,
labelled as one.

What *is* measured, with commands:

| quantity | value | how |
|---|---|---|
| composed tick, HOST, proven window | **142 ticks/s** (7.03 ms/tick) | `ofcuda_env --dump /tmp/envdump/fresh.ndjson --t0 300 --t1 1301` minus `--t1 700`, 601-tick delta |
| same, later/heavier sub-window | **130 ticks/s** (7.72 ms/tick) | `--t0 600 --t1 1301` (701 ticks) minus `--t0 300 --t1 700` (400) |
| same, early window (no attacks yet) | **183 ticks/s** (5.47 ms/tick) | `--t0 2 --t1 1301` (1299) minus `--t0 600 --t1 1301` (701) |
| decisions/s (1 decision = 15 ticks) | **9.5** on the 142 ticks/s figure | 142 / 15 |
| live trainer, its own SPS | **114.5 – 133.8** (rolling) | `openfront-train/current.log`, dashboard blocks |
| live env share of wall | 94–95 % | same blocks |
| live Train phase (fixed per-iteration cost) | **2.633 s** (3–4 % of wall) | same blocks, `Train 2s 633` |
| engine expansion step, HOST | **0.038 ms/tick** (0.5 % of the tick) | `ofcuda_env/target/release/bench` |
| FNV-1a-64 over the 2 MB plane, HOST | **1.978 ms** | same |
| FNV stage, DEVICE vs CPU companion | 64.5 vs 8.3 ms/tick → **k = 0.13** | `ofcuda_hash` / `ofcuda_hash_cpu` on `/tmp/ofhash_record_late` |

## 1. Method, host composed tick

`main.rs` parses the whole 178 MB dump before the tick loop, so the parse cost is
identical for every window and cancels in a difference of two windows on the
same dump. Three repetitions each, `nice -n 10`, wall clock `date +%s%N`:

```
B=./target/release/ofcuda_env
for w in "300 700" "300 1301" "600 1301" "2 1301"; do ... nice -n 10 $B --dump /tmp/envdump/fresh.ndjson --t0 $s --t1 $e --counts-n 1; done
```

| window | ticks | wall (ms, 3 runs) | agreement printed |
|---|---|---|---|
| [300,700) | 400 | 4237, 4353, 4181 | 200/200 |
| [300,1301) | 1001 | 8547, 8440, 8459 | 2002/2002, no mismatch |
| [600,1301) | 701 | 6573, 6588 | 1110/1402 (invalid window: starts mid-attack) |
| [2,1301) | 1299 | 10400 | **2598/2598, no mismatch** |

Marginal cost per tick = Δwall / Δticks:
* (8482 − 4257) / 601 = **7.03 ms → 142.3 ticks/s** (the README's proven window)
* (6580 − 4257) / 301 = **7.72 ms → 129.6 ticks/s**
* (10400 − 6580) / 698 = **5.47 ms → 182.7 ticks/s** (ticks before the first attack)

The [2,1301) run re-confirms the README's claim set on the current build:
`CLAIM-IDENTITY AGREEMENT: 2598/2598`, `FIRST MISMATCH: none`. [600,1301) is
*narrower* agreement for a known reason the README documents: a window may not
start mid-attack, so it is a method caveat, not a regression.

## 2. Where the host tick actually goes (bench bin, added)

`src/bin/bench.rs` (new, measurement only; `Cargo.toml` gained one `[[bin]]`; no
existing file touched, all 6 lib tests still pass) times the parts in-process on
the real record data, 200 iterations each:

```
tick  players  border∑  owned∑  recon_ms  planeb_ms  fnv_ms  tick_ms  refr_ms
 320        2       46     121     0.025      0.022   1.986    0.002    0.006
 600        2      400    5070     0.028      0.024   1.963    0.012    0.027
 900        2      865   13591     0.031      0.048   1.972    0.026    0.045
1200        2     1341   25322     0.038      0.034   1.978    0.038    0.067
```

* `fnv_ms` = one `state_hash` over the 1e6 u16 (2 MB) plane = **1.98 ms**;
* `recon_ms` = the driver's own reconstruction (clone `owned_tiles` +
  `border_order` + `owned_order` per player, then `state_plane`) = 0.03–0.04 ms;
* `tick_ms` = `Attack::refresh` + `Attack::tick` — the engine expansion step the
  CUDA kernel implements — = **0.038 ms at tick 1200**, i.e. **0.5 % of the
  7.03 ms measured tick**.

The driver pays **three** plane builds and **three** `state_hash` calls per tick
(`main.rs:349`, `:566`, `:576`, `:577`, `:753`): 3 × 1.98 + 3 × 0.03 + the
player clones ≈ 6.1 ms of the 7.03 ms. So **~87 % of the measured host tick is
FNV hashing of the ownership plane**, and the engine logic is a rounding error
in it. Any device port's payoff is bounded by that, not by the expansion.

## 3. Device side: what was and was not measured

* **Not measured: device ticks/s.** The only device-resident per-tick work that
  exists is `ofcuda_tick`'s `engine_life_claims` / `compose_tick` +
  `fnv_state_hash`, and its harness launches **64–256 threads on 24–27 claims**
  for 2 cases plus a 32-tick window. Its GPU binary took **18,896 ms** end to end
  (`nice -n 10 ./target/release/ofcuda_tick`), and that wall time is dominated by
  the host-side search loops (e.g. 3000 cursors × 32 ticks) and 99 MB of
  readbacks, not by any kernel. Isolating kernel time would need a new harness
  (a new copy of the device core), which I declined to create. **So no GPU
  ticks/s is stated.**
* **Measured: the one stage where a device and a host implementation exist on
  identical input.** `ofcuda_hash` vs `ofcuda_hash_cpu` on the same 200-plane
  dump (`/tmp/ofhash_record_late`), 2 MB serialize + ordered FNV chain per tick:

  | | wall | per tick | result |
  |---|---|---|---|
  | GPU path | 12,895 ms | 64.5 ms | `verdict PASS`, 200/200 |
  | CPU companion | 1,654 ms | 8.3 ms | same values |

  **k_hash = 8.3 / 64.5 = 0.13 — the device path is 7.8× slower.** This is
  harness wall time (the GPU path also does a 2 MB host readback per tick and the
  device FNV chain is one thread by construction: FNV-1a is a strict sequential
  dependency), so it is not a pure kernel comparison — but it is a counted,
  reproducible comparison of the two implementations of that stage, and the
  direction is not ambiguous.

## 4. k and the Amdahl consequence, kept apart

* **k (GPU ticks/s ÷ CPU ticks/s): undefined / unmeasurable.** The composed CUDA
  environment does not produce a tick on the device; `k` for the tick has no
  denominator-free meaning today. Writing a number here is exactly how the
  retracted ~700× happened.
* **k for the one measured stage: 0.13** (above), device slower.
* **Amdahl.** With the loop 94–95 % env-bound (measured, above), end-to-end
  speedup for an env `k` times faster is `S = 1 / (0.05 + 0.95/k)`.
  * Cap: `lim k→∞ S = 1/0.05 = 20×`. **No env speedup can beat 20× end to end.**
  * Using the measured device rate for the whole env share (k = 0.13):
    `S = 1 / (0.05 + 0.95/0.13) = 0.136×`, i.e. **7.4× slower end to end** —
    the arithmetic consequence of k < 1, stated as arithmetic on a measured
    stage, not as a claim about the whole env.
* **Fixed per-iteration cost** the loop pays regardless of env speed, measured
  from the live log: **Train 2.633 s + Model 2.631 s** per iteration (Copy 30 ms,
  Misc 2 ms). At the 20× cap the iteration becomes dominated by these, which is
  why more envs per launch — not a faster env — is the only lever past 20×.

## 5. Batch-size scaling

**Not supported, so not measured.** No harness here launches more than one
env/case/plane per kernel: `ofcuda_tick`'s launches are per-case (64–256 threads
on one attack's frontier) and `ofcuda_hash`'s chain kernels are 1 thread by
construction; `ofcuda_hash/README.md` states the batched variant (many planes per
launch, one readback) is "the obvious next step", i.e. not implemented. There is
therefore no scaling point to report rather than a fabricated one.

## 6. The live trainer, same machine, same minutes

From `openfront-train/current.log` dashboard blocks (Epoch 1457–1461, Uptime
~1 h 7 m):

```
Env openfront Evaluate 59s 13 95%   | Env 58s 22 94% | Copy 30ms 0%
SPS 132.6 | Epoch 1460 | Train 2s 633 4% | Model 2s 631 4% | Uptime 1h8m56s
To go 0d 16h 49m 56  (8M of 20M steps left)
```

Unit derivation (this is what fixes SPS's meaning, not an assertion): dashboards
are 68 s apart (Uptime 4136 → 4204 s) and one epoch = horizon × envs = 128 × 64 =
8192 agent-steps, giving 8192/68 = **120.5 agent-steps/s**, matching the printed
SPS 119.3–132.6. The countdown agrees: 8,000,000 remaining / 16.83 h = 132/s.
**So SPS = agent-steps/s for the whole 64-env batch**, i.e. aggregate
decisions/s ≈ SPS ≈ 115–134 across 64 envs (≈ 1.8–2.1 decisions/s per env), and
aggregate engine ticks/s ≈ 15 × that = **1.7k–2.0k ticks/s** — which is the
figure the task's "~115 decisions/s at horizon 128" corresponds to, now derived
rather than quoted.

## 7. Assumptions and caveats, all of them

1. **Host, not device.** Every tick number is a host number by construction
   (section 0/1); the CUDA crates in `Cargo.toml` are dependencies of the crate,
   not a device code path in `main.rs`.
2. **Contention.** CPU load 7.0–8.1/8 during all runs; trainer held 2429 MiB and
   the device showed 0–7 % util. Treat the host rates as lower bounds and the
   device rates as contended. Nothing was killed, paused, or reconfigured to
   clean this up; `openfront-train`/`sampler`/`watchdog` were untouched.
3. The three `state_hash`/`state_plane` calls per tick are **harness** work
   (record reconstruction + the FNV yardstick), not engine work. A device port
   would carry the plane once, so those 3 should not be mapped onto a device tick
   — that mapping would be a projection, and is not made here.
4. `bench.rs` uses the record's real owner border and plane but a fixed 1363.0
   start troops (the record's attack troop count is not in `PlayerRec`), so the
   `tick_ms` column is shaped like the real attack, not bit-equal to it.
5. `k_hash` is harness wall time, not kernel time (readback + serial chain
   included); it is labelled as such everywhere above.
6. SPS's unit was derived from epoch spacing and the countdown, not from trainer
   source (the dashboard strings live in the prebuilt `./puffer`); if the counter
   is instead per-env, multiply the decision rate by 64. Flagged, not hidden.
7. `[600,1301)` is not a valid agreement window (mid-attack start); it is used
   for timing only.
8. Window lengths: 400, 601(delta), 698, 701, 1001, 1299, 1301 ticks exactly as
   tabulated; nothing was extrapolated across dump boundaries.
