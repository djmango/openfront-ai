# ROCm/AMD (MI300X) support - status

Additive port of `oftrain` to also support AMD ROCm GPUs, for eventually
running on MI300X pods on RunPod. **CUDA/NVIDIA is unchanged and remains
the default** - every change here is gated behind an opt-in env var or
lives in a separate script, so the existing training path is unaffected.

This document is deliberately explicit about what has and has not been
verified, since none of this was tested against real AMD/ROCm hardware -
**no AMD GPU was available in the sandbox this was developed in.**

## Why this is possible with (almost) no code changes

`policy.rs`, `train.rs`, `batch.rs`, `main.rs`, and `vecenv.rs` only use
plain `tch::nn` layers and generic `tch::Device::Cuda(n)` /
`device.is_cuda()` - there are no custom CUDA kernels anywhere in this
crate. PyTorch's ROCm build deliberately reuses the exact same
`torch.cuda` / `at::Device(DeviceType::CUDA)` API surface for AMD GPUs
(HIP mirrors CUDA's API specifically so software like this doesn't need
backend-specific code), so `--device cuda:0` etc. keep working unchanged
against a ROCm libtorch. **This part is architectural reasoning from
reading the existing code + PyTorch's documented ROCm design, not
something re-verified end-to-end on hardware in this change.**

## What changed, file by file

### `rust/oftrain/build.rs`

Already had a fix (predating this change) forcing the linker to keep
`libtorch_cuda.so`/`libc10_cuda.so` linked into the final binary, working
around GNU `ld`'s `--as-needed` default silently dropping them (nothing in
our Rust code calls a `torch_cuda` symbol directly - CUDA's backend
self-registers via static initializers at load time). This change adds the
exact same fix for HIP (`libtorch_hip.so`/`libc10_hip.so`), **gated on
`OFTRAIN_ROCM=1`** so it's a complete no-op for the default build (the
libs don't exist in a CUDA/CPU libtorch install; emitting the link args
unconditionally would break the default build by failing the link step).

Source for the fix: <https://github.com/LaurentMazare/tch-rs/issues/1015>,
filed against our exact pinned `tch`/`torch-sys` version (0.24.0), with the
reporter confirming the link-arg workaround fixed `Cuda::is_available()`
returning `false` on their real ROCm (RX 7900 XTX, ROCm 7.2) setup.

**Verified in this sandbox:** the conditional gating - built with
`OFTRAIN_ROCM` unset against the existing CPU libtorch venv, confirmed
identical build output/behavior to before this change (see "What's
verified" below).

**NOT verified:** whether the fix actually resolves the link-drop issue
against a real ROCm libtorch build - this sandbox has no ROCm install to
link against. This is "copied from a GitHub issue confirmed working by its
reporter on their setup", not independently reproduced here.

### `rust/oftrain/src/gpu_util.rs`

Added `query_rocm_smi()`, parallel to the existing `query_nvidia_smi()`,
wired in as a fallback in `GpuUtilSampler::start()`:
`query_nvidia_smi().or_else(query_rocm_smi)`. This is a "try the next
thing" pattern, not a hard `--backend` flag - `GpuUtilSampler` just needs
*some* working utilization source, so on an AMD pod (no `nvidia-smi`)
it transparently falls through to `rocm-smi`.

Invocation: `rocm-smi --showuse --showmeminfo vram --csv`. Output shape
(confirmed via a real captured invocation, not guessed -
<https://github.com/marimo-team/marimo/issues/9237>):

```
device,GPU use (%),VRAM Total Memory (B),VRAM Total Used Memory (B)
card0,0,21458059264,27856896
```

This differs from `nvidia-smi`'s `--query-gpu=...,--format=csv` in two
ways that matter for parsing: memory is reported in raw bytes rather than
a used/total MiB pair, and the column *order* comes from a header row
rather than a fixed `--query-gpu` field list - so `parse_rocm_smi_csv`
looks up columns by name (fails closed / returns `None` on an unexpected
header, rather than misparsing) instead of reusing `query_nvidia_smi`'s
fixed-position parsing.

**Verified in this sandbox:** `parse_rocm_smi_csv` has unit tests
(`cargo test -p oftrain`) against the real captured single-GPU sample
above, a synthesized-but-format-consistent 2-GPU extension of it, and
malformed/missing-column inputs. These pass and pin the parsing logic.

**NOT verified:** the actual `rocm-smi` binary/CLI invocation itself, on
real hardware - no `rocm-smi` available in this sandbox. If a real ROCm
pod's `rocm-smi` version emits a meaningfully different column set (e.g.
older/newer ROCm releases sometimes add/rename `--showmeminfo` fields),
`query_rocm_smi` fails closed (`None`, same as if the command weren't
found at all) - training still runs, just without GPU-util reporting/
`--auto-scale-envs` support until the parser's column names are updated
to match. `amd-smi` (AMD's newer, longer-term-supported CLI replacing
`rocm-smi`) was considered instead but `rocm-smi` was kept because it's
the invocation explicitly named in this task's context and is still
shipped/documented on ROCm images at time of writing; switching to
`amd-smi` (or trying both) would be a reasonable, easy follow-up once
tested against a real pod.

### `scripts/pod_train_v8_rocm.sh` (new)

ROCm counterpart to `scripts/pod_train_v8.sh`, kept as a **separate
script** rather than branching inside the existing one, specifically to
avoid any risk of regressing the proven CUDA path. Same structure/safety
checks: deployed-commit-matches-origin assertion, crash-loop backoff,
periodic HF checkpoint sync to `djmango/openfront-rl`, resume-from-HF on a
fresh/wiped pod.

Differences from the CUDA script, and why:

- **Torch version pin.** `torch-sys` 0.24.0 (our pinned `tch` version)
  hard-requires PyTorch `"2.11.0"` exactly (see its `build.rs`
  `version_check` / `TORCH_VERSION` constant) unless
  `LIBTORCH_BYPASS_VERSION_CHECK=1` is set. Checked
  <https://download.pytorch.org/whl/rocm7.0/torch/> and
  <https://download.pytorch.org/whl/rocm6.4/torch/> directly (actual wheel
  index listings, not documentation that could be stale): as of this
  writing, the newest ROCm torch wheel is `2.10.0+rocm7.0` - **no
  `2.11.0` ROCm build exists yet**. So this script pins
  `ROCM_TORCH_VERSION=2.10.0` against the `rocm7.0` index and sets
  `LIBTORCH_BYPASS_VERSION_CHECK=1` in the generated `.cargo/config.toml`.
  If a matching or newer ROCm wheel ships later, prefer bumping
  `ROCM_TORCH_VERSION` and dropping the bypass.
- **`OFTRAIN_ROCM=1`** is set (both via `.cargo/config.toml`'s `[env]` and
  explicitly on the `cargo build` invocation) so `build.rs` emits the HIP
  force-link fix.
- **Sanity check** greps for `libtorch_hip.so` in `readelf -d` output
  instead of `libtorch_cuda.so` (ROCm equivalent of the existing
  CUDA-not-actually-linked footgun check).
- **`LD_LIBRARY_PATH`** includes `/opt/rocm/lib` (where the ROCm runtime's
  own shared libs like `libamdhip64.so`/`libhsa-runtime64.so` typically
  live on ROCm docker images, outside the pip-installed torch wheel)
  instead of the CUDA script's `nvidia/cuda_nvrtc/lib` (NVRTC has no direct
  ROCm equivalent needed here - this trainer doesn't use `torch.compile`).
- Everything else (repo bootstrap, Rust toolchain install, Node/bridge
  setup for `--node-fraction`, HF checkpoint sync/resume, crash-loop
  backoff, `--device cuda:0` - unchanged, see main.rs's `parse_device`)
  is unchanged from `pod_train_v8.sh`.

**NOT verified:** this entire script, end-to-end, since running it
requires an actual ROCm pod (explicitly out of scope for this task - no
GPU pod was provisioned). Every version/URL/flag in it was checked against
a primary source (the actual torch wheel index, the actual `torch-sys`
build.rs source in this repo's Cargo registry cache, the actual GitHub
issue for the link bug) rather than assumed, but "checked against
documented sources" is not the same as "ran successfully on hardware".

## What "done" looks like - verification status

| Item | Status |
|---|---|
| `build.rs` conditional on `OFTRAIN_ROCM=1`, default build unaffected | **Verified**: built + tested with `OFTRAIN_ROCM` unset against the CPU libtorch venv; identical outcome to before this change. |
| `build.rs` HIP fix itself, against a real ROCm libtorch | **Not verified** - no ROCm libtorch available. Sourced from a GitHub issue confirmed working by its reporter, not independently reproduced. |
| `gpu_util.rs` `parse_rocm_smi_csv` parsing logic | **Verified** via unit tests against a real captured sample. |
| `gpu_util.rs` `query_rocm_smi` actually invoking `rocm-smi` correctly on a real pod | **Not verified** - no `rocm-smi`/AMD GPU available. |
| `pod_train_v8_rocm.sh` end-to-end | **Not verified** - requires a real ROCm pod (out of scope for this task). |
| Default CUDA/CPU build+test path unaffected | **Verified** - see below. |

## Confirming the default (CUDA/CPU) path is unaffected

With `OFTRAIN_ROCM` unset (the default - nothing sets it unless you're
running `pod_train_v8_rocm.sh`):

```bash
cd rust
cargo build --release -p oftrain --features native-engine   # succeeds, same as before this change
cargo test --release -p oftrain                              # 29 tests pass (24 pre-existing + 5 new gpu_util rocm-smi-parser tests)
```

No existing test, existing default-build behavior, or CUDA-path code was
touched - `gpu_util.rs`'s change is an added fallback function plus one
`.or_else(...)` in the sampler's polling loop (still tries `nvidia-smi`
first, unconditionally, exactly as before), and `build.rs`'s change is
gated behind a new env var nothing else sets.
