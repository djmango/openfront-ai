//! Forces the linker to keep `libtorch_cuda.so` (and friends) as a NEEDED
//! entry in the final `oftrain` binary.
//!
//! `torch-sys`'s own build script (a *library* crate) does pass
//! `cargo:rustc-link-lib=torch_cuda`, but nothing in our Rust code
//! references a `torch_cuda` symbol directly (CUDA kernels register
//! themselves with ATen's dispatcher via static initializers at load time,
//! not via any symbol we call), so the default `--as-needed` linker
//! behavior silently drops the dependency - `ldd` on the resulting binary
//! shows only `libtorch_cpu.so`/`libc10.so`, and every CUDA op fails at
//! runtime with "Could not run 'aten::empty.memory_format' ... CUDA
//! backend". Link args from a *binary* crate's build script (unlike a
//! library's) do reach the final link step, so re-asserting `-ltorch_cuda`
//! wrapped in `--no-as-needed` here is enough to keep it. See
//! https://github.com/LaurentMazare/tch-rs/issues/907 and #1015.
//!
//! The exact same problem hits ROCm/HIP (AMD): `torch_hip`/`c10_hip` get
//! dropped for the identical reason (the HIP backend self-registers via
//! static initializers too, and nothing calls into it directly) - see
//! https://github.com/LaurentMazare/tch-rs/issues/1015, filed against this
//! exact tch-rs/torch-sys version, with the link-arg workaround below
//! confirmed working by the reporter on their ROCm setup. That block is
//! gated on `OFTRAIN_ROCM=1` (set by
//! `scripts/pod_train_v8_rocm.sh`'s bootstrap) rather than unconditional:
//! `-ltorch_hip`/`-lc10_hip` don't exist in a CUDA or CPU libtorch install,
//! so emitting them unconditionally would break the default build by
//! failing the link entirely. This has NOT been independently verified
//! against a real ROCm libtorch in this sandbox (no AMD GPU/ROCm install
//! available) - see rust/oftrain/ROCM.md.
fn main() {
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if os != "linux" {
        return;
    }
    let lib_dir = std::env::var("LIBTORCH").ok().map(|p| format!("{p}/lib"));
    if let Some(ref lib_dir) = lib_dir {
        println!("cargo:rustc-link-search=native={lib_dir}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
    // Only force-link CUDA when the shared libs are actually present
    // (CUDA/GPU libtorch). A CPU-only wheel has no libtorch_cuda.so and
    // would fail the link otherwise - still fine for unit tests / AE
    // parity checks on CPU boxes.
    let has_cuda = lib_dir
        .as_ref()
        .map(|d| std::path::Path::new(d).join("libtorch_cuda.so").exists())
        .unwrap_or(false);
    if has_cuda {
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        println!("cargo:rustc-link-arg=-ltorch_cuda");
        println!("cargo:rustc-link-arg=-lc10_cuda");
        println!("cargo:rustc-link-arg=-Wl,--as-needed");
    }

    println!("cargo:rerun-if-env-changed=OFTRAIN_ROCM");
    if std::env::var("OFTRAIN_ROCM").as_deref() == Ok("1") {
        println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
        println!("cargo:rustc-link-arg=-ltorch_hip");
        println!("cargo:rustc-link-arg=-lc10_hip");
        println!("cargo:rustc-link-arg=-Wl,--as-needed");
    }
}
