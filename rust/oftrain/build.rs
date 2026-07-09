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
fn main() {
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if os != "linux" {
        return;
    }
    if let Ok(libtorch) = std::env::var("LIBTORCH") {
        let lib_dir = format!("{libtorch}/lib");
        println!("cargo:rustc-link-search=native={lib_dir}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib_dir}");
    }
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-ltorch_cuda");
    println!("cargo:rustc-link-arg=-lc10_cuda");
    println!("cargo:rustc-link-arg=-Wl,--as-needed");
}
