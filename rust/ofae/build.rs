//! Keep `libtorch_cuda.so` NEEDED in the ofae binary (same reason as
//! `oftrain/build.rs`: CUDA kernels self-register via static init, so
//! `--as-needed` otherwise drops the dependency and every CUDA op fails).

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
}
