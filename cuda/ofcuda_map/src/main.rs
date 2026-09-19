//! `ofcuda_map` - GPU (cuda-oxide) side of the map/terrain parity level.
//!
//! Proves that Rust CUDA kernels load the *same* map files the engine loads and
//! reproduce the terrain plane byte-for-byte on **every** map, not just one.
//!
//! Numbers produced, over the tile grid flattened to `index = y*width + x`:
//!   * FNV-1a 64 of the terrain plane (`u8` per tile)
//!   * FNV-1a 64 of the state plane (`u16` per tile, each element fed as two
//!     little-endian bytes)
//!   * the number of land tiles (terrain bit 0x80)
//!     with offset basis 0xcbf29ce484222325 and prime 0x100000001b3.
//!
//! ## Reduction choice
//!
//! The hash is produced by an **exact sequential FNV-1a chain**, and the land
//! count by a **deterministic chunked strided reduction with an ordered fold**.
//!
//! A chunked "order-independent combine" is *not* available for FNV-1a. The
//! usual `h = h1 * P^len2 ^ h2` concatenation identity only holds when the
//! state update is GF(2)-linear, i.e. when `(a ^ b) * P == a*P ^ b*P`. Integer
//! multiplication does not distribute over XOR (`(3^5)*3 = 18` but
//! `3*3 ^ 5*3 = 6`), so the identity is false for FNV-1a and a combined value
//! would not be the FNV-1a hash of the plane at all. This was verified
//! numerically, and the CPU companion binary's earlier chunk-combine attempt
//! caught it: the combine disagreed with the sequential value. FNV-1a is
//! therefore computed by a single deterministic chain, which is exact by
//! construction, and parallelism is spent where it is exact:
//!
//!   * `chunk_digests_*` - one thread per contiguous chunk of the grid computes
//!     that chunk's own FNV-1a from state 0; the host recomputes the same
//!     per-chunk digests and compares every one. This proves the parallel grid
//!     read every byte of the plane identically to the CPU.
//!   * `count_land` - one thread per chunk counts land tiles in parallel; the
//!     per-chunk counts are folded by `sum_land` in fixed chunk order (an
//!     ordered, hence deterministic, sum).
//!
//! ## CLI
//!
//! ```text
//! ofcuda_map --map africa                 # one map, normal plane, all checks
//! ofcuda_map --map africa --size 4x       # the Compact game plane
//! ofcuda_map --map africa --size all      # normal + 4x + 16x
//! ofcuda_map --all [--size all]           # every map directory, TSV on stdout
//! ```
//!
//! Sweep output is one TSV line per (map, size) with the columns the independent
//! reference prints (`ofcuda_ref`, which links the real engine crate):
//!
//! ```text
//! name  size  WxH  land  manifest_land  terrain_hash  status
//! ```
//!
//! so `diff <(ofcuda_map --all) <(ofcuda_ref --all)` is the whole comparison.
//! Progress/summary lines go to stderr in sweep mode, keeping stdout diffable.
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use ofcuda_map::{MapSize, Summary, load_map_plane, synthetic_state};
use std::path::{Path, PathBuf};

// Duplicated here (not imported) so device-side constant folding never has to
// reach into another crate; `main` asserts they match the shared crate.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const IS_LAND: u8 = 0x80;

const THREADS: u32 = 256;
/// Tiles per chunk for the parallel per-chunk kernels.
const CHUNK: u32 = 4096;

const DEFAULT_REPO_ROOT: &str = "/opt/data/workspaces/skg/openfront-ai";

#[cuda_module]
mod kernels {
    use super::*;

    /// Exact sequential FNV-1a 64 of the terrain plane, walked in row-major
    /// grid order (`index = y*width + x`). One thread: FNV-1a has no valid
    /// parallel combine, see the module docs.
    #[kernel]
    #[launch_bounds(1)]
    #[launch_contract(domain = 1, block = (1, 1, 1))]
    pub fn fnv_terrain_chain(
        terrain: &[u8],
        width: u32,
        height: u32,
        mut out: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d();
        if idx.get() != 0 {
            return;
        }
        let n = width * height;
        let mut h: u64 = FNV_OFFSET;
        let mut y: u32 = 0;
        while y < height {
            let mut x: u32 = 0;
            while x < width {
                let b = terrain[(y * width + x) as usize];
                h = (h ^ (b as u64)).wrapping_mul(FNV_PRIME);
                x += 1;
            }
            y += 1;
        }
        let _ = n;
        if let Some(slot) = out.get_mut(idx) {
            *slot = h;
        }
    }

    /// Exact sequential FNV-1a 64 of a `u16` plane, each element fed as two
    /// little-endian bytes, walked in row-major grid order.
    #[kernel]
    #[launch_bounds(1)]
    #[launch_contract(domain = 1, block = (1, 1, 1))]
    pub fn fnv_state_chain(
        state: &[u16],
        width: u32,
        height: u32,
        mut out: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d();
        if idx.get() != 0 {
            return;
        }
        let mut h: u64 = FNV_OFFSET;
        let mut y: u32 = 0;
        while y < height {
            let mut x: u32 = 0;
            while x < width {
                let v = state[(y * width + x) as usize];
                h = (h ^ ((v & 0xff) as u64)).wrapping_mul(FNV_PRIME);
                h = (h ^ (v >> 8) as u64).wrapping_mul(FNV_PRIME);
                x += 1;
            }
            y += 1;
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = h;
        }
    }

    /// Parallel per-chunk FNV-1a 64 (from state 0) of the terrain plane. One
    /// thread per contiguous chunk of the flattened grid. The host recomputes
    /// the same digests and compares every one.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn chunk_digests_u8(
        data: &[u8],
        n: u32,
        chunk: u32,
        mut out: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d();
        let t = idx.get();
        let n = n as usize;
        let chunk = chunk as usize;
        let start = t * chunk;
        if start >= n {
            return;
        }
        let end = if start + chunk < n { start + chunk } else { n };
        let mut h: u64 = 0;
        let mut i = start;
        while i < end {
            h = (h ^ (data[i as usize] as u64)).wrapping_mul(FNV_PRIME);
            i += 1;
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = h;
        }
    }

    /// Parallel per-chunk FNV-1a 64 (from state 0) of a `u16` plane, two
    /// little-endian bytes per element. One thread per chunk.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn chunk_digests_u16(
        data: &[u16],
        n: u32,
        chunk: u32,
        mut out: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d();
        let t = idx.get();
        let n = n as usize;
        let chunk = chunk as usize;
        let start = t * chunk;
        if start >= n {
            return;
        }
        let end = if start + chunk < n { start + chunk } else { n };
        let mut h: u64 = 0;
        let mut i = start;
        while i < end {
            let v = data[i as usize];
            h = (h ^ ((v & 0xff) as u64)).wrapping_mul(FNV_PRIME);
            h = (h ^ (v >> 8) as u64).wrapping_mul(FNV_PRIME);
            i += 1;
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = h;
        }
    }

    /// Parallel per-chunk land-tile count (terrain bit 0x80). One thread per
    /// contiguous chunk of the flattened grid.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn count_land(
        terrain: &[u8],
        n: u32,
        chunk: u32,
        mut out: DisjointSlice<u32>,
    ) {
        let idx = thread::index_1d();
        let t = idx.get();
        let n = n as usize;
        let chunk = chunk as usize;
        let start = t * chunk;
        if start >= n {
            return;
        }
        let end = if start + chunk < n { start + chunk } else { n };
        let mut land: u32 = 0;
        let mut i = start;
        while i < end {
            if terrain[i as usize] & IS_LAND != 0 {
                land += 1;
            }
            i += 1;
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = land;
        }
    }

    /// Deterministic ordered fold of the per-chunk land counts (fixed chunk
    /// order, so the result does not depend on scheduling).
    #[kernel]
    #[launch_bounds(1)]
    #[launch_contract(domain = 1, block = (1, 1, 1))]
    pub fn sum_land(lands: &[u32], n: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if idx.get() != 0 {
            return;
        }
        let mut land: u32 = 0;
        let mut i: u32 = 0;
        while i < n {
            land += lands[i as usize];
            i += 1;
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = land;
        }
    }
}

/// Host-side reference for a chunked digest: `D(0, slice)`.
fn host_chunk_digest_u8(bytes: &[u8]) -> u64 {
    let mut h = 0u64;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(ofcuda_map::FNV_PRIME);
    }
    h
}

fn host_chunk_digest_u16(words: &[u16]) -> u64 {
    let mut h = 0u64;
    for &v in words {
        h = (h ^ (v & 0xff) as u64).wrapping_mul(ofcuda_map::FNV_PRIME);
        h = (h ^ (v >> 8) as u64).wrapping_mul(ofcuda_map::FNV_PRIME);
    }
    h
}

/// One TSV row: the reference-comparable fields plus a status word.
struct Row {
    land: u64,
    manifest_land: u32,
    terrain_hash: u64,
    status: String,
}

#[derive(Default)]
struct Args {
    repo_root: PathBuf,
    maps_root: Option<PathBuf>,
    map: Option<String>,
    all: bool,
    sizes: Vec<MapSize>,
    /// Also hash the zero state plane + the synthetic state plane (u16 path).
    state: bool,
    /// Positional legacy form: a direct map directory.
    dir: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        repo_root: PathBuf::from(DEFAULT_REPO_ROOT),
        ..Default::default()
    };
    let mut it = std::env::args().skip(1);
    let mut explicit_size = false;
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--repo-root" => a.repo_root = PathBuf::from(it.next().ok_or("--repo-root needs a value")?),
            "--maps-root" => a.maps_root = it.next().map(PathBuf::from),
            "--map" => a.map = it.next(),
            "--all" => a.all = true,
            "--size" => {
                let s = it.next().ok_or("--size needs a value")?;
                if s.eq_ignore_ascii_case("all") {
                    a.sizes = vec![MapSize::Normal, MapSize::X4, MapSize::X16];
                } else {
                    a.sizes.push(s.parse::<MapSize>()?);
                }
                explicit_size = true;
            }
            "--state" => a.state = true,
            "-h" | "--help" => {
                println!("usage: ofcuda_map [--map NAME | --all] [--size normal|4x|16x|all]... [--repo-root DIR] [--maps-root DIR] [--state] [MAP_DIR]");
                std::process::exit(0);
            }
            other if other.starts_with("--") => return Err(format!("unknown flag {other}")),
            other => a.dir = Some(PathBuf::from(other)),
        }
    }
    if a.sizes.is_empty() {
        // A sweep covers every plane variant by default; a single map defaults
        // to the normal plane only, which is what the tick runs on.
        a.sizes = if a.all {
            vec![MapSize::Normal, MapSize::X4, MapSize::X16]
        } else {
            vec![MapSize::Normal]
        };
    }
    let _ = explicit_size;
    if a.map.is_none() && !a.all && a.dir.is_none() {
        a.map = Some("World".into());
    }
    Ok(a)
}

fn maps_root_of(a: &Args) -> PathBuf {
    a.maps_root
        .clone()
        .unwrap_or_else(|| a.repo_root.join("openfront/resources/maps"))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert_eq!(FNV_OFFSET, ofcuda_map::FNV_OFFSET_BASIS, "offset basis drift");
    assert_eq!(FNV_PRIME, ofcuda_map::FNV_PRIME, "prime drift");
    assert_eq!(IS_LAND, ofcuda_map::IS_LAND_BIT, "land bit drift");

    let args = parse_args()?;
    let maps_root = maps_root_of(&args);

    // (name, dir) work list in the order the reference prints it.
    let mut work: Vec<(String, PathBuf)> = Vec::new();
    if let Some(dir) = &args.dir {
        let name = dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "?".into());
        work.push((name, dir.clone()));
    } else if args.all {
        for name in ofcuda_map::list_map_dirs_in(&maps_root)? {
            work.push((name.clone(), maps_root.join(&name)));
        }
    } else {
        let name = args.map.clone().unwrap();
        let key = name.to_lowercase().replace(' ', "");
        work.push((name, maps_root.join(key)));
    }

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: this package owns the embedded device bundle for `kernels`.
    let module = unsafe { kernels::load(&ctx)?};

    let mut pass = 0u64;
    let mut fail = 0u64;
    // Row printer, so single-map and sweep modes agree byte for byte.
    let emit = |name: &str, size: &str, dims: String, row: &Row| {
        println!(
            "{name}\t{size}\t{dims}\t{}\t{}\t{:#018x}\t{}",
            row.land, row.manifest_land, row.terrain_hash, row.status
        );
    };

    // One (map, size) unit of work: load, run every kernel, compare with the
    // host reference in this same process.
    let mut run = |name: &str, size: MapSize, dir: &Path| -> Result<bool, Box<dyn std::error::Error>> {
        let (meta, bytes) = match load_map_plane(dir, size) {
            Ok(v) => v,
            Err(e) => {
                emit(
                    name,
                    size.label(),
                    "ERROR".into(),
                    &Row {
                        land: 0,
                        manifest_land: 0,
                        terrain_hash: 0,
                        status: format!("ERR:{e}"),
                    },
                );
                return Ok(false);
            }
        };
        let tiles = meta.tiles();
        let chunks = tiles.div_ceil(CHUNK as usize) as u32;
        let threads = (chunks * THREADS) as usize;
        let n_tiles = tiles as u32;
        let one = LaunchConfig1D::new(1, 1, 0);
        let par = LaunchConfig1D::new(chunks, THREADS, 0);

        let d_terrain = DeviceBuffer::from_host(&stream, &bytes)?;

        // --- exact sequential FNV-1a chain, row-major, on the GPU ----------
        let mut t_out = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
        let p = module.prepare_fnv_terrain_chain(one)?;
        module.fnv_terrain_chain(&stream, &p, &d_terrain, meta.width, meta.height, &mut t_out)?;
        let terrain_hash_gpu = t_out.to_host_vec(&stream)?[0];

        // --- parallel per-chunk digests, every one compared on the host ----
        let mut g_tc = DeviceBuffer::<u64>::zeroed(&stream, threads)?;
        let p = module.prepare_chunk_digests_u8(par)?;
        module.chunk_digests_u8(&stream, &p, &d_terrain, n_tiles, CHUNK, &mut g_tc)?;
        let gpu_tc = g_tc.to_host_vec(&stream)?;
        let mut chunk_mismatch = 0u64;
        for c in 0..chunks as usize {
            let s = c * CHUNK as usize;
            let e = (s + CHUNK as usize).min(tiles);
            if gpu_tc[c] != host_chunk_digest_u8(&bytes[s..e]) {
                chunk_mismatch += 1;
            }
        }

        // --- parallel land count + ordered fold ----------------------------
        let mut g_land = DeviceBuffer::<u32>::zeroed(&stream, threads)?;
        let p = module.prepare_count_land(par)?;
        module.count_land(&stream, &p, &d_terrain, n_tiles, CHUNK, &mut g_land)?;
        let mut l_out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
        let p = module.prepare_sum_land(one)?;
        module.sum_land(&stream, &p, &g_land, chunks, &mut l_out)?;
        let land_gpu = l_out.to_host_vec(&stream)?[0] as u64;

        // --- u16 state-plane path (all-zero engine plane + synthetic) ------
        let mut state_detail = String::new();
        if args.state {
            let state_host: Vec<u16> = vec![0u16; tiles];
            let synth_host = synthetic_state(tiles);
            let d_state = DeviceBuffer::from_host(&stream, &state_host)?;
            let d_synth = DeviceBuffer::from_host(&stream, &synth_host)?;
            let mut s_out = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
            let p = module.prepare_fnv_state_chain(one)?;
            module.fnv_state_chain(&stream, &p, &d_state, meta.width, meta.height, &mut s_out)?;
            let state_hash_gpu = s_out.to_host_vec(&stream)?[0];
            let mut y_out = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
            module.fnv_state_chain(&stream, &p, &d_synth, meta.width, meta.height, &mut y_out)?;
            let synth_hash_gpu = y_out.to_host_vec(&stream)?[0];
            let mut g_sc = DeviceBuffer::<u64>::zeroed(&stream, threads)?;
            let p = module.prepare_chunk_digests_u16(par)?;
            module.chunk_digests_u16(&stream, &p, &d_synth, n_tiles, CHUNK, &mut g_sc)?;
            let gpu_sc = g_sc.to_host_vec(&stream)?;
            for c in 0..chunks as usize {
                let s = c * CHUNK as usize;
                let e = (s + CHUNK as usize).min(tiles);
                if gpu_sc[c] != host_chunk_digest_u16(&synth_host[s..e]) {
                    chunk_mismatch += 1;
                }
            }
            let state_cpu = ofcuda_map::state_hash(&state_host);
            let synth_cpu = ofcuda_map::state_hash(&synth_host);
            state_detail = format!(
                " state={:#018x} state_cpu={:#018x} synth={:#018x} synth_cpu={:#018x}",
                state_hash_gpu, state_cpu, synth_hash_gpu, synth_cpu
            );
            if state_hash_gpu != state_cpu || synth_hash_gpu != synth_cpu {
                chunk_mismatch += 1;
            }
        }

        // Same numbers from the host, same file, same process.
        let cpu = Summary {
            terrain_hash: ofcuda_map::terrain_hash(&bytes),
            state_hash: 0,
            land_tiles: ofcuda_map::land_tiles(&bytes),
        };

        let ok = terrain_hash_gpu == cpu.terrain_hash
            && land_gpu == cpu.land_tiles
            && chunk_mismatch == 0;
        let status = if ok {
            "OK".to_string()
        } else {
            format!(
                "DIFF cpu_hash={:#018x} cpu_land={} chunk_mismatch={}",
                cpu.terrain_hash, cpu.land_tiles, chunk_mismatch
            )
        };
        eprintln!(
            "# {name} {} {}x{} tiles={tiles} chunks={chunks} hash={:#018x} land={land_gpu} manifest_land={} {}{}",
            size.label(),
            meta.width,
            meta.height,
            terrain_hash_gpu,
            meta.num_land_tiles,
            status,
            state_detail
        );
        emit(
            name,
            size.label(),
            format!("{}x{}", meta.width, meta.height),
            &Row {
                land: land_gpu,
                manifest_land: meta.num_land_tiles,
                terrain_hash: terrain_hash_gpu,
                status,
            },
        );
        Ok(ok)
    };

    for (name, dir) in &work {
        for size in &args.sizes {
            if run(name, *size, dir)? {
                pass += 1;
            } else {
                fail += 1;
            }
        }
    }

    eprintln!("sweep_pass {pass} sweep_fail {fail}");
    if fail > 0 {
        eprintln!("FAILED: {fail} (map, size) units did not match the host reference");
        std::process::exit(1);
    }
    Ok(())
}
