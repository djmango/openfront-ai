//! `ofcuda_hash` - the verification yardstick for the GPU port: a CUDA
//! implementation of the engine's per-tick state hash plus the state-plane
//! serializer, able to consume a GPU-resident state buffer.
//!
//! Input: a dump directory written by `ofcuda_hash/oracle`, which runs the real
//! engine and writes
//!   * `terrain.bin`  - `width*height` raw terrain bytes (`game.terrainByte`)
//!   * `planes.bin`   - one `width*height` `u16` state plane per tick, LE
//!   * `post_reset.bin` (parity mode) - the post-reset state plane
//!   * `expected.jsonl` - the engine's own FNV-1a 64 hash of each of those
//!                        planes, plus per-tick `gameHashBits`
//!
//! The GPU never sees the expected hashes: it reads the *planes* and produces
//! hashes, which are then compared. Both the GPU path and the CPU reference in
//! `src/lib.rs` consume the same bytes through the same serializer.
//!
//! ## What this harness proves
//!
//! 1. **Serializer exactness.** `serialize_state_le` lays the `u16` plane out on
//!    the device as two LE bytes per word, byte-identically to
//!    `serialize_u16_le` on the host. A `u16` fed as two LE bytes is not the
//!    same byte stream as an `f32`, so this is a real assertion, not a
//!    formality.
//! 2. **Per-tick agreement.** For every tick in the dump the GPU hash equals the
//!    engine's hash, printed side by side with the running match count and the
//!    first disagreement.
//! 3. **Chunk-count invariance.** The same buffer is hashed with an ordered
//!    fold parameterized by chunk size; the value must not move as the chunk
//!    count varies.
//! 4. **The wrong design is detectably wrong.** FNV-1a has no valid chunk
//!    combine; the naive `h*P^len ^ digest` reduction is computed too and shown
//!    to disagree, so "it looked plausible" cannot pass for a result here.
//! 5. **The published constants.** Terrain `0xebffa87c2568cc58` and post-reset
//!    state `0x6334dfb980453d25` on Pangaea are recomputed from planes on the
//!    device and compared against the literals.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use ofcuda_hash::{
    Dump, FNV_OFFSET_BASIS, FNV_PRIME, fnv1a_bytes_chunked_ordered, hex64, load_dump, naive_combine,
    parse_hex64, serialize_u16_le,
};
use std::path::PathBuf;

/// The two published constants on Pangaea, reproduced from scratch below.
const EXPECTED_TERRAIN_HASH: u64 = 0xebff_a87c_2568_cc58;
const EXPECTED_POST_RESET_STATE_HASH: u64 = 0x6334_dfb9_8045_3d25;

/// Threads per block for the parallel (per-word / per-chunk) kernels.
const THREADS: u32 = 256;
/// Bytes per chunk for the parallel per-chunk digest comparison.
const DIGEST_CHUNK: u32 = 65536;
/// Chunk sizes the ordered fold is exercised at. One identical value is
/// required from all of them: chunk count must not move the hash.
const CHUNKS: [u32; 11] = [
    1, 2, 3, 5, 7, 64, 1024, 4096, 65536, 250_000, 1_000_000,
];

#[cuda_module]
mod kernels {
    use super::*;

    /// The state-plane serializer, on the device.
    ///
    /// **One thread per output byte** (`DisjointSlice` only permits a thread to
    /// write at its own index): byte `k` belongs to word `k/2`, and is the low
    /// byte for even `k` and the high byte for odd `k`. That is exactly the
    /// little-endian layout the engine's `tileStateBuffer()` has on the wire
    /// (`bridge/common.ts` wraps the same `Uint16Array.buffer`). No reordering,
    /// no padding, no float narrowing.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn serialize_state_le(state: &[u16], mut bytes: DisjointSlice<u8>) {
        let idx = thread::index_1d();
        let k = idx.get();
        if k >= 2 * state.len() {
            return;
        }
        let w = state[k / 2];
        let b = if k % 2 == 0 {
            (w & 0xff) as u8
        } else {
            (w >> 8) as u8
        };
        if let Some(slot) = bytes.get_mut(idx) {
            *slot = b;
        }
    }

    /// Exact sequential FNV-1a 64 over `n` bytes, walked in order. One thread:
    /// FNV-1a has no valid parallel combine (see the crate docs), so the *chain*
    /// is inherently serial; parallelism lives in the serializer and in the
    /// per-chunk digests below.
    #[kernel]
    #[launch_bounds(1)]
    #[launch_contract(domain = 1, block = (1, 1, 1))]
    pub fn fnv_chain_bytes(data: &[u8], n: u32, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if idx.get() != 0 {
            return;
        }
        let n = n as usize;
        let mut h: u64 = FNV_OFFSET_BASIS;
        let mut i: usize = 0;
        while i < n {
            h = (h ^ (data[i] as u64)).wrapping_mul(FNV_PRIME);
            i += 1;
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = h;
        }
    }

    /// The same ordered chain, but walked as consecutive chunks of `chunk`
    /// bytes with the running state carried across chunk boundaries. `chunk` is
    /// a *parameter*: the harness varies it and requires the result to be
    /// invariant. This is the "design for order" kernel - it never combines
    /// per-chunk digests, it only ever carries one state forward.
    #[kernel]
    #[launch_bounds(1)]
    #[launch_contract(domain = 1, block = (1, 1, 1))]
    pub fn fnv_chain_bytes_chunked(data: &[u8], n: u32, chunk: u32, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if idx.get() != 0 {
            return;
        }
        let n = n as usize;
        let chunk = chunk as usize;
        let mut h: u64 = FNV_OFFSET_BASIS;
        let mut start: usize = 0;
        while start < n {
            let mut end = start + chunk;
            if end > n {
                end = n;
            }
            let mut i = start;
            while i < end {
                h = (h ^ (data[i] as u64)).wrapping_mul(FNV_PRIME);
                i += 1;
            }
            start = end;
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = h;
        }
    }

    /// Direct `u16`-plane chain, no separate serializer pass: each word fed as
    /// two LE bytes. Must equal `fnv_chain_bytes` over `serialize_state_le`'s
    /// output - two independent device paths to the same contract.
    #[kernel]
    #[launch_bounds(1)]
    #[launch_contract(domain = 1, block = (1, 1, 1))]
    pub fn fnv_chain_u16(state: &[u16], n: u32, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if idx.get() != 0 {
            return;
        }
        let n = n as usize;
        let mut h: u64 = FNV_OFFSET_BASIS;
        let mut i: usize = 0;
        while i < n {
            let w = state[i];
            h = (h ^ ((w & 0xff) as u64)).wrapping_mul(FNV_PRIME);
            h = (h ^ ((w >> 8) as u64)).wrapping_mul(FNV_PRIME);
            i += 1;
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = h;
        }
    }

    /// Parallel per-chunk FNV-1a 64, each chunk seeded from the offset basis
    /// (`D(0, slice)`). One thread per contiguous chunk. These digest values are
    /// *not* combined into a hash - they exist so the host can recompute the same
    /// digests and prove the device read every byte of every chunk.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn chunk_digests_bytes(
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
        let mut end = start + chunk;
        if end > n {
            end = n;
        }
        let mut h: u64 = 0;
        let mut i = start;
        while i < end {
            h = (h ^ (data[i] as u64)).wrapping_mul(FNV_PRIME);
            i += 1;
        }
        if let Some(slot) = out.get_mut(idx) {
            *slot = h;
        }
    }
}

/// Launch a single-thread kernel that writes one u64 and read it back.
macro_rules! one_u64 {
    ($stream:expr, $out:expr) => {
        $out.to_host_vec(&$stream)?[0]
    };
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dump_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .ok_or(
            "usage: ofcuda_hash <dump-dir> [--max-ticks N] [--dump-device-planes <out-dir>]",
        )?;
    let mut max_ticks = usize::MAX;
    // Where (if anywhere) to dump the *device-written* per-tick plane bytes.
    // Nothing here is copied back into this file from the reference: the bytes
    // written are the ones `serialize_state_le` produced on the device, read
    // back verbatim with `to_host_vec`.
    let mut device_dump_dir: Option<PathBuf> = None;
    let argv: Vec<String> = std::env::args().skip(2).collect();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--max-ticks" => {
                max_ticks = argv[i + 1].parse()?;
                i += 2;
            }
            "--dump-device-planes" => {
                device_dump_dir = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            other => return Err(format!("unknown arg {other}").into()),
        }
    }

    let dump: Dump = load_dump(&dump_dir)?;
    let n = dump.header.tiles;
    let n_bytes = 2 * n;
    let n_ticks = dump.ticks.len().min(max_ticks);

    println!("# ofcuda_hash - device-side per-tick state hash yardstick");
    println!("# dump dir        {}", dump.dir.display());
    println!("# mode            {}", dump.header.mode);
    if let Some(r) = &dump.header.record {
        println!("# record          {r}");
    }
    if let Some(g) = &dump.header.game_id {
        println!("# game_id         {g}");
    }
    println!(
        "# grid            {}x{} tiles {} ({} plane bytes)",
        dump.header.width, dump.header.height, n, n_bytes
    );
    println!(
        "# dump ticks      {} (checking {})",
        dump.ticks.len(),
        n_ticks
    );
    println!(
        "# fnv             1a64 offset={:#018x} prime={:#018x}",
        FNV_OFFSET_BASIS, FNV_PRIME
    );

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: this package owns the embedded device bundle for `kernels`.
    let module = unsafe { kernels::load(&ctx)? };

    let one = LaunchConfig1D::new(1, 1, 0);
    let serializer_grid = (n_bytes as u32).div_ceil(THREADS);
    let ser = LaunchConfig1D::new(serializer_grid, THREADS, 0);
    let n_chunks = (n_bytes as u32).div_ceil(DIGEST_CHUNK);
    let digest_grid = n_chunks.div_ceil(THREADS);
    let par = LaunchConfig1D::new(digest_grid, THREADS, 0);
    let digest_threads = (digest_grid * THREADS) as usize;

    // Device buffers, allocated once.
    let mut d_state = DeviceBuffer::<u16>::zeroed(&stream, n)?;
    let mut d_bytes = DeviceBuffer::<u8>::zeroed(&stream, n_bytes)?;
    let d_terrain = DeviceBuffer::from_host(&stream, &dump.terrain)?;
    let mut d_digests = DeviceBuffer::<u64>::zeroed(&stream, digest_threads)?;
    let mut d_out = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
    let mut d_out2 = DeviceBuffer::<u64>::zeroed(&stream, 1)?;

    let p_ser = module.prepare_serialize_state_le(ser)?;
    let p_chain = module.prepare_fnv_chain_bytes(one)?;
    let p_chain_chunked = module.prepare_fnv_chain_bytes_chunked(one)?;
    let p_u16 = module.prepare_fnv_chain_u16(one)?;
    let p_dig = module.prepare_chunk_digests_bytes(par)?;

    let mut failures: u64 = 0;

    // =====================================================================
    // 1. TERRAIN HASH - raw bytes, w*h, `game.terrainByte(ref)` order.
    // =====================================================================
    println!();
    println!("== terrain hash ({} raw bytes) ==", dump.terrain.len());
    module.fnv_chain_bytes(&stream, &p_chain, &d_terrain, n as u32, &mut d_out)?;
    let terrain_gpu_serial = one_u64!(stream, d_out);

    // Chunk-count invariance on the terrain plane, at several chunk sizes.
    let mut chunk_values: Vec<(u32, u64)> = Vec::new();
    for chunk in CHUNKS {
        module.fnv_chain_bytes_chunked(
            &stream,
            &p_chain_chunked,
            &d_terrain,
            n as u32,
            chunk,
            &mut d_out2,
        )?;
        chunk_values.push((chunk, one_u64!(stream, d_out2)));
    }
    let chunk_invariant = chunk_values.iter().all(|(_, v)| *v == terrain_gpu_serial);
    println!("terrain_chunk_count_invariance {chunk_invariant}");
    let terrain_cpu = ofcuda_hash::terrain_hash(&dump.terrain);
    println!("terrain_hash_gpu_serial      {}", hex64(terrain_gpu_serial));
    println!("terrain_hash_cpu_reference   {}", hex64(terrain_cpu));
    println!(
        "terrain_hash_engine_expected {}",
        hex64(dump.header.terrain_hash)
    );
    println!(
        "terrain_hash_published_const {}",
        hex64(EXPECTED_TERRAIN_HASH)
    );
    println!(
        "terrain_matches_engine       {}",
        terrain_gpu_serial == dump.header.terrain_hash
    );
    println!(
        "terrain_matches_published    {}",
        terrain_gpu_serial == EXPECTED_TERRAIN_HASH
    );
    if terrain_gpu_serial != dump.header.terrain_hash {
        failures += 1;
    }
    if terrain_gpu_serial != EXPECTED_TERRAIN_HASH {
        failures += 1;
    }

    // =====================================================================
    // 2. POST-RESET STATE HASH (parity mode only) - the second constant.
    // =====================================================================
    if let Some(post) = &dump.post_reset {
        println!();
        println!("== post-reset state hash ({} u16 words) ==", post.len());
        d_state.copy_from_host(&stream, post)?;
        module.serialize_state_le(&stream, &p_ser, &d_state, &mut d_bytes)?;
        module.fnv_chain_bytes(&stream, &p_chain, &d_bytes, n_bytes as u32, &mut d_out)?;
        let via_bytes = one_u64!(stream, d_out);
        module.fnv_chain_u16(&stream, &p_u16, &d_state, n as u32, &mut d_out2)?;
        let via_u16 = one_u64!(stream, d_out2);
        let cpu = ofcuda_hash::state_hash(post);
        // Serializer equality: device bytes vs host bytes.
        let gpu_bytes = d_bytes.to_host_vec(&stream)?;
        let cpu_bytes = serialize_u16_le(post);
        let ser_equal = gpu_bytes == cpu_bytes;
        let nz = ofcuda_hash::nonzero_bytes(&gpu_bytes);
        println!("post_reset_plane_nonzero_bytes {nz}/{}", gpu_bytes.len());
        if nz == 0 {
            println!(
                "post_reset_plane_is_all_zero true  (hash is the closed form h0 * P^n mod 2^64; \
                 this check is therefore a length check, not evidence about device reads - see the \
                 non-degenerate probes in the next section)"
            );
        }
        println!("post_reset_state_hash_gpu_bytes {}", hex64(via_bytes));
        println!("post_reset_state_hash_gpu_u16   {}", hex64(via_u16));
        println!("post_reset_state_hash_cpu       {}", hex64(cpu));
        println!(
            "post_reset_state_engine_expected {}",
            hex64(dump.header.post_reset_state_hash.unwrap_or(0))
        );
        println!(
            "post_reset_state_published_const {}",
            hex64(EXPECTED_POST_RESET_STATE_HASH)
        );
        println!("serializer_bytes_identical      {ser_equal}");
        println!(
            "post_reset_matches_engine       {}",
            via_bytes == dump.header.post_reset_state_hash.unwrap_or(0)
        );
        println!(
            "post_reset_matches_published    {}",
            via_bytes == EXPECTED_POST_RESET_STATE_HASH
        );
        if !ser_equal {
            failures += 1;
        }
        if via_bytes != EXPECTED_POST_RESET_STATE_HASH || via_u16 != via_bytes || cpu != via_bytes {
            failures += 1;
        }
    }

    // =====================================================================
    // 3. CHUNK-COUNT INVARIANCE + PER-CHUNK DIGESTS, on non-degenerate probes.
    //
    // A probe must actually exercise the u16 -> 2xLE-byte path. The post-reset
    // state plane does not: it is 2,000,000 zero bytes and the all-zero chain
    // is just h0 * P^n, so every design - and a buffer that was never read -
    // hashes identically. Each probe prints its nonzero-byte density so a
    // degenerate one is visible rather than assumed.
    // =====================================================================
    println!();
    println!("== chunk-count invariance and per-chunk digests (non-degenerate probes) ==");

    let synthetic: Vec<u16> = ofcuda_hash::synthetic_state(n);

    // Densest real state plane in this dump.
    let mut densest = (0usize, 0usize);
    for i in 0..dump.ticks.len() {
        let bytes = serialize_u16_le(&dump.plane(i));
        let nz = ofcuda_hash::nonzero_bytes(&bytes);
        if nz > densest.1 {
            densest = (i, nz);
        }
    }
    let densest_plane = dump.plane(densest.0);
    let densest_label = format!(
        "real plane tick {} (densest of {})",
        dump.ticks[densest.0].tick,
        dump.ticks.len()
    );

    macro_rules! probe {
        ($label:expr, $words:expr) => {{
            d_state.copy_from_host(&stream, $words)?;
            module.serialize_state_le(&stream, &p_ser, &d_state, &mut d_bytes)?;
            let gpu_bytes = d_bytes.to_host_vec(&stream)?;
            let cpu_bytes = serialize_u16_le($words);
            let nz = ofcuda_hash::nonzero_bytes(&cpu_bytes);
            let ser_equal = gpu_bytes == cpu_bytes;
            module.fnv_chain_bytes(&stream, &p_chain, &d_bytes, n_bytes as u32, &mut d_out)?;
            let charged = one_u64!(stream, d_out);
            let mut inv_rows = String::new();
            let mut all_invariant = true;
            for chunk in CHUNKS {
                module.fnv_chain_bytes_chunked(
                    &stream,
                    &p_chain_chunked,
                    &d_bytes,
                    n_bytes as u32,
                    chunk,
                    &mut d_out2,
                )?;
                let v = one_u64!(stream, d_out2);
                let cpu_v =
                    fnv1a_bytes_chunked_ordered(FNV_OFFSET_BASIS, &gpu_bytes, chunk as usize);
                let chunks = (n_bytes as u32).div_ceil(chunk);
                let ok = v == charged && cpu_v == charged;
                all_invariant &= ok;
                inv_rows.push_str(&format!(
                    "  chunk_bytes={chunk:<8} chunks={chunks:<8} gpu={} cpu={} equal={ok}\n",
                    hex64(v),
                    hex64(cpu_v)
                ));
            }
            // The WRONG design, for contrast: per-chunk digests combined with the
            // concatenation identity. It must disagree (that identity needs
            // multiplication to distribute over XOR, which it does not).
            let mut naive_disagreements = 0;
            let mut naive_last = 0u64;
            for chunk in CHUNKS {
                let v = naive_combine(FNV_OFFSET_BASIS, &gpu_bytes, chunk as usize);
                if v != charged {
                    naive_disagreements += 1;
                }
                naive_last = v;
            }
            // Parallel per-chunk digests from the device, each against the host.
            module.chunk_digests_bytes(
                &stream,
                &p_dig,
                &d_bytes,
                n_bytes as u32,
                DIGEST_CHUNK,
                &mut d_digests,
            )?;
            let gpu_digests = d_digests.to_host_vec(&stream)?;
            let mut digest_mismatch = 0u64;
            for c in 0..n_chunks as usize {
                let s = c * DIGEST_CHUNK as usize;
                let e = (s + DIGEST_CHUNK as usize).min(n_bytes);
                if gpu_digests[c] != ofcuda_hash::fnv1a_bytes(0, &gpu_bytes[s..e]) {
                    digest_mismatch += 1;
                }
            }
            println!();
            println!("probe {label}", label = $label);
            println!("  nonzero_bytes {nz}/{n_bytes}");
            println!("  serializer_bytes_identical {ser_equal}");
            println!(
                "  serial_chain_gpu {}  cpu_reference {}",
                hex64(charged),
                hex64(ofcuda_hash::fnv1a_bytes(FNV_OFFSET_BASIS, &gpu_bytes))
            );
            print!("{inv_rows}");
            println!("  chunk_count_invariance_gpu {all_invariant}");
            println!(
                "  naive_combine_wrong_by_design disagreeing_chunk_sizes={naive_disagreements}/{} last={} (serial={})",
                CHUNKS.len(),
                hex64(naive_last),
                hex64(charged)
            );
            println!(
                "  per_chunk_digests chunks={n_chunks} chunk_bytes={DIGEST_CHUNK} mismatches={digest_mismatch}"
            );
            if !all_invariant || !ser_equal || digest_mismatch != 0 {
                failures += 1;
            }
        }};
    }
    probe!("synthetic (both bytes of every other word set)", &synthetic);
    probe!(&densest_label, &densest_plane);

    // =====================================================================
    // 4. PER-TICK STATE HASH - GPU vs the engine, every tick.
    // =====================================================================
    println!();
    println!("== per-tick state hash: GPU vs engine ({} ticks) ==", n_ticks);
    if let Some(dir) = &device_dump_dir {
        println!(
            "# dumping device-written planes to {}/device_planes.bin ({} bytes/tick, LE u16)",
            dir.display(),
            n_bytes
        );
    }
    println!(
        "  {:<8} {:<20} {:<20} {:<20} {:<6} {}",
        "tick", "gpu_serialized", "gpu_direct_u16", "engine_expected", "match", "gameHashBits"
    );

    // Device-plane dump: the file is written from `d_bytes` (device-written),
    // never from the reference plane, and the per-tick tile diff is counted by
    // decoding those device bytes back to u16 and comparing word for word
    // against the reference plane for the same tick.
    let mut device_dump: Option<std::fs::File> = match &device_dump_dir {
        Some(dir) => {
            std::fs::create_dir_all(dir)?;
            Some(std::fs::File::create(dir.join("device_planes.bin"))?)
        }
        None => None,
    };
    let mut dump_rows: Vec<(u32, usize, usize, bool)> = Vec::new();

    let mut matches = 0usize;
    let mut first_disagreement: Option<(u32, u64, u64)> = None;
    for idx in 0..n_ticks {
        let tl = &dump.ticks[idx];
        let plane = dump.plane(idx);
        d_state.copy_from_host(&stream, &plane)?;
        module.serialize_state_le(&stream, &p_ser, &d_state, &mut d_bytes)?;
        // Read the device-written serializer output back (this is the only
        // readback the dump needs, and it is the device's own bytes).
        let device_bytes = d_bytes.to_host_vec(&stream)?;
        module.fnv_chain_bytes(&stream, &p_chain, &d_bytes, n_bytes as u32, &mut d_out)?;
        let h_bytes = one_u64!(stream, d_out);
        module.fnv_chain_u16(&stream, &p_u16, &d_state, n as u32, &mut d_out2)?;
        let h_u16 = one_u64!(stream, d_out2);
        let ok = h_bytes == tl.state_hash && h_u16 == tl.state_hash;
        if ok {
            matches += 1;
        } else if first_disagreement.is_none() {
            first_disagreement = Some((tl.tick, h_bytes, tl.state_hash));
        }
        // Cross-check the CPU reference on the same plane, same code path.
        let cpu = ofcuda_hash::state_hash(&plane);
        let cpu_ok = cpu == tl.state_hash;
        if let Some(f) = device_dump.as_mut() {
            use std::io::{Seek, SeekFrom, Write};
            f.seek(SeekFrom::Start((idx * n_bytes) as u64))?;
            f.write_all(&device_bytes)?;
            let mut diff_tiles = 0usize;
            for k in 0..n {
                if u16::from_le_bytes([device_bytes[2 * k], device_bytes[2 * k + 1]]) != plane[k] {
                    diff_tiles += 1;
                }
            }
            let ref_bytes = serialize_u16_le(&plane);
            let diff_bytes = device_bytes
                .iter()
                .zip(ref_bytes.iter())
                .filter(|(a, b)| a != b)
                .count();
            let row_ok = diff_tiles == 0 && diff_bytes == 0;
            println!(
                "  device_planes tick={:<8} diff_tiles={:<8} diff_bytes={:<8} identical={}",
                tl.tick, diff_tiles, diff_bytes, row_ok
            );
            dump_rows.push((tl.tick, diff_tiles, diff_bytes, row_ok));
        }
        println!(
            "  {:<8} {:<20} {:<20} {:<20} {:<6} {}",
            tl.tick,
            hex64(h_bytes),
            hex64(h_u16),
            hex64(tl.state_hash),
            if ok && cpu_ok { "ok" } else { "MISMATCH" },
            tl.game_hash_bits.clone().unwrap_or_default()
        );
        if !cpu_ok {
            failures += 1;
        }
    }
    println!("per_tick_checked  {n_ticks}");
    println!("per_tick_matches  {matches}");
    println!("per_tick_mismatches {}", n_ticks - matches);
    if let Some(f) = device_dump.as_mut() {
        use std::io::Write;
        f.flush()?;
        f.sync_all()?;
    }
    if let Some(dir) = &device_dump_dir {
        let written = dump_rows.len();
        let bad_rows = dump_rows.iter().filter(|r| !r.3).count();
        let total_diff_tiles: usize = dump_rows.iter().map(|r| r.1).sum();
        let total_diff_bytes: usize = dump_rows.iter().map(|r| r.2).sum();
        println!(
            "device_planes_dump {} ({} ticks, {} bytes/plane)",
            dir.join("device_planes.bin").display(),
            written,
            n_bytes
        );
        println!("device_planes_ticks_dumped {written}");
        println!("device_planes_rows_not_identical {bad_rows}");
        println!("device_planes_total_diff_tiles {total_diff_tiles}");
        println!("device_planes_total_diff_bytes {total_diff_bytes}");
        println!("device_planes_all_identical {}", bad_rows == 0 && written == n_ticks);
        if bad_rows != 0 || written != n_ticks {
            failures += 1;
        }
    }
    match first_disagreement {
        None => println!("first_disagreement none"),
        Some((t, gpu, exp)) => {
            println!(
                "first_disagreement tick={t} gpu={} expected={}",
                hex64(gpu),
                hex64(exp)
            );
            failures += 1;
        }
    }
    if matches != n_ticks {
        failures += 1;
    }

    // =====================================================================
    // 5. Verdict.
    // =====================================================================
    println!();
    println!("chunk_counts_tested {}", CHUNKS.len());
    println!("verdict {}", if failures == 0 { "PASS" } else { "FAIL" });
    if failures != 0 {
        eprintln!("FAILED: {failures} failing check group(s)");
        std::process::exit(1);
    }
    // Keep `parse_hex64` honest: it is the reader the whole harness trusts.
    debug_assert_eq!(
        parse_hex64(&hex64(EXPECTED_TERRAIN_HASH)),
        Some(EXPECTED_TERRAIN_HASH)
    );
    Ok(())
}