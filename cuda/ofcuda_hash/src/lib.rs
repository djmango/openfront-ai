//! Host-side shared code for the `ofcuda_hash` verification yardstick.
//!
//! Nothing in this file touches CUDA. The GPU binary (`src/main.rs`) and the
//! CPU companion (`src/bin/cpu.rs`) both link this crate, so the serializer,
//! the hash definition and the dump reader are *literally the same code* on
//! both sides. Only the computation differs (a cuda-oxide kernel vs. a plain
//! sequential Rust loop), which is what makes a mismatch meaningful.
//!
//! ## The contract being reproduced
//!
//! `rust/engine/src/bin/parity_trace.rs` (and its published traces) define the
//! per-tick state hash that every ported mechanic is validated against:
//!
//! * **FNV-1a 64**, offset basis `0xcbf29ce484222325`, prime `0x100000001b3`.
//! * The **state plane** is `width*height` `u16`; each word is fed to the hash
//!   as **two little-endian bytes** (`bridge/common.ts` ships the same plane as
//!   `Buffer` over the `Uint16Array.buffer`, i.e. LE on x86). A `u16` stored as
//!   two LE bytes is *not* the same byte stream as an `f32` holding the same
//!   number, so the serializer has to be exact, not "close enough".
//! * The **terrain plane** is `width*height` raw bytes (`game.terrainByte(ref)`
//!   for `ref in 0..w*h`), hashed as bytes - *not* the whole terrain struct.
//! * Row-major order, `index = y*width + x`, no reordering anywhere.
//!
//! ## Why FNV-1a cannot be reduced across chunks
//!
//! The tempting parallel design - hash each chunk from the offset basis and
//! combine the digests - is **wrong**. The only combine identity FNV-1a has is
//! the concatenation identity `h(a||b) = h(a) * P^len(b) ^ h(b)`, and it holds
//! only if the state update is GF(2)-linear, i.e. if
//! `(x ^ y) * P == x*P ^ y*P`. Integer multiplication does not distribute over
//! XOR (`(3 ^ 5) * 3 = 18`, `3*3 ^ 5*3 = 6`), so the identity is false and a
//! combined digest is not the FNV-1a hash of anything. `naive_combine` below
//! implements exactly that false identity so the harness can *demonstrate* it
//! disagrees - a wrong design that fails loudly rather than plausibly.
//!
//! The correct design keeps one strict in-order chain and spends parallelism
//! where it is exact: the byte serializer, and per-chunk digests that are
//! compared against a host recomputation of the same per-chunk digests. The
//! ordered fold is verified by hashing one buffer with a parameterized chunk
//! size and showing the value is identical for every chunk count.

use std::path::{Path, PathBuf};

/// FNV-1a 64 offset basis.
pub const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64 prime.
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// FNV-1a update step.
#[inline]
pub fn fnv_step(h: u64, b: u8) -> u64 {
    (h ^ b as u64).wrapping_mul(FNV_PRIME)
}

/// Terrain hash: raw bytes, row-major.
pub fn terrain_hash(terrain: &[u8]) -> u64 {
    fnv1a_bytes(FNV_OFFSET_BASIS, terrain)
}

/// State-plane hash: each `u16` fed as two little-endian bytes, row-major.
pub fn state_hash(state: &[u16]) -> u64 {
    fnv1a_u16_le(FNV_OFFSET_BASIS, state)
}

/// Sequential FNV-1a 64 over a byte slice, starting from `h0`.
pub fn fnv1a_bytes(h0: u64, bytes: &[u8]) -> u64 {
    let mut h = h0;
    for &b in bytes {
        h = fnv_step(h, b);
    }
    h
}

/// Sequential FNV-1a 64 over a `u16` plane, each word as two LE bytes.
pub fn fnv1a_u16_le(h0: u64, words: &[u16]) -> u64 {
    let mut h = h0;
    for &w in words {
        h = fnv_step(h, (w & 0xff) as u8);
        h = fnv_step(h, (w >> 8) as u8);
    }
    h
}

/// The state-plane serializer, CPU reference.
///
/// Lays out `words` exactly the way the engine's state buffer is laid out on
/// the wire: word `i` occupies bytes `2i` (low) and `2i+1` (high). This is the
/// byte string FNV-1a is defined over.
pub fn serialize_u16_le(words: &[u16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 2);
    for &w in words {
        out.push((w & 0xff) as u8);
        out.push((w >> 8) as u8);
    }
    out
}

/// Chunked **ordered** fold, CPU reference: walk `bytes` as consecutive chunks
/// of `chunk` bytes, carrying the running FNV state from one chunk into the
/// next. The value must not depend on `chunk` - that is the invariance the GPU
/// kernel has to reproduce.
pub fn fnv1a_bytes_chunked_ordered(h0: u64, bytes: &[u8], chunk: usize) -> u64 {
    assert!(chunk > 0, "chunk must be positive");
    let mut h = h0;
    let mut start = 0usize;
    while start < bytes.len() {
        let end = (start + chunk).min(bytes.len());
        for &b in &bytes[start..end] {
            h = fnv_step(h, b);
        }
        start = end;
    }
    h
}

/// The **wrong** design, implemented so it can be shown to be wrong: hash each
/// chunk from the offset basis and combine with the concatenation identity
/// `h = h * P^len(chunk) ^ digest`. Returns the combined value.
pub fn naive_combine(h0: u64, bytes: &[u8], chunk: usize) -> u64 {
    assert!(chunk > 0, "chunk must be positive");
    let mut h = h0;
    let mut start = 0usize;
    while start < bytes.len() {
        let end = (start + chunk).min(bytes.len());
        let d = fnv1a_bytes(0, &bytes[start..end]);
        let mut p = 1u64;
        for _ in 0..(end - start) {
            p = p.wrapping_mul(FNV_PRIME);
        }
        h = h.wrapping_mul(p) ^ d;
        start = end;
    }
    h
}

/// Deterministic, non-degenerate state plane used as a probe.
///
/// The post-reset state plane is **2,000,000 zero bytes**, so hashing it proves
/// nothing about device reads: the all-zero chain is just `h0 * P^n`, and a
/// buffer that was never read hashes identically. This pattern sets both bytes
/// of every other word (owner nibbles in the low byte, fallout/magnitude bits in
/// the high byte) so the u16 -> 2xLE-byte path is genuinely exercised.
pub fn synthetic_state(tiles: usize) -> Vec<u16> {
    (0..tiles)
        .map(|t| {
            let owner = ((t % 977) as u16) & 0x0fff;
            let hi = if t % 2 == 0 {
                (((t / 977) % 15) as u16) << 12
            } else {
                0x2000u16
            };
            owner | hi
        })
        .collect()
}

/// Number of nonzero bytes in a byte slice (probe degeneracy check).
pub fn nonzero_bytes(bytes: &[u8]) -> usize {
    bytes.iter().filter(|&&b| b != 0).count()
}

/// Print helper, matching the engine's `0x` + 16 lowercase hex digits.
pub fn hex64(h: u64) -> String {
    format!("0x{h:016x}")
}

/// Parse a `0x...` hex string back to a u64 (the dump stores hashes as text).
pub fn parse_hex64(s: &str) -> Option<u64> {
    u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok()
}

// ---------------------------------------------------------------------------
// Dump reader
// ---------------------------------------------------------------------------

/// Header line of the oracle dump's `expected.jsonl`.
#[derive(Debug, Clone)]
pub struct DumpHeader {
    pub mode: String,
    pub width: u32,
    pub height: u32,
    pub tiles: usize,
    pub ticks: u32,
    pub terrain_hash: u64,
    pub post_reset_state_hash: Option<u64>,
    pub record: Option<String>,
    pub game_id: Option<String>,
}

/// One per-tick expected value, produced by the engine itself.
#[derive(Debug, Clone)]
pub struct TickLine {
    pub tick: u32,
    pub index: u32,
    pub state_hash: u64,
    pub game_hash_bits: Option<String>,
}

/// Everything the harness needs: the planes (read on demand) and the engine's
/// own hashes.
pub struct Dump {
    pub dir: PathBuf,
    pub header: DumpHeader,
    pub ticks: Vec<TickLine>,
    pub terrain: Vec<u8>,
    pub post_reset: Option<Vec<u16>>,
    planes_path: PathBuf,
}

impl Dump {
    /// Plane `i` of the dump (row-major `u16`, `width*height` words), read from
    /// `planes.bin` on demand.
    ///
    /// Deliberately *not* held in memory: a 200-tick dump at 1000x1000 is
    /// 400 MB of plane data, and the harness does not need more than one plane
    /// in flight at a time. Reading per tick is also what a GPU-resident
    /// consumer would do - one plane into device memory, hash it, next.
    pub fn plane(&self, i: usize) -> Vec<u16> {
        use std::io::{Read, Seek, SeekFrom};
        let n = self.header.tiles;
        let mut buf = vec![0u8; n * 2];
        let mut f = std::fs::File::open(&self.planes_path).expect("open planes.bin");
        f.seek(SeekFrom::Start((i * n * 2) as u64)).expect("seek plane");
        f.read_exact(&mut buf).expect("read plane");
        buf.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }
}

fn read_jsonl(path: &Path) -> Result<Vec<serde_json::Value>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        out.push(
            serde_json::from_str::<serde_json::Value>(line)
                .map_err(|e| format!("{}: bad json: {e}", path.display()))?,
        );
    }
    Ok(out)
}

/// Load a dump written by `ofcuda_hash/oracle`.
pub fn load_dump(dir: &Path) -> Result<Dump, String> {
    let lines = read_jsonl(&dir.join("expected.jsonl"))?;
    let head = lines
        .first()
        .ok_or_else(|| format!("{}: empty expected.jsonl", dir.display()))?;
    let u = |v: &serde_json::Value, k: &str| -> Result<u64, String> {
        v.get(k)
            .and_then(|x| x.as_u64())
            .ok_or_else(|| format!("header.{k} missing"))
    };
    let s = |v: &serde_json::Value, k: &str| -> Option<String> {
        v.get(k).and_then(|x| x.as_str()).map(|x| x.to_string())
    };
    let header = DumpHeader {
        mode: s(head, "mode").unwrap_or_default(),
        width: u(head, "width")? as u32,
        height: u(head, "height")? as u32,
        tiles: u(head, "tiles")? as usize,
        ticks: u(head, "ticks")? as u32,
        terrain_hash: parse_hex64(
            head.get("terrain_hash")
                .and_then(|v| v.as_str())
                .ok_or("header.terrain_hash missing")?,
        )
        .ok_or("header.terrain_hash not hex")?,
        post_reset_state_hash: head
            .get("post_reset_state_hash")
            .and_then(|v| v.as_str())
            .and_then(parse_hex64),
        record: s(head, "record"),
        game_id: s(head, "game_id"),
    };

    let mut ticks = Vec::new();
    for l in lines.iter().skip(1) {
        let tick = l.get("tick").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let index = l.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let state_hash = l
            .get("state_hash")
            .and_then(|v| v.as_str())
            .and_then(parse_hex64)
            .ok_or("tick.state_hash missing/not hex")?;
        ticks.push(TickLine {
            tick,
            index,
            state_hash,
            game_hash_bits: s(l, "game_hash_bits"),
        });
    }

    let terrain = std::fs::read(dir.join("terrain.bin"))
        .map_err(|e| format!("{}: {e}", dir.join("terrain.bin").display()))?;
    if terrain.len() != header.tiles {
        return Err(format!(
            "terrain.bin is {} bytes, header says {} tiles",
            terrain.len(),
            header.tiles
        ));
    }
    let planes_path = dir.join("planes.bin");
    let planes_len = std::fs::metadata(&planes_path)
        .map_err(|e| format!("{}: {e}", planes_path.display()))?
        .len();
    let expect_len = (header.tiles as u64) * 2 * (ticks.len() as u64);
    if planes_len != expect_len {
        return Err(format!(
            "planes.bin is {planes_len} bytes, expected {expect_len} ({} tiles x {} planes)",
            header.tiles,
            ticks.len()
        ));
    }
    let post_reset = match std::fs::read(dir.join("post_reset.bin")) {
        Ok(b) => {
            let w: Vec<u16> = b
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            if w.len() != header.tiles {
                return Err(format!(
                    "post_reset.bin holds {} words, header says {} tiles",
                    w.len(),
                    header.tiles
                ));
            }
            Some(w)
        }
        Err(_) => None,
    };

    Ok(Dump {
        dir: dir.to_path_buf(),
        header,
        ticks,
        terrain,
        post_reset,
        planes_path,
    })
}