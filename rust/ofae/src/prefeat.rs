//! Prefaturize snapshot games into AE training caches (port of scripts/prefeaturize.py).

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::Args;
use flate2::read::GzDecoder;
use rayon::prelude::*;
use serde_json::Value;
use walkdir::WalkDir;

const OWNER_MASK: u16 = 0x0fff;
const FALLOUT_BIT: u16 = 13;
const MAX_SLOTS: usize = 128;

const UNIT_CLASSES: &[&str] = &[
    "City",
    "Port",
    "Defense Post",
    "Missile Silo",
    "SAM Launcher",
    "Factory",
    "Warship",
    "Transport",
    "Trade Ship",
    "Atom Bomb",
    "Hydrogen Bomb",
    "MIRV",
    "SAMMissile",
    "MIRV Warhead",
    "Train",
];

const STATIC_INDICES: &[usize] = &[0, 1, 2, 3, 4, 5];

#[derive(Debug, Args)]
pub struct PrefeaturizeArgs {
    #[arg(long, default_value = "data")]
    pub data: String,
    #[arg(long, default_value_t = 8)]
    pub workers: usize,
}

pub fn run(args: PrefeaturizeArgs) -> Result<()> {
    let mut game_dirs = Vec::new();
    for root in args.data.split(',') {
        let root = root.trim();
        if root.is_empty() {
            continue;
        }
        for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
            if entry.file_name() == "meta.json" {
                if let Some(parent) = entry.path().parent() {
                    game_dirs.push(parent.to_path_buf());
                }
            }
        }
    }
    game_dirs.sort();
    game_dirs.dedup();
    eprintln!("ofae prefeaturize: {} games", game_dirs.len());

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.workers.max(1))
        .build_global()
        .ok();

    game_dirs.par_iter().for_each(|d| match build_cache(d) {
        Ok(msg) => println!("{msg}"),
        Err(e) => eprintln!("FAIL {}: {e:#}", d.display()),
    });
    Ok(())
}

fn build_cache(game_dir: &Path) -> Result<String> {
    let cache = game_dir.join("cache");
    if cache.join("index.json").exists() {
        ensure_static_sidecar(&cache)?;
        return Ok(format!("skip {}", game_dir.file_name().unwrap().to_string_lossy()));
    }
    let meta: Value = serde_json::from_str(
        &std::fs::read_to_string(game_dir.join("meta.json"))
            .with_context(|| format!("meta {}", game_dir.display()))?,
    )?;
    let h = meta["height"].as_u64().unwrap() as usize;
    let w = meta["width"].as_u64().unwrap() as usize;
    let snaps = meta["snapshots"].as_array().context("snapshots")?;
    let n = snaps.len();

    let terrain = std::fs::read(game_dir.join("terrain.bin"))?;
    if terrain.len() != h * w {
        bail!("terrain size mismatch");
    }

    let lut = slot_lut(&meta, 0)?;
    let lut8: Vec<u8> = lut.iter().map(|&x| x as u8).collect();

    std::fs::create_dir_all(&cache)?;
    let mut frame_offsets = vec![0i64; n + 1];
    let mut unit_rows: Vec<[i32; 7]> = Vec::new();
    let mut unit_offsets = vec![0i64; n + 1];

    let states_dir = game_dir.join("states");
    let mut frames_out = File::create(cache.join("frames.zst"))?;

    for i in 0..n {
        let state = load_state(game_dir, &states_dir, i, h, w, n)?;
        let mut slots = vec![0u8; h * w];
        let mut fall_bits = vec![0u8; h * w];
        for (ti, &s) in state.iter().enumerate() {
            slots[ti] = lut8[(s & OWNER_MASK) as usize];
            fall_bits[ti] = ((s >> FALLOUT_BIT) & 1) as u8;
        }
        let fall_packed = packbits_rows(&fall_bits, h, w);
        let mut frame = slots;
        frame.extend_from_slice(&fall_packed);

        // Reuse encoder by compressing each frame separately (zstd-1).
        let blob = zstd::encode_all(&frame[..], 1)?;
        frames_out.write_all(&blob)?;
        frame_offsets[i + 1] = frame_offsets[i] + blob.len() as i64;

        let ents = load_entities(game_dir, &states_dir, i, snaps)?;
        let mut rows = Vec::new();
        if let Some(units) = ents.get("units").and_then(|u| u.as_array()) {
            for u in units {
                let ty = u.get("type").and_then(|t| t.as_str()).unwrap_or("");
                let Some(ci) = UNIT_CLASSES.iter().position(|&c| c == ty) else {
                    continue;
                };
                let owner = u.get("owner").and_then(|o| o.as_u64()).unwrap_or(0) as usize;
                let x = u.get("x").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                let y = u.get("y").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
                let tx = u
                    .get("tx")
                    .and_then(|v| v.as_i64())
                    .map(|v| v as i32)
                    .unwrap_or(-1);
                let ty_ = u
                    .get("ty")
                    .and_then(|v| v.as_i64())
                    .map(|v| v as i32)
                    .unwrap_or(-1);
                rows.push([
                    i as i32,
                    ci as i32,
                    x,
                    y,
                    tx,
                    ty_,
                    lut.get(owner).copied().unwrap_or(0) as i32,
                ]);
            }
        }
        unit_rows.extend_from_slice(&rows);
        unit_offsets[i + 1] = unit_offsets[i] + rows.len() as i64;
    }

    write_npy_i32_matrix(&cache.join("units.npy"), &unit_rows, 7)?;
    let index = serde_json::json!({
        "n": n,
        "h": h,
        "w": w,
        "max_slots": MAX_SLOTS,
        "frame_offsets": frame_offsets,
        "unit_offsets": unit_offsets,
    });
    std::fs::write(cache.join("index.json"), serde_json::to_string(&index)?)?;
    write_static_from_units(&cache, &unit_rows, &unit_offsets)?;
    let _ = terrain;
    Ok(format!(
        "done {}: {n} snaps, {} unit rows",
        game_dir.file_name().unwrap().to_string_lossy(),
        unit_rows.len()
    ))
}

fn ensure_static_sidecar(cache: &Path) -> Result<()> {
    if cache.join("static.bin").exists() || cache.join("static.npz").exists() {
        return Ok(());
    }
    let units = read_units_npy_simple(&cache.join("units.npy"))?;
    let idx: Value = serde_json::from_str(&std::fs::read_to_string(cache.join("index.json"))?)?;
    let unit_offsets: Vec<i64> = idx["unit_offsets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();
    write_static_from_units(cache, &units, &unit_offsets)
}

fn write_static_from_units(cache: &Path, units: &[[i32; 7]], unit_offsets: &[i64]) -> Result<()> {
    let mut xy = Vec::new();
    let mut cls = Vec::new();
    let mut offsets = vec![0i64];
    let n = unit_offsets.len().saturating_sub(1);
    for si in 0..n {
        let lo = unit_offsets[si] as usize;
        let hi = unit_offsets[si + 1] as usize;
        for row in &units[lo..hi.min(units.len())] {
            let c = row[1] as usize;
            if let Some(pos) = STATIC_INDICES.iter().position(|&s| s == c) {
                xy.push([row[2], row[3]]);
                cls.push(pos as i8);
            }
        }
        offsets.push(xy.len() as i64);
    }
    // Simple binary sidecar: magic, counts, then xy i32 pairs, cls i8, offsets i64.
    let mut out = File::create(cache.join("static.bin"))?;
    out.write_all(b"OFAESTAT")?;
    out.write_all(&(xy.len() as u64).to_le_bytes())?;
    out.write_all(&(offsets.len() as u64).to_le_bytes())?;
    for [x, y] in &xy {
        out.write_all(&x.to_le_bytes())?;
        out.write_all(&y.to_le_bytes())?;
    }
    out.write_all(bytemuck_cast_i8(&cls))?;
    for o in &offsets {
        out.write_all(&o.to_le_bytes())?;
    }
    Ok(())
}

fn bytemuck_cast_i8(v: &[i8]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len()) }
}

fn slot_lut(meta: &Value, snap_i: usize) -> Result<Vec<i64>> {
    let snaps = meta["snapshots"].as_array().context("snapshots")?;
    let players = snaps
        .get(snap_i)
        .and_then(|s| s.get("players"))
        .and_then(|p| p.as_array())
        .cloned()
        .unwrap_or_default();
    let mut small_ids: Vec<u64> = players
        .iter()
        .filter_map(|p| p.get("id").or_else(|| p.get("smallID")).and_then(|v| v.as_u64()))
        .collect();
    small_ids.sort_unstable();
    small_ids.dedup();
    let mut lut = vec![0i64; 4096];
    for (slot, sid) in small_ids.into_iter().enumerate() {
        let slot = (slot + 1).min(MAX_SLOTS - 1) as i64;
        if (sid as usize) < lut.len() {
            lut[sid as usize] = slot;
        }
    }
    Ok(lut)
}

fn load_state(
    game_dir: &Path,
    states_dir: &Path,
    i: usize,
    h: usize,
    w: usize,
    n: usize,
) -> Result<Vec<u16>> {
    if states_dir.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(states_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.starts_with('t') && s.ends_with(".bin.gz"))
                    .unwrap_or(false)
            })
            .collect();
        files.sort();
        if files.len() != n {
            bail!(
                "{}: {} state files but {n} snapshots",
                game_dir.display(),
                files.len()
            );
        }
        let raw = std::fs::read(&files[i])?;
        let mut dec = GzDecoder::new(&raw[..]);
        let mut buf = Vec::new();
        std::io::Read::read_to_end(&mut dec, &mut buf)?;
        if buf.len() != h * w * 2 {
            bail!("state bytes {} != {}", buf.len(), h * w * 2);
        }
        let mut out = vec![0u16; h * w];
        for (j, chunk) in buf.chunks_exact(2).enumerate() {
            out[j] = u16::from_le_bytes([chunk[0], chunk[1]]);
        }
        return Ok(out);
    }
    // old concatenated mmap format
    let path = game_dir.join("states.bin");
    let bytes = std::fs::read(&path)?;
    let need = n * h * w * 2;
    if bytes.len() < need {
        bail!("states.bin too small");
    }
    let start = i * h * w * 2;
    let slice = &bytes[start..start + h * w * 2];
    let mut out = vec![0u16; h * w];
    for (j, chunk) in slice.chunks_exact(2).enumerate() {
        out[j] = u16::from_le_bytes([chunk[0], chunk[1]]);
    }
    Ok(out)
}

fn load_entities(
    game_dir: &Path,
    states_dir: &Path,
    i: usize,
    snaps: &[Value],
) -> Result<Value> {
    if states_dir.is_dir() {
        let mut files: Vec<PathBuf> = std::fs::read_dir(states_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|s| s.to_str())
                    .map(|s| s.starts_with('t') && s.ends_with(".bin.gz"))
                    .unwrap_or(false)
            })
            .collect();
        files.sort();
        if let Some(bin) = files.get(i) {
            // t123.bin.gz -> t123.json.gz
            let ent = bin.with_extension("").with_extension("json.gz");
            // with_extension on .bin.gz is tricky; build manually
            let name = bin.file_name().unwrap().to_string_lossy();
            let stem = name.trim_end_matches(".bin.gz");
            let ent = states_dir.join(format!("{stem}.json.gz"));
            if ent.exists() {
                let raw = std::fs::read(&ent)?;
                let mut dec = GzDecoder::new(&raw[..]);
                let mut s = String::new();
                std::io::Read::read_to_string(&mut dec, &mut s)?;
                return Ok(serde_json::from_str(&s)?);
            }
            let _ = game_dir;
        }
    }
    let snap = &snaps[i];
    let players = snap.get("players").cloned().unwrap_or(Value::Array(vec![]));
    Ok(serde_json::json!({
        "players": players,
        "alliances": [],
        "units": [],
        "attacks": [],
    }))
}

fn packbits_rows(bits: &[u8], h: usize, w: usize) -> Vec<u8> {
    let packed_w = w.div_ceil(8);
    let mut out = vec![0u8; h * packed_w];
    for y in 0..h {
        for x in 0..w {
            if bits[y * w + x] != 0 {
                let byte = y * packed_w + x / 8;
                let bit = 7 - (x % 8);
                out[byte] |= 1 << bit;
            }
        }
    }
    out
}

fn write_npy_i32_matrix(path: &Path, rows: &[[i32; 7]], cols: usize) -> Result<()> {
    // Minimal npy writer for int32 C-order array.
    let n = rows.len();
    let header = format!(
        "{{'descr': '<i4', 'fortran_order': False, 'shape': ({n}, {cols}), }}"
    );
    let mut header_bytes = header.into_bytes();
    // pad to 16-byte alignment: magic(6)+ver(2)+hlen(2)+header+\\n
    let mut total = 10 + header_bytes.len() + 1;
    let pad = (16 - (total % 16)) % 16;
    header_bytes.extend(std::iter::repeat(b' ').take(pad));
    header_bytes.push(b'\n');
    total = 10 + header_bytes.len();
    assert_eq!(total % 16, 0);

    let mut f = File::create(path)?;
    f.write_all(b"\x93NUMPY")?;
    f.write_all(&[1u8, 0u8])?;
    f.write_all(&(header_bytes.len() as u16).to_le_bytes())?;
    f.write_all(&header_bytes)?;
    for row in rows {
        for &v in row {
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

fn read_units_npy_simple(path: &Path) -> Result<Vec<[i32; 7]>> {
    if !path.exists() {
        return Ok(vec![]);
    }
    let bytes = std::fs::read(path)?;
    let npy = npyz::NpyFile::new(&bytes[..])?;
    let data: Vec<i32> = npy.into_vec()?;
    Ok(data
        .chunks_exact(7)
        .map(|c| [c[0], c[1], c[2], c[3], c[4], c[5], c[6]])
        .collect())
}
