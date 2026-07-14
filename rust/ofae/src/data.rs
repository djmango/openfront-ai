//! AE training caches: zstd frames + terrain + static structure sidecars.

use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use walkdir::WalkDir;

use crate::model::NUM_STATIC;

pub const IS_LAND_BIT: u8 = 7;
pub const MAGNITUDE_MASK: u8 = 0x1f;
pub const MAGNITUDE_NORM: f32 = 31.0;
pub const BORDER_SAMPLE_TRIES: usize = 8;
pub const BORDER_SAMPLE_FLOOR: f64 = 0.15;
pub const BORDER_SAMPLE_FULL_DENSITY: f64 = 0.05;

#[derive(Deserialize)]
struct IndexJson {
    n: usize,
    h: usize,
    w: usize,
    frame_offsets: Vec<i64>,
    #[serde(default)]
    unit_offsets: Vec<i64>,
}

pub struct CachedGame {
    pub path: PathBuf,
    pub n: usize,
    pub h: usize,
    pub w: usize,
    frames: Mmap,
    frame_offsets: Vec<i64>,
    terr: Vec<u8>,
    static_xy: Vec<[i32; 2]>,
    static_cls: Vec<i8>,
    static_offsets: Vec<i64>,
}

impl CachedGame {
    pub fn open(game_dir: impl AsRef<Path>) -> Result<Self> {
        let game_dir = game_dir.as_ref();
        let cache = game_dir.join("cache");
        let idx: IndexJson = serde_json::from_str(
            &std::fs::read_to_string(cache.join("index.json"))
                .with_context(|| format!("read {}", cache.join("index.json").display()))?,
        )?;
        let frames_file = File::open(cache.join("frames.zst"))
            .with_context(|| format!("open {}", cache.join("frames.zst").display()))?;
        let frames = unsafe { Mmap::map(&frames_file)? };
        let terr = std::fs::read(game_dir.join("terrain.bin"))
            .with_context(|| format!("read terrain {}", game_dir.display()))?;
        if terr.len() != idx.h * idx.w {
            bail!(
                "{}: terrain.bin len {} != h*w {}",
                game_dir.display(),
                terr.len(),
                idx.h * idx.w
            );
        }

        let (static_xy, static_cls, static_offsets) =
            load_or_build_static(&cache, &idx.unit_offsets)?;

        Ok(Self {
            path: game_dir.to_path_buf(),
            n: idx.n,
            h: idx.h,
            w: idx.w,
            frames,
            frame_offsets: idx.frame_offsets,
            terr,
            static_xy,
            static_cls,
            static_offsets,
        })
    }

    fn frame(&self, si: usize) -> Result<(Vec<u8>, Vec<u8>)> {
        let lo = self.frame_offsets[si] as usize;
        let hi = self.frame_offsets[si + 1] as usize;
        let raw = zstd::decode_all(&self.frames[lo..hi])
            .with_context(|| format!("zstd frame {si} in {}", self.path.display()))?;
        let hw = self.h * self.w;
        let packed_w = self.w.div_ceil(8);
        if raw.len() < hw + self.h * packed_w {
            bail!(
                "{} snap {si}: frame bytes {} too small",
                self.path.display(),
                raw.len()
            );
        }
        let slots = raw[..hw].to_vec();
        let packed = raw[hw..hw + self.h * packed_w].to_vec();
        Ok((slots, packed))
    }

    /// Sample one training crop: (owners HxW i64-as-u8, terrain 3xHxW f32, planes Cxghxgw f32).
    pub fn sample(
        &self,
        rng: &mut SmallRng,
        crop: usize,
        latent_down: i64,
        border_sample: bool,
    ) -> Result<(Vec<i64>, Vec<f32>, Vec<f32>, usize, usize)> {
        let mut si = rng.gen_range(0..self.n);
        let (mut slots_full, mut fall_full) = self.frame(si)?;
        let mut y0 = 0usize;
        let mut x0 = 0usize;
        for attempt in 0..BORDER_SAMPLE_TRIES {
            if attempt == BORDER_SAMPLE_TRIES / 2 {
                si = rng.gen_range(0..self.n);
                let f = self.frame(si)?;
                slots_full = f.0;
                fall_full = f.1;
            }
            let y_span = ((self.h.saturating_sub(crop)) / 16).max(0) + 1;
            let x_span = ((self.w.saturating_sub(crop)) / 16).max(0) + 1;
            y0 = rng.gen_range(0..y_span) * 16;
            x0 = rng.gen_range(0..x_span) * 16;
            if y0 + crop > self.h || x0 + crop > self.w {
                continue;
            }
            if !border_sample {
                break;
            }
            let mut edges = 0usize;
            for yy in 0..crop {
                for xx in 0..crop {
                    let v = slots_full[(y0 + yy) * self.w + (x0 + xx)];
                    if yy + 1 < crop {
                        let n = slots_full[(y0 + yy + 1) * self.w + (x0 + xx)];
                        if v != n {
                            edges += 1;
                        }
                    }
                    if xx + 1 < crop {
                        let n = slots_full[(y0 + yy) * self.w + (x0 + xx + 1)];
                        if v != n {
                            edges += 1;
                        }
                    }
                }
            }
            let p = (edges as f64 / (crop * crop) as f64 / BORDER_SAMPLE_FULL_DENSITY)
                .clamp(BORDER_SAMPLE_FLOOR, 1.0);
            if rng.gen::<f64>() < p {
                break;
            }
        }

        let mut owners = Vec::with_capacity(crop * crop);
        let packed_w = self.w.div_ceil(8);
        let mut fallout = vec![0f32; crop * crop];
        for yy in 0..crop {
            for xx in 0..crop {
                owners.push(slots_full[(y0 + yy) * self.w + (x0 + xx)] as i64);
                let bit_x = x0 + xx;
                let byte = fall_full[(y0 + yy) * packed_w + bit_x / 8];
                // numpy packbits is MSB-first within the byte
                let bit = 7 - (bit_x % 8);
                fallout[yy * crop + xx] = if (byte >> bit) & 1 != 0 { 1.0 } else { 0.0 };
            }
        }

        let mut terrain = vec![0f32; 3 * crop * crop];
        for yy in 0..crop {
            for xx in 0..crop {
                let t = self.terr[(y0 + yy) * self.w + (x0 + xx)];
                let i = yy * crop + xx;
                terrain[i] = ((t >> IS_LAND_BIT) & 1) as f32;
                terrain[crop * crop + i] = (t & MAGNITUDE_MASK) as f32 / MAGNITUDE_NORM;
                terrain[2 * crop * crop + i] = fallout[i];
            }
        }

        let g16 = crop / 16;
        let mut planes16 = vec![0f32; (NUM_STATIC as usize) * g16 * g16];
        let lo = self.static_offsets[si] as usize;
        let hi = self.static_offsets[si + 1] as usize;
        for i in lo..hi {
            let [ux, uy] = self.static_xy[i];
            let gx = (ux as isize - x0 as isize) / 16;
            let gy = (uy as isize - y0 as isize) / 16;
            if gx >= 0 && gy >= 0 && (gx as usize) < g16 && (gy as usize) < g16 {
                let c = self.static_cls[i] as usize;
                if c < NUM_STATIC as usize {
                    let idx = c * g16 * g16 + (gy as usize) * g16 + gx as usize;
                    planes16[idx] = (planes16[idx] + 1.0).min(1.0);
                }
            }
        }

        let planes = if latent_down == 8 {
            // nearest 2x upsample
            let g8 = crop / 8;
            let mut out = vec![0f32; (NUM_STATIC as usize) * g8 * g8];
            for c in 0..NUM_STATIC as usize {
                for gy in 0..g16 {
                    for gx in 0..g16 {
                        let v = planes16[c * g16 * g16 + gy * g16 + gx];
                        for dy in 0..2 {
                            for dx in 0..2 {
                                out[c * g8 * g8 + (gy * 2 + dy) * g8 + (gx * 2 + dx)] = v;
                            }
                        }
                    }
                }
            }
            out
        } else {
            planes16
        };

        let gh = crop / latent_down as usize;
        Ok((owners, terrain, planes, crop, gh))
    }
}

fn load_or_build_static(
    cache: &Path,
    unit_offsets: &[i64],
) -> Result<(Vec<[i32; 2]>, Vec<i8>, Vec<i64>)> {
    let bin_path = cache.join("static.bin");
    if bin_path.exists() {
        return read_static_bin(&bin_path);
    }
    let static_path = cache.join("static.npz");
    if static_path.exists() {
        return read_static_npz(&static_path);
    }
    // Build from units.npy (same as Python CachedGame constructor).
    let units_path = cache.join("units.npy");
    if !units_path.exists() {
        let n = unit_offsets.len().saturating_sub(1);
        return Ok((vec![], vec![], vec![0i64; n + 1]));
    }
    let units = read_units_npy(&units_path)?;
    const STATIC_INDICES: [i32; 6] = [0, 1, 2, 3, 4, 5];
    let mut xy = Vec::new();
    let mut cls = Vec::new();
    let mut offsets = vec![0i64];
    let n_snaps = unit_offsets.len().saturating_sub(1);
    for si in 0..n_snaps {
        let lo = unit_offsets[si] as usize;
        let hi = unit_offsets[si + 1] as usize;
        for row in &units[lo..hi.min(units.len())] {
            let c = row[1];
            if let Some(pos) = STATIC_INDICES.iter().position(|&s| s == c) {
                xy.push([row[2], row[3]]);
                cls.push(pos as i8);
            }
        }
        offsets.push(xy.len() as i64);
    }
    Ok((xy, cls, offsets))
}

fn read_static_bin(path: &Path) -> Result<(Vec<[i32; 2]>, Vec<i8>, Vec<i64>)> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 24 || &bytes[..8] != b"OFAESTAT" {
        bail!("bad static.bin magic in {}", path.display());
    }
    let n_xy = u64::from_le_bytes(bytes[8..16].try_into()?) as usize;
    let n_off = u64::from_le_bytes(bytes[16..24].try_into()?) as usize;
    let mut o = 24usize;
    let mut xy = Vec::with_capacity(n_xy);
    for _ in 0..n_xy {
        let x = i32::from_le_bytes(bytes[o..o + 4].try_into()?);
        let y = i32::from_le_bytes(bytes[o + 4..o + 8].try_into()?);
        xy.push([x, y]);
        o += 8;
    }
    let cls = bytes[o..o + n_xy].iter().map(|&b| b as i8).collect();
    o += n_xy;
    let mut offsets = Vec::with_capacity(n_off);
    for _ in 0..n_off {
        offsets.push(i64::from_le_bytes(bytes[o..o + 8].try_into()?));
        o += 8;
    }
    Ok((xy, cls, offsets))
}

fn read_static_npz(path: &Path) -> Result<(Vec<[i32; 2]>, Vec<i8>, Vec<i64>)> {
    // Existing Python caches ship units.npy beside static.npz; rebuild from that.
    let cache = path.parent().context("static.npz parent")?;
    let units_path = cache.join("units.npy");
    let idx_path = cache.join("index.json");
    if !units_path.exists() || !idx_path.exists() {
        bail!(
            "cannot read {} without sibling units.npy + index.json",
            path.display()
        );
    }
    let idx: IndexJson = serde_json::from_str(&std::fs::read_to_string(idx_path)?)?;
    let units = read_units_npy(&units_path)?;
    const STATIC_INDICES: [i32; 6] = [0, 1, 2, 3, 4, 5];
    let mut xy = Vec::new();
    let mut cls = Vec::new();
    let mut offsets = vec![0i64];
    let n_snaps = idx.unit_offsets.len().saturating_sub(1);
    for si in 0..n_snaps {
        let lo = idx.unit_offsets[si] as usize;
        let hi = idx.unit_offsets[si + 1] as usize;
        for row in &units[lo..hi.min(units.len())] {
            let c = row[1];
            if let Some(pos) = STATIC_INDICES.iter().position(|&s| s == c) {
                xy.push([row[2], row[3]]);
                cls.push(pos as i8);
            }
        }
        offsets.push(xy.len() as i64);
    }
    Ok((xy, cls, offsets))
}

fn read_units_npy(path: &Path) -> Result<Vec<[i32; 7]>> {
    let bytes = std::fs::read(path)?;
    let npy = npyz::NpyFile::new(&bytes[..])?;
    let shape = npy.shape().to_vec();
    if shape.len() != 2 || shape[1] != 7 {
        bail!("{}: expected (m,7), got {:?}", path.display(), shape);
    }
    let data: Vec<i32> = npy.into_vec()?;
    Ok(data
        .chunks_exact(7)
        .map(|c| [c[0], c[1], c[2], c[3], c[4], c[5], c[6]])
        .collect())
}

pub fn discover_games(data_roots: &str) -> Result<Vec<PathBuf>> {
    let mut dirs = Vec::new();
    for root in data_roots.split(',') {
        let root = root.trim();
        if root.is_empty() {
            continue;
        }
        for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
            if entry.file_name() == "index.json" {
                if let Some(cache) = entry.path().parent() {
                    if cache.file_name().and_then(|s| s.to_str()) == Some("cache") {
                        if let Some(game) = cache.parent() {
                            dirs.push(game.to_path_buf());
                        }
                    }
                }
            }
        }
    }
    dirs.sort();
    dirs.dedup();
    if dirs.is_empty() {
        bail!("no caches under {data_roots}; run `ofae prefeaturize`");
    }
    Ok(dirs)
}

pub struct GameBank {
    games: Vec<CachedGame>,
}

impl GameBank {
    pub fn load(data_roots: &str) -> Result<Self> {
        let dirs = discover_games(data_roots)?;
        eprintln!("ofae: {} cached games under {data_roots}", dirs.len());
        let mut games = Vec::with_capacity(dirs.len());
        for d in dirs {
            games.push(CachedGame::open(&d)?);
        }
        Ok(Self { games })
    }

    pub fn sample_batch(
        &self,
        rng: &mut SmallRng,
        batch: usize,
        crop: usize,
        latent_down: i64,
    ) -> Result<(Vec<i64>, Vec<f32>, Vec<f32>)> {
        let gh = crop / latent_down as usize;
        let mut owners = Vec::with_capacity(batch * crop * crop);
        let mut terrain = Vec::with_capacity(batch * 3 * crop * crop);
        let mut planes = Vec::with_capacity(batch * (NUM_STATIC as usize) * gh * gh);
        for _ in 0..batch {
            let gi = rng.gen_range(0..self.games.len());
            let (o, t, p, _, _) =
                self.games[gi].sample(rng, crop, latent_down, true)?;
            owners.extend(o);
            terrain.extend(t);
            planes.extend(p);
        }
        Ok((owners, terrain, planes))
    }

    pub fn worker_rng(seed: u64, worker: u64) -> SmallRng {
        SmallRng::seed_from_u64(seed.wrapping_mul(100_003).wrapping_add(worker))
    }
}
