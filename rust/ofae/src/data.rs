//! AE training caches: zstd frames + terrain + static structure sidecars.

use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;
use walkdir::WalkDir;

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

        Ok(Self {
            path: game_dir.to_path_buf(),
            n: idx.n,
            h: idx.h,
            w: idx.w,
            frames,
            frame_offsets: idx.frame_offsets,
            terr,
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

    /// Sample one training crop: (owners HxW i64, terrain 3xHxW f32, y0, x0).
    /// Static structure planes are no longer AE targets (v3.2); sidecars remain
    /// on disk for tooling but are not sampled into the training batch.
    pub fn sample(
        &self,
        rng: &mut SmallRng,
        crop: usize,
        border_sample: bool,
    ) -> Result<(Vec<i64>, Vec<f32>, usize, usize)> {
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

        Ok((owners, terrain, y0, x0))
    }
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
    ) -> Result<(Vec<i64>, Vec<f32>)> {
        let mut owners = Vec::with_capacity(batch * crop * crop);
        let mut terrain = Vec::with_capacity(batch * 3 * crop * crop);
        for _ in 0..batch {
            let gi = rng.gen_range(0..self.games.len());
            let (o, t, _, _) = self.games[gi].sample(rng, crop, true)?;
            owners.extend(o);
            terrain.extend(t);
        }
        Ok((owners, terrain))
    }

    pub fn worker_rng(seed: u64, worker: u64) -> SmallRng {
        SmallRng::seed_from_u64(seed.wrapping_mul(100_003).wrapping_add(worker))
    }
}
