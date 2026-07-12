//! Frozen SpatialAE encoder: tch port of `ae/model_v3.py::SpatialAE.encode`.
//!
//! PPO only needs the encoder (owner embedding + stem + fuse). Weights are
//! loaded from the safetensors files produced by
//! `scripts/export_safetensors.py` (keys match PyTorch state_dict / tch's
//! '.' path separator exactly). Decoder weights are intentionally absent.
//!
//! Production checkpoints:
//! - fine:  `ae_v31_d8c32`  — 32ch @ 1/8
//! - coarse: `ae_v31_d16c32` — 32ch @ 1/16 (optional v7 coarse stream)

use anyhow::{bail, Context, Result};
use std::path::Path;
use tch::nn::{self, Module};
use tch::{Device, Kind, Tensor};

pub const MAX_SLOTS: i64 = 128;
pub const OWNER_EMB_DIM: i64 = 8;
pub const TERRAIN_CHANNELS: i64 = 3;
pub const NUM_STATIC: i64 = 6;
pub const LATENT_C: i64 = 32;
pub const REGION: i64 = 8;
pub const COARSE_REGION: i64 = 16;

struct ConvGn {
    conv: nn::Conv2D,
    gn: nn::GroupNorm,
}

impl ConvGn {
    fn new(vs: &nn::Path, c_in: i64, c_out: i64, stride: i64) -> Self {
        let mut cfg = nn::ConvConfig::default();
        cfg.stride = stride;
        cfg.padding = 1;
        Self {
            conv: nn::conv2d(vs / 0, c_in, c_out, 3, cfg),
            gn: nn::group_norm(vs / 1, 8, c_out, Default::default()),
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        self.gn.forward(&self.conv.forward(xs)).silu()
    }
}

/// Encoder-only SpatialAE. `VarStore` must contain exactly the encoder
/// tensors (no decoder) so `vs.load(safetensors)` succeeds strictly.
pub struct SpatialAE {
    owner_emb: nn::Embedding,
    enc_stem: Vec<ConvGn>,
    enc_fuse: ConvGn,
    enc_out: nn::Conv2D,
    pub latent_c: i64,
    pub latent_down: i64,
}

impl SpatialAE {
    /// Build the encoder module tree under `vs` (typically `vs.root()`).
    /// Call [`SpatialAE::load`] afterward to copy weights in.
    pub fn new(vs: &nn::Path, latent_c: i64, latent_down: i64) -> Result<Self> {
        if latent_down != 8 && latent_down != 16 {
            bail!("latent_down must be 8 or 16, got {latent_down}");
        }
        let owner_emb = nn::embedding(vs / "owner_emb", MAX_SLOTS, OWNER_EMB_DIM, Default::default());

        let stem_specs: &[(i64, i64, i64)] = if latent_down == 16 {
            &[
                (OWNER_EMB_DIM + TERRAIN_CHANNELS, 32, 1),
                (32, 64, 2),
                (64, 96, 2),
                (96, 128, 2),
                (128, 128, 2),
            ]
        } else {
            &[
                (OWNER_EMB_DIM + TERRAIN_CHANNELS, 32, 1),
                (32, 64, 2),
                (64, 96, 2),
                (96, 128, 2),
            ]
        };
        let enc_stem: Vec<ConvGn> = stem_specs
            .iter()
            .enumerate()
            .map(|(i, &(ci, co, st))| ConvGn::new(&(vs / "enc_stem" / i), ci, co, st))
            .collect();

        let enc_fuse = ConvGn::new(&(vs / "enc_fuse" / 0), 128 + NUM_STATIC, 128, 1);
        let mut out_cfg = nn::ConvConfig::default();
        out_cfg.stride = 1;
        out_cfg.padding = 0;
        let enc_out = nn::conv2d(vs / "enc_fuse" / 1, 128, latent_c, 1, out_cfg);

        Ok(Self {
            owner_emb,
            enc_stem,
            enc_fuse,
            enc_out,
            latent_c,
            latent_down,
        })
    }

    /// Construct on `device`, load encoder weights from a `.safetensors`
    /// file produced by `scripts/export_safetensors.py`, then freeze.
    pub fn load(path: impl AsRef<Path>, device: Device, expected_down: i64) -> Result<(nn::VarStore, Self)> {
        let path = path.as_ref();
        let meta_path = path.with_extension("json");
        let (latent_c, latent_down) = if meta_path.exists() {
            let raw = std::fs::read_to_string(&meta_path)
                .with_context(|| format!("read AE meta {}", meta_path.display()))?;
            let v: serde_json::Value = serde_json::from_str(&raw)?;
            (
                v.get("latent_c").and_then(|x| x.as_i64()).unwrap_or(LATENT_C),
                v.get("latent_down").and_then(|x| x.as_i64()).unwrap_or(expected_down),
            )
        } else {
            (LATENT_C, expected_down)
        };
        if latent_down != expected_down {
            bail!(
                "AE {} has latent_down={latent_down}, expected {expected_down}",
                path.display()
            );
        }
        if latent_c != LATENT_C {
            bail!(
                "AE {} has latent_c={latent_c}, expected {LATENT_C}",
                path.display()
            );
        }

        let mut vs = nn::VarStore::new(device);
        let ae = Self::new(&vs.root(), latent_c, latent_down)?;
        vs.load(path)
            .with_context(|| format!("load AE encoder weights from {}", path.display()))?;
        vs.freeze();
        Ok((vs, ae))
    }

    /// `owners` (B,H,W) int64, `terrain` (B,3,H,W) f32, `static_planes`
    /// (B,6,H/down,W/down) f32 → `z` (B,latent_c,H/down,W/down) f32.
    pub fn encode(&self, owners: &Tensor, terrain: &Tensor, static_planes: &Tensor) -> Tensor {
        let emb = self.owner_emb.forward(owners).permute([0, 3, 1, 2]);
        let mut g = Tensor::cat(&[&emb, terrain], 1);
        for block in &self.enc_stem {
            g = block.forward(&g);
        }
        self.enc_out.forward(&self.enc_fuse.forward(&Tensor::cat(&[&g, static_planes], 1)))
    }
}

/// Optional fine (+ coarse) AE pair used by the rollout encode path.
pub struct AePair {
    pub fine: SpatialAE,
    /// Keep VarStores alive for the lifetime of the encoders.
    _fine_vs: nn::VarStore,
    pub coarse: Option<SpatialAE>,
    _coarse_vs: Option<nn::VarStore>,
}

impl AePair {
    pub fn load(fine_path: impl AsRef<Path>, coarse_path: Option<&Path>, device: Device) -> Result<Self> {
        let (fine_vs, fine) = SpatialAE::load(fine_path.as_ref(), device, REGION)?;
        let (coarse_vs, coarse) = match coarse_path {
            Some(p) => {
                let (vs, ae) = SpatialAE::load(p, device, COARSE_REGION)?;
                (Some(vs), Some(ae))
            }
            None => (None, None),
        };
        Ok(Self {
            fine,
            _fine_vs: fine_vs,
            coarse,
            _coarse_vs: coarse_vs,
        })
    }
}

/// Full-res AE inputs staged by `vecenv::prepare` for batched GPU encode.
#[derive(Clone)]
pub struct AeRaw {
    pub owners: Vec<i64>, // (hr*wr) slotted
    pub terrain: Vec<f32>, // (3, hr, wr) row-major channels-first
    pub stat: Vec<f32>, // (6, gh, gw)
    pub hr: usize,
    pub wr: usize,
}

/// Build land/magnitude/fallout terrain tensor planes on the host
/// (channels-first). `mag` is already the raw magnitude byte; we normalize
/// by /31 to match `rl/obs.py`.
pub fn pack_terrain(land: &[u8], mag: &[u8], fallout: &[u8], hr: usize, wr: usize) -> Vec<f32> {
    let n = hr * wr;
    debug_assert_eq!(land.len(), n);
    debug_assert_eq!(mag.len(), n);
    debug_assert_eq!(fallout.len(), n);
    let mut out = vec![0.0f32; 3 * n];
    for i in 0..n {
        out[i] = if land[i] != 0 { 1.0 } else { 0.0 };
        out[n + i] = (mag[i] as f32) / 31.0;
        out[2 * n + i] = if fallout[i] != 0 { 1.0 } else { 0.0 };
    }
    out
}

/// Batched encode helpers used by `batch::build_obs`. Groups by shape,
/// chunks by pixel budget (mirrors `rl/obs.py::MAX_ENC_PIX`).
pub const MAX_ENC_PIX: usize = 16_000_000;

pub fn encode_latent_batch(
    ae: &SpatialAE,
    items: &[&AeRaw],
    device: Device,
) -> Result<Vec<Tensor>> {
    // Returns one (latent_c, gh, gw) CPU f32 tensor per item, in input order.
    let n = items.len();
    let mut out: Vec<Option<Tensor>> = (0..n).map(|_| None).collect();
    let mut groups: std::collections::HashMap<(usize, usize), Vec<usize>> =
        std::collections::HashMap::new();
    for (i, it) in items.iter().enumerate() {
        groups.entry((it.hr, it.wr)).or_default().push(i);
    }

    for ((hr, wr), idxs) in groups {
        let pix = hr * wr;
        let per = (MAX_ENC_PIX / pix.max(1)).max(1);
        for chunk in idxs.chunks(per) {
            let b = chunk.len() as i64;
            let mut owners = Vec::with_capacity(chunk.len() * pix);
            let mut terrain = Vec::with_capacity(chunk.len() * 3 * pix);
            let gh = hr / ae.latent_down as usize;
            let gw = wr / ae.latent_down as usize;
            // static must be at latent resolution. Fine AE uses /8 = REGION
            // which matches `stat` from featurize. Coarse AE needs /16:
            // max-pool the /8 static 2x (ceil), matching encode_grids.
            let mut static_p = Vec::with_capacity(chunk.len() * NUM_STATIC as usize * gh * gw);
            for &i in chunk {
                let it = items[i];
                owners.extend_from_slice(&it.owners);
                terrain.extend_from_slice(&it.terrain);
                if ae.latent_down == REGION {
                    static_p.extend_from_slice(&it.stat);
                } else {
                    static_p.extend(max_pool2_stat(&it.stat, it.hr / REGION as usize, it.wr / REGION as usize));
                }
            }
            let owners_t = Tensor::from_slice(&owners)
                .view([b, hr as i64, wr as i64])
                .to_device(device)
                .to_kind(Kind::Int64);
            let terrain_t = Tensor::from_slice(&terrain)
                .view([b, 3, hr as i64, wr as i64])
                .to_device(device)
                .to_kind(Kind::Float);
            let static_t = Tensor::from_slice(&static_p)
                .view([b, NUM_STATIC, gh as i64, gw as i64])
                .to_device(device)
                .to_kind(Kind::Float);
            let z = tch::no_grad(|| ae.encode(&owners_t, &terrain_t, &static_t));
            // Split batch back to per-item CPU tensors.
            for (j, &i) in chunk.iter().enumerate() {
                let zj = z.select(0, j as i64).to_device(Device::Cpu).to_kind(Kind::Float);
                out[i] = Some(zj);
            }
        }
    }

    out.into_iter()
        .enumerate()
        .map(|(i, t)| t.with_context(|| format!("missing AE latent for item {i}")))
        .collect()
}

/// 2x max-pool over (C,H,W) host planes with ceil mode (odd dims keep a
/// 1-wide edge), matching `F.max_pool2d(..., ceil_mode=True)`.
fn max_pool2_stat(stat: &[f32], gh: usize, gw: usize) -> Vec<f32> {
    let cgh = gh.div_ceil(2);
    let cgw = gw.div_ceil(2);
    let mut out = vec![0.0f32; NUM_STATIC as usize * cgh * cgw];
    for c in 0..NUM_STATIC as usize {
        for y in 0..cgh {
            for x in 0..cgw {
                let y0 = y * 2;
                let x0 = x * 2;
                let mut m = f32::NEG_INFINITY;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let yy = y0 + dy;
                        let xx = x0 + dx;
                        if yy < gh && xx < gw {
                            m = m.max(stat[c * gh * gw + yy * gw + xx]);
                        }
                    }
                }
                if m == f32::NEG_INFINITY {
                    m = 0.0;
                }
                out[c * cgh * cgw + y * cgw + x] = m;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_tree_key_count_d8() {
        let vs = nn::VarStore::new(Device::Cpu);
        let _ae = SpatialAE::new(&vs.root(), 32, 8).unwrap();
        // 1 emb + 4 stem*(conv.w/b + gn.w/b) + fuse_block + 1x1 out = 1+16+4+2 = 23
        assert_eq!(vs.variables().len(), 23);
    }

    #[test]
    fn encoder_tree_key_count_d16() {
        let vs = nn::VarStore::new(Device::Cpu);
        let _ae = SpatialAE::new(&vs.root(), 32, 16).unwrap();
        // +1 stem block (4 tensors) vs d8 → 27
        assert_eq!(vs.variables().len(), 27);
    }

    #[test]
    fn load_exported_d8_weights_and_encode() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../weights/ae/ae_v31_d8c32.encoder.safetensors");
        if !path.exists() {
            eprintln!("skip: {} missing (run scripts/fetch_ae_encoders.sh)", path.display());
            return;
        }
        let (_vs, ae) = SpatialAE::load(&path, Device::Cpu, REGION).unwrap();
        assert_eq!(ae.latent_down, 8);
        let owners = Tensor::zeros([1, 32, 40], (Kind::Int64, Device::Cpu));
        let terrain = Tensor::zeros([1, 3, 32, 40], (Kind::Float, Device::Cpu));
        let static_p = Tensor::zeros([1, 6, 4, 5], (Kind::Float, Device::Cpu));
        let z = tch::no_grad(|| ae.encode(&owners, &terrain, &static_p));
        assert_eq!(z.size(), &[1, 32, 4, 5]);
        assert!(z.isfinite().all().double_value(&[]) != 0.0);
    }
}
