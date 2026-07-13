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

use anyhow::{Context, Result, bail};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
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
        let owner_emb = nn::embedding(
            vs / "owner_emb",
            MAX_SLOTS,
            OWNER_EMB_DIM,
            Default::default(),
        );

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
    pub fn load(
        path: impl AsRef<Path>,
        device: Device,
        expected_down: i64,
    ) -> Result<(nn::VarStore, Self)> {
        let path = path.as_ref();
        let meta_path = path.with_extension("json");
        let (latent_c, latent_down) = if meta_path.exists() {
            let raw = std::fs::read_to_string(&meta_path)
                .with_context(|| format!("read AE meta {}", meta_path.display()))?;
            let v: serde_json::Value = serde_json::from_str(&raw)?;
            (
                v.get("latent_c")
                    .and_then(|x| x.as_i64())
                    .unwrap_or(LATENT_C),
                v.get("latent_down")
                    .and_then(|x| x.as_i64())
                    .unwrap_or(expected_down),
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
        self.enc_out
            .forward(&self.enc_fuse.forward(&Tensor::cat(&[&g, static_planes], 1)))
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
    pub fn load(
        fine_path: impl AsRef<Path>,
        coarse_path: Option<&Path>,
        device: Device,
    ) -> Result<Self> {
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

/// Identity of one environment's episode-static terrain. Every reset gets
/// a new `static_id`, even when it happens to select the same map.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TerrainCacheKey {
    pub env_id: u64,
    pub episode: u64,
    pub static_id: u64,
    pub hr: usize,
    pub wr: usize,
}

/// Episode-static AE terrain channels, shared from EnvWorker to every
/// PreparedObs without rebuilding or copying their float payload.
#[derive(Clone)]
pub struct StaticTerrain {
    pub key: TerrainCacheKey,
    pub map: Arc<str>,
    pub land_mag: Arc<[f32]>, // (2, hr, wr), land then magnitude
}

/// Full-res AE inputs staged by `vecenv::prepare` for batched GPU encode.
#[derive(Clone)]
pub struct AeRaw {
    pub owners: Vec<u8>, // (hr*wr) slotted, embedding indices 0..=127
    pub static_terrain: StaticTerrain,
    /// NumPy/ofrs-compatible row-major packbits: each row occupies
    /// `ceil(wr/8)` bytes and its leftmost pixel is bit 7.
    pub fallout: Vec<u8>, // (hr, ceil(wr/8)), fresh every decision
    pub stat: Vec<f32>,    // (6, gh, gw)
    pub hr: usize,
    pub wr: usize,
}

/// Build the two immutable terrain planes once per episode.
pub fn pack_static_terrain(land: &[u8], mag: &[u8], hr: usize, wr: usize) -> Arc<[f32]> {
    let n = hr * wr;
    debug_assert_eq!(land.len(), n);
    debug_assert_eq!(mag.len(), n);
    let mut out = vec![0.0f32; 2 * n];
    for i in 0..n {
        out[i] = if land[i] != 0 { 1.0 } else { 0.0 };
        out[n + i] = (mag[i] as f32) / 31.0;
    }
    out.into()
}

pub const fn packed_fallout_row_bytes(wr: usize) -> usize {
    wr.div_ceil(8)
}

/// Match `numpy.packbits(..., axis=1)` and ofrs exactly: MSB-first within
/// each byte, with a new byte-aligned row and zero low-bit padding.
pub fn pack_fallout(fallout: &[u8], hr: usize, wr: usize) -> Vec<u8> {
    assert_eq!(fallout.len(), hr * wr, "fallout shape mismatch");
    let packed_wr = packed_fallout_row_bytes(wr);
    let mut packed = vec![0u8; hr * packed_wr];
    for y in 0..hr {
        for x in 0..wr {
            if fallout[y * wr + x] != 0 {
                packed[y * packed_wr + x / 8] |= 1 << (7 - x % 8);
            }
        }
    }
    packed
}

/// Exact legacy/CPU fallback. The persistent CUDA actor uses
/// [`unpack_fallout_device`] instead, so its full f32 plane never exists on
/// the host and never crosses PCIe.
fn unpack_fallout_host(packed: &[u8], hr: usize, wr: usize) -> Vec<f32> {
    let packed_wr = packed_fallout_row_bytes(wr);
    debug_assert_eq!(packed.len(), hr * packed_wr);
    let mut fallout = vec![0.0f32; hr * wr];
    for y in 0..hr {
        for x in 0..wr {
            fallout[y * wr + x] =
                ((packed[y * packed_wr + x / 8] >> (7 - x % 8)) & 1) as f32;
        }
    }
    fallout
}

/// Expand `(B,H,ceil(W/8))` u8 packbits on `packed.device()` to the exact
/// `(B,H,W)` f32 0/1 plane expected by the frozen encoder. Narrowing removes
/// only the low padding bits from the final byte of each row.
fn unpack_fallout_device(packed: &Tensor, wr: usize, shifts: &Tensor) -> Tensor {
    let size = packed.size();
    debug_assert_eq!(size.len(), 3);
    debug_assert_eq!(size[2] as usize, packed_fallout_row_bytes(wr));
    let full_wr = size[2] * 8;
    packed
        .unsqueeze(-1)
        .bitwise_right_shift(shifts)
        .bitwise_and(1)
        .view([size[0], size[1], full_wr])
        .narrow(2, 0, wr as i64)
        .to_kind(Kind::Float)
}

/// Actor-thread-owned CUDA cache. Tensor is deliberately not Clone and this
/// type is never stored in PreparedObs, so no device handle can enter a
/// rollout or cross into a learner thread.
pub struct TerrainDeviceCache {
    device: Device,
    shared_pair_inputs: bool,
    entries: HashMap<u64, (TerrainCacheKey, Arc<str>, Tensor)>,
    fallout_shifts: Option<Tensor>,
    uploads: u64,
}

impl TerrainDeviceCache {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            shared_pair_inputs: false,
            entries: HashMap::new(),
            fallout_shifts: None,
            uploads: 0,
        }
    }

    /// Enable shared fine/coarse inputs only for a cache whose entire lifetime
    /// is pinned to one persistent actor thread and its thread-local CUDA stream.
    pub fn new_persistent_actor(device: Device) -> Self {
        Self {
            shared_pair_inputs: true,
            ..Self::new(device)
        }
    }

    pub(crate) fn supports_shared_pair_inputs(&self) -> bool {
        self.shared_pair_inputs
    }

    fn fallout_shifts(&mut self) -> Tensor {
        self.fallout_shifts
            .get_or_insert_with(|| {
                // Synchronous by construction: this actor-owned tensor stays
                // in the cache and no temporary host storage can outlive H2D.
                Tensor::from_slice(&[7u8, 6, 5, 4, 3, 2, 1, 0]).to_device(self.device)
            })
            .shallow_clone()
    }

    fn static_tensor(&mut self, terrain: &StaticTerrain) -> Tensor {
        let key = terrain.key;
        let stale = self
            .entries
            .get(&key.env_id)
            .map(|(k, map, _)| *k != key || map.as_ref() != terrain.map.as_ref())
            .unwrap_or(true);
        if stale {
            let cpu = Tensor::from_slice(terrain.land_mag.as_ref()).view([
                2,
                key.hr as i64,
                key.wr as i64,
            ]);
            // The device allocation is retained for the episode. Keep this
            // copy synchronous because the pinned source is temporary.
            let tensor = if self.device.is_cuda() {
                let staged = cpu.pin_memory(self.device);
                staged.to_device_(self.device, Kind::Float, false, false)
            } else {
                cpu.to_device(self.device)
            };
            self.entries
                .insert(key.env_id, (key, Arc::clone(&terrain.map), tensor));
            self.uploads += 1;
        }
        self.entries[&key.env_id].2.shallow_clone()
    }

    #[cfg(test)]
    fn uploads(&self) -> u64 {
        self.uploads
    }
}

/// Batched encode helpers used by `batch::build_obs`. Groups by shape,
/// chunks by pixel budget (mirrors `rl/obs.py::MAX_ENC_PIX`).
pub const MAX_ENC_PIX: usize = 16_000_000;

fn validate_items(items: &[&AeRaw]) -> Result<HashMap<(usize, usize), Vec<usize>>> {
    let mut groups: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (i, it) in items.iter().enumerate() {
        anyhow::ensure!(
            it.hr > 0 && it.wr > 0,
            "AE item {i} shape {}x{} must be non-zero",
            it.hr,
            it.wr
        );
        anyhow::ensure!(
            it.hr % REGION as usize == 0 && it.wr % REGION as usize == 0,
            "AE item {i} shape {}x{} is not divisible by REGION={REGION}",
            it.hr,
            it.wr
        );
        let pix = it.hr * it.wr;
        let fine_gh = it.hr / REGION as usize;
        let fine_gw = it.wr / REGION as usize;
        anyhow::ensure!(
            it.owners.len() == pix,
            "AE item {i} owners len {}, expected {pix}",
            it.owners.len()
        );
        let packed_pix = it.hr * packed_fallout_row_bytes(it.wr);
        anyhow::ensure!(
            it.fallout.len() == packed_pix,
            "AE item {i} packed fallout len {}, expected {packed_pix}",
            it.fallout.len()
        );
        anyhow::ensure!(
            it.stat.len() == NUM_STATIC as usize * fine_gh * fine_gw,
            "AE item {i} stat len {}, expected {}",
            it.stat.len(),
            NUM_STATIC as usize * fine_gh * fine_gw
        );
        anyhow::ensure!(
            it.static_terrain.key.hr == it.hr && it.static_terrain.key.wr == it.wr,
            "AE item {i} static terrain shape {}x{} does not match raw shape {}x{}",
            it.static_terrain.key.hr,
            it.static_terrain.key.wr,
            it.hr,
            it.wr
        );
        anyhow::ensure!(
            it.static_terrain.land_mag.len() == 2 * pix,
            "AE item {i} static terrain len {}, expected {}",
            it.static_terrain.land_mag.len(),
            2 * pix
        );
        if let Some((offset, owner)) = it
            .owners
            .iter()
            .copied()
            .enumerate()
            .find(|&(_, owner)| owner as i64 >= MAX_SLOTS)
        {
            anyhow::bail!(
                "AE item {i} owner[{offset}]={owner} is outside embedding range 0..{MAX_SLOTS}"
            );
        }
        groups.entry((it.hr, it.wr)).or_default().push(i);
    }
    Ok(groups)
}

/// Upload compact owner slots synchronously, then widen on the encoder's
/// device immediately before embedding lookup. The range check belongs at
/// this conversion boundary so malformed u8 payloads cannot reach CUDA's
/// embedding kernel.
fn upload_owner_indices(owners: &[u8], shape: [i64; 3], device: Device) -> Result<Tensor> {
    if let Some((offset, owner)) = owners
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, owner)| owner as i64 >= MAX_SLOTS)
    {
        anyhow::bail!("owner[{offset}]={owner} is outside embedding range 0..{MAX_SLOTS}");
    }
    let packed = Tensor::from_slice(owners).view(shape).to_device(device);
    Ok(packed.to_kind(Kind::Int64))
}

/// Encode to per-item tensors on `device`. Keeping this operation separate
/// from the host fallback lets rollout inference consume the same CUDA
/// latents that the frozen AE produced.
pub fn encode_latent_batch_device(
    ae: &SpatialAE,
    items: &[&AeRaw],
    device: Device,
    mut terrain_cache: Option<&mut TerrainDeviceCache>,
) -> Result<Vec<Tensor>> {
    // Returns one (latent_c, gh, gw) tensor per item, in input order.
    let n = items.len();
    let mut out: Vec<Option<Tensor>> = (0..n).map(|_| None).collect();
    let groups = validate_items(items)?;

    for ((hr, wr), idxs) in groups {
        let pix = hr * wr;
        let packed_wr = packed_fallout_row_bytes(wr);
        let per = (MAX_ENC_PIX / pix.max(1)).max(1);
        for chunk in idxs.chunks(per) {
            let b = chunk.len() as i64;
            let use_terrain_cache = terrain_cache.is_some() && device.is_cuda();
            let mut owners = Vec::with_capacity(chunk.len() * pix);
            let mut terrain = Vec::with_capacity(if use_terrain_cache {
                0
            } else {
                chunk.len() * 3 * pix
            });
            let mut fallout = Vec::with_capacity(if use_terrain_cache {
                chunk.len() * hr * packed_wr
            } else {
                0
            });
            let fine_gh = hr / REGION as usize;
            let fine_gw = wr / REGION as usize;
            let (gh, gw) = if ae.latent_down == REGION {
                (fine_gh, fine_gw)
            } else {
                (fine_gh.div_ceil(2), fine_gw.div_ceil(2))
            };
            // static must be at latent resolution. Fine AE uses /8 = REGION
            // which matches `stat` from featurize. Coarse AE needs /16:
            // max-pool the /8 static 2x (ceil), matching encode_grids.
            let mut static_p = Vec::with_capacity(chunk.len() * NUM_STATIC as usize * gh * gw);
            for &i in chunk {
                let it = items[i];
                owners.extend_from_slice(&it.owners);
                if !use_terrain_cache {
                    terrain.extend_from_slice(it.static_terrain.land_mag.as_ref());
                    terrain.extend(unpack_fallout_host(&it.fallout, hr, wr));
                } else {
                    fallout.extend_from_slice(&it.fallout);
                }
                if ae.latent_down == REGION {
                    static_p.extend_from_slice(&it.stat);
                } else {
                    static_p.extend(max_pool2_stat(&it.stat, fine_gh, fine_gw));
                }
            }
            // One synchronous H2D of packed u8, followed by a device-local
            // widening conversion. No temporary host buffer is used by a
            // nonblocking copy.
            let owners_t = upload_owner_indices(&owners, [b, hr as i64, wr as i64], device)?;
            let terrain_t = if use_terrain_cache {
                let cache = terrain_cache.as_deref_mut().expect("cache checked above");
                let shifts = cache.fallout_shifts();
                let static_items: Vec<Tensor> = chunk
                    .iter()
                    .map(|&i| cache.static_tensor(&items[i].static_terrain))
                    .collect();
                let static_refs: Vec<&Tensor> = static_items.iter().collect();
                let static_t = Tensor::stack(&static_refs, 0);
                // One synchronous packed-u8 H2D. Expansion and f32 conversion
                // happen on the persistent actor device.
                let packed_t = Tensor::from_slice(&fallout)
                    .view([b, hr as i64, packed_wr as i64])
                    .to_device(device);
                let fallout_t = unpack_fallout_device(&packed_t, wr, &shifts).unsqueeze(1);
                Tensor::cat(&[static_t, fallout_t], 1)
            } else {
                Tensor::from_slice(&terrain)
                    .view([b, 3, hr as i64, wr as i64])
                    .to_device(device)
                    .to_kind(Kind::Float)
            };
            let static_t = Tensor::from_slice(&static_p)
                .view([b, NUM_STATIC, gh as i64, gw as i64])
                .to_device(device)
                .to_kind(Kind::Float);
            let z = tch::no_grad(|| ae.encode(&owners_t, &terrain_t, &static_t));
            anyhow::ensure!(
                z.size() == [b, ae.latent_c, gh as i64, gw as i64],
                "AE output shape {:?}, expected [{b}, {}, {gh}, {gw}]",
                z.size(),
                ae.latent_c
            );
            // Split batch without changing device. `select` views retain the
            // batch storage after `z` leaves this scope.
            for (j, &i) in chunk.iter().enumerate() {
                out[i] = Some(z.select(0, j as i64));
            }
        }
    }

    out.into_iter()
        .enumerate()
        .map(|(i, t)| t.with_context(|| format!("missing AE latent for item {i}")))
        .collect()
}

/// Per-item fine and coarse latent views produced from shared device inputs.
#[derive(Debug)]
pub struct DualLatents {
    pub fine: Vec<Tensor>,
    pub coarse: Vec<Tensor>,
}

/// Paired actor encode using one input upload per same-shape chunk.
///
/// The AE pair, cache, packed inputs, temporary tensors, and returned latent
/// views are all actor-thread-owned. On CUDA, libtorch operations are enqueued
/// on that actor thread's current stream; none of these tensors may cross the
/// actor/learner boundary. The caller must compact or synchronously copy the
/// final rollout payload to host before returning it from the actor.
pub fn encode_dual_latent_batch_device(
    pair: &AePair,
    items: &[&AeRaw],
    device: Device,
    terrain_cache: &mut TerrainDeviceCache,
) -> Result<DualLatents> {
    let coarse = pair
        .coarse
        .as_ref()
        .context("dual AE encode requires a coarse encoder")?;
    anyhow::ensure!(
        pair.fine.latent_down == REGION,
        "dual AE fine latent_down {}, expected {REGION}",
        pair.fine.latent_down
    );
    anyhow::ensure!(
        coarse.latent_down == COARSE_REGION,
        "dual AE coarse latent_down {}, expected {COARSE_REGION}",
        coarse.latent_down
    );
    anyhow::ensure!(
        terrain_cache.device == device,
        "terrain cache device {:?} does not match encode device {:?}",
        terrain_cache.device,
        device
    );
    anyhow::ensure!(
        terrain_cache.supports_shared_pair_inputs(),
        "dual AE encode requires persistent actor cache ownership"
    );

    let groups = validate_items(items)?;
    let mut fine_out: Vec<Option<Tensor>> = (0..items.len()).map(|_| None).collect();
    let mut coarse_out: Vec<Option<Tensor>> = (0..items.len()).map(|_| None).collect();

    for ((hr, wr), idxs) in groups {
        let pix = hr * wr;
        let packed_wr = packed_fallout_row_bytes(wr);
        let per = (MAX_ENC_PIX / pix).max(1);
        for chunk in idxs.chunks(per) {
            let b = chunk.len() as i64;
            let use_terrain_cache = device.is_cuda();
            let mut owners = Vec::with_capacity(chunk.len() * pix);
            let mut terrain = Vec::with_capacity(if use_terrain_cache {
                0
            } else {
                chunk.len() * TERRAIN_CHANNELS as usize * pix
            });
            let mut fallout = Vec::with_capacity(if use_terrain_cache {
                chunk.len() * hr * packed_wr
            } else {
                0
            });
            let fine_gh = hr / REGION as usize;
            let fine_gw = wr / REGION as usize;
            let mut fine_static =
                Vec::with_capacity(chunk.len() * NUM_STATIC as usize * fine_gh * fine_gw);
            for &i in chunk {
                let item = items[i];
                owners.extend_from_slice(&item.owners);
                if use_terrain_cache {
                    fallout.extend_from_slice(&item.fallout);
                } else {
                    terrain.extend_from_slice(item.static_terrain.land_mag.as_ref());
                    terrain.extend(unpack_fallout_host(&item.fallout, hr, wr));
                }
                fine_static.extend_from_slice(&item.stat);
            }

            // Every host-backed H2D above/below is synchronous. In particular,
            // no nonblocking copy may outlive these temporary packed vectors.
            // Owners cross as u8 and widen to int64 only on the target device.
            let owners_t = upload_owner_indices(&owners, [b, hr as i64, wr as i64], device)?;
            let terrain_t = if use_terrain_cache {
                let shifts = terrain_cache.fallout_shifts();
                let static_items: Vec<Tensor> = chunk
                    .iter()
                    .map(|&i| terrain_cache.static_tensor(&items[i].static_terrain))
                    .collect();
                let static_refs: Vec<&Tensor> = static_items.iter().collect();
                let static_t = Tensor::stack(&static_refs, 0);
                let packed_t = Tensor::from_slice(&fallout)
                    .view([b, hr as i64, packed_wr as i64])
                    .to_device(device);
                let fallout_t = unpack_fallout_device(&packed_t, wr, &shifts).unsqueeze(1);
                Tensor::cat(&[static_t, fallout_t], 1)
            } else {
                Tensor::from_slice(&terrain)
                    .view([b, TERRAIN_CHANNELS, hr as i64, wr as i64])
                    .to_device(device)
                    .to_kind(Kind::Float)
            };
            let fine_static_t = Tensor::from_slice(&fine_static)
                .view([b, NUM_STATIC, fine_gh as i64, fine_gw as i64])
                .to_device(device)
                .to_kind(Kind::Float);

            let (fine_z, coarse_z) = tch::no_grad(|| {
                let fine_z = pair.fine.encode(&owners_t, &terrain_t, &fine_static_t);
                // This exactly matches F.max_pool2d(kernel=2, stride=2,
                // ceil_mode=True): odd fine edges are retained, not dropped.
                let coarse_static_t =
                    fine_static_t.max_pool2d([2, 2], [2, 2], [0, 0], [1, 1], true);
                let coarse_z = coarse.encode(&owners_t, &terrain_t, &coarse_static_t);
                (fine_z, coarse_z)
            });
            let coarse_gh = fine_gh.div_ceil(2);
            let coarse_gw = fine_gw.div_ceil(2);
            anyhow::ensure!(
                fine_z.size() == [b, pair.fine.latent_c, fine_gh as i64, fine_gw as i64],
                "fine AE output shape {:?}, expected [{b}, {}, {fine_gh}, {fine_gw}]",
                fine_z.size(),
                pair.fine.latent_c
            );
            anyhow::ensure!(
                coarse_z.size() == [b, coarse.latent_c, coarse_gh as i64, coarse_gw as i64],
                "coarse AE output shape {:?}, expected [{b}, {}, {coarse_gh}, {coarse_gw}]",
                coarse_z.size(),
                coarse.latent_c
            );
            for (j, &i) in chunk.iter().enumerate() {
                fine_out[i] = Some(fine_z.select(0, j as i64));
                coarse_out[i] = Some(coarse_z.select(0, j as i64));
            }
        }
    }

    let collect = |name: &str, values: Vec<Option<Tensor>>| {
        values
            .into_iter()
            .enumerate()
            .map(|(i, value)| {
                value.with_context(|| format!("missing {name} AE latent for item {i}"))
            })
            .collect::<Result<Vec<_>>>()
    };
    Ok(DualLatents {
        fine: collect("fine", fine_out)?,
        coarse: collect("coarse", coarse_out)?,
    })
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
    use std::sync::Arc;

    fn terrain(env_id: u64, static_id: u64, hr: usize, wr: usize, fill: f32) -> StaticTerrain {
        StaticTerrain {
            key: TerrainCacheKey {
                env_id,
                episode: static_id,
                static_id,
                hr,
                wr,
            },
            map: Arc::from(format!("map-{static_id}")),
            land_mag: vec![fill; 2 * hr * wr].into(),
        }
    }

    fn patterned_raw(env_id: u64, hr: usize, wr: usize) -> AeRaw {
        let pix = hr * wr;
        let gh = hr / REGION as usize;
        let gw = wr / REGION as usize;
        let fallout: Vec<u8> = (0..pix)
            .map(|i| ((i + env_id as usize) % 7 == 0) as u8)
            .collect();
        AeRaw {
            owners: (0..pix)
                .map(|i| ((i * 31 + env_id as usize * 17) % MAX_SLOTS as usize) as u8)
                .collect(),
            static_terrain: terrain(env_id, env_id + 10, hr, wr, env_id as f32 / 19.0),
            fallout: pack_fallout(&fallout, hr, wr),
            stat: (0..NUM_STATIC as usize * gh * gw)
                .map(|i| ((i * 13 + env_id as usize * 5) % 37) as f32 / 11.0)
                .collect(),
            hr,
            wr,
        }
    }

    fn i64_reference(ae: &SpatialAE, raw: &AeRaw) -> Tensor {
        let owners_i64: Vec<i64> = raw.owners.iter().map(|&owner| owner as i64).collect();
        let owners = Tensor::from_slice(&owners_i64).view([1, raw.hr as i64, raw.wr as i64]);
        let mut terrain = raw.static_terrain.land_mag.to_vec();
        terrain.extend(unpack_fallout_host(&raw.fallout, raw.hr, raw.wr));
        let terrain =
            Tensor::from_slice(&terrain).view([1, TERRAIN_CHANNELS, raw.hr as i64, raw.wr as i64]);
        let fine_gh = raw.hr / REGION as usize;
        let fine_gw = raw.wr / REGION as usize;
        let (stat, gh, gw) = if ae.latent_down == REGION {
            (raw.stat.clone(), fine_gh, fine_gw)
        } else {
            (
                max_pool2_stat(&raw.stat, fine_gh, fine_gw),
                fine_gh.div_ceil(2),
                fine_gw.div_ceil(2),
            )
        };
        let stat = Tensor::from_slice(&stat).view([1, NUM_STATIC, gh as i64, gw as i64]);
        tch::no_grad(|| ae.encode(&owners, &terrain, &stat).select(0, 0))
    }

    fn flat_f32(tensor: &Tensor) -> Vec<f32> {
        Vec::<f32>::try_from(tensor.reshape([-1])).unwrap()
    }

    fn random_pair() -> AePair {
        tch::manual_seed(73);
        let fine_vs = nn::VarStore::new(Device::Cpu);
        let fine = SpatialAE::new(&fine_vs.root(), LATENT_C, REGION).unwrap();
        let coarse_vs = nn::VarStore::new(Device::Cpu);
        let coarse = SpatialAE::new(&coarse_vs.root(), LATENT_C, COARSE_REGION).unwrap();
        AePair {
            fine,
            _fine_vs: fine_vs,
            coarse: Some(coarse),
            _coarse_vs: Some(coarse_vs),
        }
    }

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
    fn invalid_owner_is_rejected_at_u8_to_i64_boundary() {
        let err = upload_owner_indices(&[0, 127, 128], [1, 1, 3], Device::Cpu)
            .expect_err("out-of-range packed owner must fail");
        assert!(err.to_string().contains("owner[2]=128"));

        let vs = nn::VarStore::new(Device::Cpu);
        let ae = SpatialAE::new(&vs.root(), LATENT_C, REGION).unwrap();
        let mut raw = AeRaw {
            owners: vec![0; 8 * 8],
            static_terrain: terrain(1, 1, 8, 8, 0.0),
            fallout: pack_fallout(&vec![0; 8 * 8], 8, 8),
            stat: vec![0.0; NUM_STATIC as usize],
            hr: 8,
            wr: 8,
        };
        raw.owners[17] = MAX_SLOTS as u8;
        let err = encode_latent_batch_device(&ae, &[&raw], Device::Cpu, None)
            .expect_err("out-of-range owner must fail");
        assert!(err.to_string().contains("owner[17]"));
    }

    #[test]
    fn packed_owner_embedding_and_latents_exactly_match_i64_reference() {
        tch::manual_seed(97);
        let fine_vs = nn::VarStore::new(Device::Cpu);
        let fine = SpatialAE::new(&fine_vs.root(), LATENT_C, REGION).unwrap();
        let coarse_vs = nn::VarStore::new(Device::Cpu);
        let coarse = SpatialAE::new(&coarse_vs.root(), LATENT_C, COARSE_REGION).unwrap();

        let every_slot: Vec<u8> = (0..MAX_SLOTS as u8).collect();
        let packed = upload_owner_indices(&every_slot, [1, 8, 16], Device::Cpu).unwrap();
        let reference = Tensor::from_slice(&(0..MAX_SLOTS).collect::<Vec<_>>()).view([1, 8, 16]);
        assert_eq!(
            flat_f32(&fine.owner_emb.forward(&packed)),
            flat_f32(&fine.owner_emb.forward(&reference)),
            "embedding output changed"
        );

        // Deliberately non-sorted mixed shapes prove grouped results restore
        // input order; 24x40 also exercises odd 3x5 fine dimensions.
        let raws = [
            patterned_raw(1, 24, 40),
            patterned_raw(2, 16, 32),
            patterned_raw(3, 32, 24),
        ];
        let refs: Vec<&AeRaw> = raws.iter().collect();
        for ae in [&fine, &coarse] {
            let actual = encode_latent_batch_device(ae, &refs, Device::Cpu, None).unwrap();
            assert_eq!(actual.len(), raws.len());
            for (i, raw) in raws.iter().enumerate() {
                let expected = i64_reference(ae, raw);
                assert_eq!(actual[i].size(), expected.size(), "shape/order item {i}");
                assert_eq!(
                    flat_f32(&actual[i]),
                    flat_f32(&expected),
                    "latent output changed for item {i}, down={}",
                    ae.latent_down
                );
            }
        }
    }

    #[test]
    fn ae_raw_owner_payload_uses_one_eighth_i64_reference_bytes() {
        let raw = patterned_raw(11, 24, 40);
        let packed_bytes = std::mem::size_of_val(raw.owners.as_slice());
        let i64_reference_bytes = raw.owners.len() * std::mem::size_of::<i64>();
        assert_eq!(packed_bytes, raw.hr * raw.wr);
        assert_eq!(packed_bytes * 8, i64_reference_bytes);
    }

    #[test]
    fn fallout_packbits_exhaustively_matches_msb_first_order() {
        let shifts = Tensor::from_slice(&[7u8, 6, 5, 4, 3, 2, 1, 0]);
        for byte in 0u16..=255 {
            let pixels: Vec<u8> = (0..8)
                .map(|x| ((byte as u8 >> (7 - x)) & 1) as u8)
                .collect();
            let packed = pack_fallout(&pixels, 1, 8);
            assert_eq!(packed, [byte as u8], "pack byte {byte:#04x}");
            assert_eq!(
                unpack_fallout_host(&packed, 1, 8),
                pixels.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                "host unpack byte {byte:#04x}"
            );
            let device = unpack_fallout_device(
                &Tensor::from_slice(&packed).view([1, 1, 1]),
                8,
                &shifts,
            );
            assert_eq!(
                flat_f32(&device),
                pixels.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                "device unpack byte {byte:#04x}"
            );
        }
    }

    #[test]
    fn fallout_packbits_resets_rows_and_zero_pads_odd_widths() {
        let pixels = [
            1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, // a1 a0
            0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, // 5e 40
            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, // ff 00
        ];
        let packed = pack_fallout(&pixels, 3, 11);
        assert_eq!(packed, [0xa1, 0xa0, 0x5e, 0x40, 0xff, 0x00]);
        assert_eq!(packed.len(), 3 * 11usize.div_ceil(8));
        assert!(
            packed.chunks_exact(2).all(|row| row[1] & 0x1f == 0),
            "low five row-padding bits must be zero"
        );

        let shifts = Tensor::from_slice(&[7u8, 6, 5, 4, 3, 2, 1, 0]);
        let unpacked = unpack_fallout_device(
            &Tensor::from_slice(&packed).view([1, 3, 2]),
            11,
            &shifts,
        );
        assert_eq!(
            flat_f32(&unpacked),
            pixels.iter().map(|&v| v as f32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn packed_fallout_host_bytes_are_one_bit_per_pixel_plus_row_padding() {
        for (hr, wr) in [(1, 1), (3, 11), (7, 16), (24, 40), (32, 24)] {
            let raw = vec![1u8; hr * wr];
            let packed = pack_fallout(&raw, hr, wr);
            let packed_bytes = std::mem::size_of_val(packed.as_slice());
            let old_f32_bytes = hr * wr * std::mem::size_of::<f32>();
            assert_eq!(packed_bytes, hr * wr.div_ceil(8), "{hr}x{wr}");
            assert!(packed_bytes < old_f32_bytes, "{hr}x{wr}");
            if wr % 8 == 0 {
                assert_eq!(packed_bytes * 32, old_f32_bytes);
            }
        }
    }

    #[test]
    fn coarse_encode_supports_odd_fine_dimensions() {
        let vs = nn::VarStore::new(Device::Cpu);
        let ae = SpatialAE::new(&vs.root(), LATENT_C, 16).unwrap();
        let raw = AeRaw {
            owners: vec![0; 24 * 40],
            static_terrain: terrain(1, 1, 24, 40, 0.0),
            fallout: pack_fallout(&vec![0; 24 * 40], 24, 40),
            stat: vec![0.0; NUM_STATIC as usize * 3 * 5],
            hr: 24,
            wr: 40,
        };
        let z = encode_latent_batch_device(&ae, &[&raw], Device::Cpu, None).unwrap();
        assert_eq!(z[0].size(), [LATENT_C, 2, 3]);
    }

    #[test]
    fn dual_encode_exactly_matches_separate_for_even_odd_and_mixed_order() {
        let pair = random_pair();
        // Fine dimensions are odd 3x5, even 2x4, then 4x3. The repeated
        // first shape at the end verifies shape grouping restores input order.
        let raws = [
            patterned_raw(1, 24, 40),
            patterned_raw(2, 16, 32),
            patterned_raw(3, 32, 24),
            patterned_raw(4, 24, 40),
        ];
        let refs: Vec<&AeRaw> = raws.iter().collect();
        let separate_fine =
            encode_latent_batch_device(&pair.fine, &refs, Device::Cpu, None).unwrap();
        let separate_coarse =
            encode_latent_batch_device(pair.coarse.as_ref().unwrap(), &refs, Device::Cpu, None)
                .unwrap();
        let mut cache = TerrainDeviceCache::new_persistent_actor(Device::Cpu);
        let shared =
            encode_dual_latent_batch_device(&pair, &refs, Device::Cpu, &mut cache).unwrap();

        for i in 0..refs.len() {
            assert_eq!(
                shared.fine[i].size(),
                separate_fine[i].size(),
                "fine shape/order item {i}"
            );
            assert_eq!(
                shared.coarse[i].size(),
                separate_coarse[i].size(),
                "coarse shape/order item {i}"
            );
            assert_eq!(
                flat_f32(&shared.fine[i]),
                flat_f32(&separate_fine[i]),
                "fine values item {i}"
            );
            assert_eq!(
                flat_f32(&shared.coarse[i]),
                flat_f32(&separate_coarse[i]),
                "coarse values item {i}"
            );
        }
    }

    #[test]
    fn dual_encode_validates_owner_and_static_payload_before_encoding() {
        let pair = random_pair();
        let mut cache = TerrainDeviceCache::new_persistent_actor(Device::Cpu);

        let mut invalid_owner = patterned_raw(5, 16, 16);
        invalid_owner.owners[37] = MAX_SLOTS as u8;
        let err =
            encode_dual_latent_batch_device(&pair, &[&invalid_owner], Device::Cpu, &mut cache)
                .expect_err("invalid packed owner must fail");
        assert!(err.to_string().contains("owner[37]=128"));

        let mut invalid_static = patterned_raw(6, 16, 16);
        invalid_static.static_terrain.land_mag = vec![0.0; 2 * 16 * 16 - 1].into();
        let err =
            encode_dual_latent_batch_device(&pair, &[&invalid_static], Device::Cpu, &mut cache)
                .expect_err("short static terrain must fail");
        assert!(err.to_string().contains("static terrain len"));

        let mut mismatched_shape = patterned_raw(7, 16, 16);
        mismatched_shape.static_terrain.key.wr = 24;
        let err =
            encode_dual_latent_batch_device(&pair, &[&mismatched_shape], Device::Cpu, &mut cache)
                .expect_err("mismatched static terrain shape must fail");
        assert!(err.to_string().contains("does not match raw shape"));
    }

    #[test]
    fn dual_encode_rejects_non_persistent_cache_ownership() {
        let pair = random_pair();
        let raw = patterned_raw(8, 16, 16);
        let mut legacy_cache = TerrainDeviceCache::new(Device::Cpu);
        let err = encode_dual_latent_batch_device(&pair, &[&raw], Device::Cpu, &mut legacy_cache)
            .expect_err("legacy/non-persistent cache must retain separate encoding");
        assert!(err.to_string().contains("persistent actor cache ownership"));
    }

    #[test]
    fn load_exported_d8_weights_and_encode() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../weights/ae/ae_v31_d8c32.encoder.safetensors");
        if !path.exists() {
            eprintln!(
                "skip: {} missing (run scripts/fetch_ae_encoders.sh)",
                path.display()
            );
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

    #[test]
    fn static_terrain_arc_is_reused_and_float_packing_is_exact() {
        let land = [0, 1, 2, 0];
        let mag = [0, 1, 30, 31];
        let packed = pack_static_terrain(&land, &mag, 2, 2);
        let shared = StaticTerrain {
            key: TerrainCacheKey {
                env_id: 3,
                episode: 7,
                static_id: 13,
                hr: 2,
                wr: 2,
            },
            map: Arc::from("arc-test"),
            land_mag: Arc::clone(&packed),
        };
        let prepared_clone = shared.clone();
        assert!(Arc::ptr_eq(&shared.land_mag, &prepared_clone.land_mag));
        assert_eq!(
            packed.as_ref(),
            &[0.0, 1.0, 1.0, 0.0, 0.0, 1.0 / 31.0, 30.0 / 31.0, 1.0]
        );
    }

    #[test]
    fn cached_terrain_matches_uncached_and_fallout_stays_fresh() {
        let static_terrain = terrain(1, 1, 2, 3, 0.25);
        let mut cache = TerrainDeviceCache::new(Device::Cpu);
        let cached_static = cache.static_tensor(&static_terrain);
        let fallout_a = Tensor::from_slice(&[0.0f32, 1.0, 0.0, 1.0, 0.0, 1.0]).view([1, 1, 2, 3]);
        let fallout_b = Tensor::from_slice(&[1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0]).view([1, 1, 2, 3]);
        let cached_a = Tensor::cat(&[cached_static.unsqueeze(0), fallout_a], 1);
        let cached_b = Tensor::cat(
            &[cache.static_tensor(&static_terrain).unsqueeze(0), fallout_b],
            1,
        );
        let mut expected = static_terrain.land_mag.to_vec();
        expected.extend([0.0f32, 1.0, 0.0, 1.0, 0.0, 1.0]);
        assert_eq!(
            Vec::<f32>::try_from(cached_a.reshape([-1])).unwrap(),
            expected
        );
        let a = Vec::<f32>::try_from(cached_a.reshape([-1])).unwrap();
        let b = Vec::<f32>::try_from(cached_b.reshape([-1])).unwrap();
        assert_eq!(&a[..12], &b[..12], "static channels changed");
        assert_ne!(&a[12..], &b[12..], "fallout was stale");
        assert_eq!(cache.uploads(), 1, "cache hit reuploaded static terrain");
    }

    #[test]
    fn cached_device_unpacked_and_legacy_host_unpacked_latents_are_exact() {
        tch::manual_seed(41);
        let vs = nn::VarStore::new(Device::Cpu);
        let ae = SpatialAE::new(&vs.root(), LATENT_C, REGION).unwrap();
        let static_terrain = terrain(2, 5, 8, 8, 0.375);
        let fallout_bits: Vec<u8> = (0..64).map(|i| ((i * 5 + 3) % 11 < 4) as u8).collect();
        let packed = pack_fallout(&fallout_bits, 8, 8);
        let fallout = unpack_fallout_host(&packed, 8, 8);
        let mut uncached = static_terrain.land_mag.to_vec();
        uncached.extend_from_slice(&fallout);
        let uncached = Tensor::from_slice(&uncached).view([1, 3, 8, 8]);

        let mut cache = TerrainDeviceCache::new(Device::Cpu);
        let shifts = cache.fallout_shifts();
        let device_fallout = unpack_fallout_device(
            &Tensor::from_slice(&packed).view([1, 8, 1]),
            8,
            &shifts,
        );
        let cached = Tensor::cat(
            &[
                cache.static_tensor(&static_terrain).unsqueeze(0),
                device_fallout.unsqueeze(1),
            ],
            1,
        );
        assert_eq!(
            Vec::<f32>::try_from(uncached.reshape([-1])).unwrap(),
            Vec::<f32>::try_from(cached.reshape([-1])).unwrap(),
            "packed device expansion changed the exact f32 encoder input"
        );
        let owners = Tensor::zeros([1, 8, 8], (Kind::Int64, Device::Cpu));
        let stat = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0]).view([1, 6, 1, 1]);
        let expected = tch::no_grad(|| ae.encode(&owners, &uncached, &stat));
        let actual = tch::no_grad(|| ae.encode(&owners, &cached, &stat));
        assert_eq!(
            Vec::<f32>::try_from(expected.reshape([-1])).unwrap(),
            Vec::<f32>::try_from(actual.reshape([-1])).unwrap()
        );
    }

    #[test]
    fn cache_invalidates_on_episode_map_identity_or_dimensions() {
        let mut cache = TerrainDeviceCache::new(Device::Cpu);
        let first = terrain(4, 1, 2, 2, 1.0);
        let same = first.clone();
        let mut other_map = first.clone();
        other_map.map = Arc::from("different-map");
        let reset = terrain(4, 2, 2, 3, 2.0);
        let other_shape = terrain(5, 1, 3, 1, 3.0);

        assert_eq!(cache.static_tensor(&first).size(), [2, 2, 2]);
        let _ = cache.static_tensor(&same);
        assert_eq!(cache.uploads(), 1);
        let _ = cache.static_tensor(&other_map);
        assert_eq!(cache.uploads(), 2);
        assert_eq!(cache.static_tensor(&reset).size(), [2, 2, 3]);
        assert_eq!(cache.uploads(), 3);
        assert_eq!(cache.static_tensor(&other_shape).size(), [2, 3, 1]);
        assert_eq!(cache.uploads(), 4);
        assert_eq!(cache.entries.len(), 2, "one current entry per env");
        let reset_values = Vec::<f32>::try_from(cache.static_tensor(&reset).reshape([-1])).unwrap();
        assert!(reset_values.iter().all(|&v| v == 2.0));
    }

    #[test]
    fn dynamic_stat_plane_is_never_part_of_terrain_cache() {
        let terrain = terrain(9, 1, 8, 8, 0.0);
        let raw_a = AeRaw {
            owners: vec![0; 64],
            static_terrain: terrain.clone(),
            fallout: pack_fallout(&vec![0; 64], 8, 8),
            stat: vec![0.0; 6],
            hr: 8,
            wr: 8,
        };
        let mut raw_b = raw_a.clone();
        raw_b.stat[3] = 7.0;
        raw_b.fallout = pack_fallout(
            &(0..64).map(|i| (i == 5) as u8).collect::<Vec<_>>(),
            8,
            8,
        );
        assert!(Arc::ptr_eq(
            &raw_a.static_terrain.land_mag,
            &raw_b.static_terrain.land_mag
        ));
        assert_ne!(raw_a.stat, raw_b.stat, "structure stats were stale");
        assert_ne!(raw_a.fallout, raw_b.fallout, "fallout was stale");
    }
}
