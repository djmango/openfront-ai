//! Frozen SpatialAE encoder: tch port of ofae / former Python SpatialAE.encode.
//!
//! PPO only needs the encoder (owner embedding + stem + fuse). Weights are
//! loaded from the safetensors files produced by
//! `ofae` / HF encoder safetensors (keys match tch's
//! '.' path separator exactly). Decoder weights are intentionally absent.
//!
//! v3.2 (no-static): encoder inputs are owners + terrain(+fallout) only.
//! Structure planes stay on `AeRaw.stat` and are concatenated into the policy
//! grid as an exact bypass (`C_GRID` = latent + 6 static + ego + db + transient).
//!
//! Production checkpoints:
//! - fine:  `ae_v32_nostatic_d8c32`  — 32ch @ 1/8
//! - coarse: `ae_v32_nostatic_d16c32` — 32ch @ 1/16 (optional coarse stream)

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

/// Python enables AE bf16 only inside CUDA autocast on the long-lived
/// rollout owner. The legacy collector may move CUDA state across threads,
/// so `--amp` must not alter its AE path.
fn use_bf16_ae(amp: bool, persistent_actor: bool, device: Device) -> bool {
    amp && persistent_actor && device.is_cuda()
}

/// Frozen bf16 parameters corresponding to one f32 VarStore convolution.
///
/// Python autocast keeps the equivalent cast in its weight cache. These
/// tensors are created once, after checkpoint loading, on the persistent
/// actor thread and are never moved or refreshed.
struct CachedBf16Conv {
    ws: Tensor,
    bs: Option<Tensor>,
}

impl CachedBf16Conv {
    fn new(conv: &nn::Conv2D) -> Self {
        Self {
            ws: conv.ws.to_kind(Kind::BFloat16),
            bs: conv.bs.as_ref().map(|b| b.to_kind(Kind::BFloat16)),
        }
    }

    fn forward(&self, x: &Tensor, stride: [i64; 2], padding: [i64; 2]) -> Tensor {
        x.conv2d(&self.ws, self.bs.as_ref(), stride, padding, [1i64, 1], 1)
    }
}

struct ConvGn {
    conv: nn::Conv2D,
    gn: nn::GroupNorm,
    stride: i64,
    bf16: Option<CachedBf16Conv>,
}

impl ConvGn {
    fn new(vs: &nn::Path, c_in: i64, c_out: i64, stride: i64) -> Self {
        let mut cfg = nn::ConvConfig::default();
        cfg.stride = stride;
        cfg.padding = 1;
        Self {
            conv: nn::conv2d(vs / 0, c_in, c_out, 3, cfg),
            gn: nn::group_norm(vs / 1, 8, c_out, Default::default()),
            stride,
            bf16: None,
        }
    }

    fn cache_bf16(&mut self) {
        if self.bf16.is_none() {
            self.bf16 = Some(CachedBf16Conv::new(&self.conv));
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        if let Some(conv) = &self.bf16 {
            // Autocast uses reduced precision for the convolution, not for
            // GroupNorm. Round-trip each block so GN and silu remain f32.
            let xb = xs.to_kind(Kind::BFloat16);
            let conv = conv
                .forward(&xb, [self.stride, self.stride], [1, 1])
                .to_kind(Kind::Float);
            self.gn.forward(&conv).silu()
        } else {
            self.gn.forward(&self.conv.forward(xs)).silu()
        }
    }
}

/// Encoder-only SpatialAE. `VarStore` must contain exactly the encoder
/// tensors (no decoder) so `vs.load(safetensors)` succeeds strictly.
pub struct SpatialAE {
    owner_emb: nn::Embedding,
    enc_stem: Vec<ConvGn>,
    enc_fuse: ConvGn,
    enc_out: nn::Conv2D,
    enc_out_bf16: Option<CachedBf16Conv>,
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

        let enc_fuse = ConvGn::new(&(vs / "enc_fuse" / 0), 128, 128, 1);
        let mut out_cfg = nn::ConvConfig::default();
        out_cfg.stride = 1;
        out_cfg.padding = 0;
        let enc_out = nn::conv2d(vs / "enc_fuse" / 1, 128, latent_c, 1, out_cfg);

        Ok(Self {
            owner_emb,
            enc_stem,
            enc_fuse,
            enc_out,
            enc_out_bf16: None,
            latent_c,
            latent_down,
        })
    }

    /// Construct on `device`, load encoder weights from a `.safetensors`
    /// file produced by `ofae` (`.encoder.safetensors`), then freeze.
    pub fn load(
        path: impl AsRef<Path>,
        device: Device,
        expected_down: i64,
        amp: bool,
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
        let mut ae = Self::new(&vs.root(), latent_c, latent_down)?;
        vs.load(path)
            .with_context(|| format!("load AE encoder weights from {}", path.display()))?;
        vs.freeze();
        if amp {
            // Build only after the strict checkpoint load: caching constructor
            // values would leave stale random bf16 parameters.
            ae.cache_bf16();
        }
        Ok((vs, ae))
    }

    fn cache_bf16(&mut self) {
        // no_grad makes the frozen-inference intent explicit even for tests
        // that construct an AE without first freezing its VarStore.
        tch::no_grad(|| {
            for block in &mut self.enc_stem {
                block.cache_bf16();
            }
            self.enc_fuse.cache_bf16();
            if self.enc_out_bf16.is_none() {
                self.enc_out_bf16 = Some(CachedBf16Conv::new(&self.enc_out));
            }
            // Amp encode uses only the bf16 conv caches (+ f32 GroupNorm /
            // embedding). Move the unused f32 conv storages to CPU so the
            // actor does not pay a second full copy of encoder weights in
            // VRAM alongside the learner's policy.
            self.reclaim_cached_f32_convs_to_cpu();
        });
    }

    /// After [`Self::cache_bf16`], f32 `Conv2D` weights are dead on the
    /// amp path. `set_data` rewrites the shared VarStore storage in place
    /// so all handles (module fields + `VarStore`) observe the CPU move.
    fn reclaim_cached_f32_convs_to_cpu(&mut self) {
        let reclaim = |t: &mut Tensor| {
            if t.device().is_cuda() {
                let cpu = t.to_device(Device::Cpu);
                t.set_data(&cpu);
            }
        };
        let reclaim_conv = |conv: &mut nn::Conv2D| {
            reclaim(&mut conv.ws);
            if let Some(bs) = conv.bs.as_mut() {
                reclaim(bs);
            }
        };
        for block in &mut self.enc_stem {
            if block.bf16.is_some() {
                reclaim_conv(&mut block.conv);
            }
        }
        if self.enc_fuse.bf16.is_some() {
            reclaim_conv(&mut self.enc_fuse.conv);
        }
        if self.enc_out_bf16.is_some() {
            reclaim_conv(&mut self.enc_out);
        }
    }

    /// `owners` (B,H,W) int64, `terrain` (B,3,H,W) f32 → `z`
    /// (B,latent_c,H/down,W/down) f32. Static structures are not an AE input.
    pub fn encode(&self, owners: &Tensor, terrain: &Tensor) -> Tensor {
        // Embedding and concatenated activations remain f32. Owners stay
        // int64 as required by embedding lookup.
        let emb = self.owner_emb.forward(owners).permute([0, 3, 1, 2]);
        let mut g = Tensor::cat(&[&emb, terrain], 1);
        for block in &self.enc_stem {
            g = block.forward(&g);
        }
        g = self.enc_fuse.forward(&g);
        if let Some(conv) = &self.enc_out_bf16 {
            conv.forward(&g.to_kind(Kind::BFloat16), [1, 1], [0, 0])
                .to_kind(Kind::Float)
        } else {
            self.enc_out.forward(&g)
        }
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
        amp: bool,
        persistent_actor: bool,
    ) -> Result<Self> {
        let ae_amp = use_bf16_ae(amp, persistent_actor, device);
        let (fine_vs, fine) = SpatialAE::load(fine_path.as_ref(), device, REGION, ae_amp)?;
        let (coarse_vs, coarse) = match coarse_path {
            Some(p) => {
                let (vs, ae) = SpatialAE::load(p, device, COARSE_REGION, ae_amp)?;
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

    #[cfg(test)]
    pub(crate) fn random_for_test(device: Device) -> Result<Self> {
        let fine_vs = nn::VarStore::new(device);
        let fine = SpatialAE::new(&fine_vs.root(), LATENT_C, REGION)?;
        let coarse_vs = nn::VarStore::new(device);
        let coarse = SpatialAE::new(&coarse_vs.root(), LATENT_C, COARSE_REGION)?;
        Ok(Self {
            fine,
            _fine_vs: fine_vs,
            coarse: Some(coarse),
            _coarse_vs: Some(coarse_vs),
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
    /// Exact structure planes at /8 (6, gh, gw). Not fed to the AE; assembled
    /// into the policy grid after the latent (see `batch` / `C_GRID`).
    pub stat: Vec<f32>,
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
            fallout[y * wr + x] = ((packed[y * packed_wr + x / 8] >> (7 - x % 8)) & 1) as f32;
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
            // Cache static terrain on CPU too — without it, watch/CPU encode
            // rebuilds the full land/fallout tensor every step and crawls.
            let use_terrain_cache = terrain_cache.is_some();
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
            for &i in chunk {
                let it = items[i];
                owners.extend_from_slice(&it.owners);
                if !use_terrain_cache {
                    terrain.extend_from_slice(it.static_terrain.land_mag.as_ref());
                    terrain.extend(unpack_fallout_host(&it.fallout, hr, wr));
                } else {
                    fallout.extend_from_slice(&it.fallout);
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
            let z = tch::no_grad(|| ae.encode(&owners_t, &terrain_t));
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

/// Batched fine/coarse outputs for an exactly uniform actor batch. Unlike
/// [`DualLatents`], these tensors retain their batch dimension so callers do
/// not have to split, pad, and stack them again before policy inference.
#[derive(Debug)]
pub struct UniformDualLatents {
    pub fine: Tensor,
    pub coarse: Option<Tensor>,
}

/// Same-shape paired encode using contiguous batched inputs and outputs.
///
/// This is intentionally separate from [`encode_dual_latent_batch_device`]:
/// mixed-shape callers retain that established grouping/order-restoration
/// path unchanged. Every tensor here is owned by the persistent actor thread.
pub fn encode_uniform_latent_batch_device(
    pair: &AePair,
    items: &[&AeRaw],
    device: Device,
    terrain_cache: &mut TerrainDeviceCache,
) -> Result<UniformDualLatents> {
    anyhow::ensure!(!items.is_empty(), "uniform AE batch must not be empty");
    let groups = validate_items(items)?;
    let (hr, wr) = (items[0].hr, items[0].wr);
    anyhow::ensure!(
        groups.len() == 1 && items.iter().all(|item| item.hr == hr && item.wr == wr),
        "uniform AE batch contains mixed full-resolution shapes"
    );
    anyhow::ensure!(
        terrain_cache.device == device,
        "terrain cache device {:?} does not match encode device {:?}",
        terrain_cache.device,
        device
    );
    anyhow::ensure!(
        terrain_cache.supports_shared_pair_inputs(),
        "uniform AE encode requires persistent actor cache ownership"
    );
    anyhow::ensure!(
        pair.fine.latent_down == REGION,
        "uniform AE fine latent_down {}, expected {REGION}",
        pair.fine.latent_down
    );
    if let Some(coarse) = pair.coarse.as_ref() {
        anyhow::ensure!(
            coarse.latent_down == COARSE_REGION,
            "uniform AE coarse latent_down {}, expected {COARSE_REGION}",
            coarse.latent_down
        );
    }

    let pix = hr * wr;
    let packed_wr = packed_fallout_row_bytes(wr);
    let per = (MAX_ENC_PIX / pix).max(1);
    let fine_gh = hr / REGION as usize;
    let fine_gw = wr / REGION as usize;
    let coarse_encoder = pair.coarse.as_ref();
    let mut fine_chunks = Vec::with_capacity(items.len().div_ceil(per));
    let mut coarse_chunks = Vec::with_capacity(items.len().div_ceil(per));

    for chunk in items.chunks(per) {
        let b = chunk.len() as i64;
        let use_terrain_cache = true;
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
        for item in chunk {
            owners.extend_from_slice(&item.owners);
            if use_terrain_cache {
                fallout.extend_from_slice(&item.fallout);
            } else {
                terrain.extend_from_slice(item.static_terrain.land_mag.as_ref());
                terrain.extend(unpack_fallout_host(&item.fallout, hr, wr));
            }
        }

        // All host-backed uploads are synchronous: no temporary vector can
        // be dropped while a device copy is still reading it.
        let owners_t = upload_owner_indices(&owners, [b, hr as i64, wr as i64], device)?;
        let terrain_t = if use_terrain_cache {
            let shifts = terrain_cache.fallout_shifts();
            let static_items: Vec<Tensor> = chunk
                .iter()
                .map(|item| terrain_cache.static_tensor(&item.static_terrain))
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

        let (fine_z, coarse_z) = tch::no_grad(|| {
            let fine_z = pair.fine.encode(&owners_t, &terrain_t);
            let coarse_z = coarse_encoder.map(|encoder| encoder.encode(&owners_t, &terrain_t));
            (fine_z, coarse_z)
        });
        fine_chunks.push(fine_z);
        if let Some(z) = coarse_z {
            coarse_chunks.push(z);
        }
    }

    let join = |mut chunks: Vec<Tensor>| {
        if chunks.len() == 1 {
            chunks.pop().unwrap()
        } else {
            let refs: Vec<&Tensor> = chunks.iter().collect();
            Tensor::cat(&refs, 0)
        }
    };
    let fine = join(fine_chunks);
    let coarse = coarse_encoder.map(|_| join(coarse_chunks));
    let b = items.len() as i64;
    anyhow::ensure!(
        fine.size() == [b, pair.fine.latent_c, fine_gh as i64, fine_gw as i64],
        "uniform fine AE output has unexpected shape {:?}",
        fine.size()
    );
    if let (Some(encoder), Some(z)) = (coarse_encoder, coarse.as_ref()) {
        let expected = [
            b,
            encoder.latent_c,
            fine_gh.div_ceil(2) as i64,
            fine_gw.div_ceil(2) as i64,
        ];
        anyhow::ensure!(
            z.size() == expected,
            "uniform coarse AE output has unexpected shape {:?}",
            z.size()
        );
    }
    Ok(UniformDualLatents { fine, coarse })
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
            let use_terrain_cache = true;
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
            for &i in chunk {
                let item = items[i];
                owners.extend_from_slice(&item.owners);
                if use_terrain_cache {
                    fallout.extend_from_slice(&item.fallout);
                } else {
                    terrain.extend_from_slice(item.static_terrain.land_mag.as_ref());
                    terrain.extend(unpack_fallout_host(&item.fallout, hr, wr));
                }
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

            let (fine_z, coarse_z) = tch::no_grad(|| {
                let fine_z = pair.fine.encode(&owners_t, &terrain_t);
                let coarse_z = coarse.encode(&owners_t, &terrain_t);
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
        tch::no_grad(|| ae.encode(&owners, &terrain).select(0, 0))
    }

    fn flat_f32(tensor: &Tensor) -> Vec<f32> {
        Vec::<f32>::try_from(tensor.reshape([-1])).unwrap()
    }

    fn bf16_cache_tensors(ae: &SpatialAE) -> Vec<&Tensor> {
        let mut tensors = Vec::new();
        for block in &ae.enc_stem {
            let conv = block.bf16.as_ref().expect("stem bf16 cache");
            tensors.push(&conv.ws);
            tensors.extend(conv.bs.iter());
        }
        let fuse = ae.enc_fuse.bf16.as_ref().expect("fuse bf16 cache");
        tensors.push(&fuse.ws);
        tensors.extend(fuse.bs.iter());
        let out = ae.enc_out_bf16.as_ref().expect("output bf16 cache");
        tensors.push(&out.ws);
        tensors.extend(out.bs.iter());
        tensors
    }

    fn bf16_cache_ptrs(ae: &SpatialAE) -> Vec<usize> {
        bf16_cache_tensors(ae)
            .into_iter()
            .map(|tensor| tensor.data_ptr() as usize)
            .collect()
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
            let device =
                unpack_fallout_device(&Tensor::from_slice(&packed).view([1, 1, 1]), 8, &shifts);
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
        let unpacked =
            unpack_fallout_device(&Tensor::from_slice(&packed).view([1, 3, 2]), 11, &shifts);
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
    fn uniform_batched_encode_exactly_matches_forced_general_fine_and_coarse() {
        let pair = random_pair();
        // 3x5 fine and 2x3 coarse dimensions exercise odd ceil pooling.
        let raws = [
            patterned_raw(21, 24, 40),
            patterned_raw(22, 24, 40),
            patterned_raw(23, 24, 40),
            patterned_raw(24, 24, 40),
        ];
        let refs: Vec<&AeRaw> = raws.iter().collect();
        let mut uniform_cache = TerrainDeviceCache::new_persistent_actor(Device::Cpu);
        let uniform =
            encode_uniform_latent_batch_device(&pair, &refs, Device::Cpu, &mut uniform_cache)
                .unwrap();

        // Deliberately invoke the established general path for the same
        // shapes, then reproduce its per-item stack at the policy boundary.
        let mut general_cache = TerrainDeviceCache::new_persistent_actor(Device::Cpu);
        let general =
            encode_dual_latent_batch_device(&pair, &refs, Device::Cpu, &mut general_cache).unwrap();
        let general_fine_refs: Vec<&Tensor> = general.fine.iter().collect();
        let general_coarse_refs: Vec<&Tensor> = general.coarse.iter().collect();
        let general_fine = Tensor::stack(&general_fine_refs, 0);
        let general_coarse = Tensor::stack(&general_coarse_refs, 0);

        assert_eq!(uniform.fine.size(), [4, LATENT_C, 3, 5]);
        assert_eq!(uniform.coarse.as_ref().unwrap().size(), [4, LATENT_C, 2, 3]);
        assert_eq!(flat_f32(&uniform.fine), flat_f32(&general_fine));
        assert_eq!(
            flat_f32(uniform.coarse.as_ref().unwrap()),
            flat_f32(&general_coarse)
        );
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
            .join("../../weights/ae/ae_v32_nostatic_d8c32.encoder.safetensors");
        if !path.exists() {
            eprintln!(
                "skip: {} missing (train ofae v3.2 / export encoder)",
                path.display()
            );
            return;
        }
        let (_vs, ae) = SpatialAE::load(&path, Device::Cpu, REGION, false).unwrap();
        assert_eq!(ae.latent_down, 8);
        let owners = Tensor::zeros([1, 32, 40], (Kind::Int64, Device::Cpu));
        let terrain = Tensor::zeros([1, 3, 32, 40], (Kind::Float, Device::Cpu));
        let z = tch::no_grad(|| ae.encode(&owners, &terrain));
        assert_eq!(z.size(), &[1, 32, 4, 5]);
        assert!(z.isfinite().all().double_value(&[]) != 0.0);
    }

    #[test]
    fn checkpoint_is_loaded_before_bf16_cache_is_created() {
        tch::manual_seed(131);
        let source_vs = nn::VarStore::new(Device::Cpu);
        let source = SpatialAE::new(&source_vs.root(), LATENT_C, REGION).unwrap();
        let path = std::env::temp_dir().join(format!(
            "oftrain-ae-bf16-cache-{}.safetensors",
            std::process::id()
        ));
        source_vs.save(&path).unwrap();

        let (loaded_vs, loaded) = SpatialAE::load(&path, Device::Cpu, REGION, true).unwrap();
        std::fs::remove_file(&path).unwrap();

        let source_variables = source_vs.variables();
        let loaded_variables = loaded_vs.variables();
        assert_eq!(source_variables.len(), loaded_variables.len());
        for (name, expected) in source_variables {
            let actual = &loaded_variables[&name];
            assert!(
                actual.equal(&expected),
                "checkpoint parameter {name} did not load exactly"
            );
            assert_eq!(actual.kind(), Kind::Float, "{name}");
        }
        assert!(
            loaded.enc_stem[0]
                .bf16
                .as_ref()
                .unwrap()
                .ws
                .equal(&source.enc_stem[0].conv.ws.to_kind(Kind::BFloat16)),
            "cache was not derived from loaded checkpoint weights"
        );
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
        let device_fallout =
            unpack_fallout_device(&Tensor::from_slice(&packed).view([1, 8, 1]), 8, &shifts);
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
        let expected = tch::no_grad(|| ae.encode(&owners, &uncached));
        let actual = tch::no_grad(|| ae.encode(&owners, &cached));
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
        raw_b.fallout = pack_fallout(&(0..64).map(|i| (i == 5) as u8).collect::<Vec<_>>(), 8, 8);
        assert!(Arc::ptr_eq(
            &raw_a.static_terrain.land_mag,
            &raw_b.static_terrain.land_mag
        ));
        assert_ne!(raw_a.stat, raw_b.stat, "structure stats were stale");
        assert_ne!(raw_a.fallout, raw_b.fallout, "fallout was stale");
    }

    #[test]
    fn conv_only_bf16_is_finite_close_and_keeps_weights_f32() {
        for latent_down in [REGION, COARSE_REGION] {
            tch::manual_seed(17);
            let fp_vs = nn::VarStore::new(Device::Cpu);
            let fp_ae = SpatialAE::new(&fp_vs.root(), LATENT_C, latent_down).unwrap();

            let mut bf_vs = nn::VarStore::new(Device::Cpu);
            let mut bf_ae = SpatialAE::new(&bf_vs.root(), LATENT_C, latent_down).unwrap();
            bf_vs.copy(&fp_vs).unwrap();
            bf_ae.cache_bf16();

            let side = 32;
            let owners = Tensor::randint(MAX_SLOTS, [2, side, side], (Kind::Int64, Device::Cpu));
            let terrain = Tensor::rand(
                [2, TERRAIN_CHANNELS, side, side],
                (Kind::Float, Device::Cpu),
            );
            let fp = tch::no_grad(|| fp_ae.encode(&owners, &terrain));
            let bf = tch::no_grad(|| bf_ae.encode(&owners, &terrain));

            assert!(
                bf_vs.variables().values().all(|v| v.kind() == Kind::Float),
                "mixed precision must not mutate stored AE parameter dtypes \
                 (CUDA reclaim only moves unused f32 conv storage to CPU)"
            );
            assert_eq!(bf.kind(), Kind::Float);
            assert!(bf.isfinite().all().int64_value(&[]) != 0);
            let mean_abs_err = (&fp - &bf).abs().mean(Kind::Float).double_value(&[]);
            assert!(
                mean_abs_err < 0.02,
                "1/{latent_down} bf16 mean absolute error {mean_abs_err} is too large"
            );
        }
    }

    #[test]
    fn bf16_conv_cache_has_expected_kind_and_allocates_only_once() {
        for (latent_down, conv_count) in [(REGION, 6), (COARSE_REGION, 7)] {
            let mut vs = nn::VarStore::new(Device::Cpu);
            let mut ae = SpatialAE::new(&vs.root(), LATENT_C, latent_down).unwrap();
            vs.freeze();
            ae.cache_bf16();

            let tensors = bf16_cache_tensors(&ae);
            assert_eq!(
                tensors.len(),
                conv_count * 2,
                "each cached convolution has one weight and one bias"
            );
            assert!(tensors.iter().all(|tensor| tensor.kind() == Kind::BFloat16));
            assert!(tensors.iter().all(|tensor| !tensor.requires_grad()));
            assert!(
                vs.variables()
                    .values()
                    .all(|tensor| { tensor.kind() == Kind::Float && !tensor.requires_grad() })
            );

            let allocated = bf16_cache_ptrs(&ae);
            ae.cache_bf16();
            assert_eq!(
                bf16_cache_ptrs(&ae),
                allocated,
                "cache initialization must be idempotent"
            );

            let side = 32;
            let owners = Tensor::zeros([1, side, side], (Kind::Int64, Device::Cpu));
            let terrain = Tensor::zeros(
                [1, TERRAIN_CHANNELS, side, side],
                (Kind::Float, Device::Cpu),
            );
            let _ = tch::no_grad(|| ae.encode(&owners, &terrain));
            let _ = tch::no_grad(|| ae.encode(&owners, &terrain));
            assert_eq!(
                bf16_cache_ptrs(&ae),
                allocated,
                "forwards must reuse cached parameter allocations"
            );
        }
    }

    #[test]
    fn bf16_gate_requires_amp_cuda_and_persistent_actor() {
        for amp in [false, true] {
            for persistent in [false, true] {
                for device in [Device::Cpu, Device::Cuda(0)] {
                    assert_eq!(
                        use_bf16_ae(amp, persistent, device),
                        amp && persistent && device.is_cuda()
                    );
                }
            }
        }
    }

    #[test]
    fn shared_fine_coarse_bf16_is_finite_close_and_has_exact_shapes() {
        let fp_pair = random_pair();
        let mut bf_pair = random_pair();
        bf_pair._fine_vs.copy(&fp_pair._fine_vs).unwrap();
        bf_pair
            ._coarse_vs
            .as_mut()
            .unwrap()
            .copy(fp_pair._coarse_vs.as_ref().unwrap())
            .unwrap();
        bf_pair.fine.cache_bf16();
        bf_pair.coarse.as_mut().unwrap().cache_bf16();
        let raws = [
            patterned_raw(21, 24, 40),
            patterned_raw(22, 16, 32),
            patterned_raw(23, 24, 40),
        ];
        let refs: Vec<&AeRaw> = raws.iter().collect();
        let mut fp_cache = TerrainDeviceCache::new_persistent_actor(Device::Cpu);
        let mut bf_cache = TerrainDeviceCache::new_persistent_actor(Device::Cpu);
        let fp =
            encode_dual_latent_batch_device(&fp_pair, &refs, Device::Cpu, &mut fp_cache).unwrap();
        let bf =
            encode_dual_latent_batch_device(&bf_pair, &refs, Device::Cpu, &mut bf_cache).unwrap();

        for (i, raw) in raws.iter().enumerate() {
            let gh = raw.hr as i64 / REGION;
            let gw = raw.wr as i64 / REGION;
            for (name, actual, expected, shape) in [
                ("fine", &bf.fine[i], &fp.fine[i], [LATENT_C, gh, gw]),
                (
                    "coarse",
                    &bf.coarse[i],
                    &fp.coarse[i],
                    [LATENT_C, (gh + 1) / 2, (gw + 1) / 2],
                ),
            ] {
                assert_eq!(actual.size(), shape, "{name} item {i}");
                assert_eq!(actual.kind(), Kind::Float, "{name} API dtype item {i}");
                assert!(
                    actual.isfinite().all().int64_value(&[]) != 0,
                    "{name} non-finite item {i}"
                );
                let mean_abs_err = (actual - expected)
                    .abs()
                    .mean(Kind::Float)
                    .double_value(&[]);
                assert!(
                    mean_abs_err < 0.02,
                    "{name} item {i} bf16 mean absolute error {mean_abs_err}"
                );
            }
        }
    }
}
