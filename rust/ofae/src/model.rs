//! Full SpatialAE (encoder + v3.1 upsample decoder) for AE training.
//!
//! Encoder VarStore paths match `oftrain::ae` / `ofexport` so
//! `.encoder.safetensors` loads into PPO without remapping.

use anyhow::{bail, Result};
use tch::nn::{self, Module, OptimizerConfig};
use tch::{Device, IndexOp, Kind, Tensor};

pub const MAX_SLOTS: i64 = 128;
pub const OWNER_EMB_DIM: i64 = 8;
pub const TERRAIN_CHANNELS: i64 = 3;
pub const NUM_STATIC: i64 = 6;
pub const STATIC_TERRAIN_C: i64 = 2;
pub const STATIC_CLASS_WEIGHTS: [f64; 6] = [1.0, 1.0, 1.0, 4.0, 4.0, 2.0];

pub const ENCODER_PREFIXES: &[&str] = &["owner_emb.", "enc_stem.", "enc_fuse."];

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

struct UpsampleBlock {
    conv: ConvGn,
}

impl UpsampleBlock {
    fn new(vs: &nn::Path, c_in: i64, c_out: i64, extra_c: i64) -> Self {
        Self {
            conv: ConvGn::new(&(vs / "conv"), c_in + extra_c, c_out, 1),
        }
    }

    fn forward(&self, x: &Tensor, extra: Option<&Tensor>) -> Tensor {
        let mut x = x.upsample_nearest2d(&[x.size()[2] * 2, x.size()[3] * 2], None, None);
        if let Some(e) = extra {
            x = Tensor::cat(&[&x, e], 1);
        }
        self.conv.forward(&x)
    }
}

pub struct SpatialAE {
    owner_emb: nn::Embedding,
    enc_stem: Vec<ConvGn>,
    enc_fuse: ConvGn,
    enc_out: nn::Conv2D,
    dec_in: ConvGn,
    dec_up: Vec<UpsampleBlock>,
    dec_refine: ConvGn,
    dec_out: nn::Conv2D,
    dec_units: nn::Conv2D,
    pub latent_c: i64,
    pub latent_down: i64,
    pub terrain_cond: bool,
}

impl SpatialAE {
    /// v3.1: terrain-conditioned upsample decoder (training default).
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

        let cond_c = STATIC_TERRAIN_C;
        let dec_in = ConvGn::new(&(vs / "dec_in"), latent_c + cond_c, 128, 1);
        let chans: Vec<i64> = if latent_down == 16 {
            vec![128, 128, 96, 64, 32]
        } else {
            vec![128, 96, 64, 32]
        };
        let dec_up: Vec<UpsampleBlock> = (0..chans.len() - 1)
            .map(|i| {
                UpsampleBlock::new(&(vs / "dec_up" / i), chans[i], chans[i + 1], cond_c)
            })
            .collect();
        let dec_refine = ConvGn::new(&(vs / "dec_refine"), 32 + cond_c, 32, 1);
        let mut dcfg = nn::ConvConfig::default();
        dcfg.stride = 1;
        dcfg.padding = 0;
        let dec_out = nn::conv2d(vs / "dec_out", 32, MAX_SLOTS, 1, dcfg);
        let mut ucfg = nn::ConvConfig::default();
        ucfg.stride = 1;
        ucfg.padding = 0;
        let dec_units = nn::conv2d(vs / "dec_units", 128, NUM_STATIC, 1, ucfg);

        Ok(Self {
            owner_emb,
            enc_stem,
            enc_fuse,
            enc_out,
            dec_in,
            dec_up,
            dec_refine,
            dec_out,
            dec_units,
            latent_c,
            latent_down,
            terrain_cond: true,
        })
    }

    pub fn encode(&self, owners: &Tensor, terrain: &Tensor, static_planes: &Tensor) -> Tensor {
        let emb = self.owner_emb.forward(owners).permute([0, 3, 1, 2]);
        let mut g = Tensor::cat(&[&emb, terrain], 1);
        for block in &self.enc_stem {
            g = block.forward(&g);
        }
        g = self.enc_fuse.forward(&Tensor::cat(&[&g, static_planes], 1));
        self.enc_out.forward(&g)
    }

    pub fn decode(&self, z_grid: &Tensor, terrain: &Tensor) -> (Tensor, Tensor) {
        let static_t = terrain.i((.., ..STATIC_TERRAIN_C, .., ..));
        let mut pyramid: Vec<(i64, Tensor)> = vec![(1, static_t.shallow_clone())];
        let mut down = 2i64;
        while down <= self.latent_down {
            pyramid.push((
                down,
                static_t.avg_pool2d([down, down], [down, down], [0, 0], false, true, None),
            ));
            down *= 2;
        }
        let lat = pyramid
            .iter()
            .find(|(d, _)| *d == self.latent_down)
            .map(|(_, t)| t)
            .expect("latent pyramid");
        let h = self.dec_in.forward(&Tensor::cat(&[z_grid, lat], 1));
        let mut x = h.shallow_clone();
        let mut scale = self.latent_down;
        for up in &self.dec_up {
            scale /= 2;
            let extra = pyramid
                .iter()
                .find(|(d, _)| *d == scale)
                .map(|(_, t)| t)
                .expect("scale pyramid");
            x = up.forward(&x, Some(extra));
        }
        let full = pyramid
            .iter()
            .find(|(d, _)| *d == 1)
            .map(|(_, t)| t)
            .expect("full pyramid");
        x = self.dec_refine.forward(&Tensor::cat(&[&x, full], 1));
        (self.dec_out.forward(&x), self.dec_units.forward(&h))
    }

    pub fn forward(
        &self,
        owners: &Tensor,
        terrain: &Tensor,
        static_planes: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let z = self.encode(owners, terrain, static_planes);
        let (tile, units) = self.decode(&z, terrain);
        (tile, units, z)
    }
}

pub fn border_mask(owners: &Tensor) -> Tensor {
    // owners: (B, H, W) - tile is border if any 4-neighbour ownership differs.
    let h = owners.size()[1];
    let w = owners.size()[2];
    let mut diff = Tensor::zeros(
        owners.size(),
        (Kind::Bool, owners.device()),
    );
    if h > 1 {
        let d = owners
            .i((.., 1.., ..))
            .ne_tensor(&owners.i((.., ..h - 1, ..)));
        let _ = diff.i((.., 1.., ..)).copy_(&d);
        let _ = diff.i((.., ..h - 1, ..)).logical_or_(&d);
    }
    if w > 1 {
        let d = owners
            .i((.., .., 1..))
            .ne_tensor(&owners.i((.., .., ..w - 1)));
        let _ = diff.i((.., .., 1..)).logical_or_(&d);
        let _ = diff.i((.., .., ..w - 1)).logical_or_(&d);
    }
    diff
}

pub fn build_optimizer(vs: &nn::VarStore, lr: f64) -> Result<nn::Optimizer> {
    Ok(nn::AdamW::default().build(vs, lr)?)
}

pub fn device_from_str(s: &str) -> Result<Device> {
    match s {
        "cpu" => Ok(Device::Cpu),
        "cuda" | "cuda:0" => Ok(Device::Cuda(0)),
        other if other.starts_with("cuda:") => {
            let i: usize = other[5..].parse()?;
            Ok(Device::Cuda(i))
        }
        "mps" => Ok(Device::Mps),
        other => bail!("unknown device {other}"),
    }
}

pub fn pick_device() -> Device {
    if tch::utils::has_cuda() {
        Device::Cuda(0)
    } else if tch::utils::has_mps() {
        Device::Mps
    } else {
        Device::Cpu
    }
}
