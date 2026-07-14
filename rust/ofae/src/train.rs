//! AE training loop (parity with `ae.train_v3`).

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::Args;
use rand::SeedableRng;
use serde_json::json;
use tch::nn::{self, OptimizerConfig};
use tch::{IndexOp, Kind, Tensor};

use crate::checkpoint::save_checkpoint;
use crate::data::GameBank;
use crate::model::{
    border_mask, build_optimizer, device_from_str, pick_device, SpatialAE, STATIC_CLASS_WEIGHTS,
    NUM_STATIC,
};

#[derive(Debug, Args)]
pub struct TrainArgs {
    #[arg(long, default_value = "data")]
    pub data: String,
    #[arg(long, default_value_t = 10000)]
    pub steps: i64,
    #[arg(long, default_value_t = 32)]
    pub batch_size: i64,
    #[arg(long, default_value_t = 256)]
    pub crop: i64,
    #[arg(long, default_value_t = 32)]
    pub latent_c: i64,
    #[arg(long, default_value_t = 8)]
    pub latent_down: i64,
    #[arg(long, default_value_t = 3e-4)]
    pub lr: f64,
    #[arg(long, default_value_t = 4.0)]
    pub border_weight: f64,
    #[arg(long, default_value_t = 1.5)]
    pub focal_gamma: f64,
    #[arg(long, default_value_t = 1.0)]
    pub w_units: f64,
    #[arg(long, default_value_t = 20.0)]
    pub unit_pos_weight: f64,
    #[arg(long, default_value = "runs/ae_v3")]
    pub out: PathBuf,
    #[arg(long)]
    pub init: Option<PathBuf>,
    #[arg(long)]
    pub device: Option<String>,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

pub fn run(args: TrainArgs) -> Result<()> {
    if args.latent_down != 8 && args.latent_down != 16 {
        anyhow::bail!("--latent-down must be 8 or 16");
    }
    let device = match &args.device {
        Some(s) => device_from_str(s)?,
        None => pick_device(),
    };
    eprintln!("ofae: device={device:?}");

    let bank = GameBank::load(&args.data)?;
    let mut rng = rand::rngs::SmallRng::seed_from_u64(args.seed);

    let mut vs = nn::VarStore::new(device);
    let ae = SpatialAE::new(&vs.root(), args.latent_c, args.latent_down)?;
    if let Some(init) = &args.init {
        vs.load(init)
            .with_context(|| format!("init load {}", init.display()))?;
        eprintln!("ofae: warm-started from {}", init.display());
    }
    let n_params: i64 = vs
        .trainable_variables()
        .iter()
        .map(|t| t.numel() as i64)
        .sum();
    eprintln!("ofae: model {:.2}M params", n_params as f64 / 1e6);

    let mut opt = build_optimizer(&vs, args.lr)?;
    let warmup = (args.steps / 10).clamp(1, 500);

    let unit_pos_w = Tensor::from_slice(&STATIC_CLASS_WEIGHTS)
        .to_device(device)
        .to_kind(Kind::Float)
        .view([1, NUM_STATIC, 1, 1])
        * args.unit_pos_weight;

    let meta = json!({
        "latent_c": args.latent_c,
        "latent_down": args.latent_down,
        "terrain_cond": true,
        "upsample_decoder": true,
        "crop": args.crop,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "border_weight": args.border_weight,
        "focal_gamma": args.focal_gamma,
        "w_units": args.w_units,
        "unit_pos_weight": args.unit_pos_weight,
        "data": args.data,
    });

    let crop = args.crop as usize;
    let t0 = Instant::now();
    for step in 1..=args.steps {
        // LR schedule: linear warmup then cosine to 5% of peak.
        let lr_scale = if step <= warmup {
            step as f64 / warmup as f64
        } else {
            let t = (step - warmup) as f64 / (args.steps - warmup).max(1) as f64;
            0.05 + 0.95 * 0.5 * (1.0 + (std::f64::consts::PI * t).cos())
        };
        opt.set_lr(args.lr * lr_scale);

        let (owners_v, terrain_v, planes_v) =
            bank.sample_batch(&mut rng, args.batch_size as usize, crop, args.latent_down)?;
        let owners = Tensor::from_slice(&owners_v)
            .to_device(device)
            .view([args.batch_size, args.crop, args.crop]);
        let terrain = Tensor::from_slice(&terrain_v)
            .to_device(device)
            .view([args.batch_size, 3, args.crop, args.crop]);
        let gh = args.crop / args.latent_down;
        let planes = Tensor::from_slice(&planes_v)
            .to_device(device)
            .view([args.batch_size, NUM_STATIC, gh, gh]);

        let (tile_logits, unit_logits, _) = ae.forward(&owners, &terrain, &planes);

        let mut per_tile = tile_logits.cross_entropy_loss(
            &owners,
            None::<Tensor>,
            tch::Reduction::None,
            -100,
            0.0,
        );
        if args.focal_gamma > 0.0 {
            let hard: Tensor = (Tensor::from(1.0f64) - (-&per_tile).exp()).clamp_min(0.0);
            per_tile = &per_tile * hard.pow_tensor_scalar(args.focal_gamma);
        }
        let border = border_mask(&owners);
        let weights = Tensor::ones_like(&per_tile)
            + border.to_kind(Kind::Float) * args.border_weight;
        let loss_tiles = (&per_tile * &weights).sum(Kind::Float) / weights.sum(Kind::Float);
        let loss_units = unit_logits.binary_cross_entropy_with_logits(
            &planes,
            None::<Tensor>,
            Some(unit_pos_w.shallow_clone()),
            tch::Reduction::Mean,
        );
        let loss = &loss_tiles + args.w_units * &loss_units;

        opt.zero_grad();
        loss.backward();
        opt.step();

        if step % 50 == 0 || step == 1 {
            tch::no_grad(|| {
                let pred = tile_logits.argmax(1, false);
                let ok = pred.eq_tensor(&owners);
                let acc = ok.to_kind(Kind::Float).mean(Kind::Float).double_value(&[]);
                let bacc = if border.any().double_value(&[]) > 0.5 {
                    ok.masked_select(&border)
                        .to_kind(Kind::Float)
                        .mean(Kind::Float)
                        .double_value(&[])
                } else {
                    1.0
                };
                let occ = planes.gt(0.5);
                let n_occ = occ.sum(Kind::Float).double_value(&[]);
                let unit_rec = if n_occ > 0.0 {
                    (unit_logits.gt(0.0).logical_and(&occ))
                        .sum(Kind::Float)
                        .double_value(&[])
                        / n_occ
                } else {
                    f64::NAN
                };
                let rate =
                    step as f64 * args.batch_size as f64 / t0.elapsed().as_secs_f64().max(1e-6);
                eprintln!(
                    "step {step:5}  loss {:.4}  tiles {:.4}  units {:.4}  acc {acc:.4}  bacc {bacc:.4}  unit-rec {unit_rec:.2}  lr {:.2e}  {rate:.1} ex/s",
                    loss.double_value(&[]),
                    loss_tiles.double_value(&[]),
                    loss_units.double_value(&[]),
                    args.lr * lr_scale,
                );
            });
        }

        if step % 500 == 0 || step == args.steps {
            save_checkpoint(&vs, &args.out, step, &meta)?;
            if step % 5000 == 0 {
                let milestone = args.out.join(format!("ae_v3_step{step}.safetensors"));
                vs.save(&milestone)?;
            }
        }
    }
    eprintln!("ofae: done -> {}", args.out.join("ae_v3.safetensors").display());
    Ok(())
}
