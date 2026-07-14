//! Filter encoder prefixes from a safetensors file into `.encoder.safetensors`.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use memmap2::MmapOptions;
use safetensors::tensor::{SafeTensors, TensorView};
use safetensors::serialize_to_file;
use serde_json::json;

pub const ENCODER_PREFIXES: &[&str] = &["owner_emb.", "enc_stem.", "enc_fuse."];

pub struct ExportArgs {
    pub input: PathBuf,
    pub out: PathBuf,
    pub meta_out: Option<PathBuf>,
    pub expected_down: Option<i64>,
    pub expected_c: Option<i64>,
}

pub fn export_encoder(args: ExportArgs) -> Result<()> {
    let file = File::open(&args.input)
        .with_context(|| format!("open {}", args.input.display()))?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let tensors = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("deserialize {}", args.input.display()))?;
    let (_hdr_len, metadata) = SafeTensors::read_metadata(&mmap)?;

    let mut filtered: Vec<(String, TensorView<'_>)> = Vec::new();
    for name in tensors.names() {
        if ENCODER_PREFIXES.iter().any(|p| name.starts_with(p)) {
            let view = tensors.tensor(name)?;
            filtered.push((name.to_string(), view));
        }
    }
    if filtered.is_empty() {
        bail!("no encoder tensors found in {}", args.input.display());
    }
    filtered.sort_by(|a, b| a.0.cmp(&b.0));

    let mut meta: HashMap<String, String> = HashMap::new();
    meta.insert("format".into(), "spatial_ae_encoder_v3".into());
    if let Some(m) = metadata.metadata() {
        for (k, v) in m {
            if k == "format" {
                continue;
            }
            meta.insert(k.clone(), v.clone());
        }
    }
    meta.insert(
        "source".into(),
        args.input.canonicalize().unwrap_or(args.input.clone()).display().to_string(),
    );

    if let Some(parent) = args.out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // serialize_to_file needs owned data; copy bytes out of the mmap views.
    let owned: Vec<(String, Vec<u8>, Vec<usize>, safetensors::Dtype)> = filtered
        .iter()
        .map(|(name, view)| {
            (
                name.clone(),
                view.data().to_vec(),
                view.shape().to_vec(),
                view.dtype(),
            )
        })
        .collect();
    let views: Vec<(String, TensorView<'_>)> = owned
        .iter()
        .map(|(name, data, shape, dtype)| {
            (
                name.clone(),
                TensorView::new(*dtype, shape.clone(), data).expect("tensor view"),
            )
        })
        .collect();
    serialize_to_file(
        views.iter().map(|(n, v)| (n.as_str(), v.clone())),
        Some(meta.clone()),
        &args.out,
    )
    .with_context(|| format!("write {}", args.out.display()))?;

    let latent_c = meta
        .get("latent_c")
        .and_then(|s| s.parse::<i64>().ok());
    let latent_down = meta
        .get("latent_down")
        .and_then(|s| s.parse::<i64>().ok());
    if let Some(expected) = args.expected_c {
        if let Some(got) = latent_c {
            if got != expected {
                bail!("latent_c mismatch: expected {expected}, got {got}");
            }
        }
    }
    if let Some(expected) = args.expected_down {
        if let Some(got) = latent_down {
            if got != expected {
                bail!("latent_down mismatch: expected {expected}, got {got}");
            }
        }
    }

    let keys: Vec<_> = owned.iter().map(|(n, _, _, _)| n.clone()).collect();
    let meta_full = json!({
        "format": "spatial_ae_encoder_v3",
        "latent_c": latent_c,
        "latent_down": latent_down,
        "terrain_cond": meta.get("terrain_cond").map(|s| s == "True" || s == "true"),
        "upsample_decoder": meta.get("upsample_decoder").map(|s| s == "True" || s == "true"),
        "source": meta.get("source"),
        "keys": keys,
        "num_tensors": owned.len(),
    });
    let meta_path = args
        .meta_out
        .unwrap_or_else(|| args.out.with_extension("json"));
    std::fs::write(
        &meta_path,
        serde_json::to_string_pretty(&meta_full)? + "\n",
    )?;
    println!(
        "wrote {} encoder tensors -> {}",
        owned.len(),
        args.out.display()
    );
    println!(
        "meta -> {} (latent_c={:?} latent_down={:?})",
        meta_path.display(),
        latent_c,
        latent_down
    );
    Ok(())
}

pub fn is_safetensors(path: &Path) -> bool {
    path.extension().and_then(|e| e.to_str()) == Some("safetensors")
}
