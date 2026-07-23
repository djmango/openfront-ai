//! Checkpoint helpers: full AE safetensors + encoder-only filter.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde_json::json;
use tch::nn::VarStore;
use tch::Tensor;

use crate::model::ENCODER_PREFIXES;

pub fn save_checkpoint(
    vs: &VarStore,
    out_dir: &Path,
    step: i64,
    meta: &serde_json::Value,
) -> Result<()> {
    std::fs::create_dir_all(out_dir)?;
    let full = out_dir.join("ae_v3.safetensors");
    let named: Vec<(String, Tensor)> = vs
        .variables()
        .into_iter()
        .map(|(k, t)| (k, t.shallow_clone()))
        .collect();
    Tensor::write_safetensors(&named, &full)
        .with_context(|| format!("write {}", full.display()))?;

    let sidecar = out_dir.join("ae_v3.state.json");
    let mut m = meta.clone();
    if let Some(obj) = m.as_object_mut() {
        obj.insert("step".into(), json!(step));
        obj.insert("format".into(), json!("ofae-safetensors"));
    }
    std::fs::write(&sidecar, serde_json::to_string_pretty(&m)? + "\n")?;

    let enc = out_dir.join("ae_v3.encoder.safetensors");
    export_encoder_named(&named, &enc, meta)?;
    eprintln!("ofae: saved {} + encoder", full.display());
    Ok(())
}

pub fn export_encoder_from_vs(vs: &VarStore, out: &Path, meta: &serde_json::Value) -> Result<()> {
    let named: Vec<(String, Tensor)> = vs
        .variables()
        .into_iter()
        .map(|(k, t)| (k, t.shallow_clone()))
        .collect();
    export_encoder_named(&named, out, meta)
}

fn export_encoder_named(
    named: &[(String, Tensor)],
    out: &Path,
    meta: &serde_json::Value,
) -> Result<()> {
    let mut enc: Vec<(String, Tensor)> = named
        .iter()
        .filter(|(k, _)| ENCODER_PREFIXES.iter().any(|p| k.starts_with(p)))
        .map(|(k, t)| (k.clone(), t.shallow_clone()))
        .collect();
    if enc.is_empty() {
        bail!("no encoder tensors in VarStore");
    }
    enc.sort_by(|a, b| a.0.cmp(&b.0));
    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Tensor::write_safetensors(&enc, out)
        .with_context(|| format!("write {}", out.display()))?;

    let mut sidecar = HashMap::new();
    sidecar.insert("format", json!("spatial_ae_encoder_v32_nostatic"));
    sidecar.insert("static_in_latent", json!(false));
    if let Some(obj) = meta.as_object() {
        for key in [
            "latent_c",
            "latent_down",
            "terrain_cond",
            "upsample_decoder",
            "schema",
            "static_in_latent",
        ] {
            if let Some(v) = obj.get(key) {
                sidecar.insert(key, v.clone());
            }
        }
    }
    let meta_path = out.with_extension("json");
    std::fs::write(&meta_path, serde_json::to_string_pretty(&sidecar)? + "\n")?;
    Ok(())
}

pub fn export_encoder_file(input: &Path, out: &Path, meta: &serde_json::Value) -> Result<()> {
    let named = Tensor::read_safetensors(input)
        .with_context(|| format!("read {}", input.display()))?;
    export_encoder_named(&named, out, meta)
}
