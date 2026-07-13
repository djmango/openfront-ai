//! Stacks `Vec<PreparedObs>` (host f32 arrays, one per env) into GPU/CPU
//! tensors (`policy::Obs`). Envs in one batch do NOT all share the same
//! grid resolution (gh, gw) in general - every curriculum stage past
//! stage 1 samples from multiple maps of different native sizes
//! (`ofcore::curriculum::stages()`), and even a single-map stage mixes in
//! differently-sized *past*-stage maps via rehearsal sampling
//! (`REHEARSAL_P`) - so a mixed-shape batch isn't an edge case, it's the
//! common case as soon as training reaches stage 2. `build_obs` pads
//! every item up to the batch's max (gh, gw) (top-left aligned, zero-fill)
//! and sets `grid_valid` to 0 over the padded region instead of the old
//! all-ones placeholder - `policy.rs` already multiplies `grid_valid`
//! into every tile-legality mask and the foveation coverage math (see its
//! module doc), so padded cells are already excluded from action
//! selection; the network's fully-convolutional towers see zero-padding
//! at the border like any other edge, no architecture change needed. The
//! common same-shape-batch case (most single-map early stages, no
//! rehearsal hit) takes a fast path with no per-item copy loop.

use ofcore::feat;
use tch::{Device, Kind, Tensor};

use crate::ae::{self, AePair, TerrainDeviceCache};
use crate::policy::{self, CompactObsMeta, Obs, PolicyNet};
use crate::vecenv::{CompactGrid, PreparedObs};

/// `--pinned-h2d`: host->device upload for one freshly-built CPU tensor.
/// When `pinned && device` is CUDA, pins the CPU tensor's backing memory
/// for `device` (`Tensor::pin_memory`, needs a page-locked allocation so
/// the CUDA driver can DMA out of it) and issues a non-blocking copy
/// (`to_device_`'s `non_blocking=true`, ATen's `aten::to.device` op) so
/// the H2D copy can overlap with other CUDA-stream work instead of the
/// calling thread blocking on a synchronous `cudaMemcpy`. Falls back to
/// the original plain `to_device` when `pinned` is off or `device` isn't
/// CUDA - pinning is meaningless (and untestable - this Mac's libtorch
/// build has no CUDA support at all) without a real CUDA device, so this
/// keeps the CPU-only path byte-for-byte unchanged from before this flag
/// existed.
fn to_device_maybe_pinned(t: &Tensor, device: Device, pinned: bool) -> Tensor {
    if pinned && device.is_cuda() {
        t.pin_memory(device)
            .to_device_(device, t.kind(), true, false)
    } else {
        t.to_device(device)
    }
}

/// Host f32 slice → device Float tensor. When `fp16_rollout` and the
/// destination is CUDA, stage as Half on the host, H2D as Half (half the
/// PCIe bytes), then `.to_kind(Float)` on device. CPU / flag-off keep the
/// plain Float path (byte-identical to pre-flag behavior).
fn upload_float_grid(
    v: &[f32],
    shape: &[i64],
    device: Device,
    pinned: bool,
    fp16_rollout: bool,
) -> Tensor {
    let cpu = Tensor::from_slice(v).view(shape).to_kind(Kind::Float);
    if fp16_rollout && device.is_cuda() {
        let half = cpu.to_kind(Kind::Half);
        to_device_maybe_pinned(&half, device, pinned).to_kind(Kind::Float)
    } else {
        to_device_maybe_pinned(&cpu, device, pinned)
    }
}

/// Assemble per-item `C_GRID` host planes: AE latent (or prebuilt grid)
/// + ego + db + transient. When `ae` is set, runs batched GPU encode
/// (mirrors `rl/obs.py::encode_grids`); otherwise requires each item's
/// `PreparedObs::grid` to already be filled (unit-test path).
fn assemble_grids(
    items: &[&PreparedObs],
    device: Device,
    ae: Option<&AePair>,
    mut terrain_cache: Option<&mut TerrainDeviceCache>,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let c_grid = policy::C_GRID as usize;
    if let Some(pair) = ae {
        let raws: Vec<&ae::AeRaw> = items.iter().map(|it| &it.ae_raw).collect();
        let zs = ae::encode_latent_batch_device(
            &pair.fine,
            &raws,
            device,
            terrain_cache.as_deref_mut(),
        )?
        .into_iter()
        .map(|z| z.to_device(Device::Cpu).to_kind(Kind::Float))
        .collect::<Vec<_>>();
        let mut grids = Vec::with_capacity(items.len());
        for (it, z) in items.iter().zip(zs.into_iter()) {
            let plane = it.gh * it.gw;
            let z_flat = z.flatten(0, -1);
            let z_vec: Vec<f32> = Vec::<f32>::try_from(z_flat)
                .map_err(|e| anyhow::anyhow!("AE latent host copy failed: {e}"))?;
            debug_assert_eq!(z_vec.len(), policy::LATENT_C as usize * plane);
            let mut grid = Vec::with_capacity(c_grid * plane);
            grid.extend_from_slice(&z_vec);
            grid.extend_from_slice(&it.ego);
            grid.extend_from_slice(&it.db);
            grid.extend_from_slice(&it.transient);
            debug_assert_eq!(grid.len(), c_grid * plane);
            grids.push(grid);
        }
        // Optional native /16 coarse stream is attached later on Obs when
        // present; fine grid always uses the /8 AE above.
        Ok(grids)
    } else {
        items
            .iter()
            .map(|it| {
                it.grid.clone().ok_or_else(|| {
                    anyhow::anyhow!("PreparedObs.grid missing and no AE provided for encode")
                })
            })
            .collect()
    }
}

/// Optional coarse grids at /16 (C_GRID, cgh, cgw) when a coarse AE is
/// loaded; otherwise `None` and the policy falls back to 2x avg-pool.
fn assemble_coarse_grids(
    items: &[&PreparedObs],
    device: Device,
    ae: Option<&AePair>,
    mut terrain_cache: Option<&mut TerrainDeviceCache>,
) -> anyhow::Result<Option<Vec<Vec<f32>>>> {
    let Some(pair) = ae else { return Ok(None) };
    let Some(coarse) = pair.coarse.as_ref() else {
        return Ok(None);
    };
    let raws: Vec<&ae::AeRaw> = items.iter().map(|it| &it.ae_raw).collect();
    let zs =
        ae::encode_latent_batch_device(coarse, &raws, device, terrain_cache.as_deref_mut())?
            .into_iter()
            .map(|z| z.to_device(Device::Cpu).to_kind(Kind::Float))
            .collect::<Vec<_>>();
    let c_grid = policy::C_GRID as usize;
    let mut grids = Vec::with_capacity(items.len());
    for (it, z) in items.iter().zip(zs.into_iter()) {
        let cgh = it.gh.div_ceil(2);
        let cgw = it.gw.div_ceil(2);
        let z_flat = z.flatten(0, -1);
        let z_vec: Vec<f32> = Vec::<f32>::try_from(z_flat)
            .map_err(|e| anyhow::anyhow!("coarse AE latent host copy failed: {e}"))?;
        debug_assert_eq!(z_vec.len(), policy::LATENT_C as usize * cgh * cgw);
        // Pool ego/db (avg) and transient (max) to /16, matching encode_grids.
        let ego_c = avg_pool2_planes(&it.ego, 3, it.gh, it.gw);
        let db_c = avg_pool2_planes(&it.db, 1, it.gh, it.gw);
        let tr_c = max_pool2_planes(&it.transient, ofcore::feat::N_TRANSIENT, it.gh, it.gw);
        let mut grid = Vec::with_capacity(c_grid * cgh * cgw);
        grid.extend_from_slice(&z_vec);
        grid.extend_from_slice(&ego_c);
        grid.extend_from_slice(&db_c);
        grid.extend_from_slice(&tr_c);
        debug_assert_eq!(grid.len(), c_grid * cgh * cgw);
        grids.push(grid);
    }
    Ok(Some(grids))
}

fn avg_pool2_planes(src: &[f32], channels: usize, gh: usize, gw: usize) -> Vec<f32> {
    let cgh = gh.div_ceil(2);
    let cgw = gw.div_ceil(2);
    let mut out = vec![0.0f32; channels * cgh * cgw];
    for c in 0..channels {
        for y in 0..cgh {
            for x in 0..cgw {
                let mut s = 0.0f32;
                let mut n = 0.0f32;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let yy = y * 2 + dy;
                        let xx = x * 2 + dx;
                        if yy < gh && xx < gw {
                            s += src[c * gh * gw + yy * gw + xx];
                            n += 1.0;
                        }
                    }
                }
                out[c * cgh * cgw + y * cgw + x] = if n > 0.0 { s / n } else { 0.0 };
            }
        }
    }
    out
}

fn max_pool2_planes(src: &[f32], channels: usize, gh: usize, gw: usize) -> Vec<f32> {
    let cgh = gh.div_ceil(2);
    let cgw = gw.div_ceil(2);
    let mut out = vec![0.0f32; channels * cgh * cgw];
    for c in 0..channels {
        for y in 0..cgh {
            for x in 0..cgw {
                let mut m = f32::NEG_INFINITY;
                for dy in 0..2 {
                    for dx in 0..2 {
                        let yy = y * 2 + dy;
                        let xx = x * 2 + dx;
                        if yy < gh && xx < gw {
                            m = m.max(src[c * gh * gw + yy * gw + xx]);
                        }
                    }
                }
                out[c * cgh * cgw + y * cgw + x] = if m == f32::NEG_INFINITY { 0.0 } else { m };
            }
        }
    }
    out
}

struct UniformAeGrids {
    fine: Tensor,
    fine_latent: Tensor,
    coarse: Option<Tensor>,
    coarse_latent: Option<Tensor>,
}

/// Same-shape AE path: stack the encoder outputs on their original device,
/// upload only the host-native feature planes, and concatenate there.
fn encode_uniform_ae_grids(
    items: &[&PreparedObs],
    device: Device,
    pinned_h2d: bool,
    fp16_rollout: bool,
    pair: &AePair,
    mut terrain_cache: Option<&mut TerrainDeviceCache>,
) -> anyhow::Result<UniformAeGrids> {
    debug_assert!(!items.is_empty());
    debug_assert!(
        items
            .iter()
            .all(|it| it.gh == items[0].gh && it.gw == items[0].gw)
    );
    let b = items.len() as i64;
    let (gh, gw) = (items[0].gh, items[0].gw);
    let raws: Vec<&ae::AeRaw> = items.iter().map(|it| &it.ae_raw).collect();

    let fine_items = ae::encode_latent_batch_device(
        &pair.fine,
        &raws,
        device,
        terrain_cache.as_deref_mut(),
    )?;
    let fine_refs: Vec<&Tensor> = fine_items.iter().collect();
    let fine_latent = Tensor::stack(&fine_refs, 0);
    let fine = assemble_uniform_device_grid(
        items,
        &fine_latent,
        gh,
        gw,
        device,
        pinned_h2d,
        fp16_rollout,
        false,
    );

    let (coarse, coarse_latent) = if let Some(coarse_ae) = pair.coarse.as_ref() {
        let coarse_items = ae::encode_latent_batch_device(
            coarse_ae,
            &raws,
            device,
            terrain_cache.as_deref_mut(),
        )?;
        let coarse_refs: Vec<&Tensor> = coarse_items.iter().collect();
        let latent = Tensor::stack(&coarse_refs, 0);
        let (cgh, cgw) = (gh.div_ceil(2), gw.div_ceil(2));
        let grid = assemble_uniform_device_grid(
            items,
            &latent,
            cgh,
            cgw,
            device,
            pinned_h2d,
            fp16_rollout,
            true,
        );
        (Some(grid), Some(latent))
    } else {
        (None, None)
    };

    debug_assert_eq!(fine.size(), [b, policy::C_GRID, gh as i64, gw as i64]);
    Ok(UniformAeGrids {
        fine,
        fine_latent,
        coarse,
        coarse_latent,
    })
}

fn assemble_uniform_device_grid(
    items: &[&PreparedObs],
    latent: &Tensor,
    gh: usize,
    gw: usize,
    device: Device,
    pinned_h2d: bool,
    fp16_rollout: bool,
    coarse: bool,
) -> Tensor {
    let extra_c = policy::C_GRID as usize - policy::LATENT_C as usize;
    let mut extras = Vec::with_capacity(items.len() * extra_c * gh * gw);
    for it in items {
        if coarse {
            extras.extend(avg_pool2_planes(&it.ego, 3, it.gh, it.gw));
            extras.extend(avg_pool2_planes(&it.db, 1, it.gh, it.gw));
            extras.extend(max_pool2_planes(
                &it.transient,
                feat::N_TRANSIENT,
                it.gh,
                it.gw,
            ));
        } else {
            extras.extend_from_slice(&it.ego);
            extras.extend_from_slice(&it.db);
            extras.extend_from_slice(&it.transient);
        }
    }
    let extras = upload_float_grid(
        &extras,
        &[items.len() as i64, extra_c as i64, gh as i64, gw as i64],
        device,
        pinned_h2d,
        fp16_rollout,
    );
    // Match the old `--fp16-rollout` path, which quantized the complete
    // assembled grid during H2D, including the AE channels.
    let latent = if fp16_rollout && device.is_cuda() {
        latent.to_kind(Kind::Half).to_kind(Kind::Float)
    } else {
        latent.shallow_clone()
    };
    Tensor::cat(&[latent, extras], 1)
}

/// Variable-shape counterpart to `assemble_uniform_device_grid`: pad each
/// encoder output on its current device, and upload/pad only host-native
/// bypass planes. In particular, no full latent is copied through CPU just
/// because a curriculum batch contains differently-sized maps.
fn assemble_mixed_device_grid(
    items: &[&PreparedObs],
    latents: &[Tensor],
    gh: usize,
    gw: usize,
    device: Device,
    pinned_h2d: bool,
    fp16_rollout: bool,
    coarse: bool,
) -> Tensor {
    let mut padded_latents = Vec::with_capacity(items.len());
    let extra_c = policy::C_GRID as usize - policy::LATENT_C as usize;
    let mut extras = Vec::with_capacity(items.len() * extra_c * gh * gw);
    for (it, latent) in items.iter().zip(latents) {
        let (ih, iw) = if coarse {
            (it.gh.div_ceil(2), it.gw.div_ceil(2))
        } else {
            (it.gh, it.gw)
        };
        padded_latents.push(latent.constant_pad_nd([0, (gw - iw) as i64, 0, (gh - ih) as i64]));

        let src = if coarse {
            let mut v = avg_pool2_planes(&it.ego, 3, it.gh, it.gw);
            v.extend(avg_pool2_planes(&it.db, 1, it.gh, it.gw));
            v.extend(max_pool2_planes(
                &it.transient,
                feat::N_TRANSIENT,
                it.gh,
                it.gw,
            ));
            v
        } else {
            let mut v = Vec::with_capacity(extra_c * ih * iw);
            v.extend_from_slice(&it.ego);
            v.extend_from_slice(&it.db);
            v.extend_from_slice(&it.transient);
            v
        };
        for c in 0..extra_c {
            let mut dst = vec![0.0f32; gh * gw];
            for y in 0..ih {
                let src_off = c * ih * iw + y * iw;
                dst[y * gw..y * gw + iw].copy_from_slice(&src[src_off..src_off + iw]);
            }
            extras.extend(dst);
        }
    }
    let refs: Vec<&Tensor> = padded_latents.iter().collect();
    let latent = Tensor::stack(&refs, 0);
    let latent = if fp16_rollout && device.is_cuda() {
        latent.to_kind(Kind::Half).to_kind(Kind::Float)
    } else {
        latent
    };
    let extras = upload_float_grid(
        &extras,
        &[items.len() as i64, extra_c as i64, gh as i64, gw as i64],
        device,
        pinned_h2d,
        fp16_rollout,
    );
    Tensor::cat(&[latent, extras], 1)
}

fn build_device_ae_obs(
    items: &[&PreparedObs],
    device: Device,
    pinned_h2d: bool,
    fp16_rollout: bool,
    pair: &AePair,
    mut terrain_cache: Option<&mut TerrainDeviceCache>,
) -> anyhow::Result<Obs> {
    let gh = items.iter().map(|it| it.gh).max().unwrap_or(0);
    let gw = items.iter().map(|it| it.gw).max().unwrap_or(0);
    let raws: Vec<&ae::AeRaw> = items.iter().map(|it| &it.ae_raw).collect();
    let fine_latents = ae::encode_latent_batch_device(
        &pair.fine,
        &raws,
        device,
        terrain_cache.as_deref_mut(),
    )?;
    let fine = assemble_mixed_device_grid(
        items,
        &fine_latents,
        gh,
        gw,
        device,
        pinned_h2d,
        fp16_rollout,
        false,
    );
    let coarse = if let Some(coarse_ae) = pair.coarse.as_ref() {
        let latents = ae::encode_latent_batch_device(
            coarse_ae,
            &raws,
            device,
            terrain_cache.as_deref_mut(),
        )?;
        Some(assemble_mixed_device_grid(
            items,
            &latents,
            gh.div_ceil(2),
            gw.div_ceil(2),
            device,
            pinned_h2d,
            fp16_rollout,
            true,
        ))
    } else {
        None
    };
    build_obs_from_parts(
        items,
        device,
        pinned_h2d,
        fp16_rollout,
        None,
        None,
        Some((fine, coarse)),
    )
}

fn host_grids_from_latent(
    items: &[&PreparedObs],
    latent: &Tensor,
    coarse: bool,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let latent: Vec<f32> = Vec::<f32>::try_from(
        latent
            .to_device(Device::Cpu)
            .to_kind(Kind::Float)
            .reshape([-1]),
    )
    .map_err(|e| anyhow::anyhow!("AE latent host copy failed: {e}"))?;
    let mut offset = 0;
    let mut grids = Vec::with_capacity(items.len());
    for it in items {
        let (gh, gw) = if coarse {
            (it.gh.div_ceil(2), it.gw.div_ceil(2))
        } else {
            (it.gh, it.gw)
        };
        let latent_len = policy::LATENT_C as usize * gh * gw;
        let mut grid = Vec::with_capacity(policy::C_GRID as usize * gh * gw);
        grid.extend_from_slice(&latent[offset..offset + latent_len]);
        offset += latent_len;
        if coarse {
            grid.extend(avg_pool2_planes(&it.ego, 3, it.gh, it.gw));
            grid.extend(avg_pool2_planes(&it.db, 1, it.gh, it.gw));
            grid.extend(max_pool2_planes(
                &it.transient,
                feat::N_TRANSIENT,
                it.gh,
                it.gw,
            ));
        } else {
            grid.extend_from_slice(&it.ego);
            grid.extend_from_slice(&it.db);
            grid.extend_from_slice(&it.transient);
        }
        grids.push(grid);
    }
    debug_assert_eq!(offset, latent.len());
    Ok(grids)
}

/// Encode AE latents into each item's `grid` / `grid_coarse` host buffers
/// in place. Called on the actor thread (exclusive `&mut`) before
/// `build_obs` so learner threads never need to hold an AE.
pub fn encode_prepared_obs(
    items: &mut [PreparedObs],
    device: Device,
    ae: &AePair,
    terrain_cache: &mut TerrainDeviceCache,
) -> anyhow::Result<()> {
    let refs: Vec<&PreparedObs> = items.iter().collect();
    let fine = assemble_grids(&refs, device, Some(ae), Some(terrain_cache))?;
    let coarse = assemble_coarse_grids(&refs, device, Some(ae), Some(terrain_cache))?;
    for (i, it) in items.iter_mut().enumerate() {
        it.grid = Some(fine[i].clone());
        if let Some(ref cg) = coarse {
            it.grid_coarse = Some(cg[i].clone());
            it.cgh = it.gh.div_ceil(2);
            it.cgw = it.gw.div_ceil(2);
        } else {
            it.grid_coarse = None;
            it.cgh = 0;
            it.cgw = 0;
        }
    }
    Ok(())
}

pub fn build_obs(
    items: &[&PreparedObs],
    device: Device,
    pinned_h2d: bool,
    fp16_rollout: bool,
) -> Obs {
    if !items.is_empty() && items.iter().all(|it| it.compact.is_some()) {
        return build_compact_host_obs(items, device, pinned_h2d).expect("build compact obs");
    }
    build_obs_with_ae(items, device, pinned_h2d, fp16_rollout, None).expect("build_obs")
}

/// Actor-side compact rollout path. The full assembled tensors and AE
/// outputs remain local to this call/device. Only fp16 windows, fp16 coarse
/// grids, masks, and scalar origins are written into `PreparedObs`.
pub fn build_compact_rollout_obs(
    items: &mut [PreparedObs],
    device: Device,
    pinned_h2d: bool,
    fp16_rollout: bool,
    ae: &AePair,
    terrain_cache: &mut TerrainDeviceCache,
) -> anyhow::Result<Obs> {
    let compact = {
        let refs: Vec<&PreparedObs> = items.iter().collect();
        let full = build_device_ae_obs(
            &refs,
            device,
            pinned_h2d,
            fp16_rollout,
            ae,
            Some(terrain_cache),
        )?;
        PolicyNet::compact_observation(&full)
    };
    store_compact_host(items, &compact)?;
    Ok(compact)
}

/// Compact an observation for immediate actor/evaluation/bootstrap use
/// without retaining any host rollout copy.
pub fn build_compact_obs_with_ae(
    items: &[&PreparedObs],
    device: Device,
    pinned_h2d: bool,
    fp16_rollout: bool,
    ae: &AePair,
    terrain_cache: &mut TerrainDeviceCache,
) -> anyhow::Result<Obs> {
    let full = build_device_ae_obs(
        items,
        device,
        pinned_h2d,
        fp16_rollout,
        ae,
        Some(terrain_cache),
    )?;
    Ok(PolicyNet::compact_observation(&full))
}

/// Rollout-only builder. Uniform batches retain AE output on `device` for
/// inference while copying the latent once to the host-backed rollout
/// buffer. Mixed shapes keep the established padding/general path.
pub fn build_rollout_obs(
    items: &mut [PreparedObs],
    device: Device,
    pinned_h2d: bool,
    fp16_rollout: bool,
    ae: &AePair,
    terrain_cache: &mut TerrainDeviceCache,
) -> anyhow::Result<Obs> {
    let uniform = !items.is_empty()
        && items
            .iter()
            .all(|it| it.gh == items[0].gh && it.gw == items[0].gw);
    if !uniform {
        encode_prepared_obs(items, device, ae, terrain_cache)?;
        let refs: Vec<&PreparedObs> = items.iter().collect();
        return Ok(build_obs(&refs, device, pinned_h2d, fp16_rollout));
    }

    let (resident, fine_host, coarse_host) = {
        let refs: Vec<&PreparedObs> = items.iter().collect();
        let resident = encode_uniform_ae_grids(
            &refs,
            device,
            pinned_h2d,
            fp16_rollout,
            ae,
            Some(terrain_cache),
        )?;
        let fine_host = host_grids_from_latent(&refs, &resident.fine_latent, false)?;
        let coarse_host = resident
            .coarse_latent
            .as_ref()
            .map(|z| host_grids_from_latent(&refs, z, true))
            .transpose()?;
        (resident, fine_host, coarse_host)
    };
    if let Some(coarse_host) = coarse_host {
        for ((it, fine), coarse) in items
            .iter_mut()
            .zip(fine_host.into_iter())
            .zip(coarse_host.into_iter())
        {
            it.grid = Some(fine);
            it.grid_coarse = Some(coarse);
            it.cgh = it.gh.div_ceil(2);
            it.cgw = it.gw.div_ceil(2);
        }
    } else {
        for (it, fine) in items.iter_mut().zip(fine_host) {
            it.grid = Some(fine);
            it.grid_coarse = None;
            it.cgh = 0;
            it.cgw = 0;
        }
    }
    let refs: Vec<&PreparedObs> = items.iter().collect();
    build_obs_from_parts(
        &refs,
        device,
        pinned_h2d,
        fp16_rollout,
        None,
        None,
        Some((resident.fine, resident.coarse)),
    )
}

pub fn build_obs_with_ae(
    items: &[&PreparedObs],
    device: Device,
    pinned_h2d: bool,
    fp16_rollout: bool,
    ae: Option<&AePair>,
) -> anyhow::Result<Obs> {
    build_obs_with_ae_cache(items, device, pinned_h2d, fp16_rollout, ae, None)
}

pub fn build_obs_with_ae_cached(
    items: &[&PreparedObs],
    device: Device,
    pinned_h2d: bool,
    fp16_rollout: bool,
    ae: &AePair,
    terrain_cache: &mut TerrainDeviceCache,
) -> anyhow::Result<Obs> {
    build_obs_with_ae_cache(
        items,
        device,
        pinned_h2d,
        fp16_rollout,
        Some(ae),
        Some(terrain_cache),
    )
}

fn build_obs_with_ae_cache(
    items: &[&PreparedObs],
    device: Device,
    pinned_h2d: bool,
    fp16_rollout: bool,
    ae: Option<&AePair>,
    mut terrain_cache: Option<&mut TerrainDeviceCache>,
) -> anyhow::Result<Obs> {
    let uniform = !items.is_empty()
        && items
            .iter()
            .all(|it| it.gh == items[0].gh && it.gw == items[0].gw);
    if uniform {
        if let Some(pair) = ae {
            let resident = encode_uniform_ae_grids(
                items,
                device,
                pinned_h2d,
                fp16_rollout,
                pair,
                terrain_cache.as_deref_mut(),
            )?;
            return build_obs_from_parts(
                items,
                device,
                pinned_h2d,
                fp16_rollout,
                None,
                None,
                Some((resident.fine, resident.coarse)),
            );
        }
    }

    let assembled = if ae.is_some() {
        assemble_grids(items, device, ae, terrain_cache.as_deref_mut())?
    } else {
        items
            .iter()
            .map(|it| {
                it.grid.clone().ok_or_else(|| {
                    anyhow::anyhow!("PreparedObs.grid missing — call encode_prepared_obs first")
                })
            })
            .collect::<anyhow::Result<Vec<_>>>()?
    };
    let coarse_assembled = if ae.is_some() {
        assemble_coarse_grids(items, device, ae, terrain_cache.as_deref_mut())?
    } else if items.iter().all(|it| it.grid_coarse.is_some()) {
        Some(
            items
                .iter()
                .map(|it| it.grid_coarse.clone().unwrap())
                .collect(),
        )
    } else {
        None
    };
    build_obs_from_parts(
        items,
        device,
        pinned_h2d,
        fp16_rollout,
        Some(assembled),
        coarse_assembled,
        None,
    )
}

fn build_obs_from_parts(
    items: &[&PreparedObs],
    device: Device,
    pinned_h2d: bool,
    fp16_rollout: bool,
    assembled: Option<Vec<Vec<f32>>>,
    coarse_assembled: Option<Vec<Vec<f32>>>,
    resident: Option<(Tensor, Option<Tensor>)>,
) -> anyhow::Result<Obs> {
    let b = items.len();
    let gh = items.iter().map(|it| it.gh).max().unwrap_or(0);
    let gw = items.iter().map(|it| it.gw).max().unwrap_or(0);
    let uniform = items.iter().all(|it| it.gh == gh && it.gw == gw);
    let c_grid = policy::C_GRID as usize;
    let resident_grid = resident.is_some();

    let mut grid = Vec::with_capacity(b * c_grid * gh * gw);
    let mut legal_tile = Vec::with_capacity(b * gh * gw);
    let mut grid_valid = Vec::with_capacity(b * gh * gw);
    let mut players = Vec::with_capacity(b * feat::MAX_SLOTS * feat::P_FEAT);
    let mut pmask = Vec::with_capacity(b * feat::MAX_SLOTS);
    let mut local = Vec::with_capacity(b * 5 * policy::LOCAL as usize * policy::LOCAL as usize);
    let mut scalars = Vec::with_capacity(b * feat::N_SCALARS);
    let mut legal_actions = Vec::with_capacity(b * feat::N_ACTIONS);
    let mut legal_ptarget = Vec::with_capacity(b * feat::N_ACTIONS * feat::MAX_SLOTS);
    let mut legal_build = Vec::with_capacity(b * feat::N_BUILD);
    let mut legal_nuke = Vec::with_capacity(b * feat::N_NUKE);

    for (idx, it) in items.iter().enumerate() {
        if !resident_grid {
            let g_src = &assembled.as_ref().expect("host grid required")[idx];
            if uniform {
                grid.extend_from_slice(g_src);
            } else {
                let mut g = vec![0.0f32; c_grid * gh * gw];
                for c in 0..c_grid {
                    for y in 0..it.gh {
                        let src_off = c * it.gh * it.gw + y * it.gw;
                        let dst_off = c * gh * gw + y * gw;
                        g[dst_off..dst_off + it.gw]
                            .copy_from_slice(&g_src[src_off..src_off + it.gw]);
                    }
                }
                grid.extend_from_slice(&g);
            }
        }
        if uniform {
            legal_tile.extend_from_slice(&it.legal_tile);
            grid_valid.extend(std::iter::repeat_n(1.0f32, gh * gw));
        } else {
            let mut lt = vec![0.0f32; gh * gw];
            let mut gv = vec![0.0f32; gh * gw];
            for y in 0..it.gh {
                let src_off = y * it.gw;
                let dst_off = y * gw;
                lt[dst_off..dst_off + it.gw]
                    .copy_from_slice(&it.legal_tile[src_off..src_off + it.gw]);
                gv[dst_off..dst_off + it.gw].fill(1.0);
            }
            legal_tile.extend_from_slice(&lt);
            grid_valid.extend_from_slice(&gv);
        }
        players.extend_from_slice(&it.players);
        pmask.extend_from_slice(&it.pmask);
        local.extend_from_slice(&it.local);
        scalars.extend_from_slice(&it.scalars);
        legal_actions.extend_from_slice(&it.legal_actions);
        legal_ptarget.extend_from_slice(&it.legal_ptarget);
        legal_build.extend_from_slice(&it.legal_build);
        legal_nuke.extend_from_slice(&it.legal_nuke);
    }

    let t = |v: Vec<f32>, shape: &[i64]| -> Tensor {
        let cpu = Tensor::from_slice(&v).view(shape).to_kind(Kind::Float);
        to_device_maybe_pinned(&cpu, device, pinned_h2d)
    };
    // Fine/coarse AE grids: optional Half H2D when `--fp16-rollout`.
    let t_grid = |v: Vec<f32>, shape: &[i64]| -> Tensor {
        upload_float_grid(&v, shape, device, pinned_h2d, fp16_rollout)
    };
    let bi = b as i64;
    let (ghi, gwi) = (gh as i64, gw as i64);
    let ms = feat::MAX_SLOTS as i64;
    let na = feat::N_ACTIONS as i64;
    let local_sz = policy::LOCAL;

    let (resident_fine, resident_coarse) = match resident {
        Some((fine, coarse)) => (Some(fine), coarse),
        None => (None, None),
    };
    let grid_coarse = if resident_grid {
        resident_coarse
    } else if let Some(coarse_items) = coarse_assembled {
        let cgh = items.iter().map(|it| it.gh.div_ceil(2)).max().unwrap_or(0);
        let cgw = items.iter().map(|it| it.gw.div_ceil(2)).max().unwrap_or(0);
        let mut packed = Vec::with_capacity(b * c_grid * cgh * cgw);
        let uniform_c = items
            .iter()
            .all(|it| it.gh.div_ceil(2) == cgh && it.gw.div_ceil(2) == cgw);
        for (idx, it) in items.iter().enumerate() {
            let src = &coarse_items[idx];
            let igh = it.gh.div_ceil(2);
            let igw = it.gw.div_ceil(2);
            if uniform_c {
                packed.extend_from_slice(src);
            } else {
                let mut g = vec![0.0f32; c_grid * cgh * cgw];
                for c in 0..c_grid {
                    for y in 0..igh {
                        let src_off = c * igh * igw + y * igw;
                        let dst_off = c * cgh * cgw + y * cgw;
                        g[dst_off..dst_off + igw].copy_from_slice(&src[src_off..src_off + igw]);
                    }
                }
                packed.extend_from_slice(&g);
            }
        }
        Some(t_grid(
            packed,
            &[bi, policy::C_GRID, cgh as i64, cgw as i64],
        ))
    } else {
        None
    };

    Ok(Obs {
        grid: resident_fine.unwrap_or_else(|| t_grid(grid, &[bi, policy::C_GRID, ghi, gwi])),
        grid_valid: t(grid_valid, &[bi, ghi, gwi]),
        legal_tile: t(legal_tile, &[bi, ghi, gwi]),
        grid_coarse,
        players: t(players, &[bi, ms, feat::P_FEAT as i64]),
        pmask: t(pmask, &[bi, ms]),
        local: t(local, &[bi, 5, local_sz, local_sz]),
        scalars: t(scalars, &[bi, feat::N_SCALARS as i64]),
        legal_actions: t(legal_actions, &[bi, na]),
        legal_ptarget: t(legal_ptarget, &[bi, na, ms]),
        legal_build: t(legal_build, &[bi, feat::N_BUILD as i64]),
        legal_nuke: t(legal_nuke, &[bi, feat::N_NUKE as i64]),
        compact: None,
    })
}

fn tensor_vec_f16(t: &Tensor) -> anyhow::Result<Vec<half::f16>> {
    Vec::<half::f16>::try_from(t.to_device(Device::Cpu).to_kind(Kind::Half).reshape([-1]))
        .map_err(|e| anyhow::anyhow!("compact fp16 host copy failed: {e}"))
}

fn tensor_vec_f32(t: &Tensor) -> anyhow::Result<Vec<f32>> {
    Vec::<f32>::try_from(t.to_device(Device::Cpu).to_kind(Kind::Float).reshape([-1]))
        .map_err(|e| anyhow::anyhow!("compact mask host copy failed: {e}"))
}

fn store_compact_host(items: &mut [PreparedObs], obs: &Obs) -> anyhow::Result<()> {
    let meta = obs.compact.as_ref().expect("compact metadata");
    let (b, _, fh, fw) = obs.grid.size4()?;
    let coarse = obs.grid_coarse.as_ref().expect("compact coarse grid");
    let (_, _, ch, cw) = coarse.size4()?;
    anyhow::ensure!(
        b as usize == items.len(),
        "compact batch/item length mismatch"
    );

    // One D2H per payload, then split on CPU. No Tensor escapes this actor call.
    let fine = tensor_vec_f16(&obs.grid)?;
    let coarse = tensor_vec_f16(coarse)?;
    let fine_valid = tensor_vec_f32(&obs.grid_valid)?;
    let fine_legal = tensor_vec_f32(&obs.legal_tile)?;
    let coarse_valid = tensor_vec_f32(&meta.coarse_valid)?;
    let coarse_legal = tensor_vec_f32(&meta.coarse_legal)?;
    let origin_y: Vec<i64> = Vec::try_from(meta.origin_y.to_device(Device::Cpu))?;
    let origin_x: Vec<i64> = Vec::try_from(meta.origin_x.to_device(Device::Cpu))?;
    let fine_n = policy::C_GRID as usize * fh as usize * fw as usize;
    let fine_mask_n = fh as usize * fw as usize;
    let coarse_n = policy::C_GRID as usize * ch as usize * cw as usize;
    let coarse_mask_n = ch as usize * cw as usize;

    for (i, it) in items.iter_mut().enumerate() {
        it.compact = Some(CompactGrid {
            fine: fine[i * fine_n..(i + 1) * fine_n].to_vec(),
            fine_valid: fine_valid[i * fine_mask_n..(i + 1) * fine_mask_n].to_vec(),
            fine_legal: fine_legal[i * fine_mask_n..(i + 1) * fine_mask_n].to_vec(),
            fine_h: fh as usize,
            fine_w: fw as usize,
            origin_y: origin_y[i],
            origin_x: origin_x[i],
            coarse: coarse[i * coarse_n..(i + 1) * coarse_n].to_vec(),
            coarse_valid: coarse_valid[i * coarse_mask_n..(i + 1) * coarse_mask_n].to_vec(),
            coarse_legal: coarse_legal[i * coarse_mask_n..(i + 1) * coarse_mask_n].to_vec(),
            coarse_h: ch as usize,
            coarse_w: cw as usize,
        });
        // These full-resolution buffers are actor inputs, not rollout data.
        it.grid = None;
        it.grid_coarse = None;
        it.ae_raw.owners.clear();
        it.ae_raw.static_terrain.land_mag = Vec::<f32>::new().into();
        it.ae_raw.fallout.clear();
        it.ae_raw.stat.clear();
        it.ego.clear();
        it.db.clear();
        it.transient.clear();
        it.legal_tile.clear();
    }
    Ok(())
}

fn build_compact_host_obs(
    items: &[&PreparedObs],
    device: Device,
    pinned_h2d: bool,
) -> anyhow::Result<Obs> {
    anyhow::ensure!(!items.is_empty(), "empty compact observation batch");
    let compact: Vec<&CompactGrid> = items
        .iter()
        .map(|it| {
            it.compact
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("missing compact grid"))
        })
        .collect::<anyhow::Result<_>>()?;
    let fh = compact[0].fine_h;
    let fw = compact[0].fine_w;
    anyhow::ensure!(
        compact.iter().all(|c| c.fine_h == fh && c.fine_w == fw),
        "compact fine windows must have a uniform shape"
    );
    let ch = compact.iter().map(|c| c.coarse_h).max().unwrap();
    let cw = compact.iter().map(|c| c.coarse_w).max().unwrap();
    let b = items.len();
    let cg = policy::C_GRID as usize;

    let mut fine = Vec::with_capacity(b * cg * fh * fw);
    let mut fine_valid = Vec::with_capacity(b * fh * fw);
    let mut fine_legal = Vec::with_capacity(b * fh * fw);
    let mut coarse = Vec::with_capacity(b * cg * ch * cw);
    let mut coarse_valid = Vec::with_capacity(b * ch * cw);
    let mut coarse_legal = Vec::with_capacity(b * ch * cw);
    for c in &compact {
        fine.extend_from_slice(&c.fine);
        fine_valid.extend_from_slice(&c.fine_valid);
        fine_legal.extend_from_slice(&c.fine_legal);
        for plane in 0..cg {
            let mut dst = vec![half::f16::ZERO; ch * cw];
            for y in 0..c.coarse_h {
                let src = plane * c.coarse_h * c.coarse_w + y * c.coarse_w;
                let out = y * cw;
                dst[out..out + c.coarse_w].copy_from_slice(&c.coarse[src..src + c.coarse_w]);
            }
            coarse.extend(dst);
        }
        for (src, dst) in [
            (&c.coarse_valid, &mut coarse_valid),
            (&c.coarse_legal, &mut coarse_legal),
        ] {
            let mut padded = vec![0.0f32; ch * cw];
            for y in 0..c.coarse_h {
                padded[y * cw..y * cw + c.coarse_w]
                    .copy_from_slice(&src[y * c.coarse_w..(y + 1) * c.coarse_w]);
            }
            dst.extend(padded);
        }
    }

    let half_up = |v: Vec<half::f16>, shape: &[i64]| {
        to_device_maybe_pinned(&Tensor::from_slice(&v).view(shape), device, pinned_h2d)
            .to_kind(Kind::Float)
    };
    let up = |v: Vec<f32>, shape: &[i64]| {
        to_device_maybe_pinned(&Tensor::from_slice(&v).view(shape), device, pinned_h2d)
    };
    let mut players = Vec::new();
    let mut pmask = Vec::new();
    let mut local = Vec::new();
    let mut scalars = Vec::new();
    let mut legal_actions = Vec::new();
    let mut legal_ptarget = Vec::new();
    let mut legal_build = Vec::new();
    let mut legal_nuke = Vec::new();
    for it in items {
        players.extend_from_slice(&it.players);
        pmask.extend_from_slice(&it.pmask);
        local.extend_from_slice(&it.local);
        scalars.extend_from_slice(&it.scalars);
        legal_actions.extend_from_slice(&it.legal_actions);
        legal_ptarget.extend_from_slice(&it.legal_ptarget);
        legal_build.extend_from_slice(&it.legal_build);
        legal_nuke.extend_from_slice(&it.legal_nuke);
    }
    let bi = b as i64;
    Ok(Obs {
        grid: half_up(fine, &[bi, policy::C_GRID, fh as i64, fw as i64]),
        grid_valid: up(fine_valid, &[bi, fh as i64, fw as i64]),
        legal_tile: up(fine_legal, &[bi, fh as i64, fw as i64]),
        grid_coarse: Some(half_up(coarse, &[bi, policy::C_GRID, ch as i64, cw as i64])),
        players: up(players, &[bi, feat::MAX_SLOTS as i64, feat::P_FEAT as i64]),
        pmask: up(pmask, &[bi, feat::MAX_SLOTS as i64]),
        local: up(local, &[bi, 5, policy::LOCAL, policy::LOCAL]),
        scalars: up(scalars, &[bi, feat::N_SCALARS as i64]),
        legal_actions: up(legal_actions, &[bi, feat::N_ACTIONS as i64]),
        legal_ptarget: up(
            legal_ptarget,
            &[bi, feat::N_ACTIONS as i64, feat::MAX_SLOTS as i64],
        ),
        legal_build: up(legal_build, &[bi, feat::N_BUILD as i64]),
        legal_nuke: up(legal_nuke, &[bi, feat::N_NUKE as i64]),
        compact: Some(CompactObsMeta {
            origin_y: to_device_maybe_pinned(
                &Tensor::from_slice(&compact.iter().map(|c| c.origin_y).collect::<Vec<_>>()),
                device,
                pinned_h2d,
            ),
            origin_x: to_device_maybe_pinned(
                &Tensor::from_slice(&compact.iter().map(|c| c.origin_x).collect::<Vec<_>>()),
                device,
                pinned_h2d,
            ),
            coarse_valid: up(coarse_valid, &[bi, ch as i64, cw as i64]),
            coarse_legal: up(coarse_legal, &[bi, ch as i64, cw as i64]),
        }),
    })
}

/// Builds a `ChoiceBatch` from previously-sampled per-env choices (stored
/// as plain scalars during rollout collection).
#[derive(Clone)]
pub struct ChoiceScalars {
    pub action: i64,
    pub player_slot: i64,   // -1 unused
    pub tile_region: i64,   // -1 unused
    pub build_type: i64,    // -1 unused
    pub nuke_type: i64,     // -1 unused
    pub quantity_frac: f32, // -1.0 unused
}

pub fn build_choice_batch(
    items: &[ChoiceScalars],
    device: Device,
    pinned_h2d: bool,
) -> policy::ChoiceBatch {
    let action: Vec<i64> = items.iter().map(|c| c.action).collect();
    let player_slot: Vec<i64> = items.iter().map(|c| c.player_slot).collect();
    let tile_region: Vec<i64> = items.iter().map(|c| c.tile_region).collect();
    let build_type: Vec<i64> = items.iter().map(|c| c.build_type).collect();
    let nuke_type: Vec<i64> = items.iter().map(|c| c.nuke_type).collect();
    let quantity_frac: Vec<f32> = items.iter().map(|c| c.quantity_frac).collect();
    let up = |t: Tensor| to_device_maybe_pinned(&t, device, pinned_h2d);
    policy::ChoiceBatch {
        action: up(Tensor::from_slice(&action)),
        player_slot: up(Tensor::from_slice(&player_slot)),
        tile_region: up(Tensor::from_slice(&tile_region)),
        build_type: up(Tensor::from_slice(&build_type)),
        nuke_type: up(Tensor::from_slice(&nuke_type)),
        quantity_frac: up(Tensor::from_slice(&quantity_frac)),
    }
}

#[cfg(test)]
mod tests {
    //! `--pinned-h2d` only changes behavior when `device.is_cuda()` (see
    //! `to_device_maybe_pinned`), which this CPU-only dev environment
    //! can't exercise - these tests instead pin down that turning the
    //! flag on for a CPU device is a no-op producing byte-identical
    //! tensors to the flag being off, so the plumbing is provably inert
    //! (not just untested) until it's A/B'd on a real CUDA box.
    use super::*;

    fn tiny_prepared_obs(gh: usize, gw: usize) -> PreparedObs {
        let plane = gh * gw;
        PreparedObs {
            compact: None,
            grid: Some(vec![0.5f32; policy::C_GRID as usize * plane]),
            grid_coarse: None,
            cgh: 0,
            cgw: 0,
            ae_raw: crate::ae::AeRaw {
                owners: vec![0i64; (gh * 8) * (gw * 8)],
                static_terrain: crate::ae::StaticTerrain {
                    key: crate::ae::TerrainCacheKey {
                        env_id: 0,
                        episode: 0,
                        static_id: 1,
                        hr: gh * 8,
                        wr: gw * 8,
                    },
                    map: std::sync::Arc::from("test"),
                    land_mag: vec![0.0f32; 2 * (gh * 8) * (gw * 8)].into(),
                },
                fallout: vec![0.0f32; (gh * 8) * (gw * 8)],
                stat: vec![0.0f32; 6 * plane],
                hr: gh * 8,
                wr: gw * 8,
            },
            ego: vec![0.0f32; 3 * plane],
            db: vec![0.0f32; plane],
            transient: vec![0.0f32; feat::N_TRANSIENT * plane],
            legal_tile: vec![1.0f32; plane],
            gh,
            gw,
            players: vec![0.1f32; feat::MAX_SLOTS * feat::P_FEAT],
            pmask: [1.0f32; feat::MAX_SLOTS],
            scalars: [0.2f32; feat::N_SCALARS],
            me_slot: 0,
            legal_actions: [1.0f32; feat::N_ACTIONS],
            legal_ptarget: vec![1.0f32; feat::N_ACTIONS * feat::MAX_SLOTS],
            legal_build: [1.0f32; feat::N_BUILD],
            legal_nuke: [1.0f32; feat::N_NUKE],
            local: vec![0.3f32; 5 * policy::LOCAL as usize * policy::LOCAL as usize],
        }
    }

    #[test]
    fn pinned_h2d_flag_is_noop_on_cpu() {
        let items = vec![tiny_prepared_obs(4, 5), tiny_prepared_obs(4, 5)];
        let refs: Vec<&PreparedObs> = items.iter().collect();
        let plain = build_obs(&refs, Device::Cpu, false, false);
        let pinned = build_obs(&refs, Device::Cpu, true, false);
        assert_eq!(
            Vec::<f32>::try_from(plain.grid.reshape([-1])).unwrap(),
            Vec::<f32>::try_from(pinned.grid.reshape([-1])).unwrap()
        );
        assert_eq!(plain.grid.size(), pinned.grid.size());
        assert_eq!(
            Vec::<f32>::try_from(plain.legal_tile.reshape([-1])).unwrap(),
            Vec::<f32>::try_from(pinned.legal_tile.reshape([-1])).unwrap()
        );

        let choices = vec![
            ChoiceScalars {
                action: 1,
                player_slot: -1,
                tile_region: 3,
                build_type: -1,
                nuke_type: -1,
                quantity_frac: -1.0,
            },
            ChoiceScalars {
                action: 2,
                player_slot: 0,
                tile_region: -1,
                build_type: 1,
                nuke_type: -1,
                quantity_frac: 0.5,
            },
        ];
        let plain_c = build_choice_batch(&choices, Device::Cpu, false);
        let pinned_c = build_choice_batch(&choices, Device::Cpu, true);
        assert_eq!(
            Vec::<i64>::try_from(&plain_c.action).unwrap(),
            Vec::<i64>::try_from(&pinned_c.action).unwrap()
        );
        assert_eq!(
            Vec::<f32>::try_from(&plain_c.quantity_frac).unwrap(),
            Vec::<f32>::try_from(&pinned_c.quantity_frac).unwrap()
        );
    }

    /// Regression test for the exact crash this padding fix replaces: two
    /// envs in one batch on differently-sized maps (any curriculum stage
    /// past stage 1, or any stage hit by rehearsal sampling into a
    /// different-shaped past map) used to panic
    /// ("obs N grid shape mismatch (v1 requires uniform batch shape)")
    /// instead of training. Verifies the smaller item is zero-padded
    /// top-left-aligned into the batch max shape and `grid_valid` is 1
    /// only over its real region.
    #[test]
    fn mixed_shape_batch_pads_instead_of_panicking() {
        let small = tiny_prepared_obs(2, 3); // filled with 0.5/1.0 by the helper
        let mut big = tiny_prepared_obs(4, 5);
        // Distinct fill so we can tell "big's real data" apart from "small's
        // real data" apart from "zero padding" unambiguously.
        big.grid.as_mut().unwrap().iter_mut().for_each(|v| *v = 9.0);
        big.legal_tile.iter_mut().for_each(|v| *v = 9.0);

        let items = vec![&small, &big];
        let obs = build_obs(&items, Device::Cpu, false, false);

        assert_eq!(obs.grid.size(), [2, policy::C_GRID, 4, 5]);
        assert_eq!(obs.grid_valid.size(), [2, 4, 5]);

        let gv: Vec<f32> = Vec::try_from(obs.grid_valid.reshape([-1])).unwrap();
        let gv = |b: usize, y: usize, x: usize| gv[b * 4 * 5 + y * 5 + x];
        // Item 0 (small, real 2x3): valid inside, zero outside.
        for y in 0..4 {
            for x in 0..5 {
                assert_eq!(
                    gv(0, y, x),
                    if y < 2 && x < 3 { 1.0 } else { 0.0 },
                    "item0 valid[{y}][{x}]"
                );
            }
        }
        // Item 1 (big, real 4x5 == the batch max): valid everywhere.
        for y in 0..4 {
            for x in 0..5 {
                assert_eq!(gv(1, y, x), 1.0, "item1 valid[{y}][{x}]");
            }
        }

        let grid: Vec<f32> = Vec::try_from(obs.grid.reshape([-1])).unwrap();
        let plane = 4 * 5;
        let grid_at = |b: usize, c: usize, y: usize, x: usize| {
            grid[b * policy::C_GRID as usize * plane + c * plane + y * 5 + x]
        };
        // Item 0's real cells keep their value (0.5); padded cells are 0.
        assert_eq!(grid_at(0, 0, 0, 0), 0.5);
        assert_eq!(grid_at(0, 0, 1, 2), 0.5);
        assert_eq!(grid_at(0, 0, 0, 3), 0.0); // outside small's gw=3
        assert_eq!(grid_at(0, 0, 3, 0), 0.0); // outside small's gh=2
        // Item 1 fills the whole max shape, no padding needed.
        assert_eq!(grid_at(1, 0, 3, 4), 9.0);

        let lt: Vec<f32> = Vec::try_from(obs.legal_tile.reshape([-1])).unwrap();
        assert_eq!(lt[0 * plane + 0 * 5 + 0], 1.0); // small's real cell
        assert_eq!(lt[0 * plane + 3 * 5 + 4], 0.0); // padded corner
        assert_eq!(lt[1 * plane + 3 * 5 + 4], 9.0); // big's real corner
    }

    #[test]
    fn uniform_resident_fine_and_coarse_match_host_fallback() {
        let mut items = vec![tiny_prepared_obs(4, 4), tiny_prepared_obs(4, 4)];
        for (bi, it) in items.iter_mut().enumerate() {
            for (i, x) in it.ego.iter_mut().enumerate() {
                *x = bi as f32 * 100.0 + i as f32;
            }
            for (i, x) in it.db.iter_mut().enumerate() {
                *x = bi as f32 * 10.0 + i as f32 * 0.25;
            }
            for (i, x) in it.transient.iter_mut().enumerate() {
                *x = bi as f32 * 3.0 - i as f32 * 0.01;
            }
        }
        let fine_values: Vec<f32> = (0..2 * policy::LATENT_C as usize * 4 * 4)
            .map(|i| i as f32 * 0.5)
            .collect();
        let coarse_values: Vec<f32> = (0..2 * policy::LATENT_C as usize * 2 * 2)
            .map(|i| 1000.0 + i as f32)
            .collect();
        let fine_latent = Tensor::from_slice(&fine_values).view([2, policy::LATENT_C, 4, 4]);
        let coarse_latent = Tensor::from_slice(&coarse_values).view([2, policy::LATENT_C, 2, 2]);

        let (fine_resident, coarse_resident, fine_host, coarse_host) = {
            let refs: Vec<&PreparedObs> = items.iter().collect();
            (
                assemble_uniform_device_grid(
                    &refs,
                    &fine_latent,
                    4,
                    4,
                    Device::Cpu,
                    false,
                    false,
                    false,
                ),
                assemble_uniform_device_grid(
                    &refs,
                    &coarse_latent,
                    2,
                    2,
                    Device::Cpu,
                    false,
                    false,
                    true,
                ),
                host_grids_from_latent(&refs, &fine_latent, false).unwrap(),
                host_grids_from_latent(&refs, &coarse_latent, true).unwrap(),
            )
        };
        for (i, it) in items.iter_mut().enumerate() {
            it.grid = Some(fine_host[i].clone());
            it.grid_coarse = Some(coarse_host[i].clone());
            it.cgh = 2;
            it.cgw = 2;
        }
        let refs: Vec<&PreparedObs> = items.iter().collect();
        let fallback = build_obs(&refs, Device::Cpu, false, false);
        let fast = build_obs_from_parts(
            &refs,
            Device::Cpu,
            false,
            false,
            None,
            None,
            Some((fine_resident, Some(coarse_resident))),
        )
        .unwrap();

        let values = |t: &Tensor| Vec::<f32>::try_from(t.reshape([-1])).unwrap();
        assert_eq!(fast.grid.size(), [2, policy::C_GRID, 4, 4]);
        assert_eq!(values(&fast.grid), values(&fallback.grid));
        assert_eq!(
            fast.grid_coarse.as_ref().unwrap().size(),
            [2, policy::C_GRID, 2, 2]
        );
        assert_eq!(
            values(fast.grid_coarse.as_ref().unwrap()),
            values(fallback.grid_coarse.as_ref().unwrap())
        );
        assert_eq!(values(&fast.grid_valid), values(&fallback.grid_valid));
        assert_eq!(values(&fast.legal_tile), values(&fallback.legal_tile));
        assert_eq!(values(&fast.legal_actions), values(&fallback.legal_actions));
        assert_eq!(values(&fast.legal_ptarget), values(&fallback.legal_ptarget));
        assert_eq!(values(&fast.legal_build), values(&fallback.legal_build));
        assert_eq!(values(&fast.legal_nuke), values(&fallback.legal_nuke));
    }

    /// `--fp16-rollout` on CPU is a no-op (Half H2D only fires on CUDA),
    /// but still exercises the encode→build path and requires finite Obs.
    #[test]
    fn fp16_rollout_flag_produces_finite_obs_on_cpu() {
        let items = vec![tiny_prepared_obs(4, 5)];
        let refs: Vec<&PreparedObs> = items.iter().collect();
        let obs = build_obs(&refs, Device::Cpu, false, true);
        let grid: Vec<f32> = Vec::try_from(obs.grid.reshape([-1])).unwrap();
        assert!(grid.iter().all(|v| v.is_finite()), "grid must be finite");
        assert_eq!(obs.grid.size(), [1, policy::C_GRID, 4, 5]);
        // CPU path must match flag-off byte-for-byte (no Half round-trip).
        let plain = build_obs(&refs, Device::Cpu, false, false);
        assert_eq!(
            grid,
            Vec::<f32>::try_from(plain.grid.reshape([-1])).unwrap()
        );
    }

    /// Exercises the complete host ownership boundary with mixed source
    /// shapes and rollout ordering. Expected grids are explicitly fp16
    /// quantized because that is the compact storage contract.
    #[test]
    fn compact_host_roundtrip_matches_device_foveation_and_preserves_order() {
        let mut items = vec![
            tiny_prepared_obs(52, 58),
            tiny_prepared_obs(56, 60),
            tiny_prepared_obs(50, 54),
            tiny_prepared_obs(54, 56),
        ];
        for (i, it) in items.iter_mut().enumerate() {
            it.scalars[0] = 100.0 + i as f32; // T-major/N-minor identity.
            let mine = policy::EGO_OWN_CH as usize * it.gh * it.gw;
            it.grid.as_mut().unwrap()[mine + (2 + i) * it.gw + (3 + i)] = 1.0;
            for y in 0..it.gh {
                for x in 0..it.gw {
                    it.legal_tile[y * it.gw + x] = ((x + y + i) % 3 != 0) as u8 as f32;
                }
            }
        }
        let full = {
            let refs: Vec<&PreparedObs> = items.iter().collect();
            build_obs(&refs, Device::Cpu, false, false)
        };
        let direct = PolicyNet::compact_observation(&full);
        store_compact_host(&mut items, &direct).unwrap();

        assert!(items.iter().all(|it| {
            it.compact.is_some()
                && it.grid.is_none()
                && it.ae_raw.owners.is_empty()
                && it.ego.is_empty()
                && it.transient.is_empty()
                && it.legal_tile.is_empty()
        }));
        let refs: Vec<&PreparedObs> = items.iter().collect();
        let rebuilt = build_obs(&refs, Device::Cpu, false, false);
        assert!(rebuilt.compact.is_some());
        assert_eq!(
            rebuilt.grid.size(),
            [
                4,
                policy::C_GRID,
                policy::FOVEATE_SIZE,
                policy::FOVEATE_SIZE
            ]
        );

        let quantized = |t: &Tensor| t.to_kind(Kind::Half).to_kind(Kind::Float);
        let diff = |a: &Tensor, b: &Tensor| (a - b).abs().max().double_value(&[]);
        assert_eq!(diff(&quantized(&direct.grid), &rebuilt.grid), 0.0);
        assert_eq!(
            diff(
                &quantized(direct.grid_coarse.as_ref().unwrap()),
                rebuilt.grid_coarse.as_ref().unwrap()
            ),
            0.0
        );
        assert_eq!(diff(&direct.grid_valid, &rebuilt.grid_valid), 0.0);
        assert_eq!(diff(&direct.legal_tile, &rebuilt.legal_tile), 0.0);
        let dm = direct.compact.as_ref().unwrap();
        let rm = rebuilt.compact.as_ref().unwrap();
        assert_eq!(diff(&dm.coarse_valid, &rm.coarse_valid), 0.0);
        assert_eq!(diff(&dm.coarse_legal, &rm.coarse_legal), 0.0);
        assert_eq!(
            Vec::<i64>::try_from(&dm.origin_y).unwrap(),
            Vec::<i64>::try_from(&rm.origin_y).unwrap()
        );
        assert_eq!(
            Vec::<i64>::try_from(&dm.origin_x).unwrap(),
            Vec::<i64>::try_from(&rm.origin_x).unwrap()
        );
        assert_eq!(
            Vec::<f32>::try_from(rebuilt.scalars.select(1, 0)).unwrap(),
            vec![100.0, 101.0, 102.0, 103.0]
        );

        let compact_grid_bytes: usize = items
            .iter()
            .map(|it| {
                let c = it.compact.as_ref().unwrap();
                2 * (c.fine.len() + c.coarse.len())
            })
            .sum();
        let old_fine_bytes: usize = [52usize * 58, 56 * 60, 50 * 54, 54 * 56]
            .into_iter()
            .map(|p| 4 * policy::C_GRID as usize * p)
            .sum();
        assert!(
            compact_grid_bytes < old_fine_bytes,
            "compact fp16 grids must reduce rollout bytes ({compact_grid_bytes} vs {old_fine_bytes})"
        );
    }
}
