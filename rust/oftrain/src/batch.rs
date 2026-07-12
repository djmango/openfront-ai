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

use crate::ae::{self, AePair};
use crate::policy::{self, Obs};
use crate::vecenv::PreparedObs;

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
        t.pin_memory(device).to_device_(device, t.kind(), true, false)
    } else {
        t.to_device(device)
    }
}

/// Host f32 slice → device Float tensor. When `fp16_rollout` and the
/// destination is CUDA, stage as Half on the host, H2D as Half (half the
/// PCIe bytes), then `.to_kind(Float)` on device. CPU / flag-off keep the
/// plain Float path (byte-identical to pre-flag behavior).
fn upload_float_grid(v: &[f32], shape: &[i64], device: Device, pinned: bool, fp16_rollout: bool) -> Tensor {
    let cpu = Tensor::from_slice(v).view(shape).to_kind(Kind::Float);
    if fp16_rollout && device.is_cuda() {
        let half = cpu.to_kind(Kind::Half);
        to_device_maybe_pinned(&half, device, pinned).to_kind(Kind::Float)
    } else {
        to_device_maybe_pinned(&cpu, device, pinned)
    }
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

/// Frozen-AE output retained on the shard device for one rollout sample.
/// The host-native feature planes remain in `PreparedObs`; only the tensors
/// that would otherwise make a GPU -> CPU -> GPU round trip live here.
pub struct ResidentLatents {
    pub fine: Tensor,
    pub coarse: Option<Tensor>,
}

/// Build the complete fine/coarse grid from resident per-item AE latents.
/// Items may have different shapes; tensors are top-left aligned and padded
/// exactly like the host fallback.
fn assemble_resident_device_grid(
    items: &[&PreparedObs],
    latents: &[&ResidentLatents],
    device: Device,
    pinned_h2d: bool,
    fp16_rollout: bool,
    coarse: bool,
) -> anyhow::Result<Tensor> {
    anyhow::ensure!(items.len() == latents.len(), "item/latent count mismatch");
    anyhow::ensure!(!items.is_empty(), "cannot assemble an empty resident batch");
    for (i, (it, z)) in items.iter().zip(latents).enumerate() {
        let plane = it.gh * it.gw;
        anyhow::ensure!(it.gh > 0 && it.gw > 0, "rollout item {i} has empty grid shape");
        anyhow::ensure!(it.ego.len() == 3 * plane, "rollout item {i} ego shape mismatch");
        anyhow::ensure!(it.db.len() == plane, "rollout item {i} db shape mismatch");
        anyhow::ensure!(
            it.transient.len() == feat::N_TRANSIENT * plane,
            "rollout item {i} transient shape mismatch"
        );
        anyhow::ensure!(
            z.fine.device() == device && z.fine.kind() == Kind::Float,
            "rollout item {i} fine latent is {:?}/{:?}, expected {:?}/Float",
            z.fine.device(),
            z.fine.kind(),
            device
        );
        anyhow::ensure!(
            z.fine.size() == [policy::LATENT_C, it.gh as i64, it.gw as i64],
            "rollout item {i} fine latent shape {:?}, expected [{}, {}, {}]",
            z.fine.size(),
            policy::LATENT_C,
            it.gh,
            it.gw
        );
        if let Some(coarse_z) = z.coarse.as_ref() {
            let (cgh, cgw) = (it.gh.div_ceil(2), it.gw.div_ceil(2));
            anyhow::ensure!(
                coarse_z.device() == device && coarse_z.kind() == Kind::Float,
                "rollout item {i} coarse latent is {:?}/{:?}, expected {:?}/Float",
                coarse_z.device(),
                coarse_z.kind(),
                device
            );
            anyhow::ensure!(
                coarse_z.size() == [policy::LATENT_C, cgh as i64, cgw as i64],
                "rollout item {i} coarse latent shape {:?}, expected [{}, {cgh}, {cgw}]",
                coarse_z.size(),
                policy::LATENT_C
            );
        }
    }
    let gh = items
        .iter()
        .map(|it| if coarse { it.gh.div_ceil(2) } else { it.gh })
        .max()
        .unwrap_or(0);
    let gw = items
        .iter()
        .map(|it| if coarse { it.gw.div_ceil(2) } else { it.gw })
        .max()
        .unwrap_or(0);
    let extra_c = policy::C_GRID as usize - policy::LATENT_C as usize;
    let uniform = items.iter().all(|it| {
        let (igh, igw) = if coarse {
            (it.gh.div_ceil(2), it.gw.div_ceil(2))
        } else {
            (it.gh, it.gw)
        };
        igh == gh && igw == gw
    });
    let mut extras = if uniform {
        Vec::with_capacity(items.len() * extra_c * gh * gw)
    } else {
        vec![0.0f32; items.len() * extra_c * gh * gw]
    };
    let mut padded_latents = Vec::with_capacity(items.len());
    for (bi, (it, z)) in items.iter().zip(latents).enumerate() {
        let (igh, igw) = if coarse {
            (it.gh.div_ceil(2), it.gw.div_ceil(2))
        } else {
            (it.gh, it.gw)
        };
        let mut src = Vec::with_capacity(extra_c * igh * igw);
        if coarse {
            src.extend(avg_pool2_planes(&it.ego, 3, it.gh, it.gw));
            src.extend(avg_pool2_planes(&it.db, 1, it.gh, it.gw));
            src.extend(max_pool2_planes(
                &it.transient,
                feat::N_TRANSIENT,
                it.gh,
                it.gw,
            ));
        } else if uniform {
            extras.extend_from_slice(&it.ego);
            extras.extend_from_slice(&it.db);
            extras.extend_from_slice(&it.transient);
        } else {
            src.extend_from_slice(&it.ego);
            src.extend_from_slice(&it.db);
            src.extend_from_slice(&it.transient);
        }
        if uniform && coarse {
            extras.extend_from_slice(&src);
        } else if !uniform {
            for c in 0..extra_c {
                for y in 0..igh {
                    let src_off = c * igh * igw + y * igw;
                    let dst_off = bi * extra_c * gh * gw + c * gh * gw + y * gw;
                    extras[dst_off..dst_off + igw].copy_from_slice(&src[src_off..src_off + igw]);
                }
            }
        }

        let z = if coarse {
            z.coarse
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("coarse latent missing for rollout item {bi}"))?
        } else {
            &z.fine
        };
        anyhow::ensure!(
            z.size() == [policy::LATENT_C, igh as i64, igw as i64],
            "rollout latent {bi} shape {:?}, expected [{}, {igh}, {igw}]",
            z.size(),
            policy::LATENT_C
        );
        padded_latents.push(if igh == gh && igw == gw {
            z.shallow_clone()
        } else {
            z.constant_pad_nd([0, (gw - igw) as i64, 0, (gh - igh) as i64])
        });
    }
    let extras = upload_float_grid(
        &extras,
        &[items.len() as i64, extra_c as i64, gh as i64, gw as i64],
        device,
        pinned_h2d,
        fp16_rollout,
    );
    let latent_refs: Vec<&Tensor> = padded_latents.iter().collect();
    let latent = Tensor::stack(&latent_refs, 0);
    // Match the old `--fp16-rollout` path, which quantized the complete
    // assembled grid during H2D, including the AE channels.
    let latent = if fp16_rollout && device.is_cuda() {
        latent.to_kind(Kind::Half).to_kind(Kind::Float)
    } else {
        latent.shallow_clone()
    };
    Ok(Tensor::cat(&[latent, extras], 1))
}

/// Encode and build an actor/eval observation without moving AE output to
/// the host. Returned latents own references to the same device storage and
/// can outlive the temporary full observation.
pub fn encode_rollout_obs(
    items: &[&PreparedObs],
    device: Device,
    pinned_h2d: bool,
    fp16_rollout: bool,
    pair: &AePair,
) -> anyhow::Result<(Obs, Vec<ResidentLatents>)> {
    anyhow::ensure!(!items.is_empty(), "cannot encode an empty rollout batch");
    let raws: Vec<&ae::AeRaw> = items.iter().map(|it| &it.ae_raw).collect();
    let fine = ae::encode_latent_batch_device(&pair.fine, &raws, device)?;
    let coarse = pair
        .coarse
        .as_ref()
        .map(|coarse| ae::encode_latent_batch_device(coarse, &raws, device))
        .transpose()?;
    let resident: Vec<ResidentLatents> = fine
        .into_iter()
        .enumerate()
        .map(|(i, fine)| ResidentLatents {
            fine,
            coarse: coarse.as_ref().map(|zs| zs[i].shallow_clone()),
        })
        .collect();
    let refs: Vec<&ResidentLatents> = resident.iter().collect();
    let obs = build_obs_with_resident_latents(items, &refs, device, pinned_h2d, fp16_rollout)?;
    Ok((obs, resident))
}

/// Learner-side batch construction. Only host-native planes/masks are
/// uploaded; fine/coarse AE channels remain on the shard GPU throughout.
pub fn build_obs_with_resident_latents(
    items: &[&PreparedObs],
    latents: &[&ResidentLatents],
    device: Device,
    pinned_h2d: bool,
    fp16_rollout: bool,
) -> anyhow::Result<Obs> {
    let fine =
        assemble_resident_device_grid(items, latents, device, pinned_h2d, fp16_rollout, false)?;
    let has_coarse = latents.iter().any(|z| z.coarse.is_some());
    anyhow::ensure!(
        !has_coarse || latents.iter().all(|z| z.coarse.is_some()),
        "inconsistent coarse latent presence in rollout"
    );
    let coarse = if has_coarse {
        Some(assemble_resident_device_grid(
            items,
            latents,
            device,
            pinned_h2d,
            fp16_rollout,
            true,
        )?)
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

#[cfg(test)]
fn resident_latents_from_host_grids(items: &[&PreparedObs]) -> Vec<ResidentLatents> {
    items
        .iter()
        .map(|it| {
            let plane = it.gh * it.gw;
            let fine = Tensor::from_slice(
                &it.grid.as_ref().unwrap()[..policy::LATENT_C as usize * plane],
            )
            .view([policy::LATENT_C, it.gh as i64, it.gw as i64]);
            let coarse = it.grid_coarse.as_ref().map(|grid| {
                let cgh = it.gh.div_ceil(2);
                let cgw = it.gw.div_ceil(2);
                Tensor::from_slice(&grid[..policy::LATENT_C as usize * cgh * cgw]).view([
                    policy::LATENT_C,
                    cgh as i64,
                    cgw as i64,
                ])
            });
            ResidentLatents { fine, coarse }
        })
        .collect()
}

pub fn build_obs(items: &[&PreparedObs], device: Device, pinned_h2d: bool, fp16_rollout: bool) -> Obs {
    build_obs_with_ae(items, device, pinned_h2d, fp16_rollout, None).expect("build_obs")
}

pub fn build_obs_with_ae(
    items: &[&PreparedObs],
    device: Device,
    pinned_h2d: bool,
    fp16_rollout: bool,
    ae: Option<&AePair>,
) -> anyhow::Result<Obs> {
    if let Some(pair) = ae {
        return Ok(encode_rollout_obs(items, device, pinned_h2d, fp16_rollout, pair)?.0);
    }

    let assembled = items
        .iter()
        .map(|it| {
            it.grid.clone().ok_or_else(|| {
                anyhow::anyhow!("PreparedObs.grid missing and no resident AE latent provided")
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let coarse_assembled = if items.iter().all(|it| it.grid_coarse.is_some()) {
        Some(items.iter().map(|it| it.grid_coarse.clone().unwrap()).collect())
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
                lt[dst_off..dst_off + it.gw].copy_from_slice(&it.legal_tile[src_off..src_off + it.gw]);
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
        let uniform_c = items.iter().all(|it| it.gh.div_ceil(2) == cgh && it.gw.div_ceil(2) == cgw);
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
        Some(t_grid(packed, &[bi, policy::C_GRID, cgh as i64, cgw as i64]))
    } else {
        None
    };

    Ok(Obs {
        grid: resident_fine
            .unwrap_or_else(|| t_grid(grid, &[bi, policy::C_GRID, ghi, gwi])),
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

pub fn build_choice_batch(items: &[ChoiceScalars], device: Device, pinned_h2d: bool) -> policy::ChoiceBatch {
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
            grid: Some(vec![0.5f32; policy::C_GRID as usize * plane]),
            grid_coarse: None,
            cgh: 0,
            cgw: 0,
            ae_raw: crate::ae::AeRaw {
                owners: vec![0i64; (gh * 8) * (gw * 8)],
                terrain: vec![0.0f32; 3 * (gh * 8) * (gw * 8)],
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
            ChoiceScalars { action: 1, player_slot: -1, tile_region: 3, build_type: -1, nuke_type: -1, quantity_frac: -1.0 },
            ChoiceScalars { action: 2, player_slot: 0, tile_region: -1, build_type: 1, nuke_type: -1, quantity_frac: 0.5 },
        ];
        let plain_c = build_choice_batch(&choices, Device::Cpu, false);
        let pinned_c = build_choice_batch(&choices, Device::Cpu, true);
        assert_eq!(Vec::<i64>::try_from(&plain_c.action).unwrap(), Vec::<i64>::try_from(&pinned_c.action).unwrap());
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
                assert_eq!(gv(0, y, x), if y < 2 && x < 3 { 1.0 } else { 0.0 }, "item0 valid[{y}][{x}]");
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
        let grid_at = |b: usize, c: usize, y: usize, x: usize| grid[b * policy::C_GRID as usize * plane + c * plane + y * 5 + x];
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
    fn resident_mixed_shape_fine_and_coarse_match_host_fallback() {
        let mut items = vec![tiny_prepared_obs(2, 3), tiny_prepared_obs(4, 4)];
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
            let plane = it.gh * it.gw;
            let mut fine = Vec::with_capacity(policy::C_GRID as usize * plane);
            fine.extend(
                (0..policy::LATENT_C as usize * plane)
                    .map(|i| bi as f32 * 1000.0 + i as f32 * 0.5),
            );
            fine.extend_from_slice(&it.ego);
            fine.extend_from_slice(&it.db);
            fine.extend_from_slice(&it.transient);
            it.grid = Some(fine);

            let (cgh, cgw) = (it.gh.div_ceil(2), it.gw.div_ceil(2));
            let mut coarse = Vec::with_capacity(policy::C_GRID as usize * cgh * cgw);
            coarse.extend(
                (0..policy::LATENT_C as usize * cgh * cgw)
                    .map(|i| 5000.0 + bi as f32 * 1000.0 + i as f32),
            );
            coarse.extend(avg_pool2_planes(&it.ego, 3, it.gh, it.gw));
            coarse.extend(avg_pool2_planes(&it.db, 1, it.gh, it.gw));
            coarse.extend(max_pool2_planes(
                &it.transient,
                feat::N_TRANSIENT,
                it.gh,
                it.gw,
            ));
            it.grid_coarse = Some(coarse);
            it.cgh = cgh;
            it.cgw = cgw;
        }
        let refs: Vec<&PreparedObs> = items.iter().collect();
        let fallback = build_obs(&refs, Device::Cpu, false, false);
        let resident = resident_latents_from_host_grids(&refs);
        let resident_refs: Vec<&ResidentLatents> = resident.iter().collect();
        let fast = build_obs_with_resident_latents(
            &refs,
            &resident_refs,
            Device::Cpu,
            false,
            false,
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

    #[test]
    fn resident_rollout_preserves_t_major_n_minor_order() {
        const T: usize = 2;
        const N: usize = 3;
        let shapes = [(2, 3), (4, 2), (3, 4), (4, 4), (2, 2), (3, 3)];
        let mut items = Vec::with_capacity(T * N);
        for t in 0..T {
            for e in 0..N {
                let flat = t * N + e;
                let marker = (100 * t + 10 * e) as f32;
                let (gh, gw) = shapes[flat];
                let plane = gh * gw;
                let mut item = tiny_prepared_obs(gh, gw);
                item.ego.fill(marker + 1.0);
                item.db.fill(marker + 2.0);
                item.transient.fill(marker + 3.0);
                let mut grid = vec![marker; policy::LATENT_C as usize * plane];
                grid.extend_from_slice(&item.ego);
                grid.extend_from_slice(&item.db);
                grid.extend_from_slice(&item.transient);
                item.grid = Some(grid);
                items.push(item);
            }
        }

        let refs: Vec<&PreparedObs> = items.iter().collect();
        let fallback = build_obs(&refs, Device::Cpu, false, false);
        let resident = resident_latents_from_host_grids(&refs);
        let resident_refs: Vec<&ResidentLatents> = resident.iter().collect();
        let fast = build_obs_with_resident_latents(
            &refs,
            &resident_refs,
            Device::Cpu,
            false,
            false,
        )
        .unwrap();
        let fast_values: Vec<f32> = Vec::try_from(fast.grid.reshape([-1])).unwrap();
        let fallback_values: Vec<f32> = Vec::try_from(fallback.grid.reshape([-1])).unwrap();
        assert_eq!(fast_values, fallback_values);

        let max_plane = 4 * 4;
        for t in 0..T {
            for e in 0..N {
                let flat = t * N + e;
                let marker = (100 * t + 10 * e) as f32;
                let row = flat * policy::C_GRID as usize * max_plane;
                assert_eq!(fast_values[row], marker, "wrong row for t={t}, e={e}");
                assert_eq!(
                    fast_values[row + policy::LATENT_C as usize * max_plane],
                    marker + 1.0,
                    "wrong extra-plane row for t={t}, e={e}"
                );
            }
        }
    }

    #[test]
    fn resident_shape_validation_fails_before_tensor_ops() {
        let item = tiny_prepared_obs(2, 3);
        let bad = ResidentLatents {
            fine: Tensor::zeros([policy::LATENT_C, 2, 2], (Kind::Float, Device::Cpu)),
            coarse: None,
        };
        let err = build_obs_with_resident_latents(
            &[&item],
            &[&bad],
            Device::Cpu,
            false,
            false,
        )
        .err()
        .expect("invalid resident shape must fail");
        assert!(err.to_string().contains("fine latent shape"));
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
}
