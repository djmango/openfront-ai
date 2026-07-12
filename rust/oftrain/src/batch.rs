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

/// Assemble per-item `C_GRID` host planes: AE latent (or prebuilt grid)
/// + ego + db + transient. When `ae` is set, runs batched GPU encode
/// (mirrors `rl/obs.py::encode_grids`); otherwise requires each item's
/// `PreparedObs::grid` to already be filled (unit-test path).
fn assemble_grids(items: &[&PreparedObs], device: Device, ae: Option<&AePair>) -> anyhow::Result<Vec<Vec<f32>>> {
    let c_grid = policy::C_GRID as usize;
    if let Some(pair) = ae {
        let raws: Vec<&ae::AeRaw> = items.iter().map(|it| &it.ae_raw).collect();
        let zs = ae::encode_latent_batch(&pair.fine, &raws, device)?;
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
) -> anyhow::Result<Option<Vec<Vec<f32>>>> {
    let Some(pair) = ae else { return Ok(None) };
    let Some(coarse) = pair.coarse.as_ref() else { return Ok(None) };
    let raws: Vec<&ae::AeRaw> = items.iter().map(|it| &it.ae_raw).collect();
    let zs = ae::encode_latent_batch(coarse, &raws, device)?;
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

/// Encode AE latents into each item's `grid` / `grid_coarse` host buffers
/// in place. Called on the actor thread (exclusive `&mut`) before
/// `build_obs` so learner threads never need to hold an AE.
pub fn encode_prepared_obs(items: &mut [PreparedObs], device: Device, ae: &AePair) -> anyhow::Result<()> {
    let refs: Vec<&PreparedObs> = items.iter().collect();
    let fine = assemble_grids(&refs, device, Some(ae))?;
    let coarse = assemble_coarse_grids(&refs, device, Some(ae))?;
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

pub fn build_obs(items: &[&PreparedObs], device: Device, pinned_h2d: bool) -> Obs {
    build_obs_with_ae(items, device, pinned_h2d, None).expect("build_obs")
}

pub fn build_obs_with_ae(
    items: &[&PreparedObs],
    device: Device,
    pinned_h2d: bool,
    ae: Option<&AePair>,
) -> anyhow::Result<Obs> {
    let b = items.len();
    let gh = items.iter().map(|it| it.gh).max().unwrap_or(0);
    let gw = items.iter().map(|it| it.gw).max().unwrap_or(0);
    let uniform = items.iter().all(|it| it.gh == gh && it.gw == gw);
    let c_grid = policy::C_GRID as usize;

    let assembled = if ae.is_some() {
        assemble_grids(items, device, ae)?
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
        assemble_coarse_grids(items, device, ae)?
    } else if items.iter().all(|it| it.grid_coarse.is_some()) {
        Some(items.iter().map(|it| it.grid_coarse.clone().unwrap()).collect())
    } else {
        None
    };

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
        let g_src = &assembled[idx];
        if uniform {
            grid.extend_from_slice(g_src);
            legal_tile.extend_from_slice(&it.legal_tile);
            grid_valid.extend(std::iter::repeat(1.0f32).take(gh * gw));
        } else {
            let mut g = vec![0.0f32; c_grid * gh * gw];
            for c in 0..c_grid {
                for y in 0..it.gh {
                    let src_off = c * it.gh * it.gw + y * it.gw;
                    let dst_off = c * gh * gw + y * gw;
                    g[dst_off..dst_off + it.gw].copy_from_slice(&g_src[src_off..src_off + it.gw]);
                }
            }
            grid.extend_from_slice(&g);
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
    let bi = b as i64;
    let (ghi, gwi) = (gh as i64, gw as i64);
    let ms = feat::MAX_SLOTS as i64;
    let na = feat::N_ACTIONS as i64;
    let local_sz = policy::LOCAL;

    let grid_coarse = if let Some(coarse_items) = coarse_assembled {
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
        Some(t(packed, &[bi, policy::C_GRID, cgh as i64, cgw as i64]))
    } else {
        None
    };

    Ok(Obs {
        grid: t(grid, &[bi, policy::C_GRID, ghi, gwi]),
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
        let plain = build_obs(&refs, Device::Cpu, false);
        let pinned = build_obs(&refs, Device::Cpu, true);
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
        let obs = build_obs(&items, Device::Cpu, false);

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
}
