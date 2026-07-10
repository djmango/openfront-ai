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

pub fn build_obs(items: &[&PreparedObs], device: Device, pinned_h2d: bool) -> Obs {
    let b = items.len();
    let gh = items.iter().map(|it| it.gh).max().unwrap_or(0);
    let gw = items.iter().map(|it| it.gw).max().unwrap_or(0);
    let uniform = items.iter().all(|it| it.gh == gh && it.gw == gw);
    let c_grid = policy::C_GRID as usize;

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

    for it in items {
        if uniform {
            // Fast path: no per-cell copy, just append the already-correctly-
            // shaped contiguous plane (this is the common case - most
            // stages' maps end up the same padded (gh, gw) after
            // `ofcore::feat::REGION` alignment, and single-map early
            // stages before any rehearsal hit always take it).
            grid.extend_from_slice(&it.grid);
            legal_tile.extend_from_slice(&it.legal_tile);
            grid_valid.extend(std::iter::repeat(1.0f32).take(gh * gw));
        } else {
            // Top-left-aligned zero-pad into the batch's max (gh, gw).
            // `it.grid` is (C_GRID, it.gh, it.gw) row-major; pad each
            // channel plane independently since the padded width differs
            // per-row-block, not just appended at the end.
            let mut g = vec![0.0f32; c_grid * gh * gw];
            for c in 0..c_grid {
                for y in 0..it.gh {
                    let src_off = c * it.gh * it.gw + y * it.gw;
                    let dst_off = c * gh * gw + y * gw;
                    g[dst_off..dst_off + it.gw].copy_from_slice(&it.grid[src_off..src_off + it.gw]);
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

    Obs {
        grid: t(grid, &[bi, policy::C_GRID, ghi, gwi]),
        grid_valid: t(grid_valid, &[bi, ghi, gwi]),
        legal_tile: t(legal_tile, &[bi, ghi, gwi]),
        players: t(players, &[bi, ms, feat::P_FEAT as i64]),
        pmask: t(pmask, &[bi, ms]),
        local: t(local, &[bi, 5, local_sz, local_sz]),
        scalars: t(scalars, &[bi, feat::N_SCALARS as i64]),
        legal_actions: t(legal_actions, &[bi, na]),
        legal_ptarget: t(legal_ptarget, &[bi, na, ms]),
        legal_build: t(legal_build, &[bi, feat::N_BUILD as i64]),
        legal_nuke: t(legal_nuke, &[bi, feat::N_NUKE as i64]),
    }
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
        PreparedObs {
            grid: vec![0.5f32; policy::C_GRID as usize * gh * gw],
            legal_tile: vec![1.0f32; gh * gw],
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
        big.grid.iter_mut().for_each(|v| *v = 9.0);
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
