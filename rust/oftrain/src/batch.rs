//! Stacks `Vec<PreparedObs>` (host f32 arrays, one per env) into GPU/CPU
//! tensors (`policy::Obs`). v1 assumes every env in a batch shares the
//! same grid resolution (gh, gw) - true as long as `EnvWorker`s in a run
//! share a curriculum stage with a single map (see `policy.rs` module
//! doc); panics with a clear message otherwise rather than silently
//! producing garbage. Padding-to-max-shape for mixed-map batches is a
//! follow-up once multi-map curriculum stages are exercised.

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
    let (gh, gw) = (items[0].gh, items[0].gw);
    for (i, it) in items.iter().enumerate() {
        assert_eq!((it.gh, it.gw), (gh, gw), "obs {i} grid shape mismatch (v1 requires uniform batch shape)");
    }

    let mut grid = Vec::with_capacity(b * policy::C_GRID as usize * gh * gw);
    let mut legal_tile = Vec::with_capacity(b * gh * gw);
    let mut players = Vec::with_capacity(b * feat::MAX_SLOTS * feat::P_FEAT);
    let mut pmask = Vec::with_capacity(b * feat::MAX_SLOTS);
    let mut local = Vec::with_capacity(b * 5 * policy::LOCAL as usize * policy::LOCAL as usize);
    let mut scalars = Vec::with_capacity(b * feat::N_SCALARS);
    let mut legal_actions = Vec::with_capacity(b * feat::N_ACTIONS);
    let mut legal_ptarget = Vec::with_capacity(b * feat::N_ACTIONS * feat::MAX_SLOTS);
    let mut legal_build = Vec::with_capacity(b * feat::N_BUILD);
    let mut legal_nuke = Vec::with_capacity(b * feat::N_NUKE);

    for it in items {
        grid.extend_from_slice(&it.grid);
        legal_tile.extend_from_slice(&it.legal_tile);
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
        grid_valid: Tensor::ones([bi, ghi, gwi], (Kind::Float, device)),
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
}
