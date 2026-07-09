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

pub fn build_obs(items: &[&PreparedObs], device: Device) -> Obs {
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
        Tensor::from_slice(&v).view(shape).to_kind(Kind::Float).to_device(device)
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

pub fn build_choice_batch(items: &[ChoiceScalars], device: Device) -> policy::ChoiceBatch {
    let action: Vec<i64> = items.iter().map(|c| c.action).collect();
    let player_slot: Vec<i64> = items.iter().map(|c| c.player_slot).collect();
    let tile_region: Vec<i64> = items.iter().map(|c| c.tile_region).collect();
    let build_type: Vec<i64> = items.iter().map(|c| c.build_type).collect();
    let nuke_type: Vec<i64> = items.iter().map(|c| c.nuke_type).collect();
    let quantity_frac: Vec<f32> = items.iter().map(|c| c.quantity_frac).collect();
    policy::ChoiceBatch {
        action: Tensor::from_slice(&action).to_device(device),
        player_slot: Tensor::from_slice(&player_slot).to_device(device),
        tile_region: Tensor::from_slice(&tile_region).to_device(device),
        build_type: Tensor::from_slice(&build_type).to_device(device),
        nuke_type: Tensor::from_slice(&nuke_type).to_device(device),
        quantity_frac: Tensor::from_slice(&quantity_frac).to_device(device),
    }
}
