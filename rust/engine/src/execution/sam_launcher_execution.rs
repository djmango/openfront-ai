//! SAM Launcher interception (`SAMLauncherExecution.ts` + its inner `SAMTargetingSystem`).

use super::sam_missile_execution::SamMissileExecution;
use super::{ExecEnum, Execution};
use crate::core::schemas::unit_type;
use crate::game::Game;
use crate::map::TileRef;
use std::collections::HashMap;

const MIRV_WARHEAD_SEARCH_RADIUS: u32 = 400;
const MIRV_WARHEAD_PROTECTION_RADIUS: u32 = 50;
const NUKE_TYPES: [&str; 2] = [unit_type::ATOM_BOMB, unit_type::HYDROGEN_BOMB];

#[derive(Clone, Copy)]
struct InterceptionTile {
    tile: TileRef,
    tick: i64,
}

pub struct SamLauncherExecution {
    owner_small_id: u16,
    sam_unit_id: i32,

    active: bool,
    /// TS `SAMTargetingSystem.precomputedNukes` - keyed by the (globally unique) nuke unit id.
    precomputed_nukes: HashMap<i32, Option<InterceptionTile>>,
}

impl SamLauncherExecution {
    /// TS `ConstructionExecution.completeConstruction` always passes the already-built
    /// structure unit as the constructor's `sam` param, so the (dead in practice) TS
    /// code path that builds a fresh SAM Launcher from `sam === null` is not ported.
    pub fn new(owner_small_id: u16, sam_unit_id: i32) -> Self {
        Self {
            owner_small_id,
            sam_unit_id,
            active: true,
            precomputed_nukes: HashMap::new(),
        }
    }
}

impl Execution for SamLauncherExecution {
    fn init(&mut self, _game: &mut Game, _tick: u32) {}

    fn tick(&mut self, game: &mut Game, ticks: u32) {
        let sam_id = self.sam_unit_id;
        // TS SAMLauncherExecution holds the Unit; after capture, retarget owner.
        let Some(owner) = game.find_unit_owner(sam_id) else {
            self.active = false;
            return;
        };
        self.owner_small_id = owner;
        let Some(sam) = game.unit(self.owner_small_id, sam_id) else {
            self.active = false;
            return;
        };
        if sam.under_construction {
            return;
        }
        let sam_tile = sam.tile as TileRef;
        let sam_level = sam.level;

        let front_time = sam.missile_timer_queue.first().copied();
        if let Some(front_time) = front_time {
            let cooldown = game.wire.sam_cooldown() as i64 - (ticks as i64 - front_time as i64);
            if cooldown <= 0 {
                game.unit_reload_missile(self.owner_small_id, sam_id);
            }
        }
        if game.unit_is_in_cooldown(self.owner_small_id, sam_id) {
            return;
        }

        let mirv_targets = mirv_warhead_targets(game, self.owner_small_id, sam_tile);

        let target = if mirv_targets.is_empty() {
            get_single_target(
                game,
                self.owner_small_id,
                sam_tile,
                sam_level,
                ticks,
                &mut self.precomputed_nukes,
            )
        } else {
            None
        };

        if target.is_none() && mirv_targets.is_empty() {
            return;
        }

        game.unit_launch(self.owner_small_id, sam_id);

        if !mirv_targets.is_empty() {
            for &(owner_sid, unit_id) in &mirv_targets {
                game.remove_unit(owner_sid, unit_id);
            }
        } else if let Some((target_owner, target_unit_id, intercept_tile)) = target {
            if let Some(u) = game.unit_mut(target_owner, target_unit_id) {
                u.targeted_by_sam = true;
            }
            game.add_execution(ExecEnum::SamMissile(SamMissileExecution::new(
                sam_tile,
                self.owner_small_id,
                sam_id,
                target_owner,
                target_unit_id,
                intercept_tile,
            )));
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        false
    }
}

fn mirv_warhead_targets(game: &Game, sam_owner_sid: u16, sam_tile: TileRef) -> Vec<(u16, i32)> {
    let mut out = Vec::new();
    for &(owner_sid, unit_id, tile, d2) in &game.nearby_structures_any(
        sam_tile,
        MIRV_WARHEAD_SEARCH_RADIUS,
        &[unit_type::MIRV_WARHEAD],
    ) {
        let _ = (tile, d2);
        if owner_sid == sam_owner_sid {
            continue;
        }
        if game.is_friendly(sam_owner_sid, owner_sid)
            && !(game.winner.is_some() && game.players_on_same_team(sam_owner_sid, owner_sid))
        {
            continue;
        }
        let Some(dst) = game.unit(owner_sid, unit_id).and_then(|u| u.target_tile) else {
            continue;
        };
        if game.manhattan_dist(dst, sam_tile) < MIRV_WARHEAD_PROTECTION_RADIUS {
            out.push((owner_sid, unit_id));
        }
    }
    out
}

/// TS `SAMTargetingSystem.getSingleTarget`.
fn get_single_target(
    game: &Game,
    sam_owner_sid: u16,
    sam_tile: TileRef,
    sam_level: i32,
    ticks: u32,
    precomputed: &mut HashMap<i32, Option<InterceptionTile>>,
) -> Option<(u16, i32, TileRef)> {
    let range = game.wire.sam_range(sam_level);
    let range_sq = range * range;
    let detection_range = (game.wire.max_sam_range() * 2.0) as u32;

    let candidates: Vec<(u16, i32)> = game
        .nearby_structures_any(sam_tile, detection_range, &NUKE_TYPES)
        .iter()
        .filter_map(|&(owner_sid, unit_id, _, _)| {
            let u = game.unit(owner_sid, unit_id)?;
            if u.targeted_by_sam {
                return None;
            }
            if owner_sid == sam_owner_sid {
                return None;
            }
            if game.is_friendly(sam_owner_sid, owner_sid) {
                if !(game.winner.is_some() && game.players_on_same_team(sam_owner_sid, owner_sid)) {
                    return None;
                }
            }
            Some((owner_sid, unit_id))
        })
        .collect();

    // TS `updateUnreachableNukes` - purge cached entries for nukes no longer nearby.
    if !precomputed.is_empty() {
        let nearby_ids: std::collections::HashSet<i32> =
            candidates.iter().map(|&(_, uid)| uid).collect();
        precomputed.retain(|uid, _| nearby_ids.contains(uid));
    }

    let mut best: Option<(u16, i32, TileRef, bool)> = None;
    for &(owner_sid, unit_id) in &candidates {
        let nuke_type_is_hydrogen = game
            .unit(owner_sid, unit_id)
            .is_some_and(|u| u.unit_type == unit_type::HYDROGEN_BOMB);

        if let Some(cached) = precomputed.get(&unit_id).copied() {
            match cached {
                None => continue,
                Some(it) => {
                    if it.tick == ticks as i64 {
                        precomputed.remove(&unit_id);
                        if best.is_none() || (nuke_type_is_hydrogen && !best.unwrap().3) {
                            best = Some((owner_sid, unit_id, it.tile, nuke_type_is_hydrogen));
                        }
                        continue;
                    }
                    if it.tick > ticks as i64 {
                        continue;
                    }
                    precomputed.remove(&unit_id);
                    // fall through to recompute
                }
            }
        }

        let Some(interception) =
            compute_interception_tile(game, owner_sid, unit_id, sam_tile, range_sq)
        else {
            precomputed.insert(unit_id, None);
            continue;
        };
        if interception.tick <= 1 {
            if best.is_none() || (nuke_type_is_hydrogen && !best.unwrap().3) {
                best = Some((owner_sid, unit_id, interception.tile, nuke_type_is_hydrogen));
            }
        } else {
            precomputed.insert(
                unit_id,
                Some(InterceptionTile {
                    tile: interception.tile,
                    tick: interception.tick + ticks as i64,
                }),
            );
        }
    }

    best.map(|(o, u, t, _)| (o, u, t))
}

/// TS `SAMTargetingSystem.computeInterceptionTile`.
fn compute_interception_tile(
    game: &Game,
    owner_sid: u16,
    unit_id: i32,
    sam_tile: TileRef,
    range_sq: f64,
) -> Option<InterceptionTile> {
    let missile_speed = game.wire.default_sam_missile_speed();
    let u = game.unit(owner_sid, unit_id)?;
    let trajectory = &u.trajectory;
    let targetable = &u.trajectory_targetable;
    let current_index = u.trajectory_index as usize;
    let explosion_tick = trajectory.len() as i64 - current_index as i64;

    for i in current_index..trajectory.len() {
        let traj_tile = trajectory[i];
        if !targetable.get(i).copied().unwrap_or(false) {
            continue;
        }
        let d2 = game.map.euclidean_dist_squared(sam_tile, traj_tile) as f64;
        if d2 > range_sq {
            continue;
        }
        let nuke_tick_to_reach = (i - current_index) as i64;
        let sam_tick_to_reach =
            (game.manhattan_dist(sam_tile, traj_tile) as f64 / missile_speed).ceil() as i64;
        let tick_before_shooting = nuke_tick_to_reach - sam_tick_to_reach;
        if sam_tick_to_reach < explosion_tick && tick_before_shooting >= 0 {
            return Some(InterceptionTile {
                tile: traj_tile,
                tick: tick_before_shooting,
            });
        }
    }
    None
}
