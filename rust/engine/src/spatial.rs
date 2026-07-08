//! Spatial queries for transport ships (TS `SpatialQuery.ts` subset).

use crate::game::Game;
use crate::map::TileRef;
use crate::water::{bounded_water_path, land_bfs_nearest_shore};

const REFINE_MAX_SEARCH_AREA: u32 = 100 * 100;
const CANDIDATE_RADIUS: u32 = 20;
const MIN_WAYPOINT_DIST: usize = 50;
const MAX_WAYPOINT_DIST: usize = 200;
const PADDING: u32 = 10;

fn refine_start_tile(game: &mut Game, path: &[TileRef], shores: &[TileRef]) -> TileRef {
    if path.is_empty() {
        return shores[0];
    }
    if path.len() <= MIN_WAYPOINT_DIST {
        return path[0];
    }

    let best_tile = path[0];
    let candidates: Vec<TileRef> = shores
        .iter()
        .copied()
        .filter(|&s| game.manhattan_dist(s, best_tile) <= CANDIDATE_RADIUS)
        .collect();
    if candidates.len() <= 1 {
        return best_tile;
    }

    let mut cand_min_x = game.map.x(candidates[0]);
    let mut cand_max_x = cand_min_x;
    let mut cand_min_y = game.map.y(candidates[0]);
    let mut cand_max_y = cand_min_y;
    for &c in &candidates[1..] {
        let sx = game.map.x(c);
        let sy = game.map.y(c);
        cand_min_x = cand_min_x.min(sx);
        cand_max_x = cand_max_x.max(sx);
        cand_min_y = cand_min_y.min(sy);
        cand_max_y = cand_max_y.max(sy);
    }

    let mut lo = MIN_WAYPOINT_DIST;
    let mut hi = MAX_WAYPOINT_DIST.min(path.len() - 1);
    let mut best_waypoint_idx = lo;
    for _ in 0..5 {
        if lo > hi {
            break;
        }
        let mid = (lo + hi) / 2;
        let wp = path[mid];
        let wp_x = game.map.x(wp);
        let wp_y = game.map.y(wp);
        let min_x = (cand_min_x.min(wp_x) as i32) - PADDING as i32;
        let max_x = (cand_max_x.max(wp_x) + PADDING) as i32;
        let min_y = (cand_min_y.min(wp_y) as i32) - PADDING as i32;
        let max_y = (cand_max_y.max(wp_y) + PADDING) as i32;
        let area = ((max_x - min_x + 1) * (max_y - min_y + 1)) as u32;
        if area <= REFINE_MAX_SEARCH_AREA {
            best_waypoint_idx = mid;
            lo = mid + 1;
        } else {
            hi = mid.saturating_sub(1);
        }
    }

    let waypoint = path[best_waypoint_idx];
    let wp_x = game.map.x(waypoint);
    let wp_y = game.map.y(waypoint);
    let bounds_min_x = cand_min_x.min(wp_x).saturating_sub(PADDING);
    let bounds_max_x = (cand_max_x.max(wp_x) + PADDING).min(game.map.width - 1);
    let bounds_min_y = cand_min_y.min(wp_y).saturating_sub(PADDING);
    let bounds_max_y = (cand_max_y.max(wp_y) + PADDING).min(game.map.height - 1);
    let bounds_area =
        (bounds_max_x - bounds_min_x + 1) * (bounds_max_y - bounds_min_y + 1);
    if bounds_area > REFINE_MAX_SEARCH_AREA {
        return best_tile;
    }

    if let Some(refined) = bounded_water_path(
        &game.map,
        &candidates,
        waypoint,
        bounds_min_x,
        bounds_max_x,
        bounds_min_y,
        bounds_max_y,
    ) {
        if let Some(&first) = refined.first() {
            return first;
        }
    }
    best_tile
}

pub fn closest_tile(game: &Game, refs: &[TileRef], tile: TileRef) -> (Option<TileRef>, u32) {
    let mut min_distance = u32::MAX;
    let mut min_ref = None;
    for &r in refs {
        let distance = game.manhattan_dist(r, tile);
        if distance < min_distance {
            min_distance = distance;
            min_ref = Some(r);
        }
    }
    (min_ref, min_distance)
}

/// TS `closestTwoTiles`  -  sweep sorted by x (not brute force).
pub fn closest_two_tiles(game: &Game, xs: &[TileRef], ys: &[TileRef]) -> Option<(TileRef, TileRef)> {
    if xs.is_empty() || ys.is_empty() {
        return None;
    }
    let mut x_sorted: Vec<TileRef> = xs.to_vec();
    let mut y_sorted: Vec<TileRef> = ys.to_vec();
    x_sorted.sort_by_key(|t| game.map.x(*t));
    y_sorted.sort_by_key(|t| game.map.x(*t));

    let mut i = 0usize;
    let mut j = 0usize;
    let mut min_distance = u32::MAX;
    let mut result = (x_sorted[0], y_sorted[0]);

    while i < x_sorted.len() && j < y_sorted.len() {
        let current_x = x_sorted[i];
        let current_y = y_sorted[j];
        let distance = game.manhattan_dist(current_x, current_y);
        if distance < min_distance {
            min_distance = distance;
            result = (current_x, current_y);
        }
        if i == x_sorted.len() - 1 {
            j += 1;
        } else if j == y_sorted.len() - 1 {
            i += 1;
        } else if game.map.x(current_x) < game.map.x(current_y) {
            i += 1;
        } else {
            j += 1;
        }
    }
    Some(result)
}

pub fn shore_border_tiles(game: &Game, small_id: u16) -> Vec<TileRef> {
    let mut out = Vec::new();
    game.for_each_border_tile(small_id, |t| {
        if game.is_shore(t) {
            out.push(t);
        }
    });
    out
}

pub fn target_transport_tile(game: &mut Game, tile: TileRef) -> Option<TileRef> {
    let owner = game.map.owner_id(tile);
    land_bfs_nearest_shore(&game.map, &mut game.bfs, tile, 50, owner)
}

pub fn closest_shore_by_water(game: &mut Game, owner_small_id: u16, target: TileRef) -> Option<TileRef> {
    if !game.is_water(target) && !game.is_shore(target) {
        return None;
    }
    let target_comp = game.get_water_component(target)?;

    let mut shores: Vec<TileRef> = Vec::with_capacity(32);
    game.for_each_border_tile(owner_small_id, |t| {
        if game.is_shore(t) && game.is_land(t) && game.map.owner_id(t) == owner_small_id {
            if game.get_water_component(t) == Some(target_comp) {
                shores.push(t);
            }
        }
    });
    if shores.is_empty() {
        return None;
    }

    // TS `PathFinding.Water(gm).findPath(shores, target)` + `refineStartTile`.
    if !game.plan_water_path_multi(&shores, target) {
        return None;
    }
    let path = game.planned_water_path().to_vec();
    Some(refine_start_tile(game, &path, &shores))
}

pub fn can_build_transport_ship(
    game: &mut Game,
    owner_small_id: u16,
    tile: TileRef,
) -> Option<TileRef> {
    if game.wire.is_unit_disabled(crate::core::schemas::unit_type::TRANSPORT) {
        return None;
    }
    if game.unit_count(owner_small_id, crate::core::schemas::unit_type::TRANSPORT)
        >= game.wire.boat_max_number()
    {
        return None;
    }
    let dst = target_transport_tile(game, tile)?;
    if game.map.owner_id(tile) == owner_small_id {
        return None;
    }
    closest_shore_by_water(game, owner_small_id, dst)
}
