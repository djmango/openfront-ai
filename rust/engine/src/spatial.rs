//! Spatial queries for transport ships (TS `SpatialQuery.ts` subset).

use crate::game::Game;
use crate::map::TileRef;
use crate::water::{bounded_water_path, land_bfs_nearest_shore, water_component_at};

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
        let min_x = cand_min_x.min(wp_x).saturating_sub(PADDING);
        let max_x = cand_max_x.max(wp_x) + PADDING;
        let min_y = cand_min_y.min(wp_y).saturating_sub(PADDING);
        let max_y = cand_max_y.max(wp_y) + PADDING;
        let area = (max_x - min_x + 1) * (max_y - min_y + 1);
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

pub fn closest_two_tiles(game: &Game, xs: &[TileRef], ys: &[TileRef]) -> Option<(TileRef, TileRef)> {
    if xs.is_empty() || ys.is_empty() {
        return None;
    }
    let mut best = (xs[0], ys[0]);
    let mut best_dist = game.manhattan_dist(best.0, best.1);
    for &x in xs {
        for &y in ys {
            let d = game.manhattan_dist(x, y);
            if d < best_dist {
                best_dist = d;
                best = (x, y);
            }
        }
    }
    Some(best)
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
    let target_comp = water_component_at(&game.map, &game.water_component, target)?;

    let mut shores: Vec<TileRef> = Vec::with_capacity(32);
    game.for_each_border_tile(owner_small_id, |t| {
        if game.is_shore(t) && game.map.owner_id(t) == owner_small_id {
            if water_component_at(&game.map, &game.water_component, t) == Some(target_comp) {
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
