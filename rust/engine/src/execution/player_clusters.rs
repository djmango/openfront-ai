//! Surrounded cluster removal (`PlayerExecution.ts` subset for hash parity).

use crate::game::Game;
use crate::map::TileRef;
use crate::util::simple_hash;
use std::collections::HashSet;

const TICKS_PER_CLUSTER_CALC: u32 = 20;

#[derive(Clone, Default)]
struct BBox {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
}

/// TS `Set<TileRef>` insertion order  -  iteration and `values().next()` depend on it.
#[derive(Clone, Default)]
struct OrderedTiles {
    tiles: Vec<TileRef>,
    set: HashSet<TileRef>,
}

impl OrderedTiles {
    fn insert(&mut self, t: TileRef) -> bool {
        if self.set.insert(t) {
            self.tiles.push(t);
            true
        } else {
            false
        }
    }

    fn contains(&self, t: TileRef) -> bool {
        self.set.contains(&t)
    }

    fn len(&self) -> usize {
        self.tiles.len()
    }

    fn first(&self) -> Option<TileRef> {
        self.tiles.first().copied()
    }

    fn iter(&self) -> impl Iterator<Item = TileRef> + '_ {
        self.tiles.iter().copied()
    }
}

fn bbox_from_tiles(game: &Game, tiles: &OrderedTiles) -> BBox {
    let mut bb = BBox {
        min_x: u32::MAX,
        min_y: u32::MAX,
        max_x: 0,
        max_y: 0,
    };
    for t in tiles.iter() {
        let x = game.map.x(t);
        let y = game.map.y(t);
        bb.min_x = bb.min_x.min(x);
        bb.min_y = bb.min_y.min(y);
        bb.max_x = bb.max_x.max(x);
        bb.max_y = bb.max_y.max(y);
    }
    bb
}

fn inscribed(outer: BBox, inner: BBox) -> bool {
    outer.min_x <= inner.min_x
        && outer.min_y <= inner.min_y
        && outer.max_x >= inner.max_x
        && outer.max_y >= inner.max_y
}

fn flood_border_cluster(
    game: &Game,
    border: &[TileRef],
    start: TileRef,
    visited: &mut HashSet<TileRef>,
) -> OrderedTiles {
    let mut cluster = OrderedTiles::default();
    let mut stack = vec![start];
    while let Some(t) = stack.pop() {
        if visited.contains(&t) || !border.contains(&t) {
            continue;
        }
        visited.insert(t);
        cluster.insert(t);
        game.map.for_each_neighbor8(t, |n| stack.push(n));
    }
    cluster
}

fn calculate_clusters(game: &Game, small_id: u16) -> Vec<OrderedTiles> {
    let Some(border) = game.border_tiles_of(small_id) else {
        return vec![];
    };
    if border.is_empty() {
        return vec![];
    }
    let mut visited = HashSet::new();
    let mut clusters = Vec::new();
    // TS iterates `player.borderTiles()` Set insertion order.
    for start in border.iter() {
        if visited.contains(&start) {
            continue;
        }
        let cluster = flood_border_cluster(game, border.as_slice(), start, &mut visited);
        if !cluster.tiles.is_empty() {
            clusters.push(cluster);
        }
    }
    clusters
}

fn surrounded_by_same_enemy(
    game: &Game,
    small_id: u16,
    cluster: &OrderedTiles,
    cluster_bb: BBox,
) -> Option<u16> {
    let mut enemy: Option<u16> = None;
    let mut enemy_bb = BBox::default();
    let mut enemy_init = false;

    for tile in cluster.iter() {
        if game.map.is_ocean_shore(tile) || game.map.is_on_edge_of_map(tile) {
            return None;
        }
        let mut neighbors = [TileRef::MAX; 4];
        let mut n = 0usize;
        game.map.for_each_neighbor4(tile, |nb| {
            if n < 4 {
                neighbors[n] = nb;
                n += 1;
            }
        });
        for i in 0..n {
            let owner = game.map.owner_id(neighbors[i]);
            if owner == 0 {
                return None;
            }
            if owner == small_id {
                continue;
            }
            match enemy {
                None => enemy = Some(owner),
                Some(e) if e != owner => return None,
                _ => {}
            }
            let x = game.map.x(neighbors[i]);
            let y = game.map.y(neighbors[i]);
            if !enemy_init {
                enemy_bb = BBox {
                    min_x: x,
                    min_y: y,
                    max_x: x,
                    max_y: y,
                };
                enemy_init = true;
            } else {
                enemy_bb.min_x = enemy_bb.min_x.min(x);
                enemy_bb.min_y = enemy_bb.min_y.min(y);
                enemy_bb.max_x = enemy_bb.max_x.max(x);
                enemy_bb.max_y = enemy_bb.max_y.max(y);
            }
        }
        if enemy.is_none() {
            return None;
        }
    }
    let e = enemy?;
    if game.is_friendly(e, small_id) {
        return None;
    }
    if inscribed(enemy_bb, cluster_bb) {
        Some(e)
    } else {
        None
    }
}

fn is_surrounded(game: &Game, small_id: u16, cluster: &OrderedTiles) -> bool {
    let mut has_enemy = false;
    let mut enemy_bb = BBox::default();
    let mut enemy_init = false;

    for tr in cluster.iter() {
        if game.is_shore(tr) || game.map.is_on_edge_of_map(tr) {
            return false;
        }
        let mut neighbors = [TileRef::MAX; 4];
        let mut n = 0usize;
        game.map.for_each_neighbor4(tr, |nb| {
            if n < 4 {
                neighbors[n] = nb;
                n += 1;
            }
        });
        for i in 0..n {
            let owner = game.map.owner_id(neighbors[i]);
            if owner == 0 || owner == small_id {
                continue;
            }
            has_enemy = true;
            let x = game.map.x(neighbors[i]);
            let y = game.map.y(neighbors[i]);
            if !enemy_init {
                enemy_bb = BBox {
                    min_x: x,
                    min_y: y,
                    max_x: x,
                    max_y: y,
                };
                enemy_init = true;
            } else {
                enemy_bb.min_x = enemy_bb.min_x.min(x);
                enemy_bb.min_y = enemy_bb.min_y.min(y);
                enemy_bb.max_x = enemy_bb.max_x.max(x);
                enemy_bb.max_y = enemy_bb.max_y.max(y);
            }
        }
    }
    if !has_enemy {
        return false;
    }
    let cluster_bb = bbox_from_tiles(game, cluster);
    inscribed(enemy_bb, cluster_bb)
}

fn get_capturing_player(game: &Game, small_id: u16, cluster: &OrderedTiles) -> Option<u16> {
    // TS iterates cluster Set insertion order; Map keeps first-seen neighbor order.
    let mut neighbors: Vec<(u16, u32)> = Vec::new();
    for t in cluster.iter() {
        game.map.for_each_neighbor4(t, |n| {
            let owner = game.map.owner_id(n);
            if owner == 0 || owner == small_id {
                return;
            }
            if game.is_friendly(owner, small_id) {
                return;
            }
            if let Some(entry) = neighbors.iter_mut().find(|(sid, _)| *sid == owner) {
                entry.1 += 1;
            } else {
                neighbors.push((owner, 1));
            }
        });
    }
    if neighbors.is_empty() {
        return None;
    }
    let attackers: Vec<u16> = neighbors.iter().map(|(sid, _)| *sid).collect();
    if let Some(attacker) = game.largest_incoming_land_attack_from_neighbors(
        small_id,
        &attackers,
        &neighbors,
    ) {
        return Some(attacker);
    }
    // TS `getMode`: first neighbor with strictly greatest count wins ties.
    let mut best: Option<(u16, u32)> = None;
    for (sid, count) in neighbors {
        if best.map(|(_, c)| count > c).unwrap_or(true) {
            best = Some((sid, count));
        }
    }
    best.map(|(sid, _)| sid)
}

fn flood_owned(game: &Game, small_id: u16, start: TileRef) -> OrderedTiles {
    let mut result = OrderedTiles::default();
    let mut stack = vec![start];
    while let Some(t) = stack.pop() {
        if result.contains(t) || game.map.owner_id(t) != small_id {
            continue;
        }
        result.insert(t);
        game.map.for_each_neighbor4(t, |n| stack.push(n));
    }
    result
}

fn remove_cluster(game: &mut Game, small_id: u16, cluster: &OrderedTiles) {
    for t in cluster.iter() {
        if game.map.owner_id(t) != small_id {
            return;
        }
    }
    let Some(captor) = get_capturing_player(game, small_id, cluster) else {
        return;
    };
    let Some(first) = cluster.first() else {
        return;
    };
    let tiles = flood_owned(game, small_id, first);
    let wipe_all = game
        .player_by_small_id(small_id)
        .is_some_and(|p| p.tiles_owned == tiles.len() as i32);
    if wipe_all {
        game.conquer_player(captor, small_id);
    }
    for t in tiles.iter() {
        game.conquer(captor, t);
    }
}

pub fn maybe_remove_clusters(game: &mut Game, small_id: u16, tick: u32) {
    let (tiles_owned, last_calc, last_change, name_hash) = {
        let Some(p) = game.player_by_small_id(small_id) else {
            return;
        };
        if !p.alive || p.tiles_owned == 0 {
            return;
        };
        (
            p.tiles_owned,
            p.last_cluster_calc,
            p.last_tile_change,
            simple_hash(&p.name),
        )
    };

    let mut last_calc = last_calc;
    if last_calc == 0 {
        last_calc = tick + (name_hash as u32 % TICKS_PER_CLUSTER_CALC);
        if let Some(p) = game.player_by_small_id_mut(small_id) {
            p.last_cluster_calc = last_calc;
        }
    }

    if tick.saturating_sub(last_calc) <= TICKS_PER_CLUSTER_CALC && tiles_owned >= 100 {
        return;
    }
    if last_change < last_calc {
        return;
    }

    if let Some(p) = game.player_by_small_id_mut(small_id) {
        p.last_cluster_calc = tick;
    }

    let clusters = calculate_clusters(game, small_id);
    if clusters.is_empty() {
        return;
    }

    let mut largest_idx = 0usize;
    let mut largest_size = clusters[0].len();
    for (i, c) in clusters.iter().enumerate().skip(1) {
        if c.len() > largest_size {
            largest_size = c.len();
            largest_idx = i;
        }
    }

    let largest = clusters[largest_idx].clone();
    let largest_bb = bbox_from_tiles(game, &largest);
    if surrounded_by_same_enemy(game, small_id, &largest, largest_bb).is_some() {
        remove_cluster(game, small_id, &largest);
    }

    for (i, cluster) in clusters.into_iter().enumerate() {
        if i == largest_idx {
            continue;
        }
        if is_surrounded(game, small_id, &cluster) {
            remove_cluster(game, small_id, &cluster);
        }
    }
}
