//! Rail/train network port (`RailNetworkImpl.ts` + `TrainStation.ts` + `Railroad.ts` +
//! `RailroadSpatialGrid.ts` + `AStar.Rail.ts` + `PathFinder.Station.ts`).
//!
//! Owned by `Game::rail_network`. Because the network needs to look up live unit
//! tiles/existence via `Game`, most mutating entry points follow the `std::mem::take`
//! pattern already used for `Player::owned_tiles` elsewhere in this crate: pull the
//! network out of `Game`, operate with a plain `&mut Game` alongside it, then put it back.
//! This avoids the self-referential `&mut Game` + `&mut Game.rail_network` borrow conflict.

use crate::core::schemas::unit_type;
use crate::game::Game;
use crate::map::{GameMap, TileRef};
use crate::prng::PseudoRandom;
use std::collections::HashMap;

pub const STATION_RADIUS: u32 = 3;
pub const MAX_CONNECTION_DISTANCE: i32 = 4;
pub const GRID_CELL_SIZE: u32 = 4;

fn is_trade_type(t: &str) -> bool {
    t == unit_type::CITY || t == unit_type::PORT
}

fn is_station_type(t: &str) -> bool {
    t == unit_type::CITY || t == unit_type::PORT || t == unit_type::FACTORY
}

#[derive(Debug, Clone)]
pub struct Station {
    pub id: u32,
    pub owner_small_id: u16,
    pub unit_id: i32,
    pub unit_type: String,
    pub cluster: Option<u32>,
    /// Railroad ids incident to this station, insertion order (mirrors TS `Set` order).
    pub railroads: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct Railroad {
    pub id: u32,
    pub from: u32,
    pub to: u32,
    pub tiles: Vec<TileRef>,
}

#[derive(Debug, Clone, Default)]
pub struct Cluster {
    pub stations: Vec<u32>,
}

#[derive(Debug, Clone, Default)]
pub struct RailNetwork {
    pub stations: HashMap<u32, Station>,
    station_by_unit: HashMap<i32, u32>,
    next_station_id: u32,
    pub railroads: HashMap<u32, Railroad>,
    next_railroad_id: u32,
    pub clusters: HashMap<u32, Cluster>,
    next_cluster_id: u32,
    dirty_clusters: Vec<u32>,
    grid: RailSpatialGrid,
}

impl RailNetwork {
    fn new_station_id(&mut self) -> u32 {
        if self.next_station_id == 0 {
            self.next_station_id = 1; // TS: 0 reserved as invalid/sentinel
        }
        let id = self.next_station_id;
        self.next_station_id += 1;
        id
    }

    fn new_railroad_id(&mut self) -> u32 {
        let id = self.next_railroad_id;
        self.next_railroad_id += 1;
        id
    }

    fn new_cluster(&mut self) -> u32 {
        let id = self.next_cluster_id;
        self.next_cluster_id += 1;
        self.clusters.insert(id, Cluster::default());
        id
    }

    /// Upper bound for A* node array sizing (TS `StationManagerImpl.count()`).
    pub fn station_slot_count(&self) -> usize {
        self.next_station_id.max(1) as usize
    }

    pub fn find_station_by_unit(&self, unit_id: i32) -> Option<u32> {
        self.station_by_unit.get(&unit_id).copied()
    }

    fn station_neighbors(&self, station_id: u32) -> Vec<u32> {
        let Some(st) = self.stations.get(&station_id) else {
            return Vec::new();
        };
        st.railroads
            .iter()
            .filter_map(|rid| {
                self.railroads.get(rid).map(|r| {
                    if r.from != station_id {
                        r.from
                    } else {
                        r.to
                    }
                })
            })
            .collect()
    }

    /// TS `Cluster.addStation` + `TrainStation.setCluster`. TS adds the station to the new
    /// cluster's `stations` Set *first*, then unconditionally removes it from whatever
    /// cluster it previously pointed to (`this.cluster.removeStation`) - even if that's the
    /// *same* cluster we just added it to. When a new station connects to two neighbors that
    /// both resolve to the same cluster within one `connectToNearbyStations` call, the second
    /// `addStation` call re-triggers this remove-after-add and the station ends up with
    /// `cluster` pointing at the cluster but absent from its `stations` set. This is real TS
    /// behavior we must reproduce byte-for-byte, so the removal below is unconditional (no
    /// `prev_id != cluster_id` guard) and happens *after* the add, matching TS's exact order.
    fn cluster_add_station(&mut self, cluster_id: u32, station_id: u32) {
        if let Some(c) = self.clusters.get_mut(&cluster_id) {
            if !c.stations.contains(&station_id) {
                c.stations.push(station_id);
            }
        }
        let prev = self.stations.get(&station_id).and_then(|s| s.cluster);
        if let Some(prev_id) = prev {
            if let Some(c) = self.clusters.get_mut(&prev_id) {
                c.stations.retain(|&s| s != station_id);
            }
        }
        if let Some(st) = self.stations.get_mut(&station_id) {
            st.cluster = Some(cluster_id);
        }
    }

    fn merge_clusters(&mut self, cluster_ids: &[u32]) -> u32 {
        let merged = self.new_cluster();
        for &cid in cluster_ids {
            let members = self
                .clusters
                .get(&cid)
                .map(|c| c.stations.clone())
                .unwrap_or_default();
            for s in members {
                self.cluster_add_station(merged, s);
            }
        }
        merged
    }

    fn delete_cluster(&mut self, cluster_id: u32) {
        // TS `deleteCluster` clears the Cluster object's station set and nulls
        // `setCluster(null)` on members still in that set — but leaves the Cluster
        // *object* alive. Orphans that still hold a reference to it can later
        // `addStation` and resurrect membership (curr-b030 city 141 snapped onto a
        // rail whose `from` pointed at deleted cluster 28: TS revived size-1 cluster,
        // native had removed the HashMap entry so `cluster_add_station` no-op'd and
        // left a stale pointer). Keep an empty entry in `clusters` to match.
        if let Some(c) = self.clusters.get_mut(&cluster_id) {
            let members = std::mem::take(&mut c.stations);
            for s in members {
                if let Some(st) = self.stations.get_mut(&s) {
                    if st.cluster == Some(cluster_id) {
                        st.cluster = None;
                    }
                }
            }
        }
        self.dirty_clusters.retain(|&c| c != cluster_id);
    }

    fn compute_cluster(&self, start: u32) -> Vec<u32> {
        let mut visited = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        while let Some(current) = queue.pop_front() {
            if visited.contains(&current) {
                continue;
            }
            visited.push(current);
            for n in self.station_neighbors(current) {
                if !visited.contains(&n) {
                    queue.push_back(n);
                }
            }
        }
        visited
    }

    fn distance_from(&self, start: u32, dest: u32, max_distance: i32) -> i32 {
        if start == dest {
            return 0;
        }
        let mut visited: Vec<u32> = Vec::new();
        let mut queue: std::collections::VecDeque<(u32, i32)> = std::collections::VecDeque::new();
        queue.push_back((start, 0));
        while let Some((station, dist)) = queue.pop_front() {
            if visited.contains(&station) {
                continue;
            }
            visited.push(station);
            if dist >= max_distance {
                continue;
            }
            for n in self.station_neighbors(station) {
                if n == dest {
                    return dist + 1;
                }
                if !visited.contains(&n) {
                    queue.push_back((n, dist + 1));
                }
            }
        }
        -1
    }

    pub fn station_cluster(&self, station_id: u32) -> Option<u32> {
        self.stations.get(&station_id).and_then(|s| s.cluster)
    }
}

pub fn station_tile(game: &Game, rn: &RailNetwork, station_id: u32) -> Option<TileRef> {
    let st = rn.stations.get(&station_id)?;
    // TS reads `station.unit.tile()` via the live unit reference — owner may
    // have changed after capture without reconnecting the station.
    let owner = game
        .find_unit_owner(st.unit_id)
        .unwrap_or(st.owner_small_id);
    game.unit_tile_of(owner, st.unit_id)
}

pub fn station_unit_type(rn: &RailNetwork, station_id: u32) -> Option<String> {
    rn.stations.get(&station_id).map(|s| s.unit_type.clone())
}

pub fn station_active(game: &Game, rn: &RailNetwork, station_id: u32) -> bool {
    // TS `TrainStation.isActive()` is `this.unit.isActive()` — resolve the
    // unit by id globally so a captured station isn't treated as dead just
    // because `Station.owner_small_id` is stale.
    rn.stations
        .get(&station_id)
        .is_some_and(|st| game.find_unit_owner(st.unit_id).is_some())
}

/// Keep `Station.owner_small_id` in sync when a station building is captured
/// (TS `Unit.setOwner` leaves the TrainStation pointing at the same Unit).
pub fn update_station_owner_for_unit(rn: &mut RailNetwork, unit_id: i32, new_owner: u16) {
    if let Some(station_id) = rn.station_by_unit.get(&unit_id).copied() {
        if let Some(st) = rn.stations.get_mut(&station_id) {
            st.owner_small_id = new_owner;
        }
    }
}

/// TS `TrainStation.tradeAvailable` - a destination station is available to `source_owner`
/// if it's the same owner (self-trade), or neither side embargoes the other.
fn station_trade_available(game: &Game, station: &Station, source_owner: u16) -> bool {
    let owner = game
        .find_unit_owner(station.unit_id)
        .unwrap_or(station.owner_small_id);
    owner == source_owner || game.can_trade(owner, source_owner)
}

/// TS `Cluster.hasAnyTradeDestination`.
pub fn cluster_has_any_trade_destination(game: &Game, cluster_id: u32, source_owner: u16) -> bool {
    let rn = &game.rail_network;
    rn.clusters.get(&cluster_id).is_some_and(|c| {
        c.stations.iter().any(|sid| {
            rn.stations.get(sid).is_some_and(|s| {
                is_trade_type(&s.unit_type) && station_trade_available(game, s, source_owner)
            })
        })
    })
}

/// TS `Cluster.randomTradeDestination` - reservoir sampling over stations eligible for trade
/// with `source_owner` (i.e. `TrainStation.tradeAvailable`).
pub fn cluster_random_trade_destination(
    game: &Game,
    cluster_id: u32,
    source_owner: u16,
    random: &mut PseudoRandom,
) -> Option<u32> {
    let rn = &game.rail_network;
    let cluster = rn.clusters.get(&cluster_id)?;
    let mut selected = None;
    let mut eligible_seen: i32 = 0;
    for &sid in &cluster.stations {
        let Some(st) = rn.stations.get(&sid) else {
            continue;
        };
        if !is_trade_type(&st.unit_type) {
            continue;
        }
        if !station_trade_available(game, st, source_owner) {
            continue;
        }
        eligible_seen += 1;
        if random.next_int(0, eligible_seen) == 0 {
            selected = Some(sid);
        }
    }
    selected
}

/// TS `RailNetworkImpl.connectStation` - registers a new station, then either snaps it onto
/// an existing railroad segment or connects it directly to nearby stations.
pub fn connect_station(
    game: &mut Game,
    owner_small_id: u16,
    unit_id: i32,
    unit_type_str: &str,
) -> u32 {
    let mut rn = std::mem::take(&mut game.rail_network);
    let id = rn.new_station_id();
    rn.stations.insert(
        id,
        Station {
            id,
            owner_small_id,
            unit_id,
            unit_type: unit_type_str.to_string(),
            cluster: None,
            railroads: Vec::new(),
        },
    );
    // TS `StationManager.findStation(unit)` scans the insertion-ordered Set and returns
    // the first station object for a Unit. Duplicate TrainStationExecutions can exist for
    // the same Unit, so do not overwrite an earlier mapping with a later duplicate.
    rn.station_by_unit.entry(unit_id).or_insert(id);

    if !connect_to_existing_rails(game, &mut rn, id) {
        connect_to_nearby_stations(game, &mut rn, id);
    }
    game.rail_network = rn;
    id
}

fn connect_to_existing_rails(game: &Game, rn: &mut RailNetwork, station_id: u32) -> bool {
    let Some(tile) = station_tile(game, rn, station_id) else {
        return false;
    };
    let rails = rn.grid.query(game, tile, STATION_RADIUS);
    let mut edited_clusters: Vec<u32> = Vec::new();

    for rail_id in rails {
        let Some(rail) = rn.railroads.get(&rail_id).cloned() else {
            continue;
        };
        let closest_idx = closest_tile_index(game, &rail.tiles, tile);
        if closest_idx == 0 || closest_idx >= rail.tiles.len() {
            continue;
        }

        // Disconnect the old rail - it becomes invalid.
        if let Some(from_st) = rn.stations.get_mut(&rail.from) {
            from_st.railroads.retain(|&r| r != rail.id);
        }
        if let Some(to_st) = rn.stations.get_mut(&rail.to) {
            to_st.railroads.retain(|&r| r != rail.id);
        }
        rn.grid.unregister(rail.id);
        rn.railroads.remove(&rail.id);

        let new_from_id = rn.new_railroad_id();
        let new_from = Railroad {
            id: new_from_id,
            from: rail.from,
            to: station_id,
            tiles: rail.tiles[..closest_idx].to_vec(),
        };
        let new_to_id = rn.new_railroad_id();
        let new_to = Railroad {
            id: new_to_id,
            from: station_id,
            to: rail.to,
            tiles: rail.tiles[closest_idx..].to_vec(),
        };

        if let Some(st) = rn.stations.get_mut(&station_id) {
            st.railroads.push(new_from_id);
            st.railroads.push(new_to_id);
        }
        if let Some(st) = rn.stations.get_mut(&rail.from) {
            st.railroads.push(new_from_id);
        }
        if let Some(st) = rn.stations.get_mut(&rail.to) {
            st.railroads.push(new_to_id);
        }
        rn.grid.register(game, &new_to);
        rn.grid.register(game, &new_from);
        rn.railroads.insert(new_from_id, new_from);
        rn.railroads.insert(new_to_id, new_to);

        if let Some(cluster) = rn.stations.get(&rail.from).and_then(|s| s.cluster) {
            rn.cluster_add_station(cluster, station_id);
            if !edited_clusters.contains(&cluster) {
                edited_clusters.push(cluster);
            }
        }
    }

    if edited_clusters.len() > 1 {
        rn.merge_clusters(&edited_clusters);
    }
    !edited_clusters.is_empty()
}

fn connect_to_nearby_stations(game: &Game, rn: &mut RailNetwork, station_id: u32) {
    let Some(tile) = station_tile(game, rn, station_id) else {
        return;
    };
    let max_range = game.wire.train_station_max_range();
    let min_range_sq = (game.wire.train_station_min_range() as f64).powi(2);
    let this_unit_id = rn.stations.get(&station_id).map(|s| s.unit_id);

    let mut neighbors = game.nearby_structures_any(
        tile,
        max_range,
        &[unit_type::CITY, unit_type::FACTORY, unit_type::PORT],
    );
    neighbors.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap());

    let mut edited_clusters: Vec<u32> = Vec::new();
    for (_owner, unit_id, _t, dist_sq) in neighbors {
        if Some(unit_id) == this_unit_id {
            continue;
        }
        let Some(neighbor_station_id) = rn.find_station_by_unit(unit_id) else {
            continue;
        };
        let distance_to_station =
            rn.distance_from(neighbor_station_id, station_id, MAX_CONNECTION_DISTANCE);
        let Some(neighbor_cluster) = rn.station_cluster(neighbor_station_id) else {
            continue;
        };
        let connection_available =
            distance_to_station > MAX_CONNECTION_DISTANCE || distance_to_station == -1;
        if connection_available && dist_sq > min_range_sq {
            if connect(game, rn, station_id, neighbor_station_id) {
                rn.cluster_add_station(neighbor_cluster, station_id);
                if !edited_clusters.contains(&neighbor_cluster) {
                    edited_clusters.push(neighbor_cluster);
                }
            }
        }
    }

    if edited_clusters.len() > 1 {
        rn.merge_clusters(&edited_clusters);
    } else if edited_clusters.is_empty() {
        let new_cluster = rn.new_cluster();
        rn.cluster_add_station(new_cluster, station_id);
    }
}

fn connect(game: &Game, rn: &mut RailNetwork, a: u32, b: u32) -> bool {
    let (Some(tile_a), Some(tile_b)) = (station_tile(game, rn, a), station_tile(game, rn, b))
    else {
        return false;
    };
    let path = find_rail_tile_path(game, tile_a, tile_b);
    if !path.is_empty() && (path.len() as f64) < game.wire.railroad_max_size() {
        let id = rn.new_railroad_id();
        let railroad = Railroad {
            id,
            from: a,
            to: b,
            tiles: path,
        };
        if let Some(st) = rn.stations.get_mut(&a) {
            st.railroads.push(id);
        }
        if let Some(st) = rn.stations.get_mut(&b) {
            st.railroads.push(id);
        }
        rn.grid.register(game, &railroad);
        rn.railroads.insert(id, railroad);
        true
    } else {
        false
    }
}

fn closest_tile_index(game: &Game, tiles: &[TileRef], to: TileRef) -> usize {
    if tiles.is_empty() {
        return usize::MAX;
    }
    let to_x = game.x(to) as i64;
    let to_y = game.y(to) as i64;
    let mut closest = 0;
    let mut min_d = i64::MAX;
    for (i, &t) in tiles.iter().enumerate() {
        let dx = game.x(t) as i64 - to_x;
        let dy = game.y(t) as i64 - to_y;
        let d = dx * dx + dy * dy;
        if d < min_d {
            min_d = d;
            closest = i;
        }
    }
    closest
}

/// TS `RailNetworkImpl.removeStation` (`Game.removeUnit` hook).
pub fn remove_station(game: &mut Game, unit_id: i32) {
    let mut rn = std::mem::take(&mut game.rail_network);
    if let Some(station_id) = rn.station_by_unit.remove(&unit_id) {
        // Disconnect from network: delete every incident railroad.
        let railroads = rn
            .stations
            .get(&station_id)
            .map(|s| s.railroads.clone())
            .unwrap_or_default();
        for rid in railroads {
            if let Some(r) = rn.railroads.remove(&rid) {
                if let Some(st) = rn.stations.get_mut(&r.from) {
                    st.railroads.retain(|&x| x != rid);
                }
                if let Some(st) = rn.stations.get_mut(&r.to) {
                    st.railroads.retain(|&x| x != rid);
                }
                rn.grid.unregister(rid);
            }
        }
        if let Some(st) = rn.stations.get_mut(&station_id) {
            st.railroads.clear();
        }

        let cluster = rn.stations.get(&station_id).and_then(|s| s.cluster);
        rn.stations.remove(&station_id);
        if let Some(next_station_id) = rn
            .stations
            .iter()
            .filter_map(|(&sid, st)| (st.unit_id == unit_id).then_some(sid))
            .min()
        {
            rn.station_by_unit.insert(unit_id, next_station_id);
        }

        if let Some(cluster_id) = cluster {
            if let Some(c) = rn.clusters.get_mut(&cluster_id) {
                c.stations.retain(|&s| s != station_id);
            }
            let empty = rn
                .clusters
                .get(&cluster_id)
                .is_some_and(|c| c.stations.is_empty());
            if empty {
                rn.delete_cluster(cluster_id);
            } else if !rn.dirty_clusters.contains(&cluster_id) {
                rn.dirty_clusters.push(cluster_id);
            }
        }
    }
    game.rail_network = rn;
}

/// TS `RecomputeRailClusterExecution.tick` -> `RailNetworkImpl.recomputeClusters`.
pub fn recompute_clusters(game: &mut Game) {
    let mut rn = std::mem::take(&mut game.rail_network);
    if !rn.dirty_clusters.is_empty() {
        let dirty = std::mem::take(&mut rn.dirty_clusters);
        for cluster_id in dirty {
            let Some(cluster) = rn.clusters.get(&cluster_id) else {
                continue;
            };
            let mut remaining = cluster.stations.clone();
            while let Some(&next_station) = remaining.first() {
                let connected = rn.compute_cluster(next_station);
                remaining.retain(|s| !connected.contains(s));
                if !remaining.is_empty() {
                    let new_cluster = rn.new_cluster();
                    for s in &connected {
                        rn.cluster_add_station(new_cluster, *s);
                    }
                }
            }
        }
    }
    game.rail_network = rn;
}

/// TS `RailNetworkImpl.findStationsPath` -> `PathFinder.Station.ts`.
pub fn find_stations_path(game: &Game, rn: &RailNetwork, from: u32, to: u32) -> Vec<u32> {
    let to_cluster = rn.station_cluster(to);
    let from_cluster = rn.station_cluster(from);
    if to_cluster.is_none() || from_cluster != to_cluster {
        return Vec::new();
    }
    let num_nodes = rn.station_slot_count();
    let max_priority = (game.map.width + game.map.height).max(1);

    let neighbors_fn = |node: u32, buf: &mut [u32]| -> usize {
        let neighbors = rn.station_neighbors(node);
        let n = neighbors.len().min(buf.len());
        buf[..n].copy_from_slice(&neighbors[..n]);
        n
    };
    let cost_fn = |_from: u32, _to: u32, _prev: Option<u32>| -> f64 { 1.0 };
    let heuristic_fn = |node: u32, goal: u32| -> f64 {
        let (Some(a), Some(b)) = (station_tile(game, rn, node), station_tile(game, rn, goal))
        else {
            return 0.0;
        };
        (game.x(a) as i64 - game.x(b) as i64).unsigned_abs() as f64
            + (game.y(a) as i64 - game.y(b) as i64).unsigned_abs() as f64
    };

    astar_generic(num_nodes, &[from], to, max_priority, 32, neighbors_fn, cost_fn, heuristic_fn)
        .unwrap_or_default()
}

/// TS `Railroad.getOrientedRailroad` - oriented tiles for a `from -> to` hop along the graph.
/// TS `TrainStation.getRailroadTo` looks up a `Map<TrainStation, Railroad>` keyed by
/// neighbor, which `addRailroad` overwrites on every call - so when two railroads connect
/// the same station pair (e.g. two nearby rails both split by one new station), the
/// *most recently added* edge wins, not the first. Iterate `st.railroads` back-to-front
/// (append-only, so last-added is scanned first) to match that overwrite semantics.
pub fn oriented_railroad_tiles(rn: &RailNetwork, from: u32, to: u32) -> Option<Vec<TileRef>> {
    let st = rn.stations.get(&from)?;
    for &rid in st.railroads.iter().rev() {
        let Some(r) = rn.railroads.get(&rid) else {
            continue;
        };
        if r.from == to || r.to == to {
            return Some(if r.to == to {
                r.tiles.clone()
            } else {
                let mut t = r.tiles.clone();
                t.reverse();
                t
            });
        }
    }
    None
}

// ---------------------------------------------------------------------------------------------
// Generic A* (TS `algorithms/AStar.ts` + `algorithms/PriorityQueue.ts::BucketQueue`).
// ---------------------------------------------------------------------------------------------

struct BucketQueue {
    buckets: Vec<Vec<u32>>,
    min_bucket: usize,
    max_bucket: usize,
    size: usize,
}

impl BucketQueue {
    fn new(max_priority: u32) -> Self {
        let max_bucket = max_priority as usize + 1;
        Self {
            buckets: vec![Vec::new(); max_bucket],
            min_bucket: max_bucket,
            max_bucket,
            size: 0,
        }
    }

    fn push(&mut self, node: u32, priority: f64) {
        let p = priority.floor();
        let bucket = if p <= 0.0 {
            0
        } else {
            (p as usize).min(self.max_bucket - 1)
        };
        self.buckets[bucket].push(node);
        self.size += 1;
        if bucket < self.min_bucket {
            self.min_bucket = bucket;
        }
    }

    fn pop(&mut self) -> Option<u32> {
        while self.min_bucket < self.max_bucket {
            if let Some(v) = self.buckets[self.min_bucket].pop() {
                self.size -= 1;
                return Some(v);
            }
            self.min_bucket += 1;
        }
        None
    }

    fn is_empty(&self) -> bool {
        self.size == 0
    }
}

#[allow(clippy::too_many_arguments)]
fn astar_generic(
    num_nodes: usize,
    starts: &[u32],
    goal: u32,
    max_priority: u32,
    max_neighbors: usize,
    mut neighbors_fn: impl FnMut(u32, &mut [u32]) -> usize,
    mut cost_fn: impl FnMut(u32, u32, Option<u32>) -> f64,
    mut heuristic_fn: impl FnMut(u32, u32) -> f64,
) -> Option<Vec<u32>> {
    if num_nodes == 0 {
        return None;
    }
    let mut queue = BucketQueue::new(max_priority);
    let mut g_score = vec![f64::INFINITY; num_nodes];
    let mut came_from: Vec<i64> = vec![-1; num_nodes];
    let mut closed = vec![false; num_nodes];

    for &s in starts {
        if (s as usize) >= num_nodes {
            continue;
        }
        g_score[s as usize] = 0.0;
        came_from[s as usize] = -1;
        queue.push(s, heuristic_fn(s, goal));
    }

    let mut iterations: i64 = 500_000;
    let mut buf = vec![0u32; max_neighbors];
    while !queue.is_empty() {
        iterations -= 1;
        if iterations <= 0 {
            return None;
        }
        let current = queue.pop()?;
        if closed[current as usize] {
            continue;
        }
        closed[current as usize] = true;

        if current == goal {
            let mut path = Vec::new();
            let mut cur = goal as i64;
            while cur != -1 {
                path.push(cur as u32);
                cur = came_from[cur as usize];
            }
            path.reverse();
            return Some(path);
        }

        let cur_g = g_score[current as usize];
        let prev = came_from[current as usize];
        let prev_opt = if prev == -1 { None } else { Some(prev as u32) };
        let count = neighbors_fn(current, &mut buf);
        for i in 0..count {
            let neighbor = buf[i];
            if (neighbor as usize) >= num_nodes || closed[neighbor as usize] {
                continue;
            }
            let tentative = cur_g + cost_fn(current, neighbor, prev_opt);
            if tentative < g_score[neighbor as usize] {
                came_from[neighbor as usize] = current as i64;
                g_score[neighbor as usize] = tentative;
                queue.push(neighbor, tentative + heuristic_fn(neighbor, goal));
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------------------------
// Rail-tile A* on the mini map (TS `AStar.Rail.ts` + `transformers/MiniMapTransformer.ts`).
// ---------------------------------------------------------------------------------------------

const RAIL_WATER_PENALTY: f64 = 5.0;
const RAIL_HEURISTIC_WEIGHT: f64 = 2.0;
const RAIL_DIRECTION_CHANGE_PENALTY: f64 = 3.0;

fn rail_is_traversable(mini: &GameMap, to: TileRef, from_shoreline: bool) -> bool {
    if !mini.is_water(to) {
        return true;
    }
    from_shoreline || mini.is_shoreline(to)
}

fn rail_neighbors(mini: &GameMap, node: u32, buf: &mut [u32; 4]) -> usize {
    let width = mini.width;
    let num_nodes = width * mini.height;
    let x = node % width;
    let from_shoreline = mini.is_shoreline(node);
    let mut count = 0;
    if node >= width {
        let n = node - width;
        if rail_is_traversable(mini, n, from_shoreline) {
            buf[count] = n;
            count += 1;
        }
    }
    if node < num_nodes - width {
        let n = node + width;
        if rail_is_traversable(mini, n, from_shoreline) {
            buf[count] = n;
            count += 1;
        }
    }
    if x != 0 {
        let n = node - 1;
        if rail_is_traversable(mini, n, from_shoreline) {
            buf[count] = n;
            count += 1;
        }
    }
    if x != width - 1 {
        let n = node + 1;
        if rail_is_traversable(mini, n, from_shoreline) {
            buf[count] = n;
            count += 1;
        }
    }
    count
}

fn rail_cost(mini: &GameMap, from: u32, to: u32, prev: Option<u32>) -> f64 {
    let penalized = mini.is_water(to) || mini.is_shoreline(to);
    let mut c = if penalized { 1.0 + RAIL_WATER_PENALTY } else { 1.0 };
    if let Some(p) = prev {
        let d1 = from as i64 - p as i64;
        let d2 = to as i64 - from as i64;
        if d1 != d2 {
            c += RAIL_DIRECTION_CHANGE_PENALTY;
        }
    }
    c
}

fn rail_heuristic(mini: &GameMap, node: u32, goal: u32) -> f64 {
    let nx = mini.x(node) as i64;
    let ny = mini.y(node) as i64;
    let gx = mini.x(goal) as i64;
    let gy = mini.y(goal) as i64;
    RAIL_HEURISTIC_WEIGHT * ((nx - gx).abs() + (ny - gy).abs()) as f64
}

fn to_mini_ref(full: &GameMap, mini: &GameMap, tile: TileRef) -> TileRef {
    mini.ref_xy(full.x(tile) / 2, full.y(tile) / 2)
}

fn upscale_cells(cells: &[(u32, u32)], scale: u32) -> Vec<(u32, u32)> {
    if cells.is_empty() {
        return Vec::new();
    }
    let scaled: Vec<(u32, u32)> = cells.iter().map(|&(x, y)| (x * scale, y * scale)).collect();
    let mut smooth = Vec::new();
    for i in 0..scaled.len().saturating_sub(1) {
        let current = scaled[i];
        let next = scaled[i + 1];
        smooth.push(current);
        let dx = next.0 as i32 - current.0 as i32;
        let dy = next.1 as i32 - current.1 as i32;
        let steps = dx.abs().max(dy.abs()) as u32;
        for step in 1..steps {
            smooth.push((
                (current.0 as f64 + (dx as f64 * step as f64) / steps as f64).round() as u32,
                (current.1 as f64 + (dy as f64 * step as f64) / steps as f64).round() as u32,
            ));
        }
    }
    if let Some(&last) = scaled.last() {
        smooth.push(last);
    }
    smooth
}

fn upscale_mini_path(full: &GameMap, mini: &GameMap, mini_path: &[TileRef]) -> Vec<TileRef> {
    let cells: Vec<(u32, u32)> = mini_path.iter().map(|&r| (mini.x(r), mini.y(r))).collect();
    upscale_cells(&cells, 2)
        .into_iter()
        .map(|(cx, cy)| full.ref_xy(cx, cy))
        .collect()
}

fn fix_path_extremes(
    mut path: Vec<TileRef>,
    cell_src: TileRef,
    cell_dst: TileRef,
) -> Vec<TileRef> {
    match path.iter().position(|&t| t == cell_src) {
        Some(idx) if idx != 0 => path = path[idx..].to_vec(),
        Some(_) => {}
        None => path.insert(0, cell_src),
    }
    match path.iter().position(|&t| t == cell_dst) {
        Some(idx) if idx + 1 != path.len() => path.truncate(idx + 1),
        Some(_) => {}
        None => path.push(cell_dst),
    }
    path
}

/// TS `RailPathFinderServiceImpl.findTilePath` (single-source; every real call site passes a
/// single tile, never an array, so multi-source support from `MiniMapTransformer` is omitted).
pub fn find_rail_tile_path(game: &Game, from: TileRef, to: TileRef) -> Vec<TileRef> {
    let full = &game.map;
    let mini = &game.mini_map;
    let mini_from = to_mini_ref(full, mini, from);
    let mini_to = to_mini_ref(full, mini, to);

    let num_nodes = (mini.width * mini.height) as usize;
    let max_cost = 1.0 + RAIL_WATER_PENALTY + RAIL_DIRECTION_CHANGE_PENALTY;
    let max_priority =
        (RAIL_HEURISTIC_WEIGHT * (mini.width + mini.height) as f64 * max_cost) as u32;

    let neighbors_fn = |node: u32, buf: &mut [u32]| -> usize {
        let mut b4 = [0u32; 4];
        let n = rail_neighbors(mini, node, &mut b4);
        buf[..n].copy_from_slice(&b4[..n]);
        n
    };
    let cost_fn = |from: u32, to: u32, prev: Option<u32>| rail_cost(mini, from, to, prev);
    let heuristic_fn = |node: u32, goal: u32| rail_heuristic(mini, node, goal);

    let Some(mini_path) = astar_generic(
        num_nodes,
        &[mini_from],
        mini_to,
        max_priority,
        4,
        neighbors_fn,
        cost_fn,
        heuristic_fn,
    ) else {
        return Vec::new();
    };
    if mini_path.is_empty() {
        return Vec::new();
    }

    let upscaled = upscale_mini_path(full, mini, &mini_path);
    if upscaled.is_empty() {
        return Vec::new();
    }
    fix_path_extremes(upscaled, from, to)
}

// ---------------------------------------------------------------------------------------------
// Rail spatial grid (TS `RailroadSpatialGrid.ts`).
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
struct RailSpatialGrid {
    cells: HashMap<(i32, i32), Vec<u32>>,
    rail_cells: HashMap<u32, Vec<(i32, i32)>>,
}

impl RailSpatialGrid {
    fn cell_of(x: i32, y: i32) -> (i32, i32) {
        (
            x.div_euclid(GRID_CELL_SIZE as i32),
            y.div_euclid(GRID_CELL_SIZE as i32),
        )
    }

    fn register(&mut self, game: &Game, rail: &Railroad) {
        self.unregister(rail.id);
        let mut keys = Vec::new();
        for &t in &rail.tiles {
            let k = Self::cell_of(game.x(t) as i32, game.y(t) as i32);
            if keys.contains(&k) {
                continue;
            }
            self.cells.entry(k).or_default().push(rail.id);
            keys.push(k);
        }
        if !keys.is_empty() {
            self.rail_cells.insert(rail.id, keys);
        }
    }

    fn unregister(&mut self, rail_id: u32) {
        if let Some(keys) = self.rail_cells.remove(&rail_id) {
            for k in keys {
                if let Some(v) = self.cells.get_mut(&k) {
                    v.retain(|&r| r != rail_id);
                    if v.is_empty() {
                        self.cells.remove(&k);
                    }
                }
            }
        }
    }

    fn query(&self, game: &Game, tile: TileRef, radius: u32) -> Vec<u32> {
        let x = game.x(tile) as i32;
        let y = game.y(tile) as i32;
        let r = radius as i32;
        let (c0x, c0y) = Self::cell_of(x - r, y - r);
        let (c1x, c1y) = Self::cell_of(x + r, y + r);
        let mut out = Vec::new();
        // TS `RailSpatialGrid.query` iterates `cx` outer, `cy` inner - matching that order
        // matters because callers (e.g. `connectToExistingRails`) use encounter order, not a
        // distance sort, to decide which cluster a new station joins first.
        for cx in c0x..=c1x {
            for cy in c0y..=c1y {
                if let Some(v) = self.cells.get(&(cx, cy)) {
                    for &rid in v {
                        if !out.contains(&rid) {
                            out.push(rid);
                        }
                    }
                }
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------
// Trade stop handling (TS `TrainStation.ts::TradeStationStopHandler`/`FactoryStopHandler`).
// ---------------------------------------------------------------------------------------------

use crate::core::config::TrainRelation;

fn train_relation(game: &Game, train_owner: u16, station_owner: u16) -> TrainRelation {
    if train_owner == station_owner {
        return TrainRelation::SelfTrade;
    }
    if game.players_on_same_team(train_owner, station_owner) {
        return TrainRelation::Team;
    }
    if game.is_allied_with(train_owner, station_owner) {
        return TrainRelation::Ally;
    }
    TrainRelation::Other
}

/// TS `TrainStation.onTrainStop` - City/Port pay gold; Factory is a no-op stop.
pub fn on_train_stop(game: &mut Game, station_id: u32, train_owner: u16, trade_stops_visited: u32) {
    let Some((unit_id, cached_owner, utype)) = game
        .rail_network
        .stations
        .get(&station_id)
        .map(|s| (s.unit_id, s.owner_small_id, s.unit_type.clone()))
    else {
        return;
    };
    if !is_trade_type(&utype) {
        return; // Factory stop handler is a no-op in TS too.
    }
    // Live owner (TS `station.unit.owner()`), not the connect-time cache.
    let owner = game.find_unit_owner(unit_id).unwrap_or(cached_owner);
    let rel = train_relation(game, train_owner, owner);
    let gold = game.wire.train_gold(rel, trade_stops_visited);
    if train_owner != owner {
        game.add_gold(owner, gold);
    }
    game.add_gold(train_owner, gold);
}

pub fn is_trade_station_type(t: &str) -> bool {
    is_trade_type(t)
}

pub fn is_valid_station_type(t: &str) -> bool {
    is_station_type(t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schemas::unit_type;
    use crate::game::{Game, PlayerInfo, PlayerType};
    use crate::test_util::plains_game;

    fn add_nation(game: &mut Game, id: &str) -> u16 {
        game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type: PlayerType::Nation,
            client_id: None,
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        })
    }

    /// Capturing a city that is a train station must not make `station_active`
    /// flip false — TS `TrainStation.isActive()` follows the Unit, not a
    /// frozen owner id. Without this, in-flight trains delete on the next tick
    /// (Canada +7/−7 unit blip on curr-b030-s0-pangaea ~4242).
    #[test]
    fn station_stays_active_after_structure_capture() {
        let mut game = plains_game(40, 40);
        let a = add_nation(&mut game, "a");
        let b = add_nation(&mut game, "b");
        for sid in [a, b] {
            let tile = game.map.ref_xy(if sid == a { 0 } else { 20 }, 0);
            game.conquer(sid, tile);
            if let Some(p) = game.player_by_small_id_mut(sid) {
                p.spawn_tile = Some(tile);
                p.alive = true;
                p.gold = 1_000_000;
            }
        }
        let city_tile = game.map.ref_xy(0, 0);
        let city_id = game.build_unit(a, unit_type::CITY, city_tile);
        let station_id = connect_station(&mut game, a, city_id, unit_type::CITY);
        assert!(station_active(&game, &game.rail_network, station_id));

        game.capture_unit(a, b, city_id);
        assert!(
            station_active(&game, &game.rail_network, station_id),
            "captured station must remain active for in-flight trains"
        );
        assert_eq!(
            game.rail_network.stations.get(&station_id).unwrap().owner_small_id,
            b
        );
        assert_eq!(station_tile(&game, &game.rail_network, station_id), Some(city_tile));
    }

    /// TS `deleteCluster` leaves the Cluster object alive (cleared). Orphans that
    /// still point at it must be able to `addStation` again — removing the HashMap
    /// entry made `cluster_add_station` a no-op and stale-pointer'd new snaps
    /// (curr-b030 stations 141/171).
    #[test]
    fn delete_cluster_keeps_empty_entry_for_orphan_resurrection() {
        let mut rn = RailNetwork::default();
        let cluster = rn.new_cluster();
        let sid = rn.new_station_id();
        rn.stations.insert(
            sid,
            Station {
                id: sid,
                owner_small_id: 1,
                unit_id: 1,
                unit_type: unit_type::CITY.to_string(),
                cluster: None,
                railroads: Vec::new(),
            },
        );
        // Simulate an orphan still pointing at `cluster` after delete (neighbor `from`).
        let orphan = rn.new_station_id();
        rn.stations.insert(
            orphan,
            Station {
                id: orphan,
                owner_small_id: 1,
                unit_id: 2,
                unit_type: unit_type::CITY.to_string(),
                cluster: Some(cluster),
                railroads: Vec::new(),
            },
        );
        assert!(rn.clusters.get(&cluster).unwrap().stations.is_empty());
        rn.delete_cluster(cluster);
        assert!(
            rn.clusters.contains_key(&cluster),
            "empty cluster entry must remain after delete"
        );
        // New station snaps using orphan neighbor's stale cluster id (TS Cluster object).
        rn.cluster_add_station(cluster, sid);
        assert!(
            rn.clusters.get(&cluster).unwrap().stations.contains(&sid),
            "addStation onto cleared cluster must revive membership"
        );
        assert_eq!(rn.stations.get(&sid).unwrap().cluster, Some(cluster));
    }

    /// TS `StationManager.findStation(unit)` scans stations in insertion order,
    /// so duplicate station objects for one unit resolve to the first one until
    /// it is removed. A simple HashMap overwrite picks the newest duplicate and
    /// can connect later stations to the wrong rail graph node.
    #[test]
    fn duplicate_unit_station_lookup_matches_first_inserted_station() {
        let mut game = plains_game(20, 20);
        let owner = add_nation(&mut game, "owner");
        let tile = game.map.ref_xy(5, 5);
        game.conquer(owner, tile);

        let factory = game.build_unit(owner, unit_type::FACTORY, tile);
        let first = connect_station(&mut game, owner, factory, unit_type::FACTORY);
        let second = connect_station(&mut game, owner, factory, unit_type::FACTORY);

        assert_ne!(first, second, "test should create duplicate stations");
        assert_eq!(
            game.rail_network.find_station_by_unit(factory),
            Some(first),
            "lookup should preserve TS first-insertion semantics"
        );

        remove_station(&mut game, factory);
        assert_eq!(
            game.rail_network.find_station_by_unit(factory),
            Some(second),
            "after the first duplicate is removed, lookup advances to the next"
        );
    }
}
