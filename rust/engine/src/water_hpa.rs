//! Hierarchical water pathfinding (TS `AbstractGraph` + `AStar.WaterHierarchical`).

use crate::map::{GameMap, TileRef};
use crate::water::AstarHeap;
use std::collections::HashMap;

pub const LAND_MARKER: u32 = 0xff;
pub const CLUSTER_SIZE: u32 = 32;

// ── Connected components (mini-map, TS `ConnectedComponents`) ───────────────

struct ConnectedComponents {
    component_ids: Vec<u32>,
    component_sizes: Vec<u32>,
}

impl ConnectedComponents {
    fn new(map: &GameMap) -> Self {
        let n = (map.width * map.height) as usize;
        let mut ids = vec![0u32; n];
        for t in 0..n {
            if !map.is_water(t as TileRef) {
                ids[t] = LAND_MARKER;
            }
        }
        Self {
            component_ids: ids,
            component_sizes: vec![0],
        }
    }

    fn initialize(&mut self, map: &GameMap) {
        let width = map.width;
        let height = map.height;
        let n = (width * height) as usize;
        let last_row_start = ((height - 1) * width) as usize;
        let mut next_id = 0u32;
        let mut queue = vec![0i32; n];
        let ids = &mut self.component_ids;
        self.component_sizes.clear();
        self.component_sizes.push(0);

        for start in 0..n {
            let val = ids[start];
            if val == LAND_MARKER || val > 0 {
                continue;
            }
            next_id += 1;
            let component_id = next_id;
            self.component_sizes.push(0);
            let mut head = 0usize;
            let mut tail = 0usize;
            queue[tail] = start as i32;
            tail += 1;

            while head < tail {
                let seed = queue[head] as usize;
                head += 1;
                if ids[seed] != 0 {
                    continue;
                }
                let row_start = seed - (seed % width as usize);
                let mut left = seed;
                while left > row_start && ids[left - 1] == 0 {
                    left -= 1;
                }
                let row_end = row_start + width as usize - 1;
                let mut right = seed;
                while right < row_end && ids[right + 1] == 0 {
                    right += 1;
                }
                for x in left..=right {
                    ids[x] = component_id;
                    self.component_sizes[component_id as usize] += 1;
                    if x >= width as usize {
                        let above = x - width as usize;
                        if ids[above] == 0 {
                            queue[tail] = above as i32;
                            tail += 1;
                        }
                    }
                    if x < last_row_start {
                        let below = x + width as usize;
                        if ids[below] == 0 {
                            queue[tail] = below as i32;
                            tail += 1;
                        }
                    }
                }
            }
        }
    }

    fn get_component_id(&self, tile: TileRef) -> u32 {
        self.component_ids.get(tile as usize).copied().unwrap_or(0)
    }

    fn get_component_size(&self, component_id: u32) -> u32 {
        self.component_sizes
            .get(component_id as usize)
            .copied()
            .unwrap_or(0)
    }
}

// ── Stamp BFS grid (TS `BFS.Grid`) ──────────────────────────────────────────

struct BfsGrid {
    stamp: u32,
    visited_stamp: Vec<u32>,
    queue: Vec<i32>,
    dist: Vec<u16>,
}

impl BfsGrid {
    fn new(num_nodes: usize) -> Self {
        Self {
            stamp: 0,
            visited_stamp: vec![0; num_nodes],
            queue: vec![0; num_nodes],
            dist: vec![0; num_nodes],
        }
    }

    fn next_stamp(&mut self) -> u32 {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.visited_stamp.fill(0);
            self.stamp = 1;
        }
        self.stamp
    }

    /// Visitor: `Some(r)` = found; `None` with explore=false = reject; `None` with explore=true = continue.
    fn search<R>(
        &mut self,
        width: u32,
        height: u32,
        start: TileRef,
        max_distance: u32,
        is_valid: impl Fn(TileRef) -> bool,
        mut visitor: impl FnMut(TileRef, u32) -> BfsVisit<R>,
    ) -> Option<R> {
        let stamp = self.next_stamp();
        let last_row_start = (height - 1) * width;
        let mut head = 0usize;
        let mut tail = 0usize;
        self.visited_stamp[start as usize] = stamp;
        self.dist[start as usize] = 0;
        self.queue[tail] = start as i32;
        tail += 1;

        while head < tail {
            let node = self.queue[head] as TileRef;
            head += 1;
            let dist = self.dist[node as usize] as u32;

            match visitor(node, dist) {
                BfsVisit::Found(r) => return Some(r),
                BfsVisit::Reject => continue,
                BfsVisit::Continue => {}
            }

            let next_dist = dist + 1;
            if next_dist > max_distance {
                continue;
            }

            let x = node % width;
            let push = |bfs: &mut Self, nb: TileRef, tail: &mut usize, next_dist: u32| {
                if bfs.visited_stamp[nb as usize] != stamp && is_valid(nb) {
                    bfs.visited_stamp[nb as usize] = stamp;
                    bfs.dist[nb as usize] = next_dist as u16;
                    bfs.queue[*tail] = nb as i32;
                    *tail += 1;
                }
            };
            if node >= width {
                push(self, node - width, &mut tail, next_dist);
            }
            if node < last_row_start {
                push(self, node + width, &mut tail, next_dist);
            }
            if x != 0 {
                push(self, node - 1, &mut tail, next_dist);
            }
            if x != width - 1 {
                push(self, node + 1, &mut tail, next_dist);
            }
        }
        None
    }
}

enum BfsVisit<R> {
    Found(R),
    Reject,
    Continue,
}

// ── Abstract graph (TS `AbstractGraph`) ─────────────────────────────────────

#[derive(Clone, Debug)]
pub struct AbstractNode {
    pub id: usize,
    pub x: u32,
    pub y: u32,
    pub tile: TileRef,
    pub component_id: u32,
}

#[derive(Clone, Debug)]
pub struct AbstractEdge {
    pub id: usize,
    pub node_a: usize,
    pub node_b: usize,
    pub cost: u32,
    pub cluster_x: u32,
    pub cluster_y: u32,
}

#[derive(Clone, Debug)]
struct Cluster {
    x: u32,
    y: u32,
    node_ids: Vec<usize>,
}

pub struct AbstractGraph {
    pub cluster_size: u32,
    pub clusters_x: u32,
    pub clusters_y: u32,
    nodes: Vec<AbstractNode>,
    edges: Vec<AbstractEdge>,
    node_edge_ids: Vec<Vec<usize>>,
    clusters: Vec<Cluster>,
    path_cache: Vec<Option<Vec<TileRef>>>,
    water_components: ConnectedComponents,
}

impl AbstractGraph {
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn get_node(&self, id: usize) -> Option<&AbstractNode> {
        self.nodes.get(id)
    }

    pub fn get_edge(&self, id: usize) -> Option<&AbstractEdge> {
        self.edges.get(id)
    }

    pub fn get_node_edges(&self, node_id: usize) -> &[usize] {
        self.node_edge_ids
            .get(node_id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn get_edge_between(&self, node_a: usize, node_b: usize) -> Option<&AbstractEdge> {
        for &edge_id in self.get_node_edges(node_a) {
            let edge = &self.edges[edge_id];
            if edge.node_a == node_b || edge.node_b == node_b {
                return Some(edge);
            }
        }
        None
    }

    pub fn other_node(&self, edge: &AbstractEdge, node_id: usize) -> usize {
        if edge.node_a == node_id {
            edge.node_b
        } else {
            edge.node_a
        }
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn get_cached_path(&self, edge_id: usize, from_node_id: usize) -> Option<&[TileRef]> {
        let edge = self.edges.get(edge_id)?;
        let direction = if from_node_id == edge.node_a { 0 } else { 1 };
        let cache_index = edge_id * 2 + direction;
        self.path_cache.get(cache_index).and_then(|p| p.as_deref())
    }

    pub fn set_cached_path(&mut self, edge_id: usize, from_node_id: usize, path: Vec<TileRef>) {
        if let Some(edge) = self.edges.get(edge_id) {
            let direction = if from_node_id == edge.node_a { 0 } else { 1 };
            let cache_index = edge_id * 2 + direction;
            if cache_index < self.path_cache.len() {
                self.path_cache[cache_index] = Some(path);
            }
        }
    }

    fn init_path_cache(&mut self) {
        self.path_cache = vec![None; self.edges.len() * 2];
    }

    pub fn get_component_id(&self, tile: TileRef) -> u32 {
        self.water_components.get_component_id(tile)
    }

    pub fn get_component_size(&self, component_id: u32) -> u32 {
        self.water_components.get_component_size(component_id)
    }

    fn cluster_key(&self, cluster_x: u32, cluster_y: u32) -> usize {
        (cluster_y * self.clusters_x + cluster_x) as usize
    }

    fn get_cluster(&self, cluster_x: u32, cluster_y: u32) -> Option<&Cluster> {
        self.clusters.get(self.cluster_key(cluster_x, cluster_y))
    }

    fn add_node(&mut self, node: AbstractNode) {
        let id = node.id;
        self.nodes.push(node);
        if self.node_edge_ids.len() <= id {
            self.node_edge_ids.resize(id + 1, Vec::new());
        }
    }

    fn add_edge(&mut self, edge: AbstractEdge) {
        let id = edge.id;
        let a = edge.node_a;
        let b = edge.node_b;
        self.edges.push(edge);
        self.node_edge_ids[a].push(id);
        self.node_edge_ids[b].push(id);
    }

    fn set_cluster(&mut self, key: usize, cluster: Cluster) {
        if self.clusters.len() <= key {
            self.clusters.resize(
                key + 1,
                Cluster {
                    x: 0,
                    y: 0,
                    node_ids: Vec::new(),
                },
            );
        }
        self.clusters[key] = cluster;
    }

    fn add_node_to_cluster(&mut self, cluster_key: usize, node_id: usize) {
        if let Some(cluster) = self.clusters.get_mut(cluster_key) {
            if !cluster.node_ids.contains(&node_id) {
                cluster.node_ids.push(node_id);
            }
        }
    }
}

// ── Abstract graph builder (TS `AbstractGraphBuilder`) ───────────────────────

pub struct AbstractGraphBuilder<'a> {
    map: &'a GameMap,
    cluster_size: u32,
    graph: AbstractGraph,
    tile_to_node: HashMap<TileRef, usize>,
    next_node_id: usize,
    next_edge_id: usize,
    edge_between: HashMap<usize, HashMap<usize, usize>>,
    tile_bfs: BfsGrid,
}

impl<'a> AbstractGraphBuilder<'a> {
    pub fn new(map: &'a GameMap) -> Self {
        let width = map.width;
        let height = map.height;
        let cluster_size = CLUSTER_SIZE;
        let clusters_x = width.div_ceil(cluster_size);
        let clusters_y = height.div_ceil(cluster_size);
        let mut water_components = ConnectedComponents::new(map);
        water_components.initialize(map);
        Self {
            map,
            cluster_size,
            graph: AbstractGraph {
                cluster_size,
                clusters_x,
                clusters_y,
                nodes: Vec::new(),
                edges: Vec::new(),
                node_edge_ids: Vec::new(),
                clusters: Vec::new(),
                path_cache: Vec::new(),
                water_components,
            },
            tile_to_node: HashMap::new(),
            next_node_id: 0,
            next_edge_id: 0,
            edge_between: HashMap::new(),
            tile_bfs: BfsGrid::new((width * height) as usize),
        }
    }

    pub fn build(mut self) -> AbstractGraph {
        for cy in 0..self.graph.clusters_y {
            for cx in 0..self.graph.clusters_x {
                let key = self.graph.cluster_key(cx, cy);
                self.graph.set_cluster(
                    key,
                    Cluster {
                        x: cx,
                        y: cy,
                        node_ids: Vec::new(),
                    },
                );
            }
        }
        for cy in 0..self.graph.clusters_y {
            for cx in 0..self.graph.clusters_x {
                self.process_cluster(cx, cy);
            }
        }
        for cy in 0..self.graph.clusters_y {
            for cx in 0..self.graph.clusters_x {
                if self
                    .graph
                    .get_cluster(cx, cy)
                    .map(|c| !c.node_ids.is_empty())
                    .unwrap_or(false)
                {
                    self.build_cluster_connections(cx, cy);
                }
            }
        }
        self.graph.init_path_cache();
        self.graph
    }

    fn get_or_create_node(&mut self, x: u32, y: u32) -> usize {
        let tile = self.map.ref_xy(x, y);
        if let Some(&id) = self.tile_to_node.get(&tile) {
            return id;
        }
        let id = self.next_node_id;
        self.next_node_id += 1;
        let node = AbstractNode {
            id,
            x,
            y,
            tile,
            component_id: self.graph.water_components.get_component_id(tile),
        };
        self.graph.add_node(node);
        self.tile_to_node.insert(tile, id);
        id
    }

    fn process_cluster(&mut self, cx: u32, cy: u32) {
        let base_x = cx * self.cluster_size;
        let base_y = cy * self.cluster_size;
        if cx < self.graph.clusters_x - 1 {
            let edge_x = (base_x + self.cluster_size - 1).min(self.map.width - 1);
            let nodes = self.find_nodes_on_vertical_edge(edge_x, base_y);
            for node_id in nodes {
                let key_a = self.graph.cluster_key(cx, cy);
                let key_b = self.graph.cluster_key(cx + 1, cy);
                self.graph.add_node_to_cluster(key_a, node_id);
                self.graph.add_node_to_cluster(key_b, node_id);
            }
        }
        if cy < self.graph.clusters_y - 1 {
            let edge_y = (base_y + self.cluster_size - 1).min(self.map.height - 1);
            let nodes = self.find_nodes_on_horizontal_edge(edge_y, base_x);
            for node_id in nodes {
                let key_a = self.graph.cluster_key(cx, cy);
                let key_b = self.graph.cluster_key(cx, cy + 1);
                self.graph.add_node_to_cluster(key_a, node_id);
                self.graph.add_node_to_cluster(key_b, node_id);
            }
        }
    }

    fn find_nodes_on_vertical_edge(&mut self, x: u32, base_y: u32) -> Vec<usize> {
        let mut nodes = Vec::new();
        let max_y = (base_y + self.cluster_size).min(self.map.height);
        let mut span_start: Option<u32> = None;
        let try_add = |span_start: &mut Option<u32>,
                       y: u32,
                       nodes: &mut Vec<usize>,
                       this: &mut Self,
                       x: u32| {
            let Some(start) = span_start.take() else {
                return;
            };
            let span_length = y - start;
            let mid_y = start + span_length / 2;
            let node_id = this.get_or_create_node(x, mid_y);
            nodes.push(node_id);
        };
        for y in base_y..max_y {
            let tile = self.map.ref_xy(x, y);
            let next_tile = if x + 1 < self.map.width {
                self.map.ref_xy(x + 1, y)
            } else {
                TileRef::MAX
            };
            let is_entrance = self.map.is_water(tile)
                && next_tile != TileRef::MAX
                && self.map.is_water(next_tile);
            if is_entrance {
                if span_start.is_none() {
                    span_start = Some(y);
                }
            } else {
                try_add(&mut span_start, y, &mut nodes, self, x);
            }
        }
        try_add(&mut span_start, max_y, &mut nodes, self, x);
        nodes
    }

    fn find_nodes_on_horizontal_edge(&mut self, y: u32, base_x: u32) -> Vec<usize> {
        let mut nodes = Vec::new();
        let max_x = (base_x + self.cluster_size).min(self.map.width);
        let mut span_start: Option<u32> = None;
        let try_add = |span_start: &mut Option<u32>,
                       x: u32,
                       nodes: &mut Vec<usize>,
                       this: &mut Self,
                       y: u32| {
            let Some(start) = span_start.take() else {
                return;
            };
            let span_length = x - start;
            let mid_x = start + span_length / 2;
            let node_id = this.get_or_create_node(mid_x, y);
            nodes.push(node_id);
        };
        for x in base_x..max_x {
            let tile = self.map.ref_xy(x, y);
            let next_tile = if y + 1 < self.map.height {
                self.map.ref_xy(x, y + 1)
            } else {
                TileRef::MAX
            };
            let is_entrance = self.map.is_water(tile)
                && next_tile != TileRef::MAX
                && self.map.is_water(next_tile);
            if is_entrance {
                if span_start.is_none() {
                    span_start = Some(x);
                }
            } else {
                try_add(&mut span_start, x, &mut nodes, self, y);
            }
        }
        try_add(&mut span_start, max_x, &mut nodes, self, y);
        nodes
    }

    fn build_cluster_connections(&mut self, cx: u32, cy: u32) {
        let cluster = match self.graph.get_cluster(cx, cy) {
            Some(c) => c,
            None => return,
        };
        let node_ids = cluster.node_ids.clone();
        let cluster_min_x = cx * self.cluster_size;
        let cluster_min_y = cy * self.cluster_size;
        let cluster_max_x = (cluster_min_x + self.cluster_size - 1).min(self.map.width - 1);
        let cluster_max_y = (cluster_min_y + self.cluster_size - 1).min(self.map.height - 1);

        for i in 0..node_ids.len() {
            let from_id = node_ids[i];
            let from_tile = self.graph.nodes[from_id].tile;
            let from_comp = self.graph.nodes[from_id].component_id;
            let mut target_nodes = Vec::new();
            for j in (i + 1)..node_ids.len() {
                let other = &self.graph.nodes[node_ids[j]];
                if from_comp != other.component_id {
                    continue;
                }
                target_nodes.push(other.clone());
            }
            if target_nodes.is_empty() {
                continue;
            }
            let reachable = self.find_all_reachable_nodes_in_bounds(
                from_tile,
                &target_nodes,
                cluster_min_x,
                cluster_max_x,
                cluster_min_y,
                cluster_max_y,
            );
            for (target_id, cost) in reachable {
                self.add_or_update_edge(from_id, target_id, cost, cx, cy);
            }
        }
    }

    fn add_or_update_edge(
        &mut self,
        node_id_a: usize,
        node_id_b: usize,
        cost: u32,
        cluster_x: u32,
        cluster_y: u32,
    ) {
        let (lo, hi) = if node_id_a < node_id_b {
            (node_id_a, node_id_b)
        } else {
            (node_id_b, node_id_a)
        };
        let node_map = self.edge_between.entry(lo).or_default();
        if let Some(&edge_id) = node_map.get(&hi) {
            let edge = &mut self.graph.edges[edge_id];
            if cost < edge.cost {
                edge.cost = cost;
                edge.cluster_x = cluster_x;
                edge.cluster_y = cluster_y;
            }
            return;
        }
        let edge_id = self.next_edge_id;
        self.next_edge_id += 1;
        let edge = AbstractEdge {
            id: edge_id,
            node_a: lo,
            node_b: hi,
            cost,
            cluster_x,
            cluster_y,
        };
        node_map.insert(hi, edge_id);
        self.graph.add_edge(edge);
    }

    fn find_all_reachable_nodes_in_bounds(
        &mut self,
        from: TileRef,
        target_nodes: &[AbstractNode],
        min_x: u32,
        max_x: u32,
        min_y: u32,
        max_y: u32,
    ) -> Vec<(usize, u32)> {
        let from_x = self.map.x(from);
        let from_y = self.map.y(from);
        let mut tile_to_node_id = HashMap::new();
        let mut max_manhattan = 0u32;
        for node in target_nodes {
            tile_to_node_id.insert(node.tile, node.id);
            let dx = node.x.abs_diff(from_x);
            let dy = node.y.abs_diff(from_y);
            max_manhattan = max_manhattan.max(dx + dy);
        }
        let max_distance = max_manhattan * 4;
        // Discovery order matters: this must match the TS oracle's `Map` (which
        // iterates in insertion order) so that edges are created/updated in the
        // same order as TS. A `HashMap` here would iterate in a randomized,
        // per-process order and make edge tie-breaking non-deterministic.
        let mut reachable: Vec<(usize, u32)> = Vec::new();
        let mut found_count = 0usize;
        let target_len = target_nodes.len();
        let width = self.map.width;
        let height = self.map.height;
        let map = self.map;

        let stamp = self.tile_bfs.next_stamp();
        let mut head = 0usize;
        let mut tail = 0usize;
        self.tile_bfs.visited_stamp[from as usize] = stamp;
        self.tile_bfs.dist[from as usize] = 0;
        self.tile_bfs.queue[tail] = from as i32;
        tail += 1;
        let last_row_start = (height - 1) * width;

        while head < tail {
            let node = self.tile_bfs.queue[head] as TileRef;
            head += 1;
            let dist = self.tile_bfs.dist[node as usize] as u32;
            let next_dist = dist + 1;
            if next_dist > max_distance {
                continue;
            }

            let x = node % width;
            let y = node / width;
            let is_start_or_target = node == from || tile_to_node_id.contains_key(&node);
            if !is_start_or_target && (x < min_x || x > max_x || y < min_y || y > max_y) {
                continue;
            }
            if let Some(&node_id) = tile_to_node_id.get(&node) {
                reachable.push((node_id, dist));
                found_count += 1;
                if found_count == target_len {
                    break;
                }
            }

            let try_push = |nb: TileRef, bfs: &mut BfsGrid, tail: &mut usize| {
                if bfs.visited_stamp[nb as usize] != stamp && map.is_water(nb) {
                    bfs.visited_stamp[nb as usize] = stamp;
                    bfs.dist[nb as usize] = next_dist as u16;
                    bfs.queue[*tail] = nb as i32;
                    *tail += 1;
                }
            };
            if node >= width {
                try_push(node - width, &mut self.tile_bfs, &mut tail);
            }
            if node < last_row_start {
                try_push(node + width, &mut self.tile_bfs, &mut tail);
            }
            if x != 0 {
                try_push(node - 1, &mut self.tile_bfs, &mut tail);
            }
            if x != width - 1 {
                try_push(node + 1, &mut self.tile_bfs, &mut tail);
            }
        }
        reachable
    }
}

// ── Bounded water A* (TS `AStar.WaterBounded`) ──────────────────────────────

const BOUNDED_LAND_BIT: u8 = 7;
const BOUNDED_COST_SCALE: u32 = 100;
const BOUNDED_BASE_COST: u32 = 100;

fn bounded_magnitude_penalty(magnitude: u8) -> u32 {
    if magnitude < 3 {
        3 * BOUNDED_BASE_COST
    } else if magnitude <= 10 {
        0
    } else {
        BOUNDED_BASE_COST
    }
}

struct BoundedWaterAstar {
    closed_stamp: Vec<u32>,
    g_stamp: Vec<u32>,
    g: Vec<u32>,
    came_from: Vec<i32>,
    heap: AstarHeap,
    stamp: u32,
    max_area: usize,
    heuristic_weight: u32,
    max_iterations: u32,
}

impl BoundedWaterAstar {
    fn new(max_search_area: usize) -> Self {
        Self {
            closed_stamp: vec![0; max_search_area],
            g_stamp: vec![0; max_search_area],
            g: vec![0; max_search_area],
            came_from: vec![-1; max_search_area],
            heap: AstarHeap::new(max_search_area * 4),
            stamp: 0,
            max_area: max_search_area,
            heuristic_weight: 3,
            max_iterations: 100_000,
        }
    }

    fn search_bounded(
        &mut self,
        map: &GameMap,
        starts: &[TileRef],
        goal: TileRef,
        min_x: u32,
        max_x: u32,
        min_y: u32,
        max_y: u32,
    ) -> Option<Vec<TileRef>> {
        if std::env::var("DEBUG_LOCAL_BOUNDED").is_ok() {
            eprintln!(
                "[native local] starts={:?} goal={} (x={},y={}) bounds=[{},{}]x[{},{}]",
                starts
                    .iter()
                    .map(|&s| (s, map.x(s), map.y(s)))
                    .collect::<Vec<_>>(),
                goal,
                map.x(goal),
                map.y(goal),
                min_x,
                max_x,
                min_y,
                max_y
            );
        }
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.closed_stamp.fill(0);
            self.g_stamp.fill(0);
            self.stamp = 1;
        }
        let stamp = self.stamp;
        let width = map.width;
        let bounds_w = max_x - min_x + 1;
        let bounds_h = max_y - min_y + 1;
        let num_local = (bounds_w * bounds_h) as usize;
        if num_local == 0 || num_local > self.max_area {
            return None;
        }

        let land_mask = 1u8 << BOUNDED_LAND_BIT;
        let goal_x = map.x(goal);
        let goal_y = map.y(goal);

        let to_local = |tile: TileRef, clamp: bool| -> Option<usize> {
            let mut x = map.x(tile);
            let mut y = map.y(tile);
            if clamp {
                x = x.clamp(min_x, max_x);
                y = y.clamp(min_y, max_y);
            }
            if x < min_x || x > max_x || y < min_y || y > max_y {
                return None;
            }
            Some(((y - min_y) * bounds_w + (x - min_x)) as usize)
        };
        let to_global = |local: usize| -> TileRef {
            let lx = (local as u32) % bounds_w;
            let ly = (local as u32) / bounds_w;
            map.ref_xy(lx + min_x, ly + min_y)
        };

        // TS `AStar.WaterBounded.searchBounded`: `toLocal(goal, true)` clamps the
        // goal into bounds. Without clamping, a goal one tile outside (e.g. x =
        // min_x - 1) underflows u32 subtraction and wraps `goal_local` to the
        // opposite edge of the cluster, producing a teleport when the real
        // destination is later appended. That desynced transport-ship paths
        // (HxWdr5PK tick 190 and many other early TINY failures).
        let goal_local = to_local(goal, true)?;
        if goal_local >= num_local {
            return None;
        }

        let s0 = starts[0];
        let start_x = map.x(s0);
        let start_y = map.y(s0);
        let dx_goal = goal_x as i32 - start_x as i32;
        let dy_goal = goal_y as i32 - start_y as i32;
        let cross_norm = (dx_goal.abs() + dy_goal.abs()).max(1) as u32;
        let cross_tie = |nx: u32, ny: u32| -> u32 {
            let dx_n = nx as i32 - goal_x as i32;
            let dy_n = ny as i32 - goal_y as i32;
            let cross = (dx_goal * dy_n - dy_goal * dx_n).unsigned_abs();
            ((cross * (BOUNDED_COST_SCALE - 1)) / cross_norm / cross_norm) as u32
        };

        self.heap.clear();
        for &s in starts {
            let Some(start_local) = to_local(s, true) else {
                continue;
            };
            self.g[start_local] = 0;
            self.g_stamp[start_local] = stamp;
            self.came_from[start_local] = -1;
            let h = self.heuristic_weight
                * BOUNDED_BASE_COST
                * (map.x(s).abs_diff(goal_x) + map.y(s).abs_diff(goal_y));
            self.heap.push(start_local as TileRef, h);
        }

        let mut iterations = self.max_iterations;
        while !self.heap.is_empty() {
            iterations -= 1;
            if iterations == 0 {
                return None;
            }
            let current_local = self.heap.pop() as usize;
            if self.closed_stamp[current_local] == stamp {
                continue;
            }
            self.closed_stamp[current_local] = stamp;
            if current_local == goal_local {
                let mut path = Vec::new();
                let mut p = goal_local as i32;
                while p >= 0 {
                    path.push(to_global(p as usize));
                    if self.came_from[p as usize] < 0 {
                        break;
                    }
                    p = self.came_from[p as usize];
                }
                path.reverse();
                return Some(path);
            }
            let current = to_global(current_local);
            let current_g = self.g[current_local];
            let cx = map.x(current);
            let cy = map.y(current);

            let relax = |this: &mut Self, nbr: TileRef, nlx: u32, nly: u32, n_local: usize| {
                if this.closed_stamp[n_local] == stamp {
                    return;
                }
                let terrain = map.terrain_byte(nbr);
                if nbr != goal && terrain & land_mask != 0 {
                    return;
                }
                let magnitude = terrain & 0x1f;
                let cost = BOUNDED_BASE_COST + bounded_magnitude_penalty(magnitude);
                let tentative = current_g.saturating_add(cost);
                if this.g_stamp[n_local] != stamp || tentative < this.g[n_local] {
                    this.came_from[n_local] = current_local as i32;
                    this.g[n_local] = tentative;
                    this.g_stamp[n_local] = stamp;
                    let h = this.heuristic_weight
                        * BOUNDED_BASE_COST
                        * (nlx.abs_diff(goal_x) + nly.abs_diff(goal_y));
                    let f = tentative + h + cross_tie(nlx, nly);
                    this.heap.push(n_local as TileRef, f);
                }
            };

            if cy > min_y {
                let nbr = current - width;
                if let Some(n_local) = to_local(nbr, false) {
                    relax(self, nbr, cx, cy - 1, n_local);
                }
            }
            if cy < max_y {
                let nbr = current + width;
                if let Some(n_local) = to_local(nbr, false) {
                    relax(self, nbr, cx, cy + 1, n_local);
                }
            }
            if cx > min_x {
                let nbr = current - 1;
                if let Some(n_local) = to_local(nbr, false) {
                    relax(self, nbr, cx - 1, cy, n_local);
                }
            }
            if cx < max_x {
                let nbr = current + 1;
                if let Some(n_local) = to_local(nbr, false) {
                    relax(self, nbr, cx + 1, cy, n_local);
                }
            }
        }
        None
    }
}

// ── Abstract graph A* (TS `AStar.AbstractGraph`) ──────────────────────────────

struct AbstractGraphAstar {
    closed_stamp: Vec<u32>,
    g_stamp: Vec<u32>,
    g: Vec<u32>,
    came_from: Vec<i32>,
    start_node: Vec<usize>,
    heap: AstarHeap,
    stamp: u32,
    heuristic_weight: u32,
    max_iterations: u32,
}

impl AbstractGraphAstar {
    fn new(num_nodes: usize, num_edges: usize) -> Self {
        Self {
            closed_stamp: vec![0; num_nodes],
            g_stamp: vec![0; num_nodes],
            g: vec![0; num_nodes],
            came_from: vec![-1; num_nodes],
            start_node: vec![0; num_nodes],
            heap: AstarHeap::new(num_nodes + num_edges * 2),
            stamp: 0,
            heuristic_weight: 1,
            max_iterations: 100_000,
        }
    }

    fn find_path(
        &mut self,
        graph: &AbstractGraph,
        start: usize,
        goal: usize,
    ) -> Option<Vec<usize>> {
        self.find_path_multi(graph, &[start], goal)
    }

    fn find_path_multi(
        &mut self,
        graph: &AbstractGraph,
        starts: &[usize],
        goal: usize,
    ) -> Option<Vec<usize>> {
        if starts.is_empty() {
            return None;
        }
        if starts.len() == 1 {
            return self.find_path_single(graph, starts[0], goal);
        }
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.closed_stamp.fill(0);
            self.g_stamp.fill(0);
            self.stamp = 1;
        }
        let stamp = self.stamp;
        let goal_node = graph.get_node(goal)?;
        let goal_x = goal_node.x;
        let goal_y = goal_node.y;

        self.heap.clear();
        for &start_id in starts {
            let node = graph.get_node(start_id)?;
            self.g[start_id] = 0;
            self.g_stamp[start_id] = stamp;
            self.came_from[start_id] = -1;
            self.start_node[start_id] = start_id;
            let h = self.heuristic_weight * (node.x.abs_diff(goal_x) + node.y.abs_diff(goal_y));
            self.heap.push(start_id as TileRef, h);
        }

        let mut iterations = self.max_iterations;
        while !self.heap.is_empty() {
            iterations -= 1;
            if iterations == 0 {
                return None;
            }
            let current = self.heap.pop() as usize;
            if self.closed_stamp[current] == stamp {
                continue;
            }
            self.closed_stamp[current] = stamp;
            if current == goal {
                return self.build_path(goal);
            }
            let current_g = self.g[current];
            for &edge_id in graph.get_node_edges(current) {
                let edge = graph.get_edge(edge_id)?;
                let neighbor = graph.other_node(edge, current);
                if self.closed_stamp[neighbor] == stamp {
                    continue;
                }
                let tentative = current_g.saturating_add(edge.cost);
                if self.g_stamp[neighbor] != stamp || tentative < self.g[neighbor] {
                    self.came_from[neighbor] = current as i32;
                    self.g[neighbor] = tentative;
                    self.g_stamp[neighbor] = stamp;
                    if let Some(neighbor_node) = graph.get_node(neighbor) {
                        let h = self.heuristic_weight
                            * (neighbor_node.x.abs_diff(goal_x) + neighbor_node.y.abs_diff(goal_y));
                        self.heap.push(neighbor as TileRef, tentative + h);
                    }
                }
            }
        }
        None
    }

    fn find_path_single(
        &mut self,
        graph: &AbstractGraph,
        start: usize,
        goal: usize,
    ) -> Option<Vec<usize>> {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.closed_stamp.fill(0);
            self.g_stamp.fill(0);
            self.stamp = 1;
        }
        let stamp = self.stamp;
        let goal_node = graph.get_node(goal)?;
        let goal_x = goal_node.x;
        let goal_y = goal_node.y;
        let start_node = graph.get_node(start)?;

        self.heap.clear();
        self.g[start] = 0;
        self.g_stamp[start] = stamp;
        self.came_from[start] = -1;
        let h =
            self.heuristic_weight * (start_node.x.abs_diff(goal_x) + start_node.y.abs_diff(goal_y));
        self.heap.push(start as TileRef, h);

        let mut iterations = self.max_iterations;
        while !self.heap.is_empty() {
            iterations -= 1;
            if iterations == 0 {
                return None;
            }
            let current = self.heap.pop() as usize;
            if self.closed_stamp[current] == stamp {
                continue;
            }
            self.closed_stamp[current] = stamp;
            if current == goal {
                return self.build_path(goal);
            }
            let current_g = self.g[current];
            for &edge_id in graph.get_node_edges(current) {
                let edge = graph.get_edge(edge_id)?;
                let neighbor = graph.other_node(edge, current);
                if self.closed_stamp[neighbor] == stamp {
                    continue;
                }
                let tentative = current_g.saturating_add(edge.cost);
                if self.g_stamp[neighbor] != stamp || tentative < self.g[neighbor] {
                    self.came_from[neighbor] = current as i32;
                    self.g[neighbor] = tentative;
                    self.g_stamp[neighbor] = stamp;
                    if let Some(neighbor_node) = graph.get_node(neighbor) {
                        let h = self.heuristic_weight
                            * (neighbor_node.x.abs_diff(goal_x) + neighbor_node.y.abs_diff(goal_y));
                        self.heap.push(neighbor as TileRef, tentative + h);
                    }
                }
            }
        }
        None
    }

    fn build_path(&self, goal: usize) -> Option<Vec<usize>> {
        let mut path = Vec::new();
        let mut current = goal as i32;
        let max_len = self.came_from.len();
        while current >= 0 {
            if current as usize >= max_len {
                return None;
            }
            path.push(current as usize);
            if path.len() > max_len {
                return None;
            }
            current = self.came_from[current as usize];
        }
        path.reverse();
        Some(path)
    }
}

#[cfg(test)]
mod abstract_astar_tests {
    use super::{
        AbstractEdge, AbstractGraph, AbstractGraphAstar, AbstractNode, Cluster,
        ConnectedComponents, CLUSTER_SIZE,
    };

    fn graph_with_equal_f_neighbors() -> AbstractGraph {
        let nodes = vec![
            AbstractNode {
                id: 0,
                x: 900,
                y: 0,
                tile: 900,
                component_id: 1,
            },
            AbstractNode {
                id: 1,
                x: 250,
                y: 0,
                tile: 793_328,
                component_id: 1,
            },
            AbstractNode {
                id: 2,
                x: 300,
                y: 0,
                tile: 806_943,
                component_id: 1,
            },
            AbstractNode {
                id: 3,
                x: 0,
                y: 0,
                tile: 0,
                component_id: 1,
            },
        ];
        let mut graph = AbstractGraph {
            cluster_size: CLUSTER_SIZE,
            clusters_x: 1,
            clusters_y: 1,
            nodes,
            edges: Vec::new(),
            node_edge_ids: vec![Vec::new(); 4],
            clusters: vec![Cluster {
                x: 0,
                y: 0,
                node_ids: vec![0, 1, 2, 3],
            }],
            path_cache: Vec::new(),
            water_components: ConnectedComponents {
                component_ids: Vec::new(),
                component_sizes: vec![0, 4],
            },
        };

        for (id, node_a, node_b, cost) in [
            (0, 0, 1, 251), // 251 + h(250) = f 501
            (1, 0, 2, 201), // 201 + h(300) = f 501
            (2, 1, 3, 250),
            (3, 2, 3, 300),
        ] {
            graph.add_edge(AbstractEdge {
                id,
                node_a,
                node_b,
                cost,
                cluster_x: 0,
                cluster_y: 0,
            });
        }
        graph
    }

    #[test]
    fn equal_f_neighbors_follow_ts_edge_insertion_order() {
        let graph = graph_with_equal_f_neighbors();

        let start_neighbors: Vec<usize> = graph
            .get_node_edges(0)
            .iter()
            .map(|&edge_id| graph.other_node(graph.get_edge(edge_id).unwrap(), 0))
            .collect();
        assert_eq!(start_neighbors, vec![1, 2]);

        let mut astar = AbstractGraphAstar::new(graph.node_count(), graph.edge_count());
        let path = astar.find_path(&graph, 0, 3).unwrap();

        assert_eq!(path, vec![0, 1, 3]);
        assert_eq!(graph.get_node(path[1]).unwrap().tile, 793_328);
    }
}

// ── Hierarchical water pathfinder (TS `AStar.WaterHierarchical`) ──────────────

pub struct WaterHierarchical {
    pub graph: AbstractGraph,
    tile_bfs: BfsGrid,
    abstract_astar: AbstractGraphAstar,
    local_astar: BoundedWaterAstar,
    local_astar_multi: BoundedWaterAstar,
    local_astar_short: BoundedWaterAstar,
    cache_paths: bool,
}

impl WaterHierarchical {
    pub fn new(map: &GameMap, cache_paths: bool) -> Self {
        let graph = AbstractGraphBuilder::new(map).build();
        let cluster_size = graph.cluster_size;
        let num_tiles = (map.width * map.height) as usize;
        Self {
            abstract_astar: AbstractGraphAstar::new(graph.node_count(), graph.edge_count()),
            local_astar: BoundedWaterAstar::new((cluster_size * cluster_size) as usize),
            local_astar_multi: BoundedWaterAstar::new(
                (cluster_size * 3 * cluster_size * 3) as usize,
            ),
            local_astar_short: BoundedWaterAstar::new(260 * 260),
            tile_bfs: BfsGrid::new(num_tiles),
            graph,
            cache_paths,
        }
    }

    pub fn find_path(
        &mut self,
        map: &GameMap,
        from: &[TileRef],
        to: TileRef,
    ) -> Option<Vec<TileRef>> {
        if from.len() == 1 {
            return self.find_path_single(map, from[0], to);
        }
        self.find_path_multi_source(map, from, to)
    }

    fn find_path_multi_source(
        &mut self,
        map: &GameMap,
        sources: &[TileRef],
        target: TileRef,
    ) -> Option<Vec<TileRef>> {
        if let Some(short) = self.try_short_path_multi_source(map, sources, target) {
            return Some(short);
        }
        let target_node = self.get_cluster_node(map, target)?;
        let mut node_to_source: HashMap<usize, TileRef> = HashMap::new();
        let mut node_to_dist: HashMap<usize, u32> = HashMap::new();
        let mut node_ids: Vec<usize> = Vec::new();
        for &source in sources {
            let node = match self.get_cluster_node(map, source) {
                Some(n) => n,
                None => continue,
            };
            let dist = node.x.abs_diff(map.x(source)) + node.y.abs_diff(map.y(source));
            let prev = node_to_dist.get(&node.id).copied();
            if prev.is_none() || dist < prev.unwrap() {
                if prev.is_none() {
                    node_ids.push(node.id);
                }
                node_to_source.insert(node.id, source);
                node_to_dist.insert(node.id, dist);
            }
        }
        if node_to_source.is_empty() {
            return None;
        }
        if std::env::var("DEBUG_WATER_MULTI").is_ok() {
            eprintln!(
                "[native] target_node.id={} tile={}",
                target_node.id, target_node.tile
            );
            for &nid in &node_ids {
                let src = node_to_source[&nid];
                let node = self.graph.get_node(nid).unwrap();
                eprintln!(
                    "[native] node_id={} tile={} x={} y={} <- source={} (x={},y={})",
                    nid,
                    node.tile,
                    node.x,
                    node.y,
                    src,
                    map.x(src),
                    map.y(src)
                );
            }
        }
        let node_path =
            self.abstract_astar
                .find_path_multi(&self.graph, &node_ids, target_node.id)?;
        let winning_source = node_to_source.get(&node_path[0])?;
        if std::env::var("DEBUG_WATER_MULTI").is_ok() {
            eprintln!(
                "[native] winning node_path[0]={} winning_source={} (x={},y={}) node_path.len()={}",
                node_path[0],
                winning_source,
                map.x(*winning_source),
                map.y(*winning_source),
                node_path.len()
            );
        }
        self.find_path_single(map, *winning_source, target)
    }

    fn try_short_path_multi_source(
        &mut self,
        map: &GameMap,
        sources: &[TileRef],
        target: TileRef,
    ) -> Option<Vec<TileRef>> {
        const SHORT_PATH_THRESHOLD: u32 = 120;
        const PADDING: u32 = 10;
        let candidates: Vec<TileRef> = sources
            .iter()
            .copied()
            .filter(|&s| map.manhattan_dist(s, target) <= SHORT_PATH_THRESHOLD)
            .collect();
        if candidates.is_empty() {
            return None;
        }
        let to_x = map.x(target);
        let to_y = map.y(target);
        let mut min_x = to_x;
        let mut max_x = to_x;
        let mut min_y = to_y;
        let mut max_y = to_y;
        for s in &candidates {
            let sx = map.x(*s);
            let sy = map.y(*s);
            min_x = min_x.min(sx);
            max_x = max_x.max(sx);
            min_y = min_y.min(sy);
            max_y = max_y.max(sy);
        }
        self.local_astar_short.search_bounded(
            map,
            &candidates,
            target,
            min_x.saturating_sub(PADDING).max(0),
            (max_x + PADDING).min(map.width - 1),
            min_y.saturating_sub(PADDING).max(0),
            (max_y + PADDING).min(map.height - 1),
        )
    }

    fn find_path_single(
        &mut self,
        map: &GameMap,
        from: TileRef,
        to: TileRef,
    ) -> Option<Vec<TileRef>> {
        let dist = map.manhattan_dist(from, to);
        if dist <= self.graph.cluster_size {
            let start_x = map.x(from);
            let start_y = map.y(from);
            let cluster_x = start_x / self.graph.cluster_size;
            let cluster_y = start_y / self.graph.cluster_size;
            if let Some(path) = self.find_local_path(map, from, to, cluster_x, cluster_y, true) {
                return Some(path);
            }
        }

        let start_node = self.find_nearest_node(map, from)?;
        let end_node = self.find_nearest_node(map, to)?;

        if start_node.id == end_node.id {
            let cluster_x = start_node.x / self.graph.cluster_size;
            let cluster_y = start_node.y / self.graph.cluster_size;
            return self.find_local_path(map, from, to, cluster_x, cluster_y, true);
        }

        let node_path = self
            .abstract_astar
            .find_path(&self.graph, start_node.id, end_node.id)?;

        let mut initial_path = Vec::new();
        let first_node = self.graph.get_node(node_path[0])?;
        let start_cluster_x = map.x(from) / self.graph.cluster_size;
        let start_cluster_y = map.y(from) / self.graph.cluster_size;
        let start_segment = self.find_local_path(
            map,
            from,
            first_node.tile,
            start_cluster_x,
            start_cluster_y,
            false,
        )?;
        initial_path.extend_from_slice(&start_segment);

        for i in 0..node_path.len().saturating_sub(1) {
            let from_node_id = node_path[i];
            let to_node_id = node_path[i + 1];
            let edge_id = self.graph.get_edge_between(from_node_id, to_node_id)?.id;
            let (from_tile, to_tile, cluster_x, cluster_y) = {
                let from_node = self.graph.get_node(from_node_id)?;
                let to_node = self.graph.get_node(to_node_id)?;
                let edge = self.graph.get_edge(edge_id)?;
                (from_node.tile, to_node.tile, edge.cluster_x, edge.cluster_y)
            };

            if self.cache_paths {
                if let Some(cached) = self.graph.get_cached_path(edge_id, from_node_id) {
                    if !cached.is_empty() {
                        initial_path.extend_from_slice(&cached[1..]);
                        continue;
                    }
                }
            }

            let segment =
                self.find_local_path(map, from_tile, to_tile, cluster_x, cluster_y, false)?;
            if self.cache_paths {
                self.graph
                    .set_cached_path(edge_id, from_node_id, segment.clone());
            }
            initial_path.extend_from_slice(&segment[1..]);
        }

        let last_node = self.graph.get_node(*node_path.last()?)?;
        let end_cluster_x = map.x(to) / self.graph.cluster_size;
        let end_cluster_y = map.y(to) / self.graph.cluster_size;
        let end_segment =
            self.find_local_path(map, last_node.tile, to, end_cluster_x, end_cluster_y, false)?;
        initial_path.extend_from_slice(&end_segment[1..]);
        Some(initial_path)
    }

    fn find_nearest_node(&mut self, map: &GameMap, tile: TileRef) -> Option<AbstractNode> {
        let x = map.x(tile);
        let y = map.y(tile);
        let cluster_x = x / self.graph.cluster_size;
        let cluster_y = y / self.graph.cluster_size;
        let cluster_size = self.graph.cluster_size;
        let min_x = cluster_x * cluster_size;
        let min_y = cluster_y * cluster_size;
        let max_x = (min_x + cluster_size - 1).min(map.width - 1);
        let max_y = (min_y + cluster_size - 1).min(map.height - 1);
        let cluster = self.graph.get_cluster(cluster_x, cluster_y)?;
        if cluster.node_ids.is_empty() {
            return None;
        }
        let candidate_nodes: Vec<AbstractNode> = cluster
            .node_ids
            .iter()
            .filter_map(|&id| self.graph.get_node(id).cloned())
            .collect();
        let max_distance = cluster_size * cluster_size;
        let graph = &self.graph;
        self.tile_bfs.search(
            map.width,
            map.height,
            tile,
            max_distance,
            |t| graph.get_component_id(t) != LAND_MARKER,
            |t, _dist| {
                let tile_x = map.x(t);
                let tile_y = map.y(t);
                for node in &candidate_nodes {
                    if node.x == tile_x && node.y == tile_y {
                        return BfsVisit::Found(node.clone());
                    }
                }
                if tile_x < min_x || tile_x > max_x || tile_y < min_y || tile_y > max_y {
                    return BfsVisit::Reject;
                }
                BfsVisit::Continue
            },
        )
    }

    fn get_cluster_node(&self, map: &GameMap, tile: TileRef) -> Option<AbstractNode> {
        let x = map.x(tile);
        let y = map.y(tile);
        let cluster_x = x / self.graph.cluster_size;
        let cluster_y = y / self.graph.cluster_size;
        let cluster = self.graph.get_cluster(cluster_x, cluster_y)?;
        if cluster.node_ids.is_empty() {
            return None;
        }
        let mut best: Option<AbstractNode> = None;
        let mut best_dist = u32::MAX;
        for &node_id in &cluster.node_ids {
            let node = self.graph.get_node(node_id)?;
            let dist = node.x.abs_diff(x) + node.y.abs_diff(y);
            if dist < best_dist {
                best_dist = dist;
                best = Some(node.clone());
            }
        }
        best
    }

    fn find_local_path(
        &mut self,
        map: &GameMap,
        from: TileRef,
        to: TileRef,
        cluster_x: u32,
        cluster_y: u32,
        multi_cluster: bool,
    ) -> Option<Vec<TileRef>> {
        let cluster_size = self.graph.cluster_size;
        let (min_x, min_y, max_x, max_y) = if multi_cluster {
            (
                ((cluster_x as i32 - 1) * cluster_size as i32).max(0) as u32,
                ((cluster_y as i32 - 1) * cluster_size as i32).max(0) as u32,
                ((cluster_x + 2) * cluster_size - 1).min(map.width - 1),
                ((cluster_y + 2) * cluster_size - 1).min(map.height - 1),
            )
        } else {
            (
                cluster_x * cluster_size,
                cluster_y * cluster_size,
                (cluster_x * cluster_size + cluster_size - 1).min(map.width - 1),
                (cluster_y * cluster_size + cluster_size - 1).min(map.height - 1),
            )
        };
        let astar = if multi_cluster {
            &mut self.local_astar_multi
        } else {
            &mut self.local_astar
        };
        let mut path = astar.search_bounded(map, &[from], to, min_x, max_x, min_y, max_y)?;
        if path.first() != Some(&from) {
            path.insert(0, from);
        }
        if path.last() != Some(&to) {
            path.push(to);
        }
        Some(path)
    }
}
