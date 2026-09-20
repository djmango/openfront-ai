//! Port of the engine's water path chain for ships, computed from DEVICE state.
//!
//! Reference: `openfront-ai/rust/engine/src/water_hpa.rs` (the half-resolution
//! abstract graph + hierarchical search: `AbstractGraphBuilder`, `BfsGrid`,
//! `BoundedWaterAstar`, `AbstractGraphAstar`, `WaterHierarchical`) and
//! `openfront-ai/rust/engine/src/water.rs` (`transport_path_multi_into`,
//! `minimap_inner_path`, `smooth_water_path`, `upscale_cells`,
//! `fix_path_extremes`, `closest_full_source`).
//!
//! The engine plans a ship's route on the HALF-RESOLUTION map (`map4x.bin`,
//! half the linear size - `core/terrain.rs:114`, `manifest.map4x`) via
//! `Game::plan_water_path` -> `water::transport_path_into` ->
//! `transport_path_multi_into`, then upscales the mini cells by 2
//! (`upscale_cells`) and fixes the extremes (`fix_path_extremes`). The ported
//! substitute was an 8-connected Chebyshev BFS on the FULL-resolution water,
//! which lands the ship after `path.len()` ticks but is only equivalent when
//! the engine's hierarchical route happens to be the full-res shortest route.
//!
//! Everything here reads ONLY device state: the device's own full-resolution
//! terrain (`map.bin`) and the half-resolution plane (`map4x.bin`), plus the
//! device's border array and owner plane. No oracle record is consulted.
//!
//! The module is compiled into the matrix driver (`mod water_hpa_port;`) and
//! every function below is a literal transcription of the engine source. Where
//! the engine allocates per query (`bounded_water_path`) the port allocates the
//! same arrays; the graph, the stamp grids and the A* workspaces live in
//! `WaterHierarchical` exactly as they do engine-side.

#![allow(dead_code)]

use std::collections::HashMap;

pub const LAND_MARKER: u32 = 0xffff;
pub const CLUSTER_SIZE: u32 = 32;

/// A read-only terrain plane + dimensions. The engine's `GameMap` reduced to
/// the surface the water chain touches (`map.rs:101-146`, `:331-435`).
#[derive(Clone, Copy)]
pub struct WMap<'a> {
    pub t: &'a [u8],
    pub w: u32,
    pub h: u32,
}

impl<'a> WMap<'a> {
    pub fn new(t: &'a [u8], w: u32, h: u32) -> Self {
        Self { t, w, h }
    }
    #[inline]
    pub fn n(&self) -> usize {
        (self.w * self.h) as usize
    }
    #[inline]
    pub fn ref_xy(&self, x: u32, y: u32) -> u32 {
        y * self.w + x
    }
    #[inline]
    pub fn x(&self, t: u32) -> u32 {
        t % self.w
    }
    #[inline]
    pub fn y(&self, t: u32) -> u32 {
        t / self.w
    }
    #[inline]
    pub fn byte(&self, t: u32) -> u8 {
        self.t[t as usize]
    }
    #[inline]
    pub fn is_land(&self, t: u32) -> bool {
        self.byte(t) & 0x80 != 0
    }
    #[inline]
    pub fn is_water(&self, t: u32) -> bool {
        !self.is_land(t)
    }
    #[inline]
    pub fn is_shore(&self, t: u32) -> bool {
        let b = self.byte(t);
        b & 0x80 != 0 && b & 0x40 != 0
    }
    #[inline]
    pub fn manhattan(&self, a: u32, b: u32) -> u32 {
        self.x(a).abs_diff(self.x(b)) + self.y(a).abs_diff(self.y(b))
    }
    /// `map.rs:370` `neighbors_nswe` - N, S, W, E.
    #[inline]
    pub fn nswe(&self, t: u32, out: &mut [u32; 4]) -> usize {
        let x = t % self.w;
        let mut n = 0usize;
        if t >= self.w {
            out[n] = t - self.w;
            n += 1;
        }
        if t < (self.h - 1) * self.w {
            out[n] = t + self.w;
            n += 1;
        }
        if x != 0 {
            out[n] = t - 1;
            n += 1;
        }
        if x != self.w - 1 {
            out[n] = t + 1;
            n += 1;
        }
        n
    }
}

// ── Connected components (water_hpa.rs:12-104) ──────────────────────────────

struct ConnectedComponents {
    component_ids: Vec<u32>,
    component_sizes: Vec<u32>,
}

impl ConnectedComponents {
    fn new(map: &WMap) -> Self {
        let n = map.n();
        let mut ids = vec![0u32; n];
        for t in 0..n {
            if !map.is_water(t as u32) {
                ids[t] = LAND_MARKER;
            }
        }
        Self {
            component_ids: ids,
            component_sizes: vec![0],
        }
    }

    fn initialize(&mut self, map: &WMap) {
        let width = map.w;
        let height = map.h;
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

    fn get_component_id(&self, tile: u32) -> u32 {
        self.component_ids.get(tile as usize).copied().unwrap_or(0)
    }

    fn get_component_size(&self, component_id: u32) -> u32 {
        self.component_sizes
            .get(component_id as usize)
            .copied()
            .unwrap_or(0)
    }
}

// ── Stamp BFS grid (water_hpa.rs:108-199) ───────────────────────────────────

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

    fn search<R>(
        &mut self,
        width: u32,
        height: u32,
        start: u32,
        max_distance: u32,
        is_valid: impl Fn(u32) -> bool,
        mut visitor: impl FnMut(u32, u32) -> BfsVisit<R>,
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
            let node = self.queue[head] as u32;
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
            let mut push = |bfs: &mut Self, nb: u32, tail: &mut usize, next_dist: u32| {
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

// ── Abstract graph (water_hpa.rs:201-358) ───────────────────────────────────

#[derive(Clone, Debug)]
pub struct AbstractNode {
    pub id: usize,
    pub x: u32,
    pub y: u32,
    pub tile: u32,
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
    #[allow(dead_code)]
    path_cache: Vec<Option<Vec<u32>>>,
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
    pub fn get_component_id(&self, tile: u32) -> u32 {
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

    fn init_path_cache(&mut self) {
        self.path_cache = vec![None; self.edges.len() * 2];
    }
}

// ── Abstract graph builder (water_hpa.rs:360-725) ───────────────────────────

pub struct AbstractGraphBuilder<'a> {
    map: &'a WMap<'a>,
    cluster_size: u32,
    graph: AbstractGraph,
    tile_to_node: HashMap<u32, usize>,
    next_node_id: usize,
    next_edge_id: usize,
    edge_between: HashMap<usize, HashMap<usize, usize>>,
    tile_bfs: BfsGrid,
}

impl<'a> AbstractGraphBuilder<'a> {
    pub fn new(map: &'a WMap<'a>) -> Self {
        let width = map.w;
        let height = map.h;
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
            let edge_x = (base_x + self.cluster_size - 1).min(self.map.w - 1);
            let nodes = self.find_nodes_on_vertical_edge(edge_x, base_y);
            for node_id in nodes {
                let key_a = self.graph.cluster_key(cx, cy);
                let key_b = self.graph.cluster_key(cx + 1, cy);
                self.graph.add_node_to_cluster(key_a, node_id);
                self.graph.add_node_to_cluster(key_b, node_id);
            }
        }
        if cy < self.graph.clusters_y - 1 {
            let edge_y = (base_y + self.cluster_size - 1).min(self.map.h - 1);
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
        let max_y = (base_y + self.cluster_size).min(self.map.h);
        let mut span_start: Option<u32> = None;
        for y in base_y..max_y {
            let tile = self.map.ref_xy(x, y);
            let next_tile = if x + 1 < self.map.w {
                self.map.ref_xy(x + 1, y)
            } else {
                u32::MAX
            };
            let is_entrance =
                self.map.is_water(tile) && next_tile != u32::MAX && self.map.is_water(next_tile);
            if is_entrance {
                if span_start.is_none() {
                    span_start = Some(y);
                }
            } else if let Some(start) = span_start.take() {
                let span_length = y - start;
                let mid_y = start + span_length / 2;
                let node_id = self.get_or_create_node(x, mid_y);
                nodes.push(node_id);
            }
        }
        if let Some(start) = span_start.take() {
            let span_length = max_y - start;
            let mid_y = start + span_length / 2;
            let node_id = self.get_or_create_node(x, mid_y);
            nodes.push(node_id);
        }
        nodes
    }

    fn find_nodes_on_horizontal_edge(&mut self, y: u32, base_x: u32) -> Vec<usize> {
        let mut nodes = Vec::new();
        let max_x = (base_x + self.cluster_size).min(self.map.w);
        let mut span_start: Option<u32> = None;
        for x in base_x..max_x {
            let tile = self.map.ref_xy(x, y);
            let next_tile = if y + 1 < self.map.h {
                self.map.ref_xy(x, y + 1)
            } else {
                u32::MAX
            };
            let is_entrance =
                self.map.is_water(tile) && next_tile != u32::MAX && self.map.is_water(next_tile);
            if is_entrance {
                if span_start.is_none() {
                    span_start = Some(x);
                }
            } else if let Some(start) = span_start.take() {
                let span_length = x - start;
                let mid_x = start + span_length / 2;
                let node_id = self.get_or_create_node(mid_x, y);
                nodes.push(node_id);
            }
        }
        if let Some(start) = span_start.take() {
            let span_length = max_x - start;
            let mid_x = start + span_length / 2;
            let node_id = self.get_or_create_node(mid_x, y);
            nodes.push(node_id);
        }
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
        let cluster_max_x = (cluster_min_x + self.cluster_size - 1).min(self.map.w - 1);
        let cluster_max_y = (cluster_min_y + self.cluster_size - 1).min(self.map.h - 1);

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
        from: u32,
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
        let mut reachable: Vec<(usize, u32)> = Vec::new();
        let mut found_count = 0usize;
        let target_len = target_nodes.len();
        let width = self.map.w;
        let height = self.map.h;
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
            let node = self.tile_bfs.queue[head] as u32;
            head += 1;
            let dist = self.tile_bfs.dist[node as usize] as u32;

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

            let next_dist = dist + 1;
            if next_dist > max_distance {
                continue;
            }

            let mut try_push = |nb: u32, bfs: &mut BfsGrid, tail: &mut usize| {
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

// ── Bounded water A* (water_hpa.rs:727-953) ─────────────────────────────────

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

/// `water.rs:294-366` `AstarHeap` - a binary heap whose priorities are f32
/// (TS `MinHeap` stores them in a Float32Array).
#[derive(Default)]
struct AstarHeap {
    f: Vec<f32>,
    tile: Vec<u32>,
}

impl AstarHeap {
    fn new(cap: usize) -> Self {
        Self {
            f: Vec::with_capacity(cap),
            tile: Vec::with_capacity(cap),
        }
    }
    fn clear(&mut self) {
        self.f.clear();
        self.tile.clear();
    }
    fn is_empty(&self) -> bool {
        self.f.is_empty()
    }
    fn push(&mut self, tile: u32, priority: u32) {
        let priority = priority as f32;
        let mut i = self.f.len();
        self.f.push(priority);
        self.tile.push(tile);
        while i > 0 {
            let parent = (i - 1) >> 1;
            if self.f[parent] <= self.f[i] {
                break;
            }
            self.f.swap(parent, i);
            self.tile.swap(parent, i);
            i = parent;
        }
    }
    fn pop(&mut self) -> u32 {
        let top = self.tile[0];
        let last_f = self.f.pop().unwrap();
        let last_tile = self.tile.pop().unwrap();
        if self.f.is_empty() {
            return top;
        }
        self.f[0] = last_f;
        self.tile[0] = last_tile;
        let mut i = 0usize;
        loop {
            let left = (i << 1) + 1;
            if left >= self.f.len() {
                break;
            }
            let right = left + 1;
            let mut smallest = i;
            if self.f[left] < self.f[smallest] {
                smallest = left;
            }
            if right < self.f.len() && self.f[right] < self.f[smallest] {
                smallest = right;
            }
            if smallest == i {
                break;
            }
            self.f.swap(smallest, i);
            self.tile.swap(smallest, i);
            i = smallest;
        }
        top
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
        map: &WMap,
        starts: &[u32],
        goal: u32,
        min_x: u32,
        max_x: u32,
        min_y: u32,
        max_y: u32,
    ) -> Option<Vec<u32>> {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.closed_stamp.fill(0);
            self.g_stamp.fill(0);
            self.stamp = 1;
        }
        let stamp = self.stamp;
        let width = map.w;
        let bounds_w = max_x - min_x + 1;
        let bounds_h = max_y - min_y + 1;
        let num_local = (bounds_w * bounds_h) as usize;
        if num_local == 0 || num_local > self.max_area {
            return None;
        }

        let land_mask = 1u8 << BOUNDED_LAND_BIT;
        let goal_x = map.x(goal);
        let goal_y = map.y(goal);

        let to_local = |tile: u32, clamp: bool| -> Option<usize> {
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
        let to_global = |local: usize| -> u32 {
            let lx = (local as u32) % bounds_w;
            let ly = (local as u32) / bounds_w;
            map.ref_xy(lx + min_x, ly + min_y)
        };

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
            ((cross as f64 * (BOUNDED_COST_SCALE - 1) as f64)
                / cross_norm as f64
                / cross_norm as f64)
                .floor() as u32
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
            self.heap.push(start_local as u32, h);
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

            let relax = |this: &mut Self, nbr: u32, nlx: u32, nly: u32, n_local: usize| {
                if this.closed_stamp[n_local] == stamp {
                    return;
                }
                let terrain = map.byte(nbr);
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
                    this.heap.push(n_local as u32, f);
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

// ── Abstract graph A* (water_hpa.rs:955-1144) ───────────────────────────────

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
            self.heap.push(start_id as u32, h);
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
                        self.heap.push(neighbor as u32, tentative + h);
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
        self.heap.push(start as u32, h);

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
                        self.heap.push(neighbor as u32, tentative + h);
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

// ── Hierarchical water pathfinder (water_hpa.rs:1240-1577) ──────────────────

pub struct WaterHierarchical {
    pub graph: AbstractGraph,
    tile_bfs: BfsGrid,
    abstract_astar: AbstractGraphAstar,
    local_astar: BoundedWaterAstar,
    local_astar_multi: BoundedWaterAstar,
    local_astar_short: BoundedWaterAstar,
    #[allow(dead_code)]
    cache_paths: bool,
}

impl WaterHierarchical {
    pub fn new(map: &WMap, cache_paths: bool) -> Self {
        let graph = AbstractGraphBuilder::new(map).build();
        let cluster_size = graph.cluster_size;
        let num_tiles = (map.w * map.h) as usize;
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

    pub fn find_path(&mut self, map: &WMap, from: &[u32], to: u32) -> Option<Vec<u32>> {
        if from.len() == 1 {
            return self.find_path_single(map, from[0], to);
        }
        self.find_path_multi_source(map, from, to)
    }

    fn find_path_multi_source(
        &mut self,
        map: &WMap,
        sources: &[u32],
        target: u32,
    ) -> Option<Vec<u32>> {
        if let Some(short) = self.try_short_path_multi_source(map, sources, target) {
            return Some(short);
        }
        let target_node = self.get_cluster_node(map, target)?;
        let mut node_to_source: HashMap<usize, u32> = HashMap::new();
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
        let node_path = self
            .abstract_astar
            .find_path_multi(&self.graph, &node_ids, target_node.id)?;
        let winning_source = node_to_source.get(&node_path[0])?;
        self.find_path_single(map, *winning_source, target)
    }

    fn try_short_path_multi_source(
        &mut self,
        map: &WMap,
        sources: &[u32],
        target: u32,
    ) -> Option<Vec<u32>> {
        const SHORT_PATH_THRESHOLD: u32 = 120;
        const PADDING: u32 = 10;
        let candidates: Vec<u32> = sources
            .iter()
            .copied()
            .filter(|&s| map.manhattan(s, target) <= SHORT_PATH_THRESHOLD)
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
            min_x.saturating_sub(PADDING),
            (max_x + PADDING).min(map.w - 1),
            min_y.saturating_sub(PADDING),
            (max_y + PADDING).min(map.h - 1),
        )
    }

    fn find_path_single(&mut self, map: &WMap, from: u32, to: u32) -> Option<Vec<u32>> {
        let dist = map.manhattan(from, to);
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
            let segment =
                self.find_local_path(map, from_tile, to_tile, cluster_x, cluster_y, false)?;
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

    fn find_nearest_node(&mut self, map: &WMap, tile: u32) -> Option<AbstractNode> {
        let x = map.x(tile);
        let y = map.y(tile);
        let cluster_x = x / self.graph.cluster_size;
        let cluster_y = y / self.graph.cluster_size;
        let cluster_size = self.graph.cluster_size;
        let min_x = cluster_x * cluster_size;
        let min_y = cluster_y * cluster_size;
        let max_x = (min_x + cluster_size - 1).min(map.w - 1);
        let max_y = (min_y + cluster_size - 1).min(map.h - 1);
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
            map.w,
            map.h,
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

    fn get_cluster_node(&self, map: &WMap, tile: u32) -> Option<AbstractNode> {
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
        map: &WMap,
        from: u32,
        to: u32,
        cluster_x: u32,
        cluster_y: u32,
        multi_cluster: bool,
    ) -> Option<Vec<u32>> {
        let cluster_size = self.graph.cluster_size;
        let (min_x, min_y, max_x, max_y) = if multi_cluster {
            (
                ((cluster_x as i32 - 1) * cluster_size as i32).max(0) as u32,
                ((cluster_y as i32 - 1) * cluster_size as i32).max(0) as u32,
                ((cluster_x + 2) * cluster_size - 1).min(map.w - 1),
                ((cluster_y + 2) * cluster_size - 1).min(map.h - 1),
            )
        } else {
            (
                cluster_x * cluster_size,
                cluster_y * cluster_size,
                (cluster_x * cluster_size + cluster_size - 1).min(map.w - 1),
                (cluster_y * cluster_size + cluster_size - 1).min(map.h - 1),
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

// ── water.rs smoothing / upscale / extremes ─────────────────────────────────

const LAND_BIT: u8 = 7;
const COST_SCALE: u32 = 100;
const BASE_COST: u32 = 100;

fn magnitude_penalty(mag: u8) -> u32 {
    if mag < 3 {
        10 * COST_SCALE
    } else if mag <= 10 {
        0
    } else {
        COST_SCALE
    }
}

fn count_water_neighbors_ts(map: &WMap, tile: u32) -> u32 {
    let mut buf = [0u32; 4];
    let n = map.nswe(tile, &mut buf);
    let mut count = 0u32;
    for i in 0..n {
        if map.is_water(buf[i]) {
            count += 1;
        }
    }
    count
}

/// `water.rs:272-292` `coerce_shore_to_water`.
pub fn coerce_shore_to_water(map: &WMap, tile: u32) -> Option<u32> {
    if map.is_water(tile) {
        return Some(tile);
    }
    let mut buf = [0u32; 4];
    let n = map.nswe(tile, &mut buf);
    let mut best: Option<u32> = None;
    let mut max_score = -1i32;
    for i in 0..n {
        let neighbor = buf[i];
        if !map.is_water(neighbor) {
            continue;
        }
        let score = count_water_neighbors_ts(map, neighbor) as i32;
        if score > max_score {
            max_score = score;
            best = Some(neighbor);
        }
    }
    best
}

/// `water.rs:556-771` `bounded_water_path` (fresh scratch each call, num_local
/// cap 10_000).
pub fn bounded_water_path(
    map: &WMap,
    starts: &[u32],
    goal: u32,
    min_x: u32,
    max_x: u32,
    min_y: u32,
    max_y: u32,
) -> Option<Vec<u32>> {
    if starts.is_empty() {
        return None;
    }
    let width = map.w;
    let bounds_w = max_x - min_x + 1;
    let bounds_h = max_y - min_y + 1;
    let num_local = (bounds_w * bounds_h) as usize;
    if num_local == 0 || num_local > 10_000 {
        return None;
    }

    let mut closed = vec![0u32; num_local];
    let mut g_stamp = vec![0u32; num_local];
    let mut g = vec![0u32; num_local];
    let mut came_from = vec![i32::MAX; num_local];
    let mut heap = AstarHeap::new(num_local * 4);
    let stamp = 1u32;
    let land_mask = 1u8 << LAND_BIT;
    let heuristic_weight = 3u32;
    let goal_x = map.x(goal);
    let goal_y = map.y(goal);

    let to_local = |tile: u32| -> Option<usize> {
        let x = map.x(tile);
        let y = map.y(tile);
        if x < min_x || x > max_x || y < min_y || y > max_y {
            return None;
        }
        Some(((y - min_y) * bounds_w + (x - min_x)) as usize)
    };
    let to_global = |local: usize| -> u32 {
        let lx = (local as u32) % bounds_w;
        let ly = (local as u32) / bounds_w;
        map.ref_xy(lx + min_x, ly + min_y)
    };
    let to_local_clamped = |tile: u32| -> Option<usize> {
        let mut x = map.x(tile);
        let mut y = map.y(tile);
        x = x.clamp(min_x, max_x);
        y = y.clamp(min_y, max_y);
        Some(((y - min_y) * bounds_w + (x - min_x)) as usize)
    };

    let goal_local = to_local_clamped(goal)?;
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
        ((cross as f64 * (COST_SCALE - 1) as f64) / cross_norm as f64 / cross_norm as f64).floor()
            as u32
    };

    heap.clear();
    for &s in starts {
        let Some(start_local) = to_local_clamped(s) else {
            continue;
        };
        g[start_local] = 0;
        g_stamp[start_local] = stamp;
        came_from[start_local] = -1;
        let h = heuristic_weight
            * BASE_COST
            * (map.x(s).abs_diff(goal_x) + map.y(s).abs_diff(goal_y));
        heap.push(start_local as u32, h);
    }

    let mut iterations = 100_000u32;
    while !heap.is_empty() {
        iterations -= 1;
        if iterations == 0 {
            return None;
        }
        let current_local = heap.pop() as usize;
        if closed[current_local] == stamp {
            continue;
        }
        closed[current_local] = stamp;
        if current_local == goal_local {
            let mut path = Vec::new();
            let mut p = goal_local as i32;
            while p >= 0 {
                path.push(to_global(p as usize));
                if came_from[p as usize] < 0 {
                    break;
                }
                p = came_from[p as usize];
            }
            path.reverse();
            return Some(path);
        }
        let current = to_global(current_local);
        let current_g = g[current_local];
        let cx = map.x(current);
        let cy = map.y(current);

        let mut try_relax = |nbr: u32, nlx: u32, nly: u32, n_local: usize| {
            if closed[n_local] == stamp {
                return;
            }
            let terrain = map.byte(nbr);
            if nbr != goal && terrain & land_mask != 0 {
                return;
            }
            let magnitude = terrain & 0x1f;
            let cost = BASE_COST + magnitude_penalty(magnitude);
            let tentative = current_g.saturating_add(cost);
            if g_stamp[n_local] != stamp || tentative < g[n_local] {
                came_from[n_local] = current_local as i32;
                g[n_local] = tentative;
                g_stamp[n_local] = stamp;
                let h = heuristic_weight
                    * BASE_COST
                    * (nlx.abs_diff(goal_x) + nly.abs_diff(goal_y));
                let f = tentative + h + cross_tie(nlx, nly);
                heap.push(n_local as u32, f);
            }
        };

        if cy > min_y {
            let nbr = current - width;
            if let Some(n_local) = to_local(nbr) {
                try_relax(nbr, cx, cy - 1, n_local);
            }
        }
        if cy < max_y {
            let nbr = current + width;
            if let Some(n_local) = to_local(nbr) {
                try_relax(nbr, cx, cy + 1, n_local);
            }
        }
        if cx > min_x {
            let nbr = current - 1;
            if let Some(n_local) = to_local(nbr) {
                try_relax(nbr, cx - 1, cy, n_local);
            }
        }
        if cx < max_x {
            let nbr = current + 1;
            if let Some(n_local) = to_local(nbr) {
                try_relax(nbr, cx + 1, cy, n_local);
            }
        }
    }
    None
}

/// `water.rs:920-978` `water_can_see` (Bresenham w/ magnitude-aware diagonal).
fn water_can_see(map: &WMap, from: u32, to: u32, min_magnitude: u8) -> bool {
    let mut x0 = map.x(from);
    let mut y0 = map.y(from);
    let x1 = map.x(to);
    let y1 = map.y(to);
    let dx = (x1 as i32 - x0 as i32).unsigned_abs();
    let dy = (y1 as i32 - y0 as i32).unsigned_abs();
    let sx = if x0 < x1 { 1i32 } else { -1i32 };
    let sy = if y0 < y1 { 1i32 } else { -1i32 };
    let mut err = dx as i32 - dy as i32;
    for _ in 0..100_000 {
        let tile = map.ref_xy(x0, y0);
        if !map.is_water(tile) {
            return false;
        }
        if (map.byte(tile) & 0x1f) < min_magnitude {
            return false;
        }
        if x0 == x1 && y0 == y1 {
            return true;
        }
        let e2 = 2 * err;
        let should_move_x = e2 > -(dy as i32);
        let should_move_y = e2 < dx as i32;
        if should_move_x && should_move_y {
            let nx = (x0 as i32 + sx) as u32;
            let intermediate = map.ref_xy(nx, y0);
            let int_mag = map.byte(intermediate) & 0x1f;
            if !map.is_water(intermediate) || int_mag < min_magnitude {
                let ny = (y0 as i32 + sy) as u32;
                err += dx as i32;
                let alt = map.ref_xy(x0, ny);
                let alt_mag = map.byte(alt) & 0x1f;
                if !map.is_water(alt) || alt_mag < min_magnitude {
                    return false;
                }
                x0 = nx;
                err -= dy as i32;
                y0 = ny;
            } else {
                x0 = nx;
                err -= dy as i32;
                y0 = (y0 as i32 + sy) as u32;
                err += dx as i32;
            }
        } else {
            if should_move_x {
                x0 = (x0 as i32 + sx) as u32;
                err -= dy as i32;
            }
            if should_move_y {
                y0 = (y0 as i32 + sy) as u32;
                err += dx as i32;
            }
        }
    }
    false
}

/// `water.rs:980-1037` `water_trace_line`.
fn water_trace_line(map: &WMap, from: u32, to: u32) -> Option<Vec<u32>> {
    let mut x0 = map.x(from);
    let mut y0 = map.y(from);
    let x1 = map.x(to);
    let y1 = map.y(to);
    let dx = (x1 as i32 - x0 as i32).unsigned_abs();
    let dy = (y1 as i32 - y0 as i32).unsigned_abs();
    let sx = if x0 < x1 { 1i32 } else { -1i32 };
    let sy = if y0 < y1 { 1i32 } else { -1i32 };
    let mut err = dx as i32 - dy as i32;
    let mut tiles = Vec::new();
    for _ in 0..100_000 {
        let tile = map.ref_xy(x0, y0);
        if !map.is_water(tile) {
            return None;
        }
        tiles.push(tile);
        if x0 == x1 && y0 == y1 {
            return Some(tiles);
        }
        let e2 = 2 * err;
        let should_move_x = e2 > -(dy as i32);
        let should_move_y = e2 < dx as i32;
        if should_move_x && should_move_y {
            x0 = (x0 as i32 + sx) as u32;
            err -= dy as i32;
            let intermediate = map.ref_xy(x0, y0);
            if !map.is_water(intermediate) {
                x0 = (x0 as i32 - sx) as u32;
                err += dy as i32;
                y0 = (y0 as i32 + sy) as u32;
                err += dx as i32;
                let alt = map.ref_xy(x0, y0);
                if !map.is_water(alt) {
                    return None;
                }
                tiles.push(alt);
                x0 = (x0 as i32 + sx) as u32;
                err -= dy as i32;
            } else {
                tiles.push(intermediate);
                y0 = (y0 as i32 + sy) as u32;
                err += dx as i32;
            }
        } else {
            if should_move_x {
                x0 = (x0 as i32 + sx) as u32;
                err -= dy as i32;
            }
            if should_move_y {
                y0 = (y0 as i32 + sy) as u32;
                err += dx as i32;
            }
        }
    }
    None
}

/// `water.rs:792-805` `smooth_water_path`.
fn smooth_water_path(map: &WMap, path: &[u32]) -> Vec<u32> {
    if path.len() <= 2 {
        return path.to_vec();
    }
    let mut smoothed = los_smooth_water(map, path, 2);
    smoothed = refine_water_endpoints(map, &smoothed);
    los_smooth_water(map, &smoothed, 3)
}

/// `water.rs:807-847` `los_smooth_water`.
fn los_smooth_water(map: &WMap, path: &[u32], min_magnitude: u8) -> Vec<u32> {
    let mut result = vec![path[0]];
    let mut current = 0usize;
    while current < path.len().saturating_sub(1) {
        let mut lo = current + 1;
        let mut hi = path.len() - 1;
        let mut farthest = lo;
        while lo <= hi {
            let mid = (lo + hi) / 2;
            if water_can_see(map, path[current], path[mid], min_magnitude) {
                farthest = mid;
                lo = mid + 1;
            } else {
                hi = mid.saturating_sub(1);
            }
        }
        if farthest > current + 1 {
            if let Some(trace) = water_trace_line(map, path[current], path[farthest]) {
                for &t in trace.iter().skip(1).take(trace.len().saturating_sub(2)) {
                    result.push(t);
                }
            }
        }
        current = farthest;
        if current < path.len().saturating_sub(1) {
            result.push(path[current]);
        }
    }
    if let Some(&last) = path.last() {
        result.push(last);
    }
    result
}

/// `water.rs:849-886` `refine_water_endpoints`.
fn refine_water_endpoints(map: &WMap, path: &[u32]) -> Vec<u32> {
    if path.len() <= 2 {
        return path.to_vec();
    }
    const REFINE_DIST: u32 = 50;
    const PADDING: u32 = 10;
    let mut result = path.to_vec();

    let start_end = tile_at_path_distance(map, path, 0, REFINE_DIST, true);
    if start_end > 1 {
        if let Some(seg) = bounded_segment_path(map, path[0], path[start_end], PADDING) {
            if !seg.is_empty() {
                result = seg[..seg.len().saturating_sub(1)]
                    .iter()
                    .copied()
                    .chain(result[start_end..].iter().copied())
                    .collect();
            }
        }
    }

    let end_start = tile_at_path_distance(map, &result, result.len() - 1, REFINE_DIST, false);
    if end_start + 2 < result.len() {
        if let Some(mut seg) =
            bounded_segment_path(map, *result.last().unwrap(), result[end_start], PADDING)
        {
            if !seg.is_empty() {
                seg.reverse();
                result = result[..end_start]
                    .iter()
                    .copied()
                    .chain(seg.iter().copied())
                    .collect();
            }
        }
    }
    result
}

fn tile_at_path_distance(map: &WMap, path: &[u32], start: usize, dist: u32, forward: bool) -> usize {
    let mut cum = 0u32;
    let mut idx = start;
    if forward {
        while idx + 1 < path.len() && cum < dist {
            cum += map.manhattan(path[idx], path[idx + 1]);
            idx += 1;
        }
    } else {
        while idx > 0 && cum < dist {
            cum += map.manhattan(path[idx], path[idx - 1]);
            idx -= 1;
        }
    }
    idx
}

fn bounded_segment_path(map: &WMap, from: u32, to: u32, padding: u32) -> Option<Vec<u32>> {
    let x0 = map.x(from);
    let y0 = map.y(from);
    let x1 = map.x(to);
    let y1 = map.y(to);
    let min_x = x0.min(x1).saturating_sub(padding);
    let max_x = (x0.max(x1) + padding).min(map.w - 1);
    let min_y = y0.min(y1).saturating_sub(padding);
    let max_y = (y0.max(y1) + padding).min(map.h - 1);
    bounded_water_path(map, &[from], to, min_x, max_x, min_y, max_y)
}

/// `water.rs:1040-1070` `upscale_cells` (scale factor 2).
fn upscale_cells(cells: &[(u32, u32)], scale: u32) -> Vec<(u32, u32)> {
    if cells.is_empty() {
        return Vec::new();
    }
    let scaled: Vec<(u32, u32)> = cells
        .iter()
        .map(|&(x, y)| (x * scale, y * scale))
        .collect();
    let mut smooth = Vec::new();
    for i in 0..scaled.len().saturating_sub(1) {
        let current = scaled[i];
        let next = scaled[i + 1];
        smooth.push(current);
        let dx = next.0 as i32 - current.0 as i32;
        let dy = next.1 as i32 - current.1 as i32;
        let steps = dx.abs().max(dy.abs()) as u32;
        if steps <= 1 {
            continue;
        }
        for step in 1..steps {
            smooth.push((
                (current.0 as f64 + (dx as f64 * step as f64) / steps as f64).round() as u32,
                (current.1 as f64 + (dy as f64 * step as f64) / steps as f64).round() as u32,
            ));
        }
    }
    if let Some(last) = scaled.last() {
        smooth.push(*last);
    }
    smooth
}

/// `water.rs:1072-1074` `to_mini_ref`.
fn to_mini_ref(full: &WMap, mini: &WMap, tile: u32) -> u32 {
    mini.ref_xy(full.x(tile) / 2, full.y(tile) / 2)
}

/// `water.rs:1076-1106` `fix_path_extremes` (keeps consecutive duplicates).
fn fix_path_extremes(
    full: &WMap,
    mut path: Vec<u32>,
    cell_src: Option<u32>,
    cell_dst: u32,
) -> Vec<u32> {
    if let Some(src) = cell_src {
        if let Some(idx) = path.iter().position(|&t| t == src) {
            if idx != 0 {
                path = path[idx..].to_vec();
            }
        } else {
            path.insert(0, src);
        }
    }
    if let Some(idx) = path.iter().position(|&t| t == cell_dst) {
        if idx + 1 != path.len() {
            path.truncate(idx + 1);
        }
    } else {
        path.push(cell_dst);
    }
    path
}

/// `water.rs:1136-1152` `closest_full_source`.
fn closest_full_source(full: &WMap, froms: &[u32], path_start: u32) -> Option<u32> {
    if froms.len() == 1 {
        return Some(froms[0]);
    }
    let px = full.x(path_start);
    let py = full.y(path_start);
    let mut best: Option<u32> = None;
    let mut best_dist = u32::MAX;
    for &f in froms {
        let dist = full.x(f).abs_diff(px) + full.y(f).abs_diff(py);
        if dist < best_dist {
            best_dist = dist;
            best = Some(f);
        }
    }
    best
}

/// `water.rs:1201-1299` `minimap_inner_path`.
fn minimap_inner_path(
    mini: &WMap,
    hpa: &mut WaterHierarchical,
    mini_froms: &[u32],
    mini_to: u32,
    mini_path_out: &mut Vec<u32>,
) -> bool {
    mini_path_out.clear();
    let mut water_starts = Vec::with_capacity(mini_froms.len());
    let mut water_to_orig: HashMap<u32, Option<u32>> = HashMap::new();
    for &mf in mini_froms {
        let Some(water) = coerce_shore_to_water(mini, mf) else {
            continue;
        };
        let orig = if mini.is_water(mf) { None } else { Some(mf) };
        water_to_orig.insert(water, orig);
        water_starts.push(water);
    }
    if water_starts.is_empty() {
        return false;
    }
    let goal_water = match coerce_shore_to_water(mini, mini_to) {
        Some(w) => w,
        None => return false,
    };
    let to_original = if mini.is_water(mini_to) { None } else { Some(mini_to) };

    // TS `ComponentCheckTransformer`.
    let to_comp = hpa.graph.get_component_id(goal_water);
    water_starts.retain(|&s| hpa.graph.get_component_id(s) == to_comp);
    if water_starts.is_empty() {
        return false;
    }

    let graph_large = hpa.graph.node_count() >= 100;
    let use_hpa = graph_large;

    let found = if water_starts.len() == 1 && water_starts[0] == goal_water {
        mini_path_out.push(goal_water);
        true
    } else if use_hpa {
        match hpa.find_path(mini, &water_starts, goal_water) {
            Some(p) => {
                *mini_path_out = p;
                true
            }
            None => false,
        }
    } else {
        false
    };
    if !found {
        return false;
    }

    if graph_large {
        let smoothed = smooth_water_path(mini, mini_path_out);
        *mini_path_out = smoothed;
    }

    if let Some(&first_water) = mini_path_out.first() {
        if let Some(&Some(orig)) = water_to_orig.get(&first_water) {
            mini_path_out.insert(0, orig);
        }
    }
    if let Some(orig) = to_original {
        if mini_path_out.last() != Some(&orig) {
            mini_path_out.push(orig);
        }
    }
    true
}

/// `water.rs:1301-1310` `upscale_mini_path`.
fn upscale_mini_path(full: &WMap, mini: &WMap, mini_path: &[u32]) -> Vec<u32> {
    let cells: Vec<(u32, u32)> = mini_path
        .iter()
        .map(|&r| (mini.x(r), mini.y(r)))
        .collect();
    upscale_cells(&cells, 2)
        .into_iter()
        .map(|(cx, cy)| full.ref_xy(cx, cy))
        .collect()
}

/// `water.rs:1313-1349` `transport_path_multi_into`.
pub fn transport_path_multi_into(
    full: &WMap,
    mini: &WMap,
    hpa: &mut WaterHierarchical,
    froms: &[u32],
    to: u32,
) -> Option<Vec<u32>> {
    if froms.is_empty() {
        return None;
    }
    let mini_froms: Vec<u32> = froms.iter().map(|&f| to_mini_ref(full, mini, f)).collect();
    let mini_to = to_mini_ref(full, mini, to);
    let mut mini_path = Vec::with_capacity(64);
    if !minimap_inner_path(mini, hpa, &mini_froms, mini_to, &mut mini_path) {
        return None;
    }
    let upscaled = upscale_mini_path(full, mini, &mini_path);
    if upscaled.is_empty() {
        return None;
    }
    let cell_src = closest_full_source(full, froms, upscaled[0]);
    let path = fix_path_extremes(full, upscaled, cell_src, to);
    if path.is_empty() {
        None
    } else {
        Some(path)
    }
}

// ── Engine-equivalent boat pairing ─────────────────────────────────────────

/// `game.rs:606-633` `get_water_component`, over the ported mini HPA.
pub fn get_water_component(full: &WMap, mini: &WMap, hpa: &WaterHierarchical, tile: u32) -> Option<u32> {
    let mini_x = full.x(tile) / 2;
    let mini_y = full.y(tile) / 2;
    let mini_tile = mini.ref_xy(mini_x, mini_y);
    if mini.is_water(mini_tile) {
        return Some(hpa.graph.get_component_id(mini_tile));
    }
    let mut one_hop = [0u32; 4];
    let n1 = mini.nswe(mini_tile, &mut one_hop);
    for i in 0..n1 {
        if mini.is_water(one_hop[i]) {
            return Some(hpa.graph.get_component_id(one_hop[i]));
        }
    }
    for i in 0..n1 {
        let mut two_hop = [0u32; 4];
        let n2 = mini.nswe(one_hop[i], &mut two_hop);
        for j in 0..n2 {
            if mini.is_water(two_hop[j]) {
                return Some(hpa.graph.get_component_id(two_hop[j]));
            }
        }
    }
    None
}

/// `spatial.rs:13-19` `refine_start_tile` (the `path.len() <= 50` branch, the
/// only regime the port reaches).
fn refine_start_tile(path: &[u32], shores: &[u32]) -> u32 {
    if path.is_empty() {
        return shores[0];
    }
    path[0]
}

/// `spatial.rs:161-185` `closest_shore_by_water`.
///
/// `owner_border` is the owner's border tile list in the engine's own order
/// (the device's `oborder` slice for the owner, which is exactly
/// `for_each_border_tile`'s order).
pub fn closest_shore_by_water(
    full: &WMap,
    mini: &WMap,
    hpa: &mut WaterHierarchical,
    owner_border: &[u32],
    target: u32,
) -> Option<u32> {
    if !full.is_water(target) && !full.is_shore(target) {
        return None;
    }
    let target_comp = get_water_component(full, mini, hpa, target)?;

    let mut shores: Vec<u32> = Vec::with_capacity(32);
    for &t in owner_border {
        if full.is_shore(t) && full.is_land(t) {
            if get_water_component(full, mini, hpa, t) == Some(target_comp) {
                shores.push(t);
            }
        }
    }
    if shores.is_empty() {
        return None;
    }
    let path = transport_path_multi_into(full, mini, hpa, &shores, target)?;
    Some(refine_start_tile(&path, &shores))
}

/// `Game::plan_water_path(from, to)` for one source - the ship's own route.
pub fn plan_water_path(
    full: &WMap,
    mini: &WMap,
    hpa: &mut WaterHierarchical,
    from: u32,
    to: u32,
) -> Option<Vec<u32>> {
    transport_path_multi_into(full, mini, hpa, &[from], to)
}
