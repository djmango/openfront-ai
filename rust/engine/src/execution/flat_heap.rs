//! Min-heap for attack tile priorities (TS `FlatBinaryHeap`).

use crate::map::TileRef;

pub struct FlatBinaryHeap {
    pri: Vec<f32>,
    tiles: Vec<TileRef>,
}

impl FlatBinaryHeap {
    pub fn new(capacity: usize) -> Self {
        Self {
            pri: Vec::with_capacity(capacity),
            tiles: Vec::with_capacity(capacity),
        }
    }

    pub fn clear(&mut self) {
        self.pri.clear();
        self.tiles.clear();
    }

    pub fn len(&self) -> usize {
        self.pri.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pri.is_empty()
    }

    pub fn enqueue(&mut self, tile: TileRef, priority: f32) {
        let mut i = self.pri.len();
        self.pri.push(0.0);
        self.tiles.push(0);
        while i > 0 {
            let parent = (i - 1) >> 1;
            if priority >= self.pri[parent] {
                break;
            }
            self.pri[i] = self.pri[parent];
            self.tiles[i] = self.tiles[parent];
            i = parent;
        }
        self.pri[i] = priority;
        self.tiles[i] = tile;
    }

    pub fn dequeue(&mut self) -> Option<TileRef> {
        if self.pri.is_empty() {
            return None;
        }
        let top = self.tiles[0];
        let last_pri = self.pri.pop()?;
        let last_tile = self.tiles.pop()?;
        if self.pri.is_empty() {
            return Some(top);
        }
        let mut i = 0usize;
        loop {
            let left = (i << 1) + 1;
            if left >= self.pri.len() {
                break;
            }
            let right = left + 1;
            let child = if right < self.pri.len() && self.pri[right] < self.pri[left] {
                right
            } else {
                left
            };
            if last_pri <= self.pri[child] {
                break;
            }
            self.pri[i] = self.pri[child];
            self.tiles[i] = self.tiles[child];
            i = child;
        }
        self.pri[i] = last_pri;
        self.tiles[i] = last_tile;
        Some(top)
    }
}
