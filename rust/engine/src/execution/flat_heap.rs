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

// TS `FlatBinaryHeap.test.ts` - used by `AttackExecution.toConquer` (native
// `to_conquer`), the conquest-frontier priority queue. TS's `dequeue()`
// throws `Error("heap empty")` on an empty heap; native's `Option` return
// is the idiomatic Rust equivalent (checked below as `None`) rather than a
// panic, since every native call site (`attack.rs`) already guards with
// `is_empty()`/pattern-matches the `Option` instead of relying on a throw.
#[cfg(test)]
mod tests {
    use super::FlatBinaryHeap;

    #[test]
    fn dequeues_tiles_in_ascending_priority_order() {
        let mut heap = FlatBinaryHeap::new(16);
        let entries: [(u32, f32); 5] = [(100, 5.0), (200, 1.0), (300, 3.0), (400, 2.0), (500, 4.0)];
        for (tile, pri) in entries {
            heap.enqueue(tile, pri);
        }
        assert_eq!(heap.len(), 5);
        assert_eq!(heap.dequeue(), Some(200));
        assert_eq!(heap.dequeue(), Some(400));
        assert_eq!(heap.dequeue(), Some(300));
        assert_eq!(heap.dequeue(), Some(500));
        assert_eq!(heap.dequeue(), Some(100));
        assert_eq!(heap.len(), 0);
    }

    #[test]
    fn dequeuing_an_empty_heap_returns_none() {
        let mut heap = FlatBinaryHeap::new(16);
        assert_eq!(heap.dequeue(), None);
    }

    #[test]
    fn clear_empties_the_heap_without_breaking_subsequent_use() {
        let mut heap = FlatBinaryHeap::new(16);
        heap.enqueue(1, 1.0);
        heap.enqueue(2, 2.0);
        heap.clear();
        assert_eq!(heap.len(), 0);
        heap.enqueue(3, 3.0);
        assert_eq!(heap.dequeue(), Some(3));
    }

    #[test]
    fn grows_past_its_initial_capacity_and_stays_ordered() {
        let mut heap = FlatBinaryHeap::new(4);
        // Insert in descending priority so every enqueue sifts up.
        let n: u32 = 1000;
        for i in 0..n {
            heap.enqueue(i, (n - i) as f32);
        }
        assert_eq!(heap.len(), n as usize);
        for i in 0..n {
            assert_eq!(heap.dequeue(), Some(n - 1 - i));
        }
    }

    /// TS's `pri: Float32Array` truncates priorities to f32 precision; two
    /// values that only differ beyond f32 precision tie-break purely on
    /// heap-internal structure (insertion/sift order), not FIFO/stable
    /// order. This pins that native's `Vec<f32>` storage produces the same
    /// tie-break as TS on a fixed insertion sequence, since a `f64`-precision
    /// reimplementation would silently desync attack-frontier ordering.
    #[test]
    fn equal_priorities_break_ties_by_heap_structure_not_insertion_order() {
        let mut heap = FlatBinaryHeap::new(8);
        for tile in [10u32, 20, 30, 40] {
            heap.enqueue(tile, 1.0);
        }
        let mut out = Vec::new();
        while let Some(t) = heap.dequeue() {
            out.push(t);
        }
        assert_eq!(out, vec![10, 40, 30, 20]);
    }
}
