//! Insertion-ordered tile set (TS `Set<TileRef>` semantics).

use crate::map::TileRef;
use std::collections::HashSet;

#[derive(Clone, Default, Debug)]
pub struct OrderedTiles {
    tiles: Vec<TileRef>,
    set: HashSet<TileRef>,
}

impl OrderedTiles {
    pub fn insert(&mut self, t: TileRef) -> bool {
        if self.set.insert(t) {
            self.tiles.push(t);
            true
        } else {
            false
        }
    }

    pub fn contains(&self, t: TileRef) -> bool {
        self.set.contains(&t)
    }

    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }

    pub fn clear(&mut self) {
        self.tiles.clear();
        self.set.clear();
    }

    pub fn iter(&self) -> impl Iterator<Item = TileRef> + '_ {
        self.tiles.iter().copied()
    }

    pub fn as_slice(&self) -> &[TileRef] {
        &self.tiles
    }

    pub fn remove(&mut self, tile: TileRef) {
        if self.set.remove(&tile) {
            if let Some(i) = self.tiles.iter().position(|&t| t == tile) {
                self.tiles.remove(i);
            }
        }
    }
}
