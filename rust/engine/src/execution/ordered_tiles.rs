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

// TS `TileSet.test.ts` - TS's `TileSet` is a custom insertion-ordered set
// (dense array + open-addressing hash table) used for both
// `Player._tiles`/`_borderTiles`. `OrderedTiles` is native's structural
// equivalent (`border_tiles`; see `game.rs`'s `owned_tiles_preserve_*` tests
// for the plain-`Vec` `owned_tiles` case used for `Player._tiles`). Ported
// invariants: insertion order preserved, delete+re-add moves a value to the
// end (matching JS `Set` semantics, which `TileSet`'s own doc comment
// explicitly targets), re-adding an existing value leaves its position
// unchanged, and membership/size stay correct throughout. `OrderedTiles` has
// no tombstone-compaction step (`Vec::remove` shifts immediately), so the
// TS suite's compaction-specific test has no analogous internal state to
// probe here; the externally observable order/membership invariants it
// protects are the same ones covered below and by the randomized test.
#[cfg(test)]
mod tests {
    use super::OrderedTiles;
    use crate::prng::PseudoRandom;
    use std::collections::HashSet;

    #[test]
    fn adds_reports_membership_and_size() {
        let mut s = OrderedTiles::default();
        assert_eq!(s.len(), 0);
        assert!(!s.contains(5));
        s.insert(5);
        s.insert(9);
        s.insert(5); // duplicate
        assert_eq!(s.len(), 2);
        assert!(s.contains(5));
        assert!(s.contains(9));
        assert!(!s.contains(6));
    }

    #[test]
    fn deletes_and_reports_whether_the_value_was_present() {
        let mut s = OrderedTiles::default();
        for t in [1, 2, 3] {
            s.insert(t);
        }
        assert!(s.contains(2));
        s.remove(2);
        assert!(!s.contains(2));
        s.remove(2); // no-op, already gone
        s.remove(99); // no-op, never present
        assert_eq!(s.len(), 2);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![1, 3]);
    }

    #[test]
    fn iterates_in_insertion_order() {
        let values = [42u32, 7, 100000, 0, 13];
        let mut s = OrderedTiles::default();
        for &v in &values {
            s.insert(v);
        }
        assert_eq!(s.iter().collect::<Vec<_>>(), values.to_vec());
        assert_eq!(s.as_slice(), values.as_slice());
    }

    #[test]
    fn moves_a_value_to_the_end_on_delete_plus_re_add() {
        let mut s = OrderedTiles::default();
        for t in [1, 2, 3] {
            s.insert(t);
        }
        s.remove(1);
        s.insert(1);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![2, 3, 1]);
    }

    #[test]
    fn re_adding_an_existing_value_does_not_change_its_position() {
        let mut s = OrderedTiles::default();
        for t in [1, 2, 3] {
            s.insert(t);
        }
        s.insert(1);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    #[test]
    fn clear_empties_the_set() {
        let mut s = OrderedTiles::default();
        for t in [1, 2, 3] {
            s.insert(t);
        }
        s.clear();
        assert_eq!(s.len(), 0);
        assert!(!s.contains(1));
        assert_eq!(s.iter().collect::<Vec<_>>(), Vec::<u32>::new());
        s.insert(7);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![7]);
    }

    #[test]
    fn handles_large_tile_refs_up_to_the_65535x65535_map_bound() {
        let big = 65535u32 * 65535 - 1;
        let mut s = OrderedTiles::default();
        for t in [big, 0, big - 1] {
            s.insert(t);
        }
        assert!(s.contains(big));
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![big, 0, big - 1]);
    }

    #[test]
    fn matches_reference_set_behavior_on_a_randomized_operation_sequence() {
        let mut random = PseudoRandom::new(12345);
        let mut tile_set = OrderedTiles::default();
        // Reference: insertion-ordered like TS `Set`, verified the same way.
        let mut reference: Vec<u32> = Vec::new();
        let mut reference_set: HashSet<u32> = HashSet::new();
        for op in 0..20_000 {
            let value = random.next_int(0, 500) as u32;
            if random.chance(3) {
                let was_present = reference_set.remove(&value);
                if was_present {
                    reference.retain(|&v| v != value);
                }
                let had = tile_set.contains(value);
                tile_set.remove(value);
                assert_eq!(had, was_present, "op {op}: delete({value})");
            } else {
                if reference_set.insert(value) {
                    reference.push(value);
                }
                tile_set.insert(value);
            }
            if op % 500 == 0 {
                assert_eq!(tile_set.len(), reference.len(), "op {op}: size");
                assert_eq!(tile_set.iter().collect::<Vec<_>>(), reference, "op {op}: order");
            }
        }
        assert_eq!(tile_set.iter().collect::<Vec<_>>(), reference);
        for v in 0..500u32 {
            assert_eq!(tile_set.contains(v), reference_set.contains(&v));
        }
    }
}
