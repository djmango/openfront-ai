//! Insertion-ordered unit-id set (TS `Set<Unit>` semantics, keyed by unit id).
//!
//! `NationWarshipBehavior.ts`'s `trackedTransportShips`/`trackedTradeShips`/
//! `trackedIncomingTransportShips` are all `Set<Unit>`. TS objects have stable identity, so a
//! `Set<Unit>` genuinely tracks the same live object across ticks; native has no such identity
//! for a unit that outlives the `Player.units` entry that briefly held it, so these are tracked
//! by unit id (`i32`) instead. Some of that file's `for (const ship of trackedX)` loops feed a
//! PRNG draw (`maybeRetaliateWithWarship`'s difficulty roll) or pick the first match via `break`
//! (`trackIncomingTransportsAndRetaliate`) for whichever tracked entry is reached first - a plain
//! `HashSet<i32>` would iterate in a per-process-random order and desync the PRNG stream /
//! target choice from TS's insertion-order iteration. Same rationale and shape as
//! `ordered_tiles::OrderedTiles`, just keyed by unit id instead of `TileRef`.

use std::collections::HashSet;

#[derive(Clone, Default, Debug)]
pub struct OrderedUnitSet {
    ids: Vec<i32>,
    set: HashSet<i32>,
}

impl OrderedUnitSet {
    pub fn insert(&mut self, id: i32) -> bool {
        if self.set.insert(id) {
            self.ids.push(id);
            true
        } else {
            false
        }
    }

    pub fn contains(&self, id: i32) -> bool {
        self.set.contains(&id)
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = i32> + '_ {
        self.ids.iter().copied()
    }

    /// Snapshot for iteration that removes while visiting (TS "deleting the current entry
    /// while iterating a `Set` is safe" - mirrored by iterating a copy here).
    pub fn to_vec(&self) -> Vec<i32> {
        self.ids.clone()
    }

    pub fn remove(&mut self, id: i32) {
        if self.set.remove(&id) {
            if let Some(i) = self.ids.iter().position(|&v| v == id) {
                self.ids.remove(i);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::OrderedUnitSet;

    #[test]
    fn adds_reports_membership_and_size() {
        let mut s = OrderedUnitSet::default();
        assert_eq!(s.len(), 0);
        s.insert(5);
        s.insert(9);
        s.insert(5); // duplicate
        assert_eq!(s.len(), 2);
        assert!(s.contains(5));
        assert!(s.contains(9));
        assert!(!s.contains(6));
    }

    #[test]
    fn iterates_in_insertion_order() {
        let mut s = OrderedUnitSet::default();
        for v in [42, 7, -13, 0, 100000] {
            s.insert(v);
        }
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![42, 7, -13, 0, 100000]);
    }

    #[test]
    fn deletes_and_moves_a_re_added_value_to_the_end() {
        let mut s = OrderedUnitSet::default();
        for v in [1, 2, 3] {
            s.insert(v);
        }
        s.remove(2);
        assert!(!s.contains(2));
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![1, 3]);
        s.insert(2);
        assert_eq!(s.iter().collect::<Vec<_>>(), vec![1, 3, 2]);
    }

    #[test]
    fn removing_a_missing_value_is_a_no_op() {
        let mut s = OrderedUnitSet::default();
        s.insert(1);
        s.remove(99);
        assert_eq!(s.len(), 1);
        assert!(s.contains(1));
    }
}
