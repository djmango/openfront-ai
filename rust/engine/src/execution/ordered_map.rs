//! Insertion-ordered map (TS `Map<K, V>` semantics).
//!
//! Rust's `HashMap` has no defined iteration order, and as of `std`'s default
//! `RandomState` hasher, that order is randomly re-seeded *per process* - so iterating a
//! `HashMap` (e.g. to pick an attack target after a stable sort with tied keys) is genuinely
//! nondeterministic across runs of the same binary on the same input. TS `Map` always iterates
//! in insertion order (setting an existing key's value does NOT move it), which game logic
//! (e.g. `PlayerImpl.allRelationsSorted()`) depends on as a stable tie-break after `.sort()`.
//! This type replicates that: `set` on a new key appends it; `set` on an existing key updates
//! it in place; iteration is always insertion order; `remove` followed by re-`set` of the same
//! key goes to the end, matching JS `Map` delete+re-insert semantics.

use std::collections::HashMap;
use std::hash::Hash;

#[derive(Clone, Debug)]
pub struct OrderedMap<K, V> {
    order: Vec<K>,
    map: HashMap<K, V>,
}

impl<K, V> Default for OrderedMap<K, V> {
    fn default() -> Self {
        Self {
            order: Vec::new(),
            map: HashMap::new(),
        }
    }
}

impl<K: Copy + Eq + Hash, V> OrderedMap<K, V> {
    pub fn new() -> Self {
        Self::default()
    }

    /// TS `Map.set` - inserts at the end if `key` is new; updates in place otherwise.
    pub fn set(&mut self, key: K, value: V) {
        if let Some(existing) = self.map.get_mut(&key) {
            *existing = value;
        } else {
            self.order.push(key);
            self.map.insert(key, value);
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.map.get_mut(key)
    }

    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// TS `Map.delete` - removing then re-`set`ting the same key sends it to the end.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let removed = self.map.remove(key);
        if removed.is_some() {
            if let Some(i) = self.order.iter().position(|k| k == key) {
                self.order.remove(i);
            }
        }
        removed
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Insertion-order iteration, matching TS `Array.from(map, ...)` / `map.forEach`.
    pub fn iter(&self) -> impl Iterator<Item = (K, &V)> + '_ {
        self.order.iter().map(move |k| (*k, self.map.get(k).expect("OrderedMap invariant")))
    }

    pub fn keys(&self) -> impl Iterator<Item = K> + '_ {
        self.order.iter().copied()
    }

    /// Gets the current value for `key`, inserting `default` first if absent, and returns a
    /// mutable reference to it - mirrors the common `map.get(k) ?? default; map.set(k, v)`
    /// read-modify-write pattern used by TS callers (e.g. `updateRelation`), without ever
    /// touching the position of an already-present key.
    pub fn entry_or_insert(&mut self, key: K, default: V) -> &mut V {
        if !self.map.contains_key(&key) {
            self.order.push(key);
            self.map.insert(key, default);
        }
        self.map.get_mut(&key).expect("OrderedMap invariant")
    }
}
