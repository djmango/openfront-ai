//! sfc32 PRNG - bit-identical to `openfront/src/core/PseudoRandom.ts`.

const POW36_8: f64 = 2_821_109_907_456.0; // 36^8

#[derive(Clone, Debug)]
pub struct PseudoRandom {
    s0: i32,
    s1: i32,
    s2: i32,
    s3: i32,
}

impl PseudoRandom {
    pub fn new(seed: i32) -> Self {
        let mut h = seed;
        let mut split = || {
            h = h.wrapping_add(0x9e37_79b9u32 as i32);
            let mut t = h ^ ((h as u32) >> 16) as i32;
            t = t.wrapping_mul(0x21f0_aaad);
            t ^= ((t as u32) >> 15) as i32;
            t = t.wrapping_mul(0x735a_2d97);
            t ^ ((t as u32) >> 15) as i32
        };
        let mut pr = Self {
            s0: split(),
            s1: split(),
            s2: split(),
            s3: split(),
        };
        for _ in 0..12 {
            pr.next();
        }
        pr
    }

    pub fn next(&mut self) -> f64 {
        let t = ((self.s0.wrapping_add(self.s1)) as i32).wrapping_add(self.s3);
        self.s3 = self.s3.wrapping_add(1);
        self.s0 = self.s1 ^ ((self.s1 as u32) >> 9) as i32;
        self.s1 = self.s2.wrapping_add(self.s2.wrapping_shl(3));
        let s2_bits = self.s2 as u32;
        self.s2 = ((s2_bits << 21) | (s2_bits >> 11)) as i32;
        self.s2 = self.s2.wrapping_add(t);
        (t as u32) as f64 / 4_294_967_296.0
    }

    pub fn next_int(&mut self, min: i32, max: i32) -> i32 {
        let lo = min;
        let hi = max;
        (self.next() * (hi - lo) as f64).floor() as i32 + lo
    }

    /// Random float in `[min, max)` (TS `nextFloat`).
    pub fn next_float(&mut self, min: f64, max: f64) -> f64 {
        self.next() * (max - min) + min
    }

    /// Returns true with probability 1/odds (TS `chance`).
    pub fn chance(&mut self, odds: i32) -> bool {
        self.next_int(0, odds) == 0
    }

    pub fn next_id(&mut self) -> String {
        let mut v = (self.next() * POW36_8).floor() as u64;
        let mut out = String::with_capacity(8);
        for _ in 0..8 {
            let digit = (v % 36) as u8;
            out.insert(
                0,
                std::str::from_utf8(&[b"0123456789abcdefghijklmnopqrstuvwxyz"[digit as usize]])
                    .unwrap()
                    .chars()
                    .next()
                    .unwrap(),
            );
            v /= 36;
        }
        out
    }

    /// Random element (TS `randElement`).
    pub fn rand_element<T: Clone>(&mut self, arr: &[T]) -> Option<T> {
        if arr.is_empty() {
            return None;
        }
        let idx = self.next_int(0, arr.len() as i32) as usize;
        Some(arr[idx].clone())
    }

    /// Random element from a JS `Set` (TS `randFromSet`) - the caller passes the set's
    /// members in insertion order (JS `Set` iteration order), so this is equivalent to
    /// `rand_element` given that same order; kept as a distinct name to mirror the TS API
    /// call sites (e.g. `NationStructureBehavior.arraySampler`'s reservoir-style sampling).
    pub fn rand_from_set<T: Clone>(&mut self, items: &[T]) -> Option<T> {
        self.rand_element(items)
    }

    /// Fisher–Yates shuffle (TS `shuffleArray`).
    pub fn shuffle_array<T: Clone>(&mut self, array: &[T]) -> Vec<T> {
        let mut result = array.to_vec();
        for i in (1..result.len()).rev() {
            let j = self.next_int(0, (i + 1) as i32) as usize;
            result.swap(i, j);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_stream() {
        let mut a = PseudoRandom::new(999);
        let mut b = PseudoRandom::new(999);
        for _ in 0..20 {
            assert_eq!(a.next().to_bits(), b.next().to_bits());
        }
    }

    #[test]
    fn next_stream_matches_ts_jby2g() {
        use crate::util::simple_hash;
        let mut prng = PseudoRandom::new(simple_hash("jby2gMJF"));
        let expected = [
            0.8149932413361967,
            0.2939943221863359,
            0.055542743066325784,
        ];
        for e in expected {
            let got = prng.next();
            assert!((got - e).abs() < 1e-10, "got {got} expected {e}");
        }
    }

    #[test]
    fn next_id_matches_ts_jby2g() {
        use crate::util::simple_hash;
        let mut prng = PseudoRandom::new(simple_hash("jby2gMJF"));
        assert_eq!(prng.next_id(), "tc8borp7");
        assert_eq!(prng.next_id(), "al0lkff3");
        assert_eq!(prng.next_id(), "1zzeh9zz");
    }

    // Everything below is ported 1:1 from `openfront/tests/PseudoRandom.test.ts`. This is
    // the single highest-leverage file in the whole port - every other mechanic's parity
    // depends on this stream being bit-identical, so scenarios are ported exhaustively
    // rather than just broadly.

    #[test]
    fn same_seed_produces_an_identical_sequence() {
        let mut a = PseudoRandom::new(42);
        let mut b = PseudoRandom::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next().to_bits(), b.next().to_bits());
        }
    }

    #[test]
    fn same_seed_produces_identical_derived_values() {
        let mut a = PseudoRandom::new(987654);
        let mut b = PseudoRandom::new(987654);
        for _ in 0..100 {
            assert_eq!(a.next_int(0, 1000), b.next_int(0, 1000));
        }
        assert_eq!(a.next_id(), b.next_id());
        let arr = [1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(a.shuffle_array(&arr), b.shuffle_array(&arr));
    }

    #[test]
    fn different_seeds_produce_different_sequences() {
        let mut a = PseudoRandom::new(1);
        let mut b = PseudoRandom::new(2);
        let mut same = 0;
        for _ in 0..100 {
            if a.next().to_bits() == b.next().to_bits() {
                same += 1;
            }
        }
        assert!(same < 5, "same={same}");
    }

    #[test]
    fn consecutive_integer_seeds_are_not_correlated() {
        // Weak seeding schemes make adjacent seeds (common: tick numbers, sequential
        // hashes) produce similar streams.
        let mut values = Vec::new();
        for seed in 1000..1100 {
            values.push(PseudoRandom::new(seed).next_int(0, 100));
        }
        let distinct: std::collections::HashSet<_> = values.iter().collect();
        assert!(distinct.len() > 50, "distinct={}", distinct.len());
    }

    #[test]
    fn next_stays_within_0_1() {
        let mut r = PseudoRandom::new(7);
        for _ in 0..10000 {
            let v = r.next();
            assert!((0.0..1.0).contains(&v), "v={v}");
        }
    }

    #[test]
    fn next_is_roughly_uniform() {
        let mut r = PseudoRandom::new(1234);
        let n = 20000;
        let mut sum = 0.0;
        let mut buckets = [0u32; 10];
        for _ in 0..n {
            let v = r.next();
            sum += v;
            buckets[(v * 10.0).floor() as usize] += 1;
        }
        let mean = sum / n as f64;
        assert!(mean > 0.48 && mean < 0.52, "mean={mean}");
        for count in buckets {
            // Expected 2000 per bucket; allow generous slack.
            assert!(count > 1700 && count < 2300, "count={count}");
        }
    }

    #[test]
    fn next_int_returns_integers_in_min_max() {
        let mut r = PseudoRandom::new(99);
        let mut seen = std::collections::HashSet::new();
        for _ in 0..10000 {
            let v = r.next_int(3, 8);
            assert!((3..8).contains(&v), "v={v}");
            seen.insert(v);
        }
        let mut seen: Vec<_> = seen.into_iter().collect();
        seen.sort();
        assert_eq!(seen, vec![3, 4, 5, 6, 7]);
    }

    #[test]
    fn next_int_with_single_value_range_always_returns_it() {
        let mut r = PseudoRandom::new(5);
        for _ in 0..100 {
            assert_eq!(r.next_int(4, 5), 4);
        }
    }

    // TS's `nextInt` additionally does `Math.floor(min)`/`Math.floor(max)` before
    // computing, since JS numbers have no static int/float distinction - this native
    // port's `next_int` takes `i32` directly, so non-integer bounds can't even be passed;
    // that TS edge case has no native counterpart to diverge on.

    #[test]
    fn next_float_stays_within_min_max() {
        let mut r = PseudoRandom::new(11);
        for _ in 0..1000 {
            let v = r.next_float(2.5, 3.5);
            assert!((2.5..3.5).contains(&v), "v={v}");
        }
    }

    #[test]
    fn next_id_returns_8_alphanumeric_characters() {
        let mut r = PseudoRandom::new(123);
        for _ in 0..100 {
            let id = r.next_id();
            assert_eq!(id.len(), 8, "id={id}");
            assert!(
                id.chars().all(|c| c.is_ascii_digit() || c.is_ascii_lowercase()),
                "id={id}"
            );
        }
    }

    #[test]
    fn rand_element_picks_members_and_none_on_empty() {
        let mut r = PseudoRandom::new(77);
        let arr = ["a", "b", "c"];
        for _ in 0..100 {
            let picked = r.rand_element(&arr).unwrap();
            assert!(arr.contains(&picked));
        }
        let empty: [&str; 0] = [];
        assert_eq!(r.rand_element(&empty), None);
    }

    #[test]
    fn rand_from_set_picks_members_and_none_on_empty() {
        let mut r = PseudoRandom::new(78);
        let set = ["x", "y", "z"];
        for _ in 0..100 {
            let picked = r.rand_from_set(&set).unwrap();
            assert!(set.contains(&picked));
        }
        let empty: [&str; 0] = [];
        assert_eq!(r.rand_from_set(&empty), None);
    }

    #[test]
    fn chance_1_is_always_true_chance_large_is_mostly_false() {
        let mut r = PseudoRandom::new(31);
        for _ in 0..100 {
            assert!(r.chance(1));
        }
        let mut hits = 0;
        for _ in 0..1000 {
            if r.chance(1000) {
                hits += 1;
            }
        }
        assert!(hits < 10, "hits={hits}");
    }

    #[test]
    fn shuffle_array_returns_a_permutation_and_leaves_input_unchanged() {
        let mut r = PseudoRandom::new(55);
        let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let copy = input;
        let shuffled = r.shuffle_array(&input);
        assert_eq!(input, copy);
        let mut sorted = shuffled.clone();
        sorted.sort();
        assert_eq!(sorted, copy.to_vec());
    }
}
