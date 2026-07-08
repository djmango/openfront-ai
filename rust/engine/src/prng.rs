//! sfc32 PRNG - bit-identical to `openfront/src/core/PseudoRandom.ts`.

const POW36_8: f64 = 2_821_109_907_456.0; // 36^8

#[derive(Clone)]
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
}
