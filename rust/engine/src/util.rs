//! djb2-style hash used throughout the TS engine (`Util.simpleHash`).

/// Match `openfront/src/core/Util.ts::simpleHash`.
pub fn simple_hash(s: &str) -> i32 {
    let mut hash: i32 = 0;
    for ch in s.chars() {
        let c = ch as i32;
        hash = hash.wrapping_shl(5).wrapping_sub(hash).wrapping_add(c);
        // TS: hash = hash & hash (truncate to i32)
        hash = hash.wrapping_mul(1);
    }
    hash.abs()
}

/// TS `Util.within`.
pub fn within(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

/// TS `Util.sigmoid`.
pub fn sigmoid(x: f64, decay_rate: f64, midpoint: f64) -> f64 {
    1.0 / (1.0 + (decay_rate * (x - midpoint)).exp())
}

/// TS `Util.toInt` as i32 for troop math.
pub fn to_int(num: f64) -> i32 {
    if num.is_infinite() {
        return if num.is_sign_positive() {
            i32::MAX
        } else {
            i32::MIN
        };
    }
    num.floor() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_ts_examples() {
        // Golden values from running TS simpleHash on node.
        assert_eq!(simple_hash(""), 0);
        assert_eq!(simple_hash("abc"), 96_354);
    }
}
