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
    1.0 / (1.0 + (-decay_rate * (x - midpoint)).exp())
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

/// Fixture-record/map-directory tests default to `OPENFRONT_REPO` when set,
/// but need *some* fallback for a plain `cargo test` with no env var. That
/// fallback used to be a hardcoded absolute path to one developer's laptop
/// checkout (`/Users/djmango/github/openfront-ai...`), which only worked on
/// that machine. `CARGO_MANIFEST_DIR` (this crate's own source directory,
/// baked in at compile time by cargo, not read from the runtime
/// environment) is fixed relative to the repo root by this workspace's
/// layout (`<repo_root>/rust/engine`) regardless of who or where it's
/// checked out, so `.../../.. ` from it is portable everywhere the source
/// tree itself is portable.
pub fn default_repo_root() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/../..").to_string()
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
