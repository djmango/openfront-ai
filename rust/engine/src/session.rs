//! RL env session - TS engine via multiplexed daemon (default) or per-process bridge.

use crate::backend::bridge::BridgeClient;
use crate::backend::daemon::{use_daemon, DaemonSession};
use crate::backend::stub::StubSession;
use crate::record::StampedIntent;
use crate::util::simple_hash;
use serde_json::Value;
use std::path::Path;

pub const AGENT_CLIENT_ID: &str = "AGENTRL1";
pub const AGENT_CLIENT_ID_2: &str = "AGENTRL2";
pub const AGENT_CLIENT_IDS: [&str; 2] = [AGENT_CLIENT_ID, AGENT_CLIENT_ID_2];

pub enum EnvSession {
    /// Multiplexed TS engine (one daemon, many sessions).
    Daemon(DaemonSession),
    /// Legacy: one tsx subprocess per env.
    Ts(BridgeClient),
    /// Incomplete Rust port (`OPENFRONT_STUB=1` only).
    Stub(StubSession),
}

impl EnvSession {
    pub fn reset(
        repo_root: &Path,
        map_key: &str,
        seed: &str,
        bots: u32,
    ) -> Result<(Self, Value, Vec<u8>, Vec<u8>), String> {
        if std::env::var("OPENFRONT_STUB").ok().as_deref() == Some("1") {
            let (stub, head, terrain, tiles) = StubSession::reset(repo_root, map_key, seed, bots)?;
            return Ok((Self::Stub(stub), head, terrain, tiles));
        }
        if use_daemon() {
            let mut daemon = DaemonSession::open(repo_root)?;
            let (head, tiles, terrain) = daemon.reset(map_key, seed, bots)?;
            return Ok((Self::Daemon(daemon), head, terrain, tiles));
        }
        let mut bridge = BridgeClient::spawn(repo_root)?;
        let (head, tiles, terrain) = bridge.reset(map_key, seed, bots)?;
        Ok((Self::Ts(bridge), head, terrain, tiles))
    }

    pub fn step(&mut self, _repo_root: &Path, intents: Vec<StampedIntent>, ticks: u32) -> (Value, Vec<u8>, u32) {
        match self {
            Self::Daemon(d) => {
                let (mut head, tiles) = d
                    .step(intents, ticks)
                    .unwrap_or_else(|e| panic!("daemon step: {e}"));
                let wasted = head
                    .get("wasted")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
                (head, tiles, wasted)
            }
            Self::Ts(b) => {
                let (mut head, tiles) = b
                    .step(intents, ticks)
                    .unwrap_or_else(|e| panic!("bridge step: {e}"));
                let wasted = head
                    .get("wasted")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
                (head, tiles, wasted)
            }
            Self::Stub(s) => s.step(intents, ticks),
        }
    }
}

pub(crate) fn terrain_bytes(game: &crate::game::Game) -> Vec<u8> {
    game.map.terrain_bytes().to_vec()
}

/// Derive the 8-char game id from an episode seed.
///
/// Bit-for-bit port of `bridge/session.ts::seedToGameID` (also duplicated
/// identically in `bridge/env.ts:74-85`):
///
/// ```ts
/// let h = simpleHash(`rl-${seed}`);          // 0 .. 2^31, integer-valued f64
/// for (let i = 0; i < 8; i++) {
///   h = (h * 1103515245 + 12345) & 0x7fffffff;
///   out += alphabet[h % alphabet.length];
/// }
/// ```
///
/// The arithmetic is **IEEE-754 double**, not wrapping u32. For
/// `h > 2^22` the exact product `h * 1103515245` exceeds 2^53, so the
/// double multiplication ROUNDS to the nearest representable double
/// (ulp is 2^9 at the top of the range, so the value moves by up to
/// ~256) and only *then* is it truncated: JS `& 0x7fffffff` applies
/// `ToInt32` (mod 2^32, then the 31-bit mask) to the **already-rounded**
/// double. So the TS result is
/// `trunc(round64(h * 1103515245 + 12345)) mod 2^31`, which is a
/// uint32-truncation of a rounded double - not the low 31 bits of the
/// exact integer product. The two disagree on essentially every step
/// (for seed "parity", all 8 steps differ), which is why the native
/// engine used to derive a different game id (`pyr6b8nU` vs TS
/// `QqkIuyke`) and therefore seeded `PseudoRandom` differently, giving
/// different bot spawn tiles from the first decision tick on.
///
/// `f64`s are the same IEEE-754 doubles JS uses, `f64 %` is an exact
/// `fmod`, and every intermediate here is an integer-valued double, so
/// the port is exact.
///
/// Note on the input hash: `simple_hash` is `Math.abs` of a wrapping-i32
/// djb2-fold, i.e. a value in `0 ..= 2^31`. The `i64` widening below
/// reproduces `Math.abs(-2^31) == 2^31` for the one input where
/// `i32::abs` would otherwise disagree (and panic in a debug build).
pub fn seed_to_game_id(seed: &str) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let h0 = simple_hash(&format!("rl-{seed}")) as i64;
    let mut h = if h0 < 0 { (-h0) as f64 } else { h0 as f64 };
    let mut out = String::with_capacity(8);
    for _ in 0..8 {
        let d = h * 1_103_515_245.0 + 12_345.0; // rounds to nearest f64
        let masked = (d % 4_294_967_296.0) as u32 & 0x7fff_ffff; // ToInt32, then mask
        h = masked as f64;
        out.push(ALPHABET[(masked as usize) % ALPHABET.len()] as char);
    }
    out
}
