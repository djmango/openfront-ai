//! RL env session - TS engine via multiplexed daemon (default) or per-process bridge.

use crate::backend::bridge::BridgeClient;
use crate::backend::daemon::{use_daemon, DaemonSession};
use crate::backend::stub::StubSession;
use crate::record::StampedIntent;
use crate::util::simple_hash;
use serde_json::Value;
use std::path::Path;

pub const AGENT_CLIENT_ID: &str = "AGENTRL1";

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

pub(crate) fn seed_to_game_id(seed: &str) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    let mut h = simple_hash(&format!("rl-{seed}")) as u32;
    let mut out = String::with_capacity(8);
    for _ in 0..8 {
        h = h.wrapping_mul(1_103_515_245).wrapping_add(12_345) & 0x7fff_ffff;
        out.push(ALPHABET[(h as usize) % ALPHABET.len()] as char);
    }
    out
}
