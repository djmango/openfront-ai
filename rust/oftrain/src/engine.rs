//! Engine abstraction: the exact surface `EnvWorker` needs from a game
//! simulation, so the trainer can run against either the Node.js engine
//! (JSON-over-pipes subprocess, `bridge.rs`) or the in-process native Rust
//! port (`openfront-engine` crate from the `rust-ofrs-fast` branch) once
//! its parity gates pass. Heads stay `serde_json::Value` on purpose: both
//! backends already produce that shape, so swapping engines can't be
//! conflated with an obs refactor.

use anyhow::Result;
use serde_json::Value;

pub const OWNER_MASK: u16 = 0x0FFF;
pub const FALLOUT_BIT: u16 = 13;
pub const DEFENSE_BONUS_BIT: u16 = 14;

/// Decoded observation: `head` is the raw JSON header (tick, width,
/// height, spawnPhase, winner, me, alive, entities, legal, wasted), plus
/// derived tile arrays split out of the binary tile-state frame.
pub struct RawObs {
    pub head: Value,
    pub owners: Vec<u16>, // raw small-ids, NOT slotted
    pub fallout: Vec<u8>,
    pub defense_bonus: Vec<u8>,
}

impl RawObs {
    pub fn tick(&self) -> i64 {
        self.head["tick"].as_i64().unwrap_or(0)
    }
    pub fn spawn_phase(&self) -> bool {
        self.head["spawnPhase"].as_bool().unwrap_or(false)
    }
    pub fn me(&self) -> i64 {
        self.head["me"].as_i64().unwrap_or(-1)
    }
    pub fn alive(&self) -> bool {
        self.head["alive"].as_bool().unwrap_or(false)
    }
    pub fn winner(&self) -> &Value {
        &self.head["winner"]
    }
    pub fn wasted(&self) -> i64 {
        self.head["wasted"].as_i64().unwrap_or(0)
    }
    pub fn legal_actions(&self) -> &Value {
        &self.head["legal"]["actions"]
    }
    pub fn entities(&self) -> &Value {
        &self.head["entities"]
    }
}

/// Splits a raw little-endian u16 tile-state buffer into the owner /
/// fallout / defense-bonus planes. Shared by every backend.
pub fn decode_tiles(tiles_raw: &[u8], n: usize) -> (Vec<u16>, Vec<u8>, Vec<u8>) {
    debug_assert_eq!(tiles_raw.len(), n * 2);
    let mut owners = vec![0u16; n];
    let mut fallout = vec![0u8; n];
    let mut defense_bonus = vec![0u8; n];
    for i in 0..n {
        let state = u16::from_le_bytes([tiles_raw[i * 2], tiles_raw[i * 2 + 1]]);
        owners[i] = state & OWNER_MASK;
        fallout[i] = ((state >> FALLOUT_BIT) & 1) as u8;
        defense_bonus[i] = ((state >> DEFENSE_BONUS_BIT) & 1) as u8;
    }
    (owners, fallout, defense_bonus)
}

/// Everything `EnvWorker` needs from a game simulation. `width`/`height`/
/// `terrain` are only valid after the first `reset()`.
pub trait GameEngine: Send {
    fn reset(
        &mut self,
        map_name: &str,
        seed: &str,
        bots: u32,
        difficulty: &str,
        nations: Value,
    ) -> Result<RawObs>;

    fn step(&mut self, intents: &[Value], ticks: u32) -> Result<RawObs>;

    fn width(&self) -> usize;
    fn height(&self) -> usize;
    /// Untrimmed raw terrain bytes (height x width), set on reset().
    fn terrain(&self) -> &[u8];

    fn close(&mut self);
}

/// Which simulation backend to run envs against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EngineKind {
    /// Node.js engine via a JSONL subprocess per env (`bridge.rs`).
    Node,
    /// In-process native Rust engine (`openfront-engine` crate). Requires
    /// the `native-engine` cargo feature and passing parity gates.
    Native,
}

impl std::str::FromStr for EngineKind {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "node" => Ok(EngineKind::Node),
            "native" => Ok(EngineKind::Native),
            other => Err(format!("unknown engine {other:?} (expected: node, native)")),
        }
    }
}

pub fn create(kind: EngineKind) -> Result<Box<dyn GameEngine>> {
    match kind {
        EngineKind::Node => Ok(Box::new(crate::bridge::Bridge::spawn()?)),
        EngineKind::Native => {
            #[cfg(feature = "native-engine")]
            {
                Ok(Box::new(crate::native::NativeEngine::new()?))
            }
            #[cfg(not(feature = "native-engine"))]
            {
                anyhow::bail!(
                    "--engine native requires building with `--features native-engine`"
                )
            }
        }
    }
}
