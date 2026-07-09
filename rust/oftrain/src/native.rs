//! In-process native engine backend (`--engine native`): drives
//! `openfront_engine::rl::RlSession` directly - no subprocess, no pipes, no
//! JSON serialization of tile frames. Requires the `native-engine` feature
//! (path dependency on the `rust-ofrs-fast` worktree's engine crate) and is
//! gated on the obs-diff parity harness before being trusted for training -
//! see rust/DEVLOG.md.

use anyhow::{anyhow, Result};
use serde_json::Value;
use std::path::PathBuf;

use crate::engine::{decode_tiles, GameEngine, RawObs};
use openfront_engine::rl::RlSession;

pub struct NativeEngine {
    session: Option<RlSession>,
    repo_root: PathBuf,
    width: usize,
    height: usize,
    terrain: Vec<u8>,
}

impl NativeEngine {
    pub fn new() -> Result<Self> {
        Ok(NativeEngine {
            session: None,
            repo_root: crate::bridge::repo_root()?,
            width: 0,
            height: 0,
            terrain: Vec::new(),
        })
    }

    fn decode(&self, head: Value, tiles_raw: Vec<u8>) -> RawObs {
        let n = self.width * self.height;
        let (owners, fallout, defense_bonus) = decode_tiles(&tiles_raw, n);
        RawObs { head, owners, fallout, defense_bonus }
    }
}

impl GameEngine for NativeEngine {
    fn reset(
        &mut self,
        map_name: &str,
        seed: &str,
        bots: u32,
        difficulty: &str,
        nations: Value,
    ) -> Result<RawObs> {
        let (session, head, terrain, tiles) =
            RlSession::reset(&self.repo_root, map_name, seed, bots, difficulty, nations)
                .map_err(|e| anyhow!("native reset: {e}"))?;
        self.width = head["width"].as_u64().ok_or_else(|| anyhow!("no width"))? as usize;
        self.height = head["height"].as_u64().ok_or_else(|| anyhow!("no height"))? as usize;
        self.terrain = terrain;
        self.session = Some(session);
        Ok(self.decode(head, tiles))
    }

    fn step(&mut self, intents: &[Value], ticks: u32) -> Result<RawObs> {
        let session = self.session.as_mut().ok_or_else(|| anyhow!("step before reset"))?;
        let (head, tiles) = session.step(intents, ticks);
        Ok(self.decode(head, tiles))
    }

    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn terrain(&self) -> &[u8] {
        &self.terrain
    }

    fn close(&mut self) {
        self.session = None;
    }
}
