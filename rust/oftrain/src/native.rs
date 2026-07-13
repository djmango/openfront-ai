//! In-process native engine backend (`--engine native`): drives
//! `openfront_engine::rl::RlSession` directly - no subprocess, no pipes, no
//! JSON serialization of tile frames. Requires the `native-engine` feature
//! (path dependency on the `rust-ofrs-fast` worktree's engine crate) and is
//! gated on the obs-diff parity harness before being trusted for training -
//! see rust/DEVLOG.md.

use anyhow::{Result, anyhow};
use serde_json::Value;
use std::path::PathBuf;

use crate::engine::{GameEngine, RawObs, TileState};
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

    /// Copies the session's packed tile state without splitting it into three
    /// full-map planes. The worker decodes only the trimmed map while slotting
    /// owners, packing fallout, and pooling defense in one pass.
    /// Takes `width`/`height` explicitly (rather than `&self`) so callers
    /// can hold a live borrow of `self.session` while decoding.
    fn decode(width: usize, height: usize, head: Value, tiles: &[u16]) -> RawObs {
        let n = width * height;
        debug_assert_eq!(tiles.len(), n);
        RawObs {
            head,
            tiles: TileState::Packed(tiles.to_vec()),
        }
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
        let (session, head, terrain) =
            RlSession::reset(&self.repo_root, map_name, seed, bots, difficulty, nations)
                .map_err(|e| anyhow!("native reset: {e}"))?;
        self.width = head["width"].as_u64().ok_or_else(|| anyhow!("no width"))? as usize;
        self.height = head["height"]
            .as_u64()
            .ok_or_else(|| anyhow!("no height"))? as usize;
        self.terrain = terrain;
        self.session = Some(session);
        let (width, height) = (self.width, self.height);
        let session = self.session.as_ref().unwrap();
        Ok(Self::decode(width, height, head, session.tile_state()))
    }

    fn step(&mut self, intents: &[Value], ticks: u32) -> Result<RawObs> {
        let (width, height) = (self.width, self.height);
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| anyhow!("step before reset"))?;
        let head = session.step(intents, ticks);
        Ok(Self::decode(width, height, head, session.tile_state()))
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
