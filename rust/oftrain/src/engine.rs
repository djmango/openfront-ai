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

/// Tile state stays packed for the in-process engine.  Stream backends keep
/// their historical split representation so the Node fallback is unchanged.
pub enum TileState {
    Split {
        owners: Vec<u16>,
        fallout: Vec<u8>,
        defense_bonus: Vec<u8>,
    },
    Packed(Vec<u16>),
}

/// Observation header plus backend-appropriate tile storage.
pub struct RawObs {
    pub head: Value,
    pub tiles: TileState,
    /// Native engine fills this to skip JSON `parse_ents` / `parse_legal`.
    /// Node / daemon backends leave it `None`.
    pub structured: Option<(ofcore::feat::EntsData, ofcore::feat::Legal)>,
}

pub struct PreparedTileState {
    pub owners_slotted: Vec<u8>,
    pub fallout_packed: Vec<u8>,
    pub db: Vec<f32>,
    /// Pooled ego fractions at /`region` when built via
    /// [`RawObs::prepare_tiles_with_ego`]; empty otherwise.
    pub ego: Vec<f32>,
    /// Own-territory centroid used by local crop. Meaningful when `ego`
    /// is non-empty; otherwise `(hr/2, wr/2)`.
    pub center: (f64, f64),
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

    #[inline]
    pub fn owner_at(&self, i: usize) -> u16 {
        match &self.tiles {
            TileState::Split { owners, .. } => owners[i],
            TileState::Packed(tiles) => tiles[i] & OWNER_MASK,
        }
    }

    #[inline]
    pub fn defense_bonus_at(&self, i: usize) -> u8 {
        match &self.tiles {
            TileState::Split { defense_bonus, .. } => defense_bonus[i],
            TileState::Packed(tiles) => ((tiles[i] >> DEFENSE_BONUS_BIT) & 1) as u8,
        }
    }

    /// Fuse trim, owner slotting, MSB-first fallout packing, and /`region`
    /// defense pooling. This is the only full tile-state scan needed before
    /// featurization, and packed native observations are decoded in place.
    pub fn prepare_tiles(
        &self,
        lut: &[u8],
        width: usize,
        hr: usize,
        wr: usize,
        region: usize,
    ) -> PreparedTileState {
        self.prepare_tiles_with_ego(lut, width, hr, wr, region, None)
    }

    /// Same as [`Self::prepare_tiles`], but also pools ego classes and the
    /// own-territory centroid in the same `hr*wr` walk when `clut` is set.
    /// Removes the separate [`ofcore::feat::pool_ego_and_center`] scan.
    pub fn prepare_tiles_with_ego(
        &self,
        lut: &[u8],
        width: usize,
        hr: usize,
        wr: usize,
        region: usize,
        clut: Option<&[u8; ofcore::feat::MAX_SLOTS]>,
    ) -> PreparedTileState {
        assert!(region != 0 && hr % region == 0 && wr % region == 0);
        let packed_wr = wr.div_ceil(8);
        let gw = wr / region;
        let gh = hr / region;
        let plane = gh * gw;
        let mut owners_slotted = vec![0u8; hr * wr];
        let mut fallout_packed = vec![0u8; hr * packed_wr];
        let mut db_counts = vec![0u32; plane];
        let mut ego_counts = clut.map(|_| vec![0u32; 3 * plane]);
        let mut sum_y = 0.0f64;
        let mut sum_x = 0.0f64;
        let mut own_count = 0.0f64;

        for y in 0..hr {
            let src_row = y * width;
            let dst_row = y * wr;
            for x in 0..wr {
                let src = src_row + x;
                let dst = dst_row + x;
                let (owner, fallout, defense) = match &self.tiles {
                    TileState::Split {
                        owners,
                        fallout,
                        defense_bonus,
                    } => (owners[src], fallout[src], defense_bonus[src]),
                    TileState::Packed(tiles) => {
                        let state = tiles[src];
                        (
                            state & OWNER_MASK,
                            ((state >> FALLOUT_BIT) & 1) as u8,
                            ((state >> DEFENSE_BONUS_BIT) & 1) as u8,
                        )
                    }
                };
                let slotted = lut.get(owner as usize).copied().unwrap_or(0);
                owners_slotted[dst] = slotted;
                if fallout != 0 {
                    fallout_packed[y * packed_wr + x / 8] |= 1 << (7 - x % 8);
                }
                let region_at = (y / region) * gw + x / region;
                db_counts[region_at] += defense as u32;
                if let (Some(clut), Some(counts)) = (clut, ego_counts.as_mut()) {
                    let cls = clut[slotted as usize];
                    if cls > 0 {
                        counts[(cls - 1) as usize * plane + region_at] += 1;
                    }
                    if cls == 1 {
                        sum_y += y as f64;
                        sum_x += x as f64;
                        own_count += 1.0;
                    }
                }
            }
        }

        let norm = (region * region) as f32;
        let db = db_counts
            .into_iter()
            .map(|count| count as f32 / norm)
            .collect();
        let (ego, center) = if let Some(counts) = ego_counts {
            let ego = counts
                .into_iter()
                .map(|count| count as f32 / norm)
                .collect();
            let center = if own_count > 0.0 {
                (sum_y / own_count, sum_x / own_count)
            } else {
                (hr as f64 / 2.0, wr as f64 / 2.0)
            };
            (ego, center)
        } else {
            (Vec::new(), (hr as f64 / 2.0, wr as f64 / 2.0))
        };
        PreparedTileState {
            owners_slotted,
            fallout_packed,
            db,
            ego,
            center,
        }
    }
}

/// Splits a raw little-endian u16 tile-state buffer into the owner /
/// fallout / defense-bonus planes. Shared by every backend that has to
/// cross a byte-stream boundary to get tile state (the Node bridge and
/// daemon backends both hand over a raw wire frame here).
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

/// Same split as [`decode_tiles`], but for backends that never left
/// process (the native engine): decodes straight from the packed `&[u16]`
/// state buffer, skipping both the little-endian byte encode on the
/// engine side and the byte-pair reassembly this function does. Same bit
/// layout, so this is the exact same field extraction with one fewer
/// (redundant, in-process) serialization round trip.
#[cfg(test)]
pub fn decode_tiles_u16(state: &[u16], n: usize) -> (Vec<u16>, Vec<u8>, Vec<u8>) {
    debug_assert_eq!(state.len(), n);
    let mut owners = vec![0u16; n];
    let mut fallout = vec![0u8; n];
    let mut defense_bonus = vec![0u8; n];
    for i in 0..n {
        let s = state[i];
        owners[i] = s & OWNER_MASK;
        fallout[i] = ((s >> FALLOUT_BIT) & 1) as u8;
        defense_bonus[i] = ((s >> DEFENSE_BONUS_BIT) & 1) as u8;
    }
    (owners, fallout, defense_bonus)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ae;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn split_obs(state: &[u16]) -> RawObs {
        let (owners, fallout, defense_bonus) = decode_tiles_u16(state, state.len());
        RawObs {
            head: Value::Null,
            tiles: TileState::Split {
                owners,
                fallout,
                defense_bonus,
            },
            structured: None,
        }
    }

    fn packed_obs(state: &[u16]) -> RawObs {
        RawObs {
            head: Value::Null,
            tiles: TileState::Packed(state.to_vec()),
            structured: None,
        }
    }

    #[test]
    fn packed_prepare_exhaustively_matches_split_tile_semantics() {
        // Every owner id and every fallout/defense combination appears once.
        // Extra width exercises row stride and right/bottom trim behavior.
        let (hr, wr, width, region) = (128, 128, 131, 8);
        let mut state = vec![0xFFFF; hr * width];
        for owner in 0..=OWNER_MASK {
            for flags in 0..4 {
                let dst = owner as usize * 4 + flags;
                state[(dst / wr) * width + dst % wr] = owner
                    | (((flags & 1) as u16) << FALLOUT_BIT)
                    | (((flags >> 1) as u16) << DEFENSE_BONUS_BIT);
            }
        }
        let lut: Vec<u8> = (0..=OWNER_MASK)
            .map(|id| ((id as usize * 37) % 128) as u8)
            .collect();
        let old = split_obs(&state).prepare_tiles(&lut, width, hr, wr, region);
        let new = packed_obs(&state).prepare_tiles(&lut, width, hr, wr, region);
        assert_eq!(new.owners_slotted, old.owners_slotted);
        assert_eq!(new.fallout_packed, old.fallout_packed);
        assert_eq!(new.db, old.db);
    }

    #[test]
    fn packed_prepare_randomized_parity_across_luts_and_flags() {
        let mut rng = SmallRng::seed_from_u64(0x5eed_fa11_0def);
        for _ in 0..200 {
            let region = ofcore::feat::REGION;
            let hr = region * rng.gen_range(1..=5);
            let wr = region * rng.gen_range(1..=7);
            let width = wr + rng.gen_range(0..=5);
            let height = hr + rng.gen_range(0..=3);
            let state: Vec<u16> = (0..width * height).map(|_| rng.r#gen()).collect();
            let lut_len = rng.gen_range(0..=4096);
            let lut: Vec<u8> = (0..lut_len).map(|_| rng.gen_range(0..128)).collect();

            let split = split_obs(&state);
            let packed = packed_obs(&state);
            let new = packed.prepare_tiles(&lut, width, hr, wr, region);

            // Reproduce the old prepare path independently: trim three split
            // planes, then scan fallout and owner/defense again downstream.
            let TileState::Split {
                owners,
                fallout,
                defense_bonus,
            } = &split.tiles
            else {
                unreachable!()
            };
            let mut old_owners = vec![0u8; hr * wr];
            let mut old_fallout = vec![0u8; hr * wr];
            let mut old_defense = vec![0u8; hr * wr];
            for y in 0..hr {
                for x in 0..wr {
                    let src = y * width + x;
                    let dst = y * wr + x;
                    old_owners[dst] = lut.get(owners[src] as usize).copied().unwrap_or(0);
                    old_fallout[dst] = fallout[src];
                    old_defense[dst] = defense_bonus[src];
                }
            }
            assert_eq!(new.owners_slotted, old_owners);
            assert_eq!(new.fallout_packed, ae::pack_fallout(&old_fallout, hr, wr));

            let mut clut = [0u8; ofcore::feat::MAX_SLOTS];
            for cls in &mut clut[1..] {
                *cls = rng.gen_range(1..=3);
            }
            let land: Vec<u8> = (0..hr * wr).map(|_| rng.gen_range(0..=1)).collect();
            let (old_ego, old_db) =
                ofcore::feat::pool_ego_db(&old_owners, &clut, &old_defense, hr, wr);
            let (new_ego, center) =
                ofcore::feat::pool_ego_and_center(&new.owners_slotted, &clut, hr, wr);
            assert_eq!(new_ego, old_ego);
            assert_eq!(new.db, old_db);
            let fused = packed.prepare_tiles_with_ego(&lut, width, hr, wr, region, Some(&clut));
            assert_eq!(fused.owners_slotted, new.owners_slotted);
            assert_eq!(fused.fallout_packed, new.fallout_packed);
            assert_eq!(fused.db, new.db);
            assert_eq!(fused.ego, new_ego);
            assert_eq!(fused.center, center);
            let local = 16.min(hr).min(wr);
            let old_local =
                ofcore::feat::local_crop(&old_owners, &clut, &land, &old_defense, hr, wr, local);
            let new_local = ofcore::feat::local_crop_at_with_defense(
                &new.owners_slotted,
                &clut,
                &land,
                hr,
                wr,
                local,
                center,
                |i| {
                    let y = i / wr;
                    let x = i % wr;
                    packed.defense_bonus_at(y * width + x)
                },
            );
            assert_eq!(new_local, old_local);
            for i in 0..state.len() {
                assert_eq!(packed.owner_at(i), split.owner_at(i));
                assert_eq!(packed.defense_bonus_at(i), split.defense_bonus_at(i));
            }
        }
    }
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

    /// Persist a client GameRecord JSON (Node bridge + native engine).
    fn save_record(&mut self, _path: &str) -> Result<serde_json::Value> {
        anyhow::bail!("save_record is not supported on this engine backend")
    }
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
                anyhow::bail!("--engine native requires building with `--features native-engine`")
            }
        }
    }
}
