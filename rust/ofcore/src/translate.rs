//! choice -> engine intent JSON. Port of `rl/ppo_translate.py::IntentTranslator`.
//!
//! Region pointers snap to a tile inside the REGION x REGION block that
//! passes the same cheap validity checks the engine runs at execution
//! (ownership, passability, shore for ports), so region picks rarely
//! become silently-discarded intents.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};

use crate::feat::{
    self, AttackE, EntsData, Legal, ACTIONS, BUILD_TYPES, GW_MAX, IMPASSABLE_MAGNITUDE,
    NUKE_TYPES, REGION,
};

const SHORELINE_BIT: u32 = 6;

pub struct IntentTranslator {
    pub land: Vec<u8>,     // (hr, wr) 0/1, trimmed to multiples of REGION
    pub mag: Vec<u8>,
    pub shore: Vec<u8>,
    pub passable: Vec<bool>,
    pub hr: usize,
    pub wr: usize,
    pub width: usize, // untrimmed env width: tile flat index stride
    pub gh: usize,
    pub gw: usize,
    rng: SmallRng,
}

impl IntentTranslator {
    /// `terrain` is the raw (height, width) terrain byte grid from
    /// `bridge/env.ts` (before REGION trimming); `hr`/`wr` are the
    /// trimmed dims (`ObsBuilder.start_game`), `width` is the untrimmed
    /// env width used for the flat tile-index stride.
    pub fn new(terrain: &[u8], width: usize, hr: usize, wr: usize) -> Self {
        let mut land = vec![0u8; hr * wr];
        let mut mag = vec![0u8; hr * wr];
        let mut shore = vec![0u8; hr * wr];
        for y in 0..hr {
            for x in 0..wr {
                let t = terrain[y * width + x];
                let i = y * wr + x;
                land[i] = (t >> feat::IS_LAND_BIT) & 1;
                mag[i] = t & feat::MAG_MASK;
                shore[i] = (t >> SHORELINE_BIT) & 1;
            }
        }
        let passable: Vec<bool> = (0..hr * wr)
            .map(|i| land[i] == 1 && mag[i] < IMPASSABLE_MAGNITUDE)
            .collect();
        IntentTranslator {
            land,
            mag,
            shore,
            passable,
            hr,
            wr,
            width,
            gh: hr / REGION,
            gw: wr / REGION,
            rng: SmallRng::seed_from_u64(0),
        }
    }

    /// Random valid tile inside `region`'s REGION x REGION block (or the
    /// 2x2 super-block when span=2), filtered by `valid`. Returns the flat
    /// tile index (`y * width + x`) in untrimmed env coordinates.
    fn region_tile(&mut self, region: i64, valid: &[bool], span: usize) -> Option<i64> {
        let stride = GW_MAX.max(self.gw as i64);
        let (gy, gx) = (region / stride, region % stride);
        if gy < 0 || gx < 0 || gy as usize >= self.gh || gx as usize >= self.gw {
            return None;
        }
        let (gy, gx) = (gy as usize, gx as usize);
        let mut cands: Vec<(usize, usize)> = Vec::new();
        for y in gy * REGION..((gy + span) * REGION).min(self.hr) {
            for x in gx * REGION..((gx + span) * REGION).min(self.wr) {
                if valid[y * self.wr + x] {
                    cands.push((y, x));
                }
            }
        }
        if cands.is_empty() {
            return None;
        }
        let (y, x) = cands[self.rng.gen_range(0..cands.len())];
        Some(y as i64 * self.width as i64 + x as i64)
    }

    fn owners_valid(&self, owners: &[i64], pred: impl Fn(i64) -> bool) -> Vec<bool> {
        owners.iter().map(|&o| pred(o)).collect()
    }

    pub fn spawn_tile(&mut self, region: i64, owners: &[i64]) -> Option<i64> {
        let passable = self.passable.clone();
        let valid: Vec<bool> = (0..owners.len()).map(|i| passable[i] && owners[i] == 0).collect();
        self.region_tile(region, &valid, 1)
    }

    pub fn boat_tile(&mut self, region: i64, owners: &[i64], me: i64, ents: &EntsData) -> Option<i64> {
        let mut valid: Vec<bool> = self.owners_valid(owners, |o| o != me);
        for a in &ents.alliances {
            let (pa, pb) = (a.0 as i64, a.1 as i64);
            let ally = if pa == me { Some(pb) } else if pb == me { Some(pa) } else { None };
            if let Some(ally) = ally {
                for (i, &o) in owners.iter().enumerate() {
                    if o == ally {
                        valid[i] = false;
                    }
                }
            }
        }
        let passable = &self.passable;
        let shore_valid: Vec<bool> =
            valid.iter().enumerate().map(|(i, &v)| v && passable[i] && self.shore[i] == 1).collect();
        let plain_valid: Vec<bool> =
            valid.iter().enumerate().map(|(i, &v)| v && passable[i]).collect();
        self.region_tile(region, &shore_valid, 2).or_else(|| self.region_tile(region, &plain_valid, 2))
    }

    pub fn build_tile(&mut self, region: i64, owners: &[i64], me: i64, unit: &str) -> Option<i64> {
        if unit == "Warship" {
            let water: Vec<bool> = self.land.iter().map(|&l| l == 0).collect();
            return self.region_tile(region, &water, 1);
        }
        let passable = &self.passable;
        let mut valid: Vec<bool> =
            (0..owners.len()).map(|i| passable[i] && owners[i] == me).collect();
        if unit == "Port" {
            for (i, v) in valid.iter_mut().enumerate() {
                *v = *v && self.shore[i] == 1;
            }
        }
        self.region_tile(region, &valid, 1)
    }

    pub fn nuke_tile(&mut self, region: i64) -> Option<i64> {
        let passable = self.passable.clone();
        self.region_tile(region, &passable, 2)
    }

    pub fn move_warship_tile(&mut self, region: i64) -> Option<i64> {
        let water: Vec<bool> = self.land.iter().map(|&l| l == 0).collect();
        self.region_tile(region, &water, 2)
    }

    /// (y, x) tile coordinates of the region's center (untrimmed coords).
    pub fn region_center(&self, region: i64) -> (i64, i64) {
        let stride = GW_MAX.max(self.gw as i64);
        let (gy, gx) = (region / stride, region % stride);
        (gy * REGION as i64 + (REGION / 2) as i64, gx * REGION as i64 + (REGION / 2) as i64)
    }

    /// Own unit closest to the region center, restricted to `ids` (engine
    /// uids from the bridge legality lists).
    pub fn nearest_own_unit<'a>(
        &self,
        region: i64,
        ents: &'a EntsData,
        me: i64,
        ids: &[usize],
    ) -> Option<&'a feat::UnitE> {
        let (cy, cx) = self.region_center(region);
        ents.units
            .iter()
            .filter(|u| u.owner as i64 == me && ids.contains(&(u.uid.max(0) as usize)))
            .min_by_key(|u| (u.y - cy).pow(2) + (u.x - cx).pow(2))
    }
}

/// Slot -> persistent player ID (intents reference players by pid).
/// `lut` is the small-id -> slot table (see `feat::make_lut`).
pub fn slot_to_pid(slot: i64, lut: &[u8], players: &[feat::PlayerE]) -> Option<String> {
    if slot < 0 {
        return None;
    }
    let small = (0..lut.len()).find(|&id| lut[id] as i64 == slot)?;
    players.iter().find(|p| p.id == small).map(|p| p.pid.clone())
}

pub struct Choice {
    pub action: i64,
    pub player_slot: Option<i64>,
    pub tile_region: Option<i64>,
    pub build_type: Option<i64>,
    pub nuke_type: Option<i64>,
    pub quantity_frac: Option<f64>,
}

/// `owners` is the RAW (unslotted) small-id owner grid, trimmed to (hr, wr)
/// - distinct from `feat::featurize`'s slotted `owners` input.
pub fn translate(
    choice: &Choice,
    tr: &mut IntentTranslator,
    owners: &[i64],
    me: i64,
    ents: &EntsData,
    legal: &Legal,
    lut: &[u8],
) -> Vec<Value> {
    let name = ACTIONS[choice.action as usize];
    let troops = legal.troops;
    let frac = choice.quantity_frac.unwrap_or(0.25).clamp(0.01, 1.0);

    match name {
        "noop" => vec![],
        "spawn" => match choice.tile_region.and_then(|r| tr.spawn_tile(r, owners)) {
            Some(tile) => vec![json!({"type": "spawn", "tile": tile})],
            None => vec![],
        },
        "expand" => vec![json!({"type": "attack", "targetID": null, "troops": (troops * frac) as i64})],
        "attack" => match choice.player_slot.and_then(|s| slot_to_pid(s, lut, &ents.players)) {
            Some(pid) => vec![json!({"type": "attack", "targetID": pid, "troops": (troops * frac) as i64})],
            None => vec![],
        },
        "boat" => match choice.tile_region.and_then(|r| tr.boat_tile(r, owners, me, ents)) {
            Some(tile) => vec![json!({"type": "boat", "dst": tile, "troops": (troops * frac) as i64})],
            None => vec![],
        },
        "build" => {
            let bt = choice.build_type.unwrap_or(-1);
            if bt < 0 || bt as usize >= BUILD_TYPES.len() {
                return vec![];
            }
            let unit = BUILD_TYPES[bt as usize];
            match choice.tile_region.and_then(|r| tr.build_tile(r, owners, me, unit)) {
                Some(tile) => vec![json!({"type": "build_unit", "unit": unit, "tile": tile})],
                None => vec![],
            }
        }
        "launch_nuke" => {
            let nt = choice.nuke_type.unwrap_or(-1);
            if nt < 0 || nt as usize >= NUKE_TYPES.len() {
                return vec![];
            }
            match choice.tile_region.and_then(|r| tr.nuke_tile(r)) {
                Some(tile) => {
                    let (unit, up) = NUKE_TYPES[nt as usize];
                    let mut intent = json!({"type": "build_unit", "unit": unit, "tile": tile});
                    if let Some(up) = up {
                        intent["rocketDirectionUp"] = json!(up);
                    }
                    vec![intent]
                }
                None => vec![],
            }
        }
        "retreat" => {
            if legal.attacks.is_empty() {
                return vec![];
            }
            let slot = choice.player_slot.unwrap_or(-1);
            let mut aid = legal.attacks.last().cloned().unwrap_or_default();
            let matches: Vec<&AttackE> = ents
                .attacks
                .iter()
                .filter(|a| {
                    a.from as i64 == me
                        && !a.retreating
                        && (if a.to != 0 { lut[a.to] as i64 } else { 0 }) == slot
                })
                .collect();
            if let Some(m) = matches.last() {
                aid = m.aid.clone();
            }
            vec![json!({"type": "cancel_attack", "attackID": aid})]
        }
        "upgrade_structure" => match choice
            .tile_region
            .and_then(|r| tr.nearest_own_unit(r, ents, me, &legal.upgradable))
        {
            Some(u) => {
                let ty = static_type_name(u.class);
                vec![json!({"type": "upgrade_structure", "unit": ty, "unitId": u.uid})]
            }
            None => vec![],
        },
        "move_warship" => {
            if legal.warships.is_empty() {
                return vec![];
            }
            match choice.tile_region.and_then(|r| tr.move_warship_tile(r)) {
                Some(tile) => vec![json!({"type": "move_warship", "unitIds": legal.warships, "tile": tile})],
                None => vec![],
            }
        }
        "cancel_boat" => match choice.tile_region.and_then(|r| tr.nearest_own_unit(r, ents, me, &legal.boats)) {
            Some(u) => vec![json!({"type": "cancel_boat", "unitID": u.uid})],
            None => vec![],
        },
        "delete_unit" => match choice.tile_region.and_then(|r| tr.nearest_own_unit(r, ents, me, &legal.deletable)) {
            Some(u) => vec![json!({"type": "delete_unit", "unitId": u.uid})],
            None => vec![],
        },
        _ => {
            let pid = match choice.player_slot.and_then(|s| slot_to_pid(s, lut, &ents.players)) {
                Some(p) => p,
                None => return vec![],
            };
            match name {
                "alliance_request" => vec![json!({"type": "allianceRequest", "recipient": pid})],
                "alliance_reject" => vec![json!({"type": "allianceReject", "requestor": pid})],
                "break_alliance" => vec![json!({"type": "breakAlliance", "recipient": pid})],
                "donate_gold" => {
                    let gold = (legal.gold * frac) as i64;
                    vec![json!({"type": "donate_gold", "recipient": pid, "gold": gold})]
                }
                "donate_troops" => {
                    vec![json!({"type": "donate_troops", "recipient": pid, "troops": (troops * frac) as i64})]
                }
                "embargo" => vec![json!({"type": "embargo", "targetID": pid, "action": "start"})],
                "embargo_stop" => vec![json!({"type": "embargo", "targetID": pid, "action": "stop"})],
                "target_player" => vec![json!({"type": "targetPlayer", "target": pid})],
                "alliance_extension" => vec![json!({"type": "allianceExtension", "recipient": pid})],
                _ => vec![],
            }
        }
    }
}

fn static_type_name(class: usize) -> &'static str {
    match class {
        0 => "City",
        1 => "Port",
        2 => "Defense Post",
        3 => "Missile Silo",
        4 => "SAM Launcher",
        5 => "Factory",
        _ => "City",
    }
}

pub fn my_tiles(ents: &EntsData, me: i64) -> f64 {
    ents.players.iter().find(|p| p.id as i64 == me).map(|p| p.tiles).unwrap_or(0.0)
}
