//! Featurization core: a v7 Rust port of `rl/obs.py::ObsBuilder.prepare()`
//! (small-state, numpy-only side). The full-resolution GPU work (AE
//! encode, ego pooling, fine/coarse foveation crop, local crop) lives in
//! `oftrain::ae` since it needs tensors, not this crate.

use serde_json::Value;

pub const REGION: usize = 8;
pub const MAX_SLOTS: usize = 128;
pub const N_STATIC: usize = 6;
pub const N_TRANSIENT: usize = 53;
pub const P_FEAT: usize = 21;
pub const N_SCALARS: usize = 11;
pub const N_ACTIONS: usize = 21;
pub const N_BUILD: usize = 7;
pub const N_NUKE: usize = 5;
pub const GW_MAX: i64 = 250;
pub const GH_MAX: i64 = 150;
pub const IS_LAND_BIT: u8 = 7;
pub const MAG_MASK: u8 = 0x1F;
pub const IMPASSABLE_MAGNITUDE: u8 = 31;

pub const A_NOOP: i64 = 0;
pub const A_ATTACK: i64 = 1;
pub const A_EXPAND: i64 = 2;
pub const A_BOAT: i64 = 3;
pub const A_BUILD: i64 = 4;
pub const A_LAUNCH_NUKE: i64 = 5;
pub const A_ALLIANCE_REQUEST: i64 = 6;
pub const A_ALLIANCE_REJECT: i64 = 7;
pub const A_BREAK_ALLIANCE: i64 = 8;
pub const A_DONATE_GOLD: i64 = 9;
pub const A_DONATE_TROOPS: i64 = 10;
pub const A_EMBARGO: i64 = 11;
pub const A_RETREAT: i64 = 12;
pub const A_SPAWN: i64 = 13;
pub const A_UPGRADE_STRUCTURE: i64 = 14;
pub const A_MOVE_WARSHIP: i64 = 15;
pub const A_CANCEL_BOAT: i64 = 16;
pub const A_DELETE_UNIT: i64 = 17;
pub const A_EMBARGO_STOP: i64 = 18;
pub const A_TARGET_PLAYER: i64 = 19;
pub const A_ALLIANCE_EXTENSION: i64 = 20;

pub const ACTIONS: [&str; N_ACTIONS] = [
    "noop",
    "attack",
    "expand",
    "boat",
    "build",
    "launch_nuke",
    "alliance_request",
    "alliance_reject",
    "break_alliance",
    "donate_gold",
    "donate_troops",
    "embargo",
    "retreat",
    "spawn",
    "upgrade_structure",
    "move_warship",
    "cancel_boat",
    "delete_unit",
    "embargo_stop",
    "target_player",
    "alliance_extension",
];

pub const BUILD_TYPES: [&str; N_BUILD] = [
    "City",
    "Port",
    "Defense Post",
    "Missile Silo",
    "SAM Launcher",
    "Factory",
    "Warship",
];

/// (engine unit, rocketDirectionUp). MIRV ignores the arc flag.
pub const NUKE_TYPES: [(&str, Option<bool>); N_NUKE] = [
    ("Atom Bomb", Some(true)),
    ("Atom Bomb", Some(false)),
    ("Hydrogen Bomb", Some(true)),
    ("Hydrogen Bomb", Some(false)),
    ("MIRV", None),
];
pub const NUKE_UNITS: [&str; 3] = ["Atom Bomb", "Hydrogen Bomb", "MIRV"];

// Transient plane base offsets (rl/obs.py TR_*); each is an own/ally/enemy
// triplet (base + oc, oc in {0,1,2}) unless noted.
pub const TR_WARSHIP: usize = 0;
pub const TR_TRANSPORT: usize = 3;
pub const TR_TRANSPORT_DEST: usize = 6;
pub const TR_TRADE: usize = 9;
pub const TR_TRADE_DEST: usize = 12;
pub const TR_NUKE: usize = 15;
pub const TR_NUKE_IMPACT: usize = 18;
pub const TR_NUKE_SAMLOCK: usize = 21;
pub const TR_CONSTRUCTION: usize = 24;
pub const TR_SAM_MISSILE: usize = 27;
pub const TR_SAM_MISSILE_IMPACT: usize = 30;
pub const TR_MIRV_WARHEAD: usize = 33;
pub const TR_MIRV_WARHEAD_IMPACT: usize = 36;
pub const TR_TRAIN: usize = 39;
pub const TR_SILO_COOLDOWN: usize = 42;
pub const TR_SAM_COOLDOWN: usize = 45;
pub const TR_STATION: usize = 48;
pub const TR_ATTACK_SRC: usize = 51; // shared across owners
pub const TR_ATTACK_RETREAT: usize = 52; // shared, overlays TR_ATTACK_SRC

pub fn action_index(name: &str) -> Option<i64> {
    ACTIONS.iter().position(|&a| a == name).map(|i| i as i64)
}

pub fn needs_player(a: i64) -> bool {
    matches!(
        a,
        A_ATTACK
            | A_ALLIANCE_REQUEST
            | A_ALLIANCE_REJECT
            | A_BREAK_ALLIANCE
            | A_DONATE_GOLD
            | A_DONATE_TROOPS
            | A_EMBARGO
            | A_RETREAT
            | A_EMBARGO_STOP
            | A_TARGET_PLAYER
            | A_ALLIANCE_EXTENSION
    )
}

pub fn needs_tile(a: i64) -> bool {
    matches!(
        a,
        A_BOAT
            | A_BUILD
            | A_LAUNCH_NUKE
            | A_SPAWN
            | A_UPGRADE_STRUCTURE
            | A_MOVE_WARSHIP
            | A_CANCEL_BOAT
            | A_DELETE_UNIT
    )
}

pub fn needs_quantity(a: i64) -> bool {
    matches!(
        a,
        A_ATTACK | A_EXPAND | A_BOAT | A_DONATE_GOLD | A_DONATE_TROOPS
    )
}

/// Actions whose tile pick refines to the fine /8 grid (see policy.rs).
pub fn refine_tile(a: i64) -> bool {
    matches!(
        a,
        A_SPAWN | A_BUILD | A_UPGRADE_STRUCTURE | A_CANCEL_BOAT | A_DELETE_UNIT
    )
}

pub fn build_index(unit: &str) -> Option<i64> {
    BUILD_TYPES
        .iter()
        .position(|&u| u == unit)
        .map(|i| i as i64)
}

pub fn nuke_index(unit: &str, up: bool) -> Option<i64> {
    Some(match (unit, up) {
        ("Atom Bomb", true) => 0,
        ("Atom Bomb", false) => 1,
        ("Hydrogen Bomb", true) => 2,
        ("Hydrogen Bomb", false) => 3,
        ("MIRV", _) => 4,
        _ => return None,
    })
}

/// UNIT_CLASSES order from ofae / former ae/units.py (v7: +SAMMissile, MIRV Warhead,
/// Train). The first N_STATIC classes are the static structures, and
/// their class index IS their position in the static-plane array.
pub fn unit_class(ty: &str) -> Option<usize> {
    Some(match ty {
        "City" => 0,
        "Port" => 1,
        "Defense Post" => 2,
        "Missile Silo" => 3,
        "SAM Launcher" => 4,
        "Factory" => 5,
        "Warship" => 6,
        "Transport" => 7,
        "Trade Ship" => 8,
        "Atom Bomb" => 9,
        "Hydrogen Bomb" => 10,
        "MIRV" => 11,
        "SAMMissile" => 12,
        "MIRV Warhead" => 13,
        "Train" => 14,
        _ => return None,
    })
}

pub fn log_norm(x: f64) -> f32 {
    ((1.0 + x.max(0.0)).log10() / 8.0) as f32
}

pub fn quantity_frac(amt: Option<f64>, avail: f64) -> f64 {
    match amt {
        Some(a) if avail > 0.0 => (a / avail).clamp(0.0, 1.0),
        _ => 0.25,
    }
}

pub fn placement_bucket(p: f64) -> i64 {
    ((p * 8.0) as i64).min(7)
}

/// Tolerant number extraction: gold rides as a JSON string (bigint).
pub fn num(v: &Value) -> f64 {
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(0.0),
        Value::String(s) => s.parse().unwrap_or(0.0),
        Value::Bool(b) => *b as i64 as f64,
        _ => 0.0,
    }
}

fn id_list(v: &Value) -> Vec<usize> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|u| u as usize))
                .collect()
        })
        .unwrap_or_default()
}

#[derive(Clone, Debug)]
pub struct UnitE {
    pub class: usize,
    pub owner: usize,
    pub uid: i64,
    pub x: i64,
    pub y: i64,
    pub gy: i64,
    pub gx: i64,
    pub tgy: i64,
    pub tgx: i64,
    pub has_target: bool,
    pub constructing: bool,
    pub level: f64,
    pub health: Option<f64>,
    pub max_health: Option<f64>,
    pub troops: f64,
    pub sam_lock: bool,
    pub cooldown: bool,
    pub station: bool,
}

#[derive(Clone, Debug)]
pub struct PlayerE {
    pub id: usize,
    pub pid: String,
    pub troops: f64,
    pub gold: f64,
    pub tiles: f64,
    pub alive: bool,
    pub traitor: bool,
    pub embargoes: Vec<usize>,
    /// `(other_player_id, relation_score)` from engine obs (reward-only).
    pub relations: Vec<(usize, f64)>,
    pub reqs_in: usize,
    pub reqs_out: usize,
    pub targets: Vec<usize>,
    pub troop_income: f64,
    pub gold_income: f64,
    pub doomsday: bool,
    pub doomsday_ticks: f64,
}

fn relation_list(v: &Value) -> Vec<(usize, f64)> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|pair| {
                    let arr = pair.as_array()?;
                    let id = arr.first()?.as_u64()? as usize;
                    let score = arr.get(1).map(num).unwrap_or(0.0);
                    Some((id, score))
                })
                .collect()
        })
        .unwrap_or_default()
}

impl PlayerE {
    pub fn relation_to(&self, other: usize) -> f64 {
        self.relations
            .iter()
            .find(|(id, _)| *id == other)
            .map(|(_, v)| *v)
            .unwrap_or(0.0)
    }
}

#[derive(Clone, Debug)]
pub struct AttackE {
    pub aid: String,
    pub from: usize,
    pub to: usize, // 0 when null (terra-nullius expand)
    pub troops: f64,
    pub retreating: bool,
    pub src_x: Option<i64>,
    pub src_y: Option<i64>,
}

/// (a, b, expires_at_tick).
#[derive(Clone, Debug)]
pub struct AllianceE(pub usize, pub usize, pub i64);

#[derive(Clone, Debug, Default)]
pub struct EntsData {
    pub players: Vec<PlayerE>,
    pub units: Vec<UnitE>,
    pub attacks: Vec<AttackE>,
    pub alliances: Vec<AllianceE>,
    pub doomsday_enabled: bool,
}

pub fn parse_ents(v: &Value) -> EntsData {
    let players = v["players"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|p| {
                    Some(PlayerE {
                        id: p["id"].as_u64()? as usize,
                        pid: p["pid"].as_str().unwrap_or("").to_string(),
                        troops: num(&p["troops"]),
                        gold: num(&p["gold"]),
                        tiles: num(&p["tiles"]),
                        alive: p["alive"].as_bool().unwrap_or(false),
                        traitor: p["traitor"].as_bool().unwrap_or(false),
                        embargoes: id_list(&p["embargoes"]),
                        relations: relation_list(&p["relations"]),
                        reqs_in: p["reqsIn"].as_array().map_or(0, |x| x.len()),
                        reqs_out: p["reqsOut"].as_array().map_or(0, |x| x.len()),
                        targets: id_list(&p["targets"]),
                        troop_income: num(&p["troopIncome"]),
                        gold_income: num(&p["goldIncome"]),
                        doomsday: p["doomsday"].as_bool().unwrap_or(false),
                        doomsday_ticks: num(&p["doomsdayTicks"]),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let units = v["units"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|u| {
                    let class = unit_class(u["type"].as_str()?)?;
                    let (tx, ty) = (&u["tx"], &u["ty"]);
                    let has_target = !tx.is_null();
                    Some(UnitE {
                        class,
                        owner: u["owner"].as_u64()? as usize,
                        uid: u.get("uid").and_then(|v| v.as_i64()).unwrap_or(-1),
                        x: u["x"].as_i64()?,
                        y: u["y"].as_i64()?,
                        gx: u["x"].as_i64()? / REGION as i64,
                        gy: u["y"].as_i64()? / REGION as i64,
                        tgx: if has_target {
                            num(tx) as i64 / REGION as i64
                        } else {
                            -1
                        },
                        tgy: if has_target {
                            num(ty) as i64 / REGION as i64
                        } else {
                            -1
                        },
                        has_target,
                        constructing: u["constructing"].as_bool().unwrap_or(false),
                        level: u.get("level").map(num).filter(|&x| x != 0.0).unwrap_or(1.0),
                        health: u.get("health").map(num),
                        max_health: u.get("maxHealth").map(num),
                        troops: num(&u["troops"]),
                        sam_lock: u["samLock"].as_bool().unwrap_or(false),
                        cooldown: u.get("cooldown").and_then(|v| v.as_bool()).unwrap_or(false),
                        station: u.get("station").and_then(|v| v.as_bool()).unwrap_or(false),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let attacks = v["attacks"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| {
                    Some(AttackE {
                        aid: x["aid"].as_str().unwrap_or("").to_string(),
                        from: x["from"].as_u64()? as usize,
                        to: x["to"].as_u64().unwrap_or(0) as usize,
                        troops: num(&x["troops"]),
                        retreating: x["retreating"].as_bool().unwrap_or(false),
                        src_x: x.get("srcX").and_then(|v| v.as_i64()),
                        src_y: x.get("srcY").and_then(|v| v.as_i64()),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let alliances = v["alliances"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|x| {
                    let arr = x.as_array()?;
                    Some(AllianceE(
                        arr.first()?.as_u64()? as usize,
                        arr.get(1)?.as_u64()? as usize,
                        arr.get(2).and_then(|v| v.as_i64()).unwrap_or(0),
                    ))
                })
                .collect()
        })
        .unwrap_or_default();
    EntsData {
        players,
        units,
        attacks,
        alliances,
        doomsday_enabled: v["doomsdayEnabled"].as_bool().unwrap_or(false),
    }
}

#[derive(Default, Clone, Debug)]
pub struct Legal {
    pub present: bool,
    pub attackable: Vec<usize>,
    pub alliance_requestable: Vec<usize>,
    pub alliance_rejectable: Vec<usize>,
    pub breakable: Vec<usize>,
    pub donatable_gold: Vec<usize>,
    pub donatable_troops: Vec<usize>,
    pub embargoable: Vec<usize>,
    pub stop_embargoable: Vec<usize>,
    pub targetable: Vec<usize>,
    pub extendable: Vec<usize>,
    pub can_expand: bool,
    pub can_boat: bool,
    pub troops: f64,
    pub gold: f64,
    pub build_mask: [f32; N_BUILD],
    pub nuke_mask: [f32; N_NUKE],
    pub has_silo: bool,
    pub attacks: Vec<String>, // attack ids (n_attacks = len)
    pub upgradable: Vec<usize>,
    pub warships: Vec<usize>,
    pub boats: Vec<usize>,
    pub deletable: Vec<usize>,
}

fn nuke_rows(unit: &str) -> &'static [usize] {
    match unit {
        "Atom Bomb" => &[0, 1],
        "Hydrogen Bomb" => &[2, 3],
        "MIRV" => &[4],
        _ => &[],
    }
}

/// Parse a legality `actions` object (bridge/common.ts legality()).
pub fn parse_legal(actions: &Value) -> Legal {
    let obj = match actions.as_object() {
        Some(o) if !o.is_empty() => o,
        _ => return Legal::default(),
    };
    let mut build_mask = [0.0f32; N_BUILD];
    let mut nuke_mask = [0.0f32; N_NUKE];
    if let Some(types) = obj.get("buildableTypes").and_then(|v| v.as_array()) {
        for t in types.iter().filter_map(|t| t.as_str()) {
            if let Some(i) = build_index(t) {
                build_mask[i as usize] = 1.0;
            }
            for &i in nuke_rows(t) {
                nuke_mask[i] = 1.0;
            }
        }
    }
    let get = |k: &str| obj.get(k).cloned().unwrap_or(Value::Null);
    let str_list = |k: &str| -> Vec<usize> {
        // upgradable/warships/boats/deletable carry engine unit ids (u64),
        // used only for len()/id-membership downstream.
        id_list(&get(k))
    };
    let attack_ids = |k: &str| -> Vec<String> {
        get(k)
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default()
    };
    Legal {
        present: true,
        attackable: id_list(&get("attackable")),
        alliance_requestable: id_list(&get("allianceRequestable")),
        alliance_rejectable: id_list(&get("allianceRejectable")),
        breakable: id_list(&get("breakable")),
        donatable_gold: id_list(&get("donatableGold")),
        donatable_troops: id_list(&get("donatableTroops")),
        embargoable: id_list(&get("embargoable")),
        stop_embargoable: id_list(&get("stopEmbargoable")),
        targetable: id_list(&get("targetable")),
        extendable: id_list(&get("extendable")),
        // Default closed: a missing key must not open expand/boat masks.
        can_expand: get("canExpand").as_bool().unwrap_or(false),
        can_boat: get("canBoat").as_bool().unwrap_or(false),
        troops: num(&get("troops")),
        gold: num(&get("gold")),
        build_mask,
        nuke_mask,
        has_silo: get("hasSilo").as_bool().unwrap_or(false),
        attacks: attack_ids("attacks"),
        upgradable: str_list("upgradable"),
        warships: str_list("warships"),
        boats: str_list("boats"),
        deletable: str_list("deletable"),
    }
}

pub struct Feat {
    pub stat: Vec<f32>,      // (N_STATIC, gh, gw)
    pub transient: Vec<f32>, // (N_TRANSIENT, gh, gw)
    pub clut: [u8; MAX_SLOTS],
    pub players: Vec<f32>, // (MAX_SLOTS, P_FEAT)
    pub pmask: [f32; MAX_SLOTS],
    pub scalars: [f32; N_SCALARS],
    pub me_slot: i64,
    pub legal_actions: [f32; N_ACTIONS],
    pub legal_ptarget: Vec<f32>, // (N_ACTIONS, MAX_SLOTS)
    pub legal_build: [f32; N_BUILD],
    pub legal_nuke: [f32; N_NUKE],
    pub legal_tile: Vec<f32>, // (gh, gw)
}

/// Slot lookup table: raw small-player-id -> stable slot in [0, MAX_SLOTS).
/// Mirrors `ObsBuilder._make_lut` (slot 0 = neutral/unowned).
pub fn make_lut(player_ids: &[usize]) -> Vec<u8> {
    let mut ids: Vec<usize> = player_ids.to_vec();
    ids.sort_unstable();
    let mut lut = vec![0u8; 4096];
    for (slot, &sid) in ids.iter().enumerate() {
        if sid < lut.len() {
            lut[sid] = (slot + 1).min(MAX_SLOTS - 1) as u8;
        }
    }
    lut
}

/// Slot → ego class LUT (0 neutral, 1 own, 2 ally, 3 enemy). Shared by
/// [`featurize`] and the fused tile+ego prepare path so both see identical
/// classing without a second full-map scan.
pub fn make_clut(lut: &[u8], me: i64, ents: &EntsData) -> [u8; MAX_SLOTS] {
    let slot_of = |id: usize| -> usize { *lut.get(id).unwrap_or(&0) as usize };
    let me_slot: usize = if me >= 0 { slot_of(me as usize) } else { 0 };
    let mut clut = [3u8; MAX_SLOTS];
    clut[0] = 0;
    for a in &ents.alliances {
        let (sa, sb) = (slot_of(a.0), slot_of(a.1));
        if sa == me_slot && sb < MAX_SLOTS {
            clut[sb] = 2;
        } else if sb == me_slot && sa < MAX_SLOTS {
            clut[sa] = 2;
        }
    }
    if me_slot > 0 && me_slot < MAX_SLOTS {
        clut[me_slot] = 1;
    }
    clut
}

#[allow(clippy::too_many_arguments)]
pub fn featurize(
    gh: usize,
    gw: usize,
    lut: &[u8],
    land: &[u8],
    mag: &[u8],
    owners: &[u8], // already slotted (uint8) at this resolution
    tick: i64,
    spawn_phase: bool,
    alive: bool,
    me: i64,
    ents: &EntsData,
    legal: &Legal,
) -> Feat {
    let slot_of = |id: usize| -> usize { *lut.get(id).unwrap_or(&0) as usize };
    let me_slot: usize = if me >= 0 { slot_of(me as usize) } else { 0 };
    let plane = gh * gw;

    // Ally slots + expiry, and the slot -> ego class LUT (0 neutral, 1
    // own, 2 ally, 3 enemy) used both for unit oc classing and for the
    // player feature block below.
    let mut allies: Vec<usize> = Vec::new();
    let mut ally_expiry: std::collections::HashMap<usize, i64> = std::collections::HashMap::new();
    for a in &ents.alliances {
        let (sa, sb) = (slot_of(a.0), slot_of(a.1));
        if sa == me_slot {
            allies.push(sb);
            ally_expiry.insert(sb, a.2);
        } else if sb == me_slot {
            allies.push(sa);
            ally_expiry.insert(sa, a.2);
        }
    }
    let clut = make_clut(lut, me, ents);
    let is_ally = |slot: usize| allies.contains(&slot);

    let mut stat = vec![0.0f32; N_STATIC * plane];
    let mut transient = vec![0.0f32; N_TRANSIENT * plane];
    for u in &ents.units {
        if !(0 <= u.gy && (u.gy as usize) < gh && 0 <= u.gx && (u.gx as usize) < gw) {
            continue;
        }
        let at = u.gy as usize * gw + u.gx as usize;
        let cls = clut[slot_of(u.owner).min(MAX_SLOTS - 1)] as i64;
        let oc = if cls != 0 { (cls - 1) as usize } else { 2 };
        if u.class < N_STATIC && !u.constructing {
            let level_frac = (u.level / 10.0).min(1.0) as f32;
            let idx = u.class * plane + at;
            stat[idx] = stat[idx].max(level_frac);
        }
        let target_ok = u.has_target
            && 0 <= u.tgy
            && (u.tgy as usize) < gh
            && 0 <= u.tgx
            && (u.tgx as usize) < gw;
        let tat = if target_ok {
            u.tgy as usize * gw + u.tgx as usize
        } else {
            0
        };
        match u.class {
            6 => {
                // Warship: value = health fraction.
                let hf = match (u.health, u.max_health) {
                    (Some(h), Some(m)) if m > 0.0 => (h / m) as f32,
                    _ => 1.0,
                };
                let idx = (TR_WARSHIP + oc) * plane + at;
                transient[idx] = transient[idx].max(hf);
            }
            7 => {
                // Transport (+ destination).
                let idx = (TR_TRANSPORT + oc) * plane + at;
                transient[idx] = transient[idx].max(log_norm(u.troops));
                if target_ok {
                    transient[(TR_TRANSPORT_DEST + oc) * plane + tat] = 1.0;
                }
            }
            8 => {
                transient[(TR_TRADE + oc) * plane + at] = 1.0;
                if target_ok {
                    transient[(TR_TRADE_DEST + oc) * plane + tat] = 1.0;
                }
            }
            9 | 10 | 11 => {
                // Nukes (+ impact point, + samlock).
                transient[(TR_NUKE + oc) * plane + at] = 1.0;
                if target_ok {
                    transient[(TR_NUKE_IMPACT + oc) * plane + tat] = 1.0;
                }
                if u.sam_lock {
                    transient[(TR_NUKE_SAMLOCK + oc) * plane + at] = 1.0;
                }
            }
            12 => {
                transient[(TR_SAM_MISSILE + oc) * plane + at] = 1.0;
                if target_ok {
                    transient[(TR_SAM_MISSILE_IMPACT + oc) * plane + tat] = 1.0;
                }
            }
            13 => {
                transient[(TR_MIRV_WARHEAD + oc) * plane + at] = 1.0;
                if target_ok {
                    transient[(TR_MIRV_WARHEAD_IMPACT + oc) * plane + tat] = 1.0;
                }
            }
            14 => {
                let idx = (TR_TRAIN + oc) * plane + at;
                transient[idx] = transient[idx].max(log_norm(u.troops));
            }
            _ => {}
        }
        if u.constructing {
            transient[(TR_CONSTRUCTION + oc) * plane + at] = 1.0;
        }
        if u.cooldown {
            if u.class == 3 {
                transient[(TR_SILO_COOLDOWN + oc) * plane + at] = 1.0;
            } else if u.class == 4 {
                transient[(TR_SAM_COOLDOWN + oc) * plane + at] = 1.0;
            }
        }
        if u.station {
            transient[(TR_STATION + oc) * plane + at] = 1.0;
        }
    }

    // Attack fronts (v7): source-tile intensity across ALL active attacks.
    for a in &ents.attacks {
        let (src_x, src_y) = match (a.src_x, a.src_y) {
            (Some(x), Some(y)) => (x, y),
            _ => continue,
        };
        let (gy, gx) = (src_y / REGION as i64, src_x / REGION as i64);
        if !(0 <= gy && (gy as usize) < gh && 0 <= gx && (gx as usize) < gw) {
            continue;
        }
        let at = gy as usize * gw + gx as usize;
        let val = log_norm(a.troops);
        let idx = TR_ATTACK_SRC * plane + at;
        transient[idx] = transient[idx].max(val);
        if a.retreating {
            let idx = TR_ATTACK_RETREAT * plane + at;
            transient[idx] = transient[idx].max(val);
        }
    }

    // Player features (v7: +9 columns, see below).
    let mut atk_between: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    let mut out_troops: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    let mut in_troops: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    let mut atk_total: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut atk_retreating: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for a in &ents.attacks {
        let sa = slot_of(a.from);
        let sb = if a.to != 0 { slot_of(a.to) } else { 0 };
        if sa == me_slot {
            *atk_between.entry(sb).or_insert(0.0) += a.troops;
        } else if sb == me_slot {
            *atk_between.entry(sa).or_insert(0.0) -= a.troops;
        }
        *out_troops.entry(sa).or_insert(0.0) += a.troops;
        if sb != 0 {
            *in_troops.entry(sb).or_insert(0.0) += a.troops;
        }
        *atk_total.entry(sa).or_insert(0) += 1;
        if a.retreating {
            *atk_retreating.entry(sa).or_insert(0) += 1;
        }
    }
    // Target marks: who ego or an ally has painted as a priority target.
    let mut marked: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for p in &ents.players {
        let slot = slot_of(p.id);
        if slot == me_slot || is_ally(slot) {
            for &t in &p.targets {
                marked.insert(slot_of(t));
            }
        }
    }

    let mut players = vec![0.0f32; MAX_SLOTS * P_FEAT];
    let mut pmask = [0.0f32; MAX_SLOTS];
    let mut n_alive = 0usize;
    for p in &ents.players {
        n_alive += p.alive as usize;
        let slot = slot_of(p.id);
        if slot == 0 || slot >= MAX_SLOTS {
            continue;
        }
        pmask[slot] = 1.0;
        let atk = atk_between.get(&slot).copied().unwrap_or(0.0);
        let n_atk = atk_total.get(&slot).copied().unwrap_or(0);
        let embargoed_me = p.embargoes.iter().any(|&e| slot_of(e) == me_slot);
        let ally_expiry_frac = ally_expiry
            .get(&slot)
            .map(|&exp| (((exp - tick) as f64 / 3000.0).clamp(0.0, 1.0)) as f32)
            .unwrap_or(0.0);
        let f = &mut players[slot * P_FEAT..(slot + 1) * P_FEAT];
        f[0] = p.alive as u8 as f32;
        f[1] = log_norm(p.troops);
        f[2] = log_norm(p.gold);
        f[3] = log_norm(p.tiles);
        f[4] = p.traitor as u8 as f32;
        f[5] = is_ally(slot) as u8 as f32;
        f[6] = embargoed_me as u8 as f32;
        f[7] = (slot == me_slot) as u8 as f32;
        f[8] = log_norm(atk.abs());
        f[9] = (atk > 0.0) as u8 as f32;
        f[10] = p.reqs_in as f32 / 4.0;
        f[11] = p.reqs_out as f32 / 4.0;
        f[12] = log_norm(out_troops.get(&slot).copied().unwrap_or(0.0));
        f[13] = log_norm(in_troops.get(&slot).copied().unwrap_or(0.0));
        f[14] = if n_atk > 0 {
            atk_retreating.get(&slot).copied().unwrap_or(0) as f32 / n_atk as f32
        } else {
            0.0
        };
        f[15] = ally_expiry_frac;
        f[16] = marked.contains(&slot) as u8 as f32;
        f[17] = log_norm(p.troop_income);
        f[18] = log_norm(p.gold_income);
        f[19] = p.doomsday as u8 as f32;
        f[20] = (p.doomsday_ticks / 3000.0).clamp(0.0, 1.0) as f32;
    }

    let me_troop_income = ents
        .players
        .iter()
        .find(|p| p.id as i64 == me)
        .map(|p| p.troop_income);
    let me_gold_income = ents
        .players
        .iter()
        .find(|p| p.id as i64 == me)
        .map(|p| p.gold_income);
    let scalars = [
        tick as f32 / 15000.0,
        spawn_phase as u8 as f32,
        alive as u8 as f32,
        log_norm(legal.troops),
        log_norm(legal.gold),
        n_alive as f32 / 128.0,
        legal.attacks.len() as f32 / 8.0,
        me_slot as f32 / MAX_SLOTS as f32,
        log_norm(me_troop_income.unwrap_or(0.0)),
        log_norm(me_gold_income.unwrap_or(0.0)),
        ents.doomsday_enabled as u8 as f32,
    ];

    // Action masks (rl/obs.py._masks).
    let mut act = [0.0f32; N_ACTIONS];
    let mut ptarget = vec![0.0f32; N_ACTIONS * MAX_SLOTS];
    if spawn_phase {
        act[A_SPAWN as usize] = 1.0;
    } else {
        act[A_NOOP as usize] = 1.0;
    }
    if alive && legal.present && !spawn_phase {
        let mut fill = |action: i64, ids: &[usize]| {
            if !ids.is_empty() {
                act[action as usize] = 1.0;
                for &i in ids {
                    let s = slot_of(i);
                    if s < MAX_SLOTS {
                        ptarget[action as usize * MAX_SLOTS + s] = 1.0;
                    }
                }
            }
        };
        fill(A_ATTACK, &legal.attackable);
        fill(A_ALLIANCE_REQUEST, &legal.alliance_requestable);
        fill(A_ALLIANCE_REJECT, &legal.alliance_rejectable);
        fill(A_BREAK_ALLIANCE, &legal.breakable);
        fill(A_DONATE_GOLD, &legal.donatable_gold);
        fill(A_DONATE_TROOPS, &legal.donatable_troops);
        fill(A_EMBARGO, &legal.embargoable);
        fill(A_EMBARGO_STOP, &legal.stop_embargoable);
        fill(A_TARGET_PLAYER, &legal.targetable);
        fill(A_ALLIANCE_EXTENSION, &legal.extendable);
        act[A_EXPAND as usize] = legal.can_expand as u8 as f32;
        act[A_BOAT as usize] = (legal.can_boat && legal.troops > 100.0) as u8 as f32;
        act[A_BUILD as usize] = legal.build_mask.iter().any(|&x| x > 0.0) as u8 as f32;
        act[A_LAUNCH_NUKE as usize] =
            (legal.nuke_mask.iter().any(|&x| x > 0.0) && legal.has_silo) as u8 as f32;
        if !legal.attacks.is_empty() {
            act[A_RETREAT as usize] = 1.0;
            for a in &ents.attacks {
                if me >= 0 && a.from == me as usize {
                    let s = if a.to != 0 { slot_of(a.to) } else { 0 };
                    if s < MAX_SLOTS {
                        ptarget[A_RETREAT as usize * MAX_SLOTS + s] = 1.0;
                    }
                }
            }
        }
        act[A_UPGRADE_STRUCTURE as usize] = !legal.upgradable.is_empty() as u8 as f32;
        act[A_MOVE_WARSHIP as usize] = !legal.warships.is_empty() as u8 as f32;
        act[A_CANCEL_BOAT as usize] = !legal.boats.is_empty() as u8 as f32;
        act[A_DELETE_UNIT as usize] = !legal.deletable.is_empty() as u8 as f32;
    }
    let (legal_build, legal_nuke) = if alive && legal.present {
        (legal.build_mask, legal.nuke_mask)
    } else {
        ([0.0; N_BUILD], [0.0; N_NUKE])
    };

    let legal_tile = legal_tile_mask(land, mag, owners, spawn_phase, gh, gw);

    Feat {
        stat,
        transient,
        clut,
        players,
        pmask,
        scalars,
        me_slot: me_slot as i64,
        legal_actions: act,
        legal_ptarget: ptarget,
        legal_build,
        legal_nuke,
        legal_tile,
    }
}

/// Ego-class occupancy fractions (own/ally/enemy) + defense-bonus fraction,
/// average-pooled to /REGION resolution from the full-res slotted owners +
/// clut + defense_bonus planes. Mirrors the `ego`/`db_pooled` tensors
/// `encode_grids` builds on the GPU in Python (`F.avg_pool2d`); done here
/// on CPU since `oftrain`'s v1 grid skips the AE latent (see policy.rs).
pub fn pool_ego_db(
    owners_slotted: &[u8],
    clut: &[u8; MAX_SLOTS],
    defense_bonus: &[u8],
    hr: usize,
    wr: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (gh, gw) = (hr / REGION, wr / REGION);
    let plane = gh * gw;
    let mut ego = vec![0.0f32; 3 * plane];
    let mut db = vec![0.0f32; plane];
    let norm = (REGION * REGION) as f32;
    for gy in 0..gh {
        for gx in 0..gw {
            let mut counts = [0u32; 3];
            let mut db_sum = 0u32;
            for y in gy * REGION..(gy + 1) * REGION {
                let row = y * wr;
                for x in gx * REGION..(gx + 1) * REGION {
                    let i = row + x;
                    let cls = clut[owners_slotted[i] as usize];
                    if cls > 0 {
                        counts[(cls - 1) as usize] += 1;
                    }
                    db_sum += defense_bonus[i] as u32;
                }
            }
            let at = gy * gw + gx;
            for c in 0..3 {
                ego[c * plane + at] = counts[c] as f32 / norm;
            }
            db[at] = db_sum as f32 / norm;
        }
    }
    (ego, db)
}

/// Pool ego classes and compute the own-territory centroid in one full-map
/// scan. The centroid is consumed by [`local_crop_at_with_defense`].
pub fn pool_ego_and_center(
    owners_slotted: &[u8],
    clut: &[u8; MAX_SLOTS],
    hr: usize,
    wr: usize,
) -> (Vec<f32>, (f64, f64)) {
    let (gh, gw) = (hr / REGION, wr / REGION);
    let plane = gh * gw;
    let mut counts = vec![0u32; 3 * plane];
    let mut sum_y = 0.0f64;
    let mut sum_x = 0.0f64;
    let mut own_count = 0.0f64;
    for y in 0..hr {
        for x in 0..wr {
            let cls = clut[owners_slotted[y * wr + x] as usize];
            if cls > 0 {
                let at = (y / REGION) * gw + x / REGION;
                counts[(cls - 1) as usize * plane + at] += 1;
            }
            if cls == 1 {
                sum_y += y as f64;
                sum_x += x as f64;
                own_count += 1.0;
            }
        }
    }
    let norm = (REGION * REGION) as f32;
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
}

/// Local crop using a precomputed own-territory center and an indexed defense
/// accessor. This lets packed native observations decode defense only for the
/// crop instead of allocating a full defense plane.
pub fn local_crop_at_with_defense<F>(
    owners_slotted: &[u8],
    clut: &[u8; MAX_SLOTS],
    land: &[u8],
    hr: usize,
    wr: usize,
    local: usize,
    center: (f64, f64),
    defense_at: F,
) -> Vec<f32>
where
    F: Fn(usize) -> u8,
{
    let (cy, cx) = center;
    let y0 = ((cy - local as f64 / 2.0).round() as i64)
        .clamp(0, hr as i64 - local as i64)
        .max(0);
    let x0 = ((cx - local as f64 / 2.0).round() as i64)
        .clamp(0, wr as i64 - local as i64)
        .max(0);
    let mut out = vec![0.0f32; 5 * local * local];
    let plane = local * local;
    for ly in 0..local {
        let y = (y0 + ly as i64).clamp(0, hr as i64 - 1) as usize;
        for lx in 0..local {
            let x = (x0 + lx as i64).clamp(0, wr as i64 - 1) as usize;
            let i = y * wr + x;
            let cls = clut[owners_slotted[i] as usize];
            let at = ly * local + lx;
            if cls == 1 {
                out[at] = 1.0;
            } else if cls == 2 {
                out[plane + at] = 1.0;
            } else if cls == 3 {
                out[2 * plane + at] = 1.0;
            }
            out[3 * plane + at] = land[i] as f32;
            out[4 * plane + at] = defense_at(i) as f32;
        }
    }
    out
}

/// (N_LOCAL=5, LOCAL, LOCAL) raw owner-map crop centered on own territory's
/// centroid (map center if the agent owns nothing yet, e.g. spawn phase).
/// Channels: own, ally, enemy, land, defense_bonus. Port of
/// `rl/obs.py::_local_crops` (host-side; the Python version runs batched on
/// GPU, but this is cheap enough per-env on CPU).
pub fn local_crop(
    owners_slotted: &[u8],
    clut: &[u8; MAX_SLOTS],
    land: &[u8],
    defense_bonus: &[u8],
    hr: usize,
    wr: usize,
    local: usize,
) -> Vec<f32> {
    let mut sum_y = 0.0f64;
    let mut sum_x = 0.0f64;
    let mut n = 0.0f64;
    for y in 0..hr {
        for x in 0..wr {
            if clut[owners_slotted[y * wr + x] as usize] == 1 {
                sum_y += y as f64;
                sum_x += x as f64;
                n += 1.0;
            }
        }
    }
    let (cy, cx) = if n > 0.0 {
        (sum_y / n, sum_x / n)
    } else {
        (hr as f64 / 2.0, wr as f64 / 2.0)
    };
    let y0 = ((cy - local as f64 / 2.0).round() as i64)
        .clamp(0, hr as i64 - local as i64)
        .max(0);
    let x0 = ((cx - local as f64 / 2.0).round() as i64)
        .clamp(0, wr as i64 - local as i64)
        .max(0);
    let mut out = vec![0.0f32; 5 * local * local];
    let plane = local * local;
    for ly in 0..local {
        let y = (y0 + ly as i64).clamp(0, hr as i64 - 1) as usize;
        for lx in 0..local {
            let x = (x0 + lx as i64).clamp(0, wr as i64 - 1) as usize;
            let i = y * wr + x;
            let cls = clut[owners_slotted[i] as usize];
            let at = ly * local + lx;
            if cls == 1 {
                out[0 * plane + at] = 1.0;
            } else if cls == 2 {
                out[1 * plane + at] = 1.0;
            } else if cls == 3 {
                out[2 * plane + at] = 1.0;
            }
            out[3 * plane + at] = land[i] as f32;
            out[4 * plane + at] = defense_bonus[i] as f32;
        }
    }
    out
}

/// Tile-pointer region mask at /8 resolution: all-ones normally; during
/// the spawn phase, regions containing at least one valid spawn tile
/// (land, unowned, passable) - mirrors SpawnExecution.getSpawn(center).
pub fn legal_tile_mask(
    land: &[u8],
    mag: &[u8],
    owners: &[u8],
    spawn_phase: bool,
    gh: usize,
    gw: usize,
) -> Vec<f32> {
    if !spawn_phase {
        return vec![1.0f32; gh * gw];
    }
    let mut lt = vec![0.0f32; gh * gw];
    let wr = gw * REGION;
    for gy in 0..gh {
        for gx in 0..gw {
            'cell: for y in gy * REGION..(gy + 1) * REGION {
                let row = y * wr;
                for x in gx * REGION..(gx + 1) * REGION {
                    let i = row + x;
                    if land[i] == 1 && mag[i] < IMPASSABLE_MAGNITUDE && owners[i] == 0 {
                        lt[gy * gw + gx] = 1.0;
                        break 'cell;
                    }
                }
            }
        }
    }
    lt
}
