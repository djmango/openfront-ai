//! Featurization core: an exact Rust port of rl/obs.py ObsBuilder.prepare()
//! (obs v4, frozen). Consumed by the cache-bc sampler (sampler.rs) via
//! serde-parsed blobs; the constants mirror rl/obs.py + rl/policy.py and are
//! parity-tested from Python (scripts/test_ofrs_parity.py).

use serde_json::Value;

pub const REGION: usize = 8;
pub const MAX_SLOTS: usize = 128;
pub const N_TRANSIENT: usize = 8;
pub const N_STATIC: usize = 6;
pub const P_FEAT: usize = 12;
pub const N_SCALARS: usize = 8;
pub const N_ACTIONS: usize = 21;
pub const N_BUILD: usize = 7;
pub const N_NUKE: usize = 5;
pub const GW_MAX: i64 = 250;
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

pub fn action_index(name: &str) -> Option<i64> {
    Some(match name {
        "noop" => A_NOOP,
        "attack" => A_ATTACK,
        "expand" => A_EXPAND,
        "boat" => A_BOAT,
        "build" => A_BUILD,
        "launch_nuke" => A_LAUNCH_NUKE,
        "alliance_request" => A_ALLIANCE_REQUEST,
        "alliance_reject" => A_ALLIANCE_REJECT,
        "break_alliance" => A_BREAK_ALLIANCE,
        "donate_gold" => A_DONATE_GOLD,
        "donate_troops" => A_DONATE_TROOPS,
        "embargo" => A_EMBARGO,
        "retreat" => A_RETREAT,
        "spawn" => A_SPAWN,
        "upgrade_structure" => A_UPGRADE_STRUCTURE,
        "move_warship" => A_MOVE_WARSHIP,
        "cancel_boat" => A_CANCEL_BOAT,
        "delete_unit" => A_DELETE_UNIT,
        "embargo_stop" => A_EMBARGO_STOP,
        "target_player" => A_TARGET_PLAYER,
        "alliance_extension" => A_ALLIANCE_EXTENSION,
        _ => return None,
    })
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
    matches!(a, A_ATTACK | A_EXPAND | A_BOAT | A_DONATE_GOLD | A_DONATE_TROOPS)
}

pub fn build_index(unit: &str) -> Option<i64> {
    Some(match unit {
        "City" => 0,
        "Port" => 1,
        "Defense Post" => 2,
        "Missile Silo" => 3,
        "SAM Launcher" => 4,
        "Factory" => 5,
        "Warship" => 6,
        _ => return None,
    })
}

/// 5-way nuke head: (unit, rocketDirectionUp). MIRV ignores the arc flag
/// and gets a single row (mirrors rl/obs.py NUKE_TYPES).
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

/// Both arc rows of a nuke unit (for buildableTypes -> legal_nuke).
fn nuke_rows(unit: &str) -> &'static [usize] {
    match unit {
        "Atom Bomb" => &[0, 1],
        "Hydrogen Bomb" => &[2, 3],
        "MIRV" => &[4],
        _ => &[],
    }
}

/// UNIT_CLASSES order from ae/units.py; the first N_STATIC are the static
/// structures (static_pos is the identity there).
fn unit_class(ty: &str) -> Option<usize> {
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
        _ => return None,
    })
}

pub fn log_norm(x: f64) -> f32 {
    ((1.0 + x.max(0.0)).log10() / 8.0) as f32
}

/// Scalar 0-1 Beta-head target: fraction of available troops/gold actually
/// committed. Mirrors rl/bc_data.py quantity_frac (None -> client default).
pub fn quantity_frac(amt: Option<f64>, avail: f64) -> f64 {
    match amt {
        Some(a) if avail > 0.0 => (a / avail).clamp(0.0, 1.0),
        _ => 0.25,
    }
}

pub fn placement_bucket(p: f64) -> i64 {
    ((p * 8.0) as i64).min(7)
}

/// Tolerant number extraction: the bridge serializes gold as a string
/// (bigint) and everything else as JSON numbers.
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
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect())
        .unwrap_or_default()
}

pub struct Unit {
    pub class: usize,
    pub gx: i64,
    pub gy: i64,
    pub tgx: i64,
    pub tgy: i64,
    pub has_target: bool,
    pub sam_lock: bool,
    pub constructing: bool,
}

pub struct PlayerE {
    pub id: usize,
    pub troops: f64,
    pub gold: f64,
    pub tiles: f64,
    pub alive: bool,
    pub traitor: bool,
    pub embargoes: Vec<usize>,
    pub reqs_in: f32,
    pub reqs_out: f32,
}

pub struct Att {
    pub from: usize,
    pub to: usize,
    pub troops: f64,
}

pub struct EntsData {
    pub players: Vec<PlayerE>,
    pub units: Vec<Unit>,
    pub attacks: Vec<Att>,
    pub alliances: Vec<(usize, usize)>,
}

pub fn parse_ents(v: &Value) -> EntsData {
    let players = v["players"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|p| {
                    Some(PlayerE {
                        id: p["id"].as_u64()? as usize,
                        troops: num(&p["troops"]),
                        gold: num(&p["gold"]),
                        tiles: num(&p["tiles"]),
                        alive: p["alive"].as_bool().unwrap_or(false),
                        traitor: p["traitor"].as_bool().unwrap_or(false),
                        embargoes: id_list(&p["embargoes"]),
                        reqs_in: p["reqsIn"].as_array().map_or(0, |x| x.len()) as f32,
                        reqs_out: p["reqsOut"].as_array().map_or(0, |x| x.len()) as f32,
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
                    Some(Unit {
                        class,
                        gx: u["x"].as_i64()? / REGION as i64,
                        gy: u["y"].as_i64()? / REGION as i64,
                        tgx: if has_target { num(tx) as i64 / REGION as i64 } else { -1 },
                        tgy: if has_target { num(ty) as i64 / REGION as i64 } else { -1 },
                        has_target,
                        sam_lock: u["samLock"].as_bool().unwrap_or(false),
                        constructing: u["constructing"].as_bool().unwrap_or(false),
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
                    Some(Att {
                        from: x["from"].as_u64()? as usize,
                        // Python: lut[to] if to else 0 - to==0 stays slot 0.
                        to: x["to"].as_u64().unwrap_or(0) as usize,
                        troops: num(&x["troops"]),
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
                    Some((arr.first()?.as_u64()? as usize, arr.get(1)?.as_u64()? as usize))
                })
                .collect()
        })
        .unwrap_or_default();
    EntsData { players, units, attacks, alliances }
}

#[derive(Default, Clone)]
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
    pub n_attacks: usize,
    pub has_upgradable: bool,
    pub has_warships: bool,
    pub has_boats: bool,
    pub has_deletable: bool,
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
    let nonempty = |k: &str| get(k).as_array().is_some_and(|a| !a.is_empty());
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
        can_expand: get("canExpand").as_bool().unwrap_or(true),
        can_boat: get("canBoat").as_bool().unwrap_or(true),
        troops: num(&get("troops")),
        gold: num(&get("gold")),
        build_mask,
        nuke_mask,
        has_silo: get("hasSilo").as_bool().unwrap_or(false),
        n_attacks: get("attacks").as_array().map_or(0, |a| a.len()),
        has_upgradable: nonempty("upgradable"),
        has_warships: nonempty("warships"),
        has_boats: nonempty("boats"),
        has_deletable: nonempty("deletable"),
    }
}

pub struct Feat {
    pub stat: Vec<f32>,          // (N_STATIC, gh, gw)
    pub transient: Vec<f32>,     // (N_TRANSIENT, gh, gw)
    pub clut: [u8; MAX_SLOTS],
    pub players: Vec<f32>,       // (MAX_SLOTS, P_FEAT)
    pub pmask: [f32; MAX_SLOTS],
    pub scalars: [f32; N_SCALARS],
    pub me_slot: i64,
    pub legal_actions: [f32; N_ACTIONS],
    pub legal_ptarget: Vec<f32>, // (N_ACTIONS, MAX_SLOTS)
    pub legal_build: [f32; N_BUILD],
    pub legal_nuke: [f32; N_NUKE],
    pub legal_tile: Vec<f32>,    // (gh, gw)
}

#[allow(clippy::too_many_arguments)]
pub fn featurize(
    gh: usize,
    gw: usize,
    lut: &[u8],
    land: &[u8],
    mag: &[u8],
    owners: &[u8],
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
    let mut stat = vec![0.0f32; N_STATIC * plane];
    let mut transient = vec![0.0f32; N_TRANSIENT * plane];
    for u in &ents.units {
        let (gy, gx) = (u.gy, u.gx);
        if !(0 <= gy && (gy as usize) < gh && 0 <= gx && (gx as usize) < gw) {
            continue;
        }
        let at = gy as usize * gw + gx as usize;
        if u.class < N_STATIC && !u.constructing {
            stat[u.class * plane + at] = 1.0;
        }
        let target_ok = u.has_target
            && 0 <= u.tgy
            && (u.tgy as usize) < gh
            && 0 <= u.tgx
            && (u.tgx as usize) < gw;
        let tat = if target_ok { u.tgy as usize * gw + u.tgx as usize } else { 0 };
        match u.class {
            6 => transient[at] = 1.0, // Warship
            7 => {
                // Transport (+ destination)
                transient[plane + at] = 1.0;
                if target_ok {
                    transient[2 * plane + tat] = 1.0;
                }
            }
            8 => transient[3 * plane + at] = 1.0, // Trade Ship
            9 | 10 | 11 => {
                // Nukes (+ impact point, + samlock)
                transient[4 * plane + at] = 1.0;
                if target_ok {
                    transient[5 * plane + tat] = 1.0;
                }
                if u.sam_lock {
                    transient[6 * plane + at] = 1.0;
                }
            }
            _ => {}
        }
        if u.constructing {
            transient[7 * plane + at] = 1.0;
        }
    }

    // Ally slots + class LUT (0 neutral, 1 own, 2 ally, 3 enemy).
    let mut clut = [3u8; MAX_SLOTS];
    clut[0] = 0;
    let mut allies = [false; MAX_SLOTS];
    for &(a, b) in &ents.alliances {
        let (sa, sb) = (slot_of(a), slot_of(b));
        if sa == me_slot {
            allies[sb] = true;
            clut[sb] = 2;
        } else if sb == me_slot {
            allies[sa] = true;
            clut[sa] = 2;
        }
    }
    if me_slot > 0 {
        clut[me_slot] = 1;
    }

    // Player features.
    let mut atk_between = [0.0f64; MAX_SLOTS];
    for a in &ents.attacks {
        let (sa, sb) = (slot_of(a.from), slot_of(a.to));
        if sa == me_slot {
            atk_between[sb] += a.troops;
        } else if sb == me_slot {
            atk_between[sa] -= a.troops;
        }
    }
    let mut players = vec![0.0f32; MAX_SLOTS * P_FEAT];
    let mut pmask = [0.0f32; MAX_SLOTS];
    let mut n_alive = 0usize;
    for p in &ents.players {
        n_alive += p.alive as usize;
        let slot = slot_of(p.id);
        if slot == 0 {
            continue;
        }
        pmask[slot] = 1.0;
        let atk = atk_between[slot];
        let embargoed_me = p.embargoes.iter().any(|&e| slot_of(e) == me_slot);
        let f = &mut players[slot * P_FEAT..(slot + 1) * P_FEAT];
        f[0] = p.alive as u8 as f32;
        f[1] = log_norm(p.troops);
        f[2] = log_norm(p.gold);
        f[3] = log_norm(p.tiles);
        f[4] = p.traitor as u8 as f32;
        f[5] = allies[slot] as u8 as f32;
        f[6] = embargoed_me as u8 as f32;
        f[7] = (slot == me_slot) as u8 as f32;
        f[8] = log_norm(atk.abs());
        f[9] = (atk > 0.0) as u8 as f32;
        f[10] = p.reqs_in / 4.0;
        f[11] = p.reqs_out / 4.0;
    }

    let scalars = [
        tick as f32 / 15000.0,
        spawn_phase as u8 as f32,
        alive as u8 as f32,
        log_norm(legal.troops),
        log_norm(legal.gold),
        n_alive as f32 / 128.0,
        legal.n_attacks as f32 / 8.0,
        me_slot as f32 / MAX_SLOTS as f32,
    ];

    // Action masks (rl/obs.py _masks).
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
                    ptarget[action as usize * MAX_SLOTS + slot_of(i)] = 1.0;
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
        // Targeted retreat: the player head picks WHICH attack to cancel
        // (slot of its target; slot 0 covers terra-nullius expands).
        if legal.n_attacks > 0 {
            act[A_RETREAT as usize] = 1.0;
            for a in &ents.attacks {
                if me >= 0 && a.from == me as usize {
                    ptarget[A_RETREAT as usize * MAX_SLOTS + slot_of(a.to)] = 1.0;
                }
            }
        }
        act[A_UPGRADE_STRUCTURE as usize] = legal.has_upgradable as u8 as f32;
        act[A_MOVE_WARSHIP as usize] = legal.has_warships as u8 as f32;
        act[A_CANCEL_BOAT as usize] = legal.has_boats as u8 as f32;
        act[A_DELETE_UNIT as usize] = legal.has_deletable as u8 as f32;
    }
    let (legal_build, legal_nuke) = if alive && legal.present {
        (legal.build_mask, legal.nuke_mask)
    } else {
        ([0.0; N_BUILD], [0.0; N_NUKE])
    };

    // legal_tile: all-ones normally; valid spawn regions during spawn phase.
    let legal_tile = if !spawn_phase {
        vec![1.0f32; plane]
    } else {
        let mut lt = vec![0.0f32; plane];
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
    };

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
