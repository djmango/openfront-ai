//! Curriculum stages and the strength-based reward. Port of
//! `rl/curriculum.py`; see that file for the design rationale in comments.

use crate::feat::EntsData;

pub const W_STR: f64 = 0.02;
pub const W_DELTA_GAIN: f64 = 5.0;
pub const W_DELTA_LOSS: f64 = 6.5;
pub const W_PLACE: f64 = 15.0;
pub const W_WIN: f64 = 30.0;
pub const W_DEATH: f64 = 1.0;
pub const W_WASTE: f64 = 0.01;
pub const PLACE_POW: f64 = 1.5;

pub const K_LAND: f64 = 0.40;
pub const K_MIL: f64 = 0.20;
pub const K_ECO: f64 = 0.25;
pub const K_BUILD: f64 = 0.15;

pub const WINDOW: usize = 40;
pub const REHEARSAL_P: f64 = 0.25;

pub fn struct_value(unit_type: &str) -> Option<f64> {
    Some(match unit_type {
        "City" => 1.0,
        "Port" => 1.0,
        "Factory" => 1.0,
        "Missile Silo" => 1.0,
        "Defense Post" => 0.25,
        "SAM Launcher" => 3.0,
        _ => return None,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nations {
    Default,
    Exact(u32),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stage {
    pub maps: &'static [&'static str],
    pub bots: u32,
    pub difficulty: &'static str,
    pub nations: Nations,
    pub decision_ticks: u32,
    pub win_at: f64,
}

pub const ALL_MAPS: [&str; 7] =
    ["Onion", "Pangaea", "Caucasus", "BlackSea", "BetweenTwoSeas", "World", "Asia"];

/// V8.1 env workers per GPU/shard. Early maps are cheap enough to keep the
/// actor full; later, larger maps use fewer workers to bound host memory and
/// engine tail latency.
pub const V81_ENV_TARGETS: [usize; 11] = [24, 24, 24, 24, 24, 24, 12, 10, 8, 8, 8];

pub fn stages() -> Vec<Stage> {
    stages_for(false)
}

/// Return the selected gate schedule. V8.1 deliberately changes only
/// `win_at`: stage indices, maps, opponents, difficulty, and decision cadence
/// remain stable so checkpoints and episode metadata retain their identity.
pub fn stages_for(v81: bool) -> Vec<Stage> {
    use Nations::{Default as ND, Exact as NE};
    let mut stages = vec![
        Stage { maps: &["Onion"], bots: 0, difficulty: "Easy", nations: NE(1), decision_ticks: 15, win_at: 0.9 },
        Stage { maps: &["Onion"], bots: 0, difficulty: "Easy", nations: NE(3), decision_ticks: 15, win_at: 0.8 },
        Stage { maps: &["Onion", "Pangaea"], bots: 5, difficulty: "Easy", nations: NE(3), decision_ticks: 15, win_at: 0.75 },
        Stage { maps: &["Pangaea", "Caucasus"], bots: 10, difficulty: "Easy", nations: NE(6), decision_ticks: 15, win_at: 0.65 },
        Stage { maps: &["Pangaea", "Caucasus", "BlackSea"], bots: 30, difficulty: "Easy", nations: ND, decision_ticks: 10, win_at: 0.55 },
        Stage { maps: &["BlackSea", "BetweenTwoSeas", "Caucasus"], bots: 30, difficulty: "Medium", nations: ND, decision_ticks: 10, win_at: 0.5 },
        Stage { maps: &["World", "Asia", "BlackSea"], bots: 50, difficulty: "Medium", nations: ND, decision_ticks: 10, win_at: 0.5 },
        Stage { maps: &["World", "Asia", "BetweenTwoSeas", "Caucasus"], bots: 80, difficulty: "Medium", nations: ND, decision_ticks: 10, win_at: 0.45 },
        Stage { maps: &ALL_MAPS, bots: 80, difficulty: "Hard", nations: ND, decision_ticks: 10, win_at: 0.4 },
        Stage { maps: &ALL_MAPS, bots: 120, difficulty: "Hard", nations: ND, decision_ticks: 10, win_at: 0.35 },
        Stage { maps: &ALL_MAPS, bots: 150, difficulty: "Impossible", nations: ND, decision_ticks: 10, win_at: 0.3 },
    ];
    if v81 {
        // Stages 0-3 are intentionally unchanged. Crowded-map wins are much
        // rarer than placement success, so require repeatable wins without
        // making the old 0.5 gate an effectively permanent lock.
        let gates = [0.35, 0.30, 0.25, 0.22, 0.20, 0.18, 0.15];
        for (stage, gate) in stages[4..].iter_mut().zip(gates) {
            stage.win_at = gate;
        }
    }
    stages
}

pub const GH_MAX: i64 = 150;
pub const GW_MAX: i64 = 250;

pub fn sample_episode(
    stg: &[Stage],
    stage: usize,
    rng: &mut impl rand::Rng,
) -> (String, u32, &'static str, Nations, bool) {
    let cur = &stg[stage];
    if stage > 0 && rng.r#gen::<f64>() < REHEARSAL_P {
        let past_i = rng.gen_range(0..stage);
        let past = &stg[past_i];
        let m = past.maps[rng.gen_range(0..past.maps.len())];
        return (m.to_string(), cur.bots, cur.difficulty, cur.nations, true);
    }
    let m = cur.maps[rng.gen_range(0..cur.maps.len())];
    (m.to_string(), cur.bots, cur.difficulty, cur.nations, false)
}

pub fn timeweight(tick: i64) -> f64 {
    0.5 + 0.5 * (tick as f64 / 8000.0).min(1.0)
}

/// Composite strength per living player: land / military / economic /
/// structural share. See rl/curriculum.py::strengths for the boat-churn
/// rationale behind counting fielded troops.
pub fn strengths(ents: &EntsData, land_total: i64) -> std::collections::HashMap<usize, f64> {
    let alive: Vec<&crate::feat::PlayerE> = ents.players.iter().filter(|p| p.alive).collect();
    let mut fielded: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    let mut sv: std::collections::HashMap<usize, f64> = std::collections::HashMap::new();
    for u in &ents.units {
        if u.troops > 0.0 {
            *fielded.entry(u.owner).or_insert(0.0) += u.troops;
        }
    }
    // Struct value needs the engine type string, which parse_ents doesn't
    // keep (only the class index); recompute from class since STATIC
    // classes 0..6 map 1:1 to BUILD_TYPES-minus-Warship order.
    for u in &ents.units {
        let ty = match u.class {
            0 => "City",
            1 => "Port",
            2 => "Defense Post",
            3 => "Missile Silo",
            4 => "SAM Launcher",
            5 => "Factory",
            _ => continue,
        };
        if let Some(v) = struct_value(ty) {
            if !u.constructing {
                *sv.entry(u.owner).or_insert(0.0) += v * u.level.max(1.0);
            }
        }
    }
    for a in &ents.attacks {
        *fielded.entry(a.from).or_insert(0.0) += a.troops;
    }
    let troops = |p: &crate::feat::PlayerE| p.troops + fielded.get(&p.id).copied().unwrap_or(0.0);
    let tot_troops: f64 = alive.iter().map(|p| troops(p)).sum::<f64>() + 1e-9;
    let tot_gold: f64 = alive.iter().map(|p| p.gold).sum::<f64>() + 1e-9;
    let tot_sv: f64 = alive.iter().map(|p| sv.get(&p.id).copied().unwrap_or(0.0)).sum::<f64>() + 1e-9;
    alive
        .iter()
        .map(|p| {
            let s = K_LAND * (p.tiles / land_total as f64)
                + K_MIL * (troops(p) / tot_troops)
                + K_ECO * (p.gold / tot_gold)
                + K_BUILD * (sv.get(&p.id).copied().unwrap_or(0.0) / tot_sv);
            (p.id, s)
        })
        .collect()
}

/// (place, n_players).
pub fn placement(ents: &EntsData, me: i64, agent_alive: bool, land_total: i64) -> (i64, i64) {
    let ids: std::collections::HashSet<usize> = ents.players.iter().map(|p| p.id).collect();
    let n = ids.len() as i64 + if ids.contains(&(me as usize)) { 0 } else { 1 };
    let s = strengths(ents, land_total);
    let me_u = me as usize;
    if !agent_alive || !s.contains_key(&me_u) {
        let others_alive = s.keys().filter(|&&pid| pid != me_u).count() as i64;
        return ((1 + others_alive).min(n), n);
    }
    let mine = s[&me_u];
    let better = s.iter().filter(|&(&pid, &v)| pid != me_u && v > mine).count() as i64;
    (1 + better, n)
}

pub fn placement_score(place: i64, n: i64) -> f64 {
    (n - place) as f64 / (n - 1).max(1) as f64
}

pub fn terminal_reward(place: i64, won: bool) -> f64 {
    let mut r = W_PLACE * (place as f64).powf(-PLACE_POW);
    if won {
        r += W_WIN;
    }
    r
}

#[cfg(test)]
mod curriculum_v81_tests {
    use super::*;

    #[test]
    fn v81_preserves_stage_identity_and_only_recalibrates_late_gates() {
        let legacy = stages();
        let v81 = stages_for(true);
        assert_eq!(legacy.len(), v81.len());
        for (index, (old, new)) in legacy.iter().zip(&v81).enumerate() {
            assert_eq!(old.maps, new.maps, "stage {index} maps changed");
            assert_eq!(old.bots, new.bots, "stage {index} bots changed");
            assert_eq!(old.difficulty, new.difficulty, "stage {index} difficulty changed");
            assert_eq!(old.nations, new.nations, "stage {index} nations changed");
            assert_eq!(old.decision_ticks, new.decision_ticks, "stage {index} cadence changed");
        }
        assert_eq!(
            v81.iter().map(|stage| stage.win_at).collect::<Vec<_>>(),
            vec![0.9, 0.8, 0.75, 0.65, 0.35, 0.30, 0.25, 0.22, 0.20, 0.18, 0.15]
        );
        for index in 0..4 {
            assert_eq!(legacy[index].win_at, v81[index].win_at);
        }
    }

    #[test]
    fn v81_env_defaults_match_map_scale_plan() {
        assert_eq!(&V81_ENV_TARGETS[..6], &[24; 6]);
        assert_eq!(&V81_ENV_TARGETS[6..], &[12, 10, 8, 8, 8]);
        assert_eq!(V81_ENV_TARGETS.len(), stages().len());
    }
}
