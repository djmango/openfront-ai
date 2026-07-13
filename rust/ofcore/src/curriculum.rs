//! Curriculum stages and the strength-based reward. Port of
//! `rl/curriculum.py`; see that file for the design rationale in comments.

use crate::feat::EntsData;
use std::collections::HashMap;

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

pub const DOMINANCE_EPS: f64 = 1e-9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RewardConfig {
    pub gamma: f64,
    pub v81_dom_coef: f64,
    pub v81_min_stage: usize,
    pub v81_potential_clamp: f64,
    pub v81_dominant_loss: bool,
    pub v81_dominance_threshold: f64,
    pub v81_delta_loss_dominant: f64,
}

impl RewardConfig {
    pub fn v81_active(self, stage: usize) -> bool {
        stage >= self.v81_min_stage
    }

    pub fn dominance_shaping_active(self, stage: usize) -> bool {
        self.v81_active(stage) && self.v81_dom_coef != 0.0
    }

    pub fn dominant_loss_active(self, stage: usize) -> bool {
        self.v81_active(stage) && self.v81_dominant_loss
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RewardComponents {
    pub strength: f64,
    pub strength_delta: f64,
    pub dominance: f64,
    pub waste: f64,
    pub death: f64,
    pub terminal: f64,
}

impl RewardComponents {
    pub fn add_assign(&mut self, other: Self) {
        self.strength += other.strength;
        self.strength_delta += other.strength_delta;
        self.dominance += other.dominance;
        self.waste += other.waste;
        self.death += other.death;
        self.terminal += other.terminal;
    }
}

/// Per-environment Φ(current), reset at each new episode.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DominanceShaper {
    prior: f64,
}

impl DominanceShaper {
    pub fn reset(&mut self, initial: f64) {
        self.prior = finite_or_zero(initial);
    }

    pub fn prior(self) -> f64 {
        self.prior
    }

    pub fn transition(&mut self, next: f64, gamma: f64, coefficient: f64) -> f64 {
        let next = finite_or_zero(next);
        let increment = coefficient * (gamma * next - self.prior);
        self.prior = next;
        finite_or_zero(increment)
    }
}

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

#[derive(Clone, Copy)]
pub enum Nations {
    Default,
    Exact(u32),
}

#[derive(Clone)]
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

pub fn stages() -> Vec<Stage> {
    use Nations::{Default as ND, Exact as NE};
    vec![
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
    ]
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
pub fn strengths(ents: &EntsData, land_total: i64) -> HashMap<usize, f64> {
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

fn finite_nonnegative(value: f64) -> f64 {
    if value.is_finite() && value > 0.0 { value } else { 0.0 }
}

fn finite_or_zero(value: f64) -> f64 {
    if value.is_finite() { value } else { 0.0 }
}

/// V8.1 Φ(s), derived from the exact composite map used by placement.
pub fn dominance_potential(composite: &HashMap<usize, f64>, me: usize, clamp: f64) -> f64 {
    let clamp = finite_nonnegative(clamp);
    if clamp == 0.0 {
        return 0.0;
    }
    let mine = finite_nonnegative(composite.get(&me).copied().unwrap_or(0.0));
    let strongest_opponent = composite
        .iter()
        .filter(|&(&pid, _)| pid != me)
        .map(|(_, &strength)| finite_nonnegative(strength))
        .fold(0.0, f64::max);
    ((mine + DOMINANCE_EPS) / (strongest_opponent + DOMINANCE_EPS))
        .ln()
        .clamp(-clamp, clamp)
}

/// Agent share after normalizing the placement composite over living players.
pub fn normalized_strength_share(composite: &HashMap<usize, f64>, me: usize) -> f64 {
    let mine = finite_nonnegative(composite.get(&me).copied().unwrap_or(0.0));
    let total: f64 = composite.values().map(|&value| finite_nonnegative(value)).sum();
    if total > 0.0 { (mine / total).clamp(0.0, 1.0) } else { 0.0 }
}

pub fn strength_delta_weight(
    delta: f64,
    normalized_share: f64,
    stage: usize,
    config: RewardConfig,
) -> f64 {
    if delta >= 0.0 {
        W_DELTA_GAIN
    } else if config.dominant_loss_active(stage)
        && normalized_share >= config.v81_dominance_threshold
    {
        config.v81_delta_loss_dominant
    } else {
        W_DELTA_LOSS
    }
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
mod tests {
    use super::*;
    use crate::feat::parse_ents;
    use serde_json::json;

    fn config() -> RewardConfig {
        RewardConfig {
            gamma: 0.9,
            v81_dom_coef: 0.25,
            v81_min_stage: 4,
            v81_potential_clamp: 2.0,
            v81_dominant_loss: true,
            v81_dominance_threshold: 0.55,
            v81_delta_loss_dominant: 5.25,
        }
    }

    fn composite(values: &[(usize, f64)]) -> HashMap<usize, f64> {
        values.iter().copied().collect()
    }

    #[test]
    fn potential_shaping_telescopes_and_cannot_reward_churn() {
        let phi = [0.2, 1.1, -0.4, 0.2];
        let (gamma, coefficient) = (0.9f64, 0.25);
        let mut shaper = DominanceShaper::default();
        shaper.reset(phi[0]);
        let increments: Vec<f64> = phi[1..].iter()
            .map(|&next| shaper.transition(next, gamma, coefficient))
            .collect();
        let discounted: f64 = increments.iter().enumerate()
            .map(|(t, value)| gamma.powi(t as i32) * value)
            .sum();
        let expected = coefficient
            * (-phi[0] + gamma.powi((phi.len() - 1) as i32) * phi[phi.len() - 1]);
        assert!((discounted - expected).abs() < 1e-12);
        assert_eq!(shaper.prior().to_bits(), phi[phi.len() - 1].to_bits());
    }

    #[test]
    fn potential_uses_the_strongest_alive_opponent() {
        let base = dominance_potential(&composite(&[(1, 0.6), (2, 0.2), (3, 0.4)]), 1, 10.0);
        let weaker_changed =
            dominance_potential(&composite(&[(1, 0.6), (2, 0.3), (3, 0.4)]), 1, 10.0);
        let strongest_changed =
            dominance_potential(&composite(&[(1, 0.6), (2, 0.2), (3, 0.5)]), 1, 10.0);
        assert_eq!(base.to_bits(), weaker_changed.to_bits());
        assert!(strongest_changed < base);
    }

    #[test]
    fn shaper_reset_drops_prior_episode_potential() {
        let mut shaper = DominanceShaper::default();
        shaper.reset(1.5);
        let _ = shaper.transition(-0.7, 0.9, 0.25);
        shaper.reset(0.3);
        assert_eq!(shaper.prior().to_bits(), 0.3f64.to_bits());
        assert!((shaper.transition(0.5, 0.9, 0.25) - 0.25 * (0.9 * 0.5 - 0.3)).abs() < 1e-15);
    }

    #[test]
    fn stage_gate_controls_both_v81_behaviors() {
        let cfg = config();
        assert!(!cfg.dominance_shaping_active(3));
        assert!(!cfg.dominant_loss_active(3));
        assert!(cfg.dominance_shaping_active(4));
        assert!(cfg.dominant_loss_active(4));
        assert_eq!(strength_delta_weight(-0.1, 0.9, 3, cfg), W_DELTA_LOSS);
    }

    #[test]
    fn dominant_threshold_relaxes_only_losses_at_or_above_threshold() {
        let cfg = config();
        assert_eq!(strength_delta_weight(-0.1, 0.549, 4, cfg), W_DELTA_LOSS);
        assert_eq!(strength_delta_weight(-0.1, 0.55, 4, cfg), cfg.v81_delta_loss_dominant);
        assert_eq!(strength_delta_weight(0.1, 0.99, 4, cfg), W_DELTA_GAIN);
    }

    #[test]
    fn disabled_configuration_preserves_legacy_reward_bits() {
        let mut cfg = config();
        cfg.v81_dom_coef = 0.0;
        cfg.v81_dominant_loss = false;
        let (mine, previous, tw) = (0.3125, 0.375, 0.78125);
        let delta = mine - previous;
        let legacy = W_STR * mine * tw
            + (if delta >= 0.0 { W_DELTA_GAIN } else { W_DELTA_LOSS }) * delta;
        let current = W_STR * mine * tw
            + strength_delta_weight(delta, 0.99, 10, cfg) * delta;
        assert_eq!(legacy.to_bits(), current.to_bits());
        assert!(!cfg.dominance_shaping_active(10));
    }

    #[test]
    fn potential_and_share_are_finite_at_all_edges() {
        for values in [
            composite(&[]),
            composite(&[(1, 0.0)]),
            composite(&[(1, f64::NAN), (2, f64::INFINITY)]),
            composite(&[(1, 1e300), (2, 1e-300)]),
        ] {
            let phi = dominance_potential(&values, 1, 2.0);
            let share = normalized_strength_share(&values, 1);
            assert!(phi.is_finite() && (-2.0..=2.0).contains(&phi));
            assert!(share.is_finite() && (0.0..=1.0).contains(&share));
        }
    }

    #[test]
    fn dominance_uses_exact_placement_composite_strength() {
        let ents = parse_ents(&json!({
            "players": [
                {"id": 1, "pid": "me", "troops": 100, "gold": 80, "tiles": 60, "alive": true},
                {"id": 2, "pid": "opp", "troops": 50, "gold": 20, "tiles": 40, "alive": true}
            ],
            "units": [], "attacks": [], "alliances": []
        }));
        let exact = strengths(&ents, 100);
        let expected = ((exact[&1] + DOMINANCE_EPS) / (exact[&2] + DOMINANCE_EPS)).ln();
        assert!((dominance_potential(&exact, 1, 10.0) - expected).abs() < 1e-15);
        assert_eq!(placement(&ents, 1, true, 100), (1, 2));
    }
}
