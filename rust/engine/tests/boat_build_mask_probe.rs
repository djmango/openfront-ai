//! Probe: how often "legal" boat/build actions become wasted intents.
//!
//! Root-cause hypotheses (verified by numbers below):
//! 1. Boat: `canBoat` only checks shore-border + boat-cap. The policy/translator
//!    then pick region tiles; many produce empty intents or engine-invalid
//!    destinations (no water path / no target shore within BFS).
//! 2. Build: `buildableTypes` is gold-only. Warship is listed with zero valid
//!    spawn sites until a Port exists. City reward (+0.01) equals waste (-0.01).

use ofcore::feat::{EntsData, REGION, GW_MAX};
use ofcore::translate::IntentTranslator;
use openfront_engine::core::schemas::unit_type as ut;
use openfront_engine::execution::nation_structures::resolve_structure_spawn_tile;
use openfront_engine::obs::{has_shore_border, legality};
use openfront_engine::rl::RlSession;
use openfront_engine::session::AGENT_CLIENT_ID;
use openfront_engine::spatial::can_build_transport_ship;
use serde_json::{json, Value};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root")
}

fn find_spawn_tile(session: &RlSession) -> u64 {
    let w = session.game.width();
    let h = session.game.height();
    for y in 0..h {
        for x in 0..w {
            let t = y * w + x;
            if session.game.is_land(t)
                && !session.game.is_impassable(t)
                && session.game.map.owner_id(t) == 0
            {
                return t as u64;
            }
        }
    }
    panic!("no free land spawn tile");
}

fn agent_sid(session: &RlSession) -> u16 {
    session
        .game
        .player_by_client_id(AGENT_CLIENT_ID)
        .expect("agent")
        .small_id
}

fn setup_coastal_agent(session: &mut RlSession, terrain: &[u8]) -> (u16, IntentTranslator) {
    let spawn = find_spawn_tile(session);
    let _ = session.step(&[json!({"type": "spawn", "tile": spawn})], 5);
    let sid = agent_sid(session);

    let w = session.game.width();
    let h = session.game.height();
    let shore = (0..h)
        .flat_map(|y| (0..w).map(move |x| y * w + x))
        .find(|&t| {
            session.game.is_shore(t)
                && session.game.is_land(t)
                && !session.game.is_impassable(t)
                && session.game.map.owner_id(t) != sid
        })
        .expect("Onion must have shore land");

    let sx = shore % w;
    let sy = shore / w;
    let mut gifted = 0u32;
    for dy in 0..24i32 {
        for dx in 0..24i32 {
            let x = sx as i32 + dx - 8;
            let y = sy as i32 + dy - 8;
            if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
                continue;
            }
            let t = y as u32 * w + x as u32;
            if session.game.is_land(t) && !session.game.is_impassable(t) {
                session.game.conquer(sid, t);
                gifted += 1;
            }
        }
    }
    assert!(gifted > 50, "failed to gift coastal territory; gifted={gifted}");

    {
        let p = session.game.player_by_small_id_mut(sid).expect("agent");
        p.gold = 1_000_000;
        p.troops = 50_000;
    }
    let _ = session.step(&[], 1);

    let agent = session
        .game
        .player_by_client_id(AGENT_CLIENT_ID)
        .expect("agent")
        .clone();
    assert!(
        has_shore_border(&session.game, &agent),
        "gifted coastal blob must produce a shore border"
    );

    let hr = (h as usize) - (h as usize) % REGION;
    let wr = (w as usize) - (w as usize) % REGION;
    let tr = IntentTranslator::new(terrain, w as usize, hr, wr);
    (sid, tr)
}

fn owners_grid(session: &RlSession, hr: usize, wr: usize) -> Vec<i64> {
    let width = session.game.width() as usize;
    let mut owners = vec![0i64; hr * wr];
    for y in 0..hr {
        for x in 0..wr {
            owners[y * wr + x] = session.game.map.owner_id((y * width + x) as u32) as i64;
        }
    }
    owners
}

#[test]
fn boat_canboat_true_but_random_regions_mostly_fail_translate_or_engine() {
    let root = repo_root();
    let (mut session, _head, terrain) =
        RlSession::reset(&root, "Onion", "boat-mask-probe", 0, "Easy", Value::from(0))
            .expect("reset");
    let (sid, mut tr) = setup_coastal_agent(&mut session, &terrain);

    let legal = legality(&session.game, AGENT_CLIENT_ID);
    assert!(
        legal
            .pointer("/actions/canBoat")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        "canBoat must be true for coastal setup"
    );

    let hr = tr.hr;
    let wr = tr.wr;
    let gh = hr / REGION;
    let gw = wr / REGION;
    let owners = owners_grid(&session, hr, wr);
    let ents = EntsData {
        players: Vec::new(),
        units: Vec::new(),
        attacks: Vec::new(),
        alliances: Vec::new(),
        doomsday_enabled: false,
    };

    let mut regions = 0u32;
    let mut empty_intent = 0u32;
    let mut engine_reject = 0u32;
    let mut engine_accept = 0u32;
    // Uniform over all coarse regions — what an untrained coarse head samples
    // when legal_tile is all-ones and there is no coarse_has_land prune.
    for gy in 0..gh {
        for gx in 0..gw {
            regions += 1;
            let region = gy as i64 * GW_MAX + gx as i64;
            match tr.boat_tile(region, &owners, sid as i64, &ents) {
                None => empty_intent += 1,
                Some(tile) => {
                    if can_build_transport_ship(&mut session.game, sid, tile as u32).is_some() {
                        engine_accept += 1;
                    } else {
                        engine_reject += 1;
                    }
                }
            }
        }
    }

    let fail_rate = (empty_intent + engine_reject) as f64 / regions as f64;
    let empty_rate = empty_intent as f64 / regions as f64;
    let reject_rate = engine_reject as f64 / regions as f64;
    eprintln!(
        "[boat-region] regions={regions} empty_intent={empty_intent} ({empty_rate:.3}) \
         engine_reject={engine_reject} ({reject_rate:.3}) engine_accept={engine_accept} \
         fail_rate={fail_rate:.3}"
    );

    // span=2 lets boat_tile reach neighboring land, so pure-empty is uncommon;
    // engine reject of translated tiles is the bulk failure mode.
    assert!(
        reject_rate > 0.25,
        "expected substantial engine rejects among translated boats; reject_rate={reject_rate:.3}"
    );
    assert!(
        fail_rate > 0.30,
        "random coarse boat regions should fail often; fail_rate={fail_rate:.3}"
    );
    assert!(
        engine_accept > 0,
        "sanity: some regions near the gifted coast must still be valid"
    );
    // Live watch evidence: death episode had 523 boat decisions / 10 boat
    // intents (97.7% empty). That stickiness is policy+reward, not map
    // geometry alone — geometry still fails >30% of uniform region picks.
    let _ = empty_rate;
}

#[test]
fn warship_listed_buildable_with_zero_valid_sites_without_port() {
    let root = repo_root();
    let (mut session, _head, terrain) =
        RlSession::reset(&root, "Onion", "build-mask-probe", 0, "Easy", Value::from(0))
            .expect("reset");
    let (sid, mut tr) = setup_coastal_agent(&mut session, &terrain);

    let legal = legality(&session.game, AGENT_CLIENT_ID);
    let buildable: Vec<String> = legal
        .pointer("/actions/buildableTypes")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    assert!(
        buildable.iter().any(|t| t == "Warship"),
        "Warship is gold-gated into buildableTypes; got {buildable:?}"
    );

    let hr = tr.hr;
    let wr = tr.wr;
    let gh = hr / REGION;
    let gw = wr / REGION;
    let owners = owners_grid(&session, hr, wr);

    let mut warship_translate = 0u32;
    let mut warship_engine_ok = 0u32;
    let mut city_translate = 0u32;
    let mut city_engine_ok = 0u32;
    let mut regions = 0u32;

    for gy in 0..gh {
        for gx in 0..gw {
            regions += 1;
            let region = gy as i64 * GW_MAX + gx as i64;
            if let Some(tile) = tr.build_tile(region, &owners, sid as i64, "Warship") {
                warship_translate += 1;
                // Engine warship_spawn: water + completed own port in component.
                let t = tile as u32;
                if !session.game.is_land(t) {
                    if let Some(comp) = session.game.get_water_component(t) {
                        let has_port = session
                            .game
                            .player_by_small_id(sid)
                            .map(|p| {
                                p.units.iter().any(|u| {
                                    u.unit_type == ut::PORT
                                        && !u.under_construction
                                        && session
                                            .game
                                            .has_water_component(u.tile as u32, comp)
                                })
                            })
                            .unwrap_or(false);
                        if has_port {
                            warship_engine_ok += 1;
                        }
                    }
                }
            }
            if let Some(tile) = tr.build_tile(region, &owners, sid as i64, "City") {
                city_translate += 1;
                if resolve_structure_spawn_tile(&session.game, sid, "City", tile as u32).is_some()
                {
                    city_engine_ok += 1;
                }
            }
        }
    }

    let warship_engine_frac = warship_engine_ok as f64 / warship_translate.max(1) as f64;
    let city_engine_frac = city_engine_ok as f64 / city_translate.max(1) as f64;
    eprintln!(
        "[build-region] regions={regions} warship_translate={warship_translate} \
         warship_engine_ok={warship_engine_ok} ({warship_engine_frac:.3}) \
         city_translate={city_translate} city_engine_ok={city_engine_ok} ({city_engine_frac:.3}) \
         buildable={buildable:?}"
    );

    assert!(
        warship_translate > 0,
        "translator happily picks water for Warship across ocean regions"
    );
    assert_eq!(
        warship_engine_ok, 0,
        "without a Port, every translated Warship site must fail engine checks"
    );
    assert!(
        city_translate > 0 && city_engine_ok > 0,
        "City should have some valid coastal sites after territory gift"
    );
}

#[test]
fn empty_boat_was_net_positive_vs_waste_while_build_was_neutral() {
    // Live watch (ppo_v10 death ep): 523 boat decisions → 10 boat intents
    // (97.7% empty). The old has_target:=tile_region.is_some() made each
    // empty boat +0.005 EV after waste, locking the recurrent policy.
    let cfg = ofcore::curriculum::RewardConfig {
        gamma: 0.999,
        v81_dom_coef: 0.25,
        v81_min_stage: 0,
        v81_potential_clamp: 2.0,
        v81_dominant_loss: true,
        v81_dominance_threshold: 0.3,
        v81_delta_loss_dominant: 5.0,
        v81_churn_coef: 0.05,
        v81_churn_window: 16,
        v81_churn_min_stage: 0,
        v83_close_coef: 4.0,
        v83_churn_coef: 0.06,
        v84_boat_useful: 0.15,
        v84_boat_destroyed: -0.20,
        v84_boat_cancelled: -0.03,
        v84_boat_own_shore: -0.05,
        v84_boat_min_stage: 0,
        v84_tempo_coef: 0.015,
        v84_tempo_min_stage: 0,
        v84_fast_win_coef: 40.0,
        v85_tempo_share_threshold: 0.30,
        v85_extra_win_bonus: 200.0,
        v85_embargo_bad_stop: -0.15,
        v85_embargo_good_stop: 0.02,
        v85_embargo_min_stage: 0,
        v85_premature_retreat: -0.03,
        v85_thrash_reengage: -0.03,
        v85_combat_min_stage: 0,
        v86_delta_loss: 5.5,
        v86_attack_symmetric_loss: true,
        v86_skip_combat_churn: true,
        v86_death_penalty: 3.0,
        v10_survival_coef: 0.01,
        v10_diplo_panic: 0.08,
        v10_diplo_panic_share: 0.35,
        v10_diplo_panic_tick_frac: 0.55,
        v10_combat_action: 0.02,
        v10_attack_commit: 0.04,
        v10_attack_switch: 0.02,
        v10_timeout_closeout: 20.0,
        v10_closeout_entry: 25.0,
    };
    let empty_boat = ofcore::curriculum::v10_empty_action_net_reward(ofcore::feat::A_BOAT, cfg);
    let empty_build = ofcore::curriculum::v10_empty_action_net_reward(ofcore::feat::A_BUILD, cfg);
    assert!((empty_boat - 0.005).abs() < 1e-9, "empty boat net={empty_boat}");
    assert_eq!(empty_build, 0.0);
    // Correct gate: no emitted intent → no combat bonus → pure -W_WASTE.
    assert_eq!(
        ofcore::curriculum::v10_combat_action_bonus(ofcore::feat::A_BOAT, false, cfg),
        0.0
    );
    assert_eq!(
        ofcore::curriculum::v10_combat_action_bonus(ofcore::feat::A_BUILD, false, cfg),
        0.0
    );
}
