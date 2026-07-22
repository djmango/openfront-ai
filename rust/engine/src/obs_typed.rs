//! Typed RL observation builders for the native trainer path.
//!
//! Mirrors [`crate::obs::entities`] / [`crate::obs::legality`] field extraction
//! but writes `ofcore::feat::{EntsData, Legal}` directly, so `--engine native`
//! collect can skip JSON encode + `parse_ents` / `parse_legal`.

use crate::core::schemas::unit_type as ut;
use crate::game::Game;
use crate::map::TileRef;
use crate::obs::{
    borders_neutral_land, can_extend_alliance, has_shore_border, shares_border_with, LAUNCHABLE,
    STRUCTURES,
};
use ofcore::feat::{
    build_index, unit_class, AllianceE, AttackE, EntsData, Legal, PlayerE, UnitE, N_BUILD, N_NUKE,
    REGION,
};

fn nuke_rows(unit: &str) -> &'static [usize] {
    match unit {
        "Atom Bomb" => &[0, 1],
        "Hydrogen Bomb" => &[2, 3],
        "MIRV" => &[4],
        _ => &[],
    }
}

/// Typed port of [`crate::obs::entities`].
pub fn entities_typed(game: &Game) -> EntsData {
    let players: Vec<PlayerE> = game
        .all_players()
        .iter()
        .map(|p| {
            let sid = p.small_id;
            PlayerE {
                id: sid as usize,
                pid: p.id.clone(),
                troops: p.troops as f64,
                gold: p.gold as f64,
                tiles: p.tiles_owned as f64,
                alive: p.alive,
                traitor: game.is_traitor(sid),
                embargoes: p.embargoes.keys().map(|id| id as usize).collect(),
                relations: p
                    .relations
                    .iter()
                    .map(|(other, &v)| (other as usize, v))
                    .collect(),
                reqs_in: game.incoming_alliance_requests(sid).len(),
                reqs_out: game.outgoing_alliance_requests(sid).len(),
                targets: game
                    .player_targets(sid)
                    .into_iter()
                    .map(|id| id as usize)
                    .collect(),
                troop_income: if p.alive {
                    game.troop_increase_rate_raw_for(sid)
                } else {
                    0.0
                },
                gold_income: if p.alive {
                    game.wire.gold_addition_rate(p.player_type) as f64
                } else {
                    0.0
                },
                doomsday: false,
                doomsday_ticks: 0.0,
            }
        })
        .collect();

    let mut alliances: Vec<AllianceE> = Vec::new();
    let mut seen: std::collections::HashSet<(u16, u16)> = std::collections::HashSet::new();
    for p in game.all_players() {
        if !p.alive {
            continue;
        }
        for al in game.player_alliances(p.small_id) {
            let (x, y) = (al.requestor_small_id, al.recipient_small_id);
            let key = if x < y { (x, y) } else { (y, x) };
            if seen.insert(key) {
                alliances.push(AllianceE(x as usize, y as usize, al.expires_at as i64));
            }
        }
    }

    let transport_troops: std::collections::HashMap<i32, f64> = game
        .live_transports()
        .filter_map(|t| t.unit_id().map(|id| (id, t.carried_troops())))
        .collect();
    let unit_capacity: usize = game
        .all_players()
        .iter()
        .filter(|p| p.alive)
        .map(|p| p.units.len())
        .sum();
    let mut units: Vec<UnitE> = Vec::with_capacity(unit_capacity);
    for p in game.all_players() {
        if !p.alive {
            continue;
        }
        for u in &p.units {
            let Some(class) = unit_class(&u.unit_type) else {
                continue;
            };
            let tile = u.tile as TileRef;
            let x = game.x(tile) as i64;
            let y = game.y(tile) as i64;
            let (has_target, tgx, tgy) = match u.target_tile {
                Some(t) => (
                    true,
                    game.x(t) as i64 / REGION as i64,
                    game.y(t) as i64 / REGION as i64,
                ),
                None => (false, -1, -1),
            };
            let troops = match u.unit_type.as_str() {
                ut::TRANSPORT => transport_troops.get(&u.id).copied().unwrap_or(0.0),
                _ => 0.0,
            };
            units.push(UnitE {
                class,
                owner: p.small_id as usize,
                uid: u.id as i64,
                x,
                y,
                gx: x / REGION as i64,
                gy: y / REGION as i64,
                tgx,
                tgy,
                has_target,
                constructing: u.under_construction,
                level: {
                    let level = u.level as f64;
                    if level != 0.0 { level } else { 1.0 }
                },
                health: None,
                max_health: None,
                troops: troops.round(),
                sam_lock: u.targeted_by_sam,
                cooldown: game.unit_is_in_cooldown(p.small_id, u.id),
                station: u.has_train_station,
            });
        }
    }

    let mut attacks: Vec<AttackE> = Vec::new();
    for a in game.live_attacks() {
        let owner_alive = game
            .player_by_small_id(a.owner_small_id())
            .is_some_and(|p| p.alive);
        if !owner_alive {
            continue;
        }
        let to = a.target_small_id();
        attacks.push(AttackE {
            aid: a.attack_id().to_string(),
            from: a.owner_small_id() as usize,
            to: if to == game.terra_nullius_id() {
                0
            } else {
                to as usize
            },
            troops: a.troops().round(),
            retreating: a.retreating(),
            src_x: a.source_tile().map(|t| game.x(t) as i64),
            src_y: a.source_tile().map(|t| game.y(t) as i64),
        });
    }

    EntsData {
        players,
        units,
        attacks,
        alliances,
        doomsday_enabled: false,
    }
}

/// Typed port of [`crate::obs::legality`]'s `actions` object (empty when
/// the agent is dead / missing — same as JSON `actions: {}`).
pub fn legality_typed(game: &Game, client_id: &str) -> Legal {
    let agent = match game.player_by_client_id(client_id) {
        Some(p) if p.alive => p,
        _ => return Legal::default(),
    };
    let sid = agent.small_id;
    let gold = agent.gold;

    let others: Vec<_> = game
        .all_players()
        .iter()
        .filter(|p| p.small_id != sid && p.alive)
        .collect();

    let mut build_mask = [0.0f32; N_BUILD];
    let mut nuke_mask = [0.0f32; N_NUKE];
    for t in STRUCTURES.iter().chain(LAUNCHABLE.iter()).chain([ut::WARSHIP].iter()) {
        if gold >= game.structure_cost(sid, t) {
            if let Some(i) = build_index(t) {
                build_mask[i as usize] = 1.0;
            }
            for &i in nuke_rows(t) {
                nuke_mask[i] = 1.0;
            }
        }
    }

    let attackable: Vec<usize> = others
        .iter()
        .filter(|p| shares_border_with(game, agent, p.small_id) && !game.is_friendly(sid, p.small_id))
        .map(|p| p.small_id as usize)
        .collect();

    let has_silo = !game.is_spawn_immunity_active()
        && agent.units.iter().any(|u| {
            u.unit_type == ut::MISSILE_SILO
                && !u.under_construction
                && !game.unit_is_in_cooldown(sid, u.id)
        });

    let upgradable: Vec<usize> = agent
        .units
        .iter()
        .filter(|u| {
            game.can_upgrade_unit(sid, u.id) && gold >= game.structure_cost(sid, &u.unit_type)
        })
        .map(|u| u.id as usize)
        .collect();

    let deletable: Vec<usize> = if game.can_delete_unit(sid) {
        agent
            .units
            .iter()
            .filter(|u| {
                let t = u.tile as TileRef;
                game.is_land(t) && game.map.owner_id(t) == sid
            })
            .map(|u| u.id as usize)
            .collect()
    } else {
        Vec::new()
    };

    Legal {
        present: true,
        attackable,
        alliance_requestable: others
            .iter()
            .filter(|p| game.can_send_alliance_request(sid, p.small_id))
            .map(|p| p.small_id as usize)
            .collect(),
        alliance_rejectable: game
            .incoming_alliance_requests(sid)
            .iter()
            .map(|r| r.requestor_small_id as usize)
            .collect(),
        breakable: game
            .player_alliances(sid)
            .iter()
            .map(|al| {
                if al.requestor_small_id == sid {
                    al.recipient_small_id as usize
                } else {
                    al.requestor_small_id as usize
                }
            })
            .collect(),
        donatable_gold: others
            .iter()
            .filter(|p| game.can_donate_gold(sid, p.small_id))
            .map(|p| p.small_id as usize)
            .collect(),
        donatable_troops: others
            .iter()
            .filter(|p| game.can_donate_troops(sid, p.small_id))
            .map(|p| p.small_id as usize)
            .collect(),
        embargoable: others
            .iter()
            .filter(|p| !game.has_embargo_against(sid, p.small_id))
            .map(|p| p.small_id as usize)
            .collect(),
        stop_embargoable: agent.embargoes.keys().map(|id| id as usize).collect(),
        targetable: others
            .iter()
            .filter(|p| game.can_target(sid, p.small_id))
            .map(|p| p.small_id as usize)
            .collect(),
        extendable: game
            .player_alliances(sid)
            .iter()
            .map(|al| {
                if al.requestor_small_id == sid {
                    al.recipient_small_id
                } else {
                    al.requestor_small_id
                }
            })
            .filter(|other| can_extend_alliance(game, agent, *other))
            .map(|id| id as usize)
            .collect(),
        can_expand: borders_neutral_land(game, agent),
        can_boat: game.unit_count(sid, ut::TRANSPORT) < game.wire.boat_max_number()
            && has_shore_border(game, agent),
        troops: agent.troops as f64,
        gold: gold as f64,
        build_mask,
        nuke_mask,
        has_silo,
        attacks: game
            .live_attacks()
            .filter(|a| a.owner_small_id() == sid)
            .map(|a| a.attack_id().to_string())
            .collect(),
        upgradable,
        warships: agent
            .units
            .iter()
            .filter(|u| u.unit_type == ut::WARSHIP)
            .map(|u| u.id as usize)
            .collect(),
        boats: agent
            .units
            .iter()
            .filter(|u| u.unit_type == ut::TRANSPORT)
            .map(|u| u.id as usize)
            .collect(),
        deletable,
    }
}
