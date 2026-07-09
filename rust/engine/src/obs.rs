//! RL observation head - full port of `bridge/common.ts::buildObsParts`'s
//! head (which delegates to `openfront/src/client/webbot/obsCore.ts`).
//! Field names, value shapes, and enum strings must match TS exactly: the
//! trainer's featurizer (`ofcore::feat`) and the obs-diff parity harness
//! both consume this JSON.

use crate::core::schemas::unit_type as ut;
use crate::game::{Game, Player, PlayerType};
use crate::map::TileRef;
use serde_json::{json, Value};

/// TS `obsCore.STRUCTURES`.
pub const STRUCTURES: [&str; 6] = [
    ut::CITY,
    ut::PORT,
    ut::DEFENSE_POST,
    ut::MISSILE_SILO,
    ut::SAM_LAUNCHER,
    ut::FACTORY,
];
/// TS `obsCore.LAUNCHABLE`.
pub const LAUNCHABLE: [&str; 3] = [ut::ATOM_BOMB, ut::HYDROGEN_BOMB, ut::MIRV];

fn player_type_str(t: PlayerType) -> &'static str {
    // TS `PlayerType` enum VALUES ("HUMAN"), not the variant names.
    match t {
        PlayerType::Human => "HUMAN",
        PlayerType::Bot => "BOT",
        PlayerType::Nation => "NATION",
    }
}

/// `winner` mirrors env.ts's step(): the TS wire tuple only on the step the
/// win update fired, `Null` otherwise (pass `Null` on reset).
pub fn build_obs_head(game: &Game, client_id: &str, winner: Value) -> Value {
    let agent = game.player_by_client_id(client_id);
    json!({
        "tick": game.ticks(),
        "width": game.width(),
        "height": game.height(),
        "spawnPhase": game.in_spawn_phase(),
        "winner": winner,
        "me": agent.map(|p| p.small_id as i32).unwrap_or(-1),
        "alive": agent.map(|p| p.alive).unwrap_or(false),
        "entities": entities(game),
        "legal": legality(game, client_id),
    })
}

/// TS `GameImpl.makeWinner()` tuple: `["player", clientID]` /
/// `["nation", name]`. RL only runs FFA so the team arm is not ported.
pub fn winner_value(game: &Game) -> Value {
    match &game.winner {
        None => Value::Null,
        Some(pid) => match game.player_by_id(pid) {
            None => Value::Null,
            Some(p) if p.client_id.is_empty() => json!(["nation", p.name]),
            Some(p) => json!(["player", p.client_id]),
        },
    }
}

/// TS `obsCore.entities()`.
pub fn entities(game: &Game) -> Value {
    let players: Vec<Value> = game
        .all_players()
        .iter()
        .map(|p| {
            let sid = p.small_id;
            json!({
                "id": sid,
                "pid": p.id,
                "type": player_type_str(p.player_type),
                "troops": p.troops,
                "gold": p.gold.to_string(),
                "tiles": p.tiles_owned,
                "alive": p.alive,
                "traitor": game.is_traitor(sid),
                "embargoes": p.embargoes.keys().collect::<Vec<_>>(),
                "reqsIn": game
                    .incoming_alliance_requests(sid)
                    .iter()
                    .map(|r| r.requestor_small_id)
                    .collect::<Vec<_>>(),
                "reqsOut": game.outgoing_alliance_requests(sid),
                "targets": game.player_targets(sid),
                "troopIncome": if p.alive { game.troop_increase_rate_raw_for(sid) } else { 0.0 },
                "goldIncome": if p.alive {
                    game.wire.gold_addition_rate(p.player_type).to_string()
                } else {
                    "0".to_string()
                },
                // Doomsday clock defaults to disabled in TS and RL configs
                // never enable it; emit the disabled constants.
                "doomsday": false,
                "doomsdayTicks": 0,
            })
        })
        .collect();

    // TS: for p of players() (alive) { for a of p.alliances() } with dedup -
    // an alliance appears iff it is active and at least one endpoint is alive.
    let mut alliances: Vec<Value> = Vec::new();
    let mut seen: std::collections::HashSet<(u16, u16)> = std::collections::HashSet::new();
    for p in game.all_players() {
        if !p.alive {
            continue;
        }
        for al in game.player_alliances(p.small_id) {
            let (x, y) = (al.requestor_small_id, al.recipient_small_id);
            let key = if x < y { (x, y) } else { (y, x) };
            if seen.insert(key) {
                alliances.push(json!([x, y, al.expires_at]));
            }
        }
    }

    // TS: game.players() (alive only) flatMap units / outgoingAttacks.
    let transport_troops: std::collections::HashMap<i32, f64> = game
        .live_transports()
        .filter_map(|t| t.unit_id().map(|id| (id, t.carried_troops())))
        .collect();
    let mut units: Vec<Value> = Vec::new();
    for p in game.all_players() {
        if !p.alive {
            continue;
        }
        for u in &p.units {
            let tile = u.tile as TileRef;
            let troops = match u.unit_type.as_str() {
                ut::TRANSPORT => transport_troops.get(&u.id).copied().unwrap_or(0.0),
                _ => 0.0,
            };
            units.push(json!({
                "uid": u.id,
                "type": u.unit_type,
                "owner": p.small_id,
                "x": game.x(tile),
                "y": game.y(tile),
                "tx": u.target_tile.map(|t| game.x(t)),
                "ty": u.target_tile.map(|t| game.y(t)),
                "samLock": u.targeted_by_sam,
                "level": u.level,
                // Unit health is not modeled natively yet (warship combat
                // unported); TS emits null for units without health.
                "health": Value::Null,
                "maxHealth": Value::Null,
                "constructing": u.under_construction,
                "cooldown": game.unit_is_in_cooldown(p.small_id, u.id),
                "station": u.has_train_station,
                "troops": troops.round() as i64,
            }));
        }
    }

    let mut attacks: Vec<Value> = Vec::new();
    for a in game.live_attacks() {
        let owner_alive = game
            .player_by_small_id(a.owner_small_id())
            .is_some_and(|p| p.alive);
        if !owner_alive {
            continue;
        }
        let to = a.target_small_id();
        attacks.push(json!({
            "aid": a.attack_id(),
            "from": a.owner_small_id(),
            "to": if to == game.terra_nullius_id() { 0 } else { to },
            "troops": a.troops().round() as i64,
            "retreating": a.retreating(),
            "srcX": a.source_tile().map(|t| game.x(t)),
            "srcY": a.source_tile().map(|t| game.y(t)),
        }));
    }

    json!({
        "players": players,
        "alliances": alliances,
        "units": units,
        "attacks": attacks,
        "doomsdayEnabled": false,
    })
}

/// TS `obsCore.hasShoreBorder()`.
pub fn has_shore_border(game: &Game, player: &Player) -> bool {
    player.border_tiles.iter().any(|t| game.is_shore(t))
}

/// TS `obsCore.bordersNeutralLand()`.
pub fn borders_neutral_land(game: &Game, player: &Player) -> bool {
    for t in player.border_tiles.iter() {
        let mut found = false;
        game.map.for_each_neighbor4(t, |n| {
            if game.is_land(n) && !game.has_owner(n) {
                found = true;
            }
        });
        if found {
            return true;
        }
    }
    false
}

/// TS `PlayerImpl.sharesBorderWith(other)`.
pub fn shares_border_with(game: &Game, player: &Player, other_small_id: u16) -> bool {
    for t in player.border_tiles.iter() {
        let mut found = false;
        game.map.for_each_neighbor4(t, |n| {
            if game.map.owner_id(n) == other_small_id {
                found = true;
            }
        });
        if found {
            return true;
        }
    }
    false
}

/// TS `Config.allianceExtensionPromptOffset()`.
const ALLIANCE_EXTENSION_PROMPT_OFFSET: u32 = 300;

/// TS `PlayerImpl.allianceInfo(other)?.canExtend === true`.
pub fn can_extend_alliance(game: &Game, agent: &Player, other_small_id: u16) -> bool {
    let Some(other) = game.player_by_small_id(other_small_id) else {
        return false;
    };
    if !agent.alive || !other.alive || agent.is_disconnected || other.is_disconnected {
        return false;
    }
    for al in game.player_alliances(agent.small_id) {
        let is_this = al.requestor_small_id == other_small_id
            || al.recipient_small_id == other_small_id;
        if !is_this {
            continue;
        }
        let in_window = al.expires_at <= game.ticks() + ALLIANCE_EXTENSION_PROMPT_OFFSET;
        let agent_agreed = if al.requestor_small_id == agent.small_id {
            al.extension_requested_requestor
        } else {
            al.extension_requested_recipient
        };
        return in_window && !agent_agreed;
    }
    false
}

/// TS `obsCore.legality()`.
pub fn legality(game: &Game, client_id: &str) -> Value {
    let agent = match game.player_by_client_id(client_id) {
        Some(p) if p.alive => p,
        _ => return json!({ "spawn": game.in_spawn_phase(), "actions": {} }),
    };
    let sid = agent.small_id;
    let gold = agent.gold;

    let others: Vec<&Player> = game
        .all_players()
        .iter()
        .filter(|p| p.small_id != sid && p.alive)
        .collect();

    let mut buildable: Vec<&str> = Vec::new();
    for t in STRUCTURES.iter().chain(LAUNCHABLE.iter()).chain([ut::WARSHIP].iter()) {
        if gold >= game.structure_cost(sid, t) {
            buildable.push(t);
        }
    }

    let attackable: Vec<u16> = others
        .iter()
        .filter(|p| shares_border_with(game, agent, p.small_id) && !game.is_friendly(sid, p.small_id))
        .map(|p| p.small_id)
        .collect();

    let has_silo = !game.is_spawn_immunity_active()
        && agent.units.iter().any(|u| {
            u.unit_type == ut::MISSILE_SILO
                && !u.under_construction
                && !game.unit_is_in_cooldown(sid, u.id)
        });

    let upgradable: Vec<i32> = agent
        .units
        .iter()
        .filter(|u| {
            game.can_upgrade_unit(sid, u.id) && gold >= game.structure_cost(sid, &u.unit_type)
        })
        .map(|u| u.id)
        .collect();

    let deletable: Vec<i32> = if game.can_delete_unit(sid) {
        agent
            .units
            .iter()
            .filter(|u| {
                let t = u.tile as TileRef;
                game.is_land(t) && game.map.owner_id(t) == sid
            })
            .map(|u| u.id)
            .collect()
    } else {
        Vec::new()
    };

    json!({
        "spawn": game.in_spawn_phase(),
        "actions": {
            "attackable": attackable,
            "allianceRequestable": others
                .iter()
                .filter(|p| game.can_send_alliance_request(sid, p.small_id))
                .map(|p| p.small_id)
                .collect::<Vec<_>>(),
            "allianceRejectable": game
                .incoming_alliance_requests(sid)
                .iter()
                .map(|r| r.requestor_small_id)
                .collect::<Vec<_>>(),
            "breakable": game
                .player_alliances(sid)
                .iter()
                .map(|al| if al.requestor_small_id == sid {
                    al.recipient_small_id
                } else {
                    al.requestor_small_id
                })
                .collect::<Vec<_>>(),
            "targetable": others
                .iter()
                .filter(|p| game.can_target(sid, p.small_id))
                .map(|p| p.small_id)
                .collect::<Vec<_>>(),
            "donatableGold": others
                .iter()
                .filter(|p| game.can_donate_gold(sid, p.small_id))
                .map(|p| p.small_id)
                .collect::<Vec<_>>(),
            "donatableTroops": others
                .iter()
                .filter(|p| game.can_donate_troops(sid, p.small_id))
                .map(|p| p.small_id)
                .collect::<Vec<_>>(),
            "embargoable": others
                .iter()
                .filter(|p| !game.has_embargo_against(sid, p.small_id))
                .map(|p| p.small_id)
                .collect::<Vec<_>>(),
            "buildableTypes": buildable,
            "canBoat": game.unit_count(sid, ut::TRANSPORT) < game.wire.boat_max_number()
                && has_shore_border(game, agent),
            "canExpand": borders_neutral_land(game, agent),
            "hasSilo": has_silo,
            "troops": agent.troops,
            "gold": gold.to_string(),
            "attacks": game
                .live_attacks()
                .filter(|a| a.owner_small_id() == sid)
                .map(|a| a.attack_id().to_string())
                .collect::<Vec<_>>(),
            "boats": agent
                .units
                .iter()
                .filter(|u| u.unit_type == ut::TRANSPORT)
                .map(|u| u.id)
                .collect::<Vec<_>>(),
            "warships": agent
                .units
                .iter()
                .filter(|u| u.unit_type == ut::WARSHIP)
                .map(|u| u.id)
                .collect::<Vec<_>>(),
            "upgradable": upgradable,
            "deletable": deletable,
            "stopEmbargoable": agent.embargoes.keys().collect::<Vec<_>>(),
            "extendable": game
                .player_alliances(sid)
                .iter()
                .map(|al| if al.requestor_small_id == sid {
                    al.recipient_small_id
                } else {
                    al.requestor_small_id
                })
                .filter(|other| can_extend_alliance(game, agent, *other))
                .collect::<Vec<_>>(),
        },
    })
}

pub fn tile_bytes_le(game: &Game) -> Vec<u8> {
    let buf = game.tile_state_buffer();
    let mut out = Vec::with_capacity(buf.len() * 2);
    for &v in buf {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}
