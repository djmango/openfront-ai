use super::{
    AllianceExtensionExecution, AllianceRejectExecution, AllianceRequestExecution,
    BoatRetreatExecution, BreakAllianceExecution, ConstructionExecution, DonateGoldExecution,
    DonateTroopsExecution, EmbargoAllExecution, EmbargoExecution, ExecEnum,
    MarkDisconnectedExecution, NoOpExecution, RetreatExecution, SpawnExecution,
    TargetPlayerExecution, TransportShipExecution, UpgradeStructureExecution,
};
use crate::execution::AttackExecution;
use crate::game::{Game, PlayerInfo};
use crate::record::StampedIntent;
use serde_json::Value;

fn parse_troops(fields: &Value) -> f64 {
    fields
        .get("troops")
        .and_then(|v| v.as_f64().or_else(|| v.as_i64().map(|n| n as f64)))
        .unwrap_or(0.0)
}

fn parse_target_id(fields: &Value) -> Option<String> {
    match fields.get("targetID") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s.clone()),
        Some(v) => v.as_str().map(|s| s.to_string()),
    }
}

pub fn intent_to_execution(game: &Game, game_id: &str, intent: &StampedIntent) -> ExecEnum {
    let client_id = &intent.client_id;
    match intent.intent_type.as_str() {
        "spawn" => {
            let tile = intent.fields.get("tile").and_then(Value::as_u64).unwrap_or(0) as u32;
            if let Some(p) = game.player_by_client_id(client_id) {
                let info = PlayerInfo {
                    name: p.client_id.clone(),
                    player_type: p.player_type,
                    client_id: Some(client_id.clone()),
                    id: p.id.clone(),
                    clan_tag: None,
                    friends: Vec::new(),
                    team: p.team.clone(),
                };
                ExecEnum::Spawn(SpawnExecution::new(game_id.to_string(), info, Some(tile)))
            } else {
                ExecEnum::NoOp(NoOpExecution)
            }
        }
        "attack" => ExecEnum::NoOp(NoOpExecution),
        "boat" => {
            let tile = intent.fields.get("dst").and_then(Value::as_u64).unwrap_or(0) as u32;
            let troops = parse_troops(&intent.fields);
            if let Some(p) = game.player_by_client_id(client_id) {
                ExecEnum::TransportShip(TransportShipExecution::new(
                    p.small_id,
                    tile,
                    troops,
                ))
            } else {
                ExecEnum::NoOp(NoOpExecution)
            }
        }
        "cancel_boat" => {
            let unit_id = intent
                .fields
                .get("unitID")
                .and_then(|v| v.as_i64().or_else(|| v.as_u64().map(|n| n as i64)))
                .unwrap_or(0) as i32;
            if let Some(p) = game.player_by_client_id(client_id) {
                ExecEnum::BoatRetreat(BoatRetreatExecution::new(p.small_id, unit_id))
            } else {
                ExecEnum::NoOp(NoOpExecution)
            }
        }
        "mark_disconnected" => {
            let disconnected = intent
                .fields
                .get("isDisconnected")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            if let Some(p) = game.player_by_client_id(client_id) {
                ExecEnum::MarkDisconnected(MarkDisconnectedExecution::new(p.small_id, disconnected))
            } else {
                ExecEnum::NoOp(NoOpExecution)
            }
        }
        "build_unit" => {
            let tile = intent.fields.get("tile").and_then(Value::as_u64).unwrap_or(0) as u32;
            let unit = intent
                .fields
                .get("unit")
                .and_then(Value::as_str)
                .unwrap_or("");
            // TS `NukeExecution` constructor default param - `undefined` (field absent) means `true`.
            let rocket_direction_up = intent
                .fields
                .get("rocketDirectionUp")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            if let Some(p) = game.player_by_client_id(client_id) {
                ExecEnum::Construction(ConstructionExecution::new(
                    p.small_id,
                    unit,
                    tile,
                    rocket_direction_up,
                ))
            } else {
                ExecEnum::NoOp(NoOpExecution)
            }
        }
        "upgrade_structure" => {
            let unit_id = intent
                .fields
                .get("unitId")
                .and_then(|v| v.as_i64().or_else(|| v.as_u64().map(|n| n as i64)))
                .unwrap_or(0) as i32;
            if let Some(p) = game.player_by_client_id(client_id) {
                ExecEnum::UpgradeStructure(UpgradeStructureExecution::new(
                    p.small_id,
                    unit_id,
                ))
            } else {
                ExecEnum::NoOp(NoOpExecution)
            }
        }
        "allianceRequest" => {
            let recipient = intent
                .fields
                .get("recipient")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if let Some(p) = game.player_by_client_id(client_id) {
                ExecEnum::AllianceRequest(AllianceRequestExecution::new(
                    p.small_id,
                    recipient,
                ))
            } else {
                ExecEnum::NoOp(NoOpExecution)
            }
        }
        "allianceReject" => {
            let requestor = intent
                .fields
                .get("requestor")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if let Some(p) = game.player_by_client_id(client_id) {
                ExecEnum::AllianceReject(AllianceRejectExecution::new(
                    requestor,
                    p.small_id,
                ))
            } else {
                ExecEnum::NoOp(NoOpExecution)
            }
        }
        "breakAlliance" => {
            let recipient = intent
                .fields
                .get("recipient")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if let Some(p) = game.player_by_client_id(client_id) {
                ExecEnum::BreakAlliance(BreakAllianceExecution::new(
                    p.small_id,
                    recipient,
                ))
            } else {
                ExecEnum::NoOp(NoOpExecution)
            }
        }
        "allianceExtension" => {
            let recipient = intent
                .fields
                .get("recipient")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if let Some(p) = game.player_by_client_id(client_id) {
                ExecEnum::AllianceExtension(AllianceExtensionExecution::new(
                    p.small_id,
                    recipient,
                ))
            } else {
                ExecEnum::NoOp(NoOpExecution)
            }
        }
        "donate_troops" => {
            let recipient = intent
                .fields
                .get("recipient")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let troops = intent.fields.get("troops").and_then(|v| {
                v.as_f64()
                    .or_else(|| v.as_i64().map(|n| n as f64))
                    .or_else(|| v.as_u64().map(|n| n as f64))
            });
            if let Some(p) = game.player_by_client_id(client_id) {
                ExecEnum::DonateTroops(DonateTroopsExecution::new(
                    p.small_id,
                    recipient,
                    troops,
                ))
            } else {
                ExecEnum::NoOp(NoOpExecution)
            }
        }
        "donate_gold" => {
            let recipient = intent
                .fields
                .get("recipient")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let gold = intent.fields.get("gold").and_then(|v| {
                v.as_i64()
                    .or_else(|| v.as_f64().map(|n| n as i64))
                    .or_else(|| v.as_u64().map(|n| n as i64))
            });
            if let Some(p) = game.player_by_client_id(client_id) {
                ExecEnum::DonateGold(DonateGoldExecution::new(p.small_id, recipient, gold))
            } else {
                ExecEnum::NoOp(NoOpExecution)
            }
        }
        "embargo" => {
            let target_id = intent
                .fields
                .get("targetID")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let action_start = intent
                .fields
                .get("action")
                .and_then(Value::as_str)
                .map(|a| a == "start")
                .unwrap_or(false);
            if let Some(p) = game.player_by_client_id(client_id) {
                ExecEnum::Embargo(EmbargoExecution::new(p.small_id, target_id, action_start))
            } else {
                ExecEnum::NoOp(NoOpExecution)
            }
        }
        "embargo_all" => {
            let action_start = intent
                .fields
                .get("action")
                .and_then(Value::as_str)
                .map(|a| a == "start")
                .unwrap_or(false);
            if let Some(p) = game.player_by_client_id(client_id) {
                ExecEnum::EmbargoAll(EmbargoAllExecution::new(p.small_id, action_start))
            } else {
                ExecEnum::NoOp(NoOpExecution)
            }
        }
        "targetPlayer" => {
            let target = intent
                .fields
                .get("target")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if let Some(p) = game.player_by_client_id(client_id) {
                ExecEnum::TargetPlayer(TargetPlayerExecution::new(p.small_id, target))
            } else {
                ExecEnum::NoOp(NoOpExecution)
            }
        }
        "cancel_attack" => {
            let attack_id = intent
                .fields
                .get("attackID")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            if let Some(p) = game.player_by_client_id(client_id) {
                ExecEnum::Retreat(RetreatExecution::new(p.small_id, attack_id))
            } else {
                ExecEnum::NoOp(NoOpExecution)
            }
        }
        _ => ExecEnum::NoOp(NoOpExecution),
    }
}

pub fn turn_to_executions(
    game: &Game,
    game_id: &str,
    intents: &[StampedIntent],
) -> Vec<ExecEnum> {
    if intents.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(intents.len());
    for intent in intents {
        if intent.intent_type == "attack" {
            let troops = parse_troops(&intent.fields);
            let target = parse_target_id(&intent.fields);
            if let Some((owner, target, troops)) =
                AttackExecution::from_intent(&intent.client_id, game, troops, target)
            {
                // TS `createExecs`: one execution per intent, in turn order.
                out.push(ExecEnum::Attack(AttackExecution::new(
                    owner, target, troops,
                )));
            } else {
                out.push(ExecEnum::NoOp(NoOpExecution));
            }
            continue;
        }
        out.push(intent_to_execution(game, game_id, intent));
    }
    out
}
