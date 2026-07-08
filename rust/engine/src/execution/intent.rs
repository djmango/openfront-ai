use super::{ExecEnum, MarkDisconnectedExecution, NoOpExecution, SpawnExecution, TransportShipExecution};
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
        _ => ExecEnum::NoOp(NoOpExecution),
    }
}

pub fn turn_to_executions(
    game: &mut Game,
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
                game.add_land_attack(owner, target, troops);
            } else {
                // TS `createExec` - missing player still appends a NoOp execution.
                out.push(ExecEnum::NoOp(NoOpExecution));
            }
            continue;
        }
        out.push(intent_to_execution(game, game_id, intent));
    }
    out
}
