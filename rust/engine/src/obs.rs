//! RL observation head (subset of `bridge/common.ts`).

use crate::game::Game;
use serde_json::{json, Value};

pub fn build_obs_head(game: &Game, client_id: &str) -> Value {
    let agent = game.player_by_client_id(client_id);
    json!({
        "tick": game.ticks(),
        "width": game.width(),
        "height": game.height(),
        "spawnPhase": game.in_spawn_phase(),
        "winner": game.winner,
        "me": agent.map(|p| p.small_id as i32).unwrap_or(-1),
        "alive": agent.map(|p| p.alive).unwrap_or(false),
        "entities": entities(game),
    })
}

fn entities(game: &Game) -> Value {
    let players: Vec<Value> = game
        .all_players()
        .iter()
        .map(|p| {
            json!({
                "id": p.small_id,
                "pid": p.id,
                "type": match p.player_type {
                    crate::game::PlayerType::Human => "Human",
                    crate::game::PlayerType::Bot => "Bot",
                    crate::game::PlayerType::Nation => "Nation",
                },
                "troops": p.troops,
                "gold": p.gold.to_string(),
                "tiles": p.tiles_owned,
                "alive": p.alive,
            })
        })
        .collect();
    json!({ "players": players, "units": [], "alliances": [] })
}

pub fn tile_bytes_le(game: &Game) -> Vec<u8> {
    let buf = game.tile_state_buffer();
    let mut out = Vec::with_capacity(buf.len() * 2);
    for &v in buf {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}
