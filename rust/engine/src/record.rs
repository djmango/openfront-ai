//! GameRecord types + sparse-turn expansion (`Util.decompressGameRecord`).

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameRecord {
    pub info: GameStartInfo,
    #[serde(default)]
    pub version: Option<String>,
    pub turns: Vec<Turn>,
    #[serde(default, rename = "gitCommit")]
    pub git_commit: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameStartInfo {
    #[serde(rename = "gameID")]
    pub game_id: String,
    pub config: Value,
    pub players: Vec<PlayerRecord>,
    #[serde(default, rename = "numTurns", alias = "num_turns")]
    pub num_turns: u32,
    #[serde(default)]
    pub winner: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayerRecord {
    #[serde(rename = "clientID")]
    pub client_id: String,
    pub username: String,
    #[serde(default)]
    pub clan_tag: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Turn {
    pub turn_number: u32,
    #[serde(default)]
    pub intents: Vec<StampedIntent>,
    #[serde(default)]
    pub hash: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StampedIntent {
    #[serde(rename = "type")]
    pub intent_type: String,
    #[serde(rename = "clientID")]
    pub client_id: String,
    #[serde(flatten)]
    pub fields: Value,
}

impl GameRecord {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(bytes)
    }

    /// Pad sparse turns so `turns[t]` is tick `t` (matches TS `decompressGameRecord`).
    pub fn decompress(mut self) -> Self {
        let mut out: Vec<Turn> = Vec::new();
        let mut last = -1i64;
        for turn in self.turns.drain(..) {
            while last < turn.turn_number as i64 - 1 {
                last += 1;
                out.push(Turn {
                    turn_number: last as u32,
                    intents: vec![],
                    hash: None,
                });
            }
            out.push(turn);
            last = out.last().unwrap().turn_number as i64;
        }
        while out.len() < self.info.num_turns as usize {
            let n = out.len() as u32;
            out.push(Turn {
                turn_number: n,
                intents: vec![],
                hash: None,
            });
        }
        self.turns = out;
        self
    }
}
