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
    #[serde(default)]
    pub friends: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Turn {
    pub turn_number: u32,
    #[serde(default)]
    pub intents: Vec<StampedIntent>,
    // TS's `hash` is a JS `number` (f64); once the accumulated hash exceeds
    // 2^53, JSON.stringify no longer prints the double's exact binary value
    // but the *shortest* decimal string that round-trips to the same double
    // (ECMA-262 Number::toString). At that point the literal digits in the
    // archive can differ from the double's true integer value by up to half
    // a ULP. We must deserialize through f64 (recovering the original double
    // via the same round-trip guarantee) and *then* truncate to i64, mirroring
    // the native side's `hash as i64`: parsing the JSON literal straight into
    // i64 instead reads the printed digits as an exact integer and can be off
    // by a few ULPs at large magnitudes, producing spurious tiny hash diffs.
    #[serde(default, skip_serializing_if = "Option::is_none", deserialize_with = "deserialize_hash_via_f64")]
    pub hash: Option<i64>,
}

fn deserialize_hash_via_f64<'de, D>(deserializer: D) -> Result<Option<i64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value: Option<f64> = Option::deserialize(deserializer)?;
    Ok(value.map(|v| v as i64))
}

#[cfg(test)]
mod hash_precision_tests {
    use super::Turn;

    // `18186392315806210` is what V8's shortest-round-trip Number::toString
    // prints for the double whose *exact* integer value is
    // `18186392315806208` (ULP = 4 at this magnitude). Naively parsing the
    // JSON literal as an exact i64 gives the wrong (unrepresentable) value;
    // parsing through f64 first recovers the double TS actually held.
    #[test]
    fn large_hash_round_trips_through_f64_like_js() {
        let json = r#"{"turnNumber":580,"intents":[],"hash":18186392315806210}"#;
        let turn: Turn = serde_json::from_str(json).unwrap();
        assert_eq!(turn.hash, Some(18186392315806208));
    }

    #[test]
    fn small_hash_is_unaffected() {
        let json = r#"{"turnNumber":1,"intents":[],"hash":42}"#;
        let turn: Turn = serde_json::from_str(json).unwrap();
        assert_eq!(turn.hash, Some(42));
    }

    #[test]
    fn missing_hash_stays_none() {
        let json = r#"{"turnNumber":1,"intents":[]}"#;
        let turn: Turn = serde_json::from_str(json).unwrap();
        assert_eq!(turn.hash, None);
    }
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
