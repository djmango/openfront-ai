//! Wire types for `GameRecord.info.config` (subset of TS `Schemas.ts`).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// `UnitType` string values from TS `Game.ts`.
pub mod unit_type {
    pub const TRANSPORT: &str = "Transport";
    pub const WARSHIP: &str = "Warship";
    pub const SHELL: &str = "Shell";
    pub const SAM_MISSILE: &str = "SAMMissile";
    pub const PORT: &str = "Port";
    pub const ATOM_BOMB: &str = "Atom Bomb";
    pub const HYDROGEN_BOMB: &str = "Hydrogen Bomb";
    pub const TRADE_SHIP: &str = "Trade Ship";
    pub const MISSILE_SILO: &str = "Missile Silo";
    pub const DEFENSE_POST: &str = "Defense Post";
    pub const SAM_LAUNCHER: &str = "SAM Launcher";
    pub const CITY: &str = "City";
    pub const MIRV: &str = "MIRV";
    pub const MIRV_WARHEAD: &str = "MIRV Warhead";
    pub const TRAIN: &str = "Train";
    pub const FACTORY: &str = "Factory";
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DoomsdayClockSpeed {
    Slow,
    Normal,
    Fast,
    #[serde(rename = "veryfast")]
    VeryFast,
}

impl Default for DoomsdayClockSpeed {
    fn default() -> Self {
        Self::Normal
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DoomsdayClockConfig {
    pub enabled: Option<bool>,
    pub speed: Option<DoomsdayClockSpeed>,
}

/// `nations` is either a count (1–400) or `"default"` / `"disabled"`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum NationsConfig {
    Count(u32),
    Mode(String),
}

impl NationsConfig {
    pub fn spawn_nations(&self) -> bool {
        !matches!(self, Self::Mode(s) if s == "disabled")
    }
}

/// TS `TeamCountConfig` - team count or preset string (`"Humans Vs Nations"`, etc.).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PlayerTeamsConfig {
    Count(u32),
    Mode(String),
}

impl PlayerTeamsConfig {
    pub fn is_humans_vs_nations(&self) -> bool {
        matches!(self, Self::Mode(s) if s == "Humans Vs Nations")
    }
}

/// Full wire config embedded in `GameRecord.info.config`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GameConfig {
    pub game_map: String,
    pub difficulty: String,
    pub donate_gold: bool,
    pub donate_troops: bool,
    pub game_type: String,
    pub game_mode: String,
    pub game_map_size: String,
    pub nations: NationsConfig,
    pub bots: u32,
    pub infinite_gold: bool,
    pub infinite_troops: bool,
    pub instant_build: bool,
    pub random_spawn: bool,
    #[serde(default)]
    pub doomsday_clock: Option<DoomsdayClockConfig>,
    #[serde(default)]
    pub disabled_units: Option<Vec<String>>,
    #[serde(default)]
    pub player_teams: Option<PlayerTeamsConfig>,
    #[serde(default)]
    pub disable_alliances: Option<bool>,
    #[serde(default)]
    pub spawn_immunity_duration: Option<u32>,
    #[serde(default)]
    pub starting_gold: Option<u64>,
    #[serde(default)]
    pub gold_multiplier: Option<f64>,
    #[serde(default)]
    pub max_timer_value: Option<u32>,
    #[serde(default)]
    pub ranked_type: Option<String>,
}

impl GameConfig {
    pub fn from_value(value: &Value) -> Result<Self, serde_json::Error> {
        serde_json::from_value(value.clone())
    }
}
