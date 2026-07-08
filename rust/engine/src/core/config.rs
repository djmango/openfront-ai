//! Runtime config wrapper (TS `configuration/Config.ts` subset for replay bootstrap).

use super::schemas::{DoomsdayClockSpeed, GameConfig};
use crate::record::GameRecord;
use serde_json::Value;

/// Resolved Doomsday Clock settings (TS `DOOMSDAY_CLOCK_DEFAULTS` + wire overrides).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DoomsdayClockResolved {
    pub enabled: bool,
    pub speed: DoomsdayClockSpeed,
    pub warn_seconds: u32,
    pub drain_start_percent: u32,
    pub drain_max_percent: u32,
    pub drain_ramp_seconds: u32,
    pub warship_drain_max_percent: u32,
}

impl Default for DoomsdayClockResolved {
    fn default() -> Self {
        Self {
            enabled: false,
            speed: DoomsdayClockSpeed::Normal,
            warn_seconds: 10,
            drain_start_percent: 2,
            drain_max_percent: 6,
            drain_ramp_seconds: 50,
            warship_drain_max_percent: 50,
        }
    }
}

/// TS `Config` - reads `GameConfig` and exposes replay-bootstrap helpers.
#[derive(Debug, Clone)]
pub struct Config {
    game_config: GameConfig,
    is_replay: bool,
}

impl Config {
    pub fn new(game_config: GameConfig, is_replay: bool) -> Self {
        Self {
            game_config,
            is_replay,
        }
    }

    pub fn from_value(value: &Value, is_replay: bool) -> Result<Self, serde_json::Error> {
        Ok(Self::new(GameConfig::from_value(value)?, is_replay))
    }

    pub fn from_record(record: &GameRecord, is_replay: bool) -> Result<Self, serde_json::Error> {
        Self::from_value(&record.info.config, is_replay)
    }

    pub fn is_replay(&self) -> bool {
        self.is_replay
    }

    pub fn game_config(&self) -> &GameConfig {
        &self.game_config
    }

    pub fn bots(&self) -> u32 {
        self.game_config.bots
    }

    pub fn spawn_nations(&self) -> bool {
        self.game_config.nations.spawn_nations()
    }

    pub fn is_random_spawn(&self) -> bool {
        self.game_config.random_spawn
    }

    pub fn is_unit_disabled(&self, unit_type: &str) -> bool {
        self.game_config
            .disabled_units
            .as_ref()
            .is_some_and(|units| units.iter().any(|u| u == unit_type))
    }

    pub fn boat_max_number(&self) -> usize {
        if self.is_unit_disabled(crate::core::schemas::unit_type::TRANSPORT) {
            0
        } else {
            3
        }
    }

    pub fn boat_attack_amount(&self, troops: i32) -> f64 {
        (troops as f64 / 5.0).floor()
    }

    pub fn doomsday_clock_config(&self) -> DoomsdayClockResolved {
        let defaults = DoomsdayClockResolved::default();
        let wire = self.game_config.doomsday_clock.as_ref();
        DoomsdayClockResolved {
            enabled: wire.and_then(|c| c.enabled).unwrap_or(defaults.enabled),
            speed: wire
                .and_then(|c| c.speed)
                .unwrap_or(defaults.speed),
            warn_seconds: defaults.warn_seconds,
            drain_start_percent: defaults.drain_start_percent,
            drain_max_percent: defaults.drain_max_percent,
            drain_ramp_seconds: defaults.drain_ramp_seconds,
            warship_drain_max_percent: defaults.warship_drain_max_percent,
        }
    }

    /// TS `numSpawnPhaseTurns()`.
    pub fn disable_alliances(&self) -> bool {
        self.game_config.disable_alliances.unwrap_or(false)
    }

    pub fn num_spawn_phase_turns(&self) -> u32 {
        if self.game_config.game_type == "Singleplayer" {
            return 100;
        }
        if self.is_random_spawn() {
            return 150;
        }
        300
    }

    /// TS `spawnImmunityDuration()` - default 5s = 50 ticks.
    pub fn spawn_immunity_duration(&self) -> u32 {
        self.game_config
            .spawn_immunity_duration
            .unwrap_or(50)
    }

    /// TS `nationSpawnImmunityDuration()` - always default 50 ticks.
    pub fn nation_spawn_immunity_duration(&self) -> u32 {
        50
    }

    /// TS `startManpower(playerInfo)`.
    pub fn start_manpower(&self, player_type: crate::game::PlayerType) -> i32 {
        use crate::game::PlayerType;
        match player_type {
            PlayerType::Bot => 10_000,
            PlayerType::Nation => match self.game_config.difficulty.as_str() {
                "Easy" => 12_500,
                "Medium" => 18_750,
                "Hard" => 25_000,
                "Impossible" => 31_250,
                _ => 18_750,
            },
            PlayerType::Human => {
                if self.game_config.infinite_troops {
                    1_000_000
                } else {
                    25_000
                }
            }
        }
    }

    pub fn min_distance_between_players(&self) -> u32 {
        30
    }

    /// TS `maxTroops()` - cities omitted until units are ported.
    pub fn max_troops(&self, player_type: crate::game::PlayerType, tiles_owned: i32) -> f64 {
        use crate::game::PlayerType;
        let mut max = 2.0 * ((tiles_owned as f64).powf(0.6) * 1000.0 + 50_000.0);
        match player_type {
            PlayerType::Bot => max /= 3.0,
            PlayerType::Human => {
                if self.game_config.infinite_troops {
                    return 1_000_000_000.0;
                }
            }
            PlayerType::Nation => {
                max *= match self.game_config.difficulty.as_str() {
                    "Easy" => 0.5,
                    "Medium" => 0.75,
                    "Hard" => 1.0,
                    "Impossible" => 1.25,
                    _ => 0.75,
                };
            }
        }
        max
    }

    /// TS `troopIncreaseRate()`.
    pub fn troop_increase_rate(
        &self,
        player_type: crate::game::PlayerType,
        troops: i32,
        tiles_owned: i32,
    ) -> i32 {
        use crate::game::PlayerType;
        let max = self.max_troops(player_type, tiles_owned);
        let mut to_add = 10.0 + (troops as f64).powf(0.73) / 4.0;
        let ratio = 1.0 - troops as f64 / max;
        to_add *= ratio;
        if player_type == PlayerType::Bot {
            to_add *= 0.5;
        }
        if player_type == PlayerType::Nation {
            to_add *= match self.game_config.difficulty.as_str() {
                "Easy" => 0.9,
                "Medium" => 0.95,
                "Hard" => 1.0,
                "Impossible" => 1.05,
                _ => 0.95,
            };
        }
        let capped = (troops as f64 + to_add).min(max);
        crate::util::to_int(capped) - troops
    }

    /// TS `attackAmount()`.
    pub fn attack_amount(&self, player_type: crate::game::PlayerType, troops: i32) -> f64 {
        use crate::game::PlayerType;
        if player_type == PlayerType::Bot {
            troops as f64 / 20.0
        } else {
            troops as f64 / 5.0
        }
    }

    /// TS `goldAdditionRate()` - multiplier omitted until cities/ports affect income.
    pub fn gold_addition_rate(&self, player_type: crate::game::PlayerType) -> i64 {
        use crate::game::PlayerType;
        if player_type == PlayerType::Bot {
            50
        } else {
            100
        }
    }

    /// TS `attackTilesPerTick()`.
    pub fn attack_tiles_per_tick(
        &self,
        attack_troops: f64,
        attacker_type: crate::game::PlayerType,
        defender_is_player: bool,
        defender_troops: i32,
        num_adjacent: f64,
    ) -> f64 {
        use crate::game::PlayerType;
        use crate::util::within;
        if defender_is_player {
            within(
                ((5.0 * attack_troops) / defender_troops as f64) * 2.0,
                0.01,
                0.5,
            ) * num_adjacent
                * 3.0
        } else {
            num_adjacent * 2.0
        }
    }

    pub fn traitor_defense_debuff(&self) -> f64 {
        0.5
    }

    pub fn traitor_speed_debuff(&self) -> f64 {
        0.8
    }

    pub fn defense_post_range(&self) -> u32 {
        30
    }

    pub fn defense_post_defense_bonus(&self) -> f64 {
        5.0
    }

    pub fn defense_post_speed_bonus(&self) -> f64 {
        3.0
    }

    pub fn fallout_defense_modifier(&self, fallout_ratio: f64) -> f64 {
        5.0 - fallout_ratio * 2.0
    }

    /// TS `attackLogic()` terra-nullius branch.
    pub fn attack_logic(
        &self,
        attack_troops: f64,
        attacker_type: crate::game::PlayerType,
        defender_is_player: bool,
        defender_troops: i32,
        defender_tiles: i32,
        terrain: crate::map::TerrainType,
    ) -> (f64, f64, f64) {
        use crate::game::PlayerType;
        use crate::util::{sigmoid, within};

        let (mut mag, mut speed) = match terrain {
            crate::map::TerrainType::Plains => (80.0, 16.5),
            crate::map::TerrainType::Highland => (100.0, 20.0),
            crate::map::TerrainType::Mountain => (120.0, 25.0),
            _ => (80.0, 16.5),
        };

        if defender_is_player {
            if (attacker_type == PlayerType::Human || attacker_type == PlayerType::Nation)
                && false
            {
                // defender bot modifier applied when we track defender type
            }

            const DEFENSE_DEBUFF_MIDPOINT: f64 = 150_000.0;
            const DEFENSE_DEBUFF_DECAY_RATE: f64 = std::f64::consts::LN_2 / 50_000.0;

            let defense_sig = 1.0
                - sigmoid(
                    defender_tiles as f64,
                    DEFENSE_DEBUFF_DECAY_RATE,
                    DEFENSE_DEBUFF_MIDPOINT,
                );
            let large_defender_speed_debuff = 0.7 + 0.3 * defense_sig;
            let large_defender_attack_debuff = 0.7 + 0.3 * defense_sig;

            let defender_troop_loss = defender_troops as f64 / defender_tiles.max(1) as f64;
            let current_attacker_loss = within(
                defender_troops as f64 / attack_troops,
                0.6,
                2.0,
            ) * mag
                * 0.8
                * large_defender_attack_debuff;
            let alt_attacker_loss = 1.3 * defender_troop_loss * (mag / 100.0);
            let attacker_troop_loss = 0.6 * current_attacker_loss + 0.4 * alt_attacker_loss;

            let tiles_per_tick = within(
                defender_troops as f64 / (5.0 * attack_troops),
                0.2,
                1.5,
            ) * speed
                * large_defender_speed_debuff;

            (attacker_troop_loss, defender_troop_loss, tiles_per_tick)
        } else {
            let attacker_troop_loss = if attacker_type == PlayerType::Bot {
                mag / 10.0
            } else {
                mag / 5.0
            };
            let tiles_per_tick = within((2000.0 * speed.max(10.0)) / attack_troops, 5.0, 100.0);
            (attacker_troop_loss, 0.0, tiles_per_tick)
        }
    }

    pub fn attack_logic_vs_player(
        &self,
        attack_troops: f64,
        attacker_type: crate::game::PlayerType,
        defender_type: crate::game::PlayerType,
        defender_troops: i32,
        defender_tiles: i32,
        terrain: crate::map::TerrainType,
    ) -> (f64, f64, f64) {
        use crate::game::PlayerType;
        use crate::util::{sigmoid, within};

        let (mut mag, mut speed) = match terrain {
            crate::map::TerrainType::Plains => (80.0, 16.5),
            crate::map::TerrainType::Highland => (100.0, 20.0),
            crate::map::TerrainType::Mountain => (120.0, 25.0),
            _ => (80.0, 16.5),
        };

        if (attacker_type == PlayerType::Human || attacker_type == PlayerType::Nation)
            && defender_type == PlayerType::Bot
        {
            mag *= 0.7;
        }

        const DEFENSE_DEBUFF_MIDPOINT: f64 = 150_000.0;
        const DEFENSE_DEBUFF_DECAY_RATE: f64 = std::f64::consts::LN_2 / 50_000.0;

        let defense_sig =
            1.0 - sigmoid(defender_tiles as f64, DEFENSE_DEBUFF_DECAY_RATE, DEFENSE_DEBUFF_MIDPOINT);
        let large_defender_speed_debuff = 0.7 + 0.3 * defense_sig;
        let large_defender_attack_debuff = 0.7 + 0.3 * defense_sig;

        let defender_troop_loss = defender_troops as f64 / defender_tiles.max(1) as f64;
        let current_attacker_loss = within(defender_troops as f64 / attack_troops, 0.6, 2.0)
            * mag
            * 0.8
            * large_defender_attack_debuff;
        let alt_attacker_loss = 1.3 * defender_troop_loss * (mag / 100.0);
        let attacker_troop_loss = 0.6 * current_attacker_loss + 0.4 * alt_attacker_loss;

        let tiles_per_tick = within(defender_troops as f64 / (5.0 * attack_troops), 0.2, 1.5)
            * speed
            * large_defender_speed_debuff;

        (attacker_troop_loss, defender_troop_loss, tiles_per_tick)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schemas::{NationsConfig, unit_type};
    use serde_json::json;

    fn sample_config_value() -> Value {
        json!({
            "gameMap": "Onion",
            "difficulty": "Medium",
            "donateGold": false,
            "donateTroops": false,
            "gameType": "Singleplayer",
            "gameMode": "Free For All",
            "gameMapSize": "Normal",
            "nations": "default",
            "bots": 100,
            "infiniteGold": false,
            "infiniteTroops": false,
            "instantBuild": false,
            "randomSpawn": true,
            "disabledUnits": ["Factory"],
            "doomsdayClock": { "enabled": true, "speed": "fast" }
        })
    }

    #[test]
    fn parses_record_config_and_matches_ts_helpers() {
        let cfg = Config::from_value(&sample_config_value(), true).unwrap();
        assert!(cfg.is_replay());
        assert_eq!(cfg.bots(), 100);
        assert!(cfg.spawn_nations());
        assert!(cfg.is_random_spawn());
        assert!(cfg.is_unit_disabled(unit_type::FACTORY));
        assert!(!cfg.is_unit_disabled(unit_type::CITY));
        let dc = cfg.doomsday_clock_config();
        assert!(dc.enabled);
        assert_eq!(dc.speed, DoomsdayClockSpeed::Fast);
        assert_eq!(dc.warn_seconds, 10);
    }

    #[test]
    fn nations_disabled_skips_spawn_nations() {
        let mut v = sample_config_value();
        v["nations"] = json!("disabled");
        let cfg = Config::from_value(&v, false).unwrap();
        assert!(!cfg.spawn_nations());
        assert!(matches!(cfg.game_config().nations, NationsConfig::Mode(_)));
    }
}
