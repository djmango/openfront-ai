//! Runtime config wrapper (TS `configuration/Config.ts` subset for replay bootstrap).

use super::schemas::{DoomsdayClockSpeed, GameConfig};
use crate::record::GameRecord;
use serde_json::Value;

/// TS `rel(player, other)` result used by `trainGold`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainRelation {
    SelfTrade,
    Team,
    Ally,
    Other,
}

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

    pub fn game_type(&self) -> &str {
        &self.game_config.game_type
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

    /// TS `UnitInfo(Warship).maxHealth` - base (veterancy-0) warship health cap.
    pub fn warship_base_max_health(&self) -> i32 {
        1000
    }

    /// TS `warshipMaxVeterancy()` - highest veterancy level a warship can reach.
    pub fn warship_max_veterancy(&self) -> i32 {
        3
    }

    /// TS `warshipVeterancyHealthBonus()` - max-health boost per level, as an integer
    /// percent of base max health.
    pub fn warship_veterancy_health_bonus(&self) -> i32 {
        20
    }

    /// TS `warshipVeterancyShellDamageBonus()` - shell-damage boost per level, as an
    /// integer percent of the rolled damage.
    pub fn warship_veterancy_shell_damage_bonus(&self) -> i32 {
        20
    }

    /// TS `warshipVeterancyTransportKills()` - transport ships a warship must destroy
    /// (alone, with no trade captures) to gain one veterancy level.
    pub fn warship_veterancy_transport_kills(&self) -> i32 {
        10
    }

    /// TS `warshipVeterancyTradeCaptures()` - trade ships a warship must capture (alone,
    /// with no transport kills) to gain one veterancy level.
    pub fn warship_veterancy_trade_captures(&self) -> i32 {
        25
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

    pub fn alliance_request_duration(&self) -> u32 {
        20 * 10
    }

    pub fn alliance_request_cooldown(&self) -> u32 {
        30 * 10
    }

    pub fn alliance_duration(&self) -> u32 {
        300 * 10
    }

    pub fn embargo_all_cooldown(&self) -> u32 {
        10 * 10
    }

    pub fn temporary_embargo_duration(&self) -> u32 {
        300 * 10
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

    /// TS `startingGold(playerInfo)` - lobby creator bonus omitted (not in records).
    /// Raw `Bot` players always get 0 starting gold in TS; only `Nation`/`Human` get the
    /// configured amount.
    pub fn starting_gold(&self, player_type: crate::game::PlayerType) -> i64 {
        use crate::game::PlayerType;
        if player_type == PlayerType::Bot {
            return 0;
        }
        self.game_config.starting_gold.unwrap_or(0) as i64
    }

    pub fn structure_min_dist(&self) -> u32 {
        15
    }

    pub fn radius_port_spawn(&self) -> u32 {
        20
    }

    pub fn atom_bomb_outer_range(&self) -> u32 {
        30
    }

    pub fn city_construction_ticks(&self) -> u32 {
        self.construction_ticks(crate::core::schemas::unit_type::CITY)
    }

    /// TS `unitInfo(...).constructionDuration`.
    pub fn construction_ticks(&self, unit_type: &str) -> u32 {
        use crate::core::schemas::unit_type;
        if self.game_config.instant_build {
            return 0;
        }
        match unit_type {
            unit_type::CITY | unit_type::FACTORY => 20,
            unit_type::DEFENSE_POST | unit_type::PORT => 50,
            unit_type::SAM_LAUNCHER => 300,
            unit_type::MISSILE_SILO => 100,
            _ => 0,
        }
    }

    /// TS `unitInfo` structure cost from `costWrapper` unit count.
    pub fn structure_cost(&self, unit_type: &str, cost_units: u32) -> i64 {
        use crate::core::schemas::unit_type;
        match unit_type {
            unit_type::CITY | unit_type::PORT | unit_type::FACTORY => {
                ((2f64.powi(cost_units as i32) * 125_000.0) as i64).min(1_000_000)
            }
            unit_type::SAM_LAUNCHER => {
                ((cost_units as i64 + 1) * 1_500_000).min(3_000_000)
            }
            unit_type::MISSILE_SILO => 1_000_000,
            unit_type::DEFENSE_POST => {
                let n = cost_units as i64;
                ((n + 1) * 50_000).min(250_000)
            }
            // TS `unitInfo(Warship).cost`: `costWrapper((n) => min(1_000_000, (n+1)*250_000),
            // Warship)`. Missing here meant every warship - AI or human-built - was free
            // (fell through to the `_ => 0` arm below), letting `maybeSpawnWarship`'s
            // `player.gold() > this.cost(Warship)` gate (and the RL env's identical
            // buildable-action gate in `rl.rs`/`obs.rs`) trivially always pass regardless of
            // gold. Found while porting `NationWarshipBehavior`'s AI decision layer, which is
            // the first caller to actually rely on this cost being nonzero.
            unit_type::WARSHIP => ((cost_units as i64 + 1) * 250_000).min(1_000_000),
            unit_type::ATOM_BOMB => 750_000,
            unit_type::HYDROGEN_BOMB => 5_000_000,
            unit_type::MIRV_WARHEAD => 0,
            _ => 0,
        }
    }

    /// Types summed for `costWrapper` pricing.
    pub fn cost_types_for(&self, unit_type: &str) -> &'static [&'static str] {
        use crate::core::schemas::unit_type;
        match unit_type {
            unit_type::CITY => &[unit_type::CITY],
            unit_type::PORT | unit_type::FACTORY => &[unit_type::PORT, unit_type::FACTORY],
            unit_type::SAM_LAUNCHER => &[unit_type::SAM_LAUNCHER],
            unit_type::MISSILE_SILO => &[unit_type::MISSILE_SILO],
            unit_type::DEFENSE_POST => &[unit_type::DEFENSE_POST],
            unit_type::WARSHIP => &[unit_type::WARSHIP],
            _ => &[],
        }
    }

    /// TS `unitInfo(City).cost` for first city (numUnits=0).
    pub fn city_cost(&self, cities_owned: u32) -> i64 {
        self.structure_cost(crate::core::schemas::unit_type::CITY, cities_owned)
    }

    /// TS `unitInfo(MIRV).cost` with zero launches (record parity default).
    pub fn mirv_cost(&self) -> i64 {
        25_000_000
    }

    pub fn hydrogen_bomb_cost(&self) -> i64 {
        5_000_000
    }

    pub fn atom_bomb_cost(&self) -> i64 {
        750_000
    }

    pub fn sam_launcher_cost(&self) -> i64 {
        1_500_000
    }

    pub fn unit_cost(&self, small_id: u16, unit_type: &str) -> i64 {
        use crate::core::schemas::unit_type;
        match unit_type {
            unit_type::CITY => {
                let owned = 0; // caller should pass via dedicated path
                let _ = small_id;
                self.city_cost(owned)
            }
            _ => 0,
        }
    }

    pub fn min_distance_between_players(&self) -> u32 {
        30
    }

    /// TS `maxTroops()`.
    pub fn max_troops(
        &self,
        player_type: crate::game::PlayerType,
        tiles_owned: i32,
        city_level_sum: i64,
    ) -> f64 {
        use crate::game::PlayerType;
        let mut max = 2.0 * ((tiles_owned as f64).powf(0.6) * 1000.0 + 50_000.0)
            + city_level_sum as f64 * self.city_troop_increase();
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

    pub fn city_troop_increase(&self) -> f64 {
        250_000.0
    }

    /// TS `troopIncreaseRate()`.
    pub fn troop_increase_rate(
        &self,
        player_type: crate::game::PlayerType,
        troops: i32,
        tiles_owned: i32,
        city_level_sum: i64,
    ) -> i32 {
        use crate::game::PlayerType;
        let max = self.max_troops(player_type, tiles_owned, city_level_sum);
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
        let capped_delta = (troops as f64 + to_add).min(max) - troops as f64;
        crate::util::to_int(capped_delta)
    }

    /// Unrounded TS `troopIncreaseRate()` - the RL obs emits the raw float
    /// (`bridge/common.ts` sends `config.troopIncreaseRate(p)` un-truncated).
    pub fn troop_increase_rate_raw(
        &self,
        player_type: crate::game::PlayerType,
        troops: i32,
        tiles_owned: i32,
        city_level_sum: i64,
    ) -> f64 {
        use crate::game::PlayerType;
        let max = self.max_troops(player_type, tiles_owned, city_level_sum);
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
        (troops as f64 + to_add).min(max) - troops as f64
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

    /// TS `Config.goldMultiplier()` - `hostCheats.goldMultiplier` (lobby-creator-only
    /// override) isn't modeled since no pinned record uses `hostCheats`.
    pub fn gold_multiplier(&self) -> f64 {
        self.game_config.gold_multiplier.unwrap_or(1.0)
    }

    /// TS `goldAdditionRate()`.
    pub fn gold_addition_rate(&self, player_type: crate::game::PlayerType) -> i64 {
        use crate::game::PlayerType;
        let base_rate: f64 = if player_type == PlayerType::Bot { 50.0 } else { 100.0 };
        (base_rate * self.gold_multiplier()).floor() as i64
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

    /// TS `trainStationMinRange()`.
    pub fn train_station_min_range(&self) -> u32 {
        15
    }

    /// TS `trainStationMaxRange()`.
    pub fn train_station_max_range(&self) -> u32 {
        110
    }

    /// TS `railroadMaxSize()`.
    pub fn railroad_max_size(&self) -> f64 {
        self.train_station_max_range() as f64 * 1.4142
    }

    /// TS `tradeShipShortRangeDebuff()`.
    pub fn trade_ship_short_range_debuff(&self) -> f64 {
        300.0
    }

    /// TS `proximityBonusPortsNb(totalPorts)`.
    pub fn proximity_bonus_ports_nb(&self, total_ports: usize) -> f64 {
        crate::util::within(total_ports as f64 / 3.0, 4.0, total_ports as f64)
    }

    /// TS `Config.tradeShipGold(dist, player)` - sigmoid-based gold reward; the
    /// per-player `goldMultiplierFor` override (`hostCheats`) isn't modeled, matching
    /// `gold_multiplier()` above.
    pub fn trade_ship_gold(&self, dist: f64) -> i64 {
        let debuff = self.trade_ship_short_range_debuff();
        let base_gold = 75_000.0 / (1.0 + (-0.03 * (dist - debuff)).exp()) + 50.0 * dist;
        (base_gold * self.gold_multiplier()).floor() as i64
    }

    /// TS `Config.tradeShipSpawnRate(rejections, numTradeShips)` - probability of
    /// spawn is `1 / tradeShipSpawnRate(...)`.
    pub fn trade_ship_spawn_rate(&self, rejections: i64, num_trade_ships: i64) -> i64 {
        let decay_rate = std::f64::consts::LN_2 / 50.0;
        let base_spawn_rate = 1.0 - crate::util::sigmoid(num_trade_ships as f64, decay_rate, 400.0);
        let rejection_modifier = 1.0 / (rejections as f64 + 1.0);
        ((100.0 * rejection_modifier) / base_spawn_rate).floor() as i64
    }

    /// TS `trainSpawnRate(numPlayerFactories)` - hyperbolic decay, midpoint at 10 factories.
    pub fn train_spawn_rate(&self, num_player_factories: i32) -> i32 {
        (num_player_factories + 10) * 15
    }

    /// TS `trainGold(rel, citiesVisited, player)`.
    pub fn train_gold(&self, rel: TrainRelation, cities_visited: u32) -> i64 {
        let cities_visited = cities_visited.saturating_sub(9);
        let base_gold: f64 = match rel {
            TrainRelation::SelfTrade => 10_000.0,
            TrainRelation::Ally => 35_000.0,
            TrainRelation::Team | TrainRelation::Other => 25_000.0,
        };
        let dist_penalty = cities_visited as f64 * 5_000.0;
        let gold = (base_gold - dist_penalty).max(5_000.0);
        (gold * self.gold_multiplier()).floor() as i64
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

    /// TS `SiloCooldown()`.
    pub fn silo_cooldown(&self) -> u32 {
        90
    }

    /// TS `SAMCooldown()`.
    pub fn sam_cooldown(&self) -> u32 {
        90
    }

    /// TS `waterNukes()`  -  not present in any pinned record's `gameConfig`; default false.
    pub fn water_nukes(&self) -> bool {
        false
    }

    /// TS `nukeMagnitudes(unitType)` -> `(inner, outer)`.
    pub fn nuke_magnitudes(&self, unit_type: &str) -> (f64, f64) {
        use crate::core::schemas::unit_type;
        match unit_type {
            unit_type::MIRV_WARHEAD => (12.0, 18.0),
            unit_type::ATOM_BOMB => (12.0, 30.0),
            unit_type::HYDROGEN_BOMB => (80.0, 100.0),
            _ => (12.0, 30.0),
        }
    }

    /// TS `nukeAllianceBreakThreshold()`.
    pub fn nuke_alliance_break_threshold(&self) -> f64 {
        100.0
    }

    /// TS `defaultNukeSpeed()`.
    pub fn default_nuke_speed(&self) -> f64 {
        10.0
    }

    /// TS `defaultNukeTargetableRange()`.
    pub fn default_nuke_targetable_range(&self) -> f64 {
        150.0
    }

    /// TS `maxSamRange()`.
    pub fn max_sam_range(&self) -> f64 {
        150.0
    }

    /// TS `samRange(level)`.
    pub fn sam_range(&self, level: i32) -> f64 {
        self.max_sam_range() - 480.0 / (level as f64 + 5.0)
    }

    /// TS `defaultSamMissileSpeed()`.
    pub fn default_sam_missile_speed(&self) -> f64 {
        12.0
    }

    /// TS `nukeDeathFactor(nukeType, humans, tilesOwned, maxTroops)`.
    pub fn nuke_death_factor(
        &self,
        nuke_type: &str,
        humans: f64,
        tiles_owned: f64,
        max_troops: f64,
    ) -> f64 {
        use crate::core::schemas::unit_type;
        if nuke_type != unit_type::MIRV_WARHEAD {
            return (5.0 * humans) / tiles_owned.max(1.0);
        }
        let target_troops = 0.03 * max_troops;
        let excess_troops = (humans - target_troops).max(0.0);
        let scaling_factor = 500.0;
        let steepness = 2.0;
        let normalized_excess = excess_troops / max_troops;
        scaling_factor * (1.0 - (-steepness * normalized_excess).exp())
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

    // TS `unitInfo(Warship).cost`: `costWrapper((n) => min(1_000_000, (n+1)*250_000),
    // Warship)`. `unit_type::WARSHIP` fell through `structure_cost`'s `_ => 0` arm before
    // this fix, so every warship (AI or human-built) was free. Values below match the
    // TS formula directly: 0 existing -> 250_000, 3 existing -> 1_000_000, 10 existing ->
    // still capped at 1_000_000 (the `min` ceiling), matching Port/City/DefensePost/
    // SAMLauncher's identical cap-then-scale shape for the same unit type elsewhere in
    // this match.
    #[test]
    fn warship_cost_scales_with_existing_count_and_caps_at_1m() {
        let cfg = Config::from_value(&sample_config_value(), true).unwrap();
        assert_eq!(cfg.structure_cost(unit_type::WARSHIP, 0), 250_000);
        assert_eq!(cfg.structure_cost(unit_type::WARSHIP, 3), 1_000_000);
        assert_eq!(cfg.structure_cost(unit_type::WARSHIP, 10), 1_000_000);
        assert_eq!(cfg.cost_types_for(unit_type::WARSHIP), &[unit_type::WARSHIP]);
    }
}
