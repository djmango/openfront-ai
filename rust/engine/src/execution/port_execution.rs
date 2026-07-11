//! Port structure behavior (`PortExecution.ts`).

use super::{
    trade_ship_execution::TradeShipExecution, train_station_execution::TrainStationExecution,
    ExecEnum, Execution,
};
use crate::core::schemas::unit_type;
use crate::game::Game;
use crate::map::TileRef;
use crate::prng::PseudoRandom;

pub struct PortExecution {
    small_id: u16,
    unit_id: i32,
    active: bool,
    random: Option<PseudoRandom>,
    check_offset: u32,
    trade_ship_spawn_rejections: i64,
}

impl PortExecution {
    pub fn new(small_id: u16, unit_id: i32) -> Self {
        Self {
            small_id,
            unit_id,
            active: true,
            random: None,
            check_offset: 0,
            trade_ship_spawn_rejections: 0,
        }
    }

    /// TS `PortExecution.createStation` - retried every tick (no one-shot guard) as long as
    /// `!hasTrainStation()`, so a Factory built nearby later still lets the port join.
    fn create_station(&self, game: &mut Game) {
        let Some(tile) = game.unit_tile_of(self.small_id, self.unit_id) else {
            return;
        };
        let range = game.wire.train_station_max_range();
        if game.has_unit_nearby_any(tile, range, unit_type::FACTORY) {
            game.set_has_train_station(self.small_id, self.unit_id, true);
            game.add_execution(ExecEnum::TrainStation(TrainStationExecution::new(
                self.small_id,
                self.unit_id,
                false,
            )));
        }
    }

    /// TS `PortExecution.shouldSpawnTradeShip`.
    fn should_spawn_trade_ship(&mut self, game: &Game) -> bool {
        let num_trade_ships = game.unit_level_sum_global(unit_type::TRADE_SHIP) as i64;
        let spawn_rate = game
            .wire
            .trade_ship_spawn_rate(self.trade_ship_spawn_rejections, num_trade_ships)
            .clamp(1, i32::MAX as i64) as i32;
        let level = game.unit_level_of(self.small_id, self.unit_id).max(1);
        let random = self.random.as_mut().expect("PortExecution not initialized");
        for _ in 0..level {
            if random.chance(spawn_rate) {
                self.trade_ship_spawn_rejections = 0;
                return true;
            }
            self.trade_ship_spawn_rejections += 1;
        }
        false
    }

    /// TS `PortExecution.tradingPorts` - candidate destination ports, weighted by
    /// probability (an entry appears multiple times to raise its odds).
    fn trading_ports(&self, game: &Game, port_tile: TileRef) -> Vec<(u16, i32)> {
        let mut source_components: Vec<u32> = Vec::new();
        game.map.for_each_neighbor4(port_tile, |n| {
            if !game.is_water(n) {
                return;
            }
            if let Some(c) = game.get_water_component(n) {
                if !source_components.contains(&c) {
                    source_components.push(c);
                }
            }
        });

        let mut candidates: Vec<(u16, i32, TileRef, i32)> = Vec::new();
        for p in game.players_alive() {
            if p.small_id == self.small_id {
                continue;
            }
            if !game.can_trade(p.small_id, self.small_id) {
                continue;
            }
            for u in p.units.iter() {
                // TS `tradingPorts`: `p.units(UnitType.Port)` is unfiltered by
                // active/under-construction status, unlike most other port lookups.
                if u.unit_type != unit_type::PORT {
                    continue;
                }
                let tile = u.tile as TileRef;
                let matches = source_components
                    .iter()
                    .any(|&c| game.has_water_component(tile, c));
                if matches {
                    candidates.push((p.small_id, u.id, tile, u.level.max(1)));
                }
            }
        }
        candidates.sort_by_key(|c| game.manhattan_dist(port_tile, c.2));

        let n = candidates.len();
        let bonus_threshold = game.wire.proximity_bonus_ports_nb(n);
        let debuff = game.wire.trade_ship_short_range_debuff();
        let mut weighted: Vec<(u16, i32)> = Vec::new();
        for (i, &(osid, uid, tile, level)) in candidates.iter().enumerate() {
            for _ in 0..level {
                weighted.push((osid, uid));
            }
            let dist = game.manhattan_dist(port_tile, tile) as f64;
            let too_close = dist < debuff;
            let close_bonus = (i as f64) < bonus_threshold;
            if !too_close && close_bonus {
                for _ in 0..level {
                    weighted.push((osid, uid));
                }
            }
            if !too_close && game.is_friendly(self.small_id, osid) {
                for _ in 0..level {
                    weighted.push((osid, uid));
                }
            }
        }
        weighted
    }
}

impl Execution for PortExecution {
    fn init(&mut self, _: &mut Game, tick: u32) {
        // TS `PortExecution.init`: `new PseudoRandom(mg.ticks())`, `checkOffset = mg.ticks() % 10`.
        self.random = Some(PseudoRandom::new(tick as i32));
        self.check_offset = tick % 10;
    }

    fn tick(&mut self, game: &mut Game, tick: u32) {
        if !self.active {
            return;
        }
        let still_active = game
            .player_by_small_id(self.small_id)
            .and_then(|p| p.units.iter().find(|u| u.id == self.unit_id))
            .is_some_and(|u| !u.under_construction);
        if !still_active {
            self.active = false;
            return;
        }
        if !game.unit_has_train_station(self.small_id, self.unit_id) {
            self.create_station(game);
        }

        // TS: "only check every 10 ticks for performance".
        if (tick + self.check_offset) % 10 != 0 {
            return;
        }
        if !self.should_spawn_trade_ship(game) {
            return;
        }
        let Some(port_tile) = game.unit_tile_of(self.small_id, self.unit_id) else {
            return;
        };
        let ports = self.trading_ports(game, port_tile);
        if ports.is_empty() {
            return;
        }
        let (_dst_small_id, dst_unit_id) = self
            .random
            .as_mut()
            .expect("PortExecution not initialized")
            .rand_element(&ports)
            .unwrap();
        game.add_execution(ExecEnum::TradeShip(TradeShipExecution::new(
            self.small_id,
            self.unit_id,
            dst_unit_id,
        )));
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        false
    }
}

// TS `PortExecution.test.ts`. Real production `trade_ship_short_range_debuff`
// (300) and `proximity_bonus_ports_nb` (`within(n/3, 4, n)`) are fixed
// constants with no native override mechanism (unlike TS's `TestConfig`,
// which the original test overrides to 0/10/100 for convenience) - these
// tests instead pick real tile distances that land on either side of the
// real 300-tile debuff threshold to exercise the exact same code paths.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schemas::unit_type;
    use crate::game::{Game, GameConfig as CoreGameConfig, PlayerInfo, PlayerType};
    use crate::map::{GameMap, MapMeta};

    /// A 1-row map, all water except two single-tile land "islands" `dist`
    /// tiles apart (one per port). The mini map is pure water, so
    /// `has_water_component`/`get_water_component` always report a single
    /// shared component everywhere - real per-map water connectivity isn't
    /// what these tests are about, only the level/proximity-bonus/
    /// short-range-debuff math downstream of "the ports share a trade
    /// network" is.
    fn game_with_two_ports(dist: u32) -> (Game, u16, TileRef, i32, u16, TileRef, i32) {
        let width = dist + 20;
        let player_x = 5u32;
        let other_x = 5 + dist;

        let mut terrain = vec![0u8; width as usize]; // all water
        terrain[player_x as usize] = 0x80; // isLand=1, magnitude=0 (Plains)
        terrain[other_x as usize] = 0x80;
        let meta = MapMeta {
            width,
            height: 1,
            num_land_tiles: 2,
        };
        let map = GameMap::from_terrain_bytes(&meta, &terrain).unwrap();

        let mini_width = width / 2 + 2;
        let mini_meta = MapMeta {
            width: mini_width,
            height: 1,
            num_land_tiles: 0,
        };
        let mini_map =
            GameMap::from_terrain_bytes(&mini_meta, &vec![0u8; mini_width as usize]).unwrap();

        let wire_cfg = crate::core::schemas::GameConfig {
            game_map: "tiny".into(),
            difficulty: "Medium".into(),
            donate_gold: false,
            donate_troops: false,
            game_type: "Singleplayer".into(),
            game_mode: "Free For All".into(),
            game_map_size: "Normal".into(),
            nations: crate::core::schemas::NationsConfig::Mode("default".into()),
            bots: 0,
            infinite_gold: false,
            infinite_troops: false,
            instant_build: false,
            random_spawn: false,
            doomsday_clock: None,
            disabled_units: None,
            player_teams: None,
            disable_alliances: None,
            spawn_immunity_duration: Some(0),
            starting_gold: None,
            gold_multiplier: None,
            max_timer_value: None,
            ranked_type: None,
        };
        let mut game = Game::new(
            String::new(),
            CoreGameConfig::default(),
            crate::core::config::Config::new(wire_cfg, false),
            map,
            mini_map,
            None,
        );
        game.end_spawn_phase();

        let player = game.add_from_info(&PlayerInfo {
            name: "player".into(),
            player_type: PlayerType::Human,
            client_id: Some("player_id".into()),
            id: "player_id".into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        });
        let other = game.add_from_info(&PlayerInfo {
            name: "other".into(),
            player_type: PlayerType::Human,
            client_id: Some("other_id".into()),
            id: "other_id".into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        });
        if let Some(p) = game.player_by_small_id_mut(player) {
            p.gold = 1_000_000_000;
        }
        if let Some(p) = game.player_by_small_id_mut(other) {
            p.gold = 1_000_000_000;
        }

        let player_tile = game.map.ref_xy(player_x, 0);
        let other_tile = game.map.ref_xy(other_x, 0);
        game.conquer(player, player_tile);
        game.conquer(other, other_tile);
        let player_port = game.build_unit(player, unit_type::PORT, player_tile);
        let other_port = game.build_unit(other, unit_type::PORT, other_tile);

        (
            game,
            player,
            player_tile,
            player_port,
            other,
            other_tile,
            other_port,
        )
    }

    fn increase_level(game: &mut Game, small_id: u16, unit_id: i32, times: i32) {
        for _ in 0..times {
            if let Some(u) = game.unit_mut(small_id, unit_id) {
                u.level += 1;
            }
        }
    }

    #[test]
    fn trading_ports_weighted_list_scales_with_level_when_not_bonus_eligible() {
        // Within the real 300-tile short-range debuff, so no proximity or
        // friendly bonus applies regardless of `proximityBonusPortsNb` -
        // isolates the pure per-level weighting.
        let (mut game, player, player_tile, player_port, other, _other_tile, other_port) =
            game_with_two_ports(10);
        let mut execution = PortExecution::new(player, player_port);
        execution.init(&mut game, 0);
        execution.tick(&mut game, 0);

        increase_level(&mut game, other, other_port, 2); // level 1 -> 3

        let ports = execution.trading_ports(&game, player_tile);
        assert_eq!(ports.len(), 3, "a level-3 port should appear 3 times, unweighted by any bonus");
        assert!(ports.iter().all(|&(sid, uid)| sid == other && uid == other_port));
    }

    #[test]
    fn trading_ports_gets_a_proximity_bonus_copy_when_far_enough_but_not_too_far() {
        // Past the real 300-tile short-range debuff, so the proximity bonus
        // (not cancelled by `tooClose`) applies for this lone candidate.
        let (mut game, player, player_tile, player_port, _other, _other_tile, _other_port) =
            game_with_two_ports(310);
        let mut execution = PortExecution::new(player, player_port);
        execution.init(&mut game, 0);
        execution.tick(&mut game, 0);

        let ports = execution.trading_ports(&game, player_tile);
        assert_eq!(
            ports.len(),
            2,
            "a lone level-1 port far enough away should get 1 base + 1 proximity-bonus copy"
        );
    }

    #[test]
    fn trading_ports_short_range_debuff_cancels_the_proximity_bonus() {
        // Within the real 300-tile short-range debuff cancels the bonus
        // that would otherwise apply to this lone candidate.
        let (mut game, player, player_tile, player_port, _other, _other_tile, _other_port) =
            game_with_two_ports(50);
        let mut execution = PortExecution::new(player, player_port);
        execution.init(&mut game, 0);
        execution.tick(&mut game, 0);

        let ports = execution.trading_ports(&game, player_tile);
        assert_eq!(
            ports.len(),
            1,
            "too-close-for-the-debuff should cancel the proximity bonus, leaving just the base copy"
        );
    }
}
