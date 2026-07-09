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
                if u.unit_type != unit_type::PORT || u.under_construction {
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
        let (dst_small_id, dst_unit_id) = self
            .random
            .as_mut()
            .expect("PortExecution not initialized")
            .rand_element(&ports)
            .unwrap();
        if std::env::var("DBG_TRADE_SHIP").is_ok() {
            let owner_id = game
                .player_by_small_id(self.small_id)
                .map(|p| p.id.clone())
                .unwrap_or_default();
            let dst_id = game
                .player_by_small_id(dst_small_id)
                .map(|p| p.id.clone())
                .unwrap_or_default();
            eprintln!(
                "DBG_TRADE_SHIP tick={} owner={}({}) port_tile={} candidates_weighted_len={} chosen_dst={}({}) dst_tile={} dst_unit_id={}",
                tick, owner_id, self.small_id, port_tile, ports.len(), dst_id, dst_small_id,
                game.unit_tile_of(dst_small_id, dst_unit_id).unwrap_or(0), dst_unit_id
            );
        }
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
