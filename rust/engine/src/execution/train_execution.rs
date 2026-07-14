//! `TrainExecution.ts` - a train's full lifecycle: pathing between stations, moving one
//! `speed`-tile hop per tick, delivering trade gold on arrival, and updating unit tiles
//! (which directly feed `game_hash` via `hash::unit_hash_js`).
//!
//! Client-only concerns are dropped: TS records a `MotionPlanRecord` purely for client
//! interpolation (never read back by the simulation, see `GameImpl.recordMotionPlan`); this
//! port skips it entirely.

use super::Execution;
use crate::core::schemas::unit_type;
use crate::game::Game;
use crate::rail;

pub struct TrainExecution {
    active: bool,
    owner_small_id: u16,
    source_station: u32,
    destination_station: u32,
    num_cars: usize,

    initialized: bool,
    /// Station id path; front gets popped off as the train advances (TS `stations.shift()`).
    stations: Vec<u32>,
    train_unit_id: Option<i32>,
    /// "back to front": index 0 = tail engine (farthest back), last = closest carriage.
    car_unit_ids: Vec<i32>,
    current_railroad_tiles: Vec<u32>,
    current_tile: usize,
    used_tiles: std::collections::VecDeque<u32>,
    trade_stops_visited: u32,

    speed: usize,
    spacing: usize,
}

impl TrainExecution {
    pub fn new(
        owner_small_id: u16,
        source_station: u32,
        destination_station: u32,
        num_cars: usize,
    ) -> Self {
        Self {
            active: true,
            owner_small_id,
            source_station,
            destination_station,
            num_cars,
            initialized: false,
            stations: Vec::new(),
            train_unit_id: None,
            car_unit_ids: Vec::new(),
            current_railroad_tiles: Vec::new(),
            current_tile: 0,
            used_tiles: std::collections::VecDeque::new(),
            trade_stops_visited: 0,
            speed: 2,
            spacing: 2,
        }
    }

    fn init_impl(&mut self, game: &mut Game) {
        self.initialized = true;
        let stations = rail::find_stations_path(
            game,
            &game.rail_network,
            self.source_station,
            self.destination_station,
        );
        if stations.len() <= 1 {
            self.active = false;
            return;
        }
        self.stations = stations;

        let Some(tiles) =
            rail::oriented_railroad_tiles(&game.rail_network, self.stations[0], self.stations[1])
        else {
            self.active = false;
            return;
        };
        self.current_railroad_tiles = tiles;

        let Some(spawn_tile) = rail::station_tile(game, &game.rail_network, self.stations[0])
        else {
            self.active = false;
            return;
        };
        if !game.is_land(spawn_tile) {
            self.active = false;
            return;
        }

        let train = game.build_unit(self.owner_small_id, unit_type::TRAIN, spawn_tile);
        self.train_unit_id = Some(train);
        // Tail engine first (farthest back), then `numCars` carriages.
        let tail = game.build_unit(self.owner_small_id, unit_type::TRAIN, spawn_tile);
        self.car_unit_ids.push(tail);
        for _ in 0..self.num_cars {
            let car = game.build_unit(self.owner_small_id, unit_type::TRAIN, spawn_tile);
            self.car_unit_ids.push(car);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schemas::unit_type;
    use crate::game::{Game, PlayerInfo, PlayerType};
    use crate::rail;
    use crate::test_util::plains_game;

    fn add_nation(game: &mut Game, id: &str) -> u16 {
        game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type: PlayerType::Nation,
            client_id: None,
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        })
    }

    #[test]
    fn destination_trade_gate_uses_live_station_owner() {
        let mut game = plains_game(40, 40);
        let stale_owner = add_nation(&mut game, "stale");
        let live_owner = add_nation(&mut game, "live");
        let train_owner = add_nation(&mut game, "train");

        let tile = game.map.ref_xy(5, 5);
        game.conquer(stale_owner, tile);
        let city_id = game.build_unit(stale_owner, unit_type::CITY, tile);
        let station_id = rail::connect_station(&mut game, stale_owner, city_id, unit_type::CITY);

        game.capture_unit(stale_owner, live_owner, city_id);
        // Regression shape from curriculum parity: a TrainStation object follows
        // the live Unit owner in TS, so this lifecycle gate must not trust a
        // stale cached station owner.
        game.rail_network
            .stations
            .get_mut(&station_id)
            .unwrap()
            .owner_small_id = stale_owner;
        game.add_embargo(stale_owner, train_owner, false, 0);

        let mut exec = TrainExecution::new(train_owner, station_id, station_id, 0);
        exec.stations = vec![station_id, station_id];

        assert!(
            exec.can_trade_with_destination(&game),
            "live owner can trade with train owner even when cached owner is embargoed"
        );
    }
}

impl Execution for TrainExecution {
    fn init(&mut self, game: &mut Game, _tick: u32) {
        self.init_impl(game);
    }

    fn tick(&mut self, game: &mut Game, _tick: u32) {
        if !self.initialized {
            // `add_execution` inits same-tick for most paths but guard defensively.
            self.init_impl(game);
        }
        if !self.active {
            return;
        }
        let Some(train_id) = self.train_unit_id else {
            self.delete_train(game);
            return;
        };
        if !game.unit_exists(self.owner_small_id, train_id)
            || !self.active_source_or_destination(game)
        {
            self.delete_train(game);
            return;
        }

        match self.get_next_tile(game) {
            Some(tile) => self.update_cars_positions(game, tile),
            None => {
                self.delete_train(game);
            }
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        false
    }
}

impl TrainExecution {
    fn active_source_or_destination(&self, game: &Game) -> bool {
        self.stations.len() > 1
            && rail::station_active(game, &game.rail_network, self.stations[1])
            && rail::station_active(game, &game.rail_network, self.stations[0])
    }

    /// TS `TrainExecution.canTradeWithDestination` - `stations[1].tradeAvailable(this.player)`.
    fn can_trade_with_destination(&self, game: &Game) -> bool {
        self.stations.len() > 1
            && game
                .rail_network
                .stations
                .get(&self.stations[1])
                .is_some_and(|st| {
                    let owner = game
                        .find_unit_owner(st.unit_id)
                        .unwrap_or(st.owner_small_id);
                    owner == self.owner_small_id || game.can_trade(owner, self.owner_small_id)
                })
    }

    fn save_traversed_tiles(&mut self, from: usize, speed: usize) {
        let mut idx = from;
        for _ in 0..speed {
            if idx >= self.current_railroad_tiles.len() {
                break;
            }
            self.save_tile(self.current_railroad_tiles[idx]);
            idx += 1;
        }
    }

    fn save_tile(&mut self, tile: u32) {
        self.used_tiles.push_back(tile);
        let cap = self.car_unit_ids.len() * self.spacing + 3;
        while self.used_tiles.len() > cap {
            self.used_tiles.pop_front();
        }
    }

    fn next_station(&mut self, game: &Game) -> bool {
        if self.stations.len() > 2 {
            self.stations.remove(0);
            if let Some(tiles) = rail::oriented_railroad_tiles(
                &game.rail_network,
                self.stations[0],
                self.stations[1],
            ) {
                self.current_railroad_tiles = tiles;
                return true;
            }
        }
        false
    }

    fn station_reached(&mut self, game: &mut Game) {
        let dest = self.stations[1];
        rail::on_train_stop(game, dest, self.owner_small_id, self.trade_stops_visited);
        if let Some(t) = rail::station_unit_type(&game.rail_network, dest) {
            if t == unit_type::CITY || t == unit_type::PORT {
                self.trade_stops_visited += 1;
            }
        }
    }

    fn get_next_tile(&mut self, game: &mut Game) -> Option<u32> {
        if self.current_railroad_tiles.is_empty() || !self.can_trade_with_destination(game) {
            return None;
        }
        self.save_traversed_tiles(self.current_tile, self.speed);
        self.current_tile += self.speed;
        let leftover = self.current_tile as i64 - self.current_railroad_tiles.len() as i64;
        if leftover >= 0 {
            self.station_reached(game);
            if !self.next_station(game) {
                return None;
            }
            self.current_tile = leftover as usize;
            self.save_traversed_tiles(0, self.current_tile);
        }
        self.current_railroad_tiles.get(self.current_tile).copied()
    }

    fn update_cars_positions(&mut self, game: &mut Game, new_tile: u32) {
        if !self.car_unit_ids.is_empty() {
            let used: Vec<u32> = self.used_tiles.iter().copied().collect();
            for i in (0..self.car_unit_ids.len()).rev() {
                let car_tile_index = (i + 1) * self.spacing + 2;
                if used.len() > car_tile_index {
                    game.move_unit(
                        self.owner_small_id,
                        self.car_unit_ids[i],
                        used[car_tile_index],
                    );
                }
            }
        }
        if let Some(train_id) = self.train_unit_id {
            game.move_unit(self.owner_small_id, train_id, new_tile);
        }
    }

    fn delete_train(&mut self, game: &mut Game) {
        self.active = false;
        if let Some(id) = self.train_unit_id.take() {
            if game.unit_exists(self.owner_small_id, id) {
                game.remove_unit(self.owner_small_id, id);
            }
        }
        for id in self.car_unit_ids.drain(..) {
            if game.unit_exists(self.owner_small_id, id) {
                game.remove_unit(self.owner_small_id, id);
            }
        }
    }
}
