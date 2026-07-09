//! `TrainStationExecution.ts` - registers a station with the rail network on its first tick,
//! and (only for factory-owned stations, `spawn_trains=true`) periodically spawns trains using
//! a per-station `PseudoRandom(mg.ticks())` seeded RNG and `config().trainSpawnRate(...)`.

use super::{train_execution::TrainExecution, ExecEnum, Execution};
use crate::core::schemas::unit_type;
use crate::game::Game;
use crate::prng::PseudoRandom;
use crate::rail;

pub struct TrainStationExecution {
    small_id: u16,
    unit_id: i32,
    /// TS `spawnTrains` on factory stations.
    spawn_trains: bool,
    active: bool,
    station_id: Option<u32>,
    random: Option<PseudoRandom>,
    num_cars: usize,
    last_spawn_tick: u32,
    ticks_cooldown: u32,
}

impl TrainStationExecution {
    pub fn new(small_id: u16, unit_id: i32, spawn_trains: bool) -> Self {
        Self {
            small_id,
            unit_id,
            spawn_trains,
            active: true,
            station_id: None,
            random: None,
            num_cars: 5,
            last_spawn_tick: 0,
            ticks_cooldown: 10,
        }
    }

    fn should_spawn_train(&mut self, game: &Game) -> bool {
        let spawn_rate = game
            .wire
            .train_spawn_rate(game.unit_level_sum(self.small_id, unit_type::FACTORY));
        let level = game.unit_level_of(self.small_id, self.unit_id);
        let Some(random) = self.random.as_mut() else {
            return false;
        };
        for _ in 0..level {
            if random.chance(spawn_rate) {
                return true;
            }
        }
        false
    }

    fn spawn_train(&mut self, game: &mut Game, station_id: u32, current_tick: u32) {
        if !self.spawn_trains || self.random.is_none() {
            return;
        }
        if current_tick < self.last_spawn_tick + self.ticks_cooldown {
            return;
        }
        let Some(cluster) = game.rail_network.station_cluster(station_id) else {
            return;
        };
        if !rail::cluster_has_any_trade_destination(game, cluster, self.small_id) {
            return;
        }
        if !self.should_spawn_train(game) {
            return;
        }
        let destination = {
            let random = self.random.as_mut().unwrap();
            rail::cluster_random_trade_destination(game, cluster, self.small_id, random)
        };
        let Some(destination) = destination else {
            return;
        };
        if destination == station_id {
            return;
        }

        game.add_execution(ExecEnum::Train(TrainExecution::new(
            self.small_id,
            station_id,
            destination,
            self.num_cars,
        )));
        self.last_spawn_tick = current_tick;
    }
}

impl Execution for TrainStationExecution {
    fn init(&mut self, game: &mut Game, _tick: u32) {
        if self.spawn_trains {
            self.random = Some(PseudoRandom::new(game.ticks() as i32));
        }
    }

    fn tick(&mut self, game: &mut Game, tick: u32) {
        if !self.active {
            return;
        }
        if !game.unit_exists(self.small_id, self.unit_id) {
            self.active = false;
            return;
        }
        if self.station_id.is_none() {
            let unit_type_str = game
                .unit_type_of(self.small_id, self.unit_id)
                .unwrap_or_default();
            let id = rail::connect_station(game, self.small_id, self.unit_id, &unit_type_str);
            self.station_id = Some(id);
        }
        let Some(station_id) = self.station_id else {
            return;
        };
        if !rail::station_active(game, &game.rail_network, station_id) {
            self.active = false;
            return;
        }
        if self.spawn_trains {
            self.spawn_train(game, station_id, tick);
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        false
    }
}
