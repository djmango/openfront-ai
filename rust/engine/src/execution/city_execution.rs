//! City structure behavior (`CityExecution.ts`).

use super::{train_station_execution::TrainStationExecution, ExecEnum, Execution};
use crate::core::schemas::unit_type;
use crate::game::Game;

pub struct CityExecution {
    small_id: u16,
    unit_id: i32,
    active: bool,
    station_created: bool,
}

impl CityExecution {
    pub fn new(small_id: u16, unit_id: i32) -> Self {
        Self {
            small_id,
            unit_id,
            active: true,
            station_created: false,
        }
    }

    /// TS `CityExecution.createStation` - one-shot; only registers a station if a Factory is
    /// nearby AT THIS TICK (a Factory built nearby later does not retroactively retrigger this,
    /// though it can still pick the city up itself via `FactoryExecution.createStation`).
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
}

impl Execution for CityExecution {
    fn init(&mut self, _: &mut Game, _: u32) {}

    fn tick(&mut self, game: &mut Game, _: u32) {
        if !self.active {
            return;
        }
        // TS CityExecution holds the Unit; after capture, retarget owner.
        let Some(owner) = game.find_unit_owner(self.unit_id) else {
            self.active = false;
            return;
        };
        self.small_id = owner;
        if !self.station_created {
            self.create_station(game);
            self.station_created = true;
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }
}
