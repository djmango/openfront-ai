//! City structure behavior (`CityExecution.ts` subset  -  train stations deferred).

use super::Execution;
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
}

impl Execution for CityExecution {
    fn init(&mut self, _: &mut Game, _: u32) {}

    fn tick(&mut self, game: &mut Game, _: u32) {
        if !self.active {
            return;
        }
        if !self.station_created {
            // TS `CityExecution.createStation`  -  TrainStationExecution deferred for parity scope.
            self.station_created = true;
        }
        let still_active = game
            .player_by_small_id(self.small_id)
            .and_then(|p| p.units.iter().find(|u| u.id == self.unit_id))
            .is_some();
        if !still_active {
            self.active = false;
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }
}
