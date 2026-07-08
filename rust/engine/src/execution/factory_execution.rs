//! Factory structure behavior (`FactoryExecution.ts` subset - rail network deferred).

use super::{train_station_execution::TrainStationExecution, ExecEnum, Execution};
use crate::game::Game;

pub struct FactoryExecution {
    small_id: u16,
    unit_id: i32,
    active: bool,
    station_created: bool,
}

impl FactoryExecution {
    pub fn new(small_id: u16, unit_id: i32) -> Self {
        Self {
            small_id,
            unit_id,
            active: true,
            station_created: false,
        }
    }
}

impl Execution for FactoryExecution {
    fn init(&mut self, _: &mut Game, _: u32) {}

    fn tick(&mut self, game: &mut Game, _: u32) {
        if !self.active {
            return;
        }
        if !self.station_created {
            game.add_execution(ExecEnum::TrainStation(TrainStationExecution::new(
                self.small_id,
                self.unit_id,
                true,
            )));
            self.station_created = true;
        }
        let still_active = game
            .player_by_small_id(self.small_id)
            .and_then(|p| p.units.iter().find(|u| u.id == self.unit_id))
            .is_some_and(|u| !u.under_construction);
        if !still_active {
            self.active = false;
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }
}
