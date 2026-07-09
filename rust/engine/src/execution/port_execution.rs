//! Port structure behavior (`PortExecution.ts` subset - trade ships deferred).

use super::{train_station_execution::TrainStationExecution, ExecEnum, Execution};
use crate::core::schemas::unit_type;
use crate::game::Game;

pub struct PortExecution {
    small_id: u16,
    unit_id: i32,
    active: bool,
}

impl PortExecution {
    pub fn new(small_id: u16, unit_id: i32) -> Self {
        Self {
            small_id,
            unit_id,
            active: true,
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
}

impl Execution for PortExecution {
    fn init(&mut self, _: &mut Game, _: u32) {}

    fn tick(&mut self, game: &mut Game, _: u32) {
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
    }

    fn is_active(&self) -> bool {
        self.active
    }
}
