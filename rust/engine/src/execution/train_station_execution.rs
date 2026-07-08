//! Train station shell (`TrainStationExecution.ts` subset - rail network deferred).

use super::Execution;
use crate::game::Game;

pub struct TrainStationExecution {
    small_id: u16,
    unit_id: i32,
    /// TS `spawnTrains` on factory stations.
    spawn_trains: bool,
    active: bool,
}

impl TrainStationExecution {
    pub fn new(small_id: u16, unit_id: i32, spawn_trains: bool) -> Self {
        Self {
            small_id,
            unit_id,
            spawn_trains,
            active: true,
        }
    }
}

impl Execution for TrainStationExecution {
    fn init(&mut self, _: &mut Game, _: u32) {}

    fn tick(&mut self, game: &mut Game, _: u32) {
        if !self.active {
            return;
        }
        let still_active = game
            .player_by_small_id(self.small_id)
            .and_then(|p| p.units.iter().find(|u| u.id == self.unit_id))
            .is_some();
        if !still_active {
            self.active = false;
            return;
        }
        let _ = self.spawn_trains;
    }

    fn is_active(&self) -> bool {
        self.active
    }
}
