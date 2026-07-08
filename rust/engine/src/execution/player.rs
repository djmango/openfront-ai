use super::Execution;
use crate::game::Game;
use crate::util::simple_hash;

const TICKS_PER_CLUSTER_CALC: u32 = 20;

/// Troop income and death checks (TS `PlayerExecution` subset for hash parity).
pub struct PlayerExecution {
    small_id: u16,
    active: bool,
}

impl PlayerExecution {
    pub fn new(small_id: u16) -> Self {
        Self {
            small_id,
            active: true,
        }
    }

    pub fn small_id(&self) -> u16 {
        self.small_id
    }
}

impl Execution for PlayerExecution {
    fn init(&mut self, game: &mut Game, tick: u32) {
        if let Some(p) = game.player_by_small_id_mut(self.small_id) {
            let name_hash = simple_hash(&p.name);
            p.last_cluster_calc = tick + (name_hash as u32 % TICKS_PER_CLUSTER_CALC);
        }
    }

    fn tick(&mut self, game: &mut Game, tick: u32) {
        if !self.active {
            return;
        }
        let spawn = game.in_spawn_phase();
        let Some(p) = game.player_by_small_id(self.small_id).cloned() else {
            self.active = false;
            return;
        };
        if !p.alive || p.spawn_tile.is_none() {
            return;
        }

        // TS `isAlive()` is `_tiles.size > 0` - no income after elimination.
        if p.tiles_owned == 0 {
            if !spawn {
                if let Some(pm) = game.player_by_small_id_mut(self.small_id) {
                    pm.alive = false;
                }
                self.active = false;
            }
            return;
        }

        // TS delays troop income until the tick after spawn phase ends.
        if game.spawn_end_tick() == Some(tick) {
            return;
        }

        let inc = game
            .wire
            .troop_increase_rate(p.player_type, p.troops, p.tiles_owned);
        let gold = game.wire.gold_addition_rate(p.player_type);
        if let Some(pm) = game.player_by_small_id_mut(self.small_id) {
            pm.troops += inc;
            pm.gold += gold;
        }

        crate::execution::player_clusters::maybe_remove_clusters(game, self.small_id, tick);
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        false
    }
}
