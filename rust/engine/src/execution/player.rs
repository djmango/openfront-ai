use super::Execution;
use crate::core::schemas::unit_type;
use crate::game::Game;
use crate::map::TileRef;
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

        reconcile_structure_ownership(game, self.small_id);

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

        let inc = game.troop_increase_rate_for(self.small_id);
        let gold = game.wire.gold_addition_rate(p.player_type);
        if let Some(pm) = game.player_by_small_id_mut(self.small_id) {
            pm.troops += inc;
            pm.gold += gold;
        }

        game.expire_alliances_for(self.small_id);

        crate::execution::player_clusters::maybe_remove_clusters(game, self.small_id, tick);
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        false
    }
}

fn is_structure(ty: &str) -> bool {
    matches!(
        ty,
        unit_type::CITY
            | unit_type::DEFENSE_POST
            | unit_type::SAM_LAUNCHER
            | unit_type::MISSILE_SILO
            | unit_type::PORT
            | unit_type::FACTORY
    )
}

/// TS `PlayerExecution.tick` structure ownership reconciliation.
fn reconcile_structure_ownership(game: &mut Game, small_id: u16) {
    let units: Vec<(i32, String, TileRef)> = game
        .player_by_small_id(small_id)
        .map(|p| {
            p.units
                .iter()
                .filter(|u| is_structure(&u.unit_type))
                .map(|u| (u.id, u.unit_type.clone(), u.tile as TileRef))
                .collect()
        })
        .unwrap_or_default();

    for (unit_id, utype, tile) in units {
        let owner = game.map.owner_id(tile);
        if owner == 0 {
            game.remove_unit(small_id, unit_id);
            continue;
        }
        if owner == small_id {
            continue;
        }
        if utype == unit_type::DEFENSE_POST {
            game.remove_unit(small_id, unit_id);
        } else {
            game.capture_unit(small_id, owner, unit_id);
        }
    }
}
