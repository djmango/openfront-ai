//! Voluntary structure deletion (`DeleteUnitExecution.ts`). Client intent type
//! `"delete_unit"` - previously fell through `intent.rs`'s catch-all to `NoOp`, silently
//! dropping every player-issued delete-unit command.

use super::Execution;
use crate::game::Game;

pub struct DeleteUnitExecution {
    small_id: u16,
    unit_id: i32,
    active: bool,
    /// Set once `init()` validates and marks the unit - mirrors TS `this.unit` being
    /// non-null only after a successful `init()` (a rejected execution has no unit to poll
    /// in `tick()`, matching TS's `!this.unit` early-return).
    marked: bool,
}

impl DeleteUnitExecution {
    pub fn new(small_id: u16, unit_id: i32) -> Self {
        Self {
            small_id,
            unit_id,
            active: true,
            marked: false,
        }
    }
}

impl Execution for DeleteUnitExecution {
    fn init(&mut self, game: &mut Game, _tick: u32) {
        if !self.active {
            return;
        }
        let Some(unit) = game.unit(self.small_id, self.unit_id) else {
            self.active = false;
            return;
        };
        let tile = unit.tile as crate::map::TileRef;

        if game.map.owner_id(tile) != self.small_id {
            self.active = false;
            return;
        }
        if !game.is_land(tile) {
            self.active = false;
            return;
        }
        if game.in_spawn_phase() {
            self.active = false;
            return;
        }
        if !game.can_delete_unit(self.small_id) {
            self.active = false;
            return;
        }

        game.record_delete_unit(self.small_id);
        game.mark_unit_for_deletion(self.small_id, self.unit_id);
        self.marked = true;
    }

    fn tick(&mut self, game: &mut Game, _tick: u32) {
        if !self.active || !self.marked {
            return;
        }
        if game.unit(self.small_id, self.unit_id).is_none() {
            self.active = false;
            return;
        }
        if game.is_unit_overdue_deletion(self.small_id, self.unit_id) {
            game.remove_unit(self.small_id, self.unit_id);
            self.active = false;
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::ExecEnum;
    use crate::game::{Game, PlayerInfo, PlayerType};
    use crate::test_util::plains_game;

    fn add_human(game: &mut Game, id: &str) -> u16 {
        game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type: PlayerType::Human,
            client_id: Some(id.into()),
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        })
    }

    /// Common fixture: a `plains_game` (already past spawn phase, matching TS's
    /// `executeTicks(game, deleteUnitCooldown + 1)` warm-up) with two players who each own
    /// one tile, plus one `City` unit owned by `p1` on `p1`'s own tile.
    fn setup() -> (Game, u16, u16, i32) {
        let mut game = plains_game(20, 20);
        let p1 = add_human(&mut game, "p1");
        let p2 = add_human(&mut game, "p2");
        let p1_tile = game.map.ref_xy(0, 0);
        let p2_tile = game.map.ref_xy(10, 0);
        game.map.set_owner_id(p1_tile, p1);
        game.map.set_owner_id(p2_tile, p2);
        let unit = game.build_unit(p1, crate::core::schemas::unit_type::CITY, p1_tile);
        // TS beforeEach's `executeTicks(game, deleteUnitCooldown() + 1)` warm-up - both
        // players' `lastDeleteUnitTick` default to "never" (-1), which only clears the
        // cooldown gate once enough ticks have actually elapsed since game start.
        for _ in 0..Game::DELETE_UNIT_COOLDOWN_TICKS + 1 {
            game.execute_next_tick();
        }
        (game, p1, p2, unit)
    }

    #[test]
    fn rejects_deleting_a_unit_not_owned_by_the_caller() {
        let (mut game, _p1, p2, _unit) = setup();
        let p2_tile = game.map.ref_xy(10, 0);
        let enemy_unit = game.build_unit(p2, crate::core::schemas::unit_type::CITY, p2_tile);

        // `p2` owns `enemy_unit`, but the execution is issued as `p1` deleting it - looking
        // it up under `p1`'s own unit list (native's equivalent of TS's `owner() !== player`
        // check) fails to find it at all.
        let p1 = _p1;
        let mut exec = DeleteUnitExecution::new(p1, enemy_unit);
        exec.init(&mut game, 0);

        assert!(!exec.is_active());
        assert!(!game.is_unit_marked_for_deletion(p2, enemy_unit));
    }

    #[test]
    fn rejects_deleting_a_unit_on_territory_the_owner_no_longer_holds() {
        let (mut game, p1, p2, unit) = setup();
        let p2_tile = game.map.ref_xy(10, 0);
        game.move_unit(p1, unit, p2_tile);

        let mut exec = DeleteUnitExecution::new(p1, unit);
        exec.init(&mut game, 0);

        assert!(!exec.is_active());
        assert!(!game.is_unit_marked_for_deletion(p1, unit));
        let _ = p2;
    }

    #[test]
    fn rejects_deleting_during_spawn_phase() {
        // `plains_game` (used by `setup()`) always calls `end_spawn_phase()`, so this test
        // builds its own tiny still-in-spawn-phase fixture instead.
        use crate::map::{GameMap, MapMeta};
        const LAND_PLAINS: u8 = 0b1000_0000;
        let meta = MapMeta {
            width: 4,
            height: 4,
            num_land_tiles: 16,
        };
        let map = GameMap::from_terrain_bytes(&meta, &vec![LAND_PLAINS; 16]).unwrap();
        let mut game = Game::default();
        game.map = map.clone();
        game.mini_map = map;
        game.bfs = crate::water::BfsScratch::new(16);
        game.water_astar = crate::water::WaterAstarScratch::new(16);
        game.mini_water_astar = crate::water::WaterAstarScratch::new(16);
        assert!(game.in_spawn_phase());

        let p1 = add_human(&mut game, "p1");
        let tile = game.map.ref_xy(0, 0);
        game.map.set_owner_id(tile, p1);
        let unit = game.build_unit(p1, crate::core::schemas::unit_type::CITY, tile);

        let mut exec = DeleteUnitExecution::new(p1, unit);
        exec.init(&mut game, 0);

        assert!(!exec.is_active());
        assert!(!game.is_unit_marked_for_deletion(p1, unit));
    }

    #[test]
    fn rejects_deleting_while_the_cooldown_is_active() {
        let (mut game, p1, _p2, unit) = setup();
        game.record_delete_unit(p1);

        let mut exec = DeleteUnitExecution::new(p1, unit);
        exec.init(&mut game, 0);

        assert!(!exec.is_active());
        assert!(!game.is_unit_marked_for_deletion(p1, unit));
    }

    #[test]
    fn marks_the_unit_when_all_conditions_are_met() {
        let (mut game, p1, _p2, unit) = setup();

        let mut exec = DeleteUnitExecution::new(p1, unit);
        exec.init(&mut game, 0);

        assert!(game.is_unit_marked_for_deletion(p1, unit));
    }

    #[test]
    fn deletes_the_unit_once_the_mark_duration_elapses() {
        let (mut game, p1, _p2, unit) = setup();

        game.add_execution(ExecEnum::DeleteUnit(DeleteUnitExecution::new(p1, unit)));
        game.execute_next_tick();
        assert!(game.is_unit_marked_for_deletion(p1, unit));
        assert!(!game.is_unit_overdue_deletion(p1, unit));

        for _ in 0..Game::DELETION_MARK_DURATION_TICKS + 1 {
            game.execute_next_tick();
        }
        assert!(game.unit(p1, unit).is_none());
    }

    #[test]
    fn capturing_the_unit_clears_the_pending_deletion() {
        let (mut game, p1, p2, unit) = setup();

        game.add_execution(ExecEnum::DeleteUnit(DeleteUnitExecution::new(p1, unit)));
        game.execute_next_tick();
        assert!(game.is_unit_marked_for_deletion(p1, unit));

        game.capture_unit(p1, p2, unit);

        assert!(game.unit(p1, unit).is_none());
        assert!(!game.is_unit_marked_for_deletion(p2, unit));
        assert!(game.unit(p2, unit).is_some());
    }
}
