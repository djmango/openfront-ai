//! Defense post structure (`DefensePostExecution.ts` subset  -  shell targeting deferred).

use super::Execution;
use crate::game::Game;

pub struct DefensePostExecution {
    small_id: u16,
    unit_id: i32,
    active: bool,
}

impl DefensePostExecution {
    pub fn new(small_id: u16, unit_id: i32) -> Self {
        Self {
            small_id,
            unit_id,
            active: true,
        }
    }
}

impl Execution for DefensePostExecution {
    fn init(&mut self, _: &mut Game, _: u32) {}

    fn tick(&mut self, game: &mut Game, _: u32) {
        if !self.active {
            return;
        }
        // TS DefensePostExecution holds the Unit; after capture, retarget owner.
        let Some(owner) = game.find_unit_owner(self.unit_id) else {
            self.active = false;
            return;
        };
        self.small_id = owner;
        let Some(u) = game.unit(self.small_id, self.unit_id) else {
            self.active = false;
            return;
        };
        if u.under_construction {
            return;
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::schemas::unit_type::{PORT, WARSHIP};

    // Ported from `openfront/tests/ShellRandom.test.ts`'s "Defense post shell attacks have
    // random damage". TS's own `DefensePostExecution.tick()` has its ship-targeting logic
    // commented out (`// TODO: Reconsider how/if defense posts target ships.`), so `target`
    // never becomes non-null and no shell is ever fired - the TS test's own assertions are
    // guarded behind `if (damages.length > 0)`, making it a no-op check today. This mirrors
    // that guard rather than asserting a shell fires, since native intentionally matches
    // TS's current (deferred) behavior, not a more-complete one TS itself doesn't have.
    #[test]
    fn defense_post_never_fires_since_ship_targeting_is_deferred_upstream() {
        let mut game = crate::test_util::plains_game(20, 20);
        game.add_player(crate::game::Player {
            id: "1".to_string(),
            small_id: 1,
            player_type: crate::game::PlayerType::Human,
            ..Default::default()
        });
        game.add_player(crate::game::Player {
            id: "2".to_string(),
            small_id: 2,
            player_type: crate::game::PlayerType::Human,
            ..Default::default()
        });
        let post_id = game.build_unit(1, PORT, game.map.ref_xy(0, 0)); // stand-in structure tile
        let target = game.build_unit(2, WARSHIP, game.map.ref_xy(1, 0));
        let initial_health = game.unit(2, target).unwrap().health;

        let mut post = DefensePostExecution::new(1, post_id);
        post.init(&mut game, 0);
        for tick in 0..100u32 {
            post.tick(&mut game, tick);
            game.execute_next_tick();
        }

        assert_eq!(
            game.unit(2, target).unwrap().health,
            initial_health,
            "no shell should ever fire while ship-targeting stays deferred, matching TS"
        );
    }
}
