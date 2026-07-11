use super::Execution;

pub struct MarkDisconnectedExecution {
    small_id: u16,
    disconnected: bool,
}

impl MarkDisconnectedExecution {
    pub fn new(small_id: u16, disconnected: bool) -> Self {
        Self {
            small_id,
            disconnected,
        }
    }
}

impl Execution for MarkDisconnectedExecution {
    fn init(&mut self, game: &mut crate::game::Game, _: u32) {
        if let Some(p) = game.player_by_small_id_mut(self.small_id) {
            p.is_disconnected = self.disconnected;
        }
    }

    fn tick(&mut self, _: &mut crate::game::Game, _: u32) {}

    fn is_active(&self) -> bool {
        false
    }
}

// Ported from Disconnected.test.ts (the state/execution tests; the
// "Disconnected team member interactions" describe block's ship/warship/team
// scenarios are ported separately, closer to the code they exercise - see
// game.rs/warship.rs/attack.rs).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::ExecEnum;
    use crate::game::{Game, PlayerInfo, PlayerType};

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

    #[test]
    fn players_initialize_as_not_disconnected() {
        let mut game = Game::default();
        let p1 = add_human(&mut game, "p1");
        let p2 = add_human(&mut game, "p2");
        assert!(!game.player_by_small_id(p1).unwrap().is_disconnected);
        assert!(!game.player_by_small_id(p2).unwrap().is_disconnected);
    }

    #[test]
    fn mark_disconnected_toggles_state() {
        let mut game = Game::default();
        let p1 = add_human(&mut game, "p1");
        game.player_by_small_id_mut(p1).unwrap().is_disconnected = true;
        assert!(game.player_by_small_id(p1).unwrap().is_disconnected);
        game.player_by_small_id_mut(p1).unwrap().is_disconnected = false;
        assert!(!game.player_by_small_id(p1).unwrap().is_disconnected);
    }

    #[test]
    fn disconnected_state_is_visible_through_shared_game_state() {
        // TS's "player view" describe block queries `game.player(id)` again to
        // confirm the update is visible; native has a single source of truth
        // (no separate view object), so this collapses to re-reading the
        // player directly.
        let mut game = Game::default();
        let p2 = add_human(&mut game, "p2");
        game.player_by_small_id_mut(p2).unwrap().is_disconnected = true;
        assert!(game.player_by_small_id(p2).unwrap().is_disconnected);
        game.player_by_small_id_mut(p2).unwrap().is_disconnected = false;
        assert!(!game.player_by_small_id(p2).unwrap().is_disconnected);
    }

    #[test]
    fn disconnected_state_persists_across_ticks() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let p2 = add_human(&mut game, "p2");
        game.player_by_small_id_mut(p2).unwrap().is_disconnected = true;
        for _ in 0..5 {
            game.execute_next_tick();
        }
        assert!(game.player_by_small_id(p2).unwrap().is_disconnected);
    }

    #[test]
    fn execution_marks_player_disconnected_and_deactivates() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let p1 = add_human(&mut game, "p1");
        game.add_execution(ExecEnum::MarkDisconnected(MarkDisconnectedExecution::new(
            p1, true,
        )));
        game.execute_next_tick();
        assert!(game.player_by_small_id(p1).unwrap().is_disconnected);
        // `is_active()` is always false (TS's execution is a one-shot: it
        // fires entirely in `init()`).
        assert!(!MarkDisconnectedExecution::new(p1, true).is_active());
    }

    #[test]
    fn multiple_executions_handle_different_players_independently() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let p1 = add_human(&mut game, "p1");
        let p2 = add_human(&mut game, "p2");
        game.add_execution(ExecEnum::MarkDisconnected(MarkDisconnectedExecution::new(
            p1, true,
        )));
        game.add_execution(ExecEnum::MarkDisconnected(MarkDisconnectedExecution::new(
            p2, false,
        )));
        game.execute_next_tick();
        assert!(game.player_by_small_id(p1).unwrap().is_disconnected);
        assert!(!game.player_by_small_id(p2).unwrap().is_disconnected);
    }

    #[test]
    fn not_active_during_spawn_phase() {
        let exec = MarkDisconnectedExecution::new(1, true);
        assert!(!exec.active_during_spawn());
    }

    #[test]
    fn last_execution_wins_for_same_player_same_tick() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let p1 = add_human(&mut game, "p1");
        game.add_execution(ExecEnum::MarkDisconnected(MarkDisconnectedExecution::new(
            p1, true,
        )));
        game.add_execution(ExecEnum::MarkDisconnected(MarkDisconnectedExecution::new(
            p1, false,
        )));
        game.execute_next_tick();
        assert!(!game.player_by_small_id(p1).unwrap().is_disconnected);
    }

    #[test]
    fn setting_same_disconnected_state_repeatedly_is_a_no_op() {
        let mut game = Game::default();
        let p1 = add_human(&mut game, "p1");
        for _ in 0..3 {
            game.player_by_small_id_mut(p1).unwrap().is_disconnected = true;
        }
        assert!(game.player_by_small_id(p1).unwrap().is_disconnected);
        for _ in 0..3 {
            game.player_by_small_id_mut(p1).unwrap().is_disconnected = false;
        }
        assert!(!game.player_by_small_id(p1).unwrap().is_disconnected);
    }

    #[test]
    fn execution_with_already_matching_state_is_idempotent() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let p1 = add_human(&mut game, "p1");
        game.player_by_small_id_mut(p1).unwrap().is_disconnected = true;
        game.add_execution(ExecEnum::MarkDisconnected(MarkDisconnectedExecution::new(
            p1, true,
        )));
        game.execute_next_tick();
        assert!(game.player_by_small_id(p1).unwrap().is_disconnected);
    }
}
