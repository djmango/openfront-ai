//! Human alliance intents (`AllianceRequestExecution.ts` subset).

use super::Execution;
use crate::game::Game;

pub struct AllianceRequestExecution {
    requestor_small_id: u16,
    recipient_id: String,
    active: bool,
    initialized: bool,
}

impl AllianceRequestExecution {
    pub fn new(requestor_small_id: u16, recipient_id: String) -> Self {
        Self {
            requestor_small_id,
            recipient_id,
            active: true,
            initialized: false,
        }
    }

    pub fn requestor_small_id(&self) -> u16 {
        self.requestor_small_id
    }

    pub fn recipient_id(&self) -> &str {
        &self.recipient_id
    }
}

impl Execution for AllianceRequestExecution {
    fn init(&mut self, game: &mut Game, tick: u32) {
        if !self.active || self.initialized {
            return;
        }
        self.initialized = true;
        if game.wire.disable_alliances() {
            self.active = false;
            return;
        }
        let Some(recipient) = game.player_by_id(&self.recipient_id) else {
            self.active = false;
            return;
        };
        let recipient_small_id = recipient.small_id;
        if !game.can_send_alliance_request(self.requestor_small_id, recipient_small_id) {
            self.active = false;
            return;
        }
        if !game.create_alliance_request(self.requestor_small_id, recipient_small_id, tick) {
            self.active = false;
            return;
        }
        // TS explicitly checks for a cross-request before calling createAllianceRequest and
        // sets active=false when one is found. Rust's create_alliance_request handles
        // cross-requests internally (via accept_alliance_pair) and returns true, so we
        // mirror TS by deactivating when no pending request was actually created.
        if game
            .pending_alliance_request(self.requestor_small_id, recipient_small_id)
            .is_none()
        {
            self.active = false;
        }
    }

    fn tick(&mut self, game: &mut Game, tick: u32) {
        if !self.active || !self.initialized {
            return;
        }
        let Some(recipient) = game.player_by_id(&self.recipient_id) else {
            self.active = false;
            return;
        };
        if let Some(req) = game.pending_alliance_request(self.requestor_small_id, recipient.small_id)
        {
            if tick.saturating_sub(req.created_at) > game.wire.alliance_request_duration() {
                game.reject_alliance_request(self.requestor_small_id, recipient.small_id);
                self.active = false;
            }
        } else {
            self.active = false;
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

pub struct AllianceRejectExecution {
    requestor_id: String,
    recipient_small_id: u16,
    active: bool,
}

impl AllianceRejectExecution {
    pub fn new(requestor_id: String, recipient_small_id: u16) -> Self {
        Self {
            requestor_id,
            recipient_small_id,
            active: true,
        }
    }
}

impl Execution for AllianceRejectExecution {
    fn init(&mut self, game: &mut Game, _: u32) {
        if !self.active {
            return;
        }
        let Some(requestor) = game.player_by_id(&self.requestor_id) else {
            self.active = false;
            return;
        };
        if game.is_allied_with(requestor.small_id, self.recipient_small_id) {
            self.active = false;
            return;
        }
        game.reject_alliance_request(requestor.small_id, self.recipient_small_id);
        self.active = false;
    }

    fn tick(&mut self, _: &mut Game, _: u32) {}

    fn is_active(&self) -> bool {
        self.active
    }
}

pub struct BreakAllianceExecution {
    requestor_small_id: u16,
    recipient_id: String,
    recipient_small_id: Option<u16>,
    active: bool,
}

impl BreakAllianceExecution {
    pub fn new(requestor_small_id: u16, recipient_id: String) -> Self {
        Self {
            requestor_small_id,
            recipient_id,
            recipient_small_id: None,
            active: true,
        }
    }
}

impl Execution for BreakAllianceExecution {
    fn init(&mut self, game: &mut Game, _: u32) {
        if !self.active {
            return;
        }
        let Some(recipient) = game.player_by_id(&self.recipient_id) else {
            self.active = false;
            return;
        };
        self.recipient_small_id = Some(recipient.small_id);
    }

    fn tick(&mut self, game: &mut Game, _: u32) {
        if !self.active {
            return;
        }
        let Some(recipient_small_id) = self.recipient_small_id else {
            self.active = false;
            return;
        };
        if game.is_allied_with(self.requestor_small_id, recipient_small_id) {
            game.break_alliance_between(self.requestor_small_id, recipient_small_id);
            game.update_relation(recipient_small_id, self.requestor_small_id, -100);

            let neighbors =
                crate::execution::ai_attack::nearby_player_small_ids(game, self.requestor_small_id);
            for neighbor in neighbors {
                if !game.players_on_same_team(neighbor, recipient_small_id) {
                    game.update_relation(neighbor, self.requestor_small_id, -40);
                }
            }
        }
        self.active = false;
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

pub struct AllianceExtensionExecution {
    from_small_id: u16,
    recipient_id: String,
    active: bool,
}

impl AllianceExtensionExecution {
    pub fn new(from_small_id: u16, recipient_id: String) -> Self {
        Self {
            from_small_id,
            recipient_id,
            active: true,
        }
    }
}

impl Execution for AllianceExtensionExecution {
    fn init(&mut self, game: &mut Game, _: u32) {
        if !self.active {
            return;
        }
        let Some(to) = game.player_by_id(&self.recipient_id) else {
            self.active = false;
            return;
        };
        if !game.player_by_small_id(self.from_small_id).is_some_and(|p| p.alive)
            || !to.alive
        {
            self.active = false;
            return;
        }
        game.add_alliance_extension_request(self.from_small_id, to.small_id);
        self.active = false;
    }

    fn tick(&mut self, _: &mut Game, _: u32) {}

    fn is_active(&self) -> bool {
        self.active
    }
}

#[cfg(test)]
mod tests {
    use super::{
        AllianceExtensionExecution, AllianceRejectExecution, AllianceRequestExecution,
        BreakAllianceExecution, Execution,
    };
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

    // Ported from AllianceRequestExecution.test.ts "Can create alliance by
    // counter-request".
    #[test]
    fn counter_request_creates_alliance_for_both_sides() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let player1 = add_human(&mut game, "player1");
        let player2 = add_human(&mut game, "player2");
        game.player_by_small_id_mut(player1).unwrap().tiles_owned = 1;
        game.player_by_small_id_mut(player2).unwrap().tiles_owned = 1;

        game.add_execution(ExecEnum::AllianceRequest(AllianceRequestExecution::new(
            player1,
            "player2".into(),
        )));
        game.execute_next_tick();

        game.add_execution(ExecEnum::AllianceRequest(AllianceRequestExecution::new(
            player2,
            "player1".into(),
        )));
        game.execute_next_tick();

        assert!(game.is_allied_with(player1, player2));
        assert!(game.is_allied_with(player2, player1));
    }

    // Ported from AllianceRequestExecution.test.ts "Can reject alliance request".
    #[test]
    fn reject_clears_outgoing_request_without_forming_alliance() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let player1 = add_human(&mut game, "player1");
        let player2 = add_human(&mut game, "player2");
        game.player_by_small_id_mut(player1).unwrap().tiles_owned = 1;
        game.player_by_small_id_mut(player2).unwrap().tiles_owned = 1;

        game.add_execution(ExecEnum::AllianceRequest(AllianceRequestExecution::new(
            player1,
            "player2".into(),
        )));
        game.execute_next_tick();

        game.add_execution(ExecEnum::AllianceReject(AllianceRejectExecution::new(
            "player1".into(),
            player2,
        )));
        game.execute_next_tick();

        assert!(!game.is_allied_with(player1, player2));
        assert!(!game.is_allied_with(player2, player1));
        assert_eq!(game.outgoing_alliance_requests(player1).len(), 0);
    }

    // Ported from AllianceRequestExecution.test.ts "Alliance request expires".
    #[test]
    fn unanswered_request_expires_after_configured_duration() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let player1 = add_human(&mut game, "player1");
        let player2 = add_human(&mut game, "player2");
        game.player_by_small_id_mut(player1).unwrap().tiles_owned = 1;
        game.player_by_small_id_mut(player2).unwrap().tiles_owned = 1;

        game.add_execution(ExecEnum::AllianceRequest(AllianceRequestExecution::new(
            player1,
            "player2".into(),
        )));
        game.execute_next_tick();

        assert_eq!(game.outgoing_alliance_requests(player1).len(), 1);

        for _ in 0..(game.wire.alliance_request_duration() + 1) {
            game.execute_next_tick();
        }

        assert_eq!(game.outgoing_alliance_requests(player1).len(), 0);
        assert!(!game.is_allied_with(player1, player2));
        assert!(!game.is_allied_with(player2, player1));
    }

    fn relation_value(game: &Game, from: u16, to: u16) -> f64 {
        game.player_by_small_id(from)
            .and_then(|player| player.relations.get(&to))
            .copied()
            .unwrap_or(0.0)
    }

    #[test]
    fn break_alliance_waits_until_tick_and_penalizes_recipient_relation() {
        let mut game = Game::default();
        let requestor = add_human(&mut game, "requestor");
        let recipient = add_human(&mut game, "recipient");
        assert!(game.create_alliance_request(requestor, recipient, 0));
        assert!(game.create_alliance_request(recipient, requestor, 0));
        game.update_relation(requestor, recipient, -120);
        game.update_relation(recipient, requestor, -50);
        game.end_spawn_phase();

        game.add_execution(ExecEnum::BreakAlliance(BreakAllianceExecution::new(
            requestor,
            "recipient".into(),
        )));
        game.execute_next_tick();

        assert!(game.is_allied_with(requestor, recipient));
        assert_eq!(relation_value(&game, requestor, recipient), -20.0);
        assert_eq!(relation_value(&game, recipient, requestor), 50.0);

        game.execute_next_tick();

        assert!(!game.is_allied_with(requestor, recipient));
        assert_eq!(relation_value(&game, requestor, recipient), -20.0);
        assert_eq!(relation_value(&game, recipient, requestor), -50.0);
    }

    #[test]
    fn invalid_break_deactivates_without_side_effects() {
        let mut game = Game::default();
        let requestor = add_human(&mut game, "requestor");
        let recipient = add_human(&mut game, "recipient");
        let mut execution = BreakAllianceExecution::new(requestor, "recipient".into());

        execution.init(&mut game, 0);
        execution.tick(&mut game, 1);

        assert!(!execution.is_active());
        assert!(!game.is_allied_with(requestor, recipient));
        assert_eq!(relation_value(&game, recipient, requestor), 0.0);
    }

    fn alliance_expires_at(game: &Game, a: u16, b: u16) -> u32 {
        game.player_alliances(a)
            .iter()
            .find(|al| al.requestor_small_id == b || al.recipient_small_id == b)
            .map(|al| al.expires_at)
            .expect("alliance should exist")
    }

    // Ported from AllianceExtensionExecution.test.ts "Successfully extends
    // existing alliance between Humans" (the "...between Human and non-Human"
    // case is identical for native, since `AllianceExtensionExecution` and
    // `Game::add_alliance_extension_request` don't special-case player type -
    // not re-tested separately).
    #[test]
    fn extension_only_applies_once_both_sides_have_requested_it() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let player1 = add_human(&mut game, "player1");
        let player2 = add_human(&mut game, "player2");
        assert!(game.create_alliance_request(player1, player2, game.ticks()));
        assert!(game.create_alliance_request(player2, player1, game.ticks()));
        assert!(game.is_allied_with(player1, player2));

        let expiration_before = alliance_expires_at(&game, player1, player2);

        game.add_execution(ExecEnum::AllianceExtension(
            AllianceExtensionExecution::new(player1, "player2".into()),
        ));
        game.execute_next_tick();
        assert_eq!(
            alliance_expires_at(&game, player1, player2),
            expiration_before
        );

        game.add_execution(ExecEnum::AllianceExtension(
            AllianceExtensionExecution::new(player2, "player1".into()),
        ));
        game.execute_next_tick();

        assert!(alliance_expires_at(&game, player1, player2) > expiration_before);
    }

    // Ported from AllianceExtensionExecution.test.ts "Fails gracefully if no
    // alliance exists".
    #[test]
    fn extension_request_without_an_existing_alliance_is_a_no_op() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let player1 = add_human(&mut game, "player1");
        let player2 = add_human(&mut game, "player2");

        game.add_execution(ExecEnum::AllianceExtension(
            AllianceExtensionExecution::new(player1, "player2".into()),
        ));
        game.execute_next_tick();

        assert!(!game.is_allied_with(player1, player2));
        assert!(!game.is_allied_with(player2, player1));
    }

    // Ported from the state-machine intent behind AllianceExtensionExecution
    // .test.ts's "Sends message to other player when one player requests
    // renewal" (the message-display assertions themselves have no native
    // equivalent - `Game::display_message` doesn't exist natively, matching the
    // documented UI-only dead-code carve-out in
    // docs/bot-ai-parity-nation-relations/README.md). What IS gameplay-relevant
    // and ported here: a second extension request from the SAME player must
    // not by itself extend the alliance (still needs the other side's
    // agreement) - i.e. `extension_requested_requestor` isn't toggled off by a
    // repeat call from the requestor.
    #[test]
    fn repeated_extension_request_from_the_same_player_does_not_self_extend() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let player1 = add_human(&mut game, "player1");
        let player2 = add_human(&mut game, "player2");
        assert!(game.create_alliance_request(player1, player2, game.ticks()));
        assert!(game.create_alliance_request(player2, player1, game.ticks()));
        let expiration_before = alliance_expires_at(&game, player1, player2);

        game.add_execution(ExecEnum::AllianceExtension(
            AllianceExtensionExecution::new(player1, "player2".into()),
        ));
        game.execute_next_tick();
        game.add_execution(ExecEnum::AllianceExtension(
            AllianceExtensionExecution::new(player1, "player2".into()),
        ));
        game.execute_next_tick();

        assert_eq!(
            alliance_expires_at(&game, player1, player2),
            expiration_before
        );
    }
}
