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
    active: bool,
}

impl BreakAllianceExecution {
    pub fn new(requestor_small_id: u16, recipient_id: String) -> Self {
        Self {
            requestor_small_id,
            recipient_id,
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
        game.break_alliance_between(self.requestor_small_id, recipient.small_id);
        self.active = false;
    }

    fn tick(&mut self, _: &mut Game, _: u32) {}

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
