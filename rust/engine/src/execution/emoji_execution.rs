//! Human/nation emoji intents (TS `EmojiExecution.ts` + `respondToEmoji`).
//!
//! Display/network emoji updates are hash-neutral, but `respondToEmoji` mutates
//! nation relations (🖕 → -100, 🤡 → -10, peaceful Easy → +15). Skipping the
//! human `"emoji"` intent left those relations missing (XjHWuiUa: Ukraine 🖕 at
//! turn 617 → Republika Srpska Hostile on TS only → hated-target / PRNG drift).
//!
//! Reply emoji content is display-only (local `PseudoRandom(ticks)` pick), but
//! the reply execution still records the nation's `outgoingEmojis_` cooldown,
//! which gates later casual-emoji PRNG draws. We schedule a no-op-index reply
//! so cooldown matches without needing the display table.

use super::{ExecEnum, Execution};
use crate::game::{Game, PlayerType};

/// Flattened tip `emojiTable` indices that trigger relation side-effects.
const EMOJI_MIDDLE_FINGER: i32 = 14; // 🖕
const EMOJI_CLOWN: i32 = 11; // 🤡
const EMOJI_PEACEFUL: [i32; 5] = [
    2,  // 🥰
    16, // 👏
    27, // 🕊️
    28, // 🏳️
    48, // ❤️
];

/// Dummy reply index — content is hash-neutral; cooldown recording is what matters.
const REPLY_EMOJI_PLACEHOLDER: i32 = 0;

pub struct EmojiExecution {
    requestor_small_id: u16,
    /// `None` = AllPlayers.
    recipient_id: Option<String>,
    emoji: i32,
    recipient_small_id: Option<Option<u16>>,
    active: bool,
}

impl EmojiExecution {
    pub fn new(requestor_small_id: u16, recipient_id: Option<String>, emoji: i32) -> Self {
        Self {
            requestor_small_id,
            recipient_id,
            emoji,
            recipient_small_id: None,
            active: true,
        }
    }
}

impl Execution for EmojiExecution {
    fn init(&mut self, game: &mut Game, _tick: u32) {
        match &self.recipient_id {
            None => self.recipient_small_id = Some(None),
            Some(id) => match game.player_by_id(id) {
                Some(p) => self.recipient_small_id = Some(Some(p.small_id)),
                None => {
                    self.active = false;
                }
            },
        }
    }

    fn tick(&mut self, game: &mut Game, _tick: u32) {
        if !self.active {
            return;
        }
        self.active = false;

        let Some(recipient_opt) = self.recipient_small_id else {
            return;
        };
        if !game.can_send_emoji(self.requestor_small_id, recipient_opt) {
            return;
        }

        // TS `requestor.sendEmoji` — records cooldown at the tick this execution runs.
        record_emoji_send_now(game, self.requestor_small_id, recipient_opt);

        // TS `respondToEmoji` — relation mutations only when recipient is a Nation.
        let Some(recipient_sid) = recipient_opt else {
            return;
        };
        let Some(recipient) = game.player_by_small_id(recipient_sid) else {
            return;
        };
        if recipient.player_type != PlayerType::Nation {
            return;
        }
        if !game.can_send_emoji(recipient_sid, Some(self.requestor_small_id)) {
            return;
        }

        let requestor_id = game
            .player_by_small_id(self.requestor_small_id)
            .map(|p| p.id.clone());

        if self.emoji == EMOJI_MIDDLE_FINGER {
            game.update_relation(recipient_sid, self.requestor_small_id, -100);
            schedule_reply(game, recipient_sid, requestor_id);
            return;
        }
        if self.emoji == EMOJI_CLOWN {
            game.update_relation(recipient_sid, self.requestor_small_id, -10);
            schedule_reply(game, recipient_sid, requestor_id);
            return;
        }
        if EMOJI_PEACEFUL.contains(&self.emoji) {
            if game.wire.game_config().difficulty == "Easy" {
                game.update_relation(recipient_sid, self.requestor_small_id, 15);
            }
            // TS always schedules a reply for peaceful emojis (love vs confused
            // pick is display-only / local PRNG).
            schedule_reply(game, recipient_sid, requestor_id);
        }
    }

    fn is_active(&self) -> bool {
        self.active
    }
}

fn schedule_reply(game: &mut Game, nation_sid: u16, human_id: Option<String>) {
    let Some(human_id) = human_id else {
        return;
    };
    game.add_execution(ExecEnum::Emoji(EmojiExecution::new(
        nation_sid,
        Some(human_id),
        REPLY_EMOJI_PLACEHOLDER,
    )));
}

fn record_emoji_send_now(game: &mut Game, sender_small_id: u16, recipient: Option<u16>) {
    let created_at = game.ticks();
    if let Some(sender) = game.player_by_small_id_mut(sender_small_id) {
        sender.outgoing_emoji_sends.push((recipient, created_at));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::PlayerInfo;

    fn add_player(game: &mut Game, id: &str, player_type: PlayerType) -> u16 {
        game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type,
            client_id: Some(id.into()),
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        })
    }

    #[test]
    fn middle_finger_makes_nation_hostile_toward_human() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let human = add_player(&mut game, "human", PlayerType::Human);
        let nation = add_player(&mut game, "nation", PlayerType::Nation);
        // Give both tiles so isAlive-equivalent paths stay happy.
        if let Some(p) = game.player_by_small_id_mut(human) {
            p.tiles_owned = 10;
            p.alive = true;
        }
        if let Some(p) = game.player_by_small_id_mut(nation) {
            p.tiles_owned = 10;
            p.alive = true;
        }

        let nation_id = game.player_by_small_id(nation).unwrap().id.clone();
        let mut exec = EmojiExecution::new(human, Some(nation_id), EMOJI_MIDDLE_FINGER);
        exec.init(&mut game, 0);
        // Human emoji intents are applied on turn N and tick on N+1 (TS EmojiExecution).
        game.execute_next_tick();
        let tick = game.ticks();
        exec.tick(&mut game, tick);
        assert_eq!(
            game.relation(nation, human),
            crate::game::Relation::Hostile
        );
    }
}
