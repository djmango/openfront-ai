//! Nation casual emoji behavior (TS `NationEmojiBehavior.ts` - RNG parity only).

use super::ai_attack::nearby_player_small_ids;
use crate::game::{Game, PlayerType};
use crate::prng::PseudoRandom;
use std::collections::HashMap;

const EMOJI_OVERWHELMED_LEN: i32 = 8;
const EMOJI_CONFUSED_LEN: i32 = 2;
const EMOJI_BORED_LEN: i32 = 1;
const EMOJI_CONGRATULATE_LEN: i32 = 1;
const EMOJI_BRAG_LEN: i32 = 3;
const EMOJI_LOVE_LEN: i32 = 3;
const EMOJI_CHARM_ALLIES_LEN: i32 = 3;
const EMOJI_CLOWN_LEN: i32 = 2;
const EMOJI_RAT_LEN: i32 = 1;
const EMOJI_GREET_LEN: i32 = 1;

#[derive(Debug, Default)]
pub struct NationEmojiState {
    last_emoji_sent: HashMap<u16, u32>,
    game_over: bool,
}

pub fn maybe_send_casual_emoji(
    game: &Game,
    random: &mut PseudoRandom,
    small_id: u16,
    state: &mut NationEmojiState,
) {
    if state.game_over {
        return;
    }

    check_overwhelmed_by_attacks(game, random, small_id, state);
    check_very_small_attack(game, random, small_id, state);
    congratulate_winner(game, random, small_id, state);
    brag(game, random, small_id, state);
    charm_allies(game, random, small_id, state);
    annoy_traitors(game, random, small_id, state);
    find_rat(game, random, small_id, state);
    greet_nearby_players(game, random, small_id, state);
}

fn check_overwhelmed_by_attacks(
    game: &Game,
    random: &mut PseudoRandom,
    small_id: u16,
    state: &mut NationEmojiState,
) {
    if !random.chance(16) {
        return;
    }

    let incoming = game.incoming_attacks(small_id, false);
    if incoming.is_empty() {
        return;
    }

    let incoming_troops: f64 = incoming.iter().map(|a| a.troops).sum();
    let our_troops = game
        .player_by_small_id(small_id)
        .map(|p| p.troops)
        .unwrap_or(0);
    if incoming_troops >= our_troops as f64 * 3.0 {
        send_emoji(random, game, small_id, state, None, EMOJI_OVERWHELMED_LEN);
    }
}

fn check_very_small_attack(
    game: &Game,
    random: &mut PseudoRandom,
    small_id: u16,
    state: &mut NationEmojiState,
) {
    if !random.chance(8) {
        return;
    }

    let incoming = game.incoming_attacks(small_id, false);
    if incoming.is_empty() {
        return;
    }

    let our_troops = game
        .player_by_small_id(small_id)
        .map(|p| p.troops)
        .unwrap_or(0);
    if our_troops <= 0 {
        return;
    }

    for attack in incoming {
        let Some(attacker) = game.player_by_small_id(attack.attacker_small_id) else {
            continue;
        };
        if attacker.player_type != PlayerType::Human {
            continue;
        }
        if attack.troops < our_troops as f64 * 0.1 {
            let emoji_len = if random.chance(2) {
                EMOJI_CONFUSED_LEN
            } else {
                EMOJI_BORED_LEN
            };
            maybe_send_emoji(
                random,
                game,
                small_id,
                state,
                Some(attack.attacker_small_id),
                emoji_len,
            );
        }
    }
}

fn congratulate_winner(
    game: &Game,
    random: &mut PseudoRandom,
    small_id: u16,
    state: &mut NationEmojiState,
) {
    let Some(winner) = game.winner.as_deref() else {
        return;
    };

    state.game_over = true;

    let is_team_game = game.wire.game_config().game_mode == "Team";
    if is_team_game {
        if game.is_on_same_team(small_id, winner) {
            return;
        }
        send_emoji(random, game, small_id, state, None, EMOJI_CONGRATULATE_LEN);
        return;
    }

    let Some(winner_player) = game.player_by_id(winner) else {
        return;
    };

    let largest_nation = game
        .players_in_order()
        .iter()
        .filter(|p| p.player_type == PlayerType::Nation)
        .max_by_key(|p| p.tiles_owned);
    let Some(largest) = largest_nation else {
        return;
    };
    if largest.small_id != small_id {
        return;
    }

    send_emoji(
        random,
        game,
        small_id,
        state,
        Some(winner_player.small_id),
        EMOJI_CONGRATULATE_LEN,
    );
}

fn brag(game: &Game, random: &mut PseudoRandom, small_id: u16, state: &mut NationEmojiState) {
    if !random.chance(300) {
        return;
    }

    let largest = game
        .players_in_order()
        .iter()
        .max_by_key(|p| p.tiles_owned);
    let Some(largest) = largest else {
        return;
    };
    if largest.small_id != small_id {
        return;
    }

    send_emoji(random, game, small_id, state, None, EMOJI_BRAG_LEN);
}

fn charm_allies(game: &Game, random: &mut PseudoRandom, small_id: u16, state: &mut NationEmojiState) {
    if !random.chance(250) {
        return;
    }

    let human_allies: Vec<u16> = game
        .allied_small_ids(small_id)
        .into_iter()
        .filter(|&sid| {
            game.player_by_small_id(sid)
                .is_some_and(|p| p.player_type == PlayerType::Human)
        })
        .collect();
    if human_allies.is_empty() {
        return;
    }

    let Some(ally) = random.rand_element(&human_allies) else {
        return;
    };
    let emoji_len = if random.chance(3) {
        EMOJI_LOVE_LEN
    } else {
        EMOJI_CHARM_ALLIES_LEN
    };
    send_emoji(random, game, small_id, state, Some(ally), emoji_len);
}

fn annoy_traitors(game: &Game, random: &mut PseudoRandom, small_id: u16, state: &mut NationEmojiState) {
    if !random.chance(40) {
        return;
    }

    let traitors: Vec<u16> = game
        .players_in_order()
        .iter()
        .filter(|p| {
            p.player_type == PlayerType::Human
                && !game.is_friendly(small_id, p.small_id)
                && game.is_traitor(p.small_id)
        })
        .map(|p| p.small_id)
        .collect();
    if traitors.is_empty() {
        return;
    }

    let Some(traitor) = random.rand_element(&traitors) else {
        return;
    };
    send_emoji(random, game, small_id, state, Some(traitor), EMOJI_CLOWN_LEN);
}

fn find_rat(game: &Game, random: &mut PseudoRandom, small_id: u16, state: &mut NationEmojiState) {
    if game.ticks() < 6000 {
        return;
    }
    if !random.chance(10000) {
        return;
    }

    let threshold = game.num_land_tiles() as f64 * 0.01;
    let small_players: Vec<u16> = game
        .players_in_order()
        .iter()
        .filter(|p| {
            p.player_type == PlayerType::Human
                && (p.tiles_owned as f64) < threshold
                && p.tiles_owned > 0
        })
        .map(|p| p.small_id)
        .collect();
    if small_players.is_empty() {
        return;
    }

    let Some(target) = random.rand_element(&small_players) else {
        return;
    };
    send_emoji(random, game, small_id, state, Some(target), EMOJI_RAT_LEN);
}

fn greet_nearby_players(
    game: &Game,
    random: &mut PseudoRandom,
    small_id: u16,
    state: &mut NationEmojiState,
) {
    if game.ticks() > 600 {
        return;
    }
    if !random.chance(250) {
        return;
    }

    let nearby_humans: Vec<u16> = nearby_player_small_ids(game, small_id)
        .into_iter()
        .filter(|&sid| {
            game.player_by_small_id(sid)
                .is_some_and(|p| p.player_type == PlayerType::Human)
        })
        .collect();
    if nearby_humans.is_empty() {
        return;
    }

    let Some(neighbor) = random.rand_element(&nearby_humans) else {
        return;
    };
    send_emoji(random, game, small_id, state, Some(neighbor), EMOJI_GREET_LEN);
}

/// TS `maybeSendAttackEmoji` - RNG only; execution is hash-neutral.
pub fn maybe_send_attack_emoji(
    game: &Game,
    random: &mut PseudoRandom,
    small_id: u16,
    state: &mut NationEmojiState,
    target_small_id: u16,
) {
    if !should_send_emoji(game, small_id, state, Some(target_small_id), true) {
        return;
    }

    if game.relation(small_id, target_small_id) >= crate::game::Relation::Neutral {
        if !random.chance(2) {
            return;
        }
        send_emoji(
            random,
            game,
            small_id,
            state,
            Some(target_small_id),
            1, // EMOJI_AGGRESSIVE_ATTACK
        );
        return;
    }

    if !random.chance(4) {
        return;
    }
    send_emoji(
        random,
        game,
        small_id,
        state,
        Some(target_small_id),
        1, // EMOJI_ATTACK
    );
}

/// TS `NationEmojiBehavior.maybeSendEmoji(otherPlayer, emojisList)` - exposed
/// crate-wide for its other two TS callers now ported natively:
/// `NationNukeBehavior.sendNuke`'s `maybeSendEmoji(targetPlayer, EMOJI_NUKE)`
/// (see `nuke_ai.rs::EMOJI_NUKE_LEN`) and
/// `NationWarshipBehavior.maybeRetaliateWithWarship`'s equivalent call (see
/// `warship_ai.rs`).
pub(crate) fn maybe_send_emoji(
    random: &mut PseudoRandom,
    game: &Game,
    small_id: u16,
    state: &mut NationEmojiState,
    recipient: Option<u16>,
    emoji_list_len: i32,
) {
    if !should_send_emoji(game, small_id, state, recipient, true) {
        return;
    }
    send_emoji(random, game, small_id, state, recipient, emoji_list_len);
}

/// TS `sendEmoji` - consumes `randElement` when allowed; execution is hash-neutral.
pub fn send_emoji(
    random: &mut PseudoRandom,
    game: &Game,
    small_id: u16,
    state: &mut NationEmojiState,
    recipient: Option<u16>,
    emoji_list_len: i32,
) {
    if !should_send_emoji(game, small_id, state, recipient, false) {
        return;
    }
    if !game.can_send_emoji(small_id, recipient) {
        return;
    }
    let _ = random.next_int(0, emoji_list_len);
}

fn should_send_emoji(
    game: &Game,
    small_id: u16,
    state: &mut NationEmojiState,
    recipient: Option<u16>,
    limit_emojis_by_time: bool,
) -> bool {
    let Some(recipient_sid) = recipient else {
        return true;
    };
    let Some(sender) = game.player_by_small_id(small_id) else {
        return false;
    };
    if sender.player_type == PlayerType::Bot {
        return false;
    }
    let Some(recipient_p) = game.player_by_small_id(recipient_sid) else {
        return false;
    };
    if recipient_p.player_type != PlayerType::Human {
        return false;
    }

    if limit_emojis_by_time {
        let last_sent = state
            .last_emoji_sent
            .get(&recipient_sid)
            .copied()
            .map(|v| v as i32)
            .unwrap_or(-300);
        if game.ticks() as i32 - last_sent <= 300 {
            return false;
        }
        state.last_emoji_sent.insert(recipient_sid, game.ticks());
    }

    true
}
