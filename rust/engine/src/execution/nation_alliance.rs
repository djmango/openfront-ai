//! Nation alliance behavior (TS `NationAllianceBehavior.ts` - RNG parity only).

use super::nation_emoji::{NationEmojiState, send_emoji};
use super::{AllianceExtensionExecution, AllianceRequestExecution, ExecEnum};
use crate::game::{Game, PlayerType, Relation};
use crate::prng::PseudoRandom;

const EMOJI_CONFUSED_LEN: i32 = 2;
const EMOJI_SCARED_OF_THREAT_LEN: i32 = 2;
const EMOJI_LOVE_LEN: i32 = 3;
const EMOJI_HANDSHAKE_LEN: i32 = 1;

pub fn handle_alliance_requests(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    emoji: &mut NationEmojiState,
) {
    if game.wire.disable_alliances() {
        return;
    }

    let spawn_cutoff = game.wire.num_spawn_phase_turns() + 1;
    let tick = game.ticks();
    let requests = game.incoming_alliance_requests(small_id);
    for req in requests {
        if req.created_at <= spawn_cutoff {
            game.reject_alliance_request(req.requestor_small_id, small_id);
            continue;
        }
        if get_alliance_decision(
            game,
            random,
            small_id,
            req.requestor_small_id,
            true,
            emoji,
        ) {
            game.accept_alliance_request(req.requestor_small_id, small_id, tick);
        } else {
            game.reject_alliance_request(req.requestor_small_id, small_id);
        }
    }
}

pub fn handle_alliance_extension_requests(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    emoji: &mut NationEmojiState,
) {
    if game.wire.disable_alliances() {
        return;
    }

    let extensions = game.alliance_extension_candidates(small_id);
    for ext in extensions {
        if get_alliance_decision(
            game,
            random,
            small_id,
            ext.other_small_id,
            true,
            emoji,
        ) {
            let Some(other) = game.player_by_small_id(ext.other_small_id) else {
                continue;
            };
            game.add_execution(ExecEnum::AllianceExtension(
                AllianceExtensionExecution::new(small_id, other.id.clone()),
            ));
        }
    }
}

pub fn maybe_send_alliance_requests(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    bordering_enemies: &[u16],
    emoji: &mut NationEmojiState,
) {
    if game.wire.disable_alliances() {
        return;
    }

    let difficulty = game.wire.game_config().difficulty.clone();
    for &enemy_sid in bordering_enemies {
        if !random.chance(30) {
            continue;
        }
        let Some(enemy) = game.player_by_small_id(enemy_sid) else {
            continue;
        };
        let acceptable =
            enemy.player_type != PlayerType::Bot || difficulty == "Easy";
        if !acceptable {
            continue;
        }
        if !game.can_send_alliance_request(small_id, enemy_sid) {
            continue;
        }
        if get_alliance_decision(game, random, small_id, enemy_sid, false, emoji) {
            game.add_execution(ExecEnum::AllianceRequest(
                AllianceRequestExecution::new(small_id, enemy.id.clone()),
            ));
        }
    }
}

fn get_alliance_decision(
    game: &Game,
    random: &mut PseudoRandom,
    small_id: u16,
    other_small_id: u16,
    is_response: bool,
    emoji: &mut NationEmojiState,
) -> bool {
    if is_confused(random, game.wire.game_config().difficulty.as_str()) {
        return random.chance(2);
    }

    if game.is_traitor(other_small_id) && random.next_int(0, 100) >= 10 {
        if is_response && random.chance(3) {
            send_emoji(
                random,
                game,
                small_id,
                emoji,
                Some(other_small_id),
                EMOJI_CONFUSED_LEN,
            );
        }
        return false;
    }

    if has_too_many_alliances(game, other_small_id) {
        return false;
    }

    if is_alliance_partner_threat(game, small_id, other_small_id) {
        if !is_response && random.chance(6) {
            send_emoji(
                random,
                game,
                small_id,
                emoji,
                Some(other_small_id),
                EMOJI_SCARED_OF_THREAT_LEN,
            );
        }
        if is_response && random.chance(6) {
            send_emoji(
                random,
                game,
                small_id,
                emoji,
                Some(other_small_id),
                EMOJI_LOVE_LEN,
            );
        }
        return true;
    }

    if should_reject_in_team_game(random, game) {
        return false;
    }

    if game.relation(small_id, other_small_id) < Relation::Neutral {
        if is_response && random.chance(3) {
            send_emoji(
                random,
                game,
                small_id,
                emoji,
                Some(other_small_id),
                EMOJI_CONFUSED_LEN,
            );
        }
        return false;
    }

    if is_alliance_partner_friendly(game, random, small_id, other_small_id) {
        if random.chance(3) {
            send_emoji(
                random,
                game,
                small_id,
                emoji,
                Some(other_small_id),
                EMOJI_HANDSHAKE_LEN,
            );
        }
        return true;
    }

    if check_already_enough_alliances(game, random, small_id, other_small_id) {
        return false;
    }

    if is_earlygame(game, random) {
        return true;
    }

    is_alliance_partner_similarly_strong(game, random, small_id, other_small_id)
}

fn is_confused(random: &mut PseudoRandom, difficulty: &str) -> bool {
    match difficulty {
        "Easy" => random.chance(10),
        "Medium" => random.chance(20),
        "Hard" => random.chance(40),
        "Impossible" => false,
        _ => random.chance(20),
    }
}

fn is_earlygame(game: &Game, random: &mut PseudoRandom) -> bool {
    let spawn_ticks = game.wire.num_spawn_phase_turns();
    let difficulty = game.wire.game_config().difficulty.as_str();
    match difficulty {
        "Easy" => {
            game.ticks() < 3000 + spawn_ticks && random.next_int(0, 100) >= 10
        }
        "Medium" => {
            game.ticks() < 1800 + spawn_ticks && random.next_int(0, 100) >= 30
        }
        "Hard" => {
            game.ticks() < 1800 + spawn_ticks && random.next_int(0, 100) >= 50
        }
        "Impossible" => {
            game.ticks() < 600 + spawn_ticks && random.next_int(0, 100) >= 70
        }
        _ => false,
    }
}

fn is_alliance_partner_threat(game: &Game, small_id: u16, other_small_id: u16) -> bool {
    let difficulty = game.wire.game_config().difficulty.as_str();
    let Some(player) = game.player_by_small_id(small_id) else {
        return false;
    };
    let Some(other) = game.player_by_small_id(other_small_id) else {
        return false;
    };

    match difficulty {
        "Easy" => false,
        "Medium" => other.troops as f64 > player.troops as f64 * 2.5,
        "Hard" => {
            other.troops > player.troops
                && game.max_troops_for(other.small_id)
                    > game.max_troops_for(player.small_id) * 2.0
        }
        "Impossible" => {
            let other_has_more_troops = other.troops as f64 > player.troops as f64 * 1.5;
            let other_has_more_max_troops = other.troops > player.troops
                && game.max_troops_for(other.small_id)
                    > game.max_troops_for(player.small_id) * 1.5;
            let other_has_more_tiles = other.troops > player.troops
                && other.tiles_owned > (player.tiles_owned as f64 * 1.5) as i32;
            other_has_more_troops || other_has_more_max_troops || other_has_more_tiles
        }
        _ => false,
    }
}

fn should_reject_in_team_game(random: &mut PseudoRandom, game: &Game) -> bool {
    if game.wire.game_config().game_mode != "Team" {
        return false;
    }
    match game.wire.game_config().difficulty.as_str() {
        "Easy" => random.next_int(0, 100) < 25,
        "Medium" => random.next_int(0, 100) < 50,
        "Hard" => random.next_int(0, 100) < 75,
        "Impossible" => true,
        _ => false,
    }
}

fn has_too_many_alliances(game: &Game, other_small_id: u16) -> bool {
    let difficulty = game.wire.game_config().difficulty.as_str();
    if difficulty != "Hard" && difficulty != "Impossible" {
        return false;
    }

    let total_players = game
        .players_in_order()
        .iter()
        .filter(|p| p.player_type != PlayerType::Bot)
        .count();
    let other_alliances = game.alliance_count(other_small_id);

    if difficulty == "Hard" {
        other_alliances as f64 >= total_players as f64 * 0.5
    } else {
        other_alliances as f64 >= total_players as f64 * 0.25
    }
}

fn check_already_enough_alliances(
    game: &Game,
    random: &mut PseudoRandom,
    small_id: u16,
    other_small_id: u16,
) -> bool {
    let difficulty = game.wire.game_config().difficulty.as_str();
    match difficulty {
        "Easy" => false,
        "Medium" => game.alliance_count(small_id) >= random.next_int(4, 6) as usize,
        "Hard" | "Impossible" => {
            let bordering: Vec<u16> = super::ai_attack::nearby_player_small_ids(game, small_id)
                .into_iter()
                .filter(|&sid| {
                    game.player_by_small_id(sid)
                        .is_some_and(|p| p.player_type != PlayerType::Bot)
                })
                .collect();
            let bordering_friends: Vec<u16> = bordering
                .iter()
                .copied()
                .filter(|&sid| game.is_friendly(small_id, sid))
                .collect();
            if bordering.len() >= 2 && bordering.contains(&other_small_id) {
                if bordering.len() <= bordering_friends.len() + 1 {
                    return true;
                }
            }
            if difficulty == "Hard" {
                game.alliance_count(small_id) >= random.next_int(3, 5) as usize
            } else {
                game.alliance_count(small_id) >= random.next_int(2, 4) as usize
            }
        }
        _ => false,
    }
}

fn is_alliance_partner_friendly(
    game: &Game,
    random: &mut PseudoRandom,
    small_id: u16,
    other_small_id: u16,
) -> bool {
    let difficulty = game.wire.game_config().difficulty.as_str();
    match difficulty {
        "Easy" | "Medium" => game.relation(small_id, other_small_id) == Relation::Friendly,
        "Hard" => {
            game.relation(small_id, other_small_id) == Relation::Friendly
                && random.next_int(0, 100) >= 17
        }
        "Impossible" => {
            game.relation(small_id, other_small_id) == Relation::Friendly
                && random.next_int(0, 100) >= 33
        }
        _ => false,
    }
}

fn is_alliance_partner_similarly_strong(
    game: &Game,
    random: &mut PseudoRandom,
    small_id: u16,
    other_small_id: u16,
) -> bool {
    let difficulty = game.wire.game_config().difficulty.as_str();
    let (troop_lo, troop_hi, tile_lo, tile_hi) = match difficulty {
        "Easy" => (60, 70, 70, 80),
        "Medium" => (70, 80, 80, 90),
        "Hard" => (75, 85, 85, 95),
        "Impossible" => (80, 90, 90, 100),
        _ => (70, 80, 80, 90),
    };

    let Some(player) = game.player_by_small_id(small_id) else {
        return false;
    };
    let Some(other) = game.player_by_small_id(other_small_id) else {
        return false;
    };

    let player_outgoing = game.outgoing_attack_troops(small_id);
    let other_outgoing = game.outgoing_attack_troops(other_small_id);
    let player_total = player.troops as f64 + player_outgoing;
    let other_total = other.troops as f64 + other_outgoing;

    let troop_threshold =
        player_total * (random.next_int(troop_lo, troop_hi) as f64 / 100.0);
    let tile_threshold =
        player.tiles_owned as f64 * (random.next_int(tile_lo, tile_hi) as f64 / 100.0);

    let has_comparable_troops = other_total > troop_threshold;
    let has_comparable_tiles = other.tiles_owned as f64 > tile_threshold
        && other_total > player_total * 0.5;
    has_comparable_troops || has_comparable_tiles
}

/// TS `NationAllianceBehavior.maybeBetray` - no RNG; returns true when betrayal would occur.
pub fn maybe_betray(
    game: &Game,
    _random: &mut PseudoRandom,
    attacker_small_id: u16,
    target_small_id: u16,
    bordering_player_count: usize,
) -> bool {
    if !game.is_allied_with(attacker_small_id, target_small_id) {
        return false;
    }
    let difficulty = game.wire.game_config().difficulty.as_str();
    let Some(attacker) = game.player_by_small_id(attacker_small_id) else {
        return false;
    };
    let Some(target) = game.player_by_small_id(target_small_id) else {
        return false;
    };

    // TS: "Betray very weak players (for example MIRVed ones)" - checked
    // first, before the Easy/Medium troop-ratio betrayal below.
    if !matches!(difficulty, "Easy" | "Medium") {
        let target_max_troops = game.max_troops_for(target_small_id);
        let target_outgoing = game.outgoing_attack_troops(target_small_id);
        if (target.troops as f64 + target_outgoing) < target_max_troops * 0.2
            && target.troops < attacker.troops
        {
            return true;
        }
    }

    if matches!(difficulty, "Easy" | "Medium")
        && !(difficulty == "Easy" && target.player_type == PlayerType::Human)
        && attacker.troops >= target.troops * 10
    {
        return true;
    }

    if difficulty != "Easy" && game.is_traitor(target_small_id) {
        if target.troops < (attacker.troops as f64 * 1.2) as i32 {
            return true;
        }
    }

    if difficulty != "Easy"
        && bordering_player_count == 1
        && target.troops * 3 < attacker.troops
    {
        return true;
    }

    false
}
