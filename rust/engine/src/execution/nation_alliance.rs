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
        let (acceptable, enemy_id) = {
            let Some(enemy) = game.player_by_small_id(enemy_sid) else {
                continue;
            };
            (
                enemy.player_type != PlayerType::Bot || difficulty == "Easy",
                enemy.id.clone(),
            )
        };
        if !acceptable {
            continue;
        }
        if !game.can_send_alliance_request(small_id, enemy_sid) {
            continue;
        }
        if get_alliance_decision(game, random, small_id, enemy_sid, false, emoji) {
            game.add_execution(ExecEnum::AllianceRequest(
                AllianceRequestExecution::new(small_id, enemy_id),
            ));
        }
    }
}

fn get_alliance_decision(
    game: &mut Game,
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

    // TS `hasTooManyAlliances`: `this.game.players().filter(p => p.type() !== Bot)`
    // where `game.players()` is already alive-only via `isAlive()` (== tiles > 0).
    // The `Player.alive` flag can remain true after a wipe (0 tiles), which inflated
    // `total_players`, lowered the alliance-density threshold, and let nations accept
    // alliances TS rejects (asia150 Kazakhstan→Tajikistan @4401).
    let total_players = game
        .players_in_order()
        .iter()
        .filter(|p| p.tiles_owned > 0 && p.player_type != PlayerType::Bot)
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
                return bordering.len() <= bordering_friends.len() + 1;
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

/// TS `NationAllianceBehavior.maybeBetray` - no RNG; returns true when
/// betrayal would occur. TS calls `this.betray(otherPlayer)` (break the
/// alliance) at each `return true` site, as a side effect of the decision
/// itself, not left for the caller - so every branch below must break the
/// alliance before returning true. Without this, `AttackExecution::init`'s
/// `is_friendly` guard silently deactivates the resulting attack (the two
/// are still allied), making `nation_strategy_betray` report success while
/// no troops/tiles actually move.
pub fn maybe_betray(
    game: &mut Game,
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
            game.break_alliance_between(attacker_small_id, target_small_id);
            return true;
        }
    }

    if matches!(difficulty, "Easy" | "Medium")
        && !(difficulty == "Easy" && target.player_type == PlayerType::Human)
        && attacker.troops >= target.troops * 10
    {
        game.break_alliance_between(attacker_small_id, target_small_id);
        return true;
    }

    if difficulty != "Easy" && game.is_traitor(target_small_id) {
        if target.troops < (attacker.troops as f64 * 1.2) as i32 {
            game.break_alliance_between(attacker_small_id, target_small_id);
            return true;
        }
    }

    if difficulty != "Easy"
        && bordering_player_count == 1
        && target.troops * 3 < attacker.troops
    {
        game.break_alliance_between(attacker_small_id, target_small_id);
        return true;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{AllianceRequestStatus, PlayerInfo};

    fn add_player(game: &mut Game, id: &str, player_type: PlayerType) -> u16 {
        let sid = game.add_from_info(&PlayerInfo {
            name: id.into(),
            player_type,
            client_id: Some(id.into()),
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        });
        let p = game.player_by_small_id_mut(sid).unwrap();
        p.troops = 1_000;
        p.tiles_owned = 10;
        sid
    }

    /// Mirrors TS `setupAllianceRequest`'s defaults (relationDelta=2 - i.e.
    /// `Relation::Neutral` since it's non-negative and below `Friendly`'s
    /// threshold - equal troop counts so nobody looks like a threat). Request
    /// `created_at` is a tick comfortably past the spawn-phase cutoff (see the
    /// dedicated cutoff test below for that boundary case).
    fn setup(
        player_troops: i32,
        requestor_troops: i32,
        relation_delta: i32,
        is_traitor: bool,
    ) -> (Game, u16, u16) {
        let mut game = Game::default();
        game.end_spawn_phase();
        let player = add_player(&mut game, "player_id", PlayerType::Bot);
        let requestor = add_player(&mut game, "requestor_id", PlayerType::Human);
        game.player_by_small_id_mut(player).unwrap().troops = player_troops;
        game.player_by_small_id_mut(requestor).unwrap().troops = requestor_troops;
        game.update_relation(player, requestor, relation_delta);
        game.update_relation(requestor, player, relation_delta);
        if is_traitor {
            game.mark_traitor(requestor);
        }
        assert!(game.create_alliance_request(requestor, player, 1_000));
        (game, player, requestor)
    }

    fn request_status(game: &Game, requestor: u16, player: u16) -> AllianceRequestStatus {
        game.alliance_requests
            .iter()
            .find(|r| r.requestor_small_id == requestor && r.recipient_small_id == player)
            .map(|r| r.status)
            .expect("request should still be tracked")
    }

    // Ported from NationAllianceBehavior.test.ts "should reject alliance
    // created on first post-spawn tick".
    #[test]
    fn rejects_alliance_request_created_during_the_spawn_phase_cutoff() {
        let mut game = Game::default();
        game.end_spawn_phase();
        let cutoff = game.wire.num_spawn_phase_turns() + 1;
        let player = add_player(&mut game, "player_id", PlayerType::Bot);
        let requestor = add_player(&mut game, "requestor_id", PlayerType::Human);
        game.update_relation(player, requestor, 2);
        game.update_relation(requestor, player, 2);
        assert!(game.create_alliance_request(requestor, player, cutoff));

        let mut random = PseudoRandom::new(46);
        let mut emoji = NationEmojiState::default();
        handle_alliance_requests(&mut game, &mut random, player, &mut emoji);

        assert!(!game.is_allied_with(player, requestor));
        assert_eq!(
            request_status(&game, requestor, player),
            AllianceRequestStatus::Rejected
        );
    }

    // Ported from NationAllianceBehavior.test.ts "should accept alliance when
    // all conditions are met".
    #[test]
    fn accepts_alliance_request_when_all_conditions_are_met() {
        let (mut game, player, requestor) = setup(1_000, 1_000, 2, false);
        let mut random = PseudoRandom::new(46);
        let mut emoji = NationEmojiState::default();
        handle_alliance_requests(&mut game, &mut random, player, &mut emoji);

        assert!(game.is_allied_with(player, requestor));
    }

    // Ported from NationAllianceBehavior.test.ts "should reject alliance if
    // requestor is a traitor".
    #[test]
    fn rejects_alliance_request_from_a_traitor() {
        let (mut game, player, requestor) = setup(1_000, 1_000, 2, true);
        let mut random = PseudoRandom::new(46);
        let mut emoji = NationEmojiState::default();
        handle_alliance_requests(&mut game, &mut random, player, &mut emoji);

        assert!(!game.is_allied_with(player, requestor));
        assert_eq!(
            request_status(&game, requestor, player),
            AllianceRequestStatus::Rejected
        );
    }

    // Ported from NationAllianceBehavior.test.ts "should reject alliance if
    // relation is hostile".
    #[test]
    fn rejects_alliance_request_when_relation_is_hostile() {
        let (mut game, player, requestor) = setup(1_000, 1_000, -2, false);
        let mut random = PseudoRandom::new(46);
        let mut emoji = NationEmojiState::default();
        handle_alliance_requests(&mut game, &mut random, player, &mut emoji);

        assert!(!game.is_allied_with(player, requestor));
        assert_eq!(
            request_status(&game, requestor, player),
            AllianceRequestStatus::Rejected
        );
    }

    // Ported from NationAllianceBehavior.test.ts "should accept alliance if
    // requestor is much larger (> 3 times size of recipient)" - TS derives the
    // size difference from tile counts feeding into the troop formula; ported
    // here as a direct troop-count difference exceeding Medium's
    // `isAlliancePartnerThreat` 2.5x ratio, exercising the exact same decision
    // branch.
    #[test]
    fn accepts_alliance_request_from_a_much_larger_requestor() {
        let (mut game, player, requestor) = setup(1_000, 4_000, 2, false);
        let mut random = PseudoRandom::new(46);
        let mut emoji = NationEmojiState::default();
        handle_alliance_requests(&mut game, &mut random, player, &mut emoji);

        assert!(game.is_allied_with(player, requestor));
    }

    // Ported from NationAllianceBehavior.test.ts "should reject alliance if
    // player has too many alliances" - TS mocks `player.alliances()` directly;
    // native computes `alliance_count` from real alliance records, so this
    // creates 10 real (unrelated) alliances for `player` instead.
    #[test]
    fn rejects_alliance_request_when_recipient_already_has_many_alliances() {
        let (mut game, player, requestor) = setup(1_000, 1_000, 2, false);
        for i in 0..10 {
            let other = add_player(&mut game, &format!("filler{i}"), PlayerType::Human);
            assert!(game.create_alliance_request(player, other, 1_000));
            assert!(game.create_alliance_request(other, player, 1_000));
        }
        assert_eq!(game.alliance_count(player), 10);

        let mut random = PseudoRandom::new(46);
        let mut emoji = NationEmojiState::default();
        handle_alliance_requests(&mut game, &mut random, player, &mut emoji);

        assert!(!game.is_allied_with(player, requestor));
        assert_eq!(
            request_status(&game, requestor, player),
            AllianceRequestStatus::Rejected
        );
    }

    // Ported from NationAllianceBehavior.test.ts "should NOT request extension
    // if onlyOneAgreedToExtend is false" - native's
    // `alliance_extension_candidates` only surfaces an alliance when exactly
    // one side has requested an extension (`extension_requested_requestor ^
    // extension_requested_recipient`), so a freshly-formed alliance where
    // neither side has requested anything should yield zero candidates and
    // queue no `AllianceExtensionExecution`.
    #[test]
    fn does_not_request_extension_when_neither_side_has_agreed() {
        let (mut game, player, requestor) = setup(1_000, 1_000, 2, false);
        // Form a real alliance (the pending request from `setup` plus its
        // counter-request auto-accepts), rather than leaving only a dangling
        // pending request, so `alliance_extension_candidates` has an actual
        // alliance to (correctly) find zero extension candidates for.
        assert!(game.create_alliance_request(player, requestor, 1_001));
        assert!(game.is_allied_with(player, requestor));

        let expiration_before = game
            .player_alliances(player)
            .first()
            .map(|al| al.expires_at)
            .expect("alliance should exist");

        let mut random = PseudoRandom::new(46);
        let mut emoji = NationEmojiState::default();
        handle_alliance_extension_requests(&mut game, &mut random, player, &mut emoji);
        // If an `AllianceExtensionExecution` had (incorrectly) been queued, its
        // `init()` would run on this next tick and flip
        // `extension_requested_requestor`/`_recipient`.
        game.execute_next_tick();

        let alliance = game
            .player_alliances(player)
            .first()
            .cloned()
            .cloned()
            .expect("alliance should still exist");
        assert!(!alliance.extension_requested_requestor);
        assert!(!alliance.extension_requested_recipient);
        assert_eq!(alliance.expires_at, expiration_before);
    }
}
