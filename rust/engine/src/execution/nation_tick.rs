//! Nation post-spawn tick flow (TS `NationExecution.ts` + behavior RNG ordering).

use super::ai_attack::nation_maybe_attack;
use super::nation_alliance::{handle_alliance_extension_requests, handle_alliance_requests};
use super::nation_emoji::{NationEmojiState, maybe_send_casual_emoji};
use crate::core::schemas::unit_type;
use crate::game::Game;
use crate::prng::PseudoRandom;
use std::collections::HashSet;

const UNDER_ATTACK_THREAT_RATIO: f64 = 0.35;

#[derive(Debug, Default)]
pub struct NationStructureState {
    pub placements_count: u32,
    pub last_structure_tick: u32,
}

#[derive(Debug, Default)]
pub struct NationBehaviorState {
    pub structure: NationStructureState,
    pub embargo_malus_applied: HashSet<String>,
    pub emoji: NationEmojiState,
}

/// TS `NationWarshipBehavior.trackShipsAndRetaliate` - no-op until warships are ported.
fn track_ships_and_retaliate(_game: &Game, _random: &mut PseudoRandom, _small_id: u16) {}

fn update_relations_from_embargos(_game: &mut Game, _state: &mut NationBehaviorState, _small_id: u16) {}

fn consider_mirv(game: &Game, random: &mut PseudoRandom, small_id: u16) -> bool {
    if game.wire.is_unit_disabled(unit_type::MIRV) {
        return false;
    }
    if game.unit_count(small_id, unit_type::MISSILE_SILO) == 0 {
        return false;
    }
    let hesitation = match game.wire.game_config().difficulty.as_str() {
        "Easy" => 20,
        "Medium" => 12,
        "Hard" => 10,
        "Impossible" => 8,
        _ => 12,
    };
    random.chance(hesitation)
}

fn handle_structures(game: &Game, random: &mut PseudoRandom, small_id: u16, state: &mut NationStructureState) {
    if state.placements_count > 0 && !game.wire.is_unit_disabled(unit_type::DEFENSE_POST) {
        if try_build_defense_post(game, random, small_id) {
            return;
        }
        if defense_post_needed(game, small_id) {
            return;
        }
    }
    if is_on_structure_cooldown(game, state) {
        return;
    }
    let built = do_handle_structures(game, random, small_id);
    if built {
        state.last_structure_tick = game.ticks();
        state.placements_count += 1;
    }
}

fn try_build_defense_post(game: &Game, random: &mut PseudoRandom, small_id: u16) -> bool {
    let difficulty = game.wire.game_config().difficulty.as_str();
    if difficulty == "Easy" {
        return false;
    }
    if difficulty == "Medium" && !random.chance(2) {
        return false;
    }
    let incoming = game.incoming_land_troops(small_id);
    if incoming <= 0.0 {
        return false;
    }
    let Some(p) = game.player_by_small_id(small_id) else {
        return false;
    };
    if p.troops <= 0 {
        return false;
    }
    if incoming / (p.troops as f64) < UNDER_ATTACK_THREAT_RATIO {
        return false;
    }
    false
}

fn defense_post_needed(game: &Game, small_id: u16) -> bool {
    let difficulty = game.wire.game_config().difficulty.as_str();
    if difficulty == "Easy" {
        return false;
    }
    let incoming = game.incoming_land_troops(small_id);
    if incoming <= 0.0 {
        return false;
    }
    let Some(p) = game.player_by_small_id(small_id) else {
        return false;
    };
    if p.troops <= 0 {
        return false;
    }
    if incoming / (p.troops as f64) >= UNDER_ATTACK_THREAT_RATIO {
        return true;
    }
    false
}

fn is_on_structure_cooldown(game: &Game, state: &NationStructureState) -> bool {
    if state.placements_count == 0 {
        return false;
    }
    let rate = match game.wire.game_config().difficulty.as_str() {
        "Easy" => 80,
        "Medium" => 60,
        "Hard" => 45,
        "Impossible" => 30,
        _ => 60,
    };
    game.ticks().saturating_sub(state.last_structure_tick) < rate
}

fn do_handle_structures(game: &Game, random: &mut PseudoRandom, small_id: u16) -> bool {
    let _ = (game, random, small_id);
    false
}

fn maybe_spawn_warship(game: &Game, random: &mut PseudoRandom, small_id: u16) -> bool {
    if game.wire.is_unit_disabled(unit_type::WARSHIP) {
        return false;
    }
    if !random.chance(50) {
        return false;
    }
    let ports: Vec<_> = game
        .player_by_small_id(small_id)
        .map(|p| {
            p.units
                .iter()
                .filter(|u| u.unit_type == unit_type::PORT)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if ports.is_empty() || game.unit_count(small_id, unit_type::WARSHIP) > 0 {
        return false;
    }
    false
}

fn handle_embargoes_to_hostile_nations(_game: &mut Game, _small_id: u16) {}

fn counter_warship_infestation(_game: &Game, _random: &mut PseudoRandom, _small_id: u16) {}

fn maybe_send_nuke(game: &Game, random: &mut PseudoRandom, small_id: u16) {
    if game.wire.is_unit_disabled(unit_type::ATOM_BOMB)
        && game.wire.is_unit_disabled(unit_type::HYDROGEN_BOMB)
    {
        return;
    }
    if game.unit_count(small_id, unit_type::MISSILE_SILO) == 0 {
        return;
    }
    let _ = random.chance(100);
}

pub fn tick_nation_post_spawn(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    behavior: &mut NationBehaviorState,
    attack_rate: i32,
    attack_tick: i32,
    trigger_ratio: f64,
    reserve_ratio: f64,
    expand_ratio: f64,
    bot_attack_troops_sent: &mut f64,
) {
    let tick_i = game.ticks() as i32;
    let difficulty = game.wire.game_config().difficulty.clone();

    if game
        .player_by_small_id(small_id)
        .is_some_and(|p| p.alive && p.tiles_owned > 0)
        && difficulty != "Easy"
        && game.unit_count(small_id, unit_type::PORT) > 0
        && !game.wire.is_unit_disabled(unit_type::WARSHIP)
    {
        track_ships_and_retaliate(game, random, small_id);
    }

    if tick_i % attack_rate != attack_tick {
        let offset = tick_i % attack_rate;
        let one_third = (attack_tick + attack_rate / 3) % attack_rate;
        let two_thirds = (attack_tick + (attack_rate * 2) / 3) % attack_rate;
        if offset == one_third || offset == two_thirds {
            handle_structures(game, random, small_id, &mut behavior.structure);
        }
        return;
    }

    maybe_send_casual_emoji(game, random, small_id, &mut behavior.emoji);
    update_relations_from_embargos(game, behavior, small_id);
    handle_alliance_requests(game, random, small_id, &mut behavior.emoji);
    handle_alliance_extension_requests(game, random, small_id, &mut behavior.emoji);
    let _ = consider_mirv(game, random, small_id);
    handle_structures(game, random, small_id, &mut behavior.structure);
    let _ = maybe_spawn_warship(game, random, small_id);
    handle_embargoes_to_hostile_nations(game, small_id);
    nation_maybe_attack(
        game,
        random,
        small_id,
        trigger_ratio,
        reserve_ratio,
        expand_ratio,
        bot_attack_troops_sent,
        &difficulty,
        Some(&mut behavior.emoji),
    );
    counter_warship_infestation(game, random, small_id);
    maybe_send_nuke(game, random, small_id);
}
