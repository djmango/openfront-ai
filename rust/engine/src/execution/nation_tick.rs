//! Nation post-spawn tick flow (TS `NationExecution.ts` + behavior RNG ordering).

use super::ai_attack::nation_maybe_attack;
use super::nation_alliance::{handle_alliance_extension_requests, handle_alliance_requests};
use super::nation_emoji::{NationEmojiState, maybe_send_casual_emoji};
use super::nuke_ai::NationNukeState;
use super::warship_ai::NationWarshipState;
use crate::core::schemas::unit_type;
use crate::game::{Game, PlayerType, Relation};
use crate::prng::PseudoRandom;
use std::collections::HashSet;

const UNDER_ATTACK_THREAT_RATIO: f64 = 0.35;
const HIGH_STARTING_GOLD_THRESHOLD: i64 = 3_000_000;
/// TS `HIGH_GOLD_STRUCTURE_COOLDOWN_TICKS`  -  gap before placing structure N (high-gold nations only).
const HIGH_GOLD_STRUCTURE_COOLDOWN_TICKS: [u32; 5] = [0, 0, 250, 150, 100];
const TEAM_POST_SAVE_UP_PHASE_TICKS: u32 = 150;

#[derive(Debug, Default)]
pub struct NationStructureState {
    pub placements_count: u32,
    pub last_structure_tick: Option<u32>,
    pub post_save_up_start_tick: Option<u32>,
}

#[derive(Debug, Default)]
pub struct NationBehaviorState {
    pub structure: NationStructureState,
    pub embargo_malus_applied: HashSet<String>,
    pub emoji: NationEmojiState,
    /// TS `NationNukeBehavior.isHydroNation`  -  `random.chance(3)` at behavior init.
    pub is_hydro_nation: bool,
    /// TS `NationNukeBehavior`'s remaining per-instance state (see `nuke_ai.rs`).
    pub nuke: NationNukeState,
    /// TS `NationWarshipBehavior`'s tracked-ship instance state (`warship_ai.rs`).
    pub warship: NationWarshipState,
}

/// TS `NationExecution.initializeBehaviors` RNG + field-init order:
/// `NationNukeBehavior`'s `atomBombPerceivedCost`/`hydrogenBombPerceivedCost`
/// initializers (`this.cost(...)`, no RNG) followed by `isHydroNation`
/// (`random.chance(3)`, the only RNG draw in that constructor).
pub fn initialize_nation_behaviors(
    game: &Game,
    random: &mut PseudoRandom,
    small_id: u16,
    state: &mut NationBehaviorState,
) {
    state.nuke.atom_bomb_perceived_cost = game.structure_cost(small_id, unit_type::ATOM_BOMB);
    state.nuke.hydrogen_bomb_perceived_cost =
        game.structure_cost(small_id, unit_type::HYDROGEN_BOMB);
    state.is_hydro_nation = random.chance(3);
}

/// TS `NationExecution.updateRelationsFromEmbargos` - applies (and later reverts) a relation
/// malus for every other player that currently has an active embargo against this nation.
/// `state.embargo_malus_applied` mirrors the TS `embargoMalusApplied: Set<PlayerID>` instance
/// field, tracking which pairs already have the malus applied so it's neither double-applied
/// nor double-reverted.
fn update_relations_from_embargos(game: &mut Game, state: &mut NationBehaviorState, small_id: u16) {
    const EMBARGO_MALUS: i32 = -20;
    let others: Vec<(u16, String)> = game
        .all_players()
        .iter()
        .filter(|p| p.small_id != small_id)
        .map(|p| (p.small_id, p.id.clone()))
        .collect();
    for (other_small_id, other_id) in others {
        let has_embargo = game.has_embargo_against(other_small_id, small_id);
        if has_embargo && !state.embargo_malus_applied.contains(&other_id) {
            game.update_relation(small_id, other_small_id, EMBARGO_MALUS);
            state.embargo_malus_applied.insert(other_id);
        } else if !has_embargo && state.embargo_malus_applied.contains(&other_id) {
            game.update_relation(small_id, other_small_id, -EMBARGO_MALUS);
            state.embargo_malus_applied.remove(&other_id);
        }
    }
}

fn handle_structures(game: &mut Game, random: &mut PseudoRandom, small_id: u16, state: &mut NationStructureState) {
    if state.placements_count > 0 && !game.wire.is_unit_disabled(unit_type::DEFENSE_POST) {
        if try_build_defense_post(game, random, small_id) {
            return;
        }
        if defense_post_needed(game, small_id) {
            return;
        }
    }
    if is_on_structure_cooldown(game, small_id, state) {
        return;
    }
    if is_in_post_save_up_blocked_phase(game, small_id, state) {
        return;
    }
    let built = do_handle_structures(game, random, small_id, state.placements_count);
    if built {
        state.last_structure_tick = Some(game.ticks());
        state.placements_count += 1;
    }
}

fn has_high_starting_gold(game: &Game) -> bool {
    // Only ever called for `Nation` players (nation AI), which always get the configured
    // starting gold (unlike raw `Bot` players, who always get 0).
    game.wire.starting_gold(crate::game::PlayerType::Nation) >= HIGH_STARTING_GOLD_THRESHOLD
}

fn is_on_structure_cooldown(game: &Game, _small_id: u16, state: &NationStructureState) -> bool {
    let Some(last) = state.last_structure_tick else {
        return false;
    };
    if !has_high_starting_gold(game) {
        return false;
    }
    let idx = state.placements_count as usize;
    let required_gap = HIGH_GOLD_STRUCTURE_COOLDOWN_TICKS
        .get(idx)
        .copied()
        .unwrap_or(0);
    if required_gap == 0 {
        return false;
    }
    game.ticks().saturating_sub(last) < required_gap
}

fn is_in_post_save_up_blocked_phase(
    game: &Game,
    small_id: u16,
    state: &mut NationStructureState,
) -> bool {
    if game.wire.is_unit_disabled(unit_type::MISSILE_SILO) {
        return false;
    }
    let save_up = super::nation_structures::save_up_target(game);
    if save_up == 0 {
        return false;
    }
    let gold = game.player_by_small_id(small_id).map(|p| p.gold).unwrap_or(0);
    if state.post_save_up_start_tick.is_none() {
        if gold < save_up {
            return false;
        }
        state.post_save_up_start_tick = Some(game.ticks());
    }
    let elapsed = game
        .ticks()
        .saturating_sub(state.post_save_up_start_tick.unwrap_or(0));
    elapsed % (TEAM_POST_SAVE_UP_PHASE_TICKS * 2) >= TEAM_POST_SAVE_UP_PHASE_TICKS
}

fn try_build_defense_post(game: &mut Game, random: &mut PseudoRandom, small_id: u16) -> bool {
    super::nation_structures::try_build_defense_post(game, random, small_id)
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

fn do_handle_structures(game: &mut Game, random: &mut PseudoRandom, small_id: u16, placements_count: u32) -> bool {
    super::nation_structures::do_handle_structures(game, random, small_id, placements_count)
}

/// TS `NationExecution.handleEmbargoesToHostileNations`.
fn handle_embargoes_to_hostile_nations(game: &mut Game, small_id: u16) {
    let tick = game.ticks();
    let difficulty = game.wire.game_config().difficulty.clone();
    let is_higher_difficulty = difficulty == "Hard" || difficulty == "Impossible";
    let team_game = game.wire.game_config().game_mode == "Team";

    let others: Vec<(u16, bool)> = game
        .all_players()
        .iter()
        .filter(|p| p.small_id != small_id)
        .map(|p| (p.small_id, p.player_type == PlayerType::Bot))
        .collect();

    for (other_small_id, other_is_bot) in others {
        // In team games on higher difficulties, refuse to trade with anyone not on this
        // nation's team (mirrors the "stop trading with all" button).
        if team_game
            && is_higher_difficulty
            && !other_is_bot
            && !game.players_on_same_team(small_id, other_small_id)
        {
            if !game.has_embargo_against(small_id, other_small_id) {
                game.add_embargo(small_id, other_small_id, false, tick);
            }
            continue;
        }

        let relation = game.relation(small_id, other_small_id);
        let has_embargo = game.has_embargo_against(small_id, other_small_id);
        // When player is hostile starts embargo. Do not stop until neutral again.
        if relation <= Relation::Hostile
            && !has_embargo
            && !game.players_on_same_team(small_id, other_small_id)
        {
            game.add_embargo(small_id, other_small_id, false, tick);
        } else if relation >= Relation::Neutral
            && has_embargo
            && difficulty != "Hard"
            && difficulty != "Impossible"
        {
            game.stop_embargo(small_id, other_small_id);
        } else if relation >= Relation::Friendly && has_embargo && difficulty != "Impossible" {
            game.stop_embargo(small_id, other_small_id);
        }
    }
}

/// TS `NationNukeBehavior.maybeSendNuke` - see `nuke_ai.rs` for the full port.
/// `maybeDestroyEnemySam` (Impossible-only SAM-overwhelm salvo planning) is
/// the one documented sub-piece deferred there; everything else (target
/// selection, MIRV-savings cost simulation, SAM-trajectory/impassable-terrain
/// avoidance, tile scoring, recent-nuke cooldown) is ported for real.
fn maybe_send_nuke(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    is_hydro_nation: bool,
    nuke_state: &mut NationNukeState,
    emoji_state: &mut NationEmojiState,
) {
    super::nuke_ai::maybe_send_nuke(game, random, small_id, is_hydro_nation, nuke_state, emoji_state);
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
        super::warship_ai::track_ships_and_retaliate(
            game,
            random,
            small_id,
            &mut behavior.warship,
            &mut behavior.emoji,
        );
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
    let _ = super::mirv_ai::consider_mirv(game, random, small_id, &mut behavior.emoji);
    handle_structures(game, random, small_id, &mut behavior.structure);
    let _ = super::warship_ai::maybe_spawn_warship(game, random, small_id);
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
    super::warship_ai::counter_warship_infestation(game, random, small_id, &mut behavior.emoji);
    maybe_send_nuke(
        game,
        random,
        small_id,
        behavior.is_hydro_nation,
        &mut behavior.nuke,
        &mut behavior.emoji,
    );
}
