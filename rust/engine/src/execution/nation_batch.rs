//! Batched nation AI - one tick driver for all nations (avoids 5× `dyn Execution` dispatch).

use crate::execution::{ExecEnum, SpawnExecution};
use super::NationRuntime;
use crate::game::Game;
use crate::map::{TerrainType, TileRef};
use crate::prng::PseudoRandom;
use crate::util::simple_hash;

struct NationState {
    game_id: String,
    nation: NationRuntime,
    random: PseudoRandom,
    active: bool,
    spawn_exec_added: bool,
    behaviors_initialized: bool,
    behavior: super::nation_tick::NationBehaviorState,
    attack_rate: i32,
    attack_tick: i32,
    trigger_ratio: f64,
    reserve_ratio: f64,
    expand_ratio: f64,
    bot_attack_troops_sent: f64,
}

pub struct NationBatch {
    nations: Vec<NationState>,
}

impl NationBatch {
    pub fn new() -> Self {
        Self {
            nations: Vec::new(),
        }
    }

    pub fn register(&mut self, game_id: String, nation: NationRuntime) {
        let seed = simple_hash(&nation.player_info.id).wrapping_add(simple_hash(&game_id));
        let mut random = PseudoRandom::new(seed);
        let trigger_ratio = random.next_int(50, 60) as f64 / 100.0;
        let reserve_ratio = random.next_int(30, 40) as f64 / 100.0;
        let expand_ratio = random.next_int(10, 20) as f64 / 100.0;
        self.nations.push(NationState {
            game_id,
            nation,
            random,
            active: true,
            spawn_exec_added: false,
            behaviors_initialized: false,
            behavior: super::nation_tick::NationBehaviorState::default(),
            attack_rate: 0,
            attack_tick: 0,
            trigger_ratio,
            reserve_ratio,
            expand_ratio,
            bot_attack_troops_sent: 0.0,
        });
    }

    pub fn init_all(&mut self, difficulty: &str) {
        for n in &mut self.nations {
            n.attack_rate = attack_rate_for_difficulty(&mut n.random, difficulty);
            n.attack_tick = n.random.next_int(0, n.attack_rate);
        }
    }

    pub fn tick(&mut self, game: &mut Game, tick: u32) {
        let tick_i = tick as i32;
        for n in &mut self.nations {
            if !n.active {
                continue;
            }
            let player_id = &n.nation.player_info.id;
            let Some(small_id) = game.player_by_id(player_id).map(|p| p.small_id) else {
                continue;
            };

            if game.in_spawn_phase() {
                if game.has_spawned(small_id) {
                    if tick_i % n.attack_rate != n.attack_tick {
                        continue;
                    }
                    if n.spawn_exec_added {
                        continue;
                    }
                } else if n.spawn_exec_added {
                    continue;
                }

                if n.nation.spawn_cell.is_none() {
                    game.add_execution(ExecEnum::Spawn(SpawnExecution::new(
                        n.game_id.clone(),
                        n.nation.player_info.clone(),
                        None,
                    )));
                    n.spawn_exec_added = true;
                    continue;
                }

                let Some(tile) = random_spawn_land(n, game) else {
                    continue;
                };

                game.add_execution(ExecEnum::Spawn(SpawnExecution::new(
                    n.game_id.clone(),
                    n.nation.player_info.clone(),
                    Some(tile),
                )));
                n.spawn_exec_added = true;
                continue;
            }

            if n.spawn_exec_added && !game.has_spawned(small_id) {
                continue;
            }

            if game
                .player_by_id(player_id)
                .is_some_and(|p| !p.alive || p.tiles_owned == 0)
            {
                n.active = false;
                continue;
            }

            if !n.behaviors_initialized {
                n.behaviors_initialized = true;
                let troops = game
                    .player_by_small_id(small_id)
                    .map(|p| p.troops as f64 / 2.0)
                    .unwrap_or(0.0);
                if troops >= 1.0 {
                    game.add_land_attack(small_id, None, Some(troops));
                }
                continue;
            }

            super::nation_tick::tick_nation_post_spawn(
                game,
                &mut n.random,
                small_id,
                &mut n.behavior,
                n.attack_rate,
                n.attack_tick,
                n.trigger_ratio,
                n.reserve_ratio,
                n.expand_ratio,
                &mut n.bot_attack_troops_sent,
            );
        }
    }
}

fn attack_rate_for_difficulty(random: &mut PseudoRandom, difficulty: &str) -> i32 {
    match difficulty {
        "Easy" => random.next_int(65, 100),
        "Medium" => random.next_int(55, 70),
        "Hard" => random.next_int(45, 60),
        "Impossible" => random.next_int(30, 50),
        _ => random.next_int(55, 70),
    }
}

fn random_spawn_land(n: &mut NationState, game: &Game) -> Option<TileRef> {
    let cell = n.nation.spawn_cell?;
    let delta = 25;
    let mut tries = 0;
    while tries < 50 {
        tries += 1;
        let x = n.random.next_int(cell[0] - delta, cell[0] + delta);
        let y = n.random.next_int(cell[1] - delta, cell[1] + delta);
        if !game.is_valid_coord(x, y) {
            continue;
        }
        let tile = game.ref_xy(x as u32, y as u32);
        if !game.is_land(tile) || game.has_owner(tile) || game.is_impassable(tile) {
            continue;
        }
        if game.terrain_type(tile) == TerrainType::Mountain && n.random.chance(2) {
            continue;
        }
        return Some(tile);
    }
    None
}
