//! Nation AI spawn-phase + expansion (`NationExecution.ts` subset for hash parity).

use super::nation_tick::{NationBehaviorState, tick_nation_post_spawn};
use super::spawn::SpawnExecution;
use super::{ExecEnum, Execution};
use crate::game::{Game, PlayerInfo};
use crate::map::{TerrainType, TileRef};
use crate::prng::PseudoRandom;
use crate::util::simple_hash;

/// Runtime nation data (TS `Nation` class).
#[derive(Debug, Clone)]
pub struct NationRuntime {
    pub spawn_cell: Option<[i32; 2]>,
    pub player_info: PlayerInfo,
}

pub struct NationExecution {
    game_id: String,
    nation: NationRuntime,
    random: PseudoRandom,
    active: bool,
    spawn_exec_added: bool,
    behaviors_initialized: bool,
    attack_rate: i32,
    attack_tick: i32,
    trigger_ratio: f64,
    reserve_ratio: f64,
    expand_ratio: f64,
    bot_attack_troops_sent: f64,
    behavior: NationBehaviorState,
}

impl NationExecution {
    pub fn new(game_id: String, nation: NationRuntime) -> Self {
        let seed = simple_hash(&nation.player_info.id).wrapping_add(simple_hash(&game_id));
        let mut random = PseudoRandom::new(seed);
        let trigger_ratio = random.next_int(50, 60) as f64 / 100.0;
        let reserve_ratio = random.next_int(30, 40) as f64 / 100.0;
        let expand_ratio = random.next_int(10, 20) as f64 / 100.0;
        Self {
            game_id,
            nation,
            random,
            active: true,
            spawn_exec_added: false,
            behaviors_initialized: false,
            attack_rate: 0,
            attack_tick: 0,
            trigger_ratio,
            reserve_ratio,
            expand_ratio,
            bot_attack_troops_sent: 0.0,
            behavior: NationBehaviorState::default(),
        }
    }

    fn attack_rate_for_difficulty(&mut self, difficulty: &str) -> i32 {
        match difficulty {
            "Easy" => self.random.next_int(65, 100),
            "Medium" => self.random.next_int(55, 70),
            "Hard" => self.random.next_int(45, 60),
            "Impossible" => self.random.next_int(30, 50),
            _ => self.random.next_int(55, 70),
        }
    }

}

impl Execution for NationExecution {
    fn init(&mut self, game: &mut Game, _: u32) {
        self.attack_rate = self.attack_rate_for_difficulty(&game.wire.game_config().difficulty);
        self.attack_tick = self.random.next_int(0, self.attack_rate);
        if !game.has_player(&self.nation.player_info.id) {
            game.add_from_info(&self.nation.player_info);
        }
    }

    fn tick(&mut self, game: &mut Game, tick: u32) {
        if !self.active {
            return;
        }

        let player_id = self.nation.player_info.id.clone();
        let Some(small_id) = game.player_by_id(&player_id).map(|p| p.small_id) else {
            return;
        };

        if game.in_spawn_phase() {
            if game.has_spawned(small_id) {
                if (tick as i32) % self.attack_rate != self.attack_tick {
                    return;
                }
            } else if self.spawn_exec_added {
                return;
            }

            if self.nation.spawn_cell.is_none() {
                game.add_execution(ExecEnum::Spawn(SpawnExecution::new(
                    self.game_id.clone(),
                    self.nation.player_info.clone(),
                    None,
                )));
                self.spawn_exec_added = true;
                return;
            }

            // TS NationExecution: if spawn cell is outside team area, random spawn instead.
            if let Some(p) = game.player_by_id(&player_id) {
                if let Some(team) = p.team.as_deref() {
                    if let Some(area) = game.team_spawn_area(team) {
                        if let Some(cell) = self.nation.spawn_cell {
                            let in_area = cell[0] >= area.x as i32
                                && cell[0] < (area.x + area.width) as i32
                                && cell[1] >= area.y as i32
                                && cell[1] < (area.y + area.height) as i32;
                            if !in_area {
                                game.add_execution(ExecEnum::Spawn(SpawnExecution::new(
                                    self.game_id.clone(),
                                    self.nation.player_info.clone(),
                                    None,
                                )));
                                self.spawn_exec_added = true;
                                return;
                            }
                        }
                    }
                }
            }

            let Some(tile) = self.random_spawn_land(game) else {
                return;
            };

            game.add_execution(ExecEnum::Spawn(SpawnExecution::new(
                self.game_id.clone(),
                self.nation.player_info.clone(),
                Some(tile),
            )));
            self.spawn_exec_added = true;
            return;
        }

        if self.spawn_exec_added && !game.has_spawned(small_id) {
            return;
        }

        if game
            .player_by_id(&player_id)
            .is_some_and(|p| !p.alive || p.tiles_owned == 0)
        {
            self.active = false;
            return;
        }

        if !self.behaviors_initialized {
            self.behaviors_initialized = true;
            super::nation_tick::initialize_nation_behaviors(&mut self.random, &mut self.behavior);
            let troops = game
                .player_by_small_id(small_id)
                .map(|p| p.troops as f64 / 2.0)
                .unwrap_or(0.0);
            if troops >= 1.0 {
                game.add_land_attack(small_id, None, Some(troops));
            }
            return;
        }

        tick_nation_post_spawn(
            game,
            &mut self.random,
            small_id,
            &mut self.behavior,
            self.attack_rate,
            self.attack_tick,
            self.trigger_ratio,
            self.reserve_ratio,
            self.expand_ratio,
            &mut self.bot_attack_troops_sent,
        );
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        true
    }
}

impl NationExecution {
    fn random_spawn_land(&mut self, game: &Game) -> Option<TileRef> {
        let cell = self.nation.spawn_cell?;
        let delta = 25;
        let mut tries = 0;
        while tries < 50 {
            tries += 1;
            let x = self.random.next_int(cell[0] - delta, cell[0] + delta);
            let y = self.random.next_int(cell[1] - delta, cell[1] + delta);
            if !game.is_valid_coord(x, y) {
                continue;
            }
            let tile = game.ref_xy(x as u32, y as u32);
            if !game.is_land(tile) || game.has_owner(tile) || game.is_impassable(tile) {
                continue;
            }
            if game.terrain_type(tile) == TerrainType::Mountain && self.random.chance(2) {
                continue;
            }
            if std::env::var("SPAWN_DEBUG").ok().as_deref() == Some(self.nation.player_info.id.as_str()) {
                eprintln!(
                    "nation_spawn_land {} tile={} tries={}",
                    self.nation.player_info.id,
                    tile,
                    tries
                );
            }
            return Some(tile);
        }
        None
    }
}
