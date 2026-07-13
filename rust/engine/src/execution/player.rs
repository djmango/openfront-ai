use super::Execution;
use crate::core::schemas::unit_type;
use crate::game::Game;
use crate::map::TileRef;
use crate::util::simple_hash;

const TICKS_PER_CLUSTER_CALC: u32 = 20;

/// Troop income and death checks (TS `PlayerExecution` subset for hash parity).
pub struct PlayerExecution {
    small_id: u16,
    active: bool,
}

impl PlayerExecution {
    pub fn new(small_id: u16) -> Self {
        Self {
            small_id,
            active: true,
        }
    }

    pub fn small_id(&self) -> u16 {
        self.small_id
    }
}

impl Execution for PlayerExecution {
    fn init(&mut self, game: &mut Game, tick: u32) {
        if let Some(p) = game.player_by_small_id_mut(self.small_id) {
            // TS `PlayerExecution.init` seeds cluster-removal cadence from
            // `player.id()`, not display name. The cadence is observable when
            // surrounded clusters are captured before same-tick attacks run.
            let id_hash = simple_hash(&p.id);
            p.last_cluster_calc = tick + (id_hash as u32 % TICKS_PER_CLUSTER_CALC);
        }
    }

    fn tick(&mut self, game: &mut Game, tick: u32) {
        if !self.active {
            return;
        }
        game.decay_relations(self.small_id);
        let spawn = game.in_spawn_phase();
        let Some(p) = game.player_by_small_id(self.small_id).cloned() else {
            self.active = false;
            return;
        };
        if p.spawn_tile.is_none() {
            return;
        }

        reconcile_structure_ownership(game, self.small_id);

        // TS `isAlive()` is `_tiles.size > 0`. Check tiles before the sticky
        // `alive` flag so `removeOnDeath` still runs when conquer already
        // cleared `alive` mid-tick on the previous (or same) tick.
        let tiles_owned = game
            .player_by_small_id(self.small_id)
            .map(|p| p.tiles_owned)
            .unwrap_or(0);
        if tiles_owned == 0 {
            if !spawn {
                if let Some(pm) = game.player_by_small_id_mut(self.small_id) {
                    pm.alive = false;
                }
                remove_on_death(game, self.small_id);
                self.active = false;
            }
            return;
        }
        if !game
            .player_by_small_id(self.small_id)
            .is_some_and(|p| p.alive)
        {
            return;
        }

        // TS `PlayerExecution.tick`: `addTroops(config.troopIncreaseRate(player))`
        // with the raw float (can be negative when over maxTroops). Must go through
        // `add_troops` so negative deltas match TS's removeTroops(-delta) flooring.
        let inc = game.troop_increase_rate_raw_for(self.small_id);
        let gold = game.wire.gold_addition_rate(p.player_type);
        game.add_troops(self.small_id, inc);
        if let Some(pm) = game.player_by_small_id_mut(self.small_id) {
            pm.gold += gold;
        }

        game.expire_alliances_for(self.small_id);
        game.expire_temporary_embargoes_for(self.small_id);

        crate::execution::player_clusters::maybe_remove_clusters(game, self.small_id, tick);
    }

    fn is_active(&self) -> bool {
        self.active
    }

    fn active_during_spawn(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Player, PlayerType};

    #[test]
    fn cluster_calc_seed_uses_player_id_like_ts() {
        let mut game = Game::default();
        let small_id = 1;
        game.add_player(Player {
            id: "w5d7jk0b".into(),
            name: "Arawak Colony".into(),
            small_id,
            ..Default::default()
        });

        let mut exec = PlayerExecution::new(small_id);
        exec.init(&mut game, 100);

        let player = game.player_by_small_id(small_id).unwrap();
        let id_seed = simple_hash(&player.id) as u32 % TICKS_PER_CLUSTER_CALC;
        let name_seed = simple_hash(&player.name) as u32 % TICKS_PER_CLUSTER_CALC;
        assert_ne!(id_seed, name_seed, "fixture should catch id/name swaps");
        assert_eq!(player.last_cluster_calc, 100 + id_seed);
    }

    /// TS `PlayerImpl.addTroops` routes negatives through `removeTroops(-delta)`,
    /// so a fractional over-max income of e.g. `-6360.28` removes 6360 troops
    /// (`floor` of the magnitude), not 6361 (`floor` of the signed value).
    /// Regresses the UK soft ±1 troop fork at tick ~1793 on `curr-b030-s0-pangaea`.
    #[test]
    fn add_troops_negative_uses_magnitude_floor_like_ts() {
        let mut game = crate::test_util::plains_game(4, 4);
        game.add_player(Player {
            id: "p".into(),
            small_id: 1,
            player_type: PlayerType::Human,
            troops: 10_000,
            ..Default::default()
        });
        game.add_troops(1, -6360.282895866956);
        assert_eq!(game.player_by_small_id(1).unwrap().troops, 10_000 - 6360);
    }

    #[test]
    fn troop_increase_rate_negative_matches_add_troops_rounding() {
        // Easy nation, over maxTroops: raw delta is a non-integral negative.
        let cfg = crate::core::config::Config::new(
            crate::core::schemas::GameConfig {
                game_map: "tiny".into(),
                difficulty: "Easy".into(),
                donate_gold: false,
                donate_troops: false,
                game_type: "Singleplayer".into(),
                game_mode: "Free For All".into(),
                game_map_size: "Normal".into(),
                nations: crate::core::schemas::NationsConfig::Mode("default".into()),
                bots: 0,
                infinite_gold: false,
                infinite_troops: false,
                instant_build: false,
                random_spawn: false,
                doomsday_clock: None,
                disabled_units: None,
                player_teams: None,
                disable_alliances: None,
                spawn_immunity_duration: Some(0),
                starting_gold: None,
                gold_multiplier: None,
                max_timer_value: None,
                ranked_type: None,
            },
            false,
        );
        let troops = 159_354;
        let tiles = 2_263;
        let raw = cfg.troop_increase_rate_raw(PlayerType::Nation, troops, tiles, 0);
        assert!(raw < -1.0, "expected meaningful negative income, got {raw}");
        let floored_signed = crate::util::to_int(raw);
        let ts_style = cfg.troop_increase_rate(PlayerType::Nation, troops, tiles, 0);
        assert_eq!(ts_style, -crate::util::to_int(-raw));
        assert!(
            floored_signed < ts_style,
            "signed floor ({floored_signed}) must be stricter than TS-style ({ts_style}) for {raw}"
        );
    }
}

/// TS `PlayerExecution.removeOnDeath` - once a player has no tiles, drop
/// their gold, delete every unit except in-flight nukes (which keep
/// simulating/detonating against their last target), and silently detach
/// all alliances.
fn remove_on_death(game: &mut Game, small_id: u16) {
    if let Some(pm) = game.player_by_small_id_mut(small_id) {
        pm.gold = 0;
    }
    let unit_ids: Vec<i32> = game
        .player_by_small_id(small_id)
        .map(|p| {
            p.units
                .iter()
                .filter(|u| {
                    !matches!(
                        u.unit_type.as_str(),
                        unit_type::ATOM_BOMB
                            | unit_type::HYDROGEN_BOMB
                            | unit_type::MIRV
                            | unit_type::MIRV_WARHEAD
                    )
                })
                .map(|u| u.id)
                .collect()
        })
        .unwrap_or_default();
    for unit_id in unit_ids {
        game.remove_unit(small_id, unit_id);
    }
    game.remove_all_alliances_for(small_id);
}

fn is_structure(ty: &str) -> bool {
    matches!(
        ty,
        unit_type::CITY
            | unit_type::DEFENSE_POST
            | unit_type::SAM_LAUNCHER
            | unit_type::MISSILE_SILO
            | unit_type::PORT
            | unit_type::FACTORY
    )
}

/// TS `PlayerExecution.tick` structure ownership reconciliation.
fn reconcile_structure_ownership(game: &mut Game, small_id: u16) {
    let units: Vec<(i32, String, TileRef)> = game
        .player_by_small_id(small_id)
        .map(|p| {
            p.units
                .iter()
                .filter(|u| is_structure(&u.unit_type))
                .map(|u| (u.id, u.unit_type.clone(), u.tile as TileRef))
                .collect()
        })
        .unwrap_or_default();

    for (unit_id, utype, tile) in units {
        let owner = game.map.owner_id(tile);
        if owner == 0 {
            game.remove_unit(small_id, unit_id);
            continue;
        }
        if owner == small_id {
            continue;
        }
        if utype == unit_type::DEFENSE_POST {
            game.remove_unit(small_id, unit_id);
        } else {
            game.capture_unit(small_id, owner, unit_id);
        }
    }
}
