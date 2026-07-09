//! In-process RL environment session - native port of `bridge/env.ts`'s
//! `EnvSession` (reset / step / countWasted). This is the engine surface the
//! Rust trainer's `--engine native` backend drives directly, replacing the
//! JSON-over-pipes Node subprocess.

use crate::bot::tribe_spawner::TribeSpawner;
use crate::core::config::Config;
use crate::core::nation::create_nations_for_game;
use crate::core::schemas::{unit_type as ut, GameConfig as WireGameConfig};
use crate::core::terrain::{load_fresh_terrain_from_dir, GameMapSize};
use crate::execution::nation_structures::resolve_structure_spawn_tile;
use crate::execution::nuke_execution::can_build_nuke;
use crate::execution::{
    turn_to_executions, ExecEnum, NationExecution, NationRuntime, RecomputeRailClusterExecution,
    WinCheckExecution,
};
use crate::game::{Game, GameConfig, PlayerInfo, PlayerType};
use crate::map::TileRef;
use crate::obs::{
    borders_neutral_land, build_obs_head, can_extend_alliance, tile_bytes_le,
};
use crate::prng::PseudoRandom;
use crate::record::StampedIntent;
use crate::session::{seed_to_game_id, terrain_bytes, AGENT_CLIENT_ID};
use crate::spatial::can_build_transport_ship;
use crate::util::simple_hash;
use serde_json::{json, Value};
use std::path::Path;

pub struct RlSession {
    pub game: Game,
    game_id: String,
}

impl RlSession {
    /// Mirrors `bridge/env.ts::EnvSession.reset()`. Returns the session plus
    /// the obs head, raw terrain bytes, and the binary tile-state frame.
    pub fn reset(
        repo_root: &Path,
        map_key: &str,
        seed: &str,
        bots: u32,
        difficulty: &str,
        nations: Value,
    ) -> Result<(Self, Value, Vec<u8>, Vec<u8>), String> {
        let map_dir = repo_root
            .join("openfront/resources/maps")
            .join(map_key.to_lowercase());
        let wire_json = json!({
            "gameMap": map_key,
            "gameMapSize": "Normal",
            "gameMode": "Free For All",
            "gameType": "Singleplayer",
            "difficulty": difficulty,
            "nations": nations,
            "donateGold": true,
            "donateTroops": true,
            "bots": bots,
            "infiniteGold": false,
            "infiniteTroops": false,
            "instantBuild": false,
            "randomSpawn": false,
        });
        let wire: WireGameConfig =
            serde_json::from_value(wire_json.clone()).map_err(|e| format!("wire config: {e}"))?;
        let cfg = Config::from_value(&wire_json, false).map_err(|e| format!("config: {e}"))?;
        let terrain = load_fresh_terrain_from_dir(&map_dir, GameMapSize::Normal)?;

        let game_id = seed_to_game_id(seed);
        let stub_config = GameConfig {
            game_map: map_key.into(),
            bots,
            num_spawn_phase_turns: cfg.num_spawn_phase_turns(),
            random_spawn: cfg.is_random_spawn(),
        };

        // Mirror createGameRunner(): humans consume random.nextID() first,
        // then nations, so records replay bit-identically.
        let mut random = PseudoRandom::new(simple_hash(&game_id));
        let humans = vec![PlayerInfo {
            name: "Agent".into(),
            player_type: PlayerType::Human,
            client_id: Some(AGENT_CLIENT_ID.into()),
            id: random.next_id(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        }];
        let nations = create_nations_for_game(
            &wire,
            &terrain.nations,
            &terrain.additional_nations,
            humans.len() as u32,
            &mut random,
        );
        let nation_infos: Vec<PlayerInfo> = nations
            .iter()
            .map(|n| PlayerInfo {
                name: n.nation.name.clone(),
                player_type: PlayerType::Nation,
                client_id: None,
                id: n.player_id.clone(),
                clan_tag: None,
                friends: Vec::new(),
                team: None,
            })
            .collect();

        let mut game = Game::new(
            game_id.clone(),
            stub_config,
            cfg.clone(),
            terrain.game_map,
            terrain.mini_game_map,
            terrain.team_game_spawn_areas,
        );
        game.init_player_teams(humans.len(), nations.len());
        for info in humans.iter().chain(nation_infos.iter()) {
            game.add_from_info(info);
        }

        // GameRunner.init(): singleplayer has no spawn timer - the spawn
        // phase ends the moment the agent picks a spawn.
        if cfg.spawn_nations() {
            for (n, info) in nations.iter().zip(nation_infos.iter()) {
                game.add_execution(ExecEnum::Nation(NationExecution::new(
                    game_id.clone(),
                    NationRuntime {
                        spawn_cell: n.nation.coordinates,
                        player_info: info.clone(),
                    },
                )));
            }
        }
        if cfg.bots() > 0 {
            let mut spawner = TribeSpawner::new(&game_id);
            for spawn in spawner.spawn_tribes(cfg.bots()) {
                game.add_execution(ExecEnum::Spawn(spawn));
            }
        }
        game.add_execution(ExecEnum::WinCheck(WinCheckExecution::new()));
        // Doomsday clock defaults to disabled and RL never enables it.
        if !cfg.is_unit_disabled(ut::FACTORY) {
            game.add_execution(ExecEnum::RecomputeRailCluster(
                RecomputeRailClusterExecution::new(),
            ));
        }

        // env.ts: advance one tick so executions initialize; the agent
        // spawns via step().
        game.execute_next_tick();

        let mut session = Self { game, game_id };
        let head = build_obs_head(&session.game, AGENT_CLIENT_ID, Value::Null);
        let terrain_raw = terrain_bytes(&session.game);
        let tiles = tile_bytes_le(&session.game);
        let _ = &mut session; // session.game mutated above via execute_next_tick
        Ok((session, head, terrain_raw, tiles))
    }

    /// Mirrors `bridge/env.ts::EnvSession.step()`: count wasted intents
    /// against pre-step state, submit one turn, run `ticks` engine ticks
    /// (breaking on a win update), return head + tiles.
    pub fn step(&mut self, intents: &[Value], ticks: u32) -> (Value, Vec<u8>) {
        let wasted = self.count_wasted(intents);
        if !intents.is_empty() {
            let stamped: Vec<StampedIntent> = intents
                .iter()
                .filter_map(|v| {
                    let mut obj = v.clone();
                    obj.as_object_mut()?
                        .insert("clientID".into(), Value::String(AGENT_CLIENT_ID.into()));
                    serde_json::from_value(obj).ok()
                })
                .collect();
            for exec in turn_to_executions(&self.game, &self.game_id, &stamped) {
                self.game.add_execution(exec);
            }
        }

        // env.ts reports the winner tuple only on the step the win update
        // fired, and stops ticking immediately.
        let mut winner = Value::Null;
        for _ in 0..ticks {
            let updates = self.game.execute_next_tick();
            if updates.win.is_some() {
                winner = crate::obs::winner_value(&self.game);
                break;
            }
        }

        let mut head = build_obs_head(&self.game, AGENT_CLIENT_ID, winner);
        if let Some(obj) = head.as_object_mut() {
            obj.insert("wasted".into(), Value::from(wasted));
        }
        (head, tile_bytes_le(&self.game))
    }

    /// Port of `bridge/env.ts::EnvSession.countWasted()` - intents the engine
    /// would silently discard, checked against the same pre-step state the
    /// action masks were built from.
    fn count_wasted(&mut self, intents: &[Value]) -> u32 {
        let Some(agent) = self.game.player_by_client_id(AGENT_CLIENT_ID) else {
            return 0;
        };
        if !agent.alive {
            return 0;
        }
        let sid = agent.small_id;
        let gold = agent.gold;
        let mut wasted = 0u32;
        for intent in intents {
            let itype = intent.get("type").and_then(Value::as_str).unwrap_or("");
            let is_wasted = match itype {
                "boat" => {
                    let dst = intent.get("dst").and_then(Value::as_u64).unwrap_or(0) as TileRef;
                    !self.can_build_boat(sid, dst)
                }
                "build_unit" => {
                    let unit = intent.get("unit").and_then(Value::as_str).unwrap_or("");
                    let tile = intent.get("tile").and_then(Value::as_u64).unwrap_or(0) as TileRef;
                    !self.can_build(sid, gold, unit, tile)
                }
                "attack" => {
                    let terra = intent
                        .get("targetID")
                        .map(|v| v.is_null())
                        .unwrap_or(false);
                    if terra {
                        let agent = self.game.player_by_client_id(AGENT_CLIENT_ID).unwrap();
                        !borders_neutral_land(&self.game, agent)
                    } else {
                        false
                    }
                }
                "upgrade_structure" => {
                    let uid = intent.get("unitId").and_then(Value::as_i64).unwrap_or(-1) as i32;
                    let owned = self.game.unit(sid, uid).is_some();
                    !(owned && self.game.can_upgrade_unit(sid, uid))
                }
                "move_warship" => {
                    let tile =
                        intent.get("tile").and_then(Value::as_u64).unwrap_or(0) as TileRef;
                    let unit_ids: Vec<i32> = intent
                        .get("unitIds")
                        .and_then(Value::as_array)
                        .map(|a| {
                            a.iter().filter_map(|v| v.as_i64()).map(|v| v as i32).collect()
                        })
                        .unwrap_or_default();
                    let comp = self.game.get_water_component(tile);
                    let movable = comp.is_some_and(|c| {
                        let agent = self.game.player_by_client_id(AGENT_CLIENT_ID).unwrap();
                        agent.units.iter().any(|u| {
                            u.unit_type == ut::WARSHIP
                                && unit_ids.contains(&u.id)
                                && self.game.has_water_component(u.tile as TileRef, c)
                        })
                    });
                    !movable
                }
                "cancel_boat" => {
                    let uid = intent.get("unitID").and_then(Value::as_i64).unwrap_or(-1) as i32;
                    let agent = self.game.player_by_client_id(AGENT_CLIENT_ID).unwrap();
                    !agent
                        .units
                        .iter()
                        .any(|u| u.unit_type == ut::TRANSPORT && u.id == uid)
                }
                "delete_unit" => {
                    let uid = intent.get("unitId").and_then(Value::as_i64).unwrap_or(-1) as i32;
                    let ok = self.game.unit(sid, uid).is_some_and(|u| {
                        let t = u.tile as TileRef;
                        self.game.is_land(t) && self.game.map.owner_id(t) == sid
                    }) && !self.game.in_spawn_phase()
                        && self.game.can_delete_unit(sid);
                    !ok
                }
                "cancel_attack" => {
                    let aid = intent.get("attackID").and_then(Value::as_str).unwrap_or("");
                    !self
                        .game
                        .live_attacks()
                        .any(|a| a.owner_small_id() == sid && a.attack_id() == aid)
                }
                "embargo" => {
                    let target_id =
                        intent.get("targetID").and_then(Value::as_str).unwrap_or("");
                    let start = intent.get("action").and_then(Value::as_str)
                        == Some("start");
                    match self.game.player_by_id(target_id) {
                        None => true,
                        Some(t) => {
                            let has = self.game.has_embargo_against(sid, t.small_id);
                            if start { has } else { !has }
                        }
                    }
                }
                "targetPlayer" => {
                    let target_id =
                        intent.get("target").and_then(Value::as_str).unwrap_or("");
                    match self.game.player_by_id(target_id) {
                        None => true,
                        Some(t) => !self.game.can_target(sid, t.small_id),
                    }
                }
                "allianceExtension" => {
                    let other_id =
                        intent.get("recipient").and_then(Value::as_str).unwrap_or("");
                    match self.game.player_by_id(other_id) {
                        None => true,
                        Some(o) => {
                            let other_sid = o.small_id;
                            let agent =
                                self.game.player_by_client_id(AGENT_CLIENT_ID).unwrap();
                            !can_extend_alliance(&self.game, agent, other_sid)
                        }
                    }
                }
                _ => false,
            };
            if is_wasted {
                wasted += 1;
            }
        }
        wasted
    }

    /// TS `agent.canBuild(TransportShip, dst)`: gold + not-disabled
    /// (canBuildUnitType) plus `canBuildTransportShip`, including the
    /// friendly/attackable destination-owner check TS applies.
    fn can_build_boat(&mut self, sid: u16, dst: TileRef) -> bool {
        let gold = self
            .game
            .player_by_small_id(sid)
            .map(|p| p.gold)
            .unwrap_or(0);
        if gold < self.game.structure_cost(sid, ut::TRANSPORT) {
            return false;
        }
        let dst_owner = self.game.map.owner_id(dst);
        if dst_owner != 0 && dst_owner != self.game.terra_nullius_id() && dst_owner != sid {
            if self.game.is_friendly(sid, dst_owner)
                || !self.game.can_attack_player(sid, dst_owner)
            {
                return false;
            }
        }
        can_build_transport_ship(&mut self.game, sid, dst).is_some()
    }

    /// TS `agent.canBuild(unit, tile)` for build_unit intents (structures,
    /// nukes, warships).
    fn can_build(&mut self, sid: u16, gold: i64, unit: &str, tile: TileRef) -> bool {
        if self.game.wire.is_unit_disabled(unit) {
            return false;
        }
        if gold < self.game.structure_cost(sid, unit) {
            return false;
        }
        match unit {
            ut::ATOM_BOMB | ut::HYDROGEN_BOMB | ut::MIRV => {
                can_build_nuke(&self.game, sid, unit, tile).is_some()
            }
            ut::WARSHIP => self.warship_spawn(sid, tile),
            ut::CITY | ut::PORT | ut::DEFENSE_POST | ut::MISSILE_SILO | ut::SAM_LAUNCHER
            | ut::FACTORY => resolve_structure_spawn_tile(&self.game, sid, unit, tile).is_some(),
            _ => false,
        }
    }

    /// TS `PlayerImpl.warshipSpawn()`: water tile + an active, completed own
    /// port in the same water component.
    fn warship_spawn(&self, sid: u16, tile: TileRef) -> bool {
        if self.game.is_land(tile) {
            return false;
        }
        let Some(comp) = self.game.get_water_component(tile) else {
            return false;
        };
        let Some(p) = self.game.player_by_small_id(sid) else {
            return false;
        };
        p.units.iter().any(|u| {
            u.unit_type == ut::PORT
                && !u.under_construction
                && self.game.has_water_component(u.tile as TileRef, comp)
        })
    }
}
