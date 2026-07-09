//! Nation structure spawning (`NationStructureBehavior.ts` city subset).

use crate::core::schemas::unit_type;
use crate::execution::{ExecEnum, construction::ConstructionExecution};
use crate::game::Game;
use crate::map::TileRef;
use crate::prng::PseudoRandom;
use crate::spatial::closest_tile;
use std::collections::HashMap;

const CITY_PERCEIVED_COST_INCREASE_PER_OWNED: f64 = 1.0;
const SAVE_UP_HYDROGEN_COUNT: i64 = 5;
/// TS `UPGRADE_DENSITY_THRESHOLD`  -  prefer upgrading over new builds when exceeded.
const UPGRADE_DENSITY_THRESHOLD: f64 = 1.0 / 1500.0;

struct Bbox {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
}

fn border_bbox(game: &Game, small_id: u16) -> Option<Bbox> {
    let p = game.player_by_small_id(small_id)?;
    if p.border_tiles.is_empty() {
        return None;
    }
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    for t in p.border_tiles.iter() {
        let x = game.map.x(t);
        let y = game.map.y(t);
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    Some(Bbox {
        min_x,
        min_y,
        max_x,
        max_y,
    })
}

fn rand_territory_tile(
    game: &Game,
    random: &mut PseudoRandom,
    small_id: u16,
    bbox: &Bbox,
) -> Option<TileRef> {
    for _ in 0..100 {
        let x = random.next_int(bbox.min_x as i32, bbox.max_x as i32);
        let y = random.next_int(bbox.min_y as i32, bbox.max_y as i32);
        if !game.is_valid_coord(x, y) {
            continue;
        }
        let tile = game.ref_xy(x as u32, y as u32);
        if game.map.owner_id(tile) == small_id {
            return Some(tile);
        }
    }
    let p = game.player_by_small_id(small_id)?;
    if !p.owned_tiles.is_empty() && p.owned_tiles.len() <= 100 {
        return random.rand_element(&p.owned_tiles);
    }
    None
}

fn rand_territory_tile_array(
    game: &Game,
    random: &mut PseudoRandom,
    small_id: u16,
    n: usize,
) -> Vec<TileRef> {
    let Some(bbox) = border_bbox(game, small_id) else {
        return Vec::new();
    };
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if let Some(t) = rand_territory_tile(game, random, small_id, &bbox) {
            out.push(t);
        }
    }
    out
}

fn spacing_constants(game: &Game) -> (u32, u32) {
    let border = game.wire.atom_bomb_outer_range();
    (border, border * 2)
}

fn should_use_connectivity_score(random: &mut PseudoRandom, difficulty: &str) -> bool {
    let chance = match difficulty {
        "Easy" => 0,
        "Medium" => 60,
        "Hard" => 75,
        "Impossible" => 100,
        _ => 60,
    };
    random.next_int(0, 100) < chance
}

fn city_value(game: &Game, small_id: u16, tile: TileRef, use_connection_score: bool) -> u32 {
    let (border_spacing, structure_spacing) = spacing_constants(game);
    let mut w = game.map.magnitude(tile) as u32;
    if let Some(p) = game.player_by_small_id(small_id) {
        if !p.border_tiles.is_empty() {
            let (_, dist) = closest_tile(game, p.border_tiles.as_slice(), tile);
            if dist != u32::MAX {
                w += dist.min(border_spacing);
            } else {
                w += border_spacing;
            }
        }
        let city_tiles: Vec<TileRef> = p
            .units
            .iter()
            .filter(|u| u.unit_type == unit_type::CITY)
            .map(|u| u.tile as TileRef)
            .collect();
        if !city_tiles.is_empty() {
            if let Some((city, _)) = crate::spatial::closest_two_tiles(game, &city_tiles, &[tile]) {
                let d = game.manhattan_dist(city, tile);
                w += d.min(structure_spacing);
            }
        }
        let factory_tiles: Vec<TileRef> = p
            .units
            .iter()
            .filter(|u| u.unit_type == unit_type::FACTORY)
            .map(|u| u.tile as TileRef)
            .collect();
        if !factory_tiles.is_empty() {
            if let Some((factory, _)) =
                crate::spatial::closest_two_tiles(game, &factory_tiles, &[tile])
            {
                let d = game.manhattan_dist(factory, tile);
                w += d.min(structure_spacing);
            }
        }
    }
    let _ = use_connection_score;
    w
}

pub fn valid_structure_spawn_tiles(game: &Game, small_id: u16, tile: TileRef) -> Vec<TileRef> {
    if game.map.owner_id(tile) != small_id {
        return Vec::new();
    }
    let search_radius = 15u32;
    let search_radius_sq = search_radius * search_radius;
    let unit_search_radius = search_radius * 2;
    let unit_search_radius_sq = unit_search_radius * unit_search_radius;
    let min_dist_sq = game.wire.structure_min_dist().pow(2);

    let mut scratch = crate::water::BfsScratch::new((game.map.width * game.map.height) as usize);
    let nearby = game.map.bfs_with_scratch(&mut scratch, tile, |gm, t| {
        gm.euclidean_dist_squared(tile, t) < search_radius_sq && gm.owner_id(t) == small_id
    });

    let structure_types = [
        unit_type::CITY,
        unit_type::PORT,
        unit_type::FACTORY,
        unit_type::MISSILE_SILO,
        unit_type::DEFENSE_POST,
        unit_type::SAM_LAUNCHER,
    ];

    let mut nearby_units: Vec<TileRef> = Vec::new();
    for p in game.players_in_order() {
        for u in &p.units {
            if !structure_types.contains(&u.unit_type.as_str()) {
                continue;
            }
            let ut = u.tile as TileRef;
            if game.map.euclidean_dist_squared(ut, tile) < unit_search_radius_sq {
                nearby_units.push(ut);
            }
        }
    }

    let mut valid: Vec<TileRef> = Vec::new();
    'tile: for &t in &nearby {
        for &ut in &nearby_units {
            if game.map.euclidean_dist_squared(ut, t) < min_dist_sq {
                continue 'tile;
            }
        }
        valid.push(t);
    }
    valid.sort_by_key(|t| game.map.euclidean_dist_squared(*t, tile));
    valid
}

/// TS `PlayerImpl.portSpawn`.
fn port_spawn(game: &Game, small_id: u16, tile: TileRef) -> Option<TileRef> {
    let radius = game.wire.radius_port_spawn();
    let valid: std::collections::HashSet<TileRef> =
        valid_structure_spawn_tiles(game, small_id, tile).into_iter().collect();
    let mut scratch = crate::water::BfsScratch::new((game.map.width * game.map.height) as usize);
    let reachable = game.map.bfs_with_scratch(&mut scratch, tile, |_, t| {
        game.manhattan_dist(tile, t) <= radius
    });
    let mut spawns: Vec<TileRef> = reachable
        .into_iter()
        .filter(|&t| game.map.owner_id(t) == small_id && game.is_shore(t))
        .collect();
    spawns.sort_by_key(|t| game.manhattan_dist(*t, tile));
    spawns.into_iter().find(|t| valid.contains(t))
}

/// TS `PlayerImpl.landBasedStructureSpawn` / `canBuild` for land structures.
fn land_based_structure_spawn(game: &Game, small_id: u16, tile: TileRef) -> Option<TileRef> {
    valid_structure_spawn_tiles(game, small_id, tile).into_iter().next()
}

/// TS `PlayerImpl.canSpawnUnitType` tile resolution for construction intents.
pub fn resolve_structure_spawn_tile(
    game: &Game,
    small_id: u16,
    unit_type_name: &str,
    tile: TileRef,
) -> Option<TileRef> {
    match unit_type_name {
        unit_type::PORT => port_spawn(game, small_id, tile),
        unit_type::CITY
        | unit_type::FACTORY
        | unit_type::MISSILE_SILO
        | unit_type::DEFENSE_POST
        | unit_type::SAM_LAUNCHER => land_based_structure_spawn(game, small_id, tile),
        _ => None,
    }
}

pub fn can_build_land_structure(game: &Game, small_id: u16, tile: TileRef) -> Option<TileRef> {
    land_based_structure_spawn(game, small_id, tile)
}

/// TS `SharedWaterCache.TTL_TICKS`  -  rebuilt at most once every 3s (30 ticks).
const SHARED_WATER_CACHE_TTL_TICKS: i64 = 30;

/// Which water bodies (lake component ids, plus a `u32::MAX` sentinel for ocean) a
/// player's coastline touches, keyed by small id. Bots are excluded (TS `SharedWaterCache`
/// treats them as never having a trade partner and never being one).
fn player_water_touch(
    game: &Game,
    small_id: u16,
) -> (bool, std::collections::HashSet<u32>) {
    let mut has_ocean = false;
    let mut lakes = std::collections::HashSet::new();
    game.for_each_border_tile(small_id, |border| {
        if !game.is_shore(border) {
            return;
        }
        game.map.for_each_neighbor4(border, |n| {
            if !game.is_water(n) {
                return;
            }
            if game.map.is_ocean(n) {
                has_ocean = true;
                return;
            }
            if let Some(c) = game.get_water_component(n) {
                lakes.insert(c);
            }
        });
    });
    (has_ocean, lakes)
}

/// TS `SharedWaterCache.build`  -  ocean is always shared once touched (no partner
/// requirement); a lake is shared only if some other non-bot, alive player also
/// touches it (mirrors `canTrade`, which is always true since embargoes aren't modeled).
fn build_shared_water_cache(
    game: &Game,
) -> HashMap<u16, Option<std::collections::HashSet<u32>>> {
    let mut touch: HashMap<u16, (bool, std::collections::HashSet<u32>)> = HashMap::new();
    let mut lake_partners: HashMap<u32, Vec<u16>> = HashMap::new();
    for p in game.players_alive() {
        if p.player_type == crate::game::PlayerType::Bot {
            continue;
        }
        let (has_ocean, lakes) = player_water_touch(game, p.small_id);
        for &c in &lakes {
            lake_partners.entry(c).or_default().push(p.small_id);
        }
        touch.insert(p.small_id, (has_ocean, lakes));
    }

    let mut result = HashMap::new();
    for (small_id, (has_ocean, lakes)) in &touch {
        let mut shared = std::collections::HashSet::new();
        if *has_ocean {
            shared.insert(u32::MAX);
        }
        for c in lakes {
            if let Some(partners) = lake_partners.get(c) {
                if partners.iter().any(|other| other != small_id) {
                    shared.insert(*c);
                }
            }
        }
        result.insert(*small_id, if shared.is_empty() { None } else { Some(shared) });
    }
    result
}

pub fn shared_water_components(game: &Game, small_id: u16) -> Option<std::collections::HashSet<u32>> {
    let tick = game.ticks() as i64;
    let needs_rebuild = {
        let last = game.shared_water_cache_tick.borrow();
        last.is_none_or(|t| tick - t >= SHARED_WATER_CACHE_TTL_TICKS)
    };
    if needs_rebuild {
        let fresh: HashMap<u16, Option<std::collections::HashSet<u32>>> =
            build_shared_water_cache(game);
        *game.shared_water_cache.borrow_mut() = fresh;
        *game.shared_water_cache_tick.borrow_mut() = Some(tick);
    }
    game.shared_water_cache
        .borrow()
        .get(&small_id)
        .cloned()
        .unwrap_or(None)
}

fn rand_coastal_tile_array(
    game: &Game,
    random: &mut PseudoRandom,
    small_id: u16,
    n: usize,
) -> Vec<TileRef> {
    let Some(shared) = shared_water_components(game, small_id) else {
        return Vec::new();
    };
    let mut tiles: Vec<TileRef> = Vec::new();
    game.for_each_border_tile(small_id, |border| {
        if !game.is_shore(border) {
            return;
        }
        let mut ok = false;
        game.map.for_each_neighbor4(border, |neighbor| {
            if ok || !game.is_water(neighbor) {
                return;
            }
            if game.map.is_ocean(neighbor) && shared.contains(&u32::MAX) {
                ok = true;
                return;
            }
            if let Some(c) = game.get_water_component(neighbor) {
                if shared.contains(&c) {
                    ok = true;
                }
            }
        });
        if ok {
            tiles.push(border);
        }
    });
    if tiles.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if let Some(t) = random.rand_element(&tiles) {
            out.push(t);
        }
    }
    out
}

fn port_value(game: &Game, small_id: u16, tile: TileRef) -> u32 {
    let port_tiles: Vec<TileRef> = game
        .player_by_small_id(small_id)
        .map(|p| {
            p.units
                .iter()
                .filter(|u| u.unit_type == unit_type::PORT)
                .map(|u| u.tile as TileRef)
                .collect()
        })
        .unwrap_or_default();
    if port_tiles.is_empty() {
        return 1;
    }
    let (_, dist) = closest_tile(game, &port_tiles, tile);
    dist
}

fn can_build_structure(game: &Game, small_id: u16, unit_type_name: &str, tile: TileRef) -> bool {
    resolve_structure_spawn_tile(game, small_id, unit_type_name, tile).is_some()
}

fn structure_spawn_tile(
    game: &Game,
    random: &mut PseudoRandom,
    small_id: u16,
    unit_type_name: &str,
) -> Option<TileRef> {
    if unit_type_name == unit_type::PORT {
        let tiles = rand_coastal_tile_array(game, random, small_id, 25);
        if tiles.is_empty() {
            return None;
        }
        let mut best: Option<TileRef> = None;
        let mut best_value = 0u32;
        for &t in &tiles {
            let v = port_value(game, small_id, t);
            if best.is_some() && v <= best_value {
                continue;
            }
            if !can_build_structure(game, small_id, unit_type_name, t) {
                continue;
            }
            best = Some(t);
            best_value = v;
        }
        return best;
    }

    let tiles = rand_territory_tile_array(game, random, small_id, 25);
    if tiles.is_empty() {
        return None;
    }
    let difficulty = game.wire.game_config().difficulty.as_str();
    let use_connection_score = should_use_connectivity_score(random, difficulty);
    let mut best: Option<TileRef> = None;
    let mut best_value = 0u32;
    let debug = std::env::var("STRUCTURE_DEBUG")
        .ok()
        .is_some_and(|id| game.player_by_small_id(small_id).is_some_and(|p| p.id == id));
    if debug {
        if let Some(bbox) = border_bbox(game, small_id) {
            eprintln!("  bbox min=({}, {}) max=({}, {})", bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y);
        }
        eprintln!(
            "structure_spawn {:?} tiles_in={} use_conn={}",
            game.player_by_small_id(small_id).map(|p| p.id.as_str()),
            tiles.len(),
            use_connection_score
        );
    }
    for t in tiles {
        let v = if unit_type_name == unit_type::CITY {
            city_value(game, small_id, t, use_connection_score)
        } else {
            0
        };
        if debug && best.is_none() {
            eprintln!("  sample tile={} v={} can={}", t, v, can_build_structure(game, small_id, unit_type_name, t));
        }
        if best.is_some() && v <= best_value {
            continue;
        }
        if can_build_structure(game, small_id, unit_type_name, t) {
            if debug {
                eprintln!("  best tile={} v={}", t, v);
            }
            best = Some(t);
            best_value = v;
        }
    }
    best
}

pub fn save_up_target(game: &Game) -> i64 {
    if game.wire.is_unit_disabled(unit_type::MISSILE_SILO) {
        return game.wire.sam_launcher_cost();
    }
    if game.wire.game_config().game_mode == "Team" {
        return game.wire.hydrogen_bomb_cost();
    }
    if !game.wire.is_unit_disabled(unit_type::MIRV) {
        return game.wire.mirv_cost() + game.wire.hydrogen_bomb_cost();
    }
    if !game.wire.is_unit_disabled(unit_type::HYDROGEN_BOMB) {
        return game.wire.hydrogen_bomb_cost() * SAVE_UP_HYDROGEN_COUNT;
    }
    if !game.wire.is_unit_disabled(unit_type::ATOM_BOMB) {
        return game.wire.atom_bomb_cost() * 20;
    }
    game.wire.sam_launcher_cost()
}

fn perceived_city_cost(game: &Game, small_id: u16) -> i64 {
    let owned = game.units_owned_count(small_id, unit_type::CITY);
    let real = game.structure_cost(small_id, unit_type::CITY);
    let save_up = save_up_target(game);
    let gold = game.player_by_small_id(small_id).map(|p| p.gold).unwrap_or(0);
    if save_up == 0 || gold >= save_up {
        return real;
    }
    let multiplier = 1.0 + CITY_PERCEIVED_COST_INCREASE_PER_OWNED * owned as f64;
    ((real as f64) * multiplier).ceil() as i64
}

fn structure_unit_types() -> [&'static str; 6] {
    [
        unit_type::CITY,
        unit_type::PORT,
        unit_type::FACTORY,
        unit_type::MISSILE_SILO,
        unit_type::DEFENSE_POST,
        unit_type::SAM_LAUNCHER,
    ]
}

fn get_total_structure_density(game: &Game, small_id: u16) -> f64 {
    let Some(p) = game.player_by_small_id(small_id) else {
        return 0.0;
    };
    if p.tiles_owned <= 0 {
        return 0.0;
    }
    let types = structure_unit_types();
    let count = p
        .units
        .iter()
        .filter(|u| types.contains(&u.unit_type.as_str()))
        .count();
    count as f64 / p.tiles_owned as f64
}

const HIGH_NATION_DENSITY_THRESHOLD: f64 = 1.0 / 7500.0;
const FACTORY_COASTAL_RATIO_MULTIPLIER: f64 = 0.33;
const MAX_MISSILE_SILOS: u32 = 3;
const FIRST_MISSILE_SILO_RATIO: f64 = 0.4;

fn nation_count(game: &Game) -> usize {
    game.players_in_order()
        .iter()
        .filter(|p| p.player_type == crate::game::PlayerType::Nation)
        .count()
}

fn has_coastal_tiles(game: &Game, small_id: u16) -> bool {
    shared_water_components(game, small_id).is_some()
}

fn is_high_nation_density(game: &Game) -> bool {
    let land = game.num_land_tiles();
    if land == 0 {
        return false;
    }
    nation_count(game) as f64 / land as f64 > HIGH_NATION_DENSITY_THRESHOLD
}

fn sam_ratio_per_city(difficulty: &str) -> f64 {
    match difficulty {
        "Easy" => 0.15,
        "Medium" => 0.2,
        "Hard" => 0.25,
        "Impossible" => 0.3,
        _ => 0.2,
    }
}

fn should_build_structure(
    game: &Game,
    small_id: u16,
    structure_type: &str,
    city_count: u32,
    has_coastal: bool,
) -> bool {
    let difficulty = game.wire.game_config().difficulty.as_str();
    let (ratio, _perceived_inc) = match structure_type {
        unit_type::PORT => (0.75, 1.0),
        unit_type::FACTORY => (0.75, 1.0),
        unit_type::SAM_LAUNCHER => (sam_ratio_per_city(difficulty), 0.3),
        unit_type::MISSILE_SILO => (0.2, 1.0),
        _ => return false,
    };
    let mut effective_ratio = ratio;
    if structure_type == unit_type::FACTORY
        && has_coastal
        && !game.wire.is_unit_disabled(unit_type::PORT)
    {
        effective_ratio *= FACTORY_COASTAL_RATIO_MULTIPLIER;
    }
    let owned = game.units_owned_count(small_id, structure_type);
    if structure_type == unit_type::MISSILE_SILO && owned >= MAX_MISSILE_SILOS {
        return false;
    }
    if structure_type == unit_type::MISSILE_SILO && owned == 0 {
        effective_ratio = FIRST_MISSILE_SILO_RATIO;
    }
    let target = (city_count as f64 * effective_ratio).floor() as u32;
    owned < target
}

fn perceived_structure_cost(game: &Game, small_id: u16, structure_type: &str) -> i64 {
    let owned = game.units_owned_count(small_id, structure_type);
    let real = game.structure_cost(small_id, structure_type);
    let save_up = save_up_target(game);
    let gold = game.player_by_small_id(small_id).map(|p| p.gold).unwrap_or(0);
    if save_up == 0 || gold >= save_up {
        return real;
    }
    let increase = if structure_type == unit_type::CITY {
        CITY_PERCEIVED_COST_INCREASE_PER_OWNED
    } else {
        1.0
    };
    let multiplier = 1.0 + increase * owned as f64;
    ((real as f64) * multiplier).ceil() as i64
}

/// TS `findBestStructureToUpgrade`.
///
/// Prefers structures protected by a SAM launcher; in 50% of cases (when not
/// already picking randomly) picks the second/third best instead of the best,
/// for variety. The per-candidate tie-break draw (`next_int(0, 5)`) happens
/// even when there's only one candidate, so RNG draw count must not be
/// short-circuited here even in the trivial single-candidate case.
fn find_best_structure_to_upgrade(
    game: &Game,
    random: &mut PseudoRandom,
    small_id: u16,
    unit_type_name: &str,
) -> Option<i32> {
    let upgradable: Vec<(i32, TileRef, i32)> = game
        .player_by_small_id(small_id)
        .map(|p| {
            p.units
                .iter()
                .filter(|u| u.unit_type == unit_type_name)
                .filter(|u| !u.under_construction)
                .filter(|u| game.can_upgrade_unit(small_id, u.id))
                .map(|u| (u.id, u.tile as TileRef, u.level))
                .collect()
        })
        .unwrap_or_default();
    if upgradable.is_empty() {
        return None;
    }
    let difficulty = game.wire.game_config().difficulty.as_str();
    let random_chance = match difficulty {
        "Easy" => 70,
        "Medium" => 40,
        "Hard" => 25,
        "Impossible" => 10,
        _ => 40,
    };
    if random.next_int(0, 100) < random_chance {
        return random.rand_element(&upgradable).map(|(id, _, _)| id);
    }

    let sam_launchers: Vec<(TileRef, i32)> = game
        .player_by_small_id(small_id)
        .map(|p| {
            p.units
                .iter()
                .filter(|u| u.unit_type == unit_type::SAM_LAUNCHER)
                .map(|u| (u.tile as TileRef, u.level))
                .collect()
        })
        .unwrap_or_default();

    let mut scored: Vec<((i32, TileRef, i32), f64)> = upgradable
        .iter()
        .map(|&(id, tile, level)| {
            let mut score = 0.0;
            for &(sam_tile, sam_level) in &sam_launchers {
                let sam_range = sam_range_for_level(sam_level);
                let sam_range_squared = sam_range * sam_range;
                let dist_squared = game.map.euclidean_dist_squared(tile, sam_tile) as f64;
                if dist_squared <= sam_range_squared {
                    score += 10.0;
                    if sam_level > 1 {
                        score += (sam_level - 1) as f64 * 7.5;
                    }
                }
            }
            score += random.next_int(0, 5) as f64;
            ((id, tile, level), score)
        })
        .collect();
    if scored.is_empty() {
        return None;
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    if scored.len() >= 2 && random.chance(2) {
        let pick_index = if scored.len() >= 3 { random.next_int(1, 3) as usize } else { 1 };
        return Some(scored[pick_index].0.0);
    }
    Some(scored[0].0.0)
}

/// TS `Config.samRange`.
fn sam_range_for_level(level: i32) -> f64 {
    150.0 - 480.0 / (level as f64 + 5.0)
}

fn maybe_upgrade_structure_of_type(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    unit_type_name: &str,
) -> bool {
    if get_total_structure_density(game, small_id) <= UPGRADE_DENSITY_THRESHOLD {
        return false;
    }
    let Some(unit_id) = find_best_structure_to_upgrade(game, random, small_id, unit_type_name) else {
        return false;
    };
    game.add_execution(ExecEnum::UpgradeStructure(
        crate::execution::upgrade_structure::UpgradeStructureExecution::new(small_id, unit_id),
    ));
    true
}

fn maybe_spawn_structure(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    unit_type_name: &str,
) -> bool {
    if game.wire.is_unit_disabled(unit_type_name) {
        return false;
    }
    let cost = perceived_structure_cost(game, small_id, unit_type_name);
    if game.player_by_small_id(small_id).is_none_or(|p| p.gold < cost) {
        return false;
    }
    let type_count = game.unit_count(small_id, unit_type_name);
    if get_total_structure_density(game, small_id) > UPGRADE_DENSITY_THRESHOLD
        && crate::execution::upgrade_structure::is_upgradable_type(unit_type_name)
    {
        if maybe_upgrade_structure_of_type(game, random, small_id, unit_type_name) {
            return true;
        }
        if type_count > 0 {
            return false;
        }
    }
    let Some(tile) = structure_spawn_tile(game, random, small_id, unit_type_name) else {
        return false;
    };
    if !can_build_structure(game, small_id, unit_type_name, tile) {
        return false;
    }
    game.add_execution(ExecEnum::Construction(ConstructionExecution::new(
        small_id,
        unit_type_name,
        tile,
    )));
    true
}

/// TS `NationStructureBehavior.doHandleStructures`.
pub fn do_handle_structures(game: &mut Game, random: &mut PseudoRandom, small_id: u16) -> bool {
    let cities_disabled = game.wire.is_unit_disabled(unit_type::CITY);
    let city_count = if cities_disabled {
        game.player_by_small_id(small_id)
            .map(|p| (p.tiles_owned / 2000).max(1) as u32)
            .unwrap_or(1)
    } else {
        game.units_owned_count(small_id, unit_type::CITY)
    };
    let has_coastal = has_coastal_tiles(game, small_id);
    let missile_silos_enabled = !game.wire.is_unit_disabled(unit_type::MISSILE_SILO);
    let nukes_enabled = !game.wire.is_unit_disabled(unit_type::ATOM_BOMB)
        || !game.wire.is_unit_disabled(unit_type::HYDROGEN_BOMB)
        || !game.wire.is_unit_disabled(unit_type::MIRV);

    if !cities_disabled
        && game.units_owned_count(small_id, unit_type::CITY) == 0
        && is_high_nation_density(game)
    {
        let preferred = if has_coastal && !game.wire.is_unit_disabled(unit_type::PORT) {
            unit_type::PORT
        } else {
            unit_type::FACTORY
        };
        if maybe_spawn_structure(game, random, small_id, preferred) {
            return true;
        }
    }

    let build_order = [
        unit_type::PORT,
        unit_type::FACTORY,
        unit_type::SAM_LAUNCHER,
        unit_type::MISSILE_SILO,
    ];
    for structure_type in build_order {
        if game.wire.is_unit_disabled(structure_type) {
            continue;
        }
        if structure_type == unit_type::PORT && !has_coastal {
            continue;
        }
        if !nukes_enabled
            && (structure_type == unit_type::MISSILE_SILO || structure_type == unit_type::SAM_LAUNCHER)
        {
            continue;
        }
        if !missile_silos_enabled && structure_type == unit_type::SAM_LAUNCHER {
            continue;
        }
        if should_build_structure(game, small_id, structure_type, city_count, has_coastal)
            && maybe_spawn_structure(game, random, small_id, structure_type)
        {
            return true;
        }
    }

    if !cities_disabled && maybe_spawn_structure(game, random, small_id, unit_type::CITY) {
        return true;
    }
    false
}

const UNDER_ATTACK_THREAT_RATIO: f64 = 0.35;
const DEFENSE_POST_RATIO_PER_POST: f64 = 0.4;

fn get_attack_front_tiles(game: &Game, small_id: u16, attackers: &[u16]) -> Vec<TileRef> {
    let attacker_set: std::collections::HashSet<u16> = attackers.iter().copied().collect();
    if attacker_set.is_empty() {
        return Vec::new();
    }
    let mut front = Vec::new();
    game.for_each_border_tile(small_id, |border| {
        game.map.for_each_neighbor4(border, |n| {
            let owner = game.map.owner_id(n);
            if owner != 0 && attacker_set.contains(&owner) {
                front.push(border);
            }
        });
    });
    front.sort_unstable();
    front.dedup();
    front
}

fn count_defense_posts_near_front(game: &Game, small_id: u16, front_tiles: &[TileRef], cap: usize) -> usize {
    if front_tiles.is_empty() {
        return 0;
    }
    let (border_spacing, _) = spacing_constants(game);
    let range_sq = (border_spacing as f64 * 1.5).powi(2);
    let Some(p) = game.player_by_small_id(small_id) else {
        return 0;
    };
    let mut count = 0usize;
    'posts: for u in &p.units {
        if u.unit_type != unit_type::DEFENSE_POST {
            continue;
        }
        let ut = u.tile as TileRef;
        for &ft in front_tiles {
            if game.map.euclidean_dist_squared(ut, ft) <= range_sq as u32 {
                count += 1;
                if count >= cap {
                    break 'posts;
                }
                break;
            }
        }
    }
    count
}

fn sample_tiles_near_front(
    game: &Game,
    random: &mut PseudoRandom,
    small_id: u16,
    front_tiles: &[TileRef],
    count: usize,
) -> Vec<TileRef> {
    if front_tiles.is_empty() {
        return Vec::new();
    }
    let (border_spacing, _) = spacing_constants(game);
    let search_radius = (border_spacing as f64 * 1.5).ceil() as i32;
    let min_border_dist = (border_spacing as f64 * 0.75).ceil() as u32;
    let max_border_dist = (border_spacing as f64 * 1.5).ceil() as u32;
    let spread_range_sq = (border_spacing as f64 * 1.5).powi(2) as u32;

    let existing_dp: Vec<TileRef> = game
        .player_by_small_id(small_id)
        .map(|p| {
            p.units
                .iter()
                .filter(|u| u.unit_type == unit_type::DEFENSE_POST)
                .map(|u| u.tile as TileRef)
                .collect()
        })
        .unwrap_or_default();

    let mut anchors: Vec<TileRef> = if existing_dp.is_empty() {
        front_tiles.to_vec()
    } else {
        front_tiles
            .iter()
            .copied()
            .filter(|ft| {
                !existing_dp
                    .iter()
                    .any(|dp| game.map.euclidean_dist_squared(*dp, *ft) < spread_range_sq)
            })
            .collect()
    };
    if anchors.is_empty() {
        anchors = front_tiles.to_vec();
    }

    let border_tiles = game.border_tiles_for(small_id);
    let mut result = Vec::new();
    for _ in 0..(count * 6) {
        if result.len() >= count {
            break;
        }
        let Some(anchor) = random.rand_element(&anchors) else {
            break;
        };
        let ax = game.map.x(anchor) as i32;
        let ay = game.map.y(anchor) as i32;
        let x = random.next_int(ax - search_radius, ax + search_radius);
        let y = random.next_int(ay - search_radius, ay + search_radius);
        if !game.is_valid_coord(x, y) {
            continue;
        }
        let t = game.ref_xy(x as u32, y as u32);
        if game.map.owner_id(t) != small_id {
            continue;
        }
        let (_, border_dist) = closest_tile(game, &border_tiles, t);
        if border_dist < min_border_dist || border_dist > max_border_dist {
            continue;
        }
        if can_build_land_structure(game, small_id, t).is_none() {
            continue;
        }
        result.push(t);
    }

    if result.is_empty() {
        return rand_territory_tile_array(game, random, small_id, count)
            .into_iter()
            .filter(|t| can_build_land_structure(game, small_id, *t).is_some())
            .take(count)
            .collect();
    }
    result
}

/// TS `NationStructureBehavior.tryBuildDefensePost`.
pub fn try_build_defense_post(game: &mut Game, random: &mut PseudoRandom, small_id: u16) -> bool {
    if game.wire.is_unit_disabled(unit_type::DEFENSE_POST) {
        return false;
    }
    let difficulty = game.wire.game_config().difficulty.as_str();
    if difficulty == "Easy" {
        return false;
    }
    if difficulty == "Medium" && !random.chance(2) {
        return false;
    }

    let attacks = game.incoming_attacks(small_id);
    if attacks.is_empty() {
        return false;
    }
    let Some(p) = game.player_by_small_id(small_id) else {
        return false;
    };
    if p.troops <= 0 {
        return false;
    }
    let incoming: f64 = attacks.iter().map(|a| a.troops).sum();
    let ratio = incoming / p.troops as f64;
    if ratio < UNDER_ATTACK_THREAT_RATIO {
        return false;
    }

    let allowed = if difficulty == "Medium" {
        1usize
    } else {
        (ratio / DEFENSE_POST_RATIO_PER_POST).ceil() as usize
    };

    let attackers: Vec<u16> = attacks.iter().map(|a| a.attacker_small_id).collect();
    let front = get_attack_front_tiles(game, small_id, &attackers);
    if count_defense_posts_near_front(game, small_id, &front, allowed) >= allowed {
        return false;
    }

    let cost = game.structure_cost(small_id, unit_type::DEFENSE_POST);
    if p.gold < cost {
        return false;
    }

    let tiles = sample_tiles_near_front(game, random, small_id, &front, 25);
    for tile in tiles {
        if can_build_land_structure(game, small_id, tile).is_none() {
            continue;
        }
        game.add_execution(ExecEnum::Construction(ConstructionExecution::new(
            small_id,
            unit_type::DEFENSE_POST,
            tile,
        )));
        return true;
    }
    false
}

