//! Nation structure spawning (`NationStructureBehavior.ts` city subset).

use crate::core::config::TrainRelation;
use crate::core::schemas::unit_type;
use crate::execution::{ExecEnum, construction::ConstructionExecution};
use crate::game::Game;
use crate::map::TileRef;
use crate::prng::PseudoRandom;
use crate::spatial::{closest_tile, closest_two_tiles};
use std::collections::HashMap;

const CITY_PERCEIVED_COST_INCREASE_PER_OWNED: f64 = 1.0;
const SAVE_UP_HYDROGEN_COUNT: i64 = 5;
/// TS `HIGH_STARTING_GOLD_THRESHOLD` (also duplicated in `nation_tick.rs` for cooldown gating).
const HIGH_STARTING_GOLD_THRESHOLD: i64 = 3_000_000;
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

/// TS `NationUtils.randTerritoryTileArray`  -  exposed crate-wide since
/// `NationNukeBehavior.maybeSendNuke` samples random tiles inside the
/// **target's** territory (not necessarily `this.player`'s), the same
/// primitive `NationStructureBehavior` uses for its own placement scans.
pub(crate) fn rand_territory_tile_array(
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

/// TS `NationStructureBehavior.shouldUseConnectivityScore`'s `randomChance`
/// lookup, split out from the `nextInt` draw itself so tests can verify the
/// exact per-difficulty cutoff (0/60/75/100) without needing to control
/// `PseudoRandom`'s output deterministically.
fn connectivity_score_chance(difficulty: &str) -> i32 {
    match difficulty {
        "Easy" => 0,
        "Medium" => 60,
        "Hard" => 75,
        "Impossible" => 100,
        _ => 60,
    }
}

fn should_use_connectivity_score(random: &mut PseudoRandom, difficulty: &str) -> bool {
    random.next_int(0, 100) < connectivity_score_chance(difficulty)
}

fn city_value_with_connectivity(
    base_value: f64,
    connectivity_score: f64,
    structure_spacing: u32,
    use_connection_score: bool,
) -> f64 {
    if use_connection_score {
        base_value + connectivity_score * structure_spacing as f64
    } else {
        base_value
    }
}

fn city_value(
    game: &Game,
    small_id: u16,
    tile: TileRef,
    use_connection_score: bool,
    reachable_stations: &[ReachableStation],
    min_range_sq: f64,
    station_range_sq: f64,
) -> f64 {
    let (border_spacing, structure_spacing) = spacing_constants(game);
    let mut w = game.map.magnitude(tile) as f64;
    if let Some(p) = game.player_by_small_id(small_id) {
        if !p.border_tiles.is_empty() {
            let (_, dist) = closest_tile(game, p.border_tiles.as_slice(), tile);
            if dist != u32::MAX {
                w += dist.min(border_spacing) as f64;
            } else {
                w += border_spacing as f64;
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
                w += d.min(structure_spacing) as f64;
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
                w += d.min(structure_spacing) as f64;
            }
        }
    }
    let connectivity_score = if use_connection_score {
        compute_connectivity_score(
            game,
            tile,
            reachable_stations,
            min_range_sq,
            station_range_sq,
        )
    } else {
        0.0
    };
    city_value_with_connectivity(
        w,
        connectivity_score,
        structure_spacing,
        use_connection_score,
    )
}

/// A station reachable for rail-connectivity scoring (TS `buildReachableStations` entry).
struct ReachableStation {
    tile: TileRef,
    cluster: Option<u32>,
    weight: f64,
}

/// TS `NationStructureBehavior.buildReachableStations`  -  own structures weighted by
/// "self" trade gold, non-bot non-embargoed neighbor structures weighted by team/ally/other
/// trade gold.
fn build_reachable_stations(game: &Game, small_id: u16) -> Vec<ReachableStation> {
    let rn = &game.rail_network;
    let max_trade_gold = (game.wire.train_gold(TrainRelation::Ally, 0) as f64).max(1.0);
    let mut result = Vec::new();

    let station_types = [unit_type::CITY, unit_type::PORT, unit_type::FACTORY];

    let self_weight = game.wire.train_gold(TrainRelation::SelfTrade, 0) as f64 / max_trade_gold;
    if let Some(p) = game.player_by_small_id(small_id) {
        for u in &p.units {
            if !station_types.contains(&u.unit_type.as_str()) {
                continue;
            }
            if let Some(station_id) = rn.find_station_by_unit(u.id) {
                let cluster = rn.stations.get(&station_id).and_then(|s| s.cluster);
                result.push(ReachableStation { tile: u.tile as TileRef, cluster, weight: self_weight });
            }
        }
    }

    for neighbor_id in crate::execution::ai_attack::nearby_player_small_ids(game, small_id) {
        let Some(neighbor) = game.player_by_small_id(neighbor_id) else { continue };
        if neighbor.player_type == crate::game::PlayerType::Bot {
            continue;
        }
        if !game.can_trade(small_id, neighbor_id) {
            continue;
        }
        let rel = if game.players_on_same_team(small_id, neighbor_id) {
            TrainRelation::Team
        } else if game.is_allied_with(small_id, neighbor_id) {
            TrainRelation::Ally
        } else {
            TrainRelation::Other
        };
        let weight = game.wire.train_gold(rel, 0) as f64 / max_trade_gold;
        for u in &neighbor.units {
            if !station_types.contains(&u.unit_type.as_str()) {
                continue;
            }
            if let Some(station_id) = rn.find_station_by_unit(u.id) {
                let cluster = rn.stations.get(&station_id).and_then(|s| s.cluster);
                result.push(ReachableStation { tile: u.tile as TileRef, cluster, weight });
            }
        }
    }

    result
}

/// TS `NationStructureBehavior.computeConnectivityScore`  -  per-cluster max weight of any
/// in-range station, plus individual weights of in-range isolated (clusterless) stations.
fn compute_connectivity_score(
    game: &Game,
    tile: TileRef,
    stations: &[ReachableStation],
    min_range_sq: f64,
    station_range_sq: f64,
) -> f64 {
    // TS iterates a `Map<Cluster, number>` in insertion order when summing; since float
    // addition isn't associative, an unordered `HashMap` here could sum in a different
    // order and produce a different last-bit result. Use an insertion-order Vec instead
    // (cluster counts per candidate tile are small, so linear lookup is cheap).
    let mut clusters_in_range: Vec<(u32, f64)> = Vec::new();
    let mut isolated_weight = 0.0;
    for st in stations {
        let dist = game.map.euclidean_dist_squared(tile, st.tile) as f64;
        if dist < min_range_sq || dist > station_range_sq {
            continue;
        }
        if let Some(c) = st.cluster {
            if let Some(entry) = clusters_in_range.iter_mut().find(|(cid, _)| *cid == c) {
                if st.weight > entry.1 {
                    entry.1 = st.weight;
                }
            } else {
                clusters_in_range.push((c, st.weight));
            }
        } else {
            isolated_weight += st.weight;
        }
    }
    let mut score = isolated_weight;
    for (_, v) in &clusters_in_range {
        score += v;
    }
    score
}

/// Context precomputed once per `structureSpawnTile` call (TS `factoryValue()` closure).
struct FactoryValueCtx {
    border_tiles: Vec<TileRef>,
    other_factory_tiles: Vec<TileRef>,
    city_tiles: Vec<TileRef>,
    border_spacing: u32,
    structure_spacing: u32,
    station_range: u32,
    station_range_sq: f64,
    use_connection_score: bool,
    reachable_stations: Vec<ReachableStation>,
    min_range_sq: f64,
}

fn build_factory_value_ctx(game: &Game, small_id: u16, use_connection_score: bool) -> FactoryValueCtx {
    let (border_spacing, structure_spacing) = spacing_constants(game);
    let station_range = game.wire.train_station_max_range();
    let station_range_sq = (station_range as f64).powi(2);
    let min_range_sq = (game.wire.train_station_min_range() as f64).powi(2);
    let (border_tiles, other_factory_tiles, city_tiles) = game
        .player_by_small_id(small_id)
        .map(|p| {
            (
                p.border_tiles.as_slice().to_vec(),
                p.units
                    .iter()
                    .filter(|u| u.unit_type == unit_type::FACTORY)
                    .map(|u| u.tile as TileRef)
                    .collect(),
                p.units
                    .iter()
                    .filter(|u| u.unit_type == unit_type::CITY)
                    .map(|u| u.tile as TileRef)
                    .collect(),
            )
        })
        .unwrap_or_default();
    let reachable_stations = if use_connection_score {
        build_reachable_stations(game, small_id)
    } else {
        Vec::new()
    };
    FactoryValueCtx {
        border_tiles,
        other_factory_tiles,
        city_tiles,
        border_spacing,
        structure_spacing,
        station_range,
        station_range_sq,
        use_connection_score,
        reachable_stations,
        min_range_sq,
    }
}

/// TS `NationStructureBehavior.factoryValue`.
fn factory_value(game: &Game, ctx: &FactoryValueCtx, tile: TileRef) -> f64 {
    let mut w = game.map.magnitude(tile) as f64;

    let (_, border_dist) = closest_tile(game, &ctx.border_tiles, tile);
    w += border_dist.min(ctx.border_spacing) as f64;

    let others: Vec<TileRef> = ctx
        .other_factory_tiles
        .iter()
        .copied()
        .filter(|&t| t != tile)
        .collect();
    if let Some((other, _)) = closest_two_tiles(game, &others, &[tile]) {
        let d = game.manhattan_dist(other, tile);
        // TS caps same-type factory spacing by `stationRange`, not `structureSpacing`.
        w += d.min(ctx.station_range) as f64;
    }

    if let Some((city, _)) = closest_two_tiles(game, &ctx.city_tiles, &[tile]) {
        let d = game.manhattan_dist(city, tile);
        w += d.min(ctx.structure_spacing) as f64;
    }

    if !ctx.use_connection_score {
        return w;
    }

    w += compute_connectivity_score(
        game,
        tile,
        &ctx.reachable_stations,
        ctx.min_range_sq,
        ctx.station_range_sq,
    ) * ctx.structure_spacing as f64;

    w
}

/// TS `NationStructureBehavior.missileSiloValue`.
fn missile_silo_value(
    game: &Game,
    border_tiles: &[TileRef],
    other_silo_tiles: &[TileRef],
    border_spacing: u32,
    structure_spacing: u32,
    tile: TileRef,
) -> f64 {
    let mut w = game.map.magnitude(tile) as f64;

    let (_, border_dist) = closest_tile(game, border_tiles, tile);
    w += border_dist.min(border_spacing) as f64;

    let others: Vec<TileRef> = other_silo_tiles.iter().copied().filter(|&t| t != tile).collect();
    if let Some((other, _)) = closest_two_tiles(game, &others, &[tile]) {
        let d = game.manhattan_dist(other, tile);
        w += d.min(structure_spacing) as f64;
    }

    w
}

/// Context precomputed once per `structureSpawnTile` call (TS `samLauncherValue()` closure).
struct SamValueCtx {
    border_tiles: Vec<TileRef>,
    other_sam_tiles: Vec<TileRef>,
    border_spacing: u32,
    structure_spacing: u32,
    difficulty_is_easy: bool,
    protect_entries: Vec<(TileRef, f64)>,
    range_sq: f64,
    use_coverage_weighting: bool,
    structure_coverage: Option<HashMap<TileRef, f64>>,
}

fn build_sam_value_ctx(game: &Game, random: &mut PseudoRandom, small_id: u16, difficulty: &str) -> SamValueCtx {
    let (border_spacing, structure_spacing) = spacing_constants(game);
    let weight_by_level = difficulty == "Hard" || difficulty == "Impossible";

    let mut protect_entries: Vec<(TileRef, f64)> = Vec::new();
    let mut existing_sams: Vec<(TileRef, i32)> = Vec::new();
    let border_tiles = game
        .player_by_small_id(small_id)
        .map(|p| p.border_tiles.as_slice().to_vec())
        .unwrap_or_default();
    if let Some(p) = game.player_by_small_id(small_id) {
        for u in &p.units {
            match u.unit_type.as_str() {
                unit_type::CITY | unit_type::FACTORY | unit_type::MISSILE_SILO | unit_type::PORT => {
                    let weight = if weight_by_level { u.level as f64 } else { 1.0 };
                    protect_entries.push((u.tile as TileRef, weight));
                }
                unit_type::SAM_LAUNCHER => existing_sams.push((u.tile as TileRef, u.level)),
                _ => {}
            }
        }
    }
    let other_sam_tiles: Vec<TileRef> = existing_sams.iter().map(|&(t, _)| t).collect();

    // TS `Config.defaultSamRange()`.
    const DEFAULT_SAM_RANGE: f64 = 70.0;
    let range_sq = DEFAULT_SAM_RANGE * DEFAULT_SAM_RANGE;

    let difficulty_is_easy = difficulty == "Easy";
    // TS short-circuits: `difficulty !== Easy && nextInt(0, 100) < 25` - no draw on Easy.
    let use_coverage_weighting = !difficulty_is_easy && random.next_int(0, 100) < 25;

    let structure_coverage = if use_coverage_weighting {
        let mut coverage = HashMap::new();
        for &(entry_tile, _weight) in &protect_entries {
            let mut score = 0.0;
            for &(sam_tile, sam_level) in &existing_sams {
                let sam_range = sam_range_for_level(sam_level);
                let dist = game.map.euclidean_dist_squared(entry_tile, sam_tile) as f64;
                if dist <= sam_range * sam_range {
                    score += sam_level as f64;
                }
            }
            coverage.insert(entry_tile, score);
        }
        Some(coverage)
    } else {
        None
    };

    SamValueCtx {
        border_tiles,
        other_sam_tiles,
        border_spacing,
        structure_spacing,
        difficulty_is_easy,
        protect_entries,
        range_sq,
        use_coverage_weighting,
        structure_coverage,
    }
}

/// TS `NationStructureBehavior.samLauncherValue`.
fn sam_launcher_value(game: &Game, ctx: &SamValueCtx, tile: TileRef) -> f64 {
    let mut w = game.map.magnitude(tile) as f64;

    let (_, border_dist) = closest_tile(game, &ctx.border_tiles, tile);
    w += border_dist.min(ctx.border_spacing) as f64;

    let others: Vec<TileRef> = ctx.other_sam_tiles.iter().copied().filter(|&t| t != tile).collect();
    if let Some((other, _)) = closest_two_tiles(game, &others, &[tile]) {
        let d = game.manhattan_dist(other, tile);
        w += d.min(ctx.structure_spacing) as f64;
    }

    if !ctx.difficulty_is_easy {
        for &(entry_tile, weight) in &ctx.protect_entries {
            let dist_sq = game.map.euclidean_dist_squared(tile, entry_tile) as f64;
            if dist_sq > ctx.range_sq {
                continue;
            }
            if ctx.use_coverage_weighting {
                let coverage = ctx
                    .structure_coverage
                    .as_ref()
                    .and_then(|m| m.get(&entry_tile))
                    .copied()
                    .unwrap_or(0.0);
                let coverage_weight = 1.0 / (1.0 + coverage);
                w += ctx.structure_spacing as f64 * weight * coverage_weight;
            } else {
                w += ctx.structure_spacing as f64 * weight;
            }
        }
    }

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
/// requirement); a lake is shared only if some other non-bot, alive player who can still
/// trade with this one (no mutual embargo) also touches it.
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
                if partners
                    .iter()
                    .any(|other| other != small_id && game.can_trade(*small_id, *other))
                {
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
    // TS `arraySampler`: if there are at most `n` candidates, return them all
    // (in border-tile order) with *no* RNG draws; otherwise sample `n` of them
    // without replacement via `PseudoRandom.randFromSet` (index draw against
    // the shrinking remaining pool, in original order).
    if tiles.len() <= n {
        return tiles;
    }
    let mut remaining = tiles;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let idx = random.next_int(0, remaining.len() as i32) as usize;
        out.push(remaining.remove(idx));
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
    let debug = std::env::var("STRUCTURE_DEBUG")
        .ok()
        .is_some_and(|id| game.player_by_small_id(small_id).is_some_and(|p| p.id == id));

    // TS `structureSpawnTileValue`  -  each branch builds its own closure/context. Only
    // City and Factory draw a `shouldUseConnectivityScore` random number; MissileSilo and
    // SAMLauncher (and SAMLauncher's own coverage-weighting draw) must not take that draw,
    // since RNG draw order/count must match TS exactly.
    enum ValueCtx {
        City {
            use_connection_score: bool,
            reachable_stations: Vec<ReachableStation>,
            min_range_sq: f64,
            station_range_sq: f64,
        },
        Factory(FactoryValueCtx),
        MissileSilo { border_tiles: Vec<TileRef>, other_tiles: Vec<TileRef>, border_spacing: u32, structure_spacing: u32 },
        Sam(SamValueCtx),
        None,
    }

    let ctx = match unit_type_name {
        unit_type::CITY => {
            let use_connection_score = should_use_connectivity_score(random, difficulty);
            let reachable_stations = if use_connection_score {
                build_reachable_stations(game, small_id)
            } else {
                Vec::new()
            };
            ValueCtx::City {
                use_connection_score,
                reachable_stations,
                min_range_sq: (game.wire.train_station_min_range() as f64).powi(2),
                station_range_sq: (game.wire.train_station_max_range() as f64).powi(2),
            }
        }
        unit_type::FACTORY => {
            let use_connection_score = should_use_connectivity_score(random, difficulty);
            ValueCtx::Factory(build_factory_value_ctx(game, small_id, use_connection_score))
        }
        unit_type::MISSILE_SILO => {
            let (border_spacing, structure_spacing) = spacing_constants(game);
            let (border_tiles, other_tiles) = game
                .player_by_small_id(small_id)
                .map(|p| {
                    (
                        p.border_tiles.as_slice().to_vec(),
                        p.units
                            .iter()
                            .filter(|u| u.unit_type == unit_type::MISSILE_SILO)
                            .map(|u| u.tile as TileRef)
                            .collect(),
                    )
                })
                .unwrap_or_default();
            ValueCtx::MissileSilo { border_tiles, other_tiles, border_spacing, structure_spacing }
        }
        unit_type::SAM_LAUNCHER => {
            ValueCtx::Sam(build_sam_value_ctx(game, random, small_id, difficulty))
        }
        _ => ValueCtx::None,
    };

    if debug {
        if let Some(bbox) = border_bbox(game, small_id) {
            eprintln!("  bbox min=({}, {}) max=({}, {})", bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y);
        }
        eprintln!(
            "structure_spawn {:?} tiles_in={}",
            game.player_by_small_id(small_id).map(|p| p.id.as_str()),
            tiles.len(),
        );
    }

    let mut best: Option<TileRef> = None;
    let mut best_value = 0f64;
    for t in tiles {
        let v = match &ctx {
            ValueCtx::City {
                use_connection_score,
                reachable_stations,
                min_range_sq,
                station_range_sq,
            } => city_value(
                game,
                small_id,
                t,
                *use_connection_score,
                reachable_stations,
                *min_range_sq,
                *station_range_sq,
            ),
            ValueCtx::Factory(fctx) => factory_value(game, fctx, t),
            ValueCtx::MissileSilo { border_tiles, other_tiles, border_spacing, structure_spacing } => {
                missile_silo_value(game, border_tiles, other_tiles, *border_spacing, *structure_spacing, t)
            }
            ValueCtx::Sam(sctx) => sam_launcher_value(game, sctx, t),
            ValueCtx::None => 0.0,
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
    // TS `getPerceivedCost`: City uses CITY_PERCEIVED_COST_INCREASE_PER_OWNED;
    // other types use `getStructureRatios(difficulty)[type].perceivedCostIncreasePerOwned`
    // (Port/Factory/MissileSilo: 1.0, SAMLauncher: 0.3, fallback: 0.1). Native previously
    // used 1.0 for every non-city type, so SAM perceived cost was `real*(1+owned)` instead
    // of `real*(1+0.3*owned)` — Mongolia @ tick 6129 couldn't "afford" a 3M SAM with
    // ~3.93M gold (native wanted 6M; TS wanted 3.9M).
    let increase = if structure_type == unit_type::CITY {
        CITY_PERCEIVED_COST_INCREASE_PER_OWNED
    } else {
        match structure_type {
            unit_type::PORT | unit_type::FACTORY | unit_type::MISSILE_SILO => 1.0,
            unit_type::SAM_LAUNCHER => 0.3,
            _ => 0.1,
        }
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
        true,
    )));
    true
}

/// TS `NationStructureBehavior.doHandleStructures`.
pub fn do_handle_structures(
    game: &mut Game,
    random: &mut PseudoRandom,
    small_id: u16,
    placements_count: u32,
) -> bool {
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

    // TS `doHandleStructures`  -  high-starting-gold Hard/Impossible nations build a
    // SAM launcher as their very first structure (bypasses city-count-based ordering).
    let difficulty = game.wire.game_config().difficulty.clone();
    if placements_count == 0
        && (difficulty == "Hard" || difficulty == "Impossible")
        && !game.wire.is_unit_disabled(unit_type::ATOM_BOMB)
        && missile_silos_enabled
        && !game.wire.is_unit_disabled(unit_type::SAM_LAUNCHER)
        && game.wire.starting_gold(crate::game::PlayerType::Nation) >= HIGH_STARTING_GOLD_THRESHOLD
        && maybe_spawn_structure(game, random, small_id, unit_type::SAM_LAUNCHER)
    {
        return true;
    }

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
    // TS `getAttackFrontTiles`: iterates `player.borderTiles()` (Set
    // insertion order) and pushes each border tile at most once, on the
    // *first* matching neighbor (`return` inside the neighbor loop). The
    // resulting `frontTiles` order feeds PRNG-driven `sampleTilesNearFront`
    // draws, so it must preserve border-tile insertion order, not a
    // numerically-sorted/deduped list.
    let mut front = Vec::new();
    game.for_each_border_tile(small_id, |border| {
        let mut matched = false;
        game.map.for_each_neighbor4(border, |n| {
            if matched {
                return;
            }
            let owner = game.map.owner_id(n);
            if owner != 0 && attacker_set.contains(&owner) {
                matched = true;
            }
        });
        if matched {
            front.push(border);
        }
    });
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

fn sample_front_coordinate(random: &mut PseudoRandom, center: i32, radius: i32) -> i32 {
    // TS passes `center + radius + 1` because `nextInt` excludes its upper
    // bound. Keep both edges of the intended interval reachable.
    random.next_int(center - radius, center + radius + 1)
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
        let x = sample_front_coordinate(random, ax, search_radius);
        let y = sample_front_coordinate(random, ay, search_radius);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rail::connect_station;

    #[test]
    fn defense_post_sampling_includes_positive_radius_edge() {
        let center = 100;
        let radius = 7;
        let mut random = PseudoRandom::new(0);
        let samples: Vec<_> = (0..10_000)
            .map(|_| sample_front_coordinate(&mut random, center, radius))
            .collect();

        assert!(samples.contains(&(center - radius)));
        assert!(samples.contains(&(center + radius)));
        assert!(samples
            .iter()
            .all(|sample| (center - radius..=center + radius).contains(sample)));
    }

    #[test]
    fn city_connectivity_weight_can_change_candidate_ranking() {
        let spacing = 100;
        let disconnected = city_value_with_connectivity(120.0, 0.0, spacing, true);
        let connected = city_value_with_connectivity(100.0, 0.5, spacing, true);

        assert!(connected > disconnected);
        assert_eq!(
            city_value_with_connectivity(100.0, 0.5, spacing, false),
            100.0
        );
    }

    // ── Ported from NationStructureBehavior.test.ts ─────────────────────────
    //
    // TS's tests mock the entire `Game`/`Player` surface and, for
    // `tryBuildDefensePost`'s cap-enforcement/successful-build/no-valid-tile
    // cases, `vi.spyOn` individual private methods (`getAttackFrontTiles`,
    // `countDefensePostsNearFront`, `sampleTilesNearFront`) to isolate
    // `tryBuildDefensePost`'s own control flow from their real
    // implementations. Native has no method-mocking equivalent (these are
    // free functions, not overridable virtual methods), so those specific
    // cases are ported here as real integration tests against a small
    // synthetic map instead: real attacker-adjacency geometry produces real
    // front tiles, real defense-post units produce a real
    // `countDefensePostsNearFront` result, etc. Same inputs/outputs as TS's
    // intent, not the literal mocked-call-count assertions.

    fn tiny_all_land_map(width: u32, height: u32) -> crate::map::GameMap {
        let n = (width * height) as usize;
        crate::map::GameMap::from_terrain_bytes(
            &crate::map::MapMeta {
                width,
                height,
                num_land_tiles: n as u32,
            },
            &vec![0x80u8; n],
        )
        .unwrap()
    }

    fn tiny_game(width: u32, height: u32, difficulty: &str) -> Game {
        let mut game = Game::default();
        game.map = tiny_all_land_map(width, height);
        game.wire = crate::core::config::Config::new(
            crate::core::schemas::GameConfig {
                game_map: "tiny".into(),
                difficulty: difficulty.into(),
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
                spawn_immunity_duration: None,
                starting_gold: None,
                gold_multiplier: None,
                max_timer_value: None,
                ranked_type: None,
            },
            false,
        );
        game.end_spawn_phase();
        game
    }

    fn add_player(game: &mut Game, id: &str, player_type: crate::game::PlayerType) -> u16 {
        add_player_team(game, id, player_type, None)
    }

    fn add_player_team(
        game: &mut Game,
        id: &str,
        player_type: crate::game::PlayerType,
        team: Option<&str>,
    ) -> u16 {
        game.add_from_info(&crate::game::PlayerInfo {
            name: id.into(),
            player_type,
            client_id: Some(id.into()),
            id: id.into(),
            clan_tag: None,
            friends: Vec::new(),
            team: team.map(|t| t.to_string()),
        })
    }

    // ── shouldUseConnectivityScore (exact per-difficulty cutoff) ────────────

    #[test]
    fn connectivity_score_chance_matches_ts_cutoffs() {
        assert_eq!(connectivity_score_chance("Easy"), 0);
        assert_eq!(connectivity_score_chance("Medium"), 60);
        assert_eq!(connectivity_score_chance("Hard"), 75);
        assert_eq!(connectivity_score_chance("Impossible"), 100);
    }

    // ── getAttackFrontTiles ──────────────────────────────────────────────────

    #[test]
    fn get_attack_front_tiles_empty_for_no_attackers() {
        let game = tiny_game(20, 20, "Hard");
        assert!(get_attack_front_tiles(&game, 1, &[]).is_empty());
    }

    #[test]
    fn get_attack_front_tiles_includes_border_tile_adjacent_to_attacker() {
        let mut game = tiny_game(20, 20, "Hard");
        let defender = add_player(&mut game, "defender", crate::game::PlayerType::Human);
        let attacker = add_player(&mut game, "attacker", crate::game::PlayerType::Human);
        let a = game.ref_xy(1, 1);
        let b = game.ref_xy(10, 10);
        game.conquer(defender, a);
        game.conquer(defender, b);
        game.conquer(attacker, game.ref_xy(2, 1));

        assert_eq!(get_attack_front_tiles(&game, defender, &[attacker]), vec![a]);
    }

    #[test]
    fn get_attack_front_tiles_excludes_tiles_not_adjacent_to_any_attacker() {
        let mut game = tiny_game(20, 20, "Hard");
        let defender = add_player(&mut game, "defender", crate::game::PlayerType::Human);
        let attacker = add_player(&mut game, "attacker", crate::game::PlayerType::Human);
        let a = game.ref_xy(1, 1);
        let b = game.ref_xy(10, 10);
        game.conquer(defender, a);
        game.conquer(defender, b);
        game.conquer(attacker, game.ref_xy(2, 1));

        let front = get_attack_front_tiles(&game, defender, &[attacker]);
        assert!(front.contains(&a));
        assert!(!front.contains(&b));
    }

    #[test]
    fn get_attack_front_tiles_handles_multiple_attackers_from_separate_attacks() {
        let mut game = tiny_game(20, 20, "Hard");
        let defender = add_player(&mut game, "defender", crate::game::PlayerType::Human);
        let atk1 = add_player(&mut game, "atk1", crate::game::PlayerType::Human);
        let atk2 = add_player(&mut game, "atk2", crate::game::PlayerType::Human);
        let a = game.ref_xy(1, 1);
        let b = game.ref_xy(5, 5);
        let c = game.ref_xy(15, 15);
        game.conquer(defender, a);
        game.conquer(defender, b);
        game.conquer(defender, c);
        game.conquer(atk1, game.ref_xy(2, 1));
        game.conquer(atk2, game.ref_xy(5, 6));

        let front = get_attack_front_tiles(&game, defender, &[atk1, atk2]);
        assert!(front.contains(&a));
        assert!(front.contains(&b));
        assert!(!front.contains(&c));
    }

    #[test]
    fn get_attack_front_tiles_does_not_duplicate_tile_with_multiple_attacker_neighbors() {
        let mut game = tiny_game(20, 20, "Hard");
        let defender = add_player(&mut game, "defender", crate::game::PlayerType::Human);
        let attacker = add_player(&mut game, "attacker", crate::game::PlayerType::Human);
        let a = game.ref_xy(5, 5);
        game.conquer(defender, a);
        game.conquer(attacker, game.ref_xy(6, 5));
        game.conquer(attacker, game.ref_xy(5, 6));

        assert_eq!(get_attack_front_tiles(&game, defender, &[attacker]), vec![a]);
    }

    // ── countDefensePostsNearFront ───────────────────────────────────────────

    #[test]
    fn count_defense_posts_zero_with_no_posts() {
        let mut game = tiny_game(300, 300, "Hard");
        let p = add_player(&mut game, "p", crate::game::PlayerType::Human);
        let front = vec![game.ref_xy(100, 100)];
        assert_eq!(count_defense_posts_near_front(&game, p, &front, usize::MAX), 0);
    }

    #[test]
    fn count_defense_posts_zero_with_no_front_tiles() {
        let mut game = tiny_game(300, 300, "Hard");
        let p = add_player(&mut game, "p", crate::game::PlayerType::Human);
        let t = game.ref_xy(10, 10);
        game.build_unit(p, unit_type::DEFENSE_POST, t);
        assert_eq!(count_defense_posts_near_front(&game, p, &[], usize::MAX), 0);
    }

    #[test]
    fn count_defense_posts_counts_posts_within_range() {
        let mut game = tiny_game(300, 300, "Hard");
        let p = add_player(&mut game, "p", crate::game::PlayerType::Human);
        let front = vec![game.ref_xy(100, 100)];
        let t1 = game.ref_xy(105, 100);
        let t2 = game.ref_xy(100, 110);
        game.build_unit(p, unit_type::DEFENSE_POST, t1);
        game.build_unit(p, unit_type::DEFENSE_POST, t2);
        assert_eq!(count_defense_posts_near_front(&game, p, &front, usize::MAX), 2);
    }

    #[test]
    fn count_defense_posts_excludes_posts_outside_range() {
        let mut game = tiny_game(300, 300, "Hard");
        let p = add_player(&mut game, "p", crate::game::PlayerType::Human);
        let front = vec![game.ref_xy(100, 100)];
        let t = game.ref_xy(250, 250);
        game.build_unit(p, unit_type::DEFENSE_POST, t);
        assert_eq!(count_defense_posts_near_front(&game, p, &front, usize::MAX), 0);
    }

    #[test]
    fn count_defense_posts_counts_a_post_only_once_for_multiple_front_tiles() {
        let mut game = tiny_game(300, 300, "Hard");
        let p = add_player(&mut game, "p", crate::game::PlayerType::Human);
        let front = vec![game.ref_xy(100, 100), game.ref_xy(102, 100)];
        let t = game.ref_xy(101, 100);
        game.build_unit(p, unit_type::DEFENSE_POST, t);
        assert_eq!(count_defense_posts_near_front(&game, p, &front, usize::MAX), 1);
    }

    #[test]
    fn count_defense_posts_sums_posts_near_different_front_sections() {
        let mut game = tiny_game(300, 300, "Hard");
        let p = add_player(&mut game, "p", crate::game::PlayerType::Human);
        let front = vec![game.ref_xy(50, 50), game.ref_xy(250, 250)];
        let t1 = game.ref_xy(52, 50);
        let t2 = game.ref_xy(250, 252);
        game.build_unit(p, unit_type::DEFENSE_POST, t1);
        game.build_unit(p, unit_type::DEFENSE_POST, t2);
        assert_eq!(count_defense_posts_near_front(&game, p, &front, usize::MAX), 2);
    }

    // ── tryBuildDefensePost — early-exit guards ──────────────────────────────
    //
    // Both players are `Nation` type so `AttackExecution::init`'s
    // `can_attack_player` immunity gate (which only applies when the
    // *attacker* is Human) never self-cancels these manually-added attacks.

    fn defense_test_setup(difficulty: &str) -> (Game, u16, u16) {
        let mut game = tiny_game(60, 60, difficulty);
        let defender = add_player(&mut game, "defender", crate::game::PlayerType::Nation);
        let attacker = add_player(&mut game, "attacker", crate::game::PlayerType::Nation);
        game.conquer(defender, game.ref_xy(30, 30));
        (game, defender, attacker)
    }

    fn add_attack(game: &mut Game, attacker: u16, defender: u16, troops: f64, source_tile: Option<TileRef>) {
        let defender_id = game.player_by_small_id(defender).unwrap().id.clone();
        game.add_land_attack_from(attacker, Some(defender_id), Some(troops), source_tile);
        game.execute_next_tick();
    }

    #[test]
    fn try_build_defense_post_false_on_easy_regardless_of_ratio() {
        let (mut game, defender, attacker) = defense_test_setup("Easy");
        game.add_troops(defender, 100.0);
        add_attack(&mut game, attacker, defender, 5000.0, None);
        let mut random = PseudoRandom::new(0);
        assert!(!try_build_defense_post(&mut game, &mut random, defender));
    }

    #[test]
    fn try_build_defense_post_false_with_no_incoming_attacks() {
        let (mut game, defender, _attacker) = defense_test_setup("Hard");
        game.add_troops(defender, 1000.0);
        let mut random = PseudoRandom::new(0);
        assert!(!try_build_defense_post(&mut game, &mut random, defender));
    }

    #[test]
    fn try_build_defense_post_false_when_only_boat_attacks_incoming() {
        let (mut game, defender, attacker) = defense_test_setup("Hard");
        game.add_troops(defender, 100.0);
        let source_tile = game.ref_xy(0, 0);
        add_attack(&mut game, attacker, defender, 5000.0, Some(source_tile));
        let mut random = PseudoRandom::new(0);
        assert!(!try_build_defense_post(&mut game, &mut random, defender));
    }

    #[test]
    fn try_build_defense_post_false_when_ratio_below_threshold() {
        let (mut game, defender, attacker) = defense_test_setup("Hard");
        game.add_troops(defender, 1000.0);
        add_attack(&mut game, attacker, defender, 349.0, None);
        let mut random = PseudoRandom::new(0);
        assert!(!try_build_defense_post(&mut game, &mut random, defender));
    }

    #[test]
    fn try_build_defense_post_false_when_own_troops_are_zero() {
        let (mut game, defender, attacker) = defense_test_setup("Hard");
        add_attack(&mut game, attacker, defender, 500.0, None);
        let mut random = PseudoRandom::new(0);
        assert!(!try_build_defense_post(&mut game, &mut random, defender));
    }

    // ── tryBuildDefensePost — real integration scenario ─────────────────────
    //
    // A large defender territory bordering a large attacker territory gives
    // `get_attack_front_tiles` a genuine multi-tile front and
    // `sample_tiles_near_front` genuine owned/buildable candidates nearby,
    // so the full pipeline (front → cap check → gold check → sampling →
    // `can_build_land_structure`) can be exercised end-to-end without
    // mocking any of it.

    fn defense_integration_setup(difficulty: &str, defender_troops: f64, attack_troops: f64) -> (Game, u16) {
        let (mut game, defender, attacker) = defense_test_setup(difficulty);
        // Overwrite the single conquered tile from defense_test_setup with a
        // real block of territory bordering the attacker's block.
        for x in 0..30 {
            for y in 0..60 {
                game.conquer(defender, game.ref_xy(x, y));
            }
        }
        for x in 30..40 {
            for y in 0..60 {
                game.conquer(attacker, game.ref_xy(x, y));
            }
        }
        if let Some(p) = game.player_by_small_id_mut(defender) {
            p.gold = 1_000_000_000;
        }
        // `add_attack` runs a real tick, and with ~1800 owned tiles the
        // population-growth income for that tick dwarfs any troop count set
        // beforehand. Set the exact troop count *after* ticking so the
        // attack-troops/defender-troops ratio the test cares about isn't
        // perturbed by incidental growth.
        add_attack(&mut game, attacker, defender, attack_troops, None);
        if let Some(p) = game.player_by_small_id_mut(defender) {
            p.troops = crate::util::to_int(defender_troops);
        }
        (game, defender)
    }

    #[test]
    fn try_build_defense_post_builds_when_all_conditions_are_met() {
        let (mut game, defender) = defense_integration_setup("Hard", 1000.0, 1000.0);
        let mut random = PseudoRandom::new(0);
        assert!(try_build_defense_post(&mut game, &mut random, defender));
        // `try_build_defense_post` only queues a `ConstructionExecution`;
        // new executions get `init()` on the next tick and only run `tick()`
        // (where the unit is actually built) the tick after that.
        game.execute_next_tick();
        game.execute_next_tick();
        assert_eq!(game.unit_count(defender, unit_type::DEFENSE_POST), 1);
    }

    #[test]
    fn try_build_defense_post_succeeds_when_ratio_is_exactly_at_threshold() {
        let (mut game, defender) = defense_integration_setup("Hard", 1000.0, 350.0);
        let mut random = PseudoRandom::new(0);
        assert!(try_build_defense_post(&mut game, &mut random, defender));
    }

    #[test]
    fn try_build_defense_post_sums_troops_across_multiple_land_attacks() {
        // TS's `defensePostNeeded` test for this sums two mocked
        // `incomingAttacks()` entries directly. Native inlines that ratio
        // logic into `try_build_defense_post` itself, so exercising it
        // faithfully needs two *real*, non-merging attacks incoming at once:
        // same-owner attacks to the same target merge into a single
        // executing attack (see `merge_outgoing_land_attacks`), so this uses
        // two distinct attacker players bordering the defender from
        // opposite sides instead of one attacker sending twice.
        let mut game = tiny_game(60, 60, "Hard");
        let defender = add_player(&mut game, "defender", crate::game::PlayerType::Nation);
        let attacker1 = add_player(&mut game, "attacker1", crate::game::PlayerType::Nation);
        let attacker2 = add_player(&mut game, "attacker2", crate::game::PlayerType::Nation);
        for x in 10..50 {
            for y in 0..60 {
                game.conquer(defender, game.ref_xy(x, y));
            }
        }
        for x in 0..10 {
            for y in 0..60 {
                game.conquer(attacker1, game.ref_xy(x, y));
            }
        }
        for x in 50..60 {
            for y in 0..60 {
                game.conquer(attacker2, game.ref_xy(x, y));
            }
        }
        if let Some(p) = game.player_by_small_id_mut(defender) {
            p.gold = 1_000_000_000;
        }
        // Neither attack alone reaches the 0.35 ratio; together they do.
        // Both attacks are queued before ticking so a single
        // `execute_next_tick` only runs their `init()` (which sets troops
        // to the requested start amount) without yet running their own
        // `tick()` (which would decay troops as border tiles get conquered).
        let defender_id = game.player_by_small_id(defender).unwrap().id.clone();
        game.add_land_attack_from(attacker1, Some(defender_id.clone()), Some(200.0), None);
        game.add_land_attack_from(attacker2, Some(defender_id), Some(200.0), None);
        game.execute_next_tick();
        if let Some(p) = game.player_by_small_id_mut(defender) {
            p.troops = 1000;
        }
        let mut random = PseudoRandom::new(0);
        assert!(try_build_defense_post(&mut game, &mut random, defender));
    }

    #[test]
    fn try_build_defense_post_false_when_gold_below_cost() {
        let (mut game, defender) = defense_integration_setup("Hard", 1000.0, 1000.0);
        if let Some(p) = game.player_by_small_id_mut(defender) {
            p.gold = 0;
        }
        let mut random = PseudoRandom::new(0);
        assert!(!try_build_defense_post(&mut game, &mut random, defender));
    }

    #[test]
    fn try_build_defense_post_false_once_cap_reached() {
        let (mut game, defender) = defense_integration_setup("Hard", 1000.0, 1000.0);
        // ratio = 1.0 → ceil(1.0 / 0.4) = 3 allowed. Pre-place 3 defense posts
        // right on the front line so the cap is already saturated.
        for i in 0..3 {
            let t = game.ref_xy(29, 10 + i * 5);
            game.build_unit(defender, unit_type::DEFENSE_POST, t);
        }
        let mut random = PseudoRandom::new(0);
        assert!(!try_build_defense_post(&mut game, &mut random, defender));
    }

    // ── sampleTilesNearFront ─────────────────────────────────────────────────

    #[test]
    fn sample_tiles_near_front_empty_when_no_front_tiles() {
        let game = tiny_game(10, 10, "Hard");
        let mut random = PseudoRandom::new(0);
        assert!(sample_tiles_near_front(&game, &mut random, 1, &[], 25).is_empty());
    }

    // ── buildReachableStations ───────────────────────────────────────────────

    fn adjacent_players(
        difficulty: &str,
        neighbor_type: crate::game::PlayerType,
        neighbor_team: Option<&str>,
    ) -> (Game, u16, u16) {
        let mut game = tiny_game(30, 30, difficulty);
        let defender = add_player(&mut game, "defender", crate::game::PlayerType::Human);
        let neighbor = add_player_team(&mut game, "neighbor", neighbor_type, neighbor_team);
        game.conquer(defender, game.ref_xy(10, 10));
        game.conquer(neighbor, game.ref_xy(11, 10));
        (game, defender, neighbor)
    }

    fn ally_weight(game: &Game) -> f64 {
        game.wire.train_gold(TrainRelation::Ally, 0) as f64
            / (game.wire.train_gold(TrainRelation::Ally, 0) as f64).max(1.0)
    }

    #[test]
    fn build_reachable_stations_includes_own_registered_units_with_self_weight() {
        let (mut game, defender, _neighbor) = adjacent_players("Medium", crate::game::PlayerType::Human, None);
        let tile = game.ref_xy(10, 10);
        let unit_id = game.build_unit(defender, unit_type::CITY, tile);
        connect_station(&mut game, defender, unit_id, unit_type::CITY);

        let result = build_reachable_stations(&game, defender);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].tile, tile);
        let expected = game.wire.train_gold(TrainRelation::SelfTrade, 0) as f64
            / game.wire.train_gold(TrainRelation::Ally, 0) as f64;
        assert!((result[0].weight - expected).abs() < 1e-9);
    }

    #[test]
    fn build_reachable_stations_excludes_own_unit_not_registered_as_station() {
        let (mut game, defender, _neighbor) = adjacent_players("Medium", crate::game::PlayerType::Human, None);
        let tile = game.ref_xy(10, 10);
        game.build_unit(defender, unit_type::CITY, tile);
        // No `connect_station` call - the city has no train station.
        assert!(build_reachable_stations(&game, defender).is_empty());
    }

    #[test]
    fn build_reachable_stations_excludes_bot_neighbors() {
        let (mut game, defender, neighbor) = adjacent_players("Medium", crate::game::PlayerType::Bot, None);
        let tile = game.ref_xy(11, 10);
        let unit_id = game.build_unit(neighbor, unit_type::CITY, tile);
        connect_station(&mut game, neighbor, unit_id, unit_type::CITY);

        assert!(build_reachable_stations(&game, defender).is_empty());
    }

    #[test]
    fn build_reachable_stations_excludes_non_player_terra_nullius_neighbor() {
        // Equivalent of TS's "non-player neighbor" mock: simply don't give
        // the defender any actual neighboring player at all - the adjacent
        // land stays unowned (TerraNullius), which `nearby_player_small_ids`
        // already excludes unconditionally.
        let mut game = tiny_game(30, 30, "Medium");
        let defender = add_player(&mut game, "defender", crate::game::PlayerType::Human);
        game.conquer(defender, game.ref_xy(10, 10));
        assert!(build_reachable_stations(&game, defender).is_empty());
    }

    #[test]
    fn build_reachable_stations_excludes_embargoed_neighbor() {
        let (mut game, defender, neighbor) = adjacent_players("Medium", crate::game::PlayerType::Human, None);
        let tile = game.ref_xy(11, 10);
        let unit_id = game.build_unit(neighbor, unit_type::CITY, tile);
        connect_station(&mut game, neighbor, unit_id, unit_type::CITY);
        game.add_embargo(defender, neighbor, false, 0);

        assert!(build_reachable_stations(&game, defender).is_empty());
    }

    #[test]
    fn build_reachable_stations_uses_other_weight_for_non_allied_neighbor() {
        let (mut game, defender, neighbor) = adjacent_players("Medium", crate::game::PlayerType::Human, None);
        let tile = game.ref_xy(11, 10);
        let unit_id = game.build_unit(neighbor, unit_type::CITY, tile);
        connect_station(&mut game, neighbor, unit_id, unit_type::CITY);

        let result = build_reachable_stations(&game, defender);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].tile, tile);
        let expected = game.wire.train_gold(TrainRelation::Other, 0) as f64
            / game.wire.train_gold(TrainRelation::Ally, 0) as f64;
        assert!((result[0].weight - expected).abs() < 1e-9);
    }

    #[test]
    fn build_reachable_stations_uses_ally_weight_for_allied_neighbor() {
        let (mut game, defender, neighbor) = adjacent_players("Medium", crate::game::PlayerType::Human, None);
        let tile = game.ref_xy(11, 10);
        let unit_id = game.build_unit(neighbor, unit_type::CITY, tile);
        connect_station(&mut game, neighbor, unit_id, unit_type::CITY);
        assert!(game.create_alliance_request(defender, neighbor, 0));
        game.accept_alliance_request(defender, neighbor, 1);
        assert!(game.is_allied_with(defender, neighbor));

        let result = build_reachable_stations(&game, defender);
        assert_eq!(result.len(), 1);
        assert!((result[0].weight - ally_weight(&game)).abs() < 1e-9);
    }

    #[test]
    fn build_reachable_stations_uses_team_weight_and_takes_precedence_over_ally() {
        let (mut game, defender, neighbor) =
            adjacent_players("Medium", crate::game::PlayerType::Human, Some("team-a"));
        if let Some(p) = game.player_by_small_id_mut(defender) {
            p.team = Some("team-a".to_string());
        }
        assert!(game.players_on_same_team(defender, neighbor));
        let tile = game.ref_xy(11, 10);
        let unit_id = game.build_unit(neighbor, unit_type::CITY, tile);
        connect_station(&mut game, neighbor, unit_id, unit_type::CITY);

        let result = build_reachable_stations(&game, defender);
        assert_eq!(result.len(), 1);
        let expected = game.wire.train_gold(TrainRelation::Team, 0) as f64
            / game.wire.train_gold(TrainRelation::Ally, 0) as f64;
        assert!((result[0].weight - expected).abs() < 1e-9);
    }

    #[test]
    fn build_reachable_stations_excludes_neighbor_unit_not_registered_as_station() {
        let (mut game, defender, neighbor) = adjacent_players("Medium", crate::game::PlayerType::Human, None);
        game.build_unit(neighbor, unit_type::CITY, game.ref_xy(11, 10));
        // No `connect_station` call.
        assert!(build_reachable_stations(&game, defender).is_empty());
    }

    #[test]
    fn build_reachable_stations_collects_own_and_neighbor_units_together() {
        let (mut game, defender, neighbor) = adjacent_players("Medium", crate::game::PlayerType::Human, None);
        let own_tile = game.ref_xy(10, 10);
        let own_unit = game.build_unit(defender, unit_type::CITY, own_tile);
        connect_station(&mut game, defender, own_unit, unit_type::CITY);
        let neighbor_tile = game.ref_xy(11, 10);
        let neighbor_unit = game.build_unit(neighbor, unit_type::CITY, neighbor_tile);
        connect_station(&mut game, neighbor, neighbor_unit, unit_type::CITY);

        let mut result = build_reachable_stations(&game, defender);
        result.sort_by_key(|r| r.tile);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].tile.min(result[1].tile), own_tile.min(neighbor_tile));
    }

    /// TS SAM `perceivedCostIncreasePerOwned` is 0.3, not 1.0. With gold under the
    /// nuke save-up target, one existing SAM makes a 3M launcher feel like 3.9M
    /// (affordable with ~3.93M) rather than 6M.
    #[test]
    fn sam_perceived_cost_uses_0_3_increase_per_owned_not_1_0() {
        let mut game = tiny_game(20, 20, "Easy");
        let sid = add_player(&mut game, "mongolia", crate::game::PlayerType::Nation);
        let tile = game.ref_xy(5, 5);
        game.conquer(sid, tile);
        game.build_unit(sid, unit_type::SAM_LAUNCHER, tile);
        if let Some(p) = game.player_by_small_id_mut(sid) {
            p.gold = 3_932_801;
            p.tiles_owned = 100_000; // keep alive / non-trivial
        }
        // Force under save-up so the multiplier path runs (MIRV+hydro = 30M).
        assert!(
            save_up_target(&game) > 3_932_801,
            "precondition: gold must be under save-up target"
        );
        let real = game.structure_cost(sid, unit_type::SAM_LAUNCHER);
        // With 1 SAM already, next is at the 3M cap for count>=1.
        assert_eq!(real, 3_000_000);
        let perceived = perceived_structure_cost(&game, sid, unit_type::SAM_LAUNCHER);
        assert_eq!(
            perceived, 3_900_000,
            "1 existing SAM => ceil(3M * 1.3) = 3.9M, not ceil(3M * 2) = 6M"
        );
        assert!(
            game.player_by_small_id(sid).unwrap().gold >= perceived,
            "Mongolia-at-6129 gold must clear the TS perceived SAM cost"
        );
    }
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

    let attacks = game.incoming_attacks(small_id, true);
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
            true,
        )));
        return true;
    }
    false
}

