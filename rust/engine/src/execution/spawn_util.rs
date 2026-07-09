//! Spawn tile selection (`execution/Util.ts::getSpawnTiles`).

use crate::execution::{ExecEnum, PlayerExecution};
use crate::bot::TribeExecution;
use crate::game::{Game, PlayerInfo, PlayerType};
use crate::map::{GameMap, TileRef};
use crate::prng::PseudoRandom;
use crate::water::BfsScratch;

struct Spawn {
    center: TileRef,
    tiles: Vec<TileRef>,
}

/// Run one spawn hop (shared by `SpawnExecution` and `TribeMassSpawn`).
pub fn execute_player_spawn(
    game: &mut Game,
    player_info: &PlayerInfo,
    tile: Option<TileRef>,
    random: &mut PseudoRandom,
) {
    let small_id = if game.has_player(&player_info.id) {
        game.player_by_id(&player_info.id).map(|p| p.small_id)
    } else {
        Some(game.add_from_info(player_info))
    };

    let Some(small_id) = small_id else {
        return;
    };

    if game.config.random_spawn && game.has_spawned(small_id) {
        return;
    }

    game.relinquish_player_tiles(small_id);

    let Some(spawn) = find_spawn(game, &player_info.id, tile, random) else {
        return;
    };

    game.conquer_spawn_tiles(small_id, &spawn.tiles);

    if !game.has_spawned(small_id) {
        game.add_execution(ExecEnum::Player(PlayerExecution::new(small_id)));
        if player_info.player_type == PlayerType::Bot {
            game.add_execution(ExecEnum::Tribe(TribeExecution::new(
                small_id,
                player_info.id.clone(),
            )));
        }
    }

    game.set_spawn_tile(small_id, spawn.center);
}

fn find_spawn(
    game: &mut Game,
    player_id: &str,
    center: Option<TileRef>,
    random: &mut PseudoRandom,
) -> Option<Spawn> {
    if let Some(center) = center {
        let tiles = get_spawn_tiles(&game.map, &mut game.bfs, center, false)?;
        if tiles.is_empty() {
            return None;
        }
        return Some(Spawn { center, tiles });
    }

    let spawn_area = game
        .player_by_id(player_id)
        .and_then(|p| p.team.as_deref())
        .and_then(|team| game.team_spawn_area(team).cloned());

    const MAX_SPAWN_TRIES: i32 = 1_000;
    let min_dist = game.wire.min_distance_between_players();
    let mut tries = 0;
    while tries < MAX_SPAWN_TRIES {
        tries += 1;
        let center = rand_tile(game, spawn_area.as_ref(), random);

        if !game.is_land(center) || game.has_owner(center) || game.is_border(center) {
            continue;
        }
        if game.too_close_to_existing_spawn(center, min_dist) {
            continue;
        }
        let Some(tiles) = get_spawn_tiles(&game.map, &mut game.bfs, center, true) else {
            continue;
        };
        return Some(Spawn { center, tiles });
    }
    None
}

fn rand_tile(game: &Game, area: Option<&crate::map::SpawnArea>, random: &mut PseudoRandom) -> TileRef {
    if let Some(area) = area {
        let x = random.next_int(area.x, area.x + area.width);
        let y = random.next_int(area.y, area.y + area.height);
        game.ref_xy(x as u32, y as u32)
    } else {
        let x = random.next_int(0, game.width() as i32);
        let y = random.next_int(0, game.height() as i32);
        game.ref_xy(x as u32, y as u32)
    }
}

pub fn get_spawn_tiles(
    map: &GameMap,
    scratch: &mut BfsScratch,
    center: TileRef,
    require_all_valid: bool,
) -> Option<Vec<TileRef>> {
    let dist2 = 16.0;
    let spawn_tiles = map.bfs_with_scratch(scratch, center, |gm, t| {
        gm.euclidean_dist_squared_center(center, t) <= dist2
    });

    // TS `getSpawnTiles`: invalid if owned or not land (no impassable check).
    let is_invalid = |t: TileRef| map.has_owner(t) || !map.is_land(t);

    if !require_all_valid {
        return Some(
            spawn_tiles
                .into_iter()
                .filter(|&t| !is_invalid(t))
                .collect(),
        );
    }

    if spawn_tiles.iter().any(|&t| is_invalid(t)) {
        None
    } else {
        Some(spawn_tiles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_set_tick50_centers() {
        let repo = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai".into());
        let dir = std::path::Path::new(&repo).join("openfront/resources/maps/twolakes");
        let map = crate::map::GameMap::load_map_dir(&dir).unwrap().1;
        let mut scratch = crate::water::BfsScratch::new((map.width * map.height) as usize);
        for center in [464606u32, 2659313u32] {
            let tiles = get_spawn_tiles(&map, &mut scratch, center, false).unwrap();
            eprintln!("center {} count {}", center, tiles.len());
        }
    }

    #[test]
    fn spawn_set_matches_python_reference_for_1830957() {
        let repo = std::env::var("OPENFRONT_REPO")
            .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai".into());
        let dir = std::path::Path::new(&repo)
            .join("openfront/resources/maps/twolakes");
        let map = GameMap::load_map_dir(&dir).unwrap().1;
        let mut scratch = crate::water::BfsScratch::new((map.width * map.height) as usize);
        let center = 1830957u32;
        let tiles = get_spawn_tiles(&map, &mut scratch, center, true).unwrap();
        let mut sorted: Vec<u32> = tiles.iter().copied().collect();
        sorted.sort_unstable();
        // Reference from python cardinal BFS+pop on empty map
        let expected_min = 1822555u32;
        let expected_max = 1837258u32;
        assert_eq!(sorted.len(), 52);
        assert_eq!(sorted[0], expected_min);
        assert_eq!(sorted[51], expected_max);
    }
}
