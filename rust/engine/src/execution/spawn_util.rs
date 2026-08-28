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
/// Returns whether a spawn was actually placed (TS `SpawnExecution.tick`
/// returns early without ending the spawn phase when `getSpawnTiles`
/// fails, so callers need to know).
pub fn execute_player_spawn(
    game: &mut Game,
    player_info: &PlayerInfo,
    tile: Option<TileRef>,
    random: &mut PseudoRandom,
) -> bool {
    let small_id = if game.has_player(&player_info.id) {
        game.player_by_id(&player_info.id).map(|p| p.small_id)
    } else {
        Some(game.add_from_info(player_info))
    };

    let Some(small_id) = small_id else {
        return false;
    };

    // Already placed: no-op, do not relinquish + re-conquer (teleport).
    // TS only gates this after the spawn phase ends; during the phase a
    // second spawn used to hop the blob. Duo sequential spawn needs the
    // placed partner to stay put so the other head can match landmass.
    if game.has_spawned(small_id) {
        return false;
    }

    game.relinquish_player_tiles(small_id);

    let Some(spawn) = find_spawn(game, &player_info.id, tile, random) else {
        return false;
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
    true
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
        let current_small_id = game.player_by_id(player_id).map(|p| p.small_id);
        if game.too_close_to_existing_spawn(center, min_dist, current_small_id) {
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

    // Tip TS `getSpawnTiles`: invalid if owned or not land (impassable removed).
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
            .unwrap_or_else(|_| crate::util::default_repo_root());
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
            .unwrap_or_else(|_| crate::util::default_repo_root());
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

    #[test]
    fn spawn_tiles_accept_mag31_mountain_on_tip() {
        let game = crate::test_util::walled_game(20, 20, Some((10, 1)));
        let mut scratch =
            crate::water::BfsScratch::new((game.map.width * game.map.height) as usize);
        let center_next_to_wall = game.map.ref_xy(8, 10);
        let wall_tile = game.map.ref_xy(10, 10);
        assert!(!game.map.is_impassable(wall_tile));

        // Tip getSpawnTiles: mag31 is ordinary land, so a strict footprint
        // that reaches the wall is accepted.
        assert!(
            get_spawn_tiles(&game.map, &mut scratch, center_next_to_wall, true).is_some(),
            "strict spawn footprints may include tip mag31 Mountain"
        );

        let loose_tiles =
            get_spawn_tiles(&game.map, &mut scratch, center_next_to_wall, false).unwrap();
        assert!(
            loose_tiles.contains(&wall_tile),
            "loose spawn footprints keep tip mag31 Mountain tiles"
        );
    }

    #[test]
    fn second_spawn_does_not_teleport_an_already_placed_player() {
        let mut game = crate::test_util::plains_game(40, 40);
        let info = crate::game::PlayerInfo {
            name: "p".to_string(),
            player_type: crate::game::PlayerType::Human,
            client_id: Some("p".to_string()),
            id: "p".to_string(),
            clan_tag: None,
            friends: Vec::new(),
            team: None,
        };
        let mut random = crate::prng::PseudoRandom::new(1);
        let first = game.map.ref_xy(10, 10);
        let second = game.map.ref_xy(30, 30);
        assert!(execute_player_spawn(
            &mut game, &info, Some(first), &mut random
        ));
        let small_id = game.player_by_id("p").unwrap().small_id;
        assert!(game.has_spawned(small_id));
        assert_eq!(game.spawn_tile_of(small_id), Some(first));
        assert!(!execute_player_spawn(
            &mut game, &info, Some(second), &mut random
        ));
        assert_eq!(game.spawn_tile_of(small_id), Some(first));
        assert_eq!(game.map.owner_id(first), small_id);
        assert_eq!(game.map.owner_id(second), 0);
    }
}
