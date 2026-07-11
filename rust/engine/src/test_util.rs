//! Shared helpers for constructing a real, multi-tile `Game` in unit tests.
//!
//! `Game::default()` only provides a 1x1 map (see `game.rs`), which is too
//! small for the neighbor-order, border-invariant, and terrain-geometry
//! tests ported from the TS Jest suite (`openfront/tests/*.test.ts`) - those
//! need actual width/height and, in a few cases, a real impassable-terrain
//! wall. This module is test-only (`#![cfg(test)]`) and not part of the
//! production build.

#![cfg(test)]

use crate::game::Game;
use crate::map::{GameMap, MapMeta};

const LAND_PLAINS: u8 = 0b1000_0000; // isLand=1, magnitude=0 (Plains)
const IMPASSABLE: u8 = 0b1001_1111; // isLand=1, magnitude=31 (Impassable)

/// All-plains, all-land `width`x`height` map, wrapped in a `Game` with
/// correctly sized BFS/water-pathing scratch buffers (unlike swapping
/// `Game::default()`'s 1x1 `map` field in place, which would leave those
/// scratch buffers undersized for the new tile count).
pub fn plains_game(width: u32, height: u32) -> Game {
    walled_game(width, height, None)
}

/// Same as `plains_game`, but with a vertical impassable wall `wall_width`
/// tiles wide starting at column `wall_x` - mirrors TS
/// `ImpassableTerrain.test.ts`'s `buildTerrain` helper.
pub fn walled_game(width: u32, height: u32, wall: Option<(u32, u32)>) -> Game {
    let n = (width * height) as usize;
    let mut data = vec![LAND_PLAINS; n];
    let mut num_land_tiles = n as u32;
    if let Some((wall_x, wall_width)) = wall {
        for y in 0..height {
            for x in wall_x..wall_x + wall_width {
                let idx = (y * width + x) as usize;
                data[idx] = IMPASSABLE;
                num_land_tiles -= 1;
            }
        }
    }
    let meta = MapMeta {
        width,
        height,
        num_land_tiles,
    };
    let map = GameMap::from_terrain_bytes(&meta, &data).unwrap();
    let tile_count = n;
    let mut game = Game::default();
    game.map = map.clone();
    game.mini_map = map;
    game.bfs = crate::water::BfsScratch::new(tile_count);
    game.water_astar = crate::water::WaterAstarScratch::new(tile_count);
    game.mini_water_astar = crate::water::WaterAstarScratch::new(tile_count);
    game.end_spawn_phase();
    game
}
