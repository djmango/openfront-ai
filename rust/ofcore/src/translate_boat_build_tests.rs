//! Translator-level probes for boat/build "misclick" behavior.

use crate::feat::{EntsData, IS_LAND_BIT, REGION};
use crate::translate::IntentTranslator;

const SHORELINE_BIT: u8 = 6;

fn empty_ents() -> EntsData {
    EntsData {
        players: Vec::new(),
        units: Vec::new(),
        attacks: Vec::new(),
        alliances: Vec::new(),
        doomsday_enabled: false,
    }
}

fn make_terrain(hr: usize, wr: usize, width: usize, paint: impl Fn(usize, usize) -> u8) -> Vec<u8> {
    let mut terrain = vec![0u8; width * (hr + 8)];
    for y in 0..hr {
        for x in 0..wr {
            terrain[y * width + x] = paint(y, x);
        }
    }
    terrain
}

fn land(shore: bool) -> u8 {
    let mut t = 1 << IS_LAND_BIT;
    if shore {
        t |= 1 << SHORELINE_BIT;
    }
    t
}

#[test]
fn boat_tile_returns_none_for_pure_ocean_region() {
    // 3x3 regions: center region is pure water; land only in region (0,0).
    let hr = REGION * 3;
    let wr = REGION * 3;
    let width = wr + 4;
    let terrain = make_terrain(hr, wr, width, |y, x| {
        if y < REGION && x < REGION {
            land(true)
        } else {
            0 // water
        }
    });
    let mut tr = IntentTranslator::new(&terrain, width, hr, wr);
    let owners = vec![0i64; hr * wr];
    let ents = empty_ents();
    // Coarse region index uses GW_MAX stride in region_tile.
    let ocean_region = 1i64 * crate::feat::GW_MAX + 1; // (gy=1,gx=1)
    assert!(
        tr.boat_tile(ocean_region, &owners, 1, &ents).is_none(),
        "pure-ocean coarse cell must not invent a boat destination"
    );
}

#[test]
fn boat_tile_can_pick_inland_non_shore_fallback() {
    // Region with only inland enemy land (no shore bit) still returns a tile
    // via the plain_valid fallback — engine may still reject it later.
    let hr = REGION * 2;
    let wr = REGION * 2;
    let width = wr;
    let terrain = make_terrain(hr, wr, width, |_y, _x| land(false));
    let mut tr = IntentTranslator::new(&terrain, width, hr, wr);
    let mut owners = vec![2i64; hr * wr];
    // Own a strip so "me" exists conceptually; destinations are enemy.
    for i in 0..REGION {
        owners[i] = 1;
    }
    let ents = empty_ents();
    let region = 0i64;
    let tile = tr
        .boat_tile(region, &owners, 1, &ents)
        .expect("inland enemy land should pass translator boat_tile");
    let y = (tile as usize) / width;
    let x = (tile as usize) % width;
    assert_eq!(owners[y * wr + x], 2);
    assert_eq!(tr.shore[y * wr + x], 0, "picked inland (no shore bit)");
}

#[test]
fn build_tile_warship_picks_any_water_without_port_check() {
    let hr = REGION;
    let wr = REGION;
    let width = wr;
    let terrain = make_terrain(hr, wr, width, |y, x| {
        if y == 0 && x == 0 {
            land(true)
        } else {
            0
        }
    });
    let mut tr = IntentTranslator::new(&terrain, width, hr, wr);
    let owners = vec![1i64; hr * wr];
    let tile = tr
        .build_tile(0, &owners, 1, "Warship")
        .expect("warship translator only requires water in region");
    let y = (tile as usize) / width;
    let x = (tile as usize) % width;
    assert_eq!(tr.land[y * wr + x], 0);
}

#[test]
fn build_tile_city_requires_own_passable_land() {
    let hr = REGION;
    let wr = REGION;
    let width = wr;
    let terrain = make_terrain(hr, wr, width, |_y, _x| land(false));
    let mut tr = IntentTranslator::new(&terrain, width, hr, wr);
    let owners = vec![2i64; hr * wr]; // enemy-owned
    assert!(
        tr.build_tile(0, &owners, 1, "City").is_none(),
        "City must not snap onto enemy land"
    );
    let owners = vec![1i64; hr * wr];
    assert!(tr.build_tile(0, &owners, 1, "City").is_some());
}
