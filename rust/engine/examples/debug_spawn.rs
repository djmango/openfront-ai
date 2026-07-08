//! Simulate tribe w3er03ee spawn RNG after native tribes 0-43.
use openfront_engine::bootstrap::game_from_record;
use openfront_engine::execution::spawn_util::get_spawn_tiles;
use openfront_engine::prng::PseudoRandom;
use openfront_engine::record::GameRecord;
use openfront_engine::util::simple_hash;

fn main() {
    let repo = std::env::var("OPENFRONT_REPO")
        .unwrap_or_else(|_| "/Users/djmango/github/openfront-ai".into());
    let path = std::path::Path::new(&repo).join("records/0c4c7d7993c9/jby2gMJF.json.gz");
    let bytes = std::fs::read(&path).unwrap();
    let mut dec = flate2::read::GzDecoder::new(&bytes[..]);
    let mut json = Vec::new();
    std::io::Read::read_to_end(&mut dec, &mut json).unwrap();
    let rec = GameRecord::from_json_bytes(&json).unwrap().decompress();
    let mut game = game_from_record(std::path::Path::new(&repo), &rec).unwrap();
    game.execute_next_tick();

    let centers: Vec<u32> =
        serde_json::from_str(&std::fs::read_to_string("/tmp/native_centers43.json").unwrap())
            .unwrap();

    for (i, &center) in centers.iter().enumerate() {
        let sid = 130u16 + i as u16;
        if let Some(tiles) = get_spawn_tiles(&game.map, center, false) {
            for t in tiles {
                game.conquer(sid, t);
            }
            game.set_spawn_tile(sid, center);
        }
    }

    let mut rng = PseudoRandom::new(simple_hash("w3er03ee") + simple_hash("jby2gMJF"));
    let min_dist = game.wire.min_distance_between_players();
    let mut successes = 0;
    for try_n in 0..1000 {
        let x = rng.next_int(0, game.width() as i32);
        let y = rng.next_int(0, game.height() as i32);
        let center = game.ref_xy(x as u32, y as u32);
        if !game.is_land(center) || game.has_owner(center) || game.is_border(center) {
            continue;
        }
        let too_close = game.all_players().iter().any(|other| {
            let Some(st) = other.spawn_tile else {
                return false;
            };
            game.manhattan_dist(st, center) < min_dist
        });
        if too_close {
            continue;
        }
        if get_spawn_tiles(&game.map, center, true).is_some() {
            successes += 1;
            if center == 631823 {
                println!("found 631823 at try {try_n}");
            }
            if successes <= 3 {
                println!("success {successes} try {try_n} center {center}");
            }
        }
    }
    println!("total successes in 1000 tries: {successes}");
}
