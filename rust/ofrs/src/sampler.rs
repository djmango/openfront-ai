//! Full BC sampler over cache-bc games: the Rust twin of rl/bc_data.py
//! BCSampler. One `sample_batch(n)` call decodes frames/entities/steps,
//! featurizes (feat.rs), and builds label targets for n samples in parallel
//! (rayon), releasing the GIL for everything except the final numpy dict
//! construction. Python keeps a fallback path; parity is enforced by
//! scripts/test_ofrs_parity.py via `debug_step_samples`.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use md5::{Digest, Md5};
use memmap2::Mmap;
use numpy::{PyArray1, PyArray3, PyArrayMethods};
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::PyDict;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde_json::Value;

use crate::feat::{
    self, action_index, featurize, needs_player, needs_quantity, needs_tile, num,
    parse_ents, parse_legal, placement_bucket, quantity_bucket, Feat, Legal, GW_MAX,
    IS_LAND_BIT, MAG_MASK, REGION,
};

const NOOP_CHOICE: [i64; 6] = [feat::A_NOOP, -1, -1, -1, -1, -1];

fn err<E: std::fmt::Display>(what: &str) -> impl Fn(E) -> PyErr + '_ {
    move |e| PyValueError::new_err(format!("{what}: {e}"))
}

/// Minimal .npy reader for the uint8 arrays prefeaturize_bc.py writes.
fn read_npy_u8(path: &Path) -> PyResult<(Vec<u8>, Vec<usize>)> {
    let raw = fs::read(path).map_err(err("npy read"))?;
    if raw.len() < 10 || &raw[..6] != b"\x93NUMPY" {
        return Err(PyValueError::new_err(format!("{path:?}: not an npy")));
    }
    let (hlen, off) = if raw[6] == 1 {
        (u16::from_le_bytes([raw[8], raw[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&raw[off..off + hlen]).map_err(err("npy header"))?;
    if !header.contains("'|u1'") || header.contains("'fortran_order': True") {
        return Err(PyValueError::new_err(format!("{path:?}: expected C-order |u1")));
    }
    let inner = header
        .split('(')
        .nth(1)
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| PyValueError::new_err("npy shape"))?;
    let shape: Vec<usize> = inner
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    Ok((raw[off + hlen..].to_vec(), shape))
}

struct SpawnLabel {
    x: i64,
    y: i64,
    me: i64,
}

struct SpawnStep {
    tick: i64,
    labels: Vec<(String, SpawnLabel)>,
}

pub struct Game {
    // Dir name: the (game, tick) identity for the Python-side AE-latent
    // cache (rl.obs.ZCache); matches CachedGame.path.name.
    name: String,
    ds: i64,
    hr: usize,
    wr: usize,
    gh: usize,
    gw: usize,
    tick_row: HashMap<i64, usize>,
    frame_off: Vec<usize>,
    ent_off: Vec<usize>,
    step_off: Vec<usize>,
    frames: Mmap,
    ents: Mmap,
    steps: Mmap,
    land: Vec<u8>,
    mag: Vec<u8>,
    lut: Vec<u8>,
    placements: HashMap<String, f64>,
    spawn_steps: Vec<SpawnStep>,
    n_spawn: usize,
    // Shared per-game (2, hr, wr) float32, like ObsBuilder._terrain_static:
    // every sample of this game returns the SAME numpy object.
    terrain_static: GILOnceCell<Py<PyArray3<f32>>>,
}

fn mmap(path: &Path) -> PyResult<Mmap> {
    let f = fs::File::open(path).map_err(err("open"))?;
    unsafe { Mmap::map(&f) }.map_err(err("mmap"))
}

fn offsets(v: &Value) -> Vec<usize> {
    v.as_array()
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect())
        .unwrap_or_default()
}

impl Game {
    fn load(dir: &Path) -> PyResult<Game> {
        let cache = dir.join("cache-bc");
        let idx: Value =
            serde_json::from_slice(&fs::read(cache.join("index.json")).map_err(err("index"))?)
                .map_err(err("index json"))?;
        let (hr, wr) = (
            idx["hr"].as_u64().unwrap_or(0) as usize,
            idx["wr"].as_u64().unwrap_or(0) as usize,
        );
        let (terr, tshape) = read_npy_u8(&cache.join("terrain.npy"))?;
        if tshape != [hr, wr] {
            return Err(PyValueError::new_err(format!("{dir:?}: terrain shape")));
        }
        let (lut, _) = read_npy_u8(&cache.join("lut.npy"))?;
        let land: Vec<u8> = terr.iter().map(|&t| (t >> IS_LAND_BIT) & 1).collect();
        let mag: Vec<u8> = terr.iter().map(|&t| t & MAG_MASK).collect();

        let tick_row = idx["ticks"]
            .as_array()
            .map(|a| {
                a.iter()
                    .enumerate()
                    .filter_map(|(i, t)| t.as_i64().map(|t| (t, i)))
                    .collect()
            })
            .unwrap_or_default();
        let placements = idx["placements"]
            .as_object()
            .map(|o| {
                o.iter()
                    .map(|(k, v)| (k.clone(), num(&v["placement"])))
                    .collect()
            })
            .unwrap_or_default();
        let spawn_steps = idx["spawn_steps"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|s| {
                        Some(SpawnStep {
                            tick: s["tick"].as_i64()?,
                            labels: s["labels"]
                                .as_object()?
                                .iter()
                                .filter_map(|(cid, l)| {
                                    Some((
                                        cid.clone(),
                                        SpawnLabel {
                                            x: l["x"].as_i64()?,
                                            y: l["y"].as_i64()?,
                                            me: l["me"].as_i64().unwrap_or(-1),
                                        },
                                    ))
                                })
                                .collect(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(Game {
            name: dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("")
                .to_string(),
            ds: idx["ds"].as_i64().unwrap_or(1),
            hr,
            wr,
            gh: hr / REGION,
            gw: wr / REGION,
            tick_row,
            frame_off: offsets(&idx["frame_offsets"]),
            ent_off: offsets(&idx["ent_offsets"]),
            step_off: offsets(&idx["step_offsets"]),
            frames: mmap(&cache.join("frames.zst"))?,
            ents: mmap(&cache.join("ents.zst"))?,
            steps: mmap(&cache.join("steps.zst"))?,
            land,
            mag,
            lut,
            placements,
            spawn_steps,
            n_spawn: idx["n_spawn"].as_u64().unwrap_or(0) as usize,
            terrain_static: GILOnceCell::new(),
        })
    }

    fn n_steps(&self) -> usize {
        self.step_off.len().saturating_sub(1)
    }

    fn step(&self, i: usize) -> Result<Value, String> {
        let blob = &self.steps[self.step_off[i]..self.step_off[i + 1]];
        let raw = zstd::bulk::decompress(blob, 1 << 26).map_err(|e| format!("step zstd: {e}"))?;
        serde_json::from_slice(&raw).map_err(|e| format!("step json: {e}"))
    }

    fn entities(&self, tick: i64) -> Result<Value, String> {
        let i = *self.tick_row.get(&tick).ok_or("tick not in cache")?;
        let blob = &self.ents[self.ent_off[i]..self.ent_off[i + 1]];
        let raw = zstd::bulk::decompress(blob, 1 << 27).map_err(|e| format!("ent zstd: {e}"))?;
        serde_json::from_slice(&raw).map_err(|e| format!("ent json: {e}"))
    }

    fn frame(&self, tick: i64) -> Result<(Vec<u8>, Vec<u8>), String> {
        let i = *self.tick_row.get(&tick).ok_or("tick not in cache")?;
        let blob = &self.frames[self.frame_off[i]..self.frame_off[i + 1]];
        let hw = self.hr * self.wr;
        let total = hw + self.hr * (self.wr / 8);
        let raw = zstd::bulk::decompress(blob, total).map_err(|e| format!("frame zstd: {e}"))?;
        if raw.len() != total {
            return Err(format!("frame size {} != {total}", raw.len()));
        }
        Ok((raw[..hw].to_vec(), raw[hw..].to_vec()))
    }

    fn placement(&self, cid: &str) -> f64 {
        self.placements.get(cid).copied().unwrap_or(0.5)
    }
}

/// One fully-built sample, GIL-free; turned into the Python raw dict at the
/// very end.
pub struct Built {
    game: usize,
    tick: i64,
    owners: Vec<u8>,
    fallout: Vec<u8>,
    feat: Feat,
    choice: [i64; 6],
    cond: i64,
}

fn label_to_choice(
    label: &Value,
    game: &Game,
    ents_players: &[(usize, f64, f64)], // (id, troops, gold)
    client_smallid: i64,
) -> Option<[i64; 6]> {
    let a = action_index(label["a"].as_str()?)?;
    let mut choice = [a, -1, -1, -1, -1, -1];
    if needs_player(a) {
        let t = label["t"].as_u64()? as usize;
        let slot = *game.lut.get(t)? as i64;
        if slot <= 0 {
            return None;
        }
        choice[1] = slot;
    }
    if needs_tile(a) {
        let gx = (label["x"].as_i64()? / game.ds) / REGION as i64;
        let gy = (label["y"].as_i64()? / game.ds) / REGION as i64;
        if !(0 <= gy && gy < game.gh as i64 && 0 <= gx && gx < game.gw as i64) {
            return None;
        }
        choice[2] = gy * GW_MAX + gx;
    }
    if a == feat::A_BUILD {
        choice[3] = feat::build_index(label["unit"].as_str()?)?;
    }
    if a == feat::A_LAUNCH_NUKE {
        choice[4] = feat::nuke_index(label["unit"].as_str()?)?;
    }
    if needs_quantity(a) {
        let me = ents_players
            .iter()
            .find(|(id, _, _)| *id as i64 == client_smallid);
        let avail = match me {
            Some((_, troops, gold)) => {
                if a == feat::A_DONATE_GOLD {
                    *gold
                } else {
                    *troops
                }
            }
            None => 0.0,
        };
        let amt = label.get("amt").and_then(|v| {
            if v.is_null() {
                None
            } else {
                Some(num(v))
            }
        });
        choice[5] = quantity_bucket(amt, avail);
    }
    Some(choice)
}

/// Force the labeled option into the legality masks (the human did it, so it
/// was legal; our reconstruction can be narrower). Mirrors bc_data.py.
fn force_legal(feat: &mut Feat, choice: &[i64; 6]) {
    feat.legal_actions[choice[0] as usize] = 1.0;
    if choice[1] >= 0 {
        feat.legal_ptarget[choice[0] as usize * feat::MAX_SLOTS + choice[1] as usize] = 1.0;
    }
    if choice[3] >= 0 {
        feat.legal_build[choice[3] as usize] = 1.0;
    }
    if choice[4] >= 0 {
        feat.legal_nuke[choice[4] as usize] = 1.0;
    }
}

fn featurize_for(
    game: &Game,
    tick: i64,
    spawn: bool,
    me: i64,
    ents_v: &Value,
    legal: &Legal,
) -> Result<(Vec<u8>, Vec<u8>, Feat), String> {
    let (owners, fallout) = game.frame(tick)?;
    let ents = parse_ents(ents_v);
    let f = featurize(
        game.gh, game.gw, &game.lut, &game.land, &game.mag, &owners, tick, spawn, !spawn, me,
        &ents, legal,
    );
    Ok((owners, fallout, f))
}

/// Players (id, troops, gold) extracted once per step for quantity buckets.
fn player_amounts(ents_v: &Value) -> Vec<(usize, f64, f64)> {
    ents_v["players"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|p| {
                    Some((
                        p["id"].as_u64()? as usize,
                        num(&p["troops"]),
                        num(&p["gold"]),
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

fn step_samples(
    games: &[Game],
    gi: usize,
    step: &Value,
    noop_frac: f64,
    rng: &mut SmallRng,
    deterministic: bool,
) -> Result<Vec<Built>, String> {
    let game = &games[gi];
    let tick = step["tick"].as_i64().ok_or("step tick")?;
    let ents_v = game.entities(tick)?;
    let amounts = player_amounts(&ents_v);

    let labels = step["labels"].as_object().ok_or("labels")?;
    let legal_map = step["legal"].as_object().ok_or("legal")?;
    let actors: Vec<&String> = labels
        .iter()
        .filter(|(c, ls)| ls.as_array().is_some_and(|a| !a.is_empty()) && legal_map.contains_key(*c))
        .map(|(c, _)| c)
        .collect();
    let noops: Vec<&String> = legal_map
        .keys()
        .filter(|c| !labels.contains_key(*c))
        .collect();

    let mut take: Vec<&String> = actors.clone();
    if !noops.is_empty() && !deterministic {
        let n_noop = std::cmp::max(1, (take.len() as f64 * noop_frac) as usize)
            .min(noops.len());
        // Partial Fisher-Yates = rng.sample without replacement.
        let mut pool: Vec<&String> = noops.clone();
        for k in 0..n_noop {
            let j = rng.gen_range(k..pool.len());
            pool.swap(k, j);
        }
        take.extend(pool[..n_noop].iter().copied());
    }

    let mut out = Vec::with_capacity(take.len());
    for cid in take {
        let leg_v = &legal_map[cid];
        let me = leg_v["me"].as_i64().ok_or("legal me")?;
        let legal = parse_legal(&leg_v["legal"]["actions"]);
        let (owners, fallout, mut f) = featurize_for(game, tick, false, me, &ents_v, &legal)?;
        let choice = if let Some(ls) = labels.get(cid).and_then(|v| v.as_array()) {
            let label = if deterministic {
                &ls[0]
            } else {
                &ls[rng.gen_range(0..ls.len())]
            };
            match label_to_choice(label, game, &amounts, me) {
                Some(c) => {
                    force_legal(&mut f, &c);
                    c
                }
                None => continue,
            }
        } else {
            NOOP_CHOICE
        };
        out.push(Built {
            game: gi,
            tick,
            owners,
            fallout,
            feat: f,
            choice,
            cond: placement_bucket(game.placement(cid)),
        });
    }
    Ok(out)
}

fn spawn_sample(
    games: &[Game],
    gi: usize,
    rng: &mut SmallRng,
) -> Result<Vec<Built>, String> {
    let game = &games[gi];
    if game.spawn_steps.is_empty() {
        return Ok(vec![]);
    }
    let step = &game.spawn_steps[rng.gen_range(0..game.spawn_steps.len())];
    if step.labels.is_empty() {
        return Ok(vec![]);
    }
    let (cid, label) = &step.labels[rng.gen_range(0..step.labels.len())];
    let gx = (label.x / game.ds) / REGION as i64;
    let gy = (label.y / game.ds) / REGION as i64;
    if !(0 <= gy && gy < game.gh as i64 && 0 <= gx && gx < game.gw as i64) {
        return Ok(vec![]);
    }
    let ents_v = game.entities(step.tick)?;
    let legal = Legal::default();
    let (owners, fallout, mut f) =
        featurize_for(game, step.tick, true, label.me, &ents_v, &legal)?;
    let mut choice = NOOP_CHOICE;
    choice[0] = feat::A_SPAWN;
    choice[2] = gy * GW_MAX + gx;
    // The human did spawn there: force the region into the mask.
    f.legal_tile[gy as usize * game.gw + gx as usize] = 1.0;
    Ok(vec![Built {
        game: gi,
        tick: step.tick,
        owners,
        fallout,
        feat: f,
        choice,
        cond: placement_bucket(game.placement(cid)),
    }])
}

fn one_sample(
    games: &[Game],
    gi: usize,
    step: &Value,
    cid: &str,
    labeled: bool,
    rng: &mut SmallRng,
) -> Result<Option<Built>, String> {
    let game = &games[gi];
    let tick = step["tick"].as_i64().ok_or("step tick")?;
    let ents_v = game.entities(tick)?;
    let leg_v = &step["legal"][cid];
    let me = leg_v["me"].as_i64().ok_or("legal me")?;
    let legal = parse_legal(&leg_v["legal"]["actions"]);
    let (owners, fallout, mut f) = featurize_for(game, tick, false, me, &ents_v, &legal)?;
    let mut choice = NOOP_CHOICE;
    if labeled {
        if let Some(ls) = step["labels"].get(cid).and_then(|v| v.as_array()) {
            if !ls.is_empty() {
                let label = &ls[rng.gen_range(0..ls.len())];
                match label_to_choice(label, game, &player_amounts(&ents_v), me) {
                    Some(c) => {
                        force_legal(&mut f, &c);
                        choice = c;
                    }
                    None => return Ok(None),
                }
            }
        }
    }
    Ok(Some(Built {
        game: gi,
        tick,
        owners,
        fallout,
        feat: f,
        choice,
        cond: placement_bucket(game.placement(cid)),
    }))
}

fn sample_window_native(
    games: &[Game],
    gi: usize,
    k: usize,
    noop_frac: f64,
    rng: &mut SmallRng,
) -> Result<Vec<Built>, String> {
    let game = &games[gi];
    let n = game.n_steps();
    if n < 1 {
        return Ok(vec![]);
    }
    let want_actor = rng.gen::<f64>() > noop_frac;
    let mut ends: Vec<usize> = (0..n).collect();
    for i in (1..ends.len()).rev() {
        ends.swap(i, rng.gen_range(0..=i));
    }
    for &end in ends.iter().take(16) {
        let step = game.step(end)?;
        let labels = step["labels"].as_object().ok_or("labels")?;
        let legal_map = step["legal"].as_object().ok_or("legal")?;
        let cands: Vec<&String> = if want_actor {
            labels
                .iter()
                .filter(|(c, ls)| {
                    ls.as_array().is_some_and(|a| !a.is_empty()) && legal_map.contains_key(*c)
                })
                .map(|(c, _)| c)
                .collect()
        } else {
            legal_map.keys().filter(|c| !labels.contains_key(*c)).collect()
        };
        if cands.is_empty() {
            continue;
        }
        let cid = cands[rng.gen_range(0..cands.len())].clone();
        let mut idxs: Vec<usize> = (end.saturating_sub(k - 1)..=end).collect();
        while idxs.len() < k {
            idxs.insert(0, idxs[0]);
        }
        let window: Result<Vec<Value>, String> =
            idxs.iter().map(|&i| game.step(i)).collect();
        let window = window?;
        if window.iter().any(|s| s["legal"].get(&cid).is_none()) {
            continue;
        }
        let mut out = Vec::with_capacity(k);
        for (j, s) in window.iter().enumerate() {
            match one_sample(games, gi, s, &cid, j == k - 1, rng)? {
                Some(b) => out.push(b),
                None => break,
            }
        }
        if out.len() == k {
            return Ok(out);
        }
    }
    Ok(vec![])
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[pyclass]
pub struct Sampler {
    games: Vec<Game>,
    spawn_games: Vec<usize>,
    noop_frac: f64,
    spawn_frac: f64,
    seed: AtomicU64,
    // Surplus from parallel step draws (a step yields all its actors);
    // carried into the next batch instead of thrown away.
    carry: std::sync::Mutex<Vec<Built>>,
}

impl Sampler {
    fn next_seed(&self) -> u64 {
        let mut s = self.seed.fetch_add(0x9E3779B97F4A7C15, Ordering::Relaxed);
        splitmix(&mut s)
    }

    fn built_to_dict<'py>(&self, py: Python<'py>, b: Built) -> PyResult<Bound<'py, PyDict>> {
        let g = &self.games[b.game];
        let (gh, gw, hr, wr) = (g.gh, g.gw, g.hr, g.wr);
        let d = PyDict::new(py);
        let arr2 = |v: Vec<u8>, h: usize, w: usize| {
            PyArray1::from_vec(py, v).reshape([h, w]).map_err(err("reshape"))
        };
        let arrf = |v: Vec<f32>, sh: [usize; 2]| {
            PyArray1::from_vec(py, v).reshape(sh).map_err(err("reshape"))
        };
        d.set_item("owners", arr2(b.owners, hr, wr)?)?;
        d.set_item("fallout_packed", arr2(b.fallout, hr, wr / 8)?)?;
        let ts = g.terrain_static.get_or_try_init(py, || -> PyResult<Py<PyArray3<f32>>> {
            let mut v = Vec::with_capacity(2 * hr * wr);
            v.extend(g.land.iter().map(|&x| x as f32));
            v.extend(g.mag.iter().map(|&x| x as f32 / 31.0));
            Ok(PyArray1::from_vec(py, v)
                .reshape([2, hr, wr])
                .map_err(err("terrain"))?
                .unbind())
        })?;
        d.set_item("terrain_static", ts)?;
        d.set_item(
            "static",
            PyArray1::from_vec(py, b.feat.stat)
                .reshape([feat::N_STATIC, gh, gw])
                .map_err(err("static"))?,
        )?;
        d.set_item(
            "transient",
            PyArray1::from_vec(py, b.feat.transient)
                .reshape([feat::N_TRANSIENT, gh, gw])
                .map_err(err("transient"))?,
        )?;
        d.set_item("clut", PyArray1::from_slice(py, &b.feat.clut))?;
        d.set_item("players", arrf(b.feat.players, [feat::MAX_SLOTS, feat::P_FEAT])?)?;
        d.set_item("pmask", PyArray1::from_slice(py, &b.feat.pmask))?;
        d.set_item("scalars", PyArray1::from_slice(py, &b.feat.scalars))?;
        d.set_item("me_slot", b.feat.me_slot)?;
        d.set_item("legal_actions", PyArray1::from_slice(py, &b.feat.legal_actions))?;
        d.set_item(
            "legal_ptarget",
            arrf(b.feat.legal_ptarget, [feat::N_ACTIONS, feat::MAX_SLOTS])?,
        )?;
        d.set_item("legal_build", PyArray1::from_slice(py, &b.feat.legal_build))?;
        d.set_item("legal_nuke", PyArray1::from_slice(py, &b.feat.legal_nuke))?;
        d.set_item("legal_tile", arrf(b.feat.legal_tile, [gh, gw])?)?;
        let ch = PyDict::new(py);
        for (k, v) in [
            ("action", b.choice[0]),
            ("player_slot", b.choice[1]),
            ("tile_region", b.choice[2]),
            ("build_type", b.choice[3]),
            ("nuke_type", b.choice[4]),
            ("quantity", b.choice[5]),
        ] {
            ch.set_item(k, v)?;
        }
        d.set_item("choice", ch)?;
        d.set_item("cond", b.cond)?;
        // AE-latent cache identity (rl.obs.ZCache): the AE input is fully
        // determined by (game, tick) for cached BC data.
        d.set_item("z_key", (g.name.as_str(), b.tick))?;
        Ok(d)
    }
}

fn sample_step_native(
    games: &[Game],
    spawn_games: &[usize],
    noop_frac: f64,
    spawn_frac: f64,
    seed: u64,
) -> Result<Vec<Built>, String> {
    let mut rng = SmallRng::seed_from_u64(seed);
    for _ in 0..64 {
        let out = if !spawn_games.is_empty() && rng.gen::<f64>() < spawn_frac {
            spawn_sample(games, spawn_games[rng.gen_range(0..spawn_games.len())], &mut rng)?
        } else {
            let gi = rng.gen_range(0..games.len());
            let game = &games[gi];
            if game.n_steps() == 0 {
                continue;
            }
            let step = game.step(rng.gen_range(0..game.n_steps()))?;
            step_samples(games, gi, &step, noop_frac, &mut rng, false)?
        };
        if !out.is_empty() {
            return Ok(out);
        }
    }
    Err("could not draw a non-empty BC step in 64 tries".into())
}

#[pymethods]
impl Sampler {
    #[new]
    #[pyo3(signature = (roots, holdout_every=10, holdout=false, noop_frac=0.15, spawn_frac=0.03, seed=0))]
    fn new(
        roots: Vec<String>,
        holdout_every: u32,
        holdout: bool,
        noop_frac: f64,
        spawn_frac: f64,
        seed: u64,
    ) -> PyResult<Self> {
        let mut dirs: Vec<PathBuf> = Vec::new();
        for root in &roots {
            let mut found: Vec<PathBuf> = Vec::new();
            for l1 in fs::read_dir(root).map_err(err("root"))?.flatten() {
                if !l1.path().is_dir() {
                    continue;
                }
                for l2 in fs::read_dir(l1.path()).map_err(err("dir"))?.flatten() {
                    if l2.path().join("cache-bc/index.json").is_file() {
                        found.push(l2.path());
                    }
                }
            }
            found.sort();
            dirs.extend(found);
        }
        let mut games = Vec::new();
        for dir in dirs {
            let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let digest = Md5::digest(name.as_bytes());
            if (digest[0] as u32 % holdout_every == 0) == holdout {
                games.push(Game::load(&dir)?);
            }
        }
        if games.is_empty() {
            return Err(PyFileNotFoundError::new_err(format!(
                "no cache-bc under {roots:?} (holdout={holdout})"
            )));
        }
        let spawn_games = games
            .iter()
            .enumerate()
            .filter(|(_, g)| g.n_spawn > 0)
            .map(|(i, _)| i)
            .collect();
        Ok(Sampler {
            games,
            spawn_games,
            noop_frac,
            spawn_frac,
            seed: AtomicU64::new(seed ^ 0xA076_1D64_78BD_642F),
            carry: std::sync::Mutex::new(Vec::new()),
        })
    }

    fn n_games(&self) -> usize {
        self.games.len()
    }

    /// Draw >= n samples (rayon across steps, GIL released), return exactly n
    /// raw dicts shaped like ObsBuilder.prepare() + {choice, cond}. Surplus
    /// samples (a step emits all its actors at once) carry to the next call.
    fn sample_batch<'py>(&self, py: Python<'py>, n: usize) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let mut built: Vec<Built> = {
            let mut carry = self.carry.lock().unwrap();
            let take = std::cmp::min(n, carry.len());
            carry.drain(..take).collect()
        };
        while built.len() < n {
            // A step yields ~2-4 samples on average; slight underdraw and a
            // small follow-up round beats overdrawing into the carry buffer.
            let draws = std::cmp::max(2, (n - built.len()).div_ceil(3));
            let seeds: Vec<u64> = (0..draws).map(|_| self.next_seed()).collect();
            let games = &self.games;
            let (sg, nf, sf) = (&self.spawn_games, self.noop_frac, self.spawn_frac);
            let new: Result<Vec<Vec<Built>>, String> = py.allow_threads(|| {
                seeds
                    .into_par_iter()
                    .map(|s| sample_step_native(games, sg, nf, sf, s))
                    .collect()
            });
            for v in new.map_err(PyRuntimeError::new_err)? {
                built.extend(v);
            }
        }
        self.carry.lock().unwrap().extend(built.drain(n..));
        built.into_iter().map(|b| self.built_to_dict(py, b)).collect()
    }

    /// One player's last-k consecutive decision steps (rl/bc_data.py
    /// sample_window); empty list when the draw fails, caller retries.
    fn sample_window<'py>(&self, py: Python<'py>, k: usize) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let seed = self.next_seed();
        let games = &self.games;
        let nf = self.noop_frac;
        let built: Result<Vec<Built>, String> = py.allow_threads(|| {
            let mut rng = SmallRng::seed_from_u64(seed);
            let gi = rng.gen_range(0..games.len());
            sample_window_native(games, gi, k, nf, &mut rng)
        });
        built
            .map_err(PyRuntimeError::new_err)?
            .into_iter()
            .map(|b| self.built_to_dict(py, b))
            .collect()
    }

    /// `batch` windows of k steps each, drawn in parallel (rayon): the seq
    /// trainer's whole micro-batch in one call. Failed draws (want_actor
    /// misses, dead-player windows) retry internally, so exactly batch*k
    /// samples come back, step-major.
    fn sample_windows<'py>(
        &self,
        py: Python<'py>,
        batch: usize,
        k: usize,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let games = &self.games;
        let nf = self.noop_frac;
        let mut built: Vec<Built> = Vec::with_capacity(batch * k);
        for _ in 0..64 {
            let need = batch - built.len() / k;
            if need == 0 {
                break;
            }
            let seeds: Vec<u64> = (0..need).map(|_| self.next_seed()).collect();
            let new: Result<Vec<Vec<Built>>, String> = py.allow_threads(|| {
                seeds
                    .into_par_iter()
                    .map(|s| {
                        let mut rng = SmallRng::seed_from_u64(s);
                        let gi = rng.gen_range(0..games.len());
                        sample_window_native(games, gi, k, nf, &mut rng)
                    })
                    .collect()
            });
            for v in new.map_err(PyRuntimeError::new_err)? {
                if v.len() == k {
                    built.extend(v);
                }
            }
        }
        if built.len() != batch * k {
            return Err(PyRuntimeError::new_err(
                "could not draw enough seq windows in 64 rounds",
            ));
        }
        built.into_iter().map(|b| self.built_to_dict(py, b)).collect()
    }

    /// Deterministic samples for parity testing: game gi, step si, actors
    /// only (no noop draw), first label per actor.
    fn debug_step_samples<'py>(
        &self,
        py: Python<'py>,
        gi: usize,
        si: usize,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let step = self.games[gi].step(si).map_err(PyRuntimeError::new_err)?;
        let mut rng = SmallRng::seed_from_u64(0);
        let built = step_samples(&self.games, gi, &step, self.noop_frac, &mut rng, true)
            .map_err(PyRuntimeError::new_err)?;
        built.into_iter().map(|b| self.built_to_dict(py, b)).collect()
    }
}
