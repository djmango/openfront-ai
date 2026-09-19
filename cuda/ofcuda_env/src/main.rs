//! Driver: run the composed tick over a fresh engine dump and COUNT the
//! agreement, tick by tick, against the engine's own claim order and state hash.
//!
//!     cargo run -- --dump <ndjson> --map <map dir> --t0 <tick> --t1 <tick> --frac <f>
//!
//! `--frac` is the fractional part of the attack's starting troops that
//! `tick_dump.rs:326` (`troops as i64`) throws away. It is the ONE input the
//! record cannot carry; the engine's own construction of it is
//! `ofcuda_env::land_attack_troops` (`execution/ai_attack.rs:9-18`), which needs
//! the owner's float troops from the economy (stage 1) plus `max_troops_for` and
//! the tribe's `expand_ratio` (`bot/tribe.rs:42`). `--frac 0` therefore means
//! "use the truncated record value", which was the previous best's input.

use std::collections::HashMap;
use std::path::PathBuf;

use ofcuda_env::{state_hash, state_plane, Attack};

struct Args {
    dump: PathBuf,
    map: PathBuf,
    t0: u32,
    t1: u32,
    frac: f64,
    verbose: bool,
    heap_at: u32,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        dump: PathBuf::from("/tmp/envdump/b002.ndjson"),
        map: PathBuf::from(
            "/opt/data/workspaces/skg/openfront-ai/openfront/resources/maps/pangaea",
        ),
        t0: 300,
        t1: 1250,
        frac: 0.0,
        verbose: false,
        heap_at: 0,
    };
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < v.len() {
        let val = |i: usize| -> Result<String, String> {
            v.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", v[i]))
        };
        match v[i].as_str() {
            "--dump" => {
                a.dump = PathBuf::from(val(i)?);
                i += 2;
            }
            "--map" => {
                a.map = PathBuf::from(val(i)?);
                i += 2;
            }
            "--t0" => {
                a.t0 = val(i)?.parse().map_err(|e| format!("t0: {e}"))?;
                i += 2;
            }
            "--t1" => {
                a.t1 = val(i)?.parse().map_err(|e| format!("t1: {e}"))?;
                i += 2;
            }
            "--frac" => {
                a.frac = val(i)?.parse().map_err(|e| format!("frac: {e}"))?;
                i += 2;
            }
            "--verbose" => {
                a.verbose = true;
                i += 1;
            }
            "--heap-at" => {
                a.heap_at = val(i)?.parse().map_err(|e| format!("heap-at: {e}"))?;
                i += 2;
            }
            other => return Err(format!("unknown arg {other}")),
        }
    }
    Ok(a)
}

/// The parts of a dump line `ofcuda_tick::parse_dump` does not keep: the attack
/// snapshots (whose `troops` is the truncated i64) and the engine state hash.
#[derive(Default)]
struct Extra {
    attacks: HashMap<u32, Vec<(u16, i64)>>,
    hash: HashMap<u32, u64>,
}

fn extras(path: &std::path::Path) -> Result<Extra, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Extra::default();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        if v.get("type").and_then(|t| t.as_str()) == Some("header") {
            continue;
        }
        let Some(tick) = v.get("tick").and_then(|t| t.as_u64()) else {
            continue;
        };
        if let Some(arr) = v.get("attacks").and_then(|x| x.as_array()) {
            let list: Vec<(u16, i64)> = arr
                .iter()
                .filter_map(|x| {
                    Some((
                        x.get("ownerSmallId")?.as_u64()? as u16,
                        x.get("troops")?.as_i64()?,
                    ))
                })
                .collect();
            out.attacks.insert(tick as u32, list);
        }
        if let Some(h) = v.get("gameHash").and_then(|x| x.as_str()) {
            if let Some(x) = ofcuda_hash::parse_hex64(h) {
                out.hash.insert(tick as u32, x);
            }
        }
    }
    Ok(out)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

type PL = (u32, Vec<u32>, Vec<u32>, Vec<u32>); // sid, owned, border, ownedOrder

fn run() -> Result<(), String> {
    let a = parse_args()?;
    let map = ofcuda_tick::load_map(&a.map)?;
    let (w, h) = (map.width, map.height);
    let terrain = &map.terrain;
    let dump = ofcuda_tick::parse_dump(&a.dump)?;
    let ex = extras(&a.dump)?;

    let players_at = |t: u32| -> Vec<PL> {
        let mut v: Vec<PL> = Vec::new();
        if let Some(ps) = dump.get(&t) {
            for p in ps.values() {
                v.push((
                    p.small_id,
                    p.owned_tiles.clone(),
                    p.border_order.clone(),
                    p.owned_order.clone(),
                ));
            }
        }
        v.sort_by_key(|x| x.0);
        v
    };
    let plane_at = |t: u32| -> Vec<u16> {
        let players: Vec<(u32, Vec<u32>)> = players_at(t)
            .into_iter()
            .map(|(s, o, _, _)| (s, o))
            .collect();
        state_plane(&players, w, h)
    };

    let mut live: HashMap<u16, Attack> = HashMap::new();
    let mut ok = 0usize;
    let mut tot = 0usize;
    let mut first_bad: Option<String> = None;
    let mut head: Vec<String> = Vec::new();

    for s in a.t0..a.t1 {
        let Some(cur) = dump.get(&s) else { continue };
        if cur.is_empty() || !dump.contains_key(&(s + 1)) {
            break;
        }
        let now = ex.attacks.get(&s).cloned().unwrap_or_default();
        let plane = plane_at(s);
        let ps = players_at(s);
        let nxt_ps = players_at(s + 1);

        // Every attack present in the record after step `s` also existed during
        // the step that produced it. Its start troops is the record value plus
        // the truncation the dump cannot carry.
        //
        // Its `init` -> `refresh_to_conquer` ran during the step that FIRST put
        // it in the record, i.e. against the owner's border as of the state
        // BEFORE that record tick - sid2 claims nothing in its creation step
        // (rec318 has the same ownedOrder length as rec317), which is how the
        // boundary is observable.
        for &(sid, tr) in &now {
            if live.contains_key(&sid) {
                continue;
            }
            let init_s = s.saturating_sub(1);
            let init_ps = players_at(init_s);
            let Some(p) = init_ps
                .iter()
                .find(|p| p.0 == sid as u32)
                .or_else(|| ps.iter().find(|p| p.0 == sid as u32))
            else {
                continue;
            };
            let init_plane = plane_at(init_s);
            let mut atk = Attack::new(sid, 0, true, tr as f64 + a.frac, ofcuda_tick::SEED);
            // init: refresh_to_conquer (attack.rs:1265-1274). Its priorities
            // carry `game.ticks()` at the moment of the refresh, which is the
            // tick the attack was CREATED in - one before the record tick that
            // first shows it (an attack created in tick T appears in rec[T+1]).
            atk.refresh(&p.2, &init_plane, terrain, w, h, init_s);
            live.insert(sid, atk);
        }

        for &(sid, _) in &now {
            let Some(p) = ps.iter().find(|p| p.0 == sid as u32) else {
                continue;
            };
            let Some(pn) = nxt_ps.iter().find(|p| p.0 == sid as u32) else {
                continue;
            };
            let n0 = p.3.len().min(pn.3.len());
            let expected = &pn.3[n0..];
            let atk = live.get_mut(&sid).expect("created above");
            if a.heap_at == s + 1 {
                let n = atk.heap.len;
                println!(
                    "[heap @ start of rec{}->rec{} sid{}] len={} border={} prng_calls={}",
                    s,
                    s + 1,
                    sid,
                    n,
                    atk.border.len(),
                    atk.pr.calls
                );
                for i in 0..n {
                    println!("   {} {}", atk.heap.tiles[i], atk.heap.pri[i]);
                }
            }
            let out = atk.tick(&plane, terrain, w, h, s, &p.2, 0.0, false);
            tot += 1;
            let held = out.claims.len() == expected.len()
                && out.claims.iter().zip(expected).all(|(x, y)| x == y);
            if held {
                ok += 1;
            } else if first_bad.is_none() {
                first_bad = Some(format!(
                    "transition rec{}->rec{} sid{}: engine_claimed={} computed_claims={} computed_budget={:.4} border={} draw={} troops_before={:.6} troops_after={:.6}\n     computed {:?}\n     engine   {:?}",
                    s,
                    s + 1,
                    sid,
                    expected.len(),
                    out.claims.len(),
                    out.budget,
                    out.border_size,
                    out.budget_draw,
                    atk.troops + (out.troops_after - atk.troops),
                    out.troops_after,
                    &out.claims[..out.claims.len().min(12)],
                    &expected[..expected.len().min(12)]
                ));
            }
            if head.len() < 12 {
                head.push(format!(
                    "  rec{}->rec{} sid{} computed_budget={:.0} border={} claimed={} engine_claimed={} troops_after={:.4}",
                    s,
                    s + 1,
                    sid,
                    out.budget,
                    out.border_size,
                    out.claims.len(),
                    expected.len(),
                    out.troops_after
                ));
            }
        }
    }

    println!(
        "composed tick, engine order: {}",
        ofcuda_env::PIPELINE.join(" -> ")
    );
    println!("dump  : {} ({} tick records)", a.dump.display(), dump.len());
    println!("map   : {w}x{h}");
    println!("frac  : {} (truncated record troops + frac)", a.frac);
    println!("--- first 12 composed ticks ---");
    for l in &head {
        println!("{l}");
    }
    match &first_bad {
        Some(m) => println!("FIRST MISMATCH: {m}"),
        None => println!("FIRST MISMATCH: none in [{}, {})", a.t0, a.t1),
    }
    println!(
        "CLAIM-IDENTITY AGREEMENT: {ok}/{tot} (tick,attack) pairs reproduce the engine's claim order exactly"
    );

    // The serializer check: hash the engine's own ownership plane with
    // `ofcuda_hash` and compare against the engine's per-tick `gameHash`.
    let mut hash_ok = 0usize;
    let mut hash_tot = 0usize;
    for s in a.t0..a.t1 {
        if let (Some(eng), true) = (ex.hash.get(&s), dump.contains_key(&s)) {
            hash_tot += 1;
            if state_hash(&plane_at(s)) == *eng {
                hash_ok += 1;
            }
        }
    }
    println!(
        "plane-serializer check (ofcuda_hash over the dumped owner plane vs the record's gameHash): {hash_ok}/{hash_tot}"
    );
    if a.verbose {
        println!(
            "NOTE: gameHash is not the plain owner plane - ofcuda_hash's own\n\
             reference plane is the `state` plane; see ofcuda_hash/README.md."
        );
    }
    Ok(())
}
