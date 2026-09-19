//! Driver: run the composed tick over a fresh engine dump and COUNT the
//! agreement, tick by tick, against the engine's own claim order.
//!
//!     cargo run -- --dump <ndjson> --map <map dir> --t0 <tick> --t1 <tick>
//!
//! The attack's start troops is **computed**, not read: the record's
//! `AttackSnapshot.troops` is `troops as i64` (`tick_dump.rs:326`), and
//! `tiles_used = within(2000*speed/attack_troops, 5, 100)` divides by the true
//! `f64`. `--start record` restores the old truncated input for comparison.
//!
//! Multi-attack concurrency is modelled: an owner can hold several live land
//! attacks, and a newly created one ABSORBS every other live same-owner
//! same-target attack (`execution/attack.rs:177-184` ->
//! `game.rs:2121-2156` `merge_outgoing_land_attacks`). Which attacks exist at
//! all - the AI's send decision and its timing - is the ONE thing taken from
//! the record; everything else (troops, budget, borderline set, claim order)
//! is computed. See the README's measured/assumed split.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;

use ofcuda_env::{econ_row, land_attack_start_troops, state_hash, state_plane, tribe_ratios, Attack};

#[derive(Clone, Copy, PartialEq, Eq)]
enum StartMode {
    /// `ai_attack.rs:9-18` from the economy step (stage 1) + `max_troops_for`
    /// + the tribe's own ratio.
    Computed,
    /// The record's truncated integer (`tick_dump.rs:326`), plus `--frac`.
    Record,
}

struct Args {
    dump: PathBuf,
    map: PathBuf,
    t0: u32,
    t1: u32,
    frac: f64,
    verbose: bool,
    heap_at: u32,
    start: StartMode,
    all_mismatches: bool,
    counts_from: u32,
    counts_n: u32,
    trace_from: u32,
    trace_to: u32,
    trace_owner: u16,
    trace_target: u16,
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
        start: StartMode::Computed,
        all_mismatches: false,
        counts_from: 0,
        counts_n: 24,
        trace_from: 1,
        trace_to: 0,
        trace_owner: u16::MAX,
        trace_target: u16::MAX,
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
            "--start" => {
                a.start = match val(i)?.as_str() {
                    "computed" => StartMode::Computed,
                    "record" => StartMode::Record,
                    o => return Err(format!("--start: {o} (computed|record)")),
                };
                i += 2;
            }
            "--all-mismatches" => {
                a.all_mismatches = true;
                i += 1;
            }
            "--counts-from" => {
                a.counts_from = val(i)?.parse().map_err(|e| format!("counts-from: {e}"))?;
                i += 2;
            }
            "--counts-n" => {
                a.counts_n = val(i)?.parse().map_err(|e| format!("counts-n: {e}"))?;
                i += 2;
            }
            "--trace" => {
                // --trace <from>:<to>[:owner[:target]]
                let s = val(i)?;
                let parts: Vec<&str> = s.split(':').collect();
                a.trace_from = parts[0].parse().map_err(|e| format!("trace: {e}"))?;
                a.trace_to = if parts.len() > 1 && !parts[1].is_empty() {
                    parts[1].parse().map_err(|e| format!("trace: {e}"))?
                } else {
                    a.trace_from
                };
                if parts.len() > 2 && !parts[2].is_empty() {
                    a.trace_owner = parts[2].parse().map_err(|e| format!("trace: {e}"))?;
                }
                if parts.len() > 3 && !parts[3].is_empty() {
                    a.trace_target = parts[3].parse().map_err(|e| format!("trace: {e}"))?;
                }
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

/// One attack entry of a record's `attacks` array.
#[derive(Clone, Copy, Debug)]
struct AttRec {
    owner: u16,
    target: u16,
    troops: i64,
    live: bool,
}

struct PInfo {
    troops: i32,
    tiles: i32,
    gold: i64,
    id: String,
    ptype: u8,
}

/// The parts of a dump line `ofcuda_tick::parse_dump` does not keep: the attack
/// snapshots (whose `troops` is the truncated i64), the engine state hash, and
/// the per-player economy inputs (troops/tiles/gold/id/type).
#[derive(Default)]
struct Extra {
    attacks: HashMap<u32, Vec<AttRec>>,
    /// `gameHash` (`tick_dump.rs:279-284`): the ENGINE's own hash, an i64.
    hash: HashMap<u32, i64>,
    p: HashMap<u32, HashMap<u16, PInfo>>,
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
            let list: Vec<AttRec> = arr
                .iter()
                .filter_map(|x| {
                    Some(AttRec {
                        owner: x.get("ownerSmallId")?.as_u64()? as u16,
                        target: x.get("targetSmallId")?.as_u64()? as u16,
                        troops: x.get("troops")?.as_i64()?,
                        live: x.get("attackLive")?.as_bool()?,
                    })
                })
                .collect();
            out.attacks.insert(tick as u32, list);
        }
        if let Some(arr) = v.get("players").and_then(|x| x.as_array()) {
            let mut m: HashMap<u16, PInfo> = HashMap::new();
            for p in arr {
                let Some(sid) = p.get("smallId").and_then(|x| x.as_u64()) else {
                    continue;
                };
                m.insert(
                    sid as u16,
                    PInfo {
                        troops: p.get("troops").and_then(|x| x.as_i64()).unwrap_or(0) as i32,
                        tiles: p.get("tiles").and_then(|x| x.as_i64()).unwrap_or(0) as i32,
                        gold: p.get("gold").and_then(|x| x.as_i64()).unwrap_or(0),
                        id: p
                            .get("id")
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string(),
                        ptype: match p.get("playerType").and_then(|x| x.as_str()) {
                            Some("Bot") => ofcuda_econ::core_impl::PT_BOT as u8,
                            Some("Nation") => ofcuda_econ::core_impl::PT_NATION as u8,
                            _ => ofcuda_econ::core_impl::PT_HUMAN as u8,
                        },
                    },
                );
            }
            out.p.insert(tick as u32, m);
        }
        if let Some(h) = v.get("gameHash") {
            let parsed = h
                .as_i64()
                .or_else(|| h.as_u64().map(|x| x as i64))
                .or_else(|| h.as_str().and_then(ofcuda_hash::parse_hex64).map(|x| x as i64));
            if let Some(x) = parsed {
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

/// One live exec in the crate's mirror of `Game::execs`: the attack plus the
/// record slot it occupies.
struct AttState {
    atk: Attack,
    target: u16,
}

struct Stats {
    creations: usize,
    creations_start_ok: usize,
    merges: usize,
    merge_start_ok: usize,
    evictions_unmodelled: usize,
    evictions_dead: usize,
    record_anomalies: usize,
    deaths_starved: usize,
    deaths_retreated: usize,
    linger_ticks: usize,
    heap_drops: u64,
    heap_peak: usize,
    plane_tot: usize,
    plane_eq: usize,
    plane_ne: usize,
    fnv_tot: usize,
    fnv_eq: usize,
}

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

    // The crate's mirror of `Game::execs`, per owner, in creation order.
    let mut owners: BTreeMap<u16, Vec<AttState>> = BTreeMap::new();
    let mut st = Stats {
        creations: 0,
        creations_start_ok: 0,
        merges: 0,
        merge_start_ok: 0,
        evictions_unmodelled: 0,
        evictions_dead: 0,
        record_anomalies: 0,
        deaths_starved: 0,
        deaths_retreated: 0,
        linger_ticks: 0,
        heap_drops: 0,
        heap_peak: 0,
        plane_tot: 0,
        plane_eq: 0,
        plane_ne: 0,
        fnv_tot: 0,
        fnv_eq: 0,
    };
    let mut ok = 0usize;
    let mut tot = 0usize;
    let mut first_bad: Option<String> = None;
    let mut first_bad_tick: Option<u32> = None;
    let mut all_bad: Vec<String> = Vec::new();
    let mut counts: Vec<(u32, String)> = Vec::new();

    for s in a.t0..a.t1 {
        let Some(cur) = dump.get(&s) else { continue };
        if cur.is_empty() || !dump.contains_key(&(s + 1)) {
            break;
        }
        let r_now = ex.attacks.get(&s).cloned().unwrap_or_default();
        let plane = plane_at(s);
        let ps = players_at(s);
        let nxt_ps = players_at(s + 1);

        let border_of = |sid: u16| -> Vec<u32> {
            ps.iter()
                .find(|p| p.0 == sid as u32)
                .map(|p| p.2.clone())
                .unwrap_or_default()
        };

        // ---- 1. align the crate's execs mirror with the record at rec `s` ----
        //
        // An attack present at rec `s` was created during the call that ended at
        // rec `s` (i.e. during the step before this one), so its `init` ran with
        // the map/border state AT rec `s` and with `game.ticks() == s - 1`.
        let mut own_set: HashSet<u16> = HashSet::new();
        for r in &r_now {
            own_set.insert(r.owner);
        }
        for o in owners.keys() {
            own_set.insert(*o);
        }

        for &owner in &own_set {
            let r_group: Vec<AttRec> =
                r_now.iter().copied().filter(|x| x.owner == owner).collect();
            let v = owners.entry(owner).or_default();

            // Evictions: an attack the engine dropped between rec s-1 and rec s.
            while v.len() > r_group.len() {
                if let Some(i) = v.iter().position(|x| !x.atk.attack_live) {
                    v.remove(i);
                    st.evictions_dead += 1;
                } else {
                    // A live attack vanished with no merge and no death this
                    // crate can explain.
                    v.remove(0);
                    st.evictions_unmodelled += 1;
                }
            }

            // Creations: the record's tail entries have no crate counterpart.
            while v.len() < r_group.len() {
                let e = r_group[v.len()];
                if !e.live {
                    st.record_anomalies += 1;
                    break;
                }
                let prev_info = ex
                    .p
                    .get(&s.saturating_sub(1))
                    .and_then(|m| m.get(&owner));
                let (start, is_bot, ratio_txt) = match (a.start, prev_info) {
                    (StartMode::Record, _) => {
                        (Some(e.troops as f64 + a.frac), true, "record".to_string())
                    }
                    (StartMode::Computed, Some(pi)) => {
                        let ratios = tribe_ratios(&pi.id);
                        // `ai_attack.rs:398-407` (terra nullius) uses
                        // `expand_ratio`; `ai_attack.rs:499-505` (player
                        // target) uses `reserve_ratio`.
                        let ratio = if e.target == 0 {
                            ratios.expand_ratio
                        } else {
                            ratios.reserve_ratio
                        };
                        let row = econ_row(
                            s - 1,
                            pi.troops,
                            pi.tiles,
                            0,
                            pi.gold,
                            pi.ptype as u32,
                        );
                        (
                            land_attack_start_troops(&row, ratio),
                            pi.ptype == ofcuda_econ::core_impl::PT_BOT as u8,
                            format!("{ratio:.4}"),
                        )
                    }
                    (StartMode::Computed, None) => (None, true, "?".to_string()),
                };
                let Some(start) = start else {
                    st.record_anomalies += 1;
                    break;
                };
                st.creations += 1;
                // Floor of the computed start must equal the record's value
                // whenever this creation is not a merge (a merge shows the sum).
                let will_merge = v
                    .iter()
                    .any(|x| x.target == e.target && x.atk.attack_live);
                if a.start == StartMode::Computed && !will_merge && start.floor() as i64 == e.troops
                {
                    st.creations_start_ok += 1;
                }
                let mut atk = Attack::new(owner, e.target, is_bot, start, ofcuda_tick::SEED);
                // `AttackExecution::init` (`attack.rs:160-164`): a land attack
                // with no source tile calls `refresh_to_conquer`, which iterates
                // the OWNER's `border_tiles` as of the end of the creating call.
                atk.refresh(&border_of(owner), &plane, terrain, w, h, s - 1);

                // `attack.rs:177-184` -> `game.rs:2121-2156`
                // `merge_outgoing_land_attacks`: every OTHER live same-owner
                // same-target attack is absorbed (`*troops += extra`) and killed.
                let mut absorbed = 0.0f64;
                for other in v.iter_mut() {
                    if other.target == e.target && other.atk.attack_live {
                        absorbed += other.atk.troops;
                        other.atk.kill();
                    }
                }
                if absorbed > 0.0 {
                    st.merges += 1;
                    atk.troops += absorbed;
                    if a.start == StartMode::Computed
                        && atk.troops.floor() as i64 == e.troops
                    {
                        st.merge_start_ok += 1;
                    }
                    if a.verbose {
                        println!(
                            "  [merge @rec{} sid{}] new({ratio_txt}) start={:.6} + absorbed={:.6} -> {:.6} (record {})",
                            s, owner, start, absorbed, atk.troops, e.troops
                        );
                    }
                }
                if a.heap_at == s + 1 && a.verbose {
                    println!(
                        "  [init @rec{} sid{}] troops={:.6} border={} heap={} prng_calls={}",
                        s,
                        owner,
                        atk.troops,
                        atk.border.len(),
                        atk.heap.len,
                        atk.pr.calls
                    );
                }
                v.push(AttState { atk, target: e.target });
            }
        }

        // ---- 2. tick every live exec, in execs order ----
        let mut computed: HashMap<u16, Vec<u32>> = HashMap::new();
        for (owner, v) in owners.iter_mut() {
            let border = border_of(*owner);
            let mut real_deaths: Vec<usize> = Vec::new();
            for (i, at) in v.iter_mut().enumerate() {
            let was_live = at.atk.attack_live;
            if !was_live {
                st.linger_ticks += 1;
            }
            let t_before = at.atk.troops;
            let d0 = at.atk.heap.drops;
            let out = at.atk.tick(&plane, terrain, w, h, s, &border, 0.0, false);
            st.heap_drops += at.atk.heap.drops - d0;
            if at.atk.heap.peak > st.heap_peak {
                st.heap_peak = at.atk.heap.peak;
            }
            if a.trace_from <= s
                && s <= a.trace_to
                && *owner == a.trace_owner
                && at.target == a.trace_target
            {
                println!(
                    "  [tick rec{} sid{} ->{}] was_live={} troops_in={:.6} border_in={} draw={} budget={:.10} heap={} claims={} troops_out={:.6}{}",
                    s,
                    owner,
                    at.target,
                    was_live,
                    t_before,
                    out.border_size,
                    out.budget_draw,
                    out.budget,
                    at.atk.heap.len,
                    out.claims.len(),
                    out.troops_after,
                    if out.dead { " DEAD" } else { "" }
                );
            }
                if !out.claims.is_empty() {
                    computed.entry(*owner).or_default().extend(out.claims);
                }
                if out.dead {
                    if out.retreated {
                        st.deaths_retreated += 1;
                    } else if out.starved {
                        st.deaths_starved += 1;
                    }
                    // A death DURING the tick sets `active = false`, so the exec
                    // is evicted at the end of this very tick: it is gone at the
                    // next record. A merge-kill (`init`) leaves `active` true and
                    // is dropped by the alignment above instead.
                    if was_live {
                        real_deaths.push(i);
                    }
                }
            }
            for i in real_deaths.into_iter().rev() {
                v.remove(i);
            }
        }

        // ---- 3b. composed plane vs the engine's plane one tick later ----
        //
        // The composed claims ARE the engine's claims (verified above), so the
        // plane rebuilt from them must equal the engine's plane at rec s+1
        // word-for-word - that is what makes the FNV state hash reproducible
        // from composed state.
        {
            let mut pc = plane.clone();
            for (sid, cl) in computed.iter() {
                for t in cl {
                    pc[*t as usize] = *sid;
                }
            }
            let pn = plane_at(s + 1);
            st.plane_tot += 1;
            if pc == pn {
                st.plane_eq += 1;
            } else if a.verbose && st.plane_eq + st.plane_ne < 3 {
                st.plane_ne += 1;
                let bad = pc.iter().zip(&pn).position(|(x, y)| x != y);
                println!("  [plane @rec{}] composed != engine at word {bad:?}", s + 1);
            }
            st.fnv_tot += 1;
            if state_hash(&pc) == state_hash(&pn) {
                st.fnv_eq += 1;
            }
        }

        // ---- 4. compare against the engine's own claim log ----
        for &(sid, _, _, _) in &ps {
            let Some(p) = ps.iter().find(|p| p.0 == sid) else {
                continue;
            };
            let Some(pn) = nxt_ps.iter().find(|p| p.0 == sid) else {
                continue;
            };
            let n0 = p.3.len().min(pn.3.len());
            let expected = &pn.3[n0..];
            let empty = Vec::new();
            let got = computed.get(&(sid as u16)).unwrap_or(&empty);
            tot += 1;
            let held = got.len() == expected.len() && got.iter().zip(expected).all(|(x, y)| x == y);
            if held {
                ok += 1;
            } else if first_bad.is_none() {
                let n_attacks = owners.get(&(sid as u16)).map(|v| v.len()).unwrap_or(0);
                let d = got
                    .iter()
                    .zip(expected)
                    .position(|(x, y)| x != y)
                    .unwrap_or(got.len().min(expected.len()));
                let lo = d.saturating_sub(3);
                first_bad = Some(format!(
                    "transition rec{}->rec{} sid{}: engine_claimed={} computed_claims={} (crate live attacks: {}) \
                     first differing index {} (reduce these: {}, {})\n     computed[{}..] {:?}\n     engine  [{}..] {:?}",
                    s,
                    s + 1,
                    sid,
                    expected.len(),
                    got.len(),
                    n_attacks,
                    d,
                    got.get(d).map(|v| *v as i64).unwrap_or(-1),
                    expected.get(d).map(|v| *v as i64).unwrap_or(-1),
                    lo,
                    &got[lo..got.len().min(lo + 12)],
                    lo,
                    &expected[lo..expected.len().min(lo + 12)]
                ));
                first_bad_tick = Some(s);
            }
            if !held && a.all_mismatches {
                all_bad.push(format!(
                    "rec{}->rec{} sid{} engine={} computed={}",
                    s,
                    s + 1,
                    sid,
                    expected.len(),
                    got.len()
                ));
            }
        }

        // per-tick claim counts, engine vs composed (collected for every
        // transition; printed around the first mismatch below)
        {
            let mut line = format!("tick {:>4} :", s);
            for &(sid, _, _, _) in &ps {
                let p = ps.iter().find(|p| p.0 == sid).unwrap();
                let pn = match nxt_ps.iter().find(|p| p.0 == sid) {
                    Some(x) => x,
                    None => continue,
                };
                let n0 = p.3.len().min(pn.3.len());
                let expected = pn.3.len() - n0;
                let empty = Vec::new();
                let got = computed.get(&(sid as u16)).unwrap_or(&empty).len();
                line.push_str(&format!(" sid{sid} engine={expected} computed={got} |"));
            }
            counts.push((s, line));
        }
    }

    println!(
        "composed tick, engine order: {}",
        ofcuda_env::PIPELINE.join(" -> ")
    );
    println!("dump  : {} ({} tick records)", a.dump.display(), dump.len());
    println!("map   : {w}x{h}");
    println!(
        "start : {}",
        match a.start {
            StartMode::Computed => {
                "COMPUTED - econ step + max_troops_for + the tribe's own ratio".to_string()
            }
            StartMode::Record => format!("record integer (truncated) + frac {}", a.frac),
        }
    );
    println!("--- per-tick claim counts (engine vs composed) ---");
    {
        let start = if a.counts_from > 0 {
            a.counts_from
        } else {
            first_bad_tick.unwrap_or(a.t0).saturating_sub(3)
        };
        let n = if a.counts_n > 0 { a.counts_n as usize } else { 24 };
        for (_, l) in counts.iter().filter(|(t, _)| *t >= start).take(n) {
            println!("{l}");
        }
    }
    println!("--- constructions ---");
    println!(
        "  creations {} (floor(computed start) == record troops on {}/{} non-merge creations)",
        st.creations,
        st.creations_start_ok,
        st.creations - st.merges
    );
    println!(
        "  merges    {} (floor(computed merged troops) == record troops on {}/{})",
        st.merges, st.merge_start_ok, st.merges
    );
    println!(
        "  deaths    starved(troops<1)={} retreated(heap dry)={} | lingering no-op ticks {}",
        st.deaths_starved, st.deaths_retreated, st.linger_ticks
    );
    println!(
        "  heap      candidates dropped on a full heap {} | peak heap depth {} (HEAP_CAP {})",
        st.heap_drops,
        st.heap_peak,
        ofcuda_tick::HEAP_CAP
    );
    println!(
        "  evictions merge-killed={} unexplained={} | unusable record entries {}",
        st.evictions_dead, st.evictions_unmodelled, st.record_anomalies
    );
    match &first_bad {
        Some(m) => println!("FIRST MISMATCH: {m}"),
        None => println!("FIRST MISMATCH: none in [{}, {})", a.t0, a.t1),
    }
    if a.all_mismatches {
        println!("--- all mismatching transitions ({}), first 40 ---", all_bad.len());
        for l in all_bad.iter().take(40) {
            println!("{l}");
        }
    }
    println!(
        "CLAIM-IDENTITY AGREEMENT: {ok}/{tot} (tick,player) pairs reproduce the engine's claim order exactly"
    );

    // ---- the hash question ----
    //
    // There are TWO different hashes here and they are not interchangeable:
    //
    //   `gameHash` (the record's): the engine's JS sync checksum,
    //   `hash.rs:7-22` - `1.0 + sum_p(id_hash * (troops + tiles_owned)
    //   + sum_units unit_hash_js)`, with `id_hash = |simple_hash(id)|`
    //   (`util.rs:4-13`, note the `.abs()`). It reads NO tile plane, so no
    //   plane serializer can ever reproduce it.
    //
    //   `state_hash` (what `ofcuda_hash` computes): FNV-1a-64 over the
    //   `width*height` `u16` owner plane, two little-endian bytes per word.
    //   `ofcuda_hash`'s 200/200 is agreement with its own oracle over the same
    //   plane bytes (`ofcuda_hash/oracle/src/main.rs:246`), i.e. a plane
    //   serialization check - and it is the hash the composed plane can be
    //   checked against, because the composed claims reproduce the engine's.
    let mut gh_ok = 0usize;
    let mut gh_tot = 0usize;
    let mut fnv_vs_game = 0usize;
    for s in a.t0..a.t1 {
        let (Some(pi), Some(eng)) = (ex.p.get(&s), ex.hash.get(&s)) else {
            continue;
        };
        gh_tot += 1;
        let mut h = 1.0f64;
        for p in pi.values() {
            h += ofcuda_prng::simple_hash(&p.id) as f64 * (p.troops as f64 + p.tiles as f64);
        }
        if h as i64 == *eng {
            gh_ok += 1;
        }
        if state_hash(&plane_at(s)) as i64 == *eng {
            fnv_vs_game += 1;
        }
    }
    println!("--- the hash question ---");
    println!(
        "  gameHash formula re-evaluated from the record's (id, troops, tiles) \
         [hash.rs:7-22]: {gh_ok}/{gh_tot}"
    );
    println!(
        "  ofcuda_hash's FNV state hash over the record's OWNER PLANE vs the \
         record's gameHash: {fnv_vs_game}/{gh_tot} - different objects (the \
         engine's gameHash never reads the plane)"
    );
    println!(
        "  composed plane == engine plane one tick later: {}/{} (word-for-word) | \
         FNV(composed plane) == FNV(engine plane): {}/{}",
        st.plane_eq, st.plane_tot, st.fnv_eq, st.fnv_tot
    );
    Ok(())
}
