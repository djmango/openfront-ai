//! `ofcuda_matrix` - the map x agent-count driver.
//!
//! Composes the three proven pieces into ONE cell run:
//!
//! 1. **terrain** - `ofcuda_map` (`load_map_normal`), the same `map.bin` bytes
//!    the engine loads (`GameMapSize::Normal`).
//! 2. **init** - the engine-exact spawn/owned sets. `ofcuda_spawn` (which links
//!    the real engine) writes the reference, `ofcuda_prng`'s `spawnall` runs the
//!    CUDA spawn kernel against it and `--dump`s the port's initial state.
//! 3. **tick** - `ofcuda_tick`'s canonical `core_impl.rs` on the device
//!    (`attack_init` / `attack_tick`), with `ofcuda_env`'s float budget and
//!    terrain-cost helpers called from device code.
//!
//! # Ground truth
//!
//! Every comparison in this file is against the **engine oracle**
//! (`ofcuda_matrix/oracle`, which links `openfront-engine`) or against the
//! engine-produced spawn reference. Nothing is compared against the CUDA port.
//!
//! # Driving attacks (what is model and what is evidence)
//!
//! The oracle records, at every boundary: the live attacks (owner, target, troop
//! bits), every player's `border_tiles` insertion order, the per-tick claim sets
//! in the engine's own order, the per-player owned counts, and the FNV-1a-64 of
//! the whole owner plane. This driver replays that schedule:
//!
//! * an attack that first appears in `ATTACK c` is created on the device with
//!   `attack_init` stamped with the engine tick of boundary `c-1` (that is
//!   `game.ticks()` inside the `execute_next_tick` that ran the `init`);
//! * it then ticks, in `ATTACK` order (= the engine's `execs` order), with the
//!   engine tick of the boundary it is leaving - so its claim order and its
//!   `add_neighbors` priorities carry the same tick stamp the engine used;
//! * `refresh_to_conquer` reads the engine's own `BORDER` order for the owner.
//!
//! So: the **schedule** (when an attack is created, and its troop count at
//! creation) is host-supplied from the engine record. Everything else - the heap
//! order, the one extra PRNG draw per tick, the budget, the pop loop, the
//! frontier, the claims and the resulting plane - is computed ON THE DEVICE.
//! The engine's claim SETS and ORDER per tick are the check, not the input.

mod kernels;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use kernels::{BC, MAXC, SCAL, SCAL_OUT};
use ofcuda_matrix::{
    fnv1a_u16_le, owners_of, parse_init, parse_oracle, plane_from_init, AttackSnap, Oracle,
};
use ofcuda_tick::{Prng, HEAP_CAP, SEED};

/// Simultaneous attacks the device array can hold. Overflow is a hard error.
const MAX_SLOTS: usize = 1024;
/// `ob_meta` is indexed by owner small_id. A small_id >= this is a hard cell
/// error (the device array is indexed without a guard).
const COLS: usize = 8192;
/// Largest `defender_tiles` the host-precomputed `1.0 - sigmoid(defender_tiles,
/// LN_2/50_000, 150_000)` table covers. The engine's sigmoid is a pure function
/// of an INTEGER tile count, so evaluating it on the host with the engine's own
/// expression makes the device's value bit-identical instead of approximate
/// (CUDA's `exp` and Rust's libm `exp` are different implementations). Beyond
/// this count the table saturates - raise it before driving a map where a
/// defender can own more tiles than this.
const DEFSIG_N: usize = 2_000_000;
/// Threads for the one-thread kernels.
const ONE: u32 = 1;

fn cfg(block: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct Cell {
    map: String,
    agents: u32,
    nations: u32,
    ticks: u32,
}

struct Args {
    map: String,
    agents: u32,
    nations: u32,
    ticks: u32,
    seed: String,
    out: PathBuf,
    oracle_dir: PathBuf,
    refs_dir: PathBuf,
    oracle_bin: PathBuf,
    spawn_bin: PathBuf,
    spawnall_bin: PathBuf,
    env_sh: PathBuf,
    maps_root: PathBuf,
    oracle: Option<PathBuf>,
    init: Option<PathBuf>,
    dump_planes: bool,
    cells_file: Option<PathBuf>,
    force: bool,
}

impl Default for Args {
    fn default() -> Self {
        let root = PathBuf::from("/opt/data/workspaces/skg");
        Self {
            map: "pangaea".into(),
            agents: 18,
            nations: 0,
            ticks: 120,
            seed: "parity".into(),
            out: root.join("ofcuda_matrix/out"),
            oracle_dir: PathBuf::new(),
            refs_dir: PathBuf::new(),
            oracle_bin: root.join("ofcuda_matrix/oracle/target/release/oracle"),
            spawn_bin: root.join("ofcuda_spawn/target/release/ofcuda_spawn"),
            spawnall_bin: root.join("ofcuda_prng/target/release/spawnall"),
            env_sh: root.join("ofcuda_env.sh"),
            maps_root: root.join("openfront-ai"),
            oracle: None,
            init: None,
            dump_planes: false,
            cells_file: None,
            force: false,
        }
    }
}

fn parse_args() -> Result<Args, String> {
    let v: Vec<String> = std::env::args().skip(1).collect();
    let mut a = Args::default();
    let mut i = 0;
    while i < v.len() {
        let val = |i: usize| {
            v.get(i + 1)
                .cloned()
                .ok_or_else(|| format!("{} needs a value", v[i]))
        };
        match v[i].as_str() {
            "--map" => {
                a.map = val(i)?;
                i += 2;
            }
            "--agents" => {
                a.agents = val(i)?.parse().map_err(|e| format!("agents: {e}"))?;
                i += 2;
            }
            "--nations" => {
                a.nations = val(i)?.parse().map_err(|e| format!("nations: {e}"))?;
                i += 2;
            }
            "--ticks" => {
                a.ticks = val(i)?.parse().map_err(|e| format!("ticks: {e}"))?;
                i += 2;
            }
            "--seed" => {
                a.seed = val(i)?;
                i += 2;
            }
            "--out" => {
                a.out = PathBuf::from(val(i)?);
                i += 2;
            }
            "--oracle-dir" => {
                a.oracle_dir = PathBuf::from(val(i)?);
                i += 2;
            }
            "--refs-dir" => {
                a.refs_dir = PathBuf::from(val(i)?);
                i += 2;
            }
            "--oracle" => {
                a.oracle = Some(PathBuf::from(val(i)?));
                i += 2;
            }
            "--init" => {
                a.init = Some(PathBuf::from(val(i)?));
                i += 2;
            }
            "--oracle-bin" => {
                a.oracle_bin = PathBuf::from(val(i)?);
                i += 2;
            }
            "--spawn-bin" => {
                a.spawn_bin = PathBuf::from(val(i)?);
                i += 2;
            }
            "--spawnall-bin" => {
                a.spawnall_bin = PathBuf::from(val(i)?);
                i += 2;
            }
            "--env-sh" => {
                a.env_sh = PathBuf::from(val(i)?);
                i += 2;
            }
            "--maps-root" => {
                a.maps_root = PathBuf::from(val(i)?);
                i += 2;
            }
            "--dump-planes" => {
                a.dump_planes = true;
                i += 1;
            }
            "--cells-file" => {
                a.cells_file = Some(PathBuf::from(val(i)?));
                i += 2;
            }
            "--force" => {
                a.force = true;
                i += 1;
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown arg {other}")),
        }
    }
    if a.oracle_dir.as_os_str().is_empty() {
        a.oracle_dir = a.out.join("oracle");
    }
    if a.refs_dir.as_os_str().is_empty() {
        a.refs_dir = a.out.join("refs");
    }
    Ok(a)
}

const USAGE: &str = "\
usage: ofcuda_matrix --map <name> --agents <N> --nations {0|1|2} --ticks <T> --out <dir>
                     [--seed <s>] [--dump-planes] [--force]
                     [--oracle <file>] [--init <file>]
                     [--oracle-dir <d>] [--refs-dir <d>] [--maps-root <d>]
                     [--cells-file <f>]
  --cells-file <f>  lines: <map> <agents> [nations] [ticks] - run many cells
  --dump-planes     write <out>/<cell>/planes.bin (tick 0 = init) + terrain.bin
";

// ---------------------------------------------------------------------------
// Result row
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Row {
    map: String,
    w: u32,
    h: u32,
    agents: u32,
    nations: u32,
    ticks: u32,
    ran_ticks: u32,
    init_sets: bool,
    init_hash: bool,
    /// Nations (if any) matched the engine on id/sid/tile/tick/owned/troops.
    init_nations: bool,
    tick_match: usize,
    /// MEASUREMENT ONLY: attacks killed by the engine's alliance retreat
    /// (`attack.rs:222-229`) on the boundary the engine killed them.
    friendly_retreats: usize,
    tick_total: usize,
    hash_match: usize,
    hash_total: usize,
    count_match: usize,
    count_total: usize,
    first_div: String,
    troops_match: usize,
    troops_total: usize,
    churn_skipped: usize,
    engine_evictions: usize,
    reinits: usize,
    /// Re-creates whose device frontier was compared against the engine's own
    /// recorded `to_conquer`/`border_tiles` sizes, and how many agreed.
    reinit_checked: usize,
    reinit_agree: usize,
    dev_claims: usize,
    /// The player-clusters stream: the engine's own `ENG_CLUSTER_REMOVE` lines
    /// for this cell, the device's removal records, and how many of those agree
    /// on `(engine tick, victim, captor, tiles)`. This is the direct parity
    /// check for the cluster pass - a long run can be exact in every boundary
    /// while the stream is not, and each removal is visible individually.
    cluster_eng: usize,
    cluster_dev: usize,
    cluster_match: usize,
    /// Times the pass actually ran `calculate_clusters` (its trigger), and any
    /// capacity overflow it hit (border set, cluster storage, `flood_owned`).
    cluster_fires: usize,
    cluster_ovf: usize,
    /// The device's own `last_cluster_calc` after the tick vs the engine's, for
    /// every player at every boundary: the trigger rule (`player_clusters.rs:
    /// 358-393`) checked independently of whether it produced a removal.
    cadence_match: usize,
    cadence_total: usize,
    note: String,
}

impl Row {
    fn tsv_header() -> &'static str {
        "map\tw\th\tN\tnations\tticks\tinit_sets\tinit_hash\ttick_match\ttick_total\thash_match\thash_total\tcount_match\tcount_total\ttroops_match\ttroops_total\tchurn_skipped\tengine_evictions\treinits\treinit_checked\treinit_agree\tdev_claims\tcluster_eng\tcluster_dev\tcluster_match\tcluster_fires\tcluster_ovf\tcadence_match\tcadence_total\tfirst_div\tnote"
    }
    fn tsv(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.map,
            self.w,
            self.h,
            self.agents,
            self.nations,
            self.ticks,
            yn(self.init_sets),
            yn(self.init_hash),
            self.tick_match,
            self.tick_total,
            self.hash_match,
            self.hash_total,
            self.count_match,
            self.count_total,
            self.troops_match,
            self.troops_total,
            self.churn_skipped,
            self.engine_evictions,
            self.reinits,
            self.reinit_checked,
            self.reinit_agree,
            self.dev_claims,
            self.cluster_eng,
            self.cluster_dev,
            self.cluster_match,
            self.cluster_fires,
            self.cluster_ovf,
            self.cadence_match,
            self.cadence_total,
            self.first_div,
            self.note,
        )
    }
    fn pass(&self) -> bool {
        self.init_sets
            && self.init_hash
            && self.init_nations
            && self.tick_match == self.tick_total
            && self.hash_match == self.hash_total
            && self.count_match == self.count_total
            // A re-create the device could not reproduce is a failure, not a note.
            && self.reinit_agree == self.reinit_checked
            // The cluster-removal stream must match the engine's, removal for
            // removal. Nothing is weakened here: every pre-existing criterion
            // above still applies.
            && self.cluster_eng == self.cluster_dev
            && self.cluster_eng == self.cluster_match
            && self.cluster_ovf == 0
            && self.cadence_match == self.cadence_total
    }
}

/// `Config::max_troops` (`core/config.rs:363-388`), difficulty `Easy` and
/// `city_level_sum = 0`.
///
/// `city_level_sum` needs `ConstructionExecution` (completed cities), which this
/// port does not model. The traced cells agree with 0, but the assumption is
/// explicit rather than implied: a cell with completed cities would need
/// `city_troop_increase()` added here.
fn eng_max_troops(ptype: char, tiles: f64) -> f64 {
    let mut m = 2.0 * (tiles.powf(0.6) * 1000.0 + 50_000.0);
    match ptype {
        'B' => m /= 3.0,
        'N' => m *= 0.5, // difficulty "Easy"
        // 'H' only differs when `game_config.infinite_troops`, which no cell sets.
        _ => {}
    }
    m
}

/// `Config::troop_increase_rate_raw` (`core/config.rs:433-458`), difficulty
/// `Easy`, `city_level_sum = 0`. Returns the RAW f64; the caller floors it the
/// way `game.add_troops` does.
fn troop_increase_rate_raw(ptype: char, troops: f64, tiles: f64) -> f64 {
    let max = eng_max_troops(ptype, tiles);
    let mut to_add = 10.0 + troops.powf(0.73) / 4.0;
    to_add *= 1.0 - troops / max;
    if ptype == 'B' {
        to_add *= 0.5;
    }
    if ptype == 'N' {
        to_add *= 0.9; // difficulty "Easy"
    }
    (troops + to_add).min(max) - troops
}

fn yn(b: bool) -> &'static str {
    if b { "yes" } else { "NO" }
}

// ---------------------------------------------------------------------------
// Input acquisition - the composition of the three proven pieces
// ---------------------------------------------------------------------------

fn run(cmd: &mut Command, what: &str) -> Result<(), String> {
    eprintln!("$ {cmd:?}");
    let out = cmd.output().map_err(|e| format!("{what}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "{what} failed ({}): {}{}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

fn ensure_oracle(a: &Args, c: &Cell) -> Result<PathBuf, String> {
    if let Some(p) = &a.oracle {
        if !p.exists() {
            return Err(format!("--oracle {} does not exist", p.display()));
        }
        return Ok(p.clone());
    }
    std::fs::create_dir_all(&a.oracle_dir).map_err(|e| e.to_string())?;
    let p = a
        .oracle_dir
        .join(format!("{}_n{}_nat{}_t{}.dump", c.map.to_lowercase(), c.agents, c.nations, c.ticks));
    if p.exists() && !a.force {
        return Ok(p);
    }
    let mut cmd = Command::new(&a.oracle_bin);
    cmd.args([
        "replay",
        "--maps",
        &c.map,
        "--ns",
        &c.agents.to_string(),
        "--nations",
        &c.nations.to_string(),
        "--ticks",
        &c.ticks.to_string(),
        "--seed",
        &a.seed,
        "--out-dir",
    ])
    .arg(&a.oracle_dir);
    run(&mut cmd, "engine oracle replay")?;
    if !p.exists() {
        return Err(format!("oracle did not write {}", p.display()));
    }
    Ok(p)
}

fn ensure_init(a: &Args, c: &Cell) -> Result<PathBuf, String> {
    if let Some(p) = &a.init {
        if !p.exists() {
            return Err(format!("--init {} does not exist", p.display()));
        }
        return Ok(p.clone());
    }
    std::fs::create_dir_all(&a.refs_dir).map_err(|e| e.to_string())?;
    let p = a
        .refs_dir
        .join(format!("{}_a{}_nat{}.init.txt", c.map.to_lowercase(), c.agents, c.nations));
    if p.exists() && !a.force {
        return Ok(p);
    }
    let refp = a
        .refs_dir
        .join(format!("{}_a{}_nat{}.spawn.txt", c.map.to_lowercase(), c.agents, c.nations));
    if !refp.exists() || a.force {
        let mut cmd = Command::new(&a.spawn_bin);
        // `--nations` MUST match the oracle's nations spec: a ref built with the
        // default (0) and replayed with Exact(1) is a different spawn stream, and
        // the cell would fail on a config mismatch rather than on device code.
        cmd.args(["--map", &c.map, "--agents", &c.agents.to_string()])
            .args(["--nations", &c.nations.to_string()])
            .args(["--seed", &a.seed])
            .arg("--out")
            .arg(&refp);
        run(&mut cmd, "engine spawn reference (ofcuda_spawn)")?;
    }
    // spawnall runs the CUDA spawn kernel; it MUST go through the CUDA env
    // wrapper (driver library path + nix gcc wrapper for its build scripts).
    let mut cmd = Command::new("bash");
    cmd.arg(&a.env_sh)
        .arg(&a.spawnall_bin)
        .arg(&refp)
        .arg("--dump")
        .arg(&p);
    run(&mut cmd, "spawnall --dump")?;
    if !p.exists() {
        return Err(format!("spawnall did not write {}", p.display()));
    }
    Ok(p)
}

// ---------------------------------------------------------------------------
// The cell
// ---------------------------------------------------------------------------

struct Slot {
    owner: u16,
    target: u16,
    idx: usize,
    alive: bool,
    created_at: u32,
    troops_bits: u64,
    /// The engine's `attack_id` for the attack this slot currently models. A
    /// change for a live `(owner, target)` is the engine having re-created it.
    attack_id: String,
}

fn run_cell(a: &Args, c: &Cell) -> Result<Row, String> {
    let oracle_path = ensure_oracle(a, c)?;
    let init_path = ensure_init(a, c)?;

    // ---- ground truth (engine) ----
    let orc: Oracle = parse_oracle(&oracle_path)?;
    let init = parse_init(&init_path)?;
    if orc.width != init.width || orc.height != init.height {
        return Err(format!(
            "oracle {}x{} vs init dump {}x{} - different game configuration",
            orc.width, orc.height, init.width, init.height
        ));
    }
    let (w, h) = (orc.width, orc.height);
    let wh = (w as usize) * (h as usize);
    // Health signal that does not depend on the payload: a cell that resolved
    // a 0-area map measures nothing and must never be able to report PASS.
    if w == 0 || h == 0 {
        return Err(format!(
            "map {} resolved to {w}x{h} (oracle {}): refusing to run a 0-area cell",
            c.map,
            oracle_path.display()
        ));
    }
    // Same for a nations cell: if the port's init dump carries no nation row,
    // the nation layer was not exercised and the cell must not report a result.
    if c.nations > 0 && init.nation_rows.is_empty() {
        return Err(format!(
            "nations={} cell {} but {} carries no `nation` row: the nation layer \
             was not exercised - refusing to score it",
            c.nations,
            c.map,
            init_path.display()
        ));
    }

    // ---- terrain (ofcuda_map) ----
    let mdir = ofcuda_map::map_dir(&a.maps_root, &c.map);
    let (meta, terrain) = ofcuda_map::load_map_normal(&mdir)?;
    if meta.width != w || meta.height != h {
        return Err(format!(
            "{}: map.bin {}x{} vs oracle {}x{} - wrong map resolved for {:?}",
            mdir.display(),
            meta.width,
            meta.height,
            w,
            h,
            c.map
        ));
    }
    let land_from_bytes = ofcuda_map::land_tiles(&terrain);

    let mut row = Row {
        map: c.map.to_lowercase(),
        w,
        h,
        agents: c.agents,
        nations: c.nations,
        ticks: c.ticks,
        ran_ticks: 0,
        ..Default::default()
    };

    // Engine player types by small_id: drives the kernel's attacker-type flag.
    let bot_sids: std::collections::HashSet<u16> = orc
        .roster
        .iter()
        .filter(|r| r.1 == 'B')
        .map(|r| r.0)
        .collect();

    // ---- init: the port's spawn output vs the engine's spawn output ----
    let init_plane = plane_from_init(&init, w, h)?;
    let init_owners = owners_of(&init_plane);
    let b0 = &orc.boundaries[0];
    row.init_hash = fnv1a_u16_le(&init_plane) == b0.hash;
    let mut sets_ok = true;
    let mut mismatches = Vec::new();
    let mut all_sids: Vec<u16> = b0.owned0.keys().copied().collect();
    all_sids.sort_unstable();
    all_sids.extend(init_owners.keys().copied().filter(|k| !b0.owned0.contains_key(k)));
    all_sids.sort_unstable();
    all_sids.dedup();
    for sid in &all_sids {
        // The engine's `owned_tiles` is INSERTION order, not ascending; the
        // plane-derived sets are ascending. Compare as SETS (the plane hash and
        // the count check are what prove the two agree tile-for-tile).
        let mut eng = b0.owned0.get(sid).cloned().unwrap_or_default();
        let mut port = init_owners.get(sid).cloned().unwrap_or_default();
        eng.sort_unstable();
        port.sort_unstable();
        if eng != port {
            sets_ok = false;
            mismatches.push(format!(
                "player {sid}: engine {} tiles, port {} tiles, first difference {:?}",
                eng.len(),
                port.len(),
                eng.iter()
                    .zip(port.iter())
                    .find(|(x, y)| x != y)
                    .map(|(x, y)| format!("{x} vs {y}"))
                    .unwrap_or_else(|| format!("prefix equal, lengths differ"))
            ));
        }
    }
    // spawn tiles from the engine roster
    for (bi, b) in init.bots.iter().enumerate() {
        if b.2 < 0 {
            continue;
        }
        if let Some(r) = orc.roster.iter().find(|r| r.0 == b.0) {
            if r.3 != b.2 {
                sets_ok = false;
                mismatches.push(format!(
                    "bot {bi} sid {}: engine spawn tile {} vs port {}",
                    b.0, r.3, b.2
                ));
            }
        }
    }
    // ---- nations: every integer the nation spawn layer produces -----------
    // Each field is checked against the ENGINE, from two independent sources:
    // the oracle's roster + boundary-0 `OWNED`/`PLAYER` rows, and the reference
    // file's own `player` rows (which `ofcuda_spawn` wrote from the engine).
    let mut nation_fail: Vec<String> = Vec::new();
    let mut nation_ok = true;
    for (ni, n) in init.nation_rows.iter().enumerate() {
        let (sid, id, _cx, _cy, tile, tick, n_tiles) = n;
        match orc.roster.iter().find(|r| r.0 == *sid) {
            None => {
                nation_ok = false;
                nation_fail.push(format!("nation {ni} sid {sid} missing from engine roster"));
            }
            Some(r) => {
                if r.1 != 'N' {
                    nation_ok = false;
                    nation_fail.push(format!("nation {ni} sid {sid} engine type {} != N", r.1));
                }
                if r.2 != *id {
                    nation_ok = false;
                    nation_fail
                        .push(format!("nation {ni} id engine {} vs port {id}", r.2));
                }
                if r.3 != *tile {
                    nation_ok = false;
                    nation_fail.push(format!(
                        "nation {ni} spawn tile engine {} vs port {tile}",
                        r.3
                    ));
                }
            }
        }
        // The reference file carries the engine's own spawn tick for the same
        // sid (`ofcuda_spawn` records when `spawn_tile` first appeared).
        if let Some(p) = init.players.iter().find(|p| p.0 == *sid) {
            if p.4 != *tick {
                nation_ok = false;
                nation_fail.push(format!(
                    "nation {ni} spawn tick engine {} vs port {tick}",
                    p.4
                ));
            }
        } else {
            nation_ok = false;
            nation_fail.push(format!("nation {ni} sid {sid} missing from reference players"));
        }
        // Owned count + the set itself (the set is folded into the plane above,
        // so a mismatch there also fails `init_hash`/`init_sets`).
        let eng_set = b0.owned0.get(sid).cloned().unwrap_or_default();
        if eng_set.len() != *n_tiles {
            nation_ok = false;
            nation_fail.push(format!(
                "nation {ni} tiles_owned engine {} vs port {n_tiles}",
                eng_set.len()
            ));
        }
        // Troops: `config::start_manpower(PlayerType::Nation)` at difficulty Easy.
        let eng_troops = b0
            .players
            .iter()
            .find(|p| p.sid == *sid)
            .map(|p| p.troops)
            .unwrap_or(-1.0);
        let port_troops = init.nation_troops.get(&ni).copied().unwrap_or(f64::NAN);
        if eng_troops != port_troops {
            nation_ok = false;
            nation_fail.push(format!(
                "nation {ni} troops engine {eng_troops} vs port {port_troops}"
            ));
        }
    }
    row.init_nations = nation_ok;

    row.init_sets = sets_ok;

    let mut detail = String::new();
    detail.push_str(&format!(
        "# ofcuda_matrix cell {} n={} nations={} ticks={}\n\
         # engine oracle {}\n# spawn init    {}\n# map          {} ({}x{}, land {} from BYTES)\n\
         # engine land from manifest {}\n\
         ### INIT (engine spawn output vs the CUDA spawn/init dump)\n\
         init_plane_hash {:#018x} engine_boundary0_hash {:#018x} match {}\n\
         per-player owned sets match: {}   nations match: {}\n\
         engine boundary0 owned_total {}, port plane owned {}\n",
        row.map,
        c.agents,
        c.nations,
        c.ticks,
        oracle_path.display(),
        init_path.display(),
        mdir.display(),
        w,
        h,
        land_from_bytes,
        meta.num_land_tiles,
        fnv1a_u16_le(&init_plane),
        b0.hash,
        yn(row.init_hash),
        yn(row.init_sets),
        yn(row.init_nations),
        b0.owned_total,
        init_owners.values().map(|v| v.len()).sum::<usize>(),
    ));
    for m in &mismatches {
        detail.push_str(&format!("INIT_MISMATCH {m}\n"));
    }
    for b in &init.bots {
        detail.push_str(&format!(
            "BOT {bi} sid={} id={} spawn_tile={} n_owned={}\n",
            b.0,
            b.1,
            b.2,
            b.3,
            bi = init
                .bots
                .iter()
                .position(|x| x.0 == b.0 && x.2 == b.2)
                .unwrap_or(0)
        ));
    }
    for (ni, n) in init.nation_rows.iter().enumerate() {
        detail.push_str(&format!(
            "NATION {ni} sid={} id={} cell={},{} spawn_tile={} spawn_tick={} n_owned={} tries={} troops={}\n",
            n.0,
            n.1,
            n.2,
            n.3,
            n.4,
            n.5,
            n.6,
            init.nation_tries.get(&ni).copied().unwrap_or(0),
            init.nation_troops.get(&ni).copied().unwrap_or(f64::NAN),
        ));
    }
    for m in &nation_fail {
        detail.push_str(&format!("NATION_MISMATCH {m}\n"));
    }
    detail.push_str(&format!(
        "SELFCHECK engine owned_tiles-vs-plane diff {} (nonzero => the engine's own \
         two representations disagree; PLAYER counts are then not a plane count)\n",
        orc.selfcheck_diff
    ));

    if init.bots.iter().filter(|b| b.2 < 0).count() > 0 {
        row.note = "PORT FAILED TO PLACE SOME BOT".into();
    }
    if !sets_ok || !row.init_hash || !row.init_nations {
        // A failed init makes every later comparison meaningless. Report it.
        row.note = format!("{} INIT MISMATCH", row.note).trim().to_string();
        write_cell(a, c, &detail, None, None)?;
        return Ok(row);
    }

    // ---- attacks already live at boundary 0 (spawn-phase nations) ----
    if !b0.attacks.is_empty() {
        row.note = format!(
            "{} not drivable: {} attack(s) already live at boundary 0 - the spawn-phase \
             schedule is not modelled",
            row.note,
            b0.attacks.len()
        )
        .trim()
        .to_string();
        write_cell(a, c, &detail, None, None)?;
        return Ok(row);
    }

    // ---- device ----
    let ctx = CudaContext::new(0).map_err(|e| format!("CUDA context: {e}"))?;
    let module = unsafe { kernels::device::load(&ctx).map_err(|e| format!("load module: {e}"))? };
    let stream = ctx.default_stream();
    let d_terrain = DeviceBuffer::from_host(&stream, &terrain).map_err(es)?;
    let mut d_plane = DeviceBuffer::from_host(&stream, &init_plane).map_err(es)?;
    let mut d_hash = DeviceBuffer::<u64>::zeroed(&stream, 1).map_err(es)?;
    let mut d_heap_tiles =
        DeviceBuffer::<u32>::zeroed(&stream, MAX_SLOTS * HEAP_CAP).map_err(es)?;
    let mut d_heap_pri = DeviceBuffer::<f32>::zeroed(&stream, MAX_SLOTS * HEAP_CAP).map_err(es)?;
    let mut d_border = DeviceBuffer::<u32>::zeroed(&stream, MAX_SLOTS * BC).map_err(es)?;
    let mut d_prng = DeviceBuffer::<u32>::zeroed(&stream, MAX_SLOTS * 5).map_err(es)?;
    let mut d_troops = DeviceBuffer::<f64>::zeroed(&stream, MAX_SLOTS).map_err(es)?;
    let mut d_scal = DeviceBuffer::<u32>::zeroed(&stream, MAX_SLOTS * SCAL).map_err(es)?;
    let mut d_out = DeviceBuffer::<u32>::zeroed(&stream, SCAL_OUT).map_err(es)?;
    let mut d_claims = DeviceBuffer::<u32>::zeroed(&stream, MAXC).map_err(es)?;
    let obcap = wh + 1024;
    let mut d_ob_meta = DeviceBuffer::<u32>::zeroed(&stream, 2 * COLS).map_err(es)?;
    let mut prng_host = vec![0u32; MAX_SLOTS * 5];
    let mut troops_host = vec![0f64; MAX_SLOTS];
    let mut ob_meta = vec![0u32; 2 * COLS];

    // --- per-player state for a non-terra-nullius target ---------------------
    // `attack_logic_at_tile`'s player branch (`game.rs:784-878`) reads the
    // TARGET's live troops and tiles, and the budget reads its troops too. The
    // engine records both on every `PLAYER` line, so they are host-seeded from
    // the record each boundary and the DEVICE applies its own intra-tick
    // decrements (`remove_troops` / `conquer`), exactly as `tick` does.
    // `pst[sid*3]` = troops, `[sid*3+1]` = owned tiles, `[sid*3+2]` = type
    // (1 Bot, 2 Nation, 0 Human) for `player_type` comparisons.
    let mut pst_host = vec![0f64; COLS * 3];
    let mut d_pst = DeviceBuffer::<f64>::zeroed(&stream, COLS * 3).map_err(es)?;
    let defsig_host: Vec<f64> = (0..=DEFSIG_N)
        .map(|n| {
            // `1.0 - sigmoid(defender_tiles as f64, LN_2/50_000, 150_000)` with
            // the engine's own expression, so the device reads a bit-identical
            // value instead of calling a second `exp` implementation.
            let k = std::f64::consts::LN_2 / 50_000.0;
            let x = n as f64;
            1.0 - 1.0 / (1.0 + (-k * (x - 150_000.0)).exp())
        })
        .collect();
    let d_defsig = DeviceBuffer::from_host(&stream, &defsig_host).map_err(es)?;

    // --- player-clusters pass state (`kernels.rs` `cluster_pass`) ------------
    // Persists across ticks: `d_cad` is the device's own `last_cluster_calc` /
    // `last_tile_change` / `tiles_owned` / `alive` per small id (the trigger,
    // seeded from the engine's `CADENCE` rows and then advanced by the pass
    // itself), `d_mark` is the `flood_owned` generation mark, and `d_t2b` maps
    // a border tile to its position in the tick's border array (`u32::MAX`
    // outside it) - the equivalent of the engine's `border.contains` +
    // `HashSet<TileRef> visited` pair.
    let mut d_cad = DeviceBuffer::<u32>::zeroed(&stream, COLS * kernels::device::CSTATE).map_err(es)?;
    let mut d_t2b =
        DeviceBuffer::from_host(&stream, &vec![u32::MAX; wh]).map_err(es)?;
    let mut d_cmark = DeviceBuffer::<u32>::zeroed(&stream, wh).map_err(es)?;
    let mut d_cbuf = DeviceBuffer::<u32>::zeroed(&stream, kernels::device::CS).map_err(es)?;
    let mut d_clen = DeviceBuffer::<u32>::zeroed(&stream, kernels::device::CB).map_err(es)?;
    let mut d_coff = DeviceBuffer::<u32>::zeroed(&stream, kernels::device::CB).map_err(es)?;
    let mut d_cvis =
        DeviceBuffer::<u32>::zeroed(&stream, kernels::device::CB / 32).map_err(es)?;
    let mut d_cstack =
        DeviceBuffer::<u32>::zeroed(&stream, kernels::device::CSTACK).map_err(es)?;
    let mut d_cowned =
        DeviceBuffer::<u32>::zeroed(&stream, kernels::device::COWN).map_err(es)?;
    let mut d_crem = DeviceBuffer::<u32>::zeroed(&stream, kernels::device::CREM).map_err(es)?;
    let mut d_cout = DeviceBuffer::<u32>::zeroed(&stream, kernels::device::CSTAT).map_err(es)?;
    let mut d_cfriends =
        DeviceBuffer::<u32>::zeroed(&stream, kernels::device::CNB + 1).map_err(es)?;
    let mut d_cidhash = DeviceBuffer::<u32>::zeroed(&stream, COLS).map_err(es)?;
    let mut d_corder = DeviceBuffer::<u32>::zeroed(&stream, COLS).map_err(es)?;
    let mut d_atk_owner = DeviceBuffer::<u16>::zeroed(&stream, MAX_SLOTS).map_err(es)?;
    let mut d_atk_target = DeviceBuffer::<u16>::zeroed(&stream, MAX_SLOTS).map_err(es)?;
    let mut d_atk_troops = DeviceBuffer::<f64>::zeroed(&stream, MAX_SLOTS).map_err(es)?;
    // `p.last_id_hash = simple_hash(p.id)` (`player_clusters.rs:370` via
    // `util.rs:5-27`) is fixed per player id, so it is computed once here from
    // the engine's own roster ids with the engine's own hash.
    {
        let mut idh = vec![0u32; COLS];
        for r in &orc.roster {
            idh[r.0 as usize] = ofcuda_prng::simple_hash(&r.2).max(0) as u32;
        }
        d_cidhash
            .copy_from_host(&stream, &idh)
            .map_err(es)?;
    }
    // The pass walks the engine's exec order. `PEXEC` is the engine's own
    // `exec_labels()` filtered to the `Player(<sid>)` execs, i.e. exactly the
    // order `execute_next_tick` ticks them; the roster order is the fallback
    // for an oracle predating that row.
    let mut cl_order: Vec<u32> = orc.pexec.iter().map(|s| *s as u32).collect();
    if cl_order.is_empty() {
        let mut v: Vec<u16> = orc.roster.iter().map(|r| r.0).collect();
        v.sort_unstable();
        cl_order = v.into_iter().map(|s| s as u32).collect();
    }
    {
        let mut ord = vec![0u32; COLS];
        ord[..cl_order.len()].copy_from_slice(&cl_order);
        d_corder.copy_from_host(&stream, &ord).map_err(es)?;
    }
    // The engine's own cluster-removal stream for this cell (`OF_ENG_CLUSTER=1`
    // -> `ENG_CLUSTER_REMOVE tick=.. victim=.. captor=.. tiles=..`), written by
    // `tools/gen_oracles.sh` as `<dump>.clusters`.
    let mut eng_cluster: HashMap<u32, Vec<(u16, u16, usize)>> = HashMap::new();
    let mut eng_cluster_n = 0usize;
    {
        let cp = oracle_path.with_extension("dump.clusters");
        if let Ok(txt) = std::fs::read_to_string(&cp) {
            for line in txt.lines() {
                let mut it = line.split_whitespace();
                if it.next() != Some("ENG_CLUSTER_REMOVE") {
                    continue;
                }
                let mut tk = 0u32;
                let mut vi = 0u16;
                let mut ca = 0u16;
                let mut ti = 0usize;
                for tok in it {
                    let (k, v) = tok.split_once('=').unwrap_or(("", ""));
                    match k {
                        "tick" => tk = v.parse().unwrap_or(0),
                        "victim" => vi = v.parse().unwrap_or(0),
                        "captor" => ca = v.parse().unwrap_or(0),
                        "tiles" => ti = v.parse().unwrap_or(0),
                        _ => {}
                    }
                }
                eng_cluster.entry(tk).or_default().push((vi, ca, ti));
                eng_cluster_n += 1;
            }
        }
    }
    // `cluster_eng` is accumulated per REACHED tick inside the boundary loop,
    // not taken from the whole dump: a cell that stops early must be judged on
    // the ticks it actually ran, otherwise a run that never reached the
    // engine's later removals would be reported as a stream mismatch.

    let mut slots: Vec<Slot> = Vec::new();
    let mut frames: Vec<u16> = Vec::new();
    if a.dump_planes {
        frames.extend_from_slice(&init_plane);
    }

    let mut first_div = String::from("none");
    let mut first_div_at: Option<u32> = None;
    let mut oborder = vec![0u32; obcap];
    // MEASUREMENT ONLY (`OFCUDA_MATRIX_FREEZE_ATTACKS=<b>`): from boundary b on
    // the device keeps its OWN attack list - it creates no attack from `ATTACK`,
    // applies no eviction and re-stamps no engine re-create - so a run with it
    // set measures how far the port can SELF-DRIVE a game instead of replaying
    // the oracle's attack schedule. Never a pass condition; the matrix runs
    // without it. The border re-seed is deliberately LEFT ON so the number
    // isolates the attack list (say so in any report of it).
    let freeze_at: Option<usize> = std::env::var("OFCUDA_MATRIX_FREEZE_ATTACKS")
        .ok()
        .and_then(|s| s.parse().ok());
    let mut sd_create = 0usize;
    let mut sd_evict = 0usize;
    let mut sd_reinit = 0usize;

        // Diagnostic-only: the first boundary at which a live attack's DEVICE
        // frontier size differs from the engine's own recorded `to_conquer`
        // size. Never a pass condition; reported so a silent frontier drift is
        // measured instead of inferred.
        let mut frontier_seen: HashMap<(u16, u16), usize> = HashMap::new();

    for b in 1..=(c.ticks as usize) {
        let prev = &orc.boundaries[b - 1];
        let cur = &orc.boundaries[b];
        let tick = prev.engine_tick;

        // --- 1. the owner border arrays for THIS tick (engine order) ---
        let mut off = 0usize;
        for (sid, tiles) in &prev.borders {
            if *sid as usize >= COLS {
                return Err(format!(
                    "owner small_id {sid} >= COLS {COLS}: raise COLS before driving this map"
                ));
            }
            let n = tiles.len().min(obcap - off);
            oborder[off..off + n].copy_from_slice(&tiles[..n]);
            ob_meta[*sid as usize * 2] = off as u32;
            ob_meta[*sid as usize * 2 + 1] = n as u32;
            off += n;
        }
        let mut d_oborder = DeviceBuffer::from_host(&stream, &oborder[..off.max(1)]).map_err(es)?;
        d_ob_meta.copy_from_host(&stream, &ob_meta).map_err(es)?;

        // --- 1b. the per-player state this boundary's tick starts from ---
        // `prev` is the boundary the tick is leaving, i.e. the state the
        // engine's own `tick` read. Nothing is carried over: the record is the
        // source, so a device-side drift cannot accumulate silently.
        //
        // `PlayerExecution::tick` INCOME, PORTED (`execution/player.rs:84-88`):
        // the engine runs `game.add_troops(small_id, game.troop_increase_rate_raw_for(small_id))`
        // for every player. Every `PlayerExecution` is created at spawn and every
        // `AttackExecution` is appended to `execs` when it is created, so ALL the
        // player executions precede ALL the attacks in `execute_next_tick`'s exec
        // order: the income is a pure TICK-START mutation of each player's live
        // troop count, and an attack that targets a player reads `p.troops`
        // AFTER it. The port previously seeded `pst` from the per-boundary record
        // and never added income, so a player-target attack opened ~3.6e-4
        // relative lower (a smaller `defender_troops`, so a smaller
        // `alt_attacker_loss`), the device's attacker lost slightly less per pop
        // and drifted ABOVE the engine - the exact `device >= engine` drift this
        // harness was chasing.
        pst_host.fill(0.0);
        for p in &prev.players {
            let i = p.sid as usize * 3;
            if i + 2 < pst_host.len() {
                pst_host[i] = p.troops;
                pst_host[i + 1] = p.tiles as f64;
                pst_host[i + 2] = match p.ptype {
                    'B' => 1.0,
                    'N' => 2.0,
                    _ => 0.0,
                };
            }
        }
        // Applied as a second pass so each player's income uses its OWN
        // tick-start troops/tiles, exactly as `troop_increase_rate_raw_for` does.
        for p in &prev.players {
            let i = p.sid as usize * 3;
            if i + 2 >= pst_host.len() || p.tiles <= 0 {
                continue; // `PlayerExecution::tick` returns early with no tiles
            }
            let inc = troop_increase_rate_raw(p.ptype, pst_host[i], p.tiles as f64);
            // `game.add_troops` (`game.rs:1147-1155`): `p.troops += to_int(amount)`,
            // `to_int` is `floor` (`util.rs:26`). A negative rate means
            // `add_troops` forwards to `remove_troops` -> `to_int(-amount)`, i.e.
            // the magnitude is floored before being negated.
            if inc < 0.0 {
                pst_host[i] -= (-inc).floor();
            } else {
                pst_host[i] += inc.floor();
            }
        }
        d_pst.copy_from_host(&stream, &pst_host).map_err(es)?;

        // --- 2. bind the engine's live attacks to device slots ---
        // Matched on (owner, target) and NOT on troop bits: the engine can
        // re-create an attack mid-flight, so the bits legitimately change for the
        // same `(owner, target)`. Re-creation is detected on the `attack_id`
        // instead (see below), which is exact rather than inferred.
        // There is deliberately no owner-only fallback: it used to bind an
        // engine attack onto any same-owner slot and re-stamp the host struct,
        // which hid an engine-side re-create from the device instead of reporting it.
        let troops_all = d_troops.to_host_vec(&stream).map_err(es)?;
        let _scal_all = d_scal.to_host_vec(&stream).map_err(es)?;
        for s in slots.iter_mut() {
            s.troops_bits = troops_all[s.idx].to_bits();
        }
        let mut plan: Vec<(usize, AttackSnap)> = Vec::new();
        let mut taken: Vec<bool> = vec![false; slots.len()];
        // Slots the engine RE-CREATED this boundary: applied AFTER the tick, so
        // the tick that still belongs to the old attack runs on the old state.
        // (slot position, the engine's re-created attack, the id it replaced)
        let mut pending_reinit: Vec<(usize, AttackSnap, String)> = Vec::new();
        for snap in &prev.attacks {
            let exact = slots.iter().position(|s| {
                s.alive && !taken[s.idx] && s.owner == snap.owner && s.target == snap.target
            });
            match exact {
                Some(i) => {
                    taken[i] = true;
                    // `i` is a POSITION and `taken` is indexed by SLOT ID; they are
                    // the same only because `slots` is never compacted (no
                    // retain/remove/swap_remove/drain) and `idx == slots.len()` at
                    // push. Adding any removal makes this silently wrong.
                    let mut s = std::mem::replace(
                        &mut slots[i],
                        Slot {
                            owner: 0,
                            target: 0,
                            idx: usize::MAX,
                            alive: false,
                            created_at: 0,
                            troops_bits: 0,
                            attack_id: String::new(),
                        },
                    );
                    s.troops_bits = snap.troops.to_bits();
                    let prev_id = s.attack_id.clone();
                    s.attack_id = snap.attack_id.clone();
                    slots[i] = s;
                    plan.push((i, snap.clone()));

                    // --- engine-side RE-CREATE of a live attack ------------------
                    // `AttackExecution::init` mints a fresh `attack_id`
                    // (`attack.rs:156`), so a CHANGED id for a live
                    // `(owner, target)` means the engine built a NEW exec object:
                    // the bot AI re-issued the attack
                    // (`game.add_land_attack_from`, `game.rs:1514`), and
                    // `init` coalesced it with the outgoing attack of the same
                    // target (`merge_outgoing_land_attacks`, `game.rs:2122-2156`)
                    // by ADDING the old attack's remaining troops into `troops`
                    // and killing the old one. `execute_next_tick`
                    // (`game.rs:3657-3700`) runs every existing exec FIRST and the
                    // new exec's `init` at the END of that same tick, then appends
                    // it to `execs` - which is why the attack moves to the END of
                    // the boundary's `ATTACK` list (europe N=18: first at
                    // boundaries 45-48, last at 49; id `47l4ldxb` -> `75imsa56`).
                    //
                    // The consequences for the device, all measured:
                    //   * that tick's CLAIMS belong to the OLD attack, on its OLD
                    //     carried heap/border/troops. Re-seeding the troops BEFORE
                    //     the tick inflates `attack_tiles_per_tick`'s budget (it
                    //     takes `troop_count`), the old attack claims one tile more
                    //     than the engine at boundary 49, and every later boundary
                    //     inherits the wrong plane. Measured with the
                    //     `OFCUDA_MATRIX_PRETICK_REALLOC=1` control below (europe
                    //     N=18): claims 290/291, hash 48/49, and the re-created
                    //     attack's device frontier 112/88 against the engine's
                    //     116/90 - i.e. the +2 tiles the pre-fix build showed at
                    //     that boundary are this, one tile of which no longer
                    //     happens once the frontier is rebuilt here;
                    //   * the NEW attack starts at the END of the tick with
                    //     `PseudoRandom::new(123)` (`attack.rs:47-57`), an EMPTY
                    //     `to_conquer`, and `refresh_to_conquer`
                    //     (`attack.rs:1265-1274`) over the owner's border set as
                    //     of that moment - so it is `attack_init` with
                    //     `fresh_prng = 1` against the CURRENT boundary's border
                    //     set, applied after this transition's ticks.
                    //
                    // Nothing here is taken on trust: the engine records the
                    // re-created attack's own `to_conquer` and `border_tiles`
                    // sizes, and the device's post-init values are compared
                    // against them (see REINIT below).
                    let reinit = cur
                        .attacks
                        .iter()
                        .find(|a| a.owner == snap.owner && a.target == snap.target)
                        .filter(|cs| !cs.attack_id.is_empty() && cs.attack_id != prev_id)
                        .cloned();
                    if let Some(cs) = reinit {
                        if freeze_at.is_some_and(|fa| b >= fa) {
                            sd_reinit += 1;
                            if sd_reinit <= 8 {
                                detail.push_str(&format!(
                                    "SELFDRIVE_MISS_REINIT boundary {b} (engine tick {tick}): the \
                                     engine re-created attack owner {} target {} (id {} -> {}) via \
                                     add_land_attack_from + merge_outgoing_land_attacks \
                                     (game.rs:2167) - the device keeps the old exec's state\
n",
                                    cs.owner, cs.target, prev_id, cs.attack_id
                                ));
                            }
                            continue;
                        }
                        // A/B CONTROL, not a code path: `OFCUDA_MATRIX_PRETICK_REALLOC=1`
                        // re-instates the OLD ordering (re-seed the re-issued attack's
                        // troops BEFORE the tick, when the tick still belongs to the old
                        // exec) so the +2 this pass closes can be reproduced and measured
                        // on demand instead of asserted. The matrix is run without it.
                        if std::env::var_os("OFCUDA_MATRIX_PRETICK_REALLOC").is_some() {
                            let mut th = d_troops.to_host_vec(&stream).map_err(es)?;
                            th[slots[i].idx] = cs.troops;
                            d_troops.copy_from_host(&stream, &th).map_err(es)?;
                            detail.push_str(&format!(
                                "PRETICK_REALLOC(control) boundary {b}: owner {} target {} \
                                 troops -> {:#018x} BEFORE the tick\n",
                                cs.owner,
                                cs.target,
                                cs.troops.to_bits()
                            ));
                        }
                        pending_reinit.push((i, cs, prev_id));
                    }
                }
                None => {
                    // SELF-DRIVE MEASUREMENT: from `freeze_at` on the device is not
                    // allowed to create an attack from `ATTACK` - it has no bot AI,
                    // so an attack the engine's AI issued is an instruction it
                    // cannot originate. Logged so the FIRST thing that breaks is
                    // named rather than inferred from the stop boundary.
                    if freeze_at.is_some_and(|fa| b >= fa) {
                        sd_create += 1;
                        if sd_create <= 8 {
                            detail.push_str(&format!(
                                "SELFDRIVE_MISS_CREATE boundary {b} (engine tick {tick}): the \
                                 engine created attack owner {} target {} id {} \
                                 (game.add_land_attack_from -> AttackExecution::init, \
                                 game.rs:1514) - the device has no AI to originate it\n",
                                snap.owner, snap.target, snap.attack_id
                            ));
                        }
                        continue;
                    }
                    // NEW attack: created on the device, stamped with the engine
                    // tick of the boundary BEFORE the one it appears at.
                    let idx = slots.len();
                    if idx >= MAX_SLOTS {
                        return Err(format!("more than {MAX_SLOTS} simultaneous attacks"));
                    }
                    let init_tick = if b >= 2 {
                        orc.boundaries[b - 2].engine_tick
                    } else {
                        prev.engine_tick
                    };
                    // Refresh the host mirrors from the DEVICE before touching
                    // one slot: uploading a stale whole-array copy resets every
                    // other slot's PRNG state and troop count.
                    troops_host = d_troops.to_host_vec(&stream).map_err(es)?;
                    prng_host = d_prng.to_host_vec(&stream).map_err(es)?;
                    let pr = Prng::new(SEED);
                    let mut pw = [0u32; 5];
                    pr.state_words(&mut pw);
                    prng_host[idx * 5..idx * 5 + 5].copy_from_slice(&pw);
                    d_prng.copy_from_host(&stream, &prng_host).map_err(es)?;
                    troops_host[idx] = snap.troops;
                    d_troops.copy_from_host(&stream, &troops_host).map_err(es)?;
                    unsafe {
                        module
                            .attack_init(
                                &stream,
                                cfg(kernels::BLOCK),
                                &d_terrain,
                                w,
                                h,
                                init_tick,
                                idx as u32,
                                snap.owner,
                                snap.target,
                                // `attack.rs:160-164`: a recorded `source_tile`
                                // seeds the frontier from the source tile's own
                                // neighbours instead of the owner's border set.
                                snap.source.unwrap_or(u32::MAX),
                                1, // fresh stream: a new exec mints
                                   // `PseudoRandom::new(123)` (attack.rs:47-57)
                                &d_plane,
                                &mut d_heap_tiles,
                                &mut d_heap_pri,
                                &mut d_border,
                                &mut d_prng,
                                &mut d_scal,
                                &mut d_out,
                                &d_oborder,
                                &d_ob_meta,
                            )
                            .map_err(es)?;
                    }
                    let o = d_out.to_host_vec(&stream).map_err(es)?;
                    let sc = d_scal.to_host_vec(&stream).map_err(es)?;
                    detail.push_str(&format!(
                        "INIT_ATTACK at boundary {b} (engine tick {init_tick}) slot {idx}: \
                         owner={} target={} troops_bits={:#018x} heap_len={} border_len={} \
                         heap_peak={} (device-computed)\n",
                        snap.owner,
                        snap.target,
                        snap.troops.to_bits(),
                        sc[idx * SCAL],
                        sc[idx * SCAL + 1],
                        o[3],
                    ));
                    slots.push(Slot {
                        owner: snap.owner,
                        target: snap.target,
                        idx,
                        alive: true,
                        created_at: b as u32,
                        troops_bits: snap.troops.to_bits(),
                        attack_id: snap.attack_id.clone(),
                    });
                    taken.push(false);
                    let i = slots.len() - 1;
                    taken[i] = true;
                    plan.push((i, snap.clone()));
                }
            }
        }

        // --- 3. evictions: my slot alive, the engine no longer lists it ---
        let mut evicted = Vec::new();
        if freeze_at.is_some_and(|fa| b >= fa) {
            // SELF-DRIVE MEASUREMENT: the device keeps its own attack list, so a
            // list-diff eviction is not applied at all. That is precisely the
            // difference between replaying a schedule and simulating a game.
            for (i, s) in slots.iter_mut().enumerate() {
                if s.alive && !taken[i] {
                    sd_evict += 1;
                    if sd_evict <= 8 {
                        detail.push_str(&format!(
                            "SELFDRIVE_MISS_EVICT boundary {b} (engine tick {tick}): owner {} \
                             target {} is gone from the engine's list and the device keeps it\
n",
                            s.owner, s.target
                        ));
                    }
                }
            }
        } else {
            for (i, s) in slots.iter_mut().enumerate() {
                if s.alive && !taken[i] {
                    s.alive = false;
                    row.engine_evictions += 1;
                    evicted.push(format!("sid {} slot {} at boundary {b}", s.owner, s.idx));
                }
            }
        }
        for e in &evicted {
            detail.push_str(&format!("ENGINE_EVICTION {e} (not modelled by the device)\n"));
            if first_div_at.is_none() {
                first_div = format!("boundary {b}: {e} (engine dropped an attack the device kept alive)");
                first_div_at = Some(b as u32);
            }
        }
        if !evicted.is_empty() {
            row.note = format!("{} engine_evictions", row.note).trim().to_string();
        }

        // --- 3a. the player-clusters pass -------------------------------------
        // `PlayerExecution::tick` (`execution/player.rs:92`) calls
        // `maybe_remove_clusters` (`execution/player_clusters.rs:358`), and the
        // player execs are ticked BEFORE the attack execs in
        // `execute_next_tick` - so this runs here, after the income phase and
        // before ANY attack of the tick. That ordering is the whole reason the
        // pass breaks parity: a cluster handed to a captor is already the
        // captor's when the tick's attacks pop, so the device's claim set
        // without it is the engine's MINUS those tiles, and the captor's
        // `owned_tiles` gains them at claim index 0.
        //
        // Everything the pass reads is the tick's OWN start state: the border
        // sets (`prev.borders` == what the engine's live `border_tiles` held
        // when the player execs ticked), each player's `last_cluster_calc` and
        // `last_tile_change` (`prev.cadence`), the roster's `simple_hash(id)`,
        // the engine's live attacks (`prev.attacks`, for
        // `largest_incoming_land_attack_from_neighbors`) and the friendly pairs.
        let mut mine: HashMap<u16, Vec<u32>> = HashMap::new();
        {
            let cst = kernels::device::CSTATE;
            let mut cadh = d_cad.to_host_vec(&stream).map_err(es)?;
            for p in &prev.players {
                if p.sid as usize >= COLS {
                    continue;
                }
                let i = p.sid as usize * cst;
                let (lcc, ltc, _) = prev.cadence.get(&p.sid).copied().unwrap_or((0, 0, false));
                cadh[i] = lcc;
                cadh[i + 1] = ltc;
                cadh[i + 2] = p.tiles.max(0) as u32;
                cadh[i + 3] = if p.tiles > 0 { 1 } else { 0 };
            }
            d_cad.copy_from_host(&stream, &cadh).map_err(es)?;
            // `Game::is_friendly(a, b)` as the engine evaluates it for this
            // tick: the oracle's own `FRIEND` pairs.
            let nf = prev.friends.len().min(kernels::device::CNB);
            let mut fr = vec![0xFFFF_FFFFu32; kernels::device::CNB + 1];
            for (k, (fa, fb)) in prev.friends.iter().take(nf).enumerate() {
                fr[k] = ((*fa as u32) << 16) | *fb as u32;
            }
            d_cfriends.copy_from_host(&stream, &fr).map_err(es)?;
            let na = prev.attacks.len().min(MAX_SLOTS);
            let mut aow = vec![0u16; MAX_SLOTS];
            let mut atg = vec![0u16; MAX_SLOTS];
            let mut atr = vec![0f64; MAX_SLOTS];
            for (k, at) in prev.attacks.iter().take(na).enumerate() {
                aow[k] = at.owner;
                atg[k] = at.target;
                atr[k] = at.troops;
            }
            d_atk_owner.copy_from_host(&stream, &aow).map_err(es)?;
            d_atk_target.copy_from_host(&stream, &atg).map_err(es)?;
            d_atk_troops.copy_from_host(&stream, &atr).map_err(es)?;
            unsafe {
                module
                    .cluster_pass(
                        &stream,
                        cfg(ONE),
                        &d_terrain,
                        w,
                        h,
                        tick,
                        cl_order.len() as u32,
                        &d_corder,
                        &d_cidhash,
                        &mut d_cad,
                        &mut d_plane,
                        &mut d_t2b,
                        &mut d_cmark,
                        &mut d_cbuf,
                        &mut d_clen,
                        &mut d_coff,
                        &mut d_cvis,
                        &mut d_cstack,
                        &mut d_cowned,
                        &mut d_crem,
                        &mut d_cout,
                        &d_oborder,
                        &d_ob_meta,
                        &d_cfriends,
                        nf as u32,
                        &d_atk_owner,
                        &d_atk_target,
                        &d_atk_troops,
                        na as u32,
                        &mut d_pst,
                    )
                    .map_err(es)?;
            }
            let co = d_cout.to_host_vec(&stream).map_err(es)?;
            row.cluster_fires += co[0] as usize;
            row.cluster_ovf += (co[2] + co[3] + co[4] + co[5]) as usize;
            if co[2] + co[3] + co[4] + co[5] > 0 {
                detail.push_str(&format!(
                    "CLUSTER_OVERFLOW boundary {b} (engine tick {tick}): border {} cluster {} \
                     owned {} neighbour {} - a capacity refusal drops a removal\n",
                    co[2], co[3], co[4], co[5]
                ));
            }
            let nrem = co[1] as usize;
            row.cluster_dev += nrem;
            let words = co[6] as usize;
            let rem = d_crem.to_host_vec(&stream).map_err(es)?;
            let mut woff = 0usize;
            let mut dev_this: Vec<(u16, u16, usize)> = Vec::new();
            let mut ri = 0usize;
            while ri < nrem && woff + 3 <= words {
                let victim = rem[woff] as u16;
                let captor = rem[woff + 1] as u16;
                let n = rem[woff + 2] as usize;
                woff += 3;
                if woff + n > words {
                    break;
                }
                let tiles = &rem[woff..woff + n];
                // The captor's `owned_tiles` gains these BEFORE any attack of
                // this tick (`game.conquer_one` pushes `owned_tiles`).
                mine.entry(captor).or_default().extend_from_slice(tiles);
                row.dev_claims += n;
                dev_this.push((victim, captor, n));
                woff += n;
                ri += 1;
                detail.push_str(&format!(
                    "CLUSTER_DEVICE_REMOVE boundary {b} (engine tick {tick}): victim {victim} \
                     captor {captor} tiles {n} first {:?}\n",
                    &tiles[..tiles.len().min(8)]
                ));
            }
            // --- the stream diff against the engine's own `ENG_CLUSTER_REMOVE` ---
            let eng_this = eng_cluster.get(&tick).cloned().unwrap_or_default();
            row.cluster_eng += eng_this.len();
            let mut d2 = dev_this.clone();
            let mut shown = 0usize;
            for x in &eng_this {
                if let Some(p) = d2.iter().position(|y| y == x) {
                    d2.remove(p);
                } else if shown < 8 {
                    shown += 1;
                    detail.push_str(&format!(
                        "CLUSTER_ENGINE_ONLY boundary {b} (engine tick {tick}): victim {} captor \
                         {} tiles {} - the engine removed a cluster the device did not\n",
                        x.0, x.1, x.2
                    ));
                    if first_div_at.is_none() {
                        first_div = format!(
                            "boundary {b} (engine tick {tick}) cluster removal victim {} captor {} \
                             tiles {} not reproduced by the device",
                            x.0, x.1, x.2
                        );
                        first_div_at = Some(b as u32);
                    }
                }
            }
            for x in &d2 {
                if shown < 8 {
                    shown += 1;
                    detail.push_str(&format!(
                        "CLUSTER_DEVICE_ONLY boundary {b} (engine tick {tick}): victim {} captor {} \
                         tiles {} - the device removed a cluster the engine did not\n",
                        x.0, x.1, x.2
                    ));
                }
            }
            row.cluster_match += eng_this.len() - d2.len();

            // --- the trigger itself, checked against the engine ---------------
            // After the pass the device's own `last_cluster_calc` must equal the
            // engine's post-tick value for EVERY player: the gate
            // (`player_clusters.rs:384-389`, including the `last_calc == 0`
            // seeding at :375-382) is then verified tick by tick, not inferred
            // from the removal stream alone.
            let cad2 = d_cad.to_host_vec(&stream).map_err(es)?;
            for p in &cur.players {
                let (elcc, _, _) = cur.cadence.get(&p.sid).copied().unwrap_or((0, 0, false));
                row.cadence_total += 1;
                let dlcc = cad2[p.sid as usize * cst];
                if dlcc == elcc {
                    row.cadence_match += 1;
                } else if row.cadence_total - row.cadence_match <= 8 {
                    detail.push_str(&format!(
                        "CLUSTER_CADENCE boundary {b} (engine tick {tick}) player {}: device \
                         last_cluster_calc {dlcc} vs engine {elcc}\n",
                        p.sid
                    ));
                }
            }
        }

        // --- 3b. transport-ship landings -------------------------------------
        // `TransportShipExecution::land` (`transport_ship.rs:374-401`) CONQUERS
        // the shore tile the boat reaches (`game.conquer`, :383) and only then
        // creates an attack whose `source_tile` IS that tile (:386). The device
        // models no units, so a landing is visible in the record only as an
        // attack that carries a `source_tile` and is NOT in the previous
        // boundary's attack list - and the recorded `source` is exactly the tile
        // `land()` conquered in THIS tick. Applying it before the tick puts the
        // tile on the plane for this boundary's hash, like every other conquest
        // of the same tick.
        // The conquer itself is ORDERED: `execute_next_tick` ticks `execs` in
        // list order (`game.rs:3662-3669`) and `TransportShipExecution::land`
        // (`transport_ship.rs:374-401`) calls `game.conquer(owner, dst)` at the
        // SHIP's own exec position, so only attacks whose exec index is greater
        // than the ship's can see `dst` as owned - the engine's
        // `add_neighbors`/`refresh_to_conquer` owner filter
        // (`attack.rs:1300-1340`) rejects it for the ones that ticked earlier.
        // The oracle emits each ship's `(exec_index, owner, dst, natk)` where
        // `natk` is the number of live attacks ahead of it, so the landing is
        // applied after exactly `natk` plan entries instead of before all of
        // them. Applying it first is what dropped tile 422259 from player 108's
        // frontier at boundary 206 (N=488/250t): player 51's ship sat at exec
        // 998 and 108's attack at exec 989, so the engine's 108 enqueued 422259
        // while the device already saw it as 51's.
        // `mine` is created in 3a (the clusters pass fills the captors' entries
        // first, because the player execs tick BEFORE the attacks) and every
        // landing/attack claim appends to it here. It must NOT be re-declared:
        // a second `let mut mine` would shadow the cluster claims and drop the
        // captor's cluster tiles from the claim comparison.
        // (natk, owner, src, attack_id, heap_len, border_len)
        let mut landings: Vec<(usize, u16, u32, String, usize, usize)> = Vec::new();
        let mut used_ships: Vec<usize> = Vec::new();
        for cs in &cur.attacks {
            let Some(src) = cs.source else { continue };
            if prev
                .attacks
                .iter()
                .any(|p| p.owner == cs.owner && p.target == cs.target)
            {
                continue; // already bound: not a landing of THIS tick
            }
            let hit = cur
                .transports
                .iter()
                .enumerate()
                .find(|(k, (_, o, d, _))| {
                    *o == cs.owner && *d == src as i64 && !used_ships.contains(k)
                })
                .map(|(k, (_, _, _, natk))| (k, *natk));
            let natk = match hit {
                Some((k, n)) => {
                    used_ships.push(k);
                    n
                }
                None => 0, // no ship recorded: keep the old "before every attack"
            };
            // A/B CONTROL, not a code path: `OFCUDA_MATRIX_LAND_FIRST=1` restores
            // the pre-fix ordering (every landing applied before every attack,
            // i.e. `natk = 0`) so the effect of the exec-order fix on any cell can
            // be measured on demand instead of argued. The matrix is run without it.
            let natk = if std::env::var_os("OFCUDA_MATRIX_LAND_FIRST").is_some() {
                0
            } else {
                natk
            };
            landings.push((natk, cs.owner, src, cs.attack_id.clone(), cs.heap_len, cs.border_len));
        }
        landings.sort_by_key(|x| x.0);

        // --- 3c. the friendship table the ATTACK TICKS read -----------------
        // `cluster_pass` above consumed `prev.friends`, because the cluster
        // removal it models reads the tick's own START state. The attack tick's
        // alliance retreat (`attack.rs:222-229`) instead reads the state the
        // engine holds DURING this boundary: `FRIEND` for THIS boundary, which
        // the oracle emits before the execs run. So the same device table is
        // re-uploaded here, after the cluster pass and before the ticks.
        let nf_t = cur.friends.len().min(kernels::device::CNB);
        {
            let mut fr = vec![0xFFFF_FFFFu32; kernels::device::CNB + 1];
            for (k, (fa, fb)) in cur.friends.iter().take(nf_t).enumerate() {
                fr[k] = ((*fa as u32) << 16) | *fb as u32;
            }
            d_cfriends.copy_from_host(&stream, &fr).map_err(es)?;
        }

        // --- 4. one device launch per live attack, in the engine's order ---
        let mut troop_pairs: Vec<(usize, u64)> = Vec::new();
        let plan_len = plan.len();
        let mut nland = 0usize;
        for p in 0..=plan_len {
            // Landings whose ship ticked ahead of the p-th attack: the engine
            // applies them exactly here (see 3b), so the attacks that ticked
            // before `p` did NOT see the landed tile as owned.
            while nland < landings.len() && landings[nland].0 <= p {
                let (_, lowner, lsrc, laid, lhl, lbl) = landings[nland].clone();
                unsafe {
                    module
                        .land_dev(
                            &stream,
                            cfg(ONE),
                            lsrc,
                            lowner,
                            &mut d_plane,
                            &mut d_claims,
                            &mut d_out,
                        )
                        .map_err(es)?;
                }
                let lo = d_out.to_host_vec(&stream).map_err(es)?;
                if lo[0] > 0 {
                    let cl = d_claims.to_host_vec(&stream).map_err(es)?;
                    mine.entry(lowner).or_default().push(cl[0]);
                    row.dev_claims += 1;
                }
                detail.push_str(&format!(
                    "TRANSPORT_LANDING boundary {b} (engine tick {tick}): owner {} landed on tile {} \
                     (source_tile of attack {}); engine frontier heap {} border {}; applied after {} of {} plan attacks (ship exec order)\n",
                    lowner, lsrc, laid, lhl, lbl, p, plan_len
                ));
                nland += 1;
            }
            if p == plan_len {
                break;
            }
            let (i, _snap) = &plan[p];
            let s = &slots[*i];
            if !s.alive {
                continue;
            }
            unsafe {
                module
                    .attack_tick(
                        &stream,
                        cfg(kernels::BLOCK),
                        &d_terrain,
                        w,
                        h,
                        tick,
                        s.idx as u32,
                        s.owner,
                        s.target,
                        // The engine's troop-loss branch is chosen by the
                        // ATTACKER's player type (`game.rs:890`): bots take the
                        // bot branch, nations/humans the other one. The type
                        // comes from the engine roster in the oracle dump, not
                        // from a guess about which cells exist.
                        if bot_sids.contains(&s.owner) { 1 } else { 0 },
                        &mut d_pst,
                        &d_defsig,
                        &mut d_plane,
                        &mut d_heap_tiles,
                        &mut d_heap_pri,
                        &mut d_border,
                        &mut d_prng,
                        &mut d_troops,
                        &mut d_scal,
                        &mut d_out,
                        &mut d_claims,
                        &d_oborder,
                        &d_ob_meta,
                        &d_cfriends,
                        nf_t as u32,
                    )
                    .map_err(es)?;
            }
            let o = d_out.to_host_vec(&stream).map_err(es)?;
            let ncl = (o[0] as usize).min(MAXC);
            if ncl > 0 {
                let cl = d_claims.to_host_vec(&stream).map_err(es)?;
                mine.entry(s.owner).or_default().extend_from_slice(&cl[..ncl]);
                row.dev_claims += ncl;
            }
            troop_pairs.push((s.idx, 0));
            if o[1] == 0 {
                // Tell the alliance retreat apart from an ordinary death instead
                // of assuming it: the same rule the device just applied is
                // re-evaluated on the host from the boundary's own FRIEND row.
                if s.target != 0
                    && s.target != s.owner
                    && cur
                        .friends
                        .iter()
                        .any(|(fa, fb)| *fa == s.owner && *fb == s.target)
                {
                    row.friendly_retreats += 1;
                    detail.push_str(&format!(
                        "FRIENDLY_RETREAT boundary {b} (engine tick {tick}): owner {} target {} \
                         retreated at the top of its own tick with 0 claims \
                         (attack.rs:222-229, is_friendly)\n",
                        s.owner, s.target
                    ));
                }
                slots[*i].alive = false;
            }
        }

        // --- 4b. engine RE-CREATEs, applied AFTER the ticks -------------------
        // `execute_next_tick` (`game.rs:3657-3700`) ticks every existing exec and
        // only THEN initialises the new ones, so a re-created attack's frontier is
        // built from the world as of the END of the tick. Doing it here (rather
        // than before the tick) is the whole fix: the tick belongs to the OLD
        // attack, and re-seeding troops first inflated the budget and over-claimed.
        for (i, cs, prev_id) in &pending_reinit {
            let s = &mut slots[*i];
            let sidx = s.idx;
            // The engine's border set as of THIS boundary: that is the set
            // `refresh_to_conquer` walked at the end of the tick just simulated.
            let mut coff = 0usize;
            let mut coborder = vec![0u32; obcap];
            let mut cob_meta = vec![0u32; 2 * COLS];
            for (osid, tiles) in &cur.borders {
                if *osid as usize >= COLS {
                    return Err(format!("owner small_id {osid} >= COLS {COLS}"));
                }
                let n = tiles.len().min(obcap - coff);
                coborder[coff..coff + n].copy_from_slice(&tiles[..n]);
                cob_meta[*osid as usize * 2] = coff as u32;
                cob_meta[*osid as usize * 2 + 1] = n as u32;
                coff += n;
            }
            let d_coborder =
                DeviceBuffer::from_host(&stream, &coborder[..coff.max(1)]).map_err(es)?;
            let mut d_cob_meta = DeviceBuffer::<u32>::zeroed(&stream, 2 * COLS).map_err(es)?;
            d_cob_meta.copy_from_host(&stream, &cob_meta).map_err(es)?;

            troops_host = d_troops.to_host_vec(&stream).map_err(es)?;
            troops_host[sidx] = cs.troops;
            d_troops.copy_from_host(&stream, &troops_host).map_err(es)?;

            // The init ran inside the tick that produced THIS boundary, so its
            // `game.ticks()` stamp is the boundary we are leaving - exactly the
            // tick that just ran.
            let init_tick = prev.engine_tick;
            unsafe {
                module
                    .attack_init(
                        &stream,
                        cfg(kernels::BLOCK),
                        &d_terrain,
                        w,
                        h,
                        init_tick,
                        sidx as u32,
                        s.owner,
                        s.target,
                        cs.source.unwrap_or(u32::MAX),
                        1, // RE-CREATE: fresh `PseudoRandom::new(123)`
                        &d_plane,
                        &mut d_heap_tiles,
                        &mut d_heap_pri,
                        &mut d_border,
                        &mut d_prng,
                        &mut d_scal,
                        &mut d_out,
                        &d_coborder,
                        &d_cob_meta,
                    )
                    .map_err(es)?;
            }
            let sc = d_scal.to_host_vec(&stream).map_err(es)?;
            let dev_heap = sc[sidx * SCAL] as usize;
            let dev_border = sc[sidx * SCAL + 1] as usize;
            // The engine recorded the re-created attack's OWN frontier sizes, so
            // the device is checked against them rather than assumed.
            let agree = dev_heap == cs.heap_len && dev_border == cs.border_len;
            row.reinits += 1;
            row.reinit_checked += 1;
            row.reinit_agree += agree as usize;
            detail.push_str(&format!(
                "REINIT boundary {b} (engine re-created the attack in the tick stamped \
                 {init_tick}): owner {} target {} id {} -> {} troops {:#018x} -> {:#018x} | \
                 device post-init heap {dev_heap} vs engine {} border {dev_border} vs engine {} \
                 -> {}\n",
                s.owner,
                s.target,
                if prev_id.is_empty() { "?" } else { prev_id },
                cs.attack_id,
                s.troops_bits,
                cs.troops.to_bits(),
                cs.heap_len,
                cs.border_len,
                if agree { "AGREE" } else { "DISAGREE" },
            ));
            if !agree && first_div_at.is_none() {
                first_div = format!(
                    "boundary {b}: re-created attack owner {} target {}: device frontier \
                     (heap {dev_heap} border {dev_border}) != engine (heap {} border {})",
                    s.owner, s.target, cs.heap_len, cs.border_len
                );
                first_div_at = Some(b as u32);
            }
            s.attack_id = cs.attack_id.clone();
            s.troops_bits = cs.troops.to_bits();
            // `attack_init` re-arms the slot, and the engine's attack exists
            // regardless of how the OLD one's last tick ended.
            s.alive = true;
        }

        // troop count after the tick, for the surviving attacks
        let troops_after = d_troops.to_host_vec(&stream).map_err(es)?;

        // FRONTIER LEAK CHECK - diagnostic, never a pass condition. A live
        // attack's `to_conquer` size is recorded by the engine (`ATTACK ...`),
        // and the device's slot holds the same quantity after the same tick
        // (`scal[soff]`). While they agree the two frontiers hold the same
        // NUMBER of tiles; the first tick that pops a tile only one of them
        // holds is where the claim sets part, so a drift is worth naming at the
        // boundary it starts rather than at the boundary it is popped.
        {
            let scal_after = d_scal.to_host_vec(&stream).map_err(es)?;
            // MEASUREMENT-ONLY diagnostic (`OF_LEAK_OWNER=<sid>`): dump the
            // device's carried frontier (tiles + f32 priority bits) and its
            // attack border size for one owner at every boundary, next to the
            // engine's own recorded `ATTACK` sizes. This is what names whether a
            // frontier defect is a different SET (a tile present on one side
            // only) or the SAME set in a different ORDER (a priority/tie-break
            // difference). Never a pass condition.
            let leak_owner: Option<u16> = std::env::var("OF_LEAK_OWNER")
                .ok()
                .and_then(|s| s.parse().ok());
            if let Some(lo) = leak_owner {
                for a in &cur.attacks {
                    if a.owner != lo {
                        continue;
                    }
                    let found = slots
                        .iter()
                        .find(|s| s.alive && s.owner == a.owner && s.target == a.target);
                    let Some(s) = found else { continue };
                    let dev_heap = scal_after[s.idx * SCAL] as usize;
                    let dev_border = scal_after[s.idx * SCAL + 1] as usize;
                    let htiles = d_heap_tiles.to_host_vec(&stream).map_err(es)?;
                    let hpri = d_heap_pri.to_host_vec(&stream).map_err(es)?;
                    let base = s.idx * HEAP_CAP;
                    let mut tiles: Vec<String> = Vec::with_capacity(dev_heap);
                    for j in 0..dev_heap.min(HEAP_CAP) {
                        tiles.push(format!("{}:{:08x}", htiles[base + j], hpri[base + j].to_bits()));
                    }
                    detail.push_str(&format!(
                        "HEAPDUMP boundary {b} (engine tick {tick}) owner {lo} target {}: \
                         dev_heap {dev_heap} eng_heap {} dev_border {dev_border} eng_border {} | {}\n",
                        a.target,
                        a.heap_len,
                        a.border_len,
                        tiles.join(" ")
                    ));
                }
            }
            for a in &cur.attacks {
                let found = slots
                    .iter()
                    .find(|s| s.alive && s.owner == a.owner && s.target == a.target);
                let Some(s) = found else { continue };
                let dev_heap = scal_after[s.idx * SCAL] as usize;
                if dev_heap != a.heap_len && !frontier_seen.contains_key(&(a.owner, a.target)) {
                    frontier_seen.insert((a.owner, a.target), dev_heap);
                    detail.push_str(&format!(
                        "FRONTIER_LEAK boundary {b} (engine tick {tick}): owner {} target {} device \
                         heap {} vs engine {} (engine border {})\n",
                        a.owner, a.target, dev_heap, a.heap_len, a.border_len
                    ));
                }
            }
        }

        // --- 5. the plane: device hash + host counts (same bytes) ---
        unsafe {
            module
                .plane_hash(&stream, cfg(ONE), &d_plane, wh as u32, &mut d_hash)
                .map_err(es)?;
        }
        let dev_hash = d_hash.to_host_vec(&stream).map_err(es)?[0];
        let plane = d_plane.to_host_vec(&stream).map_err(es)?;
        let host_hash = fnv1a_u16_le(&plane);
        if dev_hash != host_hash {
            detail.push_str(&format!(
                "HASH_INTERNAL boundary {b}: device kernel {dev_hash:#018x} vs host over the \
                 copied plane {host_hash:#018x} - the two disagree\n"
            ));
        }
        if a.dump_planes {
            frames.extend_from_slice(&plane);
        }

        // --- 6. compare against the engine ---
        row.ran_ticks = b as u32;
        row.hash_total += 1;
        row.hash_match += (dev_hash == cur.hash) as usize;
        if dev_hash != cur.hash && first_div_at.is_none() {
            let cnt = owners_of(&plane);
            let eng_counts: HashMap<u16, i32> =
                cur.players.iter().map(|p| (p.sid, p.tiles)).collect();
            let mut diff = String::new();
            for (sid, c) in &eng_counts {
                let mine_n = cnt.get(sid).map(|v| v.len()).unwrap_or(0) as i32;
                if mine_n != *c {
                    diff = format!("player {sid} engine {c} tiles vs device {mine_n}");
                    break;
                }
            }
            first_div = format!(
                "boundary {b} (engine tick {tick}): plane hash {dev_hash:#018x} != engine \
                 {:#018x}; {diff}",
                cur.hash
            );
            first_div_at = Some(b as u32);
        }

        // per-player owned counts vs the engine's PLAYER lines
        let cnt = owners_of(&plane);
        for p in &cur.players {
            row.count_total += 1;
            let mine_n = cnt.get(&p.sid).map(|v| v.len()).unwrap_or(0) as i32;
            if mine_n == p.tiles {
                row.count_match += 1;
            } else if first_div_at.is_none() {
                first_div = format!(
                    "boundary {b} (engine tick {tick}): player {} owned count engine {} vs \
                     device {}",
                    p.sid, p.tiles, mine_n
                );
                first_div_at = Some(b as u32);
            }
        }

        // claims, per carrier, in the engine's own order
        let churn: Vec<u16> = cur.churn.iter().map(|(s, _)| *s).collect();
        let mut owners_seen: Vec<u16> = cur.claims.keys().copied().collect();
        for k in mine.keys() {
            if !owners_seen.contains(k) {
                owners_seen.push(*k);
            }
        }
        owners_seen.sort_unstable();
        for sid in owners_seen {
            if churn.contains(&sid) {
                row.churn_skipped += 1;
                continue;
            }
            let eng = cur.claims.get(&sid).cloned().unwrap_or_default();
            let dev = mine.get(&sid).cloned().unwrap_or_default();
            row.tick_total += 1;
            if eng == dev {
                row.tick_match += 1;
                continue;
            }
            let cause = if eng.is_empty() {
                "device claimed tiles the engine's tail-diff showed none for".to_string()
            } else if dev.is_empty() {
                "device claimed nothing where the engine claimed".to_string()
            } else {
                let fs: std::collections::HashSet<u32> = eng.iter().copied().collect();
                let ds: std::collections::HashSet<u32> = dev.iter().copied().collect();
                if fs == ds {
                    "same tiles, different ORDER".to_string()
                } else {
                    let only_e: Vec<u32> = eng.iter().copied().filter(|t| !ds.contains(t)).take(4).collect();
                    let only_d: Vec<u32> = dev.iter().copied().filter(|t| !fs.contains(t)).take(4).collect();
                    format!("tile sets differ: engine-only {only_e:?} device-only {only_d:?}")
                }
            };
            let at = eng
                .iter()
                .zip(dev.iter())
                .position(|(x, y)| x != y)
                .unwrap_or(eng.len().min(dev.len()));
            detail.push_str(&format!(
                "CLAIM_DIVERGENCE boundary {b} (engine tick {tick}) player {sid}: engine {} \
                 tiles, device {} tiles; first index {at}; cause: {cause}\n  engine {}\n  device {}\n",
                eng.len(),
                dev.len(),
                preview(&eng),
                preview(&dev),
            ));
            if first_div_at.is_none() {
                let tile = eng.get(at).or(dev.get(at)).copied().unwrap_or(u32::MAX);
                first_div = format!(
                    "boundary {b} (engine tick {tick}) player {sid} tile {tile} at claim index \
                     {at}: {cause}"
                );
                first_div_at = Some(b as u32);
            }
        }

        // troops after the tick, where the engine still lists the attack
        let mut troop_log = String::new();
        for snap in &cur.attacks {
            // Prefer the LIVE slot for this `(owner, target)`. `slots` is never
            // compacted, so a re-created attack leaves its dead predecessor
            // (troops already zeroed on the death path) at a LOWER index than
            // the live successor; matching the first slot blindly reported the
            // dead sibling's 0x0 as a `device` value and invented a troop
            // divergence that the simulation does not have. Matching the live
            // slot reflects what the device is actually running.
            //
            // A slot that EXISTS but is dead while the engine still lists the
            // attack is a REAL divergence (the device dropped an attack the
            // engine kept) and must still be reported - so the dead slot's
            // zeroed troops are used as the fallback, not dropped as a
            // "not created yet" deferral.
            let live = slots
                .iter()
                .find(|s| s.alive && s.owner == snap.owner && s.target == snap.target);
            let mine_bits = match live {
                Some(s) => Some(troops_after[s.idx].to_bits()),
                None => {
                    // No LIVE slot models this `(owner, target)`. Two very
                    // different situations land here and they must not be
                    // confused:
                    //
                    //  * the engine STARTED this attack this tick (a fresh
                    //    attack, or a re-create whose predecessor already died)
                    //    -> the device creates its slot in the NEXT transition
                    //    (`init` runs at the very end of the tick that produced
                    //    this boundary), a deferral, counted as neither. A stale
                    //    DEAD predecessor left at a lower index by an earlier
                    //    instance of the same `(owner, target)` must NOT be read
                    //    as the live attack - that is the phantom `device 0x0`
                    //    this check used to report.
                    //
                    //  * the device WAS modelling this attack at the start of
                    //    the tick (`prev.attacks` lists it) but its slot is dead
                    //    now -> the device dropped an attack the engine kept
                    //    alive. That IS a real divergence: report it against the
                    //    dead slot's zeroed troops.
                    let modelled_before = prev
                        .attacks
                        .iter()
                        .any(|p| p.owner == snap.owner && p.target == snap.target);
                    if modelled_before {
                        slots
                            .iter()
                            .find(|s| s.owner == snap.owner && s.target == snap.target)
                            .map(|s| troops_after[s.idx].to_bits())
                    } else {
                        None
                    }
                }
            };
            let Some(mine_bits) = mine_bits else {
                // First boundary this attack appears at: the device creates it
                // in the NEXT transition (its `init` runs at the very end of the
                // tick that produced this boundary, so it has drawn nothing
                // yet). Not a mismatch - a deferral, and counted as neither.
                troop_log.push_str(&format!(
                    " [{}/{} eng {:#018x} device: not created yet (deferred to the next transition)]",
                    snap.owner,
                    snap.target,
                    snap.troops.to_bits()
                ));
                continue;
            };
            row.troops_total += 1;
            let hit = mine_bits == snap.troops.to_bits();
            row.troops_match += hit as usize;
            troop_log.push_str(&format!(
                " [{}/{} eng {:#018x} dev {:#018x} {}]",
                snap.owner,
                snap.target,
                snap.troops.to_bits(),
                mine_bits,
                if hit { "ok" } else { "MISMATCH" }
            ));
            if !hit && first_div_at.is_none() {
                first_div = format!(
                    "boundary {b} (engine tick {tick}): attack owner {} target {} troops engine \
                     {:#018x} vs device {:#018x} (troop arithmetic diverged)",
                    snap.owner,
                    snap.target,
                    snap.troops.to_bits(),
                    mine_bits,
                );
                first_div_at = Some(b as u32);
            }
        }
        detail.push_str(&format!("TROOPS boundary {b}:{troop_log}\n"));

        detail.push_str(&format!(
            "TICK boundary {b} engine_tick {tick}: dev_hash {dev_hash:#018x} engine_hash \
             {:#018x} match={} | claims {} | dev_claims_total {} | live_attacks {} | \
             engine_attacks {}\n",
            cur.hash,
            yn(dev_hash == cur.hash),
            cur.claims.values().map(|v| v.len()).sum::<usize>(),
            row.dev_claims,
            slots.iter().filter(|s| s.alive).count(),
            cur.attacks.len(),
        ));
        if let Some(at) = first_div_at {
            if at <= b as u32 && row.hash_match + row.tick_match < row.hash_total + row.tick_total {
                break; // stop at the first divergence: the rest is not evidence
            }
        }
    }

    // --- device state diagnostics (persistent scalars, read back once) ---
    let scal_end = d_scal.to_host_vec(&stream).map_err(es)?;
    let mut max_peak = 0u32;
    let mut drops = 0u64;
    let mut max_border = 0u32;
    let mut live = 0usize;
    let mut refused = 0u64;
    for s in 0..slots.len() {
        max_peak = max_peak.max(scal_end[s * SCAL + 5]);
        drops += scal_end[s * SCAL + 4] as u64;
        refused += scal_end[s * SCAL + 6] as u64;
        max_border = max_border.max(scal_end[s * SCAL + 1]);
        live += (scal_end[s * SCAL + 3] == 1) as usize;
    }
    detail.push_str(&format!(
        "\n### DEVICE RESOURCES (gpu_env's own limits, measured here)\n\
         slots used {} (max {MAX_SLOTS}), live at the end {}, max heap peak {} of HEAP_CAP {HEAP_CAP}, \
         max border_len {} of BC {BC}, claim-list overflows {drops}, \
         heap candidates refused at HEAP_CAP {refused}\n",
        slots.len(),
        live,
        max_peak,
        max_border,
    ));
    if row.friendly_retreats > 0 {
        detail.push_str(&format!(
            "FRIENDLY_RETREATS total {} (alliance retreats applied on the boundary the engine \
             killed them)\n",
            row.friendly_retreats
        ));
    }
    if freeze_at.is_some() {
        detail.push_str(&format!(
            "SELFDRIVE totals: {sd_create} engine-created attacks not created, {sd_evict} \
             evictions not applied, {sd_reinit} re-creates not re-stamped\n"
        ));
        row.note = format!(
            "{} selfdrive(no attack reseed from {}) create_miss {sd_create} evict_miss {sd_evict} \
             reinit_miss {sd_reinit}",
            row.note,
            freeze_at.unwrap()
        )
        .trim()
        .to_string();
    }
    if drops > 0 {
        row.note = format!("{} claim-list overflows {drops}", row.note).trim().to_string();
    }
    if refused > 0 {
        row.note = format!("{} heap cap refusals {refused}", row.note).trim().to_string();
    }
    // A capacity refusal is a CORRECTNESS DEFECT, not a statistic: a refused
    // frontier candidate is a silently dropped tile, i.e. a parity break. That is
    // exactly how the 256-entry cap first desynced claim order (tick 558) and how
    // 1024 did it again in this window (boundary 806). `border_insert` refuses a
    // tile when the set is full and leaves the length AT the cap, so `>=` (not
    // `>`) is the value that means "a tile was dropped".
    if max_border as usize >= BC {
        return Err(format!(
            "border set reached BC={BC} (max border_len {max_border}): a tile was refused, which is a \
             parity break - raise BC from a measured border set, never by guesswork"
        )
        .into());
    }

    row.first_div = first_div;
    detail.push_str(&format!(
        "\n### RESULT\ninit_sets {}\ninit_hash {}\ntick claims matched {}/{}\nhash matched {}/{}\n\
         owned counts matched {}/{}\ntroops matched {}/{}\nchurn-skipped owners {}\n\
         engine evictions {}\nengine re-creates followed {}\ndevice frontier agreed with the \
         engine's recorded to_conquer/border_tiles {}/{}\ndevice claims total {}\n\
         first divergence: {}\nnote: {}\n",
        yn(row.init_sets),
        yn(row.init_hash),
        row.tick_match,
        row.tick_total,
        row.hash_match,
        row.hash_total,
        row.count_match,
        row.count_total,
        row.troops_match,
        row.troops_total,
        row.churn_skipped,
        row.engine_evictions,
        row.reinits,
        row.reinit_agree,
        row.reinit_checked,
        row.dev_claims,
        row.first_div,
        row.note
    ));
    write_cell(a, c, &detail, Some(&frames), Some(&terrain))?;
    // FAIL LOUDLY on a saturated frontier. The detail file above already holds
    // the evidence, but a cell that carried a capacity refusal to completion must
    // not be able to read as anything but failed: every refused candidate is a
    // dropped tile, so the plane beyond it is meaningless no matter how many
    // boundaries "matched" before it.
    if refused > 0 {
        return Err(format!(
            "HEAP CAPACITY REFUSAL: {refused} frontier candidates were refused at HEAP_CAP={HEAP_CAP} \
             (device high-water mark {max_peak}); a refused candidate is a dropped tile = a parity \
             break, so this cell is failed deliberately. Raise HEAP_CAP in \
             ofcuda_tick/src/core_impl.rs from a measured peak with real headroom - this run measured \
             {max_peak}."
        )
        .into());
    }
    Ok(row)
}

fn preview(v: &[u32]) -> String {
    if v.is_empty() {
        return "[]".into();
    }
    let head: Vec<String> = v.iter().take(12).map(|x| x.to_string()).collect();
    format!(
        "[{}{}] ({} tiles)",
        head.join(" "),
        if v.len() > 12 { " ..." } else { "" },
        v.len()
    )
}

fn es<E: std::fmt::Display>(e: E) -> String {
    format!("cuda: {e}")
}

fn write_cell(
    a: &Args,
    c: &Cell,
    detail: &str,
    frames: Option<&[u16]>,
    terrain: Option<&[u8]>,
) -> Result<(), String> {
    let dir = a.out.join(format!(
        "{}_n{}_nat{}_t{}",
        c.map.to_lowercase(),
        c.agents,
        c.nations,
        c.ticks
    ));
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("result.txt"), detail).map_err(|e| e.to_string())?;
    if let (Some(frames), Some(terrain)) = (frames, terrain) {
        let mut bytes = Vec::with_capacity(frames.len() * 2);
        for v in frames {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(dir.join("planes.bin"), &bytes).map_err(|e| e.to_string())?;
        std::fs::write(dir.join("terrain.bin"), terrain).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn append_row(a: &Args, row: &Row) -> Result<(), String> {
    std::fs::create_dir_all(&a.out).map_err(|e| e.to_string())?;
    let p = a.out.join("cells.tsv");
    if !p.exists() {
        std::fs::write(&p, format!("{}\n", Row::tsv_header())).map_err(|e| e.to_string())?;
    }
    let mut cur = std::fs::read_to_string(&p).map_err(|e| e.to_string())?;
    cur.push_str(&row.tsv());
    cur.push('\n');
    std::fs::write(&p, cur).map_err(|e| e.to_string())?;
    Ok(())
}

fn parse_cells_file(a: &Args) -> Result<Vec<Cell>, String> {
    let p = a.cells_file.as_ref().unwrap();
    let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.len() < 2 {
            return Err(format!("{}:{}: need <map> <agents> [nations] [ticks]", p.display(), n + 1));
        }
        out.push(Cell {
            map: t[0].to_string(),
            agents: t[1].parse().map_err(|e| format!("{}:{}: {e}", p.display(), n + 1))?,
            nations: if t.len() > 2 {
                t[2].parse().map_err(|e| format!("agents: {e}"))?
            } else {
                a.nations
            },
            ticks: if t.len() > 3 {
                t[3].parse().map_err(|e| format!("ticks: {e}"))?
            } else {
                a.ticks
            },
        });
    }
    Ok(out)
}

fn main() {
    let a = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    let cells = match &a.cells_file {
        Some(_) => match parse_cells_file(&a) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(2);
            }
        },
        None => vec![Cell {
            map: a.map.clone(),
            agents: a.agents,
            nations: a.nations,
            ticks: a.ticks,
        }],
    };
    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut skip = 0usize;
    for c in &cells {
        eprintln!(
            "=== cell {} N={} nations={} ticks={}",
            c.map, c.agents, c.nations, c.ticks
        );
        // Nations ARE ported (`ofcuda_prng::nation_spawns`): the nation layer is
        // driven from the same reference file as the bots, its spawn tile/tick/
        // owned set are checked against the engine at boundary 0, and its
        // attacker type is fed to the tick kernel so a nation's troops loss is
        // the engine's non-bot branch (game.rs:890).
        match run_cell(&a, c) {
            Ok(row) => {
                if let Err(e) = append_row(&a, &row) {
                    eprintln!("error writing cells.tsv: {e}");
                    std::process::exit(1);
                }
                let ok = row.pass();
                if ok {
                    pass += 1;
                } else {
                    fail += 1;
                }
                println!(
                    "{}\t{}\t{}\t{}\tinit {}/{}\tclaims {}/{}\thash {}/{}\tcounts {}/{}\t{}",
                    row.map,
                    format!("{}x{}", row.w, row.h),
                    row.agents,
                    row.ran_ticks,
                    yn(row.init_sets),
                    yn(row.init_hash),
                    row.tick_match,
                    row.tick_total,
                    row.hash_match,
                    row.hash_total,
                    row.count_match,
                    row.count_total,
                    row.first_div,
                );
            }
            Err(e) => {
                fail += 1;
                eprintln!("CELL FAILED {} N={}: {e}", c.map, c.agents);
                let mut row = Row {
                    map: c.map.to_lowercase(),
                    agents: c.agents,
                    nations: c.nations,
                    ticks: c.ticks,
                    first_div: format!("not driven: {e}"),
                    note: "cell could not be driven".into(),
                    ..Default::default()
                };
                row.note = format!("cell could not be driven: {e}");
                let _ = append_row(&a, &row);
            }
        }
    }
    eprintln!(
        "cells: {pass} passed, {fail} failed/not-driven, {skip} skipped"
    );
}

#[allow(dead_code)]
fn unused(_: &Path) {}
