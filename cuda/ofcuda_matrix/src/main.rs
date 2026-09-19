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
    tick_match: usize,
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
    dev_claims: usize,
    note: String,
}

impl Row {
    fn tsv_header() -> &'static str {
        "map\tw\th\tN\tnations\tticks\tinit_sets\tinit_hash\ttick_match\ttick_total\thash_match\thash_total\tcount_match\tcount_total\ttroops_match\ttroops_total\tchurn_skipped\tengine_evictions\treinits\tdev_claims\tfirst_div\tnote"
    }
    fn tsv(&self) -> String {
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
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
            self.dev_claims,
            self.first_div,
            self.note,
        )
    }
    fn pass(&self) -> bool {
        self.init_sets
            && self.init_hash
            && self.tick_match == self.tick_total
            && self.hash_match == self.hash_total
            && self.count_match == self.count_total
    }
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
    row.init_sets = sets_ok;

    let mut detail = String::new();
    detail.push_str(&format!(
        "# ofcuda_matrix cell {} n={} nations={} ticks={}\n\
         # engine oracle {}\n# spawn init    {}\n# map          {} ({}x{}, land {} from BYTES)\n\
         # engine land from manifest {}\n\
         ### INIT (engine spawn output vs the CUDA spawn/init dump)\n\
         init_plane_hash {:#018x} engine_boundary0_hash {:#018x} match {}\n\
         per-player owned sets match: {}\n\
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
    detail.push_str(&format!(
        "SELFCHECK engine owned_tiles-vs-plane diff {} (nonzero => the engine's own \
         two representations disagree; PLAYER counts are then not a plane count)\n",
        orc.selfcheck_diff
    ));

    if init.bots.iter().filter(|b| b.2 < 0).count() > 0 {
        row.note = "PORT FAILED TO PLACE SOME BOT".into();
    }
    if !sets_ok || !row.init_hash {
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

    let mut slots: Vec<Slot> = Vec::new();
    let mut frames: Vec<u16> = Vec::new();
    if a.dump_planes {
        frames.extend_from_slice(&init_plane);
    }

    let mut first_div = String::from("none");
    let mut first_div_at: Option<u32> = None;
    let mut oborder = vec![0u32; obcap];

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

        // --- 2. bind the engine's live attacks to device slots ---
        // Matched on (owner, target) and NOT on troop bits: the engine can
        // re-initialise an attack mid-flight with a fresh `attack_amount`
        // allocation (attack.rs:150) plus a `refresh_to_conquer` heap refill
        // (attack.rs:1265), so the bits legitimately change for the SAME attack.
        // There is deliberately no owner-only fallback: it used to bind an
        // engine attack onto any same-owner slot and re-stamp the host struct,
        // which hid an engine-side re-init from the device instead of reporting it.
        let troops_all = d_troops.to_host_vec(&stream).map_err(es)?;
        let _scal_all = d_scal.to_host_vec(&stream).map_err(es)?;
        for s in slots.iter_mut() {
            s.troops_bits = troops_all[s.idx].to_bits();
        }
        let mut plan: Vec<(usize, AttackSnap)> = Vec::new();
        let mut taken: Vec<bool> = vec![false; slots.len()];
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
                        },
                    );
                    s.troops_bits = snap.troops.to_bits();
                    let sidx = s.idx; // `s` is moved into `slots` below
                    slots[i] = s;
                    plan.push((i, snap.clone()));

                    // --- engine-side TROOP RE-ALLOCATION on a live attack --------
                    // The engine can replace a live attack's carried amount in
                    // place: `AttackExecution::init` (attack.rs:136-150) sets
                    // `self.troops = start`, where `start` is either
                    // `self.start_troops` - a figure the bot AI passed in
                    // explicitly through `game.add_land_attack_from`
                    // (game.rs:1477) - or `game.wire.attack_amount(...)`, which is
                    // 5% of the owner's troops for a Bot and 20% for anyone else
                    // (core/config.rs:461-468). Both are ENGINE decisions the port
                    // must REPRODUCE, not recompute.
                    //
                    // The only record-level signal is the troop bits going UP: the
                    // device's own arithmetic only ever drains them
                    // (kernels.rs:225-236, attack.rs:313-314). So an increase is
                    // taken as a re-allocation and the device's `troops` word is
                    // re-seeded from the record - and NOTHING ELSE: the heap, the
                    // prng stream, the border set and the plane are left alone.
                    // (Re-seeding the heap here was tried in a first pass and
                    // over-claims - see the note below.)
                    if let Some(cs) = cur
                        .attacks
                        .iter()
                        .find(|a| a.owner == snap.owner && a.target == snap.target)
                    {
                        if let Some(ps) = prev
                            .attacks
                            .iter()
                            .find(|a| a.owner == snap.owner && a.target == snap.target)
                        {
                            if cs.troops > ps.troops {
                                troops_host = d_troops.to_host_vec(&stream).map_err(es)?;
                                troops_host[sidx] = cs.troops;
                                d_troops
                                    .copy_from_host(&stream, &troops_host)
                                    .map_err(es)?;
                                row.reinits += 1;
                                detail.push_str(&format!(
                                    "TROOP_REALLOC boundary {b} (engine tick {}): owner {} target {} \
                                     troops {} -> {} bits {:#018x} -> {:#018x} (engine decision, \
                                     re-seeded from the record; heap/prng untouched)\n",
                                    prev.engine_tick,
                                    snap.owner,
                                    snap.target,
                                    ps.troops,
                                    cs.troops,
                                    ps.troops.to_bits(),
                                    cs.troops.to_bits()
                                ));
                            }
                        }
                    }

                    // --- engine-side RE-CREATE of a live attack: NOT followed ---
                    // Measured, not assumed. The engine re-initialises an attack in
                    // place when it is re-issued/merged: `AttackExecution::init`
                    // (attack.rs:83) takes a fresh `attack_amount` (attack.rs:150),
                    // re-registers the attack (attack.rs:158 - it moves to the END
                    // of `live_attacks()`), and re-seeds the heap from
                    // `source_tile` via `add_neighbors` (attack.rs:160-164),
                    // falling back to `refresh_to_conquer` (attack.rs:1265) only
                    // when there is no source tile.
                    //
                    // Re-seeding the slot from the engine's BORDER set instead of
                    // the source tile's neighbourhood was tried and OVER-claims
                    // (europe N=18 boundary 49: device 582 owned tiles vs engine
                    // 580). `attack_init` (kernels.rs:358) is border-set based, so
                    // the composed pipeline cannot express this event: the engine's
                    // refreshed attack claims one tile the device does not (engine
                    // 303 claims vs device 302 at boundary 49). Reported as the
                    // first divergence - hidden nowhere, and `row.reinits` stays 0.
                }
                None => {
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
                                &mut d_plane,
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
        for (i, s) in slots.iter_mut().enumerate() {
            if s.alive && !taken[i] {
                s.alive = false;
                row.engine_evictions += 1;
                evicted.push(format!("sid {} slot {} at boundary {b}", s.owner, s.idx));
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

        // --- 4. one device launch per live attack, in the engine's order ---
        let mut mine: HashMap<u16, Vec<u32>> = HashMap::new();
        let mut troop_pairs: Vec<(usize, u64)> = Vec::new();
        for (i, _snap) in &plan {
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
                        1, // every actor in this matrix is a bot (nations disabled)
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
                slots[*i].alive = false;
            }
        }
        // troop count after the tick, for the surviving attacks
        let troops_after = d_troops.to_host_vec(&stream).map_err(es)?;

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
            let mine_bits = slots
                .iter()
                .filter(|s| s.owner == snap.owner && s.target == snap.target)
                .map(|s| troops_after[s.idx].to_bits())
                .next();
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
    for s in 0..slots.len() {
        max_peak = max_peak.max(scal_end[s * SCAL + 5]);
        drops += scal_end[s * SCAL + 4] as u64;
        max_border = max_border.max(scal_end[s * SCAL + 1]);
        live += (scal_end[s * SCAL + 3] == 1) as usize;
    }
    detail.push_str(&format!(
        "\n### DEVICE RESOURCES (gpu_env's own limits, measured here)\n\
         slots used {} (max {MAX_SLOTS}), live at the end {}, max heap peak {} of HEAP_CAP {HEAP_CAP}, \
         max border_len {} of BC {BC}, claim-list overflows {drops}\n",
        slots.len(),
        live,
        max_peak,
        max_border,
    ));
    if drops > 0 {
        row.note = format!("{} claim-list overflows {drops}", row.note).trim().to_string();
    }
    if max_border as usize > BC {
        return Err("border set exceeded BC".into());
    }

    row.first_div = first_div;
    detail.push_str(&format!(
        "\n### RESULT\ninit_sets {}\ninit_hash {}\ntick claims matched {}/{}\nhash matched {}/{}\n\
         owned counts matched {}/{}\ntroops matched {}/{}\nchurn-skipped owners {}\n\
         engine evictions {}\ndevice claims total {}\nfirst divergence: {}\nnote: {}\n",
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
        row.dev_claims,
        row.first_div,
        row.note
    ));
    write_cell(a, c, &detail, Some(&frames), Some(&terrain))?;
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
    for c in &cells {
        eprintln!(
            "=== cell {} N={} nations={} ticks={}",
            c.map, c.agents, c.nations, c.ticks
        );
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
    eprintln!("cells: {pass} passed, {fail} failed/not-driven");
}

#[allow(dead_code)]
fn unused(_: &Path) {}
