// ===========================================================================
// Host side of `ofcuda_cluster`: frozen-case loading, the CPU reference
// driver, packing for the device, the engine cross-check and the report.
//
// No CUDA anywhere in this file, so `src/bin/cpu.rs` (and any other consumer)
// can link it without the cuda crates. The device module in `src/main.rs`
// embeds a VERBATIM copy of `src/core_impl.rs`; `device_core_verbatim_check`
// below fails if the two ever diverge.
// ===========================================================================

pub mod core_impl;

pub use core_impl::{
    BBox, MAX_BORDER, MAX_CLUSTER, NONE, Params, StepOut, TICKS_PER_CLUSTER_CALC,
    is_surrounded, surrounded_by_same_enemy,
};

use std::path::{Path, PathBuf};

use base64::Engine as _;

/// One recorded live attack. `troops_bits` are the IEEE-754 bits of the
/// engine's `f64` troop count (`AttackExecution::troops`), carried as bits so
/// the value survives JSON and stays bit-exact on the device.
#[derive(Clone, Copy, Debug, Default)]
pub struct AttackRec {
    pub owner: u16,
    pub target: u16,
    pub troops_bits: u64,
    pub active: bool,
    pub attack_live: bool,
    pub initialized: bool,
}

impl AttackRec {
    pub fn troops(&self) -> f64 {
        f64::from_bits(self.troops_bits)
    }

    /// Exactly the predicate `Game::largest_incoming_land_attack_from_neighbors`
    /// (`game.rs:2207-2209`) applies before reading `troops()`.
    pub fn qualifies(&self) -> bool {
        self.active && self.attack_live && self.initialized
    }
}

/// The victim's own player record as the engine held it at the frozen tick.
#[derive(Clone, Debug, Default)]
pub struct PlayerRec {
    pub small_id: u16,
    pub id: String,
    pub id_hash: i32,
    pub player_type: String,
    pub tiles_owned: i32,
    pub alive: bool,
    pub last_cluster_calc: u32,
    pub last_tile_change: u32,
    pub border: Vec<u32>,
}

/// A player whose `tiles_owned` changed, plus the engine's own `owned_tiles`
/// vector in insertion order after the call. This is the order-level oracle:
/// `conquer_one` (`game.rs:1233-1273`) appends to `owned_tiles`, so the tail of
/// the captor's list is the engine's own conquer order, and the victim's list
/// is its surviving order.
#[derive(Clone, Debug, Default)]
pub struct ChangedPlayer {
    pub small_id: u16,
    pub tiles_owned: i32,
    pub owned_order: Vec<u32>,
}

#[derive(Clone, Debug, Default)]
pub struct Case {
    pub path: PathBuf,
    pub record: String,
    pub engine_commit: String,
    pub exec_tick: u32,
    pub victim: u16,
    pub width: u32,
    pub height: u32,
    pub terrain_fnv: u64,
    pub plane_fnv: u64,
    pub plane: Vec<u16>,
    pub players: Vec<PlayerRec>,
    pub friends: Vec<(u16, u16)>,
    pub attacks: Vec<AttackRec>,
    pub engine_delta: Vec<(u32, u16)>,
    pub engine_changed: Vec<ChangedPlayer>,
}

impl Case {
    pub fn tiles(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    pub fn player(&self, small_id: u16) -> Option<&PlayerRec> {
        self.players.iter().find(|p| p.small_id == small_id)
    }

    pub fn changed(&self, small_id: u16) -> Option<&ChangedPlayer> {
        self.engine_changed.iter().find(|c| c.small_id == small_id)
    }

    /// Captor the engine's recorded delta landed on (`0` if no delta).
    pub fn delta_owner(&self) -> u16 {
        self.engine_delta.first().map(|(_, o)| *o).unwrap_or(0)
    }
}

fn u64_of(v: &serde_json::Value) -> u64 {
    let s = v.as_str().unwrap_or("");
    let s = s.trim();
    if let Some(hex) = s.strip_prefix("0x") {
        u64::from_str_radix(hex, 16).unwrap_or(0)
    } else {
        s.parse::<u64>().unwrap_or(0)
    }
}

/// Load a case emitted by `tools/dump_case`.
pub fn load_case(path: &Path) -> Result<Case, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    let meta = &v["meta"];
    let plane_b64 = v["input_plane_b64"].as_str().ok_or("input_plane_b64 missing")?;
    let plane_bytes = base64::engine::general_purpose::STANDARD
        .decode(plane_b64)
        .map_err(|e| format!("input_plane_b64: {e}"))?;
    if plane_bytes.len() % 2 != 0 {
        return Err("input_plane_b64: odd byte count".into());
    }
    let plane: Vec<u16> = plane_bytes
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();

    let players = v["players"]
        .as_array()
        .ok_or("players missing")?
        .iter()
        .map(|p| PlayerRec {
            small_id: p["small_id"].as_u64().unwrap_or(0) as u16,
            id: p["id"].as_str().unwrap_or("").to_string(),
            id_hash: p["id_hash"].as_i64().unwrap_or(0) as i32,
            player_type: p["player_type"].as_str().unwrap_or("").to_string(),
            tiles_owned: p["tiles_owned"].as_i64().unwrap_or(0) as i32,
            alive: p["alive"].as_bool().unwrap_or(false),
            last_cluster_calc: p["last_cluster_calc"].as_u64().unwrap_or(0) as u32,
            last_tile_change: p["last_tile_change"].as_u64().unwrap_or(0) as u32,
            border: p["border"]
                .as_array()
                .map(|a| a.iter().filter_map(|x| x.as_u64()).map(|x| x as u32).collect())
                .unwrap_or_default(),
        })
        .collect();

    let friends = v["friends"]
        .as_array()
        .ok_or("friends missing")?
        .iter()
        .filter_map(|p| {
            let a = p.as_array()?;
            Some((a.first()?.as_u64()? as u16, a.get(1)?.as_u64()? as u16))
        })
        .collect();

    let attacks = v["attacks"]
        .as_array()
        .ok_or("attacks missing")?
        .iter()
        .map(|a| AttackRec {
            owner: a["owner"].as_u64().unwrap_or(0) as u16,
            target: a["target"].as_u64().unwrap_or(0) as u16,
            troops_bits: u64_of(&a["troops_bits"]),
            active: a["active"].as_bool().unwrap_or(false),
            attack_live: a["attack_live"].as_bool().unwrap_or(false),
            initialized: a["initialized"].as_bool().unwrap_or(false),
        })
        .collect();

    let engine_delta = v["engine_outcome_delta"]
        .as_array()
        .ok_or("engine_outcome_delta missing")?
        .iter()
        .filter_map(|d| {
            let a = d.as_array()?;
            Some((a.first()?.as_u64()? as u32, a.get(1)?.as_u64()? as u16))
        })
        .collect();

    let engine_changed = v["engine_outcome_changed_players"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|c| ChangedPlayer {
                    small_id: c["small_id"].as_u64().unwrap_or(0) as u16,
                    tiles_owned: c["tiles_owned"].as_i64().unwrap_or(0) as i32,
                    owned_order: c["owned_order"]
                        .as_array()
                        .map(|x| x.iter().filter_map(|y| y.as_u64()).map(|y| y as u32).collect())
                        .unwrap_or_default(),
                })
                .collect()
        })
        .unwrap_or_default();

    Ok(Case {
        path: path.to_path_buf(),
        record: meta["record"].as_str().unwrap_or("").to_string(),
        engine_commit: meta["engine_commit"].as_str().unwrap_or("").to_string(),
        exec_tick: meta["exec_tick"].as_u64().unwrap_or(0) as u32,
        victim: meta["victim_small_id"].as_u64().unwrap_or(0) as u16,
        width: meta["width"].as_u64().unwrap_or(0) as u32,
        height: meta["height"].as_u64().unwrap_or(0) as u32,
        terrain_fnv: u64_of(&meta["terrain_fnv"]),
        plane_fnv: u64_of(&v["input_plane_fnv"]),
        plane,
        players,
        friends,
        attacks,
        engine_delta,
        engine_changed,
    })
}

/// Map directory the RL engine loads (`ofcuda_map` reuse: the same loader and
/// hash this project already verified, terrain hash `0xebffa87c2568cc58`).
pub fn pangaea_map_dir(repo_root: &Path) -> PathBuf {
    ofcuda_map::map_dir(repo_root, "pangaea")
}

/// Load the terrain plane and return it together with the same hash the case
/// recorded, so the caller can assert the input the kernel sees is the plane
/// the engine ran on.
pub fn load_terrain(map_dir: &Path) -> Result<(Vec<u8>, u64), String> {
    let (_meta, terrain) = ofcuda_map::load_map_normal(map_dir)?;
    let hash = ofcuda_map::terrain_hash(&terrain);
    Ok((terrain, hash))
}

// ---------------------------------------------------------------------------
// inputs packed for the device
// ---------------------------------------------------------------------------

/// Everything the kernels need, built from the case.
///
/// `slots` / `slot_len` are filled here by the **same** `flood_cluster_slot`
/// the device kernel runs (a plain host loop) - `prepare()` is therefore the
/// CPU side of kernel A, and `decide_cpu()` the CPU side of kernel B. The GPU
/// binary re-derives both on device from the same inputs, which is what makes
/// a GPU/CPU disagreement meaningful.
#[derive(Clone, Debug)]
pub struct Prepared {
    pub victim: u16,
    pub params: Params,
    pub border: Vec<u32>,
    pub border_len: u32,
    pub tile_to_border: Vec<u32>,
    pub slot_len: Vec<u32>,
    pub slots: Vec<u32>,
    pub friends_packed: Vec<u32>,
    pub atk_pt: Vec<u32>,
    pub atk_troops: Vec<u64>,
    pub atk_flags: Vec<u32>,
    /// Cap for the change list = the victim's `tiles_owned`, since
    /// `flood_owned` can only ever conquer tiles the victim owns.
    pub changes_cap: usize,
}

pub fn prepare(case: &Case, victim: u16) -> Result<Prepared, String> {
    let p = case
        .player(victim)
        .ok_or_else(|| format!("victim {victim} not in case players"))?;
    let border: Vec<u32> = p.border.clone();
    let border_len: u32 = border.len() as u32;
    if border_len == 0 {
        return Err(format!("victim {victim} has no border tiles"));
    }
    if border_len as usize > MAX_BORDER {
        return Err(format!("border_len {border_len} > MAX_BORDER {MAX_BORDER}"));
    }
    let mut tile_to_border = vec![NONE; case.tiles()];
    for (j, &t) in border.iter().enumerate() {
        tile_to_border[t as usize] = j as u32;
    }

    let mut slot_len = vec![0u32; border_len as usize];
    let mut slots = vec![0u32; border_len as usize * MAX_CLUSTER];
    for j in 0..border_len {
        let base = j as usize * MAX_CLUSTER;
        let n = core_impl::flood_cluster_slot(
            case.width,
            case.height,
            &border,
            &tile_to_border,
            j,
            &mut slots[base..base + MAX_CLUSTER],
            border_len,
        );
        slot_len[j as usize] = n;
    }

    let friends_packed: Vec<u32> = case
        .friends
        .iter()
        .map(|(a, b)| ((*a as u32) << 16) | *b as u32)
        .collect();
    let atk_pt: Vec<u32> = case
        .attacks
        .iter()
        .map(|a| ((a.owner as u32) << 16) | a.target as u32)
        .collect();
    let atk_troops: Vec<u64> = case.attacks.iter().map(|a| a.troops_bits).collect();
    let atk_flags: Vec<u32> = case
        .attacks
        .iter()
        .map(|a| a.active as u32 | ((a.attack_live as u32) << 1) | ((a.initialized as u32) << 2))
        .collect();

    let params = Params {
        width: case.width,
        height: case.height,
        tick: case.exec_tick,
        victim,
        id_hash: p.id_hash,
        tiles_owned: p.tiles_owned,
        alive: p.alive,
        last_cluster_calc: p.last_cluster_calc,
        last_tile_change: p.last_tile_change,
        border_len,
    };

    Ok(Prepared {
        victim,
        params,
        border,
        border_len,
        tile_to_border,
        slot_len,
        slots,
        friends_packed,
        atk_pt,
        atk_troops,
        atk_flags,
        changes_cap: (p.tiles_owned.max(1)) as usize,
    })
}

/// The cluster list the flood produced, as `(start border index, size)` in the
/// engine's order (ascending start index).
pub fn clusters_from(prep: &Prepared) -> Vec<(u32, u32)> {
    (0..prep.border_len)
        .filter(|&j| prep.slot_len[j as usize] > 0)
        .map(|j| (j, prep.slot_len[j as usize]))
        .collect()
}

// ---------------------------------------------------------------------------
// the CPU reference
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
pub struct Run {
    pub out: StepOut,
    /// `(tile, new_owner)` in application order.
    pub changes: Vec<(u32, u16)>,
    pub plane_after: Vec<u16>,
    pub fatal: Option<String>,
}

/// CPU side of kernel B, calling the same `decide_and_remove` the kernel does.
pub fn decide_cpu(case: &Case, terrain: &[u8], prep: &Prepared) -> Run {
    let mut owners = case.plane.clone();
    let mut marks = vec![0u32; case.tiles()];
    let mut stack = vec![0u32; prep.changes_cap + MAX_BORDER + 1];
    let mut tiles_out = vec![0u32; prep.changes_cap];
    let mut changes = vec![0u64; prep.changes_cap];
    let out = core_impl::decide_and_remove(
        &prep.params,
        terrain,
        &mut owners,
        &prep.slots,
        &prep.slot_len,
        &prep.friends_packed,
        &prep.atk_pt,
        &prep.atk_troops,
        &prep.atk_flags,
        &mut marks,
        &mut stack,
        &mut tiles_out,
        &mut changes,
    );
    let mut fatal = None;
    if out.changes as usize > prep.changes_cap {
        fatal = Some(format!(
            "change list overflow: {} > cap {}",
            out.changes, prep.changes_cap
        ));
    }
    let mut decoded = Vec::with_capacity(out.changes as usize);
    for c in changes.iter().take(out.changes as usize) {
        decoded.push(((*c >> 16) as u32, (*c & 0xffff) as u16));
    }
    Run {
        out,
        changes: decoded,
        plane_after: owners,
        fatal,
    }
}

/// Full host reference: kernel A's flood (in `prepare`) plus kernel B's
/// decide-and-remove (here).
pub fn run_cpu(case: &Case, terrain: &[u8]) -> Result<(Prepared, Run), String> {
    let prep = prepare(case, case.victim)?;
    let run = decide_cpu(case, terrain, &prep);
    Ok((prep, run))
}

// ---------------------------------------------------------------------------
// diagnosis (report only - recomputed on the input plane, never on the
// engine's decision path)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ClusterDiag {
    pub index: usize,
    pub start_index: u32,
    pub start_tile: u32,
    pub size: u32,
    pub surrounded_enemy: u32,
    pub surrounded_by_same_enemy: bool,
    pub is_surrounded: bool,
    pub size_4connected_owned: u32,
}

pub fn diagnose(case: &Case, terrain: &[u8], prep: &Prepared) -> Vec<ClusterDiag> {
    let clusters = clusters_from(prep);
    let mut out = Vec::new();
    for (i, (j, len)) in clusters.iter().enumerate() {
        let base = *j as usize * MAX_CLUSTER;
        let cluster = &prep.slots[base..base + *len as usize];
        let bb = core_impl::bbox_of(case.width, cluster);
        let enemy = core_impl::surrounded_by_same_enemy(
            &prep.params,
            terrain,
            &case.plane,
            cluster,
            bb,
            &prep.friends_packed,
        );
        let mut marks = vec![0u32; case.tiles()];
        let mut stack = vec![0u32; case.tiles()];
        let n4 = core_impl::flood_owned_count(
            &prep.params,
            &case.plane,
            cluster[0],
            &mut marks,
            1,
            &mut stack,
        );
        out.push(ClusterDiag {
            index: i,
            start_index: *j,
            start_tile: cluster[0],
            size: *len,
            surrounded_enemy: enemy,
            surrounded_by_same_enemy: enemy != NONE,
            is_surrounded: core_impl::is_surrounded(&prep.params, terrain, &case.plane, cluster),
            size_4connected_owned: n4,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// engine cross-check
// ---------------------------------------------------------------------------

pub struct Check {
    pub label: String,
    pub ok: bool,
    pub detail: String,
}

pub fn compare_to_engine(case: &Case, run: &Run) -> Vec<Check> {
    let mut checks = Vec::new();
    let mut want: Vec<(u32, u16)> = case.engine_delta.clone();
    want.sort_unstable();
    let mut got: Vec<(u32, u16)> = run.changes.clone();
    got.sort_unstable();
    checks.push(Check {
        label: "engine delta (tile, owner) set, sorted".into(),
        ok: got == want,
        detail: format!(
            "{} vs {} entries{}",
            got.len(),
            want.len(),
            first_diff_u32pairs(&got, &want)
        ),
    });

    // Order-level: the captor's owned_tiles tail is the engine's conquer order.
    let owner = case.delta_owner();
    if let Some(ch) = case.changed(owner) {
        let n = run.changes.len();
        let tail: Vec<u32> = if ch.owned_order.len() >= n {
            ch.owned_order[ch.owned_order.len() - n..].to_vec()
        } else {
            ch.owned_order.clone()
        };
        let mine: Vec<u32> = run.changes.iter().map(|(t, _)| *t).collect();
        checks.push(Check {
            label: format!("conquer ORDER vs captor {owner} owned_tiles tail"),
            ok: tail == mine,
            detail: format!(
                "{} vs {} tiles{}",
                mine.len(),
                tail.len(),
                first_diff_u32(&mine, &tail)
            ),
        });
    } else if case.engine_changed.is_empty() {
        // Older dump format: this file predates `engine_outcome_changed_players`,
        // so the order oracle (and per-player survival counts) were never
        // recorded. Nothing to compare - report it as not evaluated rather than
        // a mismatch. The tile-set delta and the whole-plane check above still
        // run in full.
        checks.push(Check {
            label: format!("conquer ORDER vs captor {owner} owned_tiles tail"),
            ok: true,
            detail: "not evaluated: this dump predates engine_outcome_changed_players".into(),
        });
    } else {
        checks.push(Check {
            label: format!("conquer ORDER vs captor {owner} owned_tiles tail"),
            ok: false,
            detail: format!("captor {owner} not in engine_outcome_changed_players"),
        });
    }

    // Victim survival count and captor gain count.
    for (sid, want_tiles) in [
        (case.victim, case.changed(case.victim).map(|c| c.tiles_owned)),
        (owner, case.changed(owner).map(|c| c.tiles_owned)),
    ] {
        let Some(want) = want_tiles else { continue };
        let after = run
            .plane_after
            .iter()
            .filter(|&&o| (o & 0x0fff) == sid)
            .count() as i32;
        checks.push(Check {
            label: format!("player {sid} tiles_owned after"),
            ok: after == want,
            detail: format!("{after} vs engine {want}"),
        });
    }

    // The plane may only differ from the input on the recorded delta: every
    // differing tile must be exactly the engine's recorded (tile, new_owner),
    // and there must be no delta tile the plane left unchanged.
    let mut want_tiles: Vec<(u32, u16)> = case.engine_delta.clone();
    want_tiles.sort_unstable();
    let mut got_tiles: Vec<(u32, u16)> = Vec::new();
    for (i, (&a, &b)) in case.plane.iter().zip(run.plane_after.iter()).enumerate() {
        if a != b {
            got_tiles.push((i as u32, b));
        }
    }
    got_tiles.sort_unstable();
    let plane_ok = got_tiles == want_tiles;
    let detail = if plane_ok {
        format!("{} differing tile(s), exactly the recorded delta", got_tiles.len())
    } else {
        format!(
            "{} differing tile(s) vs engine delta of {}{}",
            got_tiles.len(),
            want_tiles.len(),
            first_diff_u32pairs(&got_tiles, &want_tiles)
        )
    };
    checks.push(Check {
        label: "plane: differs from input only on the delta".into(),
        ok: plane_ok,
        detail,
    });
    checks
}

fn first_diff_u32pairs(a: &[(u32, u16)], b: &[(u32, u16)]) -> String {
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            return format!("; first diff at sorted index {i}: got {x:?} want {y:?}");
        }
    }
    if a.len() != b.len() {
        return format!("; one side is a prefix of the other at index {}", a.len().min(b.len()));
    }
    String::new()
}

fn first_diff_u32(a: &[u32], b: &[u32]) -> String {
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        if x != y {
            return format!("; first diff at index {i}: got {x} want {y}");
        }
    }
    String::new()
}

pub fn compare_runs(cpu: &Run, gpu: &Run) -> Vec<Check> {
    vec![
        Check {
            label: "GPU vs CPU: fired/cluster_count/largest/removed".into(),
            ok: cpu.out.fired == gpu.out.fired
                && cpu.out.cluster_count == gpu.out.cluster_count
                && cpu.out.largest_index == gpu.out.largest_index
                && cpu.out.largest_size == gpu.out.largest_size
                && cpu.out.removed_clusters == gpu.out.removed_clusters,
            detail: format!(
                "cpu fired={} clusters={} largest=({},{}) removed={} | gpu fired={} clusters={} largest=({},{}) removed={}",
                cpu.out.fired,
                cpu.out.cluster_count,
                cpu.out.largest_index,
                cpu.out.largest_size,
                cpu.out.removed_clusters,
                gpu.out.fired,
                gpu.out.cluster_count,
                gpu.out.largest_index,
                gpu.out.largest_size,
                gpu.out.removed_clusters
            ),
        },
        Check {
            label: "GPU vs CPU: change count".into(),
            ok: cpu.out.changes == gpu.out.changes && cpu.changes.len() == gpu.changes.len(),
            detail: format!("{} vs {}", cpu.changes.len(), gpu.changes.len()),
        },
        Check {
            label: "GPU vs CPU: change list bit-exact (order + owner)".into(),
            ok: cpu.changes == gpu.changes,
            detail: first_diff_u32pairs(&gpu.changes, &cpu.changes),
        },
        Check {
            label: "GPU vs CPU: owner plane bit-exact".into(),
            ok: cpu.plane_after == gpu.plane_after,
            detail: match cpu
                .plane_after
                .iter()
                .zip(gpu.plane_after.iter())
                .enumerate()
                .find(|(_, (a, b))| a != b)
            {
                None => String::new(),
                Some((i, (a, b))) => format!("first diff tile {i}: cpu {a} gpu {b}"),
            },
        },
    ]
}

// ---------------------------------------------------------------------------
// the verbatim-copy guard
// ---------------------------------------------------------------------------

pub const BEGIN_MARK: &str = "===== BEGIN VERBATIM COPY of src/core_impl.rs =====";
pub const END_MARK: &str = "===== END VERBATIM COPY of src/core_impl.rs =====";

/// Extracts the device copy out of `src/main.rs` (between the markers).
pub fn device_copy_text() -> Result<String, String> {
    let main_rs = include_str!("main.rs");
    let (_, rest) = main_rs
        .split_once(BEGIN_MARK)
        .ok_or_else(|| format!("{BEGIN_MARK} not found in src/main.rs"))?;
    let (body, _) = rest
        .split_once(END_MARK)
        .ok_or_else(|| format!("{END_MARK} not found in src/main.rs"))?;
    // The markers sit on their own comment lines; drop the rest of the line
    // that carried the BEGIN marker and everything from the END marker's line
    // (including the `// ` prefix that precedes the END marker on that line).
    let body = body.split_once('\n').map(|(_, r)| r).unwrap_or("");
    let body = body.rsplit_once('\n').map(|(l, _)| l).unwrap_or("");
    // If the copy is empty (`cargo oxide run` before `sync_device_core.sh`),
    // report that rather than pretending it matches.
    Ok(body.to_string())
}

/// Fails when the device copy in `src/main.rs` differs from
/// `src/core_impl.rs`. Any perturbation of either copy is a hard error - the
/// whole point is that GPU/CPU agreement is a check on the CUDA lowering, not
/// on a second implementation.
pub fn device_core_verbatim_check() -> Result<(), String> {
    let canonical = include_str!("core_impl.rs");
    let copy = device_copy_text()?;
    if copy.trim().is_empty() {
        return Err(format!(
            "device copy between {BEGIN_MARK} / {END_MARK} is empty - run ./sync_device_core.sh"
        ));
    }
    // The injected block may carry a trailing newline; compare exactly after
    // normalising only the final newline.
    let a = canonical.trim_end_matches('\n');
    let b = copy.trim_end_matches('\n');
    if a == b {
        return Ok(());
    }
    let al: Vec<&str> = a.lines().collect();
    let bl: Vec<&str> = b.lines().collect();
    for (i, (x, y)) in al.iter().zip(bl.iter()).enumerate() {
        if x != y {
            return Err(format!(
                "device core copy diverged at line {} of core_impl.rs:\n  core_impl.rs: {x}\n  main.rs copy: {y}",
                i + 1
            ));
        }
    }
    Err(format!(
        "device core copy diverged: core_impl.rs has {} lines, main.rs copy has {}",
        al.len(),
        bl.len()
    ))
}

// ---------------------------------------------------------------------------
// the report
// ---------------------------------------------------------------------------

pub struct ReportInput<'a> {
    pub case: &'a Case,
    pub device: &'a str,
    pub terrain_path: &'a str,
    pub terrain_hash: u64,
    pub prep: &'a Prepared,
    pub diags: &'a [ClusterDiag],
    pub cpu: &'a Run,
    pub gpu: Option<&'a Run>,
    pub gpu_checks: &'a [Check],
    pub engine_checks: &'a [Check],
    pub engine_checks_gpu: &'a [Check],
}

fn fmt_changes(c: &[(u32, u16)], keep: usize) -> String {
    if c.len() <= keep * 2 {
        return format!("{c:?}");
    }
    format!(
        "{:?} ... {:?} ({} total)",
        &c[..keep],
        &c[c.len() - keep..],
        c.len()
    )
}

/// One report, the same text from the CPU-only and the GPU binary.
pub fn format_report(r: &ReportInput) -> (String, bool) {
    let mut s = String::new();
    let case = r.case;
    let mut all: Vec<(String, bool, String)> = Vec::new();

    s.push_str("# ofcuda_cluster - CLUSTER CAPTURE step\n");
    s.push_str("# spec: rust/engine/src/execution/player_clusters.rs (maybe_remove_clusters,\n");
    s.push_str("#       calculate_clusters, flood_border_cluster, surrounded_by_same_enemy,\n");
    s.push_str("#       is_surrounded, get_capturing_player, flood_owned, remove_cluster)\n");
    s.push_str("# TS authority: openfront/src/core/execution/PlayerExecution.ts:99-416\n");
    if r.device.is_empty() {
        s.push_str("# device: (CPU-only run - no CUDA linked)\n");
    } else {
        s.push_str(&format!("# device: {}\n", r.device));
    }
    s.push_str(&format!("# case file: {}\n", case.path.display()));
    s.push_str(&format!("# engine commit: {}\n", case.engine_commit));
    s.push_str(&format!("# record: {}\n", case.record));

    let th_ok = r.terrain_hash == case.terrain_fnv;
    s.push_str(&format!(
        "# terrain: {}  hash {:#018x}  case expects {:#018x}  {}\n",
        r.terrain_path,
        r.terrain_hash,
        case.terrain_fnv,
        if th_ok { "MATCH" } else { "MISMATCH" }
    ));
    let ph = ofcuda_map::state_hash(&case.plane);
    let ph_ok = ph == case.plane_fnv;
    s.push_str(&format!(
        "# input plane: fnv {:#018x}  case expects {:#018x}  {}\n",
        ph,
        case.plane_fnv,
        if ph_ok { "MATCH" } else { "MISMATCH" }
    ));
    s.push_str("# inputs: all read from the case dump; nothing synthesised\n");
    all.push((
        "terrain hash matches the case's own terrain_fnv".into(),
        th_ok,
        format!("{:#018x} vs {:#018x}", r.terrain_hash, case.terrain_fnv),
    ));
    all.push((
        "input owner plane hash matches the case's input_plane_fnv".into(),
        ph_ok,
        format!("{ph:#018x} vs {:#018x}", case.plane_fnv),
    ));

    // ---- input -------------------------------------------------------------
    let p = &r.prep.params;
    s.push_str("\n=== INPUT (reconstructed from the dump) ===\n");
    s.push_str(&format!(
        "  grid                       {}x{} ({} tiles)\n",
        case.width,
        case.height,
        case.tiles()
    ));
    s.push_str(&format!(
        "  tick / victim              {} / {} ({})\n",
        case.exec_tick,
        case.victim,
        case.player(case.victim)
            .map(|x| format!("{} {}", x.player_type, x.id))
            .unwrap_or_default()
    ));
    s.push_str(&format!(
        "  victim player record       tiles_owned {} alive {} last_cluster_calc {} last_tile_change {} id_hash {}\n",
        p.tiles_owned, p.alive, p.last_cluster_calc, p.last_tile_change, p.id_hash
    ));
    s.push_str(&format!(
        "  tick gate                  tick-last_calc={} (>{} ?) | tiles_owned>=100 ? {}\n",
        case.exec_tick.saturating_sub(p.last_cluster_calc),
        TICKS_PER_CLUSTER_CALC,
        p.tiles_owned >= 100
    ));
    s.push_str(&format!(
        "  border (engine Vec order)  {} tiles, first 8: {:?}\n",
        r.prep.border_len,
        &r.prep.border[..r.prep.border.len().min(8)]
    ));
    let vic_friends: Vec<(u16, u16)> = case
        .friends
        .iter()
        .copied()
        .filter(|(a, b)| *a == case.victim || *b == case.victim)
        .collect();
    s.push_str(&format!(
        "  friends                    {} directed pairs; touching victim: {:?}\n",
        case.friends.len(),
        vic_friends
    ));
    let vic_atk: Vec<String> = case
        .attacks
        .iter()
        .filter(|a| a.target == case.victim)
        .map(|a| {
            format!(
                "{}->{} troops {:.6} active/live/init {}/{}/{}",
                a.owner, a.target, a.troops(), a.active, a.attack_live, a.initialized
            )
        })
        .collect();
    s.push_str(&format!(
        "  attacks (exec order)       {} records; targeting victim: {:?}\n",
        case.attacks.len(),
        vic_atk
    ));
    s.push_str(&format!(
        "  change-list cap            {} (= victim tiles_owned)\n",
        r.prep.changes_cap
    ));

    // ---- clusters ----------------------------------------------------------
    let cl = clusters_from(r.prep);
    let sizes: Vec<(u32, u32)> = cl.iter().map(|(j, n)| (*j, *n)).collect();
    s.push_str("\n=== CLUSTERS (flood_cluster_slot / kernel A) ===\n");
    s.push_str(&format!(
        "  cpu + gpu input:            {} clusters, (start_border_idx, size) = {:?}\n",
        sizes.len(),
        sizes
    ));
    s.push_str(&format!(
        "  cpu:                        fired={} cluster_count={} largest=(idx {} size {}) removed={} changes={}\n",
        r.cpu.out.fired, r.cpu.out.cluster_count, r.cpu.out.largest_index, r.cpu.out.largest_size,
        r.cpu.out.removed_clusters, r.cpu.out.changes
    ));
    if let Some(g) = r.gpu {
        s.push_str(&format!(
            "  gpu:                        fired={} cluster_count={} largest=(idx {} size {}) removed={} changes={}\n",
            g.out.fired, g.out.cluster_count, g.out.largest_index, g.out.largest_size,
            g.out.removed_clusters, g.out.changes
        ));
    }
    s.push_str("  diagnosis (recomputed on the INPUT plane, report only - not the\n");
    s.push_str("  engine's decision path; the engine's own decisions are checked below):\n");
    for d in r.diags {
        s.push_str(&format!(
            "    #{} border_idx {:>3} start_tile {:>7} size {:>4} 4-connected-owned-region {:>5} surrounded_by_same_enemy {} is_surrounded {}\n",
            d.index,
            d.start_index,
            d.start_tile,
            d.size,
            d.size_4connected_owned,
            if d.surrounded_enemy == NONE {
                "none".to_string()
            } else {
                format!("player {}", d.surrounded_enemy)
            },
            d.is_surrounded
        ));
    }

    // ---- result side by side ----------------------------------------------
    s.push_str("\n=== RESULT SIDE BY SIDE ===\n");
    s.push_str(&format!(
        "  cpu    : {} change(s); plane_after fnv {:#018x}\n",
        r.cpu.changes.len(),
        ofcuda_map::state_hash(&r.cpu.plane_after)
    ));
    if let Some(g) = r.gpu {
        s.push_str(&format!(
            "  gpu    : {} change(s); plane_after fnv {:#018x}\n",
            g.changes.len(),
            ofcuda_map::state_hash(&g.plane_after)
        ));
    }
    s.push_str(&format!(
        "  engine : {} recorded change(s) -> owner {} (tick {} of {})\n",
        case.engine_delta.len(),
        case.delta_owner(),
        case.exec_tick,
        case.player(case.delta_owner()).map(|x| x.id.as_str()).unwrap_or("?")
    ));
    s.push_str(&format!(
        "  cpu changes (first/last): {}\n",
        fmt_changes(&r.cpu.changes, 6)
    ));
    if let Some(g) = r.gpu {
        s.push_str(&format!(
            "  gpu changes (first/last): {}\n",
            fmt_changes(&g.changes, 6)
        ));
    }
    let mut want = case.engine_delta.clone();
    want.sort_unstable();
    s.push_str(&format!(
        "  engine delta (first/last): {}\n",
        fmt_changes(&want, 6)
    ));
    for (sid, label) in [
        (case.victim, "victim"),
        (case.delta_owner(), "captor"),
    ] {
        if let Some(c) = case.changed(sid) {
            s.push_str(&format!(
                "  engine {label} {sid}: tiles_owned after {} ; owned_tiles len {} ; tail digest {:#018x}\n",
                c.tiles_owned,
                c.owned_order.len(),
                ofcuda_map::state_hash(
                    &c.owned_order
                        .iter()
                        .map(|&t| (t & 0xffff) as u16)
                        .collect::<Vec<u16>>()
                )
            ));
        }
    }

    // ---- checks ------------------------------------------------------------
    s.push_str("\n=== CHECKS ===\n");
    let mut n_ok = 0usize;
    let mut n_all = 0usize;
    for c in r.gpu_checks {
        n_all += 1;
        n_ok += c.ok as usize;
        all.push((c.label.clone(), c.ok, c.detail.clone()));
    }
    for c in r.engine_checks {
        n_all += 1;
        n_ok += c.ok as usize;
        all.push((format!("cpu vs {}", c.label), c.ok, c.detail.clone()));
    }
    for c in r.engine_checks_gpu {
        n_all += 1;
        n_ok += c.ok as usize;
        all.push((format!("gpu vs {}", c.label), c.ok, c.detail.clone()));
    }
    for (label, ok, detail) in &all {
        s.push_str(&format!(
            "  [{}] {label}{}\n",
            if *ok { " ok " } else { "FAIL" },
            if detail.is_empty() {
                String::new()
            } else {
                format!("  --  {detail}")
            }
        ));
    }
    let pass = n_ok == n_all;
    s.push_str(&format!(
        "\nresult: {}  ({}/{} checks)\n",
        if pass { "PASS" } else { "FAIL" },
        n_ok,
        n_all
    ));
    (s, pass)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_core_is_a_verbatim_copy() {
        device_core_verbatim_check().expect("device copy must equal src/core_impl.rs");
    }

    #[test]
    fn neighbours4_is_nswe() {
        // 3x3 grid, tile 4 = centre: N=1, S=7, W=3, E=5.
        let mut out = [9u32; 4];
        let n = core_impl::neighbors4(3, 3, 4, &mut out);
        assert_eq!(&out[..n], &[1, 7, 3, 5]);
        // Corner (0,0): no N, no W.
        let mut out = [9u32; 4];
        let n = core_impl::neighbors4(3, 3, 0, &mut out);
        assert_eq!(&out[..n], &[3, 1]);
    }

    #[test]
    fn neighbours8_is_dx_major() {
        let mut out = [0u32; 8];
        let n = core_impl::neighbors8(3, 3, 4, &mut out);
        assert_eq!(&out[..n], &[0, 3, 6, 1, 7, 2, 5, 8]);
    }
}
