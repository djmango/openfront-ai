//! `feat_dump` - dump the engine's own `/8` feature planes for a fixed
//! seed/stage at a fixed decision index, so the standalone reference
//! (`offeat_ref`) can be diffed against the engine by differential test
//! instead of by assertion.
//!
//! This binary adds no featurizer logic. Every plane it writes is produced by
//! the engine's real code path:
//!
//! * `stat` (6 static-structure planes) and `transient` (attack-front planes)
//!   come straight out of `ofcore::feat::featurize` (`rust/ofcore/src/feat.rs:650`,
//!   the function `oftrain::vecenv::prepare_agent` calls at `vecenv.rs:1618` and
//!   the C ABI calls at `puffer_ffi.rs:534`);
//! * `ego` (own/ally/enemy) and `db` (defense bonus) come out of
//!   `ofcore::feat::pool_ego_db` (`rust/ofcore/src/feat.rs:1205`) - the
//!   reference averaging pooling that `oftrain/src/engine.rs:109-199`'s fused
//!   walk mirrors and that the trainer's own tests diff against it
//!   (`oftrain/src/engine.rs:343`);
//! * the roster/unit/attack lists come from `obs_typed::entities_typed`
//!   (`rust/engine/src/obs_typed.rs:29`), exactly the function `puffer_ffi`
//!   uses (`puffer_ffi.rs:1192`), and the legality from
//!   `obs_typed::legality_typed`;
//! * the LUTs come from `ofcore::feat::make_lut` / `make_clut`.
//!
//! The only per-tile work done here is *decoding* the packed tile state into
//! the two input planes `featurize`/`pool_ego_db` already take
//! (`owners_slotted`, `defense_bonus`) - no pooling, no channel assembly.
//!
//! # Output (a directory)
//!
//! * `dump.txt`   - text: metadata, roster, units, attacks, both LUTs, and the
//!   13 planes as raw IEEE-754 f32 bit patterns (8 lowercase hex digits per
//!   cell, row-major `gy*gw + gx`, cells space-separated on one line per plane);
//! * `terrain.bin`- raw `width * height` terrain bytes (the same plane
//!   `ofcore` reads: bit 7 = land, low 5 bits = magnitude);
//! * `state.bin`  - raw `width * height` packed `u16` tile state, little-endian
//!   (owner in bits 0..12, bit 13 fallout, bit 14 defense bonus).
//!
//! Plane names in `dump.txt`:
//!   `engine_stat_city|port|defense_post|missile_silo|sam_launcher|factory`
//!   `engine_ego_own|ally|enemy`  `engine_db`
//!   `engine_attack_src_own|ally|enemy`  `engine_attack_retreat_own|ally|enemy`
//!
//! Usage (from `rust/`):
//!   cargo run --release -p openfront-engine --bin feat_dump -- \
//!     --seed parity --stage 0 --decision 4 --out /tmp/featdump
//!
//! The live trainer's systemd units are never touched: this only reads a
//! fresh `RlSession` in-process.

use clap::Parser;
use openfront_engine::obs_typed::{entities_typed, legality_typed};
use openfront_engine::puffer_ffi::stage_params;
use openfront_engine::rl::RlSession;
use openfront_engine::session::AGENT_CLIENT_ID;
use ofcore::feat::{self, N_STATIC, REGION, TR_ATTACK_RETREAT, TR_ATTACK_SRC};
use serde_json::{json, Value};
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "feat_dump")]
#[command(about = "Dump the engine's own /8 feature planes for an offeat_ref differential test")]
struct Args {
    /// OpenFront repo root (the dir containing `openfront/resources/maps`).
    #[arg(long, default_value = "/opt/data/workspaces/skg/openfront-ai")]
    repo: PathBuf,
    /// Episode seed (`seed_to_game_id` input).
    #[arg(long, default_value = "parity")]
    seed: String,
    /// V10 curriculum stage index.
    #[arg(long, default_value_t = 0)]
    stage: usize,
    /// Map key override (default: the stage's map pool's first map).
    #[arg(long)]
    map: Option<String>,
    /// Number of RL humans (1 or 2; RlSession clamps).
    #[arg(long, default_value_t = 1)]
    n_agents: u32,
    /// Decision index to run to and dump at (decision 0 = spawn).
    #[arg(long, default_value_t = 4)]
    decision: u32,
    /// Troop count sent by every scripted `expand` intent.
    #[arg(long, default_value_t = 50)]
    expand_troops: i64,
    /// Output directory (created if missing).
    #[arg(long)]
    out: PathBuf,
}

fn main() {
    let args = Args::parse();

    let params = match stage_params(args.stage) {
        Some(p) => p,
        None => {
            eprintln!("stage {} has no V10 params", args.stage);
            std::process::exit(2);
        }
    };
    let map = args
        .map
        .clone()
        .unwrap_or_else(|| params.maps.first().cloned().unwrap_or_default());
    let nations_value: Value = if params.nations_default {
        Value::String("default".into())
    } else {
        Value::from(params.nations as i64)
    };
    let decision_ticks = params.decision_ticks.max(1);

    let (mut session, _head, _ents, _legal, _terrain, _duo) = match RlSession::reset(
        &args.repo,
        &map,
        &args.seed,
        params.bots,
        &params.difficulty,
        nations_value.clone(),
        args.n_agents,
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("RlSession::reset failed: {e}");
            std::process::exit(2);
        }
    };

    // First and last legal spawn tiles (row-major): land && magnitude < 31 &&
    // owner == 0. Same rule as `parity_trace.rs:149-165` / `feat.rs:1392-1420`.
    // The second is used as agent 2's spawn in `--n-agents 2` mode so the two
    // humans land far apart (real ally territory for the ego_ally plane).
    let width = session.game.width();
    let height = session.game.height();
    let (spawn_tile, spawn_tile_2) = {
        let mut legal_tiles = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let t = y * width + x;
                if session.game.is_land(t)
                    && !session.game.is_impassable(t)
                    && session.game.map.magnitude(t) < 31
                    && session.game.map.owner_id(t) == 0
                {
                    legal_tiles.push(t);
                }
            }
        }
        match (legal_tiles.first(), legal_tiles.last()) {
            (Some(&a), Some(&b)) => (a, b),
            _ => {
                eprintln!("no legal spawn tile on map {map}");
                std::process::exit(2);
            }
        }
    };

    // Scripted episode: decision 0 spawns, every later decision expands 50.
    // In duo mode both humans act (agent 2 via an explicit `clientID`).
    for d in 0..=args.decision {
        let mut intents: Vec<Value> = if d == 0 {
            vec![json!({"type": "spawn", "tile": spawn_tile})]
        } else {
            vec![json!({"type": "attack", "targetID": null, "troops": args.expand_troops})]
        };
        if args.n_agents > 1 {
            if d == 0 {
                intents.push(
                    json!({"type": "spawn", "tile": spawn_tile_2, "clientID": "AGENTRL2"}),
                );
            } else {
                intents.push(json!({
                    "type": "attack",
                    "targetID": null,
                    "troops": args.expand_troops,
                    "clientID": "AGENTRL2"
                }));
            }
        }
        session.step(&intents, decision_ticks);
    }

    // ------------------------------------------------------------------
    // Everything below is the engine's own state, read through the engine's
    // own typed obs builders.
    // ------------------------------------------------------------------
    let tick = session.game.ticks() as i64;
    let spawn_phase = session.game.in_spawn_phase();
    let ents = entities_typed(&session.game);
    let legal = legality_typed(&session.game, AGENT_CLIENT_ID);
    let me = session
        .game
        .player_by_client_id(AGENT_CLIENT_ID)
        .map(|p| p.small_id as i64)
        .unwrap_or(-1);
    let alive = ents
        .players
        .iter()
        .find(|p| p.id as i64 == me)
        .map(|p| p.tiles > 0.0)
        .unwrap_or(false);

    let ids: Vec<usize> = ents.players.iter().map(|p| p.id).collect();
    let lut = feat::make_lut(&ids);
    let clut = feat::make_clut(&lut, me, &ents);

    let hr = height as usize - height as usize % REGION;
    let wr = width as usize - width as usize % REGION;
    let gh = hr / REGION;
    let gw = wr / REGION;
    let plane = gh * gw;

    let terrain = session.game.map.terrain_bytes();
    let packed = session.tile_state();

    // Decode the packed tile state into the two input planes featurize /
    // pool_ego_db already consume. This is decoding, not pooling.
    let mut owners_slotted = vec![0u8; hr * wr];
    let mut land = vec![0u8; hr * wr];
    let mut mag = vec![0u8; hr * wr];
    let mut defense_bonus = vec![0u8; hr * wr];
    for y in 0..hr {
        let src_row = y * width as usize;
        let dst_row = y * wr;
        for x in 0..wr {
            let src = src_row + x;
            let dst = dst_row + x;
            let t = terrain[src];
            land[dst] = (t >> feat::IS_LAND_BIT) & 1;
            mag[dst] = t & feat::MAG_MASK;
            let s = packed[src];
            owners_slotted[dst] = lut
                .get((s & 0x0FFF) as usize)
                .copied()
                .unwrap_or(0);
            defense_bonus[dst] = ((s >> 14) & 1) as u8;
        }
    }

    // Engine featurizer (stat + transient) and engine pooling (ego + db).
    let f = feat::featurize(
        gh,
        gw,
        &lut,
        &land,
        &mag,
        &owners_slotted,
        tick,
        spawn_phase,
        alive,
        me,
        &ents,
        &legal,
    );
    let (ego, db) = feat::pool_ego_db(&owners_slotted, &clut, &defense_bonus, hr, wr);
    debug_assert_eq!(f.clut, clut);

    if std::fs::create_dir_all(&args.out).is_err() {
        eprintln!("cannot create {}", args.out.display());
        std::process::exit(2);
    }

    // ---- binary side planes -------------------------------------------------
    std::fs::write(args.out.join("terrain.bin"), terrain).expect("write terrain.bin");
    let mut state_bytes = Vec::with_capacity(packed.len() * 2);
    for w in packed {
        state_bytes.extend_from_slice(&w.to_le_bytes());
    }
    std::fs::write(args.out.join("state.bin"), &state_bytes).expect("write state.bin");

    // ---- text dump ----------------------------------------------------------
    let path = args.out.join("dump.txt");
    let mut out = std::fs::File::create(&path).expect("create dump.txt");
    let hex_u8 = |b: &[u8]| -> String {
        let mut s = String::with_capacity(b.len() * 2);
        for v in b {
            s.push_str(&format!("{v:02x}"));
        }
        s
    };
    let hex_f32_plane = |p: &[f32]| -> String {
        let mut s = String::with_capacity(p.len() * 9);
        for (i, v) in p.iter().enumerate() {
            if i > 0 {
                s.push(' ');
            }
            s.push_str(&format!("{:08x}", v.to_bits()));
        }
        s
    };

    writeln!(out, "# openfront feat_dump v1").unwrap();
    writeln!(out, "format_version 1").unwrap();
    writeln!(out, "repo {}", args.repo.display()).unwrap();
    writeln!(out, "stage {}", args.stage).unwrap();
    writeln!(out, "seed {}", args.seed).unwrap();
    writeln!(out, "map {}", map).unwrap();
    writeln!(out, "game_id {}", session.game_id()).unwrap();
    writeln!(out, "decision {}", args.decision).unwrap();
    writeln!(out, "tick {tick}").unwrap();
    writeln!(out, "spawn_phase {}", spawn_phase as u8).unwrap();
    writeln!(out, "alive {}", alive as u8).unwrap();
    writeln!(out, "me {me}").unwrap();
    writeln!(out, "n_agents {}", args.n_agents).unwrap();
    writeln!(out, "width {width}").unwrap();
    writeln!(out, "height {height}").unwrap();
    writeln!(out, "hr {hr}").unwrap();
    writeln!(out, "wr {wr}").unwrap();
    writeln!(out, "gh {gh}").unwrap();
    writeln!(out, "gw {gw}").unwrap();

    writeln!(out, "players {}", ents.players.len()).unwrap();
    for p in &ents.players {
        writeln!(
            out,
            "player {} {}",
            p.id,
            p.team.as_deref().unwrap_or("-")
        )
        .unwrap();
    }
    writeln!(out, "alliances {}", ents.alliances.len()).unwrap();
    for a in &ents.alliances {
        writeln!(out, "alliance {} {} {}", a.0, a.1, a.2).unwrap();
    }
    writeln!(out, "units {}", ents.units.len()).unwrap();
    for u in &ents.units {
        // class owner gy gx constructing level
        writeln!(
            out,
            "unit {} {} {} {} {} {}",
            u.class, u.owner, u.gy, u.gx, u.constructing as u8, u.level
        )
        .unwrap();
    }
    writeln!(out, "attacks {}", ents.attacks.len()).unwrap();
    for a in &ents.attacks {
        writeln!(
            out,
            "attack {} {} {} {} {}",
            a.from,
            a.troops,
            a.retreating as u8,
            a.src_x.map(|v| v.to_string()).unwrap_or_else(|| "-".into()),
            a.src_y.map(|v| v.to_string()).unwrap_or_else(|| "-".into())
        )
        .unwrap();
    }

    writeln!(out, "lut_len {}", lut.len()).unwrap();
    writeln!(out, "lut {}", hex_u8(&lut)).unwrap();
    writeln!(out, "clut_len {}", clut.len()).unwrap();
    writeln!(out, "clut {}", hex_u8(&clut)).unwrap();

    writeln!(out, "plane_count 13").unwrap();
    writeln!(out, "plane_size {plane}").unwrap();
    let stat_names = [
        "city",
        "port",
        "defense_post",
        "missile_silo",
        "sam_launcher",
        "factory",
    ];
    for c in 0..N_STATIC {
        let p = &f.stat[c * plane..(c + 1) * plane];
        writeln!(out, "plane engine_stat_{} {}", stat_names[c], hex_f32_plane(p)).unwrap();
    }
    let ego_names = ["own", "ally", "enemy"];
    for c in 0..3 {
        let p = &ego[c * plane..(c + 1) * plane];
        writeln!(out, "plane engine_ego_{} {}", ego_names[c], hex_f32_plane(p)).unwrap();
    }
    writeln!(out, "plane engine_db {}", hex_f32_plane(&db)).unwrap();
    for c in 0..3 {
        let base = (TR_ATTACK_SRC + c) * plane;
        writeln!(
            out,
            "plane engine_attack_src_{} {}",
            ego_names[c],
            hex_f32_plane(&f.transient[base..base + plane])
        )
        .unwrap();
    }
    for c in 0..3 {
        let base = (TR_ATTACK_RETREAT + c) * plane;
        writeln!(
            out,
            "plane engine_attack_retreat_{} {}",
            ego_names[c],
            hex_f32_plane(&f.transient[base..base + plane])
        )
        .unwrap();
    }
    out.flush().ok();

    eprintln!(
        "[feat_dump] wrote {} (stage {} seed {} map {} {}x{} -> /8 {}x{}, decision {}, tick {}, \
         players {}, units {}, attacks {}, me {}, alive {})",
        path.display(),
        args.stage,
        args.seed,
        map,
        width,
        height,
        gh,
        gw,
        args.decision,
        tick,
        ents.players.len(),
        ents.units.len(),
        ents.attacks.len(),
        me,
        alive
    );
}
