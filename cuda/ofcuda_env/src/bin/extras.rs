//! The two dump fields the benchmark needs but `ofcuda_tick::parse_dump` does
//! not keep: the per-record `attacks` snapshots and the per-player economy
//! inputs (troops/tiles/gold/id/playerType).
//!
//! This is the SAME JSON keys and the SAME defaults as the driver's
//! `extras()` in `ofcuda_env/src/main.rs`; it is re-read here rather than
//! re-exported because that function is private to that binary. It is a
//! dump reader, not a device core.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct AttRec {
    pub owner: u16,
    pub target: u16,
    pub troops: i64,
    pub live: bool,
}

#[derive(Clone, Debug)]
pub struct PInfo {
    pub troops: i32,
    pub tiles: i32,
    pub gold: i64,
    pub id: String,
    pub ptype: u8,
}

#[derive(Default)]
pub struct Extra {
    pub attacks: HashMap<u32, Vec<AttRec>>,
    pub p: HashMap<u32, HashMap<u16, PInfo>>,
}

pub fn extras(path: &std::path::Path) -> Result<Extra, String> {
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
    }
    Ok(out)
}
