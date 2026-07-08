//! Multiplexed engine daemon client - one tsx process, many env sessions.

use crate::record::StampedIntent;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;

static DAEMON: Mutex<Option<DaemonClient>> = Mutex::new(None);

pub struct DaemonClient {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl DaemonClient {
    pub fn global(repo_root: &Path) -> Result<&'static Mutex<Option<DaemonClient>>, String> {
        let mut guard = DAEMON.lock().map_err(|e| e.to_string())?;
        if guard.is_none() {
            *guard = Some(Self::spawn(repo_root)?);
        }
        Ok(&DAEMON)
    }

    fn spawn(repo_root: &Path) -> Result<Self, String> {
        let root = std::env::var("OPENFRONT_REPO")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| repo_root.to_path_buf());
        let tsx = root.join("openfront/node_modules/.bin/tsx");
        let script = root.join("bridge/engine_daemon.ts");
        if !tsx.exists() {
            return Err(format!("tsx not found at {}", tsx.display()));
        }
        if !script.exists() {
            return Err(format!("daemon not found at {}", script.display()));
        }
        let mut child = Command::new(&tsx)
            .arg(&script)
            .current_dir(&root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn daemon: {e}"))?;
        let stdin = child.stdin.take().ok_or("daemon stdin")?;
        let stdout = child
            .stdout
            .take()
            .ok_or("daemon stdout")
            .map(BufReader::new)?;
        Ok(Self {
            _child: child,
            stdin,
            stdout,
        })
    }

    pub fn new_session(&mut self) -> Result<String, String> {
        let (out, _) = self.rpc(json!({"op": "new"}), false)?;
        out.get("sid")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| "daemon new: missing sid".into())
    }

    pub fn drop_session(&mut self, sid: &str) -> Result<(), String> {
        let _ = self.rpc(json!({"op": "drop", "sid": sid}), false)?;
        Ok(())
    }

    pub fn reset(
        &mut self,
        sid: &str,
        map: &str,
        seed: &str,
        bots: u32,
    ) -> Result<(Value, Vec<u8>, Vec<u8>), String> {
        let msg = json!({
            "op": "reset",
            "sid": sid,
            "map": map,
            "seed": seed,
            "bots": bots,
            "difficulty": "Medium",
            "nations": "default",
        });
        let (mut head, tiles) = self.rpc(msg, true)?;
        let terrain_b64 = head
            .get("terrain")
            .and_then(|v| v.as_str())
            .ok_or("reset missing terrain")?
            .to_string();
        head.as_object_mut().map(|o| o.remove("terrain"));
        let terrain = decode_terrain_b64(&terrain_b64)?;
        let tiles = tiles.ok_or("reset missing tiles frame")?;
        Ok((head, tiles, terrain))
    }

    pub fn step(
        &mut self,
        sid: &str,
        intents: Vec<StampedIntent>,
        ticks: u32,
    ) -> Result<(Value, Vec<u8>), String> {
        let intents_json: Vec<Value> = intents
            .into_iter()
            .map(|i| {
                let mut v = i.fields;
                if let Some(obj) = v.as_object_mut() {
                    obj.insert("type".into(), Value::String(i.intent_type));
                    obj.insert("clientID".into(), Value::String(i.client_id));
                }
                v
            })
            .collect();
        let msg = json!({
            "op": "step",
            "sid": sid,
            "intents": intents_json,
            "ticks": ticks,
        });
        let (head, tiles) = self.rpc(msg, true)?;
        let tiles = tiles.ok_or("step missing tiles frame")?;
        Ok((head, tiles))
    }

    fn rpc(&mut self, msg: Value, expect_tiles: bool) -> Result<(Value, Option<Vec<u8>>), String> {
        let line = serde_json::to_string(&msg).map_err(|e| e.to_string())?;
        writeln!(self.stdin, "{line}").map_err(|e| e.to_string())?;
        self.stdin.flush().map_err(|e| e.to_string())?;

        let mut header = String::new();
        self.stdout
            .read_line(&mut header)
            .map_err(|e| e.to_string())?;
        if header.is_empty() {
            return Err("daemon died".into());
        }
        let mut out: Value = serde_json::from_str(&header).map_err(|e| e.to_string())?;
        if let Some(err) = out.get("error").and_then(|v| v.as_str()) {
            return Err(format!("daemon error: {err}"));
        }
        let tiles = if expect_tiles {
            if let Some(n) = out.get("tilesBin").and_then(|v| v.as_u64()) {
                out.as_object_mut().map(|o| o.remove("tilesBin"));
                Some(read_exact(&mut self.stdout, n as usize)?)
            } else {
                None
            }
        } else {
            None
        };
        Ok((out, tiles))
    }
}

pub struct DaemonSession {
    sid: String,
    repo_root: std::path::PathBuf,
}

impl DaemonSession {
    pub fn open(repo_root: &Path) -> Result<Self, String> {
        let guard = DaemonClient::global(repo_root)?;
        let mut g = guard.lock().map_err(|e| e.to_string())?;
        let client = g.as_mut().ok_or("daemon not running")?;
        let sid = client.new_session()?;
        Ok(Self {
            sid,
            repo_root: repo_root.to_path_buf(),
        })
    }

    pub fn reset(
        &mut self,
        map: &str,
        seed: &str,
        bots: u32,
    ) -> Result<(Value, Vec<u8>, Vec<u8>), String> {
        let guard = DaemonClient::global(&self.repo_root)?;
        let mut g = guard.lock().map_err(|e| e.to_string())?;
        let client = g.as_mut().ok_or("daemon not running")?;
        client.reset(&self.sid, map, seed, bots)
    }

    pub fn step(
        &mut self,
        intents: Vec<StampedIntent>,
        ticks: u32,
    ) -> Result<(Value, Vec<u8>), String> {
        let guard = DaemonClient::global(&self.repo_root)?;
        let mut g = guard.lock().map_err(|e| e.to_string())?;
        let client = g.as_mut().ok_or("daemon not running")?;
        client.step(&self.sid, intents, ticks)
    }
}

impl Drop for DaemonSession {
    fn drop(&mut self) {
        if let Ok(guard) = DaemonClient::global(&self.repo_root) {
            if let Ok(mut g) = guard.lock() {
                if let Some(client) = g.as_mut() {
                    let _ = client.drop_session(&self.sid);
                }
            }
        }
    }
}

fn read_exact<R: Read>(r: &mut R, n: usize) -> Result<Vec<u8>, String> {
    let mut buf = vec![0u8; n];
    let mut got = 0;
    while got < n {
        let k = r.read(&mut buf[got..]).map_err(|e| e.to_string())?;
        if k == 0 {
            return Err("daemon died mid-frame".into());
        }
        got += k;
    }
    Ok(buf)
}

fn decode_terrain_b64(b64: &str) -> Result<Vec<u8>, String> {
    use base64::Engine;
    let raw = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| e.to_string())?;
    let mut dec = flate2::read::GzDecoder::new(&raw[..]);
    let mut out = Vec::new();
    dec.read_to_end(&mut out).map_err(|e| e.to_string())?;
    Ok(out)
}

pub fn use_daemon() -> bool {
    std::env::var("OPENFRONT_DAEMON")
        .ok()
        .map(|v| v != "0")
        .unwrap_or(true)
}
