//! JSONL subprocess client for `bridge/env.ts` - full TS engine parity.

use crate::record::StampedIntent;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};

pub struct BridgeClient {
    _child: Child,
    stdin: ChildStdin,
    stdout: BufReader<std::process::ChildStdout>,
}

impl BridgeClient {
    pub fn spawn(repo_root: &Path) -> Result<Self, String> {
        let root = std::env::var("OPENFRONT_REPO")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| repo_root.to_path_buf());
        let tsx = root.join("openfront/node_modules/.bin/tsx");
        let script = root.join("bridge/env.ts");
        if !tsx.exists() {
            return Err(format!("tsx not found at {}", tsx.display()));
        }
        if !script.exists() {
            return Err(format!("bridge not found at {}", script.display()));
        }
        let mut child = Command::new(&tsx)
            .arg(&script)
            .current_dir(&root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("spawn bridge: {e}"))?;
        let stdin = child.stdin.take().ok_or("bridge stdin")?;
        let stdout = child
            .stdout
            .take()
            .ok_or("bridge stdout")
            .map(BufReader::new)?;
        Ok(Self {
            _child: child,
            stdin,
            stdout,
        })
    }

    pub fn reset(
        &mut self,
        map: &str,
        seed: &str,
        bots: u32,
    ) -> Result<(Value, Vec<u8>, Vec<u8>), String> {
        let msg = json!({
            "op": "reset",
            "map": map,
            "seed": seed,
            "bots": bots,
            "difficulty": "Medium",
            "nations": "default",
        });
        let (mut head, tiles) = self.rpc(msg)?;
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
            "intents": intents_json,
            "ticks": ticks,
        });
        let (head, tiles) = self.rpc(msg)?;
        let tiles = tiles.ok_or("step missing tiles frame")?;
        Ok((head, tiles))
    }

    pub fn close(&mut self) -> Result<(), String> {
        let _ = self.rpc(json!({"op": "close"}));
        Ok(())
    }

    fn rpc(&mut self, msg: Value) -> Result<(Value, Option<Vec<u8>>), String> {
        let line = serde_json::to_string(&msg).map_err(|e| e.to_string())?;
        writeln!(self.stdin, "{line}").map_err(|e| e.to_string())?;
        self.stdin.flush().map_err(|e| e.to_string())?;

        let mut header = String::new();
        self.stdout
            .read_line(&mut header)
            .map_err(|e| e.to_string())?;
        if header.is_empty() {
            return Err("bridge died".into());
        }
        let mut out: Value = serde_json::from_str(&header).map_err(|e| e.to_string())?;
        if let Some(err) = out.get("error").and_then(|v| v.as_str()) {
            return Err(format!("bridge error: {err}"));
        }
        let tiles = if let Some(n) = out.get("tilesBin").and_then(|v| v.as_u64()) {
            out.as_object_mut().map(|o| o.remove("tilesBin"));
            Some(read_exact(&mut self.stdout, n as usize)?)
        } else {
            None
        };
        Ok((out, tiles))
    }
}

fn read_exact<R: Read>(r: &mut R, n: usize) -> Result<Vec<u8>, String> {
    let mut buf = vec![0u8; n];
    let mut got = 0;
    while got < n {
        let k = r.read(&mut buf[got..]).map_err(|e| e.to_string())?;
        if k == 0 {
            return Err("bridge died mid-frame".into());
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
