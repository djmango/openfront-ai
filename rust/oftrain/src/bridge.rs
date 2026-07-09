//! Subprocess JSONL bridge client: one persistent `tsx bridge/env.ts` per
//! env, matching `rl/env.py::OpenFrontEnv`. Since Rust threads don't fight
//! a GIL, each `VecEnv` slot just gets its own OS thread blocking on this
//! process's stdio (see `vecenv.rs`) - no multiprocessing/pickle framing
//! needed like the Python side.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};
use flate2::read::GzDecoder;
use serde_json::{json, Value};

use crate::engine::{decode_tiles, GameEngine, RawObs};

/// Cap on buffered stderr lines per bridge; only kept so a crash/hang can
/// be diagnosed (the Node engine's own `console.*` calls land here - see
/// bridge/env.ts's stdout==pure-JSONL redirect). Previously stderr was
/// `Stdio::null()`'d entirely, so a crashed child gave zero information
/// beyond "bridge died" - the same blind spot that made the Python
/// engine-bridge crash loop undiagnosable without `py-spy`/`faulthandler`.
const STDERR_TAIL_LINES: usize = 200;

pub struct Bridge {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    width: usize,
    height: usize,
    /// Untrimmed raw terrain bytes (height x width), set on reset().
    terrain: Vec<u8>,
}

pub(crate) fn repo_root() -> Result<PathBuf> {
    // rust/oftrain/src/bridge.rs -> repo root is 3 dirs up from the crate.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest_dir
        .parent()
        .and_then(Path::parent)
        .ok_or_else(|| anyhow!("cannot resolve repo root from {:?}", manifest_dir))?;
    Ok(root.to_path_buf())
}

impl Bridge {
    pub fn spawn() -> Result<Self> {
        let root = repo_root()?;
        let tsx = root.join("openfront").join("node_modules").join(".bin").join("tsx");
        let mut child = Command::new(&tsx)
            .arg(root.join("bridge").join("env.ts"))
            .current_dir(&root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("spawning {tsx:?}"))?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;
        let stdout = BufReader::new(child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?);
        let stderr = child.stderr.take().ok_or_else(|| anyhow!("no stderr"))?;
        let stderr_tail = Arc::new(Mutex::new(VecDeque::with_capacity(STDERR_TAIL_LINES)));
        // Must be drained continuously: an unread stderr pipe fills its OS
        // buffer and blocks the child's next write, silently hanging the
        // env the moment it logs enough (Node's console.* all land on
        // stderr - see bridge/env.ts's stdout redirect).
        let tail_writer = stderr_tail.clone();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stderr);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        let mut tail = tail_writer.lock().unwrap();
                        if tail.len() >= STDERR_TAIL_LINES {
                            tail.pop_front();
                        }
                        tail.push_back(line.trim_end().to_string());
                    }
                }
            }
        });
        Ok(Bridge {
            child,
            stdin,
            stdout,
            stderr_tail,
            width: 0,
            height: 0,
            terrain: Vec::new(),
        })
    }

    fn stderr_context(&self) -> String {
        let tail = self.stderr_tail.lock().unwrap();
        if tail.is_empty() {
            "(no stderr output)".to_string()
        } else {
            tail.iter().cloned().collect::<Vec<_>>().join("\n")
        }
    }

    fn read_exact_n(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; n];
        if let Err(e) = self.stdout.read_exact(&mut buf) {
            bail!("bridge died mid-frame: {e}\n--- stderr tail ---\n{}", self.stderr_context());
        }
        Ok(buf)
    }

    fn rpc(&mut self, msg: &Value) -> Result<(Value, Vec<u8>)> {
        let mut line = serde_json::to_string(msg)?;
        line.push('\n');
        if let Err(e) = self.stdin.write_all(line.as_bytes()) {
            bail!("bridge stdin closed: {e}\n--- stderr tail ---\n{}", self.stderr_context());
        }
        self.stdin.flush()?;
        let mut resp = String::new();
        let n = self
            .stdout
            .read_line(&mut resp)
            .with_context(|| format!("bridge stdout closed\n--- stderr tail ---\n{}", self.stderr_context()))?;
        if n == 0 {
            bail!("bridge died\n--- stderr tail ---\n{}", self.stderr_context());
        }
        let mut out: Value = serde_json::from_str(&resp)
            .with_context(|| format!("bad bridge response: {resp}"))?;
        if let Some(e) = out.get("error") {
            bail!("bridge error: {e}");
        }
        let tiles = if let Some(nbytes) = out.get("tilesBin").and_then(|v| v.as_u64()) {
            out.as_object_mut().unwrap().remove("tilesBin");
            self.read_exact_n(nbytes as usize)?
        } else {
            Vec::new()
        };
        Ok((out, tiles))
    }

    fn decode(&self, mut head: Value, tiles_raw: Vec<u8>) -> RawObs {
        let n = self.width * self.height;
        let (owners, fallout, defense_bonus) = decode_tiles(&tiles_raw, n);
        if let Some(obj) = head.as_object_mut() {
            obj.remove("terrain");
        }
        RawObs { head, owners, fallout, defense_bonus }
    }

    #[allow(dead_code)] // debug/replay tooling, not used by the training loop
    pub fn save_record(&mut self, path: &str) -> Result<Value> {
        Ok(self.rpc(&json!({"op": "save_record", "path": path}))?.0)
    }
}

impl GameEngine for Bridge {
    fn reset(
        &mut self,
        map_name: &str,
        seed: &str,
        bots: u32,
        difficulty: &str,
        nations: Value,
    ) -> Result<RawObs> {
        let (head, tiles) = self.rpc(&json!({
            "op": "reset",
            "map": map_name,
            "seed": seed,
            "bots": bots,
            "difficulty": difficulty,
            "nations": nations,
        }))?;
        self.width = head["width"].as_u64().ok_or_else(|| anyhow!("no width"))? as usize;
        self.height = head["height"].as_u64().ok_or_else(|| anyhow!("no height"))? as usize;
        let terr_b64 = head["terrain"].as_str().ok_or_else(|| anyhow!("no terrain"))?;
        let gz = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, terr_b64)?;
        let mut dec = GzDecoder::new(&gz[..]);
        let mut terrain = Vec::with_capacity(self.width * self.height);
        dec.read_to_end(&mut terrain)?;
        self.terrain = terrain;
        Ok(self.decode(head, tiles))
    }

    fn step(&mut self, intents: &[Value], ticks: u32) -> Result<RawObs> {
        let (head, tiles) = self.rpc(&json!({"op": "step", "intents": intents, "ticks": ticks}))?;
        Ok(self.decode(head, tiles))
    }

    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn terrain(&self) -> &[u8] {
        &self.terrain
    }

    fn close(&mut self) {
        let _ = self.rpc(&json!({"op": "close"}));
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        // `kill()` alone leaves a zombie until this process exits - fine
        // for a short-lived run, but fatal for a supervised/auto-restart
        // loop that spawns many generations of workers (exactly the
        // orphaned-process/pipe-hang failure mode hit on the Python side).
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
