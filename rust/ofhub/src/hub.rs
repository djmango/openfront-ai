//! `ofshowcase hub` - landing / watch / play (port of showcase_hub.py).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{Html, IntoResponse, Redirect, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Value};
use tokio::sync::Mutex;

use crate::hf;
use crate::paths::{hub_state_path, repo_root, state_path};
use crate::util::{
    featured_showcase_entry, game_map_api_name, load_json, showcase_maps, utc_now, write_json,
};
use rand::seq::SliceRandom;

const LANDING_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OpenFront RL Agent</title>
  <link rel="icon" href="/favicon.ico" sizes="any" />
  <link rel="icon" type="image/png" href="/favicon-32.png" sizes="32x32" />
  <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
  <style>
    * { box-sizing: border-box; margin: 0; }
    body {
      line-height: 1.45;
      font-size: 18px;
      padding: 3rem 1.25rem 4rem;
      color: #000;
      background: #fff;
      text-align: center;
    }
    .page { max-width: 920px; margin: 0 auto; }
    h1 {
      font-size: clamp(2.4rem, 7vw, 4rem);
      font-weight: 800;
      letter-spacing: -.03em;
      line-height: 1.05;
      margin-bottom: 1rem;
    }
    .lead {
      font-size: clamp(1.1rem, 2.5vw, 1.35rem);
      font-weight: 500;
      max-width: 36rem;
      margin: 0 auto 2rem;
      color: #111;
    }
    .lead a { font-weight: 600; }
    .preview {
      margin: 0 auto 2rem;
      max-width: 860px;
      border: 2px solid #000;
      background: #000;
    }
    .preview video {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      border: 0;
      background: #000;
    }
    .placeholder {
      aspect-ratio: 16 / 9;
      display: grid;
      place-items: center;
      color: #888;
      font-size: 1rem;
      background: #111;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      gap: 1rem 1.5rem;
      margin-bottom: 1.25rem;
    }
    .actions a {
      font-size: 1.15rem;
      font-weight: 700;
      text-decoration: underline;
      text-underline-offset: 4px;
    }
    .meta { font-size: .95rem; color: #666; margin-bottom: 1.5rem; }
    .links { font-size: 1rem; color: #666; }
    .links a { font-weight: 600; }
    .sep { margin: 0 .5rem; }
  </style>
</head>
<body>
  <main class="page">
    <h1>OpenFront Agent</h1>
    <p class="lead">A reinforcement learning agent that plays
      <a href="https://openfront.io">OpenFront.io</a>, trained on the real
      game engine with live model overlay. Play it 1v1.</p>
    <figure class="preview">%%PREVIEW%%</figure>
    <div class="actions">
      <a href="/watch">%%WATCH_LABEL%%</a>
      <a href="/play">Play vs Agent</a>
    </div>
    <p class="meta">policy: %%RUN_NAME%%</p>
    <p class="links">
      <a href="https://skg.gg" target="_blank" rel="noopener">skg.gg</a>
      <span class="sep">·</span>
      <a href="https://skg.gg/pages/openfront-devlog/" target="_blank" rel="noopener">Devlog</a>
      <span class="sep">·</span>
      <a href="https://github.com/djmango/openfront-ai" target="_blank" rel="noopener">GitHub</a>
    </p>
  </main>
</body>
</html>
"#;

#[derive(Clone)]
struct HubInner {
    run_name: String,
    client_host: String,
    admin_key: String,
    play_map: String,
    play_bots: i64,
    play_nations: i64,
    play_start_delay: i64,
    debug_port: u16,
    live_debug_port: u16,
    live_showcase: bool,
    worker_base_port: u16,
    repo: PathBuf,
    active: Arc<Mutex<ActiveLobby>>,
}

#[derive(Default)]
struct ActiveLobby {
    game_id: Option<String>,
    child: Option<Child>,
}

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn play_config(inner: &HubInner) -> Value {
    // PLAY_MAP=random (default) picks uniformly from SHOWCASE_MAPS each lobby.
    let game_map = if inner.play_map.is_empty()
        || inner.play_map.eq_ignore_ascii_case("random")
    {
        let maps = showcase_maps();
        maps.choose(&mut rand::thread_rng())
            .cloned()
            .unwrap_or_else(|| "World".into())
    } else {
        inner.play_map.clone()
    };
    json!({
        "gameMap": game_map_api_name(&game_map),
        "gameType": "Private",
        "bots": inner.play_bots,
        "difficulty": "Easy",
        "nations": inner.play_nations,
        "startDelay": inner.play_start_delay,
    })
}

async fn http_json(
    method: &str,
    url: &str,
    body: Option<Value>,
    admin_key: &str,
) -> Result<Value> {
    let client = reqwest::Client::new();
    let mut req = match method {
        "POST" => client.post(url),
        _ => client.get(url),
    };
    req = req
        .header("Content-Type", "application/json")
        .header("x-admin-bot-key", admin_key)
        .timeout(Duration::from_secs(30));
    if let Some(b) = body {
        req = req.json(&b);
    }
    let resp = req.send().await?;
    let status = resp.status();
    let text = resp.text().await?;
    if !status.is_success() {
        anyhow::bail!("{method} {url} -> {status}: {text}");
    }
    Ok(serde_json::from_str(&text).unwrap_or_else(|_| json!({})))
}

fn play_redirect(game_id: &str, worker_path: &str) -> String {
    format!("/{worker_path}/game/{game_id}")
}

fn load_replay() -> Value {
    load_json(&state_path()).unwrap_or_else(|_| json!({}))
}

fn load_hub() -> Value {
    load_json(&hub_state_path()).unwrap_or_else(|_| json!({}))
}

fn watch_target() -> (String, String, Option<Value>) {
    let replay = load_replay();
    let featured = featured_showcase_entry(&replay);
    let gid = featured
        .as_ref()
        .and_then(|e| e.get("game_id"))
        .and_then(|v| v.as_str())
        .or_else(|| replay.get("game_id").and_then(|v| v.as_str()));
    if let Some(gid) = gid {
        return (format!("/game/{gid}"), "replay".into(), featured);
    }
    (String::new(), "none".into(), None)
}

fn preview_markup(replay: &Value) -> String {
    let featured = featured_showcase_entry(replay);
    let mut clip_url = featured
        .as_ref()
        .and_then(|e| e.get("url"))
        .and_then(|v| v.as_str())
        .map(str::to_string);
    if clip_url.is_none() {
        if let Some(entries) = replay.get("hero_clips").and_then(|v| v.as_array()) {
            for entry in entries {
                clip_url = entry
                    .as_str()
                    .map(str::to_string)
                    .or_else(|| {
                        entry
                            .get("url")
                            .and_then(|v| v.as_str())
                            .map(str::to_string)
                    });
                if clip_url.is_some() {
                    break;
                }
            }
        }
    }
    if let Some(url) = clip_url {
        let map_label = featured
            .as_ref()
            .and_then(|e| e.get("map"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let title = if map_label.is_empty() {
            "Replay preview".to_string()
        } else {
            format!("Replay preview ({map_label})")
        };
        format!(
            r#"<video autoplay muted loop playsinline preload="auto" src="{url}" title="{title}"></video>"#
        )
    } else {
        r#"<div class="placeholder">Preview loading...</div>"#.to_string()
    }
}

fn render_landing(replay: &Value, run_name: &str) -> String {
    let (_, mode, featured) = watch_target();
    let map_name = featured
        .as_ref()
        .and_then(|e| e.get("map"))
        .and_then(|v| v.as_str())
        .or_else(|| replay.get("map").and_then(|v| v.as_str()));
    let mut watch_label = if mode == "replay" {
        "Watch replay".to_string()
    } else {
        "Watch".to_string()
    };
    if let Some(m) = map_name {
        watch_label = format!("Watch {m}");
    }
    LANDING_HTML
        .replace(
            "%%RUN_NAME%%",
            replay
                .get("run_name")
                .and_then(|v| v.as_str())
                .unwrap_or(run_name),
        )
        .replace("%%PREVIEW%%", &preview_markup(replay))
        .replace("%%WATCH_LABEL%%", &watch_label)
}

fn launch_webbot(inner: &HubInner, game_id: &str, worker_path: &str) -> Result<Child> {
    let script = inner.repo.join("scripts/webbot_launcher.py");
    let py = std::env::var("PYTHON").unwrap_or_else(|_| "python3".into());
    Command::new(py)
        .arg(&script)
        .args([
            "--host",
            &inner.client_host,
            "--game",
            game_id,
            "--worker-path",
            worker_path,
            "--debug-port",
            &inner.debug_port.to_string(),
        ])
        .current_dir(&inner.repo)
        .stdout(Stdio::null())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("spawn webbot_launcher for {game_id}"))
}

async fn wait_for_webbot_join(inner: &HubInner, game_id: &str, worker_path: &str) -> bool {
    let url = format!(
        "http://{}/{}/api/game/{}",
        inner.client_host, worker_path, game_id
    );
    let client = reqwest::Client::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(40);
    while tokio::time::Instant::now() < deadline {
        if let Ok(resp) = client.get(&url).timeout(Duration::from_secs(3)).send().await {
            if let Ok(info) = resp.json::<Value>().await {
                if info
                    .get("clients")
                    .and_then(|c| c.as_array())
                    .is_some_and(|a| !a.is_empty())
                {
                    return true;
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    false
}

async fn start_play_lobby(inner: &HubInner, game_id: &str, worker_index: i64) -> Result<Value> {
    let base = format!(
        "http://127.0.0.1:{}",
        inner.worker_base_port as i64 + worker_index
    );
    http_json(
        "POST",
        &format!("{base}/api/adminbot/game/{game_id}/intent"),
        Some(json!({"type": "toggle_game_start_timer"})),
        &inner.admin_key,
    )
    .await
}

async fn create_play_lobby(inner: &HubInner, config: &Value) -> Result<Value> {
    let base = format!("http://{}", inner.client_host);
    http_json(
        "POST",
        &format!("{base}/api/adminbot/create_game"),
        Some(config.clone()),
        &inner.admin_key,
    )
    .await
}

async fn landing(State(inner): State<Arc<HubInner>>) -> impl IntoResponse {
    let replay = load_replay();
    Html(render_landing(&replay, &inner.run_name))
}

async fn watch() -> Response {
    let (target, mode, featured) = watch_target();
    if target.is_empty() {
        return Json(json!({"status":"warming","message":"no replay yet"}))
            .into_response();
    }
    let mut res = Redirect::temporary(&target).into_response();
    *res.status_mut() = StatusCode::FOUND;
    res.headers_mut().insert(
        "x-showcase-watch",
        HeaderValue::from_str(&mode).unwrap_or_else(|_| HeaderValue::from_static("replay")),
    );
    if let Some(map) = featured
        .as_ref()
        .and_then(|e| e.get("map"))
        .and_then(|v| v.as_str())
    {
        if let Ok(v) = HeaderValue::from_str(map) {
            res.headers_mut().insert("x-showcase-map", v);
        }
    }
    res
}

async fn status(State(inner): State<Arc<HubInner>>) -> impl IntoResponse {
    let (target, mode, featured) = watch_target();
    let replay = load_replay();
    Json(json!({
        "watch": {
            "url": if target.is_empty() { Value::Null } else { json!(target) },
            "mode": mode,
            "map": featured.as_ref().and_then(|e| e.get("map")).cloned()
                .or_else(|| replay.get("map").cloned()),
            "game_id": featured.as_ref().and_then(|e| e.get("game_id")).cloned()
                .or_else(|| replay.get("game_id").cloned()),
            "selection": "random",
        },
        "replay": replay,
        "hub": load_hub(),
        "play_config": play_config(&inner),
        "live_showcase": inner.live_showcase,
    }))
}

async fn play_debug(
    State(inner): State<Arc<HubInner>>,
    AxumPath(game_id): AxumPath<String>,
) -> Response {
    let hub = load_hub();
    let mut ports = vec![inner.debug_port];
    if hub.get("live_game_id").and_then(|v| v.as_str()) == Some(game_id.as_str()) {
        ports.insert(0, inner.live_debug_port);
    }
    if !ports.contains(&inner.live_debug_port) {
        ports.push(inner.live_debug_port);
    }
    let client = reqwest::Client::new();
    for port in ports {
        let url = format!("http://127.0.0.1:{port}/debug/{game_id}");
        if let Ok(resp) = client.get(&url).timeout(Duration::from_secs(2)).send().await {
            if resp.status().is_success() {
                if let Ok(bytes) = resp.bytes().await {
                    let mut res = Response::new(Body::from(bytes));
                    *res.status_mut() = StatusCode::OK;
                    res.headers_mut().insert(
                        header::CONTENT_TYPE,
                        HeaderValue::from_static("application/json"),
                    );
                    return res;
                }
            }
        }
    }
    let mut res = Json(json!({"error":"no live debug feed"})).into_response();
    *res.status_mut() = StatusCode::NOT_FOUND;
    res
}

async fn play(State(inner): State<Arc<HubInner>>) -> Response {
    // Each Play click gets a fresh random map. Tear down any previous webbot
    // so we don't keep redirecting everyone into the same Onion lobby.
    {
        let mut active = inner.active.lock().await;
        if let Some(mut child) = active.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        if let Some(gid) = active.game_id.take() {
            eprintln!("[showcase_hub] replacing previous lobby {gid}");
        }
    }

    let config = play_config(&inner);
    let map_label = config
        .get("gameMap")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    eprintln!("[showcase_hub] creating play lobby map={map_label}");

    let info = match create_play_lobby(&inner, &config).await {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[showcase_hub] lobby create failed: {e}");
            let mut res = Json(json!({"error": e.to_string()})).into_response();
            *res.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
            return res;
        }
    };
    let game_id = info
        .get("gameID")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let worker_index = info
        .get("workerIndex")
        .and_then(|v| v.as_i64())
        .unwrap_or(0);
    let worker_path = info
        .get("workerPath")
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| format!("w{worker_index}"));

    {
        let mut active = inner.active.lock().await;
        match launch_webbot(&inner, &game_id, &worker_path) {
            Ok(child) => {
                active.game_id = Some(game_id.clone());
                active.child = Some(child);
            }
            Err(e) => {
                let mut res = Json(json!({"error": e.to_string()})).into_response();
                *res.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                return res;
            }
        }
    }

    let mut hub_payload = json!({
        "game_id": game_id,
        "status": "lobby",
        "config": config,
        "run_name": inner.run_name,
        "started_at": utc_now(),
        "worker_index": worker_index,
        "worker_path": worker_path,
    });
    if !wait_for_webbot_join(&inner, &game_id, &worker_path).await {
        eprintln!(
            "[showcase_hub] webbot didn't join {game_id} within timeout; arming countdown anyway"
        );
    }
    match start_play_lobby(&inner, &game_id, worker_index).await {
        Ok(started) => {
            hub_payload["status"] = json!("countdown");
            hub_payload["starts_at"] = started.get("startsAt").cloned().unwrap_or(Value::Null);
        }
        Err(e) => {
            eprintln!("[showcase_hub] lobby start failed: {e}");
            hub_payload["start_error"] = json!(e.to_string());
        }
    }
    let _ = write_json(&hub_state_path(), &hub_payload);
    let redirect = play_redirect(&game_id, &worker_path);
    eprintln!(
        "[showcase_hub] agent joining {game_id} map={map_label}; redirect -> {redirect}"
    );
    Redirect::temporary(&redirect).into_response()
}

pub async fn run_hub(port: u16) -> Result<()> {
    let data = crate::paths::data_dir();
    std::fs::create_dir_all(&data)?;

    let run_name = env_or("RUN_NAME", "ppo_v81");
    eprintln!("[showcase_hub] loading policy + encoder via HF (best-effort)");
    let client = hf::client_with_optional_token()?;
    let _policy = hf::ensure_policy(&client, &run_name).await;
    if let Err(e) = &_policy {
        eprintln!("[showcase_hub] policy ensure failed (hub still serves): {e}");
    }

    let inner = Arc::new(HubInner {
        run_name,
        client_host: env_or("CLIENT_HOST", "127.0.0.1:9000"),
        admin_key: env_or(
            "ADMIN_BOT_API_KEY",
            "WARNING_DEV_ADMIN_BOT_KEY_DO_NOT_USE_IN_PRODUCTION",
        ),
        play_map: env_or("PLAY_MAP", "random"),
        play_bots: env_or("PLAY_BOTS", "10").parse().unwrap_or(10),
        play_nations: env_or("PLAY_NATIONS", "1").parse().unwrap_or(1),
        play_start_delay: env_or("PLAY_START_DELAY", "30").parse().unwrap_or(30),
        debug_port: env_or("PLAY_DEBUG_PORT", "8989").parse().unwrap_or(8989),
        live_debug_port: env_or("LIVE_DEBUG_PORT", "8990").parse().unwrap_or(8990),
        live_showcase: env_or("LIVE_SHOWCASE", "0") != "0",
        worker_base_port: env_or("WORKER_BASE_PORT", "3001").parse().unwrap_or(3001),
        repo: repo_root(),
        active: Arc::new(Mutex::new(ActiveLobby::default())),
    });

    if inner.live_showcase {
        eprintln!("[showcase_hub] LIVE_SHOWCASE deferred in Rust hub (use webbot /play path)");
    }

    let app = Router::new()
        .route("/", get(landing))
        .route("/watch", get(watch))
        .route("/replay", get(watch))
        .route("/status", get(status))
        .route("/play", get(play))
        .route("/play/debug/{id}", get(play_debug))
        .fallback(|| async {
            let mut headers = HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
            (StatusCode::NOT_FOUND, headers, r#"{"error":"unknown route"}"#)
        })
        .with_state(inner);

    eprintln!("[showcase_hub] hub on :{port} (watch=/watch, play=/play, selection=random)");
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
