//! `ofshowcase archive` - serve GameRecords + clips (port of serve_replay.py).

use std::collections::HashMap;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use axum::body::Body;
use axum::extract::{Path as AxumPath, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Value};
use tower_http::cors::CorsLayer;

use crate::paths::state_path;
use crate::util::{featured_game_id, load_json};

#[derive(Clone)]
struct ArchiveState {
    records_dir: PathBuf,
    clips_dir: Option<PathBuf>,
    state_file: Option<PathBuf>,
    index: Arc<Mutex<HashMap<String, PathBuf>>>,
}

fn build_index(records_dir: &Path) -> HashMap<String, PathBuf> {
    let mut idx = HashMap::new();
    let walker = walkdir::WalkDir::new(records_dir).into_iter();
    for entry in walker.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        if path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(".debug.json"))
        {
            continue;
        }
        if let Ok(text) = fs::read_to_string(path) {
            if let Ok(v) = serde_json::from_str::<Value>(&text) {
                if let Some(gid) = v
                    .pointer("/info/gameID")
                    .and_then(|x| x.as_str())
                {
                    idx.insert(gid.to_string(), path.to_path_buf());
                }
            }
        }
    }
    idx
}

fn cors_json(status: StatusCode, body: Value) -> Response {
    let mut res = Json(body).into_response();
    *res.status_mut() = status;
    res.headers_mut().insert(
        header::ACCESS_CONTROL_ALLOW_ORIGIN,
        HeaderValue::from_static("*"),
    );
    res
}

async fn options_handler() -> impl IntoResponse {
    let mut headers = HeaderMap::new();
    headers.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
    headers.insert(header::ACCESS_CONTROL_ALLOW_HEADERS, HeaderValue::from_static("*"));
    (StatusCode::NO_CONTENT, headers)
}

async fn status(State(st): State<ArchiveState>) -> impl IntoResponse {
    let mut idx = st.index.lock().unwrap();
    *idx = build_index(&st.records_dir);
    let count = idx.len();
    let mut payload = if let Some(ref sp) = st.state_file {
        load_json(sp).unwrap_or_else(|_| json!({}))
    } else {
        json!({})
    };
    if let Some(obj) = payload.as_object_mut() {
        obj.insert("records".into(), json!(count));
    }
    cors_json(StatusCode::OK, payload)
}

async fn replay_redirect(State(st): State<ArchiveState>) -> Response {
    let state = if let Some(ref sp) = st.state_file {
        load_json(sp).unwrap_or_else(|_| json!({}))
    } else {
        load_json(&state_path()).unwrap_or_else(|_| json!({}))
    };
    let gid = featured_game_id(&state)
        .or_else(|| {
            state
                .get("game_id")
                .and_then(|v| v.as_str())
                .map(str::to_string)
        });
    match gid {
        Some(id) => Redirect::temporary(&format!("/game/{id}")).into_response(),
        None => cors_json(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({"status":"warming","message":"showcase replay is generating"}),
        ),
    }
}

async fn game(
    State(st): State<ArchiveState>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    {
        let mut idx = st.index.lock().unwrap();
        *idx = build_index(&st.records_dir);
    }
    let path = st.index.lock().unwrap().get(&id).cloned();
    match path {
        Some(p) => match fs::read(&p) {
            Ok(bytes) => {
                let mut res = Response::new(Body::from(bytes));
                *res.status_mut() = StatusCode::OK;
                res.headers_mut().insert(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("application/json"),
                );
                res.headers_mut().insert(
                    header::ACCESS_CONTROL_ALLOW_ORIGIN,
                    HeaderValue::from_static("*"),
                );
                res
            }
            Err(_) => cors_json(StatusCode::NOT_FOUND, json!({"error":"not found"})),
        },
        None => cors_json(StatusCode::NOT_FOUND, json!({"error":"not found"})),
    }
}

async fn debug(
    State(st): State<ArchiveState>,
    AxumPath(id): AxumPath<String>,
) -> Response {
    {
        let mut idx = st.index.lock().unwrap();
        *idx = build_index(&st.records_dir);
    }
    let path = st.index.lock().unwrap().get(&id).cloned();
    let side = path.map(|p| {
        let stem = p.with_extension("");
        // foo.json -> foo.debug.json
        PathBuf::from(format!("{}.debug.json", stem.display()))
    });
    match side {
        Some(p) if p.exists() => match fs::read(&p) {
            Ok(bytes) => {
                let mut res = Response::new(Body::from(bytes));
                *res.status_mut() = StatusCode::OK;
                res.headers_mut().insert(
                    header::CONTENT_TYPE,
                    HeaderValue::from_static("application/json"),
                );
                res.headers_mut().insert(
                    header::ACCESS_CONTROL_ALLOW_ORIGIN,
                    HeaderValue::from_static("*"),
                );
                res
            }
            Err(_) => cors_json(StatusCode::NOT_FOUND, json!({"error":"no debug sidecar"})),
        },
        _ => cors_json(StatusCode::NOT_FOUND, json!({"error":"no debug sidecar"})),
    }
}

async fn clip(
    State(st): State<ArchiveState>,
    AxumPath(name): AxumPath<String>,
) -> Response {
    let Some(dir) = &st.clips_dir else {
        return cors_json(StatusCode::NOT_FOUND, json!({"error":"clip not found"}));
    };
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
        || !name.ends_with(".webm")
    {
        return cors_json(StatusCode::NOT_FOUND, json!({"error":"clip not found"}));
    }
    let path = dir.join(&name);
    if !path.is_file() {
        return cors_json(StatusCode::NOT_FOUND, json!({"error":"clip not found"}));
    }
    match fs::read(&path) {
        Ok(bytes) => {
            let mut res = Response::new(Body::from(bytes));
            *res.status_mut() = StatusCode::OK;
            res.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("video/webm"),
            );
            res.headers_mut().insert(
                header::ACCESS_CONTROL_ALLOW_ORIGIN,
                HeaderValue::from_static("*"),
            );
            res.headers_mut().insert(
                header::CACHE_CONTROL,
                HeaderValue::from_static("public, max-age=300"),
            );
            res
        }
        Err(_) => cors_json(StatusCode::NOT_FOUND, json!({"error":"clip not found"})),
    }
}

async fn exists() -> impl IntoResponse {
    cors_json(StatusCode::OK, json!({"exists": false}))
}

async fn games(State(st): State<ArchiveState>) -> impl IntoResponse {
    let mut idx = st.index.lock().unwrap();
    *idx = build_index(&st.records_dir);
    let mut rows: Vec<Value> = idx
        .iter()
        .map(|(gid, path)| {
            let mtime = fs::metadata(path)
                .and_then(|m| m.modified())
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs())
                .unwrap_or(0);
            json!({
                "game_id": gid,
                "path": path.display().to_string(),
                "mtime": mtime,
            })
        })
        .collect();
    rows.sort_by(|a, b| {
        b.get("mtime")
            .and_then(|v| v.as_u64())
            .cmp(&a.get("mtime").and_then(|v| v.as_u64()))
    });
    let featured = st
        .state_file
        .as_ref()
        .and_then(|p| load_json(p).ok())
        .and_then(|s| featured_game_id(&s));
    cors_json(
        StatusCode::OK,
        json!({
            "count": rows.len(),
            "featured": featured,
            "games": rows,
        }),
    )
}

pub async fn run_archive(
    records: PathBuf,
    port: u16,
    bind: String,
    state: Option<PathBuf>,
    clips: Option<PathBuf>,
) -> Result<()> {
    let index = build_index(&records);
    println!(
        "serving {} record(s) on http://{bind}:{port}",
        index.len()
    );
    for (gid, f) in &index {
        println!("  {gid}  {}", f.display());
    }
    let st = ArchiveState {
        records_dir: records,
        clips_dir: clips,
        state_file: state,
        index: Arc::new(Mutex::new(index)),
    };
    let app = Router::new()
        .route("/status", get(status))
        .route("/replay", get(replay_redirect))
        .route("/games", get(games))
        .route("/game/{id}", get(game))
        .route("/debug/{id}", get(debug))
        .route("/clips/{name}", get(clip))
        .route("/api/game/{id}/exists", get(exists))
        .route("/", axum::routing::options(options_handler))
        .fallback(axum::routing::options(options_handler).get(|| async {
            cors_json(StatusCode::NOT_FOUND, json!({"error":"unknown route"}))
        }))
        .layer(CorsLayer::permissive())
        .with_state(st);

    let addr: SocketAddr = format!("{bind}:{port}").parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
