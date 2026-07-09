//! PyO3 RL environment - JSONL-free direct API (Phase 4).

use base64::{engine::general_purpose::STANDARD, Engine as _};
use flate2::write::GzEncoder;
use flate2::Compression;
use openfront_engine::record::StampedIntent;
use openfront_engine::session::EnvSession;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;
use std::io::Write;
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    std::env::var("OPENFRONT_REPO")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../..")
                .canonicalize()
                .unwrap_or_else(|_| PathBuf::from("."))
        })
}

#[pyclass(unsendable)]
struct NativeEnv {
    session: Option<EnvSession>,
    width: u32,
    height: u32,
    terrain_gz_b64: String,
}

#[pymethods]
impl NativeEnv {
    #[new]
    fn new() -> Self {
        Self {
            session: None,
            width: 0,
            height: 0,
            terrain_gz_b64: String::new(),
        }
    }

    fn reset(
        &mut self,
        py: Python<'_>,
        map: &str,
        seed: &str,
        bots: u32,
    ) -> PyResult<Py<PyDict>> {
        let (session, head, terrain, tiles) =
            EnvSession::reset(&repo_root(), map, seed, bots).map_err(PyValueError::new_err)?;
        self.width = head
            .get("width")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        self.height = head
            .get("height")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;
        let mut enc = GzEncoder::new(Vec::new(), Compression::fast());
        enc.write_all(&terrain).map_err(PyValueError::new_err)?;
        self.terrain_gz_b64 = STANDARD.encode(enc.finish().map_err(PyValueError::new_err)?);
        self.session = Some(session);
        obs_to_py(py, head, &tiles)
    }

    fn step(&mut self, py: Python<'_>, intents: Vec<PyObject>, ticks: u32) -> PyResult<Py<PyDict>> {
        let session = self
            .session
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("call reset() first"))?;
        let parsed = intents
            .into_iter()
            .map(|o| pyobject_to_intent(py, o))
            .collect::<PyResult<Vec<_>>>()?;
        let (head, tiles, _wasted) = session.step(&repo_root(), parsed, ticks);
        obs_to_py(py, head, &tiles)
    }

    #[getter]
    fn width(&self) -> u32 {
        self.width
    }

    #[getter]
    fn height(&self) -> u32 {
        self.height
    }

    #[getter]
    fn terrain(&self) -> String {
        self.terrain_gz_b64.clone()
    }
}

fn pyobject_to_intent(py: Python<'_>, obj: PyObject) -> PyResult<StampedIntent> {
    let json = py
        .import("json")?
        .call_method1("dumps", (obj,))?
        .extract::<String>()?;
    let v: Value =
        serde_json::from_str(&json).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let intent_type = v
        .get("type")
        .and_then(|x| x.as_str())
        .unwrap_or("noop")
        .to_string();
    let client_id = v
        .get("clientID")
        .and_then(|x| x.as_str())
        .unwrap_or("")
        .to_string();
    let mut fields = v;
    if let Some(obj) = fields.as_object_mut() {
        obj.remove("type");
        obj.remove("clientID");
    }
    Ok(StampedIntent {
        intent_type,
        client_id,
        fields,
    })
}

fn obs_to_py(py: Python<'_>, head: Value, tiles: &[u8]) -> PyResult<Py<PyDict>> {
    let json = py.import("json")?;
    let head_str =
        serde_json::to_string(&head).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let dict = json
        .call_method1("loads", (head_str,))?
        .cast_into::<PyDict>()?;
    dict.set_item("tiles_raw", tiles.to_vec())?;
    Ok(dict.unbind())
}

#[pymodule(gil_used = false)]
fn ofenv(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeEnv>()?;
    Ok(())
}
