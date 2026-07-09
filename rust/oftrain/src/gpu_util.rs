//! Background `nvidia-smi` poller so the training loop can report actual
//! GPU utilization/memory alongside throughput - the whole point of the
//! RunPod scale-up exercise is verifying this stays >=90-95% *on every
//! GPU*, not just that training runs or that the cluster-wide average
//! looks good while one card idles.

use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

#[derive(Clone, Default)]
pub struct GpuSnapshot {
    /// Mean utilization/memory across all visible GPUs, this sample.
    pub util_pct: f64,
    pub mem_pct: f64,
    /// Running mean utilization since the sampler started, per GPU index
    /// (for the "sustained >=95%" check - an instantaneous readout can
    /// look fine right after a slow patch resolves).
    pub util_mean_per_gpu: Vec<f64>,
    /// Instantaneous per-GPU utilization, this sample.
    pub util_per_gpu: Vec<f64>,
    pub samples: u64,
}

impl GpuSnapshot {
    pub fn min_mean_util(&self) -> f64 {
        self.util_mean_per_gpu.iter().cloned().fold(f64::INFINITY, f64::min).max(0.0)
    }
}

pub struct GpuUtilSampler {
    state: Arc<Mutex<GpuSnapshot>>,
    _handle: JoinHandle<()>,
}

impl GpuUtilSampler {
    pub fn start(interval: Duration) -> Self {
        let state = Arc::new(Mutex::new(GpuSnapshot::default()));
        let state2 = state.clone();
        let handle = std::thread::spawn(move || loop {
            if let Some((per_gpu_util, mem_pct)) = query_nvidia_smi() {
                let mut s = state2.lock().unwrap();
                let prev_samples = s.samples;
                let n = prev_samples + 1;
                if s.util_mean_per_gpu.len() != per_gpu_util.len() {
                    s.util_mean_per_gpu = per_gpu_util.clone();
                } else {
                    for (m, &u) in s.util_mean_per_gpu.iter_mut().zip(&per_gpu_util) {
                        *m = (*m * prev_samples as f64 + u) / n as f64;
                    }
                }
                s.util_pct = per_gpu_util.iter().sum::<f64>() / per_gpu_util.len().max(1) as f64;
                s.util_per_gpu = per_gpu_util;
                s.mem_pct = mem_pct;
                s.samples = n;
            }
            std::thread::sleep(interval);
        });
        GpuUtilSampler { state, _handle: handle }
    }

    pub fn snapshot(&self) -> GpuSnapshot {
        self.state.lock().unwrap().clone()
    }
}

fn query_nvidia_smi() -> Option<(Vec<f64>, f64)> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut util_per_gpu = Vec::new();
    let mut used_sum = 0.0;
    let mut total_sum = 0.0;
    for line in text.lines() {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() != 3 {
            continue;
        }
        let util: f64 = parts[0].parse().ok()?;
        let used: f64 = parts[1].parse().ok()?;
        let total: f64 = parts[2].parse().ok()?;
        util_per_gpu.push(util);
        used_sum += used;
        total_sum += total;
    }
    if util_per_gpu.is_empty() {
        return None;
    }
    Some((util_per_gpu, 100.0 * used_sum / total_sum.max(1.0)))
}
