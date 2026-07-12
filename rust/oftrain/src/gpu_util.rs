//! Background `nvidia-smi`/`rocm-smi` poller so the training loop can
//! report actual GPU utilization/memory alongside throughput - the whole
//! point of the RunPod scale-up exercise is verifying this stays
//! >=90-95% *on every GPU*, not just that training runs or that the
//! cluster-wide average looks good while one card idles. `nvidia-smi` is
//! tried first (the proven CUDA path); `rocm-smi` is a fallback tried only
//! when it's absent, for AMD/MI300X pods - see `query_rocm_smi`'s doc
//! comment and `rust/oftrain/ROCM.md` for what is/isn't verified here.

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
            if let Some((per_gpu_util, mem_pct)) = query_nvidia_smi().or_else(query_rocm_smi) {
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

/// AMD equivalent of `query_nvidia_smi`, tried as a fallback (see
/// `GpuUtilSampler::start`) only when `nvidia-smi` isn't on PATH - RunPod's
/// ROCm/MI300X images ship `rocm-smi` instead. The output shape genuinely
/// differs from nvidia-smi's `--query-gpu=...,--format=csv` - a header row
/// naming each column (not a fixed `--query-gpu` field order) and memory
/// reported in raw bytes rather than a `memory.used`/`memory.total` MiB
/// pair - so this gets its own parser (`parse_rocm_smi_csv`) rather than
/// reusing `query_nvidia_smi`'s.
///
/// NOT independently verified against real ROCm hardware in this sandbox
/// (no AMD GPU/`rocm-smi` available) - `parse_rocm_smi_csv`'s tests
/// exercise this against a real captured
/// `rocm-smi --showuse --showmeminfo vram --csv` invocation from
/// https://github.com/marimo-team/marimo/issues/9237, not a guessed
/// format, but that's still "should work per a documented real sample",
/// not "verified end-to-end on an MI300X". See `rust/oftrain/ROCM.md`.
fn query_rocm_smi() -> Option<(Vec<f64>, f64)> {
    let out = Command::new("rocm-smi")
        .args(["--showuse", "--showmeminfo", "vram", "--csv"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    parse_rocm_smi_csv(&String::from_utf8_lossy(&out.stdout))
}

/// Parses `rocm-smi --showuse --showmeminfo vram --csv` output, e.g.:
///
/// ```text
/// device,GPU use (%),VRAM Total Memory (B),VRAM Total Used Memory (B)
/// card0,0,21458059264,27856896
/// ```
///
/// Column *names* (not positions) drive the parse - sticking to exactly
/// these two flags (no `--showproductname` etc.) keeps every field
/// numeric/comma-free, but a future rocm-smi version reordering or
/// renaming columns should fail closed (`None`) rather than silently
/// misparse. Mirrors `query_nvidia_smi`'s return contract: per-GPU
/// utilization percentages plus one aggregate memory-used percentage
/// across all GPUs combined.
fn parse_rocm_smi_csv(text: &str) -> Option<(Vec<f64>, f64)> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines.next()?.split(',').map(|s| s.trim()).collect();
    let find = |name: &str| header.iter().position(|h| h.eq_ignore_ascii_case(name));
    let use_idx = find("GPU use (%)")?;
    let total_idx = find("VRAM Total Memory (B)")?;
    let used_idx = find("VRAM Total Used Memory (B)")?;

    let mut util_per_gpu = Vec::new();
    let mut used_sum = 0.0;
    let mut total_sum = 0.0;
    for line in lines {
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() <= use_idx.max(total_idx).max(used_idx) {
            continue;
        }
        let util: f64 = parts[use_idx].parse().ok()?;
        let used: f64 = parts[used_idx].parse().ok()?;
        let total: f64 = parts[total_idx].parse().ok()?;
        util_per_gpu.push(util);
        used_sum += used;
        total_sum += total;
    }
    if util_per_gpu.is_empty() {
        return None;
    }
    Some((util_per_gpu, 100.0 * used_sum / total_sum.max(1.0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real captured single-GPU output (trimmed to just the columns our
    /// invocation - `--showuse --showmeminfo vram --csv`, no
    /// `--showproductname` - actually requests), from
    /// https://github.com/marimo-team/marimo/issues/9237.
    const SAMPLE_SINGLE_GPU: &str = "device,GPU use (%),VRAM Total Memory (B),VRAM Total Used Memory (B)\ncard0,0,21458059264,27856896\n";

    /// Same real single-GPU sample, extended to a second `card1` row (the
    /// same repo issue shows multi-GPU output following this exact
    /// `cardN,...` row shape) to exercise per-GPU + aggregate math.
    const SAMPLE_MULTI_GPU: &str = "device,GPU use (%),VRAM Total Memory (B),VRAM Total Used Memory (B)\ncard0,12,21458059264,2145805926\ncard1,88,21458059264,19312253440\n";

    #[test]
    fn parses_single_gpu_sample() {
        let (util, mem_pct) = parse_rocm_smi_csv(SAMPLE_SINGLE_GPU).unwrap();
        assert_eq!(util, vec![0.0]);
        assert!((mem_pct - (100.0 * 27856896.0 / 21458059264.0)).abs() < 1e-6);
    }

    #[test]
    fn parses_multi_gpu_sample_per_gpu_and_aggregate() {
        let (util, mem_pct) = parse_rocm_smi_csv(SAMPLE_MULTI_GPU).unwrap();
        assert_eq!(util, vec![12.0, 88.0]);
        let expected = 100.0 * (2145805926.0 + 19312253440.0) / (21458059264.0 * 2.0);
        assert!((mem_pct - expected).abs() < 1e-6);
    }

    #[test]
    fn empty_output_is_none() {
        assert!(parse_rocm_smi_csv("").is_none());
    }

    #[test]
    fn header_only_is_none() {
        assert!(parse_rocm_smi_csv(
            "device,GPU use (%),VRAM Total Memory (B),VRAM Total Used Memory (B)\n"
        )
        .is_none());
    }

    #[test]
    fn missing_expected_columns_is_none() {
        // A hypothetical rocm-smi invocation/version that doesn't include
        // our expected columns should fail closed, not misparse.
        assert!(parse_rocm_smi_csv("device,Temperature (Sensor edge) (C)\ncard0,45.0\n").is_none());
    }
}
