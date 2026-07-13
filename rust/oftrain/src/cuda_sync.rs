//! Opt-in CUDA synchronization boundaries used to localize asynchronous
//! device failures. Disabled diagnostics are a CPU-cheap no-op.

use std::collections::HashMap;
use std::fmt;
use std::panic::{self, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use anyhow::{Result, anyhow};
use tch::{Cuda, Device};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhaseContext {
    pub device: Device,
    pub shard: usize,
    pub update: u64,
    pub phase: String,
}

impl fmt::Display for PhaseContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "device={} shard={} update={} phase={}",
            device_name(self.device),
            self.shard,
            self.update,
            self.phase
        )
    }
}

#[derive(Clone, Default)]
pub struct CudaSyncDiagnostics {
    enabled: bool,
    last_success: Arc<Mutex<HashMap<usize, PhaseContext>>>,
}

impl CudaSyncDiagnostics {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            ..Self::default()
        }
    }

    pub fn context(
        &self,
        device: Device,
        shard: usize,
        update: u64,
        phase: impl fmt::Display,
    ) -> PhaseContext {
        PhaseContext {
            device,
            shard,
            update,
            phase: phase.to_string(),
        }
    }

    /// Synchronize only when diagnostics are enabled.
    pub fn boundary(
        &self,
        device: Device,
        shard: usize,
        update: u64,
        phase: impl fmt::Display,
    ) -> Result<()> {
        if !self.enabled || !device.is_cuda() {
            return Ok(());
        }
        self.synchronize(self.context(device, shard, update, phase))
    }

    /// Preserve an already-required synchronization when diagnostics are
    /// disabled, while adding the same context and tracking when enabled.
    pub fn required_boundary(
        &self,
        device: Device,
        shard: usize,
        update: u64,
        phase: impl fmt::Display,
    ) -> Result<()> {
        if !device.is_cuda() {
            return Ok(());
        }
        if !self.enabled {
            if let Device::Cuda(index) = device {
                Cuda::synchronize(index as i64);
            }
            return Ok(());
        }
        self.synchronize(self.context(device, shard, update, phase))
    }

    fn synchronize(&self, context: PhaseContext) -> Result<()> {
        let Device::Cuda(index) = context.device else {
            return Ok(());
        };
        let result = panic::catch_unwind(AssertUnwindSafe(|| Cuda::synchronize(index as i64)));
        match result {
            Ok(()) => {
                self.last_success
                    .lock()
                    .expect("CUDA diagnostics state poisoned")
                    .insert(index, context.clone());
                eprintln!("[cuda-sync-diagnostics] boundary ok: {context}");
                Ok(())
            }
            Err(payload) => {
                let panic_message = panic_payload_message(&payload);
                let last = self
                    .last_success
                    .lock()
                    .expect("CUDA diagnostics state poisoned")
                    .get(&index)
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "none".to_string());
                let error = anyhow!(
                    "CUDA synchronization failed: {context}; last successful boundary: {last}; \
                     libtorch error: {panic_message}"
                );
                eprintln!("[cuda-sync-diagnostics] {error:#}");
                Err(error)
            }
        }
    }

    #[cfg(test)]
    fn last_success(&self, device: Device) -> Option<PhaseContext> {
        let Device::Cuda(index) = device else {
            return None;
        };
        self.last_success.lock().unwrap().get(&index).cloned()
    }
}

fn device_name(device: Device) -> String {
    match device {
        Device::Cpu => "cpu".to_string(),
        Device::Cuda(index) => format!("cuda:{index}"),
        Device::Mps => "mps".to_string(),
        Device::Vulkan => "vulkan".to_string(),
    }
}

fn panic_payload_message(payload: &Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_context_format_includes_required_coordinates() {
        let context = PhaseContext {
            device: Device::Cuda(3),
            shard: 2,
            update: 41,
            phase: "learner.backward".to_string(),
        };
        assert_eq!(
            context.to_string(),
            "device=cuda:3 shard=2 update=41 phase=learner.backward"
        );
    }

    #[test]
    fn enabled_diagnostics_are_a_noop_on_cpu() {
        let diagnostics = CudaSyncDiagnostics::new(true);
        diagnostics
            .boundary(Device::Cpu, 7, 99, "cpu.test")
            .unwrap();
        diagnostics
            .required_boundary(Device::Cpu, 7, 99, "cpu.required_test")
            .unwrap();
        assert_eq!(diagnostics.last_success(Device::Cpu), None);
    }
}
