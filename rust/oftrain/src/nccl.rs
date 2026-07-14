//! Optional NCCL reducer for persistent learner owners.
//!
//! The feature is deliberately link-time gated: CUDA/PyTorch wheels do not
//! provide a stable system NCCL ABI to discover safely with `dlopen`, while
//! the shim consumes PyTorch C++ tensor objects and therefore must be compiled
//! against the exact `LIBTORCH` headers and C++ ABI. Non-NCCL builds return
//! `None` and retain the CPU gradient hub.

use anyhow::Result;
use tch::Device;

#[cfg(feature = "nccl")]
mod linked {
    use super::*;
    use anyhow::{Context, anyhow, ensure};
    use std::ffi::{CStr, c_char, c_int, c_void};
    use tch::{Kind, Tensor};

    unsafe extern "C" {
        fn oftrain_nccl_last_error() -> *const c_char;
        fn oftrain_nccl_init_all(
            world: c_int,
            devices: *const c_int,
            out: *mut *mut c_void,
        ) -> c_int;
        fn oftrain_nccl_all_reduce(comm: *mut c_void, tensor: *mut c_void) -> c_int;
        fn oftrain_nccl_destroy(comm: *mut c_void) -> c_int;
    }

    fn last_error(operation: &str) -> anyhow::Error {
        let pointer = unsafe { oftrain_nccl_last_error() };
        let detail = if pointer.is_null() {
            "native shim returned no detail".into()
        } else {
            unsafe { CStr::from_ptr(pointer) }
                .to_string_lossy()
                .into_owned()
        };
        anyhow!("{operation} failed: {detail}")
    }

    /// A rank-local communicator. It is moved once into its learner owner and
    /// is neither cloned nor shared. The native handle contains no Tensor.
    pub(crate) struct Comm {
        raw: *mut c_void,
        device: usize,
    }

    // NCCL communicators may be handed to their permanent owner thread after
    // ncclCommInitAll. All subsequent use and destruction occurs there.
    unsafe impl Send for Comm {}

    impl Comm {
        pub(crate) fn all_reduce_average(&mut self, flat: &mut Tensor) -> Result<()> {
            ensure!(
                flat.device() == Device::Cuda(self.device),
                "NCCL gradient is on {:?}, communicator is for cuda:{}",
                flat.device(),
                self.device
            );
            ensure!(
                flat.kind() == Kind::Float && flat.size().len() == 1,
                "NCCL gradient must be a flat f32 tensor"
            );
            let status =
                unsafe { oftrain_nccl_all_reduce(self.raw, flat.as_mut_ptr().cast::<c_void>()) };
            if status == 0 {
                Ok(())
            } else {
                Err(last_error("NCCL gradient all-reduce"))
            }
        }
    }

    impl Drop for Comm {
        fn drop(&mut self) {
            if self.raw.is_null() {
                return;
            }
            let status = unsafe { oftrain_nccl_destroy(self.raw) };
            if status != 0 && !std::thread::panicking() {
                eprintln!(
                    "[nccl] communicator destruction error: {}",
                    last_error("destroy")
                );
            }
            self.raw = std::ptr::null_mut();
        }
    }

    pub(super) fn init(devices: &[Device]) -> Result<Option<Vec<Comm>>> {
        if devices.len() < 2
            || !devices
                .iter()
                .all(|device| matches!(device, Device::Cuda(_)))
        {
            return Ok(None);
        }
        let ids: Vec<c_int> = devices
            .iter()
            .map(|device| match device {
                Device::Cuda(index) => c_int::try_from(*index).context("CUDA index exceeds int"),
                _ => unreachable!(),
            })
            .collect::<Result<_>>()?;
        ensure!(
            ids.iter().copied().eq(0..ids.len() as c_int),
            "NCCL persistent learners require contiguous cuda:0..N-1 devices"
        );
        let mut handles = vec![std::ptr::null_mut(); ids.len()];
        let status = unsafe {
            oftrain_nccl_init_all(ids.len() as c_int, ids.as_ptr(), handles.as_mut_ptr())
        };
        if status != 0 {
            return Err(last_error("NCCL communicator initialization"));
        }
        ensure!(
            handles.iter().all(|handle| !handle.is_null()),
            "NCCL initialization returned a null communicator"
        );
        Ok(Some(
            handles
                .into_iter()
                .zip(ids)
                .map(|(raw, device)| Comm {
                    raw,
                    device: device as usize,
                })
                .collect(),
        ))
    }
}

#[cfg(feature = "nccl")]
pub(crate) use linked::Comm;

#[cfg(not(feature = "nccl"))]
pub(crate) struct Comm;

#[cfg(not(feature = "nccl"))]
impl Comm {
    pub(crate) fn all_reduce_average(&mut self, _flat: &mut tch::Tensor) -> Result<()> {
        anyhow::bail!("NCCL support was not compiled")
    }
}

/// Initializes all rank communicators together. Initialization errors are
/// returned to the caller, which may choose the CPU fallback before training
/// starts. Collective errors are returned by `Comm::all_reduce_average` and
/// are never eligible for fallback.
pub(crate) fn try_init(devices: &[Device]) -> Result<Option<Vec<Comm>>> {
    #[cfg(feature = "nccl")]
    {
        linked::init(devices)
    }
    #[cfg(not(feature = "nccl"))]
    {
        let _ = devices;
        Ok(None)
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn cpu_and_single_device_groups_do_not_initialize_nccl() {
        assert!(try_init(&[Device::Cpu, Device::Cpu]).unwrap().is_none());
        assert!(try_init(&[Device::Cuda(0)]).unwrap().is_none());
    }
}

#[cfg(all(test, feature = "nccl"))]
mod tests {
    use super::*;
    use tch::{Cuda, Tensor};

    #[test]
    #[ignore = "requires two CUDA devices and a linked NCCL runtime"]
    fn two_gpu_sum_then_divide_uses_owner_local_tensors() {
        assert!(Cuda::is_available());
        assert!(Cuda::device_count() >= 2);
        let comms = try_init(&[Device::Cuda(0), Device::Cuda(1)])
            .unwrap()
            .unwrap();
        let handles: Vec<_> = comms
            .into_iter()
            .enumerate()
            .map(|(rank, mut comm)| {
                std::thread::spawn(move || {
                    let mut flat =
                        Tensor::from_slice(&[rank as f32 + 1.0, 4.0]).to_device(Device::Cuda(rank));
                    comm.all_reduce_average(&mut flat).unwrap();
                    Vec::<f32>::try_from(flat.to_device(Device::Cpu)).unwrap()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), vec![1.5, 4.0]);
        }
    }
}
