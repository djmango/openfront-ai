//! Recurrent-policy transport and actor-state ownership.
//!
//! Policy transitions live in `PolicyNet::{act,evaluate,value}_with_state`;
//! this module owns only device-state rows and host context packing.

use anyhow::Result;
use tch::{Device, Kind, Tensor};

use crate::vecenv::ActionOutcome;

pub const CONTEXT_FLOATS: usize = 14;

pub fn context_tensor(contexts: &[ActionOutcome], device: Device) -> Tensor {
    let mut values = Vec::with_capacity(contexts.len() * CONTEXT_FLOATS);
    for context in contexts {
        values.extend_from_slice(&context.as_floats());
    }
    Tensor::from_slice(&values)
        .view([contexts.len() as i64, CONTEXT_FLOATS as i64])
        .to_device(device)
}

/// Per-environment hidden rows. This object is constructed, used, and dropped
/// only inside a persistent actor owner thread.
pub struct ActorRecurrentState {
    hidden: Tensor,
    hidden_size: usize,
}

impl ActorRecurrentState {
    pub fn new(envs: usize, hidden_size: usize, device: Device) -> Self {
        Self {
            hidden: Tensor::zeros([envs as i64, hidden_size as i64], (Kind::Float, device)),
            hidden_size,
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn indices(&self, envs: &[usize]) -> Tensor {
        let indices: Vec<i64> = envs.iter().map(|&env| env as i64).collect();
        Tensor::from_slice(&indices).to_device(self.hidden.device())
    }

    pub fn gather(&self, envs: &[usize]) -> Tensor {
        self.hidden.index_select(0, &self.indices(envs))
    }

    pub fn scatter(&mut self, envs: &[usize], hidden_out: &Tensor) -> Result<()> {
        anyhow::ensure!(
            hidden_out.size() == [envs.len() as i64, self.hidden_size as i64],
            "recurrent hidden_out shape {:?}, expected [{}, {}]",
            hidden_out.size(),
            envs.len(),
            self.hidden_size
        );
        anyhow::ensure!(
            hidden_out.device() == self.hidden.device(),
            "recurrent hidden_out device mismatch"
        );
        self.hidden = self.hidden.index_copy(0, &self.indices(envs), hidden_out);
        Ok(())
    }

    pub fn reset(&mut self, env: usize) -> Result<()> {
        anyhow::ensure!(
            env < self.hidden.size()[0] as usize,
            "recurrent env {env} out of range"
        );
        let zero = Tensor::zeros(
            [1, self.hidden_size as i64],
            (Kind::Float, self.hidden.device()),
        );
        self.scatter(&[env], &zero)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockRecurrentPolicy;

    impl MockRecurrentPolicy {
        fn advance(&self, hidden_in: &Tensor, context: &Tensor) -> Tensor {
            hidden_in + context.narrow(1, 0, 1)
        }
    }

    #[test]
    fn staggered_batches_follow_env_identity_and_reset() {
        let policy = MockRecurrentPolicy;
        let mut state = ActorRecurrentState::new(4, 3, Device::Cpu);

        let first_envs = [2usize, 0];
        let first_context = Tensor::from_slice(&[20.0f32, 10.0]).view([2, 1]);
        let first = state.gather(&first_envs);
        state
            .scatter(&first_envs, &policy.advance(&first, &first_context))
            .unwrap();

        let second_envs = [1usize, 2];
        let second_context = Tensor::from_slice(&[5.0f32, 1.0]).view([2, 1]);
        let second = state.gather(&second_envs);
        state
            .scatter(&second_envs, &policy.advance(&second, &second_context))
            .unwrap();

        let all: Vec<f32> = state
            .gather(&[0, 1, 2, 3])
            .reshape([-1])
            .try_into()
            .unwrap();
        assert_eq!(
            all,
            vec![
                10.0, 10.0, 10.0, // env 0
                5.0, 5.0, 5.0, // env 1
                21.0, 21.0, 21.0, // env 2 retained its first-batch state
                0.0, 0.0, 0.0, // env 3 never dispatched
            ]
        );

        state.reset(2).unwrap();
        let reset: Vec<f32> = state.gather(&[2]).reshape([-1]).try_into().unwrap();
        assert_eq!(reset, vec![0.0; 3]);
    }
}
