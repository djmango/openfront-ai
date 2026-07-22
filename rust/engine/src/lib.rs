//! OpenFront engine - Rust port:
//!
//! 1. **harness** - GameRecord load, decompress, hash/prng parity, replay CLI
//! 2. **tick** - `Game`, `Execution`, `execute_next_tick`
//! 3. **bot** - tribe spawner + `TribeExecution` + attack behavior
//! 4. **RL session** - consumed in-process by `oftrain` (`native-engine`)

pub mod backend;
pub mod bootstrap;
pub mod bot;
pub mod core;
pub mod execution;
pub mod game;
pub mod hash;
pub mod map;
pub mod obs;
pub mod obs_typed;
pub mod prng;
pub mod rail;
pub mod record;
pub mod replay;
pub mod rl;
#[cfg(feature = "parallel")]
pub mod rl_batch;
pub mod session;
pub mod spatial;
#[cfg(test)]
mod test_util;
pub mod util;
pub mod unit_grid;
pub mod water;
pub mod water_hpa;

pub use backend::Backend;
pub use game::Game;
pub use replay::{ReplayOptions, ReplayResult};
