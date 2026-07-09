//! Simulation backends.

pub mod bridge;
pub mod daemon;
pub mod node;
pub mod stub;

use clap::ValueEnum;

/// Which engine runs the simulation.
#[derive(Debug, Clone, Copy, ValueEnum, Default)]
pub enum Backend {
    /// Full TypeScript engine (`hash_verify.ts` for replay, daemon/bridge for RL).
    #[default]
    Ts,
    /// Growing Rust port - uses record bootstrap; hash gates in progress.
    Native,
    /// Minimal Rust stub (`OPENFRONT_STUB=1` only).
    Stub,
}
