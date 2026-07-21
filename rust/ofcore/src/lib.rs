//! Shared, Python-free core for the V8 Rust PPO trainer (`oftrain`).
//!
//! Fresh v7 port of the former Python `rl/obs.py` + `rl/curriculum.py` +
//! `rl/ppo_translate.py`. Re-implements current (v7) featurization from
//! live bridge / native engine JSON.

pub mod curriculum;
pub mod feat;
pub mod translate;

#[cfg(test)]
#[path = "translate_boat_build_tests.rs"]
mod translate_boat_build_tests;
