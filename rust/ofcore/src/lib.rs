//! Shared, Python-free core for the V8 Rust PPO trainer (`oftrain`).
//!
//! This is a *fresh* v7 port of `rl/obs.py` + `rl/curriculum.py` +
//! `rl/ppo_translate.py`, independent of `rust/ofrs`. `ofrs::feat` is
//! deliberately frozen at obs v4 (it is the on-disk BC cache format), so
//! it cannot be "upgraded in place" to v7 without breaking every existing
//! cache-bc blob. `ofcore` instead re-implements the current (v7)
//! featurization straight from the live bridge JSON, matching
//! `rl/obs.py::ObsBuilder.prepare()` field-for-field.

pub mod curriculum;
pub mod feat;
pub mod translate;
