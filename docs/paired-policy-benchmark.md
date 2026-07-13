# Rust versus frozen PPO policy benchmark

This is an **indirect paired scripted-bot benchmark**, not a direct
head-to-head tournament. Both RL engine surfaces currently create exactly one
controllable human (`AGENTRL1`); all other players are engine nations/tribes.
Consequently two policies cannot be assigned to distinct players in one game.

The harness runs each policy separately against the same current Node/TypeScript
engine, map, seed, bot count, difficulty, nation setting, decision cadence, and
tick cap. It refuses to report if the per-episode map/seed list differs. It
reports wins, placement, normalized placement score, 95% Wilson win intervals,
bootstrap intervals for means, and paired Rust-minus-old deltas.

## Compatibility matrix

| policy | source used for inference | grid / player / scalar / local | action / build / nuke | quantity | current Rust loader |
|---|---|---|---|---|---|
| current Rust | current checkout | 89 / 21 / 11 / 5 | 21 / 7 / 5 | Beta scalar | yes |
| `ppo_v7` | current Python checkout | 89 / 21 / 11 / 5 | 21 / 7 / 5 | Beta scalar | weights are Python `.pt`; no converter |
| `ppo_v5` | `23197d4^` (pre-v6 source) | 43 / 12 / 8 / 4 | 14 / 6 / 3 | 5-way categorical | no |

The old source archive is used only for observation construction, policy
inference, and action translation. Its `openfront/` and `bridge/` paths are
replaced with symlinks to the current checkout, so all legs use one engine.
Checkpoint tensor dimensions are compared to the expected schema before strict
`load_state_dict`; Rust uses strict `VarStore::load`. A mismatch is fatal.

## Checkpoints and command

Frozen checkpoints are public:

* `hf://djmango/openfront-rl/ppo_v5/policy.pt`
* `hf://djmango/openfront-rl/ppo_v7/policy.pt`
* archived Rust v8: `hf://djmango/openfront-rl/ppo_v8_rust_8gpu/policy_latest.ot`

The harness downloads v5/v7 to
`runs/rl/frozen-eval/{ppo_v5,ppo_v7}/policy.pt`. Supply the current Rust
checkpoint explicitly; `.ot` and `.safetensors` are accepted.

```bash
cargo build --manifest-path rust/Cargo.toml --release \
  -p oftrain --features native-engine

uv run python scripts/paired_policy_benchmark.py \
  --rust-checkpoint checkpoints/latest.safetensors \
  --rust-bin rust/target/release/oftrain \
  --rust-ae weights/ae/ae_v31_d8c32.encoder.safetensors \
  --rust-coarse-ae weights/ae/ae_v31_d16c32.encoder.safetensors \
  --python-ae runs/ae_v31_d8c32/ae_v3.pt \
  --python-coarse-ae runs/ae_v31_d16c32/ae_v3.pt \
  --stage 1 --episodes 64 \
  --out eval_out/paired-policy-benchmark.json
```

Python checkpoints need the original PyTorch AE files, while Rust needs
exported encoder safetensors. Public Python AE acquisition is:

```python
hf_hub_download(
    "djmango/openfront-tile-autoencoder", "ae_v31_d8c32.pt"
)
hf_hub_download(
    "djmango/openfront-tile-autoencoder", "ae_v31_d16c32.pt"
)
```

## Limitations

* Results measure performance against scripted opponents, not which policy
  beats the other. Shared seeds reduce variance but do not make bot trajectories
  identical after the tested policy changes game state.
* Placement is the trainer's composite-strength rank and score is
  `(players - place) / (players - 1)`. Historical reward totals are deliberately
  not compared because reward definitions changed.
* The Node engine is selected even if Rust training used the native engine.
  This is necessary to give legacy Python and Rust one common backend.
* Confidence intervals describe the selected deterministic seed set. Increase
  `--episodes` for useful precision; eight-episode trainer smoke evals are too
  small for model-selection claims.
* The Rust and Python AE checkpoint formats differ, so both paths are required.
