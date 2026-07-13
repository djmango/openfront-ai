# Policy weight interchange (safetensors only)

`oftrain` saves and loads policy weights via tch `VarStore::save` /
`VarStore::load`. The preferred extension is **`.safetensors`** (tch
switches format on the path suffix). Legacy `.ot` checkpoints still load.

There is no Python `.pt` → Rust converter. PyTorch `state_dict` keys do not
match tch VarStore paths, and growing heads need overlap-copy logic that is
not automated. Prefer:

- Train / resume from Rust checkpoints (`checkpoints/latest.safetensors`)
- Warm-start with `oftrain --init path.safetensors` (weights only; no
  `TrainState` / AdamW moments)

AE encoders remain independent (`scripts/export_safetensors.py` +
`--ckpt` / `--coarse-ckpt`).

## Rust policy → Python policy

The current fixed 89-channel, 21-action Beta policy can be exported for the
Python showcase and ONNX tools:

```bash
uv run python -m scripts.export_oftrain_policy \
  checkpoints/latest.safetensors \
  checkpoints/latest.state.json \
  runs/rl/rust/policy.pt
```

The exporter has an explicit audited VarStore → `Policy.state_dict` mapping.
It rejects every missing or extra source key and every shape mismatch. Conv2d,
Linear, and LayerNorm tensors already use the same layouts; the only transform
is concatenating Rust's Q, K, and V weights/biases on dimension 0 for
PyTorch's fused `in_proj_weight`/`in_proj_bias`.

This is deliberately not a model-version migration tool: non-default
`--gc`/`--blocks`, older channel/action schemas, future architecture changes,
and `.ot` inputs are refused rather than guessed.
