# Policy weight interchange (safetensors only)

`oftrain` saves policy weights via tch `VarStore::save` exclusively as
**`.safetensors`**. Automated pod restore, Hugging Face sync, v8.1 launch,
and current-policy benchmark paths select only safetensors. `oftrain` still
loads an explicitly supplied legacy `.ot` for one-time migrations; launchers
never discover or prefer it.

There is **no** Python `.pt` → Rust converter. PyTorch `state_dict` keys
do not match tch VarStore paths, and growing heads need overlap-copy logic
that is not automated. Prefer:

- Train / resume from Rust checkpoints (`checkpoints/latest.safetensors`)
- Warm-start with `oftrain --init path.safetensors` (weights only; no
  `TrainState` / AdamW moments)
- Expect `latest.state.json`, `best_eval.safetensors` plus its state sidecar,
  and `policy_update*.safetensors` milestones to sync with the weights.
- Read `manifest.json` for `format: "oftrain-safetensors"`, manifest and
  architecture schema versions, model dimensions, AE references, update,
  and curriculum stage.

AE encoders remain independent (`scripts/export_safetensors.py` +
`--ckpt` / `--coarse-ckpt`).

`scripts/export_onnx.py` can map the current Rust safetensors names into the
Python `Policy` strictly in memory. It rejects missing, extra, or
shape-mismatched tensors and does not emit `policy.pt`.
