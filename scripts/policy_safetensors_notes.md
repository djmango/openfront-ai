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

Recurrent V8.2 checkpoints use architecture schema 2. The manifest records the
256-wide GRU, `action-outcome-v1` 14-float context, 128-wide context embedding,
`--bptt-chunk-len`, rollout length, zero-output residual initialization, and
the `episode_done` per-environment hidden reset policy. Full recurrent
`--resume` requires this v2 manifest plus matching v2 `TrainState`.

Migrate V8.1 explicitly with `--init-v81-recurrent latest.safetensors`.
That path requires a schema-v1 manifest, copies every legacy tensor exactly,
and permits only the actual `recurrent.*` tensors to remain newly initialized.
The residual projection stays zero, making all initial policy/value outputs
bit-identical to V8.1. Python consumers reject schema 2 until stateful
recurrent inference is implemented there.

AE encoders remain independent (`scripts/export_safetensors.py` +
`--ckpt` / `--coarse-ckpt`).

`scripts/export_onnx.py` can map the current Rust safetensors names into the
Python `Policy` strictly in memory. It rejects missing, extra, or
shape-mismatched tensors and does not emit `policy.pt`.
