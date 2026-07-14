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

V8.2 uses architecture schema 2. Its manifest records the GRU hidden size,
context feature schema/width, BPTT length, zero-output residual initialization,
and `episode_done` per-environment hidden reset policy. A full `--resume`
requires this v2 manifest and a v2 `TrainState`; schema-v1 weights cannot be
mistaken for a resume. Migrate V8.1 explicitly with:

```sh
oftrain --init-v81-recurrent checkpoints/v81/latest.safetensors ...
```

That path requires a schema-v1 manifest, copies every old tensor exactly, and
permits only `context.*` and `recurrent.*` destination keys to be new. The
recurrent residual projection starts at zero, preserving all V8.1 outputs.
Python showcase/ONNX consumers intentionally reject schema 2 until they gain
stateful recurrent inference support.

AE encoders remain independent (`scripts/export_safetensors.py` +
`--ckpt` / `--coarse-ckpt`).

`scripts/export_onnx.py` can map the current Rust safetensors names into the
Python `Policy` strictly in memory. It rejects missing, extra, or
shape-mismatched tensors and does not emit `policy.pt`.
