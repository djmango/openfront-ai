# Policy weight interchange (safetensors only)

`oftrain` saves and loads policy weights via tch `VarStore::save` /
`VarStore::load`. The preferred extension is **`.safetensors`** (tch
switches format on the path suffix). Legacy `.ot` checkpoints still load.

There is **no** Python `.pt` → Rust converter. PyTorch `state_dict` keys
do not match tch VarStore paths, and growing heads need overlap-copy logic
that is not automated. Prefer:

- Train / resume from Rust checkpoints (`checkpoints/latest.safetensors`)
- Warm-start with `oftrain --init path.safetensors` (weights only; no
  `TrainState` / AdamW moments)

AE encoders remain independent (`scripts/export_safetensors.py` +
`--ckpt` / `--coarse-ckpt`).
