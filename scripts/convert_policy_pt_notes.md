# Python ↔ Rust policy weight interchange

`oftrain --init path.ot` loads a **tch VarStore** dump (`.ot` / `.safetensors`
via `VarStore::load`). It does **not** restore AdamW/TrainState (use
`--resume` for that).

## What works today

| Source | How |
|--------|-----|
| Rust checkpoint (`checkpoints/latest.ot`) | `--init checkpoints/latest.ot` or `--resume ...` |
| Another oftrain `.ot` with the same `PolicyNet` layout | `--init` |

## Python `policy.pt` → Rust `.ot`

A mechanical converter is **not** shipped: PyTorch `state_dict` keys
(`module.grid_tower…`) do not match tch VarStore paths
(`grid_tower…` / integer module indices from `nn::seq_t`), and several heads
grew between BC and v8. Matching requires:

1. Construct `PolicyNet` in Rust (or a tiny Python script that mirrors the
   exact VarStore tree) and dump `vs.variables()` key names.
2. Map each Python `state_dict` tensor onto a same-shape VarStore key.
3. For grown heads (action/build/nuke), copy overlapping rows and leave new
   rows at init — see `rl/ppo.py::init_extend`.
4. Save with `VarStore::save` → `.ot`, then `oftrain --init that.ot`.

Until that mapping is automated, prefer:

- Export BC/RL weights **from Rust** (`policy_*.ot`), or
- Train from scratch / resume Rust checkpoints only.

## AE encoders (already solved)

AE weights use safetensors with **matching** PyTorch keys via
`scripts/export_safetensors.py` + `--ckpt` / `--coarse-ckpt`. That path is
independent of policy `.ot` interchange.

## Stub warm-start

```bash
# Random AE-aligned policy (same architecture), then train:
oftrain --ckpt weights/ae/ae_v31_d8c32.encoder.safetensors \
  --coarse-ckpt weights/ae/ae_v31_d16c32.encoder.safetensors \
  --device cpu --num-envs 1 --updates 1 --engine native
# Then: oftrain --init checkpoints/policy_final.ot ...
```
