---
license: mit
library_name: safetensors
tags:
  - reinforcement-learning
  - ppo
  - openfront
  - game-ai
  - pytorch
language: []
pipeline_tag: reinforcement-learning
---

# openfront-rl

PPO policy checkpoints for a self-play agent on
[OpenFront.io](https://openfront.io), trained with the Rust `oftrain` stack in
[djmango/openfront-ai](https://github.com/djmango/openfront-ai).

Encoders used by these policies live in a sibling repo:
[`djmango/openfront-tile-autoencoder`](https://huggingface.co/djmango/openfront-tile-autoencoder).

## Latest (use this)

| Field | Value |
|-------|-------|
| **Run** | `ppo_v11` |
| **Pointer** | [`ppo_v11/latest.safetensors`](ppo_v11/latest.safetensors) + [`ppo_v11/latest.state.json`](ppo_v11/latest.state.json) + [`ppo_v11/manifest.json`](ppo_v11/manifest.json) |
| **Update** | **2486** (milestone file `policy_update2485.*` is the last numbered snapshot; `latest` is one step ahead) |
| **Curriculum stage** | 23 |
| **Approx. env steps** | 16.2M |
| **Reward / schedule** | `v10-anti-spiral-v1` / curriculum `v10` |
| **Weights size** | ~154 MB (safetensors) |

Download the current policy:

```bash
huggingface-cli download djmango/openfront-rl \
  ppo_v11/latest.safetensors \
  ppo_v11/latest.state.json \
  ppo_v11/manifest.json \
  --local-dir ./ppo_v11
```

Or from the trainer / play scripts (they already default to this repo):

```bash
# restore into a checkpoint dir (validates manifest)
uv run python scripts/hf_checkpoint_sync.py \
  --checkpoint-dir rust/checkpoints/ppo_v11 \
  --run-prefix ppo_v11 \
  --restore-latest

# live play helper
RUN_NAME=ppo_v11 bash scripts/play_live.sh
```

Always prefer **`ppo_v11/latest.*`** over digging through numbered milestones.
`manifest.json` is the compatibility gate (`format=oftrain-safetensors`).

## Layout

```
ppo_v11/                         # current training run
  latest.safetensors             # ← production pointer
  latest.state.json
  manifest.json                  # architecture + AE refs + update/stage
  policy_updateNNNN.*            # thinned milestone history
  curriculum_advance_*.*         # snapshots at stage promotions
  curriculum_demote_*.*          # snapshots at stage demotions

ppo_v10/ … ppo_v81/              # prior runs: latest.* + manifest only
```

Older dense `policy_update*` histories (every ~5 updates) were pruned —
adjacent milestones are nearly identical for most uses. Kept history for
`ppo_v11` is every 100 updates, plus every 25 in the most recent ~200 updates,
plus all curriculum advance/demote snapshots.

## Architecture (from `manifest.json`)

- **Policy:** `oftrain-policy` schema v3 — spatial grid tower + player/unit
  streams, legality-masked discrete actions (full OpenFront intent surface).
- **Recurrent:** LSTM (`hidden_size=512`, BPTT 24 / rollout 48),
  `action-outcome-v1` context, reset on `episode_done`.
- **Observation:** frozen tile autoencoders (`ae_v32_nostatic` fine 1/8 +
  coarse 1/16) with exact-state bypass (diplomacy, scalars, transients).
- **Training:** PPO + GAE, win-gated multi-map curriculum, native engine
  (Node hedge optional).

See the living design notes in the git repo:
[`DESIGN.md`](https://github.com/djmango/openfront-ai/blob/master/DESIGN.md).

## Prior runs

| Run | Role | What remains here |
|-----|------|-------------------|
| `ppo_v11` | **Current** | latest + thinned milestones + curriculum snapshots |
| `ppo_v10` | Previous mainline | `latest.*` + `manifest.json` only |
| `ppo_v9`, `ppo_v86`…`ppo_v81` | Lineage / ablations | `latest.*` + `manifest.json` only |
| Early `ppo_v*` / `bc_*` | Legacy | removed in the HF cleanup (recoverable from git LFS/history only if re-uploaded) |

## Retention

Going forward, training pods should not re-flood the Hub with every-5-update
milestones. Use `scripts/hf_prune_openfront_rl.py` after large syncs, and prefer
syncing `latest.*` / `manifest.json` / curriculum markers plus sparse
milestones.

## License / credit

MIT, matching [djmango/openfront-ai](https://github.com/djmango/openfront-ai).
OpenFront itself is a separate project — this repo only hosts learned weights
and trainer metadata.
