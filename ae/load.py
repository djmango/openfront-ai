"""Load SpatialAE checkpoints for training, export, and oftrain encoders."""

from __future__ import annotations

from pathlib import Path

import torch

from ae.model_v3 import SpatialAE

# Policy / oftrain defaults: d8c32 fine AE (and d16c32 coarse).
REGION = 8
LATENT_C = 32
COARSE_FACTOR = 2
COARSE_REGION = REGION * COARSE_FACTOR


def load_ae(
    ckpt_path: str | Path,
    device: str = "cpu",
    expected_down: int = REGION,
    expected_c: int = LATENT_C,
) -> SpatialAE:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    ae = SpatialAE(
        latent_c=a["latent_c"],
        terrain_cond=a.get("terrain_cond", False),
        upsample_decoder=a.get("upsample_decoder", False),
        latent_down=a.get("latent_down", 16),
    ).to(device)
    ae.load_state_dict(ckpt["model_state_dict"])
    ae.eval()
    if ae.latent_down != expected_down or ae.latent_c != expected_c:
        raise ValueError(
            f"encoder {ckpt_path} is {ae.latent_c}ch @ 1/{ae.latent_down}; "
            f"expected {expected_c}ch @ 1/{expected_down}"
        )
    return ae
