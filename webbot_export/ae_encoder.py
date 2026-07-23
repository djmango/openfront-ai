"""Encoder-only SpatialAE for webbot ONNX export (loads ofae/oftrain safetensors).

v3.2 no-static: buildings are not an AE input; they bypass into the policy grid.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from safetensors.torch import load_file

MAX_SLOTS = 128
OWNER_EMB_DIM = 8
TERRAIN_CHANNELS = 3
# Kept for policy-side constants / docs; not an encoder input.
NUM_STATIC = 6


def conv_block(c_in: int, c_out: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(8, c_out),
        nn.SiLU(),
    )


class SpatialAEEncoder(nn.Module):
    """Matches oftrain/ofae encoder VarStore keys (v3.2 no-static)."""

    def __init__(self, latent_c: int = 32, latent_down: int = 8):
        super().__init__()
        if latent_down not in (8, 16):
            raise ValueError(f"latent_down must be 8 or 16, got {latent_down}")
        self.latent_c = latent_c
        self.latent_down = latent_down
        self.owner_emb = nn.Embedding(MAX_SLOTS, OWNER_EMB_DIM)
        stem = [
            conv_block(OWNER_EMB_DIM + TERRAIN_CHANNELS, 32, stride=1),
            conv_block(32, 64, stride=2),
            conv_block(64, 96, stride=2),
            conv_block(96, 128, stride=2),
        ]
        if latent_down == 16:
            stem.append(conv_block(128, 128, stride=2))
        self.enc_stem = nn.Sequential(*stem)
        self.enc_fuse = nn.Sequential(
            conv_block(128, 128, stride=1),
            nn.Conv2d(128, latent_c, kernel_size=1),
        )

    def encode(self, owners: torch.Tensor, terrain: torch.Tensor) -> torch.Tensor:
        emb = self.owner_emb(owners).permute(0, 3, 1, 2)
        g = self.enc_stem(torch.cat([emb, terrain], dim=1))
        return self.enc_fuse(g)

    def forward(self, owners, terrain):
        return self.encode(owners, terrain)


def load_ae_encoder(
    path: str | Path,
    device: str = "cpu",
    expected_down: int = 8,
    expected_c: int = 32,
) -> SpatialAEEncoder:
    path = Path(path)
    meta_path = path.with_suffix(".json")
    latent_c, latent_down = expected_c, expected_down
    if meta_path.exists():
        import json

        meta = json.loads(meta_path.read_text())
        latent_c = int(meta.get("latent_c", latent_c))
        latent_down = int(meta.get("latent_down", latent_down))
        if meta.get("static_in_latent", True) is not False and "v32" not in path.name:
            # Old v3.1 encoders fuse 6 static planes; refuse silent mismatch.
            raise ValueError(
                f"encoder {path} looks like a static-in-latent (v3.1) checkpoint; "
                "train/export ae_v32_nostatic instead"
            )
    if latent_down != expected_down or latent_c != expected_c:
        raise ValueError(
            f"encoder {path} is {latent_c}ch @ 1/{latent_down}; "
            f"expected {expected_c}ch @ 1/{expected_down}"
        )
    ae = SpatialAEEncoder(latent_c=latent_c, latent_down=latent_down).to(device)
    if path.suffix == ".safetensors":
        sd = load_file(str(path), device=device)
        ae.load_state_dict(sd, strict=True)
    else:
        raise ValueError(f"expected .safetensors encoder, got {path}")
    ae.eval()
    return ae
