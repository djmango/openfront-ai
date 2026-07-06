"""Spatial-only autoencoder (v3): compress only what is actually big.

Lesson from v2: squeezing tiny exact state (pairwise alliances, player
scalars) through the latent and reconstructing it lossily is manufactured
difficulty — three training runs got alliance precision to 0.55 when the
policy could just read the bit. The bypass split:

  AE compresses (high-dimensional, spatial):
    - tile ownership grid + terrain + fallout
    - static structure planes (city/port/defense post/silo/SAM/factory)

  Bypass straight to the policy (small, exact):
    - pairwise diplomacy (alliances, embargoes, expiry, pending requests)
    - per-player scalars (troops, gold, tiles, alive, ...)
    - transient units (nukes/transports/warships) as (owner, pos, target)
    - attack aggregates, globals, legality masks

Latent: z_grid (B, latent_c, H/16, W/16). No global vector: the policy can
pool spatially itself, and per-player/global state arrives via bypass.
"""

import torch
import torch.nn as nn

from ae.units import STATIC_CLASSES

MAX_SLOTS = 128
OWNER_EMB_DIM = 8
TERRAIN_CHANNELS = 3  # land, magnitude, fallout
NUM_STATIC = len(STATIC_CLASSES)

# BCE positive-weight multiplier per static class (× --unit-pos-weight).
STATIC_CLASS_WEIGHTS = [1.0, 1.0, 1.0, 4.0, 4.0, 2.0]


def conv_block(c_in: int, c_out: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1),
        nn.GroupNorm(8, c_out),
        nn.SiLU(),
    )


def deconv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(8, c_out),
        nn.SiLU(),
    )


class SpatialAE(nn.Module):
    def __init__(self, latent_c: int = 64):
        super().__init__()
        self.latent_c = latent_c
        self.owner_emb = nn.Embedding(MAX_SLOTS, OWNER_EMB_DIM)

        self.enc_stem = nn.Sequential(
            conv_block(OWNER_EMB_DIM + TERRAIN_CHANNELS, 32, stride=1),
            conv_block(32, 64, stride=2),
            conv_block(64, 96, stride=2),
            conv_block(96, 128, stride=2),
            conv_block(128, 128, stride=2),  # -> 1/16
        )
        self.enc_fuse = nn.Sequential(
            conv_block(128 + NUM_STATIC, 128, stride=1),
            nn.Conv2d(128, latent_c, kernel_size=1),
        )

        self.dec_in = conv_block(latent_c, 128, stride=1)
        self.dec_tiles = nn.Sequential(
            deconv_block(128, 128),
            deconv_block(128, 96),
            deconv_block(96, 64),
            deconv_block(64, 32),
            nn.Conv2d(32, MAX_SLOTS, kernel_size=1),
        )
        # Static structure occupancy logits at 1/16 resolution.
        self.dec_units = nn.Conv2d(128, NUM_STATIC, kernel_size=1)

    def encode(
        self,
        owners: torch.Tensor,  # (B, H, W) int64
        terrain: torch.Tensor,  # (B, 3, H, W)
        static_planes: torch.Tensor,  # (B, NUM_STATIC, H/16, W/16)
    ) -> torch.Tensor:
        emb = self.owner_emb(owners).permute(0, 3, 1, 2)
        g = self.enc_stem(torch.cat([emb, terrain], dim=1))
        return self.enc_fuse(torch.cat([g, static_planes], dim=1))

    def decode(self, z_grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.dec_in(z_grid)
        return self.dec_tiles(h), self.dec_units(h)

    def forward(self, owners, terrain, static_planes):
        z_grid = self.encode(owners, terrain, static_planes)
        tile_logits, unit_logits = self.decode(z_grid)
        return tile_logits, unit_logits, z_grid
