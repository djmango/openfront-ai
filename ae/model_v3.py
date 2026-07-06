"""Spatial-only autoencoder (v3): compress only what is actually big.

Lesson from v2: squeezing tiny exact state (pairwise alliances, player
scalars) through the latent and reconstructing it lossily is manufactured
difficulty - three training runs got alliance precision to 0.55 when the
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

v3.1 additions (all off by default so old v3 checkpoints load unchanged):
  - terrain_cond: the decoder consumes the 2 STATIC terrain planes (land,
    magnitude) as free side-information at every scale, so the latent only
    has to encode ownership relative to terrain. Fallout is dynamic state
    and is never fed to the decoder.
  - upsample_decoder: nearest-upsample + 3x3 conv stages (no checkerboard)
    plus a full-resolution 3x3 refinement block before the classifier.
  - latent_down: 8 or 16; latent grid at 1/8 or 1/16 resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class UpsampleBlock(nn.Module):
    """Nearest 2x upsample + 3x3 conv (no ConvTranspose checkerboard).

    Optionally concatenates static terrain planes (at the post-upsample
    resolution) before the conv.
    """

    def __init__(self, c_in: int, c_out: int, extra_c: int = 0):
        super().__init__()
        self.conv = conv_block(c_in + extra_c, c_out, stride=1)

    def forward(self, x: torch.Tensor, extra: torch.Tensor | None = None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if extra is not None:
            x = torch.cat([x, extra], dim=1)
        return self.conv(x)


STATIC_TERRAIN_C = 2  # land, magnitude (fallout is dynamic: never decoded from)


class SpatialAE(nn.Module):
    """Defaults preserve the original v3 architecture (old checkpoints load
    with strict state_dict matching). v3.1 runs set terrain_cond=True and
    upsample_decoder=True (and optionally latent_down=8)."""

    def __init__(
        self,
        latent_c: int = 64,
        terrain_cond: bool = False,
        upsample_decoder: bool = False,
        latent_down: int = 16,
    ):
        super().__init__()
        if latent_down not in (8, 16):
            raise ValueError(f"latent_down must be 8 or 16, got {latent_down}")
        if latent_down == 8 and not upsample_decoder:
            raise ValueError("latent_down=8 requires the v3.1 upsample decoder")
        if terrain_cond and not upsample_decoder:
            raise ValueError("terrain_cond requires the v3.1 upsample decoder")
        self.latent_c = latent_c
        self.terrain_cond = terrain_cond
        self.upsample_decoder = upsample_decoder
        self.latent_down = latent_down
        self.owner_emb = nn.Embedding(MAX_SLOTS, OWNER_EMB_DIM)

        stem = [
            conv_block(OWNER_EMB_DIM + TERRAIN_CHANNELS, 32, stride=1),
            conv_block(32, 64, stride=2),
            conv_block(64, 96, stride=2),
            conv_block(96, 128, stride=2),
        ]
        if latent_down == 16:
            stem.append(conv_block(128, 128, stride=2))  # -> 1/16
        self.enc_stem = nn.Sequential(*stem)
        self.enc_fuse = nn.Sequential(
            conv_block(128 + NUM_STATIC, 128, stride=1),
            nn.Conv2d(128, latent_c, kernel_size=1),
        )

        cond_c = STATIC_TERRAIN_C if terrain_cond else 0
        self.dec_in = conv_block(latent_c + cond_c, 128, stride=1)
        if upsample_decoder:
            chans = [128, 128, 96, 64, 32] if latent_down == 16 else [128, 96, 64, 32]
            self.dec_up = nn.ModuleList(
                UpsampleBlock(chans[i], chans[i + 1], extra_c=cond_c)
                for i in range(len(chans) - 1)
            )
            self.dec_refine = conv_block(32 + cond_c, 32, stride=1)
            self.dec_out = nn.Conv2d(32, MAX_SLOTS, kernel_size=1)
        else:
            self.dec_tiles = nn.Sequential(
                deconv_block(128, 128),
                deconv_block(128, 96),
                deconv_block(96, 64),
                deconv_block(64, 32),
                nn.Conv2d(32, MAX_SLOTS, kernel_size=1),
            )
        # Static structure occupancy logits at latent resolution.
        self.dec_units = nn.Conv2d(128, NUM_STATIC, kernel_size=1)

    def encode(
        self,
        owners: torch.Tensor,  # (B, H, W) int64
        terrain: torch.Tensor,  # (B, 3, H, W)
        static_planes: torch.Tensor,  # (B, NUM_STATIC, H/down, W/down)
    ) -> torch.Tensor:
        emb = self.owner_emb(owners).permute(0, 3, 1, 2)
        g = self.enc_stem(torch.cat([emb, terrain], dim=1))
        return self.enc_fuse(torch.cat([g, static_planes], dim=1))

    def decode(
        self,
        z_grid: torch.Tensor,
        terrain: torch.Tensor | None = None,  # (B, >=2, H, W); only [:, :2] used
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.terrain_cond:
            h = self.dec_in(z_grid)
            if self.upsample_decoder:
                x = h
                for up in self.dec_up:
                    x = up(x)
                return self.dec_out(self.dec_refine(x)), self.dec_units(h)
            return self.dec_tiles(h), self.dec_units(h)

        if terrain is None:
            raise ValueError("terrain_cond model needs terrain in decode()")
        # Static side-information pyramid: full res, 1/2, 1/4, ... latent res.
        static_t = terrain[:, :STATIC_TERRAIN_C]
        pyramid = {1: static_t}
        down = 2
        while down <= self.latent_down:
            pyramid[down] = F.avg_pool2d(static_t, kernel_size=down)
            down *= 2

        h = self.dec_in(torch.cat([z_grid, pyramid[self.latent_down]], dim=1))
        x = h
        scale = self.latent_down
        for up in self.dec_up:
            scale //= 2
            x = up(x, pyramid[scale])
        x = self.dec_refine(torch.cat([x, pyramid[1]], dim=1))
        return self.dec_out(x), self.dec_units(h)

    def forward(self, owners, terrain, static_planes):
        z_grid = self.encode(owners, terrain, static_planes)
        tile_logits, unit_logits = self.decode(
            z_grid, terrain if self.terrain_cond else None
        )
        return tile_logits, unit_logits, z_grid
