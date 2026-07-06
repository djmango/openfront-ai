"""Fully-convolutional autoencoder over OpenFront tile states.

Input encoding per tile:
  - owner slot -> learned embedding (static slot per player for a whole game,
    slot 0 = unowned, slots 1..MAX_SLOTS-1 assigned by smallID order at spawn)
  - terrain scalars: land flag, normalized magnitude, fallout flag

The encoder downsamples by 16x into a spatial latent grid (LATENT_C channels
per 16x16 tile region), so any map size divisible by 16 works. The decoder
reconstructs per-tile owner-slot logits.
"""

import torch
import torch.nn as nn

MAX_SLOTS = 128  # owner classes: 0 = unowned, 1..127 player slots
OWNER_EMB_DIM = 8
TERRAIN_CHANNELS = 3  # land, magnitude, fallout


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


class TileAutoencoder(nn.Module):
    def __init__(self, latent_c: int = 64):
        super().__init__()
        self.owner_emb = nn.Embedding(MAX_SLOTS, OWNER_EMB_DIM)
        c_in = OWNER_EMB_DIM + TERRAIN_CHANNELS

        self.encoder = nn.Sequential(
            conv_block(c_in, 32, stride=1),
            conv_block(32, 64, stride=2),  # /2
            conv_block(64, 96, stride=2),  # /4
            conv_block(96, 128, stride=2),  # /8
            conv_block(128, 128, stride=2),  # /16
            nn.Conv2d(128, latent_c, kernel_size=1),
        )

        self.decoder = nn.Sequential(
            conv_block(latent_c, 128, stride=1),
            deconv_block(128, 128),  # /8
            deconv_block(128, 96),  # /4
            deconv_block(96, 64),  # /2
            deconv_block(64, 32),  # /1
            nn.Conv2d(32, MAX_SLOTS, kernel_size=1),
        )

    def encode(self, owners: torch.Tensor, terrain: torch.Tensor) -> torch.Tensor:
        """owners: (B, H, W) int64 slots; terrain: (B, 3, H, W) float."""
        emb = self.owner_emb(owners).permute(0, 3, 1, 2)  # (B, E, H, W)
        x = torch.cat([emb, terrain], dim=1)
        return self.encoder(x)

    def forward(
        self, owners: torch.Tensor, terrain: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(owners, terrain)
        logits = self.decoder(z)  # (B, MAX_SLOTS, H, W)
        return logits, z
