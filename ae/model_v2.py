"""Unified state autoencoder: tiles + units + per-player state -> joint latent.

Encoder:
  - CNN over owner-slot embeddings + terrain (as v1), with unit-presence
    planes injected at 1/16 resolution
  - per-slot player features -> MLP -> masked mean pool -> state vector
  - fusion: state vector broadcast into the spatial latent; vector latent
    pooled from the fused grid

Latent:
  - z_grid: (B, latent_c, H/16, W/16)
  - z_vec:  (B, latent_d)

Decoders (fixed-shape only, no set decoding):
  - tile owner slots: per-tile logits (border-weighted CE)
  - unit counts: per-class log1p counts at 1/16 res (MSE)
  - per-slot player scalars: slot-embedding-conditioned regression (masked MSE)
"""

import torch
import torch.nn as nn

MAX_SLOTS = 128
OWNER_EMB_DIM = 8
TERRAIN_CHANNELS = 3

# Unit classes tracked spatially (Shell/SAMMissile/MIRVWarhead/Train excluded:
# transient projectiles and rail rolling stock, not strategic state).
UNIT_CLASSES = [
    "City",
    "Port",
    "Defense Post",
    "Missile Silo",
    "SAM Launcher",
    "Factory",
    "Warship",
    "Transport",
    "Trade Ship",
    "Atom Bomb",
    "Hydrogen Bomb",
    "MIRV",
]
NUM_UNIT_CLASSES = len(UNIT_CLASSES)

# Per-slot player features, all reconstructed:
# [alive, troops_log, gold_log, tiles_frac, traitor, disconnected,
#  n_allies, n_targets, n_embargoes, n_pending_reqs,
#  attack_troops_out_log, attack_troops_in_log]
PLAYER_FEAT_DIM = 12


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


class UnifiedStateAE(nn.Module):
    def __init__(self, latent_c: int = 64, latent_d: int = 128):
        super().__init__()
        self.latent_c = latent_c
        self.latent_d = latent_d

        self.owner_emb = nn.Embedding(MAX_SLOTS, OWNER_EMB_DIM)
        self.slot_emb = nn.Embedding(MAX_SLOTS, 16)

        c_in = OWNER_EMB_DIM + TERRAIN_CHANNELS
        self.enc_stem = nn.Sequential(
            conv_block(c_in, 32, stride=1),
            conv_block(32, 64, stride=2),
            conv_block(64, 96, stride=2),
            conv_block(96, 128, stride=2),
            conv_block(128, 128, stride=2),  # -> 1/16
        )

        # Player token encoder: (slot_emb ++ feats) -> token, masked mean pool.
        self.player_enc = nn.Sequential(
            nn.Linear(16 + PLAYER_FEAT_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )

        # Fuse grid features + unit planes + broadcast player-state vector.
        self.enc_fuse = nn.Sequential(
            conv_block(128 + NUM_UNIT_CLASSES + 64, 128, stride=1),
            nn.Conv2d(128, latent_c, kernel_size=1),
        )
        self.vec_head = nn.Sequential(
            nn.Linear(latent_c + 64, latent_d),
            nn.SiLU(),
            nn.Linear(latent_d, latent_d),
        )

        # Decoders.
        self.dec_grid_in = nn.Sequential(
            conv_block(latent_c + latent_d, 128, stride=1),
        )
        self.dec_tiles = nn.Sequential(
            deconv_block(128, 128),
            deconv_block(128, 96),
            deconv_block(96, 64),
            deconv_block(64, 32),
            nn.Conv2d(32, MAX_SLOTS, kernel_size=1),
        )
        self.dec_units = nn.Conv2d(128, NUM_UNIT_CLASSES, kernel_size=1)
        self.dec_players = nn.Sequential(
            nn.Linear(latent_d + 16, 128),
            nn.SiLU(),
            nn.Linear(128, PLAYER_FEAT_DIM),
        )

    def encode(
        self,
        owners: torch.Tensor,  # (B, H, W) int64
        terrain: torch.Tensor,  # (B, 3, H, W)
        unit_planes: torch.Tensor,  # (B, U, H/16, W/16) log1p counts
        player_feats: torch.Tensor,  # (B, S, F)
        player_mask: torch.Tensor,  # (B, S) 1 = slot exists
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.owner_emb(owners).permute(0, 3, 1, 2)
        g = self.enc_stem(torch.cat([emb, terrain], dim=1))  # (B,128,h,w)

        B, S, _ = player_feats.shape
        slot_ids = torch.arange(S, device=player_feats.device).expand(B, S)
        tokens = self.player_enc(
            torch.cat([self.slot_emb(slot_ids), player_feats], dim=-1)
        )  # (B, S, 64)
        m = player_mask.unsqueeze(-1)
        pooled = (tokens * m).sum(1) / m.sum(1).clamp(min=1)  # (B, 64)

        pooled_map = pooled[:, :, None, None].expand(-1, -1, g.shape[2], g.shape[3])
        z_grid = self.enc_fuse(torch.cat([g, unit_planes, pooled_map], dim=1))
        z_vec = self.vec_head(
            torch.cat([z_grid.mean(dim=(2, 3)), pooled], dim=-1)
        )
        return z_grid, z_vec, tokens

    def decode(
        self, z_grid: torch.Tensor, z_vec: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vec_map = z_vec[:, :, None, None].expand(
            -1, -1, z_grid.shape[2], z_grid.shape[3]
        )
        h = self.dec_grid_in(torch.cat([z_grid, vec_map], dim=1))
        tile_logits = self.dec_tiles(h)
        unit_pred = self.dec_units(h)

        B = z_vec.shape[0]
        slot_ids = torch.arange(MAX_SLOTS, device=z_vec.device).expand(B, -1)
        se = self.slot_emb(slot_ids)  # (B, S, 16)
        zv = z_vec.unsqueeze(1).expand(-1, MAX_SLOTS, -1)
        player_pred = self.dec_players(torch.cat([zv, se], dim=-1))
        return tile_logits, unit_pred, player_pred

    def forward(self, owners, terrain, unit_planes, player_feats, player_mask):
        z_grid, z_vec, _ = self.encode(
            owners, terrain, unit_planes, player_feats, player_mask
        )
        return self.decode(z_grid, z_vec), (z_grid, z_vec)
