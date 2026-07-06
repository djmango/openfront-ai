"""Unified state autoencoder: tiles + units + per-player state -> joint latent.

Encoder:
  - CNN over owner-slot embeddings + terrain (as v1), with unit-presence
    planes injected at 1/16 resolution
  - per-slot player features -> MLP -> masked mean pool -> state vector
  - fusion: state vector broadcast into the spatial latent; vector latent
    pooled from the fused grid

Latent:
  - z_grid:  (B, latent_c, H/16, W/16)
  - z_vec:   (B, latent_d)
  - z_slots: (B, MAX_SLOTS, SLOT_LATENT_DIM) per-player latent (masked).
    Mean-pooling alone destroys slot identity: diplomacy precision plateaued
    at ~0.14 for any pos-weight because specific pairs were unrecoverable
    from the pooled vector. Per-slot latents fix that and give the RL agent
    per-player embeddings for free.

Decoders (fixed-shape only, no set decoding):
  - tile owner slots: per-tile logits (border-weighted CE)
  - unit presence: per-class occupancy logits at 1/16 res (BCE, rarity
    pos-weighted; MSE on counts collapses to all-zeros since ~99.9% of
    cells are empty)
  - per-slot player scalars: slot-embedding-conditioned regression (masked MSE)
"""

import torch
import torch.nn as nn

MAX_SLOTS = 128
OWNER_EMB_DIM = 8
TERRAIN_CHANNELS = 3
SLOT_LATENT_DIM = 32
DIPLO_IN_DIM = 32

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

# Positive-class weight multiplier per unit class (scaled by --unit-pos-weight
# at train time). Strategic rarity-weighted: a missed nuke-in-flight is
# catastrophic; a missed city is cheap to re-derive from territory. Order
# matches UNIT_CLASSES.
UNIT_CLASS_WEIGHTS = [
    1.0,  # City
    1.0,  # Port
    1.0,  # Defense Post
    4.0,  # Missile Silo
    4.0,  # SAM Launcher
    2.0,  # Factory
    4.0,  # Warship
    4.0,  # Transport (invasion inbound!)
    1.0,  # Trade Ship
    25.0,  # Atom Bomb
    25.0,  # Hydrogen Bomb
    25.0,  # MIRV
]

# Per-slot player features, all reconstructed:
# [alive, troops_log, gold_log, tiles_frac, traitor, disconnected,
#  n_allies, n_embargoes, n_pending_reqs,
#  attack_troops_out_log, attack_troops_in_log]
PLAYER_FEAT_DIM = 11

# Pairwise diplomacy channels reconstructed for every (slot i, slot j):
# 0 = allied (symmetric), 1 = i embargoes j
# (Targeting dropped: only human players issue target intents, so bot-only
# datasets have zero positives and the channel is untrainable.)
NUM_DIPLO = 2


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

        # Per-slot diplomacy rows (who am I allied with / targeting /
        # embargoing) projected small. Without this the encoder only saw
        # relation *counts*, making the pairwise graph unreconstructable.
        self.diplo_in = nn.Linear(NUM_DIPLO * MAX_SLOTS, DIPLO_IN_DIM)
        # Player token encoder: (slot_emb ++ feats ++ diplo) -> token,
        # masked mean pool.
        self.player_enc = nn.Sequential(
            nn.Linear(16 + PLAYER_FEAT_DIM + DIPLO_IN_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        # Per-slot latent head: keeps player identity through the bottleneck.
        self.slot_latent = nn.Linear(64, SLOT_LATENT_DIM)

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
        dec_player_in = latent_d + 16 + SLOT_LATENT_DIM
        self.dec_tiles = nn.Sequential(
            deconv_block(128, 128),
            deconv_block(128, 96),
            deconv_block(96, 64),
            deconv_block(64, 32),
            nn.Conv2d(32, MAX_SLOTS, kernel_size=1),
        )
        self.dec_units = nn.Conv2d(128, NUM_UNIT_CLASSES, kernel_size=1)
        self.dec_player_hidden = nn.Sequential(
            nn.Linear(dec_player_in, 128),
            nn.SiLU(),
        )
        self.dec_players = nn.Linear(128, PLAYER_FEAT_DIM)
        # Pairwise diplomacy decoder: bilinear scores over per-slot hidden
        # states -> (alliance, targeting, embargo) logits per (i, j) slot pair.
        # Forces the latent to carry the diplomacy *graph*, not just counts.
        self.diplo_bilinear = nn.Parameter(torch.randn(NUM_DIPLO, 64, 64) * 0.02)
        self.diplo_proj = nn.Linear(128, 64)

    def encode(
        self,
        owners: torch.Tensor,  # (B, H, W) int64
        terrain: torch.Tensor,  # (B, 3, H, W)
        unit_planes: torch.Tensor,  # (B, U, H/16, W/16) log1p counts
        player_feats: torch.Tensor,  # (B, S, F)
        player_mask: torch.Tensor,  # (B, S) 1 = slot exists
        diplo: torch.Tensor,  # (B, NUM_DIPLO, S, S) pairwise relations
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.owner_emb(owners).permute(0, 3, 1, 2)
        g = self.enc_stem(torch.cat([emb, terrain], dim=1))  # (B,128,h,w)

        B, S, _ = player_feats.shape
        slot_ids = torch.arange(S, device=player_feats.device).expand(B, S)
        # Row i of each relation channel = slot i's outgoing relations.
        diplo_rows = diplo.permute(0, 2, 1, 3).reshape(B, S, -1)
        d_in = self.diplo_in(diplo_rows)  # (B, S, DIPLO_IN_DIM)
        tokens = self.player_enc(
            torch.cat([self.slot_emb(slot_ids), player_feats, d_in], dim=-1)
        )  # (B, S, 64)
        m = player_mask.unsqueeze(-1)
        pooled = (tokens * m).sum(1) / m.sum(1).clamp(min=1)  # (B, 64)
        z_slots = self.slot_latent(tokens) * m  # (B, S, SLOT_LATENT_DIM)

        pooled_map = pooled[:, :, None, None].expand(-1, -1, g.shape[2], g.shape[3])
        z_grid = self.enc_fuse(torch.cat([g, unit_planes, pooled_map], dim=1))
        z_vec = self.vec_head(
            torch.cat([z_grid.mean(dim=(2, 3)), pooled], dim=-1)
        )
        return z_grid, z_vec, z_slots

    def decode(
        self, z_grid: torch.Tensor, z_vec: torch.Tensor, z_slots: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        hidden = self.dec_player_hidden(
            torch.cat([zv, se, z_slots], dim=-1)
        )  # (B, S, 128)
        player_pred = self.dec_players(hidden)

        # Pairwise diplomacy logits: (B, NUM_DIPLO, S, S)
        d = self.diplo_proj(hidden)  # (B, S, 64)
        diplo_logits = torch.einsum("bik,rkl,bjl->brij", d, self.diplo_bilinear, d)
        return tile_logits, unit_pred, player_pred, diplo_logits

    def forward(self, owners, terrain, unit_planes, player_feats, player_mask, diplo):
        z_grid, z_vec, z_slots = self.encode(
            owners, terrain, unit_planes, player_feats, player_mask, diplo
        )
        return self.decode(z_grid, z_vec, z_slots), (z_grid, z_vec, z_slots)
