"""ONNX-export Policy: trunk + heads only (matches oftrain feedforward schema)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from webbot_export.consts import (
    BUILD_TYPES,
    C_GRID,
    C_GRID_FINE,
    N_ACTIONS,
    N_LOCAL,
    N_SCALARS,
    NUKE_TYPES,
    P_FEAT,
)

MASKED_NEG = -1e9


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.conv1(x))
        return F.silu(x + self.conv2(h))


class Policy(nn.Module):
    """~6.4M-param feedforward policy used for webbot ONNX export."""

    def __init__(
        self,
        hidden: int = 512,
        gc: int = 256,
        pc: int = 128,
        blocks: int = 4,
        lc: int = 64,
    ):
        super().__init__()
        self.grid_coarse_net = nn.Sequential(
            nn.Conv2d(C_GRID, gc, 3, padding=1),
            nn.SiLU(),
            *[ResBlock(gc) for _ in range(blocks)],
        )
        self.grid_fine_net = nn.Sequential(
            nn.Conv2d(C_GRID_FINE, gc, 3, padding=1),
            nn.SiLU(),
            *[ResBlock(gc) for _ in range(blocks)],
        )
        self.local_net = nn.Sequential(
            nn.Conv2d(N_LOCAL, 32, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, lc, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.player_in = nn.Linear(P_FEAT, pc)
        self.player_tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=pc,
                nhead=4,
                dim_feedforward=2 * pc,
                batch_first=True,
                dropout=0.0,
            ),
            num_layers=2,
        )
        self.trunk = nn.Sequential(
            nn.Linear(2 * gc + pc + lc + N_SCALARS, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head_action = nn.Linear(hidden, N_ACTIONS)
        self.head_player_q = nn.Linear(hidden, pc)
        self.head_tile_coarse = nn.Sequential(
            nn.Conv2d(gc + hidden, 256, 1), nn.SiLU(), nn.Conv2d(256, 1, 1)
        )
        self.head_tile_fine = nn.Sequential(
            nn.Conv2d(gc + hidden, 256, 1), nn.SiLU(), nn.Conv2d(256, 1, 1)
        )
        self.head_build = nn.Linear(hidden, len(BUILD_TYPES))
        self.head_nuke = nn.Linear(hidden, len(NUKE_TYPES))
        self.head_quantity = nn.Linear(hidden, 2)
        self.head_value = nn.Linear(hidden, 1)

    def _ensure_foveated(self, o: dict) -> dict:
        if "grid_fine" in o:
            return o
        fine_cov = o["grid_valid"]
        o = dict(o)
        if o["grid"].shape[1] == C_GRID_FINE:
            o["grid_fine"] = o["grid"]
            base_grid = o["grid"][:, :C_GRID]
        else:
            o["grid_fine"] = torch.cat([o["grid"], fine_cov.unsqueeze(1)], dim=1)
            base_grid = o["grid"]
        o["grid_fine_valid"] = fine_cov
        o["fine_coverage"] = fine_cov
        o["fine_origin"] = torch.zeros(
            o["grid"].shape[0], 2, dtype=torch.long, device=o["grid"].device
        )
        o["legal_tile_fine"] = o["legal_tile"]
        o["grid_coarse"] = F.avg_pool2d(
            base_grid, 2, stride=2, ceil_mode=True, count_include_pad=False
        )
        o["grid_coarse_valid"] = F.max_pool2d(
            o["grid_valid"].unsqueeze(1), 2, stride=2, ceil_mode=True
        ).squeeze(1)
        o["legal_tile_coarse"] = F.max_pool2d(
            o["legal_tile"].unsqueeze(1), 2, stride=2, ceil_mode=True
        ).squeeze(1)
        o["coarse_has_land"] = torch.ones_like(o["grid_coarse_valid"])
        o["coarse_has_water"] = torch.ones_like(o["grid_coarse_valid"])
        return o

    def trunk_forward(self, o: dict) -> tuple[torch.Tensor, dict, torch.Tensor]:
        o = self._ensure_foveated(o)
        gc_map = self.grid_coarse_net(o["grid_coarse"])
        gc_valid = o["grid_coarse_valid"].unsqueeze(1)
        gc_map = gc_map * gc_valid
        gc_pool = gc_map.sum(dim=(2, 3)) / gc_valid.sum(dim=(2, 3)).clamp(min=1)

        gf_map = self.grid_fine_net(o["grid_fine"])
        gf_valid = o["grid_fine_valid"].unsqueeze(1)
        gf_map = gf_map * gf_valid
        gf_pool = gf_map.sum(dim=(2, 3)) / gf_valid.sum(dim=(2, 3)).clamp(min=1)

        p = self.player_tf(
            self.player_in(o["players"]),
            src_key_padding_mask=o["pmask"] < 0.5,
        )
        m = o["pmask"].unsqueeze(-1)
        p_pool = (p * m).sum(1) / m.sum(1).clamp(min=1)

        l_pool = self.local_net(o["local"])
        h = self.trunk(
            torch.cat([gc_pool, gf_pool, p_pool, l_pool, o["scalars"]], dim=-1)
        )
        return h, {"coarse": gc_map, "fine": gf_map}, p

    def heads(self, h: torch.Tensor, g: dict, p: torch.Tensor, o: dict) -> dict:
        o = self._ensure_foveated(o)
        act_logits = self.head_action(h) + (o["legal_actions"] - 1) * -MASKED_NEG
        q = self.head_player_q(h)
        player_logits = torch.einsum("bd,bsd->bs", q, p)
        gc_map, gf_map = g["coarse"], g["fine"]
        hc = h[:, :, None, None].expand(-1, -1, gc_map.shape[2], gc_map.shape[3])
        hf = h[:, :, None, None].expand(-1, -1, gf_map.shape[2], gf_map.shape[3])
        tile_coarse = self.head_tile_coarse(torch.cat([gc_map, hc], dim=1)).flatten(1)
        tile_fine = self.head_tile_fine(torch.cat([gf_map, hf], dim=1)).flatten(1)
        return {
            "action": act_logits,
            "player": player_logits,
            "tile_coarse": tile_coarse,
            "tile_fine": tile_fine,
            "tile": tile_fine,
            "build": self.head_build(h) + (o["legal_build"] - 1) * -MASKED_NEG,
            "nuke": self.head_nuke(h) + (o["legal_nuke"] - 1) * -MASKED_NEG,
            "quantity": self.head_quantity(h),
            "value": self.head_value(h).squeeze(-1),
        }

    def forward(self, o: dict) -> dict:
        h, g, p = self.trunk_forward(o)
        return self.heads(h, g, p, o)
