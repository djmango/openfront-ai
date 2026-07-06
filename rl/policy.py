"""Policy: conv trunk over the observation grid + factorized action heads.

Heads (DESIGN.md): action type -> [player target | tile region | build type |
nuke type | quantity], each masked and only sampled/trained when the chosen
action type needs them. Tile arguments are region pointers over the latent
grid; the bridge/engine snaps to legal tiles.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ae.model_v3 import MAX_SLOTS
from rl.obs import ACTIONS, BUILD_TYPES, C_GRID, N_ACTIONS, N_SCALARS, NUKE_TYPES, P_FEAT

N_QUANTITY = 5  # {5, 10, 25, 50, 100}% of troops/gold
QUANTITY_FRACS = [0.05, 0.10, 0.25, 0.50, 1.00]

NEEDS_PLAYER = {
    "attack",
    "alliance_request",
    "alliance_reject",
    "break_alliance",
    "donate_gold",
    "donate_troops",
    "embargo",
}
NEEDS_TILE = {"boat", "build", "launch_nuke"}
NEEDS_QUANTITY = {"attack", "expand", "boat", "donate_gold", "donate_troops"}

MASKED_NEG = -1e9


class Policy(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.grid_net = nn.Sequential(
            nn.Conv2d(C_GRID, 128, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.SiLU(),
        )
        self.player_net = nn.Sequential(
            nn.Linear(P_FEAT, 64), nn.SiLU(), nn.Linear(64, 64)
        )
        self.trunk = nn.Sequential(
            nn.Linear(128 + 64 + N_SCALARS, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head_action = nn.Linear(hidden, N_ACTIONS)
        self.head_player_q = nn.Linear(hidden, 64)  # query against player embs
        self.head_tile = nn.Conv2d(128 + hidden, 1, 1)
        self.head_build = nn.Linear(hidden, len(BUILD_TYPES))
        self.head_nuke = nn.Linear(hidden, len(NUKE_TYPES))
        self.head_quantity = nn.Linear(hidden, N_QUANTITY)
        self.head_value = nn.Linear(hidden, 1)

    def trunk_forward(self, o: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g = self.grid_net(o["grid"])  # (B, 128, gh, gw)
        g_pool = g.mean(dim=(2, 3))
        p = self.player_net(o["players"])  # (B, S, 64)
        m = o["pmask"].unsqueeze(-1)
        p_pool = (p * m).sum(1) / m.sum(1).clamp(min=1)
        h = self.trunk(torch.cat([g_pool, p_pool, o["scalars"]], dim=-1))
        return h, g, p

    def heads(self, h: torch.Tensor, g: torch.Tensor, p: torch.Tensor, o: dict) -> dict:
        B = h.shape[0]
        act_logits = self.head_action(h) + (o["legal_actions"] - 1) * -MASKED_NEG

        q = self.head_player_q(h)  # (B, 64)
        player_logits = torch.einsum("bd,bsd->bs", q, p)  # (B, S)

        hm = h[:, :, None, None].expand(-1, -1, g.shape[2], g.shape[3])
        tile_logits = self.head_tile(torch.cat([g, hm], dim=1)).flatten(1)  # (B, gh*gw)

        return {
            "action": act_logits,
            "player": player_logits,
            "tile": tile_logits,
            "build": self.head_build(h) + (o["legal_build"] - 1) * -MASKED_NEG,
            "nuke": self.head_nuke(h) + (o["legal_nuke"] - 1) * -MASKED_NEG,
            "quantity": self.head_quantity(h),
            "value": self.head_value(h).squeeze(-1),
        }

    def forward(self, o: dict) -> dict:
        h, g, p = self.trunk_forward(o)
        return self.heads(h, g, p, o)

    @torch.no_grad()
    def act(self, o: dict) -> tuple[list[dict], np.ndarray, np.ndarray]:
        """Batched sampling. Returns (choices, logp (B,), value (B,))."""
        out = self.forward(o)
        B = out["action"].shape[0]
        dev = out["action"].device

        d_act = torch.distributions.Categorical(logits=out["action"])
        a = d_act.sample()
        logp = d_act.log_prob(a)

        # Player head mask depends on the sampled action.
        pmask = o["legal_ptarget"].gather(
            1, a[:, None, None].expand(-1, 1, MAX_SLOTS)
        ).squeeze(1)
        heads = {}
        for name, logits in [
            ("player", out["player"] + (pmask - 1) * -MASKED_NEG),
            ("tile", out["tile"]),
            ("build", out["build"]),
            ("nuke", out["nuke"]),
            ("quantity", out["quantity"]),
        ]:
            d = torch.distributions.Categorical(logits=logits)
            s = d.sample()
            heads[name] = (s, d.log_prob(s))

        needs_p = torch.tensor(
            [ACTIONS[i] in NEEDS_PLAYER for i in range(N_ACTIONS)], device=dev
        )[a]
        needs_t = torch.tensor(
            [ACTIONS[i] in NEEDS_TILE for i in range(N_ACTIONS)], device=dev
        )[a]
        needs_q = torch.tensor(
            [ACTIONS[i] in NEEDS_QUANTITY for i in range(N_ACTIONS)], device=dev
        )[a]
        is_build = a == ACTIONS.index("build")
        is_nuke = a == ACTIONS.index("launch_nuke")

        logp = logp + torch.where(needs_p, heads["player"][1], 0.0)
        logp = logp + torch.where(needs_t, heads["tile"][1], 0.0)
        logp = logp + torch.where(is_build, heads["build"][1], 0.0)
        logp = logp + torch.where(is_nuke, heads["nuke"][1], 0.0)
        logp = logp + torch.where(needs_q, heads["quantity"][1], 0.0)

        choices = []
        for b in range(B):
            c = {"action": int(a[b])}
            if needs_p[b]:
                c["player_slot"] = int(heads["player"][0][b])
            if needs_t[b]:
                c["tile_region"] = int(heads["tile"][0][b])
            if is_build[b]:
                c["build_type"] = int(heads["build"][0][b])
            if is_nuke[b]:
                c["nuke_type"] = int(heads["nuke"][0][b])
            if needs_q[b]:
                c["quantity"] = int(heads["quantity"][0][b])
            choices.append(c)
        return choices, logp.cpu().numpy(), out["value"].cpu().numpy()

    def evaluate(self, o: dict, choice: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched logprob/entropy/value for PPO updates.

        choice tensors: action (B,), player_slot, tile_region, build_type,
        nuke_type, quantity (B,) with -1 where unused.
        """
        out = self.forward(o)
        B = out["action"].shape[0]
        logp = torch.zeros(B, device=out["action"].device)
        ent = torch.zeros(B, device=out["action"].device)

        d = torch.distributions.Categorical(logits=out["action"])
        logp = logp + d.log_prob(choice["action"])
        ent = ent + d.entropy()

        def sub(head: str, key: str, mask: torch.Tensor | None = None):
            nonlocal logp, ent
            used = choice[key] >= 0
            if not used.any():
                return
            logits = out[head]
            if mask is not None:
                logits = logits + (mask - 1) * -MASKED_NEG
            dd = torch.distributions.Categorical(logits=logits[used])
            lp = dd.log_prob(choice[key][used])
            logp = logp.index_put((used.nonzero(as_tuple=True)[0],), lp, accumulate=True)
            ent = ent.index_put(
                (used.nonzero(as_tuple=True)[0],), dd.entropy(), accumulate=True
            )

        pmask = o["legal_ptarget"].gather(
            1, choice["action"].clamp(min=0)[:, None, None].expand(-1, 1, MAX_SLOTS)
        ).squeeze(1)
        sub("player", "player_slot", pmask)
        sub("tile", "tile_region")
        sub("build", "build_type")
        sub("nuke", "nuke_type")
        sub("quantity", "quantity")

        return logp, ent, out["value"]
