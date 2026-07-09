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
from rl.obs import (
    ACTIONS,
    BUILD_TYPES,
    C_GRID,
    C_GRID_FINE,
    N_ACTIONS,
    N_LOCAL,
    N_SCALARS,
    NUKE_TYPES,
    P_FEAT,
)

# v6: quantity is a scalar 0-1 fraction from a Beta head (2 params). The
# 1+softplus parameterization keeps alpha,beta >= 1, so the density is
# unimodal and bounded (no U-shaped spikes at 0/1).
QUANTITY_EPS = 1e-4  # clamp samples/labels away from the Beta support edges

NEEDS_PLAYER = {
    "attack",
    "alliance_request",
    "alliance_reject",
    "break_alliance",
    "donate_gold",
    "donate_troops",
    "embargo",
    "retreat",  # which attack to cancel (target's slot; 0 = expand)
    "embargo_stop",
    "target_player",
    "alliance_extension",
}
NEEDS_TILE = {
    "boat",
    "build",
    "launch_nuke",
    "spawn",
    "upgrade_structure",
    "move_warship",
    "cancel_boat",
    "delete_unit",
}
# Actions whose tile pick is refined to the fine /8 grid. Everything else
# in NEEDS_TILE (boat, launch_nuke, move_warship) stays at coarse /16
# granularity: the emitted region index is the coarse cell's top-left /8
# region, and IntentTranslator searches the whole 2x2 block (span=2).
# Keep those translate() call sites in sync when editing this set.
REFINE_TILE = {
    "spawn",
    "build",
    "upgrade_structure",
    "cancel_boat",
    "delete_unit",
}
NEEDS_QUANTITY = {"attack", "expand", "boat", "donate_gold", "donate_troops"}

MASKED_NEG = -1e9


def quantity_dist(params: torch.Tensor) -> torch.distributions.Beta:
    """(B, 2) raw head outputs -> Beta(alpha, beta), alpha/beta >= 1."""
    ab = 1.0 + F.softplus(params.float())
    return torch.distributions.Beta(ab[:, 0], ab[:, 1])


def _local_to_global(local: torch.Tensor, gw: int) -> torch.Tensor:
    """Batch-local flat tile index -> global (GW_MAX-stride) region index.

    Batches are padded only to their own max grid, so the flat index stride
    varies per batch; region indices stored in choices/buffers always use
    the fixed GW_MAX stride (what IntentTranslator decodes). Padding cells
    are masked to ~0 probability, so probabilities over real cells are
    identical whatever the padding - logprobs stay consistent between
    rollout and update even if the two batches padded differently."""
    from rl.curriculum import GW_MAX

    return (local // gw) * GW_MAX + (local % gw)


def _global_to_local(region: torch.Tensor, gw: int) -> torch.Tensor:
    from rl.curriculum import GW_MAX

    r = region.clamp(min=0)  # -1 sentinels are filtered by the caller's mask
    return (r // GW_MAX) * gw + (r % GW_MAX)


def _fine_local_to_global(
    local: torch.Tensor, gw: int, origin: torch.Tensor
) -> torch.Tensor:
    from rl.curriculum import GW_MAX

    gy = local // gw + origin[:, 0].long()
    gx = local % gw + origin[:, 1].long()
    return gy * GW_MAX + gx


def _coarse_local_to_global(local: torch.Tensor, gw: int) -> torch.Tensor:
    from rl.curriculum import GW_MAX

    cy, cx = local // gw, local % gw
    return (cy * 2) * GW_MAX + (cx * 2)


def _global_to_fine_local(
    region: torch.Tensor, gw: int, origin: torch.Tensor
) -> torch.Tensor:
    from rl.curriculum import GW_MAX

    r = region.clamp(min=0)
    gy = r // GW_MAX - origin[:, 0].long()
    gx = r % GW_MAX - origin[:, 1].long()
    return gy * gw + gx


def _global_to_coarse_local(region: torch.Tensor, gw: int) -> torch.Tensor:
    from rl.curriculum import GW_MAX

    r = region.clamp(min=0)
    gy, gx = r // GW_MAX, r % GW_MAX
    return (gy // 2) * gw + (gx // 2)


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.conv1(x))
        return F.silu(x + self.conv2(h))


class Policy(nn.Module):
    """~6.4M params (17x the v1 scaffold): 256-ch residual grid tower,
    2-layer transformer over player tokens, 512-wide trunk. Sized to soak
    the idle GPU while env simulation stays the bottleneck."""

    def __init__(
        self, hidden: int = 512, gc: int = 256, pc: int = 128, blocks: int = 4,
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
        # Exact-borders bypass: raw local owner crop (64x64 tiles around own
        # territory) through a small strided CNN, pooled into the trunk.
        self.local_net = nn.Sequential(
            nn.Conv2d(N_LOCAL, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, lc, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.player_in = nn.Linear(P_FEAT, pc)
        self.player_tf = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=pc, nhead=4, dim_feedforward=2 * pc,
                batch_first=True, dropout=0.0,
            ),
            num_layers=2,
        )
        self.trunk = nn.Sequential(
            nn.Linear(2 * gc + pc + lc + N_SCALARS, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head_action = nn.Linear(hidden, N_ACTIONS)
        self.head_player_q = nn.Linear(hidden, pc)  # query against player embs
        self.head_tile_coarse = nn.Sequential(
            nn.Conv2d(gc + hidden, 256, 1), nn.SiLU(), nn.Conv2d(256, 1, 1)
        )
        self.head_tile_fine = nn.Sequential(
            nn.Conv2d(gc + hidden, 256, 1), nn.SiLU(), nn.Conv2d(256, 1, 1)
        )
        self.head_build = nn.Linear(hidden, len(BUILD_TYPES))
        self.head_nuke = nn.Linear(hidden, len(NUKE_TYPES))
        self.head_quantity = nn.Linear(hidden, 2)  # Beta alpha/beta params
        self.head_value = nn.Linear(hidden, 1)

    def _ensure_foveated(self, o: dict) -> dict:
        """Legacy obs dicts become full-map fine + pooled coarse inputs."""
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
        # No land/water info in legacy obs: no content pruning.
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
        )  # (B, S, pc)
        m = o["pmask"].unsqueeze(-1)
        p_pool = (p * m).sum(1) / m.sum(1).clamp(min=1)

        l_pool = self.local_net(o["local"])
        h = self.trunk(torch.cat([gc_pool, gf_pool, p_pool, l_pool, o["scalars"]], dim=-1))
        return h, {"coarse": gc_map, "fine": gf_map}, p

    def heads(self, h: torch.Tensor, g: dict, p: torch.Tensor, o: dict) -> dict:
        o = self._ensure_foveated(o)
        act_logits = self.head_action(h) + (o["legal_actions"] - 1) * -MASKED_NEG

        q = self.head_player_q(h)  # (B, pc)
        player_logits = torch.einsum("bd,bsd->bs", q, p)  # (B, S)

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

    def _fine_to_coarse_mask(self, o: dict, cgh: int, cgw: int) -> torch.Tensor:
        """Which coarse cells contain at least one legal covered fine cell.

        Vectorized scatter: the per-sample nonzero() loop this replaces ran
        one GPU sync per sample inside every evaluate() minibatch (~24k
        syncs/update) and dominated the update phase."""
        B, fh, fw = o["legal_tile_fine"].shape
        dev = o["legal_tile_fine"].device
        gy = (
            torch.arange(fh, device=dev)[None, :, None]
            + o["fine_origin"][:, 0, None, None].long()
        ) // 2
        gx = (
            torch.arange(fw, device=dev)[None, None, :]
            + o["fine_origin"][:, 1, None, None].long()
        ) // 2
        # Padded fine rows/cols can map past the coarse grid; their
        # legal_tile_fine is 0, so clamping the index is harmless.
        idx = (gy * cgw + gx).clamp(max=cgh * cgw - 1)
        out = o["legal_tile_fine"].new_zeros(B, cgh * cgw)
        out.scatter_add_(1, idx.reshape(B, -1), o["legal_tile_fine"].reshape(B, -1))
        return (out.view(B, cgh, cgw) > 0).float() * o["grid_coarse_valid"]

    def _coarse_logits_for_action(
        self, out: dict, o: dict, action: torch.Tensor
    ) -> torch.Tensor:
        base = o["grid_coarse_valid"] * o["legal_tile_coarse"]
        # Content pruning for coarse-granularity targets: boats and nukes
        # target land, warship moves target water. Without this the coarse
        # head can point boats at pure-ocean cells, which translate to
        # nothing and read as noop-with-penalty - space the policy would
        # otherwise have to learn to avoid pick by pick.
        if "coarse_has_land" in o:
            needs_land = torch.tensor(
                [ACTIONS[i] in ("boat", "launch_nuke") for i in range(N_ACTIONS)],
                device=action.device,
            )[action]
            needs_water = action == ACTIONS.index("move_warship")
            base = base * torch.where(
                needs_land[:, None, None], o["coarse_has_land"], 1.0
            )
            base = base * torch.where(
                needs_water[:, None, None], o["coarse_has_water"], 1.0
            )
        refine_action = torch.tensor(
            [ACTIONS[i] in REFINE_TILE for i in range(N_ACTIONS)],
            device=action.device,
        )[action]
        fine_coarse = self._fine_to_coarse_mask(
            o, o["grid_coarse_valid"].shape[1], o["grid_coarse_valid"].shape[2]
        )
        # If a degenerate sample has no fine legal cells, fall back to the
        # coarse mask instead of constructing an all-masked categorical.
        has_fine = fine_coarse.flatten(1).sum(1) > 0
        use_fine = refine_action & has_fine
        mask = torch.where(use_fine[:, None, None], fine_coarse, base)
        return out["tile_coarse"] + (mask.flatten(1) - 1) * -MASKED_NEG

    def _fine_logits_for_coarse(
        self, out: dict, o: dict, coarse: torch.Tensor
    ) -> torch.Tensor:
        B, fh, fw = o["legal_tile_fine"].shape
        cgw = o["grid_coarse_valid"].shape[2]
        cy, cx = coarse // cgw, coarse % cgw
        yy = torch.arange(fh, device=coarse.device)[None, :, None]
        xx = torch.arange(fw, device=coarse.device)[None, None, :]
        gy = yy + o["fine_origin"][:, 0, None, None].long()
        gx = xx + o["fine_origin"][:, 1, None, None].long()
        mask = (
            (gy // 2 == cy[:, None, None])
            & (gx // 2 == cx[:, None, None])
        ).float()
        mask = mask * o["legal_tile_fine"] * o["grid_fine_valid"]
        fallback = o["legal_tile_fine"] * o["grid_fine_valid"]
        fallback = torch.where(
            fallback.flatten(1).sum(1)[:, None, None] > 0,
            fallback,
            o["grid_fine_valid"],
        )
        mask = torch.where(mask.flatten(1).sum(1)[:, None, None] > 0, mask, fallback)
        return out["tile_fine"] + (mask.flatten(1) - 1) * -MASKED_NEG

    @torch.no_grad()
    def act(
        self, o: dict, debug: bool = False, greedy: bool = False
    ) -> tuple[list[dict], np.ndarray, np.ndarray]:
        """Batched sampling. Returns (choices, logp (B,), value (B,)).

        greedy=True takes the argmax of every head (deployment-style eval).
        With debug=True each choice carries a "debug" dict (masked action
        probabilities + value estimate) for visualization; strip it before
        piping choices anywhere."""
        o = self._ensure_foveated(o)
        out = self.forward(o)
        B = out["action"].shape[0]
        dev = out["action"].device

        d_act = torch.distributions.Categorical(logits=out["action"])
        a = d_act.probs.argmax(-1) if greedy else d_act.sample()
        logp = d_act.log_prob(a)

        o = self._ensure_foveated(o)
        # Player head mask depends on the sampled action.
        pmask = o["legal_ptarget"].gather(
            1, a[:, None, None].expand(-1, 1, MAX_SLOTS)
        ).squeeze(1)
        heads = {}
        for name, logits in [
            ("player", out["player"] + (pmask - 1) * -MASKED_NEG),
            ("build", out["build"]),
            ("nuke", out["nuke"]),
        ]:
            d = torch.distributions.Categorical(logits=logits)
            s = d.probs.argmax(-1) if greedy else d.sample()
            heads[name] = (s, d.log_prob(s))

        # Scalar quantity: Beta sample (mean when greedy), clamped off the
        # support edges so log_prob stays finite.
        d_q = quantity_dist(out["quantity"])
        q = (d_q.mean if greedy else d_q.sample()).clamp(
            QUANTITY_EPS, 1.0 - QUANTITY_EPS
        )
        heads["quantity"] = (q, d_q.log_prob(q))

        needs_p = torch.tensor(
            [ACTIONS[i] in NEEDS_PLAYER for i in range(N_ACTIONS)], device=dev
        )[a]
        needs_t = torch.tensor(
            [ACTIONS[i] in NEEDS_TILE for i in range(N_ACTIONS)], device=dev
        )[a]
        refine_t = torch.tensor(
            [ACTIONS[i] in REFINE_TILE for i in range(N_ACTIONS)], device=dev
        )[a]
        needs_q = torch.tensor(
            [ACTIONS[i] in NEEDS_QUANTITY for i in range(N_ACTIONS)], device=dev
        )[a]
        is_build = a == ACTIONS.index("build")
        is_nuke = a == ACTIONS.index("launch_nuke")

        logp = logp + torch.where(needs_p, heads["player"][1], 0.0)

        coarse_logits = self._coarse_logits_for_action(out, o, a)
        d_coarse = torch.distributions.Categorical(logits=coarse_logits)
        coarse = d_coarse.probs.argmax(-1) if greedy else d_coarse.sample()
        fine_logits = self._fine_logits_for_coarse(out, o, coarse)
        d_fine = torch.distributions.Categorical(logits=fine_logits)
        fine = d_fine.probs.argmax(-1) if greedy else d_fine.sample()
        fine_lp = d_fine.log_prob(fine)
        fine_global = _fine_local_to_global(
            fine, o["grid_fine"].shape[3], o["fine_origin"]
        )
        # Logprob consistency with evaluate(): only the fine GLOBAL region
        # is stored, and evaluate() reconstructs the coarse factor as the
        # sampled fine cell's parent. Normally that IS the sampled coarse
        # cell (the fine mask is restricted to it), but when the fine mask
        # falls back (no legal fine cells anywhere) the sampled fine cell
        # can live in a different coarse cell - score THAT cell here so
        # rollout and update logprobs match.
        eff_coarse = torch.where(
            refine_t,
            _global_to_coarse_local(fine_global, o["grid_coarse"].shape[3]),
            coarse,
        )
        coarse_lp = d_coarse.log_prob(eff_coarse)
        tile_lp = torch.where(refine_t, coarse_lp + fine_lp, coarse_lp)
        tile_region = torch.where(
            refine_t,
            fine_global,
            _coarse_local_to_global(coarse, o["grid_coarse"].shape[3]),
        )

        logp = logp + torch.where(needs_t, tile_lp, 0.0)
        logp = logp + torch.where(is_build, heads["build"][1], 0.0)
        logp = logp + torch.where(is_nuke, heads["nuke"][1], 0.0)
        logp = logp + torch.where(needs_q, heads["quantity"][1], 0.0)

        choices = []
        for b in range(B):
            c = {"action": int(a[b])}
            if needs_p[b]:
                c["player_slot"] = int(heads["player"][0][b])
            if needs_t[b]:
                c["tile_region"] = int(tile_region[b])
            if is_build[b]:
                c["build_type"] = int(heads["build"][0][b])
            if is_nuke[b]:
                c["nuke_type"] = int(heads["nuke"][0][b])
            if needs_q[b]:
                c["quantity_frac"] = float(heads["quantity"][0][b])
            if debug:
                c["debug"] = {
                    "action_probs": d_act.probs[b].float().cpu().numpy(),
                    "value": float(out["value"][b]),
                }
            choices.append(c)
        return choices, logp.float().cpu().numpy(), out["value"].float().cpu().numpy()

    def evaluate(
        self, o: dict, choice: dict
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched logprob/entropy/value for PPO updates.

        choice tensors: action (B,), player_slot, tile_region, build_type,
        nuke_type (B,) long with -1 where unused; quantity_frac (B,) float
        with -1.0 where unused.

        Returns (logp, ent, ent_q, value): ent is the summed DISCRETE head
        entropy (nats); ent_q is the Beta head's differential entropy (can
        be negative), kept separate so it gets its own coefficient and
        stays out of the discrete entropy floor.
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
        t_used = choice["tile_region"] >= 0
        if t_used.any():
            coarse_target = _global_to_coarse_local(
                choice["tile_region"], o["grid_coarse"].shape[3]
            )
            coarse_logits = self._coarse_logits_for_action(
                out, o, choice["action"].clamp(min=0)
            )
            dd = torch.distributions.Categorical(logits=coarse_logits[t_used])
            lp = dd.log_prob(coarse_target[t_used])
            idx = t_used.nonzero(as_tuple=True)[0]
            logp = logp.index_put((idx,), lp, accumulate=True)
            ent = ent.index_put((idx,), dd.entropy(), accumulate=True)

            refine = torch.tensor(
                [ACTIONS[i] in REFINE_TILE for i in range(N_ACTIONS)],
                device=choice["action"].device,
            )[choice["action"].clamp(min=0)] & t_used
            if refine.any():
                fine_target = _global_to_fine_local(
                    choice["tile_region"], o["grid_fine"].shape[3], o["fine_origin"]
                )
                fine_logits = self._fine_logits_for_coarse(out, o, coarse_target)
                dd = torch.distributions.Categorical(logits=fine_logits[refine])
                idx = refine.nonzero(as_tuple=True)[0]
                logp = logp.index_put(
                    (idx,), dd.log_prob(fine_target[refine]), accumulate=True
                )
                ent = ent.index_put((idx,), dd.entropy(), accumulate=True)
        sub("build", "build_type")
        sub("nuke", "nuke_type")

        ent_q = torch.zeros(B, device=logp.device)
        q_used = choice["quantity_frac"] >= 0
        if q_used.any():
            d_q = quantity_dist(out["quantity"][q_used])
            target = choice["quantity_frac"][q_used].float().clamp(
                QUANTITY_EPS, 1.0 - QUANTITY_EPS
            )
            idx = (q_used.nonzero(as_tuple=True)[0],)
            logp = logp.index_put(idx, d_q.log_prob(target), accumulate=True)
            ent_q = ent_q.index_put(idx, d_q.entropy(), accumulate=True)

        return logp, ent, ent_q, out["value"]
