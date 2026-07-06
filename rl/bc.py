"""Behavior cloning from human games onto the PPO policy architecture.

Two variants (DESIGN: imitation as warm start for RL):

  feedforward (default)  the exact rl.policy.Policy trunk+heads, plus an
                         additive placement-conditioning embedding — trained
                         on everyone, conditioned on "winner" at deployment
  --seq K                same, plus a small causal transformer over the last
                         K decision steps' trunk embeddings before the heads

Loss is masked cross-entropy per head, sub-heads supervised only where the
human action uses them (mirrors Policy.evaluate's masking).

  PYTHONPATH=. python -m rl.bc --data data-human --name bc_v1
  PYTHONPATH=. python -m rl.bc --data data-human --name bc_seq_v1 --seq 8
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.bc_data import BCSampler, N_PLACEMENT_BUCKETS
from rl.obs import ACTIONS, N_ACTIONS, encode_grids, load_ae
from rl.obs import collate as obs_collate
from rl.policy import MASKED_NEG, Policy, _global_to_local

CHOICE_KEYS = ["action", "player_slot", "tile_region", "build_type", "nuke_type", "quantity"]
OBS_KEYS = [
    "grid", "grid_valid", "players", "pmask", "scalars",
    "legal_actions", "legal_ptarget", "legal_build", "legal_nuke",
]


class BCPolicy(Policy):
    """Policy + placement conditioning (+ optional temporal transformer).

    Extras live in separate modules so the core Policy weights load straight
    into PPO with load_state_dict(strict=False)."""

    def __init__(self, cond_dim: int = 16, seq: int = 0, **kw):
        super().__init__(**kw)
        hidden = self.head_action.in_features
        self.seq = seq
        self.cond_emb = nn.Embedding(N_PLACEMENT_BUCKETS, cond_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)
        if seq:
            self.temporal = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=hidden, nhead=8, dim_feedforward=2 * hidden,
                    batch_first=True, dropout=0.0,
                ),
                num_layers=2,
            )

    def forward_bc(self, o: dict, cond: torch.Tensor) -> dict:
        """o tensors are (B, ...) for feedforward, (B*K, ...) flattened
        step-major for --seq (the label belongs to the last step)."""
        h, g, p = self.trunk_forward(o)
        if self.seq:
            BK = h.shape[0]
            B, K = BK // self.seq, self.seq
            hs = h.view(B, K, -1)
            causal = torch.triu(
                torch.ones(K, K, dtype=torch.bool, device=h.device), diagonal=1
            )
            hs = self.temporal(hs, mask=causal)
            h = hs[:, -1]
            last = torch.arange(K - 1, BK, K, device=h.device)
            g, p = g[last], p[last]
            o = {
                k: (v[last] if torch.is_tensor(v) else v) for k, v in o.items()
            }
        h = h + self.cond_proj(self.cond_emb(cond))
        return self.heads(h, g, p, o)


def collate(raws: list[dict], device: str) -> tuple[dict, dict, torch.Tensor]:
    """Stack encoded raws into policy input, choice targets, cond buckets.
    Grids are padded to this batch's max (rl.obs.collate); tile_region
    labels use the global GW_MAX stride, so convert them to this batch's
    local flat index to match the tile head's flattening."""
    o = {
        k: torch.from_numpy(v).to(device)
        for k, v in obs_collate(raws, OBS_KEYS).items()
    }
    choice = {
        k: torch.tensor([r["choice"][k] for r in raws], dtype=torch.long, device=device)
        for k in CHOICE_KEYS
    }
    tr = choice["tile_region"]
    choice["tile_region"] = torch.where(
        tr >= 0, _global_to_local(tr, o["grid"].shape[3]), tr
    )
    cond = torch.tensor([r["cond"] for r in raws], dtype=torch.long, device=device)
    return o, choice, cond


def bc_loss(out: dict, o: dict, choice: dict) -> tuple[torch.Tensor, dict]:
    """Masked CE per head; sub-heads only where the action uses them."""
    losses = {"action": F.cross_entropy(out["action"], choice["action"])}

    pmask = o["legal_ptarget"].gather(
        1, choice["action"][:, None, None].expand(-1, 1, o["legal_ptarget"].shape[-1])
    ).squeeze(1)
    player_logits = out["player"] + (pmask - 1) * -MASKED_NEG

    for head, logits in [
        ("player_slot", player_logits),
        ("tile_region", out["tile"]),
        ("build_type", out["build"]),
        ("nuke_type", out["nuke"]),
        ("quantity", out["quantity"]),
    ]:
        if (choice[head] >= 0).any():
            losses[head] = F.cross_entropy(logits, choice[head], ignore_index=-1)
    total = sum(losses.values())
    return total, {k: float(v.detach()) for k, v in losses.items()}


@torch.no_grad()
def head_accuracy(out: dict, o: dict, choice: dict) -> dict:
    accs = {}
    acted = choice["action"] != ACTIONS.index("noop")
    pred = out["action"].argmax(-1)
    accs["action"] = float((pred == choice["action"]).float().mean())
    if acted.any():
        accs["action_no_noop"] = float(
            (pred[acted] == choice["action"][acted]).float().mean()
        )
    for head, key in [("player", "player_slot"), ("tile", "tile_region"),
                      ("build", "build_type"), ("nuke", "nuke_type"),
                      ("quantity", "quantity")]:
        used = choice[key] >= 0
        if used.any():
            accs[key] = float(
                (out[head][used].argmax(-1) == choice[key][used]).float().mean()
            )
    return accs


def encode_batch(ae, raws: list[dict], device: str) -> list[dict]:
    enc = encode_grids(ae, raws, device)
    for e, r in zip(enc, raws):
        e["choice"], e["cond"] = r["choice"], r["cond"]
    return enc


def seq_batch(sampler: BCSampler, batch: int, seq: int) -> list[dict]:
    """Draw `batch` windows of `seq` consecutive steps for single players;
    flattened step-major, label on the last step of each window."""
    out: list[dict] = []
    while len(out) < batch * seq:
        out.extend(sampler.sample_window(seq))
    return out[: batch * seq]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data-human"])
    ap.add_argument("--ae", default="runs/ae_v3/ae_v3.pt")
    ap.add_argument("--name", default="bc_v1")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seq", type=int, default=0)
    ap.add_argument("--noop-frac", type=float, default=0.15)
    ap.add_argument("--holdout-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    roots = [Path(d) for d in args.data]
    sampler = BCSampler(
        roots, holdout_every=args.holdout_every, holdout=False,
        noop_frac=args.noop_frac,
    )
    try:
        eval_sampler = BCSampler(
            roots, holdout_every=args.holdout_every, holdout=True,
            noop_frac=args.noop_frac, seed=1,
        )
    except FileNotFoundError:
        eval_sampler = None  # tiny smoke datasets may have no holdout games
    print(f"device: {device}  train games: {len(sampler.games)}  "
          f"holdout games: {len(eval_sampler.games) if eval_sampler else 0}")

    ae = load_ae(args.ae, device)
    model = BCPolicy(seq=args.seq).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    start_step = 0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["opt_state_dict"])
        sched.load_state_dict(ck["sched_state_dict"])
        start_step = ck["step"]
        print(f"resumed {args.resume} at step {start_step}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.2f}M params  seq={args.seq}")

    run_dir = Path("runs/bc") / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.jsonl"

    if args.seq:
        draw = lambda s: seq_batch(s, args.batch, args.seq)  # noqa: E731
    else:
        draw = lambda s: s.sample_batch(args.batch)  # noqa: E731
    # Threaded prefetch: gzip/json decode releases the GIL.
    pool = ThreadPoolExecutor(max_workers=args.workers)
    pending = [pool.submit(draw, sampler) for _ in range(4)]

    t0 = time.time()
    for step in range(start_step + 1, args.steps + 1):
        raws = pending.pop(0).result()
        pending.append(pool.submit(draw, sampler))
        enc = encode_batch(ae, raws, device)
        if args.seq:
            # collate keeps step-major flattening; choice/cond of last steps.
            o, choice, cond = collate(enc, device)
            last = torch.arange(args.seq - 1, len(enc), args.seq, device=device)
            choice = {k: v[last] for k, v in choice.items()}
            cond = cond[last]
        else:
            o, choice, cond = collate(enc, device)

        out = model.forward_bc(o, cond)
        loss, parts = bc_loss(out, o if not args.seq else _last_o(o, args.seq, device), choice)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 50 == 0:
            rate = 50 * args.batch / (time.time() - t0)
            t0 = time.time()
            accs = head_accuracy(out, _last_o(o, args.seq, device) if args.seq else o, choice)
            msg = (f"step {step:6d}  loss {float(loss):.4f}  "
                   f"act {accs.get('action', 0):.3f}  "
                   f"act! {accs.get('action_no_noop', 0):.3f}  "
                   f"tile {accs.get('tile_region', float('nan')):.3f}  "
                   f"{rate:.1f} ex/s")
            print(msg, flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps({"step": step, "loss": float(loss),
                                    **parts, **accs}) + "\n")

        if step % args.eval_every == 0 and eval_sampler is not None:
            model.eval()
            agg: dict[str, list[float]] = {}
            with torch.no_grad():
                for _ in range(8):
                    eraws = draw(eval_sampler)
                    eenc = encode_batch(ae, eraws, device)
                    eo, ech, econd = collate(eenc, device)
                    if args.seq:
                        last = torch.arange(args.seq - 1, len(eenc), args.seq, device=device)
                        ech = {k: v[last] for k, v in ech.items()}
                        econd = econd[last]
                    eout = model.forward_bc(eo, econd)
                    eo_h = _last_o(eo, args.seq, device) if args.seq else eo
                    for k, v in head_accuracy(eout, eo_h, ech).items():
                        agg.setdefault(k, []).append(v)
            evals = {k: float(np.mean(v)) for k, v in agg.items()}
            print(f"  [eval] {evals}", flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps({"step": step, "eval": evals}) + "\n")
            model.train()

        if step % args.save_every == 0 or step == args.steps:
            tmp = run_dir / "bc.pt.tmp"
            torch.save({
                "model_state_dict": model.state_dict(),
                "opt_state_dict": opt.state_dict(),
                "sched_state_dict": sched.state_dict(),
                "step": step,
                "args": vars(args),
            }, tmp)
            tmp.rename(run_dir / "bc.pt")  # atomic: no torn ckpt on kill
            # Off-pod durability, same pattern as rl/ppo.py.
            if os.environ.get("HF_TOKEN") and step % (args.save_every * 5) == 0:
                import threading

                def _hf_push(path=run_dir / "bc.pt", name=args.name):
                    try:
                        from huggingface_hub import HfApi

                        api = HfApi()
                        api.create_repo("djmango/openfront-rl", exist_ok=True)
                        api.upload_file(
                            path_or_fileobj=str(path),
                            path_in_repo=f"{name}/bc.pt",
                            repo_id="djmango/openfront-rl",
                        )
                    except Exception as e:  # noqa: BLE001
                        print(f"hf sync failed: {e}", flush=True)

                threading.Thread(target=_hf_push, daemon=True).start()


def _last_o(o: dict, seq: int, device: str) -> dict:
    """Slice per-step tensors down to each window's last step (already done
    inside forward_bc for the model; the loss masks need the same view)."""
    if not seq:
        return o
    B = o["grid"].shape[0] // seq
    last = torch.arange(seq - 1, B * seq, seq, device=device)
    return {k: (v[last] if torch.is_tensor(v) else v) for k, v in o.items()}


if __name__ == "__main__":
    main()
