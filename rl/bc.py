"""Behavior cloning from human games onto the PPO policy architecture.

Two variants (DESIGN: imitation as warm start for RL):

  feedforward (default)  the exact rl.policy.Policy trunk+heads, plus an
                         additive placement-conditioning embedding - trained
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
import contextlib
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
from rl.obs import ACTIONS, N_ACTIONS, ZCache, encode_grids, load_ae
from rl.obs import collate as obs_collate
from rl.policy import MASKED_NEG, Policy, _global_to_local

CHOICE_KEYS = ["action", "player_slot", "tile_region", "build_type", "nuke_type", "quantity"]
OBS_KEYS = [
    "grid", "grid_valid", "legal_tile", "local", "players", "pmask", "scalars",
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


def encode_batch(
    ae, raws: list[dict], device: str, z_cache: ZCache | None = None
) -> list[dict]:
    enc = encode_grids(ae, raws, device, z_cache=z_cache)
    for e, r in zip(enc, raws):
        e["choice"], e["cond"] = r["choice"], r["cond"]
    return enc


def seq_batch(sampler: BCSampler, batch: int, seq: int) -> list[dict]:
    """Draw `batch` windows of `seq` consecutive steps for single players;
    flattened step-major, label on the last step of each window."""
    if sampler._native is not None:
        # Whole micro-batch in one native call: windows draw in parallel
        # (rayon) instead of one failed-draw-prone window per call.
        return sampler._native.sample_windows(batch, seq)
    out: list[dict] = []
    while len(out) < batch * seq:
        out.extend(sampler.sample_window(seq))
    return out[: batch * seq]


def main() -> None:
    # Stall forensics on pods without ptrace (no py-spy): dump all thread
    # stacks to stderr every 5 minutes.
    import faulthandler

    faulthandler.dump_traceback_later(300, repeat=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=["data-human"])
    ap.add_argument("--ae", default="runs/ae_v31_d8c32/ae_v3.pt")
    ap.add_argument("--name", default="bc_v1")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation: effective batch = batch * accum "
                         "(seq runs OOM above micro-batch ~8 on 24GB cards)")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seq", type=int, default=0)
    ap.add_argument("--noop-frac", type=float, default=0.15)
    ap.add_argument("--spawn-frac", type=float, default=0.03,
                    help="fraction of draws that are spawn-placement samples "
                         "(needs formatVersion-2 sidecars)")
    ap.add_argument("--holdout-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--z-cache-gb", type=float, default=300.0,
                    help="RAM LRU budget (GB) for frozen-AE latents keyed "
                         "(game, tick); 0 disables. The full human dataset "
                         "is ~260GB of fp16 latents, so 300 never evicts")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    # 3 memmaps per cached game x ~300 games, per process (workers inherit
    # the parent's table on fork before opening their own): the default 1024
    # soft ulimit is nowhere near enough.
    import resource

    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (_hard, _hard))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    def amp():
        if device == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    roots = [Path(d) for d in args.data]
    sampler = BCSampler(
        roots, holdout_every=args.holdout_every, holdout=False,
        noop_frac=args.noop_frac, spawn_frac=args.spawn_frac,
    )
    try:
        eval_sampler = BCSampler(
            roots, holdout_every=args.holdout_every, holdout=True,
            noop_frac=args.noop_frac, spawn_frac=args.spawn_frac, seed=1,
        )
    except FileNotFoundError:
        eval_sampler = None  # tiny smoke datasets may have no holdout games
    print(f"device: {device}  train games: {len(sampler.games)}  "
          f"holdout games: {len(eval_sampler.games) if eval_sampler else 0}")

    ae = load_ae(args.ae, device)
    # AE-latent cache: encode is ~90% of feed-thread wall and the AE input
    # is per-(game, tick) - see rl.obs.ZCache. BC-only (PPO episodes are
    # unique; its raws never carry z_key).
    z_cache = ZCache(int(args.z_cache_gb * 1e9)) if args.z_cache_gb > 0 else None
    model = BCPolicy(seq=args.seq).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4, fused=(device == "cuda")
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    start_step = 0
    best_metric = -1.0
    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        opt.load_state_dict(ck["opt_state_dict"])
        sched.load_state_dict(ck["sched_state_dict"])
        start_step = ck["step"]
        best_metric = ck.get("best_metric", -1.0)
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
    # Thread-pool raw-batch prefetch. Process workers (DataLoader) looked
    # right on paper but regressed hard on the pods: 16 idle workers, the
    # main process stalled on IPC pickling of ~50MB raw batches. Threads
    # share memory (no pickling) and the decode path releases the GIL
    # (zstd/ofrs), so they actually feed. Depth 8 rides out sampling
    # variance between big and small games.
    sample_pool = ThreadPoolExecutor(max_workers=args.workers)
    raw_q: list = [sample_pool.submit(draw, sampler) for _ in range(8)]

    def next_raws():
        raw_q.append(sample_pool.submit(draw, sampler))
        return raw_q.pop(0).result()

    # Micro-batch pipeline: fetch + AE encode + collate for k+1 runs on a
    # worker thread during k's forward/backward (same surgery as PPO's
    # prefetch; the serialized loop left the GPU ~35% busy).
    # prof: per-phase seconds inside the prefetch worker, read+reset by the
    # 50-step log line - the instrument for the slow-decay hunt (bc_v4 fell
    # 100 -> 25 ex/s over ~8h and a restart fully restored it).
    prof = {"sample": 0.0, "encode": 0.0, "collate": 0.0}

    def prep_next() -> tuple:
        t = time.time()
        raws = next_raws()
        prof["sample"] += time.time() - t
        t = time.time()
        enc = encode_batch(ae, raws, device, z_cache)
        prof["encode"] += time.time() - t
        t = time.time()
        o, choice, cond = collate(enc, device)
        prof["collate"] += time.time() - t
        if args.seq:
            # collate keeps step-major flattening; choice/cond of last steps.
            last = torch.arange(args.seq - 1, len(enc), args.seq, device=device)
            choice = {k: v[last] for k, v in choice.items()}
            cond = cond[last]
        return o, choice, cond

    pf = ThreadPoolExecutor(max_workers=1)
    fut = pf.submit(prep_next)

    t0 = time.time()
    stall_s = 0.0
    zc_prev = (0, 0)  # (hits, misses) at the last 50-step log
    for step in range(start_step + 1, args.steps + 1):
        # Gradient accumulation: seq runs multiply activation memory by the
        # window length, so they train at a small micro-batch but still get
        # a real effective batch (batch * accum) per optimizer step.
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            t_w = time.time()
            o, choice, cond = fut.result()
            stall_s += time.time() - t_w
            fut = pf.submit(prep_next)

            with amp():
                out = model.forward_bc(o, cond)
                loss, parts = bc_loss(
                    out, o if not args.seq else _last_o(o, args.seq, device), choice
                )
            (loss / args.accum).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 50 == 0:
            rate = 50 * args.batch * args.accum / (time.time() - t0)
            t0 = time.time()
            accs = head_accuracy(out, _last_o(o, args.seq, device) if args.seq else o, choice)
            gpu = ""
            if device == "cuda":
                gpu = (f"  cuda {torch.cuda.memory_allocated() / 1e9:.1f}"
                       f"/{torch.cuda.memory_reserved() / 1e9:.1f}GB")
            zc = ""
            if z_cache is not None:
                dh = z_cache.hits - zc_prev[0]
                dm = z_cache.misses - zc_prev[1]
                zc_prev = (z_cache.hits, z_cache.misses)
                zc = (f"  zc {100 * dh / max(1, dh + dm):.0f}% "
                      f"{z_cache.bytes / 1e9:.0f}GB")
            msg = (f"step {step:6d}  loss {float(loss):.4f}  "
                   f"act {accs.get('action', 0):.3f}  "
                   f"act! {accs.get('action_no_noop', 0):.3f}  "
                   f"tile {accs.get('tile_region', float('nan')):.3f}  "
                   f"{rate:.1f} ex/s  stall {stall_s:.1f}s  "
                   f"[smp {prof['sample']:.1f} enc {prof['encode']:.1f} "
                   f"col {prof['collate']:.1f}]{zc}{gpu}")
            for k in prof:
                prof[k] = 0.0
            stall_s = 0.0
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
                    eenc = encode_batch(ae, eraws, device, z_cache)
                    eo, ech, econd = collate(eenc, device)
                    if args.seq:
                        last = torch.arange(args.seq - 1, len(eenc), args.seq, device=device)
                        ech = {k: v[last] for k, v in ech.items()}
                        econd = econd[last]
                    with amp():
                        eout = model.forward_bc(eo, econd)
                    eo_h = _last_o(eo, args.seq, device) if args.seq else eo
                    for k, v in head_accuracy(eout, eo_h, ech).items():
                        agg.setdefault(k, []).append(v)
            evals = {k: float(np.mean(v)) for k, v in agg.items()}
            print(f"  [eval] {evals}", flush=True)
            with log_path.open("a") as f:
                f.write(json.dumps({"step": step, "eval": evals}) + "\n")
            # Best-holdout checkpoint: the PPO warm start should take the
            # plateau model, not whatever the final step happened to be
            # (the honest heads: acted-action + tile accuracy).
            vals = [evals[k] for k in ("action_no_noop", "tile_region") if k in evals]
            metric = float(np.mean(vals)) if vals else -1.0
            if metric > best_metric:
                best_metric = float(metric)
                tmp = run_dir / "bc_best.pt.tmp"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "step": step,
                        "best_metric": best_metric,
                        "args": vars(args),
                    },
                    tmp,
                )
                tmp.rename(run_dir / "bc_best.pt")
                print(f"  [eval] new best ({best_metric:.4f}) -> bc_best.pt", flush=True)
            model.train()

        if step % args.save_every == 0 or step == args.steps:
            tmp = run_dir / "bc.pt.tmp"
            torch.save({
                "model_state_dict": model.state_dict(),
                "opt_state_dict": opt.state_dict(),
                "sched_state_dict": sched.state_dict(),
                "step": step,
                "best_metric": best_metric,
                "args": vars(args),
            }, tmp)
            tmp.rename(run_dir / "bc.pt")  # atomic: no torn ckpt on kill
            # Off-pod durability, same pattern as rl/ppo.py.
            if os.environ.get("HF_TOKEN") and step % (args.save_every * 5) == 0:
                import threading

                def _hf_push(run_dir=run_dir, name=args.name):
                    try:
                        from huggingface_hub import HfApi

                        api = HfApi()
                        api.create_repo("djmango/openfront-rl", exist_ok=True)
                        for fname in ("bc.pt", "bc_best.pt"):
                            if (run_dir / fname).exists():
                                api.upload_file(
                                    path_or_fileobj=str(run_dir / fname),
                                    path_in_repo=f"{name}/{fname}",
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
