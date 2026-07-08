"""PPO over the headless engine with the frozen spatial encoder.

Vectorized: N bridge processes stepped in parallel threads, frozen-AE
encode and policy forward batched across envs on the GPU. v6 action set:
full human intent coverage (expand/attack/boat/build/nuke/diplomacy/
targeted retreat/upgrades/warships/cancels/embargo stop/target player/
alliance extension) with a scalar Beta quantity head.

Logs to TensorBoard (runs/rl/<name>).

Usage:
  uv run python -m rl.ppo --envs 48 --updates 2000 --name ppo_v2 --stage 0

Multi-GPU (v6.1): launch under torchrun; each rank owns an equal slice of
the envs (own VecEnv + collector thread + rollout, pinned to its GPU) and
runs the same minibatch loop on its rank-local data, syncing only
gradients (one all-reduce per optimizer step). Rank 0 owns curriculum,
the win window, checkpoints, state.json, and TensorBoard; episode results
gather to it every update and stage changes broadcast back. WORLD_SIZE=1
(plain python) is byte-for-byte the old single-process behavior.

  torchrun --standalone --nproc_per_node 4 -m rl.ppo --envs 384 ...
"""

import argparse
import contextlib
import json
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from collections import deque

from rl.curriculum import STAGES, WINDOW
from rl.obs import ACTIONS, collate, encode_grids, load_ae
from rl.policy import Policy
from rl.vec import VecEnv

SUB_KEYS = ["player_slot", "tile_region", "build_type", "nuke_type"]
# Entropy-floor adaptation is suppressed for this many updates after a
# resume: synchronized env resets make the first rollouts spawn-phase-heavy
# (masked action space -> artificially low measured entropy), which used to
# spike the coefficient 3x on every restart.
ENT_GRACE_UPDATES = 20
# Latent-cell budget per update forward/backward. collate() pads a minibatch
# to its largest grid, so one BetweenTwoSeas sample (132x223) pads 127
# others to 29k cells each and backward tries multi-GiB activation allocs -
# the stage-5 OOM crash loop that twice threw away a curriculum advance.
# Minibatches are instead split by native grid shape (zero padding waste)
# and capped at this many latent cells per sub-batch; sub-batch losses are
# weighted by their sample fraction, so accumulated gradients are
# numerically identical to the single big batch. Small-map stages stay one
# sub-batch per minibatch (no slowdown); mixed batches drop the padded
# compute entirely.
MAX_UPD_PIX = 1_600_000
OBS_KEYS = [
    "grid", "grid_valid", "legal_tile", "local", "players", "pmask", "scalars",
    "legal_actions", "legal_ptarget", "legal_build", "legal_nuke",
]


@torch.no_grad()
def init_extend(policy: Policy, sd: dict) -> None:
    """Warm start from a pre-v6 checkpoint whose heads are smaller.

    Copies every shape-matching tensor verbatim. Grown heads keep the old
    rows (new action/build indices append at the end) with fresh rows
    initialized small; the nuke head maps old [Atom, Hydrogen, MIRV] onto
    the new 5-way [Atom^, Atomv, Hydrogen^, Hydrogenv, MIRV] by duplicating
    each type's row across both arcs. The Beta quantity head (2 params) is
    incompatible with the old 5-bucket logits and starts fresh. Logs
    exactly what happened to every tensor."""
    nuke_row_map = [0, 0, 1, 1, 2]

    def fresh_(t: torch.Tensor) -> torch.Tensor:
        if t.dim() > 1:
            torch.nn.init.normal_(t, std=0.01)
        else:
            t.zero_()
        return t

    own = policy.state_dict()
    new_sd, copied, extended, fresh = {}, [], [], []
    for k, v in own.items():
        old = sd.get(k)
        if old is not None:
            old = old.to(v.device, v.dtype)
        if old is not None and old.shape == v.shape:
            new_sd[k] = old
            copied.append(k)
        elif (
            old is not None
            and k.split(".")[0] in ("head_action", "head_build")
            and old.shape[1:] == v.shape[1:]
            and old.shape[0] < v.shape[0]
        ):
            t = fresh_(v.clone())
            t[: old.shape[0]] = old
            new_sd[k] = t
            extended.append(f"{k} ({old.shape[0]}->{v.shape[0]} rows)")
        elif (
            old is not None
            and k.split(".")[0] == "head_nuke"
            and old.shape[0] == 3
            and v.shape[0] == 5
        ):
            new_sd[k] = old[nuke_row_map].clone()
            extended.append(f"{k} (3->5 rows, per-type arc duplication)")
        else:
            new_sd[k] = fresh_(v.clone())
            fresh.append(k)
    policy.load_state_dict(new_sd, strict=True)
    print(f"init-extend: {len(copied)} tensors copied verbatim")
    for k in extended:
        print(f"init-extend: extended {k}")
    for k in fresh:
        print(f"init-extend: fresh {k}")


class NullWriter:
    """SummaryWriter stand-in for non-zero DDP ranks (rank 0 owns TB)."""

    def add_scalar(self, *a, **k) -> None:
        pass

    def add_histogram(self, *a, **k) -> None:
        pass

    def close(self) -> None:
        pass


def save_train_state(out_dir: Path, state: dict) -> None:
    """Atomic per-update sidecar so cheap training state (win window,
    stage, counters) never lags the 10-update weight cadence or vanishes
    in a crash."""
    tmp = out_dir / "state.json.tmp"
    tmp.write_text(json.dumps(state))
    tmp.rename(out_dir / "state.json")


@torch.no_grad()
def run_eval(
    policy, ae, device: str, stage: int, episodes: int, max_ticks: int
) -> dict:
    """Deployment-style eval: fresh fixed-seed envs (worker i always plays
    seed w{i}-ep0 at this stage), greedy actions, one episode per env.
    Returns win rate / mean score over the completed episodes."""
    vec = VecEnv(episodes, stage, max_ticks, 10)
    results: dict[int, dict] = {}
    try:
        step_cap = max_ticks // STAGES[stage].decision_ticks + 64
        for _ in range(step_cap):
            pending = [i for i in range(episodes) if i not in results]
            if not pending:
                break
            obs_list = encode_grids(ae, vec.obs_group(pending), device)
            ot = {
                k: torch.from_numpy(v).to(device)
                for k, v in collate(obs_list, OBS_KEYS).items()
            }
            choices, _, _ = policy.act(ot, greedy=True)
            vec.send_group(pending, choices)
            for i, (_, _, info) in zip(pending, vec.recv_group(pending)):
                if info is not None:
                    results[i] = info
    finally:
        vec.close()
    if not results:
        return {"win": 0.0, "score": 0.0, "episodes": 0}
    wins = [float(r["won"]) for r in results.values()]
    scores = [r["score"] for r in results.values()]
    return {
        "win": float(np.mean(wins)),
        "score": float(np.mean(scores)),
        "episodes": len(results),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0, help="starting curriculum stage")
    ap.add_argument("--ckpt", default="runs/ae_v31_d8c32/ae_v3.pt")
    ap.add_argument("--name", default="ppo_v1")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--rollout", type=int, default=32, help="steps per env per update")
    ap.add_argument("--epochs", type=int, default=2,
                    help="PPO epochs per rollout (v6.1: 3 -> 2 halves update "
                         "time; watch loss/approx_kl + loss/clip_frac)")
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--stage-lr-decay", type=float, default=0.85,
                    help="LR warmdown: lr = lr * decay^stage (AlphaFront's trick)")
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--ent-coef-final", type=float, default=0.002,
                    help="entropy coef anneals linearly to this (v2c sat at "
                         "entropy ~8 under a flat 0.01, exploring forever)")
    ap.add_argument("--ent-anneal-updates", type=int, default=4000)
    ap.add_argument("--ent-floor", type=float, default=3.5,
                    help="adaptive entropy floor (0 = off): when mean policy "
                         "entropy drops below this, the entropy coef scales "
                         "up (x1.3/update, cap x30) until it recovers. "
                         "ppo_v4 collapsed 9 -> 2.8 nats and stopped winning "
                         "(deterministic safe-2nd); the schedule alone can't "
                         "prevent that")
    ap.add_argument("--ent-coef-q", type=float, default=0.002,
                    help="entropy coef for the Beta quantity head; separate "
                         "because differential entropy can be negative and "
                         "must stay out of the discrete-nats floor")
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-episode-ticks", type=int, default=15000)
    ap.add_argument("--decision-ticks", type=int, default=10,
                    help="unused for stepping (stages carry their own); kept for compat")
    ap.add_argument("--eval-every", type=int, default=300,
                    help="updates between fixed-seed greedy eval passes (0 = off)")
    ap.add_argument("--eval-episodes", type=int, default=8)
    ap.add_argument("--resume", default=None, help="policy.pt to load before training")
    ap.add_argument("--init", default=None,
                    help="warm-start from a BC checkpoint (bc.pt): loads the "
                         "shared Policy weights, folds the winner-bucket "
                         "conditioning into the heads, fresh optimizer/stage")
    ap.add_argument("--init-cond-bucket", type=int, default=7,
                    help="placement bucket to condition the BC prior on "
                         "(7 = winner tier)")
    ap.add_argument("--init-extend", default=None,
                    help="warm-start from a pre-v6 policy.pt: copies every "
                         "shape-matching tensor, extends grown heads "
                         "(action 14->21, build 6->7, nuke 3->5 with the "
                         "arc rows seeded from the old per-type rows) and "
                         "leaves the Beta quantity head fresh")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                    help="torch.compile the update path (policy.evaluate); "
                         "default: on for CUDA, off for MPS/CPU")
    args = ap.parse_args()

    # DDP: torchrun sets RANK/WORLD_SIZE/LOCAL_RANK. Plain python (no
    # torchrun) leaves world=1 and every dist branch below is skipped.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main = rank == 0
    if world > 1:
        # Long timeout: rank 0 runs the (minutes-long) eval passes alone
        # while other ranks wait at the next collective.
        dist.init_process_group(
            "nccl" if torch.cuda.is_available() else "gloo",
            timeout=timedelta(hours=2),
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    cuda = device.startswith("cuda")
    if args.envs % world != 0:
        raise SystemExit(f"--envs {args.envs} not divisible by WORLD_SIZE {world}")
    n_local = args.envs // world
    if is_main:
        extra = f" ({world} ranks x {n_local})" if world > 1 else ""
        print(f"device: {device}, envs: {args.envs}{extra}")
    torch.set_float32_matmul_precision("high")

    def amp():
        if cuda:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    out_dir = Path("runs/rl") / args.name
    writer = SummaryWriter(out_dir) if is_main else NullWriter()

    # Env slice for this rank: idx_offset keeps worker seeds (rng + game
    # seed strings) globally unique, so ranks never replay identical
    # episodes.
    vec = VecEnv(
        n_local, args.stage, args.max_episode_ticks, args.decision_ticks,
        idx_offset=rank * n_local,
    )
    stage = args.stage
    recent_scores: deque[float] = deque(maxlen=WINDOW)
    recent_wins: deque[float] = deque(maxlen=WINDOW)
    ae = load_ae(args.ckpt, device)
    base_policy = Policy().to(device)
    opt = torch.optim.AdamW(base_policy.parameters(), lr=args.lr, fused=cuda)
    start_update = 0
    global_step = 0
    ent_scale = 1.0  # adaptive entropy-floor multiplier (see --ent-floor)
    if args.init and not args.resume:
        # BC warm start: shared Policy weights from a BCPolicy checkpoint
        # (cond/temporal modules dropped), then fold the placement
        # conditioning h += cond_proj(cond_emb[bucket]) into the head
        # biases. That fold is exact: the BC bias is additive after the
        # trunk and every head consumes h linearly. The value head stays
        # untrained (BC has no value loss), so early PPO updates mostly
        # fit the critic - expect noisy advantages for the first ~100
        # updates; the behavior prior is what we're buying.
        sd = torch.load(args.init, map_location=device, weights_only=False)[
            "model_state_dict"
        ]
        own = base_policy.state_dict()
        shared = {k: v for k, v in sd.items() if k in own}
        base_policy.load_state_dict(shared, strict=True)
        if "cond_emb.weight" in sd:
            c = (
                sd["cond_proj.weight"] @ sd["cond_emb.weight"][args.init_cond_bucket]
                + sd["cond_proj.bias"]
            ).to(device)
            with torch.no_grad():
                for head in (
                    base_policy.head_action, base_policy.head_player_q,
                    base_policy.head_build, base_policy.head_nuke,
                    base_policy.head_quantity, base_policy.head_value,
                ):
                    head.bias += head.weight @ c
                conv = base_policy.head_tile[0]  # Conv2d(gc + hidden, ., 1)
                gc_in = conv.in_channels - c.shape[0]
                conv.bias += conv.weight[:, gc_in:, 0, 0] @ c
        dropped = sorted({k.split(".")[0] for k in sd if k not in own})
        print(
            f"warm start from {args.init}: {len(shared)} tensors, "
            f"cond bucket {args.init_cond_bucket} folded, dropped {dropped}"
        )

    if args.init_extend and not args.resume:
        sd = torch.load(args.init_extend, map_location=device, weights_only=False)[
            "model_state_dict"
        ]
        init_extend(base_policy, sd)
        print(f"init-extend warm start from {args.init_extend}")

    episodes_done = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        base_policy.load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state:
            opt.load_state_dict(state["optimizer_state_dict"])
        # Checkpoint stage is authoritative on resume: --stage only seeds
        # fresh runs (supervisors pass a stale --stage on every relaunch).
        if "stage" in state:
            stage = int(state["stage"])
        # Win-gate evidence survives restarts: without this, every relaunch
        # forced the agent to re-earn the whole advancement window.
        recent_wins.extend(state.get("recent_wins", []))
        recent_scores.extend(state.get("recent_scores", []))
        start_update = int(state.get("update", 0))
        global_step = int(state.get("global_step", 0))
        ent_scale = float(state.get("ent_scale", 1.0))
        # state.json is written EVERY update (weights only every 10): when
        # it's newer than the checkpoint, its cheap state wins so the win
        # window / stage / counters don't rewind up to 10 updates.
        state_path = out_dir / "state.json"
        if state_path.exists():
            try:
                js = json.loads(state_path.read_text())
            except (OSError, ValueError) as e:
                print(f"state.json unreadable ({e}); using checkpoint state")
                js = None
            if js and int(js.get("update", -1)) >= start_update:
                stage = int(js.get("stage", stage))
                start_update = int(js["update"])
                global_step = int(js.get("global_step", global_step))
                episodes_done = int(js.get("episodes_done", 0))
                ent_scale = float(js.get("ent_scale", ent_scale))
                recent_wins.clear()
                recent_wins.extend(js.get("recent_wins", []))
                recent_scores.clear()
                recent_scores.extend(js.get("recent_scores", []))
                print(f"state.json (update {start_update}) supersedes checkpoint")
        vec.set_stage(stage)
        if is_main:
            print(
                f"resumed from {args.resume}: update {start_update}, "
                f"step {global_step}, stage {stage}, "
                f"win-window {len(recent_wins)}/{WINDOW}"
            )
    # Post-resume grace: hold the entropy-floor controller for a while so
    # startup artifacts can't spike the coefficient (see ENT_GRACE_UPDATES).
    ent_grace_until = start_update + ENT_GRACE_UPDATES if args.resume else 0

    def set_lr(current_stage: int) -> float:
        lr = args.lr * args.stage_lr_decay ** current_stage
        for grp in opt.param_groups:
            grp["lr"] = lr
        return lr

    set_lr(stage)

    if world > 1:
        # Ranks must start bitwise-identical (fresh Policy() inits differ
        # per process); after that, identical all-reduced grads + a
        # deterministic elementwise optimizer keep them in sync forever.
        with torch.no_grad():
            for t in base_policy.state_dict().values():
                if dist.get_backend() == "gloo" and t.device.type != "cpu":
                    c = t.detach().cpu()
                    dist.broadcast(c, 0)
                    t.copy_(c)
                else:
                    dist.broadcast(t, 0)

    # Compile the update path after weights load; checkpoints always save
    # base_policy's (unprefixed) state_dict. Compiling the bound method is
    # what actually captures evaluate() - torch.compile(module) only wraps
    # forward(). Default on for CUDA, off elsewhere (MPS inductor support
    # is flaky and bf16/compile only need to pay off on the pods).
    do_compile = args.compile if args.compile is not None else cuda
    evaluate = (
        torch.compile(base_policy.evaluate, dynamic=True)
        if do_compile else base_policy.evaluate
    )
    if is_main and do_compile:
        print("torch.compile: update path (policy.evaluate) enabled")

    T, N = args.rollout, n_local
    rng = np.random.default_rng(rank)
    t0 = time.time()
    t0_step = global_step

    # Host->device staging. Reusable pinned buffers double PCIe bandwidth and
    # allow async copies; without them every upload is a pageable-memory
    # single-thread crawl on the main process (the profiled bottleneck).
    # Reuse is safe: every consumer syncs the stream (.item()/.cpu()) before
    # the same buffer set is written again.
    pinned: dict[str, dict[str, torch.Tensor]] = {}

    def to_device(np_dict: dict, pool: str) -> dict:
        bufs = pinned.setdefault(pool, {})
        out = {}
        for k, v in np_dict.items():
            t = torch.from_numpy(v)
            if cuda:
                n = t.numel()
                buf = bufs.get(k)
                if buf is None or buf.numel() < n or buf.dtype != t.dtype:
                    buf = torch.empty(n, dtype=t.dtype, pin_memory=True)
                    bufs[k] = buf
                staged = buf[:n]
                staged.copy_(t.reshape(-1))
                d = staged.view(t.shape).to(device, non_blocking=True)
            else:
                d = t.to(device)
            # Rollout grids ride as fp16 (transfer cost); model runs fp32/amp.
            out[k] = d.float() if d.dtype == torch.float16 else d
        return out

    # ---- v4.1 async pipeline ----
    # A collector thread builds rollout k+1 (env stepping + GPU encode/act
    # on a snapshot policy) while the main thread runs the PPO update on
    # rollout k. One-rollout staleness: old_logp is recorded from the acting
    # snapshot, so the clipped ratio still bounds movement from the policy
    # that actually produced the data.
    act_policy = Policy().to(device)
    act_policy.load_state_dict(base_policy.state_dict())
    act_policy.eval()
    act_lock = threading.Lock()  # act forward vs. weight refresh
    win_lock = threading.Lock()  # advancement deques (collector vs. logging)
    shared = {
        "stage": stage,
        "global_step": global_step,
        "episodes_done": episodes_done,
    }
    # DDP: episode results buffered here by the collector, gathered to
    # rank 0 at every update boundary (it owns the win window/curriculum).
    pending_eps: list[dict] = []

    # Two env groups pipelined: while one group's Node processes step the
    # game, the GPU encodes + acts for the other group.
    half = max(1, N // 2)
    groups = (list(range(half)), list(range(half, N)))

    def act_group(idxs: list[int]) -> tuple:
        obs_list = encode_grids(ae, vec.obs_group(idxs), device, fp16=True)
        ot = to_device(collate(obs_list, OBS_KEYS), f"act{idxs[0]}")
        with act_lock, torch.no_grad(), amp():
            choices, logp, value = act_policy.act(ot)
        return obs_list, choices, logp, value

    def collect_rollout() -> dict:
        t_roll0 = time.time()
        obs_buf: list[list] = [[None] * N for _ in range(T)]
        choice_buf: list[list] = [[None] * N for _ in range(T)]
        logp_buf = np.zeros((T, N), dtype=np.float32)
        value_buf = np.zeros((T, N), dtype=np.float32)
        reward_buf = np.zeros((T, N), dtype=np.float32)
        done_buf = np.zeros((T, N), dtype=np.float32)
        action_counts = np.zeros(len(ACTIONS))
        quantity_fracs: dict[int, list[float]] = {}  # action idx -> sampled fracs

        def record(t: int, idxs: list[int], pack: tuple, results: list) -> None:
            obs_list, choices, logp, value = pack
            gs = shared["global_step"]
            for j, i in enumerate(idxs):
                obs_buf[t][i] = obs_list[j]
                choice_buf[t][i] = choices[j]
                logp_buf[t, i] = logp[j]
                value_buf[t, i] = value[j]
                action_counts[choices[j]["action"]] += 1
                q = choices[j].get("quantity_frac")
                if q is not None:
                    quantity_fracs.setdefault(choices[j]["action"], []).append(q)
                r, d, info = results[j]
                reward_buf[t, i] = r
                done_buf[t, i] = float(d)
                if info is None:
                    continue
                if world > 1:
                    # Curriculum/win-window/TB are rank-0-only under DDP;
                    # buffer for the update-boundary gather instead of
                    # deciding locally (a rank sees only its env slice).
                    with win_lock:
                        pending_eps.append(info)
                    continue
                shared["episodes_done"] += 1
                writer.add_scalar("episode/reward", info["reward"], gs)
                writer.add_scalar("episode/length", info["length"], gs)
                writer.add_scalar("episode/final_tiles", info["final_tiles"], gs)
                writer.add_scalar("episode/final_tick", info["final_tick"], gs)
                writer.add_scalar("episode/place", info["place"], gs)
                writer.add_scalar("episode/score", info["score"], gs)
                writer.add_scalar("episode/won", float(info["won"]), gs)
                writer.add_scalar("episode/wasted", info["wasted"], gs)
                writer.add_scalar("curriculum/episode_stage", info["stage"], gs)
                writer.add_scalar(
                    "curriculum/rehearsal", float(info["rehearsal"]), gs
                )
                # Only current-stage, non-rehearsal episodes count toward
                # advancement; the gate is win rate, not placement.
                with win_lock:
                    if info["stage"] == shared["stage"] and not info["rehearsal"]:
                        recent_scores.append(info["score"])
                        recent_wins.append(float(info["won"]))
                    advance = (
                        len(recent_wins) == WINDOW
                        and np.mean(recent_wins) > STAGES[shared["stage"]].win_at
                        and shared["stage"] < len(STAGES) - 1
                    )
                    if advance:
                        shared["stage"] += 1
                        recent_scores.clear()
                        recent_wins.clear()
                if advance:
                    # Save at the end of the in-flight update: v5.1 twice
                    # cleared stage 5 and lost it to a crash inside the
                    # 10-update save cadence.
                    shared["ckpt_advance"] = True
                    vec.set_stage(shared["stage"])
                    lr_now = set_lr(shared["stage"])
                    st = STAGES[shared["stage"]]
                    print(
                        f"=== curriculum advance -> stage {shared['stage']}: "
                        f"maps={','.join(st.maps)} nations={st.nations} "
                        f"bots={st.bots} {st.difficulty}  lr -> {lr_now:.2e}",
                        flush=True,
                    )

        pack0 = act_group(groups[0])
        vec.send_group(groups[0], pack0[1])
        for t in range(T):
            pack1 = None
            if groups[1]:
                pack1 = act_group(groups[1])  # overlaps group 0 stepping
                vec.send_group(groups[1], pack1[1])
            record(t, groups[0], pack0, vec.recv_group(groups[0]))
            if t < T - 1:
                pack0 = act_group(groups[0])  # overlaps group 1 stepping
                vec.send_group(groups[0], pack0[1])
            if pack1 is not None:
                record(t, groups[1], pack1, vec.recv_group(groups[1]))
            # Count TOTAL env steps across ranks (every rank advances in
            # lockstep, so this stays consistent without communication).
            shared["global_step"] += N * world

        # Bootstrap values and GAE per env (values from the acting snapshot,
        # consistent with value_buf).
        obs_last = encode_grids(ae, vec.obs(), device)
        with act_lock, torch.no_grad(), amp():
            ot = {
                k: (torch.from_numpy(v).to(device))
                for k, v in collate(obs_last, OBS_KEYS).items()
            }
            last_value = act_policy.forward(ot)["value"].float().cpu().numpy()
        adv = np.zeros((T, N), dtype=np.float32)
        last_gae = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else value_buf[t + 1]
            nonterminal = 1.0 - done_buf[t]
            delta = reward_buf[t] + args.gamma * next_v * nonterminal - value_buf[t]
            last_gae = delta + args.gamma * args.lam * nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + value_buf

        all_choice = [c for row in choice_buf for c in row]
        choice_t = {"action": torch.tensor([c["action"] for c in all_choice])}
        for k in SUB_KEYS:
            choice_t[k] = torch.tensor([c.get(k, -1) for c in all_choice])
        choice_t["quantity_frac"] = torch.tensor(
            [c.get("quantity_frac", -1.0) for c in all_choice], dtype=torch.float32
        )
        return {
            "all_obs": [o for row in obs_buf for o in row],
            "choice_t": choice_t,
            "logp": logp_buf.reshape(-1),
            "adv": adv.reshape(-1),
            "returns": returns.reshape(-1),
            "action_counts": action_counts,
            "quantity_fracs": quantity_fracs,
            "roll_s": time.time() - t_roll0,
        }

    rollout_q: queue.Queue = queue.Queue(maxsize=1)
    stop = threading.Event()

    def collector() -> None:
        try:
            while not stop.is_set():
                data = collect_rollout()
                while not stop.is_set():
                    try:
                        rollout_q.put(data, timeout=1.0)
                        break
                    except queue.Full:
                        continue
        except BaseException as e:  # propagate crashes instead of hanging main
            if stop.is_set():  # teardown race (vec closed mid-rollout)
                return
            rollout_q.put(e)
            raise

    threading.Thread(target=collector, daemon=True, name="collector").start()

    # Update-phase minibatches are collated + staged on a worker thread so
    # the (numpy, GIL-releasing) data work overlaps GPU forward/backward.
    mb_pool = ThreadPoolExecutor(max_workers=1)

    def sync_episodes() -> None:
        """DDP update-boundary sync: every rank's buffered episode results
        gather to rank 0, which owns the win window and the advancement
        decision; the (possibly advanced) stage broadcasts back so every
        rank's vec.set_stage + lr move together. At most one rollout of
        latency vs. the single-process mid-rollout advance."""
        with win_lock:
            eps = list(pending_eps)
            pending_eps.clear()
        gathered: list = [None] * world
        dist.all_gather_object(gathered, eps)
        advanced = False
        if is_main:
            gs = shared["global_step"]
            for info in (e for rank_eps in gathered for e in rank_eps):
                shared["episodes_done"] += 1
                writer.add_scalar("episode/reward", info["reward"], gs)
                writer.add_scalar("episode/length", info["length"], gs)
                writer.add_scalar("episode/final_tiles", info["final_tiles"], gs)
                writer.add_scalar("episode/final_tick", info["final_tick"], gs)
                writer.add_scalar("episode/place", info["place"], gs)
                writer.add_scalar("episode/score", info["score"], gs)
                writer.add_scalar("episode/won", float(info["won"]), gs)
                writer.add_scalar("episode/wasted", info["wasted"], gs)
                writer.add_scalar("curriculum/episode_stage", info["stage"], gs)
                writer.add_scalar(
                    "curriculum/rehearsal", float(info["rehearsal"]), gs
                )
                with win_lock:
                    if info["stage"] == shared["stage"] and not info["rehearsal"]:
                        recent_scores.append(info["score"])
                        recent_wins.append(float(info["won"]))
                    if (
                        len(recent_wins) == WINDOW
                        and np.mean(recent_wins) > STAGES[shared["stage"]].win_at
                        and shared["stage"] < len(STAGES) - 1
                    ):
                        shared["stage"] += 1
                        recent_scores.clear()
                        recent_wins.clear()
                        advanced = True
        payload = [shared["stage"], shared["episodes_done"], advanced]
        dist.broadcast_object_list(payload, src=0)
        new_stage, eps_done, advanced = payload
        if is_main:
            shared["episodes_done"] = eps_done
        if advanced:
            shared["stage"] = new_stage
            shared["ckpt_advance"] = True
            vec.set_stage(new_stage)
            lr_now = set_lr(new_stage)
            if is_main:
                st = STAGES[new_stage]
                print(
                    f"=== curriculum advance -> stage {new_stage}: "
                    f"maps={','.join(st.maps)} nations={st.nations} "
                    f"bots={st.bots} {st.difficulty}  lr -> {lr_now:.2e}",
                    flush=True,
                )

    for update in range(start_update + 1, args.updates + 1):
        t_wait0 = time.time()
        data = rollout_q.get()
        if isinstance(data, BaseException):
            raise data
        wait_s = time.time() - t_wait0
        if world > 1:
            sync_episodes()
        all_obs = data["all_obs"]
        choice_t = data["choice_t"]
        action_counts = data["action_counts"]
        roll_s = data["roll_s"]
        a = data["adv"]
        adv_t = torch.from_numpy((a - a.mean()) / (a.std() + 1e-8)).to(device)
        ret_t = torch.from_numpy(data["returns"]).to(device)
        old_logp = torch.from_numpy(data["logp"]).to(device)

        # Entropy anneal: linear from ent-coef to ent-coef-final so late
        # training commits instead of exploring forever. The adaptive floor
        # scale (updated after each update from measured entropy) multiplies
        # the schedule so exploration can't collapse entirely.
        frac = min(1.0, update / max(1, args.ent_anneal_updates))
        ent_coef = (args.ent_coef + (args.ent_coef_final - args.ent_coef) * frac) * ent_scale

        B_total = T * N
        idx = np.arange(B_total)
        pl_sum = vl_sum = ent_sum = entq_sum = kl_sum = clip_sum = 0.0
        n_mb = 0
        t_upd0 = time.time()
        stall_s = 0.0

        def prep(mb: np.ndarray) -> list[tuple[np.ndarray, dict]]:
            # Shape-grouped, pixel-budgeted sub-batches (see MAX_UPD_PIX).
            by_shape: dict[tuple, list[int]] = {}
            for i in mb:
                by_shape.setdefault(all_obs[i]["grid"].shape[1:], []).append(int(i))
            subs = []
            for (h, w), idxs in by_shape.items():
                per = max(1, MAX_UPD_PIX // (h * w))
                for k in range(0, len(idxs), per):
                    part = idxs[k : k + per]
                    subs.append(
                        (np.asarray(part), collate([all_obs[i] for i in part], OBS_KEYS))
                    )
            return subs

        sub_i = 0
        for _ in range(args.epochs):
            rng.shuffle(idx)
            mbs = np.split(idx, max(1, B_total // args.minibatch))
            fut = mb_pool.submit(prep, mbs[0])
            for i_mb, mb in enumerate(mbs):
                t_w = time.time()
                subs = fut.result()  # collated on the worker thread
                stall_s += time.time() - t_w
                if i_mb + 1 < len(mbs):
                    fut = mb_pool.submit(prep, mbs[i_mb + 1])
                opt.zero_grad(set_to_none=True)
                pg_mb = vl_mb = ent_mb = entq_mb = kl_mb = clip_mb = 0.0
                for sub, o_np in subs:
                    w_sub = len(sub) / len(mb)
                    # Alternating pinned buffer sets: the per-sub .item()
                    # sync guarantees a set's copy landed before its reuse.
                    o_mb = to_device(o_np, f"upd{sub_i % 2}")
                    sub_i += 1
                    sub_t = torch.from_numpy(sub)
                    c_mb = {k: v[sub_t].to(device) for k, v in choice_t.items()}
                    sub_t = sub_t.to(device)
                    with amp():
                        logp, ent, ent_q, value = evaluate(o_mb, c_mb)
                    logp, ent, value = logp.float(), ent.float(), value.float()
                    ratio = (logp - old_logp[sub_t]).exp()
                    a_mb = adv_t[sub_t]
                    pg = -torch.min(
                        ratio * a_mb,
                        ratio.clamp(1 - args.clip, 1 + args.clip) * a_mb,
                    ).mean()
                    vl = F.mse_loss(value, ret_t[sub_t])
                    ent_m = ent.mean()
                    # Beta head differential entropy, averaged over the
                    # samples that actually used the quantity head.
                    n_q = (c_mb["quantity_frac"] >= 0).float().sum().clamp(min=1.0)
                    ent_qm = ent_q.float().sum() / n_q
                    with torch.no_grad():
                        # Approx KL (mean old_logp - logp) and the fraction
                        # of ratios the PPO clip actually touched: the
                        # epochs 3->2 gate (KL should stay ~1e-2-ish and
                        # clip-frac well under ~0.2).
                        kl_m = (old_logp[sub_t] - logp).mean()
                        clip_m = ((ratio - 1.0).abs() > args.clip).float().mean()
                    (
                        (
                            pg
                            + args.vf_coef * vl
                            - ent_coef * ent_m
                            - args.ent_coef_q * ent_qm
                        )
                        * w_sub
                    ).backward()
                    pg_mb += float(pg.item()) * w_sub
                    vl_mb += float(vl.item()) * w_sub
                    ent_mb += float(ent_m.item()) * w_sub
                    entq_mb += float(ent_qm.item()) * w_sub
                    kl_mb += float(kl_m.item()) * w_sub
                    clip_mb += float(clip_m.item()) * w_sub
                if world > 1:
                    # One flat all-reduce per optimizer step: each rank
                    # accumulated grads from its own rollout's sub-batches;
                    # averaging here is the only weight-affecting DDP sync
                    # (clip + step then act on identical grads everywhere).
                    # Every param participates, zero-filled where a rank's
                    # minibatch left a head untouched (grad None): ranks
                    # must flatten identical layouts, and the averaged grad
                    # must land on all of them or their weights drift.
                    params = list(base_policy.parameters())
                    grads = [
                        p.grad if p.grad is not None else torch.zeros_like(p)
                        for p in params
                    ]
                    flat = torch._utils._flatten_dense_tensors(grads)
                    if dist.get_backend() == "gloo" and flat.device.type != "cpu":
                        c = flat.cpu()
                        dist.all_reduce(c)
                        flat = c.to(flat.device)
                    else:
                        dist.all_reduce(flat)
                    flat /= world
                    for p, r in zip(
                        params, torch._utils._unflatten_dense_tensors(flat, grads)
                    ):
                        if p.grad is None:
                            p.grad = r.clone()
                        else:
                            p.grad.copy_(r)
                torch.nn.utils.clip_grad_norm_(base_policy.parameters(), 0.5)
                opt.step()
                pl_sum += pg_mb
                vl_sum += vl_mb
                ent_sum += ent_mb
                entq_sum += entq_mb
                kl_sum += kl_mb
                clip_sum += clip_mb
                n_mb += 1

        if world > 1:
            # Average the update stats across ranks: rank 0's logs then
            # describe the whole fleet, and (crucially) the entropy-floor
            # controller below sees the same number everywhere, keeping
            # ent_scale - a loss coefficient - identical across ranks.
            stats = torch.tensor(
                [pl_sum, vl_sum, ent_sum, entq_sum, kl_sum, clip_sum],
                device=device if dist.get_backend() != "gloo" else "cpu",
            )
            dist.all_reduce(stats)
            stats /= world
            pl_sum, vl_sum, ent_sum, entq_sum, kl_sum, clip_sum = stats.tolist()

        # Entropy floor controller: nudge the coef scale toward keeping
        # measured entropy above the floor, with hysteresis so it doesn't
        # oscillate. Multiplicative so it composes with the anneal. Held
        # for ENT_GRACE_UPDATES after a resume (spawn-heavy startup rollouts
        # read artificially low entropy). Discrete heads only: the Beta
        # quantity head's differential entropy lives on another scale.
        if args.ent_floor > 0 and n_mb and update > ent_grace_until:
            ent_mean = ent_sum / n_mb
            if ent_mean < args.ent_floor:
                ent_scale = min(ent_scale * 1.3, 30.0)
            elif ent_mean > args.ent_floor * 1.4:
                ent_scale = max(ent_scale / 1.3, 1.0)

        # Publish the updated weights to the acting snapshot.
        with act_lock, torch.no_grad():
            act_policy.load_state_dict(base_policy.state_dict())

        upd_s = time.time() - t_upd0
        global_step = shared["global_step"]
        stage = shared["stage"]
        episodes_done = shared["episodes_done"]
        tps = (global_step - t0_step) * STAGES[stage].decision_ticks / (time.time() - t0)
        with win_lock:
            roll = float(np.mean(recent_scores)) if recent_scores else 0.0
            roll_win = float(np.mean(recent_wins)) if recent_wins else 0.0
        writer.add_scalar("perf/rollout_s", roll_s, global_step)
        writer.add_scalar("perf/update_s", upd_s, global_step)
        writer.add_scalar("perf/update_stall_s", stall_s, global_step)
        writer.add_scalar("perf/rollout_wait_s", wait_s, global_step)
        writer.add_scalar("loss/policy", pl_sum / n_mb, global_step)
        writer.add_scalar("loss/value", vl_sum / n_mb, global_step)
        writer.add_scalar("loss/entropy", ent_sum / n_mb, global_step)
        writer.add_scalar("loss/entropy_q", entq_sum / n_mb, global_step)
        writer.add_scalar("loss/approx_kl", kl_sum / n_mb, global_step)
        writer.add_scalar("loss/clip_frac", clip_sum / n_mb, global_step)
        writer.add_scalar("loss/ent_coef", ent_coef, global_step)
        writer.add_scalar("loss/ent_scale", ent_scale, global_step)
        writer.add_scalar("loss/lr", opt.param_groups[0]["lr"], global_step)
        writer.add_scalar("perf/game_ticks_per_s", tps, global_step)
        writer.add_scalar("perf/episodes_done", episodes_done, global_step)
        writer.add_scalar("curriculum/stage", stage, global_step)
        writer.add_scalar("curriculum/rolling_score", roll, global_step)
        writer.add_scalar("curriculum/rolling_win", roll_win, global_step)
        for i, a in enumerate(ACTIONS):
            writer.add_scalar(f"actions/{a}", action_counts[i] / (T * N), global_step)
        # Sampled Beta quantity fractions: mean/std per action type plus a
        # rollout-wide histogram (replaces the old per-bucket rates).
        all_fracs: list[float] = []
        for ai, fr in data["quantity_fracs"].items():
            all_fracs.extend(fr)
            writer.add_scalar(
                f"quantity/{ACTIONS[ai]}_mean", float(np.mean(fr)), global_step
            )
            writer.add_scalar(
                f"quantity/{ACTIONS[ai]}_std", float(np.std(fr)), global_step
            )
        if all_fracs:
            writer.add_histogram(
                "quantity/frac", np.asarray(all_fracs, dtype=np.float32), global_step
            )
        if is_main:
            print(
                f"update {update:4d}  step {global_step:7d}  eps {episodes_done:4d}  "
                f"stage {stage}  roll-win {roll_win:.2f}  roll-score {roll:.2f}  "
                f"pg {pl_sum / n_mb:+.4f}  vf {vl_sum / n_mb:.4f}  ent {ent_sum / n_mb:.3f}  "
                f"kl {kl_sum / n_mb:.4f}  clip {clip_sum / n_mb:.3f}  "
                f"ecoef {ent_coef:.4f}  "
                f"{tps:.0f} game-ticks/s  "
                f"[roll {roll_s:.1f}s upd {upd_s:.1f}s wait {wait_s:.1f}s stall {stall_s:.1f}s]",
                flush=True,
            )

        if is_main and args.eval_every and update % args.eval_every == 0:
            # Deployment-style number: fixed seeds, greedy actions. This is
            # the curve to compare local replay sessions against (the
            # rolling train win rate is exploration-sampled and its window
            # resets on ops churn).
            t_eval = time.time()
            ev = run_eval(
                base_policy, ae, device, stage, args.eval_episodes,
                args.max_episode_ticks,
            )
            writer.add_scalar("eval/win", ev["win"], global_step)
            writer.add_scalar("eval/score", ev["score"], global_step)
            writer.add_scalar("eval/stage", stage, global_step)
            print(
                f"[eval] stage {stage}  win {ev['win']:.2f}  "
                f"score {ev['score']:.2f}  ({ev['episodes']} eps, "
                f"{time.time() - t_eval:.0f}s)",
                flush=True,
            )
            t0, t0_step = time.time(), global_step  # don't count eval in tps

        with win_lock:
            wins_snap = list(recent_wins)
            scores_snap = list(recent_scores)
        # Cheap restart-proof state, every update (weights are 10x rarer).
        # Rank 0 only: it owns the win window, so other ranks' copies are
        # empty and must never clobber state.json.
        if is_main:
            save_train_state(
                out_dir,
                {
                    "stage": stage,
                    "update": update,
                    "global_step": global_step,
                    "episodes_done": episodes_done,
                    "recent_wins": wins_snap,
                    "recent_scores": scores_snap,
                    "ent_scale": ent_scale,
                    "lr": opt.param_groups[0]["lr"],
                },
            )

        # Popped on every rank (the flag is set on all of them at the DDP
        # sync) so it can't go stale; only rank 0 writes the file.
        ckpt_due = (
            update % 10 == 0
            or update == args.updates
            or shared.pop("ckpt_advance", False)
        )
        if is_main and ckpt_due:
            tmp = out_dir / "policy.pt.tmp"
            torch.save(
                {
                    "model_state_dict": base_policy.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "stage": shared["stage"],
                    "update": update,
                    "global_step": shared["global_step"],
                    "recent_wins": wins_snap,
                    "recent_scores": scores_snap,
                    "ent_scale": ent_scale,
                    "args": vars(args),
                },
                tmp,
            )
            tmp.rename(out_dir / "policy.pt")  # atomic: no torn ckpt on kill
            # Off-pod durability: push to HF so a fresh pod can resume even
            # after total disk loss. Background thread; failures are logged
            # and ignored.
            if os.environ.get("HF_TOKEN") and update % 100 == 0:

                def _hf_push(path=out_dir / "policy.pt", name=args.name):
                    try:
                        from huggingface_hub import HfApi

                        api = HfApi()
                        api.create_repo("djmango/openfront-rl", exist_ok=True)
                        api.upload_file(
                            path_or_fileobj=str(path),
                            path_in_repo=f"{name}/policy.pt",
                            repo_id="djmango/openfront-rl",
                        )
                    except Exception as e:  # noqa: BLE001
                        print(f"hf sync failed: {e}", flush=True)

                threading.Thread(target=_hf_push, daemon=True).start()

    stop.set()
    with contextlib.suppress(queue.Empty):
        rollout_q.get_nowait()  # unblock the collector if it's mid-put
    vec.close()
    writer.close()
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
