"""PPO over the headless engine with the frozen spatial encoder.

Vectorized: N bridge processes stepped in parallel threads, frozen-AE
encode and policy forward batched across envs on the GPU. Core action set
(expand/attack/boat/build/nuke/diplomacy/retreat); not yet wired:
upgrade_structure, delete_unit, move_warship, cancel_boat.

Logs to TensorBoard (runs/rl/<name>).

Usage:
  uv run python -m rl.ppo --envs 48 --updates 2000 --name ppo_v2 --stage 0
"""

import argparse
import contextlib
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from collections import deque

from rl.curriculum import STAGES, WINDOW
from rl.obs import ACTIONS, collate, encode_grids, load_ae
from rl.policy import N_QUANTITY, QUANTITY_FRACS, Policy
from rl.vec import VecEnv

SUB_KEYS = ["player_slot", "tile_region", "build_type", "nuke_type", "quantity"]
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
    ap.add_argument("--epochs", type=int, default=3)
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
    ap.add_argument("--compile", action="store_true", help="torch.compile the policy")
    args = ap.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"device: {device}, envs: {args.envs}")
    torch.set_float32_matmul_precision("high")

    def amp():
        if device == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    out_dir = Path("runs/rl") / args.name
    writer = SummaryWriter(out_dir)

    vec = VecEnv(args.envs, args.stage, args.max_episode_ticks, args.decision_ticks)
    stage = args.stage
    recent_scores: deque[float] = deque(maxlen=WINDOW)
    recent_wins: deque[float] = deque(maxlen=WINDOW)
    ae = load_ae(args.ckpt, device)
    base_policy = Policy().to(device)
    opt = torch.optim.AdamW(
        base_policy.parameters(), lr=args.lr, fused=(device == "cuda")
    )
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

    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        base_policy.load_state_dict(state["model_state_dict"])
        if "optimizer_state_dict" in state:
            opt.load_state_dict(state["optimizer_state_dict"])
        # Checkpoint stage is authoritative on resume: --stage only seeds
        # fresh runs (supervisors pass a stale --stage on every relaunch).
        if "stage" in state:
            stage = int(state["stage"])
            vec.set_stage(stage)
        # Win-gate evidence survives restarts: without this, every relaunch
        # forced the agent to re-earn the whole advancement window.
        recent_wins.extend(state.get("recent_wins", []))
        recent_scores.extend(state.get("recent_scores", []))
        start_update = int(state.get("update", 0))
        global_step = int(state.get("global_step", 0))
        ent_scale = float(state.get("ent_scale", 1.0))
        print(
            f"resumed from {args.resume}: update {start_update}, "
            f"step {global_step}, stage {stage}, "
            f"win-window {len(recent_wins)}/{WINDOW}"
        )

    def set_lr(current_stage: int) -> float:
        lr = args.lr * args.stage_lr_decay ** current_stage
        for grp in opt.param_groups:
            grp["lr"] = lr
        return lr

    set_lr(stage)

    # Compile after weights load; checkpoints always save base_policy's
    # (unprefixed) state_dict.
    policy = torch.compile(base_policy, dynamic=True) if args.compile else base_policy

    T, N = args.rollout, args.envs
    rng = np.random.default_rng(0)
    episodes_done = 0
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
            if device == "cuda":
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
        quantity_counts = np.zeros(N_QUANTITY)

        def record(t: int, idxs: list[int], pack: tuple, results: list) -> None:
            obs_list, choices, logp, value = pack
            gs = shared["global_step"]
            for j, i in enumerate(idxs):
                obs_buf[t][i] = obs_list[j]
                choice_buf[t][i] = choices[j]
                logp_buf[t, i] = logp[j]
                value_buf[t, i] = value[j]
                action_counts[choices[j]["action"]] += 1
                q = choices[j].get("quantity")
                if q is not None:
                    quantity_counts[q] += 1
                r, d, info = results[j]
                reward_buf[t, i] = r
                done_buf[t, i] = float(d)
                if info is None:
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
            shared["global_step"] += N

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
        return {
            "all_obs": [o for row in obs_buf for o in row],
            "choice_t": choice_t,
            "logp": logp_buf.reshape(-1),
            "adv": adv.reshape(-1),
            "returns": returns.reshape(-1),
            "action_counts": action_counts,
            "quantity_counts": quantity_counts,
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

    for update in range(start_update + 1, args.updates + 1):
        t_wait0 = time.time()
        data = rollout_q.get()
        if isinstance(data, BaseException):
            raise data
        wait_s = time.time() - t_wait0
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
        pl_sum = vl_sum = ent_sum = 0.0
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
                pg_mb = vl_mb = ent_mb = 0.0
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
                        logp, ent, value = policy.evaluate(o_mb, c_mb)
                    logp, ent, value = logp.float(), ent.float(), value.float()
                    ratio = (logp - old_logp[sub_t]).exp()
                    a_mb = adv_t[sub_t]
                    pg = -torch.min(
                        ratio * a_mb,
                        ratio.clamp(1 - args.clip, 1 + args.clip) * a_mb,
                    ).mean()
                    vl = F.mse_loss(value, ret_t[sub_t])
                    ent_m = ent.mean()
                    ((pg + args.vf_coef * vl - ent_coef * ent_m) * w_sub).backward()
                    pg_mb += float(pg.item()) * w_sub
                    vl_mb += float(vl.item()) * w_sub
                    ent_mb += float(ent_m.item()) * w_sub
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
                pl_sum += pg_mb
                vl_sum += vl_mb
                ent_sum += ent_mb
                n_mb += 1

        # Entropy floor controller: nudge the coef scale toward keeping
        # measured entropy above the floor, with hysteresis so it doesn't
        # oscillate. Multiplicative so it composes with the anneal.
        if args.ent_floor > 0 and n_mb:
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
        q_total = data["quantity_counts"].sum()
        if q_total:
            for i, f in enumerate(QUANTITY_FRACS):
                writer.add_scalar(
                    f"actions/quantity_{int(f * 100)}pct",
                    data["quantity_counts"][i] / q_total,
                    global_step,
                )
        print(
            f"update {update:4d}  step {global_step:7d}  eps {episodes_done:4d}  "
            f"stage {stage}  roll-win {roll_win:.2f}  roll-score {roll:.2f}  "
            f"pg {pl_sum / n_mb:+.4f}  vf {vl_sum / n_mb:.4f}  ent {ent_sum / n_mb:.3f}  "
            f"ecoef {ent_coef:.4f}  "
            f"{tps:.0f} game-ticks/s  "
            f"[roll {roll_s:.1f}s upd {upd_s:.1f}s wait {wait_s:.1f}s stall {stall_s:.1f}s]",
            flush=True,
        )

        if args.eval_every and update % args.eval_every == 0:
            # Deployment-style number: fixed seeds, greedy actions. This is
            # the curve to compare local replay sessions against (the
            # rolling train win rate is exploration-sampled and its window
            # resets on ops churn).
            t_eval = time.time()
            ev = run_eval(
                policy, ae, device, stage, args.eval_episodes, args.max_episode_ticks
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

        if (
            update % 10 == 0
            or update == args.updates
            or shared.pop("ckpt_advance", False)
        ):
            tmp = out_dir / "policy.pt.tmp"
            with win_lock:
                wins_snap = list(recent_wins)
                scores_snap = list(recent_scores)
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


if __name__ == "__main__":
    main()
