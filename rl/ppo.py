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
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from collections import deque

from rl.curriculum import STAGES, WIN_AT, WINDOW
from rl.obs import ACTIONS, collate, encode_grids, load_ae
from rl.policy import Policy
from rl.vec import VecEnv

SUB_KEYS = ["player_slot", "tile_region", "build_type", "nuke_type", "quantity"]
OBS_KEYS = [
    "grid", "grid_valid", "players", "pmask", "scalars",
    "legal_actions", "legal_ptarget", "legal_build", "legal_nuke",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0, help="starting curriculum stage")
    ap.add_argument("--ckpt", default="runs/ae_v3/ae_v3.pt")
    ap.add_argument("--name", default="ppo_v1")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--rollout", type=int, default=32, help="steps per env per update")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.999)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-episode-ticks", type=int, default=15000)
    ap.add_argument("--decision-ticks", type=int, default=10,
                    help="unused for stepping (stages carry their own); kept for compat")
    ap.add_argument("--resume", default=None, help="policy.pt to load before training")
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
        start_update = int(state.get("update", 0))
        global_step = int(state.get("global_step", 0))
        print(
            f"resumed from {args.resume}: update {start_update}, "
            f"step {global_step}, stage {stage}"
        )

    # Compile after weights load; checkpoints always save base_policy's
    # (unprefixed) state_dict.
    policy = torch.compile(base_policy, dynamic=True) if args.compile else base_policy

    T, N = args.rollout, args.envs
    rng = np.random.default_rng(0)
    episodes_done = 0
    t0 = time.time()
    t0_step = global_step

    # Two env groups pipelined: while one group's Node processes step the
    # game, the GPU encodes + acts for the other group.
    half = max(1, N // 2)
    groups = (list(range(half)), list(range(half, N)))

    def act_group(idxs: list[int]) -> tuple:
        obs_list = encode_grids(ae, vec.obs_group(idxs), device)
        ot = {
            k: torch.from_numpy(v).to(device)
            for k, v in collate(obs_list, OBS_KEYS).items()
        }
        with amp():
            choices, logp, value = policy.act(ot)
        return obs_list, choices, logp, value

    for update in range(start_update + 1, args.updates + 1):
        obs_buf: list[list] = [[None] * N for _ in range(T)]
        choice_buf: list[list] = [[None] * N for _ in range(T)]
        logp_buf = np.zeros((T, N), dtype=np.float32)
        value_buf = np.zeros((T, N), dtype=np.float32)
        reward_buf = np.zeros((T, N), dtype=np.float32)
        done_buf = np.zeros((T, N), dtype=np.float32)
        action_counts = np.zeros(len(ACTIONS))

        def record(t: int, idxs: list[int], pack: tuple, results: list) -> None:
            nonlocal episodes_done, stage
            obs_list, choices, logp, value = pack
            for j, i in enumerate(idxs):
                obs_buf[t][i] = obs_list[j]
                choice_buf[t][i] = choices[j]
                logp_buf[t, i] = logp[j]
                value_buf[t, i] = value[j]
                action_counts[choices[j]["action"]] += 1
                r, d, info = results[j]
                reward_buf[t, i] = r
                done_buf[t, i] = float(d)
                if info is None:
                    continue
                episodes_done += 1
                writer.add_scalar("episode/reward", info["reward"], global_step)
                writer.add_scalar("episode/length", info["length"], global_step)
                writer.add_scalar("episode/final_tiles", info["final_tiles"], global_step)
                writer.add_scalar("episode/final_tick", info["final_tick"], global_step)
                writer.add_scalar("episode/place", info["place"], global_step)
                writer.add_scalar("episode/score", info["score"], global_step)
                writer.add_scalar("episode/won", float(info["won"]), global_step)
                writer.add_scalar("curriculum/episode_stage", info["stage"], global_step)
                writer.add_scalar(
                    "curriculum/rehearsal", float(info["rehearsal"]), global_step
                )
                # Only current-stage, non-rehearsal episodes count toward
                # advancement; the gate is win rate, not placement.
                if info["stage"] == stage and not info["rehearsal"]:
                    recent_scores.append(info["score"])
                    recent_wins.append(float(info["won"]))
                if (
                    len(recent_wins) == WINDOW
                    and np.mean(recent_wins) > WIN_AT
                    and stage < len(STAGES) - 1
                ):
                    stage += 1
                    vec.set_stage(stage)
                    recent_scores.clear()
                    recent_wins.clear()
                    st = STAGES[stage]
                    print(
                        f"=== curriculum advance -> stage {stage}: "
                        f"maps={','.join(st.maps)} nations={st.nations} "
                        f"bots={st.bots} {st.difficulty}",
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
            global_step += N

        # Bootstrap values and GAE per env.
        obs_last = encode_grids(ae, vec.obs(), device)
        with torch.no_grad(), amp():
            ot = {
                k: torch.from_numpy(v).to(device)
                for k, v in collate(obs_last, OBS_KEYS).items()
            }
            last_value = policy.forward(ot)["value"].float().cpu().numpy()
        adv = np.zeros((T, N), dtype=np.float32)
        last_gae = np.zeros(N, dtype=np.float32)
        for t in reversed(range(T)):
            next_v = last_value if t == T - 1 else value_buf[t + 1]
            nonterminal = 1.0 - done_buf[t]
            delta = reward_buf[t] + args.gamma * next_v * nonterminal - value_buf[t]
            last_gae = delta + args.gamma * args.lam * nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + value_buf

        flat = adv.reshape(-1)
        adv_t = torch.from_numpy((flat - flat.mean()) / (flat.std() + 1e-8)).to(device)
        ret_t = torch.from_numpy(returns.reshape(-1)).to(device)
        old_logp = torch.from_numpy(logp_buf.reshape(-1)).to(device)

        # Rollout buffer stays on CPU at native grid sizes; each minibatch
        # is collated (padded to its own max grid) and moved to GPU alone.
        all_obs = [o for row in obs_buf for o in row]
        all_choice = [c for row in choice_buf for c in row]
        choice_t = {
            "action": torch.tensor([c["action"] for c in all_choice])
        }
        for k in SUB_KEYS:
            choice_t[k] = torch.tensor([c.get(k, -1) for c in all_choice])

        B_total = T * N
        idx = np.arange(B_total)
        pl_sum = vl_sum = ent_sum = 0.0
        n_mb = 0
        for _ in range(args.epochs):
            rng.shuffle(idx)
            for mb in np.split(idx, max(1, B_total // args.minibatch)):
                mbt = torch.from_numpy(mb)
                o_np = collate([all_obs[i] for i in mb], OBS_KEYS)
                o_mb = {k: torch.from_numpy(v).to(device) for k, v in o_np.items()}
                c_mb = {k: v[mbt].to(device) for k, v in choice_t.items()}
                mbt = mbt.to(device)
                with amp():
                    logp, ent, value = policy.evaluate(o_mb, c_mb)
                logp, ent, value = logp.float(), ent.float(), value.float()
                ratio = (logp - old_logp[mbt]).exp()
                a_mb = adv_t[mbt]
                pg = -torch.min(
                    ratio * a_mb,
                    ratio.clamp(1 - args.clip, 1 + args.clip) * a_mb,
                ).mean()
                vl = F.mse_loss(value, ret_t[mbt])
                loss = pg + args.vf_coef * vl - args.ent_coef * ent.mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                opt.step()
                pl_sum += float(pg.item())
                vl_sum += float(vl.item())
                ent_sum += float(ent.mean().item())
                n_mb += 1

        tps = (global_step - t0_step) * STAGES[stage].decision_ticks / (time.time() - t0)
        writer.add_scalar("loss/policy", pl_sum / n_mb, global_step)
        writer.add_scalar("loss/value", vl_sum / n_mb, global_step)
        writer.add_scalar("loss/entropy", ent_sum / n_mb, global_step)
        writer.add_scalar("perf/game_ticks_per_s", tps, global_step)
        writer.add_scalar("perf/episodes_done", episodes_done, global_step)
        writer.add_scalar("curriculum/stage", stage, global_step)
        writer.add_scalar(
            "curriculum/rolling_score",
            float(np.mean(recent_scores)) if recent_scores else 0.0,
            global_step,
        )
        writer.add_scalar(
            "curriculum/rolling_win",
            float(np.mean(recent_wins)) if recent_wins else 0.0,
            global_step,
        )
        for i, a in enumerate(ACTIONS):
            writer.add_scalar(f"actions/{a}", action_counts[i] / (T * N), global_step)
        roll = float(np.mean(recent_scores)) if recent_scores else 0.0
        roll_win = float(np.mean(recent_wins)) if recent_wins else 0.0
        print(
            f"update {update:4d}  step {global_step:7d}  eps {episodes_done:4d}  "
            f"stage {stage}  roll-win {roll_win:.2f}  roll-score {roll:.2f}  "
            f"pg {pl_sum / n_mb:+.4f}  vf {vl_sum / n_mb:.4f}  ent {ent_sum / n_mb:.3f}  "
            f"{tps:.0f} game-ticks/s",
            flush=True,
        )

        if update % 10 == 0 or update == args.updates:
            tmp = out_dir / "policy.pt.tmp"
            torch.save(
                {
                    "model_state_dict": base_policy.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "stage": stage,
                    "update": update,
                    "global_step": global_step,
                    "args": vars(args),
                },
                tmp,
            )
            tmp.rename(out_dir / "policy.pt")  # atomic: no torn ckpt on kill
            # Off-pod durability: push to HF so a fresh pod can resume even
            # after total disk loss. Background thread; failures are logged
            # and ignored.
            if os.environ.get("HF_TOKEN") and update % 100 == 0:
                import threading

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

    vec.close()
    writer.close()


if __name__ == "__main__":
    main()
