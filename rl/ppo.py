"""PPO over the headless engine with the frozen spatial encoder.

Vectorized: N bridge processes stepped in parallel threads, frozen-AE
encode and policy forward batched across envs on the GPU. Core action set
(expand/attack/boat/build/nuke/diplomacy/retreat); not yet wired:
upgrade_structure, delete_unit, move_warship, cancel_boat.

Logs to TensorBoard (runs/rl/<name>).

Usage:
  uv run python -m rl.ppo --map Onion --envs 16 --updates 500 --name ppo_v1
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from rl.obs import ACTIONS, encode_grids, load_ae
from rl.policy import Policy
from rl.vec import VecEnv

SUB_KEYS = ["player_slot", "tile_region", "build_type", "nuke_type", "quantity"]
OBS_KEYS = [
    "grid", "players", "pmask", "scalars",
    "legal_actions", "legal_ptarget", "legal_build", "legal_nuke",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="Onion")
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
    ap.add_argument("--decision-ticks", type=int, default=10)
    args = ap.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"device: {device}, envs: {args.envs}")

    out_dir = Path("runs/rl") / args.name
    writer = SummaryWriter(out_dir)

    vec = VecEnv(args.envs, args.map, args.max_episode_ticks, args.decision_ticks)
    ae = load_ae(args.ckpt, device)
    policy = Policy().to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    T, N = args.rollout, args.envs
    rng = np.random.default_rng(0)
    global_step = 0
    episodes_done = 0
    t0 = time.time()

    for update in range(1, args.updates + 1):
        obs_buf: list[list[dict]] = []
        choice_buf: list[list[dict]] = []
        logp_buf = np.zeros((T, N), dtype=np.float32)
        value_buf = np.zeros((T, N), dtype=np.float32)
        reward_buf = np.zeros((T, N), dtype=np.float32)
        done_buf = np.zeros((T, N), dtype=np.float32)
        action_counts = np.zeros(len(ACTIONS))

        for t in range(T):
            raws = vec.prepare()
            obs_list = encode_grids(ae, raws, device)
            ot = {
                k: torch.from_numpy(np.stack([o[k] for o in obs_list])).to(device)
                for k in OBS_KEYS
            }
            choices, logp, value = policy.act(ot)
            for c in choices:
                action_counts[c["action"]] += 1

            results = vec.apply(choices)
            for i, (r, d, info) in enumerate(results):
                reward_buf[t, i] = r
                done_buf[t, i] = float(d)
                if info is not None:
                    episodes_done += 1
                    writer.add_scalar("episode/reward", info["reward"], global_step)
                    writer.add_scalar("episode/length", info["length"], global_step)
                    writer.add_scalar("episode/final_tiles", info["final_tiles"], global_step)
                    writer.add_scalar("episode/final_tick", info["final_tick"], global_step)

            obs_buf.append(obs_list)
            choice_buf.append(choices)
            logp_buf[t] = logp
            value_buf[t] = value
            global_step += N

        # Bootstrap values and GAE per env.
        raws = vec.prepare()
        obs_last = encode_grids(ae, raws, device)
        with torch.no_grad():
            ot = {
                k: torch.from_numpy(np.stack([o[k] for o in obs_last])).to(device)
                for k in OBS_KEYS
            }
            last_value = policy.forward(ot)["value"].cpu().numpy()
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

        all_obs = [o for row in obs_buf for o in row]
        all_choice = [c for row in choice_buf for c in row]
        obs_t = {
            k: torch.from_numpy(np.stack([o[k] for o in all_obs])).to(device)
            for k in OBS_KEYS
        }
        choice_t = {
            "action": torch.tensor([c["action"] for c in all_choice], device=device)
        }
        for k in SUB_KEYS:
            choice_t[k] = torch.tensor(
                [c.get(k, -1) for c in all_choice], device=device
            )

        B_total = T * N
        idx = np.arange(B_total)
        pl_sum = vl_sum = ent_sum = 0.0
        n_mb = 0
        for _ in range(args.epochs):
            rng.shuffle(idx)
            for mb in np.split(idx, max(1, B_total // args.minibatch)):
                mbt = torch.from_numpy(mb).to(device)
                o_mb = {k: v[mbt] for k, v in obs_t.items()}
                c_mb = {k: v[mbt] for k, v in choice_t.items()}
                logp, ent, value = policy.evaluate(o_mb, c_mb)
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

        tps = global_step * args.decision_ticks / (time.time() - t0)
        writer.add_scalar("loss/policy", pl_sum / n_mb, global_step)
        writer.add_scalar("loss/value", vl_sum / n_mb, global_step)
        writer.add_scalar("loss/entropy", ent_sum / n_mb, global_step)
        writer.add_scalar("perf/game_ticks_per_s", tps, global_step)
        writer.add_scalar("perf/episodes_done", episodes_done, global_step)
        for i, a in enumerate(ACTIONS):
            writer.add_scalar(f"actions/{a}", action_counts[i] / (T * N), global_step)
        print(
            f"update {update:4d}  step {global_step:7d}  eps {episodes_done:4d}  "
            f"pg {pl_sum / n_mb:+.4f}  vf {vl_sum / n_mb:.4f}  ent {ent_sum / n_mb:.3f}  "
            f"{tps:.0f} game-ticks/s",
            flush=True,
        )

        if update % 10 == 0 or update == args.updates:
            torch.save(
                {"model_state_dict": policy.state_dict(), "args": vars(args)},
                out_dir / "policy.pt",
            )

    vec.close()
    writer.close()


if __name__ == "__main__":
    main()
