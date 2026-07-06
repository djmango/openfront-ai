"""PPO over the headless engine with the frozen spatial encoder.

v1 scaffold: single env, core action set (expand/attack/boat/build/nuke/
diplomacy/retreat). Not yet wired: upgrade_structure, delete_unit,
move_warship, cancel_boat, alliance_extension, emoji/chat.

Logs to TensorBoard (runs/rl/<name>): reward, episode length, tiles owned,
head entropies, losses, action distribution.

Usage:
  uv run python -m rl.ppo --map Onion --updates 200 --name ppo_v1
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from rl.env import OpenFrontEnv
from rl.obs import ACTIONS, BUILD_TYPES, NUKE_TYPES, ObsBuilder
from rl.policy import NEEDS_QUANTITY, QUANTITY_FRACS, Policy

SUB_KEYS = ["player_slot", "tile_region", "build_type", "nuke_type", "quantity"]


def to_tensors(o: dict, device: str) -> dict:
    out = {}
    for k, v in o.items():
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v)[None].to(device)
    return out


class IntentTranslator:
    """choice dict -> engine intent JSON. Region pointers snap to a land
    tile inside the 16x16 region (engine validates the rest)."""

    def __init__(self, env: OpenFrontEnv, builder: ObsBuilder):
        self.env = env
        self.builder = builder
        land = (env.terrain >> 7) & 1
        self.land = land[: builder.h16, : builder.w16]
        self.gw = builder.w16 // 16
        self.rng = np.random.default_rng(0)

    def region_tile(self, region: int) -> int | None:
        gy, gx = divmod(region, self.gw)
        block = self.land[gy * 16 : (gy + 1) * 16, gx * 16 : (gx + 1) * 16]
        ys, xs = np.nonzero(block)
        if len(ys) == 0:
            return None
        i = self.rng.integers(len(ys))
        y, x = gy * 16 + int(ys[i]), gx * 16 + int(xs[i])
        return y * self.env.width + x

    def slot_to_pid(self, slot: int, obs: dict) -> str | None:
        """Slot -> persistent player ID (intents reference players by pid)."""
        lut = self.builder.lut
        small_ids = np.nonzero(lut == slot)[0]
        if not len(small_ids):
            return None
        small = int(small_ids[0])
        for p in obs["entities"]["players"]:
            if p["id"] == small:
                return p["pid"]
        return None

    def translate(self, choice: dict, obs: dict) -> list[dict]:
        name = ACTIONS[choice["action"]]
        legal = obs["legal"]["actions"]
        troops = legal.get("troops", 0)
        frac = QUANTITY_FRACS[choice.get("quantity", 2)]

        if name == "noop":
            return []
        if name == "expand":
            return [{"type": "attack", "targetID": None, "troops": int(troops * frac)}]
        if name == "attack":
            pid = self.slot_to_pid(choice["player_slot"], obs)
            if pid is None:
                return []
            return [{"type": "attack", "targetID": pid, "troops": int(troops * frac)}]
        if name == "boat":
            tile = self.region_tile(choice["tile_region"])
            if tile is None:
                return []
            return [{"type": "boat", "dst": tile, "troops": int(troops * frac)}]
        if name == "build":
            tile = self.region_tile(choice["tile_region"])
            if tile is None:
                return []
            return [{"type": "build_unit", "unit": BUILD_TYPES[choice["build_type"]], "tile": tile}]
        if name == "launch_nuke":
            tile = self.region_tile(choice["tile_region"])
            if tile is None:
                return []
            return [{"type": "build_unit", "unit": NUKE_TYPES[choice["nuke_type"]], "tile": tile}]
        if name == "retreat":
            attacks = legal.get("attacks", [])
            return [{"type": "cancel_attack", "attackID": attacks[-1]}] if attacks else []

        pid = self.slot_to_pid(choice.get("player_slot", -1), obs)
        if pid is None:
            return []
        if name == "alliance_request":
            return [{"type": "allianceRequest", "recipient": pid}]
        if name == "alliance_reject":
            return [{"type": "allianceReject", "requestor": pid}]
        if name == "break_alliance":
            return [{"type": "breakAlliance", "recipient": pid}]
        if name == "donate_gold":
            gold = int(float(legal.get("gold", 0)) * frac)
            return [{"type": "donate_gold", "recipient": pid, "gold": gold}]
        if name == "donate_troops":
            return [{"type": "donate_troops", "recipient": pid, "troops": int(troops * frac)}]
        if name == "embargo":
            return [{"type": "embargo", "targetID": pid, "action": "start"}]
        return []


def my_tiles(obs: dict) -> int:
    for p in obs["entities"]["players"]:
        if p["id"] == obs["me"]:
            return p["tiles"]
    return 0


def spawn_randomly(env: OpenFrontEnv, rng: np.random.Generator) -> dict:
    land = (env.terrain >> 7) & 1
    ys, xs = np.nonzero(land)
    obs = None
    while True:
        i = rng.integers(len(ys))
        tile = int(ys[i]) * env.width + int(xs[i])
        obs = env.step([{"type": "spawn", "tile": tile}], ticks=10)
        if not obs["spawnPhase"]:
            return obs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="Onion")
    ap.add_argument("--ckpt", default="runs/ae_v3/ae_v3.pt")
    ap.add_argument("--name", default="ppo_v1")
    ap.add_argument("--updates", type=int, default=1000)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatch", type=int, default=64)
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
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device: {device}")

    out_dir = Path("runs/rl") / args.name
    writer = SummaryWriter(out_dir)

    env = OpenFrontEnv()
    builder = ObsBuilder(args.ckpt, device="cpu")
    policy = Policy().to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    rng = np.random.default_rng(0)
    episode = 0
    obs = env.reset(args.map, seed=f"ep{episode}")
    builder.start_game(env.terrain)
    obs = spawn_randomly(env, rng)
    translator = IntentTranslator(env, builder)
    land_total = max(1, int(((env.terrain >> 7) & 1).sum()))
    prev_tiles = my_tiles(obs)
    ep_reward, ep_len = 0.0, 0

    global_step = 0
    t0 = time.time()
    for update in range(1, args.updates + 1):
        buf: dict[str, list] = {k: [] for k in ["obs", "choice", "logp", "value", "reward", "done"]}
        action_counts = np.zeros(len(ACTIONS))

        for _ in range(args.rollout):
            o = builder.build(obs)
            ot = to_tensors(o, device)
            choice, extra = policy.act(ot)
            action_counts[choice["action"]] += 1

            intents = translator.translate(choice, obs)
            obs = env.step(intents, ticks=args.decision_ticks)

            tiles = my_tiles(obs)
            reward = (tiles - prev_tiles) / land_total * 10.0
            prev_tiles = tiles
            done = False
            if not obs["alive"]:
                reward -= 5.0
                done = True
            elif obs["winner"] is not None:
                w = obs["winner"]
                won = isinstance(w, list) and len(w) > 1 and w[1] == "Agent"
                reward += 10.0 if won else -2.0
                done = True
            elif obs["tick"] >= args.max_episode_ticks:
                done = True

            buf["obs"].append(o)
            buf["choice"].append(choice)
            buf["logp"].append(extra["logp"])
            buf["value"].append(extra["value"])
            buf["reward"].append(reward)
            buf["done"].append(done)
            ep_reward += reward
            ep_len += 1
            global_step += 1

            if done:
                writer.add_scalar("episode/reward", ep_reward, global_step)
                writer.add_scalar("episode/length", ep_len, global_step)
                writer.add_scalar("episode/final_tiles", tiles, global_step)
                episode += 1
                ep_reward, ep_len = 0.0, 0
                obs = env.reset(args.map, seed=f"ep{episode}")
                builder.start_game(env.terrain)
                obs = spawn_randomly(env, rng)
                translator = IntentTranslator(env, builder)
                prev_tiles = my_tiles(obs)

        # Bootstrap + GAE.
        with torch.no_grad():
            o = builder.build(obs)
            last_value = float(policy.forward(to_tensors(o, device))["value"].item())
        adv = np.zeros(args.rollout, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(args.rollout)):
            next_v = last_value if t == args.rollout - 1 else buf["value"][t + 1]
            nonterminal = 0.0 if buf["done"][t] else 1.0
            delta = buf["reward"][t] + args.gamma * next_v * nonterminal - buf["value"][t]
            last_gae = delta + args.gamma * args.lam * nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + np.asarray(buf["value"], dtype=np.float32)
        adv_t = torch.from_numpy((adv - adv.mean()) / (adv.std() + 1e-8)).to(device)
        ret_t = torch.from_numpy(returns).to(device)
        old_logp = torch.tensor(buf["logp"], device=device)

        # Stack obs and choices.
        keys = [k for k, v in buf["obs"][0].items() if isinstance(v, np.ndarray)]
        obs_t = {
            k: torch.from_numpy(np.stack([o[k] for o in buf["obs"]])).to(device)
            for k in keys
        }
        choice_t = {
            "action": torch.tensor([c["action"] for c in buf["choice"]], device=device)
        }
        for k in SUB_KEYS:
            choice_t[k] = torch.tensor(
                [c.get(k, -1) for c in buf["choice"]], device=device
            )

        idx = np.arange(args.rollout)
        pl_sum = vl_sum = ent_sum = 0.0
        n_mb = 0
        for _ in range(args.epochs):
            rng.shuffle(idx)
            for mb in np.split(idx, max(1, args.rollout // args.minibatch)):
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

        sps = global_step * args.decision_ticks / (time.time() - t0)
        writer.add_scalar("loss/policy", pl_sum / n_mb, global_step)
        writer.add_scalar("loss/value", vl_sum / n_mb, global_step)
        writer.add_scalar("loss/entropy", ent_sum / n_mb, global_step)
        writer.add_scalar("perf/game_ticks_per_s", sps, global_step)
        for i, a in enumerate(ACTIONS):
            writer.add_scalar(f"actions/{a}", action_counts[i] / args.rollout, global_step)
        print(
            f"update {update:4d}  step {global_step:7d}  "
            f"pg {pl_sum / n_mb:+.4f}  vf {vl_sum / n_mb:.4f}  ent {ent_sum / n_mb:.3f}  "
            f"{sps:.0f} game-ticks/s",
            flush=True,
        )

        if update % 20 == 0 or update == args.updates:
            torch.save(
                {"model_state_dict": policy.state_dict(), "args": vars(args)},
                out_dir / "policy.pt",
            )

    env.close()
    writer.close()


if __name__ == "__main__":
    main()
