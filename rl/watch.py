"""Watch the trained agent play: runs one episode locally, renders a video
(WebM via ffmpeg, or GIF by extension), and saves an engine GameRecord that
the real OpenFront client can replay (see scripts/serve_replay.py).

Usage:
  uv run python -m rl.watch --policy /tmp/policy.pt --stage 3 \
      --out replay.webm --record records-rl/replay.json
"""

import argparse
import colorsys
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rl.curriculum import STAGES
from rl.env import OpenFrontEnv
from rl.obs import ObsBuilder, encode_grids, load_ae
from rl.policy import Policy
from rl.ppo import OBS_KEYS
from rl.ppo_translate import IntentTranslator, my_tiles, spawn_randomly

AGENT_RGB = (60, 255, 60)
WATER_RGB = (18, 26, 48)
LAND_RGB = (72, 66, 60)


def palette(me_slot: int, n: int = 256) -> np.ndarray:
    pal = np.zeros((n, 3), dtype=np.uint8)
    pal[0] = LAND_RGB
    for s in range(1, n):
        h = (s * 0.61803) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.75)
        pal[s] = (int(r * 255), int(g * 255), int(b * 255))
    pal[me_slot] = AGENT_RGB
    return pal


def render(env: OpenFrontEnv, builder: ObsBuilder, obs: dict, pal: np.ndarray, scale: int) -> Image.Image:
    lut = builder.lut if builder.lut is not None else np.zeros(4096, dtype=np.int64)
    owners = lut[obs["owners"]]
    land = (env.terrain >> 7) & 1
    img = pal[owners]
    img[(owners == 0) & (land == 0)] = WATER_RGB
    im = Image.fromarray(img)
    if scale != 1:
        im = im.resize((im.width // scale, im.height // scale), Image.NEAREST)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--ckpt", default="runs/ae_v3/ae_v3.pt")
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--map", default=None, help="override map (default: first in stage pool)")
    ap.add_argument("--out", default="replay.webm", help=".webm (ffmpeg) or .gif")
    ap.add_argument("--record", default=None, help="also save engine GameRecord JSON")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--frame-every", type=int, default=2, help="render every Nth decision")
    ap.add_argument("--scale", type=int, default=1, help="downscale factor (1 = native)")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seed", default="watch0")
    args = ap.parse_args()

    st = STAGES[args.stage]
    map_name = args.map or st.maps[0]
    print(f"stage {args.stage}: {map_name}, {st.bots} bots, {st.difficulty}")

    device = "cpu"
    ae = load_ae(args.ckpt, device)
    policy = Policy()
    state = torch.load(args.policy, map_location="cpu", weights_only=False)
    policy.load_state_dict(state["model_state_dict"])
    policy.eval()
    print(f"policy from update {state.get('update', '?')}, stage {state.get('stage', '?')}")

    env = OpenFrontEnv()
    builder = ObsBuilder()
    obs = env.reset(map_name, seed=args.seed, bots=st.bots, difficulty=st.difficulty)
    builder.start_game(env.terrain)
    rng = np.random.default_rng(0)
    obs = spawn_randomly(env, rng)
    translator = IntentTranslator(env, builder)
    builder._slot_lut(obs["entities"]["players"])  # build LUT before first render
    me_slot = int(builder.lut[obs["me"]]) if obs["me"] >= 0 else 1
    pal = palette(me_slot)

    frames = [render(env, builder, obs, pal, args.scale)]
    for step in range(args.max_steps):
        raw = builder.prepare(obs)
        o = encode_grids(ae, [raw], device)[0]
        ot = {k: torch.from_numpy(o[k])[None] for k in OBS_KEYS}
        choices, _, _ = policy.act(ot)
        obs = env.step(translator.translate(choices[0], obs), ticks=10)

        if step % args.frame_every == 0:
            frames.append(render(env, builder, obs, pal, args.scale))
        if step % 100 == 0:
            print(f"step {step}, tick {obs['tick']}, my tiles {my_tiles(obs)}, alive {obs['alive']}")
        if not obs["alive"] or obs["winner"] is not None:
            frames.append(render(env, builder, obs, pal, args.scale))
            print(f"episode over at tick {obs['tick']}: alive={obs['alive']}, winner={obs['winner']}")
            break

    if args.record:
        info = env.save_record(str(Path(args.record).resolve()))
        print(f"game record: {info['saved']} (gameID {info['gameID']}, {info['turns']} turns)")
    env.close()

    if args.out.endswith(".gif"):
        frames[0].save(
            args.out,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // args.fps,
            loop=0,
            optimize=True,
        )
    else:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise SystemExit("ffmpeg not found; use a .gif output or install ffmpeg")
        with tempfile.TemporaryDirectory() as td:
            for i, im in enumerate(frames):
                im.save(f"{td}/f{i:05d}.png")
            # yuv420p requires even dims; pad by one pixel if needed.
            subprocess.run(
                [
                    ffmpeg, "-y", "-framerate", str(args.fps),
                    "-i", f"{td}/f%05d.png",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30",
                    "-pix_fmt", "yuv420p",
                    args.out,
                ],
                check=True,
                capture_output=True,
            )
    print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
