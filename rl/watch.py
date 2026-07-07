"""Watch the trained agent play: runs one episode locally, renders a video
(WebM via ffmpeg, or GIF by extension), and saves an engine GameRecord that
the real OpenFront client can replay (see scripts/serve_replay.py).

Defaults: video to replays/<policy-run>_s<stage>_<seed>.webm, record to
records-rl/<same-name>.json. Frames stream to a temp dir as they render,
so memory stays flat even on big maps / long episodes.

Usage:
  uv run python -m rl.watch --policy /tmp/policy.pt --stage 3
"""

import argparse
import colorsys
import json
import shutil
import subprocess
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from rl.curriculum import GW_MAX, STAGES
from rl.env import OpenFrontEnv
from rl.obs import (
    ACTIONS,
    BUILD_TYPES,
    NUKE_TYPES,
    REGION,
    ObsBuilder,
    encode_grids,
    load_ae,
)
from rl.policy import QUANTITY_FRACS, Policy
from rl.ppo import OBS_KEYS
from rl.ppo_translate import IntentTranslator, my_tiles, spawn_randomly

AGENT_RGB = (60, 255, 60)
WATER_RGB = (18, 26, 48)
LAND_RGB = (72, 66, 60)
PANEL_W = 260
PANEL_BG = (12, 14, 22)
TEXT_RGB = (210, 214, 224)
DIM_RGB = (120, 126, 140)
BAR_RGB = (70, 110, 190)
BAR_HI_RGB = (90, 220, 90)
MARK_RGB = (255, 80, 80)


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
    builder._last_owners = owners  # for target-centroid markers
    im = Image.fromarray(img)
    if scale != 1:
        im = im.resize((im.width // scale, im.height // scale), Image.NEAREST)
    return im


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        try:
            return ImageFont.load_default(size)
        except TypeError:
            return ImageFont.load_default()


def describe(choice: dict, obs: dict) -> str:
    """One-line human description of a policy choice."""
    name = ACTIONS[choice["action"]]
    parts = [name]
    if "player_slot" in choice:
        parts.append(f"-> P{choice['player_slot']}")
    if "tile_region" in choice:
        gy, gx = divmod(choice["tile_region"], GW_MAX)
        parts.append(f"@({gx},{gy})")
    if "build_type" in choice:
        parts.append(BUILD_TYPES[choice["build_type"]])
    if "nuke_type" in choice:
        parts.append(NUKE_TYPES[choice["nuke_type"]])
    if "quantity" in choice:
        frac = QUANTITY_FRACS[choice["quantity"]]
        troops = int(obs["legal"]["actions"].get("troops", 0) * frac)
        parts.append(f"{int(frac * 100)}% ({troops:,})")
    return " ".join(parts)


def draw_markers(
    im: Image.Image, choice: dict, builder: ObsBuilder, scale: int
) -> None:
    """Mark what the agent is acting on: crosshair on the tile-region
    target, ring on the attack target's territory centroid."""
    d = ImageDraw.Draw(im)
    if "tile_region" in choice:
        gy, gx = divmod(choice["tile_region"], GW_MAX)
        x = (gx * REGION + REGION // 2) / scale
        y = (gy * REGION + REGION // 2) / scale
        r = max(6, 16 // scale)
        d.line([(x - r, y), (x + r, y)], fill=MARK_RGB, width=2)
        d.line([(x, y - r), (x, y + r)], fill=MARK_RGB, width=2)
        d.rectangle(
            [
                gx * REGION / scale, gy * REGION / scale,
                (gx + 1) * REGION / scale, (gy + 1) * REGION / scale,
            ],
            outline=MARK_RGB,
        )
    if "player_slot" in choice and builder.lut is not None:
        # centroid of the target's territory on the (unscaled) owner grid
        slot = choice["player_slot"]
        owners = getattr(builder, "_last_owners", None)
        if owners is not None:
            ys, xs = np.nonzero(owners == slot)
            if len(ys):
                x, y = float(xs.mean()) / scale, float(ys.mean()) / scale
                r = max(8, 24 // scale)
                d.ellipse([x - r, y - r, x + r, y + r], outline=MARK_RGB, width=2)


def compose(
    map_im: Image.Image,
    obs: dict,
    choice: dict | None,
    history: deque,
    step: int,
) -> Image.Image:
    """Map frame + debug side panel (state, chosen action, action probs)."""
    H = max(map_im.height, 460)
    out = Image.new("RGB", (map_im.width + PANEL_W, H), PANEL_BG)
    out.paste(map_im, (0, (H - map_im.height) // 2))
    d = ImageDraw.Draw(out)
    f = _font(13)
    fs = _font(11)
    x0 = map_im.width + 10
    y = 8

    me = next(
        (p for p in obs["entities"]["players"] if p["id"] == obs["me"]), None
    )
    d.text((x0, y), f"tick {obs['tick']}  step {step}", font=f, fill=TEXT_RGB)
    y += 20
    if me:
        d.text(
            (x0, y),
            f"tiles {me['tiles']:,}  troops {int(me['troops']):,}",
            font=fs, fill=TEXT_RGB,
        )
        y += 16
        d.text((x0, y), f"gold {int(float(me['gold'])):,}", font=fs, fill=TEXT_RGB)
        y += 16
    dbg = (choice or {}).get("debug")
    if dbg is not None:
        d.text((x0, y), f"value {dbg['value']:+.2f}", font=fs, fill=TEXT_RGB)
        y += 16
    y += 6

    desc = (choice or {}).get("_desc")
    if desc:
        d.text((x0, y), desc, font=f, fill=BAR_HI_RGB)
    else:
        d.text((x0, y), "spawn phase", font=f, fill=DIM_RGB)
    y += 24

    # Action-probability bars.
    if dbg is not None:
        probs = dbg["action_probs"]
        bar_x = x0 + 92
        bar_w = PANEL_W - 112
        for i, name in enumerate(ACTIONS):
            chosen = choice is not None and choice["action"] == i
            col = BAR_HI_RGB if chosen else (TEXT_RGB if probs[i] > 0.01 else DIM_RGB)
            d.text((x0, y), name[:13], font=fs, fill=col)
            w = int(bar_w * float(probs[i]))
            d.rectangle([bar_x, y + 2, bar_x + bar_w, y + 10], outline=(40, 44, 58))
            if w > 0:
                d.rectangle(
                    [bar_x, y + 2, bar_x + w, y + 10],
                    fill=BAR_HI_RGB if chosen else BAR_RGB,
                )
            y += 15
        y += 6

    d.text((x0, y), "recent:", font=fs, fill=DIM_RGB)
    y += 15
    for line in list(history)[-8:]:
        d.text((x0, y), line, font=fs, fill=DIM_RGB)
        y += 14
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--ckpt", default="runs/ae_v31_d8c32/ae_v3.pt")
    ap.add_argument("--stage", type=int, default=3)
    ap.add_argument("--map", default=None, help="override map (default: first in stage pool)")
    ap.add_argument("--out", default=None,
                    help=".webm (ffmpeg) or .gif; default replays/<run>_s<stage>_<seed>.webm")
    ap.add_argument("--record", default=None,
                    help="engine GameRecord JSON; default records-rl/<run>_s<stage>_<seed>.json")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--frame-every", type=int, default=2, help="render every Nth decision")
    ap.add_argument("--scale", type=int, default=1, help="downscale factor (1 = native)")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--seed", default="watch0")
    ap.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, default=True,
        help="side panel with the agent's action, probs, and value (--no-debug to disable)",
    )
    args = ap.parse_args()

    run = Path(args.policy).parent.name or "policy"
    base = f"{run}_s{args.stage}_{args.seed}"
    if args.out is None:
        args.out = f"replays/{base}.webm"
    if args.record is None:
        args.record = f"records-rl/{base}.json"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.record).parent.mkdir(parents=True, exist_ok=True)

    st = STAGES[args.stage]
    map_name = args.map or st.maps[0]
    print(
        f"stage {args.stage}: {map_name}, nations={st.nations}, "
        f"{st.bots} bots, {st.difficulty}"
    )

    device = "cpu"
    ae = load_ae(args.ckpt, device)
    policy = Policy()
    state = torch.load(args.policy, map_location="cpu", weights_only=False)
    policy.load_state_dict(state["model_state_dict"])
    policy.eval()
    print(f"policy from update {state.get('update', '?')}, stage {state.get('stage', '?')}")

    env = OpenFrontEnv()
    builder = ObsBuilder()
    obs = env.reset(
        map_name, seed=args.seed, bots=st.bots, difficulty=st.difficulty,
        nations=st.nations,
    )
    builder.start_game(env.terrain)
    rng = np.random.default_rng(0)
    translator = IntentTranslator(env, builder)
    # v4: the policy spawns itself (spawn action + tile head); fall back to
    # a random spawn only if the phase stalls.
    for _ in range(8):
        if not obs["spawnPhase"]:
            break
        raw = builder.prepare(obs)
        o = encode_grids(ae, [raw], device)[0]
        ot = {k: torch.from_numpy(o[k])[None] for k in OBS_KEYS}
        choices, _, _ = policy.act(ot)
        obs = env.step(translator.translate(choices[0], obs), ticks=10)
    if obs["spawnPhase"]:
        obs = spawn_randomly(env, rng)
    builder._slot_lut(obs["entities"]["players"])  # freeze LUT post-spawn
    me_slot = int(builder.lut[obs["me"]]) if obs["me"] >= 0 else 1
    pal = palette(me_slot)

    history: deque[str] = deque(maxlen=8)
    debug_log: list[dict] = []  # per-decision sidecar for the client overlay

    def frame(choice: dict | None, step: int) -> Image.Image:
        im = render(env, builder, obs, pal, args.scale)
        if not args.debug:
            return im
        if choice is not None:
            draw_markers(im, choice, builder, args.scale)
        return compose(im, obs, choice, history, step)

    # Frames go straight to disk: holding a long episode of full-map PIL
    # images in memory OOMs on big maps.
    frame_dir = Path(tempfile.mkdtemp(prefix="watch_frames_"))
    n_frames = 0

    def emit(im: Image.Image) -> None:
        nonlocal n_frames
        im.save(frame_dir / f"f{n_frames:05d}.png")
        n_frames += 1

    emit(frame(None, 0))
    for step in range(args.max_steps):
        raw = builder.prepare(obs)
        o = encode_grids(ae, [raw], device)[0]
        ot = {k: torch.from_numpy(o[k])[None] for k in OBS_KEYS}
        choices, _, _ = policy.act(ot, debug=args.debug)
        choice = choices[0]
        choice["_desc"] = describe(choice, obs)  # pre-step legality/troops
        if ACTIONS[choice["action"]] != "noop":
            history.append(f"t{obs['tick']:>5} {choice['_desc']}"[:40])
        if args.record:
            me = next(
                (p for p in obs["entities"]["players"] if p["id"] == obs["me"]),
                None,
            )
            entry = {
                "tick": obs["tick"],
                "desc": choice["_desc"],
                "action": ACTIONS[choice["action"]],
                "tiles": me["tiles"] if me else 0,
                "troops": int(me["troops"]) if me else 0,
            }
            dbg = choice.get("debug")
            if dbg is not None:
                entry["value"] = round(float(dbg["value"]), 3)
                entry["probs"] = [round(float(p), 4) for p in dbg["action_probs"]]
            debug_log.append(entry)
        obs = env.step(translator.translate(choice, obs), ticks=10)

        if step % args.frame_every == 0:
            emit(frame(choice, step))
        if step % 100 == 0:
            print(
                f"step {step}, tick {obs['tick']}, my tiles {my_tiles(obs)}, alive {obs['alive']}",
                flush=True,
            )
        if not obs["alive"] or obs["winner"] is not None:
            emit(frame(choice, step))
            print(
                f"episode over at tick {obs['tick']}: alive={obs['alive']}, winner={obs['winner']}",
                flush=True,
            )
            break

    if args.record:
        info = env.save_record(str(Path(args.record).resolve()))
        print(f"game record: {info['saved']} (gameID {info['gameID']}, {info['turns']} turns)")
        sidecar = Path(args.record).with_suffix(".debug.json")
        sidecar.write_text(json.dumps({"actions": ACTIONS, "log": debug_log}))
        print(f"debug sidecar: {sidecar} ({len(debug_log)} decisions)")
    env.close()

    if args.out.endswith(".gif"):
        frames = [
            Image.open(p) for p in sorted(frame_dir.glob("f*.png"))
        ]
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
        # yuv420p requires even dims; pad by one pixel if needed.
        subprocess.run(
            [
                ffmpeg, "-y", "-framerate", str(args.fps),
                "-i", f"{frame_dir}/f%05d.png",
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30",
                "-pix_fmt", "yuv420p",
                args.out,
            ],
            check=True,
            capture_output=True,
        )
    shutil.rmtree(frame_dir, ignore_errors=True)
    print(f"wrote {args.out} ({n_frames} frames)", flush=True)


if __name__ == "__main__":
    main()
