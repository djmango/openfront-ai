# OpenFront eval showcase (Docker)

View-only replay of the latest RL checkpoint by default, with an on-demand
**Play vs Agent** button for 1v1 (+ bots) live games.

## URLs

| Path | What |
|------|------|
| `/` | Landing: Watch or Play |
| `/watch` | Latest checkpoint replay (MODEL overlay) |
| `/play` | Create lobby, agent joins, you join as human |
| `/archive/*` | Replay API for checkpoint games |

## Flow

**Watch (default):** `ofshowcase daemon` keeps the latest `oftrain --watch`
replay on HF policy changes. `/watch` opens it in the real client with the
MODEL overlay.

**Play (on click):** `ofshowcase hub` creates a private lobby (random map from
the curriculum pool), launches `scripts/webbot_launcher.py` (greedy) for the
in-browser ONNX agent, redirects you to the lobby. Only one Play lobby runs
at a time - a second click gets a short busy page.

**Archive:** `ofshowcase archive` serves GameRecords + clips for the client
replay API (`/archive/*`).

## Run locally

```bash
docker build -f docker/Dockerfile -t openfront-eval .
docker run --rm -p 8086:8086 -v openfront-eval-data:/data openfront-eval
```

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `RUN_NAME` | `ppo_v10` | HF policy run under `djmango/openfront-rl` |
| `PLAY_MAP` | `Onion` | Live play map key, or `random` to sample the curriculum pool |
| `PLAY_BOTS` | `10` | Tribe bots |
| `PLAY_NATIONS` | `1` | Nation opponents |
| `PLAY_START_DELAY` | `15` | Lobby countdown (seconds) |
| `PLAY_GREEDY` | `1` | Pass `--greedy` to webbot (`0` to sample) |
| `STAGE` | `27` | Curriculum stage label in showcase state |
| `SHOWCASE_WATCH_STAGE` | (stage) | Stage passed to `oftrain --watch` (V10 schedule) |
| `SHOWCASE_BOTS` | `24` | Watch/replay bot count (matches live Easy ramp) |
| `SHOWCASE_NATIONS` | `4` | Watch/replay nations |
| `SHOWCASE_V10` | `1` | Force `--v10-curriculum` on watch (also auto when `RUN_NAME` contains `v10`) |

Homelab: [homelab README](https://github.com/djmango/homelab), `openfrontai.skg.gg`.
Redeploy: `bash docker/homelab_deploy.sh` on the host (rebuilds image, clears
`hero_clips`, restarts `ofshowcase daemon` so clips regenerate from HF).
