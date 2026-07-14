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
`SHOWCASE_MAPS` by default), launches `scripts/webbot_launcher.py` (greedy)
for the in-browser ONNX agent, redirects you to the lobby. Only one Play
lobby runs at a time - a second click gets a short busy page.

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
| `RUN_NAME` | `ppo_v81` | HF policy run under `djmango/openfront-rl` |
| `SHOWCASE_MAPS` | curriculum `ALL_MAPS` | Comma-separated map **keys** for Watch clips + random Play |
| `PLAY_MAP` | `random` | Live play map key, or `random` to sample `SHOWCASE_MAPS` |
| `PLAY_BOTS` | `10` | Tribe bots |
| `PLAY_NATIONS` | `1` | Nation opponents |
| `PLAY_START_DELAY` | `30` | Lobby countdown (seconds) |
| `PLAY_GREEDY` | `1` | Pass `--greedy` to webbot (`0` to sample) |
| `STAGE` | `4` | Curriculum stage for replay generation |
| `SHOWCASE_WATCH_STAGE` | (stage) | Stage passed to `oftrain --watch` |

Homelab: [homelab README](https://github.com/djmango/homelab), `openfrontai.skg.gg`.
