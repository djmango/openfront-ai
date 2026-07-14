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

**Play (on click):** `ofshowcase hub` creates a private lobby (Onion, 1
nation, 10 bots by default), launches `scripts/webbot_launcher.py` for the
in-browser ONNX agent, redirects you to the lobby. Set start delay in the
host modal, wait for AgentRL, then Start Game.

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
| `RUN_NAME` | `ppo_v81` | HF policy run |
| `PLAY_MAP` | `Onion` | Live play map |
| `PLAY_BOTS` | `10` | Tribe bots |
| `PLAY_NATIONS` | `1` | Nation opponents |
| `PLAY_START_DELAY` | `90` | Lobby countdown (seconds) |
| `STAGE` | `4` | Curriculum stage for replay generation |

Homelab: [homelab README](https://github.com/djmango/homelab), `openfrontai.skg.gg`.
