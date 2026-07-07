# OpenFront RL eval showcase: latest checkpoint replay + MODEL overlay.
# Serves the real game client at :8086 with archive API under /archive.

FROM node:22-bookworm AS openfront-deps
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY openfront/package.json openfront/package-lock.json ./openfront/
RUN cd openfront && npm ci --ignore-scripts

FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS py-deps
WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev \
    && uv pip install --index-url https://download.pytorch.org/whl/cu124 'torch>=2.12.0'

FROM node:22-bookworm
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates curl caddy ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=py-deps /build/.venv /app/.venv
COPY --from=py-deps /usr/local/bin/python3.12 /usr/local/bin/python3.12
COPY --from=py-deps /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=py-deps /usr/local/lib/libpython3.12.so.1.0 /usr/local/lib/
RUN ln -sf /usr/local/bin/python3.12 /usr/local/bin/python3
COPY --from=openfront-deps /build/openfront/node_modules /app/openfront/node_modules

COPY . /app
# openfront/.git is dockerignored; apply the replay patch from the repo root.
RUN git apply --directory=openfront patches/client-replay-tooling.patch

ENV PATH="/app/.venv/bin:/app/openfront/node_modules/.bin:${PATH}"
ENV PYTHONPATH=/app
ENV DATA_DIR=/data
ENV RUN_NAME=ppo_v4
ENV STAGE=4
ENV SEED=showcase0
ENV SHOWCASE_SEEDS=showcase0,showcase1,showcase2
ENV CLIP_MAX_SEC=90
ENV REFRESH_HOURS=6
ENV AE_CKPT=runs/ae_v31_d8c32/ae_v3.pt

RUN mkdir -p /data/records /data/policy /data/clips \
    && /app/.venv/bin/playwright install-deps chromium \
    && /app/.venv/bin/playwright install chromium \
    && chmod +x /app/docker/entrypoint.sh

EXPOSE 8086
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=5 \
  CMD curl -sf http://127.0.0.1:8086/status || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
