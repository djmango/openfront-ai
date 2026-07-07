# syntax=docker/dockerfile:1
# OpenFront RL eval showcase: live agent games + archive replay + MODEL overlay.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS py-deps
WORKDIR /build
COPY pyproject.toml uv.lock ./
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
    && uv pip install --index-url https://download.pytorch.org/whl/cu124 'torch>=2.12.0' \
    && .venv/bin/python -m playwright --version

FROM node:22-bookworm AS openfront-src
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY openfront.commit patches/client-replay-tooling.patch /tmp/
ARG OPENFRONT_COMMIT
RUN set -eux; \
    commit="${OPENFRONT_COMMIT:-$(tr -d '[:space:]' < /tmp/openfront.commit)}"; \
    git clone --filter=blob:none https://github.com/openfrontio/OpenFrontIO.git openfront; \
    git -C openfront checkout "${commit}"; \
    git apply --directory=openfront /tmp/client-replay-tooling.patch

FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl ffmpeg caddy gnupg \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_22.x nodistro main" \
        > /etc/apt/sources.list.d/nodesource.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY --from=py-deps /build/.venv /app/.venv

COPY rl ./rl
COPY bridge ./bridge
COPY scripts ./scripts
COPY docker ./docker
COPY patches ./patches
COPY pyproject.toml uv.lock openfront.commit ./

COPY --from=openfront-src /src/openfront ./openfront
RUN --mount=type=cache,target=/root/.npm \
    cd openfront && npm ci --ignore-scripts

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
    && /app/.venv/bin/python -m playwright install-deps chromium \
    && /app/.venv/bin/python -m playwright install chromium \
    && chmod +x /app/docker/entrypoint.sh

EXPOSE 8086
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=5 \
  CMD curl -sf http://127.0.0.1:8086/status || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
