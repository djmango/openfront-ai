# syntax=docker/dockerfile:1
# OpenFront RL eval showcase: live agent games + archive replay + MODEL overlay.
# Rust owns HF + HTTP + watch orchestration (ofhub / oftrain --watch);
# Playwright + webbot_launcher remain thin Python subprocesses.

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
COPY openfront.commit \
     patches/client-replay-tooling.patch \
     patches/showcase-spawn-window.patch \
     patches/showcase-webbot-tile.patch \
     patches/showcase-allow-software-gl.patch \
     /tmp/
ARG OPENFRONT_COMMIT
RUN set -eux; \
    commit="${OPENFRONT_COMMIT:-$(tr -d '[:space:]' < /tmp/openfront.commit)}"; \
    mkdir openfront; \
    git -C openfront init; \
    git -C openfront remote add origin https://github.com/djmango/OpenFrontIO.git; \
    git -C openfront fetch --depth 1 origin "${commit}"; \
    git -C openfront checkout FETCH_HEAD; \
    git apply --directory=openfront /tmp/client-replay-tooling.patch; \
    git apply --directory=openfront /tmp/showcase-spawn-window.patch; \
    git apply --directory=openfront /tmp/showcase-webbot-tile.patch; \
    git apply --directory=openfront /tmp/showcase-allow-software-gl.patch

FROM rust:bookworm AS rust-builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config libssl-dev ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
# Link oftrain against the CUDA PyTorch wheel (same as runtime) so showcase
# watch can use the host GPU instead of a CPU-only download-libtorch build.
COPY --from=py-deps /build/.venv /opt/venv
ENV LIBTORCH=/opt/venv/lib/python3.12/site-packages/torch
ENV LIBTORCH_CXX11_ABI=1
ENV LIBTORCH_BYPASS_VERSION_CHECK=1
ENV LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/torch/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib:/opt/venv/lib/python3.12/site-packages/nvidia/cusparselt/lib:/opt/venv/lib/python3.12/site-packages/nvidia/nvshmem/lib
RUN ln -sf libcudart.so.13 /opt/venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcudart.so \
    && ln -sf libnccl.so.2 /opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so
COPY rust ./rust
WORKDIR /build/rust
# ofhub has no libtorch dependency.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/rust/target \
    cargo build --release -p ofhub \
    && mkdir -p /out \
    && cp target/release/ofhf target/release/ofexport target/release/ofshowcase /out/
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/rust/target \
    cargo build --release -p oftrain \
    && cp target/release/oftrain /out/

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
COPY --from=rust-builder /out/ofhf /out/ofexport /out/ofshowcase /out/oftrain /app/rust/target/release/

COPY rl ./rl
COPY bridge ./bridge
COPY scripts ./scripts
COPY docker ./docker
COPY patches ./patches
COPY pyproject.toml uv.lock openfront.commit ./
RUN mkdir -p /app/weights/ae

COPY --from=openfront-src /src/openfront ./openfront
RUN --mount=type=cache,target=/root/.npm \
    cd openfront && npm ci --ignore-scripts

ENV PATH="/app/.venv/bin:/app/openfront/node_modules/.bin:/app/rust/target/release:${PATH}"
ENV PYTHONPATH=/app
ENV DATA_DIR=/data
ENV RUN_NAME=ppo_v81
ENV STAGE=4
ENV SHOWCASE_WATCH_STAGE=4
ENV SEED=showcase0
ENV CLIP_MAX_SEC=90
ENV CLIP_DEVICE_SCALE_FACTOR=1
ENV REFRESH_HOURS=1
ENV LIVE_SHOWCASE=0
ENV AE_CKPT=weights/ae/ae_v31_d8c32.encoder.safetensors
ENV OFTRAIN_BIN=/app/rust/target/release/oftrain
ENV OFSHOWCASE=/app/rust/target/release/ofshowcase

RUN mkdir -p /data/records /data/policy /data/clips /data/replay-spool /app/rust/target/release \
    && /app/.venv/bin/python -m playwright install-deps chromium \
    && /app/.venv/bin/python -m playwright install chromium \
    && chmod +x /app/docker/entrypoint.sh \
    && chmod +x /app/rust/target/release/ofhf \
                 /app/rust/target/release/ofexport \
                 /app/rust/target/release/ofshowcase \
                 /app/rust/target/release/oftrain

EXPOSE 8086
VOLUME ["/data"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=5 \
  CMD curl -sf http://127.0.0.1:8086/status || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
