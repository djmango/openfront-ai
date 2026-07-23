#!/usr/bin/env bash
# Download fine + coarse SpatialAE encoder safetensors for `oftrain --ckpt`
# / `--coarse-ckpt`. Prefers pre-exported `.encoder.safetensors` on HF via
# `ofhf`. Optional local ofae full-ckpt filter as a last resort.
#
#   bash scripts/fetch_ae_encoders.sh
#   AE_DIR=weights/ae bash scripts/fetch_ae_encoders.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
AE_DIR="${AE_DIR:-weights/ae}"
mkdir -p "$AE_DIR"

OFHF="${OFHF:-}"
if [ -z "$OFHF" ]; then
  if [ -x "$ROOT/rust/target/release/ofhf" ]; then
    OFHF="$ROOT/rust/target/release/ofhf"
  elif [ -x "$ROOT/rust/target/debug/ofhf" ]; then
    OFHF="$ROOT/rust/target/debug/ofhf"
  fi
fi

if [ -n "$OFHF" ]; then
  "$OFHF" pull-ae --ae-dir "$AE_DIR"
fi

if [ ! -f "$AE_DIR/ae_v32_nostatic_d8c32.encoder.safetensors" ] \
  || [ ! -f "$AE_DIR/ae_v32_nostatic_d16c32.encoder.safetensors" ]; then
  echo "WARN: encoder safetensors missing under $AE_DIR after ofhf pull." >&2
  echo "Train with: cargo run -p ofae -- train --latent-down 8 --out runs/ae_v32_nostatic_d8c32" >&2
  echo "Or download *.encoder.safetensors from djmango/openfront-tile-autoencoder" >&2
  exit 1
fi

echo "AE encoders ready under $AE_DIR"
