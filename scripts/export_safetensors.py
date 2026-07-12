"""Export SpatialAE *encoder* weights to safetensors for `oftrain`.

PPO only needs `encode()` (owner_emb + enc_stem + enc_fuse). Decoder
weights are dropped so the Rust VarStore can `load()` a file whose keys
match the encoder-only module tree 1:1 (tch uses '.' path separators,
same as PyTorch state_dict).

Usage:
  python scripts/export_safetensors.py \\
      --ae runs/ae_v31_d8c32/ae_v3.pt \\
      --out weights/ae/ae_v31_d8c32.encoder.safetensors

  python scripts/export_safetensors.py \\
      --ae runs/ae_v31_d16c32/ae_v3.pt \\
      --expected-down 16 \\
      --out weights/ae/ae_v31_d16c32.encoder.safetensors
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import save_file

from rl.obs import COARSE_REGION, LATENT_C, REGION, load_ae

ENCODER_PREFIXES = ("owner_emb.", "enc_stem.", "enc_fuse.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ae", required=True, help="path to ae_v3.pt checkpoint")
    ap.add_argument("--out", required=True, help="output .safetensors path")
    ap.add_argument(
        "--expected-down",
        type=int,
        default=REGION,
        help=f"latent_down to assert (default {REGION}; use {COARSE_REGION} for coarse)",
    )
    ap.add_argument(
        "--expected-c",
        type=int,
        default=LATENT_C,
        help=f"latent_c to assert (default {LATENT_C})",
    )
    ap.add_argument(
        "--meta-out",
        default=None,
        help="optional JSON sidecar with latent_c/latent_down/keys",
    )
    args = ap.parse_args()

    ae = load_ae(args.ae, "cpu", expected_down=args.expected_down, expected_c=args.expected_c)
    sd = ae.state_dict()
    enc = {k: v.contiguous().cpu() for k, v in sd.items() if k.startswith(ENCODER_PREFIXES)}
    if not enc:
        raise SystemExit(f"no encoder tensors found in {args.ae}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "format": "spatial_ae_encoder_v3",
        "latent_c": str(ae.latent_c),
        "latent_down": str(ae.latent_down),
        "terrain_cond": str(ae.terrain_cond),
        "upsample_decoder": str(ae.upsample_decoder),
        "source": str(Path(args.ae).resolve()),
    }
    save_file(enc, str(out), metadata=meta)

    meta_full = {
        **{k: (int(v) if k in ("latent_c", "latent_down") else v == "True" if k in ("terrain_cond", "upsample_decoder") else v)
           for k, v in meta.items() if k != "format"},
        "format": meta["format"],
        "keys": sorted(enc.keys()),
        "num_tensors": len(enc),
    }
    # Keep types clean for the sidecar JSON.
    meta_full["latent_c"] = ae.latent_c
    meta_full["latent_down"] = ae.latent_down
    meta_full["terrain_cond"] = ae.terrain_cond
    meta_full["upsample_decoder"] = ae.upsample_decoder
    meta_full["source"] = meta["source"]

    meta_path = Path(args.meta_out) if args.meta_out else out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta_full, indent=2) + "\n")
    print(f"wrote {len(enc)} encoder tensors -> {out}")
    print(f"meta -> {meta_path} (latent_c={ae.latent_c} latent_down={ae.latent_down})")


if __name__ == "__main__":
    main()
