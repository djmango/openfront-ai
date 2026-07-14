"""Strict, read-only mapping from current Rust oftrain safetensors to Policy."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch


_GRID = re.compile(r"^(grid_(?:coarse|fine))\.(stem|block\.(\d+)\.(conv[12]))\.(weight|bias)$")
_TF = re.compile(r"^tf\.(\d+)\.(q|k|v|out|ln1|ln2|ff1|ff2)\.(weight|bias)$")


def _python_key(rust_key: str) -> str | None:
    match = _GRID.fullmatch(rust_key)
    if match:
        tower, layer, block, conv, parameter = match.groups()
        prefix = f"{tower}_net"
        if layer == "stem":
            return f"{prefix}.0.{parameter}"
        return f"{prefix}.{int(block) + 2}.{conv}.{parameter}"
    match = _TF.fullmatch(rust_key)
    if match:
        layer, part, parameter = match.groups()
        names = {
            "out": "self_attn.out_proj",
            "ln1": "norm1",
            "ln2": "norm2",
            "ff1": "linear1",
            "ff2": "linear2",
        }
        if part in names:
            return f"player_tf.layers.{layer}.{names[part]}.{parameter}"
        return None
    prefixes = {
        "local.c1": "local_net.0",
        "local.c2": "local_net.2",
        "local.c3": "local_net.4",
        "trunk1": "trunk.0",
        "trunk2": "trunk.2",
        "htc1": "head_tile_coarse.0",
        "htc2": "head_tile_coarse.2",
        "htf1": "head_tile_fine.0",
        "htf2": "head_tile_fine.2",
    }
    for rust, python in prefixes.items():
        if rust_key.startswith(f"{rust}."):
            return f"{python}{rust_key[len(rust):]}"
    direct = {
        "player_in",
        "head_action",
        "head_player_q",
        "head_build",
        "head_nuke",
        "head_quantity",
        "head_value",
    }
    if rust_key.split(".", 1)[0] in direct:
        return rust_key
    return None


def map_oftrain_state(
    source: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Map every tensor and reject missing, extra, or shape-mismatched values."""
    mapped: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    tf_layers = sorted(
        {int(match.group(1)) for key in source if (match := _TF.fullmatch(key))}
    )
    for layer in tf_layers:
        for parameter in ("weight", "bias"):
            keys = [f"tf.{layer}.{part}.{parameter}" for part in ("q", "k", "v")]
            if all(key in source for key in keys):
                target = f"player_tf.layers.{layer}.self_attn.in_proj_{parameter}"
                mapped[target] = torch.cat([source[key] for key in keys], dim=0)
                consumed.update(keys)
    for key, tensor in source.items():
        if key in consumed:
            continue
        target = _python_key(key)
        if target is None:
            raise ValueError(f"unmapped oftrain tensor: {key}")
        if target in mapped:
            raise ValueError(f"duplicate mapped tensor: {target}")
        mapped[target] = tensor
    missing = sorted(set(expected) - set(mapped))
    extra = sorted(set(mapped) - set(expected))
    if missing or extra:
        raise ValueError(f"strict safetensors key mismatch: missing={missing}, extra={extra}")
    wrong_shapes = {
        key: (tuple(mapped[key].shape), tuple(expected[key].shape))
        for key in expected
        if mapped[key].shape != expected[key].shape
    }
    if wrong_shapes:
        raise ValueError(f"strict safetensors shape mismatch: {wrong_shapes}")
    return mapped


def load_oftrain_safetensors(
    policy: torch.nn.Module, checkpoint: str | Path
) -> dict[str, object]:
    """Load current Rust policy weights without creating a transient policy.pt."""
    from safetensors.torch import load_file

    path = Path(checkpoint)
    if path.suffix != ".safetensors":
        raise ValueError("oftrain loader requires a .safetensors checkpoint")
    manifest_path = path.parent / "manifest.json"
    manifest: dict[str, object] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "oftrain-safetensors":
            raise ValueError(f"unsupported checkpoint format: {manifest.get('format')!r}")
        if manifest.get("manifest_schema_version") != 1:
            raise ValueError("unsupported oftrain manifest schema")
        architecture = manifest.get("architecture", {})
        if isinstance(architecture, dict) and architecture.get("schema_version") == 2:
            raise ValueError(
                "recurrent oftrain architecture schema v2 is not supported by "
                "the transient Python Policy mapping"
            )
        if not isinstance(architecture, dict) or architecture.get("schema_version") != 1:
            raise ValueError("unsupported oftrain architecture schema")
    mapped = map_oftrain_state(load_file(str(path), device="cpu"), policy.state_dict())
    policy.load_state_dict(mapped, strict=True)
    state_path = path.with_name(f"{path.stem}.state.json")
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {}
    )
    return {"manifest": manifest, "state": state}
