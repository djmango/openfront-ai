"""Strictly export a Rust ``oftrain`` policy checkpoint to Python ``policy.pt``.

The two implementations have the same tensor layouts (Conv2d OIHW, Linear
out/in, and one-dimensional LayerNorm parameters), but different parameter
names. Rust also stores attention Q/K/V separately while PyTorch's
TransformerEncoderLayer stores one Q/K/V-concatenated ``in_proj`` tensor.

Usage:
    uv run python -m scripts.export_oftrain_policy \
        checkpoints/latest.safetensors \
        checkpoints/latest.state.json \
        runs/rl/rust/policy.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
import torch
from safetensors import safe_open

from rl.obs import C_GRID, C_GRID_FINE, N_ACTIONS, N_LOCAL, N_SCALARS, P_FEAT
from rl.policy import Policy


FORMAT = "oftrain-policy-export-v1"
ARCHITECTURE = {
    "name": "Policy",
    "observation_channels": C_GRID,
    "fine_observation_channels": C_GRID_FINE,
    "actions": N_ACTIONS,
    "player_features": P_FEAT,
    "scalar_features": N_SCALARS,
    "local_channels": N_LOCAL,
    "hidden": 512,
    "grid_channels": 256,
    "player_channels": 128,
    "local_embedding": 64,
    "residual_blocks": 4,
    "transformer_layers": 2,
    "transformer_heads": 4,
    "quantity_distribution": "beta",
}


class ExportError(ValueError):
    """The source checkpoint cannot be exported without guessing."""


@dataclass(frozen=True)
class Mapping:
    destination: str
    sources: tuple[str, ...]
    transform: str = "identity"


def _mapping() -> tuple[Mapping, ...]:
    """Audited map for the fixed 89-channel/21-action Beta architecture."""
    entries: list[Mapping] = []

    def direct(destination: str, source: str) -> None:
        entries.append(Mapping(destination, (source,)))

    for rust_tower, python_tower in (
        ("grid_coarse", "grid_coarse_net"),
        ("grid_fine", "grid_fine_net"),
    ):
        for parameter in ("weight", "bias"):
            direct(f"{python_tower}.0.{parameter}", f"{rust_tower}.stem.{parameter}")
        for block in range(4):
            python_block = block + 2
            for conv in ("conv1", "conv2"):
                for parameter in ("weight", "bias"):
                    direct(
                        f"{python_tower}.{python_block}.{conv}.{parameter}",
                        f"{rust_tower}.block.{block}.{conv}.{parameter}",
                    )

    for rust_name, python_index in (("c1", 0), ("c2", 2), ("c3", 4)):
        for parameter in ("weight", "bias"):
            direct(f"local_net.{python_index}.{parameter}", f"local.{rust_name}.{parameter}")

    for parameter in ("weight", "bias"):
        direct(f"player_in.{parameter}", f"player_in.{parameter}")

    for layer in range(2):
        rust = f"tf.{layer}"
        python = f"player_tf.layers.{layer}"
        for parameter in ("weight", "bias"):
            entries.append(
                Mapping(
                    f"{python}.self_attn.in_proj_{parameter}",
                    tuple(f"{rust}.{projection}.{parameter}" for projection in ("q", "k", "v")),
                    "concat_qkv_dim0",
                )
            )
            direct(f"{python}.self_attn.out_proj.{parameter}", f"{rust}.out.{parameter}")
            direct(f"{python}.linear1.{parameter}", f"{rust}.ff1.{parameter}")
            direct(f"{python}.linear2.{parameter}", f"{rust}.ff2.{parameter}")
            # tch LayerNorm is named ln1/ln2; PyTorch calls these norm1/norm2.
            direct(f"{python}.norm1.{parameter}", f"{rust}.ln1.{parameter}")
            direct(f"{python}.norm2.{parameter}", f"{rust}.ln2.{parameter}")

    for rust_name, python_name in (("trunk1", "trunk.0"), ("trunk2", "trunk.2")):
        for parameter in ("weight", "bias"):
            direct(f"{python_name}.{parameter}", f"{rust_name}.{parameter}")

    for rust_name, python_name in (
        ("head_action", "head_action"),
        ("head_player_q", "head_player_q"),
        ("htc1", "head_tile_coarse.0"),
        ("htc2", "head_tile_coarse.2"),
        ("htf1", "head_tile_fine.0"),
        ("htf2", "head_tile_fine.2"),
        ("head_build", "head_build"),
        ("head_nuke", "head_nuke"),
        ("head_quantity", "head_quantity"),
        ("head_value", "head_value"),
    ):
        for parameter in ("weight", "bias"):
            direct(f"{python_name}.{parameter}", f"{rust_name}.{parameter}")

    return tuple(entries)


MAPPING = _mapping()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_state(path: Path) -> tuple[int, int]:
    try:
        state = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportError(f"cannot read state JSON {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise ExportError("state JSON must contain an object")
    values = []
    for field in ("update", "stage"):
        value = state.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ExportError(f"state JSON field {field!r} must be a non-negative integer")
        values.append(value)
    return values[0], values[1]


def _load_source(path: Path) -> dict[str, torch.Tensor]:
    try:
        with safe_open(path, framework="pt", device="cpu") as source:
            return {key: source.get_tensor(key) for key in source.keys()}
    except Exception as exc:
        raise ExportError(f"cannot read safetensors checkpoint {path}: {exc}") from exc


def _expected_python_state() -> dict[str, torch.Tensor]:
    if (C_GRID, C_GRID_FINE, N_ACTIONS) != (89, 90, 21):
        raise ExportError(
            "exporter only supports the audited 89-channel/90-fine-channel/"
            f"21-action architecture, current constants are {(C_GRID, C_GRID_FINE, N_ACTIONS)}"
        )
    return Policy(**{
        "hidden": ARCHITECTURE["hidden"],
        "gc": ARCHITECTURE["grid_channels"],
        "pc": ARCHITECTURE["player_channels"],
        "blocks": ARCHITECTURE["residual_blocks"],
        "lc": ARCHITECTURE["local_embedding"],
    }).state_dict()


def convert_tensors(source: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert tensors, refusing every unaccounted key or shape."""
    expected = _expected_python_state()
    mapped_destinations = {entry.destination for entry in MAPPING}
    if mapped_destinations != set(expected):
        missing = sorted(set(expected) - mapped_destinations)
        extra = sorted(mapped_destinations - set(expected))
        raise RuntimeError(f"internal mapping audit failed: missing={missing}, extra={extra}")

    required_sources = {key for entry in MAPPING for key in entry.sources}
    actual_sources = set(source)
    if actual_sources != required_sources:
        missing = sorted(required_sources - actual_sources)
        extra = sorted(actual_sources - required_sources)
        raise ExportError(f"source key mismatch: missing={missing}, extra={extra}")

    output: dict[str, torch.Tensor] = {}
    for entry in MAPPING:
        tensors = [source[key] for key in entry.sources]
        wanted = expected[entry.destination]
        for key, tensor in zip(entry.sources, tensors):
            if tensor.dtype != wanted.dtype:
                raise ExportError(
                    f"dtype mismatch for {entry.destination}: source {key} has "
                    f"{tensor.dtype}, expected {wanted.dtype}"
                )
        if entry.transform == "identity":
            value = tensors[0]
        elif entry.transform == "concat_qkv_dim0":
            source_shape = (wanted.shape[0] // 3, *wanted.shape[1:])
            for key, tensor in zip(entry.sources, tensors):
                if tensor.shape != source_shape:
                    raise ExportError(
                        f"shape mismatch for {entry.destination}: source {key} has "
                        f"{tuple(tensor.shape)}, expected {tuple(source_shape)}"
                    )
            value = torch.cat(tensors, dim=0)
        else:  # pragma: no cover - frozen Mapping literals make this unreachable.
            raise RuntimeError(f"unknown mapping transform {entry.transform}")
        if value.shape != wanted.shape:
            source_shapes = {key: tuple(tensor.shape) for key, tensor in zip(entry.sources, tensors)}
            raise ExportError(
                f"shape mismatch for {entry.destination}: source {source_shapes} produced "
                f"{tuple(value.shape)}, expected {tuple(wanted.shape)}"
            )
        output[entry.destination] = value.contiguous()

    # Exercise the exact strict load used by showcase/export_onnx before writing.
    policy = Policy()
    policy.load_state_dict(output, strict=True)
    return output


def export_policy(safetensors_path: Path, state_path: Path, output_path: Path) -> dict:
    safetensors_path = safetensors_path.resolve(strict=True)
    state_path = state_path.resolve(strict=True)
    if safetensors_path.suffix != ".safetensors":
        raise ExportError("input weights must have a .safetensors extension")
    update, stage = _read_state(state_path)
    model_state = convert_tensors(_load_source(safetensors_path))
    payload = {
        "model_state_dict": model_state,
        "update": update,
        "stage": stage,
        "architecture": dict(ARCHITECTURE),
        "source_sha256": _sha256(safetensors_path),
        "export_format": FORMAT,
    }

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output_path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("safetensors", type=Path, help="Rust oftrain policy weights")
    parser.add_argument("state_json", type=Path, help="matching Rust TrainState JSON")
    parser.add_argument("output", type=Path, help="output Python policy.pt")
    args = parser.parse_args()
    try:
        payload = export_policy(args.safetensors, args.state_json, args.output)
    except ExportError as exc:
        parser.error(str(exc))
    print(
        f"wrote {args.output} (update={payload['update']}, stage={payload['stage']}, "
        f"source_sha256={payload['source_sha256']})"
    )


if __name__ == "__main__":
    main()
