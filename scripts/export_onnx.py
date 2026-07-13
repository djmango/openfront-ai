"""Export the frozen AE encoder + PPO policy to ONNX for browser inference.

Two graphs, split exactly like the "compress only what's big" philosophy in
DESIGN.md: the AE encoder (conv stem) is the only piece that needs a real
neural net for ownership/terrain/fallout; ego-pooling and the local crop are
cheap array ops done in TypeScript (webbot/features.ts) to keep the ONNX
graphs simple and shape-flexible.

  ae_encoder.onnx: (owners int64 [1,H,W], terrain float [1,3,H,W],
                     static float [1,NUM_STATIC,H/8,W/8]) -> z [1,32,H/8,W/8]
  policy.onnx: (grid, grid_valid, local, players, pmask, scalars,
                legal_actions, legal_ptarget, legal_build, legal_nuke,
                legal_tile) -> (action_logits, player_logits, tile_logits,
                                 build_logits, nuke_logits, quantity_params,
                                 value)

Usage:
  uv run python -m scripts.export_onnx \
      --ae runs/ae_v31_d8c32/ae_v3.pt \
      --policy rust/checkpoints/ppo_v81/latest.safetensors \
      --out openfront/resources/webbot/models
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from rl.obs import load_ae, C_GRID, MAX_SLOTS, N_LOCAL, N_SCALARS, LOCAL, P_FEAT
from rl.policy import Policy


class AEEncoderWrapper(nn.Module):
    def __init__(self, ae: nn.Module):
        super().__init__()
        self.ae = ae

    def forward(
        self, owners: torch.Tensor, terrain: torch.Tensor, static: torch.Tensor
    ) -> torch.Tensor:
        return self.ae.encode(owners, terrain, static)


class PolicyWrapper(nn.Module):
    """Tensor-in/tensor-out wrapper: ONNX has no dict I/O, so trunk_forward's
    dict argument is unpacked into named parameters in a fixed order.

    legal_ptarget is deliberately NOT a graph input: Policy.heads() never
    reads it (masking the player head depends on the SAMPLED action, which
    only exists after sampling - see Policy.act()). The browser side keeps
    legal_ptarget as plain data and applies the same gather-then-mask step
    itself after sampling the action head."""

    def __init__(self, policy: Policy):
        super().__init__()
        self.policy = policy

    def forward(
        self,
        grid: torch.Tensor,
        grid_valid: torch.Tensor,
        local: torch.Tensor,
        players: torch.Tensor,
        pmask: torch.Tensor,
        scalars: torch.Tensor,
        legal_actions: torch.Tensor,
        legal_build: torch.Tensor,
        legal_nuke: torch.Tensor,
        legal_tile: torch.Tensor,
    ):
        o = {
            "grid": grid,
            "grid_valid": grid_valid,
            "local": local,
            "players": players,
            "pmask": pmask,
            "scalars": scalars,
            "legal_actions": legal_actions,
            "legal_build": legal_build,
            "legal_nuke": legal_nuke,
            "legal_tile": legal_tile,
        }
        h, g, p = self.policy.trunk_forward(o)
        out = self.policy.heads(h, g, p, o)
        return (
            out["action"],
            out["player"],
            out["tile"],
            out["build"],
            out["nuke"],
            out["quantity"],
            out["value"],
        )


def make_dummy_inputs(gh: int = 19, gw: int = 31, h: int = 152, w: int = 248):
    from ae.units import STATIC_CLASSES

    n_static = len(STATIC_CLASSES)
    owners = torch.randint(0, MAX_SLOTS, (1, h, w), dtype=torch.int64)
    terrain = torch.rand(1, 3, h, w, dtype=torch.float32)
    static = torch.rand(1, n_static, gh, gw, dtype=torch.float32)

    grid = torch.randn(1, C_GRID, gh, gw, dtype=torch.float32)
    grid_valid = torch.ones(1, gh, gw, dtype=torch.float32)
    local = torch.rand(1, N_LOCAL, LOCAL, LOCAL, dtype=torch.float32)
    players = torch.randn(1, MAX_SLOTS, P_FEAT, dtype=torch.float32)
    pmask = torch.zeros(1, MAX_SLOTS, dtype=torch.float32)
    pmask[:, :5] = 1.0
    scalars = torch.rand(1, N_SCALARS, dtype=torch.float32)
    from rl.obs import ACTIONS, BUILD_TYPES, NUKE_TYPES

    legal_actions = torch.ones(1, len(ACTIONS), dtype=torch.float32)
    legal_build = torch.ones(1, len(BUILD_TYPES), dtype=torch.float32)
    legal_nuke = torch.ones(1, len(NUKE_TYPES), dtype=torch.float32)
    legal_tile = torch.ones(1, gh, gw, dtype=torch.float32)
    return {
        "ae": (owners, terrain, static),
        "policy": (
            grid, grid_valid, local, players, pmask, scalars,
            legal_actions, legal_build, legal_nuke, legal_tile,
        ),
    }


def export_ae(ae, out_dir: Path, dummy) -> None:
    wrapper = AEEncoderWrapper(ae).eval()
    owners, terrain, static = dummy["ae"]
    path = out_dir / "ae_encoder.onnx"
    torch.onnx.export(
        wrapper,
        (owners, terrain, static),
        str(path),
        input_names=["owners", "terrain", "static"],
        output_names=["z"],
        dynamic_axes={
            "owners": {1: "h", 2: "w"},
            "terrain": {2: "h", 3: "w"},
            "static": {2: "gh", 3: "gw"},
            "z": {2: "gh", 3: "gw"},
        },
        opset_version=18,
        dynamo=False,
    )
    print(f"wrote {path}")


def export_policy(policy, out_dir: Path, dummy) -> None:
    wrapper = PolicyWrapper(policy).eval()
    args = dummy["policy"]
    names = [
        "grid", "grid_valid", "local", "players", "pmask", "scalars",
        "legal_actions", "legal_build", "legal_nuke", "legal_tile",
    ]
    out_names = [
        "action_logits", "player_logits", "tile_logits", "build_logits",
        "nuke_logits", "quantity_params", "value",
    ]
    dyn = {
        "grid": {2: "gh", 3: "gw"},
        "grid_valid": {1: "gh", 2: "gw"},
        "legal_tile": {1: "gh", 2: "gw"},
        "tile_logits": {1: "flat_tile"},
    }
    path = out_dir / "policy.onnx"
    torch.onnx.export(
        wrapper,
        args,
        str(path),
        input_names=names,
        output_names=out_names,
        dynamic_axes=dyn,
        opset_version=18,
        dynamo=False,
    )
    print(f"wrote {path}")


def verify(out_dir: Path, ae, policy, dummy) -> None:
    import onnxruntime as ort

    owners, terrain, static = dummy["ae"]
    with torch.no_grad():
        z_torch = ae.encode(owners, terrain, static).numpy()
    sess = ort.InferenceSession(str(out_dir / "ae_encoder.onnx"), providers=["CPUExecutionProvider"])
    z_onnx = sess.run(
        None,
        {
            "owners": owners.numpy(),
            "terrain": terrain.numpy(),
            "static": static.numpy(),
        },
    )[0]
    diff = np.abs(z_torch - z_onnx).max()
    print(f"AE encoder max abs diff: {diff:.2e}")
    assert diff < 1e-3, "AE encoder ONNX output diverges from PyTorch"

    args = dummy["policy"]
    names = [
        "grid", "grid_valid", "local", "players", "pmask", "scalars",
        "legal_actions", "legal_build", "legal_nuke", "legal_tile",
    ]
    with torch.no_grad():
        h, g, p = policy.trunk_forward(dict(zip(names, args)))
        out = policy.heads(h, g, p, dict(zip(names, args)))
    sess2 = ort.InferenceSession(str(out_dir / "policy.onnx"), providers=["CPUExecutionProvider"])
    onnx_out = sess2.run(None, {n: a.numpy() for n, a in zip(names, args)})
    torch_out = [out["action"], out["player"], out["tile"], out["build"], out["nuke"], out["quantity"], out["value"]]
    for name, t, o in zip(
        ["action", "player", "tile", "build", "nuke", "quantity", "value"],
        torch_out, onnx_out,
    ):
        diff = np.abs(t.numpy() - o).max()
        print(f"policy {name} max abs diff: {diff:.2e}")
        assert diff < 1e-3, f"policy head {name} ONNX output diverges from PyTorch"
    print("parity OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ae", default="runs/ae_v31_d8c32/ae_v3.pt")
    ap.add_argument(
        "--policy", default="rust/checkpoints/ppo_v81/latest.safetensors"
    )
    ap.add_argument("--out", default="openfront/resources/webbot/models")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ae = load_ae(args.ae, "cpu")
    policy = Policy()
    if Path(args.policy).suffix == ".safetensors":
        from scripts.policy_safetensors import load_oftrain_safetensors

        metadata = load_oftrain_safetensors(policy, args.policy)
        ck = metadata["state"]
    else:
        # Explicit legacy Python fixture compatibility. New oftrain runs never
        # create this format.
        ck = torch.load(args.policy, map_location="cpu", weights_only=False)
        policy.load_state_dict(ck["model_state_dict"], strict=True)
    policy.eval()
    # TransformerEncoder's nested-tensor fast path (autoselected whenever a
    # key_padding_mask is passed) diverges under ONNX tracing - not a
    # learned parameter, just an inference kernel choice, safe to disable.
    # (forward() actually reads `use_nested_tensor`, set from the
    # constructor's enable_nested_tensor arg; flip both belts-and-braces.)
    policy.player_tf.enable_nested_tensor = False
    policy.player_tf.use_nested_tensor = False
    torch.backends.mha.set_fastpath_enabled(False)
    print(f"loaded ae ({args.ae}) and policy update={ck.get('update', '?')} ({args.policy})")

    dummy = make_dummy_inputs()
    export_ae(ae, out_dir, dummy)
    export_policy(policy, out_dir, dummy)
    if not args.skip_verify:
        verify(out_dir, ae, policy, dummy)


if __name__ == "__main__":
    main()
