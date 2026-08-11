"""Unit tests for openfront-rl prune selection (no Hub I/O)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "hf_prune_openfront_rl.py"


def _load():
    spec = importlib.util.spec_from_file_location("hf_prune_openfront_rl", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_select_v11_updates_sparse_old_dense_recent():
    mod = _load()
    updates = list(range(5, 2500, 5))  # every 5 like the Hub backlog
    keep = mod.select_v11_updates(updates)
    assert 5 in keep
    assert 2485 in keep or 2495 in keep  # max present
    assert max(updates) in keep
    # old region: multiples of 100 only (plus first)
    old = {u for u in keep if u < max(updates) - 200}
    assert all(u == 5 or u % 100 == 0 for u in old)
    # recent: multiples of 25
    recent = {u for u in keep if u >= max(updates) - 200}
    assert all(u == max(updates) or u % 25 == 0 for u in recent)


def test_plan_keeps_latest_and_curriculum_deletes_dense():
    mod = _load()
    files = [
        (".gitattributes", 10),
        ("ppo_v11/latest.safetensors", 100),
        ("ppo_v11/latest.state.json", 1),
        ("ppo_v11/manifest.json", 1),
        ("ppo_v11/curriculum_advance_u100_s1_to_2.safetensors", 100),
        ("ppo_v11/policy_update100.safetensors", 100),
        ("ppo_v11/policy_update100.state.json", 1),
        ("ppo_v11/policy_update105.safetensors", 100),
        ("ppo_v11/policy_update105.state.json", 1),
        ("ppo_v11/policy_update2485.safetensors", 100),
        ("ppo_v11/policy_update2485.state.json", 1),
        ("ppo_v10/latest.safetensors", 50),
        ("ppo_v10/latest.state.json", 1),
        ("ppo_v10/manifest.json", 1),
        ("ppo_v10/policy_update12000.safetensors", 50),
        ("bc_v6/foo.safetensors", 50),
    ]
    keep, delete, keep_updates = mod.plan(files)
    keep_paths = {p for p, _ in keep}
    del_paths = {p for p, _ in delete}
    assert "ppo_v11/latest.safetensors" in keep_paths
    assert "ppo_v11/curriculum_advance_u100_s1_to_2.safetensors" in keep_paths
    assert "ppo_v11/policy_update100.safetensors" in keep_paths
    assert "ppo_v11/policy_update2485.safetensors" in keep_paths
    assert "ppo_v11/policy_update105.safetensors" in del_paths
    assert "ppo_v10/latest.safetensors" in keep_paths
    assert "ppo_v10/policy_update12000.safetensors" in del_paths
    assert "bc_v6/foo.safetensors" in del_paths
    assert 100 in keep_updates and 2485 in keep_updates
