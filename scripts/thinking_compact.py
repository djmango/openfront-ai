"""Compact MODEL-overlay debug logs into a few-KB thinking blob for parquet.

Full `.debug.json` is ~100–200 KB (21-way probs every decision). Parquet stores
a stride-sampled top-3 trace (`thinking_json`) so each game stays a few KB of
policy intent alongside `record_json`.
"""

from __future__ import annotations

import json
from typing import Any


DEFAULT_STRIDE = 15


def compact_debug(
    debug: dict[str, Any],
    *,
    stride: int = DEFAULT_STRIDE,
) -> dict[str, Any]:
    """Build a compact thinking dict from a watch `.debug.json` payload."""
    actions = list(debug.get("actions") or [])
    log = list(debug.get("log") or [])
    steps: list[list[Any]] = []
    prev_a: str | None = None
    n = len(log)
    for i, entry in enumerate(log):
        action = str(entry.get("action") or "noop")
        keep = (
            i < 3
            or i >= n - 5
            or (stride > 0 and i % stride == 0)
            or (action != "noop" and action != prev_a)
        )
        prev_a = action
        if not keep:
            continue
        probs = list(entry.get("probs") or [])
        top = sorted(range(len(probs)), key=lambda k: -float(probs[k]))[:3]
        try:
            a_idx = actions.index(action)
        except ValueError:
            a_idx = 255
        row: list[Any] = [
            int(entry.get("tick") or 0),
            a_idx,
            int(round(float(entry.get("value") or 0.0) * 100.0)),
        ]
        for k in top:
            row.extend([int(k), int(round(float(probs[k]) * 1000.0))])
        if action != "noop":
            row.append(str(entry.get("desc") or "")[:48])
        steps.append(row)
    return {
        "v": 1,
        "o": debug.get("outcome"),
        "T": int(debug.get("end_tick") or 0),
        "n": n,
        "stride": stride,
        "a": actions,
        "s": steps,
    }


def compact_debug_json(debug: dict[str, Any], *, stride: int = DEFAULT_STRIDE) -> str:
    return json.dumps(compact_debug(debug, stride=stride), separators=(",", ":"))


def load_thinking_for_record(record_path) -> str:
    """Return compact thinking JSON string for a GameRecord path, or ""."""
    from pathlib import Path

    path = Path(record_path)
    thinking = path.parent / f"{path.stem}.thinking.json"
    if thinking.is_file():
        raw = thinking.read_text().strip()
        if not raw:
            return ""
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        # Accept a full debug sidecar left under the thinking name.
        if isinstance(obj, dict) and "log" in obj and "s" not in obj:
            return compact_debug_json(obj)
        return json.dumps(obj, separators=(",", ":"))

    debug_path = path.parent / f"{path.stem}.debug.json"
    if debug_path.is_file():
        try:
            debug = json.loads(debug_path.read_text())
        except json.JSONDecodeError:
            return ""
        if debug.get("log"):
            return compact_debug_json(debug)
    return ""


def thinking_summary(thinking: dict[str, Any], *, max_steps: int = 12) -> str:
    """Human-readable snippet for chat / logs."""
    actions = list(thinking.get("a") or [])
    steps = list(thinking.get("s") or [])
    outcome = thinking.get("o")
    end_tick = thinking.get("T")
    lines = [
        f"outcome={outcome} end_tick={end_tick} decisions={thinking.get('n')} kept={len(steps)}"
    ]
    show = steps[: max_steps // 2] + ([["…"]] if len(steps) > max_steps else []) + steps[-(max_steps // 2) :]
    for row in show:
        if row == ["…"]:
            lines.append("  …")
            continue
        if not isinstance(row, list) or len(row) < 3:
            continue
        tick, a_idx, v_cents = row[0], row[1], row[2]
        name = actions[a_idx] if isinstance(a_idx, int) and 0 <= a_idx < len(actions) else str(a_idx)
        tops = []
        i = 3
        while i + 1 < len(row) and isinstance(row[i], int):
            ti, pm = row[i], row[i + 1]
            tname = actions[ti] if 0 <= ti < len(actions) else str(ti)
            tops.append(f"{tname}:{pm/1000:.2f}")
            i += 2
        desc = row[-1] if row and isinstance(row[-1], str) else ""
        extra = f"  {desc}" if desc else ""
        lines.append(
            f"  t={tick} {name} V={v_cents/100:.2f} top[{', '.join(tops[:3])}]{extra}"
        )
    return "\n".join(lines)


def main() -> None:
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("debug_json", type=Path)
    ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    debug = json.loads(args.debug_json.read_text())
    blob = compact_debug(debug, stride=args.stride)
    text = json.dumps(blob, separators=(",", ":"))
    if args.out:
        args.out.write_text(text)
    if args.summary:
        print(thinking_summary(blob))
    print(f"bytes={len(text)} kept={len(blob['s'])}/{blob['n']}")


if __name__ == "__main__":
    main()
