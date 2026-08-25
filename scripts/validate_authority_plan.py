#!/usr/bin/env python3
"""Fail a formal cell unless its logged plan is the frozen dimension plan."""

from __future__ import annotations

import json
import sys
from pathlib import Path


GRAPH_DIMS = {
    "products": (100, 128, 47),
    "arxiv": (128, 128, 40),
    "igb19": (1024, 128, 19),
    "igb2983": (1024, 128, 2983),
}


def expected_variant(k: int, d: int, layer: int) -> str:
    if layer == 0 and k <= d:
        return "aggregate_static_v3"
    if k > 128 and d <= 128:
        return "native_wide_k"
    if d > 128 and k <= d:
        return "aggregate_highd_full"
    return "native_c3"


def parse_line(line: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in line.strip().split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    return fields


log_path = Path(sys.argv[1])
graph = sys.argv[2]
layers = int(sys.argv[3])
if graph not in GRAPH_DIMS:
    raise RuntimeError(f"unknown authority graph: {graph}")
base = GRAPH_DIMS[graph]
dims = (base[0],) + (base[1],) * (layers - 1) + (base[2],)
lines = [line for line in log_path.read_text().splitlines()
         if line.startswith("TFS_PLAN ")]
if len(lines) != layers:
    raise RuntimeError(f"expected {layers} plans, found {len(lines)}")

resolved = []
for layer, line in enumerate(lines):
    fields = parse_line(line)
    expected = {
        "layer": str(layer),
        "layer_index": str(layer),
        "K": str(dims[layer]),
        "D": str(dims[layer + 1]),
        "compute_dx": "0" if layer == 0 else "1",
        "execution_variant": expected_variant(
            dims[layer], dims[layer + 1], layer),
        "selection_policy": "auto",
    }
    mismatches = {
        key: {"actual": fields.get(key), "expected": value}
        for key, value in expected.items() if fields.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"layer {layer} authority-plan mismatch: {mismatches}")
    if not fields.get("plan_id") or not fields.get("plan_hash"):
        raise RuntimeError(f"layer {layer} is missing plan provenance")
    if fields.get("fallback", "none") != "none":
        raise RuntimeError(
            f"layer {layer} unexpectedly used fallback={fields['fallback']}")
    resolved.append({key: fields[key] for key in (
        "layer", "K", "D", "order", "execution_variant", "compute_dx",
        "workspace", "workspace_bytes", "plan_id", "plan_hash")})

print(json.dumps({"status": "pass", "graph": graph, "layers": layers,
                  "plans": resolved}, sort_keys=True))
