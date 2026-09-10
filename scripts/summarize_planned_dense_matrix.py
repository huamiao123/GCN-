#!/usr/bin/env python3
"""Validate and summarize direct authority-vs-planner result JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.result_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "pass":
            raise SystemExit(f"non-passing result: {path}")
        plan = payload.get("dense_plan")
        if not isinstance(plan, dict):
            raise SystemExit(f"missing explicit dense plan: {path}")
        correctness = payload["correctness"]
        rows.append({
            "dataset": payload["dataset"],
            "layers": payload["layers"],
            "threads": payload["threads"],
            "output_dim": payload["output_dim"],
            "row_tile": payload["row_tile"],
            "authority_median_ms":
                payload["authority"]["elapsed_ms"]["median_ms"],
            "planned_median_ms":
                payload["complete_bounded"]["elapsed_ms"]["median_ms"],
            "speedup": payload["speedup"],
            "paired_speedup_median": payload["paired_speedup_median"],
            "loss_abs": correctness["loss_abs"],
            "dh_relative_l2": correctness["dh"]["relative_l2"],
            "dw_relative_l2": correctness["dw"]["relative_l2"],
            "db_max_abs": correctness["db"]["max_abs"],
            "native_dw": plan["native_dw"],
            "direct_tail_transpose": plan["direct_tail_transpose"],
            "native_logits": plan["native_logits"],
            "fused_db": plan["fused_db"],
        })
    rows.sort(key=lambda row: (
        row["dataset"], row["layers"], row["threads"]))
    if not rows:
        raise SystemExit("no result JSON files found")

    groups = {}
    for row in rows:
        key = f"{row['dataset']}:L{row['layers']}"
        groups.setdefault(key, []).append(row["speedup"])
    aggregates = {
        key: math.prod(values) ** (1.0 / len(values))
        for key, values in groups.items()
    }
    aggregates["all_cells"] = math.prod(
        row["speedup"] for row in rows) ** (1.0 / len(rows))
    error_maxima = {
        "loss_abs": max(row["loss_abs"] for row in rows),
        "dh_relative_l2": max(row["dh_relative_l2"] for row in rows),
        "dw_relative_l2": max(row["dw_relative_l2"] for row in rows),
        "db_max_abs": max(row["db_max_abs"] for row in rows),
    }

    summary = {
        "cell_count": len(rows),
        "geometric_mean_speedup": aggregates,
        "maximum_errors": error_maxima,
        "cells": rows,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n")
    if args.markdown:
        lines = [
            "| Dataset | L | T | Authority ms | Planned ms | Speedup | "
            "loss abs | dH rel-L2 | dW rel-L2 | db max-abs |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f"| {row['dataset']} | {row['layers']} | "
                f"{row['threads']} | {row['authority_median_ms']:.3f} | "
                f"{row['planned_median_ms']:.3f} | {row['speedup']:.3f}x | "
                f"{row['loss_abs']:.3g} | {row['dh_relative_l2']:.3g} | "
                f"{row['dw_relative_l2']:.3g} | {row['db_max_abs']:.3g} |")
        lines.extend(["", "Geometric-mean speedups:", ""])
        for key, value in aggregates.items():
            lines.append(f"- `{key}`: `{value:.4f}x`")
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
