#!/usr/bin/env python3
"""Compare final_pre_numa with frozen v2_3_auto on shared 200-epoch cells."""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path


if len(sys.argv) not in (4, 5):
    raise SystemExit(
        "usage: compare_final_to_v23_auto.py FINAL_ROOT V23_ROOT OUT_ROOT "
        "[THREADS_CSV]")

final_root = Path(sys.argv[1])
old_root = Path(sys.argv[2])
out_root = Path(sys.argv[3])
threads_values = tuple(int(value) for value in
                       (sys.argv[4] if len(sys.argv) == 5 else
                        "4,8,16,32").split(","))
out_root.mkdir(parents=True, exist_ok=True)


def load(root: Path, graph: str, layers: int, threads: int) -> dict:
    cell = root / f"{graph}_l{layers}_t{threads}_e200"
    status_path = cell / "status.json"
    rows_path = cell / "training_detailed.csv"
    if not status_path.is_file() or not rows_path.is_file():
        raise RuntimeError(f"missing cell: {cell}")
    status = json.loads(status_path.read_text())
    rows = list(csv.DictReader(rows_path.open(newline="")))
    if status.get("status") != "success" or len(rows) != 200:
        raise RuntimeError(f"incomplete cell: {cell}")
    epoch_ms = [float(row["train_step_ms"]) + float(row["evaluation_ms"])
                for row in rows]
    return {
        "cell": str(cell),
        "wall_ms": float(status["wall_ms"]),
        "mean_epoch_ms": statistics.fmean(epoch_ms),
        "median_epoch_ms": statistics.median(epoch_ms),
        "test_accuracy": float(rows[-1]["test_accuracy"]),
    }


rows = []
violations = []
for graph in ("products", "arxiv", "igb19", "igb2983"):
    threshold = 0.02 if graph == "igb2983" else 0.03
    for layers in (2, 3):
        for threads in threads_values:
            new = load(final_root, graph, layers, threads)
            old = load(old_root, graph, layers, threads)
            wall_ratio = new["wall_ms"] / old["wall_ms"]
            epoch_ratio = new["mean_epoch_ms"] / old["mean_epoch_ms"]
            passed = wall_ratio <= 1.0 + threshold and epoch_ratio <= 1.0 + threshold
            row = {
                "graph": graph,
                "layers": layers,
                "threads": threads,
                "regression_threshold": threshold,
                "final_wall_ms": new["wall_ms"],
                "v23_auto_wall_ms": old["wall_ms"],
                "wall_ratio_final_over_old": wall_ratio,
                "final_mean_epoch_ms": new["mean_epoch_ms"],
                "v23_auto_mean_epoch_ms": old["mean_epoch_ms"],
                "epoch_ratio_final_over_old": epoch_ratio,
                "test_accuracy_delta": new["test_accuracy"] - old["test_accuracy"],
                "non_regression_pass": passed,
                "final_cell": new["cell"],
                "v23_auto_cell": old["cell"],
            }
            rows.append(row)
            if not passed:
                violations.append(row)

payload = {
    "status": "pass" if not violations else "non_regression_failure",
    "comparison": "final_pre_numa / frozen v2_3_auto",
    "threads": threads_values,
    "cells": len(rows),
    "violations": violations,
    "rows": rows,
}
(out_root / "final_vs_v23_auto_non_regression.json").write_text(
    json.dumps(payload, indent=2) + "\n")
print(json.dumps({"status": payload["status"], "cells": len(rows),
                  "violations": len(violations)}))
raise SystemExit(0 if not violations else 3)
