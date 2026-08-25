#!/usr/bin/env python3
"""Validate and merge the 48 TFS cells with the frozen 48 DGL cells."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path


tfs_root = Path(sys.argv[1])
dgl_root = Path(sys.argv[2])
out_root = Path(sys.argv[3])
out_root.mkdir(parents=True, exist_ok=True)

graphs = ("products", "arxiv", "igb19", "igb2983")
layers_values = (2, 3)
thread_values = (1, 2, 4, 8, 16, 32)


def _required_text(cell: Path, name: str) -> str:
    path = cell / name
    if not path.is_file():
        raise RuntimeError(f"missing provenance artifact: {path}")
    return path.read_text().strip()


def _provenance(cell: Path, manifest: dict, threads: int) -> dict:
    environment = {}
    for line in _required_text(cell, "environment_start.txt").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            if key in {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "OMP_DYNAMIC",
                       "MKL_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES"}:
                environment[key] = value
    expected = {"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
                "OMP_DYNAMIC": "FALSE", "MKL_DYNAMIC": "FALSE",
                "OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}
    if environment != expected:
        raise RuntimeError(f"invalid thread environment in {cell}: {environment}")
    numactl = str(manifest.get("numactl", ""))
    if numactl != "--cpunodebind=0-3 --localalloc":
        raise RuntimeError(f"invalid numactl provenance in {cell}: {numactl!r}")
    partition = str(manifest.get("partition", "")).strip()
    if not partition:
        raise RuntimeError(f"missing partition provenance in {cell}")
    return {"numactl": numactl, "environment": environment,
            "partition": partition,
            "hostname": _required_text(cell, "hostname.txt"),
            "lscpu": _required_text(cell, "lscpu.txt"),
            "affinity": _required_text(cell, "affinity.txt")}


def _assert_comparable_provenance(tfs: dict, dgl: dict) -> None:
    for name in ("numactl", "environment", "partition", "hostname", "lscpu", "affinity"):
        if tfs["provenance"][name] != dgl["provenance"][name]:
            raise RuntimeError(f"TFS/DGL provenance mismatch for {name}: "
                               f"{tfs['cell']} vs {dgl['cell']}")


def load_cell(root: Path, method: str, graph: str,
              layers: int, threads: int) -> dict:
    cell = root / f"{graph}_l{layers}_t{threads}_e200"
    status_path = cell / "status.json"
    manifest_path = cell / "manifest.json"
    csv_path = cell / "training_detailed.csv"
    if (not status_path.is_file() or not manifest_path.is_file() or
            not csv_path.is_file()):
        raise RuntimeError(f"missing {method} authority cell: {cell}")
    status = json.loads(status_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("timing_protocol") != "paper_v1":
        raise RuntimeError(f"non-paper timing protocol in {method} cell: {cell}")
    if status.get("status") != "success" or int(status.get("epochs", 0)) != 200:
        raise RuntimeError(f"invalid {method} status: {status_path}: {status}")
    rows = list(csv.DictReader(csv_path.open(newline="")))
    if len(rows) != 200:
        raise RuntimeError(f"{method} {cell.name}: expected 200 rows, got {len(rows)}")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, 201)):
        raise RuntimeError(f"{method} {cell.name}: non-contiguous epoch sequence")
    train = [float(row["train_step_ms"]) for row in rows]
    evaluation = [float(row["evaluation_ms"]) for row in rows]
    if not all(math.isfinite(value) and value > 0.0
               for value in train + evaluation):
        raise RuntimeError(f"{method} {cell.name}: invalid timing value")
    expected_path = "hybrid" if method == "tfs_final_pre_numa" else "dgl_stock"
    expected_dtype = ("bf16_inputs_fp32_accum_fp32_master"
                      if method == "tfs_final_pre_numa" else "fp32")
    for row in rows:
        if row.get("path") != expected_path:
            raise RuntimeError(f"{method} {cell.name}: wrong path={row.get('path')}")
        if row.get("dtype") != expected_dtype:
            raise RuntimeError(f"{method} {cell.name}: wrong dtype={row.get('dtype')}")
        if (int(row["layers"]) != layers or int(row["threads"]) != threads or
                int(row["seed"]) != 101):
            raise RuntimeError(f"{method} {cell.name}: row contract mismatch")
    if method == "tfs_final_pre_numa":
        if (status.get("requested_profile") != "final_pre_numa" or
                status.get("resolved_profile") != "final_pre_numa" or
                manifest.get("profile_status") != "authority" or
                manifest.get("path") != "hybrid" or
                manifest.get("dtype_contract") != expected_dtype or
                manifest.get("numa_private") != "off" or
                manifest.get("numa_reduce") != "off" or
                int(manifest.get("seed", -1)) != 101):
            raise RuntimeError(f"invalid final_pre_numa provenance: {cell}")
        required = ("plan_validation.json", "timing_markers.json",
                    "source_input_sha256.tsv", "resource_usage.txt")
        for name in required:
            if not (cell / name).is_file():
                raise RuntimeError(f"missing TFS provenance artifact: {cell / name}")
        plan_validation = json.loads((cell / "plan_validation.json").read_text())
        if (plan_validation.get("status") != "pass" or
                len(plan_validation.get("plans", ())) != layers):
            raise RuntimeError(f"invalid TFS plan validation: {cell}")
        timing = json.loads((cell / "timing_markers.json").read_text())
        marker_names = ("process_marker_ns", "data_ready_ns",
                        "framework_graph_ready_ns", "model_ready_ns",
                        "training_runtime_start_ns", "training_runtime_end_ns")
        markers = [int(timing[name]) for name in marker_names]
        if markers != sorted(markers) or markers[-1] <= markers[0]:
            raise RuntimeError(f"non-monotonic TFS timing markers: {cell}")
    else:
        if (status.get("method") != "dgl_stock" or
                status.get("path") != "dgl_stock" or
                manifest.get("method") != "dgl_stock" or
                manifest.get("dgl_variant") != "stock"):
            raise RuntimeError(f"invalid frozen DGL provenance: {cell}")
        if manifest.get("dgl_rerun") is not True:
            raise RuntimeError(f"DGL cell was not rerun under paper protocol: {cell}")
        timing = {}
    epoch_ms = [a + b for a, b in zip(train, evaluation)]
    steady_epoch_ms = epoch_ms[1:]
    if len(steady_epoch_ms) != 199:
        raise RuntimeError(f"{method} {cell.name}: paper steady window must be epochs 2..200")
    total_epoch_ms = sum(epoch_ms)
    return {
        "cell": str(cell),
        "wall_ms": int(status["wall_ms"]),
        "mean_train_ms": statistics.fmean(train),
        "median_train_ms": statistics.median(train),
        "mean_evaluation_ms": statistics.fmean(evaluation),
        "median_evaluation_ms": statistics.median(evaluation),
        "mean_epoch_ms": statistics.fmean(epoch_ms),
        "median_epoch_ms": statistics.median(epoch_ms),
        "steady_mean_epoch_ms": statistics.fmean(steady_epoch_ms),
        "steady_median_epoch_ms": statistics.median(steady_epoch_ms),
        "first_epoch_ms": epoch_ms[0],
        "last_epoch_ms": epoch_ms[-1],
        "sum_epoch_ms": total_epoch_ms,
        "non_epoch_wall_ms": int(status["wall_ms"]) - total_epoch_ms,
        "timing_markers": timing,
        "final_train_loss": float(rows[-1]["train_loss"]),
        "final_val_accuracy": float(rows[-1]["val_accuracy"]),
        "final_test_accuracy": float(rows[-1]["test_accuracy"]),
        "provenance": _provenance(cell, manifest, threads),
    }


merged = []
for graph in graphs:
    for layers in layers_values:
        for threads in thread_values:
            tfs = load_cell(tfs_root, "tfs_final_pre_numa", graph,
                            layers, threads)
            dgl = load_cell(dgl_root, "dgl_stock", graph, layers, threads)
            _assert_comparable_provenance(tfs, dgl)
            merged.append({
                "graph": graph,
                "layers": layers,
                "threads": threads,
                "epochs": 200,
                "tfs_wall_ms": tfs["wall_ms"],
                "dgl_wall_ms": dgl["wall_ms"],
                "cold_wall_speedup_dgl_over_tfs":
                    dgl["wall_ms"] / tfs["wall_ms"],
                "tfs_steady_epoch_ms": tfs["steady_median_epoch_ms"],
                "dgl_steady_epoch_ms": dgl["steady_median_epoch_ms"],
                "steady_epoch_speedup_dgl_over_tfs":
                    dgl["steady_median_epoch_ms"] / tfs["steady_median_epoch_ms"],
                "tfs_mean_train_ms": tfs["mean_train_ms"],
                "dgl_mean_train_ms": dgl["mean_train_ms"],
                "tfs_mean_evaluation_ms": tfs["mean_evaluation_ms"],
                "dgl_mean_evaluation_ms": dgl["mean_evaluation_ms"],
                "tfs_non_epoch_wall_ms": tfs["non_epoch_wall_ms"],
                "dgl_non_epoch_wall_ms": dgl["non_epoch_wall_ms"],
                "tfs_training_runtime_wall_ms":
                    tfs["timing_markers"].get("training_runtime_wall_ms"),
                "tfs_training_ready_ms":
                    tfs["timing_markers"].get("training_ready_ms"),
                "tfs_final_val_accuracy": tfs["final_val_accuracy"],
                "dgl_final_val_accuracy": dgl["final_val_accuracy"],
                "val_accuracy_delta_tfs_minus_dgl":
                    tfs["final_val_accuracy"] - dgl["final_val_accuracy"],
                "tfs_final_test_accuracy": tfs["final_test_accuracy"],
                "dgl_final_test_accuracy": dgl["final_test_accuracy"],
                "test_accuracy_delta_tfs_minus_dgl":
                    tfs["final_test_accuracy"] - dgl["final_test_accuracy"],
                "tfs_cell": tfs["cell"],
                "dgl_cell": dgl["cell"],
            })

if len(merged) != 48:
    raise RuntimeError(f"expected 48 comparisons, found {len(merged)}")

csv_path = out_root / "final_comparison_48cells.csv"
with csv_path.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(merged[0]))
    writer.writeheader()
    writer.writerows(merged)

payload = {
    "status": "complete",
    "cells": len(merged),
    "tfs_method": "final_pre_numa",
    "dgl_method": "stock GraphConv(norm=both), FP32, DGL 2.1.0",
    "dgl_result_policy": "rerun under paper_v1 protocol on comparable provenance",
    "timing": {
        "cold_wall": "process startup through 200 epochs and CSV write",
        "steady_epoch": "median of epoch 2..200 train+evaluation rows",
    },
    "rows": merged,
}
(out_root / "final_comparison_48cells.json").write_text(
    json.dumps(payload, indent=2) + "\n")

lines = [
    "# final_pre_numa vs stock DGL — paper-protocol 48-cell authority matrix",
    "",
    "- Each cell is one cold process and exactly 200 train+evaluation epochs.",
    "- TFS is the repaired `final_pre_numa` mixed-precision runtime.",
    "- DGL is rerun FP32 stock `GraphConv(norm=\"both\")` under matching provenance.",
    "- Steady time is the median of epochs 2–200; epoch 1 remains in cold wall and raw CSV only.",
    "",
    "| Graph | L | Threads | TFS wall (s) | DGL wall (s) | Cold speedup | Epoch speedup | Δ test acc |",
    "|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in merged:
    lines.append(
        f"| {row['graph']} | {row['layers']} | {row['threads']} | "
        f"{row['tfs_wall_ms']/1000:.3f} | {row['dgl_wall_ms']/1000:.3f} | "
        f"{row['cold_wall_speedup_dgl_over_tfs']:.3f}× | "
        f"{row['steady_epoch_speedup_dgl_over_tfs']:.3f}× | "
        f"{row['test_accuracy_delta_tfs_minus_dgl']:+.6f} |")
(out_root / "final_comparison_48cells.md").write_text("\n".join(lines) + "\n")
print(json.dumps({"status": "complete", "cells": len(merged),
                  "csv": str(csv_path)}))
