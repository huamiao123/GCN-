#!/usr/bin/env python3
"""Validate and summarize one complete 200-epoch authority CSV."""

import csv
import json
import statistics
import sys
from pathlib import Path


csv_path, out_path, wall, method, graph, layers, threads, epochs = sys.argv[1:]
rows = list(csv.DictReader(Path(csv_path).open(newline="")))
requested = int(epochs)
if len(rows) != requested:
    raise RuntimeError(f"expected {requested} epoch rows, found {len(rows)}")
if [int(row["epoch"]) for row in rows] != list(range(1, requested + 1)):
    raise RuntimeError("epoch sequence is incomplete or reordered")


def values(name):
    result = [float(row[name]) for row in rows if row.get(name) not in (None, "")]
    if len(result) != requested:
        raise RuntimeError(f"missing {name} timing rows")
    return result


train = values("train_step_ms")
evaluation = values("evaluation_ms")
epoch = [left + right for left, right in zip(train, evaluation)]
if requested < 2:
    raise RuntimeError("paper timing protocol requires at least two epochs")
# Epoch 1 may include cache materialisation, allocator/page first-touch and
# framework lazy initialisation.  It remains in the raw CSV and cold-process
# wall, but is never folded into the steady-state statistic.
steady_start_epoch = 2
steady_train = train[steady_start_epoch - 1:]
steady_evaluation = evaluation[steady_start_epoch - 1:]
steady_epoch = epoch[steady_start_epoch - 1:]
payload = {
    "status": "success",
    "method": method,
    "graph": graph,
    "layers": int(layers),
    "threads": int(threads),
    "epochs_requested": requested,
    "rows": len(rows),
    "wall_ms": int(wall),
    "cold_process_wall_ms": int(wall),
    "timing_protocol": "paper_v1",
    "steady_epoch_range": f"{steady_start_epoch}-{requested}",
    "steady_epoch_count": len(steady_epoch),
    "mean_train_step_ms": statistics.fmean(train),
    "median_train_step_ms": statistics.median(train),
    "mean_evaluation_ms": statistics.fmean(evaluation),
    "median_evaluation_ms": statistics.median(evaluation),
    "mean_epoch_train_plus_eval_ms": statistics.fmean(epoch),
    "median_epoch_train_plus_eval_ms": statistics.median(epoch),
    "first_epoch_train_plus_eval_ms": epoch[0],
    "last_epoch_train_plus_eval_ms": epoch[-1],
    "steady_mean_train_step_ms": statistics.fmean(steady_train),
    "steady_median_train_step_ms": statistics.median(steady_train),
    "steady_mean_evaluation_ms": statistics.fmean(steady_evaluation),
    "steady_median_evaluation_ms": statistics.median(steady_evaluation),
    "steady_mean_epoch_train_plus_eval_ms": statistics.fmean(steady_epoch),
    "steady_median_epoch_train_plus_eval_ms": statistics.median(steady_epoch),
    "final_train_loss": float(rows[-1]["train_loss"]),
    "final_val_accuracy": float(rows[-1]["val_accuracy"]),
    "final_test_accuracy": float(rows[-1]["test_accuracy"]),
    "timing_scope": "cold wall includes process startup through CSV output; steady is epoch 2..N only",
}
Path(out_path).write_text(json.dumps(payload, indent=2) + "\n")
