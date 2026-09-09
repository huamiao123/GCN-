#!/usr/bin/env python3
"""Paired A/B for the in-place log-softmax bounded terminal path."""

import json
import os
import statistics
from pathlib import Path

import torch

from bench_panelized_scope_igb import terminal_once, timing_record
from bench_supervision_scoped_products import hidden_input, metric, summary
from tfs_train.authority_model import HybridGCN
from tfs_train.datasets import load_igb_homogeneous
from tfs_train.supervision_scope import SupervisionScope


def run(hidden, terminal, graph, scope, labels, row_tile, optimized):
    os.environ["TFS_SCOPE_LOGSOFTMAX_OUT"] = "1" if optimized else "0"
    os.environ["TFS_SCOPE_FUSED_DB"] = "1"
    return terminal_once(hidden, terminal, graph, scope, labels, True,
                         row_tile)


def record_summary(records):
    return {key: summary([record[key] for record in records])
            for key in records[0] if key.endswith("_ms")}


def main():
    threads = int(os.environ.get("SCOPE_THREADS", "32"))
    out_dim = int(os.environ.get("SCOPE_OUT_DIM", "2983"))
    row_tile = int(os.environ.get("TFS_SCOPE_LOSS_ROW_TILE", "300000"))
    warmups = int(os.environ.get("SCOPE_WARMUPS", "1"))
    repeats = int(os.environ.get("SCOPE_REPEATS", "7"))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    ds = load_igb_homogeneous(
        os.environ["IGB_ROOT"], size="small", label_file="node_label_2K.npy",
        split_seed=20260813)
    x, graph = ds.x.contiguous(), ds.graph
    labels = ds.labels if out_dim == 2983 else ds.labels.remainder(out_dim)
    model = HybridGCN(threads, 2, dropout=0.5, in_dim=x.shape[1],
                      hidden_dim=128, out_dim=out_dim,
                      num_nodes=x.shape[0])
    scope = SupervisionScope.build(ds.train_mask, graph, threads)
    hidden = hidden_input(model, x, graph)
    terminal = model.convs[-1]

    reference = run(hidden, terminal, graph, scope, labels, row_tile, False)
    candidate = run(hidden, terminal, graph, scope, labels, row_tile, True)
    correctness = {
        "loss_abs": abs(reference["loss"] - candidate["loss"]),
        "dh": metric(reference["dh"], candidate["dh"]),
        "dw": metric(reference["dw"], candidate["dw"]),
        "db": metric(reference["db"], candidate["db"]),
    }
    passed = (correctness["loss_abs"] < 1e-5 and
              correctness["dh"]["relative_l2"] < 1e-5 and
              correctness["dw"]["relative_l2"] < 1e-5 and
              correctness["db"]["relative_l2"] < 1e-5)
    old_records, new_records = [], []
    for iteration in range(warmups + repeats):
        order = (False, True) if iteration % 2 == 0 else (True, False)
        values = {flag: run(hidden, terminal, graph, scope, labels, row_tile,
                            flag) for flag in order}
        if iteration >= warmups:
            old_records.append(timing_record(values[False]))
            new_records.append(timing_record(values[True]))
    old_median = statistics.median(x["elapsed_ms"] for x in old_records)
    new_median = statistics.median(x["elapsed_ms"] for x in new_records)
    payload = {
        "status": "pass" if passed else "fail",
        "contract": "bounded_panel_logsoftmax_pair_shadow_v1",
        "output_dim": out_dim,
        "threads": threads,
        "row_tile": row_tile,
        "correctness": correctness,
        "manual": record_summary(old_records),
        "logsoftmax_out": record_summary(new_records),
        "speedup": old_median / new_median,
        "manual_records": old_records,
        "logsoftmax_records": new_records,
    }
    output = Path(os.environ["SCOPE_OUTPUT"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
