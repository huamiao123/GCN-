#!/usr/bin/env python3
"""Paired A/B for native fused db inside bounded-panel terminal CE."""

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


def run(hidden, terminal, graph, scope, labels, row_tile, fused):
    os.environ["TFS_SCOPE_FUSED_DB"] = "1" if fused else "0"
    return terminal_once(
        hidden, terminal, graph, scope, labels, True, row_tile)


def record_summary(records):
    return {
        key: summary([record[key] for record in records])
        for key in records[0] if key.endswith("_ms")
    }


def main():
    threads = int(os.environ.get("SCOPE_THREADS", "32"))
    row_tile = int(os.environ.get("TFS_SCOPE_LOSS_ROW_TILE", "300000"))
    warmups = int(os.environ.get("SCOPE_WARMUPS", "2"))
    repeats = int(os.environ.get("SCOPE_REPEATS", "10"))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    ds = load_igb_homogeneous(
        os.environ["IGB_ROOT"], size="small", label_file="node_label_2K.npy",
        split_seed=20260813)
    x, labels, graph = ds.x.contiguous(), ds.labels, ds.graph
    model = HybridGCN(
        threads, 2, dropout=0.5, in_dim=x.shape[1], hidden_dim=128,
        out_dim=2983, num_nodes=x.shape[0])
    scope = SupervisionScope.build(ds.train_mask, graph, threads)
    hidden = hidden_input(model, x, graph)
    terminal = model.convs[-1]

    plain = run(hidden, terminal, graph, scope, labels, row_tile, False)
    fused = run(hidden, terminal, graph, scope, labels, row_tile, True)
    correctness = {
        "loss_abs": abs(plain["loss"] - fused["loss"]),
        "dh": metric(plain["dh"], fused["dh"]),
        "dw": metric(plain["dw"], fused["dw"]),
        "db": metric(plain["db"], fused["db"]),
    }
    passed = (
        correctness["loss_abs"] == 0.0 and
        correctness["dh"]["relative_l2"] < 1e-6 and
        correctness["dw"]["relative_l2"] < 1e-6 and
        (correctness["db"]["max_abs"] < 5e-5 or
         correctness["db"]["relative_l2"] < 1e-4))

    plain_records, fused_records = [], []
    for iteration in range(warmups + repeats):
        order = (False, True) if iteration % 2 == 0 else (True, False)
        values = {}
        for use_fused in order:
            values[use_fused] = run(
                hidden, terminal, graph, scope, labels, row_tile, use_fused)
        if iteration >= warmups:
            plain_records.append(timing_record(values[False]))
            fused_records.append(timing_record(values[True]))
    speedup = (
        statistics.median(x["elapsed_ms"] for x in plain_records) /
        statistics.median(x["elapsed_ms"] for x in fused_records))
    payload = {
        "status": "pass" if passed else "fail",
        "contract": "bounded_panel_native_fused_db_shadow_v1",
        "threads": threads,
        "row_tile": row_tile,
        "correctness": correctness,
        "plain": record_summary(plain_records),
        "fused_db": record_summary(fused_records),
        "speedup": speedup,
        "plain_records": plain_records,
        "fused_records": fused_records,
    }
    output = Path(os.environ.get(
        "SCOPE_OUTPUT", "panelized_fused_db_igb.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
