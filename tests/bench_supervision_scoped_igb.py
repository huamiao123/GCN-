#!/usr/bin/env python3
"""IGB-small gate for the supervision-scoped terminal TFS shadow path."""

import json
import os
import statistics
import time
from pathlib import Path

import torch

from bench_supervision_scoped_products import (
    hidden_input,
    metric,
    phase_summary,
    terminal_once,
    time_terminal,
    time_training_paired,
)
from tfs_train.authority_model import HybridGCN
from tfs_train.datasets import load_igb_homogeneous
from tfs_train.supervision_scope import SupervisionScope


def main():
    threads = int(os.environ.get("SCOPE_THREADS", "32"))
    layers = int(os.environ.get("SCOPE_LAYERS", "2"))
    warmups = int(os.environ.get("SCOPE_WARMUPS", "1"))
    repeats = int(os.environ.get("SCOPE_REPEATS", "5"))
    label_file = os.environ.get("IGB_LABEL_FILE", "node_label_19.npy")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    ds = load_igb_homogeneous(
        os.environ["IGB_ROOT"], size="small", label_file=label_file,
        split_seed=int(os.environ.get("IGB_SPLIT_SEED", "20260813")))
    x, labels, graph = ds.x.contiguous(), ds.labels, ds.graph
    out_dim = int(os.environ.get("SCOPE_OUT_DIM", int(labels.max()) + 1))
    if int(labels.max()) >= out_dim:
        raise ValueError("SCOPE_OUT_DIM does not cover the loaded labels")

    model = HybridGCN(
        threads, layers, dropout=0.5, in_dim=x.shape[1], hidden_dim=128,
        out_dim=out_dim, num_nodes=x.shape[0])
    build_begin = time.perf_counter_ns()
    scope = SupervisionScope.build(ds.train_mask, graph, threads)
    scope_build_ms = (time.perf_counter_ns() - build_begin) / 1e6
    hidden = hidden_input(model, x, graph)
    terminal = model.convs[-1]

    reference = terminal_once(hidden, terminal, graph, scope, labels, False)
    candidate = terminal_once(hidden, terminal, graph, scope, labels, True)
    correctness = {
        "loss_abs": abs(reference["loss"] - candidate["loss"]),
        "logits": metric(reference["logits"], candidate["logits"]),
        "dh": metric(reference["dh"], candidate["dh"]),
        "dw": metric(reference["dw"], candidate["dw"]),
        "db": metric(reference["db"], candidate["db"]),
    }

    terminal_once(hidden, terminal, graph, scope, labels, True)
    terminal_once(hidden, terminal, graph, scope, labels, False)
    full_terminal = time_terminal(
        hidden, terminal, graph, scope, labels, False, warmups, repeats)
    scoped_terminal = time_terminal(
        hidden, terminal, graph, scope, labels, True, warmups, repeats)

    baseline_model = HybridGCN(
        threads, layers, dropout=0.5, in_dim=x.shape[1], hidden_dim=128,
        out_dim=out_dim, num_nodes=x.shape[0])
    scoped_model = HybridGCN(
        threads, layers, dropout=0.5, in_dim=x.shape[1], hidden_dim=128,
        out_dim=out_dim, num_nodes=x.shape[0])
    scoped_model.load_state_dict(baseline_model.state_dict())
    full_train, scoped_train = time_training_paired(
        baseline_model, scoped_model, x, labels, graph, scope,
        warmups, repeats)

    terminal_speedup = (
        statistics.median(item["elapsed_ms"] for item in full_terminal) /
        statistics.median(item["elapsed_ms"] for item in scoped_terminal))
    train_speedup = (
        statistics.median(item["elapsed_ms"] for item in full_train) /
        statistics.median(item["elapsed_ms"] for item in scoped_train))
    db_ok = (correctness["db"]["max_abs"] < 5e-5 or
             correctness["db"]["relative_l2"] < 1e-4)
    payload = {
        "status": "pass" if (
            correctness["logits"]["relative_l2"] < 0.03 and
            correctness["dh"]["relative_l2"] < 0.05 and
            correctness["dw"]["relative_l2"] < 0.05 and db_ok) else "fail",
        "dataset": "igb-hom-small",
        "label_file": label_file,
        "output_dim": out_dim,
        "threads": threads,
        "layers": layers,
        "scoped_backward_order": os.environ.get(
            "TFS_SCOPE_BACKWARD_ORDER", "authority_active"),
        "scope": {
            "nodes": scope.node_count,
            "selected_nodes": scope.selected_count,
            "selected_node_ratio": scope.selected_ratio,
            "selected_entries_including_self": scope.selected_edge_count,
            "scope_build_ms_excluded_from_epoch": scope_build_ms,
        },
        "correctness": correctness,
        "terminal": {
            "authority_full": phase_summary(full_terminal),
            "supervision_scoped": phase_summary(scoped_terminal),
            "speedup": terminal_speedup,
            "authority_records": full_terminal,
            "supervision_records": scoped_terminal,
        },
        "train_step": {
            "authority_full": phase_summary(full_train),
            "supervision_scoped": phase_summary(scoped_train),
            "speedup": train_speedup,
            "authority_records": full_train,
            "supervision_records": scoped_train,
            "authority_losses": [record["loss"] for record in full_train],
            "scoped_losses": [record["loss"] for record in scoped_train],
        },
    }
    output = Path(os.environ.get(
        "SCOPE_OUTPUT", "supervision_scoped_igb.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
