#!/usr/bin/env python3
"""End-to-end gate for bounded-panel CE on supervision-scoped IGB-small."""

import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from bench_supervision_scoped_products import hidden_input, metric, summary
from tfs_train.authority_model import HybridGCN
from tfs_train.datasets import load_igb_homogeneous
from tfs_train.supervision_scope import (
    SupervisionScope,
    terminal_cross_entropy,
    terminal_logits,
    train_cross_entropy,
    train_logits,
)


def terminal_once(hidden, terminal, graph, scope, labels, panelized, row_tile):
    terminal.zero_grad(set_to_none=True)
    h = hidden.detach().requires_grad_(True)
    begin = time.perf_counter_ns()
    if panelized:
        loss = terminal_cross_entropy(
            h, terminal, graph, scope, labels, row_tile)
        middle = time.perf_counter_ns()
    else:
        logits = terminal_logits(h, terminal, graph, scope)
        loss = F.cross_entropy(logits, labels[scope.row_ids])
        middle = time.perf_counter_ns()
    loss.backward()
    end = time.perf_counter_ns()
    return {
        "loss": float(loss.detach()),
        "dh": h.grad.detach(),
        "dw": terminal.weight.grad.detach().clone(),
        "db": terminal.bias.grad.detach().clone(),
        "fused_forward_loss_ms": (middle - begin) / 1e6,
        "backward_ms": (end - middle) / 1e6,
        "elapsed_ms": (end - begin) / 1e6,
    }


def train_once(model, x, labels, graph, scope, optimizer, panelized,
               row_tile, seed):
    torch.manual_seed(seed)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    begin = time.perf_counter_ns()
    if panelized:
        loss = train_cross_entropy(
            model, x, labels, graph, scope, row_tile)
    else:
        logits = train_logits(model, x, graph, scope)
        loss = F.cross_entropy(logits, labels[scope.row_ids])
    forward_loss_end = time.perf_counter_ns()
    loss.backward()
    backward_end = time.perf_counter_ns()
    optimizer.step()
    end = time.perf_counter_ns()
    return {
        "loss": float(loss.detach()),
        "fused_forward_loss_ms": (forward_loss_end - begin) / 1e6,
        "backward_ms": (backward_end - forward_loss_end) / 1e6,
        "optimizer_ms": (end - backward_end) / 1e6,
        "elapsed_ms": (end - begin) / 1e6,
    }


def record_summary(records):
    return {
        key: summary([record[key] for record in records])
        for key in records[0] if key.endswith("_ms")
    }


def timing_record(record):
    """Drop correctness tensors before retaining formal timing records."""
    return {
        key: value for key, value in record.items()
        if key == "loss" or key.endswith("_ms")
    }


def main():
    threads = int(os.environ.get("SCOPE_THREADS", "32"))
    layers = int(os.environ.get("SCOPE_LAYERS", "2"))
    row_tile = int(os.environ.get("TFS_SCOPE_LOSS_ROW_TILE", "300000"))
    warmups = int(os.environ.get("SCOPE_WARMUPS", "2"))
    repeats = int(os.environ.get("SCOPE_REPEATS", "10"))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    ds = load_igb_homogeneous(
        os.environ["IGB_ROOT"], size="small",
        label_file=os.environ.get("IGB_LABEL_FILE", "node_label_2K.npy"),
        split_seed=int(os.environ.get("IGB_SPLIT_SEED", "20260813")))
    x, labels, graph = ds.x.contiguous(), ds.labels, ds.graph
    out_dim = int(os.environ.get("SCOPE_OUT_DIM", "2983"))
    synthetic_labels = os.environ.get("SCOPE_SYNTHETIC_LABELS", "0") == "1"
    if synthetic_labels:
        # Controlled shape sweep only: preserve the real graph, features and
        # supervision mask while making the label range legal for an
        # artificial output width.  These runs measure execution boundaries,
        # never model quality.
        labels = labels.remainder(out_dim).contiguous()
    elif int(labels.max()) >= out_dim:
        raise ValueError(
            "SCOPE_OUT_DIM does not cover labels; controlled sweeps must set "
            "SCOPE_SYNTHETIC_LABELS=1")
    scope = SupervisionScope.build(ds.train_mask, graph, threads)

    model = HybridGCN(
        threads, layers, dropout=0.5, in_dim=x.shape[1], hidden_dim=128,
        out_dim=out_dim, num_nodes=x.shape[0])
    hidden = hidden_input(model, x, graph)
    terminal = model.convs[-1]
    os.environ["TFS_SCOPE_BACKWARD_ORDER"] = "q_first"
    current = terminal_once(
        hidden, terminal, graph, scope, labels, False, row_tile)
    bounded = terminal_once(
        hidden, terminal, graph, scope, labels, True, row_tile)
    correctness = {
        "loss_abs": abs(current["loss"] - bounded["loss"]),
        "dh": metric(current["dh"], bounded["dh"]),
        "dw": metric(current["dw"], bounded["dw"]),
        "db": metric(current["db"], bounded["db"]),
    }
    correctness_pass = (
        correctness["loss_abs"] < 1e-4 and
        correctness["dh"]["relative_l2"] < 0.03 and
        correctness["dw"]["relative_l2"] < 0.03 and
        (correctness["db"]["max_abs"] < 5e-5 or
         correctness["db"]["relative_l2"] < 1e-4))

    terminal_current, terminal_bounded = [], []
    for iteration in range(warmups + repeats):
        order = (False, True) if iteration % 2 == 0 else (True, False)
        values = {}
        for panelized in order:
            values[panelized] = terminal_once(
                hidden, terminal, graph, scope, labels, panelized, row_tile)
        if iteration >= warmups:
            terminal_current.append(timing_record(values[False]))
            terminal_bounded.append(timing_record(values[True]))

    current_model = HybridGCN(
        threads, layers, dropout=0.5, in_dim=x.shape[1], hidden_dim=128,
        out_dim=out_dim, num_nodes=x.shape[0])
    bounded_model = HybridGCN(
        threads, layers, dropout=0.5, in_dim=x.shape[1], hidden_dim=128,
        out_dim=out_dim, num_nodes=x.shape[0])
    bounded_model.load_state_dict(current_model.state_dict())
    current_opt = torch.optim.Adam(current_model.parameters(), lr=0.01)
    bounded_opt = torch.optim.Adam(bounded_model.parameters(), lr=0.01)
    train_current, train_bounded = [], []
    for iteration in range(warmups + repeats):
        order = (False, True) if iteration % 2 == 0 else (True, False)
        values = {}
        for panelized in order:
            values[panelized] = train_once(
                bounded_model if panelized else current_model,
                x, labels, graph, scope,
                bounded_opt if panelized else current_opt,
                panelized, row_tile, 12000 + iteration)
        if iteration >= warmups:
            train_current.append(values[False])
            train_bounded.append(values[True])

    terminal_speedup = (
        statistics.median(r["elapsed_ms"] for r in terminal_current) /
        statistics.median(r["elapsed_ms"] for r in terminal_bounded))
    terminal_paired_speedup = statistics.median(
        old["elapsed_ms"] / new["elapsed_ms"]
        for old, new in zip(terminal_current, terminal_bounded))
    train_speedup = (
        statistics.median(r["elapsed_ms"] for r in train_current) /
        statistics.median(r["elapsed_ms"] for r in train_bounded))
    train_paired_speedup = statistics.median(
        old["elapsed_ms"] / new["elapsed_ms"]
        for old, new in zip(train_current, train_bounded))
    payload = {
        "status": "pass" if correctness_pass else "fail",
        "contract": "supervision_scoped_bounded_terminal_ce_shadow_v1",
        "dataset": "igb-hom-small",
        "output_dim": out_dim,
        "synthetic_labels_for_shape_sweep": synthetic_labels,
        "layers": layers,
        "threads": threads,
        "row_tile": row_tile,
        "correctness": correctness,
        "terminal": {
            "current": record_summary(terminal_current),
            "bounded_panel": record_summary(terminal_bounded),
            "speedup": terminal_speedup,
            "paired_speedup_median": terminal_paired_speedup,
            "current_records": terminal_current,
            "bounded_records": terminal_bounded,
        },
        "train_step": {
            "current": record_summary(train_current),
            "bounded_panel": record_summary(train_bounded),
            "speedup": train_speedup,
            "paired_speedup_median": train_paired_speedup,
            "current_records": train_current,
            "bounded_records": train_bounded,
        },
    }
    output = Path(os.environ.get(
        "SCOPE_OUTPUT", "panelized_scope_igb.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    if not correctness_pass:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
