#!/usr/bin/env python3
"""Products gate for the supervision-scoped terminal TFS shadow path."""

import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tfs_train.authority_model import HybridGCN
from tfs_train.datasets import NodePropertyDataset
from tfs_train.graph import CSRGraph
from tfs_train.supervision_scope import (
    SupervisionScope, terminal_logits, train_logits,
)


def load_products(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    graph = CSRGraph(payload["rowptr"], payload["colidx"], payload["scale"],
                     payload["degree"], payload["schedule"])
    return NodePropertyDataset(payload["x"], payload["labels"], graph,
                               payload["train_mask"], payload["valid_mask"],
                               payload["test_mask"])


def metric(a, b):
    diff = (a.float() - b.float()).reshape(-1)
    denom = torch.linalg.vector_norm(a.float().reshape(-1)).clamp_min(1e-12)
    return {
        "max_abs": float(diff.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(diff) / denom),
    }


def summary(values):
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(ordered),
        "mean_ms": statistics.mean(ordered),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def hidden_input(model, x, graph):
    value = x
    with torch.no_grad():
        for conv in model.convs[:-1]:
            value = F.relu(conv(value, graph))
            # Correctness/timing isolates the terminal chain and therefore
            # deliberately omits dropout here.
    return value.detach().contiguous()


def terminal_once(hidden, terminal, graph, scope, labels, scoped):
    terminal.zero_grad(set_to_none=True)
    h = hidden.detach().requires_grad_(True)
    t0 = time.perf_counter_ns()
    if scoped:
        logits = terminal_logits(h, terminal, graph, scope)
        selected = logits
    else:
        logits = terminal(h, graph)
        selected = logits[scope.row_ids]
    t1 = time.perf_counter_ns()
    loss = F.cross_entropy(selected, labels[scope.row_ids])
    t2 = time.perf_counter_ns()
    loss.backward()
    t3 = time.perf_counter_ns()
    return {
        "loss": float(loss.detach()),
        "logits": selected.detach(),
        "dh": h.grad.detach(),
        "dw": terminal.weight.grad.detach().clone(),
        "db": terminal.bias.grad.detach().clone(),
        "timing": {
            "forward_ms": (t1 - t0) / 1e6,
            "loss_ms": (t2 - t1) / 1e6,
            "backward_ms": (t3 - t2) / 1e6,
            "elapsed_ms": (t3 - t0) / 1e6,
        },
    }


def time_terminal(hidden, terminal, graph, scope, labels, scoped,
                  warmups, repeats):
    values = []
    for iteration in range(warmups + repeats):
        result = terminal_once(hidden, terminal, graph, scope, labels, scoped)
        if iteration >= warmups:
            values.append(result["timing"])
    return values


def train_once(model, x, labels, graph, scope, optimizer, scoped, seed):
    torch.manual_seed(seed)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    begin = time.perf_counter_ns()
    if scoped:
        logits = train_logits(model, x, graph, scope)
    else:
        logits = model(x, graph)
        logits = logits[scope.row_ids]
    forward_end = time.perf_counter_ns()
    loss = F.cross_entropy(logits, labels[scope.row_ids])
    loss_end = time.perf_counter_ns()
    loss.backward()
    backward_end = time.perf_counter_ns()
    optimizer.step()
    end = time.perf_counter_ns()
    return {
        "elapsed_ms": (end - begin) / 1e6,
        "forward_ms": (forward_end - begin) / 1e6,
        "loss_ms": (loss_end - forward_end) / 1e6,
        "backward_ms": (backward_end - loss_end) / 1e6,
        "optimizer_ms": (end - backward_end) / 1e6,
        "loss": float(loss.detach()),
    }


def time_training_paired(baseline_model, scoped_model, x, labels, graph,
                         scope, warmups, repeats):
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=0.01)
    scoped_optimizer = torch.optim.Adam(scoped_model.parameters(), lr=0.01)
    baseline_values, scoped_values = [], []
    for iteration in range(warmups + repeats):
        seed = 9000 + iteration
        if iteration % 2 == 0:
            baseline = train_once(baseline_model, x, labels, graph, scope,
                                  baseline_optimizer, False, seed)
            scoped = train_once(scoped_model, x, labels, graph, scope,
                                scoped_optimizer, True, seed)
        else:
            scoped = train_once(scoped_model, x, labels, graph, scope,
                                scoped_optimizer, True, seed)
            baseline = train_once(baseline_model, x, labels, graph, scope,
                                  baseline_optimizer, False, seed)
        if iteration >= warmups:
            baseline_values.append(baseline)
            scoped_values.append(scoped)
    return baseline_values, scoped_values


def phase_summary(records):
    return {
        key: summary([record[key] for record in records])
        for key in records[0] if key.endswith("_ms")
    }


def main():
    threads = int(os.environ.get("SCOPE_THREADS", "32"))
    layers = int(os.environ.get("SCOPE_LAYERS", "2"))
    warmups = int(os.environ.get("SCOPE_WARMUPS", "2"))
    repeats = int(os.environ.get("SCOPE_REPEATS", "5"))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    ds = load_products(os.environ["PRODUCTS_CACHE"])
    x, labels, graph = ds.x.contiguous(), ds.labels, ds.graph

    model = HybridGCN(threads, layers, dropout=0.5, in_dim=x.shape[1],
                      hidden_dim=128, out_dim=int(labels.max()) + 1,
                      num_nodes=x.shape[0])
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

    # Alternate order once before recorded loops so neither arm receives the
    # only cold invocation of the persistent worker pool.
    terminal_once(hidden, terminal, graph, scope, labels, True)
    terminal_once(hidden, terminal, graph, scope, labels, False)
    full_terminal = time_terminal(hidden, terminal, graph, scope, labels,
                                  False, warmups, repeats)
    scoped_terminal = time_terminal(hidden, terminal, graph, scope, labels,
                                    True, warmups, repeats)

    baseline_model = HybridGCN(
        threads, layers, dropout=0.5, in_dim=x.shape[1], hidden_dim=128,
        out_dim=int(labels.max()) + 1, num_nodes=x.shape[0])
    scoped_model = HybridGCN(
        threads, layers, dropout=0.5, in_dim=x.shape[1], hidden_dim=128,
        out_dim=int(labels.max()) + 1, num_nodes=x.shape[0])
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
    # Bias is a global reduction.  Different deterministic reduction trees can
    # make a tiny absolute discrepancy look large in relative terms when the
    # reference norm is small, so require either a strict absolute or relative
    # bound instead of rejecting on relative error alone.
    db_ok = (correctness["db"]["max_abs"] < 5e-5 or
             correctness["db"]["relative_l2"] < 1e-4)
    payload = {
        "status": "pass" if (
            correctness["logits"]["relative_l2"] < 0.03 and
            correctness["dh"]["relative_l2"] < 0.05 and
            correctness["dw"]["relative_l2"] < 0.05 and db_ok) else "fail",
        "dataset": "ogbn-products",
        "threads": threads,
        "layers": layers,
        "scoped_backward_order": os.environ.get(
            "TFS_SCOPE_BACKWARD_ORDER", "y_first"),
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
        },
        "train_step": {
            "authority_full": phase_summary(full_train),
            "supervision_scoped": phase_summary(scoped_train),
            "speedup": train_speedup,
            "authority_losses": [record["loss"] for record in full_train],
            "scoped_losses": [record["loss"] for record in scoped_train],
        },
        "gate": {
            "terminal_required": 1.30,
            "train_step_required": 1.10,
            "terminal_pass": terminal_speedup >= 1.30,
            "train_step_pass": train_speedup >= 1.10,
        },
    }
    output = Path(os.environ.get(
        "SCOPE_OUTPUT", "supervision_scoped_products.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
