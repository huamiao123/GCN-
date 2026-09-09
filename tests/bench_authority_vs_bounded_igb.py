#!/usr/bin/env python3
"""Direct authority-TFS versus complete bounded terminal dataflow gate.

The historical filename is retained for provenance, but the benchmark accepts
both IGB-small and Products through one shared implementation.
"""

import json
import os
import resource
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from bench_supervision_scoped_products import (load_products, metric,
                                                phase_summary)
from tfs_train.authority_model import HybridGCN
from tfs_train.datasets import load_igb_homogeneous
from tfs_train.supervision_scope import (SupervisionScope,
                                         terminal_cross_entropy,
                                         train_cross_entropy)


def terminal_once(hidden, terminal, graph, scope, labels, bounded, row_tile):
    terminal.zero_grad(set_to_none=True)
    h = hidden.detach().requires_grad_(True)
    begin = time.perf_counter_ns()
    if bounded:
        loss = terminal_cross_entropy(h, terminal, graph, scope, labels,
                                      row_tile)
    else:
        logits = terminal(h, graph)
        loss = F.cross_entropy(logits[scope.row_ids], labels[scope.row_ids])
    middle = time.perf_counter_ns()
    loss.backward()
    end = time.perf_counter_ns()
    return {"loss": float(loss.detach()), "dh": h.grad.detach(),
            "dw": terminal.weight.grad.detach().clone(),
            "db": terminal.bias.grad.detach().clone(),
            "forward_loss_ms": (middle - begin) / 1e6,
            "backward_ms": (end - middle) / 1e6,
            "elapsed_ms": (end - begin) / 1e6}


def train_once(model, x, labels, graph, scope, optimizer, bounded, row_tile,
               seed):
    torch.manual_seed(seed)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    begin = time.perf_counter_ns()
    if bounded:
        loss = train_cross_entropy(model, x, labels, graph, scope, row_tile)
    else:
        logits = model(x, graph)
        loss = F.cross_entropy(logits[scope.row_ids], labels[scope.row_ids])
    middle = time.perf_counter_ns()
    loss.backward()
    backward_end = time.perf_counter_ns()
    usage_before_optimizer = resource.getrusage(resource.RUSAGE_SELF)
    optimizer.step()
    end = time.perf_counter_ns()
    usage_after_optimizer = resource.getrusage(resource.RUSAGE_SELF)
    return {"loss": float(loss.detach()),
            "forward_loss_ms": (middle - begin) / 1e6,
            "backward_ms": (backward_end - middle) / 1e6,
            "optimizer_ms": (end - backward_end) / 1e6,
            "optimizer_minor_faults":
                usage_after_optimizer.ru_minflt -
                usage_before_optimizer.ru_minflt,
            "optimizer_major_faults":
                usage_after_optimizer.ru_majflt -
                usage_before_optimizer.ru_majflt,
            "optimizer_involuntary_context_switches":
                usage_after_optimizer.ru_nivcsw -
                usage_before_optimizer.ru_nivcsw,
            "elapsed_ms": (end - begin) / 1e6}


def strip(record):
    return {k: v for k, v in record.items()
            if k == "loss" or k.endswith("_ms") or
            k.startswith("optimizer_")}


def median_ratio(old_records, new_records, selector):
    old_values = [selector(record) for record in old_records]
    new_values = [selector(record) for record in new_records]
    return {
        "ratio_of_medians": statistics.median(old_values) /
        statistics.median(new_values),
        "paired_ratio_median": statistics.median(
            old / new for old, new in zip(old_values, new_values)),
    }


def main():
    threads = int(os.environ.get("SCOPE_THREADS", "32"))
    layers = int(os.environ.get("SCOPE_LAYERS", "2"))
    warmups = int(os.environ.get("SCOPE_WARMUPS", "1"))
    repeats = int(os.environ.get("SCOPE_REPEATS", "7"))
    row_tile = int(os.environ.get("TFS_SCOPE_LOSS_ROW_TILE", "300000"))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    dataset = os.environ.get("SCOPE_DATASET", "igb-hom-small")
    if dataset == "igb-hom-small":
        ds = load_igb_homogeneous(os.environ["IGB_ROOT"], size="small",
                                  label_file="node_label_2K.npy",
                                  split_seed=20260813)
        out_dim = 2983
    elif dataset == "ogbn-products":
        ds = load_products(os.environ["PRODUCTS_CACHE"])
        out_dim = int(ds.labels.max()) + 1
    else:
        raise ValueError(f"unsupported SCOPE_DATASET={dataset!r}")
    x, labels, graph = ds.x.contiguous(), ds.labels, ds.graph
    scope = SupervisionScope.build(ds.train_mask, graph, threads)
    base = HybridGCN(threads, layers, dropout=0.5, in_dim=x.shape[1],
                     hidden_dim=128, out_dim=out_dim, num_nodes=x.shape[0])
    candidate = HybridGCN(threads, layers, dropout=0.5, in_dim=x.shape[1],
                          hidden_dim=128, out_dim=out_dim,
                          num_nodes=x.shape[0])
    candidate.load_state_dict(base.state_dict())
    with torch.no_grad():
        hidden = x
        for conv in base.convs[:-1]:
            hidden = F.relu(conv(hidden, graph))
        hidden = hidden.detach().contiguous()
    ref = terminal_once(hidden, base.convs[-1], graph, scope, labels, False,
                        row_tile)
    new = terminal_once(hidden, candidate.convs[-1], graph, scope, labels,
                        True, row_tile)
    correctness = {"loss_abs": abs(ref["loss"] - new["loss"]),
                   "dh": metric(ref["dh"], new["dh"]),
                   "dw": metric(ref["dw"], new["dw"]),
                   "db": metric(ref["db"], new["db"])}
    passed = (correctness["loss_abs"] < 1e-4 and
              correctness["dh"]["relative_l2"] < 0.03 and
              correctness["dw"]["relative_l2"] < 0.03 and
              (correctness["db"]["max_abs"] < 5e-5 or
               correctness["db"]["relative_l2"] < 1e-4))
    base_opt = torch.optim.Adam(base.parameters(), lr=0.01)
    new_opt = torch.optim.Adam(candidate.parameters(), lr=0.01)
    old_records, new_records = [], []
    for iteration in range(warmups + repeats):
        order = (False, True) if iteration % 2 == 0 else (True, False)
        values = {}
        for bounded in order:
            values[bounded] = train_once(
                candidate if bounded else base, x, labels, graph, scope,
                new_opt if bounded else base_opt, bounded, row_tile,
                19000 + iteration)
        if iteration >= warmups:
            old_records.append(strip(values[False]))
            new_records.append(strip(values[True]))
    end_to_end_speedup = median_ratio(
        old_records, new_records, lambda record: record["elapsed_ms"])
    compute_speedup = median_ratio(
        old_records, new_records,
        lambda record: record["forward_loss_ms"] + record["backward_ms"])
    old_optimizer_median = statistics.median(
        record["optimizer_ms"] for record in old_records)
    new_optimizer_median = statistics.median(
        record["optimizer_ms"] for record in new_records)
    optimizer_contaminated = (
        new_optimizer_median > max(10.0, 5.0 * old_optimizer_median))
    payload = {"status": "pass" if passed else "fail",
               "contract": "authority_vs_complete_bounded_shadow_v1",
               "dataset": dataset, "threads": threads,
               "layers": layers, "output_dim": out_dim,
               "row_tile": row_tile,
               "correctness": correctness,
               "authority": phase_summary(old_records),
               "complete_bounded": phase_summary(new_records),
               "speedup": end_to_end_speedup["ratio_of_medians"],
               "paired_speedup_median":
                   end_to_end_speedup["paired_ratio_median"],
               "compute_speedup": compute_speedup,
               "optimizer_contaminated": optimizer_contaminated,
               "authority_records": old_records,
               "bounded_records": new_records}
    output = Path(os.environ["SCOPE_OUTPUT"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
