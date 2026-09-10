#!/usr/bin/env python3
"""Real-graph A/B for the optional native compact-dW shadow path."""

import json
import os
import statistics
from pathlib import Path

import torch

from bench_panelized_scope_igb import (
    metric, record_summary, terminal_once, timing_record, train_once,
)
from bench_supervision_scoped_products import hidden_input
from tfs_train.authority_model import HybridGCN
from tfs_train.datasets import load_igb_homogeneous
from tfs_train.supervision_scope import SupervisionScope


def use_native(enabled):
    kind = os.environ.get("SCOPE_AB_KIND", "dw")
    if kind == "dw":
        os.environ["TFS_SCOPE_NATIVE_DW"] = "1" if enabled else "0"
        os.environ["TFS_SCOPE_NATIVE_LOGITS"] = "0"
    elif kind == "logits":
        os.environ["TFS_SCOPE_NATIVE_DW"] = os.environ.get(
            "SCOPE_NATIVE_DW_WITH_LOGITS", "1")
        os.environ["TFS_SCOPE_NATIVE_LOGITS"] = "1" if enabled else "0"
    else:
        raise ValueError("SCOPE_AB_KIND must be dw or logits")


def speedup(old, new):
    ratio = (statistics.median(x["elapsed_ms"] for x in old) /
             statistics.median(x["elapsed_ms"] for x in new))
    paired = statistics.median(
        a["elapsed_ms"] / b["elapsed_ms"] for a, b in zip(old, new))
    return ratio, paired


def main():
    threads = int(os.environ.get("SCOPE_THREADS", "32"))
    layers = int(os.environ.get("SCOPE_LAYERS", "2"))
    out_dim = int(os.environ.get("SCOPE_OUT_DIM", "2983"))
    row_tile = int(os.environ.get("TFS_SCOPE_LOSS_ROW_TILE", "300000"))
    warmups = int(os.environ.get("SCOPE_WARMUPS", "2"))
    repeats = int(os.environ.get("SCOPE_REPEATS", "7"))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    ds = load_igb_homogeneous(
        os.environ["IGB_ROOT"], size="small",
        label_file=os.environ.get("IGB_LABEL_FILE", "node_label_2K.npy"),
        split_seed=int(os.environ.get("IGB_SPLIT_SEED", "20260813")))
    x, labels, graph = ds.x.contiguous(), ds.labels, ds.graph
    synthetic = os.environ.get("SCOPE_SYNTHETIC_LABELS", "0") == "1"
    if synthetic:
        labels = labels.remainder(out_dim).contiguous()
    scope = SupervisionScope.build(ds.train_mask, graph, threads)
    model = HybridGCN(
        threads, layers, dropout=0.5, in_dim=x.shape[1], hidden_dim=128,
        out_dim=out_dim, num_nodes=x.shape[0])
    hidden = hidden_input(model, x, graph)
    terminal = model.convs[-1]
    os.environ["TFS_SCOPE_BACKWARD_ORDER"] = "q_first"

    use_native(False)
    old = terminal_once(hidden, terminal, graph, scope, labels, True, row_tile)
    use_native(True)
    new = terminal_once(hidden, terminal, graph, scope, labels, True, row_tile)
    correctness = {
        "loss_abs": abs(old["loss"] - new["loss"]),
        "dh": metric(old["dh"], new["dh"]),
        "dw": metric(old["dw"], new["dw"]),
        "db": metric(old["db"], new["db"]),
    }
    passed = (correctness["loss_abs"] < 1e-6 and
              correctness["dh"]["relative_l2"] < 1e-4 and
              correctness["dw"]["relative_l2"] < 0.01 and
              correctness["db"]["relative_l2"] < 1e-6)

    terminal_old, terminal_new = [], []
    for iteration in range(warmups + repeats):
        values = {}
        order = (False, True) if iteration % 2 == 0 else (True, False)
        for enabled in order:
            use_native(enabled)
            values[enabled] = terminal_once(
                hidden, terminal, graph, scope, labels, True, row_tile)
        if iteration >= warmups:
            terminal_old.append(timing_record(values[False]))
            terminal_new.append(timing_record(values[True]))

    old_model = HybridGCN(
        threads, layers, dropout=0.5, in_dim=x.shape[1], hidden_dim=128,
        out_dim=out_dim, num_nodes=x.shape[0])
    new_model = HybridGCN(
        threads, layers, dropout=0.5, in_dim=x.shape[1], hidden_dim=128,
        out_dim=out_dim, num_nodes=x.shape[0])
    new_model.load_state_dict(old_model.state_dict())
    old_opt = torch.optim.Adam(old_model.parameters(), lr=0.01)
    new_opt = torch.optim.Adam(new_model.parameters(), lr=0.01)
    train_old, train_new = [], []
    for iteration in range(warmups + repeats):
        values = {}
        order = (False, True) if iteration % 2 == 0 else (True, False)
        for enabled in order:
            use_native(enabled)
            values[enabled] = train_once(
                new_model if enabled else old_model, x, labels, graph, scope,
                new_opt if enabled else old_opt, True, row_tile,
                16000 + iteration)
        if iteration >= warmups:
            train_old.append(values[False])
            train_new.append(values[True])

    terminal_ratio, terminal_paired = speedup(terminal_old, terminal_new)
    train_ratio, train_paired = speedup(train_old, train_new)
    ab_kind = os.environ.get("SCOPE_AB_KIND", "dw")
    payload = {
        "status": "pass" if passed else "fail",
        "contract": f"real_igb_bounded_{ab_kind}_ab_shadow_v1",
        "ab_kind": ab_kind,
        "output_dim": out_dim,
        "threads": threads,
        "layers": layers,
        "row_tile": row_tile,
        "correctness": correctness,
        "terminal": {
            "torch_dw": record_summary(terminal_old),
            "native_dw": record_summary(terminal_new),
            "speedup": terminal_ratio,
            "paired_speedup_median": terminal_paired,
        },
        "train_step": {
            "torch_dw": record_summary(train_old),
            "native_dw": record_summary(train_new),
            "speedup": train_ratio,
            "paired_speedup_median": train_paired,
        },
    }
    output = Path(os.environ["SCOPE_OUTPUT"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
