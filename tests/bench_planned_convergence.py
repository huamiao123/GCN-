#!/usr/bin/env python3
"""Paired convergence check for authority and planned shadow training."""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from bench_supervision_scoped_products import load_products
from tfs_train.authority_model import HybridGCN
from tfs_train.datasets import load_igb_homogeneous
from tfs_train.supervision_dense_plan import plan_supervision_dense
from tfs_train.supervision_scope import SupervisionScope, train_cross_entropy


def load_dataset(name: str):
    if name == "igb-hom-small":
        dataset = load_igb_homogeneous(
            os.environ["IGB_ROOT"], size="small",
            label_file="node_label_2K.npy", split_seed=20260813)
        return dataset, 2983
    if name == "ogbn-products":
        dataset = load_products(os.environ["PRODUCTS_CACHE"])
        return dataset, int(dataset.labels.max()) + 1
    raise ValueError(f"unsupported dataset: {name}")


def train_step(model, optimizer, x, labels, graph, scope, plan, seed):
    torch.manual_seed(seed)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    begin = time.perf_counter_ns()
    if plan is None:
        logits = model(x, graph)
        loss = F.cross_entropy(logits[scope.row_ids], labels[scope.row_ids])
    else:
        loss = train_cross_entropy(
            model, x, labels, graph, scope, dense_plan=plan)
    loss.backward()
    optimizer.step()
    elapsed_ms = (time.perf_counter_ns() - begin) / 1e6
    return float(loss.detach()), elapsed_ms


@torch.no_grad()
def evaluate(model, x, labels, graph, masks):
    model.eval()
    logits = model(x, graph).float()
    values = {}
    for name, mask in masks.items():
        selected = logits[mask]
        target = labels[mask]
        values[name + "_loss"] = float(F.cross_entropy(selected, target))
        values[name + "_accuracy"] = float(
            (selected.argmax(dim=1) == target).float().mean())
    return values


def main():
    dataset_name = os.environ.get("SCOPE_DATASET", "igb-hom-small")
    threads = int(os.environ.get("SCOPE_THREADS", "32"))
    layers = int(os.environ.get("SCOPE_LAYERS", "2"))
    epochs = int(os.environ.get("SCOPE_EPOCHS", "200"))
    eval_interval = int(os.environ.get("SCOPE_EVAL_INTERVAL", "20"))
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    ds, output_dim = load_dataset(dataset_name)
    x, labels, graph = ds.x.contiguous(), ds.labels, ds.graph
    scope = SupervisionScope.build(ds.train_mask, graph, threads)
    plan = plan_supervision_dense(
        scope.selected_count, 128, output_dim, threads)

    torch.manual_seed(20260910)
    authority = HybridGCN(
        threads, layers, dropout=0.5, in_dim=x.shape[1], hidden_dim=128,
        out_dim=output_dim, num_nodes=x.shape[0])
    planned = HybridGCN(
        threads, layers, dropout=0.5, in_dim=x.shape[1], hidden_dim=128,
        out_dim=output_dim, num_nodes=x.shape[0])
    planned.load_state_dict(authority.state_dict())
    authority_opt = torch.optim.Adam(authority.parameters(), lr=0.01)
    planned_opt = torch.optim.Adam(planned.parameters(), lr=0.01)
    records = []
    evaluations = []
    masks = {"validation": ds.valid_mask, "test": ds.test_mask}

    for epoch in range(1, epochs + 1):
        seed = 24000 + epoch
        order = ("authority", "planned") if epoch % 2 else (
            "planned", "authority")
        values = {}
        for arm in order:
            if arm == "authority":
                values[arm] = train_step(
                    authority, authority_opt, x, labels, graph, scope, None,
                    seed)
            else:
                values[arm] = train_step(
                    planned, planned_opt, x, labels, graph, scope, plan, seed)
        records.append({
            "epoch": epoch,
            "authority_loss": values["authority"][0],
            "planned_loss": values["planned"][0],
            "authority_ms": values["authority"][1],
            "planned_ms": values["planned"][1],
        })
        if epoch == 1 or epoch % eval_interval == 0 or epoch == epochs:
            evaluations.append({
                "epoch": epoch,
                "authority": evaluate(authority, x, labels, graph, masks),
                "planned": evaluate(planned, x, labels, graph, masks),
            })

    steady = records[1:]
    authority_ms = statistics.median(r["authority_ms"] for r in steady)
    planned_ms = statistics.median(r["planned_ms"] for r in steady)
    final_eval = evaluations[-1]
    finite = all(math.isfinite(value) for record in records for value in (
        record["authority_loss"], record["planned_loss"]))
    val_delta = abs(
        final_eval["authority"]["validation_accuracy"] -
        final_eval["planned"]["validation_accuracy"])
    test_delta = abs(
        final_eval["authority"]["test_accuracy"] -
        final_eval["planned"]["test_accuracy"])
    final_loss_rel = abs(
        records[-1]["authority_loss"] - records[-1]["planned_loss"]
    ) / max(abs(records[-1]["authority_loss"]), 1e-12)
    passed = finite and val_delta <= 0.02 and test_delta <= 0.02 and (
        final_loss_rel <= 0.05)
    payload = {
        "status": "pass" if passed else "fail",
        "contract": "paired_200epoch_authority_vs_planned_shadow_v1",
        "dataset": dataset_name,
        "layers": layers,
        "threads": threads,
        "epochs": epochs,
        "output_dim": output_dim,
        "row_tile": plan.row_tile,
        "plan": plan.__dict__,
        "steady_authority_median_ms": authority_ms,
        "steady_planned_median_ms": planned_ms,
        "steady_speedup": authority_ms / planned_ms,
        "final_loss_relative_delta": final_loss_rel,
        "validation_accuracy_abs_delta": val_delta,
        "test_accuracy_abs_delta": test_delta,
        "evaluations": evaluations,
        "epoch_records": records,
    }
    output = Path(os.environ["SCOPE_OUTPUT"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "epoch_records"}, indent=2), flush=True)
    if not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
