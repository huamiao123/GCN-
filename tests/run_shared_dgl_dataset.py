#!/usr/bin/env python3
"""Stock-DGL full-graph runner for the four GraphSAINT extension datasets."""
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tfs_train.datasets import load_graphsaint
from tfs_train.standard_dgl import DGLGCN, build_stock_dgl_graph_from_pull_csr
from tfs_train.standard_runtime import configure_dgl


def score(logits, labels, mask, mode):
    if mode == "multilabel":
        prediction = (logits[mask] >= 0).to(torch.float32)
        target = labels[mask]
        return float(2 * (prediction * target).sum() /
                     (prediction.sum() + target.sum()).clamp_min(1))
    return float((logits[mask].argmax(1) == labels[mask]).to(torch.float32).mean())


threads = int(os.environ["OMP_NUM_THREADS"])
layers = int(os.environ["HYBRID_LAYERS"])
epochs = int(os.environ.get("HYBRID_TRAIN_EPOCHS", "200"))
seed = int(os.environ.get("HYBRID_SEED", "101"))
torch.manual_seed(seed)
torch.set_num_threads(threads)
dataset = load_graphsaint(os.environ["TFS_DATASET_ROOT"])
if dataset.label_mode not in {"multiclass", "multilabel"}:
    raise ValueError(f"unsupported label mode {dataset.label_mode}")
import dgl
actual_dgl_threads = configure_dgl(dgl, threads)
graph, graph_metadata = build_stock_dgl_graph_from_pull_csr(
    dataset.graph.rowptr, dataset.graph.colidx, int(dataset.x.shape[0]))
out_dim = int(dataset.labels.shape[-1] if dataset.label_mode == "multilabel"
              else int(dataset.labels.max()) + 1)
model = DGLGCN(layers=layers, in_dim=int(dataset.x.shape[1]), hidden_dim=128,
               out_dim=out_dim, norm="both", dropout=.5)
optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=5e-4)
output = Path(os.environ["HYBRID_OUTPUT"])
output.parent.mkdir(parents=True, exist_ok=True)
fields = ("epoch", "train_step_ms", "evaluation_ms", "train_loss", "val_accuracy",
          "test_accuracy", "path", "dtype", "layers", "threads", "seed", "label_mode")
with output.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter_ns()
        optimizer.zero_grad(set_to_none=True)
        logits = model(dataset.x, (graph, None))
        loss = (F.binary_cross_entropy_with_logits(logits[dataset.train_mask], dataset.labels[dataset.train_mask])
                if dataset.label_mode == "multilabel" else
                F.cross_entropy(logits[dataset.train_mask], dataset.labels[dataset.train_mask]))
        loss.backward()
        optimizer.step()
        train_ms = (time.perf_counter_ns() - start) / 1e6
        model.eval()
        start = time.perf_counter_ns()
        with torch.no_grad():
            logits = model(dataset.x, (graph, None))
            valid = score(logits, dataset.labels, dataset.valid_mask, dataset.label_mode)
            test = score(logits, dataset.labels, dataset.test_mask, dataset.label_mode)
        writer.writerow({"epoch": epoch, "train_step_ms": train_ms,
                         "evaluation_ms": (time.perf_counter_ns() - start) / 1e6,
                         "train_loss": float(loss), "val_accuracy": valid,
                         "test_accuracy": test, "path": "dgl_stock", "dtype": "fp32",
                         "layers": layers, "threads": threads, "seed": seed,
                         "label_mode": dataset.label_mode})
        stream.flush()
Path(os.environ["TFS_SUMMARY_OUTPUT"]).write_text(json.dumps({
    "status": "success", "method": "dgl_stock", "dgl_threads": actual_dgl_threads,
    "dataset": os.environ["TFS_DATASET_NAME"], "nodes": int(dataset.x.shape[0]),
    "adjacency_entries": int(dataset.graph.colidx.numel()), "features": int(dataset.x.shape[1]),
    "classes": out_dim, "label_mode": dataset.label_mode, "layers": layers,
    "threads": threads, "epochs": epochs, "seed": seed, "dgl_graph": graph_metadata,
    "timing_protocol": "paper_v1_epoch_2_to_N_median"}, indent=2) + "\n")
