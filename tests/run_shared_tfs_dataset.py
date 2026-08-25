#!/usr/bin/env python3
"""Shared-node full training runner for the five validated extension graphs."""
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tfs_train.authority_model import HybridGCN
from tfs_train.datasets import load_graphsaint, load_igb_homogeneous


def score(logits, labels, mask, mode):
    if mode == "multilabel":
        pred = (logits[mask] >= 0).to(torch.float32); target = labels[mask]
        return float(2 * (pred * target).sum() / (pred.sum() + target.sum()).clamp_min(1))
    return float((logits[mask].argmax(1) == labels[mask]).to(torch.float32).mean())


kind = os.environ["TFS_DATASET_KIND"]
torch.manual_seed(int(os.environ.get("HYBRID_SEED", "101")))
if kind == "graphsaint":
    dataset = load_graphsaint(os.environ["TFS_DATASET_ROOT"])
elif kind == "igb":
    dataset = load_igb_homogeneous(os.environ["TFS_DATASET_ROOT"],
                                   size=os.environ.get("TFS_IGB_SIZE", "medium"))
else:
    raise ValueError("TFS_DATASET_KIND must be graphsaint or igb")
threads, layers = int(os.environ["OMP_NUM_THREADS"]), int(os.environ["HYBRID_LAYERS"])
epochs, seed = int(os.environ.get("HYBRID_TRAIN_EPOCHS", "200")), int(os.environ.get("HYBRID_SEED", "101"))
torch.manual_seed(seed); torch.set_num_threads(threads)
out_dim = int(dataset.labels.shape[-1] if dataset.label_mode == "multilabel"
              else int(dataset.labels.max()) + 1)
model = HybridGCN(threads, layers, dropout=0.5, in_dim=int(dataset.x.shape[1]),
                  hidden_dim=128, out_dim=out_dim, num_nodes=int(dataset.x.shape[0]))
optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=5e-4)
output = Path(os.environ["HYBRID_OUTPUT"]); output.parent.mkdir(parents=True, exist_ok=True)
fields = ("epoch", "train_step_ms", "evaluation_ms", "train_loss", "val_accuracy", "test_accuracy",
          "path", "dtype", "layers", "threads", "seed", "label_mode")
with output.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
    for epoch in range(1, epochs + 1):
        model.train(); start = time.perf_counter_ns(); optimizer.zero_grad(set_to_none=True)
        logits = model(dataset.x, dataset.graph)
        loss = (F.binary_cross_entropy_with_logits(logits[dataset.train_mask], dataset.labels[dataset.train_mask])
                if dataset.label_mode == "multilabel" else
                F.cross_entropy(logits[dataset.train_mask], dataset.labels[dataset.train_mask]))
        loss.backward(); optimizer.step(); train_ms = (time.perf_counter_ns() - start) / 1e6
        model.eval(); start = time.perf_counter_ns()
        with torch.no_grad():
            logits = model(dataset.x, dataset.graph)
            valid, test = score(logits, dataset.labels, dataset.valid_mask, dataset.label_mode), score(logits, dataset.labels, dataset.test_mask, dataset.label_mode)
        writer.writerow({"epoch": epoch, "train_step_ms": train_ms,
                         "evaluation_ms": (time.perf_counter_ns() - start) / 1e6,
                         "train_loss": float(loss), "val_accuracy": valid, "test_accuracy": test,
                         "path": "hybrid", "dtype": "bf16_internal_fp32_master", "layers": layers,
                         "threads": threads, "seed": seed, "label_mode": dataset.label_mode})
        stream.flush()
summary = {"status": "success", "dataset_kind": kind, "nodes": int(dataset.x.shape[0]),
           "adjacency_entries": int(dataset.graph.colidx.numel()), "features": int(dataset.x.shape[1]),
           "classes": out_dim, "label_mode": dataset.label_mode, "epochs": epochs,
           "layers": layers, "threads": threads, "seed": seed,
           "timing_protocol": "paper_v1_epoch_2_to_N_median"}
Path(os.environ["TFS_SUMMARY_OUTPUT"]).write_text(json.dumps(summary, indent=2) + "\n")
