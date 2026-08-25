#!/usr/bin/env python3
"""One-step real-graph GraphSAINT compatibility smoke; not a benchmark."""
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from tfs_train.datasets import load_graphsaint
from tfs_train.modules import TFSConvCSR


class SmokeGCN(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.first = TFSConvCSR(in_dim, 128, runtime="reference")
        self.last = TFSConvCSR(128, out_dim, runtime="reference")

    def forward(self, x, graph):
        return self.last(F.relu(self.first(x, graph)), graph)


root = Path(os.environ["GRAPHSAINT_ROOT"])
name = os.environ.get("GRAPHSAINT_NAME", root.name)
threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
torch.set_num_threads(threads)
dataset = load_graphsaint(root)
out_dim = int(dataset.labels.shape[-1] if dataset.label_mode == "multilabel"
              else int(dataset.labels.max()) + 1)
model = SmokeGCN(int(dataset.x.shape[1]), out_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer.zero_grad(set_to_none=True)
logits = model(dataset.x, dataset.graph)
if dataset.label_mode == "multilabel":
    loss = F.binary_cross_entropy_with_logits(
        logits[dataset.train_mask], dataset.labels[dataset.train_mask])
    prediction = (logits[dataset.test_mask] >= 0).to(torch.float32)
    target = dataset.labels[dataset.test_mask]
    metric = (2 * (prediction * target).sum() /
              (prediction.sum() + target.sum()).clamp_min(1)).item()
    metric_name = "micro_f1"
else:
    loss = F.cross_entropy(logits[dataset.train_mask], dataset.labels[dataset.train_mask])
    metric = (logits[dataset.test_mask].argmax(1) ==
              dataset.labels[dataset.test_mask]).to(torch.float32).mean().item()
    metric_name = "accuracy"
loss.backward(); optimizer.step()
result = {"status": "pass", "dataset": name, "nodes": int(dataset.x.shape[0]),
          "adjacency_entries": int(dataset.graph.colidx.numel()),
          "features": int(dataset.x.shape[1]), "classes": out_dim,
          "label_mode": dataset.label_mode, "loss": float(loss),
          metric_name: metric, "threads": threads}
path = Path(os.environ.get("GRAPHSAINT_SMOKE_OUTPUT", f"/tmp/{name}_smoke.json"))
path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
