#!/usr/bin/env python3
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from tfs_train.datasets import load_igb_homogeneous
from tfs_train.modules import TFSConvCSR


root = os.environ["IGB_ROOT"]
size = os.environ.get("IGB_SIZE", "medium")
dataset = load_igb_homogeneous(root, size=size)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
model = torch.nn.Sequential()  # Keep modules explicit because graph is an argument.
first = TFSConvCSR(dataset.x.shape[1], 128, runtime="reference")
last = TFSConvCSR(128, 19, runtime="reference")
optimizer = torch.optim.Adam([*first.parameters(), *last.parameters()], lr=.01)
optimizer.zero_grad(set_to_none=True)
logits = last(F.relu(first(dataset.x, dataset.graph)), dataset.graph)
loss = F.cross_entropy(logits[dataset.train_mask], dataset.labels[dataset.train_mask])
loss.backward(); optimizer.step()
result = {"status": "pass", "size": size, "nodes": int(dataset.x.shape[0]),
          "adjacency_entries": int(dataset.graph.colidx.numel()),
          "features": int(dataset.x.shape[1]), "classes": 19, "loss": float(loss)}
Path(os.environ.get("IGB_SMOKE_OUTPUT", "/tmp/tfs_igb_medium_smoke.json")).write_text(json.dumps(result) + "\n")
print(json.dumps(result))
