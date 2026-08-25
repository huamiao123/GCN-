from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .graph import CSRGraph, preprocess_undirected_fast


def _load_or_save_csr(cache_path, build):
    """Persist static CSR preprocessing outside repeated training processes."""
    if cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        return CSRGraph(payload["rowptr"], payload["colidx"], payload["scale"],
                        payload["degree"], payload["schedule"])
    graph = build()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rowptr": graph.rowptr, "colidx": graph.colidx,
                "scale": graph.scale, "degree": graph.degree,
                "schedule": graph.schedule}, cache_path)
    return graph


@dataclass(frozen=True)
class NodePropertyDataset:
    x: torch.Tensor
    labels: torch.Tensor
    graph: object
    train_mask: torch.Tensor
    valid_mask: torch.Tensor
    test_mask: torch.Tensor
    label_mode: str = "multiclass"


def _csv_tensor(path, dtype):
    values = pd.read_csv(path, header=None, compression="gzip").values
    return torch.as_tensor(values, dtype=dtype)


def load_ogbn_arxiv_raw(arxiv_root):
    """Load the official OGB arxiv raw/split archive without network access."""
    root = Path(arxiv_root)
    raw, split = root / "raw", root / "split" / "time"
    x = _csv_tensor(raw / "node-feat.csv.gz", torch.float32)
    labels = _csv_tensor(raw / "node-label.csv.gz", torch.long).flatten()
    edge_index = _csv_tensor(raw / "edge.csv.gz", torch.long).t().contiguous()
    graph = preprocess_undirected_fast(edge_index, x.shape[0], torch.float32)
    masks = []
    for name in ("train", "valid", "test"):
        idx = _csv_tensor(split / f"{name}.csv.gz", torch.long).flatten()
        mask = torch.zeros(x.shape[0], dtype=torch.bool); mask[idx] = True
        masks.append(mask)
    return NodePropertyDataset(x, labels, graph, *masks)


def load_graphsaint(root):
    """Load an official GraphSAINT full-graph bundle without changing labels.

    ``adj_full.npz`` already stores both directions for these released
    undirected graphs.  Re-running the generic OGB undirecting preprocessor
    would needlessly duplicate its edge stream, so this adapter builds the
    canonical pull CSR directly after removing explicit diagonal entries.
    """
    from scipy.sparse import load_npz

    root = Path(root)
    required = ("adj_full.npz", "feats.npy", "class_map.json", "role.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"GraphSAINT bundle missing: {', '.join(missing)}")
    features = np.load(root / "feats.npy", mmap_mode="r")
    if features.ndim != 2:
        raise ValueError("GraphSAINT features must be rank-2")
    n = int(features.shape[0])
    # Copy into writable FP32 storage once.  The official Reddit/Flickr/Yelp
    # bundles are float64, while Amazon is already float32.
    x = torch.from_numpy(np.array(features, dtype=np.float32, copy=True))
    def build_graph():
        adj = load_npz(root / "adj_full.npz").tocsr()
        if adj.shape != (n, n):
            raise ValueError(f"GraphSAINT adjacency shape {adj.shape} != ({n}, {n})")
        adj.setdiag(0); adj.eliminate_zeros(); adj.sort_indices()
        if not adj.has_canonical_format:
            adj.sum_duplicates(); adj.sort_indices()
        rowptr = torch.from_numpy(np.asarray(adj.indptr, dtype=np.int64).copy())
        colidx = torch.from_numpy(np.asarray(adj.indices, dtype=np.int64).copy())
        degree = rowptr[1:] - rowptr[:-1]
        return CSRGraph(rowptr, colidx, torch.rsqrt(degree.to(torch.float32) + 1),
                        degree, torch.argsort(degree, descending=True, stable=True))
    graph = _load_or_save_csr(root / "tfs_graph_csr_v1.pt", build_graph)
    class_map = json.loads((root / "class_map.json").read_text())
    if len(class_map) != n:
        raise ValueError(f"GraphSAINT class_map has {len(class_map)} labels for {n} nodes")
    first = next(iter(class_map.values()))
    if isinstance(first, list):
        width = len(first)
        labels = torch.empty((n, width), dtype=torch.float32)
        for node, value in class_map.items():
            if len(value) != width:
                raise ValueError("inconsistent GraphSAINT multi-label width")
            labels[int(node)] = torch.tensor(value, dtype=torch.float32)
        label_mode = "multilabel"
    else:
        labels = torch.empty(n, dtype=torch.long)
        for node, value in class_map.items():
            labels[int(node)] = int(value)
        label_mode = "multiclass"
    role = json.loads((root / "role.json").read_text())
    masks = []
    for key in ("tr", "va", "te"):
        indices = torch.as_tensor(role[key], dtype=torch.long)
        mask = torch.zeros(n, dtype=torch.bool); mask[indices] = True
        masks.append(mask)
    return NodePropertyDataset(x, labels, graph, *masks, label_mode=label_mode)


def load_igb_homogeneous(root, size="small", label_file="node_label_19.npy",
                         split_seed=20260813):
    """Load an IGB homogeneous tier with the common deterministic split."""
    processed = Path(root) / size / "processed"
    paper = processed / "paper"
    # A writable mapping avoids PyTorch's read-only NumPy alias warning; the
    # loader never mutates features, so this does not alter the source file.
    x_np = np.load(paper / "node_feat.npy", mmap_mode="r+")
    labels_np = np.load(paper / label_file, mmap_mode="r+")
    edges_np = np.load(processed / "paper__cites__paper" / "edge_index.npy",
                       mmap_mode="r")
    if (x_np.ndim != 2 or x_np.dtype != np.float32 or labels_np.shape != (x_np.shape[0],)
            or edges_np.ndim != 2 or edges_np.shape[1] != 2):
        raise ValueError("invalid IGB homogeneous bundle")
    x = torch.from_numpy(x_np)
    labels = torch.from_numpy(labels_np).to(torch.long)
    graph = _load_or_save_csr(
        processed / "tfs_graph_csr_v1.pt",
        lambda: preprocess_undirected_fast(torch.from_numpy(edges_np.T.copy()).long(),
                                            int(x.shape[0]), torch.float32))
    generator = torch.Generator(device="cpu").manual_seed(int(split_seed))
    permutation = torch.randperm(x.shape[0], generator=generator)
    train_end, valid_end = int(x.shape[0] * .6), int(x.shape[0] * .8)
    masks = []
    for indices in (permutation[:train_end], permutation[train_end:valid_end],
                    permutation[valid_end:]):
        mask = torch.zeros(x.shape[0], dtype=torch.bool); mask[indices] = True
        masks.append(mask)
    return NodePropertyDataset(x, labels, graph, *masks)
