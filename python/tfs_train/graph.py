from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class CSRGraph:
    rowptr: torch.Tensor
    colidx: torch.Tensor
    scale: torch.Tensor
    degree: torch.Tensor
    schedule: torch.Tensor


def preprocess_undirected(edge_index, num_nodes, dtype=torch.float32, make_undirected=True):
    edge_index = torch.as_tensor(edge_index, dtype=torch.long, device="cpu")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    pairs = set()
    for u, v in edge_index.t().tolist():
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError("node id out of range")
        if u == v:
            continue
        pairs.add((u, v))
        if make_undirected:
            pairs.add((v, u))
    pairs = sorted(pairs)
    degree = torch.zeros(num_nodes, dtype=torch.long)
    for u, _ in pairs:
        degree[u] += 1
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    rowptr[1:] = torch.cumsum(degree, 0)
    colidx = torch.tensor([v for _, v in pairs], dtype=torch.long)
    scale = torch.rsqrt(degree.to(dtype) + 1)
    schedule = torch.argsort(degree, descending=True, stable=True)
    return CSRGraph(rowptr, colidx, scale, degree, schedule)


def preprocess_undirected_fast(edge_index, num_nodes, dtype=torch.float32):
    """Vectorized deterministic preprocessing for official large static graphs."""
    edge_index = torch.as_tensor(edge_index, dtype=torch.long, device="cpu")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2,E]")
    if edge_index.numel() and (int(edge_index.min()) < 0 or
                               int(edge_index.max()) >= num_nodes):
        raise ValueError("node id out of range")
    u, v = edge_index
    keep = u != v
    u, v = u[keep], v[keep]
    keys = torch.cat((u * num_nodes + v, v * num_nodes + u))
    keys = torch.unique(keys, sorted=True)
    rows = torch.div(keys, num_nodes, rounding_mode="floor")
    colidx = torch.remainder(keys, num_nodes)
    degree = torch.bincount(rows, minlength=num_nodes)
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    rowptr[1:] = torch.cumsum(degree, 0)
    scale = torch.rsqrt(degree.to(dtype) + 1)
    schedule = torch.argsort(degree, descending=True, stable=True)
    return CSRGraph(rowptr, colidx, scale, degree, schedule)
