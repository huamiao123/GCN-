"""One native DGL model used by the authority harness.

The strict comparison always instantiates this class with ``norm=\"both\"``
and passes ``(graph, None)``.  The optional edge-weight argument is retained
only for reproducing the historical, non-authority path; it never enters the
stock authority launcher.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


def build_stock_dgl_graph_from_pull_csr(
        rowptr: torch.Tensor, colidx: torch.Tensor, num_nodes: int,
        *, add_self_loop: bool = True) -> Tuple[Any, Dict[str, Any]]:
    """Build DGL's stock graph directly from TFS's pull-oriented CSR.

    A pull CSR is exactly a CSC graph representation: ``rowptr`` partitions
    destination rows and ``colidx`` stores source nodes.  Constructing this
    format directly avoids the former ``repeat_interleave`` CSR-to-COO
    expansion without changing GraphConv, normalization, threading, or any
    timed training operation.
    """

    import dgl

    if rowptr.device.type != "cpu" or colidx.device.type != "cpu":
        raise ValueError("DGL authority graph construction requires CPU CSR")
    if rowptr.dtype not in {torch.int32, torch.int64}:
        raise TypeError("rowptr must be int32 or int64")
    if colidx.dtype != rowptr.dtype:
        raise TypeError("rowptr and colidx must use the same integer dtype")
    if not rowptr.is_contiguous() or not colidx.is_contiguous():
        raise ValueError("rowptr and colidx must be contiguous")
    n = int(num_nodes)
    if rowptr.numel() != n + 1:
        raise ValueError("pull CSR rowptr length does not match num_nodes")

    start_ns = time.perf_counter_ns()
    edge_ids = torch.empty(0, dtype=rowptr.dtype)
    graph = dgl.graph(
        ("csc", (rowptr, colidx, edge_ids)), num_nodes=n,
        idtype=rowptr.dtype, device="cpu")
    csc_ready_ns = time.perf_counter_ns()
    if add_self_loop:
        graph = dgl.add_self_loop(graph)
    ready_ns = time.perf_counter_ns()
    metadata = {
        "construction": "direct_pull_csr_as_csc",
        "num_nodes": n,
        "input_edges": int(colidx.numel()),
        "output_edges": int(graph.num_edges()),
        "self_loop_added": bool(add_self_loop),
        "csc_build_ms": (csc_ready_ns - start_ns) / 1.0e6,
        "self_loop_ms": (ready_ns - csc_ready_ns) / 1.0e6,
        "graph_ready_ms": (ready_ns - start_ns) / 1.0e6,
    }
    return graph, metadata


class DGLGCN(torch.nn.Module):
    """Native DGL GraphConv stack with the same training wrapper everywhere."""

    def __init__(self, layers: int = 2, in_dim: int = 100,
                 hidden_dim: int = 128, out_dim: int = 47,
                 norm: str = "both", allow_zero_in_degree: bool = True,
                 dropout: float = 0.5) -> None:
        super().__init__()
        from dgl.nn.pytorch import GraphConv

        dims = [int(in_dim)] + [int(hidden_dim)] * (int(layers) - 1) + [int(out_dim)]
        self.convs = torch.nn.ModuleList([
            GraphConv(dims[i], dims[i + 1], norm=norm, weight=True,
                      bias=True, allow_zero_in_degree=allow_zero_in_degree)
            for i in range(int(layers))
        ])
        self.dropout = float(dropout)
        self.force_bf16_activations = False

    def forward(self, x: torch.Tensor,
                arg: Tuple[Any, Any]) -> torch.Tensor:
        graph, edge_weight = arg

        def apply_conv(conv: torch.nn.Module, value: torch.Tensor) -> torch.Tensor:
            if edge_weight is None:
                return conv(graph, value)
            return conv(graph, value, edge_weight=edge_weight)

        for conv in self.convs[:-1]:
            if self.force_bf16_activations:
                x = x.to(torch.bfloat16)
            x = F.relu(apply_conv(conv, x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        if self.force_bf16_activations:
            x = x.to(torch.bfloat16)
        return apply_conv(self.convs[-1], x)


__all__ = ["DGLGCN", "build_stock_dgl_graph_from_pull_csr"]
