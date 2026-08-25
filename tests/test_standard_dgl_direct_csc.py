import sys
from pathlib import Path

import pytest
import torch


pytest.importorskip("dgl")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tfs_train.standard_dgl import (DGLGCN,
                                    build_stock_dgl_graph_from_pull_csr)


def _sorted_edges(graph):
    src, dst = graph.edges()
    return sorted(zip(src.tolist(), dst.tolist()))


def test_direct_csc_matches_previous_coo_graph_and_graphconv():
    import dgl

    # Pull rows (destinations): dst=1 pulls src 0/2; dst=3 pulls src 1.
    rowptr = torch.tensor([0, 0, 2, 2, 3], dtype=torch.int64)
    colidx = torch.tensor([0, 2, 1], dtype=torch.int64)
    direct, metadata = build_stock_dgl_graph_from_pull_csr(
        rowptr, colidx, 4)

    rows = torch.repeat_interleave(
        torch.arange(4), rowptr[1:] - rowptr[:-1])
    previous = dgl.add_self_loop(
        dgl.graph((colidx, rows), num_nodes=4))
    assert _sorted_edges(direct) == _sorted_edges(previous)
    assert metadata["construction"] == "direct_pull_csr_as_csc"
    assert metadata["input_edges"] == 3
    assert metadata["output_edges"] == 7

    torch.manual_seed(7)
    model_a = DGLGCN(2, 5, 6, 3, norm="both", dropout=0.0)
    model_b = DGLGCN(2, 5, 6, 3, norm="both", dropout=0.0)
    model_b.load_state_dict(model_a.state_dict())
    model_a.eval(); model_b.eval()
    x_a = torch.randn(4, 5, requires_grad=True)
    x_b = x_a.detach().clone().requires_grad_(True)
    y_a = model_a(x_a, (direct, None))
    y_b = model_b(x_b, (previous, None))
    assert torch.equal(y_a, y_b)
    y_a.sum().backward(); y_b.sum().backward()
    assert torch.equal(x_a.grad, x_b.grad)
    for left, right in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(left.grad, right.grad)
