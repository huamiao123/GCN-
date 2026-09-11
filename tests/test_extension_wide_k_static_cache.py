"""Numerical gate for explicit wide-K static SX caching."""

import importlib.util

import pytest
import torch

from tfs_train.graph import preprocess_undirected_fast
from tfs_train.native import backend


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("tfs_train_v2_c0_ext") is None,
    reason="compiled AMX extension is unavailable",
)


@pytest.mark.parametrize("k", [200, 300, 500, 602, 1024])
def test_wide_k_static_aggregate_matches_dynamic_aggregate(k):
    torch.manual_seed(20260911 + k)
    n, d, threads = 257, 47, 4
    src = torch.arange(n, dtype=torch.long)
    edge_index = torch.stack((src, (src * 17 + 23) % n))
    graph = preprocess_undirected_fast(edge_index, n, torch.float32)
    x = torch.randn(n, k, dtype=torch.float32)
    weight = torch.randn(k, d, dtype=torch.float32)
    bias = torch.randn(d, dtype=torch.float32)
    grad = torch.randn(n, d, dtype=torch.float32)
    ext = backend()

    dynamic_out, hs = ext.c3_forward_amx_v2(
        x, weight, bias, graph.rowptr, graph.colidx, graph.scale,
        threads, False)
    pulled = ext.c3_prepare_static_aggregate_v3(
        x, graph.scale, graph.rowptr, graph.colidx, threads)
    cached_out, returned = ext.c3_forward_cached_aggregate_amx_v3(
        x, pulled, weight, bias, graph.rowptr, graph.colidx,
        graph.scale, threads)

    assert torch.equal(returned, pulled)
    assert torch.equal(cached_out, dynamic_out)

    _, dynamic_dw, dynamic_db, _ = ext.c3_backward_amx_v2(
        grad, hs, weight, graph.rowptr, graph.colidx, graph.scale,
        threads, False)
    cached_dw, cached_db, _ = ext.c3_backward_cached_aggregate_amx_v3(
        grad, pulled, graph.rowptr, graph.colidx, graph.scale, threads)

    torch.testing.assert_close(cached_db, dynamic_db, rtol=1e-6, atol=3e-5)
    torch.testing.assert_close(cached_dw, dynamic_dw, rtol=5e-2, atol=2e-1)
