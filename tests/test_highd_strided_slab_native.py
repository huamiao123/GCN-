"""Numerical contract for zero-copy column-slab native inputs."""

import importlib.util

import pytest
import torch

from tfs_train.native import backend


def _ring(n):
    rows = torch.arange(n, dtype=torch.long)
    rowptr = torch.arange(0, 3 * n + 1, 3, dtype=torch.long)
    colidx = torch.stack(
        (rows, (rows - 1) % n, (rows + 1) % n), dim=1
    ).reshape(-1).contiguous()
    return rowptr, colidx


def _assert_results_equal(actual, expected):
    assert len(actual) == len(expected)
    for lhs, rhs in zip(actual[:3], expected[:3]):
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)


@pytest.mark.skipif(
    importlib.util.find_spec("tfs_train_v2_c0_ext") is None,
    reason="compiled AMX extension is unavailable",
)
@pytest.mark.parametrize("family", ["transform", "aggregate"])
def test_native_highd_accepts_row_strided_column_slab(family):
    torch.manual_seed(20260910)
    n, k, full_d, d0, width, threads = 65, 512, 513, 128, 257, 4
    grad_full = torch.randn(n, full_d)
    weight_full = torch.randn(k, full_d)
    grad_view = grad_full.narrow(1, d0, width)
    weight_view = weight_full.narrow(1, d0, width)
    assert not grad_view.is_contiguous() and grad_view.stride(1) == 1
    assert not weight_view.is_contiguous() and weight_view.stride(1) == 1

    hs = torch.randn(n, k).to(torch.bfloat16)
    scale = torch.rand(n) + 0.5
    rowptr, colidx = _ring(n)
    ext = backend()
    if family == "transform":
        function = ext.c3_backward_transform_highd_amx_v1
        operand = hs
    else:
        function = ext.c3_backward_aggregate_highd_amx_v1
        operand = torch.randn(n, k).to(torch.bfloat16)

    actual = function(
        grad_view, operand, weight_view, rowptr, colidx, scale,
        threads, True)
    expected = function(
        grad_view.contiguous(), operand, weight_view.contiguous(),
        rowptr, colidx, scale, threads, True)
    _assert_results_equal(actual, expected)
