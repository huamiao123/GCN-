"""Numerical gate for the native Transform-HighD slab entry point."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from tfs_train.execution_plan import build_layer_plan
from tfs_train.highd_backward import (native_transform_highd_backward,
                                       streamed_transform_backward)


def _ring(n: int):
    rows = torch.arange(n, dtype=torch.long)
    return (torch.arange(0, 3 * n + 1, 3, dtype=torch.long),
            torch.stack((rows, (rows - 1) % n, (rows + 1) % n), dim=1)
            .reshape(-1).contiguous())


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() /
                 (b.float().norm() + 1.0e-6))


@pytest.mark.skipif(
    importlib.util.find_spec("tfs_train_v2_c0_ext") is None,
    reason="compiled AMX extension is unavailable",
)
def test_native_transform_highd_matches_stream(monkeypatch):
    # This is a real K>D>128 production shape with an N tail.  Keep the
    # reference stream explicit so the native gate never recursively selects
    # itself through the dispatcher.
    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_STREAM_V1", "on")
    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_NATIVE", "off")
    monkeypatch.setenv("TFS_GLUE_E2_VEC_STORE", "1")
    monkeypatch.setenv("TFS_COLIDX", "int64")
    torch.set_num_threads(1)
    torch.manual_seed(20260818)
    n, k, d, threads = 4097, 1024, 257, 4
    hs = torch.randn(n, k).to(torch.bfloat16)
    weight = torch.randn(k, d)
    grad = torch.randn(n, d)
    scale = torch.rand(n) + 0.5
    rowptr, colidx = _ring(n)
    plan = build_layer_plan(n, k, d, threads=threads, compute_dx=True)
    assert plan.selected_impl == "transform_highd_stream"

    ref_dx, ref_dw = streamed_transform_backward(
        hs, weight, grad, scale, rowptr, colidx, threads, True, plan=plan)
    new_dx, new_dw = native_transform_highd_backward(
        hs, weight, grad, scale, rowptr, colidx, threads, True,
        d_slabs=plan.d_slabs, d_tile=plan.d_tile)
    assert new_dx is not None and ref_dx is not None
    assert _rel(new_dx, ref_dx) <= 5.0e-2
    assert _rel(new_dw, ref_dw) <= 5.0e-2


@pytest.mark.skipif(
    importlib.util.find_spec("tfs_train_v2_c0_ext") is None,
    reason="compiled AMX extension is unavailable",
)
def test_single_scan_uses_immutable_plan_budget(monkeypatch):
    """A later legacy-env change must not override the selected plan."""

    monkeypatch.setenv("HYBRID_AMX_FORWARD", "1")
    monkeypatch.setenv("HYBRID_AMX_BACKWARD", "1")
    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_SINGLE_SCAN", "on")
    monkeypatch.setenv("TFS_HIGHD_BWD_D_TILE", "256")
    monkeypatch.setenv("TFS_HIGHD_BWD_BASE_BUDGET_BYTES", str(64 << 20))
    monkeypatch.setenv("TFS_HIGHD_BWD_PER_THREAD_BUDGET_BYTES", str(4 << 20))
    monkeypatch.delenv("TFS_HIGHD_BWD_BUDGET_BYTES", raising=False)
    torch.set_num_threads(1)
    n, k, d, threads = 17, 1024, 513, 4
    plan = build_layer_plan(n, k, d, threads=threads, compute_dx=True)
    assert plan.execution_variant == "transform_highd_single_scan"
    assert plan.workspace_budget_bytes == 64 << 20

    # Simulate a stale process-wide value written after construction.  The
    # execution branch must still honor the plan's immutable budget.
    monkeypatch.setenv("TFS_HIGHD_BWD_BUDGET_BYTES", "1")
    hs = torch.randn(n, k).to(torch.bfloat16)
    weight = torch.randn(k, d)
    grad = torch.randn(n, d)
    rowptr = torch.arange(n + 1, dtype=torch.long)
    colidx = torch.arange(n, dtype=torch.long)
    dx, dw = streamed_transform_backward(
        hs, weight, grad, torch.ones(n), rowptr, colidx, threads, True,
        plan=plan)
    assert dx is not None and tuple(dx.shape) == (n, k)
    assert tuple(dw.shape) == (k, d)
