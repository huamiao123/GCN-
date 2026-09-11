"""Numerical contract for native aggregate D-slab one-pull execution."""

import importlib.util

import pytest
import torch

from tfs_train.highd_backward import (
    HighDBackwardPlan, native_aggregate_d_slab_backward,
    streamed_aggregate_single_scan_backward,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("tfs_train_v2_c0_ext") is None,
    reason="compiled AMX extension is unavailable",
)


def _ring(n):
    rows = torch.arange(n, dtype=torch.long)
    rowptr = torch.arange(0, 3 * n + 1, 3, dtype=torch.long)
    colidx = torch.stack(
        (rows, (rows - 1) % n, (rows + 1) % n), dim=1
    ).reshape(-1).contiguous()
    return rowptr, colidx


@pytest.mark.parametrize("k", [65, 128])
def test_native_aggregate_dslab_uses_one_final_pull(k):
    torch.manual_seed(20260911)
    n, d, threads = 65, 513, 4
    pulled = torch.randn(n, k).to(torch.bfloat16)
    weight = torch.randn(k, d)
    grad = torch.randn(n, d)
    scale = torch.rand(n) + 0.5
    rowptr, colidx = _ring(n)
    slabs = ((0, 257), (257, 513))
    plan = HighDBackwardPlan(
        n=n, k=k, d=d, kp=((k + 63) // 64) * 64, dp=544, threads=threads,
        row_panel=512, d_tile=257, strategy="d_slab", compute_dx=True,
        budget_bytes=64 << 20, estimated_panel_bytes=0,
        persistent_dx_bytes=n * k * 4, d_slabs=slabs,
        dslab_max_width=257, workspace_bytes=2 << 20,
        execution_variant="aggregate_highd_single_scan")

    old_dx, old_dw, old_db = native_aggregate_d_slab_backward(
        pulled, weight, grad, scale, rowptr, colidx, threads, True,
        d_tile=257, d_slabs=slabs)
    new_dx, new_dw, new_db = streamed_aggregate_single_scan_backward(
        pulled, weight, grad, scale, rowptr, colidx, threads, True,
        plan=plan)

    torch.testing.assert_close(new_dw, old_dw, rtol=0, atol=0)
    torch.testing.assert_close(new_db, old_db, rtol=0, atol=0)
    relative_l2 = (torch.linalg.vector_norm(new_dx - old_dx) /
                   torch.linalg.vector_norm(old_dx).clamp_min(1.0e-12))
    assert float(relative_l2) <= 1.0e-2


def test_single_scan_source_has_only_one_final_pull_call():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "python" / "tfs_train" /
              "highd_backward.py").read_text(encoding="utf-8")
    begin = source.index("def streamed_aggregate_single_scan_backward(")
    end = source.index("\n\n__all__", begin)
    body = source[begin:end]
    assert body.count("pull(") == 1
    assert "c3_backward_aggregate_highd_dp_amx_v1" in body
    assert "d_p.add_(partial_dp)" not in body
    assert "d_p, int(threads)" in body
