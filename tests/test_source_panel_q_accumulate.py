"""Correctness gate for source-panel Q consumed into final FP32 dH."""

import torch

from tfs_train.native import backend
from tfs_train.supervision_scope import _edge_balanced_cuts


def symmetric_graph(n: int, half_degree: int):
    rows = torch.arange(n, dtype=torch.long).unsqueeze(1)
    positive = torch.arange(1, half_degree + 1, dtype=torch.long)
    offsets = torch.cat((positive, -positive)).unsqueeze(0)
    colidx = ((rows + offsets) % n).reshape(-1).contiguous()
    degree = 2 * half_degree
    rowptr = torch.arange(0, n * degree + 1, degree, dtype=torch.long)
    return rowptr, colidx


def test_source_panel_q_matches_global_q():
    torch.manual_seed(20260912)
    n, k, d, threads, panel = 97, 64, 47, 4, 29
    rowptr, colidx = symmetric_graph(n, 3)
    scale = (torch.rand(n) + 0.25).contiguous()
    grad = torch.randn(n, d)
    pulled = torch.randn(n, k).to(torch.bfloat16).contiguous()
    weight_t = torch.randn(k, d).to(torch.bfloat16).transpose(0, 1).contiguous()
    ext = backend()

    gs = ext.c3_scale_grad_bf16_v1(grad, scale, threads)
    q = torch.matmul(gs, weight_t)
    expected_dh = ext.c3_pull_only_bf16_amx_v1(
        q, rowptr, colidx, threads).float()
    expected_dw = torch.matmul(pulled.transpose(0, 1).contiguous(), gs).float()

    accum = torch.zeros((n, k), dtype=torch.float32)
    actual_dw = torch.zeros((k, d), dtype=torch.float32)
    for r0 in range(0, n, panel):
        rows = min(panel, n - r0)
        ids = torch.arange(r0, r0 + rows, dtype=torch.int64)
        transpose_rowptr, transpose_colidx = (
            ext.c3_build_selected_transpose_shadow_v1(
                ids, rowptr, colidx))
        schedule = _edge_balanced_cuts(transpose_rowptr, threads)
        gs_panel = ext.c3_scale_grad_bf16_v1(
            grad.narrow(0, r0, rows), scale.narrow(0, r0, rows), threads)
        q_panel = torch.matmul(gs_panel, weight_t)
        ext.c3_rect_pull_bf16_accumulate_fp32_shadow_v1(
            q_panel, transpose_rowptr, transpose_colidx, schedule,
            accum, threads)
        actual_dw.add_(torch.matmul(
            pulled.narrow(0, r0, rows).transpose(0, 1).contiguous(),
            gs_panel).float())
    actual_dh = accum.to(torch.bfloat16).float()
    torch.testing.assert_close(actual_dh, expected_dh, rtol=5e-3, atol=5e-3)
    dw_relative_l2 = (
        torch.linalg.vector_norm(actual_dw - expected_dw) /
        torch.linalg.vector_norm(expected_dw).clamp_min(1e-12))
    assert float(dw_relative_l2) < 5e-3
