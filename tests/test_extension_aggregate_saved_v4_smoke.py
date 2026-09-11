#!/usr/bin/env python3
"""Numerical/contract gate for the generic aggregate-saved AMX path."""

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tfs_train.aggregate_saved import AggregateSavedFunction
from tfs_train.execution_plan import build_layer_plan
from tfs_train.native import backend


def _graph(n: int):
    # Two deterministic incoming entries per row.  The normalized scale is
    # deliberately non-uniform so both scaling sites are covered.
    cols = torch.stack(
        (torch.arange(n, dtype=torch.long),
         (torch.arange(n, dtype=torch.long) * 7 + 3) % n), dim=1
    ).reshape(-1)
    rowptr = torch.arange(0, 2 * n + 1, 2, dtype=torch.long)
    scale = torch.linspace(0.25, 1.25, n, dtype=torch.float32)
    return rowptr, cols, scale


def main():
    os.environ.setdefault("TFS_INTERNAL_PROFILE", "1")
    torch.manual_seed(20260816)
    n, k, d, threads = 96, 32, 64, 2
    rowptr, colidx, scale = _graph(n)
    x = torch.randn(n, k, dtype=torch.float32)
    weight = torch.randn(k, d, dtype=torch.float32)
    bias = torch.randn(d, dtype=torch.float32)
    grad = torch.randn(n, d, dtype=torch.float32)

    out, hs, pulled = backend().c3_forward_aggregate_saved_amx_v4(
        x, weight, bias, rowptr, colidx, scale, threads
    )
    out_ref, hs_ref = backend().c3_forward_amx_v2(
        x, weight, bias, rowptr, colidx, scale, threads, False
    )
    torch.testing.assert_close(out, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(hs, hs_ref, rtol=0, atol=0)
    pulled_ref = backend().c3_prepare_static_aggregate_v3(
        x, scale, rowptr, colidx, threads
    )
    torch.testing.assert_close(pulled, pulled_ref, rtol=0, atol=0)

    gs = backend().c3_scale_grad_bf16_v1(grad, scale, threads)
    gs_fused, db_fused = backend().c3_scale_grad_bf16_db_v2(
        grad, scale, threads
    )
    torch.testing.assert_close(gs_fused, gs, rtol=0, atol=0)
    # The fused deterministic worker-order reduction can differ from ATen's
    # vectorized reduction by a few FP32 ulps.
    torch.testing.assert_close(db_fused, grad.sum(0), rtol=1e-5, atol=1e-5)
    dw_ref = torch.matmul(pulled.transpose(0, 1), gs).float()
    db_ref = grad.sum(0)
    _empty, dw, db, meta = backend().c3_backward_aggregate_saved_amx_v4(
        grad, pulled, weight, rowptr, colidx, scale, threads, False
    )
    torch.testing.assert_close(dw, dw_ref, rtol=0, atol=0)
    torch.testing.assert_close(db, db_ref, rtol=0, atol=0)
    assert _empty.numel() == 0 and meta.tolist() == [threads, 512, 64, 64]

    gs = gs.contiguous()
    dp = torch.matmul(gs, weight.to(torch.bfloat16).transpose(0, 1))
    assert dp.dtype == torch.bfloat16
    dh_bf16 = backend().c3_pull_only_bf16_amx_v1(
        dp, rowptr, colidx, threads)
    # Prove that removing the full dP BF16->FP32->BF16 round trip preserves
    # the exact historical sparse-pull result.
    dh_historical = backend().c3_pull_only_amx_v1(
        dp.float(), rowptr, colidx, threads)
    torch.testing.assert_close(dh_bf16.float(), dh_historical, rtol=0, atol=0)
    dx_ref = dh_bf16 * scale.unsqueeze(1)
    dx, dw2, db2, _ = backend().c3_backward_aggregate_saved_amx_v4(
        grad, pulled, weight, rowptr, colidx, scale, threads, True
    )
    torch.testing.assert_close(dx, dx_ref, rtol=0, atol=0)
    torch.testing.assert_close(dw2, dw_ref, rtol=0, atol=0)
    torch.testing.assert_close(db2, db_ref, rtol=0, atol=0)

    # Exercise the Python autograd contract for both dX-required and
    # parameter-only cases.
    xa = x.detach().clone().requires_grad_(True)
    wa = weight.detach().clone().requires_grad_(True)
    ba = bias.detach().clone().requires_grad_(True)
    os.environ["TFS_AGGREGATE_SAVED"] = "on"
    os.environ["HYBRID_AMX_FORWARD"] = "1"
    os.environ["HYBRID_AMX_BACKWARD"] = "1"
    plan = build_layer_plan(
        n, k, d, compute_dx=True, input_static=False,
        graph_static=True, feature_static=False, threads=threads)
    assert plan.execution_variant == "aggregate_saved_v4"
    ya = AggregateSavedFunction.apply(
        xa, wa, ba, rowptr, colidx, scale, torch.empty(0), threads, plan
    )
    ya.square().mean().backward()
    assert xa.grad is not None and wa.grad is not None and ba.grad is not None
    print("aggregate_saved_v4 smoke PASS")


if __name__ == "__main__":
    main()
