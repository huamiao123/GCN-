"""Native smoke gates for the high-K and high-D AMX contracts.

This is intentionally a small deterministic fixture.  It does not claim a
full-graph speedup; it proves that the two dimensions used by IGB
(1024 input features and 2983 output classes) stay on the native AMX family
without requiring the legacy ``HYBRID_WIDE_K_AMX`` switch.
"""

import os
import sys

import torch


def _graph(n):
    rowptr = torch.arange(0, 2 * n + 1, 2, dtype=torch.long).contiguous()
    src = torch.arange(n, dtype=torch.long)
    colidx = torch.stack((src, src.roll(1))).reshape(-1).contiguous()
    scale = torch.rand(n, dtype=torch.float32).contiguous()
    return rowptr, colidx, scale


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csrc"))
    import tfs_train_v2_c0_ext as ext

    torch.manual_seed(20260816)
    os.environ.pop("HYBRID_WIDE_K_AMX", None)
    n, threads = 96, 2
    rowptr, colidx, scale = _graph(n)

    # IGB first layer: K=1024, hidden D=128.  Both calculation orders and
    # both uncached/cached consumers must agree bitwise.
    k, d = 1024, 128
    x = torch.randn(n, k, dtype=torch.float32).contiguous()
    weight = torch.randn(k, d, dtype=torch.float32).contiguous()
    bias = torch.randn(d, dtype=torch.float32).contiguous()
    hs = ext.c3_prepare_static_hs_v1(x, scale, threads)
    grad = torch.randn(n, d, dtype=torch.float32).contiguous()
    for transform_first in (False, True):
        out0, hs0 = ext.c3_forward_amx_v2(
            x, weight, bias, rowptr, colidx, scale, threads,
            transform_first)
        out1, hs1 = ext.c3_forward_cached_hs_amx_v1(
            x, hs, weight, bias, rowptr, colidx, scale, threads,
            transform_first)
        assert torch.equal(hs0, hs) and torch.equal(hs1, hs)
        assert torch.equal(out0, out1), ("wide-k forward", transform_first)
    b0 = ext.c3_backward_amx_v2(
        grad, hs, weight, rowptr, colidx, scale, threads, True)
    b1 = ext.c3_backward_amx_v2(
        grad, hs, weight, rowptr, colidx, scale, threads, False)
    assert b0[0].shape == x.shape and b1[0].numel() == 0
    assert torch.equal(b0[1], b1[1]) and torch.equal(b0[2], b1[2])

    # IGB final layer: K=128, D=2983.  Exercise the aggregate-wide producer,
    # cached consumer, and native tiled backward against its explicit tile
    # reference.  This keeps the high-D contract independent from high-K.
    k, d = 128, 2983
    x = torch.randn(n, k, dtype=torch.float32).contiguous()
    weight = torch.randn(k, d, dtype=torch.float32).contiguous()
    bias = torch.randn(d, dtype=torch.float32).contiguous()
    hs = ext.c3_prepare_static_hs_v1(x, scale, threads)
    out0, hs0, pulled0 = ext.c3_forward_aggregate_wide_amx_v3(
        x, weight, bias, rowptr, colidx, scale, threads)
    out1, hs1, pulled1 = ext.c3_forward_aggregate_wide_cached_hs_amx_v1(
        x, hs, weight, bias, rowptr, colidx, scale, threads)
    assert torch.equal(hs0, hs) and torch.equal(hs1, hs)
    assert torch.equal(pulled0, pulled1)
    assert torch.equal(out0, out1), "wide-D cached forward mismatch"

    grad = torch.randn(n, d, dtype=torch.float32).contiguous()
    wide = ext.c3_backward_wide_amx_v3(
        grad, hs, weight, rowptr, colidx, scale, threads, True)
    dx_parts, dw_parts, db_parts = [], [], []
    for start in range(0, d, 128):
        end = min(start + 128, d)
        part = ext.c3_backward_amx_v2(
            grad[:, start:end].contiguous(), hs,
            weight[:, start:end].contiguous(), rowptr, colidx, scale,
            threads, True)
        dx_parts.append(part[0])
        dw_parts.append(part[1])
        db_parts.append(part[2])
    dx_ref = dx_parts[0].clone()
    for part in dx_parts[1:]:
        dx_ref.add_(part)
    assert torch.equal(wide[0], dx_ref)
    assert torch.equal(wide[1], torch.cat(dw_parts, dim=1))
    assert torch.equal(wide[2], torch.cat(db_parts, dim=0))

    # The aggregate-wide backward uses pull-only for K>128.  Check the new
    # unbounded-K contract directly against the two-edge BF16 reference.
    k = 256
    dp = torch.randn(n, k, dtype=torch.float32).contiguous()
    pulled = ext.c3_pull_only_amx_v1(dp, rowptr, colidx, threads)
    tiled = []
    for start in range(0, k, 128):
        eye = torch.eye(128, dtype=torch.float32).contiguous()
        zero = torch.zeros(128, dtype=torch.float32).contiguous()
        part, _ = ext.c3_forward_amx_v2(
            dp[:, start:start + 128].contiguous(), eye, zero,
            rowptr, colidx, torch.ones(n), threads, False)
        tiled.append(part)
    assert pulled.shape == dp.shape and torch.equal(pulled, torch.cat(tiled, dim=1))
    print("extension dimension generalization: PASS")


if __name__ == "__main__":
    main()
