#!/usr/bin/env python3
"""Numerical gate for the NUMA/source-reuse panel scheduler."""

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "csrc"))
import tfs_train_v2_c0_ext as ext


def main():
    torch.manual_seed(20260816)
    n, k, d, threads = 4096, 32, 64, 4
    rows = torch.arange(n, dtype=torch.long)
    colidx = torch.stack((rows, rows.roll(1), (rows * 17 + 5) % n), dim=1)
    colidx = colidx.reshape(-1).contiguous()
    rowptr = torch.arange(0, 3 * n + 1, 3, dtype=torch.long)
    scale = (torch.rand(n) + 0.25).contiguous()
    x = torch.randn(n, k).contiguous()
    weight = torch.randn(k, d).contiguous()
    bias = torch.randn(d).contiguous()
    grad = torch.randn(n, d).contiguous()

    os.environ["TFS_GLUE_E7_FORWARD_SCHEDULE"] = "0"
    os.environ["TFS_LOCALITY_SCHEDULE"] = "off"
    out_off, hs_off = ext.c3_forward_amx_v2(
        x, weight, bias, rowptr, colidx, scale, threads, False
    )
    os.environ["TFS_LOCALITY_SCHEDULE"] = "on"
    out_on, hs_on = ext.c3_forward_amx_v2(
        x, weight, bias, rowptr, colidx, scale, threads, False
    )
    torch.testing.assert_close(hs_off, hs_on, rtol=0, atol=0)
    torch.testing.assert_close(out_off, out_on, rtol=0, atol=0)

    os.environ["TFS_LOCALITY_SCHEDULE"] = "off"
    dx_off, dw_off, db_off, _ = ext.c3_backward_amx_v2(
        grad, hs_off, weight, rowptr, colidx, scale, threads, True
    )
    os.environ["TFS_LOCALITY_SCHEDULE"] = "on"
    dx_on, dw_on, db_on, _ = ext.c3_backward_amx_v2(
        grad, hs_on, weight, rowptr, colidx, scale, threads, True
    )
    torch.testing.assert_close(dx_off, dx_on, rtol=0, atol=0)
    torch.testing.assert_close(db_off, db_on, rtol=0, atol=0)
    # dW's worker-local FP32 accumulation is deterministic within each
    # schedule but can change summation order when ownership changes.
    torch.testing.assert_close(dw_off, dw_on, rtol=2e-4, atol=2e-5)
    print("extension locality-schedule smoke: PASS",
          "dw_max_abs=", float((dw_off - dw_on).abs().max()))


if __name__ == "__main__":
    main()
