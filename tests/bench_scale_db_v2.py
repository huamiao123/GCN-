#!/usr/bin/env python3
"""Small isolated performance gate for scale+BF16+db fusion."""

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from tfs_train.native import backend


def timed(fn, reps=3):
    for _ in range(1):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / reps


def main():
    torch.manual_seed(20260816)
    n, d, threads = 32768, 2983, 8
    grad = torch.randn(n, d, dtype=torch.float32)
    scale = torch.rand(n, dtype=torch.float32) + 0.5
    t_v1 = timed(lambda: backend().c3_scale_grad_bf16_v1(grad, scale, threads))
    t_v2 = timed(lambda: backend().c3_scale_grad_bf16_db_v2(grad, scale, threads))
    gs1 = backend().c3_scale_grad_bf16_v1(grad, scale, threads)
    gs2, db2 = backend().c3_scale_grad_bf16_db_v2(grad, scale, threads)
    torch.testing.assert_close(gs1, gs2, rtol=0, atol=0)
    # Worker-local deterministic summation is allowed a small FP32 reduction
    # order difference versus ATen's vectorized tree on large rows.
    torch.testing.assert_close(db2, grad.sum(0), rtol=2e-3, atol=2e-3)
    print(f"scale_db_v2_ms={t_v2:.3f} scale_v1_ms={t_v1:.3f} "
          f"speedup={t_v1/t_v2:.3f}x")


if __name__ == "__main__":
    main()
