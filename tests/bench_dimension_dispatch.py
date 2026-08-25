#!/usr/bin/env python3
"""Measure the steady-state native cost of the dimension families.

The benchmark deliberately keeps Hs cached, so the reported forward number is
the per-step consumer cost.  It is an implementation/shape benchmark, not a
TFS-vs-DGL authority run.
"""

import os
import statistics
import sys
import time

import torch


def median_ms(fn, warmup=2, measured=5):
    for _ in range(warmup):
        fn()
    values = []
    for _ in range(measured):
        t0 = time.perf_counter()
        fn()
        values.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(values)


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csrc"))
    import tfs_train_v2_c0_ext as ext

    threads = int(os.environ.get("DIM_BENCH_THREADS", "8"))
    n = int(os.environ.get("DIM_BENCH_N", "4096"))
    torch.set_num_threads(threads)
    torch.manual_seed(20260816)
    src = torch.arange(n, dtype=torch.long)
    rowptr = torch.arange(0, 2 * n + 1, 2, dtype=torch.long).contiguous()
    colidx = torch.stack((src, src.roll(1))).reshape(-1).contiguous()
    scale = (torch.rand(n, dtype=torch.float32) + 0.5).contiguous()

    cases = [
        ("c3_128x128", 128, 128, "c3"),
        ("wide_k_1024x128", 1024, 128, "wide_k"),
        ("wide_aggregate_128x2983", 128, 2983, "wide_aggregate"),
        ("wide_aggregate_256x256", 256, 256, "wide_aggregate"),
    ]
    for name, k, d, path in cases:
        x = torch.randn(n, k, dtype=torch.float32).contiguous()
        weight = torch.randn(k, d, dtype=torch.float32).contiguous()
        bias = torch.randn(d, dtype=torch.float32).contiguous()
        hs = ext.c3_prepare_static_hs_v1(x, scale, threads)
        grad = torch.randn(n, d, dtype=torch.float32).contiguous()

        if path in ("c3", "wide_k"):
            transform_first = d < k

            def forward():
                ext.c3_forward_cached_hs_amx_v1(
                    x, hs, weight, bias, rowptr, colidx, scale,
                    threads, transform_first)

            def backward():
                ext.c3_backward_amx_v2(
                    grad, hs, weight, rowptr, colidx, scale,
                    threads, False)
        else:
            def forward():
                ext.c3_forward_aggregate_wide_cached_hs_amx_v1(
                    x, hs, weight, bias, rowptr, colidx, scale, threads)

            def backward():
                ext.c3_backward_wide_amx_v3(
                    grad, hs, weight, rowptr, colidx, scale,
                    threads, True)

        fwd = median_ms(forward)
        bwd = median_ms(backward)
        print(
            f"case={name} path={path} n={n} k={k} d={d} threads={threads} "
            f"forward_cached_ms={fwd:.3f} backward_ms={bwd:.3f}")


if __name__ == "__main__":
    main()
