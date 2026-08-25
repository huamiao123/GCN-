#!/usr/bin/env python3
"""Measure the opt-in Hs NUMA replica read path on a larger CSR fixture."""

import os
import statistics
import time

import torch

from tfs_train.native import backend


def main():
    torch.set_num_threads(1)
    os.environ.setdefault("TFS_LOCALITY_SCHEDULE", "off")
    n = int(os.environ.get("HS_REPLICA_BENCH_N", "65536"))
    k = int(os.environ.get("HS_REPLICA_BENCH_K", "32"))
    d = int(os.environ.get("HS_REPLICA_BENCH_D", "64"))
    threads = int(os.environ.get("HS_REPLICA_BENCH_THREADS", "8"))
    src = torch.arange(n, dtype=torch.long)
    colidx = torch.stack((src, (src * 17 + 13) % n,
                          (src * 29 + 7) % n,
                          (src * 43 + 3) % n), dim=1).reshape(-1)
    rowptr = torch.arange(0, 4 * (n + 1), 4, dtype=torch.long)
    rowptr[-1] = colidx.numel()
    x = torch.randn(n, k, dtype=torch.float32)
    scale = torch.rand(n, dtype=torch.float32) + 0.5
    weight = torch.randn(k, d, dtype=torch.float32)
    bias = torch.randn(d, dtype=torch.float32)
    ext = backend()
    hs = ext.c3_prepare_static_hs_v1(x, scale, threads)
    replicas = ext.c3_replicate_static_hs_numa_v1(hs, threads)

    def run(storage):
        ext.c3_forward_cached_hs_amx_v1(
            x, storage, weight, bias, rowptr, colidx, scale,
            threads, False)

    for storage in (hs, replicas):
        run(storage)
    ref_ms, rep_ms = [], []
    # Alternate modes so thermal drift and allocator state do not map to one
    # side of the comparison.  The benchmark reports medians only.
    for i in range(12):
        t0 = time.perf_counter(); run(hs); t1 = time.perf_counter()
        t2 = time.perf_counter(); run(replicas); t3 = time.perf_counter()
        ref_ms.append((t1 - t0) * 1e3)
        rep_ms.append((t3 - t2) * 1e3)
    ref = statistics.median(ref_ms[2:])
    rep = statistics.median(rep_ms[2:])
    print(
        f"hs_single_median_ms={ref:.3f} "
        f"hs_replica_median_ms={rep:.3f} "
        f"speedup={ref / rep:.3f}x "
        f"replicas={replicas.shape[0]} "
        f"single_bytes={hs.numel() * hs.element_size()} "
        f"replica_bytes={replicas.numel() * replicas.element_size()}")


if __name__ == "__main__":
    main()
