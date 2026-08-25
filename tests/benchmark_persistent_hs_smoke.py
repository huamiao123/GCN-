import os
import sys
import time

import torch


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csrc"))
    import tfs_train_v2_c0_ext as ext

    torch.manual_seed(11)
    n, k, d, threads = 8192, 128, 128, 2
    x = torch.randn(n, k).contiguous()
    weight = torch.randn(k, d).contiguous()
    bias = torch.randn(d).contiguous()
    rowptr = torch.arange(n + 1, dtype=torch.long).contiguous()
    colidx = torch.arange(n, dtype=torch.long).contiguous()
    scale = torch.rand(n).contiguous()

    def regular():
        return ext.c3_forward_amx_v2(
            x, weight, bias, rowptr, colidx, scale, threads, True)

    hs = ext.c3_prepare_static_hs_v1(x, scale, threads)

    def cached():
        return ext.c3_forward_cached_hs_amx_v1(
            x, hs, weight, bias, rowptr, colidx, scale, threads, True)

    for _ in range(2):
        regular()
        cached()
    regular_ms = []
    cached_ms = []
    for _ in range(5):
        t0 = time.perf_counter_ns(); regular(); t1 = time.perf_counter_ns()
        t2 = time.perf_counter_ns(); cached(); t3 = time.perf_counter_ns()
        regular_ms.append((t1 - t0) / 1e6)
        cached_ms.append((t3 - t2) / 1e6)
    regular_ms.sort(); cached_ms.sort()
    r = regular_ms[len(regular_ms) // 2]
    c = cached_ms[len(cached_ms) // 2]
    print({"n": n, "k": k, "d": d, "threads": threads,
           "regular_median_ms": r, "cached_median_ms": c,
           "speedup": r / c, "hs_bytes": hs.numel() * hs.element_size()})


if __name__ == "__main__":
    main()

