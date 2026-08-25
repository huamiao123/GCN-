"""Microbenchmark for the old FP32-entry and direct-BF16 pull primitives."""

from __future__ import annotations

import os
import statistics
import time

import torch

import tfs_train_v2_c0_ext as ext


def ring(n: int):
    rows = torch.arange(n, dtype=torch.long)
    rowptr = torch.arange(0, 3 * n + 1, 3, dtype=torch.long)
    colidx = torch.stack((rows, (rows - 1) % n, (rows + 1) % n), dim=1)
    return rowptr, colidx.reshape(-1).contiguous()


def measure(fn, repeats: int):
    values = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        values.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(values), values


def main() -> None:
    n = int(os.environ.get("TFS_BF16_PULL_N", "4096"))
    k = int(os.environ.get("TFS_BF16_PULL_K", "257"))
    threads = int(os.environ.get("TFS_BF16_PULL_THREADS", "32"))
    repeats = int(os.environ.get("TFS_BF16_PULL_REPEATS", "5"))
    torch.set_num_threads(1)
    torch.manual_seed(20260818)
    rowptr, colidx = ring(n)
    source = torch.randn(n, k).to(torch.bfloat16).contiguous()
    for _ in range(2):
        ext.c3_pull_only_amx_v1(source.float().contiguous(), rowptr,
                                 colidx, threads)
        ext.c3_pull_only_bf16_amx_v1(source, rowptr, colidx, threads)
    old_ms, old_rows = measure(
        lambda: ext.c3_pull_only_amx_v1(
            source.float().contiguous(), rowptr, colidx, threads), repeats)
    new_ms, new_rows = measure(
        lambda: ext.c3_pull_only_bf16_amx_v1(
            source, rowptr, colidx, threads), repeats)
    print({
        "shape": {"N": n, "K": k, "threads": threads},
        "old_fp32_entry_ms": old_ms,
        "new_bf16_entry_ms": new_ms,
        "speedup": old_ms / new_ms,
        "old_rows_ms": old_rows,
        "new_rows_ms": new_rows,
    })


if __name__ == "__main__":
    main()
