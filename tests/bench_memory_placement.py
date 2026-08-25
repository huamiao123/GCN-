#!/usr/bin/env python3
"""Measure static-Hs placement under the process' external NUMA policy.

The native extension deliberately does not change the process memory policy.
This benchmark makes the policy explicit through ``numactl`` at launch and
records the resulting Hs build/forward cost plus a lightweight ``numastat``
snapshot.  It is an experiment/reporting tool, not part of the standard path.
"""

import os
import re
import statistics
import subprocess
import time

# Freeze the CPU list before importing torch/native code.
if "TFS_WORKER_CPUS" not in os.environ:
    cpus = sorted(os.sched_getaffinity(0))
    ranges = []
    if cpus:
        start = previous = cpus[0]
        for cpu in cpus[1:]:
            if cpu != previous + 1:
                ranges.append(str(start) if start == previous else f"{start}-{previous}")
                start = cpu
            previous = cpu
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
    os.environ["TFS_WORKER_CPUS"] = ",".join(ranges)

import torch

from tfs_train.native import backend


def _numastat_total():
    try:
        text = subprocess.check_output(
            ["numastat", "-p", str(os.getpid())],
            text=True, stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in text.splitlines():
        if not line.strip().startswith("Total"):
            continue
        values = re.findall(r"[-+]?\d+(?:\.\d+)?", line)
        if values:
            return [float(value) for value in values]
    return None


def _numactl_policy():
    try:
        return " ".join(subprocess.check_output(
            ["numactl", "--show"], text=True,
            stderr=subprocess.STDOUT).splitlines())
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main():
    torch.set_num_threads(1)
    n = int(os.environ.get("PLACEMENT_BENCH_N", "131072"))
    k = int(os.environ.get("PLACEMENT_BENCH_K", "128"))
    d = int(os.environ.get("PLACEMENT_BENCH_D", "64"))
    threads = int(os.environ.get("PLACEMENT_BENCH_THREADS", "8"))
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
    before = _numastat_total()
    t0 = time.perf_counter()
    hs = ext.c3_prepare_static_hs_v1(x, scale, threads)
    build_ms = (time.perf_counter() - t0) * 1e3
    after = _numastat_total()

    def run():
        ext.c3_forward_cached_hs_amx_v1(
            x, hs, weight, bias, rowptr, colidx, scale,
            threads, False)

    run()
    samples = []
    for _ in range(8):
        t0 = time.perf_counter(); run(); t1 = time.perf_counter()
        samples.append((t1 - t0) * 1e3)
    print(
        f"policy={_numactl_policy()} "
        f"n={n} k={k} d={d} threads={threads} "
        f"hs_bytes={hs.numel() * hs.element_size()} "
        f"hs_build_ms={build_ms:.3f} "
        f"forward_median_ms={statistics.median(samples[2:]):.3f} "
        f"numastat_before={before} numastat_after={after}")


if __name__ == "__main__":
    main()
