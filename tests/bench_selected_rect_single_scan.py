#!/usr/bin/env python3
"""Same-node microbenchmark for selected/rectangular sparse pull kernels."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from tfs_train.native import backend


def cuts(rows: int, threads: int) -> torch.Tensor:
    return torch.tensor([rows * i // threads for i in range(threads + 1)],
                        dtype=torch.int64)


def median_ms(call, warmups=3, repeats=9):
    for _ in range(warmups):
        call()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(samples), samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--nodes", type=int, default=50_000)
    parser.add_argument("--degree", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20260910)

    n, degree = args.nodes, args.degree
    rows = torch.arange(n, dtype=torch.int64).repeat_interleave(degree)
    offsets = torch.arange(1, degree + 1, dtype=torch.int64).repeat(n)
    colidx = torch.remainder(rows * 17 + offsets * 7919, n)
    colidx = torch.where(colidx == rows, torch.remainder(colidx + 1, n), colidx)
    rowptr = torch.arange(0, n * degree + 1, degree, dtype=torch.int64)
    selected = torch.arange(0, n, 2, dtype=torch.int64)
    rect_rowptr, rect_colidx = backend().c3_build_selected_transpose_shadow_v1(
        selected, rowptr, colidx)
    scale = torch.linspace(0.25, 1.25, n, dtype=torch.float32)

    records = []
    for k in (32, 64, 128):
        hs = torch.randn(n, k).to(torch.bfloat16).contiguous()
        q = torch.randn(selected.numel(), k).to(torch.bfloat16).contiguous()
        for threads in (1, 8, 32):
            selected_schedule = cuts(selected.numel(), threads)
            rect_schedule = cuts(n, threads)
            calls = {
                "selected_bf16": lambda: backend().c3_selected_pull_bf16_shadow_v1(
                    hs, selected, rowptr, colidx, selected_schedule, threads),
                "rect_bf16": lambda: backend().c3_rect_pull_bf16_shadow_v1(
                    q, rect_rowptr, rect_colidx, rect_schedule, threads),
                "rect_scaled_fp32": lambda: backend().c3_rect_pull_bf16_scaled_fp32_shadow_v1(
                    q, rect_rowptr, rect_colidx, scale, rect_schedule, threads),
            }
            for kernel, call in calls.items():
                median, samples = median_ms(call)
                records.append({
                    "label": args.label, "kernel": kernel, "k": k,
                    "threads": threads, "median_ms": median,
                    "samples_ms": samples})

    Path(args.output).write_text(
        json.dumps({"label": args.label, "nodes": n, "degree": degree,
                    "records": records}, indent=2), encoding="utf-8")
    print(json.dumps({"label": args.label, "records": len(records)}))


if __name__ == "__main__":
    main()
