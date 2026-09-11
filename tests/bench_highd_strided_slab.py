#!/usr/bin/env python3
"""A/B the historical contiguous D-slab adapter against strided views."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tfs_train.native import backend


def ring_csr(n, degree):
    rows = torch.arange(n, dtype=torch.int64)
    offsets = torch.arange(1, degree + 1, dtype=torch.int64)
    colidx = ((rows[:, None] + offsets[None, :]) % n).reshape(-1)
    rowptr = torch.arange(0, n * degree + 1, degree, dtype=torch.int64)
    return rowptr.contiguous(), colidx.contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("transform", "aggregate"),
                        required=True)
    parser.add_argument("--nodes", type=int, default=100_003)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.manual_seed(20260911)
    torch.set_num_threads(args.threads)
    n, full_d, d0, width = args.nodes, 513, 128, 257
    k = 1024 if args.family == "transform" else 128
    grad_parent = torch.randn(n, full_d)
    weight_parent = torch.randn(k, full_d)
    grad = grad_parent.narrow(1, d0, width)
    weight = weight_parent.narrow(1, d0, width)
    scale = (torch.rand(n) + 0.5).contiguous()
    rowptr, colidx = ring_csr(n, 8)
    operand = torch.randn(n, k).to(torch.bfloat16).contiguous()
    ext = backend()
    native = (ext.c3_backward_transform_highd_amx_v1
              if args.family == "transform" else
              ext.c3_backward_aggregate_highd_amx_v1)

    def old_adapter():
        return native(grad.contiguous(), operand, weight.contiguous(),
                      rowptr, colidx, scale, args.threads, True)

    def strided_adapter():
        return native(grad, operand, weight, rowptr, colidx, scale,
                      args.threads, True)

    old = old_adapter()
    new = strided_adapter()
    for lhs, rhs in zip(old[:3], new[:3]):
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
    del old, new
    for _ in range(args.warmups):
        old_adapter(); strided_adapter()

    old_ms, new_ms = [], []
    for repeat in range(args.repeats):
        pair = (("old", old_adapter), ("new", strided_adapter))
        if repeat % 2:
            pair = tuple(reversed(pair))
        for name, function in pair:
            start = time.perf_counter()
            function()
            elapsed = (time.perf_counter() - start) * 1.0e3
            (old_ms if name == "old" else new_ms).append(elapsed)

    old_median = float(statistics.median(old_ms))
    new_median = float(statistics.median(new_ms))
    result = {
        "family": args.family, "nodes": n, "degree": 8, "k": k,
        "full_d": full_d, "slab_start": d0, "slab_width": width,
        "threads": args.threads, "old_contiguous_ms": old_ms,
        "new_strided_ms": new_ms, "old_median_ms": old_median,
        "new_median_ms": new_median, "speedup": old_median / new_median,
        "avoided_copy_bytes_per_slab": int(
            (grad.numel() + weight.numel()) * 4),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
