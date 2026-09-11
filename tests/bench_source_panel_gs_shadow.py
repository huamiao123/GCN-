#!/usr/bin/env python3
"""Gate source-row Gs panels feeding dW/Q before one BF16 sparse pull."""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from tfs_train.native import backend


def graph(n: int, degree: int):
    rows = torch.arange(n, dtype=torch.long).unsqueeze(1)
    offsets = torch.arange(1, degree + 1, dtype=torch.long).unsqueeze(0)
    colidx = ((rows * 17 + offsets * 7919) % n).reshape(-1).contiguous()
    rowptr = torch.arange(0, n * degree + 1, degree, dtype=torch.long)
    return rowptr, colidx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, default=100_003)
    p.add_argument("--degree", type=int, default=12)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--row-panel", type=int, default=25_000)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.manual_seed(20260912 + args.d + args.degree)
    torch.set_num_threads(args.threads)
    n, k, d = args.nodes, args.k, args.d
    rowptr, colidx = graph(n, args.degree)
    scale = (torch.rand(n) + 0.25).contiguous()
    grad = torch.randn(n, d)
    pulled = torch.randn(n, k).to(torch.bfloat16).contiguous()
    weight = torch.randn(k, d)
    weight_t = weight.to(torch.bfloat16).transpose(0, 1).contiguous()
    ext = backend()

    def materialized():
        return ext.c3_backward_aggregate_saved_amx_v4(
            grad, pulled, weight, rowptr, colidx, scale,
            args.threads, True)

    def source_panel():
        q = torch.empty((n, k), dtype=torch.bfloat16)
        dw = torch.zeros((k, d), dtype=torch.float32)
        for r0 in range(0, n, args.row_panel):
            rows = min(args.row_panel, n - r0)
            gs = ext.c3_scale_grad_bf16_v1(
                grad.narrow(0, r0, rows), scale.narrow(0, r0, rows),
                args.threads)
            pp = pulled.narrow(0, r0, rows)
            dw.add_(torch.matmul(pp.transpose(0, 1).contiguous(), gs).float())
            q.narrow(0, r0, rows).copy_(torch.matmul(gs, weight_t))
        dh = ext.c3_pull_only_bf16_amx_v1(
            q, rowptr, colidx, args.threads)
        dx = dh.float().mul_(scale.unsqueeze(1))
        return dx, dw, grad.sum(0)

    old, new = materialized(), source_panel()
    relative_l2 = {}
    for name, a, b in zip(("dx", "dw", "db"), old[:3], new):
        relative_l2[name] = float(
            torch.linalg.vector_norm(a.float() - b.float()) /
            torch.linalg.vector_norm(a.float()).clamp_min(1e-12))
    del old, new
    for _ in range(args.warmups):
        materialized(); source_panel()
    samples = {"materialized": [], "source_panel": []}
    for repeat in range(args.repeats):
        order = (("materialized", materialized), ("source_panel", source_panel))
        if repeat % 2:
            order = tuple(reversed(order))
        for name, fn in order:
            start = time.perf_counter()
            result = fn()
            samples[name].append((time.perf_counter() - start) * 1e3)
            del result
    medians = {name: statistics.median(values)
               for name, values in samples.items()}
    dp = (d + 31) // 32 * 32
    payload = {
        "nodes": n, "degree": args.degree, "k": k, "d": d,
        "threads": args.threads, "row_panel": args.row_panel,
        "materialized_gs_bytes": n * dp * 2,
        "candidate_peak_gs_bytes": min(n, args.row_panel) * dp * 2,
        "candidate_q_bytes": n * k * 2,
        "relative_l2": relative_l2, "samples_ms": samples,
        "median_ms": medians,
        "speedup": medians["materialized"] / medians["source_panel"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
