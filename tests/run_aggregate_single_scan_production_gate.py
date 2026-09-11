#!/usr/bin/env python3
"""Performance/numerical gate for aggregate High-D one-final-pull.

The baseline invokes the proven aggregate native kernel once per D slab and
therefore traverses CSR once per slab.  The candidate computes the same dP
partials per slab, accumulates them in FP32, then traverses CSR exactly once.
Both measurements include allocation, dense work, sparse work and returns.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tfs_train.execution_plan import partition_d
from tfs_train.highd_backward import (
    HighDBackwardPlan,
    native_aggregate_d_slab_backward,
    streamed_aggregate_single_scan_backward,
)


def ring_csr(n: int, degree: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(n, dtype=torch.int64)
    offsets = torch.arange(1, degree + 1, dtype=torch.int64)
    colidx = ((rows[:, None] + offsets[None, :]) % n).reshape(-1)
    rowptr = torch.arange(0, n * degree + 1, degree, dtype=torch.int64)
    return rowptr.contiguous(), colidx.contiguous()


def quantiles(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "median_ms": float(statistics.median(ordered)),
        "mean_ms": float(statistics.mean(ordered)),
        "min_ms": float(ordered[0]),
        "max_ms": float(ordered[-1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=100_003)
    parser.add_argument("--degree", type=int, default=12)
    parser.add_argument("--csr", default="",
                        help="optional training CSR .pt payload")
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--d", type=int, default=513)
    parser.add_argument("--d-tile", type=int, default=257)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.d <= 128:
        raise ValueError("this gate requires High-D (D > 128)")
    torch.manual_seed(20260911)
    torch.set_num_threads(args.threads)
    n, k, d = args.nodes, args.k, args.d
    slabs = partition_d(d, min(args.d_tile, d), min_native=129)
    max_width = max(d1 - d0 for d0, d1 in slabs)
    if args.csr:
        graph = torch.load(args.csr, map_location="cpu", weights_only=True)
        rowptr = graph["rowptr"].contiguous()
        colidx = graph["colidx"].contiguous()
        scale = graph["scale"].contiguous()
        n = int(rowptr.numel() - 1)
        if tuple(scale.shape) != (n,):
            raise ValueError("training CSR scale shape mismatch")
        graph_kind = "training_csr"
        avg_degree = float(colidx.numel() / n)
    else:
        rowptr, colidx = ring_csr(n, args.degree)
        scale = (torch.rand(n) + 0.5).contiguous()
        graph_kind = "synthetic_ring"
        avg_degree = float(args.degree)
    pulled = torch.randn(n, k).to(torch.bfloat16).contiguous()
    weight = torch.randn(k, d).contiguous()
    grad = torch.randn(n, d).contiguous()
    budget = 2 << 30
    plan = HighDBackwardPlan(
        n=n, k=k, d=d, kp=((k + 63) // 64) * 64,
        dp=((d + 31) // 32) * 32, threads=args.threads,
        row_panel=min(n, 512), d_tile=args.d_tile, strategy="d_slab",
        compute_dx=True, budget_bytes=budget, estimated_panel_bytes=0,
        persistent_dx_bytes=n * k * 4, d_slabs=slabs,
        dslab_max_width=max_width, workspace_bytes=n * k * 4,
        execution_variant="aggregate_highd_single_scan",
    )

    def baseline():
        return native_aggregate_d_slab_backward(
            pulled, weight, grad, scale, rowptr, colidx, args.threads, True,
            d_tile=args.d_tile, d_slabs=slabs)

    def candidate():
        return streamed_aggregate_single_scan_backward(
            pulled, weight, grad, scale, rowptr, colidx, args.threads, True,
            plan=plan)

    old_dx, old_dw, old_db = baseline()
    new_dx, new_dw, new_db = candidate()
    torch.testing.assert_close(new_dw, old_dw, rtol=0, atol=0)
    torch.testing.assert_close(new_db, old_db, rtol=0, atol=0)
    diff = new_dx - old_dx
    rel_l2 = float(torch.linalg.vector_norm(diff) /
                   torch.linalg.vector_norm(old_dx).clamp_min(1.0e-12))
    max_abs = float(diff.abs().max())
    if rel_l2 > 1.0e-2:
        raise AssertionError(f"dX relative L2 {rel_l2:.6g} exceeds 1%")
    del old_dx, old_dw, old_db, new_dx, new_dw, new_db, diff

    for _ in range(args.warmups):
        baseline()
        candidate()

    old_ms: list[float] = []
    new_ms: list[float] = []
    for repeat in range(args.repeats):
        pair = (("baseline", baseline), ("candidate", candidate))
        if repeat % 2:
            pair = tuple(reversed(pair))
        for name, function in pair:
            start = time.perf_counter()
            result = function()
            elapsed = (time.perf_counter() - start) * 1.0e3
            del result
            (old_ms if name == "baseline" else new_ms).append(elapsed)

    old_stats, new_stats = quantiles(old_ms), quantiles(new_ms)
    speedup = old_stats["median_ms"] / new_stats["median_ms"]
    payload = {
        "status": "pass" if speedup > 1.0 else "performance_regression",
        "candidate": "aggregate_highd_one_final_pull",
        "graph_kind": graph_kind, "csr_path": args.csr or None,
        "nodes": n, "nnz": int(colidx.numel()),
        "average_degree": avg_degree, "k": k, "d": d,
        "threads": args.threads, "d_tile": args.d_tile,
        "d_slabs": [list(item) for item in slabs],
        "csr_scans_baseline": len(slabs), "csr_scans_candidate": 1,
        "baseline": {**old_stats, "samples_ms": old_ms},
        "candidate_result": {**new_stats, "samples_ms": new_ms},
        "median_speedup": speedup,
        "correctness": {"dx_relative_l2": rel_l2,
                        "dx_max_abs": max_abs,
                        "dw_db_exact": True},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
