#!/usr/bin/env python3
"""Real-graph gate for explicit wide-K static SX caching.

This times only the first-layer forward/backward compute contract.  Cache
construction is reported separately and excluded from steady samples because
the full-batch feature matrix and graph are immutable across epochs.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tfs_train.datasets import load_graphsaint
from tfs_train.native import backend


def median(values):
    return float(statistics.median(values))


def timed(fn, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1.0e3)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.manual_seed(20260911)
    torch.set_num_threads(args.threads)
    ds = load_graphsaint(Path(args.root))
    x = ds.x.float().contiguous()
    graph = ds.graph
    n, k = map(int, x.shape)
    d = 128
    weight = torch.randn(k, d, dtype=torch.float32).contiguous()
    bias = torch.randn(d, dtype=torch.float32).contiguous()
    grad = torch.randn(n, d, dtype=torch.float32).contiguous()
    ext = backend()

    build_start = time.perf_counter()
    pulled = ext.c3_prepare_static_aggregate_v3(
        x, graph.scale, graph.rowptr, graph.colidx, args.threads)
    cache_build_ms = (time.perf_counter() - build_start) * 1.0e3

    def dynamic_step():
        out, hs = ext.c3_forward_amx_v2(
            x, weight, bias, graph.rowptr, graph.colidx, graph.scale,
            args.threads, True)
        _, dw, db, _ = ext.c3_backward_amx_v2(
            grad, hs, weight, graph.rowptr, graph.colidx, graph.scale,
            args.threads, False)
        return out, dw, db

    def cached_step():
        out, _ = ext.c3_forward_cached_aggregate_amx_v3(
            x, pulled, weight, bias, graph.rowptr, graph.colidx,
            graph.scale, args.threads)
        dw, db, _ = ext.c3_backward_cached_aggregate_amx_v3(
            grad, pulled, graph.rowptr, graph.colidx, graph.scale,
            args.threads)
        return out, dw, db

    dynamic = dynamic_step()
    cached = cached_step()
    errors = {}
    for name, lhs, rhs in zip(("out", "dw", "db"), dynamic, cached):
        diff = (lhs - rhs).abs()
        errors[name + "_max_abs"] = float(diff.max())
        errors[name + "_rel_l2"] = float(
            torch.linalg.vector_norm(diff) /
            torch.linalg.vector_norm(lhs).clamp_min(1.0e-12))
        errors[name + "_over_0p2_ratio"] = float((diff > 0.2).float().mean())
    # The two paths intentionally reassociate S(HW) as (SH)W at BF16
    # boundaries.  Near-zero elements make elementwise relative tolerances
    # misleading, so gate the complete tensor norm and report outliers rather
    # than hiding them.
    assert errors["out_rel_l2"] <= 2.0e-2
    assert errors["dw_rel_l2"] <= 2.0e-2
    # db contains the same FP32 values but the two native paths use different
    # worker reduction trees.
    assert errors["db_rel_l2"] <= 2.0e-3
    del dynamic, cached

    for _ in range(args.warmups):
        dynamic_step()
        cached_step()
    dynamic_ms = timed(dynamic_step, args.repeats)
    cached_ms = timed(cached_step, args.repeats)
    result = {
        "dataset": args.name, "nodes": n,
        "adjacency_entries": int(graph.colidx.numel()),
        "k": k, "d": d, "threads": args.threads,
        "cache_bytes": int(pulled.numel() * pulled.element_size()),
        "cache_build_ms": cache_build_ms,
        "dynamic_transform_ms": dynamic_ms,
        "static_cached_ms": cached_ms,
        "dynamic_median_ms": median(dynamic_ms),
        "static_cached_median_ms": median(cached_ms),
        "speedup": median(dynamic_ms) / median(cached_ms),
        "errors": errors,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
