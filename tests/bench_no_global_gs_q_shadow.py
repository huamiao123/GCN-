#!/usr/bin/env python3
"""Gate a source-panel backward with neither global Gs nor global Q."""

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from tfs_train.native import backend
from tfs_train.supervision_scope import _edge_balanced_cuts
from test_source_panel_q_accumulate import symmetric_graph


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nodes", type=int, default=100_003)
    p.add_argument("--half-degree", type=int, default=6)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--row-panel", type=int, default=25_000)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repeats", type=int, default=11)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.manual_seed(20260912 + args.d)
    torch.set_num_threads(args.threads)
    n, k, d = args.nodes, args.k, args.d
    rowptr, colidx = symmetric_graph(n, args.half_degree)
    scale = (torch.rand(n) + 0.25).contiguous()
    grad = torch.randn(n, d)
    pulled = torch.randn(n, k).to(torch.bfloat16).contiguous()
    weight = torch.randn(k, d)
    weight_t = weight.to(torch.bfloat16).transpose(0, 1).contiguous()
    ext = backend()

    build_start = time.perf_counter()
    partitions = []
    metadata_bytes = 0
    for r0 in range(0, n, args.row_panel):
        rows = min(args.row_panel, n - r0)
        ids = torch.arange(r0, r0 + rows, dtype=torch.int64)
        rp, ci = ext.c3_build_selected_transpose_shadow_v1(
            ids, rowptr, colidx)
        schedule = _edge_balanced_cuts(rp, args.threads)
        partitions.append((r0, rows, rp, ci, schedule))
        metadata_bytes += (rp.numel() + ci.numel() + schedule.numel()) * 8
    preprocessing_ms = (time.perf_counter() - build_start) * 1e3

    def materialized():
        return ext.c3_backward_aggregate_saved_amx_v4(
            grad, pulled, weight, rowptr, colidx, scale,
            args.threads, True)[:3]

    def candidate():
        dh_accum = torch.zeros((n, k), dtype=torch.float32)
        dw = torch.zeros((k, d), dtype=torch.float32)
        for r0, rows, rp, ci, schedule in partitions:
            gs = ext.c3_scale_grad_bf16_v1(
                grad.narrow(0, r0, rows), scale.narrow(0, r0, rows),
                args.threads)
            dw.add_(torch.matmul(
                pulled.narrow(0, r0, rows).transpose(0, 1).contiguous(),
                gs).float())
            q = torch.matmul(gs, weight_t)
            ext.c3_rect_pull_bf16_accumulate_fp32_shadow_v1(
                q, rp, ci, schedule, dh_accum, args.threads)
        dx = dh_accum.to(torch.bfloat16).float().mul_(scale.unsqueeze(1))
        return dx, dw, grad.sum(0)

    old, new = materialized(), candidate()
    relative_l2 = {}
    for name, a, b in zip(("dx", "dw", "db"), old, new):
        relative_l2[name] = float(
            torch.linalg.vector_norm(a.float() - b.float()) /
            torch.linalg.vector_norm(a.float()).clamp_min(1e-12))
    del old, new
    for _ in range(args.warmups):
        materialized(); candidate()
    samples = {"materialized": [], "no_global_gs_q": []}
    for repeat in range(args.repeats):
        order = (("materialized", materialized),
                 ("no_global_gs_q", candidate))
        if repeat % 2:
            order = tuple(reversed(order))
        for name, fn in order:
            begin = time.perf_counter()
            result = fn()
            samples[name].append((time.perf_counter() - begin) * 1e3)
            del result
    medians = {name: statistics.median(values)
               for name, values in samples.items()}
    dp = (d + 31) // 32 * 32
    payload = {
        "nodes": n, "degree": 2 * args.half_degree, "k": k, "d": d,
        "threads": args.threads, "row_panel": args.row_panel,
        "panel_count": len(partitions), "preprocessing_ms": preprocessing_ms,
        "partition_metadata_bytes": metadata_bytes,
        "baseline_global_gs_bytes": n * dp * 2,
        "candidate_peak_gs_q_bytes": min(n, args.row_panel) * (dp + k) * 2,
        "candidate_final_dh_bytes": n * k * 4,
        "relative_l2": relative_l2, "samples_ms": samples,
        "median_ms": medians,
        "speedup": medians["materialized"] /
                   medians["no_global_gs_q"],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
