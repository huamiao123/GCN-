#!/usr/bin/env python3
"""Fair packed native AMX Q versus PyTorch BF16 matmul."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tfs_train.native import backend


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=100_003)
    parser.add_argument("--panels", type=int, default=1)
    parser.add_argument("--d", type=int, default=2983)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(20260911)
    torch.set_num_threads(args.threads)
    if args.panels <= 0:
        raise ValueError("panels must be positive")
    gs_panels = [
        torch.randn(args.rows, args.d).to(torch.bfloat16).contiguous()
        for _ in range(args.panels)
    ]
    weight = torch.randn(args.k, args.d).to(torch.bfloat16).contiguous()
    weight_t = weight.transpose(0, 1).contiguous()
    ext = backend()
    pack_start = time.perf_counter()
    packed = ext.c3_pack_compact_q_weight_amx_shadow_v1(weight)
    pack_ms = (time.perf_counter() - pack_start) * 1.0e3

    q = torch.empty((args.rows * args.panels, args.k), dtype=torch.bfloat16)

    def baseline():
        for panel, gs in enumerate(gs_panels):
            r0 = panel * args.rows
            q[r0:r0 + args.rows].copy_(torch.matmul(gs, weight_t))
        return q

    def candidate():
        for panel, gs in enumerate(gs_panels):
            r0 = panel * args.rows
            q[r0:r0 + args.rows].copy_(
                ext.c3_compact_q_packed_bf16_amx_shadow_v1(
                    gs, packed, args.k, args.threads))
        return q

    old = baseline().clone()
    new = candidate().clone()
    rel_l2 = float(torch.linalg.vector_norm(old.float() - new.float()) /
                   torch.linalg.vector_norm(old.float()).clamp_min(1e-12))
    if rel_l2 > 1.0e-2:
        raise AssertionError(f"Q relative L2 {rel_l2:.6g} exceeds 1%")
    del old, new
    for _ in range(args.warmups):
        baseline(); candidate()
    old_ms, new_ms = [], []
    for repeat in range(args.repeats):
        pair = ((old_ms, baseline), (new_ms, candidate))
        if repeat % 2:
            pair = tuple(reversed(pair))
        for samples, function in pair:
            start = time.perf_counter()
            result = function()
            samples.append((time.perf_counter() - start) * 1.0e3)
            del result
    old_median = float(statistics.median(old_ms))
    new_median = float(statistics.median(new_ms))
    payload = {
        "rows_per_panel": args.rows, "panels": args.panels,
        "total_rows": args.rows * args.panels, "d": args.d, "k": args.k,
        "threads": args.threads, "pack_once_ms": pack_ms,
        "baseline_samples_ms": old_ms, "candidate_samples_ms": new_ms,
        "baseline_median_ms": old_median,
        "candidate_median_ms": new_median,
        "steady_speedup": old_median / new_median,
        "first_panel_speedup_including_pack": old_median / (new_median + pack_ms),
        "relative_l2": rel_l2,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
