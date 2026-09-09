"""Isolated gate for compact supervision-scoped dW.

Compares the complete native AMX path (layout conversion, thread-local
accumulation, reduction, and scatter included) with PyTorch/oneDNN BF16
matmul.  This does not alter the default TFS execution path.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from tfs_train.native import backend


def elapsed_ms(fn):
    start = time.perf_counter_ns()
    out = fn()
    return out, (time.perf_counter_ns() - start) / 1e6


def stats(values):
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=300000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260906)
    p = torch.randn(args.rows, args.hidden, dtype=torch.bfloat16)
    gs = torch.randn(args.rows, args.classes, dtype=torch.bfloat16)
    ext = backend()

    def framework():
        return torch.matmul(p.transpose(0, 1).contiguous(), gs).float()

    def native():
        return ext.c3_compact_dw_bf16_amx_shadow_v1(
            p, gs, args.threads)

    for _ in range(args.warmups):
        framework()
        native()

    framework_ms, native_ms, paired = [], [], []
    last_framework = last_native = None
    rng = random.Random(20260906)
    for _ in range(args.repeats):
        order = ["framework", "native"]
        rng.shuffle(order)
        record = {}
        for name in order:
            if name == "framework":
                last_framework, record[name] = elapsed_ms(framework)
                framework_ms.append(record[name])
            else:
                last_native, record[name] = elapsed_ms(native)
                native_ms.append(record[name])
        paired.append(record["framework"] / record["native"])

    delta = last_native - last_framework
    ref_norm = float(torch.linalg.vector_norm(last_framework.double()))
    result = {
        "status": "pass",
        "contract": "compact_dw_amx_shadow_gate_v1",
        "rows": args.rows,
        "hidden": args.hidden,
        "classes": args.classes,
        "threads": args.threads,
        "framework": stats(framework_ms),
        "native": stats(native_ms),
        "speedup": statistics.median(framework_ms) /
                   statistics.median(native_ms),
        "paired_speedup_median": statistics.median(paired),
        "correctness": {
            "max_abs": float(delta.abs().max()),
            "relative_l2": float(torch.linalg.vector_norm(delta.double()) /
                                 max(ref_norm, 1e-30)),
        },
    }
    if result["correctness"]["relative_l2"] > 0.01:
        result["status"] = "fail"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
