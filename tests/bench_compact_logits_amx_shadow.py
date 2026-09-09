"""Gate the compact high-D AMX logits producer including its epilogue."""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from tfs_train.native import backend


def timed(fn):
    start = time.perf_counter_ns()
    out = fn()
    return out, (time.perf_counter_ns() - start) / 1e6


def error_metric(actual, reference):
    delta = actual - reference
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(
            torch.linalg.vector_norm(delta.double()) /
            torch.linalg.vector_norm(reference.double()).clamp_min(1e-30)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=300000)
    p.add_argument("--classes", type=int, default=2983)
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260906)
    pulled = torch.randn(args.rows, 128, dtype=torch.bfloat16)
    weight = torch.randn(128, args.classes, dtype=torch.bfloat16)
    bias = torch.randn(args.classes, dtype=torch.float32)
    scale = torch.rand(args.rows, dtype=torch.float32)
    ext = backend()

    def framework():
        logits = torch.matmul(pulled, weight).float()
        return logits.mul_(scale.unsqueeze(1)).add_(bias)

    def native():
        return ext.c3_compact_logits_amx_shadow_v1(
            pulled, weight, bias, scale, args.threads)

    for _ in range(args.warmups):
        framework()
        native()
    values = {"framework": [], "native": []}
    outputs = {}
    rng = random.Random(20260906)
    for _ in range(args.repeats):
        order = list(values)
        rng.shuffle(order)
        for name in order:
            outputs[name], ms = timed(framework if name == "framework"
                                      else native)
            values[name].append(ms)
    native_vs_framework = error_metric(
        outputs["native"], outputs["framework"])
    # PyTorch's BF16 matmul rounds its output to BF16 before the FP32
    # scale/bias epilogue, whereas the native AMX path keeps its accumulator in
    # FP32.  Compare both implementations to the same FP32 algebraic oracle so
    # a changed gradient is not automatically mistaken for lower accuracy.
    reference = torch.matmul(pulled.float(), weight.float())
    reference.mul_(scale.unsqueeze(1)).add_(bias)
    framework_vs_fp32 = error_metric(outputs["framework"], reference)
    native_vs_fp32 = error_metric(outputs["native"], reference)
    del reference
    medians = {name: statistics.median(v) for name, v in values.items()}
    result = {
        "status": "pass" if native_vs_framework["relative_l2"] < 0.01 else "fail",
        "rows": args.rows,
        "classes": args.classes,
        "threads": args.threads,
        "median_ms": medians,
        "speedup": medians["framework"] / medians["native"],
        "correctness": {
            "native_vs_framework": native_vs_framework,
            "framework_vs_fp32": framework_vs_fp32,
            "native_vs_fp32": native_vs_fp32,
        },
        "native_contiguous": outputs["native"].is_contiguous(),
        "native_stride": list(outputs["native"].stride()),
        "raw_ms": values,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "pass":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
