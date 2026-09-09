"""Same-process comparison of PyTorch, padded-T2 AMX, and direct-tail-T4 AMX."""

from __future__ import annotations

import argparse
import json
import os
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=300000)
    p.add_argument("--classes", type=int, default=2983)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repeats", type=int, default=9)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260906)
    pulled = torch.randn(args.rows, 128, dtype=torch.bfloat16)
    gs = torch.randn(args.rows, args.classes, dtype=torch.bfloat16)
    ext = backend()

    def run(name):
        if name == "torch":
            return torch.matmul(pulled.transpose(0, 1).contiguous(), gs).float()
        os.environ["TFS_COMPACT_DW_T4"] = "1" if name == "t4" else "0"
        return ext.c3_compact_dw_bf16_amx_shadow_v1(
            pulled, gs, args.threads)

    for _ in range(args.warmups):
        for name in ("torch", "padded_t2", "t4"):
            run(name)
    values = {name: [] for name in ("torch", "padded_t2", "t4")}
    outputs = {}
    rng = random.Random(20260906)
    for _ in range(args.repeats):
        order = list(values)
        rng.shuffle(order)
        for name in order:
            outputs[name], ms = timed(lambda name=name: run(name))
            values[name].append(ms)
    ref = outputs["torch"]
    correctness = {}
    for name in ("padded_t2", "t4"):
        delta = outputs[name] - ref
        correctness[name] = {
            "max_abs": float(delta.abs().max()),
            "relative_l2": float(
                torch.linalg.vector_norm(delta.double()) /
                torch.linalg.vector_norm(ref.double())),
        }
    medians = {name: statistics.median(v) for name, v in values.items()}
    result = {
        "status": "pass" if all(
            x["relative_l2"] < 0.01 for x in correctness.values()) else "fail",
        "rows": args.rows,
        "classes": args.classes,
        "threads": args.threads,
        "median_ms": medians,
        "speedup_vs_torch": {
            name: medians["torch"] / medians[name]
            for name in ("padded_t2", "t4")
        },
        "t4_speedup_vs_padded_t2": medians["padded_t2"] / medians["t4"],
        "correctness": correctness,
        "raw_ms": values,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "pass":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
