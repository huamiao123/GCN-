#!/usr/bin/env python3
"""Real-Products 128->128 micro gate: accepted native C3 vs V4 candidate.

The result is diagnostic rather than a reason to force V4 into auto.  V4 is
eligible for auto only if a later full matrix demonstrates a >=1.05x median
gain without a >3% graph-level regression.  This gate also asserts that the
final_pre_numa planner continues to select native C3 for a dynamic hidden
layer.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import torch

from tfs_train.execution_plan import build_layer_plan
from tfs_train.native import backend


def relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = left.double() - right.double()
    denom = torch.linalg.vector_norm(right.double()).clamp_min(1.0e-30)
    return float(torch.linalg.vector_norm(delta) / denom)


def time_variant(call, warmups: int, repeats: int):
    measured = []
    last = None
    for step in range(warmups + repeats):
        begin = time.perf_counter_ns()
        current = call()
        elapsed_ms = (time.perf_counter_ns() - begin) / 1.0e6
        if step >= warmups:
            measured.append(elapsed_ms)
        last = current
    return measured, last


def main() -> None:
    os.environ["TFS_INTERNAL_PROFILE"] = "0"
    os.environ["TFS_AGGREGATE_SAVED"] = "auto"
    os.environ["HYBRID_AMX_FORWARD"] = "1"
    os.environ["HYBRID_AMX_BACKWARD"] = "1"
    warmups = int(os.environ.get("TFS_MICRO_WARMUPS", "2"))
    repeats = int(os.environ.get("TFS_MICRO_REPEATS", "5"))
    # The extension sizes its persistent worker pool on its first invocation
    # (right-sized-pool is intentionally enabled in the authority profile).
    # This gate compares several logical thread counts in one interpreter, so
    # instantiate the pool at the largest requested count first and then only
    # shrink it.  Formal 200-epoch cells use one fresh process per count.
    thread_values = tuple(sorted(
        (int(value) for value in os.environ.get(
            "TFS_MICRO_THREADS", "1,2,4,8,16,32").split(",")),
        reverse=True))
    payload = torch.load(os.environ["PRODUCTS_CACHE"], map_location="cpu",
                         weights_only=True)
    rowptr = payload["rowptr"].contiguous()
    colidx = payload["colidx"].contiguous()
    scale = payload["scale"].float().contiguous()
    n = int(rowptr.numel() - 1)
    torch.set_num_interop_threads(1)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(20260819)
    x = torch.randn(n, 128, generator=generator).contiguous()
    weight = (torch.randn(128, 128, generator=generator) /
              (128.0 ** 0.5)).contiguous()
    bias = torch.randn(128, generator=generator).contiguous()
    grad = torch.randn(n, 128, generator=generator).contiguous()

    rows = []
    for threads in thread_values:
        plan = build_layer_plan(
            n, 128, 128, compute_dx=True, input_static=False,
            graph_static=True, feature_static=False, threads=threads)
        if plan.execution_variant != "native_c3":
            raise AssertionError(
                f"dynamic 128->128 must select native_c3, got {plan.log_line()}")

        def native_call():
            out, hs = backend().c3_forward_amx_v2(
                x, weight, bias, rowptr, colidx, scale, threads, False)
            dx, dw, db, _ = backend().c3_backward_amx_v2(
                grad, hs, weight, rowptr, colidx, scale, threads, True)
            return out, dx, dw, db

        def v4_call():
            out, _hs, pulled = backend().c3_forward_aggregate_saved_amx_v4(
                x, weight, bias, rowptr, colidx, scale, threads)
            dx, dw, db, _ = backend().c3_backward_aggregate_saved_amx_v4(
                grad, pulled, weight, rowptr, colidx, scale, threads, True)
            return out, dx, dw, db

        native_ms, native = time_variant(native_call, warmups, repeats)
        v4_ms, v4 = time_variant(v4_call, warmups, repeats)
        errors = {
            name: relative_l2(candidate, accepted)
            for name, candidate, accepted in zip(
                ("out", "dx", "dw", "db"), v4, native)
        }
        limits = {"out": 5.0e-3, "dx": 5.0e-2,
                  "dw": 5.0e-2, "db": 2.0e-3}
        for name, limit in limits.items():
            if errors[name] > limit:
                raise AssertionError(
                    f"V4 numerical mismatch threads={threads} "
                    f"tensor={name} error={errors[name]} limit={limit}")
        native_median = statistics.median(native_ms)
        v4_median = statistics.median(v4_ms)
        rows.append({
            "threads": threads,
            "warmups": warmups,
            "repeats": repeats,
            "native_c3_ms": native_ms,
            "aggregate_saved_v4_ms": v4_ms,
            "native_c3_median_ms": native_median,
            "aggregate_saved_v4_median_ms": v4_median,
            "speedup_v4_over_native": native_median / max(v4_median, 1.0e-9),
            "v4_auto_gain_requirement_met":
                native_median / max(v4_median, 1.0e-9) >= 1.05,
            "relative_l2": errors,
            "authority_execution_variant": plan.execution_variant,
            "plan_id": plan.plan_id,
        })

    payload_out = {
        "status": "pass",
        "graph": "ogbn-products",
        "shape": {"N": n, "K": 128, "D": 128,
                  "compute_dx": True, "input_static": False},
        "candidate_policy": (
            "V4 remains explicit unless >=1.05x median gain and the full "
            "graph matrix has no >3% regression"),
        "rows": rows,
    }
    output = Path(os.environ["TFS_MICRO_OUTPUT"])
    output.write_text(json.dumps(payload_out, indent=2) + "\n")
    print(json.dumps(payload_out))


if __name__ == "__main__":
    main()
