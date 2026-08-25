"""Production-dispatch correctness/performance gate for single-scan Transform-HighD."""

from __future__ import annotations

import json
import os
import time

import torch

from tfs_train.dimension_dispatch import WideOutputAMX
from tfs_train.execution_plan import build_layer_plan


def ring(n: int):
    rows = torch.arange(n, dtype=torch.long)
    return (torch.arange(0, 5 * n + 1, 5, dtype=torch.long),
            torch.stack((rows, (rows - 1) % n, (rows + 1) % n,
                         (rows - 2) % n, (rows + 2) % n), dim=1)
            .reshape(-1).contiguous())


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a.float() - b.float()) /
                 torch.linalg.vector_norm(b.float()).clamp_min(1.0))


def run_one(mode: str, shape, seed: int, threads: int):
    n, k, d = map(int, shape)
    rowptr, colidx = ring(n)
    torch.manual_seed(seed)
    x = torch.randn(n, k).requires_grad_(True)
    weight = torch.randn(k, d).requires_grad_(True)
    bias = torch.randn(d).requires_grad_(True)
    scale = torch.rand(n) + 0.5
    grad = torch.randn(n, d)
    os.environ["TFS_TRANSFORM_HIGHD_NATIVE"] = "off"
    os.environ["TFS_TRANSFORM_HIGHD_STREAM_V1"] = "on"
    os.environ["TFS_TRANSFORM_HIGHD_SINGLE_SCAN"] = mode
    plan = build_layer_plan(n, k, d, threads=threads, compute_dx=True)
    expected = ("transform_highd_single_scan" if mode == "on"
                else "transform_highd_stream")
    if plan.selected_impl != expected:
        raise AssertionError((shape, mode, plan.selected_impl, expected))
    t0 = time.perf_counter()
    out = WideOutputAMX.apply(x, weight, bias, rowptr, colidx, scale,
                              None, threads, plan)
    out.backward(grad)
    return {
        "out": out.detach(), "dx": x.grad.detach(),
        "dw": weight.grad.detach(), "db": bias.grad.detach(),
        "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        "selected_impl": plan.selected_impl,
        "plan_id": plan.plan_id,
        "d_slabs": [list(item) for item in plan.d_slabs],
    }


def main():
    os.environ["HYBRID_AMX_FORWARD"] = "1"
    os.environ["HYBRID_AMX_BACKWARD"] = "1"
    os.environ["TFS_HIGHD_NATIVE_PANEL"] = "512"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    threads = int(os.environ.get("TFS_SINGLE_SCAN_THREADS", "4"))
    shapes = ((4099, 1024, 513), (4097, 2048, 513))
    rows = []
    for i, shape in enumerate(shapes):
        seed = 20260818 + i * 17
        # One warmup per arm; subsequent values are diagnostic entry timing.
        run_one("off", shape, seed, threads)
        run_one("on", shape, seed, threads)
        old = run_one("off", shape, seed, threads)
        new = run_one("on", shape, seed, threads)
        errors = {key: rel(new[key], old[key])
                  for key in ("out", "dx", "dw", "db")}
        limits = {"out": 5.0e-3, "dx": 3.5e-2,
                  "dw": 3.0e-2, "db": 2.0e-3}
        for key, limit in limits.items():
            if errors[key] > limit:
                raise AssertionError((shape, key, errors[key], limit))
        rows.append({
            "shape": {"N": shape[0], "K": shape[1], "D": shape[2],
                      "threads": threads},
            "old_ms": old["elapsed_ms"], "single_scan_ms": new["elapsed_ms"],
            "speedup_old_over_single_scan": old["elapsed_ms"] /
                                             max(new["elapsed_ms"], 1.0e-9),
            "relative_l2": errors, "plan_id": new["plan_id"],
            "d_slabs": new["d_slabs"],
        })
    print(json.dumps({"status": "pass", "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
