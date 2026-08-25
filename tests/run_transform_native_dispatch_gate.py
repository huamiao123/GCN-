"""Integrated WideOutputAMX dispatch gate for the native transform candidate."""

from __future__ import annotations

import json
import os
import time

import torch

from tfs_train.dimension_dispatch import WideOutputAMX
from tfs_train.execution_plan import build_layer_plan


def ring(n: int):
    rows = torch.arange(n, dtype=torch.long)
    return (torch.arange(0, 3 * n + 1, 3, dtype=torch.long),
            torch.stack((rows, (rows - 1) % n, (rows + 1) % n), dim=1)
            .reshape(-1).contiguous())


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() /
                 (b.float().norm() + 1.0e-6))


def run(native_mode: str, x0, w0, b0, rowptr, colidx, scale, grad, threads):
    os.environ["TFS_TRANSFORM_HIGHD_NATIVE"] = native_mode
    os.environ["TFS_TRANSFORM_HIGHD_STREAM_V1"] = "on"
    plan = build_layer_plan(x0.size(0), w0.size(0), w0.size(1),
                            threads=threads, compute_dx=True)
    expected = "native_transform" if native_mode != "off" else "transform_stream"
    if plan.selected_impl != expected:
        raise AssertionError((native_mode, plan.selected_impl, expected))
    x = x0.detach().clone().requires_grad_(True)
    weight = w0.detach().clone().requires_grad_(True)
    bias = b0.detach().clone().requires_grad_(True)
    t0 = time.perf_counter()
    out = WideOutputAMX.apply(x, weight, bias, rowptr, colidx, scale,
                              None, threads, plan)
    out.backward(grad)
    return {
        "out": out.detach(), "dx": x.grad.detach(),
        "dw": weight.grad.detach(), "db": bias.grad.detach(),
        "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
        "selected_impl": plan.selected_impl,
    }


def main() -> None:
    torch.set_num_threads(1)
    n = int(os.environ.get("TFS_HIGHD_N", "4103"))
    k = int(os.environ.get("TFS_HIGHD_K", "1024"))
    d = int(os.environ.get("TFS_HIGHD_D", "257"))
    threads = int(os.environ.get("TFS_HIGHD_THREADS", "4"))
    torch.manual_seed(20260818)
    x = torch.randn(n, k)
    weight = torch.randn(k, d)
    bias = torch.randn(d)
    scale = torch.rand(n) + 0.5
    grad = torch.randn(n, d)
    rowptr, colidx = ring(n)
    os.environ["HYBRID_AMX_FORWARD"] = "1"
    os.environ["HYBRID_AMX_BACKWARD"] = "1"
    os.environ["TFS_GLUE_E2_VEC_STORE"] = "1"
    os.environ["TFS_COLIDX"] = "int64"
    native_mode = os.environ.get("TFS_NATIVE_MODE", "on")
    for _ in range(2):
        run("off", x, weight, bias, rowptr, colidx, scale, grad, threads)
        run(native_mode, x, weight, bias, rowptr, colidx, scale, grad, threads)
    old_rows, new_rows = [], []
    old = new = None
    for _ in range(5):
        old = run("off", x, weight, bias, rowptr, colidx, scale, grad, threads)
        new = run(native_mode, x, weight, bias, rowptr, colidx, scale, grad,
                  threads)
        old_rows.append(old)
        new_rows.append(new)
    old = old_rows[-1]
    new = new_rows[-1]
    result = {
        "shape": {"N": n, "K": k, "D": d, "threads": threads},
        "old_impl": old["selected_impl"],
        "new_impl": new["selected_impl"],
        "warmup": 2,
        "measured": 5,
        "elapsed_ms": {"stream_median": float(torch.tensor(
                           [row["elapsed_ms"] for row in old_rows]).median()),
                       "native_median": float(torch.tensor(
                           [row["elapsed_ms"] for row in new_rows]).median())},
        "speedup": float(torch.tensor(
            [row["elapsed_ms"] for row in old_rows]).median() /
            torch.tensor([row["elapsed_ms"] for row in new_rows]).median()),
        "relative_l2": {key: rel(new[key], old[key])
                        for key in ("out", "dx", "dw", "db")},
    }
    for key, limit in {"out": 5.0e-3, "dx": 5.0e-2,
                       "dw": 5.0e-2, "db": 2.0e-3}.items():
        if result["relative_l2"][key] > limit:
            raise AssertionError(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
