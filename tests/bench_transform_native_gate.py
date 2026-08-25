"""Component timing for Python versus native Transform-HighD slab."""

from __future__ import annotations

import json
import os
import statistics
import time

import torch

from tfs_train.execution_plan import build_layer_plan
from tfs_train.highd_backward import (native_transform_highd_backward,
                                       streamed_transform_backward)


def ring(n: int):
    rows = torch.arange(n, dtype=torch.long)
    return (torch.arange(0, 3 * n + 1, 3, dtype=torch.long),
            torch.stack((rows, (rows - 1) % n, (rows + 1) % n), dim=1)
            .reshape(-1).contiguous())


def main() -> None:
    n = int(os.environ.get("TFS_HIGHD_N", "4103"))
    k = int(os.environ.get("TFS_HIGHD_K", "1024"))
    d = int(os.environ.get("TFS_HIGHD_D", "257"))
    threads = int(os.environ.get("TFS_HIGHD_THREADS", "32"))
    repeats = int(os.environ.get("TFS_HIGHD_REPEATS", "5"))
    torch.set_num_threads(1)
    torch.manual_seed(20260818)
    hs = torch.randn(n, k).to(torch.bfloat16)
    weight = torch.randn(k, d)
    grad = torch.randn(n, d)
    scale = torch.rand(n) + 0.5
    rowptr, colidx = ring(n)
    os.environ["TFS_TRANSFORM_HIGHD_STREAM_V1"] = "on"
    os.environ["TFS_TRANSFORM_HIGHD_NATIVE"] = "off"
    os.environ["TFS_GLUE_E2_VEC_STORE"] = "1"
    os.environ["TFS_COLIDX"] = "int64"
    plan = build_layer_plan(n, k, d, threads=threads, compute_dx=True)
    if plan.selected_impl != "transform_stream":
        raise AssertionError(plan.selected_impl)
    for _ in range(2):
        streamed_transform_backward(hs, weight, grad, scale, rowptr, colidx,
                                    threads, True, plan=plan)
        native_transform_highd_backward(hs, weight, grad, scale, rowptr,
                                        colidx, threads, True,
                                        d_slabs=plan.d_slabs,
                                        d_tile=plan.d_tile)
    py_rows, native_rows = [], []
    for _ in range(repeats):
        t0 = time.perf_counter()
        streamed_transform_backward(hs, weight, grad, scale, rowptr, colidx,
                                    threads, True, plan=plan)
        py_rows.append((time.perf_counter() - t0) * 1000.0)
        t0 = time.perf_counter()
        native_transform_highd_backward(hs, weight, grad, scale, rowptr,
                                        colidx, threads, True,
                                        d_slabs=plan.d_slabs,
                                        d_tile=plan.d_tile)
        native_rows.append((time.perf_counter() - t0) * 1000.0)
    py_median = statistics.median(py_rows)
    native_median = statistics.median(native_rows)
    result = {
        "shape": {"N": n, "K": k, "D": d, "threads": threads},
        "warmup": 2,
        "repeats": repeats,
        "python_stream_ms": py_median,
        "native_slab_ms": native_median,
        "speedup_native_over_python": py_median / native_median,
        "python_rows_ms": py_rows,
        "native_rows_ms": native_rows,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

