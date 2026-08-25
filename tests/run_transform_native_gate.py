"""Standalone server-side numerical gate for native Transform-HighD."""

from __future__ import annotations

import json
import os

import torch

from tfs_train.execution_plan import build_layer_plan
from tfs_train.highd_backward import (native_transform_highd_backward,
                                       streamed_transform_backward)


def ring(n: int):
    rows = torch.arange(n, dtype=torch.long)
    return (torch.arange(0, 3 * n + 1, 3, dtype=torch.long),
            torch.stack((rows, (rows - 1) % n, (rows + 1) % n), dim=1)
            .reshape(-1).contiguous())


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).norm() /
                 (b.float().norm() + 1.0e-6))


def main() -> None:
    n = int(os.environ.get("TFS_HIGHD_N", "4097"))
    k = int(os.environ.get("TFS_HIGHD_K", "1024"))
    d = int(os.environ.get("TFS_HIGHD_D", "257"))
    threads = int(os.environ.get("TFS_HIGHD_THREADS", "4"))
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
    if plan.selected_impl != "transform_highd_stream":
        raise AssertionError(plan.selected_impl)
    ref_dx, ref_dw = streamed_transform_backward(
        hs, weight, grad, scale, rowptr, colidx, threads, True, plan=plan)
    new_dx, new_dw = native_transform_highd_backward(
        hs, weight, grad, scale, rowptr, colidx, threads, True,
        d_slabs=plan.d_slabs, d_tile=plan.d_tile)
    assert ref_dx is not None and new_dx is not None
    result = {
        "shape": {"N": n, "K": k, "D": d, "threads": threads},
        "slabs": list(plan.d_slabs),
        "relative_l2": {"dx": rel(new_dx, ref_dx),
                        "dw": rel(new_dw, ref_dw)},
        "max_abs": {"dx": float((new_dx - ref_dx).abs().max()),
                    "dw": float((new_dw - ref_dw).abs().max())},
    }
    if result["relative_l2"]["dx"] > 5.0e-2:
        raise AssertionError(result)
    if result["relative_l2"]["dw"] > 5.0e-2:
        raise AssertionError(result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
