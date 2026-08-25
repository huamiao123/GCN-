"""Standalone Linux runner for the opt-in production transform High-D gate.

Unlike the pytest wrapper, this file has no pytest dependency.  It is useful
on the cluster's lean PyTorch module where the compiled extension is present
but the optional data-loading stack is not installed.
"""

from __future__ import annotations

import os
import sys
import types

# Importing the package exposes dataset helpers whose pandas dependency is
# optional for this kernel-only gate.  Keep the runner independent of pandas.
sys.modules.setdefault("pandas", types.ModuleType("pandas"))

import torch

from tfs_train.dimension_dispatch import WideOutputAMX
from tfs_train.execution_plan import build_layer_plan


def _ring(n: int):
    rows = torch.arange(n, dtype=torch.long)
    return (
        torch.arange(0, 3 * n + 1, 3, dtype=torch.long),
        torch.stack((rows, (rows - 1) % n, (rows + 1) % n), dim=1)
        .reshape(-1).contiguous(),
    )


def main() -> None:
    os.environ["HYBRID_AMX_FORWARD"] = "1"
    os.environ["HYBRID_AMX_BACKWARD"] = "1"
    torch.manual_seed(20260818)
    n, k, d, threads = 4103, 1024, 257, 4
    rowptr, colidx = _ring(n)
    x0 = torch.randn(n, k)
    w0 = torch.randn(k, d)
    b0 = torch.randn(d)
    scale = torch.rand(n) + 0.5
    grad = torch.randn(n, d)

    def run(stream: bool):
        os.environ["TFS_TRANSFORM_HIGHD_STREAM_V1"] = (
            "on" if stream else "off")
        plan = build_layer_plan(n, k, d, threads=threads, compute_dx=True)
        if plan.selected_impl != ("transform_stream" if stream
                                  else "legacy_wide_output"):
            raise AssertionError(plan.selected_impl)
        x = x0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        out = WideOutputAMX.apply(
            x, w, b, rowptr, colidx, scale, None, threads, plan)
        out.backward(grad)
        return out.detach(), x.grad.detach(), w.grad.detach(), b.grad.detach()

    reference = run(False)
    streamed = run(True)
    tolerances = (5e-3, 3.5e-2, 3e-2, 2e-3)
    for actual, expected, tolerance in zip(streamed, reference, tolerances):
        relative_l2 = (actual.float() - expected.float()).norm() / (
            expected.float().norm() + 1e-6)
        if float(relative_l2) > tolerance:
            raise AssertionError(
                f"relative_l2={float(relative_l2):.6g} > {tolerance}")
    print("transform production gate: PASS")


if __name__ == "__main__":
    main()
