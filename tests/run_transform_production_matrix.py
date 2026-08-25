"""Production correctness matrix for the dimension-driven Transform-HighD gate.

The matrix deliberately compares the existing native wide-output implementation
with the planner-selected streamed implementation.  It is a correctness gate,
not an end-to-end benchmark: each shape is run in a fresh pair of tensors and
the script reports relative max-absolute errors for output and all gradients.

The shapes exercise:
* a one-slab non-multiple D tail (D=257);
* a multi-slab tail (D=513);
* wide-K with both the one-slab and two-slab cases;
* an N tail which is not a panel multiple.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

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


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    den = b.float().abs().max().item() + 1.0e-6
    return float((a.float() - b.float()).abs().max().item() / den)


def _one(stream: bool, shape, seed: int, threads: int):
    n, k, d = map(int, shape)
    rowptr, colidx = _ring(n)
    torch.manual_seed(seed)
    x = torch.randn(n, k).requires_grad_(True)
    weight = torch.randn(k, d).requires_grad_(True)
    bias = torch.randn(d).requires_grad_(True)
    scale = (torch.rand(n) + 0.5).contiguous()
    grad = torch.randn(n, d)
    os.environ["TFS_TRANSFORM_HIGHD_STREAM_V1"] = "on" if stream else "off"
    plan = build_layer_plan(n, k, d, threads=threads, compute_dx=True)
    expected = "transform_stream" if stream else "legacy_wide_output"
    if plan.selected_impl != expected:
        raise AssertionError(
            f"shape={shape} stream={stream} selected={plan.selected_impl} "
            f"expected={expected}")
    t0 = time.perf_counter()
    out = WideOutputAMX.apply(
        x, weight, bias, rowptr, colidx, scale, None, threads, plan)
    out.backward(grad)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "out": out.detach(),
        "dx": x.grad.detach(),
        "dw": weight.grad.detach(),
        "db": bias.grad.detach(),
        "elapsed_ms": elapsed_ms,
        "plan_id": plan.plan_id,
        "selected_impl": plan.selected_impl,
        "d_slabs": [list(item) for item in plan.d_slabs],
    }


def main() -> None:
    os.environ["HYBRID_AMX_FORWARD"] = "1"
    os.environ["HYBRID_AMX_BACKWARD"] = "1"
    # Keep this gate comparing the established stream implementation with the
    # legacy wide-output wrapper.  Do not inherit a shell's native-transform
    # probe switch, which would silently change the selected implementation.
    os.environ["TFS_TRANSFORM_HIGHD_NATIVE"] = "off"
    os.environ["TFS_TRANSFORM_HIGHD_SINGLE_SCAN"] = "off"
    os.environ.setdefault("TFS_HIGHD_STREAM_V1", "on")
    torch.set_num_threads(1)
    threads = int(os.environ.get("TFS_HIGHD_THREADS", "4"))
    # Keep the matrix representative but bounded on the shared node.
    shapes = (
        (4097, 512, 257),
        (4103, 1024, 257),
        (4099, 1024, 513),
        (4097, 2048, 513),
    )
    rows = []
    for index, shape in enumerate(shapes):
        seed = 20260818 + index * 11
        # Warm both implementations separately.  The elapsed values below
        # are diagnostic kernel-entry timings only; correctness is the gate.
        _one(False, shape, seed, threads)
        _one(True, shape, seed, threads)
        old = _one(False, shape, seed, threads)
        new = _one(True, shape, seed, threads)
        errors = {
            key: _rel(new[key], old[key])
            for key in ("out", "dx", "dw", "db")
        }
        limits = {"out": 5.0e-3, "dx": 3.5e-2,
                  "dw": 3.0e-2, "db": 2.0e-3}
        for key, limit in limits.items():
            if errors[key] > limit:
                raise AssertionError(
                    f"shape={shape} {key} error={errors[key]:.6g} "
                    f"> {limit}")
        rows.append({
            "shape": {"N": shape[0], "K": shape[1], "D": shape[2],
                      "threads": threads},
            "old_ms": old["elapsed_ms"],
            "stream_ms": new["elapsed_ms"],
            "speedup_kernel_entry": old["elapsed_ms"] /
                                     max(new["elapsed_ms"], 1.0e-9),
            "relative_error": errors,
            "plan_id": new["plan_id"],
            "d_slabs": new["d_slabs"],
        })
    result = {
        "status": "pass",
        "timing_note": "diagnostic kernel-entry timing after one warmup; not E2E",
        "shapes": rows,
    }
    output_name = os.environ.get("TFS_TRANSFORM_MATRIX_OUTPUT")
    if output_name:
        Path(output_name).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
