"""Correctness gate for the transform-first High-D D-slab stream."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import torch

from tfs_train.dimension_dispatch import WideOutputAMX


def _ring(n: int):
    rows = torch.arange(n, dtype=torch.long)
    return (torch.arange(0, 3 * n + 1, 3, dtype=torch.long),
            torch.stack((rows, (rows - 1) % n, (rows + 1) % n), dim=1)
            .reshape(-1).contiguous())


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max() /
                 (b.float().abs().max() + 1.0e-6))


def _run(stream: bool, x0, w0, b0, rowptr, colidx, scale, grad, threads):
    os.environ["TFS_TRANSFORM_HIGHD_STREAM_V1"] = "1" if stream else "0"
    x = x0.detach().clone().requires_grad_(True)
    weight = w0.detach().clone().requires_grad_(True)
    bias = b0.detach().clone().requires_grad_(True)
    t0 = time.perf_counter()
    out = WideOutputAMX.apply(x, weight, bias, rowptr, colidx, scale,
                              None, int(threads))
    forward_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    out.backward(grad)
    backward_ms = (time.perf_counter() - t0) * 1000.0
    return {"out": out.detach(), "dx": x.grad.detach(),
            "dw": weight.grad.detach(), "db": bias.grad.detach(),
            "forward_ms": forward_ms, "backward_ms": backward_ms}


def main() -> None:
    torch.set_num_threads(1)
    threads = int(os.environ.get("TFS_HIGHD_THREADS", "4"))
    n = int(os.environ.get("TFS_HIGHD_N", "1024"))
    k = int(os.environ.get("TFS_HIGHD_K", "128"))
    d = int(os.environ.get("TFS_HIGHD_D", "256"))
    torch.manual_seed(20260817)
    x = torch.randn(n, k)
    w = torch.randn(k, d)
    b = torch.randn(d)
    scale = torch.rand(n) + 0.5
    grad = torch.randn(n, d)
    rowptr, colidx = _ring(n)
    # Warm both implementations before recording the gate result.
    _run(False, x, w, b, rowptr, colidx, scale, grad, threads)
    _run(True, x, w, b, rowptr, colidx, scale, grad, threads)
    old = _run(False, x, w, b, rowptr, colidx, scale, grad, threads)
    new = _run(True, x, w, b, rowptr, colidx, scale, grad, threads)
    result = {
        "shape": {"N": n, "K": k, "D": d, "threads": threads},
        "old_ms": {"forward": old["forward_ms"],
                   "backward": old["backward_ms"]},
        "new_ms": {"forward": new["forward_ms"],
                   "backward": new["backward_ms"]},
        "relative_error": {key: _rel(new[key], old[key])
                           for key in ("out", "dx", "dw", "db")},
    }
    assert result["relative_error"]["out"] <= 5.0e-3
    assert result["relative_error"]["dx"] <= 3.5e-2
    assert result["relative_error"]["dw"] <= 3.0e-2
    assert result["relative_error"]["db"] <= 2.0e-3
    output = os.environ.get("HYBRID_HIGHD_SMOKE_OUTPUT")
    if output:
        Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
