"""Native bounded-slab gate for aggregate High-D workspace pressure."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from tfs_train.dimension_dispatch import AggregateWideAMX


def _ring(n: int):
    rows = torch.arange(n, dtype=torch.long)
    return (torch.arange(0, 3 * n + 1, 3, dtype=torch.long),
            torch.stack((rows, (rows - 1) % n, (rows + 1) % n), dim=1)
            .reshape(-1).contiguous())


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max() /
                 (b.float().abs().max() + 1.0e-6))


def _run(native: bool, x0, w0, b0, rowptr, colidx, scale, grad, threads):
    os.environ["TFS_HIGHD_STREAM_V1"] = "1"
    os.environ["TFS_HIGHD_NATIVE_STREAM"] = "auto" if native else "off"
    x = x0.detach().clone().requires_grad_(True)
    w = w0.detach().clone().requires_grad_(True)
    b = b0.detach().clone().requires_grad_(True)
    out = AggregateWideAMX.apply(x, w, b, rowptr, colidx, scale,
                                 None, threads)
    out.backward(grad)
    return out.detach(), x.grad.detach(), w.grad.detach(), b.grad.detach()


def main() -> None:
    torch.set_num_threads(1)
    n, k, d, threads = 1024, 1024, 1024, 32
    torch.manual_seed(20260817)
    x = torch.randn(n, k)
    w = torch.randn(k, d)
    b = torch.randn(d)
    scale = torch.rand(n) + 0.5
    grad = torch.randn(n, d)
    rowptr, colidx = _ring(n)
    old = _run(False, x, w, b, rowptr, colidx, scale, grad, threads)
    new = _run(True, x, w, b, rowptr, colidx, scale, grad, threads)
    errors = {key: _rel(new[i], old[i]) for i, key in enumerate(
        ("out", "dx", "dw", "db"))}
    assert errors["out"] <= 5.0e-3
    assert errors["dx"] <= 3.5e-2
    assert errors["dw"] <= 3.0e-2
    assert errors["db"] <= 2.0e-3
    result = {"shape": {"N": n, "K": k, "D": d, "threads": threads},
              "relative_error": errors, "status": "pass"}
    output = os.environ.get("HYBRID_HIGHD_SMOKE_OUTPUT")
    if output:
        Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
