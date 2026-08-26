"""Numerical A/B gate for tile-granularity High-D fused bias reduction."""

from __future__ import annotations

import json
import os

import torch

from tfs_train.dimension_dispatch import AggregateWideAMX


def ring_csr(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    rowptr = torch.arange(0, 3 * n + 1, 3, dtype=torch.long)
    rows = torch.arange(n, dtype=torch.long)
    colidx = torch.stack((rows, (rows - 1) % n, (rows + 1) % n), dim=1)
    return rowptr, colidx.reshape(-1).contiguous()


def run(fused: bool, x0, w0, b0, grad, rowptr, colidx, scale, threads):
    os.environ["TFS_HIGHD_FUSED_DB"] = "1" if fused else "0"
    x = x0.detach().clone().requires_grad_(True)
    weight = w0.detach().clone().requires_grad_(True)
    bias = b0.detach().clone().requires_grad_(True)
    out = AggregateWideAMX.apply(
        x, weight, bias, rowptr, colidx, scale, None, int(threads))
    out.backward(grad)
    return out.detach(), x.grad.detach(), weight.grad.detach(), bias.grad.detach()


def relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max() /
                 b.float().abs().max().clamp_min(1.0e-6))


def main() -> None:
    n, k, d, threads = 1024, 128, 2983, 32
    torch.set_num_threads(1)
    torch.manual_seed(20260826)
    x0 = torch.randn(n, k)
    w0 = torch.randn(k, d)
    b0 = torch.randn(d)
    grad = torch.randn(n, d)
    scale = torch.rand(n) + 0.5
    rowptr, colidx = ring_csr(n)
    baseline = run(False, x0, w0, b0, grad, rowptr, colidx, scale, threads)
    fused = run(True, x0, w0, b0, grad, rowptr, colidx, scale, threads)
    result = {
        "shape": {"N": n, "K": k, "D": d, "threads": threads},
        "relative_error": {
            "output": relative(fused[0], baseline[0]),
            "dx": relative(fused[1], baseline[1]),
            "dw": relative(fused[2], baseline[2]),
            "db": relative(fused[3], baseline[3]),
        },
    }
    print(json.dumps(result, indent=2))
    if max(result["relative_error"].values()) > 3.0e-5:
        raise AssertionError("tile-fused-db diverged from grad.sum baseline")


if __name__ == "__main__":
    main()
