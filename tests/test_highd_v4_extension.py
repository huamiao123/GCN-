"""Small real-extension gate for the streamed aggregate-wide backward."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import torch

from tfs_train.dimension_dispatch import AggregateWideAMX


def _ring_csr(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    rowptr = torch.arange(0, 3 * n + 1, 3, dtype=torch.long)
    rows = torch.arange(n, dtype=torch.long)
    colidx = torch.stack((rows, (rows - 1) % n, (rows + 1) % n), dim=1)
    return rowptr, colidx.reshape(-1).contiguous()


def _one(stream: bool, x0, w0, b0, rowptr, colidx, scale, seed, threads):
    os.environ["TFS_HIGHD_STREAM_V1"] = "1" if stream else "0"
    torch.manual_seed(seed)
    x = x0.detach().clone().requires_grad_(True)
    weight = w0.detach().clone().requires_grad_(True)
    bias = b0.detach().clone().requires_grad_(True)
    torch.manual_seed(seed + 1)
    grad_seed = torch.randn_like(x.new_empty((x.shape[0], weight.shape[1])))
    t0 = time.perf_counter()
    out = AggregateWideAMX.apply(
        x, weight, bias, rowptr, colidx, scale, None, int(threads))
    fwd_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    out.backward(grad_seed)
    bwd_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "stream": stream,
        "fwd_ms": fwd_ms,
        "bwd_ms": bwd_ms,
        "out": out.detach(),
        "dx": x.grad.detach(),
        "dw": weight.grad.detach(),
        "db": bias.grad.detach(),
    }


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max() /
                 (b.float().abs().max() + 1.0e-6))


def main() -> None:
    # D=2983 is the IGB high-output contract.  N is intentionally small
    # enough for a quick gate, but uses the same K/D shape and real AMX code.
    n = int(os.environ.get("TFS_HIGHD_N", "4096"))
    k = int(os.environ.get("TFS_HIGHD_K", "128"))
    d = int(os.environ.get("TFS_HIGHD_D", "2983"))
    threads = int(os.environ.get("TFS_HIGHD_THREADS", "32"))
    torch.set_num_threads(1)
    torch.manual_seed(20260817)
    x0 = torch.randn(n, k)
    w0 = torch.randn(k, d)
    b0 = torch.randn(d)
    scale = torch.rand(n) + 0.5
    rowptr, colidx = _ring_csr(n)

    # One warmup per path, then two measured calls.  Use independent tensors
    # but identical values, so the native forward and backward are comparable.
    _one(False, x0, w0, b0, rowptr, colidx, scale, 100, threads)
    _one(True, x0, w0, b0, rowptr, colidx, scale, 100, threads)
    old = _one(False, x0, w0, b0, rowptr, colidx, scale, 200, threads)
    new = _one(True, x0, w0, b0, rowptr, colidx, scale, 200, threads)

    result = {
        "shape": {"N": n, "K": k, "D": d, "threads": threads},
        "old_ms": {"forward": old["fwd_ms"], "backward": old["bwd_ms"]},
        "new_ms": {"forward": new["fwd_ms"], "backward": new["bwd_ms"]},
        "speedup": {
            "forward": old["fwd_ms"] / new["fwd_ms"],
            "backward": old["bwd_ms"] / new["bwd_ms"],
        },
        "relative_error": {
            "forward": _rel(new["out"], old["out"]),
            "dx": _rel(new["dx"], old["dx"]),
            "dw": _rel(new["dw"], old["dw"]),
            "db": _rel(new["db"], old["db"]),
        },
    }
    output = Path(os.environ["HYBRID_HIGHD_SMOKE_OUTPUT"])
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
