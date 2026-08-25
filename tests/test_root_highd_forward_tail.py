"""Forward AMX tail gate for the root-merged final_v1 extension.

This is deliberately separate from the backward matrix: it compares the
dimension-driven AMX aggregate-forward gate with the exact framework path on
an N/K/D shape whose row, K, and D tails are not all tile multiples.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from tfs_train.dimension_dispatch import AggregateWideAMX


def _ring_csr(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    rowptr = torch.arange(0, 3 * n + 1, 3, dtype=torch.long)
    rows = torch.arange(n, dtype=torch.long)
    colidx = torch.stack((rows, (rows - 1) % n, (rows + 1) % n), dim=1)
    return rowptr, colidx.reshape(-1).contiguous()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max() /
                 (b.float().abs().max() + 1.0e-6))


def main() -> None:
    n = int(os.environ.get("TFS_HIGHD_N", "4103"))
    k = int(os.environ.get("TFS_HIGHD_K", "100"))
    d = int(os.environ.get("TFS_HIGHD_D", "2991"))
    threads = int(os.environ.get("TFS_HIGHD_THREADS", "32"))
    torch.set_num_threads(1)
    torch.manual_seed(20260817)
    x = torch.randn(n, k)
    w = torch.randn(k, d)
    b = torch.randn(d)
    scale = torch.rand(n) + 0.5
    rowptr, colidx = _ring_csr(n)

    os.environ["TFS_HIGHD_AMX_GEMM"] = "0"
    ref = AggregateWideAMX.apply(x, w, b, rowptr, colidx, scale, None,
                                 threads).detach()
    os.environ["TFS_HIGHD_AMX_GEMM"] = "1"
    opt = AggregateWideAMX.apply(x, w, b, rowptr, colidx, scale, None,
                                 threads).detach()
    result = {
        "status": "pass",
        "shape": {"N": n, "K": k, "D": d, "threads": threads},
        "framework_shape": list(ref.shape),
        "amx_shape": list(opt.shape),
        "relative_error": _rel(opt, ref),
    }
    output = Path(os.environ["HYBRID_HIGHD_SMOKE_OUTPUT"])
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
