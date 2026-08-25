"""Repeated old-vs-streamed Transform-HighD timing gate."""

from __future__ import annotations

import os
import statistics

import torch

from test_transform_highd_stream import _ring, _run


def main() -> None:
    threads = int(os.environ.get("TFS_HIGHD_THREADS", "32"))
    n = int(os.environ.get("TFS_HIGHD_N", "4103"))
    k = int(os.environ.get("TFS_HIGHD_K", "1024"))
    d = int(os.environ.get("TFS_HIGHD_D", "257"))
    repeats = int(os.environ.get("TFS_HIGHD_REPEATS", "5"))
    torch.set_num_threads(1)
    torch.manual_seed(20260818)
    x = torch.randn(n, k)
    w = torch.randn(k, d)
    b = torch.randn(d)
    scale = torch.rand(n) + 0.5
    grad = torch.randn(n, d)
    rowptr, colidx = _ring(n)
    _run(False, x, w, b, rowptr, colidx, scale, grad, threads)
    _run(True, x, w, b, rowptr, colidx, scale, grad, threads)
    old_rows = []
    new_rows = []
    max_errors = {key: 0.0 for key in ("out", "dx", "dw", "db")}
    for _ in range(repeats):
        old = _run(False, x, w, b, rowptr, colidx, scale, grad, threads)
        new = _run(True, x, w, b, rowptr, colidx, scale, grad, threads)
        old_rows.append((old["forward_ms"], old["backward_ms"]))
        new_rows.append((new["forward_ms"], new["backward_ms"]))
        for key in max_errors:
            denom = new[key].float().abs().max().item() + 1.0e-6
            err = (new[key].float() - old[key].float()).abs().max().item() / denom
            max_errors[key] = max(max_errors[key], err)
    old_f = statistics.median(row[0] for row in old_rows)
    old_b = statistics.median(row[1] for row in old_rows)
    new_f = statistics.median(row[0] for row in new_rows)
    new_b = statistics.median(row[1] for row in new_rows)
    print({
        "shape": {"N": n, "K": k, "D": d, "threads": threads},
        "old_median_ms": {"forward": old_f, "backward": old_b},
        "new_median_ms": {"forward": new_f, "backward": new_b},
        "speedup": {"forward": old_f / new_f, "backward": old_b / new_b},
        "max_relative_max_abs": max_errors,
        "old_rows": old_rows,
        "new_rows": new_rows,
    })


if __name__ == "__main__":
    main()
