#!/usr/bin/env python3
"""Compare cached NNZ and source-reuse panel schedules on one CSR fixture."""

import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "csrc"))
import tfs_train_v2_c0_ext as ext


def main():
    torch.set_num_threads(1)
    torch.manual_seed(20260816)
    n, k, d, threads, degree = 65536, 32, 64, 8, 8
    rows = torch.arange(n, dtype=torch.long)
    offsets = torch.arange(degree, dtype=torch.long)
    if os.environ.get("LOCALITY_CLUSTERED") == "1":
        # Eight destination panels share a 2K-row source working set.  This
        # is the regime where source ownership can reduce cross-worker cache
        # churn; the default remains a dispersed CSR fixture.
        cluster = (rows // 4096) * 4096
        local = rows % 2048
        colidx = cluster[:, None] + ((local[:, None] +
                                      offsets[None, :] * 17) % 2048)
    else:
        colidx = ((rows[:, None] * 131 + offsets[None, :] * 17 + 3) % n)
    colidx = colidx % n
    colidx = colidx.reshape(-1).contiguous()
    rowptr = torch.arange(0, degree * n + 1, degree, dtype=torch.long)
    scale = (torch.rand(n) + 0.25).contiguous()
    x = torch.randn(n, k).contiguous()
    weight = torch.randn(k, d).contiguous()
    bias = torch.randn(d).contiguous()
    os.environ["TFS_GLUE_E7_FORWARD_SCHEDULE"] = "1"
    results = {"off": [], "on": []}
    outputs = {}
    for mode in ("off", "on"):
        os.environ["TFS_LOCALITY_SCHEDULE"] = mode
        outputs[mode], _ = ext.c3_forward_amx_v2(
            x, weight, bias, rowptr, colidx, scale, threads, False
        )
    # Alternate the modes so cache/thermal drift is not mistaken for a
    # scheduler effect.  The schedule itself is cached by E7 after warmup.
    for round_id in range(10):
        order = ("off", "on") if round_id % 2 == 0 else ("on", "off")
        for mode in order:
            os.environ["TFS_LOCALITY_SCHEDULE"] = mode
            t0 = time.perf_counter()
            outputs[mode], _ = ext.c3_forward_amx_v2(
                x, weight, bias, rowptr, colidx, scale, threads, False
            )
            results[mode].append((time.perf_counter() - t0) * 1000.0)
    torch.testing.assert_close(outputs["off"], outputs["on"], rtol=0, atol=0)
    off=statistics.median(results["off"])
    on=statistics.median(results["on"])
    print("locality_off_median_ms=%.3f locality_on_median_ms=%.3f speedup=%.3fx" %
          (off, on, off / on))


if __name__ == "__main__":
    main()
