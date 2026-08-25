#!/usr/bin/env python3
"""Regression gate for process-global worker-pool capacity."""

import os

# This was an old experimental switch.  The pool must remain capable of the
# authority maximum even if it is inherited from an archived launcher.
os.environ["TFS_GLUE_E8_RIGHTSIZE_POOL"] = "1"
os.environ["TFS_WORKER_CPUS"] = "0-3"

import torch
import tfs_train_v2_c0_ext as ext


def main() -> None:
    n = 64
    x = torch.randn(n, 8).contiguous()
    weight = torch.randn(8, 8).contiguous()
    bias = torch.zeros(8).contiguous()
    rowptr = torch.arange(n + 1, dtype=torch.long).contiguous()
    colidx = torch.arange(n, dtype=torch.long).contiguous()
    scale = torch.ones(n).contiguous()
    ext.c3_forward_amx_v2(x, weight, bias, rowptr, colidx, scale, 1, False)
    ext.c3_forward_amx_v2(x, weight, bias, rowptr, colidx, scale, 4, False)
    print("worker-pool capacity regression: PASS (1 -> 4)")


if __name__ == "__main__":
    main()
