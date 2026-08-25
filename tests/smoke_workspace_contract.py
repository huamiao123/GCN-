#!/usr/bin/env python3
"""Exercise the three common-workspace allocation contracts on AMX."""

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build" / "extension"))
sys.path.insert(0, str(ROOT / "python"))

import tfs_train_v2_c0_ext as ext


os.environ["TFS_INTERNAL_PROFILE"] = "1"
os.environ["TFS_NUMA_FIRST_TOUCH"] = "on"
os.environ["TFS_WORKSPACE_CACHE_MAX_BYTES"] = str(64 << 20)
os.environ["TFS_GLUE_E1_FUSED_DB"] = "1"
os.environ["TFS_GLUE_E5_VEC_HS"] = "1"
os.environ["TFS_SMALL_SINGLE_SCAN"] = "auto"
os.environ["TFS_ACTIVE_ROW"] = "auto"

torch.manual_seed(101)
n, k, d, threads = 257, 100, 47, 2
x = torch.randn(n, k).contiguous()
weight = torch.randn(k, d).contiguous()
grad = torch.randn(n, d).contiguous()
scale = torch.rand(n).contiguous()
rowptr = torch.arange(n + 1, dtype=torch.int64).contiguous()
colidx = torch.arange(n, dtype=torch.int64).contiguous()
hs_bf16 = ext.c3_prepare_static_hs_v1(x, scale, threads)

# 1. Direct BF16 Hs and no dX: no hb, wt, or packed_wt.
no_dx = ext.c3_backward_amx_v2(
    grad, hs_bf16, weight, rowptr, colidx, scale, threads, False)
assert no_dx[0].numel() == 0

# 2. Same direct Hs but dX required: wt and packed_wt are present, hb is not.
with_dx = ext.c3_backward_amx_v2(
    grad, hs_bf16, weight, rowptr, colidx, scale, threads, True)
assert tuple(with_dx[0].shape) == (n, k)

# 3. FP32 Hs and no dX: hb is present, dX-only buffers remain absent.
fp32_no_dx = ext.c3_backward_amx_v2(
    grad, hs_bf16.float().contiguous(), weight,
    rowptr, colidx, scale, threads, False)
assert fp32_no_dx[0].numel() == 0
assert torch.equal(no_dx[1], fp32_no_dx[1])
assert torch.equal(no_dx[2], fp32_no_dx[2])
print("workspace dataflow smoke: PASS")
