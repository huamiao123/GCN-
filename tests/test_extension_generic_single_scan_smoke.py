"""AMX runtime gate for the shape-generic small-D pull path.

This is intentionally a smoke test rather than a performance claim.  It
compares the formal single-scan gate with the conservative multi-scan fallback
for the non-special widths that previously had no dedicated implementation.
"""

import os
import sys

import torch


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csrc"))
    import tfs_train_v2_c0_ext as ext

    torch.manual_seed(20260816)
    n, k = 96, 32
    rowptr = torch.arange(0, n * 3 + 1, 3, dtype=torch.long).contiguous()
    colidx = torch.stack(
        (torch.arange(n), torch.arange(n).roll(1), torch.arange(n).roll(7)),
        dim=1,
    ).reshape(-1).contiguous()
    scale = torch.rand(n, dtype=torch.float32).contiguous()
    hs = torch.randn(n, k, dtype=torch.bfloat16).contiguous()

    os.environ["TFS_GLUE_E1_FUSED_DB"] = "1"
    os.environ["TFS_GLUE_E11_VEC_GRAD_DB"] = "0"
    os.environ["TFS_GLUE_E12_D47_SINGLE_SCAN"] = "0"
    os.environ["TFS_GLUE_E13_ACTIVE_ROW"] = "0"
    os.environ["TFS_ACTIVE_ROW"] = "off"
    for d in (1, 15, 16, 17, 19, 31, 32, 33, 40, 47, 63, 64, 65, 80, 96, 100, 117, 127):
        grad = torch.randn(n, d, dtype=torch.float32).contiguous()
        weight = torch.randn(k, d, dtype=torch.float32).contiguous()
        os.environ["TFS_SMALL_SINGLE_SCAN"] = "off"
        fallback = ext.c3_backward_amx_v2(
            grad, hs, weight, rowptr, colidx, scale, 2, False
        )
        os.environ["TFS_SMALL_SINGLE_SCAN"] = "on"
        single = ext.c3_backward_amx_v2(
            grad, hs, weight, rowptr, colidx, scale, 2, False
        )
        for lhs, rhs, name in zip(fallback[:3], single[:3], ("dx", "dw", "db")):
            assert torch.equal(lhs, rhs), (d, name)
    print("extension generic small-D single-scan smoke: PASS")


if __name__ == "__main__":
    main()

