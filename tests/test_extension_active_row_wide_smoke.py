import os
import sys

import torch


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csrc"))
    import tfs_train_v2_c0_ext as ext

    torch.manual_seed(19)
    n, k, d = 96, 32, 128
    x = torch.randn(n, k, dtype=torch.float32).contiguous()
    weight = torch.randn(k, d, dtype=torch.float32).contiguous()
    rowptr = torch.arange(0, n * 3 + 1, 3, dtype=torch.long).contiguous()
    colidx = torch.stack(
        (torch.arange(n), torch.arange(n).roll(1), torch.arange(n).roll(7)),
        dim=1,
    ).reshape(-1).contiguous()
    scale = torch.rand(n, dtype=torch.float32).contiguous()
    grad = torch.zeros(n, d, dtype=torch.float32)
    active = torch.arange(0, n, 10)
    grad[active] = torch.randn(active.numel(), d)

    os.environ["TFS_GLUE_E1_FUSED_DB"] = "1"
    os.environ["TFS_GLUE_E14_ACTIVE_ROW_WIDE"] = "0"
    os.environ["TFS_ACTIVE_ROW"] = "off"
    base = ext.c3_backward_amx_v2(
        grad, x, weight, rowptr, colidx, scale, 2, True)
    os.environ["TFS_GLUE_E14_ACTIVE_ROW_WIDE"] = "1"
    os.environ["TFS_ACTIVE_ROW"] = "on"
    active_row = ext.c3_backward_amx_v2(
        grad, x, weight, rowptr, colidx, scale, 2, True)
    for lhs, rhs, name in zip(base[:3], active_row[:3], ("dx", "dw", "db")):
        assert torch.equal(lhs, rhs), name
    print("extension active-row-wide smoke: PASS")


if __name__ == "__main__":
    main()
