"""Standalone server gate for the direct-BF16 sparse pull primitive."""

from __future__ import annotations

import torch

import tfs_train_v2_c0_ext as ext


def ring(n: int):
    rows = torch.arange(n, dtype=torch.long)
    rowptr = torch.arange(0, 3 * n + 1, 3, dtype=torch.long)
    colidx = torch.stack((rows, (rows - 1) % n, (rows + 1) % n), dim=1)
    return rowptr, colidx.reshape(-1).contiguous()


def main() -> None:
    for n, k in ((103, 19), (127, 128), (131, 257)):
        torch.manual_seed(20260818 + n + k)
        rowptr, colidx = ring(n)
        source = torch.randn(n, k).to(torch.bfloat16).contiguous()
        old = ext.c3_pull_only_amx_v1(
            source.float().contiguous(), rowptr, colidx, 4)
        new = ext.c3_pull_only_bf16_amx_v1(source, rowptr, colidx, 4)
        if new.dtype != torch.bfloat16 or new.shape != (n, k):
            raise AssertionError((n, k, new.dtype, tuple(new.shape)))
        if not torch.equal(new.float(), old):
            err = (new.float() - old).abs().max().item()
            raise AssertionError(f"BF16 pull mismatch n={n} k={k} max={err}")
    print("direct BF16 pull gate: PASS")


if __name__ == "__main__":
    main()
