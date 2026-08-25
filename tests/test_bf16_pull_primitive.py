"""Numerical gate for the direct-BF16 sparse pull primitive."""

from __future__ import annotations

import importlib.util

import pytest
import torch


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("tfs_train_v2_c0_ext") is None,
    reason="compiled AMX extension is unavailable",
)


def _ring(n: int):
    rows = torch.arange(n, dtype=torch.long)
    rowptr = torch.arange(0, 3 * n + 1, 3, dtype=torch.long)
    colidx = torch.stack((rows, (rows - 1) % n, (rows + 1) % n), dim=1)
    return rowptr, colidx.reshape(-1).contiguous()


@pytest.mark.parametrize("n,k", [(103, 19), (127, 128), (131, 257)])
def test_direct_bf16_pull_matches_legacy_bf16_roundtrip(n, k):
    import tfs_train_v2_c0_ext as ext

    torch.manual_seed(20260818 + n + k)
    rowptr, colidx = _ring(n)
    source = torch.randn(n, k).to(torch.bfloat16).contiguous()
    old = ext.c3_pull_only_amx_v1(
        source.float().contiguous(), rowptr, colidx, 4)
    new = ext.c3_pull_only_bf16_amx_v1(source, rowptr, colidx, 4)
    assert new.dtype == torch.bfloat16
    assert new.shape == (n, k)
    assert torch.equal(new.float(), old)
