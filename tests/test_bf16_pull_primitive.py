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


def test_int32_workspace_is_invalidated_after_colidx_mutation(monkeypatch):
    """A mutated CSR must not reuse an int32 conversion from an old version."""
    import tfs_train_v2_c0_ext as ext

    n, k = 67, 19
    torch.manual_seed(20260911)
    rowptr, colidx = _ring(n)
    source = torch.randn(n, k).to(torch.bfloat16).contiguous()

    monkeypatch.setenv("TFS_COLIDX", "auto")
    before = ext.c3_pull_only_bf16_amx_v1(
        source, rowptr, colidx, 4)
    colidx.copy_((colidx + 7) % n)
    after_auto = ext.c3_pull_only_bf16_amx_v1(
        source, rowptr, colidx, 4)

    monkeypatch.setenv("TFS_COLIDX", "int64")
    after_int64 = ext.c3_pull_only_bf16_amx_v1(
        source, rowptr, colidx, 4)

    assert not torch.equal(before, after_auto)
    assert torch.equal(after_auto, after_int64)
