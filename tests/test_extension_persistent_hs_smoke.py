import os
import sys
import types

import torch


def main():
    # The extension is built in csrc by the isolated remote build step.
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csrc"))
    import tfs_train_v2_c0_ext as ext
    from tfs_train.persistent_hs_cache import PersistentHsCache

    torch.manual_seed(7)
    n, k, d = 64, 32, 16
    x = torch.randn(n, k, dtype=torch.float32).contiguous()
    weight = torch.randn(k, d, dtype=torch.float32).contiguous()
    bias = torch.randn(d, dtype=torch.float32).contiguous()
    rowptr = torch.arange(n + 1, dtype=torch.long).contiguous()
    colidx = torch.arange(n, dtype=torch.long).contiguous()
    scale = torch.rand(n, dtype=torch.float32).contiguous()
    threads = 2

    graph = types.SimpleNamespace(rowptr=rowptr, colidx=colidx, scale=scale)
    cache = PersistentHsCache()
    hs_cached0, hit0 = cache.get_or_build(x, graph, threads, ext)
    hs_cached1, hit1 = cache.get_or_build(x, graph, threads, ext)
    assert not hit0 and hit1
    assert hs_cached0 is hs_cached1
    assert cache.misses == 1 and cache.hits == 1

    os.environ.setdefault("TFS_GLUE_E5_VEC_HS", "1")
    for transform_first in (False, True):
        out0, hs0 = ext.c3_forward_amx_v2(
            x, weight, bias, rowptr, colidx, scale, threads,
            transform_first)
        hs_cache = hs_cached0
        out1, hs1 = ext.c3_forward_cached_hs_amx_v1(
            x, hs_cache, weight, bias, rowptr, colidx, scale, threads,
            transform_first)
        assert torch.equal(hs0, hs_cache), (transform_first, "Hs mismatch")
        assert torch.equal(hs0, hs1), (transform_first, "returned Hs mismatch")
        assert torch.equal(out0, out1), (transform_first, "output mismatch")

    grad = torch.randn(n, d).contiguous()
    dx0, dw0, db0, _ = ext.c3_backward_amx_v2(
        grad, hs_cached0, weight, rowptr, colidx, scale, threads, False)
    dx1, dw1, db1, _ = ext.c3_backward_amx_v2(
        grad, hs_cached0, weight, rowptr, colidx, scale, threads, True)
    assert dx0.numel() == 0
    assert dx1.shape == x.shape
    assert torch.equal(dw0, dw1) and torch.equal(db0, db1)

    # V2 padded-Hs contract: logical K stays 32 while the retained storage
    # uses a 64-element AMX-safe row stride.  Forward and backward must remain
    # bitwise equal to the canonical V1 path.
    hs_padded = ext.c3_prepare_static_hs_padded_v2(
        x, scale, 64, threads)
    assert hs_padded.shape == (n, 64) and hs_padded.is_contiguous()
    assert torch.equal(hs_padded[:, :k], hs_cached0)
    assert torch.equal(hs_padded[:, k:], torch.zeros_like(hs_padded[:, k:]))
    for transform_first in (False, True):
        out_base, _ = ext.c3_forward_amx_v2(
            x, weight, bias, rowptr, colidx, scale, threads,
            transform_first)
        out_pad, hs_pad = ext.c3_forward_cached_hs_padded_amx_v2(
            x, hs_padded, weight, bias, rowptr, colidx, scale, threads,
            transform_first)
        assert torch.equal(out_base, out_pad), (transform_first,
                                                "padded output mismatch")
        assert torch.equal(hs_pad, hs_padded)
    dxp0, dwp0, dbp0, _ = ext.c3_backward_amx_v2(
        grad, hs_padded, weight, rowptr, colidx, scale, threads, False)
    dxp1, dwp1, dbp1, _ = ext.c3_backward_amx_v2(
        grad, hs_padded, weight, rowptr, colidx, scale, threads, True)
    assert dxp0.numel() == 0 and dxp1.shape == x.shape
    assert torch.equal(dwp0, dwp1) and torch.equal(dbp0, dbp1)
    assert torch.equal(dwp0, dw0) and torch.equal(dbp0, db0)

    # Exercise both native wide-output consumers as well.  These APIs are
    # not used by the current datasets' static first layers, but accepting a
    # retained Hs here makes the producer/consumer contract dimension-generic
    # instead of silently limiting it to D<=128.
    wide_d = 257
    wide_weight = torch.randn(k, wide_d, dtype=torch.float32).contiguous()
    wide_bias = torch.randn(wide_d, dtype=torch.float32).contiguous()
    out0, hs0 = ext.c3_forward_wide_amx_v3(
        x, wide_weight, wide_bias, rowptr, colidx, scale, threads)
    out1, hs1 = ext.c3_forward_wide_cached_hs_amx_v1(
        x, hs_cached0, wide_weight, wide_bias, rowptr, colidx, scale,
        threads)
    assert torch.equal(hs0, hs_cached0) and torch.equal(hs1, hs_cached0)
    assert torch.equal(out0, out1), "wide transform-first output mismatch"

    out0, hs0, pulled0 = ext.c3_forward_aggregate_wide_amx_v3(
        x, wide_weight, wide_bias, rowptr, colidx, scale, threads)
    out1, hs1, pulled1 = ext.c3_forward_aggregate_wide_cached_hs_amx_v1(
        x, hs_cached0, wide_weight, wide_bias, rowptr, colidx, scale,
        threads)
    assert torch.equal(hs0, hs_cached0) and torch.equal(hs1, hs_cached0)
    assert torch.equal(pulled0, pulled1)
    assert torch.equal(out0, out1), "wide aggregate-first output mismatch"
    print("extension persistent Hs smoke: PASS")


if __name__ == "__main__":
    main()
