import os
import sys

import torch


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csrc"))
    import tfs_train_v2_c0_ext as ext

    torch.manual_seed(17)
    n, k, d, threads = 257, 100, 128, 2
    x = torch.randn(n, k, dtype=torch.float32).contiguous()
    w = torch.randn(k, d, dtype=torch.float32).contiguous()
    b = torch.randn(d, dtype=torch.float32).contiguous()
    rowptr = torch.arange(n + 1, dtype=torch.long).contiguous()
    colidx = torch.arange(n, dtype=torch.long).contiguous()
    scale = torch.rand(n, dtype=torch.float32).contiguous()

    hs = ext.c3_prepare_static_hs_v1(x, scale, threads)
    hs_pad = ext.c3_prepare_static_hs_padded_v2(x, scale, 128, threads)
    assert torch.equal(hs_pad[:, :k], hs)
    assert torch.equal(hs_pad[:, k:], torch.zeros_like(hs_pad[:, k:]))

    out_base, hs_base = ext.c3_forward_amx_v2(
        x, w, b, rowptr, colidx, scale, threads, False)
    out_v1, hs_v1 = ext.c3_forward_cached_hs_amx_v1(
        x, hs, w, b, rowptr, colidx, scale, threads, False)
    out_v2, hs_v2 = ext.c3_forward_cached_hs_padded_amx_v2(
        x, hs_pad, w, b, rowptr, colidx, scale, threads, False)
    assert torch.equal(out_base, out_v1)
    assert torch.equal(out_base, out_v2)
    assert torch.equal(hs_base, hs_v1)
    assert torch.equal(hs_pad, hs_v2)

    grad = torch.randn(n, d, dtype=torch.float32).contiguous()
    base = ext.c3_backward_amx_v2(
        grad, hs, w, rowptr, colidx, scale, threads, True)
    padded = ext.c3_backward_amx_v2(
        grad, hs_pad, w, rowptr, colidx, scale, threads, True)
    assert torch.equal(base[0], padded[0])
    assert torch.equal(base[1], padded[1])
    assert torch.equal(base[2], padded[2])
    print("padded products tail smoke: PASS")


if __name__ == "__main__":
    main()
