import os
import sys

import torch


def main():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csrc"))
    import tfs_train_v2_c0_ext as ext

    torch.manual_seed(19)
    n, threads = 64, 2
    shapes = [(1, 1), (7, 31), (31, 7), (63, 127),
              (127, 63), (128, 128), (256, 64)]
    for k, d in shapes:
        x = torch.randn(n, k).contiguous()
        weight = torch.randn(k, d).contiguous()
        bias = torch.randn(d).contiguous()
        rowptr = torch.arange(n + 1, dtype=torch.long).contiguous()
        colidx = torch.arange(n, dtype=torch.long).contiguous()
        scale = torch.rand(n).contiguous()
        hs = ext.c3_prepare_static_hs_v1(x, scale, threads)
        assert hs.shape == x.shape and hs.dtype == torch.bfloat16
        for transform_first in (False, True):
            out0, hs0 = ext.c3_forward_amx_v2(
                x, weight, bias, rowptr, colidx, scale, threads,
                transform_first)
            out1, hs1 = ext.c3_forward_cached_hs_amx_v1(
                x, hs, weight, bias, rowptr, colidx, scale, threads,
                transform_first)
            assert torch.equal(hs0, hs)
            assert torch.equal(hs1, hs)
            assert torch.equal(out0, out1), (k, d, transform_first)

    try:
        ext.c3_prepare_static_hs_v1(
            torch.randn(n, 32, dtype=torch.bfloat16),
            torch.ones(n), threads)
    except RuntimeError:
        pass
    else:
        raise AssertionError("BF16 x must be rejected by the FP32 Hs contract")
    print("extension shape matrix: PASS", shapes)


if __name__ == "__main__":
    main()

