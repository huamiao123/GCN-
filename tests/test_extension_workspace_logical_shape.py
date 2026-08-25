#!/usr/bin/env python3
import os
import sys
import torch

ROOT = os.environ.get(
    "TFS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "build", "extension"))
import tfs_train_v2_c0_ext as ext


def run_sequence(rowptr, colidx, scale, n=257, threads=2):
    results = []
    for k, d in ((100, 110), (110, 97), (97, 120), (120, 47)):
        torch.manual_seed(k * 1000 + d)
        x = torch.randn(n, k, dtype=torch.float32).contiguous()
        w = torch.randn(k, d, dtype=torch.float32).contiguous()
        grad = torch.randn(n, d, dtype=torch.float32).contiguous()
        hs = ext.c3_prepare_static_hs_v1(x, scale, threads)
        result = ext.c3_backward_amx_v2(
            grad, hs, w, rowptr, colidx, scale, threads, True)
        assert len(result) >= 3
        assert result[0].shape == x.shape
        assert result[1].shape == w.shape
        assert result[2].shape == (d,)
        results.append(tuple(value.clone() for value in result[:3]))
    return results


def main():
    torch.set_num_threads(2)
    n = 257
    rowptr = torch.arange(n + 1, dtype=torch.long).contiguous()
    colidx = torch.arange(n, dtype=torch.long).contiguous()
    scale = torch.linspace(0.2, 0.8, n, dtype=torch.float32).contiguous()
    os.environ["TFS_WORKSPACE_CACHE_MAX_BYTES"] = "1"
    fresh = run_sequence(rowptr, colidx, scale)
    os.environ["TFS_WORKSPACE_CACHE_MAX_BYTES"] = str(512 << 20)
    cached = run_sequence(rowptr, colidx, scale)
    for index, (lhs, rhs) in enumerate(zip(fresh, cached)):
        for name, a, b in zip(("dX", "dW", "db"), lhs, rhs):
            assert torch.equal(a, b), (index, name)
    print("workspace logical-shape regression: PASS")


if __name__ == "__main__":
    main()
