#!/usr/bin/env python3
"""Correctness gate for the opt-in static-Hs NUMA replica path."""

import os

import torch

from tfs_train.native import backend


def _fixture(n=4096, k=32, d=64):
    src = torch.arange(n, dtype=torch.long)
    # Keep the source pattern deterministic while giving every destination a
    # non-trivial CSR row.  The graph is intentionally small enough for a
    # correctness gate; the placement experiment uses a larger fixture.
    col = torch.stack((src, (src * 17 + 13) % n,
                       (src * 29 + 7) % n), dim=1).reshape(-1)
    rowptr = torch.arange(0, 3 * (n + 1), 3, dtype=torch.long)
    rowptr[-1] = col.numel()
    x = torch.randn(n, k, dtype=torch.float32)
    scale = torch.rand(n, dtype=torch.float32) + 0.5
    weight = torch.randn(k, d, dtype=torch.float32)
    bias = torch.randn(d, dtype=torch.float32)
    return x, weight, bias, rowptr, col, scale


def main():
    torch.set_num_threads(1)
    os.environ.setdefault("TFS_LOCALITY_SCHEDULE", "off")
    os.environ.setdefault("TFS_INTERNAL_PROFILE", "1")
    ext = backend()
    x, weight, bias, rowptr, colidx, scale = _fixture()
    threads = 4

    hs = ext.c3_prepare_static_hs_v1(x, scale, threads)
    replicas = ext.c3_replicate_static_hs_numa_v1(hs, threads)
    assert replicas.dim() == 3 and replicas.shape[1:] == hs.shape
    assert replicas.is_contiguous()
    for replica in replicas:
        assert torch.equal(replica, hs), "replica changed BF16 Hs values"

    out_ref, hs_ref = ext.c3_forward_cached_hs_amx_v1(
        x, hs, weight, bias, rowptr, colidx, scale, threads, False)
    out_rep, hs_rep = ext.c3_forward_cached_hs_amx_v1(
        x, replicas, weight, bias, rowptr, colidx, scale, threads, False)
    assert torch.equal(out_ref, out_rep), "replicated aggregate forward mismatch"
    assert torch.equal(hs_ref, hs_rep), "returned Hs view mismatch"

    out_ref_t, _ = ext.c3_forward_cached_hs_amx_v1(
        x, hs, weight, bias, rowptr, colidx, scale, threads, True)
    out_rep_t, _ = ext.c3_forward_cached_hs_amx_v1(
        x, replicas, weight, bias, rowptr, colidx, scale, threads, True)
    assert torch.equal(out_ref_t, out_rep_t), "replicated transform forward mismatch"
    print("extension Hs NUMA-replica smoke: PASS",
          f"replicas={replicas.shape[0]}",
          f"bytes={replicas.numel() * replicas.element_size()}")


if __name__ == "__main__":
    main()
