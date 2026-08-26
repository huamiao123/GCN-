"""Correctness smoke for the opt-in V3 static aggregate cache."""

import os

import torch

from tfs_train.graph import preprocess_undirected_fast
from tfs_train.native import backend


def main():
    # This gate compares two parallel FP32 reductions.  Make the fixture
    # reproducible before assessing its numerical tolerance.
    torch.manual_seed(20260826)
    os.environ.setdefault("TFS_GLUE_E2_VEC_STORE", "0")
    os.environ.setdefault("TFS_GLUE_E4_FUSED_EPILOGUE", "1")
    os.environ.setdefault("TFS_GLUE_E7_FORWARD_SCHEDULE", "1")
    os.environ.setdefault("TFS_FWD_V2_SINGLE_SCAN", "1")
    torch.set_num_threads(8)
    n, k, d, threads = 257, 100, 47, 8
    src = torch.arange(n, dtype=torch.long)
    edge_index = torch.stack((src, (src * 17 + 23) % n))
    graph = preprocess_undirected_fast(edge_index, n, torch.float32)
    x = torch.randn(n, k, dtype=torch.float32)
    scale = graph.scale.contiguous()
    weight = torch.randn(k, d, dtype=torch.float32)
    bias = torch.randn(d, dtype=torch.float32)
    ones = torch.ones_like(scale)
    eye = torch.eye(k, dtype=torch.float32)
    zero = torch.zeros(k, dtype=torch.float32)
    ext = backend()

    dynamic_t0_as_float, dynamic_hs = ext.c3_forward_amx_v2(
        x, eye, zero, graph.rowptr, graph.colidx, ones, threads, False)
    t0_check = ext.c3_prepare_static_aggregate_v3(
        x, ones, graph.rowptr, graph.colidx, threads)
    hs_ones = ext.c3_prepare_static_hs_v1(x, ones, threads)
    cached_t0_as_float, _ = ext.c3_forward_cached_hs_amx_v1(
        x, hs_ones, eye, zero, graph.rowptr, graph.colidx, ones, threads, False)
    t0 = ext.c3_prepare_static_aggregate_v3(
        x, scale, graph.rowptr, graph.colidx, threads)
    hs = ext.c3_prepare_static_hs_v1(x, scale, threads)
    print("dynamic_hs_max", float((dynamic_hs - hs_ones).abs().max()))
    print("dynamic_vs_cached_t0_max", float(
        (dynamic_t0_as_float - cached_t0_as_float).abs().max()),
        "cached_vs_t0_check_max", float((cached_t0_as_float - t0_check.float()).abs().max()))
    if not torch.equal(dynamic_t0_as_float, t0_check.float()):
        diff = (dynamic_t0_as_float - t0_check.float()).abs()
        print("T0 mismatch", float(diff.max()), int((diff != 0).sum()))
        raise AssertionError("T0 mismatch")
    assert torch.equal(t0_check.float(), dynamic_t0_as_float)

    dynamic_out, _ = ext.c3_forward_amx_v2(
        x, weight, bias, graph.rowptr, graph.colidx, scale, threads, False)
    cached_out, cached_t0 = ext.c3_forward_cached_aggregate_amx_v3(
        x, t0, weight, bias, graph.rowptr, graph.colidx, scale, threads)
    assert torch.equal(cached_t0, t0)
    assert torch.equal(dynamic_out, cached_out), "cached aggregate forward mismatch"

    grad = torch.randn(n, d, dtype=torch.float32)
    _, dynamic_dw, dynamic_db, _ = ext.c3_backward_amx_v2(
        grad, hs, weight,
        graph.rowptr, graph.colidx, scale, threads, False)
    cached_dw, cached_db, _ = ext.c3_backward_cached_aggregate_amx_v3(
        grad, t0, graph.rowptr, graph.colidx, scale, threads)
    db_diff = (cached_db - dynamic_db).abs()
    print("db max", float(db_diff.max()), "relative",
          float(db_diff.max() / dynamic_db.abs().clamp_min(1e-6).max()))
    # The two paths reduce the same FP32 values with different parallel
    # ownership/order.  Keep this tight enough to catch real discrepancies
    # while allowing the bounded associativity error of an 8-worker sum.
    assert torch.allclose(cached_db, dynamic_db, rtol=1e-6, atol=3e-5)
    dw_diff = (cached_dw - dynamic_dw).abs()
    print("dW max", float(dw_diff.max()), "rel",
          float(dw_diff.max() / dynamic_dw.abs().clamp_min(1e-6).max()))
    assert torch.allclose(cached_dw, dynamic_dw, rtol=5e-2, atol=2e-1), (
        "cached aggregate dW outside the BF16 tolerance")
    print("extension aggregate cache smoke: PASS")


if __name__ == "__main__":
    main()
