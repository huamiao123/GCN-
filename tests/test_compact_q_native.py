"""Numerical and tail contracts for the native compact terminal Q GEMM."""

import importlib.util

import pytest
import torch

from tfs_train.native import backend


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("tfs_train_v2_c0_ext") is None,
    reason="compiled AMX extension is unavailable",
)


@pytest.mark.parametrize("shape", [(31, 47, 65), (33, 129, 128),
                                    (17, 2983, 128)])
@pytest.mark.parametrize("threads", [1, 4])
def test_compact_q_matches_bf16_matmul(shape, threads):
    torch.manual_seed(20260911)
    rows, d, k = shape
    gs = torch.randn(rows, d).to(torch.bfloat16)
    weight = torch.randn(k, d).to(torch.bfloat16)
    packed = backend().c3_pack_compact_q_weight_amx_shadow_v1(weight)
    actual = backend().c3_compact_q_packed_bf16_amx_shadow_v1(
        gs, packed, k, threads)
    reference = torch.matmul(gs, weight.transpose(0, 1).contiguous())
    assert actual.shape == (rows, k)
    assert actual.dtype == torch.bfloat16
    relative_l2 = (torch.linalg.vector_norm(actual.float() - reference.float()) /
                   torch.linalg.vector_norm(reference.float()).clamp_min(1e-12))
    assert float(relative_l2) <= 1.0e-2


def test_compact_q_rejects_wrong_packed_weight():
    gs = torch.randn(16, 129).to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="size mismatch"):
        backend().c3_compact_q_packed_bf16_amx_shadow_v1(
            gs, torch.empty(512, dtype=torch.bfloat16), 128, 1)
