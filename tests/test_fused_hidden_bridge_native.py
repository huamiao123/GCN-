"""Numerical contract for the fused inter-layer activation producer."""

import importlib.util

import pytest
import torch

from tfs_train.native import backend


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("tfs_train_v2_c0_ext") is None,
    reason="compiled AMX extension is unavailable",
)


@pytest.mark.parametrize("shape", [(33, 47), (65, 128), (17, 129)])
@pytest.mark.parametrize("threads", [1, 4])
def test_fused_bridge_matches_explicit_reference(shape, threads):
    torch.manual_seed(20260911)
    x = torch.randn(*shape)
    x[0, 0] = 0.0
    scale = (torch.rand(shape[0]) + 0.25).contiguous()
    mask = torch.randint(0, 2, shape, dtype=torch.uint8)
    dropout_scale = 2.0
    out, staged, state = backend().c3_fused_hidden_bridge_shadow_v1(
        x, scale, mask, dropout_scale, threads)

    active = (x > 0) & mask.bool()
    reference = torch.where(active, x * dropout_scale, 0.0)
    reference_staged = (reference * scale.unsqueeze(1)).to(torch.bfloat16)
    torch.testing.assert_close(out, reference, rtol=0, atol=0)
    torch.testing.assert_close(staged, reference_staged, rtol=0, atol=0)
    torch.testing.assert_close(state, active.to(torch.uint8), rtol=0, atol=0)

    grad = torch.randn(*shape)
    actual_grad = backend().c3_fused_hidden_bridge_backward_shadow_v1(
        grad, state, dropout_scale, threads)
    reference_grad = torch.where(active, grad * dropout_scale, 0.0)
    torch.testing.assert_close(actual_grad, reference_grad, rtol=0, atol=0)


def test_fused_bridge_rejects_non_byte_dropout_mask():
    x = torch.randn(8, 17)
    scale = torch.ones(8)
    with pytest.raises(RuntimeError, match="contract"):
        backend().c3_fused_hidden_bridge_shadow_v1(
            x, scale, torch.ones_like(x, dtype=torch.bool), 2.0, 1)
