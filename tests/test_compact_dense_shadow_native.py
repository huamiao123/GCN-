import torch
import pytest

from tfs_train.native import backend


def relative_l2(actual, expected):
    return (torch.linalg.vector_norm((actual - expected).double()) /
            torch.linalg.vector_norm(expected.double()).clamp_min(1e-30))


def test_compact_dw_layouts_match_framework(monkeypatch):
    torch.manual_seed(17)
    pulled = torch.randn(67, 128, dtype=torch.bfloat16)
    gs = torch.randn(67, 47, dtype=torch.bfloat16)
    reference = torch.matmul(pulled.transpose(0, 1).contiguous(), gs).float()
    ext = backend()
    for direct_tail in ("0", "1"):
        monkeypatch.setenv("TFS_COMPACT_DW_T4", direct_tail)
        actual = ext.c3_compact_dw_bf16_amx_shadow_v1(pulled, gs, 1)
        assert actual.shape == reference.shape
        assert actual.is_contiguous()
        assert relative_l2(actual, reference) < 0.01


def test_compact_dw_v2_selects_layout_without_environment(monkeypatch):
    torch.manual_seed(18)
    pulled = torch.randn(67, 128, dtype=torch.bfloat16)
    gs = torch.randn(67, 47, dtype=torch.bfloat16)
    reference = torch.matmul(pulled.transpose(0, 1).contiguous(), gs).float()
    monkeypatch.setenv("TFS_COMPACT_DW_T4", "0")
    actual = backend().c3_compact_dw_bf16_amx_shadow_v2(
        pulled, gs, 1, True)
    assert relative_l2(actual, reference) < 0.01


def test_compact_logits_is_contiguous_and_matches_framework():
    torch.manual_seed(19)
    pulled = torch.randn(67, 128, dtype=torch.bfloat16)
    weight = torch.randn(128, 47, dtype=torch.bfloat16)
    bias = torch.randn(47, dtype=torch.float32)
    scale = torch.rand(67, dtype=torch.float32)
    reference = torch.matmul(pulled, weight).float()
    reference.mul_(scale.unsqueeze(1)).add_(bias)
    actual = backend().c3_compact_logits_amx_shadow_v1(
        pulled, weight, bias, scale, 1)
    assert actual.shape == reference.shape
    assert actual.is_contiguous()
    assert relative_l2(actual, reference) < 0.01
    fp32_oracle = torch.matmul(pulled.float(), weight.float())
    fp32_oracle.mul_(scale.unsqueeze(1)).add_(bias)
    # The native path keeps the AMX FP32 accumulator through the epilogue;
    # framework BF16 matmul rounds once before converting to FP32.
    assert relative_l2(actual, fp32_oracle) < 1e-5
    assert relative_l2(actual, fp32_oracle) < relative_l2(reference,
                                                          fp32_oracle)


@pytest.mark.parametrize("m,k,d,threads", [
    (16, 32, 47, 1),
    (67, 100, 129, 4),
    (71, 128, 2983, 8),
])
def test_explicit_packed_logits_is_bitwise_equal_to_legacy(
        m, k, d, threads):
    torch.manual_seed(20 + m + k + d)
    pulled = torch.randn(m, k, dtype=torch.bfloat16)
    weight = torch.randn(k, d, dtype=torch.bfloat16)
    bias = torch.randn(d, dtype=torch.float32)
    scale = torch.rand(m, dtype=torch.float32)
    ext = backend()
    packed = ext.c3_pack_compact_logits_weight_amx_shadow_v1(weight)
    actual = ext.c3_compact_logits_packed_amx_shadow_v2(
        pulled, packed, bias, scale, d, threads)
    legacy = ext.c3_compact_logits_amx_shadow_v1(
        pulled, weight, bias, scale, threads)
    torch.testing.assert_close(actual, legacy, rtol=0, atol=0)
