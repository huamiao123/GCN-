import pytest

from tfs_train.supervision_dense_plan import plan_supervision_dense


def test_measured_high_d_region_selects_both_native_kernels():
    plan = plan_supervision_dense(600_000, 128, 2983, 32)
    assert plan.row_tile == 300_000
    assert plan.fused_db
    assert plan.logsoftmax_out
    assert plan.native_dw
    assert plan.direct_tail_transpose
    assert plan.native_logits


def test_small_high_d_scope_keeps_framework_dw():
    plan = plan_supervision_dense(8192, 128, 2983, 32)
    assert plan.row_tile == 8192
    assert not plan.fused_db
    assert not plan.native_dw
    assert plan.native_logits


def test_low_work_avoids_oversubscribed_native_logits():
    plan = plan_supervision_dense(8192, 128, 512, 32)
    assert not plan.native_dw
    assert not plan.native_logits


def test_unmeasured_low_thread_logits_keeps_framework_path():
    plan = plan_supervision_dense(600_000, 128, 2983, 4)
    assert not plan.native_logits
    assert not plan.native_dw


def test_unmeasured_hidden_width_keeps_framework_paths():
    plan = plan_supervision_dense(600_000, 64, 2983, 32)
    assert not plan.native_logits
    assert not plan.native_dw


def test_aligned_output_does_not_request_tail_transpose():
    plan = plan_supervision_dense(600_000, 128, 2048, 8)
    assert plan.native_dw
    assert not plan.direct_tail_transpose
    assert not plan.native_logits


def test_plan_rejects_use_with_a_different_shape():
    plan = plan_supervision_dense(600_000, 128, 2983, 32)
    with pytest.raises(ValueError, match="shape mismatch"):
        plan.validate(600_000, 128, 2983, 16)


@pytest.mark.parametrize("arguments", [
    (0, 128, 2983, 32),
    (600_000, 0, 2983, 32),
    (600_000, 128, -1, 32),
    (600_000, 128, 2983, 64),
])
def test_invalid_shapes_are_rejected(arguments):
    with pytest.raises(ValueError):
        plan_supervision_dense(*arguments)
