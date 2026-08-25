"""Production-dispatch gate for real ``K>D>128`` transform shapes.

The inexpensive planner checks run in the normal test suite.  The native
numerical comparison is opt-in because it requires the AMX extension and a
Linux CPU with the release build; set ``TFS_RUN_NATIVE_TRANSFORM_GATE=1`` on
the server to execute it.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest
import torch

from tfs_train.execution_plan import build_layer_plan


@pytest.mark.parametrize("n,k,d", [
    (4097, 1024, 256),
    (4103, 1024, 257),
    (8191, 2048, 513),
    (4099, 300, 129),
])
def test_production_dispatch_plan_is_transform_highd(monkeypatch, n, k, d):
    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_STREAM_V1", "on")
    # This test validates the established Python stream contract.  The new
    # native candidate has its own explicit/cost-gated test below.
    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_NATIVE", "off")
    plan = build_layer_plan(n, k, d, threads=4, compute_dx=True)
    assert plan.order == "transform"
    assert plan.dimension_path == "wide_output"
    # ``execution_variant`` is the canonical, dispatcher-facing name.  The
    # older ``transform_stream`` label predates the unified High-D namespace.
    assert plan.execution_variant == "transform_highd_stream"
    assert plan.selected_impl == "transform_highd_stream"
    plan.validate()


def _ring(n: int):
    rows = torch.arange(n, dtype=torch.long)
    return (torch.arange(0, 3 * n + 1, 3, dtype=torch.long),
            torch.stack((rows, (rows - 1) % n, (rows + 1) % n), dim=1)
            .reshape(-1).contiguous())


@pytest.mark.skipif(
    os.environ.get("TFS_RUN_NATIVE_TRANSFORM_GATE") != "1",
    reason="native transform gate is an explicit server-side test",
)
def test_native_production_transform_highd_numerical_gate(monkeypatch):
    if importlib.util.find_spec("tfs_train_v2_c0_ext") is None:
        pytest.skip("compiled TFS extension is unavailable")
    from tfs_train.dimension_dispatch import WideOutputAMX

    monkeypatch.setenv("HYBRID_AMX_FORWARD", "1")
    monkeypatch.setenv("HYBRID_AMX_BACKWARD", "1")
    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_STREAM_V1", "on")
    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_NATIVE", "off")
    torch.manual_seed(20260818)
    n, k, d, threads = 4097, 1024, 257, 4
    rowptr, colidx = _ring(n)
    x0 = torch.randn(n, k)
    w0 = torch.randn(k, d)
    b0 = torch.randn(d)
    scale = torch.rand(n) + 0.5
    grad = torch.randn(n, d)

    def run(stream: bool):
        monkeypatch.setenv("TFS_TRANSFORM_HIGHD_STREAM_V1",
                           "on" if stream else "off")
        plan = build_layer_plan(n, k, d, threads=threads, compute_dx=True)
        assert plan.order == "transform"
        x = x0.clone().requires_grad_(True)
        w = w0.clone().requires_grad_(True)
        b = b0.clone().requires_grad_(True)
        out = WideOutputAMX.apply(
            x, w, b, rowptr, colidx, scale, None, threads, plan)
        out.backward(grad)
        return out.detach(), x.grad.detach(), w.grad.detach(), b.grad.detach()

    old = run(False)
    new = run(True)
    for actual, reference, tol in zip(new, old, (5e-3, 3.5e-2, 3e-2, 2e-3)):
        rel = (actual.float() - reference.float()).norm() / (
            reference.float().norm() + 1e-6)
        assert float(rel) <= tol
