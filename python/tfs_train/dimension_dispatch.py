"""Dimension-generic AMX dispatch helpers.

The native C3 tile contract is output-width limited (D<=128), while real
models can have a 1024-wide input feature matrix or thousands of output
classes.  This module keeps those shape decisions out of dataset-specific
training scripts:

* high-K, D<=128 layers use the stride-parametric ordinary C3 entry point;
* transform-first D>128 layers use the native wide-output wrapper;
* aggregate-first D>128 layers use the native wide-aggregate wrapper.

The wrappers are only selected by callers when AMX is enabled.  Existing
reference/fallback paths remain available when the AMX gates are off.
"""

from __future__ import annotations

import os
import time

import torch

from .native import backend
from .execution_plan import build_layer_plan
from .highd_backward import (highd_plan_from_execution_plan,
                             highd_stream_enabled,
                             native_aggregate_d_slab_backward,
                             streamed_aggregate_single_scan_backward,
                             streamed_aggregate_backward,
                             streamed_transform_backward)


def amx_small_output_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Return whether the ordinary C3 output-tiled contract is legal.

    K is intentionally unbounded here: the native sparse pull and backward
    kernels iterate over a padded K stride, so IGB's K=1024 -> D=128 first
    layer belongs to this contract rather than an environment-gated probe.
    """

    return (
        x.dim() == 2 and weight.dim() == 2 and x.shape[1] >= 1 and
        weight.shape[0] == x.shape[1] and weight.shape[1] <= 128
    )


def amx_training_enabled() -> bool:
    """Whether both forward and backward AMX contracts are requested."""

    return (
        os.environ.get("HYBRID_AMX_FORWARD") == "1" and
        os.environ.get("HYBRID_AMX_BACKWARD") == "1"
    )


def _backward_result(ctx, dx, dw, db):
    """Return gradients with the same arity as the optional-plan forward."""

    result = (dx, dw, db, None, None, None, None, None, None)
    return result if getattr(ctx, "input_arity", 9) == 9 else result[:-1]


def _assert_actual_variant(plan, actual_variant: str) -> None:
    """Bind a concrete autograd branch to the immutable planner decision."""

    plan.assert_dispatch(actual_variant)


class WideOutputAMX(torch.autograd.Function):
    """Transform-first AMX with native output-column tiling."""

    @staticmethod
    def forward(ctx, x, weight, bias, rowptr, colidx, scale, schedule, threads,
                plan=None):
        if os.environ.get("HYBRID_AMX_FORWARD") != "1":
            raise RuntimeError("wide output path requires AMX forward")
        had_plan = plan is not None
        if plan is None:
            # Direct smoke tests remain supported, but the formal model path
            # always supplies the immutable construction-time plan.
            plan = build_layer_plan(
                int(x.shape[0]), int(weight.shape[0]), int(weight.shape[1]),
                compute_dx=bool(x.requires_grad), layer=1, threads=int(threads))
        if (int(plan.n), int(plan.k), int(plan.d)) != (
                int(x.shape[0]), int(weight.shape[0]), int(weight.shape[1])):
            raise ValueError("wide output execution plan shape mismatch")
        if not plan.execution_variant.startswith("transform_highd_"):
            raise RuntimeError(
                "wide output received a non-transform execution variant")
        out, hs = backend().c3_forward_wide_amx_v3(
            x, weight, bias, rowptr, colidx, scale, int(threads)
        )
        ctx.save_for_backward(hs, weight, rowptr, colidx, scale)
        ctx.threads = int(threads)
        ctx.plan = plan
        ctx.input_arity = 9 if had_plan else 8
        ctx.amx = os.environ.get("HYBRID_AMX_BACKWARD") == "1"
        return out

    @staticmethod
    def backward(ctx, grad_output):
        hs, weight, rowptr, colidx, scale = ctx.saved_tensors
        if not ctx.amx:
            raise RuntimeError("wide output path requires AMX backward")
        compute_dx = bool(ctx.needs_input_grad[0])
        layer_plan = ctx.plan
        # The immutable planner, not a second autograd-time shape heuristic,
        # owns the production transform-High-D gate.
        transform_stream = layer_plan.execution_variant in {
            "transform_highd_stream", "transform_highd_native",
            "transform_highd_single_scan"
        }
        if transform_stream and int(weight.shape[1]) > 128:
            _assert_actual_variant(layer_plan, layer_plan.execution_variant)
            highd_plan = highd_plan_from_execution_plan(layer_plan)
            dx, dw = streamed_transform_backward(
                hs, weight, grad_output.contiguous(), scale,
                rowptr, colidx, ctx.threads, compute_dx,
                d_tile=highd_plan.d_tile, plan=highd_plan)
            db = grad_output.contiguous().sum(0)
            if not compute_dx:
                dx = None
            return _backward_result(ctx, dx, dw, db)
        if layer_plan.execution_variant != "transform_highd_legacy":
            raise RuntimeError(
                "unhandled authority transform execution variant: "
                f"{layer_plan.execution_variant}")
        _assert_actual_variant(layer_plan, "transform_highd_legacy")
        dx, dw, db, _ = backend().c3_backward_wide_amx_v3(
            grad_output.contiguous(), hs, weight, rowptr, colidx, scale,
            ctx.threads, compute_dx
        )
        if not compute_dx:
            dx = None
        return _backward_result(ctx, dx, dw, db)


class AggregateWideAMX(torch.autograd.Function):
    """Aggregate-first AMX path for D>128 output classes."""

    @staticmethod
    def forward(ctx, x, weight, bias, rowptr, colidx, scale, schedule, threads,
                plan=None):
        if os.environ.get("HYBRID_AMX_FORWARD") != "1":
            raise RuntimeError("wide aggregate path requires AMX forward")
        had_plan = plan is not None
        if plan is None:
            plan = build_layer_plan(
                int(x.shape[0]), int(weight.shape[0]), int(weight.shape[1]),
                compute_dx=bool(x.requires_grad), layer=1, threads=int(threads))
        if (int(plan.n), int(plan.k), int(plan.d)) != (
                int(x.shape[0]), int(weight.shape[0]), int(weight.shape[1])):
            raise ValueError("wide aggregate execution plan shape mismatch")
        if not plan.execution_variant.startswith("aggregate_highd_"):
            raise RuntimeError(
                "wide aggregate received a non-aggregate execution variant")
        out, _hs, pulled = backend().c3_forward_aggregate_wide_amx_v3(
            x, weight, bias, rowptr, colidx, scale, int(threads)
        )
        ctx.save_for_backward(pulled, weight, rowptr, colidx, scale)
        ctx.threads = int(threads)
        ctx.plan = plan
        ctx.input_arity = 9 if had_plan else 8
        ctx.amx = os.environ.get("HYBRID_AMX_BACKWARD") == "1"
        return out

    @staticmethod
    def backward(ctx, grad_output):
        pulled, weight, rowptr, colidx, scale = ctx.saved_tensors
        if not ctx.amx:
            raise RuntimeError("wide aggregate path requires AMX backward")
        grad = grad_output.contiguous()
        scale_f = scale.contiguous()
        compute_dx = bool(ctx.needs_input_grad[0])

        # High-D V1 keeps only one scaled gradient panel live at a time.  The
        # layer plan was already selected from shape/thread/budget in the
        # canonical planner; this adapter must not make a second independent
        # order/tile decision inside autograd.
        layer_plan = ctx.plan
        if (int(layer_plan.n), int(layer_plan.k), int(layer_plan.d)) != (
                int(pulled.shape[0]), int(weight.shape[0]), int(weight.shape[1])):
            raise ValueError("aggregate execution plan shape mismatch")
        highd_plan = highd_plan_from_execution_plan(layer_plan)
        variant = layer_plan.execution_variant
        if os.environ.get("TFS_HIGHD_LOG_PLAN", "0").strip().lower() not in {
                "", "0", "off", "false", "no"}:
            print(highd_plan.log_line(), flush=True)

        if variant == "aggregate_highd_full":
            _assert_actual_variant(layer_plan, "aggregate_highd_full")
            native_fn = getattr(
                backend(), "c3_backward_aggregate_highd_amx_v1", None)
            if native_fn is None:
                raise RuntimeError(
                    "planned aggregate_highd_full kernel is unavailable")
            dx, dw, db, _meta = native_fn(
                grad, pulled, weight, rowptr, colidx, scale_f,
                ctx.threads, compute_dx)
        elif variant == "aggregate_highd_dslab":
            _assert_actual_variant(layer_plan, "aggregate_highd_dslab")
            dx, dw, db = native_aggregate_d_slab_backward(
                pulled, weight, grad, scale_f, rowptr, colidx,
                ctx.threads, compute_dx, d_tile=layer_plan.d_tile,
                d_slabs=layer_plan.d_slabs)
        elif variant == "aggregate_highd_single_scan":
            _assert_actual_variant(layer_plan, "aggregate_highd_single_scan")
            dx, dw = streamed_aggregate_single_scan_backward(
                pulled, weight, grad, scale_f, rowptr, colidx,
                ctx.threads, compute_dx, plan=highd_plan)
            db = grad.sum(0)
        elif variant == "aggregate_highd_reference":
            _assert_actual_variant(layer_plan, "aggregate_highd_reference")
            dx, dw, _resolved = streamed_aggregate_backward(
                pulled, weight, grad, scale_f, rowptr, colidx,
                ctx.threads, compute_dx, plan=highd_plan)
            db = grad.sum(0)
        else:
            raise RuntimeError(
                f"unhandled authority aggregate execution variant: {variant}")

        if not compute_dx:
            dx = None
        return _backward_result(ctx, dx, dw, db)


__all__ = [
    "AggregateWideAMX",
    "WideOutputAMX",
    "amx_small_output_supported",
    "amx_training_enabled",
]
