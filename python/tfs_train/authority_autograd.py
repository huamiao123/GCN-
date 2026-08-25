"""Shared autograd primitives for the final-pre-NUMA authority models.

Dataset benchmark scripts own data loading and baseline frameworks.  This
module owns only the common TFS C3 forward/backward contracts so their
implementations cannot drift independently.
"""

from __future__ import annotations

import os

import torch

from .native import backend


class _C3Base(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_output):
        hs, weight, rowptr, colidx, scale = ctx.saved_tensors
        compute_dx = bool(ctx.needs_input_grad[0])
        if ctx.amx:
            dx, dw, db, _ = backend().c3_backward_amx_v2(
                grad_output.contiguous(), hs, weight, rowptr, colidx, scale,
                ctx.threads, compute_dx)
        else:
            dx, dw, db, _, _, _ = backend().c3_backward_selective(
                grad_output.contiguous(), hs, weight, rowptr, colidx, scale,
                ctx.schedule, ctx.threads, False, 8, compute_dx)
        if not compute_dx:
            dx = None
        # Benchmark wrappers may omit the optional replica cache argument.
        # Autograd requires one gradient slot per argument actually supplied,
        # so keep this shared primitive arity-compatible with all callers.
        return (dx, dw, db) + (None,) * (len(ctx.needs_input_grad) - 3)


def _cached_c3_forward(x, cached_hs, weight, bias, rowptr, colidx, scale,
                       threads, transform_first, replicas=None):
    source = replicas if replicas is not None else cached_hs
    if cached_hs.shape[1] != x.shape[1]:
        return backend().c3_forward_cached_hs_padded_amx_v2(
            x, source, weight, bias, rowptr, colidx, scale,
            int(threads), transform_first)
    return backend().c3_forward_cached_hs_amx_v1(
        x, source, weight, bias, rowptr, colidx, scale,
        int(threads), transform_first)


class AggregateFirst(_C3Base):
    @staticmethod
    def forward(ctx, x, weight, bias, rowptr, colidx, scale, schedule, threads,
                cached_hs=None, backward_hs=None, cached_hs_replicas=None,
                plan=None):
        if plan is not None:
            plan.assert_dispatch("native_wide_k" if plan.dimension_path == "wide_k"
                                 else "native_c3")
        if os.environ.get("HYBRID_AMX_FORWARD") == "1":
            if cached_hs is None:
                out, hs = backend().c3_forward_amx_v2(
                    x, weight, bias, rowptr, colidx, scale, int(threads), False)
            else:
                out, hs = _cached_c3_forward(
                    x, cached_hs, weight, bias, rowptr, colidx, scale, threads,
                    False, cached_hs_replicas)
        else:
            out, hs, _ = backend().c3_forward(
                x, weight, bias, rowptr, colidx, scale, schedule, int(threads))
        ctx.save_for_backward(hs if backward_hs is None else backward_hs,
                              weight, rowptr, colidx, scale)
        ctx.threads = int(threads)
        ctx.amx = os.environ.get("HYBRID_AMX_BACKWARD") == "1"
        ctx.schedule = None if ctx.amx else schedule
        return out


class TransformFirst(_C3Base):
    @staticmethod
    def forward(ctx, x, weight, bias, rowptr, colidx, scale, schedule, threads,
                cached_hs=None, backward_hs=None, cached_hs_replicas=None,
                plan=None):
        if plan is not None:
            plan.assert_dispatch("native_wide_k" if plan.dimension_path == "wide_k"
                                 else "native_c3")
        if os.environ.get("HYBRID_AMX_FORWARD") == "1":
            if cached_hs is None:
                out, hs = backend().c3_forward_amx_v2(
                    x, weight, bias, rowptr, colidx, scale, int(threads), True)
            else:
                out, hs = _cached_c3_forward(
                    x, cached_hs, weight, bias, rowptr, colidx, scale, threads,
                    True, cached_hs_replicas)
        else:
            out, hs, _ = backend().c3_forward_transform(
                x, weight, bias, rowptr, colidx, scale, schedule, int(threads))
        ctx.save_for_backward(hs if backward_hs is None else backward_hs,
                              weight, rowptr, colidx, scale)
        ctx.threads = int(threads)
        ctx.amx = os.environ.get("HYBRID_AMX_BACKWARD") == "1"
        ctx.schedule = None if ctx.amx else schedule
        return out


class AggregateCachedT0(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, rowptr, colidx, scale, schedule,
                threads, cached_t0, plan=None):
        if plan is not None:
            plan.assert_dispatch("aggregate_static_v3")
        if os.environ.get("HYBRID_AMX_FORWARD") != "1":
            raise RuntimeError("static aggregate cache requires AMX forward")
        out, t0 = backend().c3_forward_cached_aggregate_amx_v3(
            x, cached_t0, weight, bias, rowptr, colidx, scale, int(threads))
        ctx.save_for_backward(t0, weight, rowptr, colidx, scale)
        ctx.threads = int(threads)
        ctx.amx = os.environ.get("HYBRID_AMX_BACKWARD") == "1"
        return out

    @staticmethod
    def backward(ctx, grad_output):
        t0, weight, rowptr, colidx, scale = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            raise RuntimeError("static aggregate cache cannot provide dX")
        if not ctx.amx:
            raise RuntimeError("static aggregate cache requires AMX backward")
        dw, db, _ = backend().c3_backward_cached_aggregate_amx_v3(
            grad_output.contiguous(), t0, rowptr, colidx, scale, ctx.threads)
        return (None, dw, db) + (None,) * (len(ctx.needs_input_grad) - 3)


__all__ = ["AggregateCachedT0", "AggregateFirst", "TransformFirst"]
