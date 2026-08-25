"""Autograd dispatch for the generic per-forward aggregate-saved path.

The native producer stores ``P = B * Q_BF16(S*X)`` for the current forward.
The backward reuses P for dW and performs exactly one pull for dX when the
input participates in autograd.  No static-graph assumption is made here.
"""

from __future__ import annotations

import torch

from .native import backend


class AggregateSavedFunction(torch.autograd.Function):
    """AMX aggregate-first forward with a per-step saved pulled matrix."""

    @staticmethod
    def forward(ctx, x, weight, bias, rowptr, colidx, scale, schedule, threads,
                plan=None):
        if plan is None:
            raise RuntimeError("authority aggregate-saved V4 requires a plan")
        plan.assert_dispatch("aggregate_saved_v4")
        out, _hs, pulled = backend().c3_forward_aggregate_saved_amx_v4(
            x, weight, bias, rowptr, colidx, scale, int(threads)
        )
        ctx.save_for_backward(pulled, weight, rowptr, colidx, scale)
        ctx.threads = int(threads)
        ctx.plan = plan
        ctx.input_arity = 9
        return out

    @staticmethod
    def backward(ctx, grad_output):
        pulled, weight, rowptr, colidx, scale = ctx.saved_tensors
        compute_dx = bool(ctx.needs_input_grad[0])
        dx, dw, db, _meta = backend().c3_backward_aggregate_saved_amx_v4(
            grad_output.contiguous(), pulled, weight, rowptr, colidx, scale,
            ctx.threads, compute_dx
        )
        if not compute_dx:
            dx = None
        # forward() has nine inputs; only x, weight, and bias are
        # differentiable.  schedule/threads are opaque dispatch arguments.
        result = (dx, dw, db, None, None, None, None, None, None)
        return result if ctx.input_arity == 9 else result[:-1]


__all__ = ["AggregateSavedFunction"]
