"""Experimental supervision-scoped terminal GCN dataflow.

This module is intentionally absent from the authority planner.  It computes
only terminal rows selected by a static supervision mask and propagates their
gradient through an explicitly transposed rectangular CSR.  Hidden layers and
evaluation remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

import torch
import torch.nn.functional as F

from .native import backend


def _edge_balanced_cuts(rowptr: torch.Tensor, threads: int) -> torch.Tensor:
    """Return contiguous row cuts with approximately equal nonzero counts."""
    if rowptr.dtype != torch.int64 or rowptr.dim() != 1:
        raise TypeError("rowptr must be a one-dimensional int64 tensor")
    rows = rowptr.numel() - 1
    if rows < 0:
        raise ValueError("invalid rowptr")
    total = int(rowptr[-1])
    if total == 0:
        return torch.linspace(0, rows, threads + 1, dtype=torch.float64).long()
    targets = torch.arange(threads + 1, dtype=torch.float64)
    targets.mul_(float(total) / threads)
    cuts = torch.searchsorted(rowptr, targets.to(torch.int64), right=False)
    cuts.clamp_(0, rows)
    cuts[0] = 0
    cuts[-1] = rows
    return cuts.contiguous()


def _selected_cuts(row_ids: torch.Tensor, rowptr: torch.Tensor,
                   threads: int) -> torch.Tensor:
    # The native square pull has an implicit self contribution per row.
    degrees = rowptr[row_ids + 1] - rowptr[row_ids] + 1
    prefix = torch.empty(row_ids.numel() + 1, dtype=torch.int64)
    prefix[0] = 0
    torch.cumsum(degrees, dim=0, out=prefix[1:])
    total = int(prefix[-1])
    if total == 0:
        return torch.linspace(0, row_ids.numel(), threads + 1,
                              dtype=torch.float64).long()
    targets = (torch.arange(threads + 1, dtype=torch.float64) *
               (float(total) / threads)).to(torch.int64)
    cuts = torch.searchsorted(prefix, targets, right=False)
    cuts.clamp_(0, row_ids.numel())
    cuts[0] = 0
    cuts[-1] = row_ids.numel()
    return cuts.contiguous()


@dataclass(frozen=True)
class SupervisionScope:
    row_ids: torch.Tensor
    transpose_rowptr: torch.Tensor
    transpose_colidx: torch.Tensor
    selected_schedule: torch.Tensor
    transpose_schedule: torch.Tensor
    node_count: int
    selected_edge_count: int

    @property
    def selected_count(self) -> int:
        return int(self.row_ids.numel())

    @property
    def selected_ratio(self) -> float:
        return self.selected_count / self.node_count

    @classmethod
    def build(cls, train_mask: torch.Tensor, graph, threads: int):
        if train_mask.dtype != torch.bool or train_mask.dim() != 1:
            raise TypeError("train_mask must be a one-dimensional bool tensor")
        row_ids = train_mask.nonzero(as_tuple=False).flatten().to(torch.int64)
        row_ids = row_ids.contiguous()
        if row_ids.numel() == 0:
            raise ValueError("supervision scope must contain at least one row")
        transpose_rowptr, transpose_colidx = (
            backend().c3_build_selected_transpose_shadow_v1(
                row_ids, graph.rowptr, graph.colidx))
        selected_schedule = _selected_cuts(row_ids, graph.rowptr, threads)
        transpose_schedule = _edge_balanced_cuts(transpose_rowptr, threads)
        selected_edges = int(transpose_colidx.numel())
        return cls(row_ids, transpose_rowptr, transpose_colidx,
                   selected_schedule, transpose_schedule,
                   int(train_mask.numel()), selected_edges)


class _ScopedTerminalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, rowptr, colidx, scale,
                row_ids, transpose_rowptr, transpose_colidx,
                selected_schedule, transpose_schedule, threads):
        threads = int(threads)
        hs = backend().c3_prepare_static_hs_v1(hidden, scale, threads)
        pulled = backend().c3_selected_pull_bf16_shadow_v1(
            hs, row_ids, rowptr, colidx, selected_schedule, threads)
        weight_bf16 = weight.to(torch.bfloat16)
        aggregated = torch.matmul(pulled, weight_bf16)
        selected_scale = scale.index_select(0, row_ids).contiguous()
        logits = aggregated.float()
        logits.mul_(selected_scale.unsqueeze(1)).add_(bias)
        ctx.save_for_backward(pulled, hs, weight, selected_scale, scale,
                              row_ids, rowptr, colidx, transpose_rowptr,
                              transpose_colidx, transpose_schedule)
        ctx.threads = threads
        ctx.backward_order = os.environ.get(
            "TFS_SCOPE_BACKWARD_ORDER", "y_first").strip().lower()
        if ctx.backward_order not in {"y_first", "q_first", "authority_active"}:
            raise ValueError(
                "TFS_SCOPE_BACKWARD_ORDER must be y_first, q_first, or "
                "authority_active")
        return logits

    @staticmethod
    def backward(ctx, grad_logits):
        (pulled, hs, weight, selected_scale, scale, row_ids, rowptr, colidx,
         transpose_rowptr, transpose_colidx,
         transpose_schedule) = ctx.saved_tensors
        grad_logits = grad_logits.contiguous()
        if ctx.backward_order == "authority_active":
            grad_full = torch.zeros(
                (scale.numel(), grad_logits.shape[1]), dtype=grad_logits.dtype)
            grad_full.index_copy_(0, row_ids, grad_logits)
            dh, dw, db, _ = backend().c3_backward_amx_v2(
                grad_full, hs, weight, rowptr, colidx, scale,
                ctx.threads, True)
            return (dh, dw, db, None, None, None, None, None, None, None,
                    None, None)
        gs = backend().c3_scale_grad_bf16_v1(
            grad_logits, selected_scale, ctx.threads)
        dw = torch.matmul(pulled.transpose(0, 1).contiguous(), gs).float()
        weight_t = weight.to(torch.bfloat16).transpose(0, 1).contiguous()
        if ctx.backward_order == "q_first":
            q = torch.matmul(gs, weight_t)
            dh = backend().c3_rect_pull_bf16_scaled_fp32_shadow_v1(
                q.contiguous(), transpose_rowptr, transpose_colidx, scale,
                transpose_schedule, ctx.threads)
        else:
            y = backend().c3_rect_pull_bf16_shadow_v1(
                gs, transpose_rowptr, transpose_colidx,
                transpose_schedule, ctx.threads)
            dh = torch.matmul(y, weight_t).float()
            dh.mul_(scale.unsqueeze(1))
        db = grad_logits.sum(dim=0)
        return (dh, dw, db, None, None, None, None, None, None, None,
                None, None)


class _ScopedTerminalCrossEntropyFunction(torch.autograd.Function):
    """Bounded-panel terminal classifier, CE, and Q-first preparation.

    The forward owns the exact terminal-loss boundary.  It retains only one
    row panel of logits/gradients, immediately forms compact Q and parameter
    gradients, and saves those bounded results for autograd.  The sparse
    rectangular pull remains in backward so hidden-layer autograd ordering is
    unchanged.
    """

    @staticmethod
    def forward(ctx, hidden, weight, bias, rowptr, colidx, scale, row_ids,
                transpose_rowptr, transpose_colidx, selected_schedule,
                transpose_schedule, compact_labels, threads, row_tile):
        threads = int(threads)
        row_tile = int(row_tile)
        if row_tile <= 0:
            raise ValueError("row_tile must be positive")
        rows = int(row_ids.numel())
        if compact_labels.shape != (rows,):
            raise ValueError("compact labels do not match supervision rows")

        hs = backend().c3_prepare_static_hs_v1(hidden, scale, threads)
        pulled = backend().c3_selected_pull_bf16_shadow_v1(
            hs, row_ids, rowptr, colidx, selected_schedule, threads)
        weight_bf16 = weight.to(torch.bfloat16).contiguous()
        weight_t = weight_bf16.transpose(0, 1).contiguous()
        selected_scale = scale.index_select(0, row_ids).contiguous()
        q = torch.empty((rows, weight.shape[0]), dtype=torch.bfloat16)
        dw = torch.zeros_like(weight, dtype=torch.float32)
        db = torch.zeros_like(bias, dtype=torch.float32)
        total_loss = torch.zeros((), dtype=torch.float64)
        inv_rows = 1.0 / float(rows)
        fused_db = os.environ.get("TFS_SCOPE_FUSED_DB", "0") == "1"

        for r0 in range(0, rows, row_tile):
            r1 = min(rows, r0 + row_tile)
            pp = pulled[r0:r1]
            yy = compact_labels[r0:r1]
            panel_scale = selected_scale[r0:r1].contiguous()
            if os.environ.get("TFS_SCOPE_NATIVE_LOGITS", "0") == "1":
                logits = backend().c3_compact_logits_amx_shadow_v1(
                    pp.contiguous(), weight_bf16, bias, panel_scale, threads)
            else:
                logits = torch.matmul(pp, weight_bf16).float()
                logits.mul_(panel_scale.unsqueeze(1)).add_(bias)
            if os.environ.get("TFS_SCOPE_LOGSOFTMAX_OUT", "0") == "1":
                torch.log_softmax(logits, dim=1, out=logits)
                target = logits.gather(1, yy.unsqueeze(1)).squeeze(1)
                total_loss.add_(-target.double().sum())
                logits.exp_().mul_(inv_rows)
            else:
                row_lse = torch.logsumexp(logits, dim=1)
                target = logits.gather(1, yy.unsqueeze(1)).squeeze(1)
                total_loss.add_((row_lse - target).double().sum())
                logits.sub_(row_lse.unsqueeze(1)).exp_().mul_(inv_rows)
            logits[torch.arange(r1 - r0), yy] -= inv_rows
            if fused_db:
                gs, panel_db = backend().c3_scale_grad_bf16_db_v2(
                    logits, panel_scale, threads)
                db.add_(panel_db)
            else:
                db.add_(logits.sum(dim=0))
                gs = backend().c3_scale_grad_bf16_v1(
                    logits, panel_scale, threads)
            if os.environ.get("TFS_SCOPE_NATIVE_DW", "0") == "1":
                panel_dw = backend().c3_compact_dw_bf16_amx_shadow_v1(
                    pp.contiguous(), gs, threads)
            else:
                panel_dw = torch.matmul(
                    pp.transpose(0, 1).contiguous(), gs).float()
            dw.add_(panel_dw)
            q[r0:r1].copy_(torch.matmul(gs, weight_t))

        ctx.save_for_backward(q, dw, db, transpose_rowptr,
                              transpose_colidx, scale, transpose_schedule)
        ctx.threads = threads
        return total_loss.float().div_(float(rows))

    @staticmethod
    def backward(ctx, grad_loss):
        (q, dw, db, transpose_rowptr, transpose_colidx, scale,
         transpose_schedule) = ctx.saved_tensors
        dh = backend().c3_rect_pull_bf16_scaled_fp32_shadow_v1(
            q, transpose_rowptr, transpose_colidx, scale,
            transpose_schedule, ctx.threads)
        factor = float(grad_loss)
        if factor != 1.0:
            dh.mul_(factor)
            dw = dw * factor
            db = db * factor
        return (dh, dw, db, None, None, None, None, None, None, None,
                None, None, None, None)


def terminal_logits(hidden: torch.Tensor, terminal_conv, graph,
                    scope: SupervisionScope) -> torch.Tensor:
    """Return logits only for the scope rows, in ``scope.row_ids`` order."""
    if hidden.shape[0] != scope.node_count:
        raise ValueError("hidden row count does not match supervision scope")
    return _ScopedTerminalFunction.apply(
        hidden, terminal_conv.weight, terminal_conv.bias,
        graph.rowptr, graph.colidx, graph.scale, scope.row_ids,
        scope.transpose_rowptr, scope.transpose_colidx,
        scope.selected_schedule, scope.transpose_schedule,
        terminal_conv.threads)


def train_logits(model, x: torch.Tensor, graph,
                 scope: SupervisionScope) -> torch.Tensor:
    """Run ordinary hidden layers and the supervision-scoped terminal layer."""
    hidden = x
    for conv in model.convs[:-1]:
        hidden = F.dropout(F.relu(conv(hidden, graph)), p=model.dropout,
                           training=model.training)
    return terminal_logits(hidden, model.convs[-1], graph, scope)


def terminal_cross_entropy(hidden: torch.Tensor, terminal_conv, graph,
                           scope: SupervisionScope, labels: torch.Tensor,
                           row_tile: Optional[int] = None) -> torch.Tensor:
    """Return exact mean CE without a persistent selected_rows x D tensor."""
    if hidden.shape[0] != scope.node_count:
        raise ValueError("hidden row count does not match supervision scope")
    if row_tile is None:
        row_tile = int(os.environ.get("TFS_SCOPE_LOSS_ROW_TILE", "300000"))
    compact_labels = labels.index_select(0, scope.row_ids).contiguous()
    return _ScopedTerminalCrossEntropyFunction.apply(
        hidden, terminal_conv.weight, terminal_conv.bias,
        graph.rowptr, graph.colidx, graph.scale, scope.row_ids,
        scope.transpose_rowptr, scope.transpose_colidx,
        scope.selected_schedule, scope.transpose_schedule, compact_labels,
        terminal_conv.threads, int(row_tile))


def train_cross_entropy(model, x: torch.Tensor, labels: torch.Tensor, graph,
                        scope: SupervisionScope,
                        row_tile: Optional[int] = None) -> torch.Tensor:
    """Run hidden layers and the bounded-panel terminal CE dataflow."""
    hidden = x
    for conv in model.convs[:-1]:
        hidden = F.dropout(F.relu(conv(hidden, graph)), p=model.dropout,
                           training=model.training)
    return terminal_cross_entropy(
        hidden, model.convs[-1], graph, scope, labels, row_tile)


__all__ = [
    "SupervisionScope", "terminal_logits", "train_logits",
    "terminal_cross_entropy", "train_cross_entropy",
]
