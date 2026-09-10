"""Conservative shape planner for shadow supervision-scoped dense kernels.

The thresholds are deliberately limited to the region covered by the
2026-09-07 IGB and microbenchmark gates.  This module does not mutate the
authority planner or environment; callers must explicitly opt into a plan.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DenseFusionPlan:
    selected_rows: int
    hidden_dim: int
    output_dim: int
    threads: int
    row_tile: int
    fused_db: bool
    logsoftmax_out: bool
    native_dw: bool
    direct_tail_transpose: bool
    native_logits: bool
    rationale: tuple[str, ...]

    def validate(self, selected_rows: int, hidden_dim: int,
                 output_dim: int, threads: int) -> None:
        actual = (selected_rows, hidden_dim, output_dim, threads)
        expected = (self.selected_rows, self.hidden_dim,
                    self.output_dim, self.threads)
        if actual != expected:
            raise ValueError(
                "dense plan shape mismatch: "
                f"planned={expected}, actual={actual}")


def plan_supervision_dense(selected_rows: int, hidden_dim: int,
                           output_dim: int, threads: int) -> DenseFusionPlan:
    """Select only kernels supported by the measured shadow envelope.

    ``selected_rows / output_dim`` approximates whether compact dW has enough
    row work to amortize its D-wide transpose and thread-local reduction.
    ``selected_rows * output_dim / threads`` approximates per-worker logits
    work.  The high-D floor reflects the real end-to-end crossover, not the
    much more optimistic isolated GEMM crossover.
    """
    if min(selected_rows, hidden_dim, output_dim, threads) <= 0:
        raise ValueError("all dense-plan dimensions must be positive")
    if threads > 32:
        raise ValueError("shadow dense kernels support at most 32 threads")

    rationale = []
    row_tile = min(selected_rows, 300_000)
    logsoftmax_out = True
    native_dw = (
        hidden_dim == 128 and threads >= 8 and output_dim >= 512 and
        selected_rows / output_dim >= 20.0)
    if native_dw:
        rationale.append("compact dW amortizes transpose/reduction")
    else:
        rationale.append("framework dW retained outside measured envelope")

    direct_tail = native_dw and output_dim % 32 != 0
    if direct_tail:
        rationale.append("direct-tail transpose avoids D padding copy")

    work_per_thread = selected_rows * output_dim / threads
    native_logits = (
        hidden_dim == 128 and threads >= 8 and output_dim >= 2500 and
        work_per_thread >= 500_000)
    if native_logits:
        rationale.append("high-D fused logits exceeds end-to-end crossover")
    else:
        rationale.append("framework logits retained below high-D crossover")

    # The fused reduction removes a separate MxD gradient scan, but its
    # chunk-local accumulator is only enabled in the same measured region as
    # compact dW.  Outside that envelope the framework reduction is retained.
    fused_db = native_dw
    if fused_db:
        rationale.append("tile-local db removes a separate gradient scan")
    rationale.append(f"bounded exact CE uses row_tile={row_tile}")

    return DenseFusionPlan(
        selected_rows, hidden_dim, output_dim, threads, row_tile,
        fused_db, logsoftmax_out, native_dw, direct_tail, native_logits,
        tuple(rationale))


__all__ = ["DenseFusionPlan", "plan_supervision_dense"]
