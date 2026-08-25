"""Memory-bounded aggregate-first backward for wide output layers.

This module is deliberately separate from the frozen ``final_v1`` path.  It
implements the first safe version of the high-D backward contract:

* the forward-produced BF16 ``P=A(H)`` is consumed by row panels;
* only one scaled BF16 gradient panel is live at a time;
* ``dW`` is accumulated directly in logical ``[K, D]`` layout;
* ``dP`` is materialized only when the caller really needs ``dX``;
* the existing AMX pull-only primitive remains the sparse ``A^T`` stage.

The old implementation formed a full ``Gs=[N,D]`` BF16 tensor before the
dense products.  For IGB-HOM with D=2983 that temporary is several GiB and
also made the high-D path easy to accidentally route through a different
implementation.  The stream path is gated by ``TFS_HIGHD_STREAM_V1=1`` so
the authoritative baseline remains reproducible until its numerical and
timing gates pass.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import os
import time
from typing import Any, Optional, Tuple

import torch

from .native import backend


_ROW_CANDIDATES = (65536, 32768, 16384, 8192, 4096, 2048, 1024, 512,
                   256, 128, 64, 32)


def _positive_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return int(default)
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + multiple - 1) // multiple) * multiple


@dataclass(frozen=True)
class HighDBackwardPlan:
    """Shape-only plan and memory accounting for one aggregate backward."""

    n: int
    k: int
    d: int
    kp: int
    dp: int
    threads: int
    row_panel: int
    d_tile: int
    strategy: str
    compute_dx: bool
    budget_bytes: int
    estimated_panel_bytes: int
    persistent_dx_bytes: int
    dense_d_tile: int = 0
    sparse_d_slab: int = 0
    d_slabs: tuple[tuple[int, int], ...] = ()
    dslab_max_width: int = 0
    workspace_bytes: int = 0
    panel_budget_bytes: int = 0
    plan_id: str = ""
    execution_variant: str = ""

    @property
    def selected_impl(self) -> str:
        return self.execution_variant

    @property
    def within_budget(self) -> bool:
        budget = self.panel_budget_bytes or self.budget_bytes
        return self.estimated_panel_bytes <= budget

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["within_budget"] = self.within_budget
        return payload

    def log_line(self) -> str:
        return (
            "TFS_HIGHD_PLAN "
            f"N={self.n} K={self.k} D={self.d} Kp={self.kp} Dp={self.dp} "
            f"threads={self.threads} row_panel={self.row_panel} "
            f"d_tile={self.d_tile} dense_d_tile={self.dense_d_tile or self.d_tile} "
            f"sparse_d_slab={self.sparse_d_slab or self.d_tile} "
            f"strategy={self.strategy} "
            f"compute_dx={int(self.compute_dx)} budget_bytes={self.budget_bytes} "
            f"panel_bytes={self.estimated_panel_bytes} "
            f"panel_budget_bytes={self.panel_budget_bytes or self.budget_bytes} "
            f"persistent_dx_bytes={self.persistent_dx_bytes} "
            f"dslabs={len(self.d_slabs)} dslab_max={self.dslab_max_width} "
            f"workspace_bytes={self.workspace_bytes} plan_id={self.plan_id or 'none'} "
            f"within_budget={int(self.within_budget)}"
        )


def highd_plan_from_execution_plan(plan: Any) -> HighDBackwardPlan:
    """Adapt the canonical ``LayerExecutionPlan`` for the stream kernel.

    ``execution_plan.py`` is the only place that makes the formal order,
    tile and workspace decision.  This adapter intentionally performs no
    shape heuristic; it only copies the already-selected values into the
    stream implementation's profiling record.
    """

    required = ("n", "k", "d", "kp", "dp", "panel", "d_tile",
                "workspace_strategy", "workspace_bytes")
    missing = [name for name in required if not hasattr(plan, name)]
    if missing:
        raise TypeError("execution plan is missing fields: " + ", ".join(missing))
    strategy = str(plan.workspace_strategy)
    if strategy not in {"full_d", "d_slab", "not_applicable"}:
        raise ValueError(f"invalid workspace strategy {strategy!r}")
    threads = max(1, int(getattr(plan, "threads", 1)))
    budget = max(1, int(getattr(plan, "workspace_budget_bytes", 0) or
                        os.environ.get("TFS_HIGHD_BWD_BUDGET_BYTES", 64 << 20)))
    estimated = int(getattr(plan, "panel_working_set_bytes", 0) or
                    _panel_working_set(
                        int(plan.panel), int(plan.kp), int(plan.dp), int(plan.k),
                        int(plan.d), bool(plan.compute_dx), native_scale=True))
    d_slabs = tuple(getattr(plan, "d_slabs", ()) or ((0, int(plan.d)),))
    dslab_max = int(getattr(plan, "dslab_max_width", 0) or
                    max(d1 - d0 for d0, d1 in d_slabs))
    return HighDBackwardPlan(
        n=int(plan.n), k=int(plan.k), d=int(plan.d),
        kp=int(plan.kp), dp=int(plan.dp), threads=threads,
        row_panel=min(int(plan.n), max(1, int(plan.panel))),
        d_tile=max(1, int(plan.d_tile)),
        dense_d_tile=max(1, int(getattr(plan, "dense_d_tile", 0) or
                               int(plan.d_tile))),
        sparse_d_slab=max(1, int(getattr(plan, "sparse_d_slab", 0) or
                                int(plan.d_tile))),
        strategy=strategy,
        compute_dx=bool(plan.compute_dx), budget_bytes=budget,
        estimated_panel_bytes=int(estimated),
        persistent_dx_bytes=(int(plan.n) * int(plan.k) * 4
                             if plan.compute_dx else 0),
        d_slabs=d_slabs,
        dslab_max_width=dslab_max,
        workspace_bytes=int(getattr(plan, "workspace_bytes", 0)),
        panel_budget_bytes=int(getattr(plan, "panel_budget_bytes", 0)),
        plan_id=str(getattr(plan, "plan_id", "")),
        execution_variant=str(getattr(plan, "execution_variant", "")),
    )


def _plan_budget_bytes(plan: Any) -> int:
    """Return the immutable budget from either supported plan representation."""

    budget = getattr(plan, "budget_bytes", None)
    if budget is None:
        budget = getattr(plan, "workspace_budget_bytes", None)
    if budget is None or int(budget) <= 0:
        raise ValueError("execution plan has no positive workspace budget")
    return int(budget)


def highd_stream_enabled() -> bool:
    """Return whether the experimental V1 stream path is explicitly enabled."""

    value = os.environ.get("TFS_HIGHD_STREAM_V1", "0").strip().lower()
    return value not in {"", "0", "off", "false", "no"}


def should_use_stream(n: int, k: int, d: int, compute_dx: bool) -> bool:
    """Apply the shape/workset gate before entering the streamed kernel.

    A panel loop is not automatically faster.  When K is very wide and the
    complete scaled-gradient tensor is already within the scratch budget,
    the established one-shot GEMM avoids panel launch/reduction overhead and
    is the better path.  Conversely, a large-N wide-D layer must stream even
    if its K is large because materializing ``Gs[N,D]`` is the dominant
    memory cost.  ``TFS_HIGHD_STREAM_AUTO=0`` is an explicit forced-stream
    ablation for kernel correctness tests.
    """

    if not highd_stream_enabled():
        return False
    mode = os.environ.get("TFS_HIGHD_STREAM_AUTO", "1").strip().lower()
    if mode in {"0", "off", "false", "no"}:
        return True
    if mode not in {"1", "on", "true", "yes", "auto"}:
        raise ValueError("TFS_HIGHD_STREAM_AUTO must be 0|1|auto")
    budget = _positive_env("TFS_HIGHD_BWD_BUDGET_BYTES", 64 << 20)
    dp = _round_up(int(d), 32)
    full_gs_bytes = int(n) * dp * 2
    # The high-K threshold is deliberately shape-based.  K<=256 is the
    # common AMX panel regime where streaming is useful even for a small gate
    # fixture; for wider K retain the one-shot path while it fits in budget.
    if int(k) > 256 and full_gs_bytes <= budget:
        return False
    return True


def _panel_working_set(row_panel: int, kp: int, dp: int, k: int,
                       d: int, compute_dx: bool, native_scale: bool) -> int:
    """Conservative live temporary estimate, excluding the output dW."""

    # P is BF16 and the native scale routine returns BF16 directly.  The
    # fallback ATen expression has an FP32 multiply temporary as well.
    bytes_live = row_panel * kp * 2
    bytes_live += row_panel * dp * 2
    bytes_live += row_panel * d * (2 if native_scale else 6)
    if compute_dx:
        # dP panel plus the GEMM result.  They are kept separate deliberately
        # so the copy into the full pull input is race-free.
        bytes_live += row_panel * k * 4 * 2
    return int(bytes_live)


def choose_highd_backward_plan(
    n: int,
    k: int,
    d: int,
    threads: int,
    compute_dx: bool,
    *,
    budget_bytes: Optional[int] = None,
    native_scale: bool = True,
) -> HighDBackwardPlan:
    """Choose row/D tiles from shape and memory budget, never dataset name."""

    n, k, d, threads = map(int, (n, k, d, threads))
    if min(n, k, d, threads) <= 0:
        raise ValueError("N/K/D/threads must be positive")
    kp, dp = _round_up(k, 64), _round_up(d, 32)
    if budget_bytes is None:
        budget_bytes = _positive_env("TFS_HIGHD_BWD_BUDGET_BYTES", 64 << 20)
    budget_bytes = int(budget_bytes)
    if budget_bytes <= 0:
        raise ValueError("budget_bytes must be positive")

    requested_panel = os.environ.get("TFS_HIGHD_BWD_ROW_PANEL")
    if requested_panel:
        row_panel = min(n, _positive_env("TFS_HIGHD_BWD_ROW_PANEL", 512))
    else:
        # Start at the largest candidate, but retain the smallest candidate
        # when even the budget is too small for a larger panel.  The latter
        # case is reported through ``within_budget`` instead of silently
        # selecting an unbounded N-row temporary.
        row_panel = min(n, _ROW_CANDIDATES[-1])
        found = False
        for candidate in _ROW_CANDIDATES:
            candidate = min(n, candidate)
            if _panel_working_set(candidate, kp, dp, k, d, compute_dx,
                                  native_scale) <= budget_bytes:
                row_panel = max(1, candidate)
                found = True
                break
        if not found:
            # The candidate list is intentionally coarse.  Derive a final
            # exact row count so the planner still honors a very small test
            # budget instead of overshooting merely because 32 was too large.
            per_row = _panel_working_set(1, kp, dp, k, d, compute_dx,
                                         native_scale)
            row_panel = max(1, min(n, budget_bytes // max(1, per_row)))

    # A full per-thread dW workspace is cheap for the current K=128/D=2983
    # layer (about 49 MiB at 32 threads).  For wider shapes the *thread-local*
    # footprint, rather than only the logical output size, determines whether
    # slabbed accumulation is required.  This is the budget gate that keeps a
    # high-D native candidate from silently allocating hundreds of MiB per
    # invocation.
    output_bytes = k * d * 4
    local_dw_bytes = threads * kp * dp * 4
    requested_tile = os.environ.get("TFS_HIGHD_BWD_D_TILE")
    if requested_tile:
        d_tile = min(d, _positive_env("TFS_HIGHD_BWD_D_TILE", 256))
    elif local_dw_bytes <= budget_bytes and output_bytes <= (64 << 20):
        d_tile = d
    else:
        d_tile = min(d, 256)
    strategy = "full_d" if d_tile >= d else "d_slab"
    estimated = _panel_working_set(row_panel, kp, dp, k, d, compute_dx,
                                   native_scale)
    return HighDBackwardPlan(
        n=n, k=k, d=d, kp=kp, dp=dp, threads=threads,
        row_panel=row_panel, d_tile=d_tile, strategy=strategy,
        compute_dx=bool(compute_dx), budget_bytes=budget_bytes,
        estimated_panel_bytes=estimated,
        persistent_dx_bytes=n * k * 4 if compute_dx else 0,
    )


def _scale_panel(grad: torch.Tensor, scale: torch.Tensor, threads: int) -> torch.Tensor:
    """Produce exactly the established BF16 scaled-gradient panel."""

    grad = grad.contiguous()
    scale = scale.to(dtype=torch.float32).contiguous()
    if grad.device.type == "cpu" and os.environ.get(
            "TFS_HIGHD_NATIVE_SCALE", "1").strip().lower() not in {
                "0", "off", "false", "no"}:
        native = getattr(backend(), "c3_scale_grad_bf16_v1", None)
        if native is not None:
            return native(grad, scale, int(threads))
    return (grad * scale.unsqueeze(1)).to(torch.bfloat16)


def streamed_aggregate_backward(
    pulled: torch.Tensor,
    weight: torch.Tensor,
    grad: torch.Tensor,
    scale: torch.Tensor,
    rowptr: torch.Tensor,
    colidx: torch.Tensor,
    threads: int,
    compute_dx: bool,
    *,
    plan: Optional[HighDBackwardPlan] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, HighDBackwardPlan]:
    """Compute ``dX``/``dW`` without materializing a full ``Gs`` tensor."""

    if pulled.dim() != 2 or weight.dim() != 2 or grad.dim() != 2:
        raise ValueError("pulled, weight and grad must be rank-2 tensors")
    n, k = map(int, pulled.shape)
    wk, d = map(int, weight.shape)
    if wk != k or tuple(grad.shape) != (n, d):
        raise ValueError(
            f"shape mismatch pulled={tuple(pulled.shape)} "
            f"weight={tuple(weight.shape)} grad={tuple(grad.shape)}")
    if scale.numel() != n:
        raise ValueError("scale must contain one value per row")
    if plan is None:
        strict = os.environ.get("TFS_HIGHD_RELEASE_MODE", "0").strip().lower() not in {
            "", "0", "off", "false", "no"
        }
        if strict and os.environ.get("TFS_RELEASE_CONTRACT_STRICT", "0").strip().lower() not in {
                "", "0", "off", "false", "no"}:
            raise RuntimeError(
                "authority backward requires the model construction execution plan")
        plan = choose_highd_backward_plan(n, k, d, int(threads), compute_dx)
    if plan.n != n or plan.k != k or plan.d != d:
        raise ValueError("plan shape does not match tensors")
    if os.environ.get("TFS_HIGHD_LOG_PLAN", "0").strip().lower() not in {
            "", "0", "off", "false", "no"}:
        print(plan.log_line(), flush=True)

    # The output is always logical [K,D].  In particular, never expose the
    # internal D-major AMX workspace layout to autograd callers.
    dw = torch.zeros((k, d), dtype=torch.float32, device=grad.device)
    d_pulled = (torch.empty((n, k), dtype=torch.float32, device=grad.device)
                if compute_dx else None)
    weight_bf16 = weight.to(torch.bfloat16) if compute_dx else None
    weight_bf16_t = weight_bf16.transpose(0, 1) if compute_dx else None
    scale_f = scale.to(torch.float32).contiguous()
    profile = os.environ.get("TFS_HIGHD_PROFILE", "0").strip().lower() not in {
        "", "0", "off", "false", "no"}
    scale_ms = dw_ms = dp_ms = pull_ms = 0.0
    panel_count = 0

    for row0 in range(0, n, plan.row_panel):
        panel_count += 1
        row1 = min(n, row0 + plan.row_panel)
        p_panel = pulled.narrow(0, row0, row1 - row0).contiguous()
        g_panel = grad.narrow(0, row0, row1 - row0)
        s_panel = scale_f.narrow(0, row0, row1 - row0)
        t_stage = time.perf_counter() if profile else 0.0
        gs_panel = _scale_panel(g_panel, s_panel, plan.threads)
        if profile:
            scale_ms += (time.perf_counter() - t_stage) * 1000.0

        if plan.strategy == "full_d":
            t_stage = time.perf_counter() if profile else 0.0
            dw.add_(torch.matmul(p_panel.transpose(0, 1), gs_panel).float())
            if profile:
                dw_ms += (time.perf_counter() - t_stage) * 1000.0
            if compute_dx:
                t_stage = time.perf_counter() if profile else 0.0
                d_p = torch.matmul(gs_panel, weight_bf16_t).float()
                if profile:
                    dp_ms += (time.perf_counter() - t_stage) * 1000.0
                d_pulled.narrow(0, row0, row1 - row0).copy_(d_p)
        else:
            d_p = (torch.zeros((row1 - row0, k), dtype=torch.float32,
                               device=grad.device) if compute_dx else None)
            for d0 in range(0, d, plan.d_tile):
                d1 = min(d, d0 + plan.d_tile)
                gs_tile = gs_panel.narrow(1, d0, d1 - d0)
                t_stage = time.perf_counter() if profile else 0.0
                dw[:, d0:d1].add_(
                    torch.matmul(p_panel.transpose(0, 1), gs_tile).float())
                if profile:
                    dw_ms += (time.perf_counter() - t_stage) * 1000.0
                if compute_dx:
                    t_stage = time.perf_counter() if profile else 0.0
                    d_p.add_(torch.matmul(
                        gs_tile, weight_bf16[:, d0:d1].transpose(0, 1)
                    ).float())
                    if profile:
                        dp_ms += (time.perf_counter() - t_stage) * 1000.0
            d_pulled.narrow(0, row0, row1 - row0).copy_(d_p)

    if not compute_dx:
        return None, dw, plan
    t_stage = time.perf_counter() if profile else 0.0
    d_h = backend().c3_pull_only_amx_v1(
        d_pulled, rowptr, colidx, int(threads))
    if profile:
        pull_ms = (time.perf_counter() - t_stage) * 1000.0
    dx = d_h * scale_f.unsqueeze(1)
    if profile:
        print(
            "TFS_HIGHD_INTERNAL "
            f"N={n} K={k} D={d} threads={threads} panels={panel_count} "
            f"scale_ms={scale_ms:.3f} dw_ms={dw_ms:.3f} "
            f"dp_ms={dp_ms:.3f} pull_ms={pull_ms:.3f}",
            flush=True)
    return dx, dw, plan


def native_transform_highd_backward(
    hs: torch.Tensor,
    weight: torch.Tensor,
    grad: torch.Tensor,
    scale: torch.Tensor,
    rowptr: torch.Tensor,
    colidx: torch.Tensor,
    threads: int,
    compute_dx: bool,
    *,
    d_slabs: Optional[tuple[tuple[int, int], ...]] = None,
    d_tile: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Run the native Transform-HighD slab entry point.

    The C++ entry consumes exactly one logical D slab and returns FP32 dW/dHs
    for that slab.  This thin adapter only partitions the logical output and
    performs the required FP32 dHs accumulation; it does not make a second
    shape or workspace decision.
    """

    native = getattr(backend(), "c3_backward_transform_highd_amx_v1", None)
    if native is None:
        raise RuntimeError("native transform high-D symbol is unavailable")
    if hs.dim() != 2 or weight.dim() != 2 or grad.dim() != 2:
        raise ValueError("hs, weight and grad must be rank-2 tensors")
    n, k = map(int, hs.shape)
    wk, d = map(int, weight.shape)
    if wk != k or tuple(grad.shape) != (n, d) or d <= 128 or k <= d:
        raise ValueError("native transform high-D shape mismatch")
    if scale.numel() != n:
        raise ValueError("scale must contain one value per row")
    if d_slabs:
        ranges = tuple((int(d0), int(d1)) for d0, d1 in d_slabs)
    else:
        limit = int(d_tile or os.environ.get("TFS_TRANSFORM_HIGHD_D_TILE", "256"))
        limit = max(129, min(d, limit))
        from .execution_plan import partition_d
        ranges = tuple(partition_d(d, limit, min_native=129))
    if not ranges or ranges[0][0] != 0 or ranges[-1][1] != d:
        raise ValueError("native transform slabs must cover the complete output")
    if any(d1 <= d0 or d1 - d0 <= 128 for d0, d1 in ranges):
        raise ValueError("native transform slabs must have width >128")
    max_slab_width = max(d1 - d0 for d0, d1 in ranges)
    gs_budget = _positive_env("TFS_TRANSFORM_HIGHD_NATIVE_GS_BUDGET_BYTES",
                              256 << 20)
    if n * _round_up(max_slab_width, 32) * 2 > gs_budget:
        raise RuntimeError(
            "native transform scaled-gradient slab exceeds the configured "
            "GS budget")
    hs_bf16 = hs if hs.dtype == torch.bfloat16 else hs.to(torch.bfloat16)
    dw = torch.empty((k, d), dtype=torch.float32, device=grad.device)
    dx = (torch.zeros((n, k), dtype=torch.float32, device=grad.device)
          if compute_dx else None)
    for d0, d1 in ranges:
        part_dx, part_dw, _part_db, _meta = native(
            grad.narrow(1, d0, d1 - d0).contiguous(),
            hs_bf16,
            weight.narrow(1, d0, d1 - d0).contiguous(),
            rowptr,
            colidx,
            scale,
            int(threads),
            bool(compute_dx),
        )
        dw.narrow(1, d0, d1 - d0).copy_(part_dw)
        if compute_dx:
            dx.add_(part_dx)
    return dx, dw


def native_transform_highd_single_scan_backward(
    hs: torch.Tensor,
    weight: torch.Tensor,
    grad: torch.Tensor,
    scale: torch.Tensor,
    rowptr: torch.Tensor,
    colidx: torch.Tensor,
    threads: int,
    compute_dx: bool,
    *,
    budget_bytes: Optional[int] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Run the fused immediate-consume/single-CSR-scan native candidate.

    The C++ entry keeps only one row-panel ``dT`` buffer live: source scaling,
    BF16 rounding, sparse pull, dW and dHs are performed before the next
    panel.  This helper intentionally has a separate opt-in entry point so a
    component probe cannot silently alter the authority path.
    """

    native = getattr(
        backend(), "c3_backward_transform_highd_single_scan_amx_v1", None)
    if native is None:
        raise RuntimeError("native transform single-scan symbol is unavailable")
    if hs.dim() != 2 or weight.dim() != 2 or grad.dim() != 2:
        raise ValueError("single-scan transform inputs must be rank-2")
    n, k = map(int, hs.shape)
    wk, d = map(int, weight.shape)
    if wk != k or tuple(grad.shape) != (n, d) or d <= 128 or k <= d:
        raise ValueError("single-scan transform high-D shape mismatch")
    if scale.numel() != n:
        raise ValueError("scale must contain one value per row")
    kp = _round_up(k, 64)
    dp = _round_up(d, 32)
    # Production callers receive their budget from the immutable execution
    # plan.  Keep the environment fallback solely for direct component probes
    # that do not construct such a plan.
    budget = (_positive_env("TFS_HIGHD_BWD_BUDGET_BYTES", 64 << 20)
              if budget_bytes is None else int(budget_bytes))
    if budget <= 0:
        raise ValueError("budget_bytes must be positive")
    local_dw = int(max(1, int(threads)) * kp * dp * 4)
    if local_dw > budget:
        raise RuntimeError(
            "single-scan native dW workspace exceeds the configured budget")
    return native(
        grad.contiguous(),
        hs if hs.dtype == torch.bfloat16 else hs.to(torch.bfloat16),
        weight.contiguous(), rowptr, colidx, scale.contiguous(),
        int(threads), bool(compute_dx))[:2]


def streamed_transform_backward(
    hs: torch.Tensor,
    weight: torch.Tensor,
    grad: torch.Tensor,
    scale: torch.Tensor,
    rowptr: torch.Tensor,
    colidx: torch.Tensor,
    threads: int,
    compute_dx: bool,
    *,
    d_tile: Optional[int] = None,
    plan: Optional[HighDBackwardPlan] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Transform-first High-D backward with one CSR scan per D macro slab.

    The historical wide-output wrapper invoked the D<=128 sparse backward
    once per output tile.  This implementation keeps the exact BF16 operand
    contract but processes a planner-selected 256/512-column slab: scale and
    BF16-round the gradient, perform one pull, then run the dense dW/dX
    products.  It is deliberately a Python/ATen correctness path until the
    native d-slab kernel receives its own C++ gate.
    """

    if hs.dim() != 2 or weight.dim() != 2 or grad.dim() != 2:
        raise ValueError("hs, weight and grad must be rank-2 tensors")
    n, k = map(int, hs.shape)
    wk, d = map(int, weight.shape)
    if wk != k or tuple(grad.shape) != (n, d):
        raise ValueError(
            f"shape mismatch hs={tuple(hs.shape)} weight={tuple(weight.shape)} "
            f"grad={tuple(grad.shape)}")
    if scale.numel() != n:
        raise ValueError("scale must contain one value per row")
    if d <= 128:
        raise ValueError("transform High-D stream requires D>128")
    if plan is not None:
        if plan.n != n or plan.k != k or plan.d != d:
            raise ValueError("transform plan shape does not match tensors")
        d_tile = int(plan.d_tile)
    elif d_tile is None:
        d_tile = int(os.environ.get("TFS_TRANSFORM_HIGHD_D_TILE", "256"))
    d_tile = max(128, min(d, int(d_tile)))
    if plan is None and d_tile % 32:
        raise ValueError("transform High-D d_tile must be 32-aligned")

    profile = os.environ.get("TFS_HIGHD_PROFILE", "0").strip().lower() not in {
        "", "0", "off", "false", "no"}
    t0 = time.perf_counter() if profile else 0.0
    ranges = (tuple(plan.d_slabs) if plan is not None and plan.d_slabs
              else tuple((d0, min(d, d0 + d_tile))
                         for d0 in range(0, d, d_tile)))
    planned_variant = (plan.execution_variant if plan is not None else "")
    if planned_variant:
        if planned_variant not in {
                "transform_highd_stream", "transform_highd_native",
                "transform_highd_single_scan"}:
            raise RuntimeError(
                f"invalid transform High-D execution variant: {planned_variant}")
        if planned_variant == "transform_highd_single_scan":
            return native_transform_highd_single_scan_backward(
                hs, weight, grad, scale, rowptr, colidx, threads,
                compute_dx, budget_bytes=_plan_budget_bytes(plan))
        if planned_variant == "transform_highd_native":
            return native_transform_highd_backward(
                hs, weight, grad, scale, rowptr, colidx, threads,
                compute_dx, d_slabs=ranges, d_tile=d_tile)

    native_mode = os.environ.get("TFS_TRANSFORM_HIGHD_NATIVE", "off").strip().lower()
    if native_mode in {"1", "on", "true", "yes"}:
        native_mode = "on"
    elif native_mode in {"0", "off", "false", "no", ""}:
        native_mode = "off"
    elif native_mode == "auto":
        max_width = max(d1 - d0 for d0, d1 in ranges)
        max_bytes = _positive_env("TFS_TRANSFORM_HIGHD_NATIVE_GS_BUDGET_BYTES",
                                  256 << 20)
        planner_selected_native = bool(
            plan is not None and
            plan.execution_variant == "transform_highd_native")
        native_mode = ("on" if planner_selected_native or
                        (len(ranges) == 1 and
                         n * _round_up(max_width, 32) * 2 <= max_bytes)
                        else "off")
    else:
        raise ValueError("TFS_TRANSFORM_HIGHD_NATIVE must be off|on|auto")
    single_scan_mode = os.environ.get(
        "TFS_TRANSFORM_HIGHD_SINGLE_SCAN", "off").strip().lower()
    if single_scan_mode in {"1", "true", "yes"}:
        single_scan_mode = "on"
    elif single_scan_mode in {"0", "false", "no", ""}:
        single_scan_mode = "off"
    elif single_scan_mode not in {"auto", "on", "off"}:
        raise ValueError(
            "TFS_TRANSFORM_HIGHD_SINGLE_SCAN must be off|on|auto")
    # ``auto`` is only enabled by an immutable planner selection.  The
    # default authority profile keeps it off until the complete shape/thread
    # gate has been accepted.
    planner_single_scan = bool(
        plan is not None and
        plan.execution_variant == "transform_highd_single_scan")
    if single_scan_mode == "auto":
        single_scan_mode = "on" if planner_single_scan else "off"
    elif single_scan_mode == "on" and plan is not None and not planner_single_scan:
        # A production dispatcher must not change the implementation after
        # the immutable construction-time plan has been selected.  Keep an
        # explicit ``on`` probe usable for direct callers, but turn it into a
        # normal stream (or fail in strict mode) when a supplied plan did not
        # authorize the candidate.
        strict = os.environ.get(
            "TFS_TRANSFORM_HIGHD_SINGLE_SCAN_STRICT", "0").strip().lower() not in {
                "", "0", "off", "false", "no"}
        if strict:
            raise RuntimeError(
                "single-scan candidate is not authorized by the execution plan")
        if os.environ.get(
                "TFS_HIGHD_LOG_FALLBACK", "0").strip().lower() not in {
                    "", "0", "off", "false", "no"}:
            print("TFS_TRANSFORM_HIGHD_SINGLE_SCAN_FALLBACK "
                  "reason=plan_not_selected", flush=True)
        single_scan_mode = "off"
    if single_scan_mode == "on":
        try:
            return native_transform_highd_single_scan_backward(
                hs, weight, grad, scale, rowptr, colidx, threads,
                compute_dx,
                budget_bytes=(_plan_budget_bytes(plan)
                              if plan is not None else None))
        except (RuntimeError, ValueError, AttributeError) as exc:
            strict = os.environ.get(
                "TFS_TRANSFORM_HIGHD_SINGLE_SCAN_STRICT", "0").strip().lower() not in {
                    "", "0", "off", "false", "no"}
            if strict:
                raise
            if os.environ.get(
                    "TFS_HIGHD_LOG_FALLBACK", "0").strip().lower() not in {
                        "", "0", "off", "false", "no"}:
                print("TFS_TRANSFORM_HIGHD_SINGLE_SCAN_FALLBACK "
                      f"reason={type(exc).__name__}:{exc}", flush=True)

    if native_mode == "on":
        try:
            return native_transform_highd_backward(
                hs, weight, grad, scale, rowptr, colidx, threads,
                compute_dx, d_slabs=ranges, d_tile=d_tile)
        except (RuntimeError, ValueError, AttributeError) as exc:
            strict = os.environ.get(
                "TFS_TRANSFORM_HIGHD_NATIVE_STRICT", "0").strip().lower() not in {
                    "", "0", "off", "false", "no"}
            if strict:
                raise
            if os.environ.get("TFS_HIGHD_LOG_FALLBACK", "0").strip().lower() not in {
                    "", "0", "off", "false", "no"}:
                print(f"TFS_TRANSFORM_HIGHD_NATIVE_FALLBACK "
                      f"reason={type(exc).__name__}:{exc}", flush=True)
    hs_bf16 = hs if hs.dtype == torch.bfloat16 else hs.to(torch.bfloat16)
    weight_bf16 = weight.to(torch.bfloat16)
    scale_f = scale.to(torch.float32).contiguous()
    dw = torch.empty((k, d), dtype=torch.float32, device=grad.device)
    dx = (torch.zeros((n, k), dtype=torch.float32, device=grad.device)
          if compute_dx else None)
    pull_ms = dense_ms = 0.0
    slabs = 0
    bf16_pull = getattr(backend(), "c3_pull_only_bf16_amx_v1", None)
    for d0, d1 in ranges:
        slabs += 1
        g_slab = grad.narrow(1, d0, d1 - d0).contiguous()
        gs_bf16 = _scale_panel(g_slab, scale_f, int(threads))
        pull_start = time.perf_counter() if profile else 0.0
        # The BF16-input primitive consumes the already-rounded panel
        # directly.  Keep the FP32 primitive as a compatibility fallback for
        # old extensions; the two paths use the same CSR pull order.
        if bf16_pull is not None:
            d_p_bf16 = bf16_pull(
                gs_bf16, rowptr, colidx, int(threads)
            )
        else:
            d_p = backend().c3_pull_only_amx_v1(
                gs_bf16.float().contiguous(), rowptr, colidx, int(threads)
            )
            d_p_bf16 = d_p.to(torch.bfloat16)
        if profile:
            pull_ms += (time.perf_counter() - pull_start) * 1000.0
        dense_start = time.perf_counter() if profile else 0.0
        dw[:, d0:d1] = torch.matmul(
            hs_bf16.transpose(0, 1), d_p_bf16
        ).float()
        if compute_dx:
            dx.add_(torch.matmul(
                d_p_bf16, weight_bf16[:, d0:d1].transpose(0, 1)
            ).float().mul_(scale_f.unsqueeze(1)))
        if profile:
            dense_ms += (time.perf_counter() - dense_start) * 1000.0
    if profile:
        print(
            "TFS_TRANSFORM_HIGHD_INTERNAL "
            f"N={n} K={k} D={d} threads={int(threads)} "
            f"d_tile={d_tile} slabs={slabs} pull_ms={pull_ms:.3f} "
            f"dense_ms={dense_ms:.3f} total_ms="
            f"{(time.perf_counter() - t0) * 1000.0:.3f}",
            flush=True)
    return dx, dw


def native_aggregate_d_slab_backward(
    pulled: torch.Tensor,
    weight: torch.Tensor,
    grad: torch.Tensor,
    scale: torch.Tensor,
    rowptr: torch.Tensor,
    colidx: torch.Tensor,
    threads: int,
    compute_dx: bool,
    *,
    d_tile: int,
    d_slabs: Optional[tuple[tuple[int, int], ...]] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Use the native AMX high-D kernel one bounded D slab at a time.

    The v1 C++ entry point owns a full-D per-thread dW workspace.  For shapes
    where the unified planner selects ``d_slab``, this adapter invokes that
    proven native entry point on bounded output slices and combines the
    logical results.  It is a memory-safe native slice path, not a claim that
    the C++ kernel has a single-scan d-slab implementation; the latter remains
    a separate optimization target.
    """

    n, k = map(int, pulled.shape)
    wk, d = map(int, weight.shape)
    if wk != k or tuple(grad.shape) != (n, d) or d <= 128:
        raise ValueError("native d-slab aggregate shape mismatch")
    if d_slabs:
        ranges = [(int(d0), int(d1)) for d0, d1 in d_slabs]
    else:
        # Compatibility callers may still provide only d_tile.  Use the same
        # strict partition algorithm as the canonical planner; never merge a
        # small tail into a slab larger than the requested bound.
        from .execution_plan import partition_d
        d_tile = max(129, min(int(d_tile), d))
        ranges = list(partition_d(d, d_tile, min_native=129))
    if not ranges or ranges[-1][1] != d:
        raise ValueError("d-slab ranges must cover the complete output")
    if len(ranges) > 1 and any((d1 - d0) > int(d_tile)
                               for d0, d1 in ranges):
        raise ValueError("native d-slab exceeds the requested maximum width")
    if len(ranges) > 1 and any(d1 - d0 <= 128 for d0, d1 in ranges):
        raise ValueError("native d-slab partition contains a <=128 tail")
    native = getattr(backend(), "c3_backward_aggregate_highd_amx_v1", None)
    if native is None:
        raise RuntimeError("native aggregate high-D symbol is unavailable")
    dw = torch.empty((k, d), dtype=torch.float32, device=grad.device)
    db_parts = []
    dx = (torch.zeros((n, k), dtype=torch.float32, device=grad.device)
          if compute_dx else None)
    for d0, d1 in ranges:
        part_dx, part_dw, part_db, _ = native(
            grad.narrow(1, d0, d1 - d0).contiguous(), pulled,
            weight.narrow(1, d0, d1 - d0).contiguous(), rowptr, colidx,
            scale, int(threads), compute_dx)
        if compute_dx:
            dx.add_(part_dx)
        dw.narrow(1, d0, d1 - d0).copy_(part_dw)
        db_parts.append(part_db)
    return dx, dw, torch.cat(db_parts, dim=0)


def streamed_aggregate_single_scan_backward(
    pulled: torch.Tensor,
    weight: torch.Tensor,
    grad: torch.Tensor,
    scale: torch.Tensor,
    rowptr: torch.Tensor,
    colidx: torch.Tensor,
    threads: int,
    compute_dx: bool,
    *,
    plan: Optional[HighDBackwardPlan] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Aggregate High-D d-slab stream with one final CSR pull.

    The regular native d-slab adapter invokes the C++ full-D entry once per
    output slab, which repeats the CSR pull used to form dX.  This candidate
    keeps an FP32 ``dP`` accumulator while visiting the D slabs and calls the
    established pull-only primitive exactly once.  Because the accumulator
    is N*K, it is guarded by the same explicit High-D budget and is never an
    implicit replacement for the bounded native adapter.
    """

    if plan is None:
        plan = choose_highd_backward_plan(
            int(pulled.shape[0]), int(pulled.shape[1]), int(weight.shape[1]),
            int(threads), bool(compute_dx))
    n, k = map(int, pulled.shape)
    if not compute_dx:
        # Without dX there is no sparse pull to deduplicate; retain the
        # ordinary stream so dW numerical behavior is unchanged.
        dx, dw, _ = streamed_aggregate_backward(
            pulled, weight, grad, scale, rowptr, colidx, threads,
            compute_dx, plan=plan)
        return dx, dw
    kp = _round_up(k, 64)
    # Do not re-resolve a possibly different environment budget after the
    # canonical planner selected this branch.  Direct callers without a plan
    # retain the historical environment-controlled behavior.
    budget = (_plan_budget_bytes(plan) if plan is not None else
              _positive_env("TFS_HIGHD_BWD_BUDGET_BYTES", 64 << 20))
    dp_bytes = int(n * kp * 4)
    workspace_bytes = int(getattr(plan, "workspace_bytes", 0))
    accounted = (workspace_bytes if plan.execution_variant ==
                 "aggregate_highd_single_scan" else
                 workspace_bytes + dp_bytes)
    if accounted > budget:
        raise RuntimeError(
            "aggregate single-scan dP accumulator exceeds the configured "
            "High-D workspace budget")
    dx, dw, _ = streamed_aggregate_backward(
        pulled, weight, grad, scale, rowptr, colidx, threads,
        compute_dx, plan=plan)
    return dx, dw


__all__ = [
    "HighDBackwardPlan",
    "choose_highd_backward_plan",
    "highd_plan_from_execution_plan",
    "highd_stream_enabled",
    "should_use_stream",
    "streamed_aggregate_backward",
    "native_transform_highd_backward",
    "native_transform_highd_single_scan_backward",
    "streamed_transform_backward",
    "native_aggregate_d_slab_backward",
    "streamed_aggregate_single_scan_backward",
]
