"""Shape- and runtime-aware execution planning for the standard TFS path.

The project historically kept the order heuristic and the cache/AMX switches
inside each benchmark script.  That made a 32-thread V3 run and a 1--16
thread V1 run look like the same implementation even though they were not.
This module is the single, side-effect-free source of truth for selecting a
layer's execution contract.  It deliberately keeps the V1 order rule
(``aggregate`` when ``D >= K`` and ``transform`` otherwise) so adopting the
planner does not silently change an existing experiment.

The planner only describes a path that the caller can actually dispatch.  The
canonical launcher enables the generic aggregate-saved kernel with
``TFS_AGGREGATE_SAVED=on`` after its native gate; ``auto`` remains available
as an explicit compatibility/ablation mode and keeps the established
static-cache behavior.  This distinction is important for fair reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from typing import Any, Dict, Optional, Tuple


_MODES = {"auto", "off", "on"}
_MAX_AUTHORITY_THREADS = 32


def _mode(name: str, default: str = "auto") -> str:
    """Read an ``auto|off|on`` environment switch with a useful error."""

    value = os.environ.get(name, default).strip().lower()
    # Accept the common boolean spellings used by old launch scripts while
    # emitting the canonical mode to the plan/log.
    if value in {"1", "true", "yes", "enable", "enabled"}:
        value = "on"
    elif value in {"0", "false", "no", "disable", "disabled"}:
        value = "off"
    if value not in _MODES:
        raise ValueError(f"{name} must be auto|off|on, got {value!r}")
    return value


def _enabled(mode: str, auto: bool) -> bool:
    return auto if mode == "auto" else mode == "on"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "off", "no"}


def _positive_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _panel_working_set_bytes(row_panel: int, kp: int, dp: int, k: int,
                             d: int, compute_dx: bool,
                             native_scale: bool = True) -> int:
    """Conservative live panel estimate used by the canonical planner.

    The estimate intentionally excludes the persistent output ``dW``.  The
    separate ``TFS_HIGHD_PANEL_BUDGET_BYTES`` contract bounds these short-lived
    buffers, while the High-D workspace budget bounds the per-thread dW slab.
    Keeping the budgets separate avoids changing the already validated
    128->2983 full-D path merely because a panel is live at the same time.
    """

    row_panel = max(1, int(row_panel))
    bytes_live = row_panel * int(kp) * 2       # pulled/P panel (BF16)
    bytes_live += row_panel * int(dp) * 2      # scaled gradient (BF16)
    bytes_live += row_panel * int(d) * (2 if native_scale else 6)
    if compute_dx:
        bytes_live += row_panel * int(k) * 4 * 2  # dP + dense result
    return int(bytes_live)


def _positive_env_or_default(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return int(default)
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _highd_workspace_budget(threads: int) -> int:
    """Return the locked, shape-agnostic High-D workspace budget.

    The compatibility variable remains the base budget.  The per-thread
    floor prevents a higher thread count from losing a viable native path
    solely because its worker-local dW storage is replicated.
    """

    legacy = os.environ.get("TFS_HIGHD_BWD_BUDGET_BYTES")
    if legacy is not None and legacy.strip() != "":
        return _nonnegative_int("TFS_HIGHD_BWD_BUDGET_BYTES", 64 << 20)
    base = _nonnegative_int(
        "TFS_HIGHD_BWD_BASE_BUDGET_BYTES",
        64 << 20)
    per_thread = _nonnegative_int(
        "TFS_HIGHD_BWD_PER_THREAD_BUDGET_BYTES", 4 << 20)
    return max(int(base), int(threads) * int(per_thread))


def _round_up(value: int, multiple: int) -> int:
    return ((int(value) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _choose_panel(n: int, k: int, d: int, kp: int, dp: int,
                  compute_dx: bool, requested: int) -> Tuple[int, int, int]:
    """Choose the largest practical row panel within the High-D panel budget."""

    requested = max(1, min(int(n), int(requested)))
    # Small-D kernels already have a bounded native panel contract.  Do not
    # perturb their historical caller-selected panel.
    if int(d) <= 128:
        return requested, 0, 0

    budget = _positive_env_or_default(
        "TFS_HIGHD_PANEL_BUDGET_BYTES", 32 << 20)
    candidates = [min(requested, c) for c in (1024, 512, 256, 128, 64)]
    candidates = list(dict.fromkeys(c for c in candidates if c >= 1))
    for candidate in candidates:
        estimate = _panel_working_set_bytes(
            candidate, kp, dp, k, d, compute_dx, native_scale=True)
        if estimate <= budget:
            return int(candidate), int(estimate), int(budget)

    # Extremely small ablation budgets should still have a deterministic,
    # bounded result rather than silently exceeding the contract.
    per_row = _panel_working_set_bytes(
        1, kp, dp, k, d, compute_dx, native_scale=True)
    panel = max(1, min(requested, budget // max(1, per_row)))
    return int(panel), int(_panel_working_set_bytes(
        panel, kp, dp, k, d, compute_dx, native_scale=True)), int(budget)


def partition_d(d: int, max_width: int, min_native: int = 129) -> tuple[tuple[int, int], ...]:
    """Partition a High-D output into strict native slabs.

    Every multi-slab partition has ``min_native <= width <= max_width``.  If
    such a partition does not exist (for example D=200,T=129), one slab is
    retained; its actual width is then charged to the workspace gate.  This
    is preferable to the old tail merge, which could silently turn a 256-wide
    request into a 332-wide physical slab.
    """

    d, max_width, min_native = map(int, (d, max_width, min_native))
    if d <= 0 or max_width <= 0 or min_native <= 0:
        raise ValueError("d, max_width and min_native must be positive")
    if d <= max_width or max_width < min_native:
        return ((0, d),)
    count = int(math.ceil(d / float(max_width)))
    while count > 1 and d // count < min_native:
        count -= 1
    if count <= 1:
        return ((0, d),)
    base, rem = divmod(d, count)
    widths = [base + (1 if i < rem else 0) for i in range(count)]
    if min(widths) < min_native or max(widths) > max_width:
        raise AssertionError("partition_d produced an invalid slab")
    ranges = []
    offset = 0
    for width in widths:
        ranges.append((offset, offset + width))
        offset += width
    return tuple(ranges)


@dataclass(frozen=True, init=False)
class KernelCandidate:
    """One implementation candidate considered by the planner."""

    name: str
    supported: bool
    workspace_bytes: int
    estimated_cost: float
    reason: str = ""
    # Structural cost terms are recorded even when the initial authority
    # planner only uses ``estimated_cost``.  Keeping them on the candidate
    # makes sparse-scan/launch tradeoffs auditable without changing dispatch.
    sparse_scans: int = 0
    launch_count: int = 0
    performance_state: str = "experimental"
    validated_for_auto: bool = False

    # ``dataclass(kw_only=True)`` would express this directly, but the
    # authoritative cluster still runs Python 3.9.  Keep the same contract
    # with a small hand-written initializer so a positional argument can
    # never silently bind to the wrong field on either Python 3.9 or 3.13.
    def __init__(self, *, name: str, supported: bool,
                 workspace_bytes: int, estimated_cost: float,
                 reason: str = "", sparse_scans: int = 0,
                 launch_count: int = 0,
                 performance_state: str = "experimental",
                 validated_for_auto: bool = False) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "supported", supported)
        object.__setattr__(self, "workspace_bytes", workspace_bytes)
        object.__setattr__(self, "estimated_cost", estimated_cost)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "sparse_scans", int(sparse_scans))
        object.__setattr__(self, "launch_count", int(launch_count))
        object.__setattr__(self, "performance_state", str(performance_state))
        object.__setattr__(self, "validated_for_auto",
                           bool(validated_for_auto))


def _sparse_backward_contract(variant: str, k: int, d: int,
                              kp: int, dp: int) -> tuple[str, int, int]:
    """Return the actual backward CSR operand, logical width and stride."""
    if variant == "aggregate_static_v3":
        return "none", 0, 0
    if variant in {"native_c3", "native_wide_k", "aggregate_saved_v4"}:
        return "Gs", d, dp
    if variant.startswith("transform_highd_"):
        return "Gs", d, dp
    if variant.startswith("aggregate_highd_"):
        return "dP", k, kp
    raise ValueError(f"unknown execution variant {variant!r}")


@dataclass(frozen=True)
class LayerExecutionPlan:
    """Serializable contract chosen for one graph-convolution layer.

    ``n`` is the node count, ``k`` the input feature width and ``d`` the
    output width.  ``kp``/``dp`` are the AMX physical widths.  The boolean
    fields describe capabilities/gates and are consumed by the Python
    dispatchers; they are also emitted in the mandatory ``TFS_PLAN`` line.
    """

    n: int
    k: int
    d: int
    kp: int
    dp: int
    panel: int
    d_tile: int
    order: str
    execution_variant: str
    compute_dx: bool
    small_single_scan: bool
    active_row: bool
    static_hs: bool
    static_pulled: bool
    use_int32_colidx: bool
    local_dw_budget: int
    reason: str
    # The fields below make the planner's data-flow and High-D workspace
    # decision explicit.  They are kept in the same immutable record as the
    # order rule so dispatchers cannot silently derive a second, conflicting
    # plan from the dataset name or a legacy environment switch.
    forward_family: str = ""
    sparse_fwd_width: int = 0
    sparse_bwd_width: int = 0
    sparse_fwd_tensor: str = "Hs"
    sparse_bwd_tensor: str = "dP"
    sparse_fwd_width_logical: int = 0
    sparse_fwd_width_physical: int = 0
    sparse_bwd_width_logical: int = 0
    sparse_bwd_width_physical: int = 0
    layer_index: int = 0
    layer_path: str = ""
    dtype_contract: str = "bf16_inputs_fp32_accum_fp32_master"
    workspace_strategy: str = "not_applicable"
    workspace_bytes: int = 0
    local_dw_bytes: int = 0
    native_supported: bool = False
    native_d_slab_supported: bool = False
    fallback_reason: str = ""
    numa_hint: str = ""
    threads: int = 1
    # v2.2 makes implementation selection explicit instead of deriving it
    # indirectly from full-D/d-slab booleans.  These fields are immutable and
    # are consumed by both forward and backward dispatch.
    candidates: tuple[KernelCandidate, ...] = ()
    selection_policy: str = "auto"
    # Keep dense AMX blocking separate from the maximum width of one sparse
    # traversal.  ``d_tile`` remains the compatibility alias consumed by old
    # adapters and is equal to ``dense_d_tile`` in plans produced here.
    dense_d_tile: int = 0
    sparse_d_slab: int = 0
    d_slabs: tuple[tuple[int, int], ...] = ()
    dslab_max_width: int = 0
    panel_working_set_bytes: int = 0
    panel_budget_bytes: int = 0
    workspace_budget_bytes: int = 0

    @property
    def plan_id(self) -> str:
        """Stable identity for this shape/environment execution contract."""

        payload = asdict(self)
        # ``plan_id`` is a property, so it cannot recurse into itself.  Keep
        # the digest short enough for logs while retaining collision margin.
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:16]

    @property
    def plan_hash(self) -> str:
        """Compatibility name for the immutable plan hash.

        ``plan_id`` is retained as the canonical log key used by existing
        reports; both names intentionally resolve to the same digest.
        """

        return self.plan_id

    @property
    def selected_impl(self) -> str:
        """Read-only compatibility alias for old reports.

        Production code must dispatch on :attr:`execution_variant`.  Keeping
        this alias as a property (rather than a dataclass field) prevents an
        old caller from constructing a contradictory second decision.
        """

        return self.execution_variant

    @property
    def backward_impl(self) -> str:
        if self.execution_variant == "aggregate_static_v3":
            return "aggregate_saved_v3"
        if self.execution_variant == "aggregate_saved_v4":
            return "aggregate_saved_v4"
        if self.execution_variant.startswith("aggregate_highd_"):
            return "aggregate_saved_wide"
        return "selective_amx"

    @property
    def backward_mode(self) -> str:
        """Read-only compatibility name derived from the sole variant."""

        return self.backward_impl

    @property
    def save_pulled(self) -> bool:
        return (
            self.execution_variant in {
                "aggregate_static_v3", "aggregate_saved_v4"
            } or self.execution_variant.startswith("aggregate_highd_")
        )

    @property
    def save_hs(self) -> bool:
        return not self.save_pulled

    @property
    def layer0_path(self) -> str:
        """Compatibility view without duplicating stack-level metadata."""

        return self.layer_path if self.layer_index == 0 else ""

    @property
    def layer1_path(self) -> str:
        return self.layer_path if self.layer_index == 1 else ""

    @property
    def layer1_backward(self) -> str:
        return self.backward_impl if self.layer_index == 1 else ""

    def assert_dispatch(self, actual_impl: str) -> None:
        """Fail fast if production dispatch diverges from the plan."""

        if self.execution_variant != str(actual_impl):
            raise RuntimeError(
                "TFS plan/dispatch mismatch: "
                f"plan_id={self.plan_id} planned={self.execution_variant} "
                f"actual={actual_impl}"
            )

    def validate(self) -> None:
        """Validate invariants that must hold for an authority plan."""

        if self.kp < self.k or self.kp % 64:
            raise ValueError("invalid padded K in execution plan")
        if self.dp < self.d or self.dp % 32:
            raise ValueError("invalid padded D in execution plan")
        if self.order not in {"aggregate", "transform"}:
            raise ValueError(f"invalid order {self.order!r}")
        expected_order = "aggregate" if self.d >= self.k else "transform"
        if self.order != expected_order:
            raise ValueError("execution plan order violates the shape rule")
        sparse_logical = self.k if self.order == "aggregate" else self.d
        sparse_physical = self.kp if self.order == "aggregate" else self.dp
        sparse_fwd_tensor = "Hs" if self.order == "aggregate" else "T"
        sparse_bwd_tensor, sparse_bwd_logical, sparse_bwd_physical = (
            _sparse_backward_contract(
                self.execution_variant, self.k, self.d, self.kp, self.dp))
        if self.sparse_fwd_width != sparse_physical:
            raise ValueError("forward sparse width does not match order")
        if self.sparse_bwd_width != sparse_bwd_physical:
            raise ValueError("backward sparse width does not match order")
        if self.sparse_fwd_tensor != sparse_fwd_tensor:
            raise ValueError("forward sparse tensor does not match order")
        if self.sparse_bwd_tensor != sparse_bwd_tensor:
            raise ValueError("backward sparse tensor does not match order")
        if self.sparse_fwd_width_logical != sparse_logical:
            raise ValueError("forward logical sparse width is inconsistent")
        if self.sparse_bwd_width_logical != sparse_bwd_logical:
            raise ValueError("backward logical sparse width is inconsistent")
        if self.sparse_fwd_width_physical != sparse_physical:
            raise ValueError("forward physical sparse width is inconsistent")
        if self.sparse_bwd_width_physical != sparse_bwd_physical:
            raise ValueError("backward physical sparse width is inconsistent")
        if self.dense_d_tile and self.dense_d_tile != self.d_tile:
            raise ValueError("dense_d_tile and compatibility d_tile disagree")
        if self.d > 128 and self.sparse_d_slab <= 0:
            raise ValueError("High-D plan must expose sparse_d_slab")
        if not self.candidates:
            raise ValueError("execution plan must expose candidates")
        for candidate in self.candidates:
            if not isinstance(candidate, KernelCandidate):
                raise TypeError("execution plan candidate has an invalid type")
            if not isinstance(candidate.workspace_bytes, int):
                raise TypeError("candidate workspace_bytes must be int")
            if not isinstance(candidate.estimated_cost, float):
                raise TypeError("candidate estimated_cost must be float")
            if not isinstance(candidate.reason, str):
                raise TypeError("candidate reason must be str")
            if not isinstance(candidate.sparse_scans, int):
                raise TypeError("candidate sparse_scans must be int")
            if not isinstance(candidate.launch_count, int):
                raise TypeError("candidate launch_count must be int")
            if candidate.performance_state not in {
                    "experimental", "validated", "authority", "deprecated"}:
                raise ValueError("candidate has an invalid performance state")
            if not isinstance(candidate.validated_for_auto, bool):
                raise TypeError("candidate validated_for_auto must be bool")
        selected = [candidate for candidate in self.candidates
                    if candidate.name == self.execution_variant]
        if not selected or not any(candidate.supported for candidate in selected):
            raise ValueError("selected implementation is not a supported candidate")
        if self.selection_policy not in {"auto", "explicit", "fallback"}:
            raise ValueError("invalid execution selection policy")
        if (self.selection_policy == "auto" and
                not any(candidate.supported and candidate.validated_for_auto
                        for candidate in selected)):
            raise ValueError(
                "auto selected an implementation without a performance gate")
        if not self.d_slabs:
            raise ValueError("execution plan must contain at least one D slab")
        offset = 0
        for d0, d1 in self.d_slabs:
            if d0 != offset or d1 <= d0 or d1 > self.d:
                raise ValueError("D slabs are not contiguous or in bounds")
            offset = d1
        if offset != self.d:
            raise ValueError("D slabs do not cover the logical output width")
        if self.workspace_strategy == "d_slab" and self.d <= 128:
            raise ValueError("small-D layer cannot use d-slab strategy")
        if self.native_d_slab_supported:
            if self.order != "aggregate" or self.d <= 128:
                raise ValueError("native d-slab gate is only aggregate High-D")
            if any((d1 - d0) <= 128 for d0, d1 in self.d_slabs):
                raise ValueError("native d-slab contains an unsupported tail")
        # Native candidates are charged against the per-thread High-D
        # workspace budget.  Reference streams intentionally do not allocate
        # that dW slab; they are charged against the bounded live-panel
        # budget instead.  Applying the native budget to a reference fallback
        # would make an explicit low-memory fallback fail validation rather
        # than remain usable.
        reference_impl = self.execution_variant in {
            "transform_highd_stream", "transform_highd_legacy",
            "aggregate_highd_reference"
        }
        if reference_impl:
            if (self.panel_budget_bytes and
                    self.panel_working_set_bytes > self.panel_budget_bytes):
                raise ValueError("selected reference panel exceeds the planner budget")
        elif (self.workspace_budget_bytes and
              self.workspace_bytes > self.workspace_budget_bytes):
            raise ValueError("selected workspace exceeds the planner budget")
        if self.panel_budget_bytes and self.panel_working_set_bytes > self.panel_budget_bytes:
            raise ValueError("selected row panel exceeds the planner budget")

    @property
    def dimension_path(self) -> str:
        """Return the dimension-specialized AMX family for this layer.

        The path is deliberately derived from the shape rather than from a
        launch-time opt-in.  ``wide_k`` covers high-input-width layers such
        as IGB's 1024 -> 128 first layer; ``wide_output`` and
        ``wide_aggregate`` cover the two high-output-width orders.  Keeping
        this decision in the planner prevents individual benchmark scripts
        from silently falling back merely because ``K`` is larger than the
        historical 128-wide C3 fixture.
        """

        if self.d > 128:
            return "wide_aggregate" if self.order == "aggregate" else "wide_output"
        if self.k > 128:
            return "wide_k"
        return "c3"

    @property
    def wide_k(self) -> bool:
        """Whether this layer needs the high-input-width AMX contract."""

        return self.dimension_path == "wide_k"

    @property
    def wide_output(self) -> bool:
        """Whether this layer uses transform-first wide-output tiling."""

        return self.dimension_path == "wide_output"

    @property
    def wide_aggregate(self) -> bool:
        """Whether this layer uses aggregate-first wide-output tiling."""

        return self.dimension_path == "wide_aggregate"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["selected_impl"] = self.selected_impl
        payload["backward_mode"] = self.backward_mode
        payload["save_hs"] = self.save_hs
        payload["save_pulled"] = self.save_pulled
        payload["dimension_path"] = self.dimension_path
        payload["plan_id"] = self.plan_id
        payload["plan_hash"] = self.plan_hash
        return payload

    def log_line(self, layer: Optional[int] = None) -> str:
        """Return the stable machine-readable plan log required by r5."""

        prefix = "TFS_PLAN"
        if layer is not None:
            prefix += f" layer={int(layer)}"
        fields = (
            f"K={self.k}",
            f"D={self.d}",
            f"order={self.order}",
            f"execution_variant={self.execution_variant}",
            f"backward={self.backward_mode}",
            f"compute_dx={int(self.compute_dx)}",
            f"forward_family={self.forward_family or self.order}",
            f"sparse_fwd_width={self.sparse_fwd_width or self.kp}",
            f"sparse_bwd_width={self.sparse_bwd_width}",
            f"sparse_fwd_tensor={self.sparse_fwd_tensor}",
            f"sparse_fwd_width_logical={self.sparse_fwd_width_logical or (self.k if self.order == 'aggregate' else self.d)}",
            f"sparse_fwd_width_physical={self.sparse_fwd_width_physical or (self.kp if self.order == 'aggregate' else self.dp)}",
            f"sparse_bwd_width_logical={self.sparse_bwd_width_logical}",
            f"sparse_bwd_width_physical={self.sparse_bwd_width_physical}",
            f"save_hs={int(self.save_hs)}",
            f"save_pulled={int(self.save_pulled)}",
            f"static_hs={int(self.static_hs)}",
            f"static_pulled={int(self.static_pulled)}",
            f"single_scan={int(self.small_single_scan)}",
            f"active_row={int(self.active_row)}",
            f"colidx=int32:{int(self.use_int32_colidx)}",
            f"dim_path={self.dimension_path}",
            f"panel={self.panel}",
            f"d_tile={self.d_tile}",
            f"workspace={self.workspace_strategy}",
            f"workspace_bytes={self.workspace_bytes}",
            f"workspace_budget={self.workspace_budget_bytes}",
            f"local_dw_bytes={self.local_dw_bytes}",
            f"panel_ws={self.panel_working_set_bytes}",
            f"panel_budget={self.panel_budget_bytes}",
            f"selected={self.selected_impl or 'unspecified'}",
            f"selection_policy={self.selection_policy}",
            f"dense_d_tile={self.dense_d_tile or self.d_tile}",
            f"sparse_d_slab={self.sparse_d_slab or self.d_tile}",
            f"d_slabs={len(self.d_slabs)}",
            f"dslab_max={self.dslab_max_width}",
            f"native_supported={int(self.native_supported)}",
            f"native_dslab_supported={int(self.native_d_slab_supported)}",
            f"layer_index={self.layer_index}",
            f"layer_path={self.layer_path or 'dynamic'}",
            f"dtype={self.dtype_contract}",
            f"sparse_bwd_tensor={self.sparse_bwd_tensor}",
            f"plan_id={self.plan_id}",
            f"plan_hash={self.plan_hash}",
            f"fallback={self.fallback_reason or 'none'}",
            f"numa={self.numa_hint or 'default'}",
        )
        return prefix + " " + " ".join(fields)


def planner_enabled() -> bool:
    """Whether callers should use the standard planner.

    The formal r5 contract defaults to enabled.  Setting
    ``TFS_EXEC_PLANNER=0`` is an explicit compatibility escape hatch for old
    reproduction scripts; it is included in the reason field by callers that
    choose to fall back to their legacy path.
    """

    return _env_flag("TFS_EXEC_PLANNER", True)


def build_layer_plan(
    n: int,
    k: int,
    d: int,
    *,
    compute_dx: bool = True,
    input_static: bool = False,
    graph_static: bool = True,
    feature_static: bool = True,
    layer: int = 0,
    panel: int = 512,
    d_tile: int = 128,
    colidx_max: Optional[int] = None,
    threads: int = 1,
) -> LayerExecutionPlan:
    """Build a behavior-equivalent V1 plan with formal r5 gates.

    The function has no graph/device side effects and is therefore safe to
    call during model construction.  ``input_static`` is intentionally
    supplied by the model: only a first layer fed by a fixed full-graph
    feature matrix may use the persistent Hs/T0 cache.
    """

    n, k, d, threads = int(n), int(k), int(d), int(threads)
    if n < 1 or k < 1 or d < 1:
        raise ValueError(f"invalid layer shape n={n}, k={k}, d={d}")
    if panel < 1 or d_tile < 1 or threads < 1:
        raise ValueError("panel and d_tile must be positive")
    if threads > _MAX_AUTHORITY_THREADS:
        raise ValueError(
            f"threads must be in [1, {_MAX_AUTHORITY_THREADS}] for the "
            "current native authority contract")

    kp = int(math.ceil(k / 64.0) * 64)
    dp = int(math.ceil(d / 32.0) * 32)
    # V1's order is deliberately dimension based and dataset agnostic.
    order = "aggregate" if d >= k else "transform"

    single_mode = _mode("TFS_SMALL_SINGLE_SCAN")
    active_mode = _mode("TFS_ACTIVE_ROW")
    saved_mode = _mode("TFS_AGGREGATE_SAVED")
    hs_mode = _mode("TFS_STATIC_HS")
    aggregate_mode = _mode("TFS_STATIC_AGGREGATE")
    transform_mode = _mode("TFS_TRANSFORM_HIGHD_STREAM_V1")
    native_transform_mode = _mode("TFS_TRANSFORM_HIGHD_NATIVE")
    single_scan_mode = _mode("TFS_TRANSFORM_HIGHD_SINGLE_SCAN", "off")
    aggregate_single_scan_mode = _mode(
        "TFS_AGGREGATE_DSLAB_SINGLE_SCAN", "off")
    colidx_mode = os.environ.get("TFS_COLIDX", "auto").strip().lower()
    if colidx_mode not in {"auto", "int32", "int64"}:
        raise ValueError("TFS_COLIDX must be auto|int32|int64")
    # The plan uses the document's final name; retain the shorter alias used
    # by an early r5 draft so old launch files remain reproducible.
    local_dw_budget = _nonnegative_int(
        "TFS_MAX_LOCAL_DW_BYTES",
        _nonnegative_int("TFS_LOCAL_DW_BUDGET_BYTES", 0),
    )

    # The native path now has one generic small-D single-scan loop and one
    # generic active-row gate for D<=128.  ``auto`` therefore enables them for
    # the complete supported width range; ``off`` remains available for an
    # apples-to-apples regression run.
    small_single_auto = d <= 128
    active_auto = d <= 128 and bool(compute_dx is not None)
    small_single = _enabled(single_mode, small_single_auto)
    active_row = _enabled(active_mode, active_auto)

    static_candidate = bool(input_static and graph_static and feature_static)
    # ``on`` means enable the cache when the validity contract is satisfied;
    # it must not override the layer-0/input-static eligibility check.
    static_hs = bool(static_candidate and _enabled(hs_mode, static_candidate))
    # The existing V3 T0 path is an opt-in implementation and cannot provide
    # dX.  Never select it for a layer whose input gradient is required.
    static_aggregate_requested = _enabled(
        aggregate_mode,
        os.environ.get("HYBRID_STATIC_AGG_CACHE", "0") == "1",
    )
    static_v3_supported = bool(k <= 128 and d <= 128)
    static_pulled = bool(
        static_candidate and static_hs and static_aggregate_requested and
        static_v3_supported and order == "aggregate" and not compute_dx and
        os.environ.get("HYBRID_AMX_FORWARD", "0") == "1" and
        os.environ.get("HYBRID_AMX_BACKWARD", "0") == "1"
    )

    # The accepted wide-D aggregate wrapper already returns and saves the
    # pulled BF16 tensor.  Treat it as an aggregate-saved path as well; this
    # is what makes IGB's 128 -> 2983 layer show the same contract as the
    # static T0 layer without pretending that a new generic kernel exists.
    wide_aggregate_saved = bool(order == "aggregate" and d > 128)
    # ``auto`` retains the established static-T0 behavior.  ``on`` additionally
    # opts into the generic per-forward saved-pulled contract for every
    # aggregate-first D<=128 layer, including hidden layers whose input is not
    # static.  Keeping auto conservative avoids silently changing old timing
    # matrices until the new native numerical gate has passed.
    aggregate_saved_requested = _enabled(saved_mode, static_pulled)
    amx_contract = (
        os.environ.get("HYBRID_AMX_FORWARD", "0") == "1" and
        os.environ.get("HYBRID_AMX_BACKWARD", "0") == "1"
    )
    generic_aggregate_saved = bool(
        order == "aggregate" and d <= 128 and not static_pulled and
        saved_mode == "on" and amx_contract
    )
    # The C++ path still accepts int64 CSR by default.  This field is a plan
    # contract only; the native conversion is enabled separately when the
    # formal TFS_COLIDX switch is set to int32/auto in a compatible build.
    max_col = n - 1 if colidx_max is None else int(colidx_max)
    if max_col < 0:
        raise ValueError("colidx_max must be non-negative")
    use_int32_colidx = (
        colidx_mode == "int32" or
        (colidx_mode == "auto" and max_col < (1 << 31))
    )

    # Derive the physical workspace contract once, using only shape, thread
    # count and explicit budgets.  Both forward and backward receive this
    # immutable record; no autograd callback is allowed to make a second
    # independent tile/budget decision.
    highd_budget = _highd_workspace_budget(threads)
    native_transform_gs_budget = _positive_env_or_default(
        "TFS_TRANSFORM_HIGHD_NATIVE_GS_BUDGET_BYTES", 256 << 20)
    local_dw_bytes = int(threads * kp * dp * 4)
    logical_dw_bytes = int(k * d * 4)
    highd_shape = bool(d > 128)
    requested_tile = os.environ.get("TFS_HIGHD_BWD_D_TILE")
    if requested_tile:
        requested_d_tile = min(d, _positive_int("TFS_HIGHD_BWD_D_TILE", 256))
    elif highd_shape:
        requested_d_tile = min(d, 256)
    else:
        requested_d_tile = int(d_tile)

    if highd_shape:
        native_transform_supported = False
        native_transform_enabled = False
        native_transform_single_scan_supported = False
        native_transform_single_scan_enabled = False
        # A full-D candidate is independently evaluated from the d-slab
        # candidate.  This prevents increasing the budget from accidentally
        # selecting a framework fallback merely because a different gate
        # became eligible.
        full_d_workspace = local_dw_bytes
        dslab_limit = max(129, int(requested_d_tile))
        d_slabs = partition_d(d, dslab_limit, min_native=129)
        dslab_max_width = max(d1 - d0 for d0, d1 in d_slabs)
        dslab_workspace = int(
            threads * kp * _round_up(dslab_max_width, 32) * 4)
        full_supported = bool(
            order == "aggregate" and k <= 256 and
            full_d_workspace <= highd_budget and
            logical_dw_bytes <= (64 << 20)
        )
        dslab_supported = bool(
            order == "aggregate" and d > 128 and
            all((d1 - d0) > 128 for d0, d1 in d_slabs) and
            dslab_workspace <= highd_budget
        )
        # The streamed aggregate d-slab candidate accumulates dP in FP32 and
        # performs one final CSR pull.  It is useful only when that persistent
        # dP buffer plus the bounded dW workspace fit the same explicit
        # budget; otherwise the native repeated-pull adapter remains the
        # memory-safe choice.
        aggregate_single_scan_dP = int(n * kp * 4)
        aggregate_single_scan_workspace = int(
            dslab_workspace + aggregate_single_scan_dP)
        # This is an explicit reference path, not an independently optimized
        # native kernel: it accumulates dP across slabs then performs one
        # final CSR pull.  Preserve it for correctness/data-flow probes, but
        # never expose it to automatic selection or performance claims.
        aggregate_single_scan_supported = bool(
            order == "aggregate" and compute_dx and
            aggregate_single_scan_workspace <= highd_budget)
        aggregate_single_scan_enabled = _enabled(
            aggregate_single_scan_mode, False)
        # The native Transform-HighD entry point is deliberately opt-in until
        # its repeated thread/shape gate is complete.  The gate is purely
        # dimensional and budget based; it never names a dataset.
        native_transform_supported = bool(
            order == "transform" and d > 128 and
            all((d1 - d0) > 128 for d0, d1 in d_slabs) and
            dslab_workspace <= highd_budget and
            n * _round_up(dslab_max_width, 32) * 2 <=
            native_transform_gs_budget
        )
        # Fused immediate-consume/single-scan candidate.  It uses the full
        # output width in the worker-local dW workspace, but does not allocate
        # an [N,D] BF16 Gs tensor.  Keep it opt-in until the complete
        # shape/thread non-regression matrix is accepted.
        native_transform_single_scan_supported = bool(
            order == "transform" and d > 128 and
            local_dw_bytes <= highd_budget and len(d_slabs) > 1
        )
        native_transform_single_scan_enabled = _enabled(
            single_scan_mode, False)
        if (order == "aggregate" and aggregate_single_scan_supported and
                aggregate_single_scan_enabled):
            workspace_strategy = "d_slab"
            workspace_bytes = aggregate_single_scan_workspace
            effective_d_tile = int(requested_d_tile)
            selected_impl = "aggregate_d_slab_single_scan"
        elif (order == "transform" and native_transform_single_scan_supported and
                native_transform_single_scan_enabled):
            workspace_strategy = "full_d"
            workspace_bytes = local_dw_bytes
            effective_d_tile = d
            selected_impl = "native_transform_single_scan"
        elif full_supported:
            workspace_strategy = "full_d"
            workspace_bytes = full_d_workspace
            effective_d_tile = d
            selected_impl = "native_full_d"
        elif dslab_supported:
            workspace_strategy = "d_slab"
            workspace_bytes = dslab_workspace
            effective_d_tile = int(requested_d_tile)
            selected_impl = "native_d_slab"
        else:
            # Transform High-D remains the correctness stream until its
            # native sparse-slab gate passes.  Aggregate shapes that cannot
            # fit either native candidate use the explicit reference stream.
            panel_count = (n + int(panel) - 1) // int(panel)
            # The native composite pays one callback per owned row panel and
            # has no advantage when only a handful of panels are available.
            # This deterministic cost gate is intentionally conservative:
            # require at least four workers, two panels per worker (and eight
            # panels overall), and enough K*D dense work to amortize AMX
            # setup.  ``on`` remains an explicit force for component probes.
            native_transform_auto = bool(
                False
            )
            native_transform_enabled = _enabled(
                native_transform_mode, native_transform_auto)
            # The streamed Transform-HighD path is currently validated as a
            # single sparse slab.  Multi-slab shapes repeat the CSR pull and
            # dense epilogue; repeated medians show a regression against the
            # established native 128-column wrapper (notably K=1024,D=513 at
            # 8/16/32 threads).  Keep those shapes available behind an
            # explicit ``TFS_TRANSFORM_HIGHD_STREAM_V1=on`` probe, but do not
            # let ``auto`` introduce a known performance regression.
            # The historical component measurements used unequal ATen/native
            # thread counts and too few panels.  Keep both Transform-HighD
            # paths explicit until Cycle 5's fair performance gate exists.
            transform_auto = False
            transform_enabled = _enabled(transform_mode, transform_auto)
            if (order == "transform" and transform_enabled and
                    native_transform_enabled and native_transform_supported):
                workspace_strategy = "d_slab"
                workspace_bytes = dslab_workspace
                effective_d_tile = int(requested_d_tile)
                selected_impl = "native_transform"
            else:
                workspace_strategy = (
                    "d_slab" if order == "transform" else
                    "full_d" if local_dw_bytes <= highd_budget and
                    logical_dw_bytes <= (64 << 20) else "d_slab")
                workspace_bytes = (local_dw_bytes
                                   if workspace_strategy == "full_d"
                                   else dslab_workspace)
                effective_d_tile = (d if order == "transform" or
                                    workspace_strategy == "full_d"
                                    else int(requested_d_tile))
                selected_impl = (
                    "transform_stream" if order == "transform" and transform_enabled
                    else "legacy_wide_output" if order == "transform"
                    else "python_stream"
                )
    else:
        workspace_strategy = "not_applicable"
        workspace_bytes = 0
        effective_d_tile = int(requested_d_tile)
        d_slabs = ((0, d),)
        dslab_max_width = d
        dslab_workspace = 0
        full_supported = False
        dslab_supported = False
        aggregate_single_scan_dP = 0
        aggregate_single_scan_workspace = 0
        aggregate_single_scan_supported = False
        aggregate_single_scan_enabled = False
        native_transform_supported = False
        native_transform_enabled = False
        native_transform_single_scan_supported = False
        native_transform_single_scan_enabled = False
        if static_pulled:
            selected_impl = "aggregate_static_v3"
        elif generic_aggregate_saved:
            selected_impl = "aggregate_saved_v4"
        else:
            selected_impl = "native_wide_k" if k > 128 else "native_c3"

    # Convert all historical implementation labels to one canonical enum.
    # Only this value is stored in the immutable plan; backward/cache fields
    # are derived properties and therefore cannot contradict dispatch.
    execution_variant = {
        "native_full_d": "aggregate_highd_full",
        "native_d_slab": "aggregate_highd_dslab",
        "aggregate_d_slab_single_scan": "aggregate_highd_single_scan",
        "transform_stream": "transform_highd_stream",
        "native_transform": "transform_highd_native",
        "native_transform_single_scan": "transform_highd_single_scan",
        "legacy_wide_output": "transform_highd_legacy",
        "python_stream": "aggregate_highd_reference",
    }.get(selected_impl, selected_impl)

    explicitly_selected = (
        execution_variant == "aggregate_saved_v4" or
        (execution_variant == "aggregate_highd_single_scan" and
         aggregate_single_scan_mode == "on") or
        (execution_variant == "transform_highd_single_scan" and
         single_scan_mode == "on") or
        (execution_variant == "transform_highd_native" and
         native_transform_mode == "on") or
        (execution_variant == "transform_highd_stream" and
         transform_mode == "on")
    )
    selection_policy = "explicit" if explicitly_selected else "auto"
    candidate_d_slabs = tuple(d_slabs)
    candidate_slab_count = len(candidate_d_slabs)

    effective_panel, panel_ws, panel_budget = _choose_panel(
        n, k, d, kp, dp, bool(compute_dx), int(panel))
    # Reference streams do not allocate the native per-thread dW slab.  Charge
    # their bounded live panel instead of reporting an infeasible native slab
    # size; native candidates retain the strict d-slab workspace accounting.
    if execution_variant in {
            "transform_highd_stream", "aggregate_highd_reference"}:
        workspace_bytes = int(panel_ws)
    if execution_variant == "transform_highd_single_scan":
        # The fused candidate traverses the complete logical D width in each
        # row panel; its plan therefore exposes one sparse traversal rather
        # than the speculative per-slab partition used by other candidates.
        d_slabs = ((0, d),)
        dslab_max_width = d
    # Expose the physical slabs of the selected implementation, not an
    # unused candidate's speculative partition.  The full-D and legacy paths
    # therefore report one logical slab; native d-slab and transform stream
    # retain the strict planner partition.
    if execution_variant not in {
            "aggregate_highd_dslab", "transform_highd_stream",
            "transform_highd_native", "transform_highd_single_scan"}:
        d_slabs = ((0, d),)
        dslab_max_width = d

    dense_flops = float(max(1, n) * max(1, k) * max(1, d))
    slab_scans = max(1, candidate_slab_count)
    dense_launches = max(1, (kp + 63) // 64) * max(1, (dp + 31) // 32)
    candidates = []
    if highd_shape:
        candidates.extend((
            KernelCandidate(
                name="aggregate_highd_full", supported=full_supported,
                workspace_bytes=int(full_d_workspace),
                estimated_cost=dense_flops / 1.0e9 + 1.0,
                reason="aggregate K<=256 and full workspace fits",
                sparse_scans=1, launch_count=dense_launches,
                performance_state="authority", validated_for_auto=True),
            KernelCandidate(
                name="aggregate_highd_dslab", supported=dslab_supported,
                workspace_bytes=int(dslab_workspace),
                estimated_cost=dense_flops / 1.0e9 + 2.0 + candidate_slab_count * 0.01,
                reason="strict slabs and bounded native workspace",
                sparse_scans=slab_scans, launch_count=dense_launches,
                performance_state="validated", validated_for_auto=True),
            KernelCandidate(
                name="aggregate_highd_single_scan",
                supported=aggregate_single_scan_supported,
                workspace_bytes=int(aggregate_single_scan_workspace),
                estimated_cost=dense_flops / 1.0e9 + 12.0,
                reason=("explicit reference: one final CSR pull; not an "
                        "independent native single-scan kernel"),
                sparse_scans=1, launch_count=dense_launches,
                performance_state="experimental", validated_for_auto=False),
            KernelCandidate(
                name="transform_highd_stream",
                supported=(order == "transform"),
                workspace_bytes=int(panel_ws),
                estimated_cost=dense_flops / 1.0e9 + 10.0,
                reason="correctness stream; native sparse slab pending",
                sparse_scans=slab_scans, launch_count=dense_launches,
                performance_state="experimental",
                validated_for_auto=False),
            KernelCandidate(
                name="transform_highd_native",
                supported=(order == "transform" and
                           native_transform_supported),
                workspace_bytes=int(dslab_workspace),
                estimated_cost=dense_flops / 1.0e9 + 3.0 + candidate_slab_count * 0.02,
                reason="explicit native Transform-HighD slab gate",
                sparse_scans=slab_scans, launch_count=dense_launches,
                performance_state="experimental",
                validated_for_auto=False),
            KernelCandidate(
                name="transform_highd_single_scan",
                supported=(order == "transform" and
                           native_transform_single_scan_supported),
                workspace_bytes=int(local_dw_bytes),
                estimated_cost=dense_flops / 1.0e9 + 2.5,
                reason="fused source-scale/pull with immediate dT consume",
                sparse_scans=1, launch_count=dense_launches,
                performance_state="experimental", validated_for_auto=False),
            KernelCandidate(
                name="transform_highd_legacy", supported=(order == "transform"),
                workspace_bytes=int(panel_ws),
                estimated_cost=dense_flops / 1.0e9 + 11.0,
                reason="accepted legacy transform-wide-output path",
                sparse_scans=slab_scans, launch_count=dense_launches,
                performance_state="authority", validated_for_auto=True),
            KernelCandidate(
                name="aggregate_highd_reference", supported=(order == "aggregate"),
                workspace_bytes=int(panel_ws),
                estimated_cost=dense_flops / 1.0e9 + 12.0,
                reason="explicit memory-bounded reference stream",
                sparse_scans=slab_scans, launch_count=dense_launches,
                performance_state="authority", validated_for_auto=True),
        ))
    else:
        candidates.append(KernelCandidate(
            name=execution_variant, supported=True, workspace_bytes=0,
            estimated_cost=dense_flops / 1.0e9 + 1.0,
            reason=("static aggregate V3" if static_pulled else
                    "explicit aggregate-saved V4" if generic_aggregate_saved else
                    "ordinary stride-parametric C3 contract"),
            sparse_scans=1, launch_count=dense_launches,
            performance_state=("experimental" if generic_aggregate_saved
                               else "authority"),
            validated_for_auto=not generic_aggregate_saved))

    # Forward sparse width follows the actual order (S(H) is K-wide for
    # aggregate-first, S(HW) is D-wide for transform-first).  The backward
    # sparse stage follows the same algebra: aggregate-first pulls dP at K
    # width, while transform-first pulls Gs to produce dT at D width.
    sparse_fwd_width = kp if order == "aggregate" else dp
    sparse_bwd_tensor, sparse_bwd_logical, sparse_bwd_width = (
        _sparse_backward_contract(execution_variant, k, d, kp, dp))
    sparse_width_logical = k if order == "aggregate" else d
    sparse_fwd_tensor = "Hs" if order == "aggregate" else "T"
    forward_family = "aggregate" if order == "aggregate" else "transform"
    layer_path = (
        "persistent_aggregate_t0" if execution_variant == "aggregate_static_v3" else
        "highd_native" if execution_variant == "aggregate_highd_full" else
        "highd_native_d_slab_slice" if execution_variant == "aggregate_highd_dslab" else
        "highd_aggregate_single_scan" if execution_variant == "aggregate_highd_single_scan" else
        "highd_transform_native" if execution_variant == "transform_highd_native" else
        "highd_transform_single_scan" if execution_variant == "transform_highd_single_scan" else
        "highd_d_slab" if highd_shape and order == "aggregate" else
        "highd_transform_d_slab" if highd_shape and order == "transform" else
        "dimension_native"
    )
    native_supported = bool((not highd_shape) or full_supported or
                            native_transform_supported or
                            native_transform_single_scan_supported)
    native_d_slab_supported = bool(dslab_supported)
    fallback_reason = ""
    if (highd_shape and order == "transform" and
            execution_variant not in {"transform_highd_native",
                                      "transform_highd_single_scan"}):
        fallback_reason = "transform_highd_native_pending"
    elif highd_shape and execution_variant == "aggregate_highd_dslab":
        fallback_reason = "native_d_slab_slice_adapter"
    elif (highd_shape and order == "aggregate" and
          execution_variant not in {"aggregate_highd_dslab",
                                    "aggregate_highd_single_scan",
                                    "aggregate_highd_full"} and
          not dslab_supported and not full_supported):
        fallback_reason = "native_candidates_exceed_budget"
    numa_hint = (
        "parallel_first_touch" if _env_flag("TFS_NUMA_FIRST_TOUCH", False)
        else "allocator_default"
    )

    reasons = [f"v1_order:{order}"]
    if k != kp:
        reasons.append("tail_k_padded")
    if d != dp:
        reasons.append("tail_d_padded")
    if static_hs:
        reasons.append("static_hs_candidate")
    if static_pulled:
        reasons.append("v3_static_aggregate")
    elif (static_candidate and static_hs and static_aggregate_requested and
          order == "aggregate" and not compute_dx and
          not static_v3_supported):
        reasons.append("static_v3_capability_k_or_d_exceeded")
    if wide_aggregate_saved:
        reasons.append("wide_aggregate_saved")
    if generic_aggregate_saved:
        reasons.append("generic_aggregate_saved_v4")
    if aggregate_saved_requested and not static_pulled and not wide_aggregate_saved and not generic_aggregate_saved:
        reasons.append("aggregate_saved_fallback_selective")
    if active_row:
        reasons.append("active_row_gate")
    if small_single:
        reasons.append("single_scan_gate")
    if use_int32_colidx:
        reasons.append("int32_colidx_candidate")
    if workspace_strategy != "not_applicable":
        reasons.append(f"workspace:{workspace_strategy}")
    reasons.append(f"selected:{execution_variant}")
    reasons.append(f"dslab_max:{dslab_max_width}")
    reasons.append(f"panel:{effective_panel}/{panel_budget or 'na'}")
    if fallback_reason:
        reasons.append(f"fallback:{fallback_reason}")
    reasons.append(f"dim_path:{('wide_aggregate' if (d > 128 and order == 'aggregate') else 'wide_output' if d > 128 else 'wide_k' if k > 128 else 'c3')}")

    plan = LayerExecutionPlan(
        n=n,
        k=k,
        d=d,
        kp=kp,
        dp=dp,
        panel=int(effective_panel),
        d_tile=int(effective_d_tile),
        order=order,
        execution_variant=execution_variant,
        compute_dx=bool(compute_dx),
        small_single_scan=small_single,
        active_row=active_row,
        static_hs=static_hs,
        static_pulled=static_pulled,
        use_int32_colidx=use_int32_colidx,
        local_dw_budget=local_dw_budget,
        reason=";".join(reasons),
        forward_family=forward_family,
        sparse_fwd_width=int(sparse_fwd_width),
        sparse_bwd_width=int(sparse_bwd_width),
        sparse_fwd_tensor=sparse_fwd_tensor,
        sparse_fwd_width_logical=int(sparse_width_logical),
        sparse_fwd_width_physical=int(sparse_fwd_width),
        sparse_bwd_width_logical=int(sparse_bwd_logical),
        sparse_bwd_width_physical=int(sparse_bwd_width),
        layer_index=int(layer),
        layer_path=layer_path,
        workspace_strategy=workspace_strategy,
        workspace_bytes=workspace_bytes,
        local_dw_bytes=local_dw_bytes,
        native_supported=native_supported,
        native_d_slab_supported=native_d_slab_supported,
        fallback_reason=fallback_reason,
        numa_hint=numa_hint,
        threads=threads,
        candidates=tuple(candidates),
        selection_policy=selection_policy,
        dense_d_tile=int(effective_d_tile),
        sparse_d_slab=(int(requested_d_tile) if highd_shape else 0),
        d_slabs=tuple(d_slabs),
        dslab_max_width=int(dslab_max_width),
        panel_working_set_bytes=int(panel_ws),
        panel_budget_bytes=int(panel_budget),
        workspace_budget_bytes=int(highd_budget if highd_shape else 0),
        sparse_bwd_tensor=sparse_bwd_tensor,
    )
    plan.validate()
    return plan


def plan_layers(
    n: int,
    dims: Any,
    *,
    graph_static: bool = True,
    feature_static_first: bool = True,
    threads: int = 1,
) -> list[LayerExecutionPlan]:
    """Plan a complete GCN stack using one consistent rule per layer."""

    values = [int(v) for v in dims]
    if len(values) < 2:
        raise ValueError("dims must contain input and output widths")
    plans = [
        build_layer_plan(
            n,
            values[i],
            values[i + 1],
            compute_dx=i > 0,
            input_static=feature_static_first and i == 0,
            graph_static=graph_static,
            feature_static=feature_static_first and i == 0,
            layer=i,
            threads=threads,
        )
        for i in range(len(values) - 1)
    ]
    return plans


def emit_plan_logs(plans: Any, *, stream: Any = None) -> None:
    """Print one stable ``TFS_PLAN`` line per layer (or write to a stream)."""

    if stream is None:
        import sys

        stream = sys.stdout
    print(runtime_contract_line(validate=True), file=stream, flush=True)
    contract_dir = os.environ.get("TFS_RUNTIME_CONTRACT_DIR", "").strip()
    if contract_dir:
        write_runtime_contract(os.path.join(contract_dir,
                                             "runtime_contract_resolved.json"))
    for i, plan in enumerate(plans):
        print(plan.log_line(layer=i), file=stream, flush=True)
        if _env_flag("TFS_EXPLAIN_PLAN", False):
            print("TFS_EXPLAIN_PLAN " + json.dumps(
                {"layer": i, **plan.to_dict()}, sort_keys=True,
                separators=(",", ":")), file=stream, flush=True)


def runtime_contract_payload() -> Dict[str, Any]:
    """Return the complete resolved authority contract as JSON data."""

    def value(name: str, default: str) -> str:
        return os.environ.get(name, default).strip().lower()

    runtime_threads = _positive_int("OMP_NUM_THREADS", 1)
    locked = {
        "TFS_MAX_AUTHORITY_THREADS": _MAX_AUTHORITY_THREADS,
        "TFS_EXEC_PLANNER": value("TFS_EXEC_PLANNER", "1"),
        "TFS_HIGHD_STREAM_V1": value("TFS_HIGHD_STREAM_V1", "0"),
        "TFS_HIGHD_NATIVE_STREAM": value("TFS_HIGHD_NATIVE_STREAM", "auto"),
        "TFS_TRANSFORM_HIGHD_STREAM_V1": value(
            "TFS_TRANSFORM_HIGHD_STREAM_V1", "auto"),
        # Record the resolved value used by the planner, rather than the
        # obsolete fixed default.  This is thread-sensitive unless an
        # explicit legacy override is present.
        "TFS_HIGHD_BWD_BUDGET_BYTES": _highd_workspace_budget(
            runtime_threads),
        "TFS_HIGHD_BWD_BASE_BUDGET_BYTES": _nonnegative_int(
            "TFS_HIGHD_BWD_BASE_BUDGET_BYTES", 64 << 20),
        "TFS_HIGHD_BWD_PER_THREAD_BUDGET_BYTES": _nonnegative_int(
            "TFS_HIGHD_BWD_PER_THREAD_BUDGET_BYTES", 4 << 20),
        "TFS_HIGHD_BWD_D_TILE": int(
            os.environ.get("TFS_HIGHD_BWD_D_TILE", 256)),
        "TFS_HIGHD_BWD_ROW_PANEL": int(
            os.environ.get("TFS_HIGHD_BWD_ROW_PANEL", 512)),
        "TFS_TRANSFORM_HIGHD_D_TILE": int(
            os.environ.get("TFS_TRANSFORM_HIGHD_D_TILE", 256)),
        "TFS_HIGHD_NATIVE_PANEL": int(
            os.environ.get("TFS_HIGHD_NATIVE_PANEL", 512)),
        "TFS_WORKSPACE_CACHE_MAX_BYTES": int(
            os.environ.get("TFS_WORKSPACE_CACHE_MAX_BYTES", 512 << 20)),
        "TFS_HIGHD_PANEL_BUDGET_BYTES": int(
            os.environ.get("TFS_HIGHD_PANEL_BUDGET_BYTES", 32 << 20)),
        "TFS_PULL_ONLY_BACKWARD": value("TFS_PULL_ONLY_BACKWARD", "0"),
        "TFS_SCALE_GRAD_BF16_NATIVE": value(
            "TFS_SCALE_GRAD_BF16_NATIVE", "0"),
        "TFS_STATIC_HS": value("TFS_STATIC_HS", "auto"),
        "TFS_SMALL_SINGLE_SCAN": value("TFS_SMALL_SINGLE_SCAN", "auto"),
        "TFS_ACTIVE_ROW": value("TFS_ACTIVE_ROW", "auto"),
        "TFS_AGGREGATE_SAVED": value("TFS_AGGREGATE_SAVED", "off"),
        "TFS_STATIC_AGGREGATE": value("TFS_STATIC_AGGREGATE", "off"),
        "TFS_NUMA_FIRST_TOUCH": value("TFS_NUMA_FIRST_TOUCH", "off"),
        "TFS_NUMA_PRIVATE": value("TFS_NUMA_PRIVATE", "auto"),
        "TFS_NUMA_REDUCE": value("TFS_NUMA_REDUCE", "auto"),
        "TFS_LOCALITY_SCHEDULE": value("TFS_LOCALITY_SCHEDULE", "off"),
        "TFS_HS_REPLICA": value("TFS_HS_REPLICA", "off"),
        "TFS_INTERNAL_PROFILE": value("TFS_INTERNAL_PROFILE", "0"),
        "TFS_PROFILE_NUMA_WORKSPACE": value(
            "TFS_PROFILE_NUMA_WORKSPACE", "0"),
        "TFS_HIGHD_NATIVE_SCALE": value("TFS_HIGHD_NATIVE_SCALE", "1"),
        "TFS_HIGHD_STREAM_AUTO": value("TFS_HIGHD_STREAM_AUTO", "1"),
        "HYBRID_STATIC_AGG_CACHE": value("HYBRID_STATIC_AGG_CACHE", "0"),
        "TFS_E13_ACTIVE_MAX_DENSITY": value(
            "TFS_E13_ACTIVE_MAX_DENSITY", "0.25"),
        "TFS_GLUE_E1_FUSED_DB": value("TFS_GLUE_E1_FUSED_DB", "0"),
        "TFS_GLUE_E2_VEC_STORE": value("TFS_GLUE_E2_VEC_STORE", "0"),
        "TFS_GLUE_E3_EMPTY_DX": value("TFS_GLUE_E3_EMPTY_DX", "0"),
        "TFS_GLUE_E4_FUSED_EPILOGUE": value(
            "TFS_GLUE_E4_FUSED_EPILOGUE", "0"),
        "TFS_GLUE_E5_VEC_HS": value("TFS_GLUE_E5_VEC_HS", "0"),
        "TFS_GLUE_E6_LOCAL_ZERO": value("TFS_GLUE_E6_LOCAL_ZERO", "0"),
        "TFS_GLUE_E6_PARALLEL_REDUCE": value(
            "TFS_GLUE_E6_PARALLEL_REDUCE", "0"),
        "TFS_GLUE_E6_SERIAL_WT": value("TFS_GLUE_E6_SERIAL_WT", "0"),
        "TFS_GLUE_E7_FORWARD_SCHEDULE": value(
            "TFS_GLUE_E7_FORWARD_SCHEDULE", "0"),
        "TFS_GLUE_E8_RIGHTSIZE_POOL": value(
            "TFS_GLUE_E8_RIGHTSIZE_POOL", "0"),
        "TFS_GLUE_E8_ATOMIC_DONE": value("TFS_GLUE_E8_ATOMIC_DONE", "0"),
        "TFS_GLUE_E8_SPIN": value("TFS_GLUE_E8_SPIN", "0"),
        "TFS_GLUE_E9_INT32_COLIDX": value(
            "TFS_GLUE_E9_INT32_COLIDX", "0"),
        "TFS_GLUE_E11_VEC_GRAD_DB": value(
            "TFS_GLUE_E11_VEC_GRAD_DB", "0"),
        "TFS_FWD_V2_SINGLE_SCAN": value("TFS_FWD_V2_SINGLE_SCAN", "0"),
        "TFS_GLUE_E12_D47_SINGLE_SCAN": value(
            "TFS_GLUE_E12_D47_SINGLE_SCAN", "0"),
        "TFS_GLUE_E13_ACTIVE_ROW": value(
            "TFS_GLUE_E13_ACTIVE_ROW", "0"),
        "TFS_GLUE_E14_ACTIVE_ROW_WIDE": value(
            "TFS_GLUE_E14_ACTIVE_ROW_WIDE", "0"),
        "TFS_HIGHD_FUSED_DB": value("TFS_HIGHD_FUSED_DB", "0"),
        "TFS_HIGHD_FUSED_TRANSPOSE": value(
            "TFS_HIGHD_FUSED_TRANSPOSE", "0"),
        "TFS_HIGHD_FUSED_SCALE_TRANSPOSE": value(
            "TFS_HIGHD_FUSED_SCALE_TRANSPOSE", "0"),
        "TFS_HIGHD_CONTIGUOUS_PANELS": value(
            "TFS_HIGHD_CONTIGUOUS_PANELS", "1"),
        "TFS_TRANSFORM_HIGHD_NATIVE": value(
            "TFS_TRANSFORM_HIGHD_NATIVE", "off"),
        "TFS_TRANSFORM_HIGHD_SINGLE_SCAN": value(
            "TFS_TRANSFORM_HIGHD_SINGLE_SCAN", "off"),
        "TFS_AGGREGATE_DSLAB_SINGLE_SCAN": value(
            "TFS_AGGREGATE_DSLAB_SINGLE_SCAN", "off"),
        "TFS_TRANSFORM_HIGHD_NATIVE_GS_BUDGET_BYTES": value(
            "TFS_TRANSFORM_HIGHD_NATIVE_GS_BUDGET_BYTES", str(256 << 20)),
        "TFS_COLIDX": value("TFS_COLIDX", "int64"),
    }
    return {
        "profile": os.environ.get("TFS_RELEASE_PROFILE", "").strip(),
        "release_mode": _env_flag("TFS_HIGHD_RELEASE_MODE", False),
        "dtype_contract": "bf16_inputs_fp32_accum_fp32_master",
        "locked": locked,
        "hardware": {
            key: os.environ.get(key, "")
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                        "TFS_CPU_NODES", "TFS_MEMORY_NODES",
                        "TFS_MEMORY_POLICY")
        },
    }


def write_runtime_contract(path: str) -> None:
    """Write the resolved environment contract atomically as JSON."""

    target = os.path.abspath(os.fspath(path))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    temporary = target + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(runtime_contract_payload(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, target)


def runtime_contract_line(*, validate: bool = False) -> str:
    """Return and optionally validate the process-level release contract."""

    def value(name: str, default: str) -> str:
        return os.environ.get(name, default).strip().lower()

    release_threads = _positive_int("OMP_NUM_THREADS", 1)
    fields = {
        "planner": "on" if planner_enabled() else "off",
        "highd_stream": "on" if _env_flag("TFS_HIGHD_STREAM_V1", False) else "off",
        "highd_native": value("TFS_HIGHD_NATIVE_STREAM", "auto"),
        "transform_highd_native": value(
            "TFS_TRANSFORM_HIGHD_NATIVE", "off"),
        "pull_only": "on" if _env_flag("TFS_PULL_ONLY_BACKWARD", False) else "off",
        "static_hs": value("TFS_STATIC_HS", "auto"),
        "numa_first_touch": value("TFS_NUMA_FIRST_TOUCH", "off"),
    }
    if validate and _env_flag("TFS_HIGHD_RELEASE_MODE", False):
        profile = os.environ.get("TFS_RELEASE_PROFILE", "").strip()
        dataflow_profile = profile == "v2_3_dataflow"
        final_profile = profile in {
            "final_pre_numa", "v2_4_dataflow_generalized"}
        expected = {
            "planner": "on", "highd_stream": "on", "highd_native": "auto",
            "transform_highd_native": "off",
            "pull_only": "on", "static_hs": "on",
            "numa_first_touch": "off" if dataflow_profile else "on",
        }
        mismatches = [
            f"{key}={fields[key]} expected={expected[key]}"
            for key in expected if fields[key] != expected[key]
        ]
        if mismatches:
            raise RuntimeError(
                "TFS release runtime contract mismatch: " + ", ".join(mismatches)
            )
        # Full locked-value validation is enabled by the v2.2 profile.  Keep
        # the short legacy check above for old reproduction scripts that set
        # only the original six switches by hand.
        if (_env_flag("TFS_RELEASE_CONTRACT_STRICT", False) or
                os.environ.get("TFS_RELEASE_PROFILE", "").strip()
                in {"v2_2_authority", "v2_highd_native"}):
            expected_locked = {
                "TFS_EXEC_PLANNER": "1",
                "TFS_HIGHD_STREAM_V1": "1",
                "TFS_HIGHD_NATIVE_STREAM": "auto",
                "TFS_TRANSFORM_HIGHD_STREAM_V1": "auto",
                "TFS_TRANSFORM_HIGHD_NATIVE": "off",
                "TFS_TRANSFORM_HIGHD_SINGLE_SCAN": "off",
                "TFS_TRANSFORM_HIGHD_NATIVE_GS_BUDGET_BYTES": str(256 << 20),
                "TFS_HIGHD_BWD_BUDGET_BYTES": str(64 << 20),
                "TFS_HIGHD_BWD_D_TILE": "256",
                "TFS_HIGHD_BWD_ROW_PANEL": "512",
                "TFS_TRANSFORM_HIGHD_D_TILE": "256",
                "TFS_HIGHD_NATIVE_PANEL": "512",
                "TFS_WORKSPACE_CACHE_MAX_BYTES": str(512 << 20),
                "TFS_HIGHD_PANEL_BUDGET_BYTES": str(32 << 20),
                "TFS_PULL_ONLY_BACKWARD": "1",
                "TFS_SCALE_GRAD_BF16_NATIVE": "1",
                "TFS_STATIC_HS": "on",
                "TFS_SMALL_SINGLE_SCAN": "auto",
                "TFS_ACTIVE_ROW": "auto",
                "TFS_AGGREGATE_SAVED": "off",
                "TFS_STATIC_AGGREGATE": "off",
                "TFS_NUMA_FIRST_TOUCH": "on",
                "TFS_NUMA_PRIVATE": "auto",
                "TFS_NUMA_REDUCE": "auto",
                "TFS_LOCALITY_SCHEDULE": "off",
                "TFS_HS_REPLICA": "off",
                "TFS_INTERNAL_PROFILE": "0",
                "TFS_PROFILE_NUMA_WORKSPACE": "0",
                "TFS_HIGHD_NATIVE_SCALE": "1",
                "TFS_HIGHD_STREAM_AUTO": "1",
                "HYBRID_STATIC_AGG_CACHE": "0",
                "TFS_E13_ACTIVE_MAX_DENSITY": "0.25",
                "TFS_GLUE_E1_FUSED_DB": "1",
                "TFS_GLUE_E2_VEC_STORE": "0",
                "TFS_GLUE_E3_EMPTY_DX": "0",
                "TFS_GLUE_E4_FUSED_EPILOGUE": "1",
                "TFS_GLUE_E5_VEC_HS": "0",
                "TFS_GLUE_E6_LOCAL_ZERO": "1",
                "TFS_GLUE_E6_PARALLEL_REDUCE": "0",
                "TFS_GLUE_E6_SERIAL_WT": "0",
                "TFS_GLUE_E7_FORWARD_SCHEDULE": "1",
                "TFS_GLUE_E8_RIGHTSIZE_POOL": "0",
                "TFS_GLUE_E8_ATOMIC_DONE": "0",
                "TFS_GLUE_E8_SPIN": "0",
                "TFS_GLUE_E9_INT32_COLIDX": "0",
                "TFS_GLUE_E11_VEC_GRAD_DB": "0",
                "TFS_FWD_V2_SINGLE_SCAN": "1",
                "TFS_GLUE_E12_D47_SINGLE_SCAN": "1",
                "TFS_GLUE_E13_ACTIVE_ROW": "1",
                "TFS_GLUE_E14_ACTIVE_ROW_WIDE": "0",
                "TFS_HIGHD_FUSED_DB": "0",
                "TFS_HIGHD_FUSED_TRANSPOSE": "0",
                "TFS_HIGHD_FUSED_SCALE_TRANSPOSE": "0",
                "TFS_HIGHD_CONTIGUOUS_PANELS": "1",
                "TFS_COLIDX": "int64",
            }
            # v2.3 deliberately exposes dimension-driven ``auto`` choices
            # for the fused high-D epilogues, panel layout, and CSR index
            # width.  They are still part of the locked release contract:
            # the planner/native wrapper resolve them from K/D and the index
            # range.  Do not reject the profile before a layer plan exists.
            if (os.environ.get("TFS_RELEASE_PROFILE", "").strip()
                    in {"v2_3_auto", "v2_3_dataflow",
                        "final_pre_numa", "v2_4_dataflow_generalized"}):
                expected_locked.update({
                    # The v2.3+ profile deliberately retired the obsolete
                    # fixed legacy budget.  Validate the same thread-aware
                    # base/per-thread resolution that the planner records,
                    # rather than rejecting valid multi-thread launches as
                    # though they had inherited a legacy override.
                    "TFS_HIGHD_BWD_BUDGET_BYTES": str(
                        _highd_workspace_budget(release_threads)),
                    "TFS_HIGHD_FUSED_DB": "auto",
                    "TFS_HIGHD_FUSED_TRANSPOSE": "auto",
                    "TFS_HIGHD_FUSED_SCALE_TRANSPOSE": "auto",
                    "TFS_HIGHD_CONTIGUOUS_PANELS": "auto",
                    "TFS_COLIDX": "auto",
                    "TFS_AGGREGATE_SAVED": "on",
                    "TFS_STATIC_AGGREGATE": "auto",
                    "HYBRID_STATIC_AGG_CACHE": "1",
                    "TFS_GLUE_E1_FUSED_DB": "1",
                    "TFS_GLUE_E2_VEC_STORE": "0",
                    "TFS_GLUE_E3_EMPTY_DX": "0",
                    "TFS_GLUE_E4_FUSED_EPILOGUE": "1",
                    "TFS_GLUE_E5_VEC_HS": "1",
                    "TFS_GLUE_E6_LOCAL_ZERO": "1",
                    "TFS_GLUE_E6_PARALLEL_REDUCE": "0",
                    "TFS_GLUE_E6_SERIAL_WT": "0",
                    "TFS_GLUE_E7_FORWARD_SCHEDULE": "1",
                    "TFS_GLUE_E8_RIGHTSIZE_POOL": "0",
                    "TFS_GLUE_E8_ATOMIC_DONE": "0",
                    "TFS_GLUE_E8_SPIN": "0",
                    "TFS_GLUE_E9_INT32_COLIDX": "0",
                    "TFS_GLUE_E11_VEC_GRAD_DB": "0",
                    "TFS_FWD_V2_SINGLE_SCAN": "1",
                    "TFS_GLUE_E12_D47_SINGLE_SCAN": "1",
                    "TFS_GLUE_E13_ACTIVE_ROW": "1",
                    "TFS_GLUE_E14_ACTIVE_ROW_WIDE": "0",
                    "TFS_TRANSFORM_HIGHD_NATIVE": "off",
                    "TFS_TRANSFORM_HIGHD_SINGLE_SCAN": "off",
                })
                if dataflow_profile:
                    expected_locked.update({
                        "TFS_NUMA_FIRST_TOUCH": "off",
                        "TFS_AGGREGATE_DSLAB_SINGLE_SCAN": "off",
                    })
                if final_profile:
                    expected_locked.update({
                        "TFS_AGGREGATE_SAVED": "auto",
                        "TFS_WORKSPACE_CACHE_MAX_BYTES": str(2 << 30),
                        "TFS_NUMA_FIRST_TOUCH": "on",
                        "TFS_NUMA_PRIVATE": "off",
                        "TFS_NUMA_REDUCE": "off",
                        "TFS_AGGREGATE_DSLAB_SINGLE_SCAN": "off",
                    })
            locked = runtime_contract_payload()["locked"]
            mismatches = [
                f"{key}={locked[key]} expected={expected}"
                for key, expected in expected_locked.items()
                if str(locked[key]) != expected
            ]
            if mismatches:
                raise RuntimeError(
                    "TFS release locked contract mismatch: " +
                    ", ".join(mismatches)
                )
    return "TFS_RUNTIME_CONTRACT " + " ".join(
        f"{key}={fields[key]}" for key in fields
    )


__all__ = [
    "LayerExecutionPlan",
    "KernelCandidate",
    "build_layer_plan",
    "emit_plan_logs",
    "plan_layers",
    "planner_enabled",
    "runtime_contract_line",
    "runtime_contract_payload",
    "write_runtime_contract",
    "partition_d",
]
