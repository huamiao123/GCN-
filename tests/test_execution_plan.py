import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from tfs_train.execution_plan import (KernelCandidate, build_layer_plan,
                                      partition_d, plan_layers,
                                      runtime_contract_line,
                                      runtime_contract_payload)


_FORMAL_SWITCHES = (
    "TFS_EXEC_PLANNER",
    "TFS_SMALL_SINGLE_SCAN",
    "TFS_ACTIVE_ROW",
    "TFS_AGGREGATE_SAVED",
    "TFS_STATIC_HS",
    "TFS_STATIC_AGGREGATE",
    "TFS_COLIDX",
    "TFS_MAX_LOCAL_DW_BYTES",
    "TFS_LOCAL_DW_BUDGET_BYTES",
    "HYBRID_STATIC_AGG_CACHE",
    "HYBRID_AMX_FORWARD",
    "HYBRID_AMX_BACKWARD",
    "TFS_HIGHD_STREAM_V1",
    "TFS_HIGHD_NATIVE_STREAM",
    "TFS_HIGHD_BWD_BUDGET_BYTES",
    "TFS_HIGHD_BWD_BASE_BUDGET_BYTES",
    "TFS_HIGHD_BWD_PER_THREAD_BUDGET_BYTES",
    "OMP_NUM_THREADS",
    "TFS_HIGHD_BWD_D_TILE",
    "TFS_HIGHD_BWD_ROW_PANEL",
    "TFS_TRANSFORM_HIGHD_D_TILE",
    "TFS_TRANSFORM_HIGHD_SINGLE_SCAN",
    "TFS_HIGHD_NATIVE_PANEL",
    "TFS_HIGHD_PANEL_BUDGET_BYTES",
    "TFS_WORKSPACE_CACHE_MAX_BYTES",
    "TFS_HIGHD_FUSED_DB",
    "TFS_HIGHD_FUSED_TRANSPOSE",
    "TFS_HIGHD_FUSED_SCALE_TRANSPOSE",
    "TFS_HIGHD_CONTIGUOUS_PANELS",
    "TFS_RELEASE_CONTRACT_STRICT",
    "TFS_HIGHD_RELEASE_MODE",
    "TFS_PULL_ONLY_BACKWARD",
    "TFS_SCALE_GRAD_BF16_NATIVE",
    "TFS_NUMA_FIRST_TOUCH",
    "TFS_RELEASE_PROFILE",
)


@pytest.fixture(autouse=True)
def clean_formal_switches(monkeypatch):
    for name in _FORMAL_SWITCHES:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "k,d,order",
    [
        (100, 128, "aggregate"),
        (128, 47, "transform"),
        (128, 40, "transform"),
        (1024, 128, "transform"),
        (128, 2983, "aggregate"),
        (512, 1024, "aggregate"),
        (2048, 256, "transform"),
    ],
)
def test_v1_order_is_shape_based(k, d, order):
    plan = build_layer_plan(1000, k, d)
    assert plan.order == order
    assert plan.kp >= k and plan.kp % 64 == 0
    assert plan.dp >= d and plan.dp % 32 == 0
    assert "v1_order:" + order in plan.reason


def test_planner_rejects_threads_above_native_capability():
    assert build_layer_plan(1000, 128, 47, threads=32).threads == 32
    with pytest.raises(ValueError, match="current native authority contract"):
        build_layer_plan(1000, 128, 47, threads=33)
    assert (runtime_contract_payload()["locked"]
            ["TFS_MAX_AUTHORITY_THREADS"] == 32)


@pytest.mark.parametrize(
    "k,d,path",
    [
        (1024, 128, "wide_k"),
        (128, 19, "c3"),
        (128, 2983, "wide_aggregate"),
        (1024, 256, "wide_output"),
    ],
)
def test_dimension_path_is_shape_driven_without_opt_in(k, d, path):
    """High-K/high-D paths must not depend on a legacy environment switch."""

    plan = build_layer_plan(1_000_000, k, d)
    assert plan.dimension_path == path
    assert plan.to_dict()["dimension_path"] == path
    assert f"dim_path={path}" in plan.log_line()
    assert f"dim_path:{path}" in plan.reason


def test_runtime_contract_records_thread_resolved_highd_budget(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "32")
    monkeypatch.setenv("TFS_HIGHD_BWD_BASE_BUDGET_BYTES", str(64 << 20))
    monkeypatch.setenv("TFS_HIGHD_BWD_PER_THREAD_BUDGET_BYTES", str(4 << 20))
    resolved = runtime_contract_payload()["locked"]
    assert resolved["TFS_HIGHD_BWD_BUDGET_BYTES"] == 128 << 20
    assert resolved["TFS_HIGHD_BWD_BASE_BUDGET_BYTES"] == 64 << 20
    assert resolved["TFS_HIGHD_BWD_PER_THREAD_BUDGET_BYTES"] == 4 << 20

    monkeypatch.setenv("TFS_HIGHD_BWD_BUDGET_BYTES", str(96 << 20))
    assert (runtime_contract_payload()["locked"]
            ["TFS_HIGHD_BWD_BUDGET_BYTES"] == 96 << 20)


def test_static_aggregate_requires_no_dx_and_amx(monkeypatch):
    monkeypatch.setenv("HYBRID_STATIC_AGG_CACHE", "1")
    monkeypatch.setenv("HYBRID_AMX_FORWARD", "1")
    monkeypatch.setenv("HYBRID_AMX_BACKWARD", "1")
    enabled = build_layer_plan(
        1000, 128, 128, compute_dx=False, input_static=True
    )
    disabled_dx = build_layer_plan(
        1000, 128, 128, compute_dx=True, input_static=True
    )
    assert enabled.static_pulled and enabled.save_pulled
    assert enabled.backward_mode == "aggregate_saved_v3"
    assert not disabled_dx.static_pulled
    assert disabled_dx.backward_mode == "selective_amx"


def test_formal_switches_and_plan_log(monkeypatch):
    monkeypatch.setenv("TFS_SMALL_SINGLE_SCAN", "off")
    monkeypatch.setenv("TFS_ACTIVE_ROW", "on")
    monkeypatch.setenv("TFS_COLIDX", "int64")
    # ``TFS_MAX_LOCAL_DW_BYTES`` outranks the deprecated alias below, and the
    # authority profile exports it as 0.  Drop it so this case actually
    # exercises the alias instead of silently reading the profile's 0 --
    # otherwise the test passes in a bare shell and fails inside
    # scripts/run_final_pre_numa_gate.sh, which sources the profile first.
    monkeypatch.delenv("TFS_MAX_LOCAL_DW_BYTES", raising=False)
    monkeypatch.setenv("TFS_LOCAL_DW_BUDGET_BYTES", "1048576")
    plan = build_layer_plan(4096, 100, 47)
    assert not plan.small_single_scan
    assert plan.active_row
    assert not plan.use_int32_colidx
    assert plan.local_dw_budget == 1048576
    line = plan.log_line(layer=0)
    assert line.startswith("TFS_PLAN layer=0 K=100 D=47")
    for token in (
        "order=transform",
        "backward=selective_amx",
        "compute_dx=1",
        "save_hs=1",
        "single_scan=0",
        "active_row=1",
    ):
        assert token in line


def test_plan_layers_has_one_source_of_truth():
    plans = plan_layers(1000, [100, 128, 47], feature_static_first=True)
    assert [p.order for p in plans] == ["aggregate", "transform"]
    assert plans[0].static_hs
    assert not plans[1].static_hs
    assert plans[0].compute_dx is False
    assert plans[1].compute_dx is True
    assert [p.layer_index for p in plans] == [0, 1]
    assert all(p.layer_path for p in plans)
    assert plans[0].layer1_path == ""
    assert plans[1].layer0_path == ""
    assert plans[1].layer1_backward == plans[1].backward_impl


def test_final_profile_dynamic_hidden_layer_never_uses_v4(monkeypatch):
    """The P0 regression: static V3 stays on, dynamic 128->128 returns C3."""

    for name, value in {
        "TFS_STATIC_HS": "on",
        "TFS_STATIC_AGGREGATE": "auto",
        "TFS_AGGREGATE_SAVED": "auto",
        "HYBRID_STATIC_AGG_CACHE": "1",
        "HYBRID_AMX_FORWARD": "1",
        "HYBRID_AMX_BACKWARD": "1",
    }.items():
        monkeypatch.setenv(name, value)
    plans = plan_layers(4097, [128, 128, 128, 47], threads=32)
    assert [p.execution_variant for p in plans] == [
        "aggregate_static_v3", "native_c3", "native_c3"
    ]
    assert [p.layer_index for p in plans] == [0, 1, 2]
    assert all(p.selected_impl == p.execution_variant for p in plans)
    assert all(p.selection_policy == "auto" for p in plans)


@pytest.mark.parametrize("depth", [2, 3, 4, 5])
def test_arbitrary_depth_has_per_layer_metadata(depth):
    dims = [100] + [128] * (depth - 1) + [47]
    plans = plan_layers(4097, dims, threads=8)
    assert len(plans) == depth
    assert [p.layer_index for p in plans] == list(range(depth))
    assert all(p.layer_path for p in plans)
    assert all(p.candidates for p in plans)


def test_dispatch_identity_is_fail_fast():
    plan = build_layer_plan(4097, 128, 47, threads=8)
    plan.assert_dispatch("native_c3")
    with pytest.raises(RuntimeError, match="plan/dispatch mismatch"):
        plan.assert_dispatch("aggregate_saved_v4")


def test_static_hs_on_never_leaks_to_non_static_layers(monkeypatch):
    monkeypatch.setenv("TFS_STATIC_HS", "on")
    plans = plan_layers(1000, [1024, 128, 2983],
                        feature_static_first=True, threads=32)
    assert plans[0].static_hs
    assert not plans[1].static_hs


def test_static_v3_capability_is_checked_by_planner(monkeypatch):
    for name, value in {
            "TFS_STATIC_HS": "on", "TFS_STATIC_AGGREGATE": "on",
            "HYBRID_STATIC_AGG_CACHE": "1", "HYBRID_AMX_FORWARD": "1",
            "HYBRID_AMX_BACKWARD": "1"}.items():
        monkeypatch.setenv(name, value)
    plan = build_layer_plan(4097, 100, 129, compute_dx=False,
                            input_static=True, graph_static=True,
                            feature_static=True)
    assert plan.execution_variant != "aggregate_static_v3"
    assert "static_v3_capability_k_or_d_exceeded" in plan.reason


def test_wide_aggregate_saves_pulled_contract():
    plan = build_layer_plan(1000, 128, 2983, compute_dx=True)
    assert plan.order == "aggregate"
    assert plan.save_pulled and not plan.save_hs
    assert plan.backward_mode == "aggregate_saved_wide"


def test_generic_aggregate_saved_is_explicitly_opt_in(monkeypatch):
    monkeypatch.setenv("TFS_AGGREGATE_SAVED", "on")
    monkeypatch.setenv("HYBRID_AMX_FORWARD", "1")
    monkeypatch.setenv("HYBRID_AMX_BACKWARD", "1")
    plan = build_layer_plan(1000, 64, 64, compute_dx=True)
    assert plan.order == "aggregate"
    assert plan.save_pulled and not plan.save_hs
    assert plan.backward_mode == "aggregate_saved_v4"
    assert "generic_aggregate_saved_v4" in plan.reason


def test_generic_aggregate_saved_auto_remains_legacy(monkeypatch):
    monkeypatch.setenv("TFS_AGGREGATE_SAVED", "auto")
    plan = build_layer_plan(1000, 64, 64, compute_dx=True)
    assert plan.backward_mode == "selective_amx"
    assert not plan.save_pulled


def test_zero_local_dw_budget_is_a_valid_disabled_gate(monkeypatch):
    monkeypatch.setenv("TFS_LOCAL_DW_BUDGET_BYTES", "0")
    monkeypatch.setenv("TFS_MAX_LOCAL_DW_BYTES", "0")
    plan = build_layer_plan(1000, 64, 64)
    assert plan.local_dw_budget == 0


def test_planner_emits_actual_dataflow_and_workspace_contract(monkeypatch):
    monkeypatch.setenv("TFS_HIGHD_BWD_BUDGET_BYTES", str(64 << 20))
    plan = build_layer_plan(4097, 1024, 256, threads=32, compute_dx=True)
    assert plan.forward_family == "transform"
    assert plan.sparse_fwd_width == plan.dp
    assert plan.sparse_bwd_width == plan.dp
    assert plan.sparse_fwd_tensor == "T"
    assert plan.sparse_bwd_tensor == "Gs"
    assert plan.sparse_fwd_width_logical == 256
    assert plan.sparse_bwd_width_logical == 256
    assert plan.sparse_fwd_width_physical == plan.dp
    assert plan.sparse_bwd_width_physical == plan.dp
    assert plan.workspace_strategy in {"full_d", "d_slab"}
    assert plan.local_dw_bytes == 32 * plan.kp * plan.dp * 4
    line = plan.log_line(layer=1)
    for token in ("forward_family=transform", "sparse_fwd_width=256",
                  "sparse_bwd_width=256", "sparse_bwd_tensor=Gs",
                  "workspace=", "dtype="):
        assert token in line


@pytest.mark.parametrize("k,d,logical,physical,tensor", [
    (128, 19, 19, 32, "T"),
    (1024, 128, 128, 128, "T"),
    (128, 2983, 128, 128, "Hs"),
])
def test_sparse_width_metadata_matches_algebraic_order(
        k, d, logical, physical, tensor):
    plan = build_layer_plan(4097, k, d, compute_dx=True)
    assert plan.sparse_fwd_width == physical
    assert plan.sparse_bwd_width == physical
    assert plan.sparse_fwd_width_logical == logical
    assert plan.sparse_bwd_width_logical == logical
    assert plan.sparse_fwd_width_physical == physical
    assert plan.sparse_bwd_width_physical == physical
    assert plan.sparse_fwd_tensor == tensor
    assert plan.sparse_bwd_tensor == ("Gs" if tensor == "T" else "dP")
    plan.validate()


def test_sparse_backward_contract_is_variant_driven(monkeypatch):
    c3 = build_layer_plan(4097, 100, 110, compute_dx=True)
    assert c3.execution_variant == "native_c3"
    assert (c3.sparse_bwd_tensor, c3.sparse_bwd_width_logical,
            c3.sparse_bwd_width_physical) == ("Gs", 110, c3.dp)

    monkeypatch.setenv("TFS_AGGREGATE_SAVED", "on")
    monkeypatch.setenv("HYBRID_AMX_FORWARD", "1")
    monkeypatch.setenv("HYBRID_AMX_BACKWARD", "1")
    v4 = build_layer_plan(4097, 100, 110, compute_dx=True)
    assert v4.execution_variant == "aggregate_saved_v4"
    assert (v4.sparse_bwd_tensor, v4.sparse_bwd_width_logical,
            v4.sparse_bwd_width_physical) == ("Gs", 110, v4.dp)

    monkeypatch.setenv("TFS_AGGREGATE_SAVED", "auto")
    monkeypatch.setenv("TFS_STATIC_HS", "on")
    monkeypatch.setenv("TFS_STATIC_AGGREGATE", "auto")
    monkeypatch.setenv("HYBRID_STATIC_AGG_CACHE", "1")
    static = build_layer_plan(4097, 100, 110, compute_dx=False,
                              input_static=True, graph_static=True,
                              feature_static=True)
    assert static.execution_variant == "aggregate_static_v3"
    assert (static.sparse_bwd_tensor, static.sparse_bwd_width,
            static.sparse_bwd_width_logical,
            static.sparse_bwd_width_physical) == ("none", 0, 0, 0)
    assert "sparse_bwd_tensor=none" in static.log_line()
    static.validate()


def test_kernel_candidate_is_keyword_only_and_typed():
    candidate = KernelCandidate(
        name="native", supported=True, workspace_bytes=64,
        estimated_cost=1.5, reason="unit")
    assert isinstance(candidate.workspace_bytes, int)
    assert isinstance(candidate.estimated_cost, float)
    assert isinstance(candidate.reason, str)
    with pytest.raises(TypeError):
        KernelCandidate("native", True, 64, 1.5, "unit")


def test_highd_budget_is_thread_aware_and_monotonic(monkeypatch):
    monkeypatch.delenv("TFS_HIGHD_BWD_BUDGET_BYTES", raising=False)
    monkeypatch.setenv("TFS_HIGHD_BWD_BASE_BUDGET_BYTES", str(64 << 20))
    monkeypatch.setenv("TFS_HIGHD_BWD_PER_THREAD_BUDGET_BYTES", str(4 << 20))
    plans = [build_layer_plan(1_000_003, k, 2983, threads=threads)
             for k in (128, 192, 256) for threads in (1, 2, 4, 8, 16, 32)]
    for plan in plans:
        assert plan.execution_variant == "aggregate_highd_full"
    by_k = {k: [build_layer_plan(1_000_003, k, 2983, threads=t)
                for t in (1, 2, 4, 8, 16, 32)] for k in (128, 192, 256)}
    for series in by_k.values():
        assert [p.workspace_budget_bytes for p in series] == sorted(
            p.workspace_budget_bytes for p in series)


def test_every_planner_candidate_uses_explicit_fields():
    for plan in (
            build_layer_plan(4096, 1024, 256, threads=32),
            build_layer_plan(4096, 128, 2983, threads=32),
            build_layer_plan(4096, 128, 47, threads=4)):
        assert plan.candidates
        for candidate in plan.candidates:
            assert isinstance(candidate, KernelCandidate)
            assert isinstance(candidate.workspace_bytes, int)
            assert isinstance(candidate.estimated_cost, float)
            assert isinstance(candidate.reason, str)


def test_dense_tile_and_sparse_slab_are_separate_contracts():
    transform = build_layer_plan(4097, 1024, 257, threads=32)
    # The transform correctness stream keeps one logical dense tile while its
    # sparse traversal remains bounded by the requested 256-wide slab.
    assert transform.dense_d_tile == transform.d_tile == 257
    assert transform.sparse_d_slab == 256
    aggregate_full = build_layer_plan(4097, 128, 2983, threads=32)
    assert aggregate_full.dense_d_tile == aggregate_full.d_tile == 2983
    assert aggregate_full.sparse_d_slab == 256
    transform.validate()
    aggregate_full.validate()


def test_planner_marks_native_d_slab_shape(monkeypatch):
    monkeypatch.setenv("TFS_HIGHD_BWD_BUDGET_BYTES", str(64 << 20))
    plan = build_layer_plan(1024, 1024, 1024, threads=32)
    assert plan.workspace_strategy == "d_slab"
    assert plan.native_d_slab_supported
    assert "native_d_slab_slice_adapter" in plan.reason


@pytest.mark.parametrize(
    "n,k,d,selected,workspace",
    [
        (17, 1, 1, "native_c3", "not_applicable"),
        (4097, 1024, 128, "native_wide_k", "not_applicable"),
        (4097, 128, 2983, "aggregate_highd_full", "full_d"),
        (4097, 1024, 1024, "aggregate_highd_dslab", "d_slab"),
        (4097, 1024, 513, "transform_highd_legacy", "d_slab"),
    ],
)
def test_shape_matrix_has_one_valid_selected_candidate(
        n, k, d, selected, workspace, monkeypatch):
    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_STREAM_V1", "auto")
    plan = build_layer_plan(n, k, d, threads=32)
    assert plan.selected_impl == selected
    assert plan.workspace_strategy == workspace
    assert any(c.name == selected and c.supported for c in plan.candidates)
    plan.validate()


def test_reference_fallback_uses_panel_budget_not_native_budget(monkeypatch):
    """A low native budget must select a usable bounded reference stream."""

    monkeypatch.setenv("TFS_HIGHD_BWD_BUDGET_BYTES", "1")
    plan = build_layer_plan(4097, 1024, 1024, threads=32)
    assert plan.selected_impl == "aggregate_highd_reference"
    assert plan.fallback_reason == "native_candidates_exceed_budget"
    assert plan.workspace_bytes > plan.workspace_budget_bytes
    assert plan.panel_working_set_bytes <= plan.panel_budget_bytes
    plan.validate()


def test_transform_highd_multislab_is_explicit_opt_in(monkeypatch):
    """Known multi-slab regression stays probeable but not automatic."""

    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_STREAM_V1", "auto")
    auto = build_layer_plan(4097, 1024, 513, threads=32)
    assert auto.selected_impl == "transform_highd_legacy"
    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_STREAM_V1", "on")
    forced = build_layer_plan(4097, 1024, 513, threads=32)
    assert forced.selected_impl == "transform_highd_stream"
    assert len(forced.d_slabs) > 1
    forced.validate()


def test_transform_native_is_explicit_and_budgeted(monkeypatch):
    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_STREAM_V1", "on")
    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_NATIVE", "on")
    plan = build_layer_plan(4097, 1024, 257, threads=32, compute_dx=True)
    assert plan.selected_impl == "transform_highd_native"
    assert plan.workspace_strategy == "d_slab"
    assert plan.workspace_bytes <= plan.workspace_budget_bytes
    assert plan.fallback_reason == ""
    assert any(c.name == "transform_highd_native" and c.supported
               for c in plan.candidates)


def test_transform_single_scan_is_explicit_and_budgeted(monkeypatch):
    """The fused immediate-consume path is probeable but never implicit."""

    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_SINGLE_SCAN", "on")
    plan = build_layer_plan(4097, 1024, 513, threads=4, compute_dx=True)
    assert plan.selected_impl == "transform_highd_single_scan"
    assert plan.d_slabs == ((0, 513),)
    assert plan.workspace_strategy == "full_d"
    assert plan.workspace_bytes <= plan.workspace_budget_bytes
    assert any(c.name == "transform_highd_single_scan" and c.supported
               for c in plan.candidates)
    plan.validate()


def test_transform_single_scan_requires_plan_authorization(monkeypatch):
    """An explicit probe cannot silently override a supplied plan."""

    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_SINGLE_SCAN", "on")
    plan = build_layer_plan(4097, 1024, 257, threads=4, compute_dx=True)
    assert plan.selected_impl != "transform_highd_single_scan"


def test_aggregate_dslab_single_scan_is_explicit_reference_only(monkeypatch):
    monkeypatch.setenv("TFS_AGGREGATE_DSLAB_SINGLE_SCAN", "on")
    plan = build_layer_plan(4097, 1024, 1024, threads=32,
                            compute_dx=True)
    assert plan.execution_variant == "aggregate_highd_single_scan"
    candidate = next(c for c in plan.candidates
                     if c.name == "aggregate_highd_single_scan")
    assert candidate.supported
    assert candidate.sparse_scans == 1
    assert candidate.performance_state == "experimental"
    assert not candidate.validated_for_auto


def test_transform_native_auto_uses_panel_cost_gate(monkeypatch):
    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_STREAM_V1", "auto")
    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_NATIVE", "auto")
    small_t4 = build_layer_plan(4103, 1024, 257, threads=4,
                                compute_dx=True)
    small_t32 = build_layer_plan(4103, 1024, 257, threads=32,
                                 compute_dx=True)
    multi_t4 = build_layer_plan(4097, 1024, 513, threads=4,
                                compute_dx=True)
    multi_t32 = build_layer_plan(4097, 1024, 513, threads=32,
                                 compute_dx=True)
    assert small_t4.selected_impl == "transform_highd_legacy"
    assert small_t32.selected_impl == "transform_highd_legacy"
    assert multi_t4.selected_impl == "transform_highd_legacy"
    assert multi_t32.selected_impl == "transform_highd_legacy"
    for plan in (small_t4, small_t32, multi_t4, multi_t32):
        for candidate in plan.candidates:
            if candidate.name in {"transform_highd_stream",
                                  "transform_highd_native"}:
                assert not candidate.validated_for_auto


def test_transform_native_gs_budget_is_shape_bounded(monkeypatch):
    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_STREAM_V1", "on")
    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_NATIVE", "on")
    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_NATIVE_GS_BUDGET_BYTES", "1")
    plan = build_layer_plan(4097, 1024, 257, threads=4,
                            compute_dx=True)
    assert plan.selected_impl == "transform_highd_stream"
    assert plan.fallback_reason == "transform_highd_native_pending"
    assert not any(c.name == "transform_highd_native" and c.supported
                   for c in plan.candidates)


def test_release_runtime_contract_is_fail_fast(monkeypatch):
    for key, value in {
        "TFS_HIGHD_RELEASE_MODE": "1",
        "TFS_EXEC_PLANNER": "1",
        "TFS_HIGHD_STREAM_V1": "1",
        "TFS_HIGHD_NATIVE_STREAM": "auto",
        "TFS_PULL_ONLY_BACKWARD": "1",
        "TFS_SCALE_GRAD_BF16_NATIVE": "1",
        "TFS_STATIC_HS": "on",
        "TFS_NUMA_FIRST_TOUCH": "on",
    }.items():
        monkeypatch.setenv(key, value)
    assert "planner=on" in runtime_contract_line(validate=True)
    monkeypatch.setenv("TFS_HIGHD_STREAM_V1", "0")
    with pytest.raises(RuntimeError, match="runtime contract mismatch"):
        runtime_contract_line(validate=True)


def test_v23_auto_release_contract_accepts_dimension_driven_switches(monkeypatch):
    """v2.3 keeps release fail-fast while allowing planner-owned ``auto``."""

    for key, value in {
        "TFS_RELEASE_PROFILE": "v2_3_auto",
        "TFS_RELEASE_CONTRACT_STRICT": "1",
        "TFS_HIGHD_RELEASE_MODE": "1",
        "TFS_EXEC_PLANNER": "1",
        "TFS_HIGHD_STREAM_V1": "1",
        "TFS_HIGHD_NATIVE_STREAM": "auto",
        "TFS_TRANSFORM_HIGHD_STREAM_V1": "auto",
        "TFS_TRANSFORM_HIGHD_NATIVE": "off",
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
        "TFS_AGGREGATE_SAVED": "on",
        "TFS_STATIC_AGGREGATE": "auto",
        "TFS_NUMA_FIRST_TOUCH": "on",
        "TFS_NUMA_PRIVATE": "auto",
        "TFS_NUMA_REDUCE": "auto",
        "TFS_LOCALITY_SCHEDULE": "off",
        "TFS_HS_REPLICA": "off",
        "TFS_INTERNAL_PROFILE": "0",
        "TFS_PROFILE_NUMA_WORKSPACE": "0",
        "TFS_HIGHD_NATIVE_SCALE": "1",
        "TFS_HIGHD_STREAM_AUTO": "1",
        "HYBRID_STATIC_AGG_CACHE": "1",
        "TFS_E13_ACTIVE_MAX_DENSITY": "0.25",
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
        "TFS_HIGHD_FUSED_DB": "auto",
        "TFS_HIGHD_FUSED_TRANSPOSE": "auto",
        "TFS_HIGHD_FUSED_SCALE_TRANSPOSE": "auto",
        "TFS_HIGHD_CONTIGUOUS_PANELS": "auto",
        "TFS_COLIDX": "auto",
    }.items():
        monkeypatch.setenv(key, value)
    contract = runtime_contract_line(validate=True)
    assert "planner=on" in contract
    assert "transform_highd_native=off" in contract


def test_v23_release_contract_rejects_inherited_optional_switch(monkeypatch):
    """Performance gates outside the short legacy contract are locked too."""

    for key, value in {
        "TFS_RELEASE_PROFILE": "v2_3_auto",
        "TFS_RELEASE_CONTRACT_STRICT": "1",
        "TFS_HIGHD_RELEASE_MODE": "1",
        "TFS_EXEC_PLANNER": "1",
        "TFS_HIGHD_STREAM_V1": "1",
        "TFS_HIGHD_NATIVE_STREAM": "auto",
        "TFS_TRANSFORM_HIGHD_STREAM_V1": "auto",
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
        "TFS_AGGREGATE_SAVED": "on",
        "TFS_STATIC_AGGREGATE": "auto",
        "TFS_NUMA_FIRST_TOUCH": "on",
        "TFS_NUMA_PRIVATE": "auto",
        "TFS_NUMA_REDUCE": "auto",
        "TFS_LOCALITY_SCHEDULE": "off",
        "TFS_HS_REPLICA": "off",
        "TFS_INTERNAL_PROFILE": "0",
        "TFS_PROFILE_NUMA_WORKSPACE": "0",
        "TFS_HIGHD_NATIVE_SCALE": "1",
        "TFS_HIGHD_STREAM_AUTO": "1",
        "HYBRID_STATIC_AGG_CACHE": "1",
        "TFS_E13_ACTIVE_MAX_DENSITY": "0.25",
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
        "TFS_HIGHD_FUSED_DB": "auto",
        "TFS_HIGHD_FUSED_TRANSPOSE": "auto",
        "TFS_HIGHD_FUSED_SCALE_TRANSPOSE": "auto",
        "TFS_HIGHD_CONTIGUOUS_PANELS": "auto",
        "TFS_COLIDX": "auto",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("TFS_ACTIVE_ROW", "on")
    with pytest.raises(RuntimeError, match="locked contract mismatch"):
        runtime_contract_line(validate=True)


@pytest.mark.parametrize("n,k,d", [
    (4097, 1024, 256),
    (4103, 1024, 257),
    (8191, 2048, 513),
    (4099, 300, 129),
])
def test_production_transform_highd_shapes_are_planned_without_dataset_name(
        monkeypatch, n, k, d):
    """The real dispatcher gate is shape-based and covers K>D>128 tails."""

    monkeypatch.setenv("TFS_TRANSFORM_HIGHD_STREAM_V1", "on")
    plan = build_layer_plan(n, k, d, threads=32, compute_dx=True)
    assert plan.order == "transform"
    assert plan.dimension_path == "wide_output"
    assert plan.selected_impl == "transform_highd_stream"
    assert plan.d_slabs[-1][1] == d
    assert plan.plan_id == build_layer_plan(
        n, k, d, threads=32, compute_dx=True).plan_id
    plan.validate()


def test_dslab_partition_has_no_oversized_merged_tail():
    slabs = partition_d(1100, 256)
    assert [d1 - d0 for d0, d1 in slabs] == [220, 220, 220, 220, 220]
    assert max(d1 - d0 for d0, d1 in slabs) <= 256
    assert min(d1 - d0 for d0, d1 in slabs) > 128


def test_native_dslab_workspace_uses_actual_slab_width(monkeypatch):
    monkeypatch.setenv("TFS_HIGHD_BWD_BUDGET_BYTES", str(64 << 20))
    plan = build_layer_plan(4096, 1024, 1100, threads=32, compute_dx=True)
    assert plan.selected_impl == "aggregate_highd_dslab"
    assert plan.native_d_slab_supported
    assert plan.dslab_max_width == 220
    assert plan.workspace_bytes == 32 * plan.kp * 224 * 4
    assert plan.workspace_bytes <= plan.workspace_budget_bytes
    plan.validate()


def test_budget_increase_does_not_downgrade_native_to_python(monkeypatch):
    selected = []
    for budget in (32, 64, 128, 256, 512):
        monkeypatch.setenv("TFS_HIGHD_BWD_BUDGET_BYTES", str(budget << 20))
        plan = build_layer_plan(4096, 1024, 1024, threads=32,
                                compute_dx=True)
        selected.append(plan.selected_impl)
    assert selected[0] in {"aggregate_highd_dslab", "aggregate_highd_full"}
    assert all(name not in {"aggregate_highd_reference", "transform_highd_legacy"}
               for name in selected)


def test_panel_budget_is_part_of_the_plan(monkeypatch):
    monkeypatch.setenv("TFS_HIGHD_PANEL_BUDGET_BYTES", str(1 << 20))
    plan = build_layer_plan(100_000, 1024, 1024, threads=32,
                            compute_dx=True)
    assert plan.panel < 512
    assert plan.panel_budget_bytes == 1 << 20
    assert plan.panel_working_set_bytes <= plan.panel_budget_bytes
    plan.validate()
