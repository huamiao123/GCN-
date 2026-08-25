#!/usr/bin/env bash
# Source this file from every TFS benchmark/training launch.
#
# It is deliberately free of dataset names and thread counts.  The Python
# planner chooses the order and caches from the layer shape; the native
# extension consumes the same formal gates.  Overrides are valid only in debug
# or explicit ablation profiles; the authority profile assigns locked values.

export TFS_EXEC_PLANNER="${TFS_EXEC_PLANNER:-1}"
export TFS_SMALL_SINGLE_SCAN="${TFS_SMALL_SINGLE_SCAN:-auto}"
export TFS_ACTIVE_ROW="${TFS_ACTIVE_ROW:-auto}"
# Conservative cross-graph default: aggregate-saved V4 is an explicit
# ablation until a complete no-regression matrix exists.
export TFS_AGGREGATE_SAVED="${TFS_AGGREGATE_SAVED:-off}"
export TFS_STATIC_HS="${TFS_STATIC_HS:-auto}"
export TFS_STATIC_AGGREGATE="${TFS_STATIC_AGGREGATE:-off}"
export TFS_COLIDX="${TFS_COLIDX:-auto}"
export TFS_NUMA_PRIVATE="${TFS_NUMA_PRIVATE:-auto}"
export TFS_NUMA_REDUCE="${TFS_NUMA_REDUCE:-auto}"
# NUMA first-touch is measured separately because its benefit is placement-
# and graph-dependent; keep it out of the common authority path.
export TFS_NUMA_FIRST_TOUCH="${TFS_NUMA_FIRST_TOUCH:-off}"
export TFS_LOCALITY_SCHEDULE="${TFS_LOCALITY_SCHEDULE:-off}"
export TFS_HS_REPLICA="${TFS_HS_REPLICA:-off}"
# Internal native timing is diagnostic-only.  Authority runs must leave it
# off because the per-call std::endl logging is not present in DGL.
export TFS_INTERNAL_PROFILE="${TFS_INTERNAL_PROFILE:-0}"
export TFS_MAX_LOCAL_DW_BYTES="${TFS_MAX_LOCAL_DW_BYTES:-0}"
export TFS_LOCAL_DW_BUDGET_BYTES="${TFS_LOCAL_DW_BUDGET_BYTES:-0}"
export TFS_WORKSPACE_CACHE_MAX_BYTES="${TFS_WORKSPACE_CACHE_MAX_BYTES:-536870912}"
export TFS_HIGHD_BWD_BASE_BUDGET_BYTES="${TFS_HIGHD_BWD_BASE_BUDGET_BYTES:-67108864}"
export TFS_HIGHD_BWD_PER_THREAD_BUDGET_BYTES="${TFS_HIGHD_BWD_PER_THREAD_BUDGET_BYTES:-4194304}"
export TFS_HIGHD_PANEL_BUDGET_BYTES="${TFS_HIGHD_PANEL_BUDGET_BYTES:-33554432}"
export TFS_PROFILE_NUMA_WORKSPACE="${TFS_PROFILE_NUMA_WORKSPACE:-0}"

# A release profile is applied after the compatibility defaults above.  This
# keeps old ablation scripts reproducible while making the formal v2.1 launch
# independent of an inherited login environment.  The profile is deliberately
# shape agnostic: the Python planner still chooses transform/aggregate and the
# native wrapper still checks the actual K/D shape.
case "${TFS_RELEASE_PROFILE:-}" in
  v2_2_authority|v2_highd_native)
    # Authority values are deliberately assigned (not ${VAR:-default}) so a
    # contaminated login shell cannot change the execution path.  Hardware
    # and dataset parameters remain the launcher's responsibility.
    export TFS_EXEC_PLANNER=1
    export TFS_SMALL_SINGLE_SCAN=auto TFS_ACTIVE_ROW=auto
    export TFS_AGGREGATE_SAVED=off TFS_STATIC_AGGREGATE=off
    export TFS_HIGHD_STREAM_V1=1
    export TFS_HIGHD_NATIVE_STREAM=auto
    export TFS_TRANSFORM_HIGHD_STREAM_V1=auto
    # The new native Transform-HighD slab has passed component gates but is
    # not yet part of the authority profile; enable it only in an explicit
    # shape/thread ablation.
    export TFS_TRANSFORM_HIGHD_NATIVE=off
    export TFS_TRANSFORM_HIGHD_NATIVE_GS_BUDGET_BYTES=$((256 * 1024 * 1024))
    export TFS_HIGHD_BWD_BASE_BUDGET_BYTES=$((64 * 1024 * 1024))
    export TFS_HIGHD_BWD_PER_THREAD_BUDGET_BYTES=$((4 * 1024 * 1024))
    # The legacy fixed budget overrides the thread-aware contract whenever it
    # is present.  Do not inherit it from an archived launcher/login shell.
    unset TFS_HIGHD_BWD_BUDGET_BYTES
    export TFS_HIGHD_BWD_D_TILE=256
    export TFS_HIGHD_BWD_ROW_PANEL=512
    export TFS_TRANSFORM_HIGHD_D_TILE=256
    export TFS_HIGHD_NATIVE_PANEL=512
    export TFS_WORKSPACE_CACHE_MAX_BYTES=$((512 * 1024 * 1024))
    export TFS_HIGHD_PANEL_BUDGET_BYTES=$((32 * 1024 * 1024))
    export TFS_PULL_ONLY_BACKWARD=1
    export TFS_SCALE_GRAD_BF16_NATIVE=1
    export TFS_STATIC_HS=on
    export TFS_NUMA_FIRST_TOUCH=on
    export TFS_NUMA_PRIVATE=auto TFS_NUMA_REDUCE=auto
    export TFS_LOCALITY_SCHEDULE=off TFS_HS_REPLICA=off
    export TFS_INTERNAL_PROFILE=0 TFS_PROFILE_NUMA_WORKSPACE=0
    export TFS_HIGHD_FUSED_DB=0
    export TFS_HIGHD_FUSED_TRANSPOSE=0
    export TFS_HIGHD_FUSED_SCALE_TRANSPOSE=0
    export TFS_HIGHD_CONTIGUOUS_PANELS=1
    export TFS_COLIDX=int64
    export TFS_MAX_LOCAL_DW_BYTES=0
    export TFS_LOCAL_DW_BUDGET_BYTES=0
    export TFS_HIGHD_NATIVE_SCALE=1
    export TFS_HIGHD_STREAM_AUTO=1
    export HYBRID_STATIC_AGG_CACHE=0
    export TFS_GLUE_E1_FUSED_DB=1 TFS_GLUE_E2_VEC_STORE=0
    export TFS_GLUE_E3_EMPTY_DX=0 TFS_GLUE_E4_FUSED_EPILOGUE=1
    export TFS_GLUE_E5_VEC_HS=0 TFS_GLUE_E6_LOCAL_ZERO=1
    export TFS_GLUE_E6_PARALLEL_REDUCE=0 TFS_GLUE_E6_SERIAL_WT=0
    # Pool capacity is now fixed at 32; the old first-request right-sizing
    # switch is retired and must not appear enabled in provenance.
    export TFS_GLUE_E7_FORWARD_SCHEDULE=1 TFS_GLUE_E8_RIGHTSIZE_POOL=0
    export TFS_GLUE_E8_ATOMIC_DONE=0 TFS_GLUE_E8_SPIN=0
    export TFS_GLUE_E9_INT32_COLIDX=0 TFS_GLUE_E11_VEC_GRAD_DB=0
    export TFS_FWD_V2_SINGLE_SCAN=1 TFS_GLUE_E12_D47_SINGLE_SCAN=1
    export TFS_GLUE_E13_ACTIVE_ROW=1 TFS_GLUE_E14_ACTIVE_ROW_WIDE=0
    export TFS_E13_ACTIVE_MAX_DENSITY=0.25
    export TFS_RELEASE_CONTRACT_STRICT=1
    export TFS_HIGHD_RELEASE_MODE=1
    export TFS_RUNTIME_CONTRACT="planner=on highd_stream=on highd_native=auto transform_native=off pull_only=on static_hs=on numa_first_touch=on budget=67108864 d_tile=256 row_panel=512 panel_budget=33554432 colidx=int64"
    ;;
  v2_3_auto|v2_3_dataflow|final_pre_numa|v2_4_dataflow_generalized)
    # Dimension/contract-driven optimized profile.  This keeps the v2.2
    # High-D contract but enables the two static-input optimizations that
    # were previously held as explicit ablations.  The planner remains the
    # sole source of the K/D/order decision; these switches are dataset-free.
    # V3 is selected only for static aggregate-first layers with no dX.
    # V4 is selected only for AMX aggregate-first layers that do not qualify
    # for the static T0 path.
    export TFS_EXEC_PLANNER=1
    export TFS_SMALL_SINGLE_SCAN=auto TFS_ACTIVE_ROW=auto
    export TFS_HIGHD_STREAM_V1=1
    export TFS_HIGHD_NATIVE_STREAM=auto
    export TFS_TRANSFORM_HIGHD_STREAM_V1=auto
    export TFS_TRANSFORM_HIGHD_NATIVE=off
    export TFS_TRANSFORM_HIGHD_SINGLE_SCAN=off
    export TFS_AGGREGATE_DSLAB_SINGLE_SCAN=off
    export TFS_TRANSFORM_HIGHD_NATIVE_GS_BUDGET_BYTES=$((256 * 1024 * 1024))
    export TFS_HIGHD_BWD_BASE_BUDGET_BYTES=$((64 * 1024 * 1024))
    export TFS_HIGHD_BWD_PER_THREAD_BUDGET_BYTES=$((4 * 1024 * 1024))
    # See the authority profile above: fixed legacy overrides are incompatible
    # with the monotonic base/per-thread budget contract.
    unset TFS_HIGHD_BWD_BUDGET_BYTES
    export TFS_HIGHD_BWD_D_TILE=256
    export TFS_HIGHD_BWD_ROW_PANEL=512
    export TFS_TRANSFORM_HIGHD_D_TILE=256
    export TFS_HIGHD_NATIVE_PANEL=512
    # Products has a single ordinary backward workspace just above 512 MiB.
    # Keep the authority cache large enough for its two hidden layers while
    # remaining a dataset-independent, fixed release contract.
    export TFS_WORKSPACE_CACHE_MAX_BYTES=$((2 * 1024 * 1024 * 1024))
    export TFS_HIGHD_PANEL_BUDGET_BYTES=$((32 * 1024 * 1024))
    export TFS_PULL_ONLY_BACKWARD=1
    export TFS_SCALE_GRAD_BF16_NATIVE=1
    export TFS_STATIC_HS=on
    export TFS_STATIC_AGGREGATE=auto
    if [[ "${TFS_RELEASE_PROFILE:-}" == "final_pre_numa" ]]; then
      # V4 remains an explicit ablation.  ``auto`` preserves the validated
      # static aggregate V3 layer-0 cache without routing dynamic 128->128
      # hidden layers through the slower per-step saved-pulled path.
      export TFS_AGGREGATE_SAVED=auto
      export TFS_PROFILE_STATUS=authority
    elif [[ "${TFS_RELEASE_PROFILE:-}" == "v2_4_dataflow_generalized" ]]; then
      export TFS_AGGREGATE_SAVED=auto
      export TFS_PROFILE_STATUS=deprecated_ablation
    else
      export TFS_AGGREGATE_SAVED=on
      export TFS_PROFILE_STATUS=$(
        [[ "${TFS_RELEASE_PROFILE:-}" == "v2_3_dataflow" ]] &&
          printf deprecated_ablation || printf historical)
    fi
    export HYBRID_STATIC_AGG_CACHE=1
    # v2_3_dataflow freezes the dataflow/planner work before the separate
    # NUMA phase.  Keep the historical v2_3_auto profile unchanged, while
    # giving the current authority matrix an explicit NUMA-off contract.
    if [[ "${TFS_RELEASE_PROFILE:-}" == "v2_3_dataflow" ]]; then
      export TFS_NUMA_FIRST_TOUCH=off
    else
      export TFS_NUMA_FIRST_TOUCH=on
    fi
    if [[ "${TFS_RELEASE_PROFILE:-}" == "final_pre_numa" ||
          "${TFS_RELEASE_PROFILE:-}" == "v2_4_dataflow_generalized" ]]; then
      # These legacy knobs have no production consumer, but lock them off so
      # the pre-NUMA manifest cannot be mistaken for an ownership/reduction
      # experiment.  Only parallel first-touch remains active here.
      export TFS_NUMA_PRIVATE=off TFS_NUMA_REDUCE=off
    else
      export TFS_NUMA_PRIVATE=auto TFS_NUMA_REDUCE=auto
    fi
    export TFS_LOCALITY_SCHEDULE=off TFS_HS_REPLICA=off
    export TFS_INTERNAL_PROFILE=0 TFS_PROFILE_NUMA_WORKSPACE=0
    # Keep the proven r5 glue choices in the unified profile.  These are
    # shape/density-gated inside the native wrapper; they are not dataset
    # names or a forced execution order.  In particular, do not lock CSR
    # indices to int64: the planner validates the index range and selects an
    # int32 workspace whenever it is safe (the old IGB/Products path did so).
    export TFS_GLUE_E1_FUSED_DB=1 TFS_GLUE_E2_VEC_STORE=0
    export TFS_GLUE_E3_EMPTY_DX=0 TFS_GLUE_E4_FUSED_EPILOGUE=1
    # The BF16 Hs producer is bitwise-equivalent to the scalar conversion
    # (covered by the extension smoke) and removes a serial conversion tail
    # from both static-cache misses and uncached narrow layers.
    export TFS_GLUE_E5_VEC_HS=1 TFS_GLUE_E6_LOCAL_ZERO=1
    export TFS_GLUE_E6_PARALLEL_REDUCE=0 TFS_GLUE_E6_SERIAL_WT=0
    # Pool capacity is now fixed at 32; the old first-request right-sizing
    # switch is retired and must not appear enabled in provenance.
    export TFS_GLUE_E7_FORWARD_SCHEDULE=1 TFS_GLUE_E8_RIGHTSIZE_POOL=0
    export TFS_GLUE_E8_ATOMIC_DONE=0 TFS_GLUE_E8_SPIN=0
    export TFS_GLUE_E9_INT32_COLIDX=0 TFS_GLUE_E11_VEC_GRAD_DB=0
    export TFS_FWD_V2_SINGLE_SCAN=1 TFS_GLUE_E12_D47_SINGLE_SCAN=1
    export TFS_GLUE_E13_ACTIVE_ROW=1 TFS_GLUE_E14_ACTIVE_ROW_WIDE=0
    export TFS_E13_ACTIVE_MAX_DENSITY=0.25
    # ``auto`` restores the dimension-driven high-D defaults.  The previous
    # v2.3 profile forced all three to zero, which disabled fused db/transpose
    # even for narrow D<=512 layers and regressed Products/IGB-19.
    export TFS_HIGHD_FUSED_DB=auto
    export TFS_HIGHD_FUSED_TRANSPOSE=auto
    export TFS_HIGHD_FUSED_SCALE_TRANSPOSE=auto
    export TFS_HIGHD_CONTIGUOUS_PANELS=auto
    export TFS_COLIDX=auto
    export TFS_MAX_LOCAL_DW_BYTES=0
    export TFS_LOCAL_DW_BUDGET_BYTES=0
    export TFS_HIGHD_NATIVE_SCALE=1
    export TFS_HIGHD_STREAM_AUTO=1
    export TFS_RELEASE_CONTRACT_STRICT=1
    export TFS_HIGHD_RELEASE_MODE=1
    export TFS_RUNTIME_CONTRACT="planner=on highd_stream=on highd_native=auto transform_native=off transform_single_scan=off aggregate_single_scan=off pull_only=on static_hs=on static_aggregate=auto aggregate_saved=${TFS_AGGREGATE_SAVED} numa_first_touch=${TFS_NUMA_FIRST_TOUCH} highd_fused=auto legacy_glue=on budget=67108864 d_tile=256 row_panel=512 panel_budget=33554432 colidx=auto"
    ;;
  ""|debug|legacy)
    # Compatibility/ablation mode.  No release-only fail-fast is enabled.
    export TFS_HIGHD_RELEASE_MODE="${TFS_HIGHD_RELEASE_MODE:-0}"
    export TFS_RELEASE_CONTRACT_STRICT="${TFS_RELEASE_CONTRACT_STRICT:-0}"
    export TFS_RUNTIME_CONTRACT="planner=${TFS_EXEC_PLANNER} highd_stream=${TFS_HIGHD_STREAM_V1:-0} highd_native=${TFS_HIGHD_NATIVE_STREAM:-auto} pull_only=${TFS_PULL_ONLY_BACKWARD:-0} static_hs=${TFS_STATIC_HS} numa_first_touch=${TFS_NUMA_FIRST_TOUCH}"
    ;;
  *)
    echo "unknown TFS_RELEASE_PROFILE=${TFS_RELEASE_PROFILE@Q}; expected final_pre_numa, v2_4_dataflow_generalized, a historical profile, debug, or legacy" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

# Keep the contract available to launchers and Python startup logs without
# requiring every benchmark to reconstruct the individual switches.
export TFS_RUNTIME_CONTRACT
