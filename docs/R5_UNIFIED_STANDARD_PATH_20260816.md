# R5 unified standard path (2026-08-16)

This file records the implementation boundary for
`TFS-Train_统一泛化_NUMA_训练数据流优化总计划_20260816.md`.  It is part of the
source package so a timing report can state exactly which gates were active.

## Standard contract

The Python entry point is `tfs_train.execution_plan.build_layer_plan()` (or
`plan_layers()` for a whole GCN).  It selects the V1, shape-only order:

* `aggregate` when `D >= K`;
* `transform` when `D < K`.

The same plan records padded AMX widths, cache eligibility, single-scan and
active-row gates, int32 CSR eligibility, and whether a saved-pulled backward
contract is legal.  The generic per-forward saved-pulled path is available as
an explicit `aggregate_saved_v4` ablation; the conservative authority path
keeps `TFS_AGGREGATE_SAVED=off` until a complete cross-graph no-regression
matrix exists.  Setting it to `auto` preserves the established V3 static
aggregate cache behavior.  Each
formal detailed run emits one machine-readable line such as:

```
TFS_PLAN layer=0 K=100 D=128 order=aggregate backward=selective_amx compute_dx=0 sparse_fwd_width=128 sparse_bwd_width=128 save_hs=1 save_pulled=0 static_hs=1 static_pulled=0 single_scan=1 active_row=1 colidx=int32:1 dim_path=c3 panel=512
```

The canonical launch defaults are in `scripts/tfs_standard_env.sh`; the
authoritative launcher is `scripts/run_conservative_e2e_20260816.sh`.  It is
the only entry point for the strict matrix: it selects one of the three
dataset loaders, applies the same TFS flags, and runs exactly one native
`dgl.nn.pytorch.GraphConv(norm="both")` path for DGL.  DGL is not given any
of the TFS switches.

The shared wrapper is `python/tfs_train/standard_dgl.py:DGLGCN`; the detailed
entry points no longer carry dataset-specific stock-DGL model copies.  A
strict-template guard rejects the historical edge-weight and `dgl_cached`
variants if they are accidentally selected.

The authority launcher uses `numactl --cpunodebind=0-3 --interleave=0-3` for
both variants by default.  First-touch, locality scheduling, and Hs NUMA
replication are separate ablations and are never silently mixed into the
stock-DGL denominator.

## Implemented in this revision

1. All three formal TFS test entry points use the same planner instead of
   dataset-specific order expressions.  The actual graph node count is passed
   before module construction and the exact same plan is emitted on the first
   forward; there is no construction-time `n=1` placeholder in the authority
   path.
2. The AMX backward pull now has one generic single-CSR-scan loop for every
   `1 <= D <= 128` (up to eight 16-wide blocks).  The old D=47/100 special
   cases remain source-compatible but are no longer required by the standard
   path.  `TFS_SMALL_SINGLE_SCAN=off` is available for an ablation.
3. Generic active-row selection and formal `TFS_ACTIVE_ROW=auto|off|on` are
   wired into the same path.  The existing density threshold remains
   `TFS_E13_ACTIVE_MAX_DENSITY`.
4. BF16 logical-width Hs with a K tail (for example K=100) is packed directly
   per panel with zero-filled AMX tail columns; the previous full `N x Kp`
   conversion buffer is needed only for FP32 Hs.
5. `TFS_COLIDX=auto|int32|int64` controls the cached int32 CSR conversion.
   `auto` validates the values before conversion and never narrows an out-of-
   range index.
6. NUMA topology is now observable in the native profile (`TFS_NUMA` worker
   lines) and the reduction partition derives its workers-per-NUMA value from
   the actual CPU affinity rather than a fixed eight-worker assumption.
7. Generic aggregate-saved V4 now has a real native producer and backward for
   all aggregate-first `D<=128` layers.  It saves per-forward BF16 `P=B*Hs`,
   reuses `P` for dW, and performs only the necessary pull for `dX`; the
   producer is not used for transform-first or wide-D layers.  It is an
   explicit opt-in (`TFS_AGGREGATE_SAVED=on`); the strict common path keeps it
   off until the cross-graph no-regression gate is complete.
8. Backward workspace slabs have a guarded first-touch initializer
   (`TFS_NUMA_FIRST_TOUCH=auto|off|on`).  It is implemented and measurable,
   but the conservative authority path uses `off` because its benefit is
   graph- and placement-dependent.
9. A source-reuse-aware panel scheduler is available through
   `TFS_LOCALITY_SCHEDULE=off|auto|on`.  It builds a 256-bit source-block
   signature, scores NUMA ownership by load/fresh-source/overlap, and then
   balances workers inside the selected NUMA domain.  Forward outputs and
   dX/db are exact; dW remains within the documented FP32 reduction-order
   tolerance.  The standard launch leaves this gate off until graph-family
   measurements establish a universal benefit.
10. Static Hs can be replicated per runtime NUMA ownership group with the
    opt-in native producer `c3_replicate_static_hs_numa_v1`.  The replicated
    `[R,N,K]` tensor is consumed by the existing cached-Hs forward entry point;
    each worker selects its precomputed ownership slot, so the hot loop does
    not rediscover topology.  The Python cache accounts for replica bytes and
    invalidates/rebuilds them with the ordinary O(1) Hs contract.  Exact output
    and transform-first/aggregate-first smoke gates pass.
11. Dimension dispatch is shape-driven rather than environment-gated.  A
    layer is logged as `c3`, `wide_k`, `wide_output`, or `wide_aggregate`:
    `wide_k` covers K>128 with D<=128 (including IGB's 1024->128 first layer),
    while the two wide-output families cover D>128 in the selected order.
    Shared Python wrappers are used by the Products, Arxiv, and IGB detailed
    entry points.  Aggregate-wide backward automatically uses the
    stride-parametric pull-only primitive for K>128, so it does not construct
    an oversized identity matrix or rescan the CSR once per feature tile.

## Existing optimizations retained (not reimplemented)

Persistent Hs V1, the V2 padded-Hs consumer, V3 static aggregate/T0 cache,
AMX wide-K/wide-D wrappers (including their saved-pulled backward), BF16 gradient scaling, pull-only backward, and the
accepted GLUE E1--E14 kernels remain in the extension.  The planner only
selects V3 when its no-`dX` contract is satisfied, and reports V4 only when its
explicit AMX/saved-pulled gate is enabled; the conservative standard leaves
that gate off and otherwise reports the regular selective AMX backward path.
This prevents the old 32-thread V3 table and the
1--16-thread V1 table from being mislabeled as one configuration.

The optional `c3_scale_grad_bf16_db_v2` experiment fuses BF16 gradient
preparation and `db`.  Its numerical gate allows the expected FP32 reduction
order difference, but its isolated 32K x 2983 benchmark was 0.75x of the
existing `c3_scale_grad_bf16_v1`; it therefore remains opt-in and is not in
the standard launch.

## Not yet claimed as complete

The document's later phases still require separate work and gates: HBM/DDR
placement.  The shared node exposes DDR-only NUMA domains, so the HBM branch is
not testable here; the controlled DDR policy result is recorded in
`docs/R5_MEMORY_PLACEMENT_20260816.md`.  Static-Hs NUMA replication is implemented as an auditable opt-in,
but is deliberately not enabled in the standard launch: on the shared
8-NUMA node its 8-thread isolated benchmark was 1.037x only for a large
`N=262144,K=128` fixture, while a smaller `N=65536,K=32` fixture regressed to
0.875x; a two-replica run was effectively neutral (0.999x).  The extra
`R*N*K*2` bytes therefore do not justify a universal default.  The cap
`TFS_HS_REPLICA_MAX` is available for controlled experiments.  Dimension
families now have native and full-graph gates, including the complete IGB
19/2983-class, 2/3-layer, 1/2/4/8/16/32-thread dispatch matrix. The remaining
performance claims that need separate reporting are the longer 200-epoch
authority matrix and any policy-specific NUMA ablation; neither is silently
substituted by the one-step scaling table.

## Verification

* Python planner tests: `tests/test_execution_plan.py` (14 passed locally).
* Native Intel build: `CXX=icpx CC=icx python csrc/setup_backward_opt_aggregate_wide.py build_ext --inplace`.
* AMX smoke gates on the shared node: generic small-D single-scan, active-row,
  padded Products K=100 tail, `tests/test_extension_aggregate_saved_v4_smoke.py`,
  and `tests/test_extension_hs_numa_replica_smoke.py` all pass, including
  `TFS_COLIDX=auto`.  The full Products AMX reference gate
  still reports the pre-existing ~3.3% input-gradient discrepancy; V4 changes
  it by only ~0.004 percentage points and does not alter that threshold.
