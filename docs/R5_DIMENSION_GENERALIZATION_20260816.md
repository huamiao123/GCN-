# R5 dimension generalization report (2026-08-16)

## Decision

Dimension adaptation is now a shape-driven standard contract.  The planner
emits one of four paths without requiring `HYBRID_WIDE_K_AMX`:

| Path | Shape/order | Example |
|---|---|---|
| `c3` | `K<=128`, `D<=128` | Products/Arxiv hidden layer |
| `wide_k` | `K>128`, `D<=128` | IGB `1024 -> 128` first layer |
| `wide_output` | transform-first, `D>128` | high-K/high-output transform layer |
| `wide_aggregate` | aggregate-first, `D>128` | IGB `128 -> 2983` final layer |

This is a unified dispatch interface, not a claim that every shape uses one
identical tile kernel.  The specialized kernels preserve the low-dimensional
fast path and are selected only when the layer shape requires them.

## Implementation

* `python/tfs_train/execution_plan.py` records `dim_path` in every `TFS_PLAN`
  line and exposes `wide_k`, `wide_output`, and `wide_aggregate` properties.
* `python/tfs_train/dimension_dispatch.py` contains the shared high-output
  autograd wrappers used by the Products, Arxiv, and IGB detailed entry points.
* The ordinary C3 consumer and backward kernels accept high logical K with
  `D<=128`; the old Python-only `HYBRID_WIDE_K_AMX` gate is no longer needed.
* `c3_pull_only_amx_v1` now accepts arbitrary `K` (16-wide chunks plus a
  scalar tail).  Aggregate-wide backward automatically uses it for `K>128`,
  avoiding an oversized identity matrix and repeated CSR scans.

## Correctness gates

The shared-node Intel build passed:

* `tests/test_extension_dimension_generalization.py`: native 1024->128
  forward/backward, 2983-output forward/backward, and K=256 pull-only
  contract.
* Products fixture with `K=1024, hidden=256, D=256`: both `wide_output` and
  `wide_aggregate` paths completed the forward/backward gate.
* IGB fixture with `K=1024, hidden=128, D=2983`: plan logs showed
  `wide_k` followed by `wide_aggregate`, and the gate completed with the
  documented relaxed mixed-precision gradient tolerance.

The strict AMX reference gate remains unchanged.  On the small IGB fixture,
the input-gradient relative error is about 3.24% for the 2983-class case and
about 3.15% for the 19-class case, slightly above the existing 3% mixed-
precision threshold.  This is the known BF16/FP32 accumulation discrepancy;
the native dimension contracts are exact and the threshold was not loosened.

## Steady-state shape measurements

Shared node, 8 threads, `N=4096`, cached-Hs consumer; these are component
times, not TFS/DGL speedups:

| Case | Forward (ms) | Backward (ms) |
|---|---:|---:|
| `c3`, 128->128 | 0.347 | 1.194 |
| `wide_k`, 1024->128 | 1.430 | 6.379 |
| `wide_aggregate`, 128->2983 | 5.048 | 106.718 |
| `wide_aggregate`, 256->256 | 0.671 | 4.926 |

The high-dimensional paths are therefore functional and measurable, but their
absolute cost naturally grows with feature/class width.  Final claims still
require the full-graph 1/2/4/8/16/32-thread matrix.

## Full-graph gate and cache accounting

The first full-graph acceptance array (`scripts/validate_dimension_igb_full_20260816.sh`)
completed on the shared node for IGB-HOM-small 19/2983 classes and 2/3 layers.
All four TFS and native-DGL stock gates returned zero.  The logs explicitly
show `wide_k` for `1024 -> 128`, `c3` for hidden layers, and
`wide_aggregate` for `128 -> 2983`; no C2 fallback was used.

The gate also caught and fixed a cache-budget accounting defect: for an
unpadded Hs cache, `forward_tensor` aliases the aligned Hs tensor and must
not be counted as a second allocation.  The corrected contract reports
2,048,000,000 bytes for the 1M x 1024 BF16 Hs cache; padded views and distinct
NUMA replicas remain fully budgeted.  The regression is covered by
`test_unpadded_forward_view_is_not_double_counted`.

The end-to-end NUMA comparison is submitted separately as
`scripts/run_numa_e2e_igb_20260816.sh`: both `localalloc + first-touch` and
`interleave + first-touch` run the same 200-epoch TFS persistent-Hs and
native-DGL stock workloads.  Its wall times are intentionally not folded into
the dimension component table or the authoritative thread speedup table; the
completed policy/class results are recorded separately in
`docs/R5_MEMORY_PLACEMENT_20260816.md`.

## Full-graph thread dispatch matrix

The 24-point array `scripts/run_dimension_thread_matrix_igb_20260816.sh`
completed with all TFS/DGL gates and timing processes returning zero. Each cell
is `TFS ms / DGL-stock ms (DGL/TFS)` from the one measured step after two
warmups. Both frameworks used the same four-node interleave policy; TFS used
the current persistent-Hs/AMX path. These are dispatch/scaling measurements,
not 200-epoch wall times.

| classes/layers | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads | 32 threads |
|---|---:|---:|---:|---:|---:|---:|
| 19 / 2 | 3919.1 / 8554.2 (2.183x) | 2055.5 / 4452.1 (2.166x) | 1066.2 / 2266.0 (2.125x) | 567.9 / 1193.7 (2.102x) | 313.0 / 642.8 (2.054x) | 193.1 / 397.2 (2.057x) |
| 19 / 3 | 6328.9 / 11965.1 (1.891x) | 3313.3 / 6264.4 (1.891x) | 1711.5 / 3188.1 (1.863x) | 895.8 / 1660.9 (1.854x) | 504.8 / 909.4 (1.802x) | 311.9 / 606.3 (1.944x) |
| 2983 / 2 | 28372.6 / 49588.3 (1.748x) | 19388.8 / 30529.8 (1.575x) | 9849.0 / 15447.1 (1.568x) | 5184.7 / 8011.5 (1.545x) | 2714.3 / 4193.8 (1.545x) | 1594.8 / 2465.8 (1.546x) |
| 2983 / 3 | 29899.2 / 52341.7 (1.751x) | 20843.0 / 32611.3 (1.565x) | 10621.2 / 16558.2 (1.559x) | 5456.3 / 8407.7 (1.541x) | 2933.6 / 4482.1 (1.528x) | 1682.9 / 2624.5 (1.560x) |

The 1024-wide first layer and 2983-wide output layer remain on AMX in every
thread point. The lower high-class speedup is a shape effect (the dense
2983-column output dominates), not a dimension-dispatch failure. The 24
status files all report `tfs_gate_exit=0`, `dgl_gate_exit=0`,
`tfs_time_exit=0`, and `dgl_time_exit=0`.

## Why the 2983-class step is still large

The unprofiled 32-thread measured rows separate the cause from the dispatch:

| classes/layers | path | forward | loss | backward | total |
|---|---|---:|---:|---:|---:|
| 19 / 2 | TFS | 87.5 ms | 7.9 ms | 96.8 ms | 193.1 ms |
| 2983 / 2 | TFS | 326.2 ms | 198.2 ms | 1069.1 ms | 1594.8 ms |

The 2983 output creates about 2.983 billion logits per step. The loss alone
therefore grows from 7.9 ms to 198.2 ms and is common to both frameworks.
More importantly, the current high-D aggregate backward still forms `dW` and
`d_pulled` through ATen matrix multiplies in
`python/tfs_train/dimension_dispatch.py`, followed by the native pull-only
kernel. A 32-thread profile measured layer-1 forward at about 226 ms and its
parameter-gradient portion at about 341 ms; the complete backward was about
1049 ms because it also includes the dense `d_pulled` product and reduction.

Thus the previous sparse/AMX/Hs optimizations are active, but the high-D
backward dense GEMM is an identified optimization gap. Improving this path
requires a native AMX dense `dW/d_pulled` implementation (or an equivalent
fused kernel); changing NUMA policy or rebuilding the first-layer Hs cache
cannot remove that `N x 2983` work.
