# TFS-Train `final_pre_numa` manifest

- Version: `final_pre_numa`
- Freeze date: `2026-08-19`
- Base: frozen `v2_3_auto`; `v2_3_dataflow` is a deprecated ablation only.
- Production profile: `TFS_RELEASE_PROFILE=final_pre_numa`
- Production implementation selector: one immutable `execution_variant` per layer.
- Numerical contract: BF16 inputs/intermediates, FP32 accumulation and FP32 master parameters.
- DGL source baseline: stock DGL 2.1.0 `GraphConv(norm="both")`, FP32, direct pull-CSR-as-CSC construction and explicit self-loop.
- Formal TFS launcher: `scripts/run_final_pre_numa_authority_matrix.sh`
- Preflight launcher: `scripts/run_final_pre_numa_gate.sh`
- Formal result root: `/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa_authority_20260819`
- Frozen reused DGL result root: `/home/huangjianqiang_group/hdacp1/data/wzh/authority_v23_triplet_20260818/dgl_stock`

The reused DGL matrix predates the direct-CSC source cleanup and is not rerun,
as explicitly requested.  The new helper is numerically gated and is the only
DGL construction path retained for future runs.  Deep NUMA ownership,
replication, locality scheduling and partitioning remain outside this freeze;
parallel first-touch plus external `localalloc` are baseline allocation
hygiene, not claimed as the later NUMA algorithm.

## 2026-08-26 TC-review candidate updates

This branch now also carries isolated, source-level candidate updates from
`tfs_tc_v3_fix_shadow_20260826`; it is no longer a byte-for-byte copy of the
2026-08-19 timing authority. The changes are numerically gated but must pass
separate performance A/B gates before replacing any reported authority result:

- High-D Q-first keeps `Q` in BF16 at the dense-to-sparse boundary and applies
  destination normalization in the FP32 pull epilogue.
- High-D bias reduction uses tile-local accumulation to avoid a second full
  `grad.sum(0)` scan; `TFS_HIGHD_FUSED_DB=0` remains the ablation arm.
- Sparse source-row software prefetch is strictly opt-in via
  `TFS_SPARSE_PREFETCH=1` and defaults off.
- `scripts/run_p2_highd_smoke.sh` and `tests/test_highd_fused_db_ab.py` are
  numerical gates only, never formal timing launchers.
