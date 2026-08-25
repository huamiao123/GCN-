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
