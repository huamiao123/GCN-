# TFS-Train `final_pre_numa` server handoff

**State:** source and correctness gate accepted; 48-cell final TFS performance
matrix is still running, so final speedup results are not yet accepted.

**Written:** 2026-08-20 (Asia/Shanghai)

## Authoritative locations

| Item | Location |
|---|---|
| Server source | `/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa` |
| Frozen old source | `/home/huangjianqiang_group/hdacp1/data/wzh/audit_v2_3_buggy` |
| New TFS results | `/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa_authority_20260819/tfs_final_pre_numa` |
| Frozen DGL results | `/home/huangjianqiang_group/hdacp1/data/wzh/authority_v23_triplet_20260818/dgl_stock` |
| Local audit archive | `C:\Users\花喵\Desktop\TFS_PreNUMA_Audit_Package_20260820.zip` |

Use only `final_pre_numa` for current work. `audit_v2_3_buggy` is an immutable
historical snapshot; do not modify it or mix its results with the new matrix.

## Runtime contract

- Profile: `TFS_RELEASE_PROFILE=final_pre_numa`.
- TFS: BF16 inputs/intermediates, FP32 accumulation and FP32 master weights.
- The per-layer immutable `execution_variant` is the only dispatch authority.
- Planner policy may use dimensions, staticness, gradient demand, thread count
  and memory budget; it must not use dataset names.
- Deep NUMA candidates are disabled: `TFS_NUMA_PRIVATE=off`,
  `TFS_NUMA_REDUCE=off`, locality scheduler off and Hs replica off. Parallel
  first-touch and `numactl --localalloc` are retained.
- Dynamic 128→128 selects `native_c3`, not V4 auto.
- DGL is stock DGL 2.1.0 FP32 `GraphConv(norm="both")`; it must not use a TFS
  kernel, cache, planner or TFS-only NUMA implementation.

## Completed acceptance jobs

| Job | Purpose | Result |
|---:|---|---|
| `10048940` | Products 128→128 native-C3 / V4 micro-gate | `COMPLETED`, exit `0:0` |
| `10048941` | profile, planner, numerics, workspace and integrated gate | `COMPLETED`, exit `0:0` |

Gate artifacts:

```text
/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa/reports/gates/
```

These prove implementation/correctness conditions, not final performance.

## Running formal matrix

Slurm `10048942` is a 48-cell TFS-only array, throttle 2:

```text
Products / Arxiv / IGB-HOM-small-19 / IGB-HOM-small-2983
× 2 and 3 layers × 1/2/4/8/16/32 threads = 48 cells
```

Each cell is a fresh cold process, seed 101, exactly 200 train/eval epochs,
with plan/source/profile hashes, CPU/NUMA metadata, RSS, timing markers and
external cold wall time. Do not cancel/resubmit successful cells without
preserving their artifacts and recording a reason.

Monitor:

```bash
ssh hdacp1@176.0.250.88
squeue -j 10048942 -o '%.18i|%.9T|%.20j|%M|%R'
cd /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa
python scripts/check_authority_progress.py \
  /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa_authority_20260819/tfs_final_pre_numa
```

## Actions after all TFS cells finish

1. Progress checker must show `expected=48`, `complete=48`, `incomplete=0`,
   `missing=0`.
2. Validate that frozen DGL has the same 48 successful 200-row cells.
3. Merge:

```bash
python scripts/merge_final_authority_results.py \
  /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa_authority_20260819/tfs_final_pre_numa \
  /home/huangjianqiang_group/hdacp1/data/wzh/authority_v23_triplet_20260818/dgl_stock \
  /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa_authority_20260819/merged
```

4. Run the non-regression comparison in `scripts/`; investigate failures.
5. Only then publish a speedup table, with cold wall and steady epoch timing
   reported separately.

## Mandatory DGL caveat

The direct pull-CSR-as-CSC + explicit-self-loop DGL wrapper passed structural
and numerical gates. Per instruction, its 200-epoch matrix was **not rerun**
after that cleanup. Thus frozen DGL timings must be labeled “stock-DGL timings
from the pre-direct-CSC-cleanup wrapper,” not timings from the cleaned wrapper.

## Important exclusions

- Historical acceleration reports are only in
  `02_pre_generalization_results/`; they are not final_pre_numa results.
- Aggregate/transform single-scan candidates remain non-authority auto paths.
- The old dynamic-graph project `dynamic_tfs_migration_nodata/HANDOFF.md` is
  unrelated to this GCN training task.

## Start here

```text
VERSION_MANIFEST.md
scripts/tfs_standard_env.sh
scripts/run_final_pre_numa_gate.sh
scripts/run_final_pre_numa_authority_matrix.sh
scripts/check_authority_progress.py
scripts/validate_authority_plan.py
scripts/merge_final_authority_results.py
python/tfs_train/execution_plan.py
python/tfs_train/standard_dgl.py
```
