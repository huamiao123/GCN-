#!/usr/bin/env bash
#SBATCH -p intel_expr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=220G
#SBATCH --exclusive
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=32
#SBATCH --threads-per-core=1
#SBATCH -t 02:00:00
#SBATCH -J final_pre_numa_gate
#SBATCH -o /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa/slurm/gate_%j.out
#SBATCH -e /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa/slurm/gate_%j.err

set -euo pipefail
root=${TFS_ROOT:-/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa}
# A gate must not inherit an experimental switch from the submitting shell.
while IFS='=' read -r variable _; do
  case "$variable" in TFS_*|HYBRID_*) unset "$variable" ;; esac
done < <(env)
mkdir -p "$root/slurm" "$root/reports/gates"
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export LD_LIBRARY_PATH="/home/huangjianqiang_group/hdacp1/.local/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$root/build/extension:$root/python:$root/csrc"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close OMP_PLACES=cores
export TFS_WORKER_CPUS
TFS_WORKER_CPUS=$(taskset -pc $$ | sed 's/.*: //')

bash "$root/tests/test_final_pre_numa_profile.sh"
export TFS_RELEASE_PROFILE=final_pre_numa
source "$root/scripts/tfs_standard_env.sh"
export HYBRID_AMX_FORWARD=1 HYBRID_AMX_BACKWARD=1
export HYBRID_PERSISTENT_HS_CACHE=1
# One authority-wide cache ceiling.  Eligibility remains shape-derived in the
# cache contract; individual dataset names must not choose resource policy.
export TFS_HS_CACHE_MAX_BYTES=$((3 * 1024 * 1024 * 1024))
python "$root/tests/smoke_final_pre_numa_planner.py"
python -m pytest -q "$root/tests/test_execution_plan.py"
python "$root/tests/smoke_standard_dgl_direct_csc.py" \
  > "$root/reports/gates/dgl_direct_csc.log"
python "$root/tests/test_extension_shape_matrix.py"
python "$root/tests/test_extension_padded_products_tail.py"
python "$root/tests/test_extension_workspace_logical_shape.py"
python "$root/tests/test_extension_dimension_generalization.py"
python "$root/tests/test_extension_generic_single_scan_smoke.py"
python "$root/tests/test_extension_aggregate_cache_smoke.py"
python "$root/tests/test_extension_aggregate_saved_v4_smoke.py"
HYBRID_HIGHD_SMOKE_OUTPUT="$root/reports/gates/highd_v4.json" \
  python "$root/tests/test_highd_v4_extension.py"
HYBRID_HIGHD_SMOKE_OUTPUT="$root/reports/gates/highd_d_slab.json" \
  python "$root/tests/test_highd_d_slab_extension.py"
# Historical candidate timings used unequal thread counts and are invalid for
# auto decisions.  Cycle 5 replaces them with the fair protocol; do not emit
# fresh-looking candidate artifacts from this correctness gate meanwhile.
printf '{"status":"disabled","reason":"awaiting Cycle 5 fair performance protocol"}\n' \
  > "$root/reports/gates/transform_native_candidate.json"
printf '{"status":"disabled","reason":"awaiting Cycle 5 fair performance protocol"}\n' \
  > "$root/reports/gates/transform_single_scan_candidate.json"
/usr/bin/time -v -o "$root/reports/gates/workspace_resource.txt" \
  python "$root/tests/smoke_workspace_contract.py" \
  | tee "$root/reports/gates/workspace_contract.log"

grep -q 'hb_bytes=0,wt_bytes=0,packed_wt_bytes=0' \
  "$root/reports/gates/workspace_contract.log"
grep -Eq 'hb_bytes=0,wt_bytes=[1-9][0-9]*,packed_wt_bytes=[1-9][0-9]*' \
  "$root/reports/gates/workspace_contract.log"
grep -Eq 'hb_bytes=[1-9][0-9]*,wt_bytes=0,packed_wt_bytes=0' \
  "$root/reports/gates/workspace_contract.log"
grep -q 'Maximum resident set size' \
  "$root/reports/gates/workspace_resource.txt"

# Integrated autograd gates use the same immutable planner/dispatch path as
# the 200-epoch models.  The small Products/IGB fixtures cover all production
# dimension families; Arxiv deliberately validates its real canonical graph.
for depth in 2 3 4 5; do
  grad_tol=0.05
  if [[ "$depth" -ge 4 ]]; then grad_tol=0.10; fi
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  HYBRID_CHECK=1 HYBRID_CHECK_FULL=1 HYBRID_LAYERS="$depth" \
  HYBRID_INPUT_DIM=100 HYBRID_HIDDEN_DIM=128 HYBRID_OUT_DIM=47 \
  HYBRID_GRAD_TOL=0.05 HYBRID_FORWARD_TOL=0.01 \
  PRODUCTS_CACHE=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/ogbn-products_official/processed_tfs_v1/ogbn_products_canonical_v1.pt \
  HYBRID_OUTPUT="$root/reports/gates/products_l${depth}_numerics.json" \
    python "$root/tests/hybrid_fullstep_products_detailed.py" \
      > "$root/reports/gates/products_l${depth}_numerics.log"
done

for classes in 19 2983; do
  for depth in 2 3; do
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    HYBRID_CHECK=1 HYBRID_CHECK_FULL=1 HYBRID_LAYERS="$depth" \
    HYBRID_INPUT_DIM=1024 HYBRID_HIDDEN_DIM=128 \
    HYBRID_OUT_DIM="$classes" HYBRID_GRAD_TOL=0.05 \
    HYBRID_FORWARD_TOL=0.01 IGB_SPLIT_SEED=20260813 \
    IGB_ROOT=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/igb_homogeneous_small \
    HYBRID_OUTPUT="$root/reports/gates/igb${classes}_l${depth}_numerics.json" \
      python "$root/tests/hybrid_aggregatewide_bf16grad_strongdgl_v3.py" \
        > "$root/reports/gates/igb${classes}_l${depth}_numerics.log"
  done
done

for depth in 2 3; do
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  HYBRID_CHECK=1 HYBRID_CHECK_FULL=1 HYBRID_LAYERS="$depth" \
  HYBRID_INPUT_DIM=128 HYBRID_HIDDEN_DIM=128 HYBRID_OUT_DIM=40 \
  HYBRID_GRAD_TOL=0.05 HYBRID_FORWARD_TOL=0.01 \
  ARXIV_ROOT=/online1/huangjianqiang_group/hdacp1/wzh/TFS-Train/runs/c5_fullstep_arxiv_20260804_v3/dataset/arxiv \
  HYBRID_OUTPUT="$root/reports/gates/arxiv_l${depth}_numerics.json" \
    python "$root/tests/hybrid_fullstep_arxiv_detailed.py" \
      > "$root/reports/gates/arxiv_l${depth}_numerics.log"
done

printf '{"status":"pass","profile":"final_pre_numa","job_id":"%s"}\n' \
  "${SLURM_JOB_ID:-interactive}" > "$root/reports/gates/status.json"
