#!/usr/bin/env bash
# Exclusive-node gate for the post-document native Transform-HighD slab.
# This is a component gate, not a four-graph authority E2E run.
#SBATCH -p intel_expr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=240G
#SBATCH --exclusive
#SBATCH -t 00:20:00
#SBATCH -J tfs_transform_native_slab
#SBATCH -o /home/huangjianqiang_group/hdacp1/data/wzh/final_v2_2/slurm/transform_native_slab_%A.out
#SBATCH -e /home/huangjianqiang_group/hdacp1/data/wzh/final_v2_2/slurm/transform_native_slab_%A.err

set -euo pipefail
root="${TFS_ROOT:-${SLURM_SUBMIT_DIR:-/home/huangjianqiang_group/hdacp1/data/wzh/final_v2_2}}"
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export PYTHONPATH="${root}/python:${root}/build/extension"
export LD_LIBRARY_PATH="/home/huangjianqiang_group/hdacp1/.local/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export TFS_GLUE_E2_VEC_STORE=1 TFS_COLIDX=int64
mkdir -p "${root}/reports"

python -u "${root}/tests/run_transform_native_gate.py" \
  > "${root}/reports/transform_native_gate_one_slab_${SLURM_JOB_ID}.json"
for threads in 1 2 4 8 16 32; do
  TFS_HIGHD_THREADS="${threads}" TFS_HIGHD_N=4103 \
    TFS_HIGHD_K=1024 TFS_HIGHD_D=257 TFS_HIGHD_REPEATS=5 \
    python -u "${root}/tests/bench_transform_native_gate.py" \
    > "${root}/reports/transform_native_bench_4103_1024_257_t${threads}_${SLURM_JOB_ID}.json"
done
for threads in 4 16 32; do
  TFS_HIGHD_THREADS="${threads}" TFS_HIGHD_N=4097 \
    TFS_HIGHD_K=1024 TFS_HIGHD_D=513 TFS_HIGHD_REPEATS=5 \
    python -u "${root}/tests/bench_transform_native_gate.py" \
    > "${root}/reports/transform_native_bench_4097_1024_513_t${threads}_${SLURM_JOB_ID}.json"
done
echo "TRANSFORM_NATIVE_SLAB_GATE_DONE job=${SLURM_JOB_ID}"
