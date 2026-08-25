#!/usr/bin/env bash
#SBATCH -p intel_expr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=240G
#SBATCH --exclusive
#SBATCH -t 00:30:00
#SBATCH -J tfs_highd_v4_smoke
#SBATCH -o /home/huangjianqiang_group/hdacp1/data/wzh/final_v2_1/slurm/highd_v4_smoke_%A.out
#SBATCH -e /home/huangjianqiang_group/hdacp1/data/wzh/final_v2_1/slurm/highd_v4_smoke_%A.err

set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TFS_RELEASE_PROFILE=v2_highd_native
source "${root}/scripts/tfs_standard_env.sh"
ext="${TFS_EXTENSION_ROOT:-${root}/build/extension}"
out="${root}/runs/highd_v4_smoke.json"

module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export PYTHONPATH="${root}/python:${root}/csrc:${ext}"
export LD_LIBRARY_PATH="/home/huangjianqiang_group/hdacp1/.local/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close OMP_PLACES=cores
export HYBRID_AMX_FORWARD=1 HYBRID_AMX_BACKWARD=1 TFS_PULL_ONLY_BACKWARD=1
export TFS_HIGHD_STREAM_STRICT=1 TFS_HIGHD_LOG_FALLBACK=1 TFS_HIGHD_LOG_PLAN=1
export TFS_HIGHD_BWD_BUDGET_BYTES=$((64*1024*1024))
export TFS_GLUE_E1_FUSED_DB=0 TFS_GLUE_E11_VEC_GRAD_DB=0
export TFS_GLUE_E2_VEC_STORE=0 TFS_GLUE_E5_VEC_HS=0
export TFS_GLUE_E7_FORWARD_SCHEDULE=0 TFS_GLUE_E9_INT32_COLIDX=0
export TFS_FWD_V2_SINGLE_SCAN=0 TFS_NUMA_FIRST_TOUCH=on
export HYBRID_HIGHD_SMOKE_OUTPUT="${out}"

mkdir -p "${root}/results"
python -u "${root}/tests/test_highd_v4_extension.py"
