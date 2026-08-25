#!/usr/bin/env bash
# Isolate the static-Hs BF16 conversion gate on the real IGB-HOM-small shape.
# This is a diagnostic probe; it does not change an authority profile.
# Array 0 uses the scalar conversion and array 1 uses E5 vector conversion.
#SBATCH -p intel_expr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH -t 00:10:00
#SBATCH --array=0-1%2
#SBATCH -J tfs_hs_e5_probe
#SBATCH -o /home/huangjianqiang_group/hdacp1/data/wzh/final_v2_2/slurm/hs_e5_probe_%A_%a.out
#SBATCH -e /home/huangjianqiang_group/hdacp1/data/wzh/final_v2_2/slurm/hs_e5_probe_%A_%a.err

set -euo pipefail
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp

ROOT=${TFS_ROOT:-/home/huangjianqiang_group/hdacp1/data/wzh/final_v2_2}
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) export TFS_GLUE_E5_VEC_HS=0; mode=off ;;
  1) export TFS_GLUE_E5_VEC_HS=1; mode=on ;;
  *) echo "invalid array task" >&2; exit 2 ;;
esac
export TFS_GLUE_E8_RIGHTSIZE_POOL=1
export TFS_NUMA_FIRST_TOUCH=on TFS_WORKER_CPUS=0-31
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close OMP_PLACES=cores
export PYTHONPATH="$ROOT/build/extension:$ROOT/python:$ROOT/csrc"
export LD_LIBRARY_PATH="/home/huangjianqiang_group/hdacp1/.local/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PLACEMENT_BENCH_N=1000000 PLACEMENT_BENCH_K=1024
export PLACEMENT_BENCH_D=128 PLACEMENT_BENCH_THREADS=32
export TFS_E5_PROBE_MODE="$mode"

numactl --cpunodebind=0-3 --localalloc \
  python -u "$ROOT/tests/bench_memory_placement.py"
