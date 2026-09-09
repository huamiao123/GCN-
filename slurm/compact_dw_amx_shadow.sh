#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=120G
#SBATCH --exclusive
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=32
#SBATCH --threads-per-core=1
#SBATCH -t 01:00:00
#SBATCH --array=0-11%2
#SBATCH -J compact_dw_t4
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/compact_dw_t4_%A_%a.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/compact_dw_t4_%A_%a.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
classes=(512 1024 2048 2983)
thread_values=(1 8 32)
index=${SLURM_ARRAY_TASK_ID:?}
dim=${classes[$((index / 3))]}
threads=${thread_values[$((index % 3))]}
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
source "$root/scripts/tfs_standard_env.sh"
export PYTHONPATH="$root/build_compact_dw_t4/extension:$root/python:$root/csrc:$root/tests"
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
export TFS_WORKER_CPUS="$(taskset -pc $$ | sed 's/.*: //')"
export TFS_INTERNAL_PROFILE=1
out="$root/results/compact_dw_amx_t4/d${dim}_${threads}t.json"
mkdir -p "$(dirname "$out")"
numactl --cpunodebind=0-3 --localalloc python -u \
  "$root/tests/bench_compact_dw_amx_shadow.py" \
  --classes "$dim" --threads "$threads" --output "$out"
