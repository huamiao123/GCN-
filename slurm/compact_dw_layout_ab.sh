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
#SBATCH -t 00:30:00
#SBATCH --array=0-2%3
#SBATCH -J dw_layout_ab
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/dw_layout_ab_%A_%a.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/dw_layout_ab_%A_%a.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
thread_values=(1 8 32)
threads=${thread_values[${SLURM_ARRAY_TASK_ID:?}]}
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
source "$root/scripts/tfs_standard_env.sh"
export PYTHONPATH="$root/build_compact_dw_layout_ab/extension:$root/python:$root/csrc:$root/tests"
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
export TFS_WORKER_CPUS="$(taskset -pc $$ | sed 's/.*: //')"
out="$root/results/compact_dw_layout_ab/d2983_${threads}t.json"
mkdir -p "$(dirname "$out")"
numactl --cpunodebind=0-3 --localalloc python -u \
  "$root/tests/bench_compact_dw_layout_ab.py" \
  --threads "$threads" --output "$out"
