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
#SBATCH -t 00:20:00
#SBATCH --array=0-1
#SBATCH -J phase_logsm
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/phase_logsm_%A_%a.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/phase_logsm_%A_%a.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
dims=(512 2983)
dim=${dims[${SLURM_ARRAY_TASK_ID:?}]}
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
source "$root/scripts/tfs_standard_env.sh"
export PYTHONPATH="$root/build_supervision/extension:$root/python:$root/csrc:$root/tests"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
export TFS_WORKER_CPUS="$(taskset -pc $$ | sed 's/.*: //')"
mkdir -p "$root/results/terminal_phase_logsoftmax"
numactl --cpunodebind=0-3 --localalloc python -u \
  "$root/tests/bench_terminal_phase_breakdown.py" --classes "$dim" \
  --threads 32 --row-tile 300000 --warmups 1 --repeats 7 \
  --logsoftmax-out \
  --output "$root/results/terminal_phase_logsoftmax/d${dim}_32t.json"
