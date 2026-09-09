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
#SBATCH --array=0-3
#SBATCH -J terminal_phase
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/terminal_phase_%A_%a.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/terminal_phase_%A_%a.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
dims=(512 1024 2048 2983)
dim=${dims[${SLURM_ARRAY_TASK_ID:?}]}
threads=32
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
source "$root/scripts/tfs_standard_env.sh"
export PYTHONPATH="$root/build_supervision/extension:$root/python:$root/csrc:$root/tests"
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
export TFS_WORKER_CPUS="$(taskset -pc $$ | sed 's/.*: //')"
mkdir -p "$root/results/terminal_phase_breakdown"
numactl --cpunodebind=0-3 --localalloc python -u \
  "$root/tests/bench_terminal_phase_breakdown.py" \
  --classes "$dim" --threads "$threads" --row-tile 300000 \
  --warmups 1 --repeats 5 \
  --output "$root/results/terminal_phase_breakdown/d${dim}_32t.json"
