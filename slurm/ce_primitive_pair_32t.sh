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
#SBATCH --array=0-3
#SBATCH -J ce_primitive
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/ce_primitive_%A_%a.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/ce_primitive_%A_%a.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
dims=(512 1024 2048 2983)
dim=${dims[${SLURM_ARRAY_TASK_ID:?}]}
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
source "$root/scripts/tfs_standard_env.sh"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
mkdir -p "$root/results/ce_primitive_pair"
numactl --cpunodebind=0-3 --localalloc python -u \
  "$root/tests/bench_ce_primitive_pair.py" --classes "$dim" --threads 32 \
  --warmups 1 --repeats 7 \
  --output "$root/results/ce_primitive_pair/d${dim}_32t.json"
