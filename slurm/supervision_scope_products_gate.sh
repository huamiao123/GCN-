#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=220G
#SBATCH --exclusive
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=32
#SBATCH --threads-per-core=1
#SBATCH -t 00:45:00
#SBATCH --array=0-1
#SBATCH -J scope_products
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/products_%A_%a.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/products_%A_%a.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
threads_list=(8 32)
threads=${threads_list[${SLURM_ARRAY_TASK_ID:-0}]}
requested_scope_order=${TFS_SCOPE_BACKWARD_ORDER:-y_first}
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp

while IFS='=' read -r variable _; do
  case "$variable" in TFS_*|HYBRID_*|SCOPE_*) unset "$variable" ;; esac
done < <(env)
export TFS_RELEASE_PROFILE=final_pre_numa
source "$root/scripts/tfs_standard_env.sh"
export HYBRID_AMX_FORWARD=1 HYBRID_AMX_BACKWARD=1
export HYBRID_PERSISTENT_HS_CACHE=1
export HYBRID_PATH=hybrid
export PRODUCTS_CACHE=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/ogbn-products_official/processed_tfs_v1/ogbn_products_canonical_v1.pt
export PYTHONPATH="$root/build_supervision/extension:$root/python:$root/csrc"
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
export SCOPE_THREADS=$threads SCOPE_LAYERS=2 SCOPE_WARMUPS=2 SCOPE_REPEATS=5
export TFS_SCOPE_BACKWARD_ORDER=$requested_scope_order
export SCOPE_OUTPUT="$root/runs/products_scope_${TFS_SCOPE_BACKWARD_ORDER}_${threads}t.json"
export TFS_WORKER_CPUS="$(taskset -pc $$ | sed 's/.*: //')"

numactl --cpunodebind=0-3 --localalloc \
  python -u "$root/tests/bench_supervision_scoped_products.py"
