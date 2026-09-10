#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=200G
#SBATCH --exclusive
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=32
#SBATCH --threads-per-core=1
#SBATCH -t 01:20:00
#SBATCH --array=0-23%6
#SBATCH -J plan_matrix
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/plan_matrix_%A_%a.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/plan_matrix_%A_%a.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
threads_all=(1 2 4 8 16 32)
index=${SLURM_ARRAY_TASK_ID:?}
dataset_index=$((index / 12))
within=$((index % 12))
layers=$((2 + within / 6))
threads=${threads_all[$((within % 6))]}
if [[ $dataset_index -eq 0 ]]; then
  dataset=ogbn-products
  slug=products
else
  dataset=igb-hom-small
  slug=igb_small_d2983
fi

module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
while IFS='=' read -r variable _; do
  case "$variable" in TFS_*|HYBRID_*|SCOPE_*|IGB_*|PRODUCTS_*) unset "$variable" ;; esac
done < <(env)
export TFS_RELEASE_PROFILE=final_pre_numa
source "$root/scripts/tfs_standard_env.sh"
export HYBRID_AMX_FORWARD=1 HYBRID_AMX_BACKWARD=1
export HYBRID_PERSISTENT_HS_CACHE=1 HYBRID_PATH=hybrid
export IGB_ROOT=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/igb_homogeneous_small
export PRODUCTS_CACHE=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/ogbn-products_official/processed_tfs_v1/ogbn_products_canonical_v1.pt
export PYTHONPATH="$root/build_planned_dense/extension:$root/python:$root/csrc:$root/tests"
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
export SCOPE_THREADS=$threads SCOPE_LAYERS=$layers SCOPE_DATASET=$dataset
export SCOPE_WARMUPS=2 SCOPE_REPEATS=9 SCOPE_USE_DENSE_PLAN=1
export SCOPE_OUTPUT="$root/results/authority_vs_planned_dense_matrix/${slug}_l${layers}_t${threads}.json"
export TFS_WORKER_CPUS="$(taskset -pc $$ | sed 's/.*: //')"
mkdir -p "$root/results/authority_vs_planned_dense_matrix" "$root/runs"
numactl --cpunodebind=0-3 --localalloc python -u \
  "$root/tests/bench_authority_vs_bounded_igb.py"
