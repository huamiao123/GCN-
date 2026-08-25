#!/usr/bin/env bash
#SBATCH -p intel_expr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=220G
#SBATCH --exclusive
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=32
#SBATCH --threads-per-core=1
#SBATCH -t 02:00:00
#SBATCH -J fpnuma_128gate
#SBATCH -o /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa/slurm/128gate_%j.out
#SBATCH -e /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa/slurm/128gate_%j.err

set -euo pipefail
root=${TFS_ROOT:-/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa}
while IFS='=' read -r variable _; do
  case "$variable" in TFS_*|HYBRID_*) unset "$variable" ;; esac
done < <(env)
mkdir -p "$root/slurm" "$root/reports/gates"
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export LD_LIBRARY_PATH="/home/huangjianqiang_group/hdacp1/.local/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$root/build/extension:$root/python:$root/csrc"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close OMP_PLACES=cores
export TFS_WORKER_CPUS
TFS_WORKER_CPUS=$(taskset -pc $$ | sed 's/.*: //')
export TFS_RELEASE_PROFILE=final_pre_numa
source "$root/scripts/tfs_standard_env.sh"
[[ "$TFS_PROFILE_STATUS" == authority ]]
export HYBRID_AMX_FORWARD=1 HYBRID_AMX_BACKWARD=1
export PRODUCTS_CACHE=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/ogbn-products_official/processed_tfs_v1/ogbn_products_canonical_v1.pt
export TFS_MICRO_OUTPUT="$root/reports/gates/products_128_native_c3_vs_v4.json"
numactl --cpunodebind=0-3 --localalloc \
  python "$root/tests/bench_products_128_authority_gate.py" \
  > "$root/reports/gates/products_128_native_c3_vs_v4.log"
