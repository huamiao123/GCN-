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
#SBATCH -t 00:50:00
#SBATCH -J auth_vs_ndw
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/auth_vs_ndw_%j.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/auth_vs_ndw_%j.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
while IFS='=' read -r variable _; do
  case "$variable" in TFS_*|HYBRID_*|SCOPE_*|IGB_*) unset "$variable" ;; esac
done < <(env)
export TFS_RELEASE_PROFILE=final_pre_numa
source "$root/scripts/tfs_standard_env.sh"
export HYBRID_AMX_FORWARD=1 HYBRID_AMX_BACKWARD=1 HYBRID_PERSISTENT_HS_CACHE=1 HYBRID_PATH=hybrid
export IGB_ROOT=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/igb_homogeneous_small
export PYTHONPATH="$root/build_compact_dw/extension:$root/python:$root/csrc:$root/tests"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 SCOPE_THREADS=32
export SCOPE_WARMUPS=1 SCOPE_REPEATS=7
export TFS_SCOPE_BACKWARD_ORDER=q_first TFS_SCOPE_FUSED_DB=1
export TFS_SCOPE_LOGSOFTMAX_OUT=1 TFS_SCOPE_NATIVE_DW=1
export TFS_SCOPE_LOSS_ROW_TILE=300000
export SCOPE_OUTPUT="$root/results/authority_vs_native_dw/igb_d2983_l2_32t.json"
export TFS_WORKER_CPUS="$(taskset -pc $$ | sed 's/.*: //')"
mkdir -p "$root/results/authority_vs_native_dw"
numactl --cpunodebind=0-3 --localalloc python -u \
  "$root/tests/bench_authority_vs_bounded_igb.py"
