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
#SBATCH -J panel_fused_db
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/panel_fused_db_%j.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/panel_fused_db_%j.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export TFS_RELEASE_PROFILE=final_pre_numa
source "$root/scripts/tfs_standard_env.sh"
export HYBRID_AMX_FORWARD=1 HYBRID_AMX_BACKWARD=1
export HYBRID_PERSISTENT_HS_CACHE=1 HYBRID_PATH=hybrid
export IGB_ROOT=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/igb_homogeneous_small
export PYTHONPATH="$root/build_supervision/extension:$root/python:$root/csrc:$root/tests"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
export SCOPE_THREADS=32 SCOPE_WARMUPS=2 SCOPE_REPEATS=10
export TFS_SCOPE_LOSS_ROW_TILE=300000 TFS_SCOPE_BACKWARD_ORDER=q_first
export SCOPE_OUTPUT="$root/results/panelized_fused_db_igb_d2983_32t.json"
export TFS_WORKER_CPUS="$(taskset -pc $$ | sed 's/.*: //')"
numactl --cpunodebind=0-3 --localalloc \
  python -u "$root/tests/bench_panelized_fused_db_igb.py"
