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
#SBATCH --array=0-5%4
#SBATCH -J native_dw_t4
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/native_dw_t4_%A_%a.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/native_dw_t4_%A_%a.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
dims=(512 2983)
thread_values=(1 8 32)
index=${SLURM_ARRAY_TASK_ID:?}
dim=${dims[$((index / 3))]}
threads=${thread_values[$((index % 3))]}
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
source "$root/scripts/tfs_standard_env.sh"
export HYBRID_AMX_FORWARD=1 HYBRID_AMX_BACKWARD=1 HYBRID_PERSISTENT_HS_CACHE=1 HYBRID_PATH=hybrid
export IGB_ROOT=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/igb_homogeneous_small
export IGB_LABEL_FILE=node_label_2K.npy IGB_SPLIT_SEED=20260813
export PYTHONPATH="$root/build_compact_dw_t4_safe/extension:$root/python:$root/csrc:$root/tests"
export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
export SCOPE_THREADS=$threads SCOPE_LAYERS=2 SCOPE_WARMUPS=1 SCOPE_REPEATS=7
export SCOPE_OUT_DIM=$dim SCOPE_SYNTHETIC_LABELS=1
export TFS_SCOPE_BACKWARD_ORDER=q_first TFS_SCOPE_FUSED_DB=1 TFS_SCOPE_LOGSOFTMAX_OUT=1
export TFS_SCOPE_LOSS_ROW_TILE=300000
export SCOPE_OUTPUT="$root/results/native_dw_scope_igb_t4/d${dim}_${threads}t.json"
export TFS_WORKER_CPUS="$(taskset -pc $$ | sed 's/.*: //')"
mkdir -p "$root/results/native_dw_scope_igb_t4"
numactl --cpunodebind=0-3 --localalloc python -u \
  "$root/tests/bench_native_dw_scope_igb.py"
