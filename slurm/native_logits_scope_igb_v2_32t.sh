#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=160G
#SBATCH --exclusive
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=32
#SBATCH --threads-per-core=1
#SBATCH -t 00:45:00
#SBATCH -J logits_v2_gate
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/logits_v2_gate_%j.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/logits_v2_gate_%j.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
source "$root/scripts/tfs_standard_env.sh"
export HYBRID_AMX_FORWARD=1 HYBRID_AMX_BACKWARD=1 HYBRID_PERSISTENT_HS_CACHE=1 HYBRID_PATH=hybrid
export IGB_ROOT=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/igb_homogeneous_small
export IGB_LABEL_FILE=node_label_2K.npy IGB_SPLIT_SEED=20260813
export PYTHONPATH="$root/build_compact_logits_v2/extension:$root/python:$root/csrc:$root/tests"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 SCOPE_THREADS=32
export SCOPE_LAYERS=2 SCOPE_WARMUPS=1 SCOPE_REPEATS=7 SCOPE_OUT_DIM=2983
export SCOPE_SYNTHETIC_LABELS=1 SCOPE_AB_KIND=logits
export TFS_SCOPE_BACKWARD_ORDER=q_first TFS_SCOPE_FUSED_DB=1
export TFS_SCOPE_LOGSOFTMAX_OUT=1 TFS_SCOPE_LOSS_ROW_TILE=300000
export SCOPE_OUTPUT="$root/results/native_logits_scope_igb_v2/d2983_32t.json"
export TFS_WORKER_CPUS="$(taskset -pc $$ | sed 's/.*: //')"
mkdir -p "$root/results/native_logits_scope_igb_v2"
numactl --cpunodebind=0-3 --localalloc python -u \
  "$root/tests/bench_native_dw_scope_igb.py"
