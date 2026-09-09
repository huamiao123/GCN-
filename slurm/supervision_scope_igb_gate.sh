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
#SBATCH -t 01:00:00
#SBATCH --array=0-2%3
#SBATCH -J scope_igb_gate
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/igb_gate_%A_%a.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/igb_gate_%A_%a.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) label=node_label_19.npy; out_dim=19; order=y_first ;;
  1) label=node_label_19.npy; out_dim=19; order=authority_active ;;
  2) label=node_label_2K.npy; out_dim=2983; order=q_first ;;
  *) exit 2 ;;
esac
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
while IFS='=' read -r variable _; do
  case "$variable" in TFS_*|HYBRID_*|SCOPE_*|IGB_*) unset "$variable" ;; esac
done < <(env)
export TFS_RELEASE_PROFILE=final_pre_numa
source "$root/scripts/tfs_standard_env.sh"
export HYBRID_AMX_FORWARD=1 HYBRID_AMX_BACKWARD=1
export HYBRID_PERSISTENT_HS_CACHE=1 HYBRID_PATH=hybrid
export IGB_ROOT=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/igb_homogeneous_small
export IGB_LABEL_FILE=$label IGB_SPLIT_SEED=20260813
export PYTHONPATH="$root/build_supervision/extension:$root/python:$root/csrc"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
export SCOPE_THREADS=32 SCOPE_LAYERS=2 SCOPE_WARMUPS=1 SCOPE_REPEATS=5
export SCOPE_OUT_DIM=$out_dim
export TFS_SCOPE_BACKWARD_ORDER=$order
stem=${label%.npy}
export SCOPE_OUTPUT="$root/runs/igb_scope_gate_${stem}_${order}_l2_32t.json"
export TFS_WORKER_CPUS="$(taskset -pc $$ | sed 's/.*: //')"
numactl --cpunodebind=0-3 --localalloc \
  python -u "$root/tests/bench_supervision_scoped_igb.py"
