#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=8
#SBATCH --threads-per-core=1
#SBATCH -t 00:30:00
#SBATCH -J build_plan
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/build_plan_%j.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/build_plan_%j.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
mkdir -p "$root/build_planned_dense/extension" \
  "$root/build_planned_dense/temp" "$root/runs"
python "$root/csrc/setup_backward_opt_aggregate_wide.py" build_ext \
  --build-lib "$root/build_planned_dense/extension" \
  --build-temp "$root/build_planned_dense/temp"
sha256sum "$root"/build_planned_dense/extension/tfs_train_v2_c0_ext*.so
