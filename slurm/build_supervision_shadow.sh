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
#SBATCH -J build_scope
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/build_scope_%j.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/build_scope_%j.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
mkdir -p "$root/build_compact_logits_v2/extension" "$root/build_compact_logits_v2/temp"
python "$root/csrc/setup_backward_opt_aggregate_wide.py" build_ext \
  --build-lib "$root/build_compact_logits_v2/extension" \
  --build-temp "$root/build_compact_logits_v2/temp"
sha256sum "$root"/build_compact_logits_v2/extension/tfs_train_v2_c0_ext*.so
