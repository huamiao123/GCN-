#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --sockets-per-node=1
#SBATCH --threads-per-core=1
#SBATCH -t 00:30:00
#SBATCH -J scope_build_gate
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/build_gate_%j.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/build_gate_%j.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
mkdir -p "$root/build_supervision/extension" "$root/build_supervision/temp" "$root/runs"
python "$root/csrc/setup_backward_opt_aggregate_wide.py" \
  build_ext --build-lib "$root/build_supervision/extension" \
  --build-temp "$root/build_supervision/temp"
export PYTHONPATH="$root/build_supervision/extension:$root/python:$root/csrc"
export TFS_WORKER_CPUS="$(taskset -pc $$ | sed 's/.*: //')"
export HYBRID_AMX_FORWARD=1 HYBRID_AMX_BACKWARD=1
export TFS_GLUE_E2_VEC_STORE=1 TFS_FWD_V2_SINGLE_SCAN=1
python -m pytest -q "$root/tests/test_supervision_scope_shadow.py"
sha256sum "$root/build_supervision/extension"/tfs_train_v2_c0_ext*.so
