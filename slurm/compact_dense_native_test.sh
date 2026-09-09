#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=2
#SBATCH --threads-per-core=1
#SBATCH -t 00:10:00
#SBATCH -J compact_test
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/compact_test_%j.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/compact_test_%j.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
source "$root/scripts/tfs_standard_env.sh"
export PYTHONPATH="$root/build_compact_logits_v2/extension:$root/python:$root/csrc:$root/tests"
export TFS_WORKER_CPUS="$(taskset -pc $$ | sed 's/.*: //')"
python -m pytest -q "$root/tests/test_compact_dense_shadow_native.py"
