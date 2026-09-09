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
#SBATCH -t 00:30:00
#SBATCH -J compact_logits
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/compact_logits_%j.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/compact_logits_%j.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
source "$root/scripts/tfs_standard_env.sh"
export PYTHONPATH="$root/build_compact_logits_v2/extension:$root/python:$root/csrc:$root/tests"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
export TFS_WORKER_CPUS="$(taskset -pc $$ | sed 's/.*: //')"
export TFS_INTERNAL_PROFILE=1
out="$root/results/compact_logits_v2/d2983_32t.json"
mkdir -p "$(dirname "$out")"
numactl --cpunodebind=0-3 --localalloc python -u \
  "$root/tests/bench_compact_logits_amx_shadow.py" --output "$out"
