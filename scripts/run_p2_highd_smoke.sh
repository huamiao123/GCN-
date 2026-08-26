#!/usr/bin/env bash
# Isolated numerical gate for the P2 High-D BF16-Q pull path.
# This is a correctness smoke, never an authority timing launcher.
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --exclusive
#SBATCH -J tfs_p2_gate
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_tc_v3_fix_shadow_20260826/runs/p2_highd_smoke_20260826/slurm-%j.out

set -euo pipefail

root=${TFS_ROOT:-/online1/huangjianqiang_group/hdacp1/wzh/tfs_tc_v3_fix_shadow_20260826}
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
mkdir -p "$root/runs/p2_highd_smoke_20260826"
export PYTHONPATH="$root/csrc:$root/python"
export HYBRID_AMX_FORWARD=1 HYBRID_AMX_BACKWARD=1
export TFS_HIGHD_N=1024 TFS_HIGHD_THREADS=32
export HYBRID_HIGHD_SMOKE_OUTPUT="$root/runs/p2_highd_smoke_20260826/result.json"
numactl --cpunodebind=0-3 --localalloc \
  python "$root/tests/test_highd_v4_extension.py"
numactl --cpunodebind=0-3 --localalloc \
  python "$root/tests/test_highd_fused_db_ab.py"
