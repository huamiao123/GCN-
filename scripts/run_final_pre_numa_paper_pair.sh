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
#SBATCH -t 2-00:00:00
#SBATCH --array=0-47%2
#SBATCH -J paper_v1_pair_200e
#SBATCH -o /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa/slurm/paper_pair_%A_%a.out
#SBATCH -e /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa/slurm/paper_pair_%A_%a.err

# Each method is a separate cold Python process, but both run in this one
# exclusive allocation.  This is the required pairing for host/CPU/NUMA
# comparable TFS-vs-stock-DGL evidence.
set -euo pipefail

root=${TFS_ROOT:-/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa}
launcher="$root/scripts/run_final_pre_numa_authority_matrix.sh"
[[ -x "$launcher" || -f "$launcher" ]] || {
  echo "missing authority launcher: $launcher" >&2; exit 2;
}

AUTHORITY_METHOD=tfs_final_pre_numa bash "$launcher"
AUTHORITY_METHOD=dgl_stock bash "$launcher"
