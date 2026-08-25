#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=160G
#SBATCH --exclusive
#SBATCH -t 1-00:00:00
#SBATCH --array=0-4%5
#SBATCH -J tfs_ext_csr
#SBATCH -o /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa/slurm/ext_csr_%A_%a.out
#SBATCH -e /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa/slurm/ext_csr_%A_%a.err

# One-time, dataset-only CSR construction.  Formal cell timing explicitly
# records this as offline preprocessing; it is never repeated per cell.
set -euo pipefail
root=${TFS_ROOT:-/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa}
graphs=(flickr reddit yelp amazon igb_medium)
graph=${graphs[${SLURM_ARRAY_TASK_ID:?}]}
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export PYTHONPATH="$root/python:$root/build/extension:$root/csrc"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close OMP_PLACES=cores
case "$graph" in
  flickr) root_data=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/graphsaint/flickr; kind=graphsaint ;;
  reddit) root_data=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/graphsaint/reddit; kind=graphsaint ;;
  yelp) root_data=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/graphsaint/yelp; kind=graphsaint ;;
  amazon) root_data=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/graphsaint/amazon; kind=graphsaint ;;
  igb_medium) root_data=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/igb/igb_hom_medium; kind=igb ;;
esac
python - "$kind" "$root_data" <<'PY'
import sys, torch
from tfs_train.datasets import load_graphsaint, load_igb_homogeneous
torch.set_num_threads(32)
kind, root = sys.argv[1:]
d = load_graphsaint(root) if kind == "graphsaint" else load_igb_homogeneous(root, size="medium")
print({"nodes": int(d.x.shape[0]), "adjacency_entries": int(d.graph.colidx.numel())})
PY
