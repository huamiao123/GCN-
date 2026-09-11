#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=128G
#SBATCH -t 01:30:00
#SBATCH -J widek_sx

set -euo pipefail
root="${TFS_ROOT:?TFS_ROOT is required}"
dataset="${DATASET:?DATASET is required}"
threads="${THREADS:-32}"
data_root="/home/huangjianqiang_group/hdacp1/data/wzh/datasets/graphsaint/$dataset"
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export PYTHONPATH="$root/python:$root/build/extension${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.9/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
export TFS_COLIDX=auto TFS_INTERNAL_PROFILE=0
output="$root/results/wide_k_static/${dataset}_${threads}t.json"
python "$root/tests/bench_wide_k_static_real.py" \
  --root "$data_root" --name "$dataset" --threads "$threads" \
  --warmups 1 --repeats 5 --output "$output"
