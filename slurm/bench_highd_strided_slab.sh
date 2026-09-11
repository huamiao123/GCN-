#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J slab_view

set -euo pipefail
root="${TFS_ROOT:?TFS_ROOT is required}"
family="${FAMILY:?FAMILY is required}"
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export PYTHONPATH="$root/python:$root/build/extension${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.9/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 TFS_COLIDX=auto
python "$root/tests/bench_highd_strided_slab.py" --family "$family" \
  --threads 32 --output "$root/results/highd_strided_slab/${family}.json"
