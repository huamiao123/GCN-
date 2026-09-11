#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=48G
#SBATCH --exclusive
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=32
#SBATCH --threads-per-core=1
#SBATCH -t 00:45:00
#SBATCH -J source_gs_gate

set -euo pipefail
root="${TFS_ROOT:?TFS_ROOT is required}"
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export PYTHONPATH="$root/python:$root/build/extension${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.9/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
for d in 47 128; do
  for panel in 25000 50000 100003; do
    python "$root/tests/bench_source_panel_gs_shadow.py" \
      --nodes 100003 --degree 12 --k 128 --d "$d" --row-panel "$panel" \
      --threads 32 --warmups 3 --repeats 15 \
      --output "$root/results/gs_panel/source_exclusive_d${d}_r${panel}.json"
  done
done
