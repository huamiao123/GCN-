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
#SBATCH -J no_gs_q_gate

set -euo pipefail
root="${TFS_ROOT:?TFS_ROOT is required}"
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export PYTHONPATH="$root/python:$root/build/extension:$root/tests${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.9/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
for d in 47 128; do
  for panel in 25000 50000; do
    python "$root/tests/bench_no_global_gs_q_shadow.py" \
      --nodes 100003 --half-degree 6 --k 128 --d "$d" \
      --row-panel "$panel" --threads 32 --warmups 2 --repeats 11 \
      --output "$root/results/gs_panel/no_global_d${d}_r${panel}.json"
  done
done
