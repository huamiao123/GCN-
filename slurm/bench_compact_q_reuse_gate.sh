#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J compact_q_gate

set -euo pipefail
root="${TFS_ROOT:?TFS_ROOT is required}"
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export PYTHONPATH="$root/python:$root/build/extension${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.9/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32

for panels in 1 2 3; do
  python "$root/tests/bench_compact_q_native.py" \
    --rows 300000 --panels "$panels" --d 2983 --k 128 --threads 32 \
    --warmups 2 --repeats 7 \
    --output "$root/results/compact_q/reuse_r300000_p${panels}_d2983.json"
done

for d in 512 1024 2048; do
  python "$root/tests/bench_compact_q_native.py" \
    --rows 300000 --panels 2 --d "$d" --k 128 --threads 32 \
    --warmups 2 --repeats 7 \
    --output "$root/results/compact_q/shape_r300000_p2_d${d}.json"
done
