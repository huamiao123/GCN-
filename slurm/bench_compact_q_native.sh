#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -J compact_q

set -euo pipefail
root="${TFS_ROOT:?TFS_ROOT is required}"
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export PYTHONPATH="$root/python:$root/build/extension${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.9/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
rows="${ROWS:-100003}"
d="${D:-2983}"
panels="${PANELS:-1}"
python "$root/tests/bench_compact_q_native.py" \
  --rows "$rows" --panels "$panels" --d "$d" --k "${K:-128}" --threads 32 \
  --warmups "${WARMUPS:-2}" --repeats "${REPEATS:-9}" \
  --output "${OUTPUT:-$root/results/compact_q/r${rows}_p${panels}_d${d}.json}"
