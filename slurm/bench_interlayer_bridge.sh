#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J hidden_bridge

set -euo pipefail
root="${TFS_ROOT:?TFS_ROOT is required}"
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export PYTHONPATH="$root/python:$root/build/extension${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.9/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
nodes="${NODES:-1000000}"
python "$root/tests/bench_interlayer_bridge.py" \
  --nodes "$nodes" --features "${FEATURES:-128}" --threads 32 \
  --warmups "${WARMUPS:-2}" --repeats "${REPEATS:-9}" \
  --output "${OUTPUT:-$root/results/interlayer_bridge/n${nodes}_k128.json}"
