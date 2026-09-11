#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J pack_once

set -euo pipefail
root="${TFS_ROOT:?TFS_ROOT is required}"
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.9/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TFS_INTERNAL_PROFILE=0
mkdir -p "$root/results/compact_logits_packed"
PYTHONPATH="$root/build/extension:$root/python" python \
  "$root/tests/bench_compact_logits_packed.py" \
  --rows "${BENCH_ROWS:-4096}" \
  --warmups "${BENCH_WARMUPS:-2}" \
  --repeats "${BENCH_REPEATS:-7}" \
  --output "$root/results/compact_logits_packed/${BENCH_RESULT_NAME:-results.json}"
