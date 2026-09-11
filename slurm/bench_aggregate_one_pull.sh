#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J agg_one_pull

set -euo pipefail
root="${TFS_ROOT:?TFS_ROOT is required}"
d="${D:-513}"
d_tile="${D_TILE:-257}"
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export PYTHONPATH="$root/python:$root/build/extension${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.9/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 TFS_COLIDX=auto
csr_args=()
if [[ -n "${CSR_PATH:-}" ]]; then
  csr_args=(--csr "$CSR_PATH")
fi
python "$root/tests/run_aggregate_single_scan_production_gate.py" \
  --nodes "${NODES:-100003}" --degree "${DEGREE:-12}" \
  --k 128 --d "$d" --d-tile "$d_tile" --threads 32 \
  --warmups "${WARMUPS:-1}" --repeats "${REPEATS:-7}" \
  "${csr_args[@]}" \
  --output "${OUTPUT:-$root/results/aggregate_one_pull/d${d}.json}"
