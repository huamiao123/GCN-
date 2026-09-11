#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 00:30:00
#SBATCH -J scan_ab

set -euo pipefail
root="${TFS_ROOT:?TFS_ROOT is required}"
baseline_ext="${BASELINE_EXT:?BASELINE_EXT is required}"
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.9/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TFS_INTERNAL_PROFILE=0
mkdir -p "$root/results/selected_rect_single_scan"
PYTHONPATH="$baseline_ext:$root/python" python \
  "$root/tests/bench_selected_rect_single_scan.py" --label baseline \
  --output "$root/results/selected_rect_single_scan/baseline.json"
PYTHONPATH="$root/build/extension:$root/python" python \
  "$root/tests/bench_selected_rect_single_scan.py" --label single_scan \
  --output "$root/results/selected_rect_single_scan/single_scan.json"
