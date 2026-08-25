#!/usr/bin/env bash
#SBATCH -p intel_expr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=100G
#SBATCH -t 2:00:00
#SBATCH -J igb_medium_smoke
#SBATCH -o /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa/reports/shared_smoke/igb_medium_%j.out
#SBATCH -e /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa/reports/shared_smoke/igb_medium_%j.err

# Compatibility only: one genuine step on a shared node, never a benchmark.
set -euo pipefail
root=/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa
mkdir -p "$root/reports/shared_smoke"
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export IGB_ROOT=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/igb/igb_hom_medium
export IGB_SIZE=medium
export IGB_SMOKE_OUTPUT="$root/reports/shared_smoke/igb_medium_shared_result.json"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$root/python:$root/build/extension"
python -u "$root/tests/smoke_igb_medium_real.py"
