#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -J audit_fix_test

set -euo pipefail
root="${TFS_ROOT:?TFS_ROOT must point to the audit repair shadow}"
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export PYTHONPATH="$root/python:$root/build/extension${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$HOME/.local/lib/python3.9/site-packages/torch/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$root"
python -m pytest -q \
  tests/test_execution_plan.py \
  tests/test_colidx_cache_contract.py \
  tests/test_supervision_scope_contract.py \
  tests/test_supervision_scope_shadow.py \
  tests/test_bf16_pull_primitive.py \
  tests/test_extension_wide_k_static_cache.py \
  tests/test_compact_dense_shadow_native.py \
  tests/test_highd_strided_slab_native.py \
  tests/test_aggregate_highd_single_scan_native.py \
  tests/test_fused_hidden_bridge_native.py \
  tests/test_compact_q_native.py \
  tests/test_source_panel_q_accumulate.py \
  tests/test_selected_rect_single_scan.py
python tests/test_extension_aggregate_saved_v4_smoke.py
