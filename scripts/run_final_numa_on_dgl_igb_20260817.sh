#!/usr/bin/env bash
# Final paired experiment: integrated TFS NUMA-ON vs native DGL stock.
#
# This is the single release launcher for the 2026-08-16 NUMA comparison:
#   - TFS: current persistent-Hs/AMX implementation, first-touch enabled;
#   - DGL: native stock GraphConv (norm="both"), no TFS implementation;
#   - both processes use the same external CPU and localalloc policy;
#   - full-graph 200-epoch wall time, including startup and evaluation.
#
# Array points: 0 = IGB-HOM-small/19 classes, 1 = /2983 classes.
# The dataset path, output root, and server checkout can be overridden by env.
#
#SBATCH -p intel_expr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH --array=0-1%1
#SBATCH -J final_tfs_numa_on_dgl
#SBATCH -o /home/huangjianqiang_group/hdacp1/data/wzh/final_v2_2/slurm/final_tfs_numa_on_dgl_%A_%a.out
#SBATCH -e /home/huangjianqiang_group/hdacp1/data/wzh/final_v2_2/slurm/final_tfs_numa_on_dgl_%A_%a.err

set -euo pipefail

# Resolve the release checkout by default so the launcher remains reproducible
# after this directory is copied to a different server location.  The
# explicit override is retained for clusters that stage the source elsewhere.
script_file="$(readlink -f -- "${BASH_SOURCE[0]}")"
script_dir="$(cd -- "$(dirname -- "$script_file")/.." && pwd)"
root=${FINAL_TFS_ROOT:-$script_dir}
dataset=${IGB_ROOT:-/home/huangjianqiang_group/hdacp1/data/wzh/datasets/igb_homogeneous_small}
test_py="$root/tests/hybrid_aggregatewide_bf16grad_strongdgl_v3.py"
runroot=${FINAL_NUMA_RUN_ROOT:-$root/runs/final_numa_on_dgl_20260817}
point=${SLURM_ARRAY_TASK_ID:-0}

classes=(19 2983)
label_files=(node_label_19.npy node_label_2K.npy)
classes_n=${classes[$point]}
label_file=${label_files[$point]}
run="$runroot/c${classes_n}_l2_numa_on"
mkdir -p "$run"

module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp

export IGB_ROOT="$dataset" IGB_LABEL_FILE="$label_file"
export HYBRID_INPUT_DIM=1024 HYBRID_HIDDEN_DIM=128 HYBRID_OUT_DIM="$classes_n"
export HYBRID_LAYERS=2 HYBRID_SEED=101 IGB_SPLIT_SEED=20260813
export HYBRID_DTYPE=fp32 HYBRID_TRAIN_EPOCHS=200 HYBRID_WARMUPS=0 HYBRID_REPEATS=0
export HYBRID_LAYER_PROFILE=0 HYBRID_WIDE_K_AMX=1
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close OMP_PLACES=cores
export LD_LIBRARY_PATH="/home/huangjianqiang_group/hdacp1/.local/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
extension_dir=${FINAL_TFS_EXTENSION_DIR:-$root/build/extension}
if [[ ! -d "$extension_dir" ]]; then
  echo "missing compiled extension directory: $extension_dir" >&2
  echo "build the extension from csrc/setup_backward_opt_aggregate_wide.py first" >&2
  exit 3
fi
export PYTHONPATH="$extension_dir:$root/python:$root/csrc"

# Apply the release contract from a clean, repository-owned file.  The
# profile is shape agnostic; execution_plan.py remains the only component
# that selects transform-first versus aggregate-first.
export TFS_RELEASE_PROFILE=v2_2_authority
source "$root/scripts/tfs_standard_env.sh"

# Stable integrated optimization contract used by the accepted NUMA run.
export TFS_AMX_PERMISSION_CACHE=1 TFS_NUMA_FIRST_TOUCH=on
export TFS_MEMORY_POLICY=localalloc TFS_CPU_NODES=0-3 TFS_MEMORY_NODES=0-3
export TFS_LOCALITY_SCHEDULE=off TFS_HS_REPLICA=off
export TFS_PULL_ONLY_BACKWARD=1 TFS_SCALE_GRAD_BF16_NATIVE=1
export TFS_GLUE_E1_FUSED_DB=1 TFS_GLUE_E2_VEC_STORE=0 TFS_GLUE_E3_EMPTY_DX=0
export TFS_GLUE_E4_FUSED_EPILOGUE=1 TFS_GLUE_E5_VEC_HS=0
export TFS_GLUE_E6_LOCAL_ZERO=1 TFS_GLUE_E6_PARALLEL_REDUCE=0 TFS_GLUE_E6_SERIAL_WT=0
export TFS_GLUE_E7_FORWARD_SCHEDULE=1 TFS_GLUE_E8_RIGHTSIZE_POOL=1
export TFS_GLUE_E8_ATOMIC_DONE=0 TFS_GLUE_E8_SPIN=0 TFS_GLUE_E9_INT32_COLIDX=0
export TFS_GLUE_E11_VEC_GRAD_DB=0 TFS_FWD_V2_SINGLE_SCAN=1
export TFS_GLUE_E12_D47_SINGLE_SCAN=1 TFS_GLUE_E13_ACTIVE_ROW=1
export TFS_E13_ACTIVE_MAX_DENSITY=0.25 TFS_HS_CACHE_MAX_BYTES=3000000000
export TFS_HIGHD_STREAM_V1=1 TFS_HIGHD_NATIVE_STREAM=auto
export TFS_HIGHD_RELEASE_MODE=1
export TFS_RUNTIME_CONTRACT_DIR="$run"
# Re-apply the locked profile after the legacy glue switches above.  This is
# intentional: future launcher edits cannot contaminate the authority path.
source "$root/scripts/tfs_standard_env.sh"

cat > "$run/release_contract.json" <<JSON
{
  "profile": "$TFS_RELEASE_PROFILE",
  "planner": true,
  "highd_stream": true,
  "native_mode": "$TFS_HIGHD_NATIVE_STREAM",
  "budget_bytes": $TFS_HIGHD_BWD_BUDGET_BYTES,
  "aggregate_d_tile": $TFS_HIGHD_BWD_D_TILE,
  "transform_d_tile": $TFS_TRANSFORM_HIGHD_D_TILE,
  "row_panel": $TFS_HIGHD_BWD_ROW_PANEL,
  "panel_budget_bytes": $TFS_HIGHD_PANEL_BUDGET_BYTES,
  "workspace_cache_max_bytes": $TFS_WORKSPACE_CACHE_MAX_BYTES,
  "colidx": "$TFS_COLIDX",
  "numa_first_touch": true,
  "pull_only": true
}
JSON

printf '%s\n' "{\"release\":\"final_v2_2\",\"release_profile\":\"$TFS_RELEASE_PROFILE\",\"runtime_contract\":\"$TFS_RUNTIME_CONTRACT\",\"dataset\":\"IGB-HOM-small\",\"classes\":$classes_n,\"layers\":2,\"threads\":32,\"epochs\":200,\"policy\":\"localalloc\",\"first_touch\":\"on\",\"tfs_variant\":\"numa_on_persistent_hs_amx\",\"dgl_variant\":\"native_stock\"}" > "$run/manifest.json"
hostname > "$run/hostname.txt"
lscpu > "$run/lscpu.txt"
numactl -H > "$run/numa.txt"
printf '%s\n' "TFS_RUNTIME_CONTRACT $TFS_RUNTIME_CONTRACT" > "$run/runtime_contract.txt"

run_one() {
  local variant="$1"
  local out="$run/${variant}_200epoch.csv"
  local log="$run/${variant}_200epoch.log"

  if [[ "$variant" == tfs_numa_on ]]; then
    export HYBRID_PATH=hybrid HYBRID_AMX_FORWARD=1 HYBRID_AMX_BACKWARD=1
    export HYBRID_PERSISTENT_HS_CACHE=1
    export TFS_RELEASE_PROFILE=v2_2_authority
    export TFS_EXEC_PLANNER=1 TFS_HIGHD_STREAM_V1=1 TFS_HIGHD_NATIVE_STREAM=auto
    export TFS_TRANSFORM_HIGHD_STREAM_V1=auto
    export TFS_PULL_ONLY_BACKWARD=1 TFS_SCALE_GRAD_BF16_NATIVE=1
    export TFS_STATIC_HS=on TFS_NUMA_FIRST_TOUCH=on TFS_HIGHD_RELEASE_MODE=1
    export TFS_RUNTIME_CONTRACT_DIR="$run"
    source "$root/scripts/tfs_standard_env.sh"
  elif [[ "$variant" == dgl_stock ]]; then
    export HYBRID_PATH=dgl_stock HYBRID_DGL_VARIANT=stock
    export HYBRID_AMX_FORWARD=0 HYBRID_AMX_BACKWARD=0
    unset HYBRID_PERSISTENT_HS_CACHE TFS_RELEASE_PROFILE TFS_RUNTIME_CONTRACT
    unset TFS_EXEC_PLANNER TFS_HIGHD_STREAM_V1 TFS_HIGHD_NATIVE_STREAM TFS_TRANSFORM_HIGHD_STREAM_V1
    unset TFS_PULL_ONLY_BACKWARD TFS_SCALE_GRAD_BF16_NATIVE TFS_STATIC_HS
    unset TFS_NUMA_FIRST_TOUCH TFS_HIGHD_RELEASE_MODE
    unset TFS_RELEASE_CONTRACT_STRICT TFS_HIGHD_BWD_BUDGET_BYTES TFS_HIGHD_BWD_D_TILE
    unset TFS_HIGHD_BWD_ROW_PANEL TFS_TRANSFORM_HIGHD_D_TILE TFS_HIGHD_NATIVE_PANEL
    unset TFS_HIGHD_PANEL_BUDGET_BYTES TFS_HIGHD_FUSED_DB TFS_HIGHD_FUSED_TRANSPOSE
    unset TFS_HIGHD_FUSED_SCALE_TRANSPOSE TFS_HIGHD_CONTIGUOUS_PANELS TFS_COLIDX
    unset TFS_MAX_LOCAL_DW_BYTES TFS_LOCAL_DW_BUDGET_BYTES TFS_HIGHD_NATIVE_SCALE
    unset TFS_HIGHD_STREAM_AUTO
    unset TFS_RUNTIME_CONTRACT_DIR
    unset TFS_AMX_PERMISSION_CACHE TFS_MEMORY_POLICY TFS_CPU_NODES TFS_MEMORY_NODES
    unset TFS_LOCALITY_SCHEDULE TFS_HS_REPLICA TFS_HS_CACHE_MAX_BYTES
    unset TFS_GLUE_E1_FUSED_DB TFS_GLUE_E2_VEC_STORE TFS_GLUE_E3_EMPTY_DX
    unset TFS_GLUE_E4_FUSED_EPILOGUE TFS_GLUE_E5_VEC_HS TFS_GLUE_E6_LOCAL_ZERO
    unset TFS_GLUE_E6_PARALLEL_REDUCE TFS_GLUE_E6_SERIAL_WT TFS_GLUE_E7_FORWARD_SCHEDULE
    unset TFS_GLUE_E8_RIGHTSIZE_POOL TFS_GLUE_E8_ATOMIC_DONE TFS_GLUE_E8_SPIN
    unset TFS_GLUE_E9_INT32_COLIDX TFS_GLUE_E11_VEC_GRAD_DB TFS_FWD_V2_SINGLE_SCAN
    unset TFS_GLUE_E12_D47_SINGLE_SCAN TFS_GLUE_E13_ACTIVE_ROW TFS_E13_ACTIVE_MAX_DENSITY
    unset TFS_PROFILE_NUMA_WORKSPACE TFS_WORKSPACE_CACHE_MAX_BYTES
  else
    echo "unknown variant: $variant" >&2
    return 2
  fi

  export HYBRID_OUTPUT="$out"
  env | sort > "$run/${variant}_environment_start.txt"
  if [[ "$variant" == tfs_numa_on ]]; then
    printf '%s\n' "TFS_RUNTIME_CONTRACT $TFS_RUNTIME_CONTRACT" > "$run/${variant}_runtime_contract.txt"
  else
    printf '%s\n' "TFS_RUNTIME_CONTRACT absent (native DGL stock)" > "$run/${variant}_runtime_contract.txt"
  fi
  local start_ns end_ns code wall_ms
  start_ns=$(date +%s%N)
  set +e
  # The same external policy is applied to both frameworks.  DGL does not
  # receive any TFS internal kernel/cache flag; it only shares the fair
  # process-level CPU/memory placement.
  numactl --cpunodebind=0-3 --localalloc python -u "$test_py" > "$log" 2>&1
  code=$?
  set -e
  end_ns=$(date +%s%N)
  wall_ms=$(( (end_ns - start_ns) / 1000000 ))
  printf '%s\n' "{\"variant\":\"$variant\",\"policy\":\"localalloc\",\"wall_ms\":$wall_ms,\"exit_code\":$code}" > "$run/${variant}_timing.json"
  return "$code"
}

set +e
run_one tfs_numa_on; tfs_code=$?
run_one dgl_stock; dgl_code=$?
set -e
printf '%s\n' "{\"policy\":\"localalloc\",\"tfs_variant\":\"numa_on_persistent_hs_amx\",\"dgl_variant\":\"native_stock\",\"tfs_exit\":$tfs_code,\"dgl_exit\":$dgl_code,\"epochs\":200,\"threads\":32}" > "$run/status.json"
test "$tfs_code" -eq 0 -a "$dgl_code" -eq 0
