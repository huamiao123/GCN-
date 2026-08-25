#!/usr/bin/env bash
# Paired 32-thread Products comparison: frozen v2 versus final_v2_2.
# Array: v2/TFS, v2_2/TFS, v2/DGL stock, v2_2/DGL stock.
# Each task is serialized so every process owns the same node and NUMA policy.
# The timed process includes dataset load, graph construction, model setup,
# 200 train+eval epochs, and output; training_detailed.csv additionally keeps
# per-epoch train/eval and layer timings.
#SBATCH -p intel_expr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=180G
#SBATCH --exclusive
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=32
#SBATCH --threads-per-core=1
#SBATCH -t 02:00:00
#SBATCH --array=0-3%1
#SBATCH -J products_v2_v22_200e
#SBATCH -o /home/huangjianqiang_group/hdacp1/data/wzh/final_v2_2/slurm/products_v2_v22_200e_%A_%a.out
#SBATCH -e /home/huangjianqiang_group/hdacp1/data/wzh/final_v2_2/slurm/products_v2_v22_200e_%A_%a.err

set -euo pipefail

V2_ROOT=${V2_ROOT:-/home/huangjianqiang_group/hdacp1/data/wzh/final_v2}
V22_ROOT=${V22_ROOT:-/home/huangjianqiang_group/hdacp1/data/wzh/final_v2_2}
RUN_ROOT=${PRODUCTS_COMPARE_RUN_ROOT:-$V22_ROOT/runs/products_v2_v2_2_200e_20260818}
CACHE=${PRODUCTS_CACHE:-/home/huangjianqiang_group/hdacp1/data/wzh/datasets/ogbn-products_official/processed_tfs_v1/ogbn_products_canonical_v1.pt}
point=${SLURM_ARRAY_TASK_ID:-0}

case "$point" in
  0) version=v2;  method=hybrid; root="$V2_ROOT"  ;;
  1) version=v2_2; method=hybrid; root="$V22_ROOT" ;;
  2) version=v2;  method=dgl_stock; root="$V2_ROOT"  ;;
  3) version=v2_2; method=dgl_stock; root="$V22_ROOT" ;;
  *) echo "invalid array point: $point" >&2; exit 2 ;;
esac

tag="${version}_${method}_products_l2_t32_e200_detailed"
run="$RUN_ROOT/$tag"
test_py="$root/tests/hybrid_fullstep_products_detailed.py"
mkdir -p "$run"
test ! -e "$run/status.json" || { echo "run already exists: $run" >&2; exit 90; }
exec > >(tee "$run/stdout.log") 2> >(tee "$run/stderr.log" >&2)
trap 'code=$?; printf "{\"status\":\"failed\",\"exit_code\":%d}\n" "$code" > "$run/status.json"; exit "$code"' ERR

module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp

export PRODUCTS_CACHE="$CACHE"
export HYBRID_INPUT_DIM=100 HYBRID_HIDDEN_DIM=128 HYBRID_OUT_DIM=47
export HYBRID_LAYERS=2 HYBRID_SEED=101 HYBRID_TRAIN_EPOCHS=200
export HYBRID_WARMUPS=0 HYBRID_REPEATS=0 HYBRID_LAYER_PROFILE=0
export HYBRID_DETAILED_PROFILE="${HYBRID_DETAILED_PROFILE:-1}" HYBRID_DGL_OP_PROFILE=0 HYBRID_DTYPE=fp32
export HYBRID_GRAD_TOL=0.03 HYBRID_FORWARD_TOL=0.03
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close OMP_PLACES=cores
export LD_LIBRARY_PATH="/home/huangjianqiang_group/hdacp1/.local/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export TFS_MEMORY_POLICY=localalloc TFS_CPU_NODES=0-3 TFS_MEMORY_NODES=0-3
export TFS_NUMA_FIRST_TOUCH="${TFS_NUMA_FIRST_TOUCH:-on}" TFS_LOCALITY_SCHEDULE=off TFS_HS_REPLICA=off
export TFS_AMX_PERMISSION_CACHE=1 TFS_PULL_ONLY_BACKWARD=1
export TFS_SCALE_GRAD_BF16_NATIVE=1
export TFS_GLUE_E1_FUSED_DB=1 TFS_GLUE_E2_VEC_STORE=0 TFS_GLUE_E3_EMPTY_DX=0
export TFS_GLUE_E4_FUSED_EPILOGUE=1 TFS_GLUE_E5_VEC_HS=0
export TFS_GLUE_E6_LOCAL_ZERO=1 TFS_GLUE_E6_PARALLEL_REDUCE=0 TFS_GLUE_E6_SERIAL_WT=0
export TFS_GLUE_E7_FORWARD_SCHEDULE=1 TFS_GLUE_E8_RIGHTSIZE_POOL=1
export TFS_GLUE_E8_ATOMIC_DONE=0 TFS_GLUE_E8_SPIN=0 TFS_GLUE_E9_INT32_COLIDX=0
export TFS_GLUE_E11_VEC_GRAD_DB=0 TFS_FWD_V2_SINGLE_SCAN=1
export TFS_GLUE_E12_D47_SINGLE_SCAN=1 TFS_GLUE_E13_ACTIVE_ROW=1
export TFS_E13_ACTIVE_MAX_DENSITY=0.25 TFS_HS_CACHE_MAX_BYTES=3000000000

# Remove inherited contract switches before applying the selected version.
unset TFS_RELEASE_PROFILE TFS_RELEASE_CONTRACT_STRICT TFS_RUNTIME_CONTRACT
unset TFS_HIGHD_RELEASE_MODE TFS_HIGHD_STREAM_V1 TFS_HIGHD_NATIVE_STREAM
unset TFS_TRANSFORM_HIGHD_STREAM_V1 TFS_HIGHD_BWD_BUDGET_BYTES TFS_HIGHD_BWD_D_TILE
unset TFS_HIGHD_BWD_ROW_PANEL TFS_TRANSFORM_HIGHD_D_TILE TFS_HIGHD_NATIVE_PANEL
unset TFS_HIGHD_PANEL_BUDGET_BYTES TFS_WORKSPACE_CACHE_MAX_BYTES
unset TFS_HIGHD_FUSED_DB TFS_HIGHD_FUSED_TRANSPOSE TFS_HIGHD_FUSED_SCALE_TRANSPOSE
unset TFS_HIGHD_CONTIGUOUS_PANELS TFS_COLIDX TFS_MAX_LOCAL_DW_BYTES TFS_LOCAL_DW_BUDGET_BYTES

if [[ "$version" == v2_2 ]]; then
  export TFS_RELEASE_PROFILE=v2_2_authority
  source "$root/scripts/tfs_standard_env.sh"
  export TFS_RUNTIME_CONTRACT_DIR="$run"
else
  # v2 is measured with its frozen profile and the same accepted NUMA/glue
  # switches; no v2.2-only locked profile is injected into this baseline.
  # The v2 planner's ``auto`` colidx candidate is only a metadata gate; the
  # v2 native wrapper still consumes the canonical int64 Products CSR.  Lock
  # the baseline to int64 so the comparison exercises v2's actual kernels
  # instead of passing an int32 plan bit to an int64 tensor (which can abort
  # inside the AMX wrapper on the official Products cache).
  source "$root/scripts/tfs_standard_env.sh"
  export TFS_COLIDX=int64
  export TFS_NUMA_FIRST_TOUCH="${TFS_NUMA_FIRST_TOUCH:-on}" TFS_PULL_ONLY_BACKWARD=1
  export TFS_SCALE_GRAD_BF16_NATIVE=1
fi

if [[ "$method" == hybrid ]]; then
  export HYBRID_PATH=hybrid HYBRID_AMX_FORWARD=1 HYBRID_AMX_BACKWARD=1
  export HYBRID_PERSISTENT_HS_CACHE="${HYBRID_PERSISTENT_HS_CACHE:-1}"
  export PYTHONPATH="${TFS_EXTENSION_ROOT:-$root/build/extension}:$root/python:$root/csrc"
else
  export HYBRID_PATH=dgl_stock HYBRID_DGL_VARIANT=stock
  export HYBRID_AMX_FORWARD=0 HYBRID_AMX_BACKWARD=0
  unset HYBRID_PERSISTENT_HS_CACHE TFS_AMX_PERMISSION_CACHE
  unset TFS_RELEASE_PROFILE TFS_RELEASE_CONTRACT_STRICT TFS_RUNTIME_CONTRACT_DIR
  unset TFS_EXEC_PLANNER TFS_HIGHD_STREAM_V1 TFS_HIGHD_NATIVE_STREAM
  unset TFS_TRANSFORM_HIGHD_STREAM_V1 TFS_PULL_ONLY_BACKWARD TFS_SCALE_GRAD_BF16_NATIVE
  unset TFS_STATIC_HS TFS_NUMA_FIRST_TOUCH TFS_HIGHD_RELEASE_MODE
  unset TFS_HIGHD_BWD_BUDGET_BYTES TFS_HIGHD_BWD_D_TILE TFS_HIGHD_BWD_ROW_PANEL
  unset TFS_TRANSFORM_HIGHD_D_TILE TFS_HIGHD_NATIVE_PANEL TFS_HIGHD_PANEL_BUDGET_BYTES
  unset TFS_WORKSPACE_CACHE_MAX_BYTES TFS_HIGHD_FUSED_DB TFS_HIGHD_FUSED_TRANSPOSE
  unset TFS_HIGHD_FUSED_SCALE_TRANSPOSE TFS_HIGHD_CONTIGUOUS_PANELS TFS_COLIDX
  unset TFS_MEMORY_POLICY TFS_CPU_NODES TFS_MEMORY_NODES TFS_LOCALITY_SCHEDULE
  unset TFS_HS_REPLICA TFS_HS_CACHE_MAX_BYTES TFS_NUMA_PRIVATE TFS_NUMA_REDUCE
  unset TFS_GLUE_E1_FUSED_DB TFS_GLUE_E2_VEC_STORE TFS_GLUE_E3_EMPTY_DX
  unset TFS_GLUE_E4_FUSED_EPILOGUE TFS_GLUE_E5_VEC_HS TFS_GLUE_E6_LOCAL_ZERO
  unset TFS_GLUE_E6_PARALLEL_REDUCE TFS_GLUE_E6_SERIAL_WT TFS_GLUE_E7_FORWARD_SCHEDULE
  unset TFS_GLUE_E8_RIGHTSIZE_POOL TFS_GLUE_E8_ATOMIC_DONE TFS_GLUE_E8_SPIN
  unset TFS_GLUE_E9_INT32_COLIDX TFS_GLUE_E11_VEC_GRAD_DB TFS_FWD_V2_SINGLE_SCAN
  unset TFS_GLUE_E12_D47_SINGLE_SCAN TFS_GLUE_E13_ACTIVE_ROW TFS_E13_ACTIVE_MAX_DENSITY
  export PYTHONPATH="$root/python"
fi

export HYBRID_OUTPUT="$run/training_detailed.csv"
export HYBRID_OP_PROFILE="$run/dgl_ops.jsonl"
printf '%s\n' "{\"version\":\"$version\",\"method\":\"$method\",\"dataset\":\"ogbn-products\",\"layers\":2,\"input_dim\":100,\"hidden_dim\":128,\"out_dim\":47,\"threads\":32,\"epochs\":200,\"dtype\":\"fp32\",\"policy\":\"localalloc\",\"detailed\":true}" > "$run/manifest.json"
hostname > "$run/hostname.txt"
lscpu > "$run/lscpu.txt"
numactl -H > "$run/numa.txt"
env | sort > "$run/environment_start.txt"
sha256sum "$test_py" "$root/python/tfs_train/execution_plan.py" "$root/python/tfs_train/dimension_dispatch.py" "$CACHE" "$root/build/extension"/*.so 2>/dev/null > "$run/source_input_sha256.tsv" || true

numa_launch_mode="${TFS_NUMA_LAUNCH_MODE:-localalloc}"
case "$numa_launch_mode" in
  localalloc) numa_memory_args=(--localalloc) ;;
  interleave) numa_memory_args=(--interleave=0-3) ;;
  *) echo "TFS_NUMA_LAUNCH_MODE must be localalloc|interleave" >&2; exit 2 ;;
esac
start_ns=$(date +%s%N)
set +e
numactl --cpunodebind=0-3 "${numa_memory_args[@]}" python -u "$test_py"
code=$?
set -e
end_ns=$(date +%s%N)
wall_ms=$(( (end_ns - start_ns) / 1000000 ))
printf '%s\n' "{\"status\":\"$([[ $code -eq 0 ]] && echo success || echo failed)\",\"version\":\"$version\",\"method\":\"$method\",\"threads\":32,\"epochs\":200,\"wall_ms\":$wall_ms,\"exit_code\":$code}" > "$run/status.json"
test "$code" -eq 0
