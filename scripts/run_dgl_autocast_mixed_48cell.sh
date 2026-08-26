#!/usr/bin/env bash
# Strict DGL CPU autocast mixed-precision 48-cell matrix.
#
# Contract: FP32 input/model/Adam/loss; CPU autocast(BF16) only around model
# calls.  Do not enable DNNL_VERBOSE in formal timing runs.
#
# Submit: sbatch scripts/run_dgl_autocast_mixed_48cell.sh
# Override TFS_ROOT and FINAL_PRE_NUMA_RUN_ROOT for another installation.
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=220G
#SBATCH --exclusive
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=32
#SBATCH --threads-per-core=1
#SBATCH -t 2-00:00:00
#SBATCH --array=0-47%2
#SBATCH -J dgl_autocast_mixed

set -euo pipefail

root=${TFS_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}
run_root=${FINAL_PRE_NUMA_RUN_ROOT:-"$root/runs/dgl_autocast_mixed_48cell"}
epochs=${AUTHORITY_EPOCHS:-200}
point=${SLURM_ARRAY_TASK_ID:-0}
unset DNNL_VERBOSE

graphs=(products arxiv igb19 igb2983)
layers_list=(2 3)
threads_list=(1 2 4 8 16 32)
graph_index=$((point / 12))
remainder=$((point % 12))
layer_index=$((remainder / 6))
thread_index=$((remainder % 6))
graph=${graphs[$graph_index]}
layers=${layers_list[$layer_index]}
threads=${threads_list[$thread_index]}

case "$graph" in
  products)
    test_rel=tests/hybrid_fullstep_products_detailed.py
    dataset_name=ogbn-products
    export PRODUCTS_CACHE=${PRODUCTS_CACHE:?set PRODUCTS_CACHE}
    export HYBRID_INPUT_DIM=100 HYBRID_HIDDEN_DIM=128 HYBRID_OUT_DIM=47
    ;;
  arxiv)
    test_rel=tests/hybrid_fullstep_arxiv_detailed.py
    dataset_name=ogbn-arxiv
    export ARXIV_ROOT=${ARXIV_ROOT:?set ARXIV_ROOT}
    export HYBRID_INPUT_DIM=128 HYBRID_HIDDEN_DIM=128 HYBRID_OUT_DIM=40
    ;;
  igb19)
    test_rel=tests/hybrid_aggregatewide_bf16grad_strongdgl_v3.py
    dataset_name=IGB-HOM-small-19
    export IGB_ROOT=${IGB_ROOT:?set IGB_ROOT} IGB_LABEL_FILE=node_label_19.npy
    export IGB_SPLIT_SEED=20260813
    export HYBRID_INPUT_DIM=1024 HYBRID_HIDDEN_DIM=128 HYBRID_OUT_DIM=19
    ;;
  igb2983)
    test_rel=tests/hybrid_aggregatewide_bf16grad_strongdgl_v3.py
    dataset_name=IGB-HOM-small-2983
    export IGB_ROOT=${IGB_ROOT:?set IGB_ROOT} IGB_LABEL_FILE=node_label_2K.npy
    export IGB_SPLIT_SEED=20260813
    export HYBRID_INPUT_DIM=1024 HYBRID_HIDDEN_DIM=128 HYBRID_OUT_DIM=2983
    ;;
  *) echo "invalid array index: $point" >&2; exit 2 ;;
esac

run="$run_root/dgl_autocast_mixed/${graph}_l${layers}_t${threads}_e${epochs}"
mkdir -p "$run"
if [[ -f "$run/status.json" ]] && grep -q '"status":"success"' "$run/status.json"; then
  exit 0
fi

module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export HYBRID_PATH=dgl_stock HYBRID_DGL_VARIANT=stock HYBRID_DTYPE=bf16_mixed
export HYBRID_LAYERS="$layers" HYBRID_SEED=101 HYBRID_TRAIN_EPOCHS="$epochs"
export HYBRID_WARMUPS=0 HYBRID_REPEATS=0 HYBRID_CHECK=0 HYBRID_CHECK_FULL=0
export HYBRID_LAYER_PROFILE=0 HYBRID_DETAILED_PROFILE=0 HYBRID_DGL_OP_PROFILE=0
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
export TFS_MEMORY_POLICY=localalloc TFS_CPU_NODES=0-3 TFS_MEMORY_NODES=0-3
export PYTHONPATH="$root/build/extension:$root/python:$root/csrc"
export HYBRID_OUTPUT="$run/training_detailed.csv"
export HYBRID_TIMING_METADATA="$run/timing_markers.json"

cat >"$run/manifest.json" <<EOF
{"method":"dgl_autocast_mixed","graph":"$graph","dataset":"$dataset_name","layers":$layers,"threads":$threads,"epochs":$epochs,"seed":101,"dtype_contract":"fp32_input_model_optimizer_cpu_autocast_bf16_fp32_loss","numactl":"--cpunodebind=0-3 --localalloc","steady_epoch_scope":"median(epoch 2..$epochs) of train_step_ms + evaluation_ms"}
EOF
sha256sum "$root/$test_rel" "$root/python/tfs_train/standard_dgl.py" >"$run/source_input_sha256.tsv"

start_ns=$(date +%s%N)
set +e
numactl --cpunodebind=0-3 --localalloc python -u "$root/$test_rel" >"$run/stdout.log" 2>"$run/stderr.log"
code=$?
set -e
wall_ms=$(( ($(date +%s%N) - start_ns) / 1000000 ))
if [[ "$code" -eq 0 ]]; then
  python "$root/scripts/summarize_authority_cell.py" "$run/training_detailed.csv" \
    "$run/summary.json" "$wall_ms" dgl_autocast_mixed "$graph" "$layers" \
    "$threads" "$epochs" || code=95
fi
status=$([[ "$code" -eq 0 ]] && echo success || echo failed)
printf '{"status":"%s","method":"dgl_autocast_mixed","graph":"%s","layers":%s,"threads":%s,"epochs":%s,"wall_ms":%s,"exit_code":%s}\n' \
  "$status" "$graph" "$layers" "$threads" "$epochs" "$wall_ms" "$code" >"$run/status.json"
exit "$code"
