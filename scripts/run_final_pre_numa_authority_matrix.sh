#!/usr/bin/env bash
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
#SBATCH -J final_pre_numa_200e
#SBATCH -o /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa/slurm/authority_%A_%a.out
#SBATCH -e /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa/slurm/authority_%A_%a.err

# One paper-protocol process per cell.  Submit this launcher twice, once with
# AUTHORITY_METHOD=tfs_final_pre_numa and once with AUTHORITY_METHOD=dgl_stock.
# Both jobs use the identical Slurm allocation, binding and timing boundary.
set -euo pipefail

root=${TFS_ROOT:-/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa}
run_root=${FINAL_PRE_NUMA_RUN_ROOT:-/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa_paper_v1_20260822}
epochs=${AUTHORITY_EPOCHS:-200}
point=${SLURM_ARRAY_TASK_ID:-0}
method=${AUTHORITY_METHOD:-tfs_final_pre_numa}
case "$method" in tfs_final_pre_numa|dgl_stock) ;; *)
  echo "AUTHORITY_METHOD must be tfs_final_pre_numa or dgl_stock" >&2; exit 2 ;;
esac

# Remove inherited experimental knobs before defining this cell.  This must
# happen before the graph-specific HYBRID_* shape variables are assigned;
# otherwise `unset` would erase the cell contract we just constructed.
while IFS='=' read -r variable _; do
  case "$variable" in TFS_*|HYBRID_*) unset "$variable" ;; esac
done < <(env)

graphs=(products arxiv igb19 igb2983)
layers_list=(2 3)
threads_list=(1 2 4 8 16 32)
thread_count=${#threads_list[@]}
layer_count=${#layers_list[@]}
graph_index=$((point / (layer_count * thread_count)))
remainder=$((point % (layer_count * thread_count)))
layer_index=$((remainder / thread_count))
thread_index=$((remainder % thread_count))
graph=${graphs[$graph_index]}
layers=${layers_list[$layer_index]}
threads=${threads_list[$thread_index]}
# Authority resource policy is shape/process driven, never graph-name driven.
export TFS_HS_CACHE_MAX_BYTES=3221225472

case "$graph" in
  products)
    test_rel=tests/hybrid_fullstep_products_detailed.py
    dataset_name=ogbn-products
    export PRODUCTS_CACHE=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/ogbn-products_official/processed_tfs_v1/ogbn_products_canonical_v1.pt
    export HYBRID_INPUT_DIM=100 HYBRID_HIDDEN_DIM=128 HYBRID_OUT_DIM=47
    ;;
  arxiv)
    test_rel=tests/hybrid_fullstep_arxiv_detailed.py
    dataset_name=ogbn-arxiv
    export ARXIV_ROOT=/online1/huangjianqiang_group/hdacp1/wzh/TFS-Train/runs/c5_fullstep_arxiv_20260804_v3/dataset/arxiv
    export HYBRID_INPUT_DIM=128 HYBRID_HIDDEN_DIM=128 HYBRID_OUT_DIM=40
    ;;
  igb19)
    test_rel=tests/hybrid_aggregatewide_bf16grad_strongdgl_v3.py
    dataset_name=IGB-HOM-small-19
    export IGB_ROOT=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/igb_homogeneous_small
    export IGB_LABEL_FILE=node_label_19.npy IGB_SPLIT_SEED=20260813
    export HYBRID_INPUT_DIM=1024 HYBRID_HIDDEN_DIM=128 HYBRID_OUT_DIM=19
    ;;
  igb2983)
    test_rel=tests/hybrid_aggregatewide_bf16grad_strongdgl_v3.py
    dataset_name=IGB-HOM-small-2983
    export IGB_ROOT=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/igb_homogeneous_small
    export IGB_LABEL_FILE=node_label_2K.npy IGB_SPLIT_SEED=20260813
    export HYBRID_INPUT_DIM=1024 HYBRID_HIDDEN_DIM=128 HYBRID_OUT_DIM=2983
    ;;
  *) echo "invalid graph index=$graph_index" >&2; exit 2 ;;
esac

run="$run_root/$method/${graph}_l${layers}_t${threads}_e${epochs}"
mkdir -p "$run" "$root/slurm"
if [[ -f "$run/status.json" ]] && grep -q '"status":"success"' "$run/status.json"; then
  echo "already complete: $run"
  exit 0
fi
if [[ -e "$run/running.lock" ]]; then
  echo "cell already has a running lock: $run" >&2
  exit 90
fi
touch "$run/running.lock"
trap 'code=$?; rm -f "$run/running.lock"; exit "$code"' EXIT

module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export HYBRID_INPUT_DIM HYBRID_HIDDEN_DIM HYBRID_OUT_DIM TFS_HS_CACHE_MAX_BYTES
case "$graph" in
  products) export PRODUCTS_CACHE ;;
  arxiv) export ARXIV_ROOT ;;
  igb19|igb2983) export IGB_ROOT IGB_LABEL_FILE IGB_SPLIT_SEED ;;
esac
export HYBRID_LAYERS="$layers" HYBRID_SEED=101
export HYBRID_TRAIN_EPOCHS="$epochs" HYBRID_WARMUPS=0 HYBRID_REPEATS=0
export HYBRID_LAYER_PROFILE=0 HYBRID_DETAILED_PROFILE=0 HYBRID_DGL_OP_PROFILE=0
export HYBRID_DTYPE=fp32 HYBRID_CHECK=0 HYBRID_CHECK_FULL=0
export HYBRID_GRAD_TOL=0.03 HYBRID_FORWARD_TOL=0.03
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
export LD_LIBRARY_PATH="/home/huangjianqiang_group/hdacp1/.local/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

requested_profile=""
if [[ "$method" == tfs_final_pre_numa ]]; then
  # This template intentionally accepts only the TFS execution path.  Keeping
  # it out of the DGL branch is essential: the paired wrapper launches stock
  # DGL in the same allocation but it is not a TFS path.
  export HYBRID_AUTHORITY_TEMPLATE=final-pre-numa-v1
  requested_profile=final_pre_numa
  export TFS_RELEASE_PROFILE="$requested_profile"
  source "$root/scripts/tfs_standard_env.sh"
  [[ "$TFS_RELEASE_PROFILE" == "$requested_profile" ]]
  [[ "$TFS_PROFILE_STATUS" == authority ]]
  [[ "$TFS_AGGREGATE_SAVED" == auto ]]
  [[ "$TFS_NUMA_FIRST_TOUCH" == on ]]
  [[ "$TFS_NUMA_PRIVATE" == off ]]
  [[ "$TFS_NUMA_REDUCE" == off ]]
  [[ "$TFS_LOCALITY_SCHEDULE" == off ]]
  [[ "$TFS_HS_REPLICA" == off ]]
  export TFS_RUNTIME_CONTRACT_DIR="$run" TFS_EXPLAIN_PLAN=1
  export HYBRID_PATH=hybrid HYBRID_AMX_FORWARD=1 HYBRID_AMX_BACKWARD=1
  export HYBRID_PERSISTENT_HS_CACHE=1
else
  export HYBRID_PATH=dgl_stock HYBRID_DGL_VARIANT=stock
fi
export TFS_MEMORY_POLICY=localalloc TFS_CPU_NODES=0-3 TFS_MEMORY_NODES=0-3
export TFS_AMX_PERMISSION_CACHE=1
export PYTHONPATH="$root/build/extension:$root/python:$root/csrc"
export HYBRID_OUTPUT="$run/training_detailed.csv"
export HYBRID_TIMING_METADATA="$run/timing_markers.json"
test_py="$root/$test_rel"

hostname > "$run/hostname.txt"
lscpu > "$run/lscpu.txt"
numactl -H > "$run/numa.txt"
numactl --show > "$run/numactl_show.txt" 2>&1 || true
taskset -pc $$ > "$run/affinity.txt" 2>&1 || true
env | sort | grep -E \
  '^(TFS_|HYBRID_|OMP_|MKL_|PRODUCTS_CACHE=|ARXIV_ROOT=|IGB_ROOT=|IGB_LABEL_FILE=|IGB_SPLIT_SEED=|PYTHONPATH=|LD_LIBRARY_PATH=)' \
  > "$run/environment_start.txt"
python -c 'import json,torch; print(json.dumps({"torch":torch.__version__}))' \
  > "$run/framework_versions.json"
sha256sum "$test_py" "$root/python/tfs_train/execution_plan.py" \
  "$root/python/tfs_train/dimension_dispatch.py" \
  "$root/python/tfs_train/highd_backward.py" \
  "$root/python/tfs_train/aggregate_saved.py" \
  "$root/python/tfs_train/standard_runtime.py" \
  "$root/python/tfs_train/standard_dgl.py" \
  "$root/csrc/experiments/backward_opt_20260814/v6_wide_aggregate_probe.cpp" \
  "$root/scripts/tfs_standard_env.sh" \
  "$root/scripts/run_final_pre_numa_authority_matrix.sh" \
  "$root/scripts/validate_authority_plan.py" \
  "$root/scripts/summarize_authority_cell.py" \
  "$root/build/extension"/tfs_train_v2_c0_ext*.so \
  > "$run/source_input_sha256.tsv"

if [[ "$method" == tfs_final_pre_numa ]]; then
  cat > "$run/manifest.json" <<EOF
{"method":"$method","requested_profile":"$requested_profile","resolved_profile":"$TFS_RELEASE_PROFILE","profile_status":"$TFS_PROFILE_STATUS","graph":"$graph","dataset":"$dataset_name","layers":$layers,"threads":$threads,"epochs":$epochs,"seed":101,"path":"$HYBRID_PATH","dtype_contract":"bf16_inputs_fp32_accum_fp32_master","warmups":0,"repeats":0,"offline_preprocessing":false,"partition":"${SLURM_JOB_PARTITION:-unknown}","numactl":"--cpunodebind=0-3 --localalloc","numa_first_touch":"$TFS_NUMA_FIRST_TOUCH","numa_private":"$TFS_NUMA_PRIVATE","numa_reduce":"$TFS_NUMA_REDUCE","locality_scheduler":"$TFS_LOCALITY_SCHEDULE","hs_replica":"$TFS_HS_REPLICA","timing_protocol":"paper_v1","dgl_rerun_requested":true,"timing_scope":"launcher wall begins immediately before numactl/python and ends after Python exits; includes process/import startup, dataset and graph construction, model and persistent-cache setup, all 200 train+evaluation epochs, and CSV write; excludes queue/module/metadata","steady_epoch_scope":"median(epoch 2..200) of train_step_ms + evaluation_ms; epoch 1 stays only in raw CSV and cold wall"}
EOF
else
  cat > "$run/manifest.json" <<EOF
{"method":"dgl_stock","dgl_variant":"stock","dgl_rerun":true,"graph":"$graph","dataset":"$dataset_name","layers":$layers,"threads":$threads,"epochs":$epochs,"seed":101,"path":"$HYBRID_PATH","dtype_contract":"fp32","warmups":0,"repeats":0,"offline_preprocessing":false,"partition":"${SLURM_JOB_PARTITION:-unknown}","numactl":"--cpunodebind=0-3 --localalloc","timing_protocol":"paper_v1","timing_scope":"launcher wall begins immediately before numactl/python and ends after Python exits; includes process/import startup, dataset and graph construction, model setup, all 200 train+evaluation epochs, and CSV write; excludes queue/module/metadata","steady_epoch_scope":"median(epoch 2..200) of train_step_ms + evaluation_ms; epoch 1 stays only in raw CSV and cold wall"}
EOF
fi

start_ns=$(date +%s%N)
set +e
if [[ "${AUTHORITY_RESOURCE_TIME:-1}" == "1" ]]; then
  /usr/bin/time -v -o "$run/resource_usage.txt" \
    numactl --cpunodebind=0-3 --localalloc python -u "$test_py" \
    > "$run/stdout.log" 2> "$run/stderr.log"
else
  # Matches the DGL-Official-AMP launcher exactly: cold wall spans one
  # numactl-bound Python process only, without an extra measurement wrapper.
  numactl --cpunodebind=0-3 --localalloc python -u "$test_py" \
    > "$run/stdout.log" 2> "$run/stderr.log"
fi
code=$?
set -e
end_ns=$(date +%s%N)
wall_ms=$(( (end_ns - start_ns) / 1000000 ))

if [[ "$code" -eq 0 && "$method" == tfs_final_pre_numa ]]; then
  grep '^TFS_PLAN ' "$run/stdout.log" > "$run/tfs_plan.log" || true
  plan_lines=$(wc -l < "$run/tfs_plan.log")
  [[ "$plan_lines" -eq "$layers" ]] || code=91
  grep -q 'execution_variant=' "$run/tfs_plan.log" || code=92
  if grep -q 'execution_variant=aggregate_saved_v4' "$run/tfs_plan.log"; then code=93; fi
  if [[ "$code" -eq 0 ]]; then
    python "$root/scripts/validate_authority_plan.py" \
      "$run/tfs_plan.log" "$graph" "$layers" \
      > "$run/plan_validation.json" || code=94
  fi
fi

if [[ "$code" -eq 0 ]]; then
  python "$root/scripts/summarize_authority_cell.py" \
    "$run/training_detailed.csv" "$run/summary.json" "$wall_ms" \
    tfs_final_pre_numa "$graph" "$layers" "$threads" "$epochs" || code=95
fi

status=$([[ "$code" -eq 0 ]] && echo success || echo failed)
if [[ -f "$run/tfs_plan.log" ]]; then
  plan_lines=$(wc -l < "$run/tfs_plan.log")
else
  plan_lines=0
fi
printf '%s\n' \
  "{\"status\":\"$status\",\"method\":\"$method\",\"graph\":\"$graph\",\"layers\":$layers,\"threads\":$threads,\"epochs\":$epochs,\"requested_profile\":\"$requested_profile\",\"resolved_profile\":\"${TFS_RELEASE_PROFILE:-}\",\"path\":\"$HYBRID_PATH\",\"wall_ms\":$wall_ms,\"exit_code\":$code,\"plan_lines\":$plan_lines}" \
  > "$run/status.json"
exit "$code"
