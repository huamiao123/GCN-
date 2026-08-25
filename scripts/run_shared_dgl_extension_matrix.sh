#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=160G
#SBATCH --exclusive
#SBATCH -t 2-00:00:00
#SBATCH --array=0-47%5
#SBATCH -J dgl_ext_200e
#SBATCH -o /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa/slurm/dgl_ext_%A_%a.out
#SBATCH -e /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa/slurm/dgl_ext_%A_%a.err

# Stock-DGL companion matrix for GraphSAINT extensions.  It uses the same
# graph bundle, seed, hidden width, optimizer, epoch count, CPU reservation,
# and per-epoch timing fields as the TFS extension matrix.
set -euo pipefail
root=${TFS_ROOT:-/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa}
run_root=${DGL_EXTENSION_RUN_ROOT:-/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa_extension_dgl_v1_20260823}
epochs=${DGL_EXTENSION_EPOCHS:-200}
graphs=(flickr reddit yelp amazon)
layers_list=(2 3)
threads_list=(1 2 4 8 16 32)
point=${SLURM_ARRAY_TASK_ID:?}; graph=${graphs[$((point / 12))]}
rest=$((point % 12)); layers=${layers_list[$((rest / 6))]}; threads=${threads_list[$((rest % 6))]}
run="$run_root/${graph}_l${layers}_t${threads}_e${epochs}"
mkdir -p "$run" "$root/slurm"
if [[ -f "$run/status.json" ]] && grep -q '"status": "success"' "$run/status.json"; then echo "already complete: $run"; exit 0; fi
if [[ -e "$run/running.lock" ]]; then echo "cell already locked: $run" >&2; exit 90; fi
touch "$run/running.lock"; trap 'code=$?; rm -f "$run/running.lock"; exit "$code"' EXIT
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
while IFS='=' read -r variable _; do case "$variable" in TFS_*|HYBRID_*) unset "$variable";; esac; done < <(env)
case "$graph" in
  flickr) data=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/graphsaint/flickr ;;
  reddit) data=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/graphsaint/reddit ;;
  yelp) data=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/graphsaint/yelp ;;
  amazon) data=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/graphsaint/amazon ;;
esac
export TFS_DATASET_ROOT="$data" TFS_DATASET_NAME="$graph"
export HYBRID_LAYERS="$layers" HYBRID_SEED=101 HYBRID_TRAIN_EPOCHS="$epochs"
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
export PYTHONPATH="$root/build/extension:$root/python:$root/csrc"
export LD_LIBRARY_PATH="/home/huangjianqiang_group/hdacp1/.local/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export HYBRID_OUTPUT="$run/training_detailed.csv" TFS_SUMMARY_OUTPUT="$run/summary.json"
hostname > "$run/hostname.txt"; lscpu > "$run/lscpu.txt"; numactl -H > "$run/numa.txt"; taskset -pc $$ > "$run/affinity.txt" 2>&1 || true
env | sort | grep -E '^(TFS_|HYBRID_|OMP_|MKL_|PYTHONPATH=|LD_LIBRARY_PATH=)' > "$run/environment_start.txt"
sha256sum "$root/tests/run_shared_dgl_dataset.py" "$root/python/tfs_train/datasets.py" "$root/python/tfs_train/standard_dgl.py" "$root/python/tfs_train/standard_runtime.py" > "$run/source_input_sha256.tsv"
cat > "$run/manifest.json" <<EOF
{"method":"dgl_stock","dataset":"$graph","layers":$layers,"threads":$threads,"epochs":$epochs,"seed":101,"partition":"${SLURM_JOB_PARTITION:-intel}","allocation":"exclusive single-node in ordinary intel pool","graph_construction":"direct pull CSR as DGL CSC plus self-loops","timing_scope":"each epoch records train_step_ms and evaluation_ms; steady statistic is median epoch 2..$epochs"}
EOF
start_ns=$(date +%s%N); set +e
/usr/bin/time -v -o "$run/resource_usage.txt" srun --cpu-bind=none -n 1 -c 32 "$root/tests/run_shared_tfs_dataset_step.sh" "$root/tests/run_shared_dgl_dataset.py" > "$run/stdout.log" 2> "$run/stderr.log"
code=$?; set -e; end_ns=$(date +%s%N); wall_ms=$(((end_ns-start_ns)/1000000))
if [[ "$code" -eq 0 ]]; then python "$root/scripts/summarize_authority_cell.py" "$run/training_detailed.csv" "$run/timing_summary.json" "$wall_ms" dgl_stock "$graph" "$layers" "$threads" "$epochs" || code=95; fi
status=failed; [[ "$code" -eq 0 ]] && status=success
printf '{"status": "%s", "graph": "%s", "layers": %s, "threads": %s, "epochs": %s, "wall_ms": %s, "exit_code": %s}\n' "$status" "$graph" "$layers" "$threads" "$epochs" "$wall_ms" "$code" > "$run/status.json"
exit "$code"
