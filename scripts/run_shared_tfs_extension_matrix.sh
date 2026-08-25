#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=160G
#SBATCH --exclusive
#SBATCH -t 2-00:00:00
#SBATCH --array=0-59%5
#SBATCH -J tfs_ext_200e
#SBATCH -o /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa/slurm/ext_matrix_%A_%a.out
#SBATCH -e /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa/slurm/ext_matrix_%A_%a.err

# Formal TFS-only extension matrix.  Exclusive allocation within the ordinary
# intel pool keeps each cell off a co-located workload; this is not a TFS-DGL
# comparison because no matching DGL cell is launched here.
set -euo pipefail
root=${TFS_ROOT:-/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa}
run_root=${TFS_EXTENSION_RUN_ROOT:-/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa_extension_tfs_v1_20260822}
epochs=${TFS_EXTENSION_EPOCHS:-200}
graphs=(flickr reddit yelp amazon igb_medium)
layers_list=(2 3)
threads_list=(1 2 4 8 16 32)
point=${SLURM_ARRAY_TASK_ID:?}
graph=${graphs[$((point / 12))]}
rest=$((point % 12)); layers=${layers_list[$((rest / 6))]}; threads=${threads_list[$((rest % 6))]}
run="$run_root/${graph}_l${layers}_t${threads}_e${epochs}"
mkdir -p "$run" "$root/slurm"
if [[ -f "$run/status.json" ]] && grep -q '"status": "success"' "$run/status.json"; then echo "already complete: $run"; exit 0; fi
if [[ -e "$run/running.lock" ]]; then echo "cell already locked: $run" >&2; exit 90; fi
touch "$run/running.lock"; trap 'code=$?; rm -f "$run/running.lock"; exit "$code"' EXIT
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
while IFS='=' read -r variable _; do case "$variable" in TFS_*|HYBRID_*) unset "$variable";; esac; done < <(env)
export TFS_RELEASE_PROFILE=final_pre_numa
source "$root/scripts/tfs_standard_env.sh"
[[ "$TFS_PROFILE_STATUS" == authority ]]
export TFS_DATASET_KIND=graphsaint
case "$graph" in
  flickr) export TFS_DATASET_ROOT=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/graphsaint/flickr ;;
  reddit) export TFS_DATASET_ROOT=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/graphsaint/reddit ;;
  yelp) export TFS_DATASET_ROOT=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/graphsaint/yelp ;;
  amazon) export TFS_DATASET_ROOT=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/graphsaint/amazon ;;
  igb_medium) export TFS_DATASET_KIND=igb TFS_DATASET_ROOT=/home/huangjianqiang_group/hdacp1/data/wzh/datasets/igb/igb_hom_medium TFS_IGB_SIZE=medium ;;
esac
export HYBRID_LAYERS="$layers" HYBRID_SEED=101 HYBRID_TRAIN_EPOCHS="$epochs"
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
export HYBRID_AMX_FORWARD=1 HYBRID_AMX_BACKWARD=1 HYBRID_PERSISTENT_HS_CACHE=1
export TFS_HS_CACHE_MAX_BYTES=3221225472 TFS_AMX_PERMISSION_CACHE=1
export PYTHONPATH="$root/build/extension:$root/python:$root/csrc"
export LD_LIBRARY_PATH="/home/huangjianqiang_group/hdacp1/.local/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export HYBRID_OUTPUT="$run/training_detailed.csv" HYBRID_TIMING_METADATA="$run/timing_markers.json" TFS_SUMMARY_OUTPUT="$run/summary.json"
hostname > "$run/hostname.txt"; lscpu > "$run/lscpu.txt"; numactl -H > "$run/numa.txt"; taskset -pc $$ > "$run/affinity.txt" 2>&1 || true
env | sort | grep -E '^(TFS_|HYBRID_|OMP_|MKL_|PYTHONPATH=|LD_LIBRARY_PATH=)' > "$run/environment_start.txt"
sha256sum "$root/tests/run_shared_tfs_dataset.py" "$root/python/tfs_train/datasets.py" "$root/python/tfs_train/authority_model.py" "$root/scripts/tfs_standard_env.sh" "$root/build/extension"/tfs_train_v2_c0_ext*.so > "$run/source_input_sha256.tsv"
cat > "$run/manifest.json" <<EOF
{"method":"tfs_final_pre_numa","dataset":"$graph","layers":$layers,"threads":$threads,"epochs":$epochs,"seed":101,"profile":"final_pre_numa","partition":"${SLURM_JOB_PARTITION:-intel}","allocation":"exclusive single-node in ordinary intel pool","offline_preprocessing":true,"preprocessing":"canonical CSR cache built by prerequisite job; excluded from this cell timing","timing_scope":"each epoch records train_step_ms and evaluation_ms; formal steady statistic is median epoch 2..$epochs of their sum","comparison_scope":"TFS-only extension matrix; not a fair DGL comparison"}
EOF
start_ns=$(date +%s%N); set +e
# Slurm runs the batch shell with a single-CPU affinity on this cluster even
# when the allocation reserves 32 CPUs.  Start the measured process as a
# 32-CPU job step so the extension's frozen-affinity safety check observes
# the allocation that this cell actually reserved.
/usr/bin/time -v -o "$run/resource_usage.txt" srun --cpu-bind=none -n 1 -c 32 "$root/tests/run_shared_tfs_dataset_step.sh" "$root/tests/run_shared_tfs_dataset.py" > "$run/stdout.log" 2> "$run/stderr.log"
code=$?; set -e; end_ns=$(date +%s%N); wall_ms=$(((end_ns-start_ns)/1000000))
if [[ "$code" -eq 0 ]]; then
  python "$root/scripts/summarize_authority_cell.py" "$run/training_detailed.csv" "$run/timing_summary.json" "$wall_ms" tfs_final_pre_numa "$graph" "$layers" "$threads" "$epochs" || code=95
fi
status=failed; [[ "$code" -eq 0 ]] && status=success
printf '{"status": "%s", "graph": "%s", "layers": %s, "threads": %s, "epochs": %s, "wall_ms": %s, "exit_code": %s}\n' "$status" "$graph" "$layers" "$threads" "$epochs" "$wall_ms" "$code" > "$run/status.json"
exit "$code"
