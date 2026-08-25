#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=80G
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH -J yq_control
#SBATCH -o /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa/slurm/yq_control_%j.out
#SBATCH -e /home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa/slurm/yq_control_%j.err
set -euo pipefail
root=/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa
nodes=${YQ_NODES:-8192}
degrees=${YQ_DEGREES:-2,8,16,32}
dims=${YQ_DIMS:-128,160,256,512,1024,2983}
warmups=${YQ_WARMUPS:-2}
repeats=${YQ_REPEATS:-5}
out=${YQ_OUTPUT:-"$root/reports/yq_control_n${nodes}_k128_t32_20260823.csv"}
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
export PYTHONPATH="$root/build/extension:$root/python:$root/csrc"
export LD_LIBRARY_PATH="/home/huangjianqiang_group/hdacp1/.local/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
srun --cpu-bind=none -n 1 -c 32 "$root/tests/run_shared_tfs_dataset_step.sh" "$root/tests/bench_yq_dataflow_control.py" \
  --nodes "$nodes" --degrees "$degrees" --dims "$dims" \
  --threads 32 --warmups "$warmups" --repeats "$repeats" --output "$out"
