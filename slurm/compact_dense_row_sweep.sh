#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=120G
#SBATCH --exclusive
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=32
#SBATCH --threads-per-core=1
#SBATCH -t 00:35:00
#SBATCH -J dense_m_sweep
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/dense_m_sweep_%j.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/dense_m_sweep_%j.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
source "$root/scripts/tfs_standard_env.sh"
export PYTHONPATH="$root/build_compact_logits_v2/extension:$root/python:$root/csrc:$root/tests"
export TFS_WORKER_CPUS="$(taskset -pc $$ | sed 's/.*: //')"
export TFS_COMPACT_DW_T4=1
for rows in 8192 32768 131072; do
  for dim in 512 2983; do
    for threads in 8 32; do
      export OMP_NUM_THREADS=$threads MKL_NUM_THREADS=$threads
      dw_out="$root/results/compact_dense_row_sweep/dw_m${rows}_d${dim}_${threads}t.json"
      logits_out="$root/results/compact_dense_row_sweep/logits_m${rows}_d${dim}_${threads}t.json"
      mkdir -p "$(dirname "$dw_out")"
      numactl --cpunodebind=0-3 --localalloc python -u \
        "$root/tests/bench_compact_dw_amx_shadow.py" \
        --rows "$rows" --classes "$dim" --threads "$threads" \
        --output "$dw_out"
      numactl --cpunodebind=0-3 --localalloc python -u \
        "$root/tests/bench_compact_logits_amx_shadow.py" \
        --rows "$rows" --classes "$dim" --threads "$threads" \
        --output "$logits_out"
    done
  done
done
