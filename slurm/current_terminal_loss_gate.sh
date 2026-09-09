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
#SBATCH -t 00:30:00
#SBATCH -J current_ce_gate
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/current_ce_gate_%j.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/current_ce_gate_%j.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
result="$root/results/streaming_terminal_loss/current"
mkdir -p "$result"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
export OMP_PROC_BIND=close OMP_PLACES=cores

for mode in current materialized; do
  /usr/bin/time -v numactl --cpunodebind=0-3 --localalloc python -u \
    "$root/tests/bench_streaming_terminal_loss.py" \
    --mode "$mode" --rows 600000 --hidden 128 --classes 2983 \
    --row-tile 300000 --class-tile 2983 --threads 32 \
    --warmups 1 --repeats 5 \
    --output "$result/igb2983_${mode}_32t.json" \
    2> "$result/igb2983_${mode}_32t.time.txt"
done

echo "CURRENT_TERMINAL_LOSS_GATE_COMPLETE"
