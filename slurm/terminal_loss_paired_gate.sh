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
#SBATCH -t 01:00:00
#SBATCH -J terminal_ce_pair
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/terminal_ce_pair_%j.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/terminal_ce_pair_%j.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
result="$root/results/streaming_terminal_loss/paired"
mkdir -p "$result"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
export OMP_PROC_BIND=close OMP_PLACES=cores

# Fresh processes share the same exclusive node and binding.  The strong
# materialized arm separates generic CE cleanup from the new bounded-panel
# dataflow.
for spec in current:600000 materialized:600000 panelized:131072 panelized:300000; do
  IFS=: read -r mode rt <<< "$spec"
  stem="${mode}_r${rt}"
  /usr/bin/time -v numactl --cpunodebind=0-3 --localalloc python -u \
    "$root/tests/bench_streaming_terminal_loss.py" \
    --mode "$mode" --rows 600000 --hidden 128 --classes 2983 \
    --row-tile "$rt" --class-tile 2983 --threads 32 \
    --warmups 2 --repeats 10 \
    --output "$result/igb2983_${stem}_32t.json" \
    2> "$result/igb2983_${stem}_32t.time.txt"
done

echo "TERMINAL_LOSS_PAIRED_GATE_COMPLETE"
