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
#SBATCH -J panel_ce_gate
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/panel_ce_gate_%j.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/panel_ce_gate_%j.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
result="$root/results/streaming_terminal_loss/panelized"
mkdir -p "$result"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
export OMP_PROC_BIND=close OMP_PLACES=cores

numactl --cpunodebind=0-3 --localalloc python -u \
  "$root/tests/bench_streaming_terminal_loss.py" \
  --mode check --rows 2048 --hidden 128 --classes 2983 \
  --row-tile 512 --class-tile 256 --threads 32 \
  --output "$result/check_panelized.json"

for rt in 8192 16384 32768 65536 131072 300000; do
  /usr/bin/time -v numactl --cpunodebind=0-3 --localalloc python -u \
    "$root/tests/bench_streaming_terminal_loss.py" \
    --mode panelized --rows 600000 --hidden 128 --classes 2983 \
    --row-tile "$rt" --class-tile 256 --threads 32 \
    --warmups 1 --repeats 5 \
    --output "$result/igb2983_panelized_r${rt}_32t.json" \
    2> "$result/igb2983_panelized_r${rt}_32t.time.txt"
done

echo "PANELIZED_TERMINAL_LOSS_GATE_COMPLETE"
