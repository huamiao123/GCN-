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
#SBATCH -t 02:00:00
#SBATCH -J stream_ce_gate
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/stream_ce_gate_%j.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/stream_ce_gate_%j.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
result="$root/results/streaming_terminal_loss"
mkdir -p "$result"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
export OMP_PROC_BIND=close OMP_PLACES=cores

# Small exactness gate.  This deliberately exercises all 2983 classes while
# keeping both outputs available for elementwise comparison.
numactl --cpunodebind=0-3 --localalloc python -u \
  "$root/tests/bench_streaming_terminal_loss.py" \
  --mode check --rows 2048 --hidden 128 --classes 2983 \
  --row-tile 512 --class-tile 256 --threads 32 \
  --output "$result/check_m2048_k128_d2983.json"

# Two real terminal shapes.  Each arm is a fresh process, so ru_maxrss is
# comparable and is not contaminated by the other arm's allocator high-water.
for spec in products:196615:47 igb2983:600000:2983; do
  IFS=: read -r name rows classes <<< "$spec"
  for mode in materialized streaming; do
    /usr/bin/time -v numactl --cpunodebind=0-3 --localalloc python -u \
      "$root/tests/bench_streaming_terminal_loss.py" \
      --mode "$mode" --rows "$rows" --hidden 128 --classes "$classes" \
      --row-tile 32768 --class-tile 256 --threads 32 \
      --warmups 1 --repeats 5 \
      --output "$result/${name}_${mode}_32t.json" \
      2> "$result/${name}_${mode}_32t.time.txt"
  done
done

echo "STREAMING_TERMINAL_LOSS_GATE_COMPLETE"
