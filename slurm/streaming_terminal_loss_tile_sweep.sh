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
#SBATCH -J stream_ce_tiles
#SBATCH -o /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/stream_ce_tiles_%j.out
#SBATCH -e /online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905/runs/stream_ce_tiles_%j.err

set -euo pipefail
root=/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
result="$root/results/streaming_terminal_loss/tile_sweep"
mkdir -p "$result"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32
export OMP_PROC_BIND=close OMP_PLACES=cores

for rt in 16384 32768 65536 131072; do
  for ct in 256 512 1024 2983; do
    /usr/bin/time -v numactl --cpunodebind=0-3 --localalloc python -u \
      "$root/tests/bench_streaming_terminal_loss.py" \
      --mode streaming --rows 600000 --hidden 128 --classes 2983 \
      --row-tile "$rt" --class-tile "$ct" --threads 32 \
      --warmups 1 --repeats 3 \
      --output "$result/igb2983_streaming_r${rt}_c${ct}_32t.json" \
      2> "$result/igb2983_streaming_r${rt}_c${ct}_32t.time.txt"
  done
done

echo "STREAMING_TERMINAL_LOSS_TILE_SWEEP_COMPLETE"
