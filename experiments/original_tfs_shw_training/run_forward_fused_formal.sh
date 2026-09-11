#!/usr/bin/env bash
#SBATCH -p intel_expr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=16G
#SBATCH --exclusive
#SBATCH -t 00:45:00
#SBATCH -J fwd_fused_formal
#SBATCH -o /home/huangjianqiang_group/hdacp1/data/wzh/TFS-Train/runs/forward_fused_formal_20260810_v2-%j.out
#SBATCH -e /home/huangjianqiang_group/hdacp1/data/wzh/TFS-Train/runs/forward_fused_formal_20260810_v2-%j.err
set -euo pipefail
root=/home/huangjianqiang_group/hdacp1/data/wzh/TFS-Train
run=$root/runs/forward_fused_formal_20260810_v2
agents=/home/huangjianqiang_group/hdacp1/data/wzh/AGENTS.md
skill=/home/huangjianqiang_group/hdacp1/data/wzh/skills/tfs-research-engineering/SKILL.md
test ! -e "$run" || exit 90
mkdir -p "$run/raw" "$run/binding"
exec > >(tee "$run/stdout.log") 2> >(tee "$run/stderr.log" >&2)
cat "$agents" > "$run/AGENTS.read.txt"; cat "$skill" > "$run/SKILL.read.txt"; sha256sum "$agents" "$skill" > "$run/preflight_sha256.tsv"
for s in MATH_SPEC_V2 KERNEL_LAYOUT_V2 EXPERIMENT_SPEC_V2 PYTORCH_EXTENSION_SPEC; do cat "$root/docs/spec/$s.md" > "$run/$s.read.md"; done
git -C "$root" status --short > "$run/git_status.txt"; git -C "$root" rev-parse HEAD > "$run/git_commit.txt"
sinfo > "$run/sinfo.txt"; squeue -u hdacp1 > "$run/squeue.txt" || true; sacct -u hdacp1 -S 2026-08-10 > "$run/sacct.txt" || true
hostname > "$run/hostname.txt"; lscpu > "$run/lscpu.txt"; numactl -H > "$run/numa.txt"; taskset -pc $$ > "$run/affinity.txt"
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
module load intel/intel-oneapi-mkl/2023.2.0/oneapi2023.2.0-vuihbr3
flags=(-std=c++17 -O3 -DNDEBUG -qopenmp -qmkl -mamx-tile -mamx-bf16 -mavx512f -mavx512bw -mavx512bf16)
icpx --version > "$run/compiler.txt"; printf '%s\n' "${flags[*]}" > "$run/compiler_flags.txt"
icpx "${flags[@]}" "$root/csrc/forward_exp/normalized_fused_forward_bench.cpp" -o "$run/bench" 2> "$run/build.log"
fixture=$root/runs/forward_fused_smoke_20260810_v5/ogbn_arxiv_forward.bin
sha256sum "$run/bench" > "$run/binary_sha256.tsv"; sha256sum "$fixture" > "$run/input_sha256.tsv"; sha256sum "$root/csrc/forward_exp/normalized_fused_forward_bench.cpp" > "$run/source_sha256.tsv"
export OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
for d in 40 128; do
  if test "$d" = 40; then r=32; else r=16; fi
  for t in 1 2 4 8 16 32; do
    export OMP_NUM_THREADS=$t MKL_NUM_THREADS=$t
    if test "$t" -le 8; then
      numactl --cpunodebind=0 --membind=0 "$run/bench" --fixture "$fixture" --d "$d" --threads "$t" --warmups 2 --repeats 7 --group-rows "$r" | tee "$run/raw/d${d}_t${t}_local0.log"
    else
      if test "$t" = 16; then nodes=0-1; else nodes=0-3; fi
      numactl --cpunodebind="$nodes" --membind=0 "$run/bench" --fixture "$fixture" --d "$d" --threads "$t" --warmups 2 --repeats 7 --group-rows "$r" | tee "$run/raw/d${d}_t${t}_memory_local0.log"
      numactl --cpunodebind="$nodes" --interleave="$nodes" "$run/bench" --fixture "$fixture" --d "$d" --threads "$t" --warmups 2 --repeats 7 --group-rows "$r" | tee "$run/raw/d${d}_t${t}_memory_interleave.log"
    fi
  done
done
printf '%s\n' '{"status":"success","class":"formal-performance","paper_eligible":true,"scope":"official ogbn-arxiv normalized forward operator"}' > "$run/status.json"
