#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=12G
#SBATCH -J fwd_r_sweep
#SBATCH -o /home/huangjianqiang_group/hdacp1/data/wzh/TFS-Train/runs/forward_fused_r_sweep_20260810_v1-%j.out
#SBATCH -e /home/huangjianqiang_group/hdacp1/data/wzh/TFS-Train/runs/forward_fused_r_sweep_20260810_v1-%j.err
set -euo pipefail
root=/home/huangjianqiang_group/hdacp1/data/wzh/TFS-Train
run=$root/runs/forward_fused_r_sweep_20260810_v1
agents=/home/huangjianqiang_group/hdacp1/data/wzh/AGENTS.md
skill=/home/huangjianqiang_group/hdacp1/data/wzh/skills/tfs-research-engineering/SKILL.md
test ! -e "$run" || exit 90
mkdir -p "$run/raw"
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
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
for d in 40 128; do for r in 16 32 64 128 256 512; do
  numactl --cpunodebind=0 --membind=0 "$run/bench" --fixture "$fixture" --d "$d" --threads 8 --warmups 1 --repeats 3 --group-rows "$r" | tee "$run/raw/d${d}_r${r}.log"
done; done
printf '%s\n' '{"status":"success","class":"development-parameter-sweep","paper_eligible":false}' > "$run/status.json"
