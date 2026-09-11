#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=12G
#SBATCH -J fwd_fused_smoke
#SBATCH -o /home/huangjianqiang_group/hdacp1/data/wzh/TFS-Train/runs/forward_fused_smoke_20260810_v5-%j.out
#SBATCH -e /home/huangjianqiang_group/hdacp1/data/wzh/TFS-Train/runs/forward_fused_smoke_20260810_v5-%j.err
set -euo pipefail
root=/home/huangjianqiang_group/hdacp1/data/wzh/TFS-Train
run=$root/runs/forward_fused_smoke_20260810_v5
agents=/home/huangjianqiang_group/hdacp1/data/wzh/AGENTS.md
skill=/home/huangjianqiang_group/hdacp1/data/wzh/skills/tfs-research-engineering/SKILL.md
test ! -e "$run" || exit 90
mkdir -p "$run"
exec > >(tee "$run/stdout.log") 2> >(tee "$run/stderr.log" >&2)
cat "$agents" > "$run/AGENTS.read.txt"
cat "$skill" > "$run/SKILL.read.txt"
sha256sum "$agents" "$skill" > "$run/preflight_sha256.tsv"
for s in MATH_SPEC_V2 KERNEL_LAYOUT_V2 EXPERIMENT_SPEC_V2 PYTORCH_EXTENSION_SPEC; do cat "$root/docs/spec/$s.md" > "$run/$s.read.md"; done
git -C "$root" status --short > "$run/git_status.txt"
git -C "$root" rev-parse HEAD > "$run/git_commit.txt"
sinfo > "$run/sinfo.txt"
squeue -u hdacp1 > "$run/squeue.txt" || true
sacct -u hdacp1 -S 2026-08-10 > "$run/sacct.txt" || true
hostname > "$run/hostname.txt"
lscpu > "$run/lscpu.txt"
numactl -H > "$run/numa.txt"
taskset -pc $$ > "$run/affinity.txt"
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
module load intel/intel-oneapi-mkl/2023.2.0/oneapi2023.2.0-vuihbr3
icpx --version > "$run/compiler.txt"
flags=(-std=c++17 -O3 -DNDEBUG -qopenmp -qmkl -mamx-tile -mamx-bf16 -mavx512f -mavx512bw -mavx512bf16)
printf '%s\n' "${flags[*]}" > "$run/compiler_flags.txt"
icpx "${flags[@]}" "$root/csrc/forward_exp/normalized_fused_forward_bench.cpp" -o "$run/normalized_fused_forward_bench" 2> "$run/build.log"
sha256sum "$run/normalized_fused_forward_bench" > "$run/binary_sha256.tsv"
sha256sum "$root/csrc/forward_exp/normalized_fused_forward_bench.cpp" "$root/csrc/forward_exp/export_arxiv_forward_fixture.py" > "$run/source_sha256.tsv"
old=$root/runs/c5_fullstep_arxiv_20260804_v3
export PYTHONPATH="$old/extension/lib:$old/packages:$root/python"
python "$root/csrc/forward_exp/export_arxiv_forward_fixture.py" --arxiv-root "$old/dataset/arxiv" --output "$run/ogbn_arxiv_forward.bin" | tee "$run/export.json"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_DYNAMIC=FALSE OMP_PROC_BIND=close OMP_PLACES=cores
numactl --cpunodebind=0 --membind=0 "$run/normalized_fused_forward_bench" --fixture "$run/ogbn_arxiv_forward.bin" --d 40 --threads 8 --warmups 0 --repeats 1 --group-rows 64 | tee "$run/d40.log"
numactl --cpunodebind=0 --membind=0 "$run/normalized_fused_forward_bench" --fixture "$run/ogbn_arxiv_forward.bin" --d 128 --threads 8 --warmups 0 --repeats 1 --group-rows 64 | tee "$run/d128.log"
grep -q 'ERROR,path=fused_amx' "$run/d40.log"
grep -q 'ERROR,path=fused_amx' "$run/d128.log"
printf '%s\n' '{"status":"success","class":"development-smoke","paper_eligible":false}' > "$run/status.json"
