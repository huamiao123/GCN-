#!/usr/bin/env bash
# Run the High-D tail-shape gate against the canonical v2 extension
# extension.  This is a correctness/dispatch gate, not an end-to-end claim.
#SBATCH -p intel_expr
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --mem=240G
#SBATCH --exclusive
#SBATCH -t 00:30:00
#SBATCH -J tfs_root_tail_matrix
#SBATCH -o /home/huangjianqiang_group/hdacp1/data/wzh/final_v2_1/slurm/root_tail_matrix_%A.out
#SBATCH -e /home/huangjianqiang_group/hdacp1/data/wzh/final_v2_1/slurm/root_tail_matrix_%A.err

set -euo pipefail
script_file="$(readlink -f -- "${BASH_SOURCE[0]}")"
root="${FINAL_TFS_ROOT:-$(cd -- "$(dirname -- "$script_file")/.." && pwd)}"
export TFS_RELEASE_PROFILE=v2_highd_native
source "${root}/scripts/tfs_standard_env.sh"
outdir="${root}/runs/root_tail_matrix_${SLURM_JOB_ID:-manual}"
mkdir -p "${outdir}"
export ROOT_TAIL_OUTDIR="${outdir}"
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
export PYTHONPATH="${root}/build/extension:${root}/python:${root}/csrc"
export LD_LIBRARY_PATH="/home/huangjianqiang_group/hdacp1/.local/lib/python3.9/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export OMP_PROC_BIND=close OMP_PLACES=cores
# The standalone gate calls torch.set_num_threads(1) to keep ATen reference
# work deterministic.  Pin the extension pool explicitly so that this does
# not collapse the native worker allocation to one inherited CPU.
export TFS_WORKER_CPUS=0-63
export HYBRID_AMX_FORWARD=1 HYBRID_AMX_BACKWARD=1 TFS_PULL_ONLY_BACKWARD=1
export TFS_HIGHD_STREAM_STRICT=1 TFS_HIGHD_LOG_FALLBACK=1 TFS_HIGHD_LOG_PLAN=1
export TFS_HIGHD_BWD_BUDGET_BYTES=$((64*1024*1024))
export TFS_GLUE_E1_FUSED_DB=0 TFS_GLUE_E11_VEC_GRAD_DB=0
export TFS_GLUE_E2_VEC_STORE=0 TFS_GLUE_E5_VEC_HS=0
export TFS_GLUE_E7_FORWARD_SCHEDULE=0 TFS_GLUE_E9_INT32_COLIDX=0
export TFS_FWD_V2_SINGLE_SCAN=0 TFS_NUMA_FIRST_TOUCH=on
export TFS_HIGHD_AMX_GEMM=0

for spec in 4103,100,2991 4097,128,512 8191,100,1900 8192,256,513; do
  IFS=, read -r n k d <<< "${spec}"
  export TFS_HIGHD_N="${n}" TFS_HIGHD_K="${k}" TFS_HIGHD_D="${d}"
  export TFS_HIGHD_THREADS=32 TFS_HIGHD_AMX_GEMM_TEST=0 TFS_HIGHD_FORWARD_ONLY=0
  export HYBRID_HIGHD_SMOKE_OUTPUT="${outdir}/shape_${n}_${k}_${d}.json"
  python -u "${root}/tests/test_highd_v4_extension.py" \
    > "${outdir}/shape_${n}_${k}_${d}.log" 2>&1
done

# Exercise the dimension-driven AMX forward gate on the non-multiple tail.
export TFS_HIGHD_N=4103 TFS_HIGHD_K=100 TFS_HIGHD_D=2991
export TFS_HIGHD_THREADS=32 TFS_WORKER_CPUS=0-63
export HYBRID_HIGHD_SMOKE_OUTPUT="${outdir}/shape_4103_100_2991_forward_amx.json"
python -u "${root}/tests/test_root_highd_forward_tail.py" \
  > "${outdir}/shape_4103_100_2991_forward_amx.log" 2>&1

python - <<'PY'
import json, os, pathlib
out = pathlib.Path(os.environ["ROOT_TAIL_OUTDIR"])
rows = []
for p in sorted(out.glob("*.json")):
    rows.append(json.loads(p.read_text()))
summary = {"status": "pass", "root_extension": True, "results": rows}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
