#!/usr/bin/env bash
#SBATCH -p intel
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J audit_fix_build

set -euo pipefail
script_file="$(readlink -f -- "${BASH_SOURCE[0]}")"
if [[ -n "${TFS_ROOT:-}" ]]; then
  root="$TFS_ROOT"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  root="$SLURM_SUBMIT_DIR"
else
  root="$(cd -- "$(dirname -- "$script_file")/.." && pwd)"
fi
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
mkdir -p "$root/build/extension" "$root/build/temp"
python "$root/csrc/setup_backward_opt_aggregate_wide.py" build_ext \
  --build-lib "$root/build/extension" \
  --build-temp "$root/build/temp"
sha256sum "$root"/build/extension/tfs_train_v2_c0_ext*.so
