#!/usr/bin/env bash
# Build the v2 AMX extension without modifying any other release directory.
set -euo pipefail
script_file="$(readlink -f -- "${BASH_SOURCE[0]}")"
root="${FINAL_TFS_ROOT:-$(cd -- "$(dirname -- "$script_file")/.." && pwd)}"
module load intel/intel-oneapi-compilers/2024.1.0/gcc8.5.0-5ndeojp
mkdir -p "${root}/build/extension" "${root}/build/temp"
python "${root}/csrc/setup_backward_opt_aggregate_wide.py" \
  build_ext --build-lib "${root}/build/extension" \
  --build-temp "${root}/build/temp"
sha256sum "${root}/build/extension"/tfs_train_v2_c0_ext*.so \
  > "${root}/build/extension.sha256"
cat "${root}/build/extension.sha256"
