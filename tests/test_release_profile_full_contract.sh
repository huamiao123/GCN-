#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TFS_RELEASE_PROFILE=v2_2_authority

# Deliberately poison every planner-owned value before sourcing.  The
# authority profile must overwrite all of them, while hardware/dataset values
# are intentionally left to the launcher.
export TFS_HIGHD_BWD_BUDGET_BYTES=12345
export TFS_HIGHD_BWD_D_TILE=64
export TFS_HIGHD_BWD_ROW_PANEL=17
export TFS_TRANSFORM_HIGHD_D_TILE=512
export TFS_HIGHD_NATIVE_PANEL=64
export TFS_WORKSPACE_CACHE_MAX_BYTES=1
export TFS_HIGHD_PANEL_BUDGET_BYTES=1
export TFS_PULL_ONLY_BACKWARD=0
export TFS_SCALE_GRAD_BF16_NATIVE=0
export TFS_STATIC_HS=off
export TFS_NUMA_FIRST_TOUCH=off
export TFS_HIGHD_FUSED_DB=1
export TFS_HIGHD_FUSED_TRANSPOSE=1
export TFS_HIGHD_FUSED_SCALE_TRANSPOSE=1
export TFS_HIGHD_CONTIGUOUS_PANELS=0
export TFS_COLIDX=int32
export TFS_MAX_LOCAL_DW_BYTES=123
export TFS_LOCAL_DW_BUDGET_BYTES=456

source "$root/scripts/tfs_standard_env.sh"

[[ "$TFS_RELEASE_PROFILE" == "v2_2_authority" ]]
[[ -z "${TFS_HIGHD_BWD_BUDGET_BYTES+x}" ]]
[[ "$TFS_HIGHD_BWD_D_TILE" == 256 ]]
[[ "$TFS_HIGHD_BWD_ROW_PANEL" == 512 ]]
[[ "$TFS_TRANSFORM_HIGHD_D_TILE" == 256 ]]
[[ "$TFS_HIGHD_NATIVE_PANEL" == 512 ]]
[[ "$TFS_WORKSPACE_CACHE_MAX_BYTES" == "$((512*1024*1024))" ]]
[[ "$TFS_HIGHD_PANEL_BUDGET_BYTES" == "$((32*1024*1024))" ]]
[[ "$TFS_PULL_ONLY_BACKWARD" == 1 ]]
[[ "$TFS_SCALE_GRAD_BF16_NATIVE" == 1 ]]
[[ "$TFS_STATIC_HS" == on ]]
[[ "$TFS_NUMA_FIRST_TOUCH" == on ]]
[[ "$TFS_HIGHD_FUSED_DB" == 0 ]]
[[ "$TFS_HIGHD_FUSED_TRANSPOSE" == 0 ]]
[[ "$TFS_HIGHD_FUSED_SCALE_TRANSPOSE" == 0 ]]
[[ "$TFS_HIGHD_CONTIGUOUS_PANELS" == 1 ]]
[[ "$TFS_COLIDX" == int64 ]]
[[ "$TFS_MAX_LOCAL_DW_BYTES" == 0 ]]
[[ "$TFS_LOCAL_DW_BUDGET_BYTES" == 0 ]]

echo "release profile full contract: PASS"
