#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export TFS_RELEASE_PROFILE=final_pre_numa

# Poison the values that caused the v2.3 provenance/dataflow bugs.  The
# authority profile must assign, not inherit, every one of them.
export TFS_AGGREGATE_SAVED=on
export TFS_STATIC_AGGREGATE=off
export TFS_NUMA_FIRST_TOUCH=off
export TFS_NUMA_PRIVATE=on
export TFS_NUMA_REDUCE=on
export TFS_TRANSFORM_HIGHD_NATIVE=on
export TFS_TRANSFORM_HIGHD_SINGLE_SCAN=on
export TFS_AGGREGATE_DSLAB_SINGLE_SCAN=on
export TFS_LOCALITY_SCHEDULE=on
export TFS_HS_REPLICA=on
export TFS_WORKSPACE_CACHE_MAX_BYTES=1

source "$root/scripts/tfs_standard_env.sh"

[[ "$TFS_RELEASE_PROFILE" == final_pre_numa ]]
[[ "$TFS_PROFILE_STATUS" == authority ]]
[[ "$TFS_AGGREGATE_SAVED" == auto ]]
[[ "$TFS_STATIC_AGGREGATE" == auto ]]
[[ "$TFS_NUMA_FIRST_TOUCH" == on ]]
[[ "$TFS_NUMA_PRIVATE" == off ]]
[[ "$TFS_NUMA_REDUCE" == off ]]
[[ "$TFS_TRANSFORM_HIGHD_NATIVE" == off ]]
[[ "$TFS_TRANSFORM_HIGHD_SINGLE_SCAN" == off ]]
[[ "$TFS_AGGREGATE_DSLAB_SINGLE_SCAN" == off ]]
[[ "$TFS_LOCALITY_SCHEDULE" == off ]]
[[ "$TFS_HS_REPLICA" == off ]]
[[ "$TFS_WORKSPACE_CACHE_MAX_BYTES" == "$((2*1024*1024*1024))" ]]
[[ -z "${TFS_HIGHD_BWD_BUDGET_BYTES+x}" ]]

PYTHONPATH="$root/python" python -c \
  'from tfs_train.execution_plan import runtime_contract_line; print(runtime_contract_line(validate=True))'

echo "final_pre_numa profile contract: PASS"
