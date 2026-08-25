#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
expected='planner=on highd_stream=on highd_native=auto pull_only=on static_hs=on numa_first_touch=on budget=67108864 d_tile=256 row_panel=512 panel_budget=33554432 colidx=int64'

actual="$(env -i PATH=/usr/bin:/bin \
  TFS_RELEASE_PROFILE=v2_2_authority \
  TFS_HIGHD_STREAM_V1=0 TFS_STATIC_HS=off \
  bash -c 'source "$1/scripts/tfs_standard_env.sh"; printf "%s" "$TFS_RUNTIME_CONTRACT"' \
  bash "$root")"
test "$actual" = "$expected"

if env -i PATH=/usr/bin:/bin TFS_RELEASE_PROFILE=unknown \
    bash -c 'source "$1/scripts/tfs_standard_env.sh"' bash "$root"; then
  echo "unknown release profile unexpectedly succeeded" >&2
  exit 1
fi

echo "release profile: PASS"
