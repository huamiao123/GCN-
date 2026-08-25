#!/usr/bin/env bash
set -euo pipefail

if (($# == 0)); then
  echo "usage: TFS_MEMORY_POLICY=localalloc|interleave|membind0 $0 command [args...]" >&2
  exit 2
fi

policy="${TFS_MEMORY_POLICY:-interleave}"
cpu_nodes="${TFS_CPU_NODES:-0-3}"
memory_nodes="${TFS_MEMORY_NODES:-${cpu_nodes}}"

case "${policy}" in
  localalloc)
    exec numactl --cpunodebind="${cpu_nodes}" --localalloc "$@"
    ;;
  interleave)
    exec numactl --cpunodebind="${cpu_nodes}" \
      --interleave="${memory_nodes}" "$@"
    ;;
  membind0)
    exec numactl --cpunodebind="${cpu_nodes}" --membind=0 "$@"
    ;;
  *)
    echo "TFS_MEMORY_POLICY must be localalloc|interleave|membind0" >&2
    exit 2
    ;;
esac
