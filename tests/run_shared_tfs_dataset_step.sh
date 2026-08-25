#!/usr/bin/env bash
# Establish the worker map from the actual Slurm job-step process, not from
# the batch shell.  The extension freezes this map when it creates its
# persistent pool, so it must be valid before Python imports the backend.
set -euo pipefail
export TFS_WORKER_CPUS="$(taskset -pc $$ | sed 's/.*: //')"
exec numactl --localalloc python -u "$@"
