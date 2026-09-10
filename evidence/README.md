# Verified shadow evidence

This directory contains machine-readable evidence collected from the exact
source integrated in this branch.

## Authority-versus-planner matrix

- Slurm array: `10387812`
- Shape: Products D=47 and IGB-small D=2983, L2/L3,
  1/2/4/8/16/32 threads
- Protocol: one exclusive single-socket node per cell; authority and planned
  arm interleaved inside the same process; 2 warm-ups and 9 measured repeats
- Outcome: 24/24 JSON files report `status=pass`

Artifacts:

- `authority_vs_planned_dense_matrix/`: raw per-cell JSON
- `planned_dense_matrix_summary.json`: validated compact summary
- `planned_dense_matrix_table.md`: complete readable table

## Paired 200-epoch convergence

- Slurm array: `10387962`
- Shape: Products D=47 L2 32T and IGB-small D=2983 L2 32T
- Protocol: identical initialization and dropout seed per epoch; evaluation is
  outside train-step timing
- Outcome: 2/2 JSON files report `status=pass`

Artifacts are under `planned_dense_convergence/` and include every epoch's
loss/time plus periodic validation/test measurements.

## Build provenance

- Build job: `10387317`, exit `0:0`
- Native extension SHA-256:
  `79385b1cee4b624264f906b4882d353b2b3c1dcb3b485c0e2990af39bdfdc402`
- Unit/native gate: `10387533`, `13 passed`
- Final gate after tightening unmeasured fallbacks: `10388041`, `15 passed`
- First explicit-planner gate: `10387552`, `status=pass`

The source hashes used on the server matched the local files byte-for-byte at
acceptance time; see `docs/PLANNED_SHADOW_ACCEPTANCE_20260910.md` for the
interpretation and claim boundary.

## Deliberately rejected low-thread extension

`native_logits_low_threads/` contains Slurm array `10388006`. Native logits
improved train-step time at 1/2/4 threads, but every cell exceeded the
experiment's strict `dH relative-L2 < 1e-4` gate by a small amount. These are
kept as negative/boundary evidence; the planner does not enable this path.
