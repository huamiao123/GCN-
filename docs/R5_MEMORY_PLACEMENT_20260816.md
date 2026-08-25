# R5 memory-placement experiment (2026-08-16)

## Hardware gate

The shared node exposes eight NUMA nodes (0--7), two sockets, and about
128--129 GB per node.  All nodes are ordinary DDR domains; no HBM/high-
bandwidth memory NUMA domain is exposed by `numactl -H`, `lscpu`, or `/sys`.
Therefore this machine cannot validate an HBM-vs-DDR placement claim.

## Workload and contract

`tests/bench_memory_placement.py` uses the native static-Hs producer and
cached aggregate-first forward on the same deterministic CSR fixture:

```text
N=131072, K=128, D=64, threads=8, Hs=32 MiB
```

The CPU list is one worker per NUMA node (`0,8,16,24,32,40,48,56`) and the
native Hs producer writes rows in parallel, so `--localalloc` is a genuine
first-touch placement rather than an offline copy.

## Observed placement effect

Three independent runs per policy gave these forward medians:

| policy | forward median | relative to localalloc |
|---|---:|---:|
| `numactl --localalloc` | 22.4--23.3 ms | 1.00× |
| `numactl --interleave=0-7` | 33.5--35.0 ms | 0.66--0.70× |
| `numactl --membind=0` | 39.8--62.4 ms | 0.36--0.59× |

With the four-node/32-thread launch shape used by the formal runs, one
additional pair measured 16.3 ms for local first-touch versus 19.8 ms for
interleave (about 1.22×).

The result is a memory-policy effect, not a new arithmetic kernel.  Forcing
all Hs pages to node 0 is especially harmful because every other NUMA group
performs remote random reads.  Interleave is safer than node binding but still
loses to owner-local first-touch for this source-read-heavy workload.

## Fairness boundary

The formal TFS-vs-DGL results must launch both frameworks under the identical
CPU and memory policy.  Enabling `--localalloc` only for TFS would inflate its
speedup and is therefore not allowed.  The standard scripts remain on their
existing policy until a paired TFS/DGL matrix is rerun under `--localalloc`.
The native standard gate is unchanged; this report is an auditable placement
ablation, not a replacement for the authoritative results.

## End-to-end 200-epoch check

To test whether the placement effect survives the complete workload, the
shared node ran `scripts/run_numa_e2e_igb_20260816.sh` at 32 threads. Every
row includes process startup, full-graph loading/preprocessing, 200 training
epochs, and the per-epoch validation/test pass. Both TFS and native DGL stock
used the same policy in a row; only TFS enabled the persistent-Hs/AMX
implementation. `TFS_NUMA_FIRST_TOUCH=on` was held fixed for both policies,
so the policy delta is not confounded by enabling first-touch only for TFS.

| graph/classes | memory policy | TFS wall (s) | DGL stock wall (s) | DGL/TFS |
|---|---|---:|---:|---:|
| IGB-HOM-small/19 | `localalloc` | 57.797 | 119.665 | 2.070x |
| IGB-HOM-small/19 | `interleave=0-3` | 58.137 | 124.230 | 2.137x |
| IGB-HOM-small/2983 | `localalloc` | 418.268 | 673.639 | 1.611x |
| IGB-HOM-small/2983 | `interleave=0-3` | 429.412 | 687.817 | 1.602x |

Relative to interleave, local first-touch improved TFS end-to-end wall time by
`0.59%` on 19 classes and `2.66%` on 2983 classes. The corresponding DGL
changes were `3.81%` and `2.10%`, respectively, because the external policy
also affects DGL. NUMA placement is therefore a valid generalized
optimization, but its end-to-end benefit is workload-dependent and much
smaller than the isolated Hs-forward microbenchmark; it must not be reported
as a universal multi-x TFS speedup.

All four policy runs completed 200/200 epochs with exit code 0. The TFS
1024-wide Hs cache reported `2,048,000,000` bytes and one build followed by
399 cache hits in each run; no offline preprocessing or replica cache was
used.

The experiment can be reproduced with:

```bash
numactl --cpunodebind=0-7 --localalloc \
  env TFS_WORKER_CPUS=0,8,16,24,32,40,48,56 \
  PYTHONPATH=python:csrc python tests/bench_memory_placement.py
```

For a paired TFS/DGL run, wrap both commands with the same
`scripts/run_with_memory_policy.sh` invocation and change only
`TFS_MEMORY_POLICY`; the launcher supports `localalloc`, `interleave`, and the
diagnostic `membind0` policy.
