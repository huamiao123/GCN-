# R5 source-level fairness audit (2026-08-16)

## Scope

This audit covers the 200-epoch run launched by
`scripts/run_conservative_e2e_20260816.sh` (diagnostic Slurm job 10008651), not older
tables that used a different launcher or extension.  The run uses one
exclusive node, CPU nodes 0--3, `numactl --interleave=0-3`, 32 PyTorch
threads, and one process for each of `tfs_dynamic`, `tfs_cache`, and
`dgl_stock`.

## What is aligned

* The remote environment is DGL **2.1.0** and PyTorch 2.5.1+cpu.
* The timed DGL path is native `dgl.nn.pytorch.GraphConv(norm="both")`;
  `HYBRID_PATH=dgl_stock` clears the TFS switches and reports zero cache
  entries.  It does not use the historical external edge-weight path or the
  custom `dgl_cached` wrapper.
* DGL's own source selects transform-first when `K>D` and aggregate-first
  otherwise, matching `execution_plan.py`'s `D>=K` rule.  Both paths use the
  same CSR edge list, one explicit self-loop in DGL, and the same symmetric
  degree scale (`rsqrt(degree+1)`) as TFS's implicit self-loop.
* Weight/bias shapes, Adam settings, dropout, seed, epoch count, CPU affinity,
  and process-level wall timer are shared.  The numerical DGL stock gates
  pass; the mixed-precision TFS gate is within its documented BF16 tolerance.
* TFS cache contracts are actually exercised.  Products/Arxiv cache rows
  report 399 hits/1 miss for the static first layer; IGB 1024->128 reports the
  same for persistent Hs.  IGB 128->2983 is the shape-driven
  `wide_aggregate` saved-pulled path, not a hidden C2 fallback.

## Problems found in the old authority template

1. **Native profile logging was inside the timed TFS process.**  The launcher
   exported `TFS_INTERNAL_PROFILE=1`; the C++ extension then emitted and
   flushed per-forward/per-backward profile lines (`std::endl`).  A Products
   cache log contains about 8,000 lines while the paired DGL log contains one.
   This slows TFS and makes the comparison conservative, but it is still an
   asymmetry.  The launcher now defaults `R5_INTERNAL_PROFILE=0`; profiling is
   diagnostic-only.
2. **Active-row `auto` is not universally profitable.**  The native backward
   builds an active mask whenever `D<=128`, then disables the actual skip when
   density exceeds 0.25.  The current logs show `use_active=0` for Arxiv
   (`density=0.537`) and IGB (`0.60`/`0.98`), so those rows should not be sold
   as an unconditional speedup.  Products D=47 has `use_active=1` at density
   about 0.08.  The native fix now accumulates density during the existing E1
   row pass (no second full-N count scan), and disables the vectorized E1/E11
   D=128 mask read that previously could inspect an uninitialized workspace.
   The feature remains a runtime density gate, not a dataset-name heuristic.
3. **The planner was re-evaluated with the real N only for logging.**  Models
   were constructed with `plan_layers(1, ...)`, then `HybridGCN.forward()`
   replaced only `self._plans`; the already-created convolution objects kept
   their construction-time plan.  The authority entry points now pass the
   actual node count before module construction and verify it remains fixed on
   the first forward.
4. **The previous `TFS_COLIDX=auto` fix had an over-broad process-global
   result.**  A valid first graph could make a later graph skip validation.
   The native source now keys the validation cache by CSR pointer and length
   instead of a single global boolean.  The rebuilt extension must be used for
   the next authority run.
5. **DGL thread configuration was implicit.**  `OMP_NUM_THREADS=32` normally
   initializes DGL to 32, but DGL has an independent OpenMP runtime.  The
   shared `tfs_train.standard_runtime.configure_dgl()` helper now calls and
   verifies `dgl.utils.set_num_threads(threads)` in every detailed entry
   point, so a launcher/default change cannot silently alter the comparison.
6. **There are multiple historical launchers and duplicated model wrappers.**
   They differ in profile flags, `PYTHONPATH`, cache switches, and DGL
   variants.  Only the conservative launcher and its three detailed entry
   points are in scope for the next authority matrix; old scripts are
   reproduction/ablation scripts, not interchangeable templates.

## Verification status after the fixes

The shared DGL wrapper and the native active-row/planner fixes compile on the
Intel node.  The generic single-scan, active-row-wide, and aggregate-saved-V4
extension smoke tests all pass.  The independent stock-DGL numerical gate
passes (Products fixture: logits relative L2 `2.22e-7`, input-gradient
relative L2 `2.29e-7`, parameter gradients below `4e-7`) with the shared
wrapper and an explicitly verified 32-thread DGL runtime.

The mixed-precision TFS reference gate on the Products fixture remains the
known BF16 reduction-order case: input-gradient relative L2 is about `3.32%`
and the default `3%` gate therefore fails; the result is unchanged with
active-row on/off.  This is a numerical-tolerance issue to report explicitly,
not evidence that DGL was slowed or that the cache returned stale values.

The first strict-template submission (10009509) was cancelled before the
matrix advanced because its memory policy accidentally used first-touch while
10008651 used interleave.  The replacement strict authority job **10009732**
uses profile off, explicit DGL thread setup, the shared stock wrapper, the
rebuilt native extension, and the same `numactl --interleave=0-3` policy for
both implementations.  Its output root is
`runs/strict_template_authority_interleave_20260816`; it supersedes 10008651
when all rows complete.

During 10009732, both Arxiv stock-DGL rows exited before timing because the
new shared wrapper was called without an explicit input dimension and fell
back to its Products default (`100`), while Arxiv features are `128`.  The
resulting `169343x128 · 100x128` matmul error is a wrapper-call defect, not a
performance or fairness finding.  All Arxiv DGL rows from 10009732 are
invalid and excluded; the detailed Arxiv entry points now pass
`in_dim=hidden_dim=128`, and repair array job **10009870** reruns only the
Arxiv 2/3-layer points under a separate output root.

## TFS cache verdict

The cache implementation itself has no observed cache-hit correctness failure:
the native producer/consumer smoke tests pass, cache rows show exactly one
build and subsequent hits, and the training losses/metrics remain aligned with
the paired FP32 DGL run within the stated mixed-precision tolerance.  The
cache is not accidentally enabled in DGL.

The 10008651 wall numbers should nevertheless be treated as **diagnostic, not
the final strict authority table**, because they include the old TFS profile
logging overhead and rely on implicit DGL thread initialization.  The clean
10009732 run uses the profile-off launcher, the rebuilt extension, the
explicit DGL thread helper, one shared DGL wrapper, and the same shape-driven
TFS template for all three graphs.  It also records active-row density/use
separately; the repaired Arxiv rows from 10009870 will be joined only after
their exit code is zero.

The comparison remains an optimized-TFS-versus-native-DGL comparison: the
TFS static T0/Hs cache is intentionally the method under test, while stock DGL
recomputes its native `GraphConv` aggregation.  For a kernel-level
no-cache comparison, use `tfs_dynamic` versus `dgl_stock` as a separate row;
do not call the cache ratio a stock-kernel equivalence.
