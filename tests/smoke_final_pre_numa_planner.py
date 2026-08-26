#!/usr/bin/env python3
"""Dependency-free production planner/generalization gate."""

from __future__ import annotations

import contextlib
import os

from tfs_train.execution_plan import (build_layer_plan, plan_layers,
                                      runtime_contract_line)
from tfs_train.standard_runtime import validate_authority_variant


@contextlib.contextmanager
def environment(**updates):
    previous = {name: os.environ.get(name) for name in updates}
    try:
        for name, value in updates.items():
            os.environ[name] = str(value)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


assert os.environ.get("TFS_RELEASE_PROFILE") == "final_pre_numa"
assert os.environ.get("TFS_AGGREGATE_SAVED") == "auto"
assert os.environ.get("TFS_PROFILE_STATUS") == "authority"
assert "numa_first_touch=on" in runtime_contract_line(validate=True)
os.environ["TFS_HS_CACHE_MAX_BYTES"] = str(3_000_000_000)

expected = {
    (100, 128, 47, 2): ("aggregate_static_v3", "native_c3"),
    (100, 128, 47, 3):
        ("aggregate_static_v3", "native_c3", "native_c3"),
    (1024, 128, 19, 3):
        ("native_wide_k", "native_c3", "native_c3"),
    (1024, 128, 2983, 3):
        ("native_wide_k", "native_c3", "aggregate_highd_full"),
}
for (input_dim, hidden, output_dim, layers), variants in expected.items():
    dims = [input_dim] + [hidden] * (layers - 1) + [output_dim]
    plans = plan_layers(1_000_003, dims, threads=32)
    actual = tuple(plan.execution_variant for plan in plans)
    assert actual == variants, (dims, actual, variants)
    assert tuple(plan.layer_index for plan in plans) == tuple(range(layers))
    assert not any(plan.execution_variant == "aggregate_saved_v4"
                   for plan in plans)
    for plan in plans:
        plan.assert_dispatch(plan.execution_variant)
        plan.validate()

for depth in (2, 3, 4, 5):
    plans = plan_layers(4103, [100] + [128] * (depth - 1) + [47],
                        threads=8)
    assert len(plans) == depth

# N/K/D tails and K>256 are legal without a dataset-specific switch.
tail_shapes = ((4103, 1025, 127), (4099, 300, 129),
               (8191, 2048, 513), (4097, 4096, 4096))
for n, k, d in tail_shapes:
    plan = build_layer_plan(n, k, d, threads=16, compute_dx=True)
    assert plan.k == k and plan.d == d and plan.n == n
    assert plan.d_slabs[-1][1] == d
    plan.validate()

# There is no independent native one-CSR-scan aggregate kernel in this
# release.  The variant exists only as an explicit d-slab reference stream
# with one final CSR pull, so the contract it must satisfy is "never auto,
# and never labelled as validated" -- not "explicit requests fail".  The same
# contract is asserted from the other side by
# tests/test_execution_plan.py::test_aggregate_dslab_single_scan_is_explicit_reference_only.
auto_aggregate = build_layer_plan(
    4097, 1024, 1024, threads=32, compute_dx=True)
assert auto_aggregate.execution_variant != "aggregate_highd_single_scan"
assert auto_aggregate.selection_policy == "auto"
with environment(TFS_AGGREGATE_DSLAB_SINGLE_SCAN="on",
                 TFS_HIGHD_NATIVE_STREAM="off"):
    explicit_aggregate = build_layer_plan(
        4097, 1024, 1024, threads=32, compute_dx=True)
assert explicit_aggregate.execution_variant == "aggregate_highd_single_scan"
assert explicit_aggregate.selection_policy == "explicit"
single_scan = next(c for c in explicit_aggregate.candidates
                   if c.name == "aggregate_highd_single_scan")
assert single_scan.performance_state == "experimental"
assert not single_scan.validated_for_auto

auto_transform = build_layer_plan(
    4097, 1024, 513, threads=32, compute_dx=True)
assert auto_transform.execution_variant == "transform_highd_legacy"
with environment(TFS_TRANSFORM_HIGHD_STREAM_V1="on"):
    explicit = build_layer_plan(
        4097, 1024, 513, threads=32, compute_dx=True)
    assert explicit.execution_variant == "transform_highd_stream"
    assert explicit.selection_policy == "explicit"

os.environ["HYBRID_AUTHORITY_TEMPLATE"] = "final-pre-numa-v1"
os.environ["HYBRID_DETAILED_PROFILE"] = "0"
os.environ["HYBRID_DGL_OP_PROFILE"] = "0"
validate_authority_variant("hybrid")
try:
    validate_authority_variant("dgl_stock")
except RuntimeError:
    pass
else:
    raise AssertionError("final authority accepted a non-TFS path")

print("final_pre_numa planner/generalization smoke: PASS")
