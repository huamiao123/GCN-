"""Standalone remote gate for the Transform-HighD auto/explicit split."""

from __future__ import annotations

import os

from tfs_train.execution_plan import build_layer_plan


def main() -> None:
    os.environ["TFS_TRANSFORM_HIGHD_STREAM_V1"] = "auto"
    one = build_layer_plan(4103, 1024, 257, threads=32)
    multi = build_layer_plan(4099, 1024, 513, threads=32)
    assert one.selected_impl == "transform_stream", one.selected_impl
    assert len(one.d_slabs) == 1, one.d_slabs
    assert multi.selected_impl == "legacy_wide_output", multi.selected_impl
    assert multi.fallback_reason == "transform_highd_native_pending"
    os.environ["TFS_TRANSFORM_HIGHD_STREAM_V1"] = "on"
    forced = build_layer_plan(4099, 1024, 513, threads=32)
    assert forced.selected_impl == "transform_stream", forced.selected_impl
    forced.validate()
    print("transform auto gate: PASS")


if __name__ == "__main__":
    main()
