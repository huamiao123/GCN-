"""Runtime contracts shared by the authority TFS/DGL harnesses.

The benchmark uses a fixed logical thread count for both implementations.
PyTorch's ``torch.set_num_threads`` does not configure DGL's independent
OpenMP runtime, so every DGL entry point must call :func:`configure_dgl`
explicitly after importing DGL.  Keeping this in one module prevents a
dataset-specific script from silently falling back to DGL's host default.
"""

from __future__ import annotations

import os
from typing import Any


def configure_dgl(dgl_module: Any, threads: int) -> int:
    """Set and verify DGL's own OpenMP thread count.

    ``OMP_NUM_THREADS`` normally gives the desired default, but relying on
    that implicit initialization is fragile when a launcher, Slurm cgroup,
    or a future DGL build changes the default.  The returned value is useful
    for manifests and preflight logs.
    """

    requested = int(threads)
    if requested < 1:
        raise ValueError("DGL thread count must be positive")
    utils = getattr(dgl_module, "utils", None)
    if utils is None or not hasattr(utils, "set_num_threads"):
        raise RuntimeError("this DGL build does not expose set_num_threads")
    utils.set_num_threads(requested)
    actual = int(utils.get_num_threads())
    if actual != requested:
        raise RuntimeError(
            f"DGL OpenMP thread contract failed: requested={requested}, "
            f"actual={actual}"
        )
    return actual


def validate_authority_variant(path: str) -> None:
    """Reject legacy framework variants when the strict template is active.

    Historical edge-weight and cached-DGL paths remain available for
    provenance/ablation scripts, but silently selecting one from an authority
    launcher would change the denominator.  The canonical launcher opts into
    this guard with ``HYBRID_AUTHORITY_TEMPLATE``.
    """

    template = os.environ.get("HYBRID_AUTHORITY_TEMPLATE", "")
    if template not in {"r5-standard-v1", "final-pre-numa-v1"}:
        return
    if template == "final-pre-numa-v1" and path != "hybrid":
        raise RuntimeError(
            "final_pre_numa authority template requires HYBRID_PATH=hybrid, "
            f"got {path!r}"
        )
    if path.startswith("dgl") and path != "dgl_stock":
        raise RuntimeError(
            f"strict authority template requires HYBRID_PATH=dgl_stock, got {path!r}"
        )
    if os.environ.get("HYBRID_DETAILED_PROFILE", "0") != "0":
        raise RuntimeError("strict authority template requires detailed profiling off")
    if os.environ.get("HYBRID_DGL_OP_PROFILE", "0") != "0":
        raise RuntimeError("strict authority template requires DGL op profiling off")


__all__ = ["configure_dgl", "validate_authority_variant"]
