#!/usr/bin/env python3
"""Static contract: every ``backend().<name>`` must be a registered export.

This gate exists because ``c3_forward``, ``c3_forward_transform`` and
``c3_backward_selective`` were called from ``python/tfs_train`` and from four
benchmark harnesses while having no definition in ``csrc/`` and no ``m.def``
entry in the pybind module.  Those call sites sit behind
``HYBRID_AMX_FORWARD``/``HYBRID_AMX_BACKWARD``, which every authority launcher
pins to ``1``, so the defect could not surface in the authority matrix and
survived every numerical gate.

The check is deliberately source-level: it needs neither the compiled
extension nor AMX hardware, so it runs anywhere the sources do.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINDINGS = ROOT / "csrc/experiments/products_saved_t_20260812/bindings_aggregate.cpp"
SCANNED = ("python/tfs_train", "tests")

EXPORT_RE = re.compile(r'\bm\.def\(\s*"([A-Za-z0-9_]+)"')
USE_RE = re.compile(r'backend\(\)\.([A-Za-z_][A-Za-z0-9_]*)')


def exported_symbols() -> set[str]:
    text = BINDINGS.read_text(encoding="utf-8", errors="replace")
    names = set(EXPORT_RE.findall(text))
    if not names:
        raise AssertionError(f"no pybind exports parsed from {BINDINGS}")
    return names


def used_symbols() -> dict[str, list[str]]:
    used: dict[str, list[str]] = {}
    for area in SCANNED:
        for path in sorted((ROOT / area).rglob("*.py")):
            for lineno, line in enumerate(
                    path.read_text(encoding="utf-8", errors="replace")
                    .splitlines(), 1):
                for name in USE_RE.findall(line):
                    used.setdefault(name, []).append(
                        f"{path.relative_to(ROOT).as_posix()}:{lineno}")
    return used


def main() -> None:
    exported = exported_symbols()
    used = used_symbols()
    missing = {name: sites for name, sites in used.items()
               if name not in exported}
    if missing:
        for name in sorted(missing):
            print(f"{name}: not exported by bindings_aggregate.cpp", file=sys.stderr)
            for site in missing[name]:
                print(f"    {site}", file=sys.stderr)
        raise SystemExit(
            f"{len(missing)} extension symbol(s) are called but never "
            "registered; add the m.def entry or route the call through "
            "tfs_train.native.require_non_amx_c3")
    print(f"extension symbol contract: PASS "
          f"({len(used)} used / {len(exported)} exported)")


if __name__ == "__main__":
    main()
