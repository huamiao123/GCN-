#!/usr/bin/env python3
"""Retired gate for an aggregate one-scan candidate not yet implemented."""

from __future__ import annotations

import json


def main() -> None:
    print(json.dumps({
        "status": "unavailable",
        "candidate": "aggregate_highd_single_scan",
        "reason": "no one-CSR-scan aggregate kernel exists in this release",
    }))
    raise SystemExit(2)


if __name__ == "__main__":
    main()
