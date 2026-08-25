#!/usr/bin/env python3
"""Report exact 48-cell authority-matrix completion without mutating runs."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


root = Path(sys.argv[1])
expected = [
    (graph, layers, threads,
     root / f"{graph}_l{layers}_t{threads}_e200")
    for graph in ("products", "arxiv", "igb19", "igb2983")
    for layers in (2, 3)
    for threads in (1, 2, 4, 8, 16, 32)
]
complete = []
incomplete = []
missing = []
for graph, layers, threads, cell in expected:
    status_path = cell / "status.json"
    rows_path = cell / "training_detailed.csv"
    item = {"graph": graph, "layers": layers, "threads": threads,
            "cell": str(cell)}
    if not cell.is_dir():
        missing.append(item)
        continue
    status = {}
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text())
        except Exception as exc:  # preserve diagnostic state
            status = {"status": "invalid_json", "error": repr(exc)}
    row_count = None
    if rows_path.is_file():
        try:
            row_count = sum(1 for _ in csv.DictReader(rows_path.open(newline="")))
        except Exception as exc:
            item["csv_error"] = repr(exc)
    item.update({"status": status.get("status"),
                 "exit_code": status.get("exit_code"),
                 "rows": row_count})
    if status.get("status") == "success" and row_count == 200:
        complete.append(item)
    else:
        incomplete.append(item)

payload = {
    "root": str(root),
    "expected": len(expected),
    "complete": len(complete),
    "incomplete": len(incomplete),
    "missing": len(missing),
    "complete_cells": complete,
    "incomplete_cells": incomplete,
    "missing_cells": missing,
}
print(json.dumps(payload, indent=2))
