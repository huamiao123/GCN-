"""Collect shadow dense-fusion JSON artifacts into one auditable CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RESULT_GROUPS = {
    "native_dw_scope_igb_t4": "native_dw_t4",
    "native_dw_t4_generalization": "native_dw_t4",
    "native_logits_scope_igb_v2": "native_logits_v2",
    "native_logits_v2_generalization": "native_logits_v2",
    "authority_vs_native_dw": "authority_vs_native_dw",
    "authority_vs_final_dense": "authority_vs_final_dense",
    "authority_vs_final_dense_rowtile": "authority_vs_final_dense",
    "compact_dense_row_sweep": "compact_dense_micro",
}


def nested(document, *keys):
    value = document
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def row_for(path, optimization, document):
    terminal = document.get("terminal", {})
    train_step = document.get("train_step", {})
    correctness = document.get("correctness", {})
    if optimization == "compact_dense_micro":
        operation = "native_dw_t4" if path.name.startswith("dw_") else \
            "native_logits_v2"
    else:
        operation = optimization
    return {
        "artifact": path.as_posix(),
        "optimization": operation,
        "status": document.get("status"),
        "selected_rows": document.get("rows"),
        "output_dim": document.get("output_dim", document.get("classes")),
        "threads": document.get("threads"),
        "layers": document.get("layers"),
        "row_tile": document.get("row_tile"),
        "kernel_speedup": (document.get("speedup")
                           if optimization == "compact_dense_micro"
                           else None),
        "terminal_speedup": terminal.get("speedup"),
        "terminal_paired_speedup": terminal.get("paired_speedup_median"),
        "train_speedup": train_step.get("speedup", document.get("speedup")),
        "train_paired_speedup": train_step.get(
            "paired_speedup_median", document.get("paired_speedup_median")),
        "compute_speedup": nested(
            document, "compute_speedup", "ratio_of_medians"),
        "optimizer_contaminated": document.get("optimizer_contaminated"),
        "loss_abs": correctness.get("loss_abs"),
        "dh_relative_l2": nested(correctness, "dh", "relative_l2"),
        "dw_relative_l2": nested(correctness, "dw", "relative_l2"),
        "db_relative_l2": nested(correctness, "db", "relative_l2"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for directory, optimization in RESULT_GROUPS.items():
        group = args.results / directory
        if not group.is_dir():
            continue
        for path in sorted(group.glob("*.json")):
            rows.append(row_for(
                path.relative_to(args.results), optimization,
                json.loads(path.read_text())))
    if not rows:
        raise SystemExit("no dense-fusion result JSON files found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
