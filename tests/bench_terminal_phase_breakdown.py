"""Phase breakdown for the bounded-panel terminal CE prototype.

This deliberately starts from the already-selected compact P[M,K].  It is
not an end-to-end benchmark; its purpose is to identify which dense terminal
phase must be moved into a native fused implementation.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tfs_train.native import backend


def _ms(start: int) -> float:
    return (time.perf_counter_ns() - start) / 1e6


@torch.inference_mode()
def run_once(p, w, bias, labels, scale, row_tile: int, threads: int,
             logsoftmax_out: bool):
    rows, hidden = p.shape
    wt = w.transpose(0, 1).contiguous()
    q = torch.empty((rows, hidden), dtype=torch.bfloat16)
    dw = torch.zeros((hidden, w.shape[1]), dtype=torch.float32)
    db = torch.zeros(w.shape[1], dtype=torch.float32)
    inv_rows = 1.0 / float(rows)
    phases = {name: 0.0 for name in (
        "logits_gemm", "scale_bias", "ce_gradient", "scale_db",
        "dw", "q", "total")}
    total_start = time.perf_counter_ns()
    for r0 in range(0, rows, row_tile):
        r1 = min(rows, r0 + row_tile)
        pp, yy, ss = p[r0:r1], labels[r0:r1], scale[r0:r1]

        start = time.perf_counter_ns()
        logits = torch.matmul(pp, w).float()
        phases["logits_gemm"] += _ms(start)

        start = time.perf_counter_ns()
        logits.mul_(ss.unsqueeze(1)).add_(bias)
        phases["scale_bias"] += _ms(start)

        start = time.perf_counter_ns()
        if logsoftmax_out:
            torch.log_softmax(logits, dim=1, out=logits)
            logits.exp_().mul_(inv_rows)
        else:
            row_lse = torch.logsumexp(logits, dim=1)
            logits.sub_(row_lse.unsqueeze(1)).exp_().mul_(inv_rows)
        logits[torch.arange(r1 - r0), yy] -= inv_rows
        phases["ce_gradient"] += _ms(start)

        start = time.perf_counter_ns()
        gs, panel_db = backend().c3_scale_grad_bf16_db_v2(
            logits, ss.contiguous(), threads)
        db.add_(panel_db)
        phases["scale_db"] += _ms(start)

        start = time.perf_counter_ns()
        ppt = pp.transpose(0, 1).contiguous()
        dw.add_(torch.matmul(ppt, gs).float())
        phases["dw"] += _ms(start)

        start = time.perf_counter_ns()
        q[r0:r1].copy_(torch.matmul(gs, wt))
        phases["q"] += _ms(start)
    phases["total"] = _ms(total_start)
    phases["accounted"] = sum(phases[name] for name in phases if name not in
                               ("total", "accounted"))
    phases["finite"] = bool(torch.isfinite(dw).all() and
                            torch.isfinite(db).all())
    return phases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=600000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--row-tile", type=int, default=300000)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--logsoftmax-out", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    gen = torch.Generator().manual_seed(20260906 + args.classes)
    p = torch.randn(args.rows, args.hidden, dtype=torch.bfloat16,
                    generator=gen).mul_(0.125)
    w = torch.randn(args.hidden, args.classes, dtype=torch.bfloat16,
                    generator=gen).mul_(0.0625)
    bias = torch.randn(args.classes, dtype=torch.float32,
                       generator=gen).mul_(0.01)
    labels = torch.randint(args.classes, (args.rows,), dtype=torch.int64,
                           generator=gen)
    scale = torch.rand(args.rows, dtype=torch.float32,
                       generator=gen).add_(0.5)
    for _ in range(args.warmups):
        run_once(p, w, bias, labels, scale, args.row_tile, args.threads,
                 args.logsoftmax_out)
    records = [run_once(p, w, bias, labels, scale, args.row_tile,
                        args.threads, args.logsoftmax_out)
               for _ in range(args.repeats)]
    names = ("logits_gemm", "scale_bias", "ce_gradient", "scale_db",
             "dw", "q", "accounted", "total")
    summary = {name: statistics.median(r[name] for r in records)
               for name in names}
    total = summary["total"]
    shares = {name: summary[name] / total for name in names[:-2]}
    payload = {
        "contract": "bounded_terminal_phase_breakdown_shadow_v1",
        "shape": {"rows": args.rows, "hidden": args.hidden,
                  "classes": args.classes},
        "row_tile": args.row_tile,
        "threads": args.threads,
        "logsoftmax_out": args.logsoftmax_out,
        "summary_median_ms": summary,
        "shares": shares,
        "status": "pass" if all(r["finite"] for r in records) else "fail",
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
