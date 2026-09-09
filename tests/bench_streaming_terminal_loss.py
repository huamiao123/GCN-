"""Gate for a materialization-free terminal classifier/loss dataflow.

This is deliberately a standalone dense microbenchmark.  It consumes the
compact terminal aggregate P[M,K] that supervision-scoped TFS already
produces and compares two exact algebraic paths:

  materialized: Z=P@W -> CE -> G -> {Q=G@W.T, dW=P.T@G, db=sum(G)}
  streaming:     class/row tiled two-pass CE, immediately consuming each G
                 tile into Q/dW/db without retaining Z[M,D] or G[M,D].

The streaming implementation uses PyTorch BF16 GEMMs as a conservative
prototype; it is not presented as the final fused native AMX kernel.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import resource
import statistics
import time
from pathlib import Path

import torch


def _rss_mb() -> float:
    # Linux reports KiB, macOS bytes.  Formal runs are Linux-only.
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0 if os.uname().sysname == "Linux" else value / 2**20


def _make_inputs(rows: int, hidden: int, classes: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    p = torch.randn(rows, hidden, dtype=torch.bfloat16, generator=generator)
    p.mul_(0.125)
    w = torch.randn(hidden, classes, dtype=torch.bfloat16,
                    generator=generator)
    w.mul_(0.0625)
    bias = torch.randn(classes, dtype=torch.float32, generator=generator)
    bias.mul_(0.01)
    labels = torch.randint(classes, (rows,), dtype=torch.int64,
                           generator=generator)
    return p.contiguous(), w.contiguous(), bias.contiguous(), labels


@torch.inference_mode()
def materialized(p, w, bias, labels):
    """Strong baseline that reuses its single MxD buffer as the gradient."""
    rows = p.shape[0]
    z = torch.matmul(p, w).float()
    z.add_(bias)
    row_lse = torch.logsumexp(z, dim=1)
    target = z.gather(1, labels.unsqueeze(1)).squeeze(1)
    loss = (row_lse - target).mean()

    # Turn Z into the exact FP32 softmax/CE gradient in-place.  This is more
    # memory-efficient than keeping both logits and grad and therefore avoids
    # weakening the baseline.
    z.sub_(row_lse.unsqueeze(1)).exp_().div_(float(rows))
    z[torch.arange(rows), labels] -= 1.0 / float(rows)
    db = z.sum(dim=0)
    gs = z.to(torch.bfloat16)
    del z
    q = torch.matmul(gs, w.transpose(0, 1)).float()
    dw = torch.matmul(p.transpose(0, 1).contiguous(), gs).float()
    return loss, q, dw, db


def current_autograd(p, w, bias, labels):
    """Emulate the current PyTorch CE boundary and retain its real gradient."""
    with torch.enable_grad():
        z = torch.matmul(p, w).float()
        z.add_(bias)
        z.requires_grad_(True)
        loss = torch.nn.functional.cross_entropy(z, labels)
        (g,) = torch.autograd.grad(loss, z)
    db = g.sum(dim=0)
    gs = g.to(torch.bfloat16)
    q = torch.matmul(gs, w.transpose(0, 1)).float()
    dw = torch.matmul(p.transpose(0, 1).contiguous(), gs).float()
    return loss.detach(), q, dw, db


@torch.inference_mode()
def streaming(p, w, bias, labels, row_tile: int, class_tile: int):
    """Two-pass exact CE with no MxD logits or gradient materialization."""
    rows, hidden = p.shape
    classes = w.shape[1]
    q = torch.empty((rows, hidden), dtype=torch.float32)
    dw = torch.zeros((hidden, classes), dtype=torch.float32)
    db = torch.zeros(classes, dtype=torch.float32)
    total_loss = torch.zeros((), dtype=torch.float64)
    inv_rows = 1.0 / float(rows)

    for r0 in range(0, rows, row_tile):
        r1 = min(rows, r0 + row_tile)
        pp = p[r0:r1]
        yy = labels[r0:r1]
        count = r1 - r0
        running_max = torch.full((count,), -math.inf, dtype=torch.float32)
        running_sum = torch.zeros(count, dtype=torch.float32)
        target = torch.empty(count, dtype=torch.float32)

        # Pass 1: online log-sum-exp and target-logit extraction.
        for c0 in range(0, classes, class_tile):
            c1 = min(classes, c0 + class_tile)
            z = torch.matmul(pp, w[:, c0:c1].contiguous()).float()
            z.add_(bias[c0:c1])
            tile_max = z.amax(dim=1)
            new_max = torch.maximum(running_max, tile_max)
            running_sum.mul_(torch.exp(running_max - new_max))
            running_sum.add_(torch.exp(z - new_max.unsqueeze(1)).sum(dim=1))
            running_max = new_max
            hit = (yy >= c0) & (yy < c1)
            if bool(hit.any()):
                local = yy[hit] - c0
                target[hit] = z[hit, local]

        row_lse = running_max + running_sum.log()
        total_loss.add_((row_lse - target).double().sum())
        q_panel = torch.zeros((count, hidden), dtype=torch.float32)

        # Pass 2: recompute one class tile and consume its gradient at once.
        for c0 in range(0, classes, class_tile):
            c1 = min(classes, c0 + class_tile)
            wt = w[:, c0:c1].contiguous()
            z = torch.matmul(pp, wt).float()
            z.add_(bias[c0:c1])
            z.sub_(row_lse.unsqueeze(1)).exp_().mul_(inv_rows)
            hit = (yy >= c0) & (yy < c1)
            if bool(hit.any()):
                local = yy[hit] - c0
                z[hit, local] -= inv_rows
            db[c0:c1].add_(z.sum(dim=0))
            gs = z.to(torch.bfloat16)
            dw[:, c0:c1].add_(
                torch.matmul(pp.transpose(0, 1).contiguous(), gs).float())
            q_panel.add_(torch.matmul(gs, wt.transpose(0, 1)).float())
        q[r0:r1].copy_(q_panel)

    return total_loss.float().div_(float(rows)), q, dw, db


@torch.inference_mode()
def panelized(p, w, bias, labels, row_tile: int):
    """Bound logits/gradient lifetime to one row panel without recomputation.

    This variant preserves the strong baseline's arithmetic count.  It trades
    a bounded row_tile x D temporary for the streaming path's second P@W.
    """
    rows, hidden = p.shape
    classes = w.shape[1]
    q = torch.empty((rows, hidden), dtype=torch.float32)
    dw = torch.zeros((hidden, classes), dtype=torch.float32)
    db = torch.zeros(classes, dtype=torch.float32)
    total_loss = torch.zeros((), dtype=torch.float64)
    inv_rows = 1.0 / float(rows)
    w_t = w.transpose(0, 1).contiguous()

    for r0 in range(0, rows, row_tile):
        r1 = min(rows, r0 + row_tile)
        pp = p[r0:r1]
        yy = labels[r0:r1]
        z = torch.matmul(pp, w).float()
        z.add_(bias)
        row_lse = torch.logsumexp(z, dim=1)
        target = z.gather(1, yy.unsqueeze(1)).squeeze(1)
        total_loss.add_((row_lse - target).double().sum())
        z.sub_(row_lse.unsqueeze(1)).exp_().mul_(inv_rows)
        z[torch.arange(r1 - r0), yy] -= inv_rows
        db.add_(z.sum(dim=0))
        gs = z.to(torch.bfloat16)
        del z
        q[r0:r1].copy_(torch.matmul(gs, w_t).float())
        dw.add_(torch.matmul(pp.transpose(0, 1).contiguous(), gs).float())
    return total_loss.float().div_(float(rows)), q, dw, db


def _metrics(reference, candidate):
    result = {}
    for name, lhs, rhs in zip(("loss", "q", "dw", "db"), reference,
                              candidate):
        lhs = lhs.float()
        rhs = rhs.float()
        diff = lhs - rhs
        denom = max(float(torch.linalg.vector_norm(lhs)), 1e-30)
        result[name] = {
            "max_abs": float(diff.abs().max()),
            "relative_l2": float(torch.linalg.vector_norm(diff)) / denom,
        }
    return result


def _timed(fn, warmups: int, repeats: int):
    for _ in range(warmups):
        outputs = fn()
        del outputs
        gc.collect()
    records = []
    for _ in range(repeats):
        gc.collect()
        before = _rss_mb()
        start = time.perf_counter_ns()
        outputs = fn()
        elapsed = (time.perf_counter_ns() - start) / 1e6
        after = _rss_mb()
        records.append({"elapsed_ms": elapsed, "rss_before_mb": before,
                        "rss_hwm_mb": after})
        del outputs
    return records


def _summary(records):
    values = [item["elapsed_ms"] for item in records]
    return {
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "rss_hwm_mb": max(item["rss_hwm_mb"] for item in records),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("check", "current", "materialized",
                                           "streaming", "panelized"),
                        required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--row-tile", type=int, default=32768)
    parser.add_argument("--class-tile", type=int, default=256)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    p, w, bias, labels = _make_inputs(
        args.rows, args.hidden, args.classes, args.seed)
    base_rss = _rss_mb()
    payload = {
        "contract": "terminal_loss_streaming_shadow_v1",
        "mode": args.mode,
        "shape": {"rows": args.rows, "hidden": args.hidden,
                  "classes": args.classes},
        "tiles": {"rows": args.row_tile, "classes": args.class_tile},
        "threads": args.threads,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "input_rss_mb": base_rss,
        "logical_wide_tensor_gib": args.rows * args.classes * 4 / 2**30,
    }

    if args.mode == "check":
        reference = materialized(p, w, bias, labels)
        candidate = streaming(p, w, bias, labels, args.row_tile,
                              args.class_tile)
        bounded = panelized(p, w, bias, labels, args.row_tile)
        payload["correctness"] = {
            "streaming": _metrics(reference, candidate),
            "panelized": _metrics(reference, bounded),
        }
        stream_metrics = payload["correctness"]["streaming"]
        panel_metrics = payload["correctness"]["panelized"]
        payload["pass"] = (
            all(metrics["loss"]["relative_l2"] < 2e-5 and
                metrics["q"]["relative_l2"] < 3e-2 and
                metrics["dw"]["relative_l2"] < 3e-2 and
                metrics["db"]["relative_l2"] < 3e-2
                for metrics in (stream_metrics, panel_metrics)))
    else:
        if args.mode == "current":
            fn = lambda: current_autograd(p, w, bias, labels)
        elif args.mode == "materialized":
            fn = lambda: materialized(p, w, bias, labels)
        elif args.mode == "streaming":
            fn = lambda: streaming(p, w, bias, labels, args.row_tile,
                                   args.class_tile)
        else:
            fn = lambda: panelized(p, w, bias, labels, args.row_tile)
        records = _timed(fn, args.warmups, args.repeats)
        payload["summary"] = _summary(records)
        payload["records"] = records

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.mode == "check" and not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
