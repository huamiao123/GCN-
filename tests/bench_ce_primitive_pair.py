"""Paired gate for the exact in-place CE-gradient primitive."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


@torch.inference_mode()
def manual(base, labels, scale, bias):
    z = base.clone()
    z.mul_(scale.unsqueeze(1)).add_(bias)
    lse = torch.logsumexp(z, dim=1)
    target = z.gather(1, labels.unsqueeze(1)).squeeze(1)
    loss_sum = (lse - target).double().sum()
    z.sub_(lse.unsqueeze(1)).exp_()
    z[torch.arange(z.shape[0]), labels] -= 1.0
    return loss_sum, z


@torch.inference_mode()
def logsoftmax_out(base, labels, scale, bias):
    z = base.clone()
    z.mul_(scale.unsqueeze(1)).add_(bias)
    torch.log_softmax(z, dim=1, out=z)
    loss_sum = -z.gather(1, labels.unsqueeze(1)).double().sum()
    z.exp_()
    z[torch.arange(z.shape[0]), labels] -= 1.0
    return loss_sum, z


def timed(fn):
    start = time.perf_counter_ns()
    out = fn()
    return (time.perf_counter_ns() - start) / 1e6, out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=300000)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    gen = torch.Generator().manual_seed(20260906 + args.classes)
    base = torch.randn(args.rows, args.classes, generator=gen)
    labels = torch.randint(args.classes, (args.rows,), generator=gen)
    scale = torch.rand(args.rows, generator=gen).add_(0.5)
    bias = torch.randn(args.classes, generator=gen).mul_(0.01)
    ref = manual(base, labels, scale, bias)
    alt = logsoftmax_out(base, labels, scale, bias)
    loss_sum_abs = float((ref[0] - alt[0]).abs())
    loss_mean_abs = loss_sum_abs / float(args.rows)
    grad_rel = float(torch.linalg.vector_norm(ref[1] - alt[1]) /
                     torch.linalg.vector_norm(ref[1]))
    for _ in range(args.warmups):
        for fn in (manual, logsoftmax_out):
            out = fn(base, labels, scale, bias)
            del out
    records = {"manual": [], "logsoftmax_out": []}
    for iteration in range(args.repeats):
        order = (("manual", manual), ("logsoftmax_out", logsoftmax_out))
        if iteration % 2:
            order = tuple(reversed(order))
        for name, fn in order:
            elapsed, out = timed(lambda: fn(base, labels, scale, bias))
            records[name].append(elapsed)
            del out
    med = {name: statistics.median(values)
           for name, values in records.items()}
    payload = {
        "contract": "ce_primitive_pair_shadow_v1",
        "shape": {"rows": args.rows, "classes": args.classes},
        "threads": args.threads,
        "correctness": {"loss_sum_abs": loss_sum_abs,
                        "loss_mean_abs": loss_mean_abs,
                        "grad_relative_l2": grad_rel},
        "median_ms": med,
        "speedup": med["manual"] / med["logsoftmax_out"],
        "status": "pass" if loss_mean_abs <= 1e-5 and grad_rel <= 1e-5 else "fail",
        "records_ms": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
