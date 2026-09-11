#!/usr/bin/env python3
"""A/B a strong explicit activation bridge against the one-pass native path."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tfs_train.native import backend


def stats(values: list[float]) -> dict[str, object]:
    return {
        "median_ms": float(statistics.median(values)),
        "mean_ms": float(statistics.mean(values)),
        "samples_ms": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=1_000_000)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(20260911)
    torch.set_num_threads(args.threads)
    shape = (args.nodes, args.features)
    x = torch.randn(shape)
    scale = (torch.rand(args.nodes) + 0.25).contiguous()
    mask = torch.randint(0, 2, shape, dtype=torch.uint8)
    grad = torch.randn(shape)
    dropout_scale = 2.0
    ext = backend()

    def baseline_forward():
        # Strong explicit baseline: reuse the ReLU output and perform dropout
        # multiplications in place before materializing the scaled BF16 input.
        out = torch.relu(x)
        out.mul_(mask)
        out.mul_(dropout_scale)
        staged = (out * scale.unsqueeze(1)).to(torch.bfloat16)
        return out, staged

    def candidate_forward():
        return ext.c3_fused_hidden_bridge_shadow_v1(
            x, scale, mask, dropout_scale, args.threads)

    def baseline_backward():
        # Constructing the combined ReLU/dropout state is part of the explicit
        # baseline.  The native forward writes the equivalent byte state in
        # its single pass, so precomputing it outside timing would be unfair.
        active = (x > 0) & mask.bool()
        return torch.where(active, grad * dropout_scale, 0.0)

    _, _, state = candidate_forward()

    def candidate_backward():
        return ext.c3_fused_hidden_bridge_backward_shadow_v1(
            grad, state, dropout_scale, args.threads)

    x_autograd = x.detach().requires_grad_(True)

    def baseline_complete():
        out = F.dropout(F.relu(x_autograd), p=0.5, training=True)
        staged = (out * scale.unsqueeze(1)).to(torch.bfloat16)
        dx, = torch.autograd.grad(out, x_autograd, grad)
        return staged, dx

    def candidate_complete():
        generated_mask = torch.empty(shape, dtype=torch.uint8)
        generated_mask.bernoulli_(0.5)
        out, staged, generated_state = (
            ext.c3_fused_hidden_bridge_shadow_v1(
                x, scale, generated_mask, dropout_scale, args.threads))
        dx = ext.c3_fused_hidden_bridge_backward_shadow_v1(
            grad, generated_state, dropout_scale, args.threads)
        return out, staged, dx

    old_out, old_hs = baseline_forward()
    new_out, new_hs, new_state = candidate_forward()
    active = (x > 0) & mask.bool()
    torch.testing.assert_close(new_out, old_out, rtol=0, atol=0)
    torch.testing.assert_close(new_hs, old_hs, rtol=0, atol=0)
    torch.testing.assert_close(new_state, active.to(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(candidate_backward(), baseline_backward(),
                               rtol=0, atol=0)
    del old_out, old_hs, new_out, new_hs, new_state

    timings = {name: [] for name in (
        "baseline_forward", "candidate_forward",
        "baseline_backward", "candidate_backward",
        "baseline_complete", "candidate_complete")}
    pairs = (("baseline_forward", baseline_forward),
             ("candidate_forward", candidate_forward),
             ("baseline_backward", baseline_backward),
             ("candidate_backward", candidate_backward),
             ("baseline_complete", baseline_complete),
             ("candidate_complete", candidate_complete))
    for _ in range(args.warmups):
        for _, function in pairs:
            function()
    for repeat in range(args.repeats):
        order = pairs if repeat % 2 == 0 else tuple(reversed(pairs))
        for name, function in order:
            begin = time.perf_counter()
            result = function()
            timings[name].append((time.perf_counter() - begin) * 1.0e3)
            del result

    summary = {name: stats(values) for name, values in timings.items()}
    old_total = (summary["baseline_forward"]["median_ms"] +
                 summary["baseline_backward"]["median_ms"])
    new_total = (summary["candidate_forward"]["median_ms"] +
                 summary["candidate_backward"]["median_ms"])
    payload = {
        "nodes": args.nodes, "features": args.features,
        "threads": args.threads, "dropout_probability": 0.5,
        "mask_generation_timed": False,
        "timings": summary,
        "forward_speedup": (summary["baseline_forward"]["median_ms"] /
                            summary["candidate_forward"]["median_ms"]),
        "backward_speedup": (summary["baseline_backward"]["median_ms"] /
                             summary["candidate_backward"]["median_ms"]),
        "bridge_forward_backward_speedup": old_total / new_total,
        "complete_training_bridge_speedup": (
            summary["baseline_complete"]["median_ms"] /
            summary["candidate_complete"]["median_ms"]),
        "complete_contract": (
            "mask allocation and RNG + ReLU/dropout + source scale/BF16 + "
            "activation backward are all timed"),
        "correctness": "bitwise_equal",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
