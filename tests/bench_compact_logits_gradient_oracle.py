"""Compare compact AMX logits and framework BF16 logits to an FP32 oracle.

The downstream derivatives are deliberately evaluated in FP32 in all arms.
This isolates the numerical effect of the logits producer from the separately
validated BF16 Q/dW kernels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tfs_train.native import backend


def metric(actual, reference):
    delta = actual.double() - reference.double()
    return {
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(
            torch.linalg.vector_norm(delta) /
            torch.linalg.vector_norm(reference.double()).clamp_min(1e-30)),
    }


def derivatives(logits, pulled, weight, scale, labels):
    rows = logits.shape[0]
    row_index = torch.arange(rows, dtype=torch.int64)
    log_probabilities = torch.log_softmax(logits, dim=1)
    probabilities = torch.softmax(logits, dim=1)
    loss = -log_probabilities[row_index, labels].double().mean()
    grad_logits = probabilities
    grad_logits[row_index, labels] -= 1.0
    grad_logits.div_(float(rows))
    grad_scaled = grad_logits * scale.unsqueeze(1)
    return {
        "loss": loss,
        "dhidden": grad_scaled @ weight.float().transpose(0, 1),
        "dweight": pulled.float().transpose(0, 1) @ grad_scaled,
        "dbias": grad_logits.sum(dim=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--classes", type=int, default=2983)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260907)
    pulled = torch.randn(args.rows, 128, dtype=torch.bfloat16)
    weight = torch.randn(128, args.classes, dtype=torch.bfloat16)
    bias = torch.randn(args.classes, dtype=torch.float32)
    scale = torch.rand(args.rows, dtype=torch.float32)
    labels = torch.randint(args.classes, (args.rows,), dtype=torch.int64)

    fp32_logits = pulled.float() @ weight.float()
    fp32_logits.mul_(scale.unsqueeze(1)).add_(bias)
    framework_logits = (pulled @ weight).float()
    framework_logits.mul_(scale.unsqueeze(1)).add_(bias)
    native_logits = backend().c3_compact_logits_amx_shadow_v1(
        pulled, weight, bias, scale, args.threads)

    oracle = derivatives(fp32_logits, pulled, weight, scale, labels)
    arms = {
        "framework_bf16_output": derivatives(
            framework_logits, pulled, weight, scale, labels),
        "native_fp32_accumulator": derivatives(
            native_logits, pulled, weight, scale, labels),
    }
    result = {
        "contract": "compact_logits_gradient_fp32_oracle_v1",
        "rows": args.rows,
        "classes": args.classes,
        "threads": args.threads,
        "logits": {
            name: metric(value, fp32_logits)
            for name, value in {
                "framework_bf16_output": framework_logits,
                "native_fp32_accumulator": native_logits,
            }.items()
        },
        "derivatives": {
            name: {
                "loss_abs": float((values["loss"] - oracle["loss"]).abs()),
                "dhidden": metric(values["dhidden"], oracle["dhidden"]),
                "dweight": metric(values["dweight"], oracle["dweight"]),
                "dbias": metric(values["dbias"], oracle["dbias"]),
            }
            for name, values in arms.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
