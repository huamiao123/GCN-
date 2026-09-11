import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch

from tfs_train.native import backend


def _median_ms(samples):
    return statistics.median(samples) * 1e3


def _run_case(ext, *, panels, rows, k, d, threads, warmups, repeats):
    torch.manual_seed(1000 + panels * 17 + d)
    pulled = [torch.randn(rows, k, dtype=torch.bfloat16)
              for _ in range(panels)]
    weight = torch.randn(k, d, dtype=torch.bfloat16)
    bias = torch.randn(d, dtype=torch.float32)
    scale = [torch.rand(rows, dtype=torch.float32) for _ in range(panels)]

    def legacy():
        result = None
        for index in range(panels):
            result = ext.c3_compact_logits_amx_shadow_v1(
                pulled[index], weight, bias, scale[index], threads)
        return result

    def packed():
        packed_weight = ext.c3_pack_compact_logits_weight_amx_shadow_v1(
            weight)
        result = None
        for index in range(panels):
            result = ext.c3_compact_logits_packed_amx_shadow_v2(
                pulled[index], packed_weight, bias, scale[index], d, threads)
        return result

    reference = legacy()
    candidate = packed()
    torch.testing.assert_close(candidate, reference, rtol=0, atol=0)

    for _ in range(warmups):
        legacy()
        packed()

    legacy_samples = []
    packed_samples = []
    for iteration in range(repeats):
        order = (("legacy", legacy), ("packed", packed))
        if iteration % 2:
            order = tuple(reversed(order))
        for label, function in order:
            start = time.perf_counter()
            function()
            elapsed = time.perf_counter() - start
            (legacy_samples if label == "legacy" else packed_samples).append(
                elapsed)

    legacy_ms = _median_ms(legacy_samples)
    packed_ms = _median_ms(packed_samples)
    return {
        "panels": panels,
        "rows_per_panel": rows,
        "k": k,
        "d": d,
        "threads": threads,
        "legacy_median_ms": legacy_ms,
        "pack_once_median_ms": packed_ms,
        "speedup": legacy_ms / packed_ms,
        "legacy_samples_ms": [value * 1e3 for value in legacy_samples],
        "pack_once_samples_ms": [value * 1e3 for value in packed_samples],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    torch.set_num_threads(1)
    ext = backend()
    rows = []
    for d in (1024, 2983):
        for panels in (1, 2, 8):
            for threads in (1, 8, 32):
                result = _run_case(
                    ext, panels=panels, rows=args.rows, k=128, d=d,
                    threads=threads, warmups=args.warmups,
                    repeats=args.repeats)
                rows.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)

    payload = {
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "cases": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
