#!/usr/bin/env python3
"""Controlled Y-first/Q-first backward dataflow experiment.

This is an experiment-only harness.  It uses the same native pull primitive
for both arms, a symmetric regular CSR, identical BF16 conversion points, and
identical logical threads.  It does not alter the authority planner.
"""
import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import torch

from tfs_train.native import backend


def regular_symmetric_csr(n: int, degree: int):
    if degree < 2 or degree % 2:
        raise ValueError("degree must be positive, even, and at least two")
    rows = torch.arange(n, dtype=torch.long).repeat_interleave(degree)
    offsets = torch.cat((torch.arange(-degree // 2, 0),
                         torch.arange(1, degree // 2 + 1))).to(torch.long)
    colidx = (torch.arange(n, dtype=torch.long).unsqueeze(1) + offsets).reshape(-1) % n
    rowptr = torch.arange(n + 1, dtype=torch.long) * degree
    # ``rows`` deliberately establishes the direct CSR construction invariant.
    assert rows.numel() == colidx.numel() == int(rowptr[-1])
    return rowptr.contiguous(), colidx.contiguous()


def pull(value, rowptr, colidx, threads):
    return backend().c3_pull_only_amx_v1(
        value.float().contiguous(), rowptr, colidx, int(threads))


def q_first(hs, pulled, weight, grad, scale, rowptr, colidx, threads):
    gs = (grad * scale.unsqueeze(1)).to(torch.bfloat16)
    q = torch.matmul(gs, weight.to(torch.bfloat16).transpose(0, 1)).float()
    dx = pull(q, rowptr, colidx, threads).mul_(scale.unsqueeze(1))
    dw = torch.matmul(pulled.float().transpose(0, 1), gs.float())
    return dx, dw


def y_first(hs, weight, grad, scale, rowptr, colidx, threads):
    gs = (grad * scale.unsqueeze(1)).to(torch.bfloat16)
    ybar = pull(gs, rowptr, colidx, threads)
    dx = torch.matmul(ybar.to(torch.bfloat16),
                      weight.to(torch.bfloat16).transpose(0, 1)).float()
    dx.mul_(scale.unsqueeze(1))
    dw = torch.matmul(hs.float().transpose(0, 1), ybar)
    return dx, dw


def timed(call, warmups, repeats):
    for _ in range(warmups):
        call()
    values = []
    for _ in range(repeats):
        start = time.perf_counter_ns(); call()
        values.append((time.perf_counter_ns() - start) / 1e6)
    return statistics.median(values), values


def relative_l2(actual, reference):
    return float((actual - reference).norm() / reference.norm().clamp_min(1e-12))


parser = argparse.ArgumentParser()
parser.add_argument('--nodes', type=int, default=32768)
parser.add_argument('--degrees', default='2,8,16,32')
parser.add_argument('--dims', default='128,160,256,512,1024,2983')
parser.add_argument('--k', type=int, default=128)
parser.add_argument('--threads', type=int, default=32)
parser.add_argument('--warmups', type=int, default=2)
parser.add_argument('--repeats', type=int, default=5)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.manual_seed(20260823); torch.set_num_threads(args.threads)
rowptr_cache = {}
args.output.parent.mkdir(parents=True, exist_ok=True)
fields = ('nodes', 'degree', 'k', 'd', 'threads', 'y_first_ms', 'q_first_ms',
          'y_over_q', 'dx_relative_l2', 'dw_relative_l2', 'y_samples_ms', 'q_samples_ms')
with args.output.open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
    for degree in map(int, args.degrees.split(',')):
        rowptr, colidx = rowptr_cache.setdefault(degree, regular_symmetric_csr(args.nodes, degree))
        scale = torch.full((args.nodes,), 1.0 / (degree ** .5), dtype=torch.float32)
        x = torch.randn((args.nodes, args.k), dtype=torch.float32)
        hs = (x * scale.unsqueeze(1)).to(torch.bfloat16)
        pulled = pull(hs, rowptr, colidx, args.threads).to(torch.bfloat16)
        for d in map(int, args.dims.split(',')):
            weight = torch.randn((args.k, d), dtype=torch.float32)
            grad = torch.randn((args.nodes, d), dtype=torch.float32)
            q_dx, q_dw = q_first(hs, pulled, weight, grad, scale, rowptr, colidx, args.threads)
            y_dx, y_dw = y_first(hs, weight, grad, scale, rowptr, colidx, args.threads)
            q_ms, q_samples = timed(lambda: q_first(hs, pulled, weight, grad, scale, rowptr, colidx, args.threads), args.warmups, args.repeats)
            y_ms, y_samples = timed(lambda: y_first(hs, weight, grad, scale, rowptr, colidx, args.threads), args.warmups, args.repeats)
            writer.writerow({'nodes': args.nodes, 'degree': degree, 'k': args.k, 'd': d, 'threads': args.threads,
                             'y_first_ms': y_ms, 'q_first_ms': q_ms, 'y_over_q': y_ms / q_ms,
                             'dx_relative_l2': relative_l2(y_dx, q_dx), 'dw_relative_l2': relative_l2(y_dw, q_dw),
                             'y_samples_ms': json.dumps(y_samples), 'q_samples_ms': json.dumps(q_samples)})
            stream.flush()
            print(f'YQ degree={degree} D={d} Y/Q={y_ms / q_ms:.3f}', flush=True)
