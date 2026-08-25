#!/usr/bin/env python3
"""Summarize completed shared-pool TFS scale-up without implying DGL parity."""
import json
from pathlib import Path

ROOT = Path('/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa_extension_tfs_v1_20260822')
OUT = ROOT / 'shared_tfs_scaleup_completed_four_graphs.md'
GRAPHS = ('flickr', 'reddit', 'yelp', 'amazon')
LAYERS, THREADS = (2, 3), (1, 2, 4, 8, 16, 32)

lines = [
    '# Shared-pool TFS scale-up', '',
    '> TFS-only strong-scaling result. Ratio = 1-thread steady median epoch '
    'time / current-thread steady median epoch time. These are not DGL speedups.', '',
    '| Graph / layers | 1 thread | 2 threads | 4 threads | 8 threads | 16 threads | 32 threads |',
    '|---|---:|---:|---:|---:|---:|---:|',
]
for graph in GRAPHS:
    for layer in LAYERS:
        values = []
        for thread in THREADS:
            path = ROOT / f'{graph}_l{layer}_t{thread}_e200' / 'timing_summary.json'
            data = json.loads(path.read_text())
            values.append(float(data['steady_median_epoch_train_plus_eval_ms']))
        base = values[0]
        ratios = [base / value for value in values]
        lines.append('| {} L{} | {} |'.format(graph, layer, ' | '.join(f'{x:.3f}×' for x in ratios)))
lines.extend(['', '## Raw steady median epoch time (ms)', '',
              '| Graph / layers | 1 | 2 | 4 | 8 | 16 | 32 |',
              '|---|---:|---:|---:|---:|---:|---:|'])
for graph in GRAPHS:
    for layer in LAYERS:
        values = []
        for thread in THREADS:
            path = ROOT / f'{graph}_l{layer}_t{thread}_e200' / 'timing_summary.json'
            values.append(float(json.loads(path.read_text())['steady_median_epoch_train_plus_eval_ms']))
        lines.append('| {} L{} | {} |'.format(graph, layer, ' | '.join(f'{x:.3f}' for x in values)))
OUT.write_text('\n'.join(lines) + '\n')
print(OUT)
