#!/usr/bin/env python3
"""Produce a clearly-labelled, non-paper comparison to archived DGL cells."""
import csv
import json
import math
from pathlib import Path


NEW = Path('/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa_paper_v1_20260822/tfs_final_pre_numa')
OLD = Path('/home/huangjianqiang_group/hdacp1/data/wzh/authority_v23_triplet_20260818/dgl_stock')
OUT = NEW.parent / 'historical_dgl_reference_comparison.md'
graphs = ('products', 'arxiv', 'igb19', 'igb2983')
layers = (2, 3)
threads = (1, 2, 4, 8, 16, 32)


def payload(path):
    return json.loads(path.read_text())


def final_test(path):
    rows = list(csv.DictReader(path.open(newline='')))
    if not rows:
        return None
    value = rows[-1].get('test_accuracy') or rows[-1].get('test_score')
    return float(value) if value not in (None, '') else None


rows = []
for graph in graphs:
    for layer in layers:
        for thread in threads:
            name = f'{graph}_l{layer}_t{thread}_e200'
            new, old = NEW / name, OLD / name
            if not ((new / 'summary.json').is_file() and (old / 'summary.json').is_file()):
                continue
            tfs, dgl = payload(new / 'summary.json'), payload(old / 'summary.json')
            tfs_ms = float(tfs['steady_median_epoch_train_plus_eval_ms'])
            dgl_ms = float(dgl['mean_epoch_train_plus_eval_ms'])
            rows.append((graph, layer, thread, dgl_ms / tfs_ms, tfs_ms, dgl_ms,
                         final_test(new / 'training_detailed.csv'),
                         final_test(old / 'training_detailed.csv')))

lines = [
    '# Historical DGL reference comparison',
    '',
    '> This is a diagnostic-only historical comparison, not a paper result. '
    'The archived DGL cells used an earlier run/environment and report mean epoch '
    'time including epoch 1; new TFS uses paper_v1 steady median over epochs 2–200. '
    'Do not describe these ratios as fair TFS-vs-DGL speedups.',
    '',
    f'Comparable completed cells: {len(rows)}/48.',
    '',
    '| Graph | Layers | cells | Historical DGL / new TFS ratio (geomean) | TFS test − DGL test (mean pp) |',
    '|---|---:|---:|---:|---:|',
]
for graph in graphs:
    for layer in layers:
        subset = [row for row in rows if row[0] == graph and row[1] == layer]
        if not subset:
            continue
        ratio = math.exp(sum(math.log(row[3]) for row in subset) / len(subset))
        deltas = [(row[6] - row[7]) * 100 for row in subset
                  if row[6] is not None and row[7] is not None]
        delta = f'{sum(deltas) / len(deltas):+.3f}' if deltas else 'n/a'
        lines.append(f'| {graph} | {layer} | {len(subset)}/6 | {ratio:.3f}× | {delta} |')

lines += ['', '## Per-cell timing reference', '',
          '| Graph | L | threads | historical DGL mean epoch (ms) | new TFS steady median (ms) | ratio |',
          '|---|---:|---:|---:|---:|---:|']
for graph, layer, thread, ratio, tfs_ms, dgl_ms, _, _ in rows:
    lines.append(f'| {graph} | {layer} | {thread} | {dgl_ms:.3f} | {tfs_ms:.3f} | {ratio:.3f}× |')
OUT.write_text('\n'.join(lines) + '\n')
print(OUT)
