#!/usr/bin/env python3
"""Compare completed matching shared-pool GraphSAINT TFS and stock-DGL cells."""
import json
from pathlib import Path

TFS = Path('/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa_extension_tfs_v1_20260822')
DGL = Path('/home/huangjianqiang_group/hdacp1/data/wzh/final_pre_numa_extension_dgl_v1_20260823')
OUT = DGL / 'shared_tfs_vs_dgl_completed.md'
graphs, layers, threads = ('flickr', 'reddit', 'yelp', 'amazon'), (2, 3), (1, 2, 4, 8, 16, 32)

lines = ['# Shared-pool TFS vs stock DGL', '',
         '> Ratio = DGL steady median epoch time / TFS steady median epoch time. '
         'Each method has the same graph, layer, threads, seed and epoch protocol, but '
         'they are separate shared-pool jobs; these are shared-node reference results.', '',
         '| Graph / layers | 1 | 2 | 4 | 8 | 16 | 32 |',
         '|---|---:|---:|---:|---:|---:|---:|']
complete = 0
for graph in graphs:
    for layer in layers:
        ratios = []
        for thread in threads:
            name = f'{graph}_l{layer}_t{thread}_e200'
            tfs_path, dgl_path = TFS / name / 'timing_summary.json', DGL / name / 'timing_summary.json'
            if tfs_path.is_file() and dgl_path.is_file():
                tfs = json.loads(tfs_path.read_text())['steady_median_epoch_train_plus_eval_ms']
                dgl = json.loads(dgl_path.read_text())['steady_median_epoch_train_plus_eval_ms']
                ratios.append(f'{float(dgl) / float(tfs):.3f}×')
                complete += 1
            else:
                ratios.append('pending')
        lines.append(f'| {graph} L{layer} | ' + ' | '.join(ratios) + ' |')
lines += ['', f'Comparable completed cells: {complete}/48.']
OUT.write_text('\n'.join(lines) + '\n')
print(OUT)
