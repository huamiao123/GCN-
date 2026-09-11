# Evidence index: original TFS SHW fusion in training

This directory preserves the compact, reviewable evidence used by
`docs/ORIGINAL_TFS_SHW_TRAINING_EXPERIMENTS_20260911.md`.

- `forward_fused_formal_summary.csv`: topology-correct Job 9959570, 12 cells.
- `forward_replay_exclusive_summary.csv`: independent Job 9963827 replay.
- `experiment_chronology.csv`: failed, diagnostic, formal and superseded jobs.
- `raw_forward_replay_v4/*.log`: immutable copied stdout records for all replay variants.
- `SOURCE_MANIFEST.sha256`: SHA-256 identities for all archived source/launcher files.

The older Job 9959570 raw directory was not present in the local publication tree. Its
values are transcribed from the project experiment ledger and cross-checked against the
independent raw Job 9963827 replay. This limitation is disclosed instead of pretending
that a raw artifact is locally available.
