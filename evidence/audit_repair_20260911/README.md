# 2026-09-11 审计修复与数据流实验凭证

本目录保存与当前分支源码对应的机器可读结果。它们用于复核局部 kernel、数据流
边界和负结果，不替代论文正式 48-cell 端到端训练矩阵。

## 目录说明

| 目录 | 内容 | 当前结论 |
|---|---|---|
| `aggregate_one_pull/` | Aggregate High-D 多 D-slab 共用 FP32 dP、最终仅一次 pull | 局部收益明确；待 convergence/E2E |
| `highd_strided_slab/` | D-slab strided native adapter 对比强制 contiguous | Aggregate 路径收益明确 |
| `wide_k_static/` | 四个真实图的静态首层 SX 数据流复用 | 局部收益明确；待 convergence/E2E |
| `compact_q/` | Terminal Native-Q 子算子、维度边界及完整 terminal gate | 子算子快，完整边界 0.998x，不接入 |
| `interlayer_bridge/` | ReLU/Dropout/BF16 bridge 完整训练边界 | 0.784x，不接入 |
| `gs_panel/` | destination/source/no-global Gs/Q panel 系列 | bounded 路径无稳定收益，不接入 |

## 解读约束

- `speedup > 1` 表示候选路径快于同一结果文件中的对照路径。
- 局部 backward、首层 forward+backward、terminal 边界不能写成完整训练 E2E。
- shadow-only ABI 和 benchmark 路径不等于默认 planner 已启用。
- 正确性误差、线程数、预热/重复次数和是否计入预处理，以各 JSON 字段及
  `docs/AUDIT_REPAIR_STATUS_20260910.md` 为准。
