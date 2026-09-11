# 原始 TFS `SHW` 融合向 GCN 训练迁移：完整实验设置、过程、结果与结论

更新日期：2026-09-11

结论状态：`FORMAL NEGATIVE FOR DEFAULT FORWARD`
适用范围：官方 `ogbn-arxiv`、完整对称归一化 GCN 前向、`128→40` 与 `128→128`、Intel Xeon Max 9462、单 Socket 1–32 物理线程。

## 1. 摘要

项目完整实现和测试了原始 TFS 推理工作的主要机制：`C=A·H·W`/训练语义下的 `Y=S·H·W+b` 寄存器级融合、16-row AMX tile batch、ascending degree sort、full-K neighbor gather、BF16 输入、VNNI-packed BF16 权重、`TDPBF16PS` FP32 累加、software prefetch、tile zero、output scatter、D40 尾部、persistent workspace、row-group 扫描、1–32 线程和单 Socket NUMA placement。

结论：融合实现数值正确（典型相对 L2 约 `5e-7`），但正式 12 个 shape/thread cell 中 `0/12` 获胜。`128→40` 应使用 transform-first，`128→128` 应使用 aggregate-first。persistent workspace、尾部修复、R sweep、额外 accumulator 和 replay 均未改变结论。失败是 per-edge dense transform amplification 的结构问题，不是一次性分配或简单线程 Bug。

## 2. 原始 TFS 方法和源码事实

原始推理路径计算：

```text
C = A · H · W
```

传统路径物化 `Z=A·H`，再执行 `C=Z·W`。TFS 将每个邻居的 feature row 搬入 16-row BF16 buffer，直接与预打包 W tile 执行 AMX 乘加，输出留在 C tile 中，避免全局 `Z[N,K]`。

归档源码：[`amx_tfs_v3k_original.cpp`](../experiments/original_tfs_shw_training/amx_tfs_v3k_original.cpp)

源码循环结构：

```text
row group
  └─ 16-row tile batch
       └─ output pass
            └─ neighbor position
                 ├─ gather 16 个 source feature rows 的完整 K
                 └─ K/32 个 AMX reduction blocks
```

`K=D=128` 时输出需要 8 个 16-column C tiles，但同时只有 4 个 C accumulator，因此有两个 output passes。output pass 位于 neighbor traversal 外层，CSR、indices、H gather 会执行两遍。full-K gather 只消除了同一 pass 内不同 K-block 的重复读取。

### Degree sort 的准确作用

目标节点按度数升序形成逻辑 permutation，使同一 16-row lockstep batch 内的度数相近。它降低 padding、空 AMX lane，并支持 `active_from` 快速失活。

它不能消除：per-edge dense transform、D128 两次 output pass、跨 batch 的重复 source 读取、全局线程 NNZ 不均衡和 source NUMA locality。

原始 `active_from` 依赖 batch 内 degree 单调。旧 `no_degsort` ablation 只换成 identity permutation、仍使用 `active_from`，曾在 7 张图中产生 5 张 correctness failure，不能作为纯净的关闭排序性能对照。

## 3. 训练版融合语义与比较路径

训练适配语义：

```text
S = diag(scale) · (A_off + I) · diag(scale)
Y = S · H · W + bias
```

实现先计算 `Hs=diag(scale)·H`，在 `(A_off+I)` 上融合，再在 store 时执行 destination scale 和 bias。源码：[`normalized_fused_forward_bench.cpp`](../experiments/original_tfs_shw_training/normalized_fused_forward_bench.cpp)。

比较三条路径：

```text
Aggregate-first: T=(A_off+I)·Hs; Y=diag(scale)·(T·W)+bias
Transform-first: U=Hs·W; Y=diag(scale)·((A_off+I)·U)+bias
Fused SHW:       neighbor gather 与 Hs[source,:]·W 在 AMX 中逐边融合
```

三者使用同一图、H、W、bias 和线程数；最终与 aggregate/transform 中实际更快的一条比较。

## 4. 正式实验设置

### 数据与图合同

| 项目 | 设置 |
|---|---|
| 数据集 | official ogbn-arxiv RELEASE_v1 |
| 节点数 | 169,343 |
| canonical off-diagonal NNZ | 2,315,598 |
| self-loop | 每节点显式加入一条 |
| 训练 CSR 邻接条目 | 2,484,941 |
| 平均训练度数 | 约 14.674 |
| 输入 K | 128 |
| 输出 D | 40、128 |
| 归一化 | symmetric normalization |
| 输出顺序 | scatter 回原节点顺序 |

### 硬件、线程和统计

权威 Job 9959570：Xeon Max 9462，`intel_expr --exclusive`，qhcn816，只使用 Socket 0 物理核；1/2/4/8T 位于 NUMA0，16T 在 NUMA0–1 interleave，32T 在 NUMA0–3 interleave，不跨 Socket。每点 warmup 2 次、timed repeat 7 次并报告中位数。

正式运行前扫描 `R=16/32/64/128/256/512`，固定使用 D40 的最佳 `R=32` 和 D128 的最佳 `R=16`，没有用统一参数压低 fused。

### 计时边界

每条路径 total 包括 Hs BF16 staging、各自 W packing、sparse traversal、dense GEMM 或 fused AMX、scale/bias epilogue 和输出 store。dataset loading、CSR canonicalization、一次性 degree permutation、persistent workspace 首次构造和编译不计入稳态。

早期版本在 invocation 内分配大 workspace；Job 9959554 持久化后重新测量，fused 仍明显落后，因此正式结论不依赖该缺陷。

## 5. 开发、失败和修正过程

机器可读完整列表：[`experiment_chronology.csv`](../evidence/original_tfs_shw_training/experiment_chronology.csv)

| Job | 目的/问题 | 结果与处理 |
|---|---|---|
| 9959539 | 首次 exact-normalized fused build | tile intrinsic 使用动态 tile ID，编译失败；改成常量展开，无性能数据 |
| 9959543 | 修复后重跑 | `val_mask/valid_mask` fixture 字段错误，无 kernel 数据 |
| 9959545 | 第一轮成功 smoke | 数值约 `5e-7`；D40 fused/strongest `0.827×`，D128 `0.744×`，但有 timed allocation |
| 9959552 | D40 8-column AMX tail | illegal instruction，整个变体废弃 |
| 9959554 | persistent workspace + 安全 D40 tail | fused/strongest D40 `0.624×`、D128 `0.513×`；排除 allocation 主因 |
| 9959556 | R sweep | 最佳 D40 R32 仍 `0.654×`；D128 R16 仍 `0.604×` |
| 9959563 | 首个 formal | 16/32T 错绑在仅 8 核 NUMA0，两个 cell 废弃 |
| 9959570 | 拓扑正确 formal | 12-cell 全部有效，fused `0/12` 获胜 |
| 9963827 | 独立 D128 replay | legacy fused 1/8/32T 慢 `1.553×/1.988×/3.765×`；其他 replay 更慢 |

Job 9959545 的 8T 单次结果：

| Shape | Aggregate | Transform | Fused | Fused/strongest |
|---|---:|---:|---:|---:|
| 128→40 | 238.45 | 150.39 | 181.85 | 0.827× |
| 128→128 | 253.57 | 285.45 | 340.89 | 0.744× |

Job 9959554 持久化后：

| Shape | Aggregate | Transform | Fused | Fused/strongest |
|---|---:|---:|---:|---:|
| 128→40 | 145.55 | 78.52 | 125.77 | 0.624× |
| 128→128 | 179.74 | 158.30 | 308.38 | 0.513× |

此时 fused prep 已约 7.5–8.0 ms，但 kernel 本体仍为 D40 `118.30 ms`、D128 `300.34 ms`。

R sweep 的 fused 时间：

```text
D40  R16/32/64/128/256/512:
137.25, 119.45, 141.33, 164.90, 157.26, 160.32 ms

D128 R16/32/64/128/256/512:
262.84, 302.60, 315.82, 350.53, 367.14, 339.59 ms
```

## 6. 权威 12-cell 结果

CSV：[`forward_fused_formal_summary.csv`](../evidence/original_tfs_shw_training/forward_fused_formal_summary.csv)

### `128→40`：最强为 transform-first

| T | Aggregate ms | Transform ms | Fused ms | Fused/strongest | Fused 慢 |
|---:|---:|---:|---:|---:|---:|
| 1 | 104.12 | 59.96 | 96.35 | 0.622× | 1.607× |
| 2 | 52.69 | 30.36 | 50.64 | 0.600× | 1.668× |
| 4 | 26.34 | 15.39 | 28.01 | 0.549× | 1.820× |
| 8 | 13.27 | 7.79 | 16.17 | 0.482× | 2.076× |
| 16 | 5.67 | 3.72 | 9.75 | 0.381× | 2.621× |
| 32 | 2.98 | 2.48 | 7.24 | 0.342× | 2.919× |

### `128→128`：最强为 aggregate-first

| T | Aggregate ms | Transform ms | Fused ms | Fused/strongest | Fused 慢 |
|---:|---:|---:|---:|---:|---:|
| 1 | 126.57 | 134.86 | 196.87 | 0.643× | 1.555× |
| 2 | 63.64 | 66.86 | 103.00 | 0.618× | 1.618× |
| 4 | 31.89 | 33.47 | 54.12 | 0.589× | 1.697× |
| 8 | 16.07 | 17.53 | 32.21 | 0.499× | 2.004× |
| 16 | 7.27 | 8.03 | 19.87 | 0.366× | 2.733× |
| 32 | 4.15 | 5.18 | 15.25 | 0.272× | 3.675× |

融合路径在 `0/12` cell 中获胜，且线程越多差距越大。

## 7. 独立 replay 和细分时间

汇总：[`forward_replay_exclusive_summary.csv`](../evidence/original_tfs_shw_training/forward_replay_exclusive_summary.csv)；原始日志：[`raw_forward_replay_v4/`](../evidence/original_tfs_shw_training/raw_forward_replay_v4/)。

典型中位附近：

```text
1T aggregate = prep 7.18 + sparse 84.74 + dense 21.73 + epilogue 12.81 = 126.47 ms
1T fused     = prep 7.15 + fused core 189.38                         = 196.53 ms

8T aggregate = prep 1.07 + sparse 10.35 + dense 2.89 + epilogue 1.69 = 16.00 ms
8T fused     = prep 1.08 + fused core about 30.7                     = 31.81 ms

32T aggregate = prep 0.31 + sparse 2.22 + dense 0.96 + epilogue 0.58 = 4.08 ms
32T fused     = prep 0.31 + fused core about 15.0                    = 15.35 ms
```

差距位于 fused core，不是共同 prep。R4/4C 和 R4/6C 在 1/8/32T 也全部更慢。

## 8. 数值正确性

官方 arxiv、显式 self-loop、完整 normalized semantics：

- fused vs transform D40 rel-L2：`4.88e-7`；
- D128 rel-L2：`5.04e-7`；
- max absolute：小于 `5.68e-5`；
- Job 9963827 典型 fused rel-L2：约 `2.60e-7`；
- Job 9963827 max absolute：约 `2.96e-5`。

所以这是性能 NO-GO，不是 correctness NO-GO。

## 9. 根因分析

### 9.1 Dense 工作从 per-node 放大为 per-edge

Materialized dense transform 约为 `N×K×D`，edge-wise fused 约为 `NNZ×K×D`，比例约为平均度：

```text
NNZ/N ≈ 14.674
```

AMX 吞吐只能抵消部分放大，无法抵消约 14.7 倍逻辑重复。

### 9.2 D40 有更强的 transform-first

`D=40<K=128` 时，transform-first 先做每节点一次 `H·W`，再执行 40-wide sparse traversal。fused 虽能击败较弱的 aggregate-first，却无法击败该路径。

### 9.3 D128 重复完整 traversal

原始四个 C accumulator 迫使 128 输出列分两个 pass；CSR、indices、prefetch、H gather 和 Hbuf 写入均重复。

### 9.4 Degree sort 只降低 padding

Degree sort 提高 16-row lane efficiency，但不改变 `NNZ×K×D`。它优化主导成本中的常数项，不能消除主导项。

### 9.5 Strong baseline 更容易扩展

1→32T：

- D40 transform `59.96→2.48 ms`，`24.18×`；
- D40 fused `96.35→7.24 ms`，`13.31×`；
- D128 aggregate `126.57→4.15 ms`，`30.50×`；
- D128 fused `196.87→15.25 ms`，`12.91×`。

NUMA interleave 对强基线帮助更大，说明 fused 仍受 per-edge gather/compute 和重复 traversal 限制。

## 10. 原始推理结果为何不矛盾

原始论文/代码报告固定 K=128、32T、TFS BF16/FP32 accumulate 对 MKL FP32：kernel 几何均值 `2.21×`、ogbn-products `6.67×`、两层推理最高 `4.09×`、Reddit E2E `1.51×`。没有 backward、optimizer、现代 DGL Official AMP 或 layer-adaptive baseline。

项目重新运行 frozen Original-TFS kernel，在 ogbn-products 上得到：

```text
N=2,449,029; NNZ=123,718,152; avg degree=50.5
1T=5142.31 ms; 8T=646.99 ms; scaling=7.95×
sampled relative error<=0.003342
```

但这是 fixed 128→128、unweighted native forward、kernel-only，没有 matched DGL-Mixed、完整归一化训练和 layer-adaptive aggregate/transform。因此它证明原始 kernel 可运行和扩展，不证明其适合训练默认路径。

## 11. 后续完整训练集成

融合 NO-GO 后，系统采用 hidden `128→128 aggregate-first`、output `128→40 transform-first`。

Job 9959660 完整边界为 zero_grad、forward、ReLU、dropout、masked CE、backward、Adam，3 seeds、warmup2/repeat7：

| T | Hybrid | Reference | DGL | PyG |
|---:|---:|---:|---:|---:|
| 1 | 899.981 | 1037.210 | 1397.702 | 2270.106 |
| 2 | 999.122 | 820.383 | 760.831 | 1196.755 |
| 4 | 1121.585 | 697.222 | 408.484 | 646.261 |
| 8 | 1263.065 | 635.857 | 231.736 | 387.784 |
| 16 | 1488.909 | 522.753 | 136.322 | 241.019 |
| 32 | 2165.259 | 416.962 | 99.766 | 181.592 |

该 early hybrid 的负扩展是独立 runtime 问题：static degree-sorted chunks 严重不均衡；`dynamic,64` 在 8/32T 提升 `1.639×/1.693×`，连续 NNZ-balanced 再提升 `1.105×/1.088×`，相对初版合计约 `1.81×/1.89×`。后续 persistent worker/workspace/staging 修复才消除负扩展。

因此，全局 degree order 不能直接作为训练线程 partition。合理组合是外层 panel/NNZ balance，内层才考虑 degree-homogeneous 16-row tile grouping。

## 12. 最终决策与可复用部分

不进入默认训练路径：edge-wise fused `S·H·W`、全局 degree-sort + static thread chunks、D128 两次 output pass、用旧 MKL FP32 推理数字替代 mixed-precision 训练基线。

可以复用：16-row AMX tile micro-batch、BF16 operands + FP32 accumulation、VNNI packing、full-K gather 思想、software prefetch、tile-local zero/store、NNZ-balanced panel 内局部 degree grouping、shape/graph/runtime planner gate。

当前两级 panel、Y/Gs、Q-first、High-D one-final-pull 的目标，是消除训练 forward/backward 中间张量和重复 sparse scan，同时避免原始 TFS 的 per-edge dense amplification。准确表述应为：

> 保留 AMX tile-aware 机制，但重新设计训练专用数据流、panel 生命周期和按层计算顺序，而不是把原始 TFS 直接用于训练。

## 13. 复现和证据导航

| 内容 | 路径 |
|---|---|
| 原始 TFS kernel | `experiments/original_tfs_shw_training/amx_tfs_v3k_original.cpp` |
| 归一化三路径 benchmark | `experiments/original_tfs_shw_training/normalized_fused_forward_bench.cpp` |
| smoke/R sweep/formal launcher | `experiments/original_tfs_shw_training/run_forward_fused_*.sh` |
| replay 源码/launcher | `experiments/original_tfs_shw_training/normalized_fused_forward_replay.cpp`、`run_forward_replay_exclusive_v4.sh` |
| 12-cell 汇总 | `evidence/original_tfs_shw_training/forward_fused_formal_summary.csv` |
| replay 汇总与 raw | `evidence/original_tfs_shw_training/forward_replay_exclusive_summary.csv`、`raw_forward_replay_v4/` |
| 全过程 | `evidence/original_tfs_shw_training/experiment_chronology.csv` |

## 14. 证据完整性声明

Job 9963827 的九份原始日志已保存。Job 9959570 的原始 run directory 不在本地 publication tree，因此没有伪造 raw log；其数字来自项目 immutable experiment ledger，并由 Job 9963827 的独立 D128 replay 交叉确认。failed、invalid、diagnostic、formal、superseded 均分开标记，不把 kernel-only、shared-node或单次 smoke 包装成正式训练 E2E。

最终结论：在已验证的官方 ogbn-arxiv 训练边界内，原始 normalized edge-wise TFS `SHW` 融合应停止作为默认前向方向；其底层 AMX tile 机制可以继续服务于新的训练专用数据流。
