# Supervision-Scoped TFS：稠密终端融合探索

## 1. 研究范围

本轮只修改独立 shadow 源码，不修改权威 TFS。目标是在已经成立的
supervision-scoped terminal dataflow 上，继续消除框架级中间张量和数据
布局往返。

当前完整候选链为：

```text
H[N,K]
  -> 只聚合监督行 P[M,K]
  -> bounded terminal panel
  -> logits / exact CE / Gs
  -> panel-local dW
  -> compact Q-first dH
```

## 2. Bounded terminal 的输出维度边界

真实 IGB-small CSR，M=600,000，K=128，两层，1/8/32 线程。表中比值为
旧 supervision-scoped terminal / bounded terminal，大于 1 表示 bounded 更快。

| D | 1T terminal | 8T terminal | 32T terminal | 结论 |
|---:|---:|---:|---:|---|
| 19 | 0.993x | 1.002x | 1.028x | 基本无收益 |
| 47 | 1.026x | 1.080x | 1.130x | 小幅收益 |
| 128 | 1.116x | 1.092x | 1.076x | 稳定终端收益 |
| 256 | 1.194x | 1.137x | 1.049x | 稳定终端收益 |

D=19 不应启用。D>=47 可进入候选，但低维完整训练步只有约 0%--8% 收益，
容易被隐藏层和系统噪声吞没。D=128/256 的 32T 完整训练步记录含明显 optimizer
长尾，不能用来确定 planner。

## 3. Compact dW 原生 AMX 路径

旧实现对每个 bounded panel 执行：

```text
P[M,K] -> P^T.contiguous() -> torch.matmul(P^T, Gs)
```

新实验路径执行：

```text
512-row micro-panel
  -> panel-local Gs transpose
  -> panel-local P packing
  -> AMX BF16 / FP32-accumulate dW
  -> thread-local dW
  -> deterministic reduction
  -> KxD scatter
```

因此它消除的是全局 `P^T.contiguous()` 和框架/oneDNN 往返，同时保持
bounded working set。下表为 M=300,000、K=128 的完整原生路径与 PyTorch
BF16 matmul 对比；全部包含布局转换、归约和输出。

| D | 1T | 8T | 32T |
|---:|---:|---:|---:|
| 512 | 1.580x | 4.598x | 5.701x |
| 1024 | 1.350x | 2.636x | 3.795x |
| 2048 | 1.072x | 1.413x | 2.362x |
| 2983 | 0.845x | 0.903x | 2.224x |

数值相对 L2 误差约 0.166%。D=2983 明确说明 native dW 必须由 D、线程数和
布局共同规划，不能无条件启用。

### 真实图增量结果

IGB-small、D=512：

| 线程 | terminal speedup | 完整训练步 speedup |
|---:|---:|---:|
| 8 | 1.237x | 1.112x |
| 32 | 1.360x | 1.364x |

32T 完整训练步存在较明显系统长尾，需独立复测；但 terminal 与微基准方向一致。

IGB-small、D=2983、8T 的 padded-T2 native dW 则为负收益：terminal `0.981x`，
完整训练步 `0.987x`。因此即使同为 high-D，也不能仅按 D 启用 native dW；
线程数和尾维布局必须进入 planner。

同一 D=2983 在 32T 时，padded-T2 native dW 转为明确正收益：terminal 从
582.05 ms 降至 493.12 ms（`1.180x`），完整训练步从 883.46 ms 降至
793.30 ms（`1.114x`，配对中位数 `1.111x`）。dW 相对 L2 误差为 `0.117%`。

## 4. T4 direct-tail transpose

D 不是 32 的倍数时，旧路径先把完整 Gs panel 复制到 padded stride，再转置。
T4 直接从原始 stride 转置，只有最后的 16x16 尾块补零。

它存在明确权衡：

- 节省完整 padded copy；
- 原始 D=2983 stride 不对齐，转置加载本身可能变慢；
- 低线程下省拷贝更重要，高线程下旧 aligned-T2 可能更好。

跨节点微基准中，D2983 的原生总时间从 760.69/132.97 ms 降到
645.76/101.75 ms（1/8T），但 32T 跨节点绝对时间反而从 50.89 ms 变为
65.97 ms。必须使用同一二进制、同一节点的 T2/T4 配对实验后才能冻结选择。

## 5. Compact logits + fused epilogue

原 TFS 已有成熟 high-D AMX 4-C-tile GEMM，但 supervision-scoped terminal
此前仍使用：

```text
torch.matmul(P, W) -> FP32 -> scale -> bias
```

新 shadow 接口复用原 AMX kernel，并在 tile epilogue 内完成 scale+bias。
D=2983、M=300,000、K=128、32T 的首轮完整 producer 微基准：

```text
PyTorch matmul + scale/bias : 73.32 ms
AMX 4-tile fused epilogue   : 24.66 ms
speedup                     : 2.973x
relative L2 error           : 0.164%
```

首版输出使用 padded stride 2992。v2 已改为 epilogue 直接写连续 stride 2983，
避免后续 exact CE / scale kernel 因非连续 view 发生复制；仍需真实训练步验证。

## 6. 当前结论

已确认的方向不是单独调 kernel，而是把监督域裁剪与 bounded panel 数据流继续
贯穿到稠密终端：

1. 只生成监督行；
2. 不物化 full logits / full G；
3. 不物化全局 P transpose；
4. panel-local AMX dW 和确定性归约；
5. 复用原 TFS high-D AMX logits，并融合 scale+bias；
6. 根据 D、M、线程数和布局成本选择 PyTorch、padded-T2 或 direct-tail-T4。

最终结论必须等待真实 D2983 端到端 A/B 和权威 TFS 直接对照，不能把不同节点
的独立 speedup 相乘。

## 7. 权威 TFS 直接对照：native dW 后

IGB-small、D=2983、L2、32T，同一独占单路节点内交替运行，1 次 warm-up、
7 次正式重复：

| 路径 | step 中位时间 |
|---|---:|
| 权威 TFS | 1325.26 ms |
| supervision scope + bounded CE + logsoftmax + native dW | 632.12 ms |

直接加速比为 **2.0965x**。权威路径 7 次范围为 1304--1345 ms，新路径范围为
626--675 ms。正确性指标：loss 绝对误差 `9.54e-7`，dH 相对 L2 `0.0123%`，
dW 相对 L2 `0.000011%`，db 最大绝对误差 `3.05e-6`。

此前未接 native dW 的同口径新路径中位时间为 737.52 ms、直接加速比
1.763x。native dW 后总时间进一步降至 632.12 ms。以上两次实验的最终结论
均来自各自的同节点直接对照；不把跨实验 speedup 相乘作为证据。

## 8. T4 的同节点结论

D=2983、M=300,000 的同一进程交替 A/B 排除了跨节点漂移：

| 线程 | padded-T2 / Torch | direct-tail-T4 / Torch | T4 / T2 |
|---:|---:|---:|---:|
| 1 | 0.769x | 0.914x | 1.189x |
| 8 | 0.933x | 1.187x | 1.273x |
| 32 | 2.376x | 2.520x | 1.061x |

因此 T4 确实消除了非 32 对齐 D 的无效 padded copy，但它不是独立的万能
kernel：1T 下即使 T4 改善了 T2，native dW 整体仍未打赢框架路径。

真实 IGB-small、M=600,000 的 T4 native dW 增量结果如下。完整步容易被
optimizer 长尾污染，terminal 与 paired speedup 是更稳定的 dispatch 证据。

| D | 8T terminal | 8T train | 32T terminal | 32T train |
|---:|---:|---:|---:|---:|
| 512 | 1.235x | 1.117x | 1.707x | 1.250x |
| 1024 | 1.128x | 1.076x | 1.443x | 1.102x |
| 2048 | 1.057x | 1.034x | 1.274x | 1.293x* |
| 2983 | 1.025x | 1.017x | 1.224x | 1.154x |

`*` D=2048/32T 的 optimizer 有长尾，其配对完整步 speedup 为 1.222x。

## 9. 连续 logits v2 与 FP32 oracle

v1 的 padded-stride view 在孤立 producer 中很快，但在真实 CE 中触发了昂贵
的非连续访问，完整步仅为 0.824x。v2 让 AMX epilogue 直接写逻辑 stride D，
输出连续；D=2983、M=300,000、32T 的 producer 结果为：

| 路径 | 中位时间 |
|---|---:|
| PyTorch BF16 matmul + scale/bias | 73.99 ms |
| native AMX FP32-accumulate + fused epilogue | 29.41 ms |

producer speedup 为 **2.516x**。真实 IGB bounded 路径中，terminal 从 508.08 ms
降至 440.87 ms（1.152x），完整步从 795.20 ms 降至 737.68 ms（1.078x）。

旧 A/B gate 要求 native 梯度贴近 PyTorch BF16 输出到 `1e-4`，实测 dH
relative L2 为 `1.2246e-4`，因此旧 gate 报 fail。FP32 oracle 证明这不是精度
退化，而是 native 路径保留了 FP32 accumulator：

| 路径 | logits relative L2 vs FP32 | dH relative L2 vs FP32 | dW relative L2 vs FP32 |
|---|---:|---:|---:|
| PyTorch BF16-output | 1.638e-3 | 1.180e-2 | 1.174e-2 |
| native FP32-accumulator | 8.94e-8 | 5.97e-7 | 6.01e-7 |

必须同时保存旧 gate 与 FP32-oracle 结果，不能为了晋级修改旧阈值；正式数值
合同应明确采用哪一种 accumulation 语义。

## 10. M-D-thread 边界与 planner 约束

短微基准补测 M=8,192/32,768/131,072，结论是 dispatch 至少依赖
`(M,D,K,threads,padding)`：

- logits 的唯一负点是 D=512、M=8,192、32T（0.599x）；它近似对应
  `M*D/threads` 小于约 0.5M 输出元素/线程。
- dW 在 D=2983、M=8,192 时为 0.617x/0.375x（8/32T），M=32,768
  才转为 1.026x/1.339x。
- dW 在 D=512、M=8,192 已为 1.688x/1.136x，说明 M 边界还受 D、
  transpose、thread-local accumulator 和 reduction 成本共同影响。

因此不能把 `D>K`、`D>=512` 或数据集名写成最终规则。推荐的 planner 层次为：

1. 用监督比例与选中边比例决定是否使用 supervision scope；
2. 用 `M*D` working set 决定是否 bounded；
3. 用 D/K、平均度与线程决定 Y-first/Q-first；
4. 用 M、D、K、线程与 padding 决定 framework/native dW；
5. native dW 内部仅在尾维存在时选择 padded-T2/direct-tail-T4；
6. 用每线程输出工作量决定 framework/native logits。

在更多真实 shape 校准完成前，所有 native dense 路径继续保持 shadow opt-in，
不修改权威默认行为。
