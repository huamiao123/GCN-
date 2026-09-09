# Bounded-Panel Terminal CE/Q-first 探索结果

日期：2026-09-06

状态：影子源码实验通过；未修改 authority TFS；尚未进入正式 48-cell。

## 1. 问题

Supervision-scoped TFS 已经把终层输出从全图 `N x D` 缩小到训练节点
`M x D`，但 IGB-small 的 `M=600,000, D=2,983` 仍会产生巨大的 logits
与 logits gradient。一个 FP32 `M x D` 张量约 7.16 GB（十进制），当前
PyTorch CE/autograd 边界会让多个宽张量同时存活。

## 2. 三条被严格区分的路径

1. `current`：当前 supervision-scoped `logits -> PyTorch CE -> grad_logits
   -> Q/dW/db`。
2. `streaming`：按类别 tile 做两遍在线 log-sum-exp；不保存 logits，但为
   生成梯度重算一次 `P@W`。
3. `panelized`：只在一个 row panel 中临时保存完整类别 logits，立即计算
   CE gradient、Q、dW、db，然后释放该 panel；不重算 `P@W`，也不保存
   全局 `M x D`。

`panelized` 的持久工作集为 compact `P[M,K]`、compact `Q[M,K]` 和参数
梯度，宽临时量被限制为 `R x D`：

```text
P panel
  -> AMX/BF16 classifier GEMM
  -> exact FP32 log-sum-exp / CE
  -> gradient in the same bounded panel
  -> dW/db accumulation + compact Q generation
  -> release wide panel

backward:
compact Q -> selected-transpose rectangular sparse pull -> dH
```

## 3. 独立 dense-chain gate

形状：`M=600,000, K=128, D=2,983`，32T，同一独占单路节点，10 次正式
重复。Job 10360158。

| 路径 | 中位时间 | 相对 current | 进程峰值 RSS | 相对 current 内存 |
|---|---:|---:|---:|---:|
| current PyTorch CE/autograd | 763.44 ms | 1.000x | 27.64 GB | 1.00x |
| 强原地 materialized 上限 | 635.32 ms | 1.202x | 13.99 GB | 1.97x 降低 |
| panelized, R=131,072 | 732.56 ms | 1.042x | 4.27 GB | 6.47x 降低 |
| panelized, R=300,000 | 645.94 ms | 1.182x | 9.02 GB | 3.06x 降低 |

两遍 class-streaming 的最佳点为 952.8 ms，仍比强 materialized 慢 1.50x，
因此不作为性能默认路径。它只适合作为极低内存模式。

## 4. 接入真实稀疏终层后的结果

数据集：IGB-HOM-small，`D=2,983`，两层，训练节点 600,000。当前 compact
Q-first 与 bounded-panel CE + compact Q-first 在同一节点内交替配对。

| 线程 | 当前 terminal | 新 terminal | terminal 加速 | 当前完整训练步 | 新完整训练步 | 训练步加速 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17,687.3 ms | 13,228.7 ms | 1.337x | 21,495.5 ms | 16,530.1 ms | 1.300x |
| 8 | 2,032.5 ms | 1,916.5 ms | 1.061x | 2,475.1 ms | 2,336.8 ms | 1.059x |
| 32 | 755.7 ms | 702.7 ms | 1.075x | 895.4 ms | 839.0 ms | 1.067x |

三档几何均值：terminal 1.151x；完整训练步 1.137x。Job 10360168。

另一次 32T 配对曾得到 1.329x 训练步收益，但旧路径在该节点明显偏慢；该
数字只保留作 provenance，不作为结论。正式结论采用跨线程复测中更保守的
1.067x。

## 5. 正确性

三个线程档全部通过。代表性范围：

- loss absolute error：0；
- dH relative L2：约 0.0028%--0.0029%；
- dW relative L2：约 0.199%--0.207%；
- db relative L2：约 1e-7 量级。

误差来自与当前 TFS 一致的 BF16 GEMM/FP32 accumulate 路径，不改变训练
目标，不使用采样或近似 loss。

## 6. 适用边界

- IGB-2983：启用候选；高 D、宽 logits 是主要目标。
- Products-47：独立 gate 中 panelized/streaming 均无性能优势，应回退。
- IGB-19：此前 supervision-scoped 本身只有 1.001x，应直接回退 authority。
- planner 必须至少考虑 `M/N`、`D/K`、线程数和 row-panel 内存预算。

## 7. 创新边界

不能把以下内容单独声称为首创：监督节点裁剪、在线 log-sum-exp、fused
cross entropy、普通 row tiling 或中间张量消除。相关思路分别存在于 SAR/
MFG、Cut Cross-Entropy 和通用算子融合工作。

可能成立的具体贡献是：

> 面向 CPU full-batch GCN 的 loss-coupled supervision-scoped terminal
> dataflow，把 selected sparse aggregation、AMX classifier、精确 CE、
> Q-first backward preparation 和 AVX rectangular sparse propagation 组织成
> bounded-panel pipeline，并由图范围、形状、线程和内存共同规划。

该表述仍需系统文献检索后才能使用“首次”。

## 8. 下一阶段

1. 将 Python panel loop 下沉为 native terminal-loss kernel，减少 dispatcher、
   transpose/contiguous 和临时 allocator 开销。
2. 让 AMX 直接输出 compact BF16 Q panel，并在同一 panel 内累计 dW/db。
3. 增加 per-NUMA dW/db partial，避免多 domain 写同一参数梯度。
4. 在同一权威 launcher 中直接比较 authority、compact Q-first 和
   bounded-panel，不能把不同节点的历史比例相乘当正式 speedup。
5. gate 通过后再跑 48-cell；D 较小或监督边覆盖高时必须自动回退。

## 8.1 Native epilogue 补充结果

后续将 panel 内原先分开的 `db=sum(grad)` 与
`selected-scale + FP32->BF16` 接到已有 native fused epilogue。旧实现按
worker 使用一个长 FP32 accumulator；8T 下 db 最大误差达到
`6.11e-5`，没有通过严格门禁，因此没有接受该版本。

修复方式不是放宽 tolerance，而是把 db 改成与线程数无关的固定 2,048-row
chunk：chunk 内 AVX-512 FP32 累加，最后按固定顺序使用 FP64 合并。重新构建
后，8T/32T 的 db 最大误差约为 `2.2e-6--2.6e-6`，且 native epilogue
相对未融合 panel 路径仍有 1.044x（8T）和 1.022x（32T）收益。

接入完整两层 IGB-2983 训练后的最终影子矩阵：

| 线程 | 当前 compact Q-first | bounded-panel + native epilogue | 加速比 |
|---:|---:|---:|---:|
| 1 | 21,525.4 ms | 16,023.5 ms | 1.343x |
| 8 | 2,471.1 ms | 2,260.8 ms | 1.093x |
| 32 | 917.0 ms | 842.0 ms | 1.089x |

三线程几何均值为 1.169x。三个 cell 的 loss、dH、dW、db 全部通过；dH
relative L2 约 0.0026%--0.0029%，dW 约 0.186%--0.219%。

当前保守启用边界：只把 `D=2,983` 视为已验证区域。Products-47 和 IGB-19
必须回退；未测过的中间 D 不得仅凭公式自动打开。正式 planner 需要先补
`D=256/512/1024/2048` controlled shape sweep。

## 9. 文件与 provenance

- 实现：`python/tfs_train/supervision_scope.py`
- dense gate：`tests/bench_streaming_terminal_loss.py`
- 真实终层 gate：`tests/bench_panelized_scope_igb.py`
- paired dense job：10360158
- thread job：10360168
- fixed native fused-db operator gate：10360207
- fixed native fused-db full thread matrix：10360211
- 第一次端到端 job 10360163 仅在最终 JSON 序列化阶段失败，原因是逐次记录
  错误包含 Tensor；修复后 job 10360165/10360168 通过。失败任务不进入结果。

## 10. Controlled output-dimension/thread sweep

为避免用单一 `D > K` 规则误判，固定真实 IGB-HOM-small 训练 CSR、`K=128`、两层模型、
600,000 个监督节点和同一实现，仅改变终层输出维度与线程数。`D<2983` 的标签使用
`label % D`，只用于保持 CE 合法；这是一组性能与数值工作负载，不是准确率实验。

表中为完整训练步加速比：`current compact Q-first / bounded-panel terminal CE + native epilogue`。

| D | 1T | 2T | 4T | 8T | 16T | 32T | 六线程几何均值 | 最小值 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1.039x | 1.029x | 1.029x | 1.021x | 1.031x | 1.024x | 1.029x | 1.021x |
| 512 | 1.070x | 1.053x | 1.052x | 1.043x | 1.030x | 0.945x | 1.031x | 0.945x |
| 1024 | 1.125x | 1.108x | 1.093x | 1.136x | 1.074x | 0.991x | 1.087x | 0.991x |
| 2048 | 1.178x | 1.148x | 1.145x | 1.126x | 1.109x | 1.130x | 1.139x | 1.109x |
| 2983 | 1.343x | 1.107x | 1.107x | 1.093x | 1.105x | 1.089x | 1.137x | 1.089x |

全部 30 个 cell 的 loss、dH、dW 和 db 门限通过。D=256/512/1024/2048 的
1/8/32T 来自 job 10360310，2/4/16T 来自 job 10360339；D=2983 的 1/8/32T
来自同一修正版影子实现的 job 10360211，2/4/16T 来自 job 10360339。

### 10.1 结论与保守 planner 边界

- `D=2048` 和 `D=2983` 在 1--32T 全范围稳定获益，最差仍分别为 1.109x 和 1.089x。
- `D=1024` 在 1--16T 有 1.074x--1.136x 收益，但 32T 仅 0.991x，属于线程相关边界区。
- `D=512` 在 32T 退化到 0.945x；`D=256` 虽全为正，但仅 1.021x--1.039x，
  收益不足以承担新路径复杂度。
- 因此 `D > K`、`D/K` 单阈值或仅按平均度决策都不充分。至少还要考虑线程数，
  而跨图 planner 还必须加入监督行数、`M/N`、图传播成本和内存预算。

该结论仅对应旧手工 CE 路径；当时的保守建议是 `D>=2048` 才默认启用。
后续 in-place log-softmax 已改变边界，最终 planner 必须以下文第12节和低维补测为准，
不得继续使用本节的旧 cutoff，也不得把 IGB-small 经验当作通用公式。

## 11. In-place log-softmax follow-up

阶段剖析表明，旧 bounded-panel 路径虽然大幅缩短 backward，但手工
`logsumexp -> subtract -> exp` 会多次扫描 FP32 logits。影子实现增加了显式开关
`TFS_SCOPE_LOGSOFTMAX_OUT=1`，使用 PyTorch 优化的
`log_softmax(logits, out=logits)`，随后原地 `exp` 形成精确 CE gradient。

在 32T、真实 IGB-small CSR、`M=600000`、`K=128`、row tile 300000 上，
相对于 current compact Q-first 的完整两层训练步配对结果为：

| D | current | bounded + log-softmax | 加速比 |
|---:|---:|---:|---:|
| 512 | 616.3 ms | 530.3 ms | 1.162x |
| 1024 | 486.2 ms | 436.5 ms | 1.114x |
| 2048 | 712.3 ms | 590.2 ms | 1.207x |
| 2983 | 903.5 ms | 725.2 ms | 1.246x |

四个 cell 均通过既有完整 gate。loss 最大绝对误差不超过 `9.54e-7`；dH relative
L2 最大为 `6.67e-5`，dW 最大约 `0.211%`，db 最大绝对误差约 `3.43e-6`。
job 10361565。

独立 CE 原语配对显示 1.487x--1.540x 收益；完整 terminal 配对显示
1.162x--1.293x。收益来自减少 CE 阶段的 logits 扫描，不改变损失定义，也不使用
近似 softmax。

该结果将 `D=512/1024` 的 32T 边界从旧手工 CE 的无收益区域转为正收益候选，
但 D=512/2983 的若干重复存在系统抖动，且目前只覆盖一个图和 32T。因此该开关继续
保留在影子实现，必须完成 1/2/4/8/16/32T 与跨图复测后才能进入默认 planner。

row-tile sweep（job 10360841）同时证明不存在统一最佳 tile：D=512/1024/2048
在本轮以 600000 较好，而 D=2983 以 450000 较好；D=2048/450000 还出现明显
不稳定。故不采用“统一增大 panel”的规则。

## 12. Direct authority comparison and completed thread matrix

### 12.1 IGB-2983 direct authority comparison

job 10361689 在同一独占节点、同一进程内交替比较原权威 full-terminal TFS 与完整新路径：

```text
supervision-scoped selected pull
  + bounded terminal CE
  + in-place log-softmax
  + native scale/BF16/db epilogue
  + compact Q-first rectangular backward
```

32T、两层 IGB-small、D=2983 的完整训练步：

| 路径 | 中位数 | 7次范围 |
|---|---:|---:|
| authority full-terminal TFS | 1300.5 ms | 1298.4--1342.7 ms |
| complete bounded path | 737.5 ms | 717.3--752.2 ms |
| direct speedup | **1.763x** | -- |

loss 误差为0；dH relative L2 为0.0127%，dW为0.1237%，db最大绝对误差
`3.62e-6`。这是直接配对结果，取代此前把两个历史比例相乘得到的估算。

### 12.2 High-D shape/thread matrix

job 10361688 完成 `D={512,1024,2048,2983}` × `T={1,2,4,8,16,32}` 的
24-cell矩阵。表中为新路径相对于 supervision-scoped current compact Q-first 的完整训练步
加速比：

| D | 1T | 2T | 4T | 8T | 16T | 32T | 几何均值 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1.136x | 1.113x | 1.108x | 1.103x | 1.095x | 1.069x | 1.104x |
| 1024 | 1.227x | 1.211x | 1.204x | 1.197x | 1.151x | 见下 | -- |
| 2048 | 1.521x | 1.282x | 1.280x | 1.257x | 1.219x | 1.228x | 1.294x |
| 2983 | 1.309x | 1.287x | 1.280x | 1.268x | 1.271x | 1.288x | 1.284x |

D=1024/32T 的首场结果受到运行期漂移影响：独立中位数之比为0.892x，而同轮配对比
中位数为1.028x。随后 job 10362171 在三个独占节点以10次记录复测，分别得到
1.118x、1.115x、1.118x，故将首场0.892x标记为系统漂移异常，不用于planner。

### 12.3 Products cross-graph direct comparison

job 10362178 使用真实 Products 图、真实47类输出，直接比较 authority full-terminal 与
完整新路径：

| 层数 | 1T | 8T | 32T | 三线程几何均值 |
|---:|---:|---:|---:|---:|
| L2 | 1.319x | 1.305x | 1.161x | 1.260x |
| L3 | 1.103x | 1.099x | 1.046x | 1.082x |

全部6个cell通过数值门限。L3收益较小是因为终层优化在三层完整训练中的占比下降，
符合Amdahl规律；该结果证明完整路径并非只对IGB高D有效，但不同图和层数收益不同。
