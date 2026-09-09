# Supervision-Scoped TFS：相关工作边界与第一轮实现结论

## 结论

“只计算影响标签节点的子图”不是新思想。最接近的已发表工作是 MLSys 2022 的
SAR：它在分布式 full-batch GNN 中使用 DGL Message Flow Graph，只更新会影响
标签输出的节点，并说明梯度只从标签节点产生。

截至 2026-09-05 的检索中，没有发现已发表工作同时实现以下组合：

1. 单路 CPU full-batch GCN；
2. 只生成末层监督行，而隐藏层保持正常 full-graph；
3. compact terminal aggregation、compact logits/CE 与反向数据流联合设计；
4. 在 Y-first、Q-first、成熟 active-row backward 之间按形状和活跃范围选择；
5. 复用 AMX dense、AVX-512 sparse、panel 与 BF16/FP32 数值合同；
6. 不采样、不改变训练目标、不进行节点重排。

因此不能声称“首次利用标签节点裁剪”，可以研究和主张的是：

> 面向 CPU AMX full-batch GCN 的 supervision-scoped terminal dataflow，
> 通过跨越末层 sparse/dense、loss 和 backward 边界的代数与执行规划，
> 消除非监督目标行的末层计算，并选择最合适的梯度数据流。

这仍需更完整文献检索和同行评议，不能把“暂未检索到”写成绝对首创。

## 相关工作边界

### SAR / Message Flow Graph

- 论文：Sequential Aggregation and Rematerialization，MLSys 2022。
- 已有内容：从标签输出反向确定必要节点；使用 DGL MFG 避免无关节点更新；重点是
  分布式 full-batch 的内存缩放和顺序重物化。
- 与本工作的重叠：监督节点决定活跃计算域。
- 差异：当前候选只收缩 terminal layer，并把 compact forward/loss 与 TFS 的
  Y/Q/active-row backward 和 AMX/AVX kernel 选择联合起来；目标首先是单路 CPU
  端到端时间，而不是分布式重物化。
- 来源：https://proceedings.mlsys.org/paper_files/paper/2022/file/1d781258d409a6efc66cd1aa14a1681c-Paper.pdf

### DGL/PyG 的 block/MFG mini-batch 执行

- 已有内容：以 output/seeds 为目标构造多层 block，只计算其依赖节点。
- 与本工作的重叠：矩形 sparse operator 和紧凑 destination rows 都是成熟抽象。
- 差异：当前候选保持 exact full-batch objective 和既有 full-graph hidden layers，
  只针对监督稀疏且 frontier 很快饱和的真实图优化末层，不引入采样。

### Graphiler / FeatGraph

- Graphiler：使用 message-passing DFG 做算子重排、拆分和融合。
- FeatGraph：联合优化图遍历与 feature dimension。
- 重叠：跨算子数据流变换、图维度联合规划属于已有系统方法论。
- 差异：当前候选 planner 的输入额外包含监督活跃域
  `|M|/N、E_M/E、U_M/N`，并针对训练末层的 forward-loss-backward 精确代数。
- 来源：
  - https://proceedings.mlsys.org/paper_files/paper/2022/file/a1126573153ad7e9f44ba80e99316482-Paper.pdf
  - https://arxiv.org/abs/2008.11359

### GNN computational-graph reorganization

- MLSys 2022 的工作已经研究 propagation-postponed reorganization、统一线程映射、
  fusion 和中间量重计算。
- 当前工作不能泛泛地把“算子重排/中间张量消除”作为首创；贡献必须落到监督域、
  末层双向数据流和 CPU AMX/AVX planner 的具体组合。
- 来源：https://proceedings.mlsys.org/paper_files/paper/2022/file/b559156047e50cf316207249d0b5a6c5-Paper.pdf

## 当前实现

权威源码没有修改。实验 worktree：

```text
C:\Users\花喵\tfs_supervision_scope_shadow_20260905
```

服务器 shadow：

```text
/online1/huangjianqiang_group/hdacp1/wzh/tfs_supervision_scope_shadow_20260905
```

新增执行链：

```text
hidden H[N,K]
  -> selected pull: P_M=(SH)_M
  -> compact logits Z_M=P_M W+b
  -> CE(G_M)
  -> planner chooses:
       compact Y-first
       compact Q-first
       authority active-row backward
```

一次性构建的矩形转置 CSR 不改变图、不重排节点，只把从训练目标行出发的边显式
存成反向 pull 结构。其构建耗时必须进入 cold-start 报告，但不进入稳态 epoch。

## Products 完整矩阵结果

图结构：

- N = 2,449,029；
- 训练节点 = 196,615（8.0283%）；
- 训练目标行涉及的邻接条目（含隐式 self）= 33,914,532；
- 一次性 scope/transpose 构建约 0.57 s。

全部 cell 均为同一节点内配对、3 次 warm-up、20 次正式重复。加速比是
`authority full / supervision scoped`。

### 两层

| 线程 | Y-first terminal | Y-first train | mature-backward terminal | mature-backward train | 最优 train |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.039x | 1.483x | 1.645x | 1.334x | 1.483x |
| 2 | 2.060x | 1.479x | 1.627x | 1.323x | 1.479x |
| 4 | 1.857x | 1.407x | 1.620x | 1.318x | 1.407x |
| 8 | 1.722x | 1.296x | 1.601x | 1.267x | 1.296x |
| 16 | 1.561x | 1.230x | 1.547x | 1.277x | 1.277x |
| 32 | 1.256x | 1.123x | 1.457x | 1.195x | 1.195x |

两层跨线程几何均值：Y-first 1.330x，mature-backward 1.285x，逐 cell
planner 选择后 1.352x。

### 三层

| 线程 | Y-first terminal | Y-first train | mature-backward terminal | mature-backward train | 最优 train |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.009x | 1.135x | 1.641x | 1.101x | 1.135x |
| 2 | 2.035x | 1.139x | 1.622x | 1.099x | 1.139x |
| 4 | 1.875x | 1.122x | 1.644x | 1.097x | 1.122x |
| 8 | 1.710x | 1.095x | 1.607x | 1.085x | 1.095x |
| 16 | 1.569x | 1.079x | 1.579x | 1.088x | 1.088x |
| 32 | 1.256x | 1.051x | 1.460x | 1.084x | 1.084x |

三层跨线程几何均值：Y-first 1.103x，mature-backward 1.092x，逐 cell
planner 选择后 1.110x。三层绝对节省与两层接近，但多出一个不受 terminal
scope 影响的全图隐藏层，因此相对加速更小。

### 实测 planner 边界

Products 上的最优选择在两层和三层完全一致：

```text
1/2/4/8T   -> compact Y-first
16/32T     -> compact forward + mature active-row backward
```

这说明线程数必须进入 planner。只使用 `D>K`、训练节点比例或平均度都不足以
解释该 crossover。建议 planner 输入至少包括 `|M|/N`、`E_M/E`、`D/K`、
线程数和层数。

### 冷启动与数值

一次性 0.57 s 的 scope 构建若计入冷启动，按逐 cell 最优路径计算，Products
两层约 0.2--11.2 个 epoch、三层约 0.3--9.7 个 epoch 即可摊销；不能把它
藏入 steady-state。

- logits relative L2 约 0.24%--0.28%；
- mature-backward 的 dH relative L2 约 0.028%--0.036%；
- compact Y-first 的 dH relative L2 约 0.169%--0.172%；
- dW 约 0.001%（mature）或 0.17%（Y-first）；
- 旧 gate 有三个低线程 cell 因 db 的单一相对阈值标红，但其最大绝对误差仅
  `4.7e-6`--`1.07e-5`。验收已改为严格的绝对/相对联合条件，原始 JSON 不改写。

## IGB-small 输出维度边界

IGB-small 使用固定 60/20/20 split，训练目标为 600,000/1,000,000 节点；
训练目标行涉及 7,834,010 个含 self 邻接条目。scope 构建约 0.132 s。

32T、两层、3 次 warm-up、20 次正式重复的确认结果：

| 输出维度 | 后向选择 | terminal speedup | train-step speedup | 结论 |
|---:|---|---:|---:|---|
| 19 | mature active-row | 0.880x | 1.001x | 回退原路径 |
| 2983 | compact Q-first | 1.526x | 1.479x | 明显启用 |

高维权威 cell 的 train-step 中位数从 1314.5 ms 降到 888.7 ms：forward
283.5 -> 232.6 ms，backward 933.8 -> 561.4 ms。数值误差为 logits 0.182%、
dH 0.0126%、dW 0.165%，db 可忽略。20 个逐次配对 speedup 的中位数为
1.477x，最小 1.463x，P95 1.508x，说明收益不是少数异常点造成。

最初一次高维 gate 错误地用 `labels.max()+1` 推导出 D=2943；该结果只保留作
provenance，不进入结论。确认实验显式锁定权威 `D=2983`。

IGB-19 首轮 5 次样本曾出现 1.338x 的假高收益，但 20 次复测后回到 1.001x；
逐次配对范围为 0.948x--1.046x。这说明正式 gate 必须保留原始逐次记录，
不能用少量重复决定 planner。

## 当前判断

路线值得继续，但必须由 planner 有条件启用，并拆成两阶段贡献：

1. 已成立：supervision-scoped terminal forward + backward dataflow planner；
2. 已成立：高输出维时 compact Q-first 不物化 full logits/full G，IGB-2983
   32T 两层训练步达到 1.479x；
3. 待优化：低输出维 compact backward，目标是不物化 full G 且不比 mature
   active-row kernel 慢；
4. 必须回退：监督边覆盖较高且输出维低时（IGB-19），收益为零。

Products 的 mature-backward 路径仍会临时物化 full G，因此不能泛称整个实现
已经消除了完整 G；只有 compact Y/Q 路径满足该条件。论文主张应聚焦
“监督域 + 形状 + 线程”共同驱动的末层双向数据流，而不是标签节点裁剪本身。
