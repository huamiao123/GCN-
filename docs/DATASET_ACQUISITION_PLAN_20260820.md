# TFS-Train 数据集扩展与下载计划（泛化后 auto）

**日期：** 2026-08-20  
**适用权威版本：** `final_pre_numa` / `TFS_RELEASE_PROFILE=final_pre_numa`  
**状态：** 规划文档；本文件不授权下载、提交 Slurm 任务或修改实验配置。

## 1. 目的与边界

目标是在论文中形成可复现的、面向 **静态同构全图 CPU GCN 训练** 的数据集矩阵。每个正式比较均须同时有：

1. 相同 canonical CSR、相同自环和归一化语义；
2. TFS `final_pre_numa` auto 与 stock DGL `GraphConv(norm="both")`；
3. 相同官方 split、seed、层数、线程数、CPU 绑定与 NUMA 策略；
4. 真实图 FP32 数值 gate、200-epoch 训练质量、冷启动与稳态计时分别报告；
5. 每个候选数据集先通过接入 gate，才可进入正式 48-cell（或其扩展）矩阵。

不将异构图、带必需边特征的算子、链接预测、采样训练或只有结构没有官方特征/标签/split 的网络，混入本轮主表。

## 2. 当前已具备的主表数据

| 数据集 / 配置 | 角色 | 备注 |
|---|---|---|
| `ogbn-arxiv` | 引文图、中等规模 | 169,343 节点、1,166,243 边、128 输入、40 类。 |
| `ogbn-products` | 商品图、大规模 | 2,449,029 节点、61,859,140 边、100 输入、47 类。 |
| IGB-HOM-small / 19 类 | 百万节点、窄输出 | 与 2983 类共享图拓扑；应视为不同输出宽度配置，不应写成两个独立图。 |
| IGB-HOM-small / 2983 类 | 百万节点、极宽输出 | High-D 路径与内存压力代表。 |

## 3. 推荐下载批次

### Batch A：立即下载并完成接入（补领域与稀疏结构）

| 数据集 | 官方规模 | 论文价值 | 接入前专项检查 |
|---|---:|---|---|
| **Reddit** | 232,965 节点 / 114,615,892 边 / 602 输入 / 41 类 | 高边密度社交图；重点看源侧 gather 与线程扩展。 | 固定官方 mask；检查 DGL 与 TFS 的有向/无向预处理一致。 |
| **Flickr** | 89,250 / 899,756 / 500 / 7 | 中等图，补足不同输入宽度与低类别数。 | 固定 split；禁止用作性能主结论的唯一小图。 |
| **Yelp** | 716,847 / 13,954,819 / 300 / 100 | 真实业务/社交图，规模和类别宽度均不同于现有 OGB。 | **先核对发布版本的单标签/多标签语义**；若为多标签，需单独定义 BCE/ROC-AUC 契约，不与当前 cross-entropy 主表混合。 |

### Batch B：大图补强（论文应至少完成其中两项）

| 数据集 | 规模 / 访问方式 | 论文价值 | 准入条件 |
|---|---|---|---|
| **AmazonProducts（GraphSAINT 版本）** | 1,569,960 节点 / 264,339,468 边 / 200 输入 / 107 tasks | 高边数商品图，补足 OGB Products 之外的强稀疏压力。 | 先核对标签是否多标签；若需要 BCE，作为独立任务族报告。 |
| **IGB-HOM-medium** | IGB 官方可直接下载的 homogeneous `medium` 规模；19/2983 类可选 | 将当前 IGB-small 的百万节点压力向上扩展，同时保留可解释的输入/输出宽度压力。 | 下载前记录压缩包、解压后磁盘、FP32/BF16 特征、CSR、DGL 图和 workspace 的内存预算；先只跑 1 epoch 与 32 线程。 |
| **ogbn-papers100M** | 111,059,956 节点 / 1,615,685,872 边 / 128 输入 / 172 类 | 最有说服力的公开同构超大图标尺。 | 仅作 scale-out 专项：先做数据加载、CSR 构建、峰值 RSS、单 epoch 和 stock DGL 可行性 gate；通过后才考虑 200 epoch。 |

### Batch C：正确性与泛化附录（不作为性能主表）

`Cora`、`Citeseer`、`PubMed`、`Amazon Computers`、`Amazon Photo`、`Coauthor CS`、`Coauthor Physics`。

这些图应服务于：任意输入/隐藏/输出维度、尾部 padding、层数 2--5、真实图 FP32 对照和 workspace 复用回归。它们太小，不应支持“CPU 稀疏训练端到端加速”的主结论。

## 4. 明确不纳入当前主表

| 数据集 / 类别 | 原因 |
|---|---|
| `ogbn-proteins` | 有必需的 8 维边特征、112 个多标签任务和 ROC-AUC，超出当前无权 GCN 契约。 |
| `ogbn-mag`、HGB、MovieLens | 异构图；需另行定义关系类型、聚合和公平基线。 |
| IGB large / full、IGB260M / IGBH600M | 官方说明需要超过 500GB 磁盘；在单节点 full-graph DGL 可行性实验前不下载。 |
| SNAP/Friendster 等纯拓扑网络 | 缺少与当前训练论文同口径的官方特征、标签与 split；可做 SpMM 微基准，不进训练主表。 |

## 5. 下载和接入的固定流程

对每个数据集严格执行以下顺序；未通过前不得进入正式性能表。

1. **获取与留档**：仅从官方发布源下载；记录 URL、许可证、版本、SHA-256、原始/解压后字节数。
2. **canonical 化**：在独立 preprocessing 目录生成无重复自环、确定性对称化、CSR、`scale=1/sqrt(degree_offdiag+1)`、官方 split 的不可变快照。
3. **结构 gate**：断言节点数、边数、CSR 单调性、无原始自环、DGL 加自环后边数与度数关系、split 无交叠。
4. **真实图数值 gate**：每个目标层数至少一次完整 forward、logits、dW、db、隐藏层 dX 对 FP32 reference；禁止用合成图替代并标成数据集名。
5. **预实验**：1/8/32 线程、1--5 epoch，记录峰值 RSS、workspace reuse、计划 variant、fallback reason、DGL 线程与绑定。
6. **正式矩阵**：2/3 层 × 1/2/4/8/16/32 线程 × 200 epoch；每个 TFS cell 配同名 DGL cell；冷启动 wall 与稳态 epoch 分表。
7. **发布门槛**：全部 cell 成功且 200 行、合并器通过 provenance 校验、三次独立 32 线程重复报告中位数与区间。

## 6. 建议的论文主表结构

主表至少覆盖以下六类真实图：

1. 引文：`ogbn-arxiv`；
2. 商品：`ogbn-products`；
3. 社交高边密度：Reddit；
4. 图像/社交：Flickr；
5. 业务社交：Yelp（仅在任务契约匹配后）；
6. 百万节点、高输出宽度：IGB-HOM-small 19/2983；
7. 大图专项：AmazonProducts、IGB-HOM-medium 或 `ogbn-papers100M` 中至少两项通过准入。

论文中应把 IGB 的 19/2983 类明确标为“同图、不同输出维度的 shape stress”，并将大图专项与常规 200-epoch 主表分开，避免将无法公平完成的超大图混入平均数。

## 7. 资源与风险门槛

- `ogbn-papers100M` 的官方数据规模为 111M 节点、1.616B 边；它是专项扩展性实验，不是立即的 200-epoch 默认任务。
- IGB 的 `tiny/small/medium` 支持官方脚本下载与 MD5 校验；`large/full` 需 bash 下载，官方要求超过 500GB 磁盘。
- 单标签 accuracy、 多标签 ROC-AUC/BCE、异构关系分类必须各自成表，不能用同一个“平均准确率/加速比”混合。
- 不得按数据集名选择内核路径；数据集名只能位于 launcher 的路径映射。执行计划只可使用 N/K/D/NNZ、线程数、拓扑和显式预算。

## 8. 官方来源

- OGB node property prediction：<https://ogb.stanford.edu/docs/nodeprop/>
- OGB 数据下载索引：<https://snap.stanford.edu/ogb/data/nodeproppred/>
- IGB 官方仓库与下载说明：<https://github.com/IllinoisGraphBenchmark/IGB-Datasets>
- DGL 官方内置数据集清单：<https://www.dgl.ai/dgl_docs/en/2.3.x/features/dataset.html>
- DGL Yelp 数据集说明：<https://www.dgl.ai/dgl_docs/en/2.3.x/generated/dgl.data.YelpDataset.html>
- PyG 数据集规模清单：<https://pytorch-geometric.readthedocs.io/en/stable/cheatsheet/data_cheatsheet.html>

