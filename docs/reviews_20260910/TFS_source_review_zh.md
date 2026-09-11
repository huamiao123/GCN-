# TFS 训练运行时源码审查与研究建议

## 1. 审查范围与总判断

审查对象为 `huamiao123/GCN-` 的 `codex/tfs-integrated-optimizations-20260909` 分支，审查时可见最新提交为 `e399d4f`（Wire and validate supervision dense planner）。`main` 不能代表当前完整实现。实际训练调用链需要沿 `authority_model.py`、`authority_autograd.py`、`dimension_dispatch.py`、`highd_backward.py` 和 supervision 模块核对；不能把简化 reference `autograd.py` 当作 authority 实现。[1][2][3][4][5][6][7]

本报告基于公开源码静态审查、仓库证据记录和一手论文/官方文档。没有编译该扩展，没有在 AMX 服务器或真实图上复测性能。另附独立的小规模 CPU 参考检查，仅验证代数、BF16 舍入边界和 micro-panel CE 一阶梯度，不构成原生实现验收。

总体判断：原问题清单的大部分结构性诊断成立，但需要三个修正。第一，默认路径、实验候选与已测 shadow 路径必须分开。第二，减少物化、减少逻辑扫描和减少实际 DRAM 流量不是同一件事。第三，训练中的融合必须考虑反向消费者，不能照搬“前向消费完就丢弃”的推理逻辑。

最有价值的主线应是：**在目标函数、梯度依赖和数值契约约束下，联合决定计算域、保存状态、重算、稀疏方向、精度边界及内存生命周期。** AMX、AVX、selected CSR 和 overlap packing 是实现手段，而不是各自独立堆叠的论文贡献。

## 2. 对原 20 项判断的逐项核对

| 编号 | 判断 | 源码对应与重要限定 |
|---|---|---|
| 1 | 成立，且障碍比描述更硬 | `execution_plan.py` 不仅按 D/K 选序，`validate()` 还强制校验该顺序；静态 aggregate 候选受 K≤128、D≤128 约束。Wide-K 静态首层需要新合法候选和新数值合同，不能只放宽一个条件。[2] |
| 2 | 成立 | hidden 路径仍分开执行卷积、ReLU、Dropout；下一层再构造 Hs。更完整的方向是前向和反向边界同时融合。[3][6] |
| 3 | 对 V4 成立，不应泛化 | V4 的 dP 先从 BF16 matmul 结果升到 FP32，再调用 FP32 pull；部分 streamed transform 路径已经优先使用 BF16-input pull。[4][8] |
| 4 | 成立 | Aggregate native D-slab adapter 为每个 slab 调用完整 backward，再累加完整 part_dx。[4] |
| 5 | 成立 | grad 与 weight 的列 slab 调用 contiguous；grad 所有互不重叠 slab 合计至少复制一遍逻辑梯度，不应无依据说每个 slab 都复制整张梯度。[4] |
| 6 | 合理，尚不是成熟完成态 | Aggregate single-scan reference 仍保存全局 FP32 dP；native 化与真正的内存预算贯通尚需实现。[4] |
| 7 | 成立，但现有候选需更准确描述 | 已有 native transform single-scan 候选；其融合 pull helper 仍在 feature block 外循环内重访邻接表，不能直接理解成邻接索引只读一遍。[4][8] |
| 8 | 成立，收益有条件 | terminal 仍先构造 full Hs。应先测 U_M/N；按边现场转换可能将 O(NK) 转换变成 O(E_M K)，反而更差。[3] |
| 9 | 成立，terminal 最适合做 | selected pull 先生成 P[M,K]。因为 terminal loss 能立即产生梯度，可让 P panel 同时服务前向和 dW；隐藏层不能无条件照搬。[3] |
| 10 | 成立 | wide Gs 被物化后分别供 dW 和 Q 消费；应区分逻辑字节数与硬件 DRAM 字节数。[3] |
| 11 | 成立 | bounded terminal 的 Q 仍由 `torch.matmul` 计算。框架调用不等于没有 AMX 加速，原生版必须对照真实后端测量。[3][20] |
| 12 | 成立 | native logits、框架 CE、native scaling/dW 和框架 Q 尚未形成单次调用的 micro-engine。[3] |
| 13 | 成立 | compact logits 每次入口重新准备 W padding/packing。micro-panel 化前必须建立执行期共享 packing。[8] |
| 14 | 成立 | authority_active 回退创建 full G[N,D] 并散射监督梯度；应作为独立路径看待，不代表 bounded-Q 路径也这样做。[3] |
| 15 | 成立 | selected/rectangular 原生接口要求 int64。colidx 可按列域范围降为 int32，rowptr 则须按边数范围单独决定。[8] |
| 16 | 成立 | selected forward 经 row_ids 访问 full CSR。构造独立连续 selected CSR 有利于流式 panel 和格式验证摊销。[8] |
| 17 | 成立，必须有生命周期设计 | 一次 optimizer 更新前可以复用 W/Wt 及 packing；键不能只有地址，需版本、布局和精度等。[3][8][9] |
| 18 | 成立，需区分两种转换 | 大张量落地的往返可优先删；有些局部 BF16→FP32 是保留既有舍入语义，删除会改变数值合同。[8] |
| 19 | 基本合理，但缺少硬件计数证据 | 共享 source 的潜在重复访问成立；不能由 E/U 直接推出 DRAM 节省。当前已有 source-signature locality scheduler 研究实现，并非完全没有相关代码。[8] |
| 20 | 成立，但 authority 并非完全没建模 | authority 已记录 workspace、sparse scans 等字段；supervision dense planner 是另一套 M/K/D/T gate。问题是成本模型、语义字段与执行路径还未贯通，而不是“从零没有 planner”。[2][10] |

数量校正：N=1,000,000、K=128 的 FP32 张量为 512,000,000 B，即十进制 0.512 GB，不是单个就超过 1 GB。多份中间态或数百万节点时，GB 级判断成立。下文所有 GB 使用十进制；这些是按形状推导的字节量，不是测得的 RSS 或 DRAM 流量。

## 3. 清单之外的具体问题

### 3.1 Selected 与 rectangular pull 尚未真正做到 edge-single-scan

`v6_wide_aggregate_probe.cpp` 的 selected pull（约 1706–1764 行）和 rectangular pull（约 1769–1886 行）将 16-feature block 循环放在 edge 循环之外。K=128 对应 8 个 feature block，所以同一行的邻接索引被逻辑遍历 8 次。[8]

这不表示同一条边的整行特征被读 8 次：每次读取不同的特征分片。也不表示 DRAM 索引流量或延迟一定变成 8 倍，因为索引可能留在 cache。确定存在的是重复的索引加载、地址计算、循环控制，以及部分重复检查。

K=128 可以先采用“一行 8 个 FP32 向量累加器，edge 循环在外”的专用实现。这样保持每个特征的邻居求和顺序不变，数值风险较低。对 selected 和 transpose 两方向一起实现，并覆盖 K=47、尾块、空行和 self 项。这比先投入复杂 overlap packing 更适合作为直接基线修复。

Rectangular kernel 还在 feature-block 内对每条边检查 compact column 范围。若 scope 是一次验证后不可变的图计划，可在构建期验证，在热循环中使用已验证版本；保留独立的 checked/debug 入口，不能对任意外部 CSR 直接移除检查。[8]

### 3.2 Transform “single-scan” 的计数口径不完整

`pull_panel_scaled_grad_impl`（约 747–789 行）同样是 feature-block 外循环。它还在每次消费 source 时读取 FP32 grad、乘 scale 并做 BF16 舍入。因而“没有每个 macro D slab 重新启动整个 kernel”不等于“每条边索引只读一次”。[8]

该融合移除了全局 Gs，但把可预计算的 source-scale/conversion 转移到边消费端，复杂度可能由 O(ND) 变成 O((E+N)D)。这是一种存储与重算交换，不是免费的优化。高复用图、FP32 源特征的 cache-line 流量以及多 feature block 都可能抵消收益。

建议把 planner 的 scan 指标分成：macro sparse invocation 数、邻接索引逻辑访问数、source feature 标量消费数、估计 cache-line 缺失数、实测 DRAM/NUMA 字节数。“single-scan” 必须声明指哪一种。

### 3.3 原生 int32 索引缓存有失效与无界增长风险

`int32_colidx_workspace`（约 803–827 行）用原 int64 数组的裸地址和元素数查找静态 vector 中的缓存，没有保存源 tensor 的版本或所有权，也没有容量淘汰。[8]

据此可构造两类风险：同地址同长度的原地修改仍命中旧 int32 内容；释放后地址复用可能产生 ABA 命中。多图长寿命进程还会积累条目。对“进程内始终只有一张严格不变的图”未必触发，但作为通用 runtime，这是正确性问题，不只是性能问题。

Python 的 `PersistentHsCache` 已经有 tensor version、graph generation、签名和 weakref 身份保护，不能把这层成熟缓存也说成只有地址键。[9] 应将原生 CSR 副本纳入统一的不可变 GraphHandle：构建时验证，持有源存储与 generation，管理内存预算，执行期间持有有效 lease。若引入 LRU，必须避免淘汰正在使用的指针。

### 3.4 Planner 的 V4 sparse operand 元数据与执行不符

`_sparse_backward_contract()` 把 `aggregate_saved_v4` 与普通 C3 放在同一分支，记录为 Gs、宽度 D；但 V4 实际先计算 dP，再沿图拉取 K 维 dP。[2][8]

这至少会污染日志、解释和后续成本建模。K≠D 时还会误报宽度；K=D 时宽度巧合相同，数据流标签仍然不正确。现阶段不能直接据此断言数值结果错误，因为 kernel 并不必然使用这个元数据执行。

应新增“计划—执行一致性测试”：核对实际 sparse operand、逻辑/物理宽度、panel/slab、是否保存中间态、执行的索引循环层数和真实申请字节数。

### 3.5 缩小 Python panel 会放大 dW 清零与归约成本

compact dW 已复用 workspace，不应误报成每次都重新分配其大缓冲区；但每次外部调用仍会清零线程私有 dW、执行归约并返回结果。它内部还有自己的 512-row panel。[8]

因此，把外部 row_tile 从 300000 改为 32，并不等于得到 cache-local engine，可能只是让固定成本执行上万次。解决办法应是一次 terminal 执行拥有共享 W/Wt packing、线程私有 dW/db 和 scratch，内部迭代 micro-panel，最后统一归约。

### 3.6 Scope、预算和 Autograd 契约需要收紧

Supervision scope 保存固定监督行与转置结构，但调用处主要检查节点数。相同 N 并不能证明 graph、mask、normalization 和线程调度仍匹配。[3] 应绑定 graph generation、监督集合身份与 schedule 的线程数；张量字段即使处于 frozen dataclass 内也仍可能被原地修改。

High-D 的 Python plan 与部分 native 环境变量控制的 panel 入口需要继续做逐接口一致性审计：可见调用链并非所有入口都显式接收 plan.panel。这是需验证的契约风险，不能仅凭静态阅读断言所有预算配置已经失效。[4][5][8]

当前 fused CE 在 forward 内预先形成一阶参数梯度和 Q。它应明确限定支持的 CE 语义与微分阶数，不应默认承诺任意 class weights、label smoothing、ignore_index、soft targets 或二阶梯度。对不需要的输入梯度，也应显式利用需求信息跳过工作。[3]

## 4. Wide-K 静态首层：应优先做，但需要独立契约

用 B=A_off+I 表示带隐式自环的二值邻接，R=diag(s) 表示归一化因子，S=RBR。理想实数计算为 Y=SXW+b。固定 X 和 S 时，P0=SX 可以缓存，从而前向只做 P0W，参数梯度只需 P0ᵀG。

但源码实现有明确的 BF16 舍入边界，例如先形成 Hs=Q_BF16(RX)，再聚合并按既有规则舍入，最后进行 dense 和输出 scale。实际应缓存与这些边界一致的 T0/P0，而不是用任意 FP32 SpMM 结果替换。[8][11]

Aggregate-first 与 Transform-first 在实数上等价，却通常不在 BF16 上逐位等价。独立参考检查在 31×16→7 的小例子中：FP64 重关联最大差约 8.88e-16；刻意简化的 BF16 边界例子出现约 1.18e-2 的最大差。这不是 TFS 误差测量，只说明“代数相等”不能替代精度验收。

缓存成立需要：输入特征、图结构、归一化稳定；不存在改变输入/邻接的随机增强、input dropout 或可训练结构；或者这些变更能正确触发重建。无需 dX 是简化首层训练的重要条件，但不应作为“只要 false 就一定可缓存”的充分条件。

成本必须同时考虑冷启动与常驻内存。N=1,000,000、K=1024 的单份 BF16 缓存就是 2.048 GB。需要检查是否仍持有 X、Hs、padded copy、NUMA replica，以及 evaluation 是否复用。合理的启用条件是：

`预计复用步数 × 每步净节省 > 构建代价 + 首次布局/分配代价`，且峰值 live bytes 不超预算。

工程上建议新增显式 `static_input_aggregate` 家族，让 D<K 的形状合法进入候选集；将旧 D/K 规则保留为普通动态输入的一个候选先验。先验证 1024→128，再扩到其他 K、D 尾块。该点主要是高价值部分求值/常量预计算，不适合作为独立首创叙事。

## 5. Whole-terminal micro-engine：当前最合适的强融合边界

### 5.1 三十万行不是 cache-local panel

R=300000、D=2983 时，BF16 Gs 为 1.7898 GB，FP32 logits 为 3.5796 GB，两者合计 5.3694 GB。仅 Gs 写一次、dW 读一次、Q 再读一次，逻辑访问量就是 5.3694 GB；还未计转置、padding 和其他中间态。

即使 R=16，FP32 logits+BF16 Gs 也有 286368 B，约 279.7 KiB。因此 D=2983 的整体 terminal micro-panel 更适合描述为 L2/scratch-resident，而非必然 L1-resident。P[16,128] 的 BF16 4 KiB 只是其中一个局部张量，不代表整个工作集。

### 5.2 推荐执行结构

一次 terminal session 负责准备 W/Wt 的 BF16 与 AMX packing、初始化线程私有 dW/db、分配固定 scratch。随后每个 worker 执行：

`selected pull(P_micro) → AMX logits → stable exact CE → Gs_micro → {dW update, native Q} → 释放 P/logits/Gs micro-buffer`。

最后统一归约 dW/db；Q[M,K] 可以暂时保留，用成熟的 rectangular pull 计算 dH。第一版不必消灭 Q：它是窄张量，保留它可能比原子 push、反复累加 full dH 或重访 CSR 更划算。

Native Q 应首先独立 A/B，包含 Wt packing、尾块、输出舍入及调用成本。PyTorch CPU BF16 dense 算子可以借助 oneDNN 和 AMX，因此不能用“framework→native”推导必然加速。[20] 更强的收益来自共享布局、消费刚产生的 Gs、减小落地流量及控制线程归约。

### 5.3 Exact CE 的全类别依赖不能跳过

对每一行，CE 梯度需要所有 D 类的归一化分母。一个 class tile 刚算完时，不能在未知完整分母的情况下最终提交该 tile 的 Gs。

D=2983 可以优先在小 row micro-panel 中保存完整类别维，再做稳定 logsumexp/softmax；更宽 D 再考虑 online logsumexp 加第二次 logits 计算，或显式保存受控的类别分片。在线归约解决分母存储，并不自动让 dW/Q 无需重算、同步或状态。

“精确”建议定义为不裁剪监督节点、边或类别梯度，使用稳定的全类别 CE；不应把不同浮点求和序、BF16 舍入点也宣称为逐位一致。全新 exp/归约实现需要数值和收敛测试。

### 5.4 U_M 优化应避免转换重算陷阱

先构建被监督目标访问的 unique-source 集，并测量 U_M/N。若 U_M 接近 N，compact Hs 的 remap、间接访问和构建成本可能没有意义。若显著较小，可构造 Hs_compact[U_M,K] 和 compact selected-forward CSR。

不建议默认在每条 selected edge 上直接从 FP32 hidden 转换。重复邻居会重复转换；更合理的是以 unique source 为单位，或把转换和具有显式复用的 source packing 结合。

## 6. 隐藏层融合：从单向转换优化扩展到双向接口

前向目标是直接从卷积 FP32 accumulator 生成经过 ReLU、Dropout、下一层输入 scale 的 BF16 Hs。单独写一个融合扫描 kernel 能减少边界遍历；真正避免 FP32 conv output 落地，需要将 epilogue 与下一层输入表示设计连起来。

反向同样存在可融合链：`dH → dropout backward → ReLU backward → 上一层 Gs 的 scale/BF16 conversion + db reduction`。应考虑把这两条链作为一个训练块内部的接口，而不是继续依赖每个独立模块之间完整 FP32 tensor 的协议。

必须保留足够反向状态。ReLU mask 可压为位，dropout 可保存 mask 或使用基于 global node/channel/layer/step 的可复现 counter RNG。相同 seed 不保证节点重排或线程调整后每个元素得到同一个 dropout 样本；这会污染 A/B 数值对照。

是否能删除某个 FP32 activation，取决于所有消费者：残差连接、辅助 loss、外部 hook、后续统计以及所选 backward 可能仍需要它。更合理的合同是“避免无消费者的 FP32 激活副本”，不是强迫整个框架只接受 BF16。

## 7. 训练期 tile-local fusion 的关键边界

对隐藏层 P=AH、Y=PW，前向立即消费 P 后，反向仍需要 dW=PᵀG。G 要等下游所有层及 loss 完成后才出现。

删除全局 P 至少需要以下之一：保存替代状态；在 backward 重算 P；使用 dW=Hᵀ(AᵀG) 的另一条梯度数据流。后两种分别增加一次 K 维稀疏工作或引入 D 维 sparse backward，并可能改变 BF16 数值边界。不同 authority 变体原本保存的状态也不同，必须相对于实际 baseline 计账。[6][8]

这解释了三个不同最优点：

| 层类型 | 有希望的状态策略 | 原因 |
|---|---|---|
| 静态首层 | 长期保存窄/宽的固定 P0 | 跨 epoch 重用，故意物化反而是优化 |
| 普通隐藏层 | 保存 P 或 Hs，或受控重算 | 梯度有跨层时间依赖，需联合比较前后向 |
| Loss-coupled terminal | P_micro 即产即用，不保留全局 P | CE 梯度可在同一个 micro-panel 内产生，前向和 dW 消费者同时就绪 |

因此，“register-resident，否则 L1，但永不 global”不是通用训练原则。正确原则是：**避免无收益的全局物化；对能减少重算、重复图遍历或同步的窄状态，允许有计划地保存。**

## 8. Native single-scan D-slab 的可实现设计

Aggregate backward 在实数上有 dP=Σ_s Gs_s W_sᵀ，再执行 AᵀdP。推荐以 row-panel 为外循环，在 panel 内遍历所有 D slab，使用 FP32 dP accumulator，最后一次 BF16 rounding 写入全局窄 dP。所有 panel 完成后，只对该 dP 进行一次完整的 transpose pull。

临时 dP_acc 约为 4RK，全局 BF16 dP 为 2NK；另有最终 FP32 dX=4NK。后者本来就是输出，但仍须计入峰值 live bytes。这样可以不用全局 FP32 dP，却不是“完全不存在 N×K 中间态”。

该设计还没解决所有 dW 问题。线程私有 FP32 dW 约为 4TK_pD_p；可能需要 D ownership、分组线程或受控归约，且不能为节省 dW 又在外层重新启动图遍历。所有梯度矩阵、常驻缓存、packing、scratch、归约副本都必须纳入同一个 live-buffer 预算。

grad 与 W 的 slab 接口应传 base pointer、leading dimension、d0、有效宽度，让 packing/scaling 直接处理非连续列片，避免大 FP32 contiguous 临时张量。不能只删 contiguous 而把不连续输入交给只支持紧致行跨度的 kernel。

精度方面，`BF16(Σ_s FP32_dot_s)` 通常不同于逐 slab 舍入后累加。新 native 路径要声明舍入合同，并测试不同 slab partition 的稳定性。

Transform-High-D 是不同问题：其 sparse 输出宽度是 D。要在不重复 edge 索引循环的同时保留完整 D accumulator，需要足够寄存器或 scratch；D 很宽时会有 spill/缓存权衡。因此不应承诺一个 layout 同时实现任意宽 D、零重算、零全局中间态和严格小 workspace。

## 9. Overlap-aware packing 的合理性、陷阱与文献边界

### 9.1 指标应解释为复用机会，而不是现成加速比

对 16-row batch 定义 E_B 为真实边/贡献数，U_B 为 unique-source 数。更易解释的指标是 r_B=E_B/U_B，理想 source-row 加载消除比例为 1−U_B/E_B。原定义 ρ_B=E_B/(16U_B)=r_B/16，是二值 incidence 的密度；在无重复边且 16 行都定义完整的条件下取值在 [1/16,1]。

如果使用 dense AMX incidence tile，必须计入 reduction 维 padding。一个满宽 BF16 reduction-block 模型可写为 ρ_eff=E_B/[16·32·ceil(U_B/32)]；实际尾块可变时应使用真正的物理 tile 数。稀疏算术与 padded dense 算术的比值大致受到 1/ρ_eff 影响，所以“复用高”不自动代表 dense AMX 比 AVX masked aggregation 好。

η=E_B/(16 maxdegree) 衡量的是按最长邻居行补齐的负载利用率。若新 kernel 已显式枚举 unique sources 和有效 mask，η 不再必然是与原实现同权重的优化目标。应从真实执行家族推导目标函数。

### 9.2 AMX 的 16 行不是稀疏分组的硬性规定

Intel AMX palette 1 提供 8 个 tile，每个最大 16×64 bytes，实际 rows/columns 可配置。[19] 可以把 sparse micro-group 设为 4/8/16 行，用更大 source-reuse superpanel 再向 16-row GEMM 提供数据。

P[16,128] 的 BF16 存储确为 4 KiB，但 sparse FP32 accum 为 8 KiB。若试图用 AVX-512 为 16 个目的行同时维持全部 K=128 特征，每行需要 8 个 16-float vector accumulator，合计 128 个，寄存器压力明显过高。必须设计行/特征阻塞或 scratch，而不是假设一次 gather 后所有目的行 accumulator 都免费常驻。

16-bit mask 可以描述当前二值去重邻接的目的行集合。一般加权边、重复多重边或注意力权重还需存值/计数；仅有 mask 不足以表达。当前 RBR 的可分离归一化可以由 source/output scale 处理，但要保留 implicit self 语义。[11]

### 9.3 从加载指令到 DRAM 节省有一段距离

原 CSR 对同一 source 的多次加载可能命中 L1/L2/LLC，尤其是 hub。union 格式还会引入 source ID、mask、padding、解码和重排。若 baseline 已命中 cache，减少 source load 指令不一定减少同等 DRAM 流量。

建议成本函数同时估计 source cache-line 缺失、NUMA 远端成本、metadata、mask 解码、累加器 spill、packing 及 dense 消费；不要只用 E/U 排序。预处理应在受限的 degree/locality 候选窗口中做近似匹配，例如签名或 MinHash 辅助局部贪心，避免全图两两比较。必须报告冷启动成本和预计 amortization。

局部复用与中间态融合的收益不能默认相乘：它们可能作用于同一瓶颈，且融合改变 cache 状态。需要“无复用/无融合、仅复用、仅融合、两者”四格消融。

### 9.4 相关一手研究已经覆盖哪些部分

| 工作 | 与本项目相近的内容 | 仍可能区分的内容 |
|---|---|---|
| HAG，Redundancy-Free Computation Graphs for GNNs | 共享邻居冗余、减少聚合与数据传输。[14] | mask source-load reuse 与层次化公共子聚合不同，但不能把消除共同邻居冗余本身称为首创 |
| TC-GNN，USENIX ATC 2023 | row window 中排序/去重邻居、condense unique sources，映射 Tensor Core block。[15] | CPU AMX/AVX 的成本与训练生命周期联合设计，而不是“16 行+unique source” |
| FeatGraph，SC 2020 | 图遍历与 feature-dimension schedule 的协同优化。[16] | 具体梯度状态选择、目标范围、精度与 NUMA 约束 |
| Cut Your Losses / CCE，2024 | classifier、CE 的少物化，去除无 loss 行；提供无 gradient-filtering 对照。[17] | CPU AMX、selected sparse producer、dW/Q 双消费者的执行期联合调度 |
| On Efficient Scaling of GNNs via IO-Aware Layers Implementations，2026-05 预印本 | WSB 明确使用 16 行窗口、unique sources、bitmap 与矩阵块；研究 IO-aware fusion 和 kernel-dependent reordering。[18] | 不能再把这些布局元素单独当新点，需证明训练状态/精度/NUMA 联合方案的新增收益 |

这不是穷尽的新颖性检索，不能据此保证“首次”。但现有文献已足以说明：仅换成 AMX 并命名为 Overlap-Aware Tile Packing，创新性论证不充分。

## 10. 推荐收敛的创新点

### 10.1 主贡献：Derivative-aware、lifetime-aware GCN dataflow planner

将静态首层、普通隐藏层、监督 terminal 作为不同依赖域；以真实梯度消费者决定保存 P、保存 Hs、重算、改写 sparse backward，或在 loss 边界消除中间态。优化目标应是整个 train step 加冷启动摊销，而不是单个 forward kernel。

可形式化为：在峰值 live bytes、精度合同和可复现性约束下，最小化执行时间与构建摊销。输入包含 N、M、E、E_M、U_M、K、D、线程与 NUMA、复用次数；路径代价包含实际扫描、转换、packing、归约、索引、临时张量与保留状态。

为了具有研究说服力，需要展示“局部最快前向为何会导致更慢训练步”“完全消除 P 为何有时输给保存 BF16 P”“同一 D/K 在不同监督率、内存预算或图复用下为何选不同路径”。这比增加一套经验阈值更像实质性系统贡献。

### 10.2 最强实现贡献：Loss-coupled terminal 双消费者引擎

将 compact sparse producer、全类别 CE、dW 和 Q 共同调度；一次执行期共享 packing 和私有归约状态；以 micro-panel 临时保存宽 logits/Gs，仅保留必要窄 Q。创新论证聚焦该联合依赖图，而不是“把 CE 或 GEMM 用 C++ 重写”。

### 10.3 跨层贡献：双向精度边界协议

围绕 Hs 与 Gs 建立内部表示合同，将前向 activation/dropout 与后向 activation-gradient/scale/db 融合，记录 RNG、mask、舍入与版本。通过 L3/L4/更深模型证明收益随边界数量增长，并量化减少的 FP32 全图扫描。

### 10.4 可选探索：Topology×shape 自适应稀疏执行家族

保留 CSR edge-single-scan AVX、union-mask AVX、高密度 incidence AMX 三个家族。按真实复用、padding 和缓存/NUMA 成本选择，而不是全图统一一个 AMX 格式。先统计实际图的 r_B/ρ_eff 分布和热冷 source，再决定是否值得开发。

### 10.5 额外研究方向：监督依赖域向隐藏层扩展

设最后一层所需输出集合为训练节点 R_L=M，向前递推 R_{l−1}=union_{v∈R_l}(N(v)∪{v})。若只存在局部消息传递和逐点激活，使用原图归一化、保留所有依赖边，并保持相同的全局 dropout 样本，这个依赖域可以给出同一训练目标，而不是邻居采样近似。

但该基本原理与 DGL 的全邻居多层 dependency block 相通，不能称为首次。[21] 潜在新增价值是为固定 full-batch 监督集合预编译这些域，并与 AMX 布局、静态缓存和 backward 状态联合规划。高连通图中域可能很快接近全 N；BatchNorm 等跨节点统计、全局 loss 或动态图则需要扩展依赖或禁用裁剪。

## 11. 实施顺序与验收

| 阶段 | 工作 | 首要验收标准 |
|---|---|---|
| P0 | 修正 native int32 cache、V4 元数据、scope/执行合同；selected/rect K=128 真 edge-single-scan | 不同 graph/version、线程、尾块、self、空行正确；计划与执行一致；保持求和顺序 |
| P1 | Native Q 独立对照；一次执行期 W/Wt packing；Wide-K 静态缓存原型 | 含准备成本的真实 train-step 改善；冷启动摊销；既定精度与收敛门槛 |
| P2 | whole-terminal micro-engine 与持久 dW；双向 hidden boundary fusion | 宽 logits/Gs 无全局物化；多层模型扫描减少；无 per-micro-panel 全量归约 |
| P3 | Native aggregate single-scan D-slab；统一 live-byte 预算 | 限制 workspace 时不重复 sparse pull；不生成每 slab full part_dx；数值随切分稳定 |
| P4 | overlap 家族及成本模型；监督依赖域扩展 | 在实际图上证明收益分布、预处理摊销与回退边界，而非只展示高 overlap 特例 |

这不是对所有模型固定不变的收益排序。宽 terminal 的 L2/L3 优先 Q/micro-engine；低类别数深层模型更值得优先隐藏层边界和 sparse locality；不常触发 D-slab 的 workload 不应把它当首要投入。

实验需要覆盖真实图与控制变量图。形状至少包含 K=64/128/256/1024，D=19/47/128/129/512/2983，线程 1/2/4/8/16/32，L2/L3/L4，不同监督率、U_M/N、缓存预算和 D-slab 切分。控制变量图应在类似 N/E/degree 下改变 neighbor overlap，避免把高复用和高密度效果混淆。

性能报告分别给出冷启动、稳态 step、evaluation、time-to-target-quality、峰值 live bytes/RSS。采用同机交错 A/B，报告分布而非单个最佳值。硬件计数重点是 DRAM 字节、本地/远端流量、LLC/DTLB、内存停顿、AMX 活跃及 packing/reduction 开销。

数值验收应同时对照既定 mixed-precision baseline 与 FP32 reference，记录 loss、dH/dW/db 的绝对和相对误差，覆盖极小梯度、尾块、空邻居、dropout mask、不同线程和 optimizer 更新后的缓存失效。逐位一致、容差一致和收敛一致是三种不同承诺。

## 12. 对当前证据的准确表述

仓库 2026-09-10 验收记录报告 24-cell 同节点交错 A/B 全部通过其正确性门槛，训练步加速几何均值 1.5848×；IGB-small D=2983 的 L2/L3 分别为约 2.2406×/2.0329×，Products D=47 的 L2/L3 分别约 1.2765×/1.0850×；并有两组 200-epoch paired 收敛检查。[12]

这些数字是仓库报告，未在本审查中复现。基线是冻结 authority TFS，不是新的 DGL-Mixed 同口径矩阵。不同实验的加速比不能连乘，也不能把通过两个收敛配置推广为任意模型/监督率/CPU 上的保证。

当前最稳妥的定位是：已有可信的 supervision-scoped shadow 进展；下一步通过 terminal 双消费者引擎、静态输入部分求值、双向精度接口与梯度生命周期规划，减少无效物化和重复扫描。Overlap 应保留为有明确成本模型和回退路径的研究支线。

## 13. 参考检查与源码定位索引

附件 `reference_checks.py` 不依赖该仓库，执行两个参考检查：FP64 与简化 BF16 的重关联比较；FP64 full CE/autograd 与 micro-panel 解析梯度比较。结果文件 `reference_checks_results.json` 记录本环境输出。micro-panel 的 dP/dW/db 最大绝对差约为 2.8e-17 / 4.2e-17 / 2.8e-17；这是算法参考，不证明原生精度或速度。

| 位置 | 核查内容 |
|---|---|
| `execution_plan.py:216–226` | V4 sparse backward operand 元数据 |
| `execution_plan.py:380–382, 645–661` | D/K 硬性 order 合同和静态 cache gate |
| `supervision_scope.py:123–128` | authority_active full G 回退 |
| `supervision_scope.py:172–225` | full Hs/P、bounded CE、dW、框架 Q |
| `supervision_scope.py:231–243` | 一阶 backward 返回协议 |
| `highd_backward.py:286–399` | streamed aggregate 与全局 FP32 dP |
| `highd_backward.py:411–507` | transform D-slab 与 native single-scan 候选 |
| `highd_backward.py:706–815` | aggregate slab 与 reference single-scan |
| `v6_wide_aggregate_probe.cpp:747–827` | fused source conversion 与 int32 cache |
| `v6_wide_aggregate_probe.cpp:1706–1886` | selected/rectangular loops |
| `v6_wide_aggregate_probe.cpp:2595–2803` | dW session 状态与 logits packing |
| `v6_wide_aggregate_probe.cpp:2810–2871` | V4 dP dtype 往返 |
| `persistent_hs_cache.py:48–112` | 已存在的版本与 ABA 保护 |

行号对应审查时分支快照，后续修改可能移动。

## 来源

[1] huamiao123/GCN-. Commit e399d4f, “Wire and validate supervision dense planner.” https://github.com/huamiao123/GCN-/commit/e399d4f

[2] `python/tfs_train/execution_plan.py`. https://github.com/huamiao123/GCN-/blob/e399d4f/python/tfs_train/execution_plan.py

[3] `python/tfs_train/supervision_scope.py`. https://github.com/huamiao123/GCN-/blob/e399d4f/python/tfs_train/supervision_scope.py

[4] `python/tfs_train/highd_backward.py`. https://github.com/huamiao123/GCN-/blob/e399d4f/python/tfs_train/highd_backward.py

[5] `python/tfs_train/dimension_dispatch.py`. https://github.com/huamiao123/GCN-/blob/e399d4f/python/tfs_train/dimension_dispatch.py

[6] `python/tfs_train/authority_model.py`. https://github.com/huamiao123/GCN-/blob/e399d4f/python/tfs_train/authority_model.py

[7] `python/tfs_train/authority_autograd.py`. https://github.com/huamiao123/GCN-/blob/e399d4f/python/tfs_train/authority_autograd.py

[8] `csrc/experiments/backward_opt_20260814/v6_wide_aggregate_probe.cpp`. https://github.com/huamiao123/GCN-/blob/e399d4f/csrc/experiments/backward_opt_20260814/v6_wide_aggregate_probe.cpp

[9] `python/tfs_train/persistent_hs_cache.py`. https://github.com/huamiao123/GCN-/blob/e399d4f/python/tfs_train/persistent_hs_cache.py

[10] `python/tfs_train/supervision_dense_plan.py`. https://github.com/huamiao123/GCN-/blob/e399d4f/python/tfs_train/supervision_dense_plan.py

[11] `python/tfs_train/graph.py`. https://github.com/huamiao123/GCN-/blob/e399d4f/python/tfs_train/graph.py

[12] `docs/PLANNED_SHADOW_ACCEPTANCE_20260910.md`. https://github.com/huamiao123/GCN-/blob/e399d4f/docs/PLANNED_SHADOW_ACCEPTANCE_20260910.md

[13] `INTEGRATED_OPTIMIZATIONS_20260909.md`. https://github.com/huamiao123/GCN-/blob/e399d4f/INTEGRATED_OPTIMIZATIONS_20260909.md

[14] Jia, Z. et al. Redundancy-Free Computation Graphs for Graph Neural Networks. arXiv:1906.03707. https://arxiv.org/abs/1906.03707

[15] Wang, Y. et al. TC-GNN: Bridging Sparse GNN Computation and Dense Tensor Cores on GPUs. USENIX ATC 2023, Section 4.1 and Figure 4. https://www.usenix.org/conference/atc23/presentation/wang-yuke ; https://www.usenix.org/system/files/atc23-wang-yuke.pdf

[16] Hu, Y. et al. FeatGraph: A Flexible and Efficient Backend for Graph Neural Network Systems. SC 2020 / arXiv:2008.11359. https://arxiv.org/abs/2008.11359

[17] Cut Your Losses in Large-Vocabulary Language Models. arXiv:2411.09009, 2024. https://arxiv.org/html/2411.09009v1

[18] Fomina, D. et al. On Efficient Scaling of GNNs via IO-Aware Layers Implementations. arXiv:2605.31500v1, 29 May 2026, Appendix D.3. https://arxiv.org/html/2605.31500

[19] Intel. Code Sample: Intel Advanced Matrix Extensions (Intel AMX) – Intrinsics Functions. https://www.intel.com/content/www/us/en/developer/articles/code-sample/advanced-matrix-extensions-intrinsics-functions.html

[20] PyTorch. Empowering PyTorch on Intel Xeon Scalable processors with Bfloat16. https://pytorch.org/blog/empowering-pytorch-on-intel-xeon-scalable-processors-with-bfloat16/

[21] DGL. MultiLayerFullNeighborSampler documentation. https://www.dgl.ai/dgl_docs/generated/dgl.dataloading.MultiLayerFullNeighborSampler.html
