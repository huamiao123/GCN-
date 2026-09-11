# TFS 全批量 GCN 训练的学术创新候选与否定性分析

## 结论

**截至本次调研，尚不能确认已经找到一个同时满足“高新颖度、适合当前实现、预期收益充分、没有直接先例”的成熟论文主贡献。** 但进一步推导后，有两个比“融合更多 kernel”更值得验证的研究候选：

**候选 A：以损失为闭合边界的原值—伴随值联合流式执行。** 利用末层损失可按监督节点分解的性质，把同一个 source 的前向特征使用与反向梯度完成安排在同一个编译后的生命周期中。研究对象不再只是 logits/Gs 的 micro-panel，而是 selected sparse forward、分类器、CE、两个 dense gradient 和 sparse adjoint 组成的完整局部训练环。

**候选 B：共享 source 成本耦合的同层混合算子顺序。** 同一层不同目标节点组分别选择 Aggregate-first 或 Transform-first，但不独立打分；显式考虑一个 source 的变换可以被多个选择 Transform-first 的目标组共享，并将一个受限成本模型精确归约为最小割。

这两项都还只是**研究假设**。A 更贴近当前宽末层；B 的可解释性和数学结构较清楚，但适用范围明显更窄。特别是，本文推导出了 B 的一个否定结论：在全节点输出、带自环、统一算术成本的模型下，混合顺序不可能优于最佳全局顺序。因此，不能把 B 宣传成所有 GCN 层都适用的普遍优化。

本文还纠正上一轮建议中的一个重要问题：**“联合选择计算顺序、缓存和反向路径”的大方向，与 Cached Operator Reordering 已有直接重合；“生命周期感知 planner”本身也不能脱离已有重计算调度研究而声称高新颖度。**[^R01][^R02]

文中严格区分四类内容：已核对的论文/源码事实；本文给出的数学推导；尚待验证的设计假设；实际运行的小规模参考检查。没有把合成成本当作运行时间，没有把 FP64 参考当作 BF16 原生验收，也没有把本次没有检索到直接先例当作首创证明。

---

## 1. 研究范围、约束与判断标准

### 1.1 对接的实现边界

本次以仓库提交 `e399d4f` 为固定源码锚点，而不是假定它永远是最新版本。该快照的 supervision 路径已经提供 selected scope、bounded CE、native logits 和 compact dW 等组件；可见调用链仍分阶段处理 Hs、P、Gs、Q 和矩形稀疏反向。因此，新研究应建立在这些组件之上，而不是重新实现一套弱基线来制造收益。[^C01][^C02]

本报告没有在 Intel AMX 服务器上编译或运行该仓库，没有获得真实 IGB/Products 图文件，也没有重新验证仓库记录的训练加速比。当前可交付的实证是小规模 CPU 数学参考与穷举检查。

### 1.2 保持不变的任务

默认研究对象是固定图结构、固定当前监督集合、全批量更新的 GCN 训练。末层使用按监督节点可分解的标准 softmax cross-entropy；一次训练步内 W 不变。允许重新安排等价计算，但不默认允许邻居采样、类别采样、删小梯度、改变 dropout 分布或提前更新部分权重。

“精确”必须分成三层：

| 层次 | 含义 | 本文状态 |
|---|---|---|
| 数学等价 | 在实数运算中得到同一目标和一阶梯度 | 为两个候选给出公式与条件 |
| 数值合同兼容 | 保持规定的归一化、量化位置、精度与误差容忍 | 原生实现尚需验证 |
| 逐位一致 | 与指定基线所有浮点结果逐位相同 | 不作承诺；重关联/归约顺序通常需要特别处理 |

例如，实数上的 `(S H) W = S (H W)`，不意味着两种 BF16 计算图相同。节点重排后使用同一个随机 seed，也不自动意味着每个原始 node/channel 获得同一个 dropout mask。研究必须将这些作为合同，而不是最后用一条准确率曲线掩盖差异。

### 1.3 怎样才算学术贡献

不以新缩写、代码量、AMX 指令数量或单个 kernel 的加速比判断新颖度。更强的候选应至少回答：究竟改变了什么执行机制；相比最接近的工作，多出了哪个不可省略的决策或数据依赖；为什么某些实例获益、另一些实例不获益；以及用什么实验可以直接推翻该机制的解释。

因此，本文优先保留可以写出模型、推导适用条件并构造反例的想法。已有优化仍应推进，但应作为系统基础设施或强基线，不应强行包装成独立首创。

---

## 2. 相关工作碰撞：哪些方向不宜再作为独立高新颖度主张

| 已有工作 | 已核对的直接相关内容 | 对当前项目的影响 |
|---|---|---|
| Cached Operator Reordering，2023 | 联合分析 GCN 前后向算子顺序、共享反向传播和中间态缓存 | 上一轮“保存 P/H 与梯度路径联合选择”的宽泛主张重合明显。[^R01] |
| Checkmate，2019/2020 | 根据计算图和成本，优化训练中的保存与重计算 | 一般性的 lifetime/rematerialization planner 不是新问题。[^R02] |
| TC-GNN，ATC 2023 | row window 内邻居去重、unique-source 压缩、Tensor Core 映射 | “16 行邻居 union + 矩阵硬件”不能作为首创表述。[^R03] |
| HAG，2019 | 消除共同邻居引起的重复聚合，构造共享中间计算 | “找公共邻居减少重复加法”已有直接先例。[^R04] |
| CBM，IPDPS 2025 | 用计算友好的矩阵压缩与带符号行差异复用相似邻域 | 新提出邻接行差分树/MST 复用时，必须先与其比较。[^R05] |
| FeatGraph，SC 2020 | 联合安排图遍历和特征维执行 | topology×shape 的一般说法不够，需新的模型或机制。[^R06] |
| Cut Your Losses / CCE，2024 | 避免完整大词表 logits，融合分类与归一化，并研究 backward | “大分类器 CE 少物化”已有强相关工作。[^R07] |
| Tile Fusion，2024 | 在共享内存多核上，利用稀疏依赖构造局部 fused tile | “利用静态图把 sparse/dense 局部融合”本身不足。[^R08] |
| SAR，2021/2022 | 全批量 GNN 的顺序聚合和分段反向重物化 | “分块前后向、随用随释放”的宽泛概念已有先例。[^R09] |
| DGL 全邻居多层 dependency block | 从输出 seed 反推每层必需的完整邻居依赖 | 全网络监督依赖域裁剪的基本计算依赖不是首创。[^R12] |

此外，2026 年的 IO-aware GNN 预印本已经描述 16-row、unique-source 和 bitmap 风格的格式。这里只用它确认相近机制仍在被研究，不把其未经独立复核的理论和性能断言当作本项目证据。[^R13]

### 2.1 对原有主要方案重新定位

Native Q、whole-terminal micro-engine、Wide-K 首层缓存、inter-layer fusion、native single-scan D-slab 都有工程价值，而且本来就是现有项目提出的方向。它们可以组成一个很强的系统，但单独将“框架算子改成 native”“静态输入预计算”“把中间张量缩成 panel”称为高新颖度，证据不足。

Overlap-aware packing 也不能只靠“GPU 上做过，但这里是 CPU AMX”成立。CPU 的缓存、NUMA、向量累加器和线程调度差异可能带来研究问题，但必须产生新的决策机制、受限最优性结果或可解释的执行方法，而不是仅仅换硬件。

### 2.2 本文保留的差异

A 试图改变的是：**一个 source 的正向读取和反向完成不再属于两次独立的全图执行，而成为同一条编译生命周期的两个端点。**

B 试图改变的是：**算子顺序不是一层一个枚举值，也不是各行独立选择，而是由拓扑共享成本耦合的一组选择。**

这两个差异比一般的“更智能 planner”具体，但尚不足以自动构成高新颖度。后文分别说明相关工作可能如何覆盖它们，以及必须补足的证据。

---

## 3. 候选 A：损失闭合的原值—伴随值联合流式执行

### 3.1 与现有 micro-engine 的本质区别

现有主要目标可以写成：

```text
selected sparse pull → P panel
→ logits → CE → Gs → dW + Q
→ 后续独立的 transpose sparse pull
```

即使宽 logits/Gs 已经局部化，最后的 `SᵀQ` 仍可能是一条独立数据流。本文候选不是再次提出这个 micro-engine，而是研究能否继续把左边的 sparse producer 和右边的 sparse adjoint 合在一个有界、可编译的训练单元内。

直观上，每个 source u 有两个需求：前向在若干目标行中需要 `H[u]`；反向要收集这些目标行产生的所有 `Q[i]` 贡献。若处理顺序已知，则 `H[u]` 第一次被使用到 `dH[u]` 最后一个贡献到达之间，可以被表示成一个确定的存活区间。

**这里的“伴随值”就是反向梯度累加器，不是另一种模型参数或近似梯度。**

### 3.2 数学对象与闭合条件

用固定的矩形稀疏算子

$$
S\in\mathbb R^{M\times N}
$$

表示所选监督目标对全部输入 source 的依赖。归一化、自环和边权均属于 S 的既定语义。令

$$
P=SH,\qquad Z=PW+\mathbf 1b^\top,
$$

$$
L=\frac1M\sum_{i=1}^{M}\operatorname{CE}(Z_i,y_i).
$$

那么

$$
G_i=\frac{\operatorname{softmax}(Z_i)-e_{y_i}}{M},
$$

$$
Q_i=G_iW^\top,
$$

$$
dW=\sum_i P_i^\top G_i,\qquad db=\sum_i G_i,
$$

$$
dH_u=\sum_{i:S_{iu}\ne0}S_{iu}Q_i.
$$

对于单个目标行或小 row panel，一旦完整类别维的 logits 可用，其 loss、G、Q 和 dW 贡献便可以计算。这正是允许前向与反向局部闭合的条件。

这里没有按 panel 更新参数。所有 panel 读取同一份 W，梯度全部汇总后才执行本来的 optimizer step。全局梯度裁剪也必须等待完整梯度。

### 3.3 从访问序列得到 source 生命周期

固定目标 panel 的执行顺序 $\pi$。对每个至少被访问一次的 source u，定义：

$$
f_\pi(u)=\min\{t:u\text{ 出现在 panel }\pi_t\},
$$

$$
\ell_\pi(u)=\max\{t:u\text{ 出现在 panel }\pi_t\}.
$$

在 panel t 执行期间，其 source 活跃集合为

$$
C_t=\{u:f_\pi(u)\le t\le\ell_\pi(u)\}.
$$

对应峰值

$$
w_\pi=\max_t|C_t|.
$$

两个 panel 之间的 frontier 可以定义为

$$
F_t=\{u:f_\pi(u)\le t<\ell_\pi(u)\}.
$$

必须区分 C 与 F：一个仅在当前 panel 使用的 source 虽然不跨 panel 存活，执行该 panel 时仍然需要临时状态，不能漏计工作集。

**本文提出的结构指标是 $w_\pi$，而不是只有 N、E、degree 或某个 16-row 窗口的 E/U。** 它描述“同时尚未完成的 source 有多少”，直接关系到能否将前向特征与反向累加状态保留在限定存储内。

### 3.4 编译的不是另一个大张量，而是一串事件

图和监督集合固定时，在 cold start 编译：

```text
open(u, slot):
    从原始 H[u] 读取，并按指定合同生成局部 source 表示
    清零该 slot 的 dH 累加器

consume(panel, edge-slot descriptors):
    使用局部 source slot 聚合 P_panel
    classifier → exact CE → G_panel → dW/db + Q_panel
    使用该 panel 的局部 incidence 把 Q 贡献累加回 source slot

close(u, slot):
    dH[u] 的所有当前 terminal 贡献已完成
    写出最终结果，或交给合法的局部 activation backward
    释放 slot
```

source 的区间已知，可以用经典区间分配/linear-scan 思路复用 slot；这部分算法不是新发明。新研究对象是如何在 GCN 的 loss-closed 原值—伴随执行图上构造这些区间，并将它们变成可并行、可回退的执行计划。[^R10]

热路径中的边引用可以预编译成 slot id。全局 source id 主要用于 open/close 事件，避免每个 panel 都通过哈希表构造 unique-source 映射。不过，编译事件和 edge descriptor 的空间必须计入成本；不能因为 metadata 不叫 tensor 就不统计。

若单个计划确实保证 slot 数少于 65536，可以考虑局部 16-bit slot id。但这只是有范围证明的编码选项，不是所有大图都能降成 16-bit。

### 3.5 一个受限的状态下界，而不是夸大的通用 I/O 定理

考虑如下受限执行模型：以 panel 为原子调度粒度；每个 source 从首次引用的 panel 开始，到最后引用的 panel 完成局部反向为止，持有一个不可拆分的“特征＋梯度累加器”配对 slot；不允许 reload、spill、重算，也不允许通过其他代数关系压缩多个 source 状态。所有 panel 按给定顺序处理。

在任一时刻，同时存活的 source 不能占用同一个 slot。因此任何此类实现至少需要 $w_\pi$ 个 slot。按区间结束释放、按首次出现分配的方案可以达到这一 slot 数。

这是给定顺序、原子 panel 和配对 slot 模型中的结论。它不是任意 GCN 算法的全局最小内存定理，也不是任意矩阵压缩形式的 I/O 下界。若允许特征与梯度独立分配、panel 内更细粒度交错、spill/reload、邻接压缩或重算，最优方案可能使用更少状态。因此这里的最优性只用于验证这一种计划，不用于排斥其他算法。

这个受限结论的价值在于产生一个**可计算的执行准入证据**：若 $w_\pi$ 已大到不可能满足预算，就不应启动“所有 source 常驻”的路径。

### 3.6 工作集：不要只算那一块 4 KiB 的 P

假设 source 表示是 BF16，反向累加器是 FP32，每个 source slot 的基本数据量为：

$$
2K+4K=6K\text{ bytes}.
$$

因此基本 source 状态量是：

$$
B_{\mathrm{source}}=6Kw_\pi.
$$

这不是完整内存预算。还要加上：

$$
B_{\mathrm{total,live}}
\approx 6Kw_\pi
+B_{\mathrm{CE}}
+B_{\mathrm{P/Q}}
+B_{\mathrm{Wpack}}
+B_{\mathrm{dWprivate}}
+B_{\mathrm{metadata}}
+B_{\mathrm{queues}}.
$$

若一个 r-row micro-panel 同时保存 FP32 logits 和 BF16 Gs，其对应项约为 $6rD$ bytes；若采用 T 个线程的完整 FP32 私有 dW，额外项可能达到 $4TK_pD_p$。尾块 padding、双缓冲和归约副本还要另算。

K=128、D=2983、r=16 时，logits 加 Gs 已有 286368 bytes，约 279.7 KiB。这并不是整个 terminal 都能进 L1 的证据。源 slot 和 dense scratch 应分别考虑 L2、共享 LLC 与 NUMA 所属关系，不能把所有缓存容量简单相加后当成一个统一 scratchpad。

原始 H、前面隐藏层为反向保存的状态以及最终 dH 输出，也不会因为 terminal 流式化自动消失。上述预算区分的是 terminal 的局部工作状态，不是声称整个训练 RSS 只有 $O(w_\pi K)$。

### 3.7 对多核的要求：不能把随机原子写伪装成融合收益

最直接的单线程算法容易正确，但不是最终多核方案。若每个 worker 随意处理目标行，多个 worker 会同时修改同一个 source 的 FP32 adjoint。全局 atomics 或线程私有 full dH 很可能抹掉收益。

一个值得验证的并行版本是两级 panel：较大的 superpanel 决定同步与 source ownership；内部 micro-panel 执行 AMX dense 和 CE。dense 阶段只留下有界的窄 Q_super，然后由 source owner 消费本 superpanel 的局部反向 incidence，更新自己持有的 slot。

这允许小范围保留 Q，而不是坚持“不允许任何中间张量”。也意味着以下成本必须被量化：worker 之间的等待、Q_super 传递、边分组 metadata、source 所有权不均以及跨 NUMA 通信。

对于跨多个分区出现的 source，需明确选择唯一 owner 汇总、边界 spill/reload 或有限复制。任何复制方案的“一次 source load”都只能在相应 owner/分区边界内表述，不能宣称整机绝对只读取一次。

### 3.8 真正可以继续研究的调度目标

只最大化邻居重叠不够：高度共享的 hub 可能从第一块一直存活到最后一块，反而抬高 frontier。只最小化 frontier 也不够：过分串行化会降低 AMX 和多核利用率。

建议研究下列联合目标，而不是先写大 kernel：

$$
\min_{\pi,\mathcal P,\mathcal O}
\left[
\widehat T_{\mathrm{dense}}
+\widehat T_{\mathrm{sparse}}
+\widehat T_{\mathrm{spill}}
+\widehat T_{\mathrm{sync/NUMA}}
\right],
$$

其中 $\mathcal P$ 是 panel/partition，$\mathcal O$ 是 source owner，约束为每个 worker/domain 的实际 live-byte 预算。

一个初始启发式可以维护每个 source 尚未处理的目标引用数，在受限候选行窗口内平衡“新开启多少 source”“本轮关闭多少 source”“AMX tile 填充率”和“owner 负载”。这个启发式是待测候选，不宣称求出了最优图排列。

更稳健的运行时应允许少量长生命周期 source spill，其余 source 常驻；也允许对不适合的分区回到成熟的 Q materialize + transpose pull。不能为满足一个统一叙述而强迫所有图走同一种路径。

### 3.9 与最接近工作的差异和碰撞风险

CCE 主要回答 dense classifier/CE 的宽中间态如何少物化；这里额外把选中稀疏算子的正向 source 输入和其伴随输出纳入同一生命周期。Tile Fusion 已有稀疏依赖驱动的融合与同步处理；这里需要证明 loss 使原本跨前后向的依赖局部闭合，且联合 source 生存期提供了额外能力。SAR 已有顺序反向重物化；这里的理想驻留路径不依赖重新生成整块前向计算图。[^R07][^R08][^R09]

然而，若实现最后只剩“按照现有依赖 DAG 做普通融合”，或所有收益都来自原来已经计划的 native Q/CE，A 就不足以构成新的主贡献。**必须在相同 micro-engine 上做消融，隔离联合 source 生命周期本身的效果。**

### 3.10 与强基线相比，到底还有多少上限

必须先把用户已有的 whole-terminal micro-engine 当作强基线。它已经可以删除大量宽 logits/Gs 流量，也可以局部消费 P。因此不能再把这些 bytes 全部记成 A 的新增收益。

若强基线只保留 BF16 的 Q[M,K]，Q 的一次全局写和一次读对应 $4MK$ 个逻辑 bytes。A 可能额外减少 source 的重复冷访问和独立转置阶段的部分开销，但具体 DRAM 字节取决于缓存命中；成熟的 transpose pull 本来就可以只写一次最终 dH，不能把所有梯度写入都计作可删除。

相反，A 新增了活跃 adjoint 状态、source-owner 组织和同步。因此应先计算剩余阶段的占比。设它在强基线整步中占比为 f，则即使把该部分完全免费化，整步理论加速上限也仅为 $1/(1-f)$。这是按定义推导的上限，不是当前 f 的实测值。

### 3.11 预期适用与失败条件

A 可能适合监督稀疏算子存在局部团簇、可构造受控 frontier，且终端宽 dense/CE 已具备高效微引擎的情形。

它可能在下列情形失败：重排后仍有大量长生命周期 source；高度扩展的图使 w 接近 N；owner 负载被少数 hub 主导；为了维持驻留而降低并行度；strong baseline 的 Q 与 source 本来就大量命中 cache；或者 global normalization、跨样本 loss 等破坏局部损失闭合。

固定全局 dropout mask 下，可以在 source 的最终 dH 完成后接一段逐点 activation backward。**但不能据此推断更前面的整个 GCN 层也能立即完成反向**，其空间依赖需要另行证明。

---

## 4. 候选 B：共享 source 成本驱动的同层混合顺序

### 4.1 不再为每一行独立选择

对某些目标节点，先聚合只需在输出侧进行少量 dense 工作；对另一些共享邻居很多的目标节点，先变换 source 可以让一次 dense 结果被多个目标复用。

关键是选择之间有耦合：当目标 v 选择 Transform-first 时，它需要的 source 可能已经因为其他目标的选择被变换。其边际成本不是“本行 degree × 单个 source 变换成本”。

这在 M<N 的矩形监督域中尤其值得考虑：全局 Transform-first 可能要为远多于 M 个 source 付出变换，而全局 Aggregate-first 又可能在一组高复用、K>D 的行上浪费稀疏特征宽度。

### 4.2 一个能够精确求解的受限模型

对每个目标行或固定目标组 v，设：

- $x_v=0$：Aggregate-first；$x_v=1$：Transform-first。
- $a_v\ge0$：v 选择 Aggregate-first 的局部成本。
- $b_v\ge0$：v 选择 Transform-first 后的局部消费成本，不包含共享 source 的首次变换。
- $c_u\ge0$：source u 的变换与规定表示准备的共享开启成本。
- $z_u\in\{0,1\}$：source u 是否被任意 Transform-first 目标需要。

代理目标为：

$$
\min_{x,z}
\sum_v[(1-x_v)a_v+x_vb_v]
+\sum_uc_uz_u,
$$

并满足：

$$
z_u\ge x_v\quad\text{对所有 }S_{vu}\ne0.
$$

其中 c 只支付一次，正是与逐行独立评分不同的部分。由于 c 非负，最优解不必开启没有消费者的 source；若 c=0，可能存在等价的冗余开启，不影响最优值。

### 4.3 最小割构造与证明

构造有向网络，包含超级源 s、超级汇 t、目标节点 v 和 source 节点 u：

| 弧 | 容量 |
|---|---:|
| s → v | $a_v$ |
| v → t | $b_v$ |
| v → u，当 $S_{vu}\ne0$ | $C_\infty$ |
| u → t | $c_u$ |

取 $C_\infty$ 严格大于全部有限容量之和，而不是在代码里随意写一个可能不够大的常数。

把位于源侧的目标解释为 Transform-first。若 v 在源侧而依赖的 u 在汇侧，割会切断大容量弧，因此任何最优有限割都会开启这些 u。一个可行割支付的有限容量，恰好等于上述目标值。反过来，任何可行选择都能构造同成本的割。因此该代理模型的最优选择可以由最小割得到。

**最小闭包/最小割处理共享固定成本是经典技术，本身不是本文创新。** 潜在新增内容是将 GCN 的同层异构关联顺序与 source 开启依赖表述为这个问题，并证明什么时候这种自由度有价值、什么时候没有价值。[^R11]

### 4.4 一个小例子：为什么独立逐行选择会失效

设 M=6、N=16。前四个目标都依赖 `{0,1,2,3}`；第五个目标依赖 `{4,6,7,8,9,10}`；第六个依赖 `{5,11,12,13,14,15}`。每个目标保留其自环依赖。

取 K=128、D=32，并人为规定 dense 每行成本 g=10、每条边每特征成本 s=0.05：

$$
a_v=g+s\,\deg(v)K,
\qquad b_v=s\,\deg(v)D,
\qquad c_u=g.
$$

实际运行的最小割与穷举检查得到：

| 选择 | 代理成本 |
|---|---:|
| 所有目标 Aggregate-first | 239.2 |
| 所有目标 Transform-first | 204.8 |
| 前四行 Transform-first，其余 Aggregate-first | 162.4 |
| 忽略跨行共享的逐行独立选择 | 239.2 |

这些是**人为成本单位，不是毫秒，也不是 TFS 的加速比**。这个例子只证明共享开启成本可以改变最优选择，而不是证明真实 kernel 必然获益。

独立打分会认为前四行每行都需支付四次 source 变换；联合选择则只为这四个共享 source 支付一次。后两行的 source 基本不共享，因此留在 Aggregate-first。

### 4.5 必须保留的否定结论：全节点统一成本下，混合没有优势

设输出为全部 N 个节点，S 的每个对角元素均非零。令 A 为 Aggregate-first 目标集合，T 为 Transform-first 目标集合；令 $U_T$ 为 T 使用的 source 并集。

因为自环存在：

$$
|U_T|\ge|T|.
$$

设单行 dense 成本均为 g，每边每特征稀疏成本均为 s，且不考虑不同尺寸 kernel、缓存和常驻状态的额外差异。令 $E_A,E_T$ 为两组目标的非零数，则混合成本为：

$$
C_{mix}=g(|A|+|U_T|)+s(KE_A+DE_T).
$$

当 K≥D 时，全局 Transform-first 的成本为：

$$
C_T=gN+sD(E_A+E_T),
$$

于是：

$$
C_{mix}-C_T
=g(|U_T|-|T|)+s(K-D)E_A\ge0.
$$

当 K≤D 时，同理对全局 Aggregate-first：

$$
C_{mix}-C_A
=g(|U_T|-|T|)+s(D-K)E_T\ge0.
$$

**结论：在这个模型下，混合不能优于最佳全局顺序。** 这包括 K=D 的边界。

这是本文给出的简单代数推导，不声称它本身是一条新的图论定理。它的重要作用是防止把一个矩形监督域候选夸大成普适优化。

矩形且每个目标保留对应自环时，K≤D 的类似支配关系仍使 Aggregate-first 成为该统一成本模型的强候选。因此 B 的首要探索域应是 **M<N、K>D、source 复用高度不均**；不是当前 K=128、D=2983 的宽类别末层。

硬件成本不均、既有缓存或跨层状态可能打破该简单模型，但那必须由新成本项和实验解释，不能假装否定结论不存在。

### 4.6 混合前向必须配套完整梯度

按 A/T 分割 S 和输出梯度 G：

$$
P_A=S_AH,
\qquad R_T=S_T^\top G_T.
$$

则一套实数上等价的反向为：

$$
dW=P_A^\top G_A+H^\top R_T,
$$

$$
dH=S_A^\top(G_AW^\top)+R_TW^\top,
$$

$$
db=\sum_iG_i.
$$

这不是允许重复计入边或漏掉某一组梯度。source 可以同时被 A 和 T 使用，最后需要正确相加两条梯度贡献。

这些公式仍然受到数值合同约束。尤其是每行采用不同关联顺序，会引入异构 BF16 舍入路径。若项目要求单一基线的逐位一致，B 很可能不属于可接受的优化空间；若允许固定精度策略下的容差和训练质量验收，则应固定 routing、逐项测量误差并报告。

### 4.7 为什么不能把这个最小割称为“全训练最优 planner”

上述最优性只针对可分解的代理目标。实际执行还存在 GEMM shape 效率、padding、线程数、pack 共享、选择后产生的布局、dH 合流、NUMA、live interval，以及不随 row 数线性变化的成本。

例如，增加硬约束：

$$
\sum_um_uz_u\le B
$$

后，不能直接声称同一个普通最小割仍能求出所有最优预算解。将 $c_u$ 改为 $c_u+\lambda m_u$ 可以产生拉格朗日候选，但需要单独检查可行性，也不保证覆盖所有硬预算最优解。

同理，基于每行成本的选择可以被收粗为 AMX 友好的目标组与 source block；此时最小割只对该固定分组的模型最优，不能再声称保留了逐行模型的全部最优性。

### 4.8 预处理规模和真正的工程风险

在最细粒度建模时，依赖网络有 O(M+U_M) 个节点和 O(E_M) 条弧。对千万级 selected edges，通用求解器的内存与冷启动可能不可接受。不能把小例子的 NetworkX 求解直接当成可扩展方案。

研究可以从固定粗分组开始，先评估模型能否找到明显优于两个全局选择的候选。然后再考虑图压缩、分区求解或成本近似。若 coarsening 使最优选择退化为全 A/全 T，或者训练步收益不能覆盖规划摊销，就应停止该方向。

### 4.9 与 COR 的差别，以及还不足的地方

COR 是必须直接对照的强相关工作。B 不应再声称首次重排或首次缓存；候选差异限定为“同一矩形层中的拓扑耦合异构顺序选择，以及共享 source 开启成本的可解模型”。[^R01]

但是，如果继续查重发现已有同层细粒度选择和相同共享成本建模，B 的新颖性会大幅下降。即使没有直接同构论文，若主要结果只是将经典闭包模型应用到一个收益很小的特例，也未必足以支持高影响力系统论文。

---

## 5. 已运行的参考验证及其边界

随报告附带 `verify_candidates.py` 与 `verification_results.json`。脚本只依赖 NumPy、PyTorch 和 NetworkX；固定随机种子为 20260910。它不是项目补丁，没有调用 TFS 原生扩展。

### 5.1 候选 A 的数学与槽位验证

在 48 组小矩形矩阵上检查随机权重、随机执行顺序、非规则 shape 和刻意构造的空目标行。将流式结果与 FP64 dense PyTorch 自动微分对比：

| 检查项 | 最大绝对误差 |
|---|---:|
| loss | 4.44×10⁻¹⁶ |
| dH | 2.22×10⁻¹⁶ |
| dW | 2.22×10⁻¹⁶ |
| db | 1.11×10⁻¹⁶ |
| 固定 mask 的 ReLU/dropout 链式梯度 | 3.33×10⁻¹⁶ |

同时检查每个被使用的 source 仅在 open 事件加载一次；slot 不冲突；已分配峰值 slot 数等于该顺序下的最大活跃区间数。

这验证了单流、FP64 参考算法，而不是 source-owner 多核实现、缓存命中、BF16 舍入合同或实际加速。

### 5.2 候选 B 的梯度与优化模型验证

在 48 组问题上分别测试全 A、全 T 和随机混合，共 144 组执行；logits/loss 最大绝对差不超过 8.89×10⁻¹⁶，一阶梯度最大绝对差不超过 2.23×10⁻¹⁶。

另外，80 个小成本实例的最小割与全部 10240 个选择的穷举最优值一致，最大目标值差为 2.14×10⁻¹⁴，属于本次浮点累计误差范围。

对 60 张带自环的小图、三个 K/D 关系，共穷举 23040 个混合选择，没有发现违反第 4.5 节支配结论的情况。最小差约 −1.42×10⁻¹⁴，也在浮点容差内。**理论依据是代数证明，不是穷举数量。**

### 5.3 同 N/E/degree 的 frontier 可以相差很大

构造 N=M=96、每行 degree=6、总非零数 576 的合成图。一个是 12 个大小为 8 的独立局部社区；另一个是相同 row degree 的随机邻接。得到：

| 合成结构与行序 | 峰值活跃 source slot | K=128、6K/slot 的理论 source 状态量 |
|---|---:|---:|
| 社区图，按社区处理 | 8 | 6144 B |
| 同一社区图，随机打乱行序 | 94 | 72192 B |
| 随机邻接，自然行序 | 90 | 69120 B |
| 随机邻接，随机行序 | 89 | 68352 B |

这个例子支持“只看 N、E 和 degree 不能决定可流式驻留程度”，并展示 A 的一种失败模式。它不证明现实图可获得相同降幅，也没有把理论状态字节换算成运行时间。

### 5.4 BF16 反例与 dropout 提醒

脚本中的一个故意简化的 BF16 重关联例子，最大输出差约为 9.88×10⁻²。它不模拟 TFS 的全部归一化与精度合同，只说明实数等价不能推出量化后逐位一致。

对 64 个当前都为非零的独立活跃坐标，若相邻步使用独立的 p=0.5 dropout mask，即使未 dropout 的值不变，整个 mask 完全相同的概率也只有 $2^{-64}\approx5.42\times10^{-20}$。这说明“激活缓慢变化，因此整行跨步缓存经常有效”的直觉，在独立 dropout 下不能直接成立。这里没有断言所有增量方法都无效；它只否定一个常见但不足的缓存前提。

---

## 6. 其他考虑过的想法：为什么没有列为主贡献

### 6.1 邻接行差分、公共邻域和树状聚合复用

把一行表示为另一行加上少量邻居差分，或用公共子聚合减少工作，是值得工程评估的机制。但 HAG 和 CBM 已形成直接相关先例；不能在没有更强结构性差异的情况下重新命名。[^R04][^R05]

### 6.2 “online softmax 一遍完成，logits/Gs 全不保存也不重算”

一个行的 Q 可以被理解为 softmax 加权的 W 行组合，适合用在线归一化维护。问题在于 dW：同一个 class 的梯度要累加多个样本的贡献，每个样本的最终 softmax 分母不同。将尚未得到最终分母的各行贡献先合并，通常不能用一个公共缩放因子事后修正。

这不构成对所有算法的绝对不可能性证明，但足以说明：不能仅凭 online softmax 就宣称 logits、Gs、重算和额外状态可以同时全部删除。保留小 micro-panel 或第二遍计算是正常的交换。CCE 应作为直接参照，并在精确对照中禁用其可选的梯度过滤。[^R07]

### 6.3 跨 epoch 的隐藏特征增量缓存

隐层 W 变化和 dropout 都会使缓存失效。可以研究元素级差分、误差控制或特定稳定阶段，但需要额外元数据与传播成本。若要冻结 dropout 或跳过小变化，还可能改变训练过程，不属于本文默认的精确运行时优化。

第 5.4 节的简单概率反例表明，不能只用“训练后期变化小”支持整行缓存的新论文主张。

### 6.4 多层监督依赖域裁剪

该思路可以减少确实不在 loss 依赖内的工作，但多层全邻居依赖构造已有实现与文献。高连通图的依赖域又可能迅速扩展到 N。它适合作为有覆盖率统计的执行优化，不宜仅凭“从 terminal 扩展到 hidden”认定为高新颖度。[^R12]

### 6.5 基于 BF16 舍入单元的有证书提前终止聚合

这是保留的一条高风险想法，而不是建议立刻实施的第三主线：若已累积的部分和以及所有剩余项的严格界，都落入同一个 BF16 舍入区间，则可以证明剩余项不会改变该次 BF16 输出，从而省略后续累加。

障碍是严格界本身的代价，且不同 feature channel 必须同时满足证书；浮点求和误差也要包含在界内。均匀同号贡献的简单情形中，舍入区间很窄，往往接近算完才满足条件。若邻居需要重排，原来求和顺序的舍入语义也会改变。

这条线存在数学上的可能性，但本次没有完成充分查重、原生可行性分析或有效数据分布证明。因此不将它列为“已找到的新创新点”，只保留为失败成本较低的离线统计实验。

---

## 7. 当前实现中会影响创新判断的额外问题

### 7.1 强基线必须先补齐

若 selected/rectangular pull 仍有可以直接修复的重复索引遍历、packing 反复执行或超大 panel，新的复杂方案很容易只是赢在这些基础缺口上。本文不把这种收益当作 A/B 的学术证据。

应在相同代码快照上保留两组对照：冻结 authority 用于历史可比性；补齐确定性基线问题后的版本用于证明新增机制。代码里已有的保守经验 gate 与正式成本模型也要区分。[^C01][^C03]

### 7.2 Scope 生命周期直接决定预处理是否可信

A 的事件表和 B 的依赖网络都绑定图、监督集合、row id 映射、自环/归一化合同以及静态分区。相同 N 不能证明这些对象相同。计划需要有明确 generation，并在所有相关对象变化时失效。

W pack 只可在规定的权重版本内复用；不能将“一次 train step 内不变”误推广为“跨 optimizer 更新仍有效”。这些不是新颖点，却是原型能否成为可信运行时的前提。

### 7.3 预计算一阶梯度的 Autograd 合同

loss-closed 执行可预先形成当前 scalar loss 的一阶梯度。如果外部只在 backward 传入一个公共标量系数，可以在返回时统一缩放。若 API 返回逐行 loss 并允许任意逐行上游梯度，或支持多个耦合 loss、二阶导等，就需要不同状态与接口，不能无条件套用当前闭合。

应明确禁止或另行实现未覆盖语义，而不是通过自定义 autograd 隐藏差异。

### 7.4 数据移动是指标，不是已经证明的主导因素

AMX 利用率、频率、同步、工作队列和稀疏地址计算都可能改变瓶颈。用户当前提出“GCN 主要受数据移动约束”是合理研究假设，但对每个优化后的 shape 仍需重新 profile。

本文的所有 bytes 公式都是按数据表示推导的逻辑量或工作集量；实际 DRAM、LLC 与远端 NUMA 流量必须由硬件计数获得。

---

## 8. 研究验收：先验证结构性机会，再写生产 kernel

### 8.1 候选 A 的第一道门槛

在实际监督 CSR 上输出：不同 row/panel 顺序的 w；source 首末使用距离；分区边界 source 比例；source owner 负载；每个 NUMA 域的 live bytes；metadata/编译成本；允许有限 spill 后的额外字节。

比较自然顺序、当前 degree/locality 顺序、source-overlap 顺序与拟议的 frontier-aware 顺序。不能只对一个特别适合的社区图展示结果。

如果在合理并行度与 scratch 预算下，绝大多数 source 都长期活跃，或者预计节省的剩余流量不足以覆盖同步/owner 成本，应暂停 A 的大型 native 开发。保留普通 micro-engine 作为成熟方案不是研究失败的掩饰，而是该机制的正确边界。

### 8.2 候选 B 的第一道门槛

先在真实 K>D、M<N 的 selected scope 上，使用实测的小型 sparse/GEMM/packing 成本建立代理模型。比较最优全 A、最优全 T、独立逐组选择与耦合选择。

必须同时报告多少图/shape 得到真正的混合解，以及把结果送入完整 backward 后还有多少收益。若可观收益只存在于人为设计的共享邻居例子，不能支撑普适论文主张。

### 8.3 建议的强对照和消融

| 研究问题 | 必须比较的版本 |
|---|---|
| A 是否超出原有 terminal fusion | 相同 native logits/CE/dW/Q 引擎，分别接传统 CSR/transpose 与联合生命周期 |
| source 生命周期还是普通重排起作用 | 同一节点顺序下，只开 feature cache、只开 adjoint slot、两者同时开 |
| frontier 是否比 overlap 更有解释力 | degree 相近、overlap 相近但首末跨度不同的图；以及相同 N/E/degree 的控制图 |
| A 的并行性是否成立 | 单流、source-owner、多 NUMA 与成熟 pull 对照，记录等待和远端流量 |
| B 的共享建模是否必要 | 全 A、全 T、独立组决策、最小割共享决策 |
| B 是否只是 forward 加速 | 完整 forward/backward/optimizer 与达到目标质量的总时间 |
| 预处理是否值得 | cold start、单步稳态、规定步数内总耗时，含计划重建与失效 |

对 CCE 类对照，精确与过滤/近似版本分开；对其他系统，硬件、线程、数学目标和 precision contract 一致。不能将 CPU 与 GPU 的原论文绝对时间直接混在同一速度表里。

### 8.4 指标与统计方法

主要指标是整步时间、峰值内存和达到预定质量的总时间；辅助指标包括 DRAM 字节、LLC/DTLB、远端 NUMA、GEMM/packing、队列等待和归约时间。

训练准确性至少检查 loss、dW/db/dH、尾块、空行、自环、不同 panel 顺序、固定 dropout mask、计划失效和多次 optimizer 更新。长期实验应报告多个随机种子或明确说明单种子限制。

原生性能建议同节点交错 A/B、固定 affinity 和线程预算，记录软件与编译版本，并区分冷缓存、热缓存和 steady state。若出现较小收益，报告离散程度而不是仅保留最快样本。

### 8.5 何时足够写成论文贡献

A 需要证明：原值—伴随联合生命周期在强 micro-engine 之外仍然带来可解释收益；frontier/owner 模型能预测适用范围；并且多核实现在真实图上没有被同步吞没。

B 需要证明：同层共享 source 的选择在真实矩形图上是必要自由度；代理模型与真实执行之间有足够相关性；以及所得收益不是改变精度或遗漏反向得到的。

两个候选都不必覆盖所有数据集，但必须诚实覆盖失败样本。一个只在特定范围成立、边界清楚的贡献，比宣称通用却依赖隐含例外更可信。

---

## 9. 建议的立项顺序与最后判断

**优先探索 A 的结构统计和强基线消融。** 它与当前 supervision-scoped terminal、native dense 组件、静态图及 persistent workers 的基础最匹配。值得问的问题不是“再省一个 tensor”，而是“能否将 source 的前向使用与反向完成编译成有界的联合生存期”。

**把 B 保留为范围明确的第二研究线。** 它具有清晰的共享成本模型、最小割解和否定性边界，适合先做低成本的离线评估。但不应把它放到 K=128→D=2983 的主战场，也不应为了一条看似漂亮的 min-cut 定理而忽略求解器规模和真实 kernel 非线性。

Wide-K 静态缓存、native Q、whole-terminal、inter-layer 和 D-slab 仍可继续作为必要的系统建设。它们为验证 A/B 提供基础，但不是本文重新发现的创新。

**最终结论不是“已经找到两个高新颖度论文点”，而是“经查重与反例筛选，保留下两个具体、可证伪、与当前代码可对接的学术候选”。** 在本次核对的一手资料中，没有找到与 A 的联合 source 生命周期或 B 的矩形层共享开启选择完全同构的机制；检索覆盖仍有限，不能据此声称首次。

剩余最重要的缺口不是给它们起名字，而是：继续针对最接近的 sparse-fusion/自动微分工作查重，并在真实图上证明有足够的结构性机会。若这些检查失败，合理结论就是保留工程成果、不强行制造一个学术故事。

---

## 附录 A：可复现实验文件

文件：

```text
TFS_academic_novelty_review_zh.md
verify_candidates.py
verification_results.json
README.md
```

运行：

```bash
python -m pip install numpy torch networkx
python verify_candidates.py --output verification_results.json
```

本次参考环境：NumPy 2.3.5、PyTorch 2.10.0+cpu、NetworkX 3.6.1。脚本设置单线程执行小型参考以减少不必要开销；这不是 CPU 扩展性测试。脚本使用 FP64 dense reference，不能直接用于千万边图的性能测量。

所有代码中的成本单位、BF16 简化反例及合成图均已在结果字段中标注。结果文件包含完整参数、用例数与最大误差。

## 附录 B：来源与进一步查重范围

检索与核对截至 2026-09-10。优先使用原论文、作者稿、会议论文及项目官方源码。下面来源用于界定已有技术和当前代码，不用于背书本文尚未验证的收益。

[^C01]: huamiao123/GCN-，固定提交 `e399d4f`，*Wire and validate supervision dense planner*。源码锚点：[GitHub commit](https://github.com/huamiao123/GCN-/commit/e399d4f)。仓库实验记录是作者报告，不是本次独立复现。

[^C02]: huamiao123/GCN-，`python/tfs_train/supervision_scope.py`，提交 `e399d4f`。[固定版本源码](https://raw.githubusercontent.com/huamiao123/GCN-/e399d4f/python/tfs_train/supervision_scope.py)。用于核对 selected scope、bounded terminal 和 gradient 数据流。

[^C03]: huamiao123/GCN-，`execution_plan.py` 与 `supervision_dense_plan.py`，提交 `e399d4f`。[authority plan](https://raw.githubusercontent.com/huamiao123/GCN-/e399d4f/python/tfs_train/execution_plan.py)；[supervision dense plan](https://raw.githubusercontent.com/huamiao123/GCN-/e399d4f/python/tfs_train/supervision_dense_plan.py)。

[^R01]: Julia Bazinska 等，*Cached Operator Reordering: A Unified View for Fast GNN Training*，arXiv:2308.12093，2023。重点核对 §3.1、缓存部分和 GCN backward：[原文](https://arxiv.org/html/2308.12093)。

[^R02]: Paras Jain 等，*Checkmate: Breaking the Memory Wall with Optimal Tensor Rematerialization*，arXiv:1910.02653，2019；ICLR 2020 工作。[论文记录](https://arxiv.org/abs/1910.02653)。用于界定已有的训练状态/重计算优化。

[^R03]: Yuke Wang 等，*TC-GNN: Bridging Sparse GNN Computation and Dense Tensor Cores on GPUs*，USENIX ATC 2023。重点核对图 4 与算法 1–2，row window 邻居去重和硬件块映射：[会议论文 PDF](https://www.usenix.org/system/files/atc23-wang-yuke.pdf)。

[^R04]: *Redundancy-Free Computation Graphs for Graph Neural Networks*，arXiv:1906.03707，2019。HAG 的共同邻域与共享聚合：[论文](https://arxiv.org/abs/1906.03707)。本文不采用该稿中任何未复核的性能数字。

[^R05]: João N. F. Alves、Samir Moustafa、Siegfried Benkner、Alexandre P. Francisco、Wilfried N. Gansterer、Luís M. S. Russo，*Accelerating Graph Neural Networks Using a Novel Computation-Friendly Matrix Compression Format*，IPDPS 2025，pp.1091–1103，DOI:10.1109/IPDPS64566.2025.00100。[作者接受稿](https://eprints.cs.univie.ac.at/8363/1/CBM_IPDPS25_accepted_.pdf)。相关预印本 arXiv:2409.02208 的题名略有不同。

[^R06]: *FeatGraph: A Flexible and Efficient Backend for Graph Neural Network Systems*，SC 2020，arXiv:2008.11359。[原论文记录](https://arxiv.org/abs/2008.11359)。

[^R07]: Erik Wijmans、Brody Huval、Alexander Hertzberg、Vladlen Koltun、Philipp Krähenbühl，*Cut Your Losses in Large-Vocabulary Language Models*，arXiv:2411.09009，2024。重点核对 §4 的 classifier/LSE/backward 与过滤机制：[原文 v1](https://arxiv.org/html/2411.09009v1)。

[^R08]: Mohammad Mahdi Salehi Dezfuli、Kazem Cheshmi，*Improving Locality in Sparse and Dense Matrix Multiplications*，arXiv:2407.00243，2024。重点核对依赖 DAG、tile fusion、同步/复制交换和 cache-size 约束：[原文](https://arxiv.org/html/2407.00243)。

[^R09]: Hesham Mostafa，*Sequential Aggregation and Rematerialization: Distributed Full-batch Training of Graph Neural Networks on Large Graphs*，arXiv:2111.06483，首版 2021，核对版本 v3 为 2022。[原文](https://arxiv.org/html/2111.06483)。

[^R10]: Massimiliano Poletto、Vivek Sarkar，*Linear Scan Register Allocation*，1999。[原论文 PDF 镜像](https://www.cs.ucla.edu/~palsberg/course/cs132/linearscan.pdf)。作为生存区间与线性扫描分配的已有技术来源，不将 slot 分配算法本身声称为新增。

[^R11]: Jean-Claude Picard，*Maximal Closure of a Graph and Applications to Combinatorial Problems*，Management Science 22(11):1268–1272，1976。[出版方论文页](https://pubsonline.informs.org/doi/10.1287/mnsc.22.11.1268)。用于界定经典共享固定成本/闭包方法。

[^R12]: DGL 官方文档，`MultiLayerFullNeighborSampler`。[官方 API](https://www.dgl.ai/dgl_docs/generated/dgl.dataloading.MultiLayerFullNeighborSampler.html)。用于确认全邻居多层依赖构造已有实现。

[^R13]: Daria Fomina 等，*On Efficient Scaling of GNNs via IO-Aware Layers Implementations*，arXiv:2605.31500，2026 年 5 月预印本。[原文](https://arxiv.org/html/2605.31500)。只引用其 Appendix D.3 的相关格式提案，不采用未独立验证的理论/性能主张。

进一步查重应集中在：loss-coupled forward/backward fusion、稀疏算子伴随的联合调度、graph live-range/edge-streaming、细粒度矩阵链关联顺序、共享 source 的闭包选择。通用搜索没有命中同构标题不构成排除已有工作的证据；上述列表不是完整文献综述，也不是专利新颖性意见。
