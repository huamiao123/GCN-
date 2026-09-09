# TFS 权威基线与监督范围优化集成说明

更新日期：2026-09-09

集成分支：`codex/tfs-integrated-optimizations-20260909`
权威基线提交：`2ffaa61`（`codex/publish-authority-qfirst-20260823`）

## 1. 版本定位

本分支包含两层代码：

1. `2ffaa61`及其之前的代码是已经冻结的`final_pre_numa`权威TFS基线；
2. 本分支新增的是监督范围末层、bounded-panel CE以及配套AMX/AVX内核的研究实现。

新增代码全部使用`shadow`命名或独立Python模块，不会被原权威
`execution_plan.py`自动选择。因此：

- 可以在同一个分支中审核全部源码和复现实验；
- 原权威运行入口和默认行为保持不变；
- 不能把本分支直接称为新的权威性能版本；
- 只有完成同机正式矩阵后，才能将shadow候选晋级为默认路径。

## 2. 原权威TFS已经包含的优化

### 2.1 双数据流与形状分派

GCN层可按矩阵结合顺序执行两种路径：

```text
aggregate-first / Q-first方向
H -> S H -> (S H) W

transform-first / Y-first方向
H -> H W -> S (H W)
```

反向同样区分先做稠密降维还是先做稀疏传播。权威planner目前仍以
`D >= K`作为基础结合顺序，再结合层位置、静态性、线程数、workspace和
候选门禁生成唯一的`execution_variant`。

源码：

- `python/tfs_train/execution_plan.py`
- `python/tfs_train/autograd.py`
- `python/tfs_train/highd.py`

注意：`D >= K`是权威版本的保守启发式，不应被描述成最终cost model。

### 2.2 稀疏与稠密协同执行

权威TFS使用AVX-512执行不规则稀疏pull，使用AMX BF16执行规则稠密计算，
并采用BF16输入/中间量、FP32累加和FP32 master parameter数值合同。它不是
单独替换一个GEMM，而是让forward、backward和保存张量遵守同一个层计划。

主要源码：

- `csrc/experiments/backward_opt_20260814/v6_wide_aggregate_probe.cpp`
- `csrc/experiments/products_saved_t_20260812/backward_v2_kernels.cpp`
- `csrc/experiments/products_saved_t_20260812/bindings_aggregate.cpp`
- `csrc/amx/`
- `csrc/avx/`

### 2.3 panel化与中间结果复用

权威路径通过row panel、D-slab和saved aggregate避免无界workspace，并在合法
路径中复用前向已经生成的聚合结果，减少反向重复稀疏遍历。静态首层可以缓存
`Hs`或其聚合结果`T0`；动态层仍遵守正常训练语义。

相关实现：

- `aggregate_static_v3`
- `aggregate_saved_v4`（实验候选）
- `aggregate_highd_full`
- `aggregate_highd_dslab`
- `transform_highd_stream/native/single_scan`（按门禁状态使用）

### 2.4 High-D与任意维度处理

权威版本支持small-D、wide-K、wide-D、非AMX整倍数尾维和任意`L>=2`。
High-D路径根据预算选择完整workspace或D-slab，避免输出维度增大时无界分配。

主要源码：

- `python/tfs_train/execution_plan.py`
- `python/tfs_train/highd.py`
- `csrc/amx/backward_v2_kernels.h`
- `include/tfs/kernel/backward_v2.h`

### 2.5 多线程与内存工程

权威实现包含持久worker、按NNZ近似均衡的连续行调度、active-row后向、并行
first-touch、workspace复用和有界LRU。这些是重要性能基础，但普通负载均衡、
线程池、缓存和first-touch本身不作为论文核心首创。

### 2.6 可复现实验合同

权威版本将每层计划冻结为不可变`execution_variant`，并记录源码哈希、扩展哈希、
CPU/NUMA/affinity、逐层计划、冷启动wall、200轮逐epoch数据和峰值RSS。

正式入口保持为：

```bash
bash scripts/build_extension.sh
sbatch scripts/run_final_pre_numa_gate.sh
sbatch --dependency=afterok:<gate_job_id> \
  scripts/run_final_pre_numa_authority_matrix.sh
```

## 3. 相比权威TFS新增的优化

### 3.1 监督范围末层数据流

设全图节点数为`N`，监督节点数为`M`，隐藏维为`K`，输出维为`D`。原权威
末层计算并保存全图输出；新路径只产生训练目标行：

```text
H[N,K]
  -> selected sparse pull
P[M,K] = (S H)[train_rows]
  -> compact classifier
Z[M,D]
```

隐藏层仍然执行完整full-graph GCN，不采样、不删除边、不改变训练目标。一次性
构造的矩形转置CSR把紧凑梯度精确传播回全图隐藏表示：

```text
Q[M,K] = Gs[M,D] W^T[D,K]
dH[N,K] = S_train_rows^T Q
```

这消除了未被loss使用的末层目标行，同时保持数学上精确的全批反向。

源码：

- `python/tfs_train/supervision_scope.py`
- `c3_selected_pull_bf16_shadow_v1`
- `c3_build_selected_transpose_shadow_v1`
- `c3_rect_pull_bf16_shadow_v1`
- `c3_rect_pull_bf16_scaled_fp32_shadow_v1`

### 3.2 末层Y-first、Q-first和权威回退

新增末层实现支持三种后向：

- `y_first`：先在输出宽度`D`上传播，再乘`W^T`；
- `q_first`：先把梯度从`D`降到`K`，再做矩形稀疏传播；
- `authority_active`：将紧凑梯度放回全图并调用成熟权威active-row后向。

Products实测表明1/2/4线程更偏向紧凑Y-first，16/32线程可能更适合成熟
active-row；IGB-2983则明显适合compact Q-first。因此新路径证明了结合顺序
不能只由`D > K`决定，监督范围和线程数同样影响crossover。

### 3.3 bounded-panel精确交叉熵

即使只保留`M`行，IGB-2983的`M x D` logits仍然很大。新路径把监督行按
row panel处理：

```text
P panel
  -> logits
  -> exact log-softmax / cross entropy
  -> 同一panel内形成Gs、Q、dW、db
  -> 释放宽临时量
```

它不保存全局`M x D` logits或gradient，也不使用近似softmax。持久工作集主要
变为`P[M,K]`、`Q[M,K]`和参数梯度，宽张量被限制为`row_tile x D`。

实现入口：

- `_ScopedTerminalCrossEntropyFunction`
- `terminal_cross_entropy`
- `train_cross_entropy`
- `TFS_SCOPE_LOSS_ROW_TILE`
- `TFS_SCOPE_LOGSOFTMAX_OUT`

### 3.4 原地log-softmax与确定性db归约

最初的手工`logsumexp/subtract/exp`会多次扫描宽logits。新候选使用原地
`log_softmax`减少扫描，再原地生成精确CE梯度。

`db`融合版本最初按worker累加，数值误差随线程变化。修复后改为固定
2,048-row chunk、chunk内FP32、固定顺序FP64合并，使归约顺序不依赖线程数，
没有通过放宽容差掩盖问题。

相关接口：

- `c3_scale_grad_bf16_db_v2`
- `TFS_SCOPE_FUSED_DB`

### 3.5 panel-local AMX compact dW

旧候选执行：

```text
P[M,K] -> P^T.contiguous() -> torch.matmul(P^T, Gs)
```

新native路径按512行micro-panel执行Gs转置、P packing、AMX dW、线程本地累加、
确定性归约和最终`K x D`写回，避免全局`P^T`物化以及框架/oneDNN往返。

源码：

- `c3_compact_dw_bf16_amx_shadow_v1`
- `TFS_SCOPE_NATIVE_DW`
- `tests/bench_compact_dw_amx_shadow.py`
- `tests/bench_native_dw_scope_igb.py`

### 3.6 T4 direct-tail转置

当`D`不是32的倍数时，旧T2路径先生成完整padded Gs再转置。T4直接读取
logical stride，只有最后的16x16尾块进入小型补零buffer，从而避免整块
padded copy。

源码：

- `transpose_y_avx512_tail_t4`
- `TFS_COMPACT_DW_T4`

该优化依赖形状和线程：它改善尾维处理，但低线程下native dW整体仍可能不如
框架实现，因此不能无条件启用。

### 3.7 紧凑AMX logits与融合epilogue

新路径复用权威High-D AMX GEMM，为紧凑`P[M,K]`直接生成FP32 logits，并在
tile epilogue中融合节点scale和bias。v2支持logical output stride，直接写连续
`M x D`输出，避免padded-stride view在CE阶段触发复制。

源码：

- `c3_compact_logits_amx_shadow_v1`
- `amx_gemm_4c_epilogue(..., output_stride, ...)`
- `TFS_SCOPE_NATIVE_LOGITS`

FP32 oracle显示该路径保持FP32 accumulator；它与PyTorch BF16-output的差异
主要来自后者更早舍入，不能把“与BF16输出不完全一致”误判为精度恶化。

### 3.8 保守的稠密融合planner

新增的独立planner根据：

```text
(M, K, D, threads, padding)
```

选择native dW、T4和native logits。它只覆盖已经测量的shadow区域，不修改
权威planner，也不按数据集名称分派。

源码：

- `python/tfs_train/supervision_dense_plan.py`
- `tests/test_supervision_dense_plan.py`

这仍是初步经验planner，不应宣称为已经完成的通用cost model。

## 4. 已验证结果与边界

以下是同节点直接A/B，不是跨节点加速比相乘：

| 工作负载 | 对比 | 结果 |
|---|---|---:|
| IGB-small，D=2983，L2，32T | 权威TFS / supervision+bounded | 1.763x |
| IGB-small，D=2983，L2，32T | 权威TFS / 加入native dW | 2.0965x |
| Products，L2，32T | 权威TFS / 完整候选 | 1.161x |
| Products，L3，32T | 权威TFS / 完整候选 | 1.046x |
| native连续logits，D=2983，32T | PyTorch BF16 producer / AMX producer | 2.516x |

代表性完整候选正确性：

- loss绝对误差：`0`或不超过`9.54e-7`；
- dH relative L2：约`0.0026%--0.0127%`，视对照路径而定；
- dW relative L2：约`0.000011%--0.219%`；
- db最大绝对误差：约`2.2e-6--3.62e-6`。

已知边界：

- IGB-19低输出维基本无收益，必须回退；
- Products L3受额外隐藏层的Amdahl限制，收益较小；
- native dW和native logits都存在`M/D/threads`交叉点；
- row tile不存在跨形状统一最优值；
- 当前结果是训练步直接A/B，不是新的200-epoch权威矩阵；
- 当前DGL-Mixed对比仍需用同一正式launcher重跑后才能发布。

完整实验记录见：

- `docs/SUPERVISION_SCOPED_TFS_RESEARCH_20260905.md`
- `docs/BOUNDED_PANEL_TERMINAL_CE_20260906.md`
- `docs/SUPERVISION_SCOPED_DENSE_FUSION_RESULTS_20260906.md`

## 5. 尚未完成或没有进入源码的设计

`docs/Q_PANEL_NUMA_PUSH_DESIGN_20260906.md`描述了destination-owned、NUMA-local
Q-panel push，但目前只是设计文档，没有对应正式内核和性能结论。它不能列入
“已经实现的优化”。

同样，以下内容没有在本分支中成为正式默认能力：

- 通用的监督比例/输出维度cost model；
- NUMA owner-compute末层反向；
- 全native的fused CE kernel；
- 新版本48-cell权威结果；
- GraphSAGE、GAT等非GCN模型泛化。

## 6. 创新与工程优化边界

可以重点论证的组合贡献是：

> 面向单路CPU full-batch GCN的loss-coupled supervision-scoped terminal
> dataflow，在保持隐藏层和训练目标精确不变的情况下，将selected sparse
> aggregation、bounded exact CE、AMX dense计算和矩形稀疏反向组织成一条
> 有界流水线，消除High-D全图末层中间张量。

支撑性技术贡献是：

- panel-local AMX compact dW；
- logical-stride AMX logits融合epilogue；
- direct-tail转置和确定性归约；
- 监督范围、形状和线程共同决定的安全回退机制。

不能单独宣称为创新的内容包括：

- BF16或AMX本身；
- 普通panel/tiling；
- 按边负载均衡；
- 静态特征缓存；
- NUMA first-touch；
- Y-first/Q-first矩阵重结合本身；
- 通用中间张量消除概念。

## 7. 源码和验证索引

| 类别 | 路径 |
|---|---|
| 权威planner | `python/tfs_train/execution_plan.py` |
| 权威autograd/dispatch | `python/tfs_train/autograd.py`、`python/tfs_train/highd.py` |
| 新监督范围数据流 | `python/tfs_train/supervision_scope.py` |
| 新稠密planner | `python/tfs_train/supervision_dense_plan.py` |
| native shadow内核 | `csrc/experiments/backward_opt_20260814/v6_wide_aggregate_probe.cpp` |
| AMX接口 | `csrc/amx/backward_v2_kernels.h` |
| AVX转置 | `csrc/avx/ybar_layout.cpp` |
| Python绑定 | `csrc/experiments/products_saved_t_20260812/bindings_aggregate.cpp` |
| 接口声明 | `include/tfs/kernel/backward_v2.h` |
| 单元测试 | `tests/test_supervision_scope_shadow.py`、`tests/test_supervision_dense_plan.py`、`tests/test_compact_dense_shadow_native.py` |
| 性能/数值探针 | `tests/bench_supervision_scoped_*.py`、`tests/bench_panelized_*.py`、`tests/bench_compact_*.py`、`tests/bench_authority_vs_bounded_igb.py` |
| Slurm复现脚本 | `slurm/` |

## 8. 审核建议

审核时先比较：

```bash
git diff 2ffaa61..codex/tfs-integrated-optimizations-20260909 -- \
  python/tfs_train/supervision_scope.py \
  python/tfs_train/supervision_dense_plan.py \
  csrc/ include/tfs/kernel/backward_v2.h
```

随后运行不依赖真实大图的测试，再在AMX服务器构建扩展并执行shadow gate。
不要用本分支的研究launcher覆盖`run_final_pre_numa_authority_matrix.sh`。
