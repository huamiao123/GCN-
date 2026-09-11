# TFS-Train：数据流修复与两级 Panel 研究版（2026-09-11）

> 原始 TFS 推理 `SHW` 融合、degree sort 和 AMX tile 数据流向完整归一化 GCN 训练迁移的全面实验档案，见 [`docs/ORIGINAL_TFS_SHW_TRAINING_EXPERIMENTS_20260911.md`](docs/ORIGINAL_TFS_SHW_TRAINING_EXPERIMENTS_20260911.md)。配套源码快照和原始证据分别位于 [`experiments/original_tfs_shw_training/`](experiments/original_tfs_shw_training/) 与 [`evidence/original_tfs_shw_training/`](evidence/original_tfs_shw_training/)。

本分支是以 `final_pre_numa` 权威实现为基础完成的独立修复与数据流研究快照，目标是
面向 Intel AMX CPU 的 full-batch GCN 训练，减少 sparse/dense 边界上的重复扫描、
全局中间张量和无效数据搬运。

发布分支：`codex/tfs-dataflow-audit-repair-20260911`

基础提交：`e399d4f`（其权威基线继承自 `2ffaa61`）

这不是对旧 R5 路线的延续，也不是把所有实验性 kernel 自动打开的性能分支。默认
planner 仍保持保守；只有通过正确性、性能和执行契约检查的路径才能成为正式候选。

## 1. 主线：训练数据流，而不是单个 kernel

GCN backward 中的核心关系可写成：

```text
Gs = scale * G
Y  = S^T * Gs
dW = Hs^T * Y
dH = scale * (Y * W^T)
```

当 forward 已保存聚合结果 `P=S*Hs` 时，还可以使用 aggregate/Q-first：

```text
Gs = scale * G
dW = P^T * Gs
Q  = Gs * W^T
dH = scale * (S^T * Q)
```

本项目的主线贡献不是简单地把 GEMM 换成 AMX，而是根据层形状和中间张量宽度选择
数据流，并让 sparse traversal、BF16 staging、AMX dense consumer 和梯度归约在
受控的 panel 生命周期中协同执行。

## 2. 主要创新点

### 2.1 两级 Panel 训练执行

两级 panel 是本版本最重要的执行组织方式：

```text
Level 1：预算约束的 row panel
  限制 Gs/P/Q/logits 等动态张量的同时存活范围
  ↓
Level 2：native 512-row micro-panel
  由 edge-balanced worker ownership 分配给线程
  ↓
AMX 16-row tile / 32-reduction tile
  panel 内完成 sparse 生产、布局转换和 dense 消费
```

High-D 下另有与 row panel 正交的 `D-slab`：它限制线程私有 dW workspace，但不能
被误认为新的 sparse traversal。planner 分别记录：

- `panel` 与 `panel_working_set_bytes`；
- `dense_d_tile`；
- `sparse_d_slab`；
- 真实 `d_slabs`、最大 slab 宽度和 sparse scan 次数。

因此这一设计不是普通循环切块，而是把内存预算、稀疏扫描次数、AMX tile 形状和
多线程归约统一到一个不可变执行计划中。

### 2.2 Aggregate High-D one-final-pull

旧 D-slab 路径中，每个 slab 都会产生一个 `N×K` 的 partial dP/dX，并再次扫描 CSR：

```text
D-slab 1 -> dP1 -> CSR pull -> partial dX1
D-slab 2 -> dP2 -> CSR pull -> partial dX2
...
```

新路径让所有 slab 直接累加到唯一、预算内的 FP32 `dP[N,Kp]`：

```text
多个 AMX D-slab
  -> 同一个 FP32 dP accumulator
  -> 一次 FP32→BF16
  -> 一次 scaled CSR pull
  -> dX
```

它消除了逐 slab `part_dx`、`dx.add_` 的全量扫描，并把 CSR pull 从 12 次降为 1 次。
IGB-HOM-small、`K=128,D=2983,32T` 的真实训练 CSR gate：

```text
旧路径：2403.62 ms
新路径： 743.76 ms
局部 backward 加速：3.232x
dX relative L2：0.322%
```

该数字是 High-D backward 路径加速，不是完整训练 E2E。当前仍为显式 opt-in，
进入自动 planner 前必须完成多 epoch 收敛和完整训练 gate。

### 2.3 Wide-K 静态首层数据流复用

对于静态图、静态输入的首层，预计算 `SX` 后可同时消除每轮 forward 和 dW backward
中的 sparse traversal。32T 首层 forward+backward gate（不含一次性 cache build）：

| 数据集 | K | 加速比 | 相对 L2 |
|---|---:|---:|---:|
| Flickr | 500 | 5.324x | 0.264% |
| Reddit | 602 | 16.504x | 0.281% |
| Yelp | 300 | 5.443x | 0.282% |
| AmazonProducts | 200 | 29.089x | 0.126% |

这项收益来自“静态数据流复用”，而不是把 cache 命中本身包装成新算法。由于 BF16
下 `S(XW)` 与 `(SX)W` 的重结合会改变舍入顺序，它仍需完整收敛验证，默认 auto
行为未改变。

### 2.4 Supervision-scoped terminal 与 bounded exact CE

末层只对监督行生成 pulled features、logits 和梯度，并以最多 30 万行的外层 panel
执行精确交叉熵：

```text
P panel
  -> logits + scale + bias
  -> exact log-softmax / CE
  -> Gs + db
  -> panel-local AMX dW
  -> compact Q
```

该路径将 terminal loss 边界、监督范围和 AMX dense kernel 放入同一数据流，同时
保留 FP32 master parameters、精确 mean CE 和正常 autograd 顺序。

### 2.5 Shape-aware、budget-aware 的唯一执行契约

planner 只依赖 `N/K/D`、层位置、静态性、线程数和明确的 workspace 预算，不读取
数据集名称。每层只保存一个不可变 `execution_variant`，forward、backward、cache、
日志和 provenance 必须与之匹配；不一致时直接失败，避免运行时二次启发式漂移。

## 3. 本轮已修复的实现问题

| 修复项 | 处理结果 |
|---|---|
| int64→int32 CSR cache 仅按裸指针识别 | 改为持有源 tensor，并使用 TensorImpl identity + version counter |
| CSR cache 无界增长/生命周期风险 | 改为有界 8-entry cache，并增加 mutation/lifetime 测试 |
| V4 backward 元数据错误报告为 Gs/D | 改为真实 sparse operand `dP/K/Kp` |
| SupervisionScope 可跨图或跨线程误复用 | 绑定 rowptr/colidx identity、version、节点数和线程数 |
| High-D Q-first BF16→FP32→BF16 往返 | BF16 dP 直接进入 BF16 pull primitive |
| D-slab Python `.contiguous()` 大拷贝 | native adapter 支持 column-slab stride，移除强制复制 |
| 每个 D-slab 重复生成 N×K partial 并重复 pull | one-final-pull 统一 dP accumulator |
| planner 对 D-slab 与 dP workspace 计费不完整 | 同时核算 bounded dW workspace 与唯一 dP accumulator |
| packed-W 可能跨 optimizer step 失效 | 生命周期限制为同一次 forward，下一步重新打包 |
| 未验证候选可能被误当正式路径 | 保持 shadow ABI/显式 opt-in，并用 immutable plan 校验 dispatch |

最新服务器回归：`127 passed`，`aggregate_saved_v4 smoke PASS`。

## 4. 新探索的真实结论

本分支保留负结果，因为它们决定哪些数据流不应进入正式代码。

| 路线 | 子算子结果 | 完整边界结果 | 决策 |
|---|---:|---:|---|
| Hidden ReLU/Dropout/BF16 bridge | 静态 mask 可加速 | 计入 RNG 后 0.784x | 不接入 |
| Terminal Native-Q | 多 panel 最高 2.224x | IGB terminal 0.998x | 不接入 planner |
| destination-pull Gs panel | 无 | 0.095x–0.567x | 原型清除 |
| source-row Gs panel + 全局 Q | 单全量 panel 最高 1.110x | bounded panel 0.918x–1.010x | 不接入 |
| 无全局 Gs/Q，Q panel 直接累加 dH | 数值通过 | 0.563x–0.695x | 性能淘汰 |

上述结果说明：中间张量消除不能只看“少分配一个 tensor”。如果引入重复 rowptr
扫描、反复读写全局 dH、额外 dense launch 或按边重复 BF16 转换，整体仍会变慢。

## 5. 当前状态边界

### 已实现并具有明确局部收益

- 两级 panel 执行与统一 planner 元数据；
- Aggregate High-D one-final-pull；
- High-D D-slab 零拷贝 adapter；
- Wide-K 静态首层 `SX` 复用；
- supervision-scoped terminal / bounded exact CE；
- BF16 pull、fused db、compact dW 和 packed-weight 生命周期基础。

### 仍需正式训练验证

- Aggregate High-D one-final-pull 的 convergence 与 E2E；
- Wide-K 静态首层数据流的 convergence 与 E2E。

### 已冻结或淘汰

- Hidden bridge、孤立 Native-Q、普通层 Gs panel 消除；
- HBM Cache Mode locality 微调；
- Transform High-D 重复扫描重构；
- 未校准的 Y-first/Q-first cost planner；
- RCM 等依赖离线重排的正式训练路线。

## 6. 源码导航

| 内容 | 路径 |
|---|---|
| 唯一执行计划与预算 | `python/tfs_train/execution_plan.py` |
| High-D row-panel / D-slab / one-final-pull | `python/tfs_train/highd_backward.py` |
| 维度分发与 autograd | `python/tfs_train/dimension_dispatch.py` |
| supervision-scoped terminal | `python/tfs_train/supervision_scope.py` |
| aggregate-saved autograd | `python/tfs_train/aggregate_saved.py` |
| AMX/AVX native 主实现 | `csrc/experiments/backward_opt_20260814/v6_wide_aggregate_probe.cpp` |
| AMX backward kernels | `csrc/experiments/products_saved_t_20260812/backward_v2_kernels.cpp` |
| pybind 接口 | `csrc/experiments/products_saved_t_20260812/bindings_aggregate.cpp` |
| 本轮完整修复日志 | `docs/AUDIT_REPAIR_STATUS_20260910.md` |
| 可复现实验结果 | `evidence/audit_repair_20260911/` |

## 7. 构建与测试

```bash
bash scripts/build_extension.sh
```

审计修复回归使用：

```bash
sbatch --export=ALL,TFS_ROOT=$PWD slurm/test_audit_repair_shadow.sh
```

主要性能 gate：

```bash
sbatch --export=ALL,TFS_ROOT=$PWD slurm/bench_aggregate_one_pull.sh
sbatch --export=ALL,TFS_ROOT=$PWD slurm/bench_wide_k_static_real.sh
sbatch --export=ALL,TFS_ROOT=$PWD slurm/bench_compact_q_reuse_gate.sh
```

正式 48-cell 入口仍以原权威脚本和冻结环境契约为准；本研究分支中的 shadow gate
不能替代论文正式训练矩阵。

## 8. 随附评审材料

- `docs/reviews_20260910/TFS_source_review_zh.md`
- `docs/reviews_20260910/TFS_academic_novelty_review_zh.md`
- `docs/reviews_20260910/TFS_academic_novelty_bundle.zip`
- `docs/reviews_20260910/TFS_当前源码问题与重大加速空间审计_20260910.md`

这些文件是外部独立评审和创新建议，不等同于代码已经实现或论文结论。实际状态以
本 README、修复日志、测试和 evidence 为准。
